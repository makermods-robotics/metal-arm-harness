"""The LLM-as-policy loop, on the Anthropic Messages API.

One model call per decision. The model interacts only through tools:

- ``move_joints`` — absolute degree targets for any subset of joints, with a
  mandatory human-readable ``note``. The safety envelope plans the motion; a
  rejection comes back as a correctable tool error, not a crash.
- ``look`` — a fresh observation with no motion.
- ``done`` / ``give_up`` — end the episode, optionally recording
  ``hindsight``: what the model wishes it had known at the start.

After every tool call the model receives labeled joint positions, its current
clearance above the table (when the guard is active), and one image per
camera. The transport is injectable, so the whole loop runs against a
scripted fake in tests.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from metal_arm_harness import executor
from metal_arm_harness.arms.base import Arm, Observation
from metal_arm_harness.episode_log import EpisodeLog
from metal_arm_harness.safety import MoveRejected, SafetyAbort, SafetyEnvelope

DEFAULT_MODEL = "claude-opus-5"

_SYSTEM_TEMPLATE = """\
You are the control policy for a physical robot arm named "{name}". An \
operator has given you a task and is watching. You act ONLY by calling \
tools, exactly one per turn. After every call you receive the arm's joint \
positions in degrees, your clearance above the table when it is measured, \
and a fresh image from each camera.

Safety rules, enforced by the harness (violations return an error you can \
correct, they never crash the run):
{safety}

Every move_joints call must include a `note`: one or two sentences on what \
you observe and why you chose this motion — the operator reads these to \
follow your reasoning.

You have a budget of {budget} model calls. When the task is complete call \
`done`; if you are stuck call `give_up`. Both take optional `hindsight`: \
what you wish you had known from the start, for the next attempt.

Arm notes:
{notes}"""


def _tools_schema(joint_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": "move_joints",
            "description": (
                "Move to absolute joint targets in degrees. Provide only the joints "
                "you want to change; the others hold. The motion is interpolated at "
                "a safe speed and may be rejected with a reason."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": (
                            f"joint name -> absolute degrees; joints: {list(joint_names)}"
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "note": {"type": "string"},
                },
                "required": ["targets", "note"],
            },
        },
        {
            "name": "look",
            "description": "Take a fresh observation (cameras and joint state) without moving.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "done",
            "description": "The task is complete. Ends the episode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "hindsight": {"type": "string"},
                },
                "required": ["summary"],
            },
        },
        {
            "name": "give_up",
            "description": "The task cannot be completed. Ends the episode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "hindsight": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    ]


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    max_llm_calls: int = 40
    max_tokens: int = 1500
    api_key: str | None = None  # None: the SDK reads ANTHROPIC_API_KEY


@dataclass
class EpisodeResult:
    status: str  # "done" | "give_up" | "budget_exhausted" | "aborted"
    detail: str
    hindsight: str | None
    llm_calls: int
    moves: int = 0
    rejections: int = 0


@dataclass
class _Turn:
    """One executed tool call's outcome, to be rendered as a tool_result."""

    tool_use_id: str
    text: str
    is_error: bool = False
    observation: Observation | None = None


class ArmAgent:
    """Runs one episode: observe, ask the model, execute, repeat."""

    def __init__(
        self,
        arm: Arm,
        safety: SafetyEnvelope,
        config: AgentConfig | None = None,
        *,
        armed: bool,
        log: EpisodeLog | None = None,
        http_client: Any = None,
    ):
        import anthropic

        self.arm = arm
        self.safety = safety
        self.config = config or AgentConfig()
        self.armed = armed
        self.log = log or EpisodeLog(None)
        kwargs: dict[str, Any] = {}
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = anthropic.Anthropic(**kwargs)
        self._tools = _tools_schema(arm.info.joint_names)
        self._messages: list[dict[str, Any]] = []

    # ── episode ─────────────────────────────────────────────────────────────

    def run(self, instruction: str) -> EpisodeResult:
        """Drive the arm until done, give_up, budget exhaustion, or an abort."""
        system = _SYSTEM_TEMPLATE.format(
            name=self.arm.info.name,
            safety=self.safety.describe(),
            budget=self.config.max_llm_calls,
            notes=self.arm.info.notes,
        )
        self.log.event("episode_start", instruction=instruction, armed=self.armed)
        observation = self.arm.observe()
        self._messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {instruction}"},
                    *self._observation_blocks(observation),
                ],
            }
        ]

        calls = moves = rejections = 0
        while calls < self.config.max_llm_calls:
            calls += 1
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system,
                tools=self._tools,
                messages=self._messages,
            )
            content = [block.model_dump() for block in response.content]
            self._messages.append({"role": "assistant", "content": content})
            self.log.event("llm_response", call=calls, content=_strip_images(content))

            tool_use = next((b for b in content if b.get("type") == "tool_use"), None)
            if tool_use is None:
                self._messages.append(
                    {"role": "user", "content": "Respond with exactly one tool call."}
                )
                continue

            name, args = tool_use["name"], tool_use.get("input") or {}
            if name == "done":
                self.log.event("done", **args)
                return EpisodeResult(
                    "done", str(args.get("summary", "")), args.get("hindsight"),
                    calls, moves, rejections,
                )
            if name == "give_up":
                self.log.event("give_up", **args)
                return EpisodeResult(
                    "give_up", str(args.get("reason", "")), args.get("hindsight"),
                    calls, moves, rejections,
                )

            try:
                turn = self._execute(tool_use["id"], name, args)
            except SafetyAbort as abort:
                self.log.event("safety_abort", reason=str(abort))
                return EpisodeResult("aborted", str(abort), None, calls, moves, rejections)
            moves += int(name == "move_joints" and not turn.is_error)
            rejections += int(turn.is_error)
            self._append_tool_result(turn)

        self.log.event("budget_exhausted", calls=calls)
        return EpisodeResult(
            "budget_exhausted", "model call budget exhausted", None, calls, moves, rejections
        )

    # ── tool execution ──────────────────────────────────────────────────────

    def _execute(self, tool_use_id: str, name: str, args: dict[str, Any]) -> _Turn:
        if name == "look":
            return _Turn(tool_use_id, "Fresh observation.", observation=self.arm.observe())
        if name != "move_joints":
            return _Turn(tool_use_id, f"unknown tool {name!r}", is_error=True)

        targets = args.get("targets")
        if not isinstance(targets, dict) or not targets:
            return _Turn(
                tool_use_id,
                "move_joints needs `targets`: {joint_name: absolute_degrees, ...}",
                is_error=True,
            )
        names = self.arm.info.joint_names
        unknown = sorted(set(targets) - set(names))
        if unknown:
            return _Turn(
                tool_use_id,
                f"unknown joint(s) {unknown}; this arm has {list(names)}",
                is_error=True,
            )
        if not str(args.get("note", "")).strip():
            return _Turn(
                tool_use_id,
                "note is required: describe what you observe and why you chose this motion",
                is_error=True,
            )

        current = self.arm.read().positions_deg
        target = np.array(current, dtype=np.float64)
        for joint, value in targets.items():
            try:
                target[names.index(joint)] = float(value)
            except (TypeError, ValueError):
                return _Turn(
                    tool_use_id, f"target for {joint} is not a number: {value!r}", is_error=True
                )

        try:
            waypoints = self.safety.plan_move(current, target)
        except MoveRejected as rejection:
            self.log.event("move_rejected", reason=str(rejection), targets=targets)
            return _Turn(tool_use_id, f"move rejected: {rejection}", is_error=True)

        report = executor.play(self.arm, self.safety, waypoints, armed=self.armed)
        self.log.event(
            "move",
            targets=targets,
            steps=report.steps,
            duration_s=round(report.duration_s, 3),
            armed=report.armed,
        )
        prefix = "Move played" if self.armed else "UNARMED: move computed and logged, not sent"
        return _Turn(
            tool_use_id,
            f"{prefix} ({report.steps} steps, {report.duration_s:.1f}s).",
            observation=self.arm.observe(),
        )

    # ── message rendering ───────────────────────────────────────────────────

    def _append_tool_result(self, turn: _Turn) -> None:
        content: list[dict[str, Any]] = [{"type": "text", "text": turn.text}]
        if turn.observation is not None:
            content += self._observation_blocks(turn.observation)
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": turn.tool_use_id,
                        "is_error": turn.is_error,
                        "content": content,
                    }
                ],
            }
        )

    def _observation_blocks(self, observation: Observation) -> list[dict[str, Any]]:
        positions = observation.state.positions_deg
        lines = [
            "joints (deg): "
            + " ".join(
                f"{name}={value:.1f}"
                for name, value in zip(self.arm.info.joint_names, positions, strict=True)
            )
        ]
        clearance = self.safety.clearance_m(positions)
        if clearance is not None:
            lines.append(f"clearance_m above table: {clearance:.3f}")
        if not self.armed:
            lines.append("mode: UNARMED (moves are computed and logged, never sent)")
        blocks: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
        for name, frame in observation.frames.items():
            blocks.append({"type": "text", "text": f"camera '{name}':"})
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode_png(frame),
                    },
                }
            )
        return blocks


def _encode_png(frame: Any) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _strip_images(content: Any) -> Any:
    """Drop base64 payloads before logging."""
    return json.loads(
        json.dumps(content, default=str)
        .replace('"type": "base64"', '"type": "elided"')
    )
