"""The whole loop with a scripted model: floor rejection, recovery, done.

httpx.MockTransport stands in for the Anthropic API, so no network and no
key — but everything else is real: the sim arm, FK on the vendored URDF, the
safety envelope's floor and slow zone, waypoint execution, and the message
plumbing the real model will see.
"""

from __future__ import annotations

import importlib
import importlib.util
import json

import numpy as np
import pytest

from metal_arm_harness.agent import AgentConfig, ArmAgent
from metal_arm_harness.arms import build_arm
from metal_arm_harness.safety import SafetyConfig, SafetyEnvelope

# Newer anthropic SDKs vendor httpx as `httpx2` and type-check the injected
# client against it; use whichever module the installed SDK expects.
httpx = importlib.import_module(
    "httpx2" if importlib.util.find_spec("httpx2") else "httpx"
)

pytest.importorskip("pinocchio")


def _message(*blocks) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": list(blocks),
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def _tool_use(name: str, arguments: dict) -> dict:
    return {"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": arguments}


class Script:
    def __init__(self, responses: list[dict]):
        self.queue = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return httpx.Response(200, json=payload)


@pytest.fixture
def rig():
    # High control rate keeps wall-clock short; speeds stay proportionate.
    arm, kinematics = build_arm("metal", backend="sim", control_hz=250.0)
    arm.connect()
    arm.enable_torque()
    safety = SafetyEnvelope(
        SafetyConfig(max_speed_deg_s=200.0, slow_speed_deg_s=50.0), arm.info, kinematics
    )
    # Table 2 cm below the arm's current lowest point: pan moves (height-
    # neutral) are fine, any real descent crosses the 10 mm floor margin.
    start = arm.read().positions_deg
    safety.set_floor(kinematics.min_height_m(start) - 0.02)
    yield arm, safety
    arm.close()


def test_floor_rejection_then_recovery_then_done(rig) -> None:
    arm, safety = rig
    start = np.asarray(arm.read().positions_deg, dtype=np.float64)
    note = "Probing per the notes."
    script = Script(
        [
            # Folding the elbow down dives toward the table (drops the arm
            # ~9 cm from the sim start pose): must hit the hard floor.
            _message(_tool_use("move_joints", {
                "targets": {"elbow_flex": float(start[2] - 30.0)}, "note": note,
            })),
            # Height-neutral base rotation: allowed (slow, we are in the zone).
            _message(_tool_use("move_joints", {
                "targets": {"shoulder_pan": float(start[0] + 8.0)}, "note": note,
            })),
            _message(_tool_use("done", {
                "summary": "rotated the base",
                "hindsight": "the table guard rejects descents near the floor",
            })),
        ]
    )
    agent = ArmAgent(
        arm,
        safety,
        AgentConfig(api_key="test-key", max_llm_calls=10),
        armed=True,
        http_client=httpx.Client(transport=httpx.MockTransport(script)),
    )

    result = agent.run("pick up the blue cube")

    assert result.status == "done"
    assert result.detail == "rotated the base"
    assert result.hindsight.startswith("the table guard")
    assert result.llm_calls == 3
    assert result.moves == 1 and result.rejections == 1

    # The dive never moved the arm; the pan move landed.
    final = np.asarray(arm.read().positions_deg, dtype=np.float64)
    assert final[2] == pytest.approx(start[2])
    assert final[0] == pytest.approx(start[0] + 8.0, abs=1e-6)

    # The rejection reached the model as a correctable tool error.
    flattened = json.dumps(script.requests[1]["messages"])
    assert "hard floor" in flattened and "aim higher" in flattened

    # The system prompt carries the contract: safety rules + arm notes.
    system = script.requests[0]["system"]
    assert "hard floor" in system and "0 is CLOSED" in system

    # Observations carry clearance and camera images.
    first_user = json.dumps(script.requests[0]["messages"][0])
    assert "clearance_m" in first_user and "image/png" in first_user


def test_unarmed_agent_never_moves_the_sim_arm(rig) -> None:
    arm, safety = rig
    start = np.asarray(arm.read().positions_deg, dtype=np.float64)
    script = Script(
        [
            _message(_tool_use("move_joints", {
                "targets": {"shoulder_pan": float(start[0] + 5.0)},
                "note": "dry run",
            })),
            _message(_tool_use("done", {"summary": "dry run complete"})),
        ]
    )
    agent = ArmAgent(
        arm,
        safety,
        AgentConfig(api_key="test-key", max_llm_calls=5),
        armed=False,
        http_client=httpx.Client(transport=httpx.MockTransport(script)),
    )

    result = agent.run("rotate a little")

    assert result.status == "done"
    assert arm.read().positions_deg == pytest.approx(start)
    assert "UNARMED" in json.dumps(script.requests[1]["messages"])


def test_text_only_reply_gets_a_nudge_and_budget_forces_an_end(rig) -> None:
    arm, safety = rig
    script = Script(
        [
            _message({"type": "text", "text": "I would rather write a poem."}),
        ]
    )
    agent = ArmAgent(
        arm,
        safety,
        AgentConfig(api_key="test-key", max_llm_calls=3),
        armed=True,
        http_client=httpx.Client(transport=httpx.MockTransport(script)),
    )

    result = agent.run("do nothing useful")

    assert result.status == "budget_exhausted"
    assert result.llm_calls == 3
    nudges = [
        message
        for request in script.requests
        for message in request["messages"]
        if message.get("content") == "Respond with exactly one tool call."
    ]
    assert nudges
