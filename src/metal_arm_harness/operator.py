"""Operator session: one connection, many commands, a human (or an agent
reading camera files) as the policy.

`metal-arm-harness serve` connects the arm and cameras once, sets the floor,
optionally arms, and then answers JSON-line commands on a Unix socket.
`metal-arm-harness op <command>` sends one command and prints the reply.
Compared with a process per command this removes the CAN handshake (~2 s),
camera re-open and auto-exposure warm-up from every step, which on the
bench was most of the wall-clock time of a pick-and-place.

Commands (all motion goes through `Controller.goto`, hence the envelope):

  observe                  joints, clearance, tool pose; frames saved as JPEGs
  goto j=deg ...           absolute joint targets, chunked past the excursion cap
  tip X Y Z PITCH          tool tip target in the base frame (m, deg) via IK
  nudge forward= left= up= [pitch=]   move the tip by metres in the arm's
                           horizontal frame; the operator's "2 cm forward"
  gripper DEG | open | close
  rest                     the zero pose, gripper unchanged
  clear-faults             clear latched motor faults (e.g. gripper rotor overtemp)
  status | quit
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from metal_arm_harness.arms.base import Arm, Kinematics
from metal_arm_harness.control import Controller
from metal_arm_harness.episode_log import EpisodeLog
from metal_arm_harness.ik import IKError, offset_target, solve_tip
from metal_arm_harness.safety import MoveRejected, SafetyAbort, SafetyEnvelope

DEFAULT_SOCKET = Path.home() / ".metal-arm-harness" / "operator.sock"
GRIPPER_OPEN_DEG = 112.0
GRIPPER_CLOSED_DEG = 0.0


class OperatorError(RuntimeError):
    """A command could not be carried out; the message is for the operator."""


@dataclass
class OperatorSession:
    """Owns a connected arm and answers operator commands."""

    arm: Arm
    kinematics: Kinematics | None
    safety: SafetyEnvelope
    controller: Controller
    frames_dir: Path
    log: EpisodeLog = field(default_factory=lambda: EpisodeLog(None))
    _frame_counter: int = 0

    # ── commands ────────────────────────────────────────────────────────────

    def handle(self, command: str, args: list[str]) -> dict[str, Any]:
        """Dispatch one command. Never raises for operator mistakes."""
        try:
            handler = getattr(self, f"cmd_{command.replace('-', '_')}", None)
            if handler is None:
                raise OperatorError(f"unknown command {command!r}")
            result = handler(args)
            result.setdefault("ok", True)
            return result
        except (OperatorError, MoveRejected, IKError, ValueError) as error:
            self.log.event("command_error", command=command, args=args, error=str(error))
            return {"ok": False, "error": str(error), **self._state()}
        except SafetyAbort as abort:
            self.log.event("safety_abort", reason=str(abort))
            relief = self._relieve_after_abort(str(abort))
            return {"ok": False, "error": f"SAFETY ABORT: {abort}{relief}", "abort": True}

    def _relieve_after_abort(self, reason: str) -> str:
        """An over-temperature veto stops motion but not the motor's push: zero the
        P-torque on the hot joint so it stops heating (the gripper stall case)."""
        try:
            state = self.arm.read()
        except Exception:
            return ""
        temps = np.asarray(state.temperatures_c, dtype=np.float64)
        hot = [i for i, t in enumerate(temps) if t > self.safety.config.max_temp_c]
        if not hot or not self.controller.armed:
            return ""
        for joint in hot:
            self.controller.relieve(joint)
        names = ", ".join(self.arm.info.joint_names[i] for i in hot)
        return f" (torque relieved on {names}; wait for it to cool, then continue)"

    def cmd_clear_faults(self, args: list[str]) -> dict[str, Any]:
        cleared = self.arm.clear_faults()
        state = self.arm.read()
        remaining = [n for n, f in zip(self.arm.info.joint_names, state.faults, strict=False) if f]
        text = f"cleared faults on: {', '.join(cleared) or 'nothing (no latched faults)'}" + (
            f"; still faulted: {', '.join(remaining)}" if remaining else ""
        )
        self.log.event("clear_faults", cleared=list(cleared), remaining=remaining)
        return {"text": text, "cleared": list(cleared), "remaining": remaining, **self._state()}

    def cmd_status(self, args: list[str]) -> dict[str, Any]:
        return {"text": self._state_text(), **self._state()}

    def cmd_observe(self, args: list[str]) -> dict[str, Any]:
        tag = args[0] if args else "look"
        return self._observe(tag)

    def cmd_goto(self, args: list[str]) -> dict[str, Any]:
        targets = _parse_targets(args, self.arm.info.joint_names)
        if not targets:
            raise OperatorError("goto needs joint=degrees pairs")
        return self._move(targets, note=" ".join(args))

    def cmd_gripper(self, args: list[str]) -> dict[str, Any]:
        if len(args) != 1:
            raise OperatorError("gripper needs one value in degrees")
        return self._move({"gripper": float(args[0])}, note=f"gripper {args[0]}")

    def cmd_open(self, args: list[str]) -> dict[str, Any]:
        return self._move({"gripper": GRIPPER_OPEN_DEG}, note="open")

    def cmd_close(self, args: list[str]) -> dict[str, Any]:
        return self._move({"gripper": GRIPPER_CLOSED_DEG}, note="close")

    def cmd_rest(self, args: list[str]) -> dict[str, Any]:
        names = self.arm.info.joint_names
        targets = {n: 0.0 for i, n in enumerate(names) if i != self.arm.info.gripper_index}
        return self._move(targets, note="rest")

    def cmd_tip(self, args: list[str]) -> dict[str, Any]:
        if len(args) != 4:
            raise OperatorError("tip needs X Y Z (metres, base frame) and PITCH (deg)")
        x, y, z, pitch = (float(a) for a in args)
        return self._move_tip([x, y, z], pitch, note=f"tip {' '.join(args)}")

    def cmd_nudge(self, args: list[str]) -> dict[str, Any]:
        kinematics = self._pose_kinematics()
        fields = _parse_targets(args, ("forward", "left", "up", "pitch"))
        if not fields:
            raise OperatorError("nudge needs forward=/left=/up= (metres) and/or pitch= (deg)")
        base = self.controller.base()
        target, pitch = offset_target(
            kinematics,
            base,
            forward_m=fields.get("forward", 0.0),
            left_m=fields.get("left", 0.0),
            up_m=fields.get("up", 0.0),
        )
        return self._move_tip(target, fields.get("pitch", pitch), note=f"nudge {' '.join(args)}")

    def cmd_quit(self, args: list[str]) -> dict[str, Any]:
        return {"text": "closing session; a holding arm keeps holding", "quit": True}

    # ── helpers ─────────────────────────────────────────────────────────────

    def _pose_kinematics(self) -> Any:
        if self.kinematics is None or not hasattr(self.kinematics, "tool_pose"):
            raise OperatorError("this arm has no tool-pose kinematics; use goto")
        return self.kinematics

    def _move_tip(self, target: Any, pitch: float, *, note: str) -> dict[str, Any]:
        kinematics = self._pose_kinematics()
        base = self.controller.base()
        solution = solve_tip(
            kinematics,
            base,
            target,
            pitch,
            limits=(self.safety._low, self.safety._high),
        )
        clearance = self.safety.clearance_m(solution.joints_deg)
        if clearance is not None and clearance < self.safety.config.floor_margin_m:
            raise OperatorError(
                f"that tip pose would leave only {clearance * 1000:.0f} mm above the table "
                f"(margin {self.safety.config.floor_margin_m * 1000:.0f} mm); aim higher"
            )
        result = self._move(solution.joints_deg, note=note)
        result["ik"] = {
            "tip_m": [round(float(v), 4) for v in solution.tip_m],
            "pitch_deg": round(solution.pitch_deg, 1),
            "iterations": solution.iterations,
        }
        return result

    def _move(self, targets: Mapping[str, float] | Any, *, note: str) -> dict[str, Any]:
        report = self.controller.goto(targets, note=note, chunk=True)
        time.sleep(0.3)  # let the wrist camera catch up with the arm
        result = self._observe("after")
        result["move"] = report.summary(self.arm.info.joint_names)
        result["text"] = result["move"] + "\n" + result["text"]
        return result

    def _observe(self, tag: str) -> dict[str, Any]:
        state_sample = self.arm.read()
        frames: dict[str, Any] = {}
        missing: list[str] = []
        # A dropped frame on one camera must not fail the command: the move already
        # happened. Report the camera as missing and carry on with the others.
        for camera in getattr(self.arm, "_cameras", ()):
            try:
                frames[camera.name] = camera.read()
            except RuntimeError as error:
                missing.append(f"{camera.name}: {error}")
        if not hasattr(self.arm, "_cameras"):
            frames = dict(self.arm.frames())
        paths = self._save_frames(frames, tag)
        state = self._state(state_sample.positions_deg)
        self.log.event("observe", tag=tag, frames=[str(p) for p in paths], missing=missing, **state)
        text = self._state_text(state_sample.positions_deg)
        if paths:
            text += "\nframes: " + " ".join(str(p) for p in paths)
        if missing:
            text += "\nCAMERA MISSING: " + "; ".join(missing)
        return {
            "text": text,
            "frames": [str(p) for p in paths],
            "missing_cameras": missing,
            **state,
        }

    def _save_frames(self, frames: Mapping[str, Any], tag: str) -> list[Path]:
        from PIL import Image

        if not frames:
            return []
        self._frame_counter += 1
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, frame in frames.items():
            path = self.frames_dir / f"{self._frame_counter:03d}_{tag}_{name}.jpg"
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path, quality=85)
            paths.append(path)
        return paths

    def _state(self, positions: Any = None) -> dict[str, Any]:
        q = np.asarray(
            self.arm.read().positions_deg if positions is None else positions, dtype=np.float64
        )
        state: dict[str, Any] = {
            "joints": {
                n: round(float(v), 1) for n, v in zip(self.arm.info.joint_names, q, strict=True)
            },
            "armed": self.controller.armed,
        }
        clearance = self.safety.clearance_m(q)
        if clearance is not None:
            state["clearance_mm"] = round(clearance * 1000)
        try:
            faults = self.arm.read().faults
        except Exception:
            faults = ()
        flagged = [
            f"{n}: {f}" for n, f in zip(self.arm.info.joint_names, faults, strict=False) if f
        ]
        if flagged:
            state["faults"] = flagged
        if self.kinematics is not None and hasattr(self.kinematics, "tool_pose"):
            tip, pitch = self.kinematics.tool_pose(q)
            state["tip_m"] = [round(float(v), 4) for v in tip]
            state["pitch_deg"] = round(float(pitch), 1)
        return state

    def _state_text(self, positions: Any = None) -> str:
        state = self._state(positions)
        lines = ["joints (deg): " + " ".join(f"{k}={v:.1f}" for k, v in state["joints"].items())]
        extra = []
        if "clearance_mm" in state:
            extra.append(f"clearance {state['clearance_mm']} mm")
        if "tip_m" in state:
            x, y, z = state["tip_m"]
            extra.append(f"tip ({x:.3f}, {y:.3f}, {z:.3f}) m pitch {state['pitch_deg']:.0f} deg")
        extra.append("ARMED" if state["armed"] else "UNARMED")
        lines.append("   ".join(extra))
        if state.get("faults"):
            lines.append("MOTOR FAULTS: " + "; ".join(state["faults"]) + "  -> `op clear-faults`")
        return "\n".join(lines)


def _parse_targets(args: list[str], allowed: tuple[str, ...] | list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for arg in args:
        if "=" not in arg:
            raise OperatorError(f"expected name=value, got {arg!r}")
        name, _, value = arg.partition("=")
        name = name.strip()
        if name not in allowed:
            raise OperatorError(f"unknown name {name!r}; expected one of {list(allowed)}")
        try:
            out[name] = float(value)
        except ValueError:
            raise OperatorError(f"{name} is not a number: {value!r}") from None
    return out


# ── transport ───────────────────────────────────────────────────────────────


def serve(session: OperatorSession, socket_path: Path = DEFAULT_SOCKET) -> None:
    """Answer JSON-line commands on a Unix socket until `quit`."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(socket_path))
        except OSError:
            socket_path.unlink()  # stale: nobody is listening
        else:
            probe.close()
            raise OperatorError(
                f"another operator session is live on {socket_path}; one arm, one session"
            )
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(1)
    print(f"operator session ready on {socket_path}", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            reply: dict[str, Any] = {}
            try:
                with conn, conn.makefile("rwb") as stream:
                    line = stream.readline()
                    if not line:
                        continue
                    try:
                        request = json.loads(line)
                        reply = session.handle(
                            str(request.get("command", "")), list(request.get("args", []))
                        )
                    except Exception as error:  # a bug must not take the arm session down
                        reply = {
                            "ok": False,
                            "error": f"internal error: {error}",
                            "trace": traceback.format_exc(),
                        }
                    stream.write((json.dumps(reply, default=str) + "\n").encode("utf-8"))
                    stream.flush()
            except OSError as error:  # client went away mid-reply: keep serving
                print(f"client connection dropped: {error}", file=sys.stderr, flush=True)
            if reply.get("quit"):
                break
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()


def send(
    command: str, args: list[str], socket_path: Path = DEFAULT_SOCKET, timeout_s: float = 600.0
) -> dict[str, Any]:
    """Send one command to a running session and return its reply."""
    if not socket_path.exists():
        raise OperatorError(
            f"no operator session at {socket_path}; start one with `metal-arm-harness serve`"
        )
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_s)
    client.connect(str(socket_path))
    with client, client.makefile("rwb") as stream:
        stream.write((json.dumps({"command": command, "args": args}) + "\n").encode("utf-8"))
        stream.flush()
        line = stream.readline()
    if not line:
        raise OperatorError("session closed without replying")
    return json.loads(line)


def print_reply(reply: Mapping[str, Any], stream: Any = None) -> int:
    stream = stream or sys.stdout
    if not reply.get("ok", False):
        print(f"ERROR: {reply.get('error')}", file=stream)
        if reply.get("trace"):
            print(reply["trace"], file=stream)
        if reply.get("joints"):
            print(
                "joints (deg): " + " ".join(f"{k}={v:.1f}" for k, v in reply["joints"].items()),
                file=stream,
            )
        return 1
    print(reply.get("text", json.dumps(reply)), file=stream)
    if reply.get("ik"):
        ik = reply["ik"]
        print(
            f"ik: tip {ik['tip_m']} pitch {ik['pitch_deg']} ({ik['iterations']} iterations)",
            file=stream,
        )
    return 0
