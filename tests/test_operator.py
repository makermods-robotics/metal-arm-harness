"""Operator session over the Unix socket, against the simulator."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest

from metal_arm_harness.arms import build_arm
from metal_arm_harness.control import Controller
from metal_arm_harness.operator import OperatorSession, send, serve
from metal_arm_harness.safety import SafetyConfig, SafetyEnvelope


@pytest.fixture
def session(tmp_path: Path):
    arm, kinematics = build_arm("metal", backend="sim")
    arm.connect()
    safety = SafetyEnvelope(SafetyConfig(), arm.info, kinematics)
    # A low synthetic table, so the zero ("rest") pose is reachable in the sim.
    safety.set_floor(kinematics.min_height_m(arm.read().positions_deg) - 0.25)
    arm.enable_torque()
    yield OperatorSession(
        arm, kinematics, safety, Controller(arm, safety, armed=True), tmp_path / "frames"
    )
    arm.close()


def test_observe_saves_one_frame_per_camera(session: OperatorSession) -> None:
    reply = session.handle("observe", [])
    assert reply["ok"] and len(reply["frames"]) == 1
    assert Path(reply["frames"][0]).is_file()
    assert "clearance_mm" in reply and "tip_m" in reply


def test_goto_and_open_and_rest(session: OperatorSession) -> None:
    assert session.handle("goto", ["shoulder_pan=15"])["joints"]["shoulder_pan"] == pytest.approx(
        15.0, abs=0.1
    )
    assert session.handle("open", [])["joints"]["gripper"] == pytest.approx(112.0, abs=0.1)
    rest = session.handle("rest", [])
    assert rest["ok"] and rest["joints"]["shoulder_pan"] == pytest.approx(0.0, abs=0.1)
    assert rest["joints"]["gripper"] == pytest.approx(112.0, abs=0.1)  # rest leaves the jaws


def test_nudge_moves_the_tip_by_the_requested_offset(session: OperatorSession) -> None:
    before = session.handle("status", [])["tip_m"]
    reply = session.handle("nudge", ["up=0.02"])
    assert reply["ok"], reply.get("error")
    assert reply["tip_m"][2] - before[2] == pytest.approx(0.02, abs=0.003)


def test_tip_rejects_a_pose_under_the_floor_margin(session: OperatorSession) -> None:
    floor = session.safety.floor_z_m
    reply = session.handle("tip", ["0.25", "0.0", f"{floor + 0.002}", "-80"])
    assert not reply["ok"] and ("above the table" in reply["error"] or "reach" in reply["error"])


def test_bad_commands_are_errors_not_crashes(session: OperatorSession) -> None:
    assert not session.handle("frobnicate", [])["ok"]
    assert "unknown name" in session.handle("goto", ["elbow=3"])["error"]
    assert not session.handle("goto", ["shoulder_pan=abc"])["ok"]


def test_socket_round_trip(session: OperatorSession) -> None:
    # AF_UNIX paths are capped at ~100 bytes; pytest's tmp_path is longer than that.
    sock = Path(tempfile.mkdtemp(prefix="mah-")) / "op.sock"
    thread = threading.Thread(target=serve, args=(session, sock), daemon=True)
    thread.start()
    for _ in range(100):
        if sock.exists():
            break
        threading.Event().wait(0.02)
    reply = send("status", [], sock)
    assert reply["ok"] and "joints" in reply
    assert send("quit", [], sock)["quit"]
    thread.join(timeout=5)
    assert not thread.is_alive()
