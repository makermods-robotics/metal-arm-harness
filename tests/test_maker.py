"""Maker integration: real follower in simulation, and a wire-level read-only fake."""

from __future__ import annotations

import dataclasses
import math
import struct
from collections import deque

import can
import numpy as np
import pytest

from metal_arm_harness import executor
from metal_arm_harness.arms import build_arm
from metal_arm_harness.arms.maker import MAKER_START_DEG, MakerArm
from metal_arm_harness.cli import _calibration_path, build_parser
from metal_arm_harness.control import Controller
from metal_arm_harness.ik import offset_target, solve_tip
from metal_arm_harness.maker_kinematics import MakerKinematics
from metal_arm_harness.operator import OperatorSession
from metal_arm_harness.safety import MoveRejected, SafetyConfig, SafetyEnvelope


@pytest.fixture
def rig(tmp_path):
    arm, kin = build_arm("maker", backend="sim", cameras="")
    arm.connect()
    safety = SafetyEnvelope(SafetyConfig(), arm.info, kin)
    safety.set_floor(kin.min_height_m(arm.read().positions_deg) - 0.1)
    session = OperatorSession(arm, kin, safety, Controller(arm, safety, armed=False), tmp_path)
    yield arm, kin, safety, session
    arm.close()


def test_sim_uses_maker_models_limits_and_no_metal_lead_caps(rig):
    arm, _, _, _ = rig
    assert arm.read().positions_deg == pytest.approx(MAKER_START_DEG)
    assert arm._follower.bus.torque_enabled is False
    with pytest.raises(RuntimeError, match="enable_torque"):
        arm.send(MAKER_START_DEG)
    arm.enable_torque()
    target = np.array(MAKER_START_DEG)
    target[0] += 50
    target[6] = 112  # Invalid Metal opening is capped at Maker's upper limit.
    arm.send(target)
    assert arm.read().positions_deg[0] == pytest.approx(2)
    assert arm.read().positions_deg[6] == pytest.approx(-2.5)
    assert set(arm._follower.config.max_relative_target.values()) == {2}
    assert arm._follower.config.gains["shoulder_lift"] == (150, 4.5)


def test_operator_open_close_use_maker_targets_and_rest_is_explicit(rig, monkeypatch):
    arm, _, _, session = rig
    seen = []
    monkeypatch.setattr(session, "_move", lambda targets, **kw: seen.append(targets) or {})
    assert session.handle("open", [])["ok"]
    assert session.handle("close", [])["ok"]
    assert seen == [{"gripper": -3.0}, {"gripper": -119.6}]
    assert not session.handle("rest", [])["ok"]
    assert not session.handle("clear-faults", [])["ok"]
    arm.info = dataclasses.replace(arm.info, gripper_index=None)
    assert "no gripper" in session.handle("open", [])["error"]


def test_maker_motion_and_grasp_through_controller(rig):
    arm, _, _, session = rig
    arm.enable_torque()
    session.controller.armed = True
    moved = session.handle("goto", ["shoulder_pan=1"])
    assert moved["ok"] and moved["joints"]["shoulder_pan"] == pytest.approx(1, abs=0.1)
    bus = arm._follower.bus
    original = bus.write

    def contact(data_name, motor, value):
        original(data_name, motor, max(value, -20) if motor == "gripper" else value)

    bus.write = contact
    target = arm.read().positions_deg.copy()
    target[6] = -30
    settled = executor.settle(
        arm, session.safety, target, armed=True, timeout_s=2, stall_window_s=0.05
    )
    assert settled.gripper_stalled and not settled.gripper_blocked


def test_gripper_closing_direction_comes_from_arm_info(rig):
    arm, _, safety, _ = rig
    arm.info = dataclasses.replace(arm.info, gripper_open_deg=-119.6, gripper_closed_deg=-3)
    arm.enable_torque()
    arm._follower.bus._positions["gripper"] = -30
    original = arm._follower.bus.write

    def contact(data_name, motor, value):
        original(data_name, motor, min(value, -20) if motor == "gripper" else value)

    arm._follower.bus.write = contact
    target = arm.read().positions_deg.copy()
    target[6] = -3
    settled = executor.settle(arm, safety, target, armed=True, timeout_s=2, stall_window_s=0.05)
    assert settled.gripper_stalled and not settled.gripper_blocked


def test_cad_fk_is_z_up_and_gripper_bounds_cover_all_openings(rig):
    arm, kin, _, _ = rig
    zero_tip, pitch = kin.tool_pose(np.zeros(7))
    assert zero_tip == pytest.approx([-0.2281227, 0, 0.1223697], abs=0.002)
    assert math.isfinite(pitch)
    assert kin.hardware_verified is False
    q = arm.read().positions_deg
    for roll in (-100, 0, 90):
        q[5] = roll
        q[6] = -3
        height = kin.min_height_m(q)
        q[6] = -119.6
        assert kin.min_height_m(q) == pytest.approx(height)
        assert height < kin.tool_tip_z_m(q)
    with pytest.raises(ValueError, match="seven finite"):
        kin.tool_pose([0] * 6)


def test_maker_ik_and_heading_do_not_assume_metal_geometry(rig):
    arm, kin, safety, _ = rig
    q = arm.read().positions_deg
    target, pitch = offset_target(kin, q, up_m=0.01)
    solution = solve_tip(kin, q, target, pitch, limits=(safety._low, safety._high))
    assert solution.tip_m == pytest.approx(target, abs=0.002)
    forward, _ = offset_target(kin, q, forward_m=0.01)
    tip, _ = kin.tool_pose(q)
    assert forward[0] < tip[0]  # CAD zero gripper points -X, unlike Metal.
    safety.set_floor(kin.min_height_m(q) + 0.01)
    # Find a downward joint step; the entire path must be refused.
    for index in (1, 2, 3):
        down = q.copy()
        down[index] -= 2
        if kin.min_height_m(down) < kin.min_height_m(q) - 1e-4:
            with pytest.raises(MoveRejected):
                safety.plan_move(q, down)
            break
    else:
        pytest.fail("no downward fixture found")


class ReadOnlyWire:
    def __init__(self):
        self.sent = []
        self.queue = deque()

    def send(self, message):
        self.sent.append(message)
        mid = message.arbitration_id & 255
        if message.arbitration_id < 128:
            assert bytes(message.data) == bytes([255] * 6 + [0, 0xFB])
            payload = bytes([mid, 0, 0, 0, 0])
            if mid % 2 == 0:
                payload += bytes(3)
            self.queue.append(can.Message(arbitration_id=0xFD, data=payload, is_extended_id=False))
        else:
            index = struct.unpack("<H", message.data[:2])[0]
            assert index in (0x7019, 0x7028)
            value = struct.pack("<f", -0.25) if index == 0x7019 else struct.pack("<I", 4000)
            self.queue.append(
                can.Message(
                    arbitration_id=0x300 | mid,
                    data=struct.pack("<H2x", index) + value,
                    is_extended_id=False,
                )
            )

    def recv(self, timeout):
        return self.queue.popleft() if self.queue else None


def test_real_connection_and_diagnostics_never_change_torque_or_clear_faults(monkeypatch, tmp_path):
    arm = MakerArm(backend="slcan", port="fake", cameras="")
    bus = arm._follower.bus
    wire = ReadOnlyWire()
    calls = []

    def connect(handshake=True):
        assert handshake is False
        bus.canbus = wire
        calls.append("connect")

    monkeypatch.setattr(bus, "connect", connect)
    monkeypatch.setattr(bus, "disconnect", lambda disable: calls.append(("disconnect", disable)))
    arm.connect()
    assert wire.sent == []
    kin = MakerKinematics()
    safety = SafetyEnvelope(SafetyConfig(), arm.info, kin)
    session = OperatorSession(
        arm, kin, safety, Controller(arm, safety, armed=False), tmp_path, diagnostics_only=True
    )
    for command in ("goto", "open", "clear-faults", "tip", "rest", "observe", "status"):
        assert session.handle(command, [])["ok"] is False
    assert wire.sent == []
    result = session.handle("inspect", [])
    assert result["ok"]
    assert len(wire.sent) == 21
    for state in result["diagnostics"]["motors"].values():
        assert state["has_fault"] is False
        assert state["raw_position_deg"] == pytest.approx(math.degrees(-0.25))
        assert state["watchdog_ticks"] == 4000
    with pytest.raises(RuntimeError, match="not commissioned"):
        arm.enable_torque()
    with pytest.raises(RuntimeError, match="cached zeros"):
        arm.read()
    with pytest.raises(RuntimeError, match="no faults cleared"):
        arm.clear_faults()
    assert len(wire.sent) == 21
    arm.close()
    assert calls == ["connect", ("disconnect", False)]


def test_status_packets_wrong_ids_and_short_packets_are_not_encoders():
    for payload in (b"", b"\x01", bytes([2, 0, 0, 0, 0]), bytes(8)):
        msg = can.Message(arbitration_id=0xFD, data=payload, is_extended_id=False)
        assert MakerArm._fault_reply(msg, 1) is None
    fault = can.Message(arbitration_id=0xFD, data=[1, 4, 0, 0, 0], is_extended_id=False)
    assert MakerArm._fault_reply(fault, 1) == bytes([4, 0, 0, 0])


def test_table_files_are_isolated_by_arm_robot_and_backend():
    parser = build_parser()
    paths = {
        _calibration_path(
            parser.parse_args(["serve", "--arm", arm, "--robot-id", robot, "--backend", backend])
        )
        for arm in ("metal", "maker")
        for robot in ("one", "two")
        for backend in ("sim", "slcan")
    }
    assert len(paths) == 8
    assert all(p.parent.name == "tables" for p in paths)


@pytest.mark.parametrize("cap", [0, -1, float("nan"), {"elbow_flex": float("inf")}, {"bad": 1}])
def test_invalid_lead_caps_are_rejected(cap):
    with pytest.raises(ValueError):
        MakerArm(backend="sim", lead_cap_deg=cap)


def test_missing_replies_do_not_become_cached_feedback(rig):
    arm, _, _, _ = rig
    wire = ReadOnlyWire()
    arm._follower.bus.canbus = wire
    wire.send = lambda message: wire.sent.append(message)
    assert arm._parameter(1, 0x7019, "f") is None
    assert len(wire.sent) == 1

    # A status reply cannot satisfy a position parameter query.
    def status_only(message):
        wire.queue.append(
            can.Message(arbitration_id=0xFD, data=[1, 0, 0, 0, 0], is_extended_id=False)
        )

    wire.send = status_only
    assert arm._parameter(1, 0x7019, "f") is None


def test_armed_diagnostics_are_rejected_before_building_an_arm(monkeypatch, capsys):
    from metal_arm_harness import cli

    monkeypatch.setattr(cli, "build_arm", lambda *a, **kw: pytest.fail("opened hardware"))
    assert cli.main(["serve", "--arm", "maker", "--diagnostics-only", "--armed"]) == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_explicit_floor_cannot_bypass_unverified_maker_geometry(rig):
    from metal_arm_harness.cli import _set_floor
    from metal_arm_harness.episode_log import EpisodeLog

    arm, kin, safety, _ = rig
    args = build_parser().parse_args(["serve", "--arm", "maker", "--table-z", "-10"])
    with pytest.raises(RuntimeError, match="unverified"):
        _set_floor(args, arm, kin, safety, EpisodeLog(None))
