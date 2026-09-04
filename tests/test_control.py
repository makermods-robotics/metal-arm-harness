"""Controller and executor.settle against the bench simulator."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from metal_arm_harness import executor
from metal_arm_harness.arms import build_arm
from metal_arm_harness.control import Controller
from metal_arm_harness.safety import MoveRejected, SafetyConfig, SafetyEnvelope


@pytest.fixture
def rig():
    arm, kinematics = build_arm("metal", backend="sim")
    arm.connect()
    safety = SafetyEnvelope(SafetyConfig(), arm.info, kinematics)
    safety.set_floor(kinematics.min_height_m(arm.read().positions_deg) - 0.10)
    arm.enable_torque()
    yield arm, safety
    arm.close()


def test_goto_reaches_a_small_target_and_records_the_commanded_base(rig) -> None:
    arm, safety = rig
    controller = Controller(arm, safety, armed=True)
    report = controller.goto({"shoulder_pan": 10.0}, note="probe")
    assert report.converged and report.legs == 1
    assert arm.read().positions_deg[0] == pytest.approx(10.0)
    assert controller.commanded is not None and controller.commanded[0] == pytest.approx(10.0)


def test_goto_without_chunking_honours_the_excursion_cap(rig) -> None:
    arm, safety = rig
    controller = Controller(arm, safety, armed=True)
    with pytest.raises(MoveRejected, match="per-move"):
        controller.goto({"shoulder_pan": 60.0})


def test_goto_with_chunking_plays_a_big_swing_in_legs(rig) -> None:
    arm, safety = rig
    controller = Controller(arm, safety, armed=True)
    report = controller.goto({"shoulder_pan": 100.0}, chunk=True)
    assert report.legs >= 3 and report.converged
    assert arm.read().positions_deg[0] == pytest.approx(100.0)


def test_unmentioned_joints_hold_the_commanded_pose_not_the_measured_one(rig) -> None:
    arm, safety = rig
    controller = Controller(arm, safety, armed=True)
    controller.goto({"shoulder_pan": 5.0})
    # Simulate gravity sag: the bus reports the shoulder a degree low.
    arm._follower.bus._positions["shoulder_lift"] -= 1.0
    controller.goto({"gripper": 40.0})
    assert controller.commanded is not None
    assert controller.commanded[1] == pytest.approx(-20.0)  # SIM_START shoulder_lift, not -21
    # The hold pulls the sagged joint back to within its 0.5 deg convergence band.
    assert arm.read().positions_deg[1] == pytest.approx(-20.0, abs=0.6)


def test_a_hand_moved_arm_makes_the_commanded_base_stale(rig) -> None:
    arm, safety = rig
    controller = Controller(arm, safety, armed=True)
    controller.goto({"shoulder_pan": 5.0})
    arm._follower.bus._positions["shoulder_pan"] = 25.0  # moved by hand, > 6 deg
    base = controller.base()
    assert base[0] == pytest.approx(25.0)


def test_unarmed_goto_moves_nothing(rig) -> None:
    arm, safety = rig
    controller = Controller(arm, safety, armed=False)
    before = arm.read().positions_deg.copy()
    report = controller.goto({"shoulder_pan": 10.0})
    assert not report.armed
    assert arm.read().positions_deg == pytest.approx(before)
    assert "UNARMED" in report.summary(arm.info.joint_names)


def test_settle_reports_a_gripper_that_stalls_on_an_object(rig) -> None:
    arm, safety = rig
    bus = arm._follower.bus
    real_write = bus.sync_write_metal

    def jammed(commands):  # the jaws cannot close past 70 degrees
        goals = {m: list(c) for m, c in commands.items()}
        if "gripper" in goals:
            goals["gripper"][2] = max(goals["gripper"][2], 70.0)
        real_write({m: tuple(c) for m, c in goals.items()})

    bus._positions["gripper"] = 100.0  # jaws open, then closing onto the object
    bus.sync_write_metal = jammed
    target = np.array(arm.read().positions_deg)
    target[6] = 40.0
    report = executor.settle(arm, safety, target, armed=True, timeout_s=2.0, stall_window_s=0.05)
    assert report.gripper_stalled and report.converged and not report.gripper_blocked
    assert arm.read().positions_deg[6] == pytest.approx(70.0)


def test_settle_reports_jaws_that_cannot_open_as_blocked_not_holding(rig) -> None:
    arm, safety = rig
    bus = arm._follower.bus
    real_write = bus.sync_write_metal

    def jammed(commands):  # the jaws cannot open past 30 degrees
        goals = {m: list(c) for m, c in commands.items()}
        if "gripper" in goals:
            goals["gripper"][2] = min(goals["gripper"][2], 30.0)
        real_write({m: tuple(c) for m, c in goals.items()})

    bus.sync_write_metal = jammed
    target = np.array(arm.read().positions_deg)
    target[6] = 100.0
    report = executor.settle(arm, safety, target, armed=True, timeout_s=2.0, stall_window_s=0.05)
    assert report.gripper_blocked and not report.gripper_stalled


class SaggingBus:
    """Wraps the sim bus: shoulder_lift reads 1.2 deg below its command (gravity)."""

    SAG: ClassVar[dict[str, float]] = {"shoulder_lift": 1.2}

    def __init__(self, inner):
        self._inner = inner
        self.commands: list[dict[str, float]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def sync_write_metal(self, commands):
        self.commands.append({m: float(c[2]) for m, c in commands.items()})
        self._inner.sync_write_metal(commands)

    def sync_read_all_states(self, motors=None, **_):
        states = self._inner.sync_read_all_states(motors, **_)
        for joint, sag in self.SAG.items():
            states[joint] = dict(states[joint], position=states[joint]["position"] - sag)
        return states


def test_settle_holds_a_sagging_joint_on_target_and_the_next_move_does_not_dip() -> None:
    arm, kinematics = build_arm("metal", backend="sim")
    arm.connect()
    arm._follower.bus = SaggingBus(arm._follower.bus)
    bus = arm._follower.bus
    safety = SafetyEnvelope(SafetyConfig(), arm.info, kinematics)
    safety.set_floor(kinematics.min_height_m(arm.read().positions_deg) - 0.10)
    arm.enable_torque()
    controller = Controller(arm, safety, armed=True)

    first = controller.goto({"shoulder_lift": -15.0}, chunk=True)
    # Integral action: the joint is measured ON target although it sags 1.2 deg below command.
    assert abs(arm.read().positions_deg[1] - (-15.0)) <= 0.5
    assert bus.commands[-1]["shoulder_lift"] > -15.0 + 0.9  # command carries the lead
    held = bus.commands[-1]["shoulder_lift"]

    bus.commands.clear()
    second = controller.goto({"shoulder_lift": -10.0}, chunk=True)
    # The new stream starts from the held command (no drop back to measured + step).
    assert bus.commands[0]["shoulder_lift"] >= held - 1e-6
    assert min(c["shoulder_lift"] for c in bus.commands) >= held - 1e-6
    assert second.counter_dip_deg < 0.3 and first.converged and second.converged
    arm.close()


def test_jaw_only_request_near_the_table_is_one_fast_leg() -> None:
    arm, kinematics = build_arm("metal", backend="sim")
    arm.connect()
    safety = SafetyEnvelope(SafetyConfig(), arm.info, kinematics)
    arm.kinematics_min_height = lambda: kinematics.min_height_m(arm.read().positions_deg)
    arm.enable_torque()
    # A floor 5 cm under the start pose puts the arm inside the 7.6 cm slow zone.
    safety.set_floor(arm.kinematics_min_height() - 0.05)
    controller = Controller(arm, safety, armed=True)
    assert safety.clearance_m(arm.read().positions_deg) < safety.config.slow_zone_m
    report = controller.goto({"gripper": 100.0}, chunk=True)
    assert report.legs == 1
    # 90 degrees of jaw travel at the full 0.8 deg step, not the 0.2 deg slow step.
    assert report.steps <= 120
    assert arm.read().positions_deg[6] == pytest.approx(100.0)


def test_a_stalled_gripper_keeps_its_closed_goal_through_later_moves(rig) -> None:
    arm, safety = rig
    bus = arm._follower.bus
    real_write = bus.sync_write_metal
    commanded_gripper: list[float] = []

    def jammed(commands):  # an object stops the jaws at 70 degrees
        goals = {m: list(c) for m, c in commands.items()}
        if "gripper" in goals:
            commanded_gripper.append(goals["gripper"][2])
            goals["gripper"][2] = max(goals["gripper"][2], 70.0)
        real_write({m: tuple(c) for m, c in goals.items()})

    controller = Controller(arm, safety, armed=True, settle_timeout_s=2.0)
    controller.goto({"gripper": 100.0}, chunk=True)  # open first, jaws free
    bus.sync_write_metal = jammed
    report = controller.goto({"gripper": 0.0}, chunk=True)
    assert report.gripper_stalled
    # The lift must keep commanding the closed goal: that standing error is the grip.
    report = controller.goto({"shoulder_lift": -15.0}, chunk=True)
    # The driver's own lead cap clamps what reaches the bus to measured - 2 deg, so
    # "still squeezing" shows as a command below the 70 deg the jaws are stopped at.
    assert commanded_gripper[-1] < 70.0
    assert report.steps < 40  # no phantom 70-degree gripper travel in the arm stream


def _jam_at(bus, stop_deg: float):
    real_write = bus.sync_write_metal

    def jammed(commands):
        goals = {m: list(c) for m, c in commands.items()}
        if "gripper" in goals:
            goals["gripper"][2] = max(goals["gripper"][2], stop_deg)
        real_write({m: tuple(c) for m, c in goals.items()})

    bus.sync_write_metal = jammed
    return real_write


def test_a_grasp_that_slips_during_a_lift_is_reported_as_object_lost(rig) -> None:
    arm, safety = rig
    bus = arm._follower.bus
    controller = Controller(arm, safety, armed=True, settle_timeout_s=2.0)
    controller.goto({"gripper": 100.0}, chunk=True)
    real_write = _jam_at(bus, 70.0)  # object stops the jaws at 70
    grasp = controller.goto({"gripper": 0.0}, chunk=True)
    assert grasp.gripper_stalled and grasp.contact_deg == pytest.approx(70.0, abs=1.0)

    bus.sync_write_metal = real_write  # object gone: jaws now close freely toward the goal
    lift = controller.goto({"shoulder_lift": -10.0}, chunk=True)
    assert lift.object_lost and not lift.gripper_stalled
    assert "OBJECT LOST" in lift.summary(arm.info.joint_names)


def test_a_held_object_stays_reported_through_a_move(rig) -> None:
    arm, safety = rig
    bus = arm._follower.bus
    controller = Controller(arm, safety, armed=True, settle_timeout_s=2.0)
    controller.goto({"gripper": 100.0}, chunk=True)
    _jam_at(bus, 70.0)
    controller.goto({"gripper": 0.0}, chunk=True)
    lift = controller.goto({"shoulder_lift": -10.0}, chunk=True)
    assert lift.gripper_stalled and not lift.object_lost
    assert "holding something" in lift.summary(arm.info.joint_names)


def test_weak_grip_torque_is_called_out(rig) -> None:
    arm, safety = rig
    bus = arm._follower.bus
    controller = Controller(arm, safety, armed=True, settle_timeout_s=2.0)
    controller.goto({"gripper": 100.0}, chunk=True)
    _jam_at(bus, 70.0)
    original = bus.sync_read_all_states

    def with_torque(motors=None, **kw):
        states = original(motors, **kw)
        states["gripper"] = dict(states["gripper"], torque=0.05)  # barely any squeeze
        return states

    bus.sync_read_all_states = with_torque
    grasp = controller.goto({"gripper": 0.0}, chunk=True)
    assert grasp.grip_torque_nm == pytest.approx(0.05)
    assert "resting ON the object" in grasp.summary(arm.info.joint_names)
