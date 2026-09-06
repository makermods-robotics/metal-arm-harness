"""Observable write behavior with deterministic feedback and a real envelope."""

import numpy as np
import pytest

from metal_arm_harness import executor
from metal_arm_harness.arms.base import ArmInfo, ArmState, JointSpec
from metal_arm_harness.control import Controller
from metal_arm_harness.safety import MoveRejected, SafetyAbort, SafetyConfig, SafetyEnvelope


class MemoryArm:
    """Ideal position feedback with a visible command log; no transport."""

    info = ArmInfo(
        "memory", tuple(JointSpec(f"joint{i}", -180, 180) for i in range(6)), 5, 50, "test only"
    )

    def __init__(self):
        self.positions = np.array([0.0, -20.0, 25.0, -15.0, 0.0, 30.0])
        self.sent = []

    def read(self):
        return ArmState(self.positions.copy(), np.zeros(6), np.full(6, 25.0))

    def send(self, q):
        self.positions = np.asarray(q).copy()
        self.sent.append(self.positions.copy())


@pytest.fixture
def rig(monkeypatch):
    arm = MemoryArm()
    clock = [0.0]
    monkeypatch.setattr(executor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(executor.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    safety = SafetyEnvelope(SafetyConfig(max_speed_deg_s=5, slow_speed_deg_s=1), arm.info, None)
    return arm, safety


def test_excursion_veto_sends_nothing(rig):
    arm, safety = rig
    target = arm.positions.copy()
    target[0] += 40
    with pytest.raises(MoveRejected, match="per-move"):
        executor.settle(arm, safety, target, armed=True)
    assert arm.sent == []


def test_actual_floor_veto_sends_nothing(rig):
    arm, safety = rig

    class Height:
        def min_height_m(self, q):
            return 0.02 - float(q[0]) / 100

    safety.kinematics = Height()
    safety.set_floor(0)
    target = arm.positions.copy()
    target[0] += 2
    with pytest.raises(MoveRejected, match="hard floor"):
        executor.settle(arm, safety, target, armed=True)
    assert arm.sent == []


@pytest.mark.parametrize("slow", [False, True])
def test_every_correction_is_paced_from_last_command(rig, slow):
    arm, safety = rig

    class Height:
        def min_height_m(self, q):
            return 0.05

    if slow:
        safety.kinematics = Height()
        safety.set_floor(0)
    initial = arm.positions.copy()
    target = initial.copy()
    target[0] += 1
    report = executor.settle(arm, safety, target, armed=True, max_lead_deg=0, tol_deg=0.001)
    assert report.converged
    step = (
        safety.config.slow_speed_deg_s if slow else safety.config.max_speed_deg_s
    ) / arm.info.control_hz
    deltas = np.diff(np.vstack([initial, *arm.sent]), axis=0)
    assert np.max(np.abs(deltas)) <= step + 1e-10
    assert len(arm.sent) > 1
    assert report.last_command == pytest.approx(arm.sent[-1])


def test_new_floor_veto_after_one_step_stops_further_writes(rig):
    arm, safety = rig
    send = arm.send

    class Height:
        def min_height_m(self, q):
            return 0.02 - float(q[0]) / 100

    def move_floor(q):
        send(q)
        safety.kinematics = Height()
        safety.set_floor(0)

    arm.send = move_floor
    target = arm.positions.copy()
    target[0] += 2
    with pytest.raises(MoveRejected):
        executor.settle(arm, safety, target, armed=True)
    assert len(arm.sent) == 1


def test_runtime_fault_between_corrections_prevents_second_write(rig):
    arm, safety = rig
    send = arm.send

    def fail_feedback(q):
        send(q)
        arm.positions[1] = np.nan

    arm.send = fail_feedback
    target = arm.positions.copy()
    target[0] += 2
    with pytest.raises(SafetyAbort, match="non-finite"):
        executor.settle(arm, safety, target, armed=True)
    assert len(arm.sent) == 1


def test_unarmed_settle_has_no_writes(rig):
    arm, safety = rig
    target = arm.positions.copy()
    target[0] += 5
    executor.settle(arm, safety, target, armed=False)
    assert arm.sent == []


def test_last_command_carries_gravity_lead_without_downward_reset(rig):
    arm, safety = rig
    measured = arm.positions.copy()
    previous = measured.copy()
    previous[1] += 1.2
    arm.positions = previous.copy()
    original_read = arm.read

    def sagged():
        state = original_read()
        state.positions_deg[1] -= 1.2
        return state

    arm.read = sagged
    goal = previous.copy()
    report = executor.settle(
        arm, safety, goal, armed=True, initial_command_deg=previous, timeout_s=4
    )
    assert report.converged
    stream = np.vstack([previous, *arm.sent])
    assert np.max(np.abs(np.diff(stream, axis=0))) <= 5 / 50 + 1e-10
    assert min(q[1] for q in arm.sent) >= previous[1]
    assert abs(arm.read().positions_deg[1] - goal[1]) <= 0.5


def test_rejected_correction_invalidates_controller_command_cache(rig, monkeypatch):
    arm, safety = rig
    controller = Controller(arm, safety, armed=True)

    def rejected(*args, **kwargs):
        # A partial correction has moved beyond the previous play's endpoint.
        q = arm.positions.copy()
        q[0] += 0.01
        arm.send(q)
        raise MoveRejected("changed floor")

    monkeypatch.setattr(executor, "settle", rejected)
    with pytest.raises(MoveRejected):
        controller.goto({"joint0": 1})
    assert controller.last_command is None
    assert controller.commanded is None
    assert controller.base() == pytest.approx(arm.positions)


@pytest.mark.parametrize("origin", [[0], [float("nan")] * 6])
def test_invalid_command_origin_is_refused_without_writes(rig, origin):
    arm, safety = rig
    with pytest.raises(ValueError, match="initial_command"):
        executor.settle(arm, safety, arm.positions, armed=True, initial_command_deg=origin)
    assert arm.sent == []


def test_measured_slow_zone_caps_a_correction_from_a_higher_command_origin(rig):
    arm, safety = rig

    class Height:
        def min_height_m(self, q):
            return 0.075 + float(q[0]) * 0.001

    safety.kinematics = Height()
    safety.set_floor(0)
    previous = arm.positions.copy()
    previous[0] += 2
    target = previous.copy()
    target[0] += 1
    assert safety.clearance_m(arm.positions) < safety.config.slow_zone_m
    assert safety.clearance_m(previous) > safety.config.slow_zone_m
    executor.settle(
        arm,
        safety,
        target,
        armed=True,
        initial_command_deg=previous,
        max_lead_deg=0,
        timeout_s=0.02,
    )
    assert len(arm.sent) == 1
    assert (
        np.max(np.abs(arm.sent[0] - previous))
        <= safety.config.slow_speed_deg_s / arm.info.control_hz + 1e-10
    )


def test_actual_next_command_is_checked_from_feedback(rig):
    arm, safety = rig

    class Height:
        def min_height_m(self, q):
            # The two plans to the goal miss this low pocket; the path from
            # feedback to the first command-origin waypoint crosses it.
            return 0.005 if 0.45 < q[0] < 0.65 and -0.55 < q[1] < -0.35 else 0.2

    arm.positions[:2] = 0
    previous = arm.positions.copy()
    previous[:2] = [1, -1]
    target = arm.positions.copy()
    target[:2] = [2, 2]
    safety.kinematics = Height()
    safety.set_floor(0)
    safety.plan_move(arm.positions, target)
    safety.plan_move(previous, target)
    with pytest.raises(MoveRejected, match="hard floor"):
        executor.settle(
            arm, safety, target, armed=True, initial_command_deg=previous, max_lead_deg=0
        )
    assert arm.sent == []


def test_settle_can_recover_upward_from_inside_floor_margin(rig):
    arm, safety = rig

    class Height:
        def min_height_m(self, q):
            return 0.005 + float(q[0]) / 100

    safety.kinematics = Height()
    safety.set_floor(0)
    initial = arm.positions.copy()
    target = initial.copy()
    target[0] += 1
    report = executor.settle(arm, safety, target, armed=True, max_lead_deg=0, tol_deg=0.001)
    assert report.converged
    heights = [safety.clearance_m(q) for q in [initial, *arm.sent]]
    assert heights[0] < safety.config.floor_margin_m
    assert np.all(np.diff(heights) >= 0)
