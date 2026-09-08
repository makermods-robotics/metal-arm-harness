"""Runtime aborts must not replay a stale target while relieving another joint."""

from dataclasses import replace

import numpy as np
import pytest

from metal_arm_harness import executor
from metal_arm_harness.arms.base import ArmInfo, ArmState, JointSpec
from metal_arm_harness.control import Controller
from metal_arm_harness.operator import OperatorSession
from metal_arm_harness.safety import SafetyAbort, SafetyConfig, SafetyEnvelope


class HeatingArm:
    info = ArmInfo(
        "memory",
        tuple(JointSpec(f"joint{i}", -180, 180) for i in range(3)),
        None,
        50,
        "synthetic feedback only",
    )

    def __init__(self):
        self.position = np.zeros(3)
        self.sag = np.zeros(3)
        self.sent = []
        self.hot = False
        self.heat_when = lambda q: False

    def read(self):
        return ArmState(
            self.position - self.sag, np.zeros(3), np.array([80 if self.hot else 25, 25, 25])
        )

    def send(self, q):
        self.position = np.asarray(q).copy()
        self.sent.append(self.position.copy())
        self.hot = self.hot or self.heat_when(self.position)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(executor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(executor.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    arm = HeatingArm()
    safety = SafetyEnvelope(SafetyConfig(max_speed_deg_s=5), arm.info, None)
    controller = Controller(arm, safety, armed=True)
    session = OperatorSession(arm, None, safety, controller, tmp_path)
    return arm, controller, session


@pytest.mark.parametrize("phase", ["play", "settle"])
def test_heat_relief_keeps_other_joints_at_the_last_successful_command(rig, phase):
    arm, controller, session = rig
    controller.goto({"joint1": 1})
    arm.sent.clear()
    if phase == "play":
        arm.heat_when = lambda q: q[1] >= 1.5
    else:
        arm.sag[1] = 0.8
        arm.heat_when = lambda q: q[1] > 3.05
    reply = session.handle("goto", ["joint1=3"])
    assert reply["abort"]
    assert len(arm.sent) >= 2
    last_move, relief = arm.sent[-2:]
    assert relief[0] == pytest.approx(arm.read().positions_deg[0])
    assert relief[1:] == pytest.approx(last_move[1:])
    assert controller.last_command == pytest.approx(relief)


def test_abort_replaces_the_previous_completed_goal_with_the_actual_hold(rig):
    arm, controller, _session = rig
    controller.goto({"joint1": 1})
    arm.heat_when = lambda q: q[1] >= 1.5
    with pytest.raises(SafetyAbort):
        controller.goto({"joint1": 3})
    assert controller.base() == pytest.approx(arm.sent[-1])


def test_interrupted_move_retains_the_standing_gripper_squeeze(rig):
    arm, controller, session = rig
    controller.goto({"joint1": 1})
    arm.info = replace(arm.info, gripper_index=2)
    controller.safety = SafetyEnvelope(controller.safety.config, arm.info, None)
    session.safety = controller.safety
    controller.gripper_stalled = True
    read = arm.read

    def gripping():
        state = read()
        state.positions_deg[2] = 5  # Object stops the jaws short of their zero-degree goal.
        return state

    arm.read = gripping
    arm.heat_when = lambda q: q[1] >= 1.5
    assert session.handle("goto", ["joint1=3"])["abort"]
    assert arm.read().positions_deg[2] == 5
    assert controller.base()[2] == 0
    assert arm.sent[-1][2] == 0


def test_command_tracking_preserves_the_intended_goal_in_telemetry(rig):
    arm, controller, _session = rig
    samples = []
    controller.log.event = lambda kind, **data: samples.append((kind, data))
    controller.goto({"joint1": 3})
    motion = [data for kind, data in samples if kind == "motion_sample"]
    assert len(motion) > 2
    assert all(data["goal_deg"] == [0.0, 3.0, 0.0] for data in motion)
    assert controller.commanded == pytest.approx([0, 3, 0])
    assert controller.last_command == pytest.approx(arm.sent[-1])
