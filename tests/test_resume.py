import json
import time

import numpy as np
import pytest

from metal_arm_harness.arms import build_arm
from metal_arm_harness.control import Controller
from metal_arm_harness.resume import resume_hold
from metal_arm_harness.safety import SafetyConfig, SafetyEnvelope


def test_resume_preserves_hold_without_sends_and_rejects_moved_arm(tmp_path):
    arm, kin = build_arm("metal", backend="sim")
    arm.connect()
    try:
        envelope = SafetyEnvelope(SafetyConfig(), arm.info, kin)
        controller = Controller(arm, envelope, armed=True)
        q = arm.read().positions_deg
        goal = q.copy()
        command = q.copy()
        command[1] += 1
        sample = {
            "event": "motion_sample",
            "phase": "hold",
            "t": time.time(),
            "joint_names": arm.info.joint_names,
            "goal_deg": goal.tolist(),
            "command_deg": command.tolist(),
            "measured_deg": q.tolist(),
        }
        path = tmp_path / "hold.jsonl"
        path.write_text(
            json.dumps({"event": "move", "gripper_stalled": False})
            + "\n"
            + json.dumps(sample)
            + "\n"
        )
        resume_hold(controller, path)
        assert controller.last_command == pytest.approx(command)
        assert arm._follower.bus.goal_writes == []
        with path.open("a") as stream:
            stream.write(json.dumps({"event": "safety_abort"}) + "\n")
        with pytest.raises(ValueError, match="completed move"):
            resume_hold(controller, path)
        path.write_text(json.dumps({"event": "move"}) + "\n" + json.dumps(sample))
        arm._follower.bus._positions["shoulder_pan"] += 10
        with pytest.raises(ValueError, match="arm moved"):
            resume_hold(Controller(arm, envelope, armed=True), path)
        arm._follower.bus._positions["shoulder_pan"] -= 10
        sample["t"] -= 301
        path.write_text(json.dumps({"event": "move"}) + "\n" + json.dumps(sample))
        with pytest.raises(ValueError, match="older"):
            resume_hold(controller, path)
        assert np.allclose(arm.read().positions_deg, q)
    finally:
        arm.close()
