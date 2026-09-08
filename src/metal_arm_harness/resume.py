"""Restore a recent, stationary powered hold without sending motor commands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from metal_arm_harness.control import Controller


def resume_hold(controller: Controller, path: Path) -> None:
    """Explicit restart only: reject stale logs, changed poses, or lost grasps."""
    sample = move = floor = None
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("event") == "motion_sample":
                sample = item
            elif item.get("event") == "move":
                move = item
            elif item.get("event") == "floor_set":
                floor = item["floor_z_m"]
            elif item.get("event") in ("safety_abort", "relieve"):
                move = None  # an abort invalidates the preceding completed hold
    if sample is None or move is None or sample.get("phase") != "hold":
        raise ValueError("resume requires a completed move followed by a read-only monitor")
    if not 0 <= time.time() - sample["t"] <= 300:
        raise ValueError("resume monitor is older than five minutes")
    if tuple(sample["joint_names"]) != controller.arm.info.joint_names:
        raise ValueError("resume joint names do not match this arm")
    if floor != controller.safety.floor_z_m:
        raise ValueError("resume table calibration does not match")
    goal = np.array(sample["goal_deg"], dtype=float)
    command = np.array(sample["command_deg"], dtype=float)
    state = controller.arm.read()
    controller.safety.check_runtime(state)
    shape = state.positions_deg.shape
    for vector in (goal, command):
        if vector.shape != shape or not np.all(np.isfinite(vector)):
            raise ValueError("invalid resume joint vector")
        if not np.allclose(vector, controller.safety.clamp_to_limits(vector), atol=1e-8):
            raise ValueError("resume command outside joint limits")
    gi = controller.arm.info.gripper_index
    mask = np.ones(shape, dtype=bool)
    if gi is not None:
        mask[gi] = False
    if (
        np.max(np.abs(command[mask] - state.positions_deg[mask])) > 4
        or np.max(np.abs(goal[mask] - state.positions_deg[mask])) > 4
        or np.max(np.abs(state.velocities_deg_s[mask])) > 2
    ):
        raise ValueError("arm moved or is moving; cannot resume this hold")
    held = bool(move.get("gripper_stalled") and not move.get("object_lost"))
    if held and gi is not None:
        if (
            abs(state.positions_deg[gi] - sample["measured_deg"][gi]) > 2
            or abs(state.efforts_nm[gi]) < 0.4
        ):
            raise ValueError("grip no longer matches the recorded hold")
    controller.commanded = goal
    controller.last_command = command
    controller.gripper_stalled = held
    controller.contact_deg = float(state.positions_deg[gi]) if held and gi is not None else None
    # This is an already energized, verified hold; startup's 1-degree cap would
    # temporarily remove gravity compensation and grip force. Lead caps remain.
    if hasattr(controller.arm, "_follower"):
        controller.arm._follower._synced = True
    controller.log.event(
        "resume_hold",
        source=str(path),
        goal=goal.tolist(),
        command=command.tolist(),
        gripper_stalled=held,
    )
