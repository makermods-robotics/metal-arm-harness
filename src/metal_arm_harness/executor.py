"""Plays an approved waypoint stream at the arm's control rate.

The executor is deliberately dumb: it takes waypoints the safety envelope
already approved, sends them one per control period, and re-checks runtime
health between sends. Unarmed mode runs the identical loop with the motor
write removed, so a dry run exercises everything but torque.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metal_arm_harness.arms.base import Arm
from metal_arm_harness.safety import SafetyEnvelope


@dataclass(frozen=True)
class MoveReport:
    """What actually happened, for the tool result and the episode log."""

    steps: int
    duration_s: float
    commanded_deg: tuple[float, ...]
    measured_deg: tuple[float, ...]
    armed: bool


def play(
    arm: Arm,
    safety: SafetyEnvelope,
    waypoints: npt.NDArray[np.float64],
    *,
    armed: bool,
) -> MoveReport:
    """Send each waypoint at the control rate; re-check health between them."""
    period = 1.0 / arm.info.control_hz
    started = time.monotonic()
    for waypoint in waypoints:
        tick = time.monotonic()
        safety.check_runtime(arm.read())
        if armed:
            arm.send([float(v) for v in waypoint])
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    measured = arm.read().positions_deg
    commanded = waypoints[-1] if len(waypoints) else measured
    return MoveReport(
        steps=len(waypoints),
        duration_s=time.monotonic() - started,
        commanded_deg=tuple(float(v) for v in commanded),
        measured_deg=tuple(float(v) for v in measured),
        armed=armed,
    )
