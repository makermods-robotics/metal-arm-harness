"""Plays an approved waypoint stream at the arm's control rate.

The executor is deliberately dumb: it takes waypoints the safety envelope
already approved, sends them one per control period, and re-checks runtime
health between sends. Unarmed mode runs the identical loop with the motor
write removed, so a dry run exercises everything but torque.

`settle` is the one piece of feedback control: the real arm lags the stream
(the driver's lead cap bounds torque) and sags under gravity, so after the
last waypoint it holds the approved target with integral action until the
measured joints converge or the timeout passes. A gripper that stops moving
short of its target while closing is holding something — reported, not
waited on; one that stops while opening is blocked, and reported as such.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metal_arm_harness.arms.base import Arm
from metal_arm_harness.safety import MoveRejected, SafetyEnvelope


@dataclass(frozen=True)
class MoveReport:
    """What actually happened, for the tool result and the episode log."""

    steps: int
    duration_s: float
    commanded_deg: tuple[float, ...]
    measured_deg: tuple[float, ...]
    armed: bool
    #: Measured joint vectors sampled once per tick while playing (for dip metrics).
    trace: tuple[tuple[float, ...], ...] = ()


def play(
    arm: Arm,
    safety: SafetyEnvelope,
    waypoints: npt.NDArray[np.float64],
    *,
    armed: bool,
    gripper_override: float | None = None,
) -> MoveReport:
    """Send each waypoint at the control rate; re-check health between them.

    `gripper_override` replaces the gripper value of every sent waypoint: the
    jaws are not geometry, so a squeeze command (already inside the gripper's
    limits) rides along an approved arm path without being re-planned.
    """
    period = 1.0 / arm.info.control_hz
    started = time.monotonic()
    trace: list[tuple[float, ...]] = []
    for waypoint in waypoints:
        tick = time.monotonic()
        state = arm.read()
        safety.check_runtime(state)
        trace.append(tuple(float(v) for v in state.positions_deg))
        if armed:
            arm.send(_with_gripper(arm, waypoint, gripper_override))
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
        trace=tuple(trace),
    )


#: Fraction of the residual added to the hold offset per tick.
HOLD_INTEGRAL_GAIN = 0.5


@dataclass(frozen=True)
class SettleReport:
    """How close the arm got to the approved target after holding it."""

    converged: bool
    residual_deg: tuple[float, ...]
    #: Jaws stopped short while CLOSING: holding something.
    gripper_stalled: bool
    #: Jaws stopped short while OPENING: blocked or faulted, never "holding".
    gripper_blocked: bool
    duration_s: float
    #: The last motor command sent, so the next move can start from it.
    last_command: tuple[float, ...] | None


def settle(
    arm: Arm,
    safety: SafetyEnvelope,
    target_deg: Sequence[float],
    *,
    armed: bool,
    timeout_s: float = 4.0,
    tol_deg: float = 0.5,
    stall_window_s: float = 0.6,
    gripper_override: float | None = None,
    initial_lead_deg: Sequence[float] | None = None,
    max_lead_deg: float = 2.0,
) -> SettleReport:
    """Hold an approved target until the arm converges (or the jaws stall).

    Arm joints are held with integral action: each tick commands
    `target + lead`, `lead` accumulating the residual (bounded by
    `max_lead_deg`), so a joint that sags under gravity is brought onto the
    target instead of resting `sag` below it. `initial_lead_deg` carries the
    lead the previous command stream had built, so torque is continuous
    across the hand-over. Every tick's command is re-planned through the
    envelope from the measured pose, so nothing unapproved is sent.
    """
    target = np.asarray(target_deg, dtype=np.float64)
    period = 1.0 / arm.info.control_hz
    gripper = arm.info.gripper_index
    arm_mask = np.ones(target.shape, dtype=bool)
    if gripper is not None:
        arm_mask[gripper] = False
    lead = np.zeros(target.shape, dtype=np.float64)
    if initial_lead_deg is not None:
        lead[arm_mask] = np.clip(
            np.asarray(initial_lead_deg, dtype=np.float64)[arm_mask], -max_lead_deg, max_lead_deg
        )
    started = time.monotonic()
    stall_since: float | None = None
    stall_reference: float | None = None
    gripper_stopped = False
    closing = True
    start_gripper = float(arm.read().positions_deg[gripper]) if gripper is not None else 0.0
    if gripper is not None:
        closing = target[gripper] < start_gripper
    last_command: np.ndarray | None = None
    while True:
        state = arm.read()
        safety.check_runtime(state)
        measured = np.asarray(state.positions_deg, dtype=np.float64)
        residual = target - measured
        arm_ok = bool(np.all(np.abs(residual[arm_mask]) <= tol_deg))
        gripper_ok = gripper is None or abs(residual[gripper]) <= tol_deg
        if gripper is not None and not gripper_ok:
            position = float(measured[gripper])
            now = time.monotonic()
            if stall_reference is None or abs(position - stall_reference) > tol_deg:
                stall_reference, stall_since = position, now
            elif stall_since is not None and now - stall_since >= stall_window_s:
                gripper_stopped = True
        if not armed or (arm_ok and (gripper_ok or gripper_stopped)):
            break
        if time.monotonic() - started >= timeout_s:
            break
        lead = np.clip(lead + HOLD_INTEGRAL_GAIN * residual * arm_mask, -max_lead_deg, max_lead_deg)
        desired = target + lead
        try:
            waypoints = safety.plan_move(measured, desired)
        except MoveRejected:
            desired = target  # fall back to the approved target itself
            waypoints = np.asarray([desired])
        command = waypoints[-1] if len(waypoints) else desired
        arm.send(_with_gripper(arm, command, gripper_override))
        last_command = np.asarray(_with_gripper(arm, command, gripper_override))
        time.sleep(period)
    final = np.abs(arm.read().positions_deg - target)
    stalled = gripper_stopped and closing
    blocked = gripper_stopped and not closing
    converged = bool(np.all(final[arm_mask] <= tol_deg)) and (
        gripper is None or final[gripper] <= tol_deg or stalled
    )
    return SettleReport(
        converged=converged,
        residual_deg=tuple(float(v) for v in final),
        gripper_stalled=stalled,
        gripper_blocked=blocked,
        duration_s=time.monotonic() - started,
        last_command=None if last_command is None else tuple(float(v) for v in last_command),
    )


def _with_gripper(arm: Arm, target: Sequence[float], gripper_override: float | None) -> list[float]:
    values = [float(v) for v in target]
    if gripper_override is not None and arm.info.gripper_index is not None:
        values[arm.info.gripper_index] = float(gripper_override)
    return values
