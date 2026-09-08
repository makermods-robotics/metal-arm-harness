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
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metal_arm_harness.arms.base import Arm, ArmState
from metal_arm_harness.safety import MoveRejected, SafetyEnvelope

SampleCallback = Callable[[str, ArmState, Sequence[float] | None], None]


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
    sample: SampleCallback | None = None,
    on_command: Callable[[Sequence[float]], None] | None = None,
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
        if sample is not None:
            sample("play", state, _with_gripper(arm, waypoint, gripper_override))
        if armed:
            command = _with_gripper(arm, waypoint, gripper_override)
            arm.send(command)
            if on_command is not None:
                on_command(command)
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


#: Time-based, bounded integral trim; independent of bus/control tick rate.
HOLD_INTEGRAL_GAIN = 1.5  # 1/s
HOLD_TRIM_SPEED_DEG_S = 0.75
HOLD_STABLE_WINDOW_S = 0.4
HOLD_STABLE_RANGE_DEG = 0.2


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
    sample: SampleCallback | None = None,
    on_command: Callable[[Sequence[float]], None] | None = None,
    correct_arm: bool = True,
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
        lead[arm_mask] = np.asarray(initial_lead_deg, dtype=np.float64)[arm_mask]
    started = time.monotonic()
    stall_since: float | None = None
    stall_reference: float | None = None
    gripper_stopped = False
    closing = True
    start_gripper = float(arm.read().positions_deg[gripper]) if gripper is not None else 0.0
    if gripper is not None:
        closing = (target[gripper] - start_gripper) * (
            arm.info.gripper_closed_deg - arm.info.gripper_open_deg
        ) > 0
    # Start from the command actually left at the motors, not target + measured lag.
    # Measured lag includes motion delay; adding it again creates an endpoint kick.
    command = np.asarray(_with_gripper(arm, target + lead, gripper_override))
    last_command: np.ndarray | None = None
    last_tick = started
    stable_samples: deque[tuple[float, np.ndarray]] = deque()
    while True:
        tick = time.monotonic()
        state = arm.read()
        safety.check_runtime(state)
        if sample is not None:
            sample("settle", state, _with_gripper(arm, command, gripper_override))
        measured = np.asarray(state.positions_deg, dtype=np.float64)
        residual = target - measured
        arm_ok = bool(np.all(np.abs(residual[arm_mask]) <= tol_deg))
        now = time.monotonic()
        stable_samples.append((now, measured.copy()))
        while len(stable_samples) > 1 and now - stable_samples[1][0] >= HOLD_STABLE_WINDOW_S:
            stable_samples.popleft()
        stable = now - stable_samples[0][0] >= HOLD_STABLE_WINDOW_S and bool(
            np.all(
                np.ptp([q[arm_mask] for _, q in stable_samples], axis=0) <= HOLD_STABLE_RANGE_DEG
            )
        )
        gripper_ok = gripper is None or abs(residual[gripper]) <= tol_deg
        if gripper is not None and not gripper_ok:
            position = float(measured[gripper])
            now = time.monotonic()
            if stall_reference is None or abs(position - stall_reference) > tol_deg:
                stall_reference, stall_since = position, now
            elif stall_since is not None and now - stall_since >= stall_window_s:
                gripper_stopped = True
        if not armed or (
            (arm_ok or not correct_arm) and stable and (gripper_ok or gripper_stopped)
        ):
            break
        if time.monotonic() - started >= timeout_s:
            break
        dt = min(now - last_tick, 2 * period)
        last_tick = now
        # Leave joints inside tolerance alone; jaw commands must not trim the arm.
        correction = np.where(arm_mask & (np.abs(residual) > tol_deg) & correct_arm, residual, 0.0)
        step = np.clip(
            HOLD_INTEGRAL_GAIN * correction * dt,
            -HOLD_TRIM_SPEED_DEG_S * dt,
            HOLD_TRIM_SPEED_DEG_S * dt,
        )
        bounded_lead = np.clip(lead + step, -max_lead_deg, max_lead_deg)
        proposed_lead = lead + np.where(
            correction != 0,
            np.clip(bounded_lead - lead, -HOLD_TRIM_SPEED_DEG_S * dt, HOLD_TRIM_SPEED_DEG_S * dt),
            0.0,
        )
        # plan_move may return no waypoints when its clamped target equals
        # the held command. Its empty-path fallback must use that same target.
        desired = safety.clamp_to_limits(target + proposed_lead)
        try:
            # Validate physical clearance from feedback AND preserve command continuity.
            safety.plan_move(measured, desired)
            waypoints = safety.plan_move(command, desired)
        except MoveRejected:
            # Keep the last approved hold; never jump to an unchecked fallback target.
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
            continue
        command = waypoints[0] if len(waypoints) else desired
        lead[arm_mask] = (command - target)[arm_mask]  # no windup past the sent trim
        arm.send(_with_gripper(arm, command, gripper_override))
        last_command = np.asarray(_with_gripper(arm, command, gripper_override))
        if on_command is not None:
            on_command(last_command)
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    final = np.abs(arm.read().positions_deg - target)
    stalled = gripper_stopped and closing
    blocked = gripper_stopped and not closing
    converged = (
        stable
        and bool(np.all(final[arm_mask] <= tol_deg))
        and (gripper is None or final[gripper] <= tol_deg or stalled)
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
