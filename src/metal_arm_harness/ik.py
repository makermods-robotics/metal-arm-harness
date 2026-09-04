"""Inverse kinematics on the harness FK: tool tip position + pitch → joints.

Damped least squares on a numeric Jacobian of `Kinematics.tool_pose`, using
the four in-plane joints (pan, lift, elbow, wrist pitch) and holding the
rest. It solves in milliseconds and stays close to the starting pose, which
is what an operator or a policy wants: "the same arm, 2 cm forward", not a
different configuration family.

Position tolerance defaults to 2 mm and pitch to 2°; failure raises
`IKError` instead of returning a nearby guess, because the caller is about
to move a real arm.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


class PoseKinematics(Protocol):
    def tool_pose(self, joints_deg: Sequence[float]) -> tuple[np.ndarray, float]: ...


class IKError(RuntimeError):
    """No solution within tolerance; the message says how far off it got."""


@dataclass(frozen=True)
class IKSolution:
    joints_deg: npt.NDArray[np.float64]
    tip_m: npt.NDArray[np.float64]
    pitch_deg: float
    iterations: int


#: Joints the solver moves, by index in the Metal joint order.
DEFAULT_IK_JOINTS = (0, 1, 2, 3)


def solve_tip(
    kinematics: PoseKinematics,
    start_deg: Sequence[float],
    target_m: Sequence[float],
    pitch_deg: float,
    *,
    limits: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    joints: Sequence[int] = DEFAULT_IK_JOINTS,
    tol_m: float = 0.002,
    tol_pitch_deg: float = 2.0,
    max_iterations: int = 200,
    damping: float = 0.02,
    pitch_weight: float = 0.3,
) -> IKSolution:
    """Find joints near `start_deg` that put the tool tip at `target_m` with `pitch_deg`."""
    q = np.asarray(start_deg, dtype=np.float64).copy()
    active = np.asarray(joints, dtype=int)
    target = np.asarray(target_m, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise IKError(f"target must be a finite xyz triple, got {target_m!r}")
    lo, hi = limits if limits is not None else (None, None)

    def residual(qq: np.ndarray) -> np.ndarray:
        tip, pitch = kinematics.tool_pose(qq)
        return np.array(
            [*(target - tip), pitch_weight * math.radians(pitch_deg - pitch)], dtype=np.float64
        )

    err = residual(q)
    for iteration in range(1, max_iterations + 1):
        pos_err = float(np.linalg.norm(err[:3]))
        pitch_err = abs(math.degrees(err[3] / pitch_weight))
        if pos_err <= tol_m and pitch_err <= tol_pitch_deg:
            tip, pitch = kinematics.tool_pose(q)
            return IKSolution(q, np.asarray(tip), float(pitch), iteration - 1)
        jac = np.zeros((4, active.size))
        for column, index in enumerate(active):
            probe = q.copy()
            probe[index] += 0.5  # degrees
            jac[:, column] = (residual(probe) - err) / math.radians(0.5)
        # jac is d(residual)/dq = -d(pose)/dq, hence the sign.
        gain = -jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(4), err)
        step = np.degrees(gain)
        largest = float(np.max(np.abs(step)))
        if largest > 10.0:  # keep the linearisation honest
            step *= 10.0 / largest
        q[active] += step
        if lo is not None and hi is not None:
            q[active] = np.clip(q[active], lo[active], hi[active])
        err = residual(q)
    pos_err = float(np.linalg.norm(err[:3]))
    raise IKError(
        f"no pose within {tol_m * 1000:.0f} mm / {tol_pitch_deg:.0f} deg of the target after "
        f"{max_iterations} iterations (best: {pos_err * 1000:.0f} mm off); the point is "
        "probably out of reach at that pitch"
    )


def offset_target(
    kinematics: PoseKinematics,
    joints_deg: Sequence[float],
    forward_m: float = 0.0,
    left_m: float = 0.0,
    up_m: float = 0.0,
) -> tuple[npt.NDArray[np.float64], float]:
    """The current tip moved by (forward, left, up) in the arm's horizontal
    frame — forward is the tool's pointing direction projected onto the
    table, left is 90° anticlockwise from it — plus the current pitch."""
    tip, pitch = kinematics.tool_pose(joints_deg)
    q = np.radians(np.asarray(joints_deg, dtype=np.float64))
    heading = math.atan2(math.sin(q[0]), math.cos(q[0]))  # shoulder_pan sets the plane
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    left = np.array([-math.sin(heading), math.cos(heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.asarray(tip) + forward_m * forward + left_m * left + up_m * up, float(pitch)
