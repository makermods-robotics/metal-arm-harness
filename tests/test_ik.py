"""IK on the vendored URDF: fast, accurate, and honest about reach."""

from __future__ import annotations

import time

import numpy as np
import pytest

from metal_arm_harness.ik import IKError, offset_target, solve_tip
from metal_arm_harness.kinematics import MetalKinematics

pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def kin() -> MetalKinematics:
    return MetalKinematics()


def test_solve_reaches_a_grasp_pose_from_a_hover(kin: MetalKinematics) -> None:
    start = [-15.8, -105.2, 62.9, -31.6, 0.2, 0.0, 83.8]
    began = time.perf_counter()
    solution = solve_tip(kin, start, [0.335, -0.095, 0.040], -80.0)
    assert time.perf_counter() - began < 0.5
    assert np.linalg.norm(solution.tip_m - [0.335, -0.095, 0.040]) <= 0.002
    assert abs(solution.pitch_deg + 80.0) <= 2.0
    # Held joints are untouched; the arm stays in the same configuration family.
    assert solution.joints_deg[4:] == pytest.approx(start[4:])
    assert abs(solution.joints_deg[1] - start[1]) < 20.0


def test_offset_target_moves_forward_along_the_pan_heading(kin: MetalKinematics) -> None:
    q = [30.0, -100.0, 60.0, -30.0, 0.0, 0.0, 50.0]
    tip, pitch = kin.tool_pose(q)
    target, target_pitch = offset_target(kin, q, forward_m=0.02)
    delta = target - tip
    assert target_pitch == pytest.approx(pitch)
    assert delta[2] == pytest.approx(0.0)
    assert np.hypot(*delta[:2]) == pytest.approx(0.02)
    assert np.degrees(np.arctan2(delta[1], delta[0])) == pytest.approx(30.0, abs=1e-6)


def test_unreachable_target_raises_instead_of_guessing(kin: MetalKinematics) -> None:
    with pytest.raises(IKError, match="out of reach"):
        solve_tip(kin, [0.0] * 7, [0.9, 0.0, 0.1], -80.0)


def test_solution_respects_joint_limits_when_given(kin: MetalKinematics) -> None:
    low = np.array([-160.0, -180.0, 0.0, -123.0, -85.0, -145.0, 0.0])
    high = np.array([160.0, 0.0, 180.0, 81.0, 85.0, 145.0, 137.0])
    solution = solve_tip(kin, [0.0] * 7, [0.22, -0.02, 0.12], -50.0, limits=(low, high))
    assert np.all(solution.joints_deg >= low - 1e-9) and np.all(solution.joints_deg <= high + 1e-9)


def test_pitch_is_honoured_from_an_extended_pose(kin: MetalKinematics) -> None:
    # Seen on the bench: from a stretched pose a low pitch weight converged on
    # position and left the pitch 6 degrees off, which the tolerance then rejected.
    start = [-1.8, -149.1, 151.8, -79.3, 0.0, 81.7, 69.4]
    solution = solve_tip(kin, start, [0.43, -0.065, 0.12], -78.0)
    assert abs(solution.pitch_deg + 78.0) <= 2.0
    assert np.linalg.norm(solution.tip_m - [0.43, -0.065, 0.12]) <= 0.002
