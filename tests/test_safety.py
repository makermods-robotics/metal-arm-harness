"""Every envelope rule, exercised without hardware or Pinocchio."""

from __future__ import annotations

import numpy as np
import pytest

from metal_arm_harness.arms.base import ArmInfo, ArmState, JointSpec, Kinematics
from metal_arm_harness.safety import MoveRejected, SafetyAbort, SafetyConfig, SafetyEnvelope

HZ = 25.0
FULL_STEP = 20.0 / HZ  # 0.8 deg
SLOW_STEP = 5.0 / HZ  # 0.2 deg


class LinearKinematics(Kinematics):
    """Toy FK: height in metres = joint0 degrees / 1000. 76.2 deg = 3 inches."""

    def min_height_m(self, joints_deg) -> float:
        return float(np.asarray(joints_deg, dtype=np.float64)[0]) / 1000.0

    def tool_tip_z_m(self, joints_deg) -> float:
        return self.min_height_m(joints_deg)


def _info() -> ArmInfo:
    return ArmInfo(
        name="toy",
        joints=(JointSpec("lift", -200.0, 200.0), JointSpec("other", -100.0, 100.0)),
        gripper_index=None,
        control_hz=HZ,
        notes="toy arm",
    )


def _envelope(with_kin: bool = True, floor: float | None = 0.0) -> SafetyEnvelope:
    envelope = SafetyEnvelope(SafetyConfig(), _info(), LinearKinematics() if with_kin else None)
    if with_kin and floor is not None:
        envelope.set_floor(floor)
    return envelope


def test_free_space_moves_step_at_full_speed() -> None:
    plan = _envelope().plan_move([150.0, 0.0], [160.0, 0.0])
    jumps = np.diff(np.vstack([[150.0, 0.0], plan]), axis=0)
    assert np.max(np.abs(jumps)) == pytest.approx(FULL_STEP)
    assert plan[-1][0] == pytest.approx(160.0)


def test_slow_zone_steps_at_slow_speed() -> None:
    # Entirely inside the 76.2 mm zone (heights 40..50 mm).
    plan = _envelope().plan_move([40.0, 0.0], [50.0, 0.0])
    jumps = np.diff(np.vstack([[40.0, 0.0], plan]), axis=0)
    assert np.max(np.abs(jumps)) == pytest.approx(SLOW_STEP)


def test_descent_switches_from_full_to_slow_at_the_zone_boundary() -> None:
    plan = _envelope().plan_move([150.0, 0.0], [120.0, 0.0])
    jumps = np.abs(np.diff(np.vstack([[150.0, 0.0], plan]), axis=0)[:, 0])
    assert jumps[0] == pytest.approx(FULL_STEP)  # starts high, far from zone
    plan2 = _envelope().plan_move([80.0, 0.0], [70.0, 0.0])
    jumps2 = np.abs(np.diff(np.vstack([[80.0, 0.0], plan2]), axis=0)[:, 0])
    assert jumps2[-1] == pytest.approx(SLOW_STEP)  # ends inside the zone


def test_leaving_the_zone_is_also_slow() -> None:
    plan = _envelope().plan_move([40.0, 0.0], [45.0, 0.0])
    assert np.max(np.abs(np.diff(plan[:, 0]))) <= SLOW_STEP + 1e-12


def test_hard_floor_rejects_the_whole_move() -> None:
    envelope = _envelope()
    with pytest.raises(MoveRejected, match="hard floor"):
        envelope.plan_move([40.0, 0.0], [5.0, 0.0])  # 5 mm < 10 mm margin


def test_floor_rejection_happens_before_any_waypoint_is_returned() -> None:
    envelope = _envelope()
    try:
        envelope.plan_move([40.0, 0.0], [5.0, 0.0])
    except MoveRejected as rejection:
        assert "aim higher" in str(rejection)
    else:  # pragma: no cover
        pytest.fail("expected MoveRejected")


def test_without_kinematics_there_is_no_zone_or_floor() -> None:
    envelope = _envelope(with_kin=False)
    plan = envelope.plan_move([40.0, 0.0], [30.0, 0.0])
    assert np.max(np.abs(np.diff(np.vstack([[40.0, 0.0], plan]), axis=0))) == pytest.approx(
        FULL_STEP
    )
    assert envelope.clearance_m([40.0, 0.0]) is None


def test_excursion_cap_rejects_long_moves() -> None:
    with pytest.raises(MoveRejected, match="per-move limit"):
        _envelope().plan_move([150.0, 0.0], [150.0, 40.0])


def test_targets_clamp_to_inset_limits() -> None:
    envelope = _envelope(with_kin=False)
    plan = envelope.plan_move([80.0, 90.0], [80.0, 130.0])
    assert plan[-1][1] == pytest.approx(100.0 - 0.5)


def test_non_finite_target_is_rejected() -> None:
    with pytest.raises(MoveRejected, match="non-finite"):
        _envelope().plan_move([100.0, 0.0], [float("nan"), 0.0])


def test_runtime_vetoes_overheating_and_bad_feedback() -> None:
    envelope = _envelope(with_kin=False)
    ok = ArmState(
        positions_deg=np.zeros(2), velocities_deg_s=np.zeros(2),
        temperatures_c=np.array([40.0, 40.0]),
    )
    envelope.check_runtime(ok)
    with pytest.raises(SafetyAbort, match="over the 70C limit"):
        envelope.check_runtime(
            ArmState(np.zeros(2), np.zeros(2), np.array([40.0, 88.0]))
        )
    with pytest.raises(SafetyAbort, match="non-finite"):
        envelope.check_runtime(
            ArmState(np.array([np.nan, 0.0]), np.zeros(2), np.array([40.0, 40.0]))
        )


def test_bad_configs_are_refused() -> None:
    with pytest.raises(ValueError, match="slow_speed_deg_s must not exceed"):
        SafetyEnvelope(
            SafetyConfig(max_speed_deg_s=5.0, slow_speed_deg_s=10.0), _info(), None
        )
    with pytest.raises(ValueError, match="floor_z_m"):
        _envelope().set_floor(float("inf"))


def test_describe_mentions_the_guard_only_when_calibrated() -> None:
    assert "table" not in _envelope(with_kin=True, floor=None).describe().lower()
    assert "hard floor" in _envelope().describe()
