"""The table-touch ritual: stable readings register, wobbly ones refuse."""

from __future__ import annotations

import numpy as np
import pytest

from metal_arm_harness.arms.base import ArmState, Kinematics
from metal_arm_harness.calibration import (
    CalibrationError,
    load_calibration,
    run_table_ritual,
    save_calibration,
)


class ScriptedArm:
    """read() walks through a scripted list of joint vectors."""

    def __init__(self, positions: list[list[float]]):
        self._positions = list(positions)

    def read(self) -> ArmState:
        joints = self._positions.pop(0) if len(self._positions) > 1 else self._positions[0]
        return ArmState(
            positions_deg=np.asarray(joints, dtype=np.float64),
            velocities_deg_s=np.zeros(len(joints)),
            temperatures_c=np.full(len(joints), 30.0),
        )


class TipKinematics(Kinematics):
    """Tool tip height = joint0 / 1000 (degrees to metres)."""

    def min_height_m(self, joints_deg) -> float:
        return self.tool_tip_z_m(joints_deg)

    def tool_tip_z_m(self, joints_deg) -> float:
        return float(np.asarray(joints_deg, dtype=np.float64)[0]) / 1000.0


def test_stable_touch_registers_the_lower_sample(capsys) -> None:
    arm = ScriptedArm([[-52.0], [-53.0]])
    calibration = run_table_ritual(
        arm, TipKinematics(), ask=lambda _: "", say=print, sample_gap_s=0.0
    )
    assert calibration.floor_z_m == pytest.approx(-0.053)
    assert "never be commanded below" in capsys.readouterr().out


def test_wobbly_readings_refuse_after_three_attempts() -> None:
    arm = ScriptedArm([[0.0], [50.0], [0.0], [50.0], [0.0], [50.0], [50.0]])
    with pytest.raises(CalibrationError, match="not moving without a floor"):
        run_table_ritual(arm, TipKinematics(), ask=lambda _: "", say=lambda _: None,
                         sample_gap_s=0.0)


def test_round_trip_persistence(tmp_path) -> None:
    arm = ScriptedArm([[10.0], [10.0]])
    calibration = run_table_ritual(
        arm, TipKinematics(), ask=lambda _: "", say=lambda _: None, sample_gap_s=0.0
    )
    path = tmp_path / "table.json"
    save_calibration(calibration, path)
    loaded = load_calibration(path)
    assert loaded.floor_z_m == pytest.approx(calibration.floor_z_m)


def test_missing_saved_calibration_is_loud(tmp_path) -> None:
    with pytest.raises(CalibrationError, match="no saved table calibration"):
        load_calibration(tmp_path / "nope.json")
