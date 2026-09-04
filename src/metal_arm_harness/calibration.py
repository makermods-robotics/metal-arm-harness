"""The table-touch ritual: measure the table, don't configure it.

Before the harness allows any motion, the operator physically places the
gripper tip on the top of the table. The tool tip's forward-kinematics height
at that pose IS the table surface in the arm's base frame — measured on the
real machine, so it is right even if the arm sits on a pedestal, the table
moved, or the URDF's base offset is imperfect at that pose.

The reading is sampled twice and must agree within a tolerance (a hand still
holding the arm, a wobbling table, or noisy feedback all fail loudly instead
of registering a wrong floor). The result feeds `SafetyEnvelope.set_floor`,
below which no commanded waypoint may ever go.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from metal_arm_harness.arms.base import Arm, Kinematics

#: The two samples must agree within this many metres.
STABILITY_TOLERANCE_M = 0.003

#: Attempts before the ritual gives up and refuses to run.
MAX_ATTEMPTS = 3


class CalibrationError(RuntimeError):
    """The table height could not be measured; the harness must not move."""


@dataclass(frozen=True)
class TableCalibration:
    floor_z_m: float
    measured_at: float  # time.time()

    def to_json(self) -> str:
        return json.dumps({"floor_z_m": self.floor_z_m, "measured_at": self.measured_at})

    @staticmethod
    def from_json(text: str) -> TableCalibration:
        payload = json.loads(text)
        return TableCalibration(
            floor_z_m=float(payload["floor_z_m"]),
            measured_at=float(payload["measured_at"]),
        )


def run_table_ritual(
    arm: Arm,
    kinematics: Kinematics,
    *,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    sample_gap_s: float = 0.3,
) -> TableCalibration:
    """Interactively measure the table surface height from a gripper touch.

    The arm must be connected and NOT executing motion; torque state is not
    touched (a gravity-holding arm keeps holding while the operator guides
    the gripper).
    """
    say(
        "\nTable calibration — this sets the hard floor the arm can never go below.\n"
        "Move the arm so the GRIPPER TIP touches the TOP of the table surface.\n"
        "Keep it resting there while the height is measured."
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ask("Press ENTER when the gripper tip is touching the table... ")
        first = kinematics.tool_tip_z_m(arm.read().positions_deg)
        time.sleep(sample_gap_s)
        second = kinematics.tool_tip_z_m(arm.read().positions_deg)
        if abs(first - second) <= STABILITY_TOLERANCE_M:
            floor = min(first, second)
            say(
                f"Table registered at z={floor:.4f} m in the arm's base frame. "
                "The arm will never be commanded below it."
            )
            return TableCalibration(floor_z_m=floor, measured_at=time.time())
        say(
            f"Reading moved {abs(first - second) * 1000:.1f} mm between samples "
            f"(limit {STABILITY_TOLERANCE_M * 1000:.0f} mm) — the arm is not at rest. "
            f"Attempt {attempt}/{MAX_ATTEMPTS}."
        )
    raise CalibrationError(
        "table height could not be measured stably; not moving without a floor"
    )


def save_calibration(calibration: TableCalibration, path: Path) -> None:
    """Persist a measurement for --reuse-table sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibration.to_json(), encoding="utf-8")


def load_calibration(path: Path) -> TableCalibration:
    """Load a previously saved measurement (explicit opt-in only)."""
    if not path.is_file():
        raise CalibrationError(f"no saved table calibration at {path}")
    return TableCalibration.from_json(path.read_text(encoding="utf-8"))


def watch_for_touch(
    arm: Arm,
    kinematics: Kinematics,
    *,
    say: Callable[[str], None] = print,
    move_m: float = 0.03,
    still_m: float = 0.003,
    still_s: float = 5.0,
    timeout_s: float = 480.0,
    poll_s: float = 0.2,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[str], str]:
    """An `ask` for `run_table_ritual` that needs no keyboard.

    It returns once the operator has moved the gripper (tip height changed by
    `move_m`) and then held it still for `still_s`: the hands-free version of
    "press ENTER when the tip is on the table", for sessions driven from a
    terminal without stdin. Raises CalibrationError on timeout.
    """

    def ask(prompt: str) -> str:
        say(prompt.strip() + f"  (watching: move the gripper, then hold still {still_s:.0f}s)")
        start_z = kinematics.tool_tip_z_m(arm.read().positions_deg)
        started = clock()
        moved = False
        window: list[tuple[float, float]] = []
        while clock() - started < timeout_s:
            z = kinematics.tool_tip_z_m(arm.read().positions_deg)
            now = clock()
            if not moved:
                moved = abs(z - start_z) > move_m
            else:
                window.append((now, z))
                window[:] = [(t, v) for t, v in window if now - t <= still_s]
                spread = max(v for _, v in window) - min(v for _, v in window)
                if now - window[0][0] >= still_s - poll_s and spread <= still_m:
                    return ""
            sleep(poll_s)
        raise CalibrationError("timed out waiting for the gripper to be placed and held still")

    return ask
