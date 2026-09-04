"""The safety envelope: every rule between the LLM's request and the motors.

Pure logic — no bus, no sleep — so every rule is unit-testable without
hardware. The executor plays what `plan_move` returns and calls
`check_runtime` between waypoints; nothing else in the harness is allowed to
produce motor targets.

The rules, in the order a move meets them:

1. **Joint soft limits** — targets are clamped to the arm's limits, inset by
   a margin.
2. **Per-move excursion cap** — a single tool call may move no joint more
   than `max_excursion_deg`. Speed limits bound deg/s; this bounds distance
   travelled between looks at the camera, which open-loop execution
   otherwise leaves unbounded. Violations reject the whole move.
3. **The hard floor** — with kinematics and a calibrated table height, any
   waypoint whose lowest monitored point would come within `floor_margin_m`
   of the table rejects the whole move before one frame reaches the bus.
   This is what stops the arm pressing down on the table.
4. **The slow zone** — waypoints within `slow_zone_m` (default 3 inches) of
   the table are stepped at `slow_speed_deg_s`; everywhere else the full
   `max_speed_deg_s` applies. Both ends of each step are zone-tested, so
   there is no fast pose from which the arm can dive at the table and no
   fast pull-out that starts low.
5. **Runtime vetoes** — non-finite feedback and over-temperature motors
   abort the episode between waypoints.

A rejection is a `MoveRejected` carrying an explanation written for the LLM:
it comes back as a correctable tool error, not a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metal_arm_harness.arms.base import ArmInfo, ArmState, Kinematics

#: Hard cap on waypoints per move; at slow speed this is minutes of motion,
#: far past anything a sane move needs, so hitting it means a logic error.
_MAX_WAYPOINTS = 100_000


class MoveRejected(Exception):
    """The move violates the envelope; the message is written for the LLM."""


class SafetyAbort(Exception):
    """The arm is not healthy enough to keep running; ends the episode."""


@dataclass(frozen=True)
class SafetyConfig:
    """Every number the envelope enforces, in one reviewable place."""

    max_speed_deg_s: float = 20.0
    slow_speed_deg_s: float = 5.0
    #: Height band above the table in which the slow speed applies (3 in).
    slow_zone_m: float = 0.0762
    #: Clearance below which a waypoint is rejected outright (FK error pad).
    floor_margin_m: float = 0.010
    max_excursion_deg: float = 35.0
    max_temp_c: float = 70.0
    limit_margin_deg: float = 0.5

    def validate(self) -> None:
        if not np.isfinite(self.max_speed_deg_s) or self.max_speed_deg_s <= 0:
            raise ValueError(f"max_speed_deg_s must be > 0, got {self.max_speed_deg_s!r}")
        if not np.isfinite(self.slow_speed_deg_s) or self.slow_speed_deg_s <= 0:
            raise ValueError(f"slow_speed_deg_s must be > 0, got {self.slow_speed_deg_s!r}")
        if self.slow_speed_deg_s > self.max_speed_deg_s:
            raise ValueError("slow_speed_deg_s must not exceed max_speed_deg_s")
        if not np.isfinite(self.slow_zone_m) or self.slow_zone_m <= 0:
            raise ValueError(f"slow_zone_m must be > 0, got {self.slow_zone_m!r}")
        if not np.isfinite(self.floor_margin_m) or self.floor_margin_m < 0:
            raise ValueError(f"floor_margin_m must be >= 0, got {self.floor_margin_m!r}")
        if not np.isfinite(self.max_excursion_deg) or self.max_excursion_deg <= 0:
            raise ValueError(f"max_excursion_deg must be > 0, got {self.max_excursion_deg!r}")


class SafetyEnvelope:
    """Owns the rules; produces waypoint plans and runtime vetoes."""

    def __init__(self, config: SafetyConfig, info: ArmInfo, kinematics: Kinematics | None):
        config.validate()
        self.config = config
        self.info = info
        self.kinematics = kinematics
        self.floor_z_m: float | None = None
        self._low = np.array(
            [j.lo + config.limit_margin_deg for j in info.joints], dtype=np.float64
        )
        self._high = np.array(
            [j.hi - config.limit_margin_deg for j in info.joints], dtype=np.float64
        )
        if bool(np.any(self._low >= self._high)):
            raise ValueError("limit_margin_deg leaves a joint no travel inside its limits")
        self._full_step = config.max_speed_deg_s / info.control_hz
        self._slow_step = config.slow_speed_deg_s / info.control_hz

    # ── configuration ───────────────────────────────────────────────────────

    def set_floor(self, floor_z_m: float) -> None:
        """Register the table surface height measured by the touch ritual."""
        if not np.isfinite(floor_z_m):
            raise ValueError(f"floor_z_m must be finite, got {floor_z_m!r}")
        self.floor_z_m = float(floor_z_m)

    @property
    def guards_height(self) -> bool:
        """True when kinematics exist and the table has been registered."""
        return self.kinematics is not None and self.floor_z_m is not None

    def clearance_m(self, joints_deg: Sequence[float]) -> float | None:
        """Current height above the table, or None without the guard."""
        if not self.guards_height:
            return None
        assert self.kinematics is not None and self.floor_z_m is not None
        return self.kinematics.min_height_m(joints_deg) - self.floor_z_m

    # ── planning ────────────────────────────────────────────────────────────

    def clamp_to_limits(self, target_deg: Sequence[float]) -> npt.NDArray[np.float64]:
        """Clip a target vector into the inset joint limits."""
        return np.clip(np.asarray(target_deg, dtype=np.float64), self._low, self._high)

    def plan_move(
        self, current_deg: Sequence[float], target_deg: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Turn a requested absolute target into an approved waypoint stream.

        Returns an (N, dof) array of absolute joint targets, one per control
        period, or raises `MoveRejected` with an LLM-readable reason. The
        first row is one step beyond `current_deg`; the last row is the
        (possibly clamped) target.
        """
        current = np.asarray(current_deg, dtype=np.float64)
        if not bool(np.all(np.isfinite(current))):
            raise SafetyAbort("joint feedback is non-finite; check bus wiring and power")
        raw = np.asarray(target_deg, dtype=np.float64)
        if raw.shape != current.shape:
            raise MoveRejected(
                f"target has {raw.shape} entries but the arm has {current.shape}"
            )
        if not bool(np.all(np.isfinite(raw))):
            raise MoveRejected("target contains a non-finite value")
        target = self.clamp_to_limits(raw)

        span = np.abs(target - current)
        over = np.nonzero(span > self.config.max_excursion_deg)[0]
        if over.size:
            worst = int(over[np.argmax(span[over])])
            raise MoveRejected(
                f"{self.info.joint_names[worst]} would move {span[worst]:.1f} degrees in "
                f"one call, over the {self.config.max_excursion_deg:.0f} degree per-move "
                "limit. Break the motion into smaller steps and re-check the cameras "
                "between them."
            )

        waypoints: list[npt.NDArray[np.float64]] = []
        q = current
        for _ in range(_MAX_WAYPOINTS):
            if bool(np.all(q == target)):
                break
            step = self._step_size(q, target)
            q = q + np.clip(target - q, -step, step)
            self._check_floor(q)
            waypoints.append(q)
        else:  # pragma: no cover - _MAX_WAYPOINTS is unreachable by design
            raise MoveRejected("move produced an unreasonable number of waypoints")
        return np.asarray(waypoints, dtype=np.float64).reshape(-1, current.size)

    def _step_size(self, q: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> float:
        """Full step in free space; slow step when either end is near the table."""
        if not self.guards_height:
            return self._full_step
        candidate = q + np.clip(target - q, -self._full_step, self._full_step)
        if self._in_slow_zone(q) or self._in_slow_zone(candidate):
            return self._slow_step
        return self._full_step

    def _in_slow_zone(self, joints_deg: npt.NDArray[np.float64]) -> bool:
        clearance = self.clearance_m(joints_deg)
        return clearance is not None and clearance < self.config.slow_zone_m

    def _check_floor(self, joints_deg: npt.NDArray[np.float64]) -> None:
        clearance = self.clearance_m(joints_deg)
        if clearance is not None and clearance < self.config.floor_margin_m:
            raise MoveRejected(
                f"that move would bring the arm to {clearance * 1000:.0f} mm above the "
                f"table, under the {self.config.floor_margin_m * 1000:.0f} mm hard floor. "
                "The table surface is a hard limit; aim higher and approach objects "
                "from above."
            )

    # ── runtime ─────────────────────────────────────────────────────────────

    def check_runtime(self, state: ArmState) -> None:
        """Veto continuing on unhealthy feedback, between waypoints."""
        if not bool(np.all(np.isfinite(state.positions_deg))):
            raise SafetyAbort("joint feedback went non-finite mid-move")
        temps = np.asarray(state.temperatures_c, dtype=np.float64)
        hot = np.nonzero(np.isfinite(temps) & (temps > self.config.max_temp_c))[0]
        if hot.size:
            worst = int(hot[np.argmax(temps[hot])])
            raise SafetyAbort(
                f"{self.info.joint_names[worst]} is at {temps[worst]:.1f}C, over the "
                f"{self.config.max_temp_c:.0f}C limit"
            )

    # ── prompt material ─────────────────────────────────────────────────────

    def describe(self) -> str:
        """The envelope stated for the system prompt, in the model's terms."""
        lines = [
            f"Speed is limited to {self.config.max_speed_deg_s:.0f} deg/s; moves are "
            "interpolated, so a large change takes proportionally longer.",
            f"No single move may change a joint by more than "
            f"{self.config.max_excursion_deg:.0f} degrees; bigger motions must be split "
            "across calls, with a look at the camera between them.",
        ]
        if self.guards_height:
            lines.append(
                f"A table guard is active. Within {self.config.slow_zone_m:.3f} m of the "
                f"table, speed drops to {self.config.slow_speed_deg_s:.0f} deg/s. Moves "
                "that would touch or pass the table surface are rejected outright — the "
                "table is a hard floor. Each observation reports your current "
                "clearance_m above it."
            )
        return "\n".join(f"- {line}" for line in lines)
