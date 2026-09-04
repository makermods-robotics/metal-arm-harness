"""One motion primitive for every caller: `Controller.goto`.

The operator session, any future policy, and the tests all move the arm
through this class, so the bench lessons live in one place:

- **Commanded base.** Joints the caller does not mention hold at their last
  *commanded* value, not their measured one. Otherwise gravity sag (about a
  degree at the shoulder under P control) is re-planned into the goal on
  every call and the arm creeps toward the table.
- **Chunking.** An operator may ask for a 130° swing; it is played as
  consecutive envelope-approved legs no longer than the per-move excursion
  cap, each settled before the next; `chunk=False` keeps the cap a hard rule.
- **Settle with integral action.** After the last waypoint the target is
  held with `goal + lead`, `lead` integrating the residual (bounded,
  re-planned through the envelope every tick), which cancels the
  steady-state gravity error: 0.1-0.3° instead of 1.3°.
- **No dip.** Each new waypoint stream starts from the last command that
  was actually sent (the held `goal + lead`), not from the measured pose,
  and the settle inherits the lag the stream ended with. Otherwise every
  move began by dropping the lead the shoulder needed against gravity and
  the arm visibly sank for a few ticks (seen on the bench).
- **Stalled gripper = holding, and it stays that way.** Jaws that stop
  short of the target while closing are reported as a grasp and excluded
  from convergence checks, but they keep being commanded to their goal on
  every later move — that standing error IS the grip force (an earlier
  version relaxed it after the stall and dropped four grasps in a row).
- **Grip verification.** After a grasp the reply carries the gripper's
  measured torque; a stall with almost no torque means the jaws are
  resting on the object, not around it. During later moves the jaw angle
  is watched: closing well past the contact angle means the object
  slipped, reported as OBJECT LOST instead of being discovered later.

Everything still goes through `SafetyEnvelope.plan_move`; this class never
produces a motor target the envelope has not approved.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from metal_arm_harness import executor
from metal_arm_harness.arms.base import Arm, Observation
from metal_arm_harness.episode_log import EpisodeLog
from metal_arm_harness.safety import MoveRejected, SafetyEnvelope

#: Commanded base is discarded when the arm is this far from it (moved by hand).
STALE_BASE_DEG = 6.0
#: Largest single correction command, on top of the goal.
MAX_CORRECTION_DEG = 2.0
#: Remaining arm-joint error below which no further leg is played (the settle covers it).
LEG_DONE_DEG = 3.0
#: The last command is used as the plan origin only while it is this close to measured.
ORIGIN_MAX_LAG_DEG = 4.0
#: Below this measured gripper torque a "stalled" gripper is not really squeezing.
GRIP_TORQUE_MIN_NM = 0.4
#: Jaws closing this much further than the contact angle during a move = object slipping.
GRIP_SLIP_DEG = 5.0


@dataclass(frozen=True)
class GotoReport:
    """What one `goto` did, for tool results and logs."""

    goal_deg: tuple[float, ...]
    measured_deg: tuple[float, ...]
    legs: int
    steps: int
    duration_s: float
    converged: bool
    gripper_stalled: bool
    armed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Largest excursion of any arm joint AWAY from its target during play, degrees:
    #: the "dip" an operator sees at the start of a move. ~0 when torque is continuous.
    counter_dip_deg: float = 0.0
    counter_dip_joint: str = ""
    #: Gripper torque after the move (N·m), when the arm reports torque; None otherwise.
    grip_torque_nm: float | None = None
    #: Jaw angle at contact when a grasp was detected, else None.
    contact_deg: float | None = None
    #: True when a held object appears to have slipped: the jaws closed well past contact.
    object_lost: bool = False

    @property
    def residual_deg(self) -> tuple[float, ...]:
        return tuple(abs(m - g) for m, g in zip(self.measured_deg, self.goal_deg, strict=True))

    def summary(self, joint_names: Sequence[str]) -> str:
        worst = max(range(len(self.goal_deg)), key=lambda i: self.residual_deg[i])
        head = "Move played" if self.armed else "UNARMED: move computed and logged, not sent"
        text = (
            f"{head} ({self.legs} leg{'s' if self.legs != 1 else ''}, {self.steps} steps, "
            f"{self.duration_s:.1f}s; residual {self.residual_deg[worst]:.1f} deg on "
            f"{joint_names[worst]})."
        )
        if self.object_lost:
            text += (
                " OBJECT LOST: the jaws closed well past the contact angle during this move; "
                "the object slipped out. Re-approach and grasp lower around its middle."
            )
        elif self.gripper_stalled:
            text += " Gripper stopped short of its target: it is holding something"
            if self.grip_torque_nm is not None:
                if self.grip_torque_nm < GRIP_TORQUE_MIN_NM:
                    text += (
                        f" but grip torque is only {self.grip_torque_nm:.2f} N·m: the jaws are "
                        "probably resting ON the object, not around it. Do not lift; open, "
                        "descend lower, close again."
                    )
                else:
                    text += f" (grip torque {self.grip_torque_nm:.2f} N·m)."
            else:
                text += "."
        if self.counter_dip_deg >= 0.5:
            text += f" Counter-dip {self.counter_dip_deg:.1f} deg on {self.counter_dip_joint}."
        for note in self.notes:
            text += f" {note}"
        return text


class Controller:
    """Moves the arm to joint goals through the envelope, with settle and correction."""

    def __init__(
        self,
        arm: Arm,
        safety: SafetyEnvelope,
        *,
        armed: bool,
        log: EpisodeLog | None = None,
        settle_timeout_s: float = 4.0,
        settle_tol_deg: float = 0.5,
    ):
        self.arm = arm
        self.safety = safety
        self.armed = armed
        self.log = log or EpisodeLog(None)
        self.settle_timeout_s = settle_timeout_s
        self.settle_tol_deg = settle_tol_deg
        self.commanded: npt.NDArray[np.float64] | None = None
        #: True while the jaws are known to be stopped on an object.
        self.gripper_stalled = False
        #: The last motor command actually sent (goal + gravity lead); plan origin.
        self.last_command: npt.NDArray[np.float64] | None = None
        #: Jaw angle where the last grasp stalled, while an object is believed held.
        self.contact_deg: float | None = None

    # ── observation ─────────────────────────────────────────────────────────

    def observe(self) -> Observation:
        return self.arm.observe()

    def base(self) -> npt.NDArray[np.float64]:
        """Where unmentioned joints hold: the last commanded pose, unless stale.

        Staleness is judged on the arm joints only. A gripper closed on an
        object sits far from its commanded (closed) goal by design, and that
        gap must survive into the next command or the grip is released.
        """
        measured = np.asarray(self.arm.read().positions_deg, dtype=np.float64)
        if self.commanded is None:
            return measured
        arm_mask = np.ones(measured.shape, dtype=bool)
        if self.arm.info.gripper_index is not None:
            arm_mask[self.arm.info.gripper_index] = False
        if float(np.max(np.abs((self.commanded - measured)[arm_mask]))) <= STALE_BASE_DEG:
            return self.commanded.copy()
        return measured

    # ── motion ──────────────────────────────────────────────────────────────

    def goto(
        self,
        targets: Mapping[str, float] | Sequence[float],
        *,
        note: str = "",
        chunk: bool = False,
    ) -> GotoReport:
        """Move to absolute joint targets (a name→deg mapping or a full vector).

        `chunk=True` lets the move exceed the per-move excursion cap by playing
        it as consecutive approved legs; with `chunk=False` the cap is a hard
        rule and the envelope rejects anything larger. Raises `MoveRejected` with
        nothing played when the first leg is not allowed; a rejection on a
        later leg is reported in `notes` with the arm settled where it got to.
        """
        names = self.arm.info.joint_names
        goal = self.base()
        if isinstance(targets, Mapping):
            for joint, value in targets.items():
                goal[names.index(joint)] = float(value)
        else:
            goal = np.asarray(targets, dtype=np.float64).copy()
        goal = self.safety.clamp_to_limits(goal)
        cap = self.safety.config.max_excursion_deg - 1.0 if chunk else None
        # A request that names only the gripper must plan as a pure jaw move: arm
        # joints are left at their MEASURED values for the leg (so the envelope sees no
        # geometry change and runs it at full speed even near the table); the correction
        # passes afterwards still pull the arm back to its commanded pose.
        gripper_index = self.arm.info.gripper_index
        jaw_only = (
            isinstance(targets, Mapping)
            and gripper_index is not None
            and set(targets) == {names[gripper_index]}
        )

        gripper = self.arm.info.gripper_index
        arm_mask = np.ones(goal.shape, dtype=bool)
        if gripper is not None:
            arm_mask[gripper] = False
        started = time.monotonic()
        legs = steps = 0
        notes: list[str] = []
        # A stalled gripper is planned as parked (no phantom travel in the stream) and
        # its goal rides along as the squeeze override; a new gripper request resets that.
        gripper_requested = (
            isinstance(targets, Mapping) and gripper is not None and names[gripper] in targets
        )
        stalled = self.gripper_stalled and not gripper_requested
        squeeze = float(goal[gripper]) if (stalled and gripper is not None) else None
        converged = False
        previous: npt.NDArray[np.float64] | None = None
        dip = 0.0
        dip_joint = ""
        for _ in range(64):
            current = np.asarray(self.arm.read().positions_deg, dtype=np.float64)
            delta = goal - current
            if cap is None:
                leg_target = goal.copy()
            else:
                leg_target = current + np.clip(delta, -cap, cap)
                if gripper is not None:
                    leg_target[gripper] = goal[gripper]  # jaws are exempt from the cap
            if jaw_only:
                leg_target[arm_mask] = current[arm_mask]
            if squeeze is not None and gripper is not None:
                leg_target[gripper] = current[gripper]  # parked in the plan, squeezed via override
            origin = self._origin(current, arm_mask)
            try:
                waypoints = self.safety.plan_move(origin, leg_target)
            except MoveRejected:
                if legs == 0:
                    raise
                notes.append("later leg rejected by the envelope; stopped early.")
                break
            report = executor.play(
                self.arm, self.safety, waypoints, armed=self.armed, gripper_override=squeeze
            )
            legs += 1
            steps += report.steps
            leg_dip, leg_dip_joint = _counter_dip(
                report.trace, current, leg_target, arm_mask, names
            )
            if leg_dip > dip:
                dip, dip_joint = leg_dip, leg_dip_joint
            approved = waypoints[-1] if len(waypoints) else origin
            if self.armed:
                self.last_command = np.asarray(
                    executor._with_gripper(self.arm, approved, squeeze), dtype=np.float64
                )
            # Final leg (or a jaw-only one): hold the COMMANDED pose, which the settle
            # re-plans through the envelope every tick; intermediate legs hold their target.
            final_leg = jaw_only or bool(np.all(np.abs((leg_target - goal)[arm_mask]) < 1e-9))
            hold_target = approved.copy()
            if final_leg:
                hold_target[arm_mask] = goal[arm_mask]
            after = np.asarray(self.arm.read().positions_deg, dtype=np.float64)
            settled = executor.settle(
                self.arm,
                self.safety,
                hold_target,
                armed=self.armed,
                timeout_s=self.settle_timeout_s,
                tol_deg=self.settle_tol_deg,
                gripper_override=squeeze,
                initial_lead_deg=(hold_target - after) * arm_mask,  # the lag the stream ended with
                max_lead_deg=MAX_CORRECTION_DEG,
            )
            if settled.last_command is not None:
                self.last_command = np.asarray(settled.last_command, dtype=np.float64)
            if settled.gripper_blocked:
                notes.append("gripper stopped while opening: blocked or faulted, not holding.")
            if settled.gripper_stalled and gripper is not None:
                stalled = True
                squeeze = float(goal[gripper])
                self.contact_deg = float(self.arm.read().positions_deg[gripper])
            measured = np.asarray(self.arm.read().positions_deg, dtype=np.float64)
            arm_done = bool(np.all(np.abs((goal - measured)[arm_mask]) <= LEG_DONE_DEG))
            if cap is None or not self.armed or arm_done or jaw_only:
                converged = settled.converged
                break
            if (
                previous is not None
                and float(np.max(np.abs((measured - previous)[arm_mask]))) < 1.0
            ):
                notes.append("no progress between legs; stopped early.")
                break
            previous = measured

        self.commanded = goal.copy()
        state = self.arm.read()
        measured = state.positions_deg
        grip_torque = (
            abs(float(state.efforts_nm[gripper]))
            if (gripper is not None and len(state.efforts_nm) > gripper)
            else None
        )
        object_lost = False
        if stalled and gripper is not None and self.contact_deg is not None:
            # A held object keeps the jaws at the contact angle; if they closed much further
            # (toward the closed goal) during this move, the object is gone.
            closed_past = (self.contact_deg - float(measured[gripper])) * np.sign(
                self.contact_deg - float(goal[gripper]) or 1.0
            )
            object_lost = closed_past > GRIP_SLIP_DEG
        if object_lost or (gripper_requested and not stalled):
            self.contact_deg = None
        self.gripper_stalled = stalled and not object_lost
        result = GotoReport(
            goal_deg=tuple(float(v) for v in goal),
            measured_deg=tuple(float(v) for v in measured),
            legs=legs,
            steps=steps,
            duration_s=time.monotonic() - started,
            converged=converged,
            gripper_stalled=stalled and not object_lost,
            armed=self.armed,
            notes=tuple(notes),
            counter_dip_deg=dip,
            counter_dip_joint=dip_joint,
            grip_torque_nm=grip_torque,
            contact_deg=self.contact_deg,
            object_lost=object_lost,
        )
        self.log.event(
            "move",
            targets=_as_dict(names, targets),
            note=note,
            armed=self.armed,
            legs=legs,
            steps=steps,
            duration_s=round(result.duration_s, 3),
            goal=result.goal_deg,
            measured=result.measured_deg,
            gripper_stalled=stalled,
            counter_dip_deg=round(dip, 2),
            counter_dip_joint=dip_joint,
            grip_torque_nm=grip_torque,
            object_lost=object_lost,
        )
        return result

    def _origin(
        self, measured: npt.NDArray[np.float64], arm_mask: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]:
        """Where the next waypoint stream starts: the last command actually sent
        (goal + gravity lead), so torque is continuous, unless it is stale."""
        if self.last_command is None or not self.armed:
            return measured
        lag = np.abs((self.last_command - measured)[arm_mask])
        if float(np.max(lag)) > ORIGIN_MAX_LAG_DEG:
            return measured  # moved by hand, or a long time ago: re-anchor on reality
        origin = self.last_command.copy()
        if self.arm.info.gripper_index is not None:
            origin[self.arm.info.gripper_index] = measured[self.arm.info.gripper_index]
        return origin

    def relieve(self, joint: int) -> None:
        """Zero the P-torque on one joint by commanding its measured position while
        the others keep their last command. Used after an over-temperature veto: the
        envelope refuses motion while the motor is hot, but the motor must stop pushing.
        Not a motion, hence outside plan_move, and only ever toward less torque."""
        measured = np.asarray(self.arm.read().positions_deg, dtype=np.float64)
        base = self.last_command.copy() if self.last_command is not None else measured.copy()
        base[joint] = measured[joint]
        self.arm.send([float(v) for v in base])
        self.last_command = base
        self.log.event("relieve", joint=self.arm.info.joint_names[joint])


def _as_dict(
    names: Sequence[str], targets: Mapping[str, float] | Sequence[float]
) -> dict[str, float]:
    if isinstance(targets, Mapping):
        return {k: float(v) for k, v in targets.items()}
    return {n: float(v) for n, v in zip(names, targets, strict=True)}


def _counter_dip(
    trace: Sequence[Sequence[float]],
    start: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    arm_mask: npt.NDArray[np.bool_],
    names: Sequence[str],
) -> tuple[float, str]:
    """Largest excursion of an arm joint away from its target during a leg."""
    if not trace:
        return 0.0, ""
    samples = np.asarray(trace, dtype=np.float64)
    direction = np.sign(target - start)
    away = (start - samples) * direction  # positive = moved opposite to the target
    away[:, direction == 0] = np.abs((samples - start)[:, direction == 0])
    away[:, ~arm_mask] = 0.0  # the jaws are not part of the dip
    worst = int(np.argmax(np.max(away, axis=0)))
    value = float(np.max(away[:, worst]))
    return (value, names[worst]) if value > 0 else (0.0, "")
