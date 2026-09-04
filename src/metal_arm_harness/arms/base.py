"""The arm-agnostic contract every arm implements.

The harness (safety envelope, executor, agent, CLI) is written against these
types only. Adding a new arm means implementing `Arm` (and optionally
`Kinematics`, without which the table guard is unavailable) and registering a
factory in `arms/__init__.py` — nothing above this layer changes.

Conventions the contract fixes, so the agent prompt can state them once:
- joint positions and targets are absolute angles in DEGREES,
- joints are ordered, named, and bounded by `ArmInfo.joints`,
- `read()` is always safe; `send()` requires the harness to have armed torque.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class JointSpec:
    """One joint: its model-facing name and soft limits in degrees."""

    name: str
    lo: float
    hi: float


@dataclass(frozen=True)
class ArmInfo:
    """Everything the harness and the LLM need to know about an arm."""

    name: str
    joints: tuple[JointSpec, ...]
    #: Index of the gripper joint in `joints`, or None for gripperless arms.
    gripper_index: int | None
    #: Rate the executor plays waypoints at, in Hz.
    control_hz: float
    #: Model-facing operating notes: joint roles, gripper convention, zero
    #: pose — the facts, stated once, that make zero-shot control possible.
    notes: str

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)


@dataclass(frozen=True)
class ArmState:
    """One proprioception sample, in joint order."""

    positions_deg: npt.NDArray[np.float64]
    velocities_deg_s: npt.NDArray[np.float64]
    temperatures_c: npt.NDArray[np.float64]


@dataclass
class Observation:
    """What the agent sees after every tool call."""

    state: ArmState
    frames: Mapping[str, npt.NDArray[np.uint8]] = field(default_factory=dict)


class Arm(abc.ABC):
    """A connected arm. Torque is opt-in and gated by the harness."""

    info: ArmInfo

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the bus and cameras for reads. Never touches torque state."""

    @abc.abstractmethod
    def enable_torque(self) -> None:
        """Energize the motors. The harness calls this only after its gates."""

    @abc.abstractmethod
    def read(self) -> ArmState:
        """Current joint state, in degrees."""

    @abc.abstractmethod
    def frames(self) -> dict[str, npt.NDArray[np.uint8]]:
        """One RGB frame per camera, keyed by camera name."""

    @abc.abstractmethod
    def send(self, targets_deg: Sequence[float]) -> None:
        """Send one absolute joint target vector. Requires enabled torque."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the bus and cameras. Must not drop a holding arm."""

    def observe(self) -> Observation:
        """State plus camera frames, the agent's per-turn input."""
        return Observation(state=self.read(), frames=self.frames())


class Kinematics(abc.ABC):
    """Forward kinematics an arm may provide to enable the table guard."""

    @abc.abstractmethod
    def min_height_m(self, joints_deg: Sequence[float]) -> float:
        """Lowest monitored point of the arm (metres, base frame)."""

    @abc.abstractmethod
    def tool_tip_z_m(self, joints_deg: Sequence[float]) -> float:
        """Height of the tool tip (metres, base frame), for the table ritual."""
