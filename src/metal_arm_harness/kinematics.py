"""Metal arm forward kinematics, on the vendored URDF, via Pinocchio.

The URDF is the canonical 6-revolute (nq=6) model with the gripper's mass
lumped into Link6 (see ``assets/urdf/PROVENANCE.md``). The mapping from motor
degrees to URDF q is the identity (degrees → radians, JOINT1..JOINT6 order,
no offsets) — the same mapping LeRobot's gravity-compensated leader proves on
hardware.

Monitored points for ``min_height_m`` are the elbow-to-wrist joint origins
plus a tool tip extended along the wrist-roll axis (JOINT6's local x is the
gripper's pointing direction). The default tool length is a deliberate
overestimate: too long starts the slow zone early, too short lets the real
gripper undercut it.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import as_file, files

import numpy as np

from metal_arm_harness.arms.base import Kinematics

#: Conservative default gripper length from the wrist-roll origin, in metres.
DEFAULT_TOOL_LENGTH_M = 0.12

#: First monitored pinocchio joint index (1-based; 3 = the elbow joint).
_FIRST_MONITORED_JOINT = 3


class MetalKinematics(Kinematics):
    """Heights of the Metal arm's distal points, from the vendored URDF."""

    def __init__(self, urdf_path: str | None = None, tool_length_m: float = DEFAULT_TOOL_LENGTH_M):
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - exercised only without pinocchio
            raise RuntimeError(
                "forward kinematics needs Pinocchio.\nfix: uv pip install pin"
            ) from exc
        if not np.isfinite(tool_length_m) or tool_length_m < 0:
            raise ValueError(f"tool_length_m must be finite and >= 0, got {tool_length_m!r}")

        self._pin = pin
        if urdf_path is None:
            resource = files("metal_arm_harness").joinpath(
                "assets/urdf/metal_with_gripper.urdf"
            )
            with as_file(resource) as path:
                self._model = pin.buildModelFromUrdf(str(path))
        else:
            self._model = pin.buildModelFromUrdf(urdf_path)
        if self._model.nq != 6:
            raise ValueError(
                f"expected the 6-revolute Metal URDF (nq=6), got nq={self._model.nq}; "
                "the metal_description variant with prismatic jaws (nq=8) is not this model"
            )
        self._data = self._model.createData()
        self._tool = np.array([tool_length_m, 0.0, 0.0], dtype=np.float64)

    def _fk(self, joints_deg: Sequence[float]) -> None:
        q = np.radians(np.asarray(joints_deg, dtype=np.float64)[:6])
        self._pin.forwardKinematics(self._model, self._data, q)

    def min_height_m(self, joints_deg: Sequence[float]) -> float:
        """Lowest z over elbow..wrist origins and the tool tip, base frame."""
        self._fk(joints_deg)
        heights = [
            float(self._data.oMi[i].translation[2])
            for i in range(_FIRST_MONITORED_JOINT, self._model.njoints)
        ]
        heights.append(self._tool_tip_z())
        return min(heights)

    def tool_tip_z_m(self, joints_deg: Sequence[float]) -> float:
        """Tool tip height in the base frame, for the table-touch ritual."""
        self._fk(joints_deg)
        return self._tool_tip_z()

    def _tool_tip_z(self) -> float:
        last = self._model.njoints - 1
        return float(self._data.oMi[last].act(self._tool)[2])
