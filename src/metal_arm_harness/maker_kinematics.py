"""Provisional Maker CAD kinematics; not a commissioned real-arm safety model.

The SDK's Y-up CAD is rotated +90 degrees about X. Motor degrees currently map
one-to-one to CAD arm radians, explicitly UNVERIFIED on hardware. Jaw drive
angle is not linearly converted to travel: the full measured jaw travel is
included in the collision bounds, independent of gripper motor angle.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from importlib.resources import as_file, files

import numpy as np

from metal_arm_harness.arms.base import Kinematics


class MakerKinematics(Kinematics):
    hardware_verified = False
    JOINT_NAMES = tuple(f"link_{i:03d}_joint" for i in range(2, 8))
    JAW_TRAVEL_M = 0.0524125

    def __init__(self):
        import pinocchio as pin

        self._pin = pin
        assets = files("metal_arm_harness").joinpath("assets/urdf")
        with as_file(assets.joinpath("maker_arm.urdf")) as path:
            self._model = pin.buildModelFromUrdf(str(path))
        self._data = self._model.createData()
        self._joint_ids = [self._model.getJointId(n) for n in self.JOINT_NAMES]
        if any(i == 0 or i >= self._model.njoints for i in self._joint_ids):
            raise ValueError("Maker URDF does not contain the expected six arm joints")
        self._tip_id = self._model.getFrameId("grasp_center")
        self._wrist_id = self._model.getFrameId("link_007")
        self._up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        bounds = json.loads(assets.joinpath("maker_bounds.json").read_text())
        self._bounds = []
        for name, (low, high) in bounds.items():
            # Base and pan housing intentionally excluded; they sit on the mount.
            if name in ("base_link", "link_002"):
                continue
            lo, hi = np.array(low), np.array(high)
            # Enclose BOTH jaw endpoints, so roll cannot move an unmodelled open
            # jaw under the floor. No uncalibrated motor-angle/gap interpolation.
            if name == "gripper_left":
                hi[1] += self.JAW_TRAVEL_M
            elif name == "gripper_right":
                lo[1] -= self.JAW_TRAVEL_M
            corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))))
            self._bounds.append((self._model.getFrameId(name), corners))

    def _fk(self, joints_deg: Sequence[float]) -> None:
        angles = np.asarray(joints_deg, dtype=float)
        if angles.shape != (7,) or not np.all(np.isfinite(angles)):
            raise ValueError("Maker FK expects seven finite motor angles in degrees")
        q = self._pin.neutral(self._model)
        for index, joint_id in enumerate(self._joint_ids):
            q[self._model.joints[joint_id].idx_q] = math.radians(angles[index])
        # Jaw coordinates stay zero: bounds already cover their entire travel.
        self._pin.framesForwardKinematics(self._model, self._data, q)

    def tool_pose(self, joints_deg: Sequence[float]) -> tuple[np.ndarray, float]:
        self._fk(joints_deg)
        tip = self._up @ self._data.oMf[self._tip_id].translation
        direction = self._up @ (-self._data.oMf[self._wrist_id].rotation[:, 2])
        pitch = math.degrees(math.atan2(direction[2], math.hypot(*direction[:2])))
        return np.asarray(tip), pitch

    def tool_heading_rad(self, joints_deg: Sequence[float]) -> float:
        self._fk(joints_deg)
        direction = self._up @ (-self._data.oMf[self._wrist_id].rotation[:, 2])
        if math.hypot(*direction[:2]) < 1e-6:
            raise ValueError(
                "tool is vertical; use absolute tip coordinates instead of forward/left"
            )
        return math.atan2(direction[1], direction[0])

    def tool_tip_z_m(self, joints_deg: Sequence[float]) -> float:
        return float(self.tool_pose(joints_deg)[0][2])

    def min_height_m(self, joints_deg: Sequence[float]) -> float:
        self._fk(joints_deg)
        lowest = float((self._up @ self._data.oMf[self._tip_id].translation)[2])
        for frame_id, corners in self._bounds:
            placement = self._data.oMf[frame_id]
            world = (corners @ placement.rotation.T + placement.translation) @ self._up.T
            lowest = min(lowest, float(world[:, 2].min()))
        return lowest
