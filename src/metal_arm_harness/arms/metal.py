"""The MakerMods Metal arm, over LeRobot's Damiao driver layer.

Low level is deliberately reused, not rewritten: `DamiaoMotorsBus` +
`MetalFollower` are the CAN/MIT stack proven under real teleoperation. This
module adapts them to the harness's `Arm` contract and adds a bench simulator
(`backend="sim"`): a fake bus whose motors instantly track every command, so
the entire harness runs with nothing plugged in.

`MetalFollower.send_action` keeps its own soft-limit clamp and its
`max_relative_target` per-tick cap (set to the harness's full-speed step), so
the driver layer enforces the speed ceiling independently of the safety
envelope above it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from metal_arm_harness.arms.base import Arm, ArmInfo, ArmState, JointSpec

_SIM_TEMP_C = 32.0

#: Where the simulated arm powers on: inside every soft limit, mid-workspace.
SIM_START_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 0.0,
    "wrist_yaw": 0.0,
    "wrist_roll": 0.0,
    "gripper": 10.0,
}

METAL_NOTES = """\
Six revolute joints plus a gripper, all commanded as absolute angles in
DEGREES. The names say what each one does:

  shoulder_pan    base yaw (rotates the whole arm about vertical)
  shoulder_lift   shoulder pitch; range -180..0, where 0 is the arm standing
                  straight up and more negative leans it further over
  elbow_flex      elbow; range 0..180, 0 is straight
  wrist_flex      wrist pitch
  wrist_yaw       wrist yaw
  wrist_roll      spins the gripper about its own axis
  gripper         jaw drive: 0 is CLOSED, larger is more open; the jaws are at
                  their widest documented opening near 116

The zero pose (all joints 0) is the arm standing fully upright with the
gripper closed. Which Cartesian direction "positive" maps to for each joint
has not been verified against this build, so spend your first turns probing:
one small motion on a single joint, look at the camera, note which way it
moved.

The gripper runs in compliant position control: grip force is a function of
how far past contact you command. To grasp, command a few degrees past where
the jaws meet the object; do not slam to 0 around a rigid object."""


class MetalArm(Arm):
    """Metal over slcan (any OS), socketcan (Linux), or the bench sim."""

    def __init__(
        self,
        *,
        backend: str = "slcan",
        port: str | None = None,
        robot_id: str = "metal_arm",
        cameras: str = "overhead=0",
        control_hz: float = 25.0,
        max_step_deg: float = 0.8,
        armed: bool = False,
    ):
        from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig

        if backend not in ("slcan", "socketcan", "sim"):
            raise ValueError(f"backend must be 'slcan', 'socketcan', or 'sim', got {backend!r}")
        if backend != "sim" and not port:
            raise ValueError(
                "a real backend needs a port: the slcan serial device "
                "(/dev/cu.usbmodemXXXX) or the socketcan interface name (can0)"
            )
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be > 0, got {control_hz!r}")

        self._backend = backend
        self._armed = armed
        config = MetalFollowerConfig(
            id=robot_id,
            port=port or "sim",
            can_interface="slcan" if backend == "sim" else backend,
            max_relative_target=max_step_deg,
            cameras={},
        )
        self._follower = MetalFollower(config)
        names = tuple(self._follower._joint_motor_names)
        limits = config.joint_limits
        self.info = ArmInfo(
            name="metal",
            joints=tuple(JointSpec(n, limits[n][0], limits[n][1]) for n in names),
            gripper_index=names.index("gripper"),
            control_hz=control_hz,
            notes=METAL_NOTES,
        )

        self._cameras: tuple[Any, ...]
        if backend == "sim":
            self._follower.bus = TrackingBus(names)
            self._cameras = (SyntheticCamera(lambda: self.read().positions_deg),)
        else:
            from metal_arm_harness.camera import open_cameras

            self._cameras = open_cameras(cameras) if cameras else ()
        self._connected = False

    # ── Arm contract ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the bus for reads; torque state is left exactly as found."""
        self._follower.bus.connect()
        self._connected = True

    def enable_torque(self) -> None:
        """Energize with the proven follow gains, via the follower's connect path."""
        if self._backend != "sim" and not self._follower.is_calibrated:
            raise RuntimeError(
                "the Metal arm has no zero-pose calibration file; run LeRobot's "
                "metal_follower calibration once (arm upright, gripper closed) first"
            )
        # The follower's connect() is where gains are written and torque
        # enabled; the bus is already connected, which it tolerates via its
        # own is_connected bookkeeping on the sim bus and errors loudly on a
        # double real connect — so route through its parts instead.
        self._follower.bus.enable_torque()
        gains = {
            m: (kp, kd)
            for m, (kp, kd) in self._follower.config.gains.items()
            if m in self._follower._joint_motor_names
        }
        self._follower._resolved_gains = gains
        self._follower.bus.sync_write("Kp", {m: kp for m, (kp, _) in gains.items()})
        self._follower.bus.sync_write("Kd", {m: kd for m, (_, kd) in gains.items()})
        self._armed = True

    def read(self) -> ArmState:
        states = self._follower.bus.sync_read_all_states()
        names = self.info.joint_names
        return ArmState(
            positions_deg=np.array([states[n]["position"] for n in names], dtype=np.float64),
            velocities_deg_s=np.array([states[n]["velocity"] for n in names], dtype=np.float64),
            temperatures_c=np.array(
                [max(states[n].get("temp_mos", 0.0), states[n].get("temp_rotor", 0.0))
                 for n in names],
                dtype=np.float64,
            ),
        )

    def frames(self) -> dict[str, npt.NDArray[np.uint8]]:
        return {camera.name: camera.read() for camera in self._cameras}

    def send(self, targets_deg: Sequence[float]) -> None:
        if not self._armed:
            raise RuntimeError("send() before enable_torque(); the harness must gate this")
        action = {
            f"{name}.pos": float(value)
            for name, value in zip(self.info.joint_names, targets_deg, strict=True)
        }
        self._follower.send_action(action)

    def close(self) -> None:
        """Close the bus with torque untouched (a holding arm keeps holding)."""
        try:
            if self._connected:
                self._follower.bus.disconnect(False)
        finally:
            self._connected = False
            for camera in self._cameras:
                camera.close()


# ── bench simulator ─────────────────────────────────────────────────────────


class TrackingBus:
    """A fake DamiaoMotorsBus whose motors instantly reach every command."""

    def __init__(self, motors: Sequence[str], start: Mapping[str, float] | None = None):
        self._positions = {m: float((start or SIM_START_DEG).get(m, 0.0)) for m in motors}
        self._is_connected = False
        self.torque_enabled = False
        self.goal_writes: list[dict[str, float]] = []

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, handshake: bool = True) -> None:
        self._is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.torque_enabled = False
        self._is_connected = False

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.torque_enabled = True

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.torque_enabled = False

    def sync_read(self, data_name: str, motors: Any = None, **_: Any) -> dict[str, float]:
        if data_name != "Present_Position":
            raise NotImplementedError(f"TrackingBus cannot sync_read {data_name!r}")
        return dict(self._positions)

    def sync_read_all_states(self, motors: Any = None, **_: Any) -> dict[str, dict[str, float]]:
        return {
            m: {
                "position": p,
                "velocity": 0.0,
                "torque": 0.0,
                "temp_mos": _SIM_TEMP_C,
                "temp_rotor": _SIM_TEMP_C,
            }
            for m, p in self._positions.items()
        }

    def sync_write(self, data_name: str, values: Mapping[str, float]) -> None:
        if data_name in ("Kp", "Kd"):
            return
        if data_name == "Goal_Position":
            self.goal_writes.append(dict(values))
            self._positions.update({m: float(v) for m, v in values.items()})
            return
        raise NotImplementedError(f"TrackingBus cannot sync_write {data_name!r}")

    def sync_write_metal(self, commands: Mapping[str, tuple[float, ...]]) -> None:
        goals = {m: float(c[2]) for m, c in commands.items()}
        self.goal_writes.append(goals)
        self._positions.update(goals)


class SyntheticCamera:
    """A cartoon overhead view: a planar arm sketch reaching for a blue cube.

    Invented geometry — it exercises the perception loop, not the robot.
    """

    name = "overhead"
    _IMG = 384
    _VIEW = 0.9

    def __init__(self, joints_deg: Any):
        self._joints = joints_deg
        self.grasped = False

    def read(self) -> npt.NDArray[np.uint8]:
        positions = np.asarray(self._joints(), dtype=np.float64)
        image = np.full((self._IMG, self._IMG, 3), 238, dtype=np.uint8)
        yaw = math.radians(float(positions[0]))
        elbow = math.radians(float(positions[4])) if positions.size > 4 else 0.0
        gripper = float(positions[6]) if positions.size > 6 else 90.0

        a, b = 0.30, 0.26
        mid = (a * math.cos(yaw), a * math.sin(yaw))
        tip = (mid[0] + b * math.cos(yaw + elbow), mid[1] + b * math.sin(yaw + elbow))
        cube = (0.34, 0.12)
        closed = gripper < 30.0
        if closed and math.dist(tip, cube) < 0.05:
            self.grasped = True
        if self.grasped:
            cube = tip

        self._square(image, cube, 0.028, (28, 76, 214))
        self._line(image, (0.0, 0.0), mid, (70, 70, 78))
        self._line(image, mid, tip, (110, 110, 120))
        self._square(image, tip, 0.012, (32, 156, 88) if closed else (208, 64, 40))
        return image

    def close(self) -> None:
        return None

    def _to_px(self, point: tuple[float, float]) -> tuple[int, int]:
        half = self._VIEW / 2.0
        col = int(round((point[0] + half) / self._VIEW * (self._IMG - 1)))
        row = int(round((half - point[1]) / self._VIEW * (self._IMG - 1)))
        return row, col

    def _square(self, image: Any, centre: tuple[float, float], half: float, colour: Any) -> None:
        row, col = self._to_px(centre)
        span = max(2, int(round(half / self._VIEW * self._IMG)))
        image[max(0, row - span) : row + span + 1, max(0, col - span) : col + span + 1] = colour

    def _line(
        self, image: Any, start: tuple[float, float], end: tuple[float, float], colour: Any
    ) -> None:
        r0, c0 = self._to_px(start)
        r1, c1 = self._to_px(end)
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for k in range(steps + 1):
            row = int(round(r0 + (r1 - r0) * k / steps))
            col = int(round(c0 + (c1 - c0) * k / steps))
            image[max(0, row - 3) : row + 4, max(0, col - 3) : col + 4] = colour
