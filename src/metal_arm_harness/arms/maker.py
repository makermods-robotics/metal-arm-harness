"""Maker Arm v1 over LeRobot's MakerFollower and RobstrideMotorsBus.

Real connections are read-only during commissioning. Upstream's state reads
send CLEAR_FAULT and may reuse stale feedback; neither is a safe observation.
Only explicit, side-effect-free MIT diagnostic queries are used here until
encoder access, watchdog/idle holding, and motor-to-CAD alignment are validated.
The sim exercises the actual follower's target clamps and degree conventions.
"""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from metal_arm_harness.arms.base import Arm, ArmInfo, ArmState, JointSpec
from metal_arm_harness.arms.metal import SyntheticCamera, TrackingBus

MAKER_COMMISSIONING = (
    "Maker powered operation is not commissioned: verify fresh encoder feedback, "
    "motor-side watchdog with continuous idle holding, and motor-to-URDF alignment first. "
    "Use an unarmed `serve --arm maker --diagnostics-only` session and `op inspect`. "
    "See docs/MAKER.md. No motor has been enabled."
)
MAKER_START_DEG = (0.0, -20.0, 30.0, 0.0, 0.0, 0.0, -3.0)
MAKER_NOTES = """Six arm joints and an RS00 gripper, absolute motor degrees.
CAN IDs 1..7: shoulder_pan, shoulder_lift (RS02), elbow_flex (RS02),
wrist_flex, wrist_yaw, wrist_roll, gripper (others RS00).
LeRobot zero: folded resting pose, gripper open. Open is -3 deg; close is
-119.6 deg, inside the captured soft limits. Do not use Metal's 112/0 commands.
CAD FK is provisional: Y-up is rotated +90 deg about X into the harness's
Z-up base frame. Physical axis signs and zero offsets are not yet validated.
Real hardware supports read-only diagnostics; powered commissioning is pending.
"""


def _follower_types():
    # The two upstream robot integrations currently live on separate branches.
    try:
        from lerobot.robots.maker_follower import MakerFollower, MakerFollowerConfig
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot.robots.maker_follower":
            raise
        from metal_arm_harness._vendor.maker_follower import MakerFollower, MakerFollowerConfig
    return MakerFollower, MakerFollowerConfig


class MakerTrackingBus(TrackingBus):
    """The same perfect tracking sim, with Maker's sequential LeRobot API."""

    def read(self, data_name: str, motor: str) -> float:
        return self.sync_read(data_name)[motor]

    def write(self, data_name: str, motor: str, value: float) -> None:
        self.sync_write(data_name, {motor: value})


class MakerArm(Arm):
    def __init__(
        self,
        *,
        backend: str = "slcan",
        port: str | None = None,
        robot_id: str = "maker_arm",
        cameras: str = "",
        control_hz: float = 25.0,
        lead_cap_deg: float | Mapping[str, float] | None = None,
        armed: bool = False,
    ):
        if backend not in ("slcan", "socketcan", "sim"):
            raise ValueError("backend must be slcan, socketcan, or sim")
        if backend != "sim" and not port:
            raise ValueError("a real backend needs a port (slcan device or socketcan interface)")
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if armed:
            raise ValueError(
                "use enable_torque() after the harness gates; armed at construction is unsafe"
            )
        follower_type, config_type = _follower_types()
        config = config_type(
            id=robot_id,
            port=port or "sim",
            cameras={},
            can_interface="slcan" if backend == "sim" else backend,
            startup_sync_speed_deg=None,
            disable_torque_on_disconnect=False,
        )
        names = tuple(config.motor_can_ids)
        # Deliberately not Metal's bench-tuned shoulder/gripper overrides.
        caps = dict.fromkeys(names, 2.0)
        if isinstance(lead_cap_deg, Mapping):
            if set(lead_cap_deg) - set(names):
                raise ValueError("unknown joint in lead cap")
            caps.update(lead_cap_deg)
        elif lead_cap_deg is not None:
            caps = dict.fromkeys(names, lead_cap_deg)
        if any(not math.isfinite(v) or v <= 0 for v in caps.values()):
            raise ValueError("lead caps must be finite and > 0")
        config.max_relative_target = caps
        self._follower = follower_type(config)
        self._backend = backend
        self._armed = False
        self._connected = False
        self.info = ArmInfo(
            name="maker",
            joints=tuple(JointSpec(n, *config.joint_limits[n]) for n in names),
            gripper_index=names.index("gripper"),
            control_hz=control_hz,
            notes=MAKER_NOTES,
            gripper_open_deg=-3.0,
            gripper_closed_deg=-119.6,
            rest_positions_deg=None,
            ik_joints=(0, 1, 2, 3),
        )
        if backend == "sim":
            self._follower.bus = MakerTrackingBus(
                names, dict(zip(names, MAKER_START_DEG, strict=True))
            )
            self._cameras = (SyntheticCamera(self._camera_pose),)
        else:
            from metal_arm_harness.camera import open_cameras

            self._cameras = open_cameras(cameras) if cameras else ()

    def _camera_pose(self) -> np.ndarray:
        q = self.read().positions_deg.copy()
        q[6] = (
            (q[6] - self.info.gripper_closed_deg)
            * 112
            / (self.info.gripper_open_deg - self.info.gripper_closed_deg)
        )
        return q

    def connect(self) -> None:
        # No follower.connect(): it enables torque. No bus handshake(): it clears faults.
        self._follower.bus.connect(handshake=False)
        self._connected = True

    def enable_torque(self) -> None:
        if not self._connected:
            raise RuntimeError("connect before enable_torque")
        if self._backend != "sim":
            raise RuntimeError(MAKER_COMMISSIONING)
        self._follower._resolved_gains = dict(self._follower.config.gains)
        for key, index in (("Kp", 0), ("Kd", 1)):
            self._follower.bus.sync_write(
                key, {n: gains[index] for n, gains in self._follower._resolved_gains.items()}
            )
        self._follower.bus.enable_torque()
        self._armed = True

    def read(self) -> ArmState:
        if not self._connected:
            raise RuntimeError("Maker arm is not connected")
        if self._backend != "sim":
            raise RuntimeError(
                "Maker encoder/temperature feedback is not commissioned; cached zeros are not "
                "measurements. Run `op inspect` in a --diagnostics-only session."
            )
        states = self._follower.bus.sync_read_all_states()
        names = self.info.joint_names
        return ArmState(
            np.array([states[n]["position"] for n in names]),
            np.zeros(len(names)),
            np.full(len(names), 32.0),
            faults=tuple("" for _ in names),
            efforts_nm=tuple(states[n]["torque"] for n in names),
        )

    def inspect(self) -> dict[str, Any]:
        """Fresh fault, position-parameter and watchdog-capability queries; no writes.

        Status packets are never decoded as encoders. No STOP, enable, zero,
        clear-fault, parameter write, or MIT position frame is sent by this path.
        Wire definitions follow maker-arm-sdk/docs/mit-protocol.md.
        """
        if not self._connected:
            raise RuntimeError("Maker arm is not connected")
        if self._backend == "sim":
            return {"arm": "maker", "backend": "sim", "simulated": True}
        result: dict[str, Any] = {
            "arm": "maker",
            "backend": self._backend,
            "powered_ready": False,
            "reason": MAKER_COMMISSIONING,
            "motors": {},
        }
        for name, motor_id in self._follower.config.motor_can_ids.items():
            fault = self._query(
                motor_id,
                bytes([255] * 6 + [0, 0xFB]),
                lambda msg, mid=motor_id: self._fault_reply(msg, mid),
            )
            position = self._parameter(motor_id, 0x7019, "f")
            watchdog = self._parameter(motor_id, 0x7028, "I")
            result["motors"][name] = {
                "can_id": motor_id,
                "fault_bytes": None if fault is None else fault.hex(),
                "fault_status_available": fault is not None,
                "has_fault": None if fault is None else any(fault),
                "raw_position_deg": None if position is None else math.degrees(position),
                "watchdog_ticks": watchdog,
            }
        return result

    @staticmethod
    def _fault_reply(msg: Any, motor_id: int) -> bytes | None:
        data = bytes(msg.data)
        if (
            not msg.is_extended_id
            and msg.arbitration_id == 0xFD
            and len(data) in (5, 8)
            and data[0] == motor_id
            and (len(data) == 5 or data[5:] == bytes(3))
        ):
            return data[1:5]
        return None

    def _parameter(self, motor_id: int, index: int, dtype: str) -> Any:
        def match(msg: Any) -> Any:
            data = bytes(msg.data)
            if (
                not msg.is_extended_id
                and msg.arbitration_id == (0x300 | motor_id)
                and len(data) == 8
                and data[:4] == struct.pack("<H2x", index)
            ):
                value = struct.unpack("<" + dtype, data[4:])[0]
                return value if math.isfinite(value) else None
            return None

        return self._query(0x300 | motor_id, struct.pack("<H6x", index), match)

    def _query(self, can_id: int, payload: bytes, match: Any) -> Any:
        import can

        bus = self._follower.bus.canbus
        # Discard already-queued replies before a request. Bound the drain in case
        # another producer is streaming; one process must own the bus.
        for _ in range(256):
            if bus.recv(timeout=0) is None:
                break
        else:
            raise RuntimeError("CAN receive queue never became quiet; check for another bus owner")
        bus.send(can.Message(arbitration_id=can_id, data=payload, is_extended_id=False))
        deadline = time.monotonic() + 0.15
        while (remaining := deadline - time.monotonic()) > 0:
            msg = bus.recv(timeout=remaining)
            if msg is not None:
                value = match(msg)
                if value is not None:
                    return value
        return None

    def clear_faults(self) -> tuple[str, ...]:
        if self._backend != "sim":
            raise RuntimeError(
                "Maker fault recovery requires supported-arm commissioning; no faults cleared"
            )
        return ()

    def send(self, targets_deg: Sequence[float]) -> None:
        if not self._armed:
            raise RuntimeError("send() before enable_torque(); the harness must gate this")
        values = np.asarray(targets_deg, dtype=float)
        if values.shape != (len(self.info.joints),) or not np.all(np.isfinite(values)):
            raise ValueError("targets must be a finite vector in joint order")
        self._follower.send_action(
            {f"{n}.pos": float(v) for n, v in zip(self.info.joint_names, values, strict=True)}
        )

    def frames(self) -> dict[str, np.ndarray]:
        return {camera.name: camera.read() for camera in self._cameras}

    def close(self) -> None:
        try:
            if self._connected:
                self._follower.bus.disconnect(False)
        finally:
            self._connected = self._armed = False
            for camera in self._cameras:
                camera.close()
