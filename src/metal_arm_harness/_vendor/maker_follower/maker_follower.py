#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.robstride import RobstrideMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from .config_maker_follower import MakerFollowerConfig

logger = logging.getLogger(__name__)

# RobStride model per joint, as named in `motors/robstride/tables.py` (MIT-protocol parameter
# ranges). RS00 maps to "O0" (+-12.57 rad, +-33 rad/s, +-14 Nm) and RS02 to "O1" (+-44 rad/s,
# +-17 Nm). Fixed by the hardware build: shoulder_lift and elbow_flex carry the RS02 motors.
MOTOR_MODELS = {
    "shoulder_pan": "O0",
    "shoulder_lift": "O1",
    "elbow_flex": "O1",
    "wrist_flex": "O0",
    "wrist_yaw": "O0",
    "wrist_roll": "O0",
    "gripper": "O0",
}

# Startup-sync stall release: a joint counts as making progress when it moved more than
# _SYNC_PROGRESS_EPS_DEG since its last progress mark; after _SYNC_STALL_RELEASE_SEC without
# progress it is released from the slow sync (it physically cannot close the gap at the capped
# step size, e.g. parked past a soft limit or stiction above kp * step).
_SYNC_PROGRESS_EPS_DEG = 0.2
_SYNC_STALL_RELEASE_SEC = 1.0

# A RobStride motor can come back from a power cycle reporting its angle a whole turn off (the
# multi-turn count resets; seen on this arm's shoulder_lift). Joint travel is under one turn, so
# a reading that lands inside the soft limits only after a +-360 deg shift is unambiguous. The
# grace widens the test so a joint parked just past a limit is not mistaken for a wrapped one.
_FULL_TURN_DEG = 360.0
_WRAP_GRACE_DEG = 20.0


class MakerFollower(Robot):
    """
    Maker Arm v1 follower: 6 joints + a permanent gripper, all RobStride motors over classic CAN
    via the stock `RobstrideMotorsBus` in MIT position control.

    All 7 motors are normalized in degrees. The gripper is passed through in raw motor degrees,
    same as the joints.
    """

    config_class = MakerFollowerConfig
    name = "maker_follower"

    def __init__(self, config: MakerFollowerConfig):
        super().__init__(config)
        self.config = config

        if not config.port:
            raise ValueError(
                "maker_follower requires `port`. With can_interface='slcan' (the default) it is "
                "the USB-CAN adapter's serial port, '/dev/ttyACM0' on Linux, "
                "'/dev/cu.usbmodem1101' on macOS, 'COM5' on Windows. With "
                "can_interface='socketcan' it is the interface name, e.g. 'can0'."
            )

        if config.can_interface not in ("socketcan", "slcan"):
            raise ValueError(
                f"maker_follower supports can_interface='socketcan' or 'slcan', got "
                f"'{config.can_interface}'. On Linux, bring the USB-CAN adapter up as a socketcan "
                "interface (`sudo slcand -o -f -s8 /dev/ttyACM0 can0 && sudo ip link set up can0`) "
                "and set port='can0'. On macOS/Windows, where SocketCAN does not exist, use "
                "can_interface='slcan' with the adapter's serial port as `port`."
            )

        # RobStride MIT feedback carries the motor id in payload byte 0, so recv_id == id.
        motors: dict[str, Motor] = {}
        for motor_name, can_id in config.motor_can_ids.items():
            motor_type_str = MOTOR_MODELS[motor_name]
            motor = Motor(can_id, motor_type_str, MotorNormMode.DEGREES)
            motor.recv_id = can_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self._joint_motor_names = list(motors)

        self.bus = RobstrideMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=False,
            bitrate=self.config.can_bitrate,
            data_bitrate=None,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        # False until the follower has caught up to the leader (slow initial sync), then full speed.
        self._synced = False
        self._synced_motors: set[str] = set()
        self._sync_progress: dict[str, tuple[float, float]] = {}
        self._resolved_gains: dict[str, tuple[float, float]] = {}
        # Per-session +-360 deg correction per joint, see `_detect_full_turn_offsets`.
        self._turn_offset: dict[str, float] = dict.fromkeys(self._joint_motor_names, 0.0)
        # Set when a joint sits far outside its limits and no whole-turn shift explains it. The
        # arm then refuses to move until it is recalibrated; connect() itself still succeeds so
        # `lerobot-calibrate` (which connects first) can perform that recalibration.
        self._stale_zero: dict[str, float] = {}
        # Last good reading per joint, used when a motor misses a reply.
        self._last_positions: dict[str, float] = {}

    def _detect_full_turn_offsets(self) -> None:
        """Pick a +-360 deg correction for joints that report a whole turn outside their limits.

        A joint far outside its limits that no single-turn shift explains has a stale zero.
        Driving the arm from there would be unsafe, so it is recorded in `_stale_zero` and
        `send_action` refuses until `calibrate()` runs.
        """
        raw = self._read_raw_positions()
        self._stale_zero = {}
        for motor, position in raw.items():
            self._turn_offset[motor] = 0.0
            if motor not in self.config.joint_limits:
                continue
            low, high = self.config.joint_limits[motor]
            low, high = low - _WRAP_GRACE_DEG, high + _WRAP_GRACE_DEG
            if low <= position <= high:
                continue
            for shift in (-_FULL_TURN_DEG, _FULL_TURN_DEG):
                if low <= position + shift <= high:
                    self._turn_offset[motor] = shift
                    logger.warning(
                        f"{self} {motor} reads {position:+.1f} deg, a full turn outside its limits "
                        f"{self.config.joint_limits[motor]}; correcting by {shift:+.0f} deg for this session."
                    )
                    break
            else:
                self._stale_zero[motor] = position
                logger.error(
                    f"{self} {motor} reads {position:+.1f} deg, outside its soft limits "
                    f"{self.config.joint_limits[motor]} by more than {_WRAP_GRACE_DEG:.0f} deg and "
                    "not by a whole turn. The motor zero no longer matches the calibration pose; "
                    "the arm will not move until it is recalibrated."
                )

    def _read_raw_positions(self) -> dict[str, float]:
        """Read every joint in motor coordinates, one request in flight at a time.

        `sync_read` bursts all seven requests and collects the replies afterwards. USB-CAN
        adapters with a shallow receive FIFO drop much of that burst: on a CANable at 30 Hz a
        third of the frames came back stale, which lands in a recorded dataset as a repeated
        observation. `read` sends one request and waits for that motor's reply, so each frame
        has the bus to itself, at a cost of about a millisecond per joint. A motor that still
        misses its reply keeps its previous value rather than failing the tick.
        """
        positions: dict[str, float] = {}
        for motor in self._joint_motor_names:
            try:
                positions[motor] = float(self.bus.read("Present_Position", motor))
            except Exception:
                previous = self._last_positions.get(motor)
                if previous is None:
                    raise
                logger.debug("%s no reply from %s, keeping the previous position", self, motor)
                positions[motor] = previous
        self._last_positions = positions
        return positions

    def _read_present_positions(self) -> dict[str, float]:
        raw = self._read_raw_positions()
        return {motor: raw[motor] + self._turn_offset[motor] for motor in self._joint_motor_names}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._joint_motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()
        try:
            for cam in self.cameras.values():
                cam.connect()

            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            # Gains ride along in every MIT frame, so they must be in the bus before the first
            # command. The bus default kp=10 is far too soft to hold the arm against gravity.
            # Entries for motors this arm does not have are dropped rather than written to a
            # missing id.
            self._resolved_gains = {
                m: (kp, kd) for m, (kp, kd) in self.config.gains.items() if m in self._joint_motor_names
            }
            self.bus.sync_write("Kp", {m: kp for m, (kp, kd) in self._resolved_gains.items()})
            self.bus.sync_write("Kd", {m: kd for m, (kp, kd) in self._resolved_gains.items()})
            self._detect_full_turn_offsets()
            self.bus.enable_torque()
        except Exception:
            # Do not hold the serial port (or a half-enabled arm) after a failed connect.
            self.bus.disconnect(self.config.disable_torque_on_disconnect)
            raise

        # Re-arm the slow initial sync on every connect.
        self._synced = False
        self._synced_motors = set()
        self._sync_progress = {}

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # RobStride motors hold their zero across power cycles; calibration here means the user
        # has confirmed the zero pose once and a calibration file exists on disk.
        return bool(self.calibration)

    def calibrate(self) -> None:
        """Interactive zero-pose calibration for the Maker follower arm.

        No range-of-motion recording is needed. This procedure confirms the arm is in its zero
        pose and sets that reference on each motor, matching the reBot B601 and Metal followers.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the Maker arm to its ZERO POSE: the folded resting pose it ships in,\n"
            "arm tucked against the base, gripper fully open.\n"
            "The soft limits in `joint_limits` are measured from this pose.\n"
        )
        input("Press ENTER when ready...")

        # Log the pre-zero readings so an offset against an earlier zero stays recoverable.
        positions = self._read_raw_positions()
        for motor in self._joint_motor_names:
            logger.info(f"Pre-zero position of {motor}: {positions[motor]:.2f} deg")
        self.bus.set_zero_position()
        self._turn_offset = dict.fromkeys(self._joint_motor_names, 0.0)
        self._stale_zero = {}
        self._last_positions = {}
        logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, can_id in self.config.motor_can_ids.items():
            range_min, range_max = self.config.joint_limits.get(motor_name, (-360.0, 360.0))
            self.calibration[motor_name] = MotorCalibration(
                id=can_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """No-op: follow gains are set in connect() from config.gains."""
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        positions = self._read_present_positions()
        for motor in self._joint_motor_names:
            obs_dict[f"{motor}.pos"] = positions[motor]

        for cam_key, cam in self.cameras.items():
            cam_start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - cam_start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._stale_zero:
            stale = ", ".join(f"{m} at {p:+.1f} deg" for m, p in self._stale_zero.items())
            raise RuntimeError(
                f"{self} refuses to move: {stale} is outside the soft limits and not by a whole "
                "turn, so the motor zero no longer matches the calibration pose. Move the arm to "
                "its zero pose and run `lerobot-calibrate` again."
            )
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Clamp every motor, gripper included, to its soft limit.
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(
                        f"Clipped {motor_name} from {position:.2f} deg to {clipped_position:.2f} deg"
                    )
                goal_pos[motor_name] = clipped_position

        # The startup sync and the relative-target cap both need the arm's present state. Read
        # it at most once per tick and share it; the bus serves it from its cache when the
        # observation read of this tick is still fresh.
        syncing = self.config.startup_sync_speed_deg is not None and not self._synced
        present_pos: dict[str, float] = {}
        if syncing or self.config.max_relative_target is not None:
            present_pos = self._read_present_positions()

        # Slow initial sync: at teleop start cap each joint's per-step motion until the follower
        # has caught up to the leader, so firm follow gains don't snap it across a large gap.
        # Sync is per joint: a joint that cannot converge (parked past a soft limit, or stiction
        # above what kp * step can overcome) is released after a stall instead of capping the
        # whole arm forever; max_relative_target still bounds its speed after release.
        if syncing:
            step = self.config.startup_sync_speed_deg
            now = time.perf_counter()
            for motor_name, position in goal_pos.items():
                if motor_name in self._synced_motors:
                    continue
                present = present_pos[motor_name]
                err = position - present
                if abs(err) <= self.config.startup_sync_tolerance_deg:
                    self._synced_motors.add(motor_name)
                    continue
                last_pos, last_progress_t = self._sync_progress.get(motor_name, (present, now))
                if abs(present - last_pos) > _SYNC_PROGRESS_EPS_DEG:
                    self._sync_progress[motor_name] = (present, now)
                elif now - last_progress_t > _SYNC_STALL_RELEASE_SEC:
                    logger.warning(
                        f"{self} startup sync stalled on {motor_name} ({abs(err):.1f} deg from goal); "
                        "releasing it to full speed."
                    )
                    self._synced_motors.add(motor_name)
                    continue
                else:
                    self._sync_progress.setdefault(motor_name, (present, now))
                goal_pos[motor_name] = present + max(-step, min(step, err))
            if len(self._synced_motors) >= len(goal_pos):
                self._synced = True
                logger.info(f"{self} synced to leader; tracking at full speed.")

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Gains were stored in the bus at connect and ride along in every MIT frame. Written one
        # motor at a time for the same reason reads are, see `_read_raw_positions`.
        for motor, position in goal_pos.items():
            self.bus.write("Goal_Position", motor, position - self._turn_offset[motor])

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
