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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@dataclass
class MakerFollowerConfigBase:
    """
    Configuration for the Maker Arm v1 follower: 6 RobStride joints plus a permanent RS00
    gripper on one classic CAN bus, driven by the stock `RobstrideMotorsBus` in MIT position
    control. The motors must run RobStride's MIT protocol (see the Maker docs for the one-time
    switch from the factory private protocol).

    Kept separate from the registered `MakerFollowerConfig` so `BiMakerFollowerConfig` can embed
    per-arm configs without referencing the RobotConfig choice registry (which would make the
    draccus CLI parser tree self-referential).
    """

    # Required; there is no portable default. With can_interface="slcan" this is the adapter's
    # serial port ("/dev/ttyACM0", "/dev/cu.usbmodem1101" on macOS, "COM5" on Windows); with
    # "socketcan" it is an interface name ("can0").
    port: str | None = None

    # "slcan" (any OS, via pyserial) or "socketcan" (Linux only). slcan is the default because
    # it is the only transport that works on all three platforms and needs no privileged setup.
    # Measured on a Maker arm over a CANable on macOS, a full tick (7 state reads + 7 MIT writes,
    # each waiting for its reply) costs 9.6 ms of a 33 ms period at 30 Hz.
    can_interface: str = "slcan"

    # Classic CAN at 1 Mbps, no CAN FD.
    can_bitrate: int = 1_000_000

    # Motor CAN id per joint, base to gripper. Both arms of a bimanual rig ship with these same
    # ids, which is why each arm needs its own bus.
    motor_can_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_yaw": 5,
            "wrist_roll": 6,
            "gripper": 7,
        }
    )

    # Soft joint limits (degrees), clipped against on every action. Zero is the calibration pose
    # (the arm folded in its resting position, gripper open). Values are the vendor's mechanical
    # stop captures (maker-arm SDK profile, arm #02, 2026-08-20, taken with a 0.05 rad backoff
    # from the stops) re-expressed against that zero. Narrow them to fence off workspace.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-158.2, 156.1),
            "shoulder_lift": (-175.3, -3.2),
            "elbow_flex": (2.8, 236.1),
            "wrist_flex": (-65.2, 104.0),
            "wrist_yaw": (-97.6, 78.0),
            "wrist_roll": (-153.2, 152.0),
            "gripper": (-120.1, -2.5),
        }
    )

    # Per-motor MIT follow gains {name: (kp, kd)}, applied at connect(). The vendor's bench-tuned
    # values: hold, sine and teleop tracking all passed with them. The gripper runs a deliberately
    # low kp so grip force (kp times position error) stays compliant. Mutating the resolved dict
    # at runtime retunes the arm live.
    gains: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (60.0, 4.0),
            "shoulder_lift": (150.0, 4.5),
            "elbow_flex": (90.0, 3.0),
            "wrist_flex": (30.0, 2.0),
            "wrist_yaw": (30.0, 2.0),
            "wrist_roll": (30.0, 2.0),
            "gripper": (20.0, 0.5),
        }
    )

    # Slow initial synchronization: at teleop start the follower can be far from the leader, and
    # firm follow gains would snap it there hard. Until it has caught up, cap each joint's per-step
    # motion to this many degrees (gentle alignment), then track at full speed. None disables it.
    startup_sync_speed_deg: float | None = 1.0
    startup_sync_tolerance_deg: float = 3.0

    # Safety limit for relative target positions (degrees). None disables the check.
    max_relative_target: float | dict[str, float] | None = None

    # Torque-off is the vendor's documented safe state and the arm has no brakes, so a torque-off
    # arm settles under gravity: support it before disconnecting.
    disable_torque_on_disconnect: bool = True

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("maker_follower")
@dataclass
class MakerFollowerConfig(RobotConfig, MakerFollowerConfigBase):
    """Registered single-arm Maker follower config (adds `id` / `calibration_dir`)."""
