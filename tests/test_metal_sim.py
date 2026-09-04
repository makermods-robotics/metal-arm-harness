"""The Metal arm adapter against the bench simulator (no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

from metal_arm_harness.arms import build_arm
from metal_arm_harness.arms.metal import SIM_START_DEG, MetalArm


@pytest.fixture
def arm() -> MetalArm:
    metal, _ = build_arm("metal", backend="sim")
    metal.connect()
    yield metal
    metal.close()


def test_info_declares_all_seven_joints(arm: MetalArm) -> None:
    assert arm.info.joint_names == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
        "wrist_yaw", "wrist_roll", "gripper",
    )
    assert arm.info.gripper_index == 6
    assert "0 is CLOSED" in arm.info.notes


def test_read_reports_the_sim_start_pose(arm: MetalArm) -> None:
    positions = arm.read().positions_deg
    expected = [SIM_START_DEG[name] for name in arm.info.joint_names]
    assert positions == pytest.approx(expected)


def test_send_before_enable_torque_is_refused(arm: MetalArm) -> None:
    with pytest.raises(RuntimeError, match="enable_torque"):
        arm.send(list(arm.read().positions_deg))


def test_enabled_arm_tracks_small_commands(arm: MetalArm) -> None:
    arm.enable_torque()
    target = np.asarray(arm.read().positions_deg, dtype=np.float64)
    target[0] += 0.5
    arm.send([float(v) for v in target])
    assert arm.read().positions_deg[0] == pytest.approx(target[0])


def test_follower_relative_cap_bounds_a_wild_command(arm: MetalArm) -> None:
    # Even if the harness were bypassed, the driver layer caps per-tick travel.
    arm.enable_torque()
    start = np.asarray(arm.read().positions_deg, dtype=np.float64)
    wild = np.array(start)
    wild[0] += 50.0
    arm.send([float(v) for v in wild])
    moved = arm.read().positions_deg[0] - start[0]
    assert abs(moved) <= 2.0 + 1e-9  # DEFAULT_LEAD_CAP_DEG


def test_frames_render_the_synthetic_overhead_camera(arm: MetalArm) -> None:
    frames = arm.frames()
    assert set(frames) == {"overhead"}
    frame = frames["overhead"]
    assert frame.shape == (384, 384, 3) and frame.dtype == np.uint8
    blue = (frame[:, :, 2].astype(int) - frame[:, :, 0].astype(int)) > 80
    assert blue.sum() > 100


def test_real_backend_requires_a_port() -> None:
    with pytest.raises(ValueError, match="needs a port"):
        MetalArm(backend="slcan")


def test_camera_name_guard_refuses_shifted_indices(monkeypatch) -> None:
    from metal_arm_harness import camera

    monkeypatch.setattr(
        camera, "avfoundation_video_devices",
        lambda: {0: "MacBook Pro Camera", 1: "KD-USB Cameras", 2: "KD-USB Cameras"},
    )
    camera.check_device_names("overhead=2,wrist=1", "KD-USB")  # fine
    with pytest.raises(RuntimeError, match="MacBook"):
        camera.check_device_names("front=0,wrist=1", "KD-USB")
    monkeypatch.setattr(camera, "avfoundation_video_devices", lambda: None)
    camera.check_device_names("front=0", "KD-USB")  # unknown platform: no-op
