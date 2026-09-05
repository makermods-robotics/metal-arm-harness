"""Standalone Orbbec depth viewer; never imports or connects to robot hardware.

Run: python -m metal_arm_harness.depth_viewer
SDK installation and camera compatibility: docs/DEPTH_CAMERA.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def depth_in_mm(frame) -> np.ndarray:
    """Copy SDK-owned uint16 data and apply its millimeters-per-unit scale."""
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
    return raw.reshape(frame.get_height(), frame.get_width()).astype(np.float32) * (
        frame.get_depth_scale()
    )


def colorize(depth_mm: np.ndarray, near: float, far: float) -> np.ndarray:
    """Fixed color scale: near is warm, far is cool, missing depth is black."""
    import cv2

    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    safe = np.where(valid, depth_mm, far)
    intensity = (255 * (1 - np.clip((safe - near) / (far - near), 0, 1))).astype(np.uint8)
    colored = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def render(depth_mm: np.ndarray, near: float, far: float, fps: float) -> np.ndarray:
    import cv2

    height, width = depth_mm.shape
    colored = colorize(depth_mm, near, far)
    cx, cy = width // 2, height // 2
    cv2.drawMarker(colored, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    canvas = np.zeros((height + 100, max(width, 640), 3), dtype=np.uint8)
    canvas[:height, :width] = colored
    center = depth_mm[cy, cx]
    distance = f"{center:.0f} mm" if np.isfinite(center) and center > 0 else "no depth"
    label = f"Center: {distance}  |  {fps:.1f} fps  |  S: save   Q/Esc: quit"
    cv2.putText(
        canvas,
        label,
        (12, height + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    ramp = np.linspace(near, far, canvas.shape[1] - 24, dtype=np.float32)[None, :]
    canvas[height + 36 : height + 52, 12:-12] = colorize(np.repeat(ramp, 16, axis=0), near, far)
    cv2.putText(
        canvas,
        f"Near: {near:.0f} mm",
        (12, height + 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Far: {far:.0f} mm",
        (canvas.shape[1] - 155, height + 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Black: missing depth. Distances outside the scale use endpoint colors.",
        (12, height + 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_frame(folder: Path, depth_mm: np.ndarray, display: np.ndarray) -> Path:
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / datetime.now().strftime("depth-%Y%m%d-%H%M%S-%f")
    np.save(stem.with_suffix(".npy"), depth_mm)
    if not cv2.imwrite(str(stem.with_suffix(".png")), display):
        raise RuntimeError(f"Could not write {stem}.png")
    print(f"Saved {stem}.png and {stem}.npy (float32 millimeters)", flush=True)
    return stem


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
        import pyorbbecsdk as ob
    except ImportError as exc:
        raise RuntimeError(
            "Install OpenCV and the camera-compatible Orbbec SDK. See docs/DEPTH_CAMERA.md."
        ) from exc

    context = ob.Context()
    devices = context.query_devices()
    if not devices.get_count():
        raise RuntimeError(
            "No Orbbec depth camera detected by this SDK. Check USB and close other camera "
            "apps. DaBai DCW2 with OpenNI firmware needs SDK v1; pyorbbecsdk2 will not "
            "detect it. See docs/DEPTH_CAMERA.md."
        )
    if args.list:
        for index in range(devices.get_count()):
            info = devices.get_device_by_index(index).get_device_info()
            print(
                f"{index}: {info.get_name()} | serial {info.get_serial_number()} "
                f"| firmware {info.get_firmware_version()}"
            )
        return 0
    if args.device >= devices.get_count():
        raise RuntimeError(f"Device index {args.device} does not exist; use --list.")
    device = devices.get_device_by_index(args.device)
    name = device.get_device_info().get_name()
    pipeline = ob.Pipeline(device)
    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    # DaBai advertises Y11 transport; the SDK converts it to metric Y16 frames.
    # Choose the device default and validate the delivered format below.
    profile = profiles.get_default_video_stream_profile()
    config = ob.Config()
    config.enable_stream(profile)
    title = f"Orbbec depth - {name}"
    started = False
    window = False
    try:
        pipeline.start(config)
        started = True
        print(f"Streaming {name}: {profile}", flush=True)
        if not args.snapshot:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            window = True
        last_frame = time.monotonic()
        previous = None
        fps = 0.0
        received = 0
        while True:
            frames = pipeline.wait_for_frames(100)
            frame = frames.get_depth_frame() if frames is not None else None
            if frame is not None:
                if frame.get_format() != ob.OBFormat.Y16:
                    raise RuntimeError(f"Expected unpacked Y16 depth, got {frame.get_format()}")
                now = time.monotonic()
                if previous is not None:
                    fps = 0.9 * fps + 0.1 / max(now - previous, 1e-6)
                previous = last_frame = now
                depth_mm = depth_in_mm(frame)
                display = render(depth_mm, args.near, args.far, fps)
                received += 1
                if args.snapshot:
                    # Let exposure and the depth pipeline settle before capturing.
                    if received >= 15:
                        save_frame(args.output, depth_mm, display)
                        valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
                        print(f"Valid depth: {valid.size / depth_mm.size:.1%}")
                        if not valid.size:
                            raise RuntimeError(
                                "Captured frame contains no valid depth measurements."
                            )
                        print(f"Measured range: {valid.min():.0f}-{valid.max():.0f} mm")
                        return 0
                else:
                    cv2.imshow(title, display)
            if time.monotonic() - last_frame > 10:
                raise RuntimeError(
                    "No depth frames for 10 seconds. Check the camera USB connection."
                )
            if window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("s"), ord("S")) and received:
                    save_frame(args.output, depth_mm, display)
    finally:
        try:
            if started:
                pipeline.stop()
        finally:
            if window:
                cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near", type=float, default=200, help="Warm end of color scale, mm")
    parser.add_argument("--far", type=float, default=3000, help="Cool end of color scale, mm")
    parser.add_argument("--device", type=int, default=0, help="Orbbec device index (see --list)")
    parser.add_argument("--list", action="store_true", help="List cameras and exit")
    parser.add_argument("--snapshot", action="store_true", help="Save one frame without a GUI")
    parser.add_argument("--output", type=Path, default=Path("frames/depth"))
    args = parser.parse_args(argv)
    if not (np.isfinite(args.near) and np.isfinite(args.far) and 0 <= args.near < args.far):
        parser.error("require finite values with 0 <= --near < --far")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    try:
        return run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Depth viewer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
