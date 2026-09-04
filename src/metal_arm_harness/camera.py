"""USB webcams via OpenCV, plus the spec parser for ``name=index`` strings.

Indices are positional: when a USB camera drops off the bus every index
after it shifts, and "overhead=2" silently becomes the laptop's own camera.
`check_device_names` maps indices to AVFoundation device names (macOS, via
ffmpeg when present) so a session can refuse to open anything that is not
the bench camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


@dataclass
class CameraSpecEntry:
    name: str
    target: int | str


def parse_camera_spec(spec: str) -> tuple[CameraSpecEntry, ...]:
    """Parse ``"overhead=0,wrist=/dev/video2"``; bare entries become camN."""
    out: list[CameraSpecEntry] = []
    for index, chunk in enumerate(part.strip() for part in spec.split(",")):
        if not chunk:
            continue
        if "=" in chunk:
            name, _, target = chunk.partition("=")
            name, target = name.strip(), target.strip()
        else:
            name, target = f"cam{index}", chunk
        if not name:
            raise ValueError(f"camera spec {chunk!r} has an empty name")
        out.append(CameraSpecEntry(name=name, target=int(target) if target.isdigit() else target))
    names = [entry.name for entry in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate camera names in {spec!r}: {names}")
    return tuple(out)


class OpenCVCamera:
    """One webcam read through ``cv2.VideoCapture``, converted to RGB."""

    def __init__(self, name: str, target: int | str, width: int, height: int):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "cameras need OpenCV.\n"
                "fix: uv pip install 'metal-arm-harness[camera]', or run with --cameras ''"
            ) from exc
        self.name = name
        self._cv2 = cv2
        self._target = target
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"camera {name!r} ({target!r}) would not open.\n"
                "fix: check the device index; on macOS grant camera access to the terminal "
                "in System Settings > Privacy & Security > Camera"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Smallest buffer the backend honors: a frame read during motion must
        # show where the arm is now, not three frames ago.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise RuntimeError(f"camera {name!r} opened but returned no frame")

    def read(self, retries: int = 3) -> npt.NDArray[np.uint8]:
        """One RGB frame. A USB camera occasionally drops a frame; retry briefly
        before declaring it gone, and reopen once if the handle itself died."""
        import time

        for attempt in range(retries + 1):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                return np.ascontiguousarray(rgb, dtype=np.uint8)
            if attempt == 1 and not self._cap.isOpened():
                self._cap.release()
                self._cap = self._cv2.VideoCapture(self._target)
            time.sleep(0.05 * (attempt + 1))
        raise RuntimeError(f"camera {self.name!r} stopped returning frames")

    def close(self) -> None:
        self._cap.release()


def open_cameras(
    spec: str, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT
) -> tuple[OpenCVCamera, ...]:
    """Open every camera in the spec, releasing already-opened ones on failure."""
    opened: list[OpenCVCamera] = []
    try:
        for entry in parse_camera_spec(spec):
            opened.append(OpenCVCamera(entry.name, entry.target, width, height))
    except Exception:
        for camera in opened:
            camera.close()
        raise
    return tuple(opened)


def avfoundation_video_devices() -> dict[int, str] | None:
    """index -> device name from ffmpeg's AVFoundation listing; None if unavailable."""
    import re
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=10, check=False,
        ).stderr
    except (OSError, subprocess.SubprocessError):
        return None
    devices: dict[int, str] = {}
    in_video = False
    for line in out.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        match = re.search(r"\[(\d+)\] (.+)$", line) if in_video else None
        if match:
            devices[int(match.group(1))] = match.group(2).strip()
    return devices or None


def check_device_names(spec: str, required_substring: str) -> None:
    """Refuse a spec whose indices do not all map to devices named like `required_substring`.

    No-op when the device list cannot be read (non-macOS, no ffmpeg).
    """
    devices = avfoundation_video_devices()
    if devices is None:
        return
    wrong = []
    for entry in parse_camera_spec(spec):
        if isinstance(entry.target, int):
            name = devices.get(entry.target, "<no device>")
            if required_substring.lower() not in name.lower():
                wrong.append(f"{entry.name}={entry.target} is '{name}'")
    if wrong:
        raise RuntimeError(
            "camera indices do not point at the bench cameras: "
            + "; ".join(wrong)
            + f". Expected names containing {required_substring!r}. A USB camera probably "
            "dropped off (indices shift): re-plug and re-check with "
            "`ffmpeg -f avfoundation -list_devices true -i \"\"`."
        )
