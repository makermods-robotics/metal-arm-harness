"""USB webcams via OpenCV, plus the spec parser for ``name=index`` strings."""

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

    def read(self) -> npt.NDArray[np.uint8]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"camera {self.name!r} stopped returning frames")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

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
