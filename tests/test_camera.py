"""HTTP camera reuse and stale-feed checks without USB or network access."""

import io
import json
import urllib.request

import numpy as np
import pytest
from PIL import Image

from metal_arm_harness.camera import HTTPSnapshotCamera, open_cameras


def test_http_camera_reuses_existing_feed_as_rgb(monkeypatch):
    jpeg = io.BytesIO()
    Image.new("RGB", (4, 3), (250, 10, 20)).save(jpeg, format="JPEG")
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if url.endswith("/status"):
            return io.BytesIO(json.dumps({"wrist": {"age_s": 0.1}}).encode())
        return io.BytesIO(jpeg.getvalue())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    cameras = open_cameras("wrist=http://127.0.0.1:8765/camera-1.jpg")
    assert len(cameras) == 1 and isinstance(cameras[0], HTTPSnapshotCamera)
    frame = cameras[0].read()
    assert frame.shape == (3, 4, 3) and frame.dtype == np.uint8
    assert frame[0, 0, 0] > 240 and frame[0, 0, 2] < 30
    assert calls == [
        "http://127.0.0.1:8765/status",
        "http://127.0.0.1:8765/camera-1.jpg",
    ] * 2
    cameras[0].close()


@pytest.mark.parametrize("status", [{}, {"wrist": {"age_s": 2.1}}])
def test_http_camera_rejects_stale_or_missing_status_before_fetching_frame(monkeypatch, status):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(json.dumps(status).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="stale frames"):
        HTTPSnapshotCamera("wrist", "http://127.0.0.1:8765/camera-1.jpg")
    assert calls == ["http://127.0.0.1:8765/status"]
