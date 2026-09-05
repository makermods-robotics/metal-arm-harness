"""Depth units and display semantics, without a camera or native SDK."""

import numpy as np
import pytest

from metal_arm_harness.depth_viewer import colorize, depth_in_mm, main


def test_sdk_scale_and_owned_memory():
    raw = np.array([[0, 1200], [2400, 65535]], dtype=np.uint16)

    class Frame:
        def get_data(self):
            return raw.data

        def get_width(self):
            return 2

        def get_height(self):
            return 2

        def get_depth_scale(self):
            return 0.25

    depth = depth_in_mm(Frame())
    np.testing.assert_array_equal(depth, [[0, 300], [600, 16383.75]])
    raw[:] = 0
    assert depth[1, 0] == 600
    assert depth.dtype == np.float32


def test_colors_are_stable_and_invalid_depth_is_black():
    pytest.importorskip("cv2")
    first = colorize(np.array([[0, np.nan, np.inf, -1, 500, 1000]]), 200, 3000)
    second = colorize(np.array([[500, 20000]]), 200, 3000)
    assert not first[0, :4].any()
    np.testing.assert_array_equal(first[0, 4], second[0, 0])
    assert first[0, 4:].any()


@pytest.mark.parametrize(
    "args",
    [
        ["--near", "1000", "--far", "1000"],
        ["--far", "nan"],
        ["--near", "-1"],
        ["--device", "-1"],
    ],
)
def test_invalid_options_fail_before_opening_camera(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
