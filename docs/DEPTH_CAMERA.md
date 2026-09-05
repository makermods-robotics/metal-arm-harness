# Orbbec depth viewer

From the repository root:

```sh
.venv/bin/python -m metal_arm_harness.depth_viewer
```

The window shows live depth, a center crosshair with distance in millimeters,
and a fixed color legend. Warm colors are near; cool colors are far. Black means
missing depth. Values outside the display range retain the endpoint color;
the saved measurements and center reading are not clipped.

- **Q / Escape / close window:** stop and release the camera.
- **S:** save a PNG and a float32 `.npy` array in `frames/depth/`.
- `--near 200 --far 1500`: change the color range (millimeters).
- `--snapshot`: save one frame without opening a window.
- `--list`: list SDK-visible devices; `--device 1` selects a device index.
- `--output PATH`: change the save directory.

Saved `.npy` arrays contain depth in millimeters, with zero meaning missing
depth. Load with `numpy.load(path)`. Depth is the camera's depth measurement,
not robot coordinates. This standalone tool does not connect to the arm.

## Camera and SDK compatibility

The camera connected on September 5, 2026 identifies over USB as **Orbbec DaBai
DCW2**, depth VID:PID `2bc5:06a0`, RGB `2bc5:0561`, on USB 2.0. The SDK v2 Python
package (`pyorbbecsdk2` 2.1.2) reported zero devices. SDK v1.10.16 detects it.
The v1 Python binding has been built and installed into this project's `.venv`.
Hardware capture succeeded at 640×400, 15 fps, with firmware RD1017. One test
frame contained 92.4% valid pixels and depths of 321–671 mm. For this nearby
scene, use `--near 200 --far 1000` for more visible color contrast.

[Orbbec documents the protocol distinction](https://github.com/orbbec/OrbbecSDK):
legacy OpenNI firmware requires SDK v1 or OpenNI. SDK v2 serves supported UVC
models. Do not install both Python packages together: both import as
`pyorbbecsdk`. No firmware update is needed for this viewer.

## Rebuild the legacy binding on Apple Silicon

Requires Git, CMake and Xcode command-line tools. Run from this repository:

```sh
uv pip install --python .venv/bin/python pybind11 opencv-python
ORBBEC_BUILD_DIR=$(mktemp -d /tmp/orbbec-v1.XXXXXX)
git clone https://github.com/orbbec/pyorbbecsdk.git "$ORBBEC_BUILD_DIR/source"
git -C "$ORBBEC_BUILD_DIR/source" checkout ee32b475fdd3a433c568842597c55d29b385052e
cmake -S "$ORBBEC_BUILD_DIR/source" -B "$ORBBEC_BUILD_DIR/source/build" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build "$ORBBEC_BUILD_DIR/source/build" -j 4
cmake --install "$ORBBEC_BUILD_DIR/source/build"
# Remove v2 first if it is installed.
uv pip uninstall --python .venv/bin/python pyorbbecsdk2
uv pip install --python .venv/bin/python "$ORBBEC_BUILD_DIR/source"
.venv/bin/python -m metal_arm_harness.depth_viewer --list
```

For a newer UVC camera, install its compatible SDK following the
[official Python installation guide](https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html).
The viewer uses the shared v1/v2 API; the connected DaBai is the hardware
validation target here.

If no device is found, close other camera applications, check the USB cable,
and verify that the SDK matches the camera's protocol. A normal RGB webcam
stream does not provide the depth map. If frames stop arriving, the viewer
reports an error after ten seconds instead of silently showing stale data.
