"""Regenerate the mesh-free Maker FK model and conservative link AABBs.

Run from the harness root with maker-arm-sdk checked out beside it:
    .venv/bin/python tools/vendor_maker_urdf.py
The source SDK checkout and its meshes are read only.
"""

import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

source = Path("../maker-arm-sdk/urdf/maker_arm/robot.urdf")
root = ET.parse(source).getroot()
bounds = {}
for link in root.findall("link"):
    clouds = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        path = source.parent / mesh.attrib["filename"]
        data = path.read_bytes()
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) != 84 + 50 * count:
            raise ValueError(f"not binary STL: {path}")
        triangles = np.frombuffer(
            data,
            dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]),
            offset=84,
        )
        points = triangles["vertices"].reshape(-1, 3).astype(float)
        points *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        rot = pin.rpy.rpyToMatrix(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
        offset = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        points = np.einsum("ij,kj->ik", points, rot) + offset
        assert np.all(np.isfinite(points)), f"non-finite vertices: {path}"
        clouds.append(points)
    if clouds:
        points = np.concatenate(clouds)
        bounds[link.attrib["name"]] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    for tag in ("visual", "collision"):
        for item in link.findall(tag):
            link.remove(item)
ET.indent(root)
out = Path("src/metal_arm_harness/assets/urdf")
ET.ElementTree(root).write(out / "maker_arm.urdf", encoding="unicode", xml_declaration=True)
(out / "maker_bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
print(hashlib.sha256(source.read_bytes()).hexdigest())
print(hashlib.sha256((out / "maker_arm.urdf").read_bytes()).hexdigest())
