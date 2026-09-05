# URDF provenance

metal_with_gripper.urdf — the canonical 6-revolute (nq=6) Metal model, gripper mass lumped into Link6.
Source: makermods-robotics/metal-python-ros @ ef4181f1305cbcfc63431d3bcfb96f5fb7f72763,
path metal_sdk/example/urdf/metal_with_gripper.urdf.
sha256: faac0ba624b28cf531834be0bf4eb90595ae78648dbfe12058b8ec656b65f7ef

## Maker Arm v1

Source: sibling `maker-arm-sdk/urdf/maker_arm/robot.urdf`, local SDK HEAD
`b30d05a23d72e8c155a8e00f807aba8e8c705f68`. The source file's SHA-256 (including
local changes, if any) is authoritative:
`f2921e393b5e2f225048b3ab1437e0aabea40ba538afa61885134e5a0edf7caa`.

The SDK is Apache-2.0; its license is included as `LICENSE-maker`.

`maker_arm.urdf` preserves all links, joints, inertias, limits, and the fixed
`grasp_center`; visual/collision mesh elements were removed so FK needs no CAD
mesh installation. Derived SHA-256:
`b108af798a30d0cea5b0cd0fce679b1f010bae7902a269e729338d13aca5c4b0`.
`maker_bounds.json` contains each link's axis-aligned bounds after applying the
source collision STL scales and origins. Regenerate both with
`.venv/bin/python tools/vendor_maker_urdf.py` from this repository root.

MakerKinematics rotates the Y-up CAD +90 degrees about X into a Z-up harness
frame. Its six arm-angle mappings are provisional identities. The model's
negative wrist-local Z points down the gripper. The pose target is the SDK's
inner-face centroid (`grasp_center`), **not the physical fingertip**. Collision
bounds extend beyond it. Jaw bounds include both endpoints of the full
52.4125 mm per-jaw travel, independent of motor angle. This avoids inventing a
linear motor-angle/gap transmission and keeps the floor guard conservative
when the wrist rolls. Only the base and pan housing are excluded as mounted
geometry. Link bounds are a floor proxy, not self-collision detection.

The SDK explicitly leaves physical zero alignment and actuator mapping
unvalidated. Its elbow CAD limit also differs from the captured motor range.
No real-arm floor calibration or torque enable is allowed using this provisional
model. A real table ritual must reference a commissioned physical contact point;
the centroid alone cannot establish fingertip clearance. See `docs/MAKER.md`.
