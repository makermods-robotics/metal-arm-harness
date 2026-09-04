"""Arm registry: name → (Arm, Kinematics | None) factory.

Adding an arm: implement `base.Arm` (and `base.Kinematics` if the arm should
get the table guard), then register a builder here. The harness above this
package never imports a concrete arm.
"""

from __future__ import annotations

from typing import Any

from metal_arm_harness.arms.base import Arm, ArmInfo, ArmState, JointSpec, Kinematics

__all__ = ["Arm", "ArmInfo", "ArmState", "JointSpec", "Kinematics", "build_arm"]


def _build_metal(**options: Any) -> tuple[Arm, Kinematics]:
    from metal_arm_harness.arms.metal import MetalArm
    from metal_arm_harness.kinematics import MetalKinematics

    tool_length = options.pop("tool_length_m", None)
    kinematics = (
        MetalKinematics(tool_length_m=tool_length) if tool_length is not None else MetalKinematics()
    )
    return MetalArm(**options), kinematics


_FACTORIES = {
    "metal": _build_metal,
}


def build_arm(name: str, **options: Any) -> tuple[Arm, Kinematics | None]:
    """Construct a registered arm and its kinematics (None disables the table guard)."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown arm {name!r}; registered arms: {sorted(_FACTORIES)}"
        ) from None
    return factory(**options)
