"""metal-arm-harness: a safety-enveloped operator harness for real robot arms."""

from metal_arm_harness.arms import build_arm
from metal_arm_harness.calibration import run_table_ritual
from metal_arm_harness.control import Controller, GotoReport
from metal_arm_harness.ik import IKError, offset_target, solve_tip
from metal_arm_harness.safety import MoveRejected, SafetyAbort, SafetyConfig, SafetyEnvelope

__all__ = [
    "Controller",
    "GotoReport",
    "IKError",
    "MoveRejected",
    "SafetyAbort",
    "SafetyConfig",
    "SafetyEnvelope",
    "build_arm",
    "offset_target",
    "run_table_ritual",
    "solve_tip",
]

__version__ = "0.2.0"
