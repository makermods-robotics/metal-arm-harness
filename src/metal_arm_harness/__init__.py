"""metal-arm-harness: a small LLM-as-policy harness for real robot arms."""

from metal_arm_harness.agent import AgentConfig, ArmAgent, EpisodeResult
from metal_arm_harness.arms import build_arm
from metal_arm_harness.calibration import run_table_ritual
from metal_arm_harness.safety import MoveRejected, SafetyAbort, SafetyConfig, SafetyEnvelope

__all__ = [
    "AgentConfig",
    "ArmAgent",
    "EpisodeResult",
    "MoveRejected",
    "SafetyAbort",
    "SafetyConfig",
    "SafetyEnvelope",
    "build_arm",
    "run_table_ritual",
]

__version__ = "0.1.0"
