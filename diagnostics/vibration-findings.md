# Vibration investigation — 2026-09-05

Placements paused at the user's request. No motion, gain changes, torque changes,
or driver restart were issued during this investigation. Yellow remains gripped
at approximately (0.348, -0.120, 0.177) m, pitch -81 degrees.

## Measured evidence

Source: logs/episode-20260905-174738.jsonl, move 5 (the approach toward the right cup).
The new eight-second monitor sampled 199 readings while sending no motion commands.
The separate plot deliberately preserves the distinction between the move and
this later hold; the long unobserved interval is not treated as measured data.

- Shoulder and elbow show repeated oscillations while the command progresses monotonically.
- Peak spacing is approximately 0.24–0.31 seconds (about 3–4 Hz).
- Shoulder overshoot past the IK goal reached 2.441 degrees.
- During the later hold, shoulder range was 0.220 degrees; the encoder-derived
  tip-height range was 1.409 mm. This is a small one-direction change, not itself
  evidence of ongoing periodic oscillation in that later window.
- Play samples were approximately 25 Hz (median interval 40.35 ms, maximum 53.34 ms).
  This cannot exclude higher-frequency vibration or structural compliance.
- IK is solved once per requested pose. It is not repeatedly changing solutions
  during the observed oscillation. The joint goal is constant in the trace.

## Confirmed configuration mismatch and remaining hypotheses

The installed MetalFollowerConfig requests shoulder/elbow Kd=11 and pan Kd=6.5.
The Damiao encoder uses MIT_KD_RANGE=(0,5), with silent clamping. An offline
packet-encoding check verified Kd=11 and Kd=5 produce identical bytes. The
results are in gain-encoding.json. This is a proven software mismatch, not proof
that it alone causes the oscillation. The vendor's motor_config.cpp also uses
Kd=5 for the shoulder and elbow; changing protocol scaling is not justified.

The path generator clips each joint's step independently and has no acceleration
or jerk limit. Joints stop at different times with abrupt speed changes. The
follower additionally estimates velocity feedforward from clipped positions,
with a tick-based low-pass filter. Its last nonzero velocity can remain at the
motor when command streaming ends. Actual transmitted velocity targets are not
in the existing trace, so this requires instrumented verification.

The previous settle fix reduced endpoint trim but did not fix vibration during
travel. Stable readings in one later hold were insufficient validation.

## Library research and recommended order

1. Correct and validate the gain/packet assumptions; capture applied position AND
   velocity targets and feedback timestamps, and ensure zero terminal velocity.
2. Add a synchronized trajectory with explicit acceleration and jerk limits,
   retaining the harness's floor, joint, torque and excursion checks.
3. Evaluate a different IK solver only if offline tests show bad solutions or
   discontinuities. A new IK solver alone cannot fix the demonstrated oscillatory
   response to a monotonic command.

- Orocos KDL: established serial-chain IK and MoveIt's default solver.
  https://moveit.picknik.ai/main/doc/examples/kinematics_configuration/kinematics_configuration_tutorial.html
- TRAC-IK: joint-limit-aware IK; Distance mode favors solutions near the seed.
  https://moveit.picknik.ai/main/doc/how_to_guides/trac_ik/trac_ik_tutorial.html
- pick_ik: configurable MoveIt 2 IK; local mode recommended for relative moves.
  https://moveit.picknik.ai/main/doc/how_to_guides/pick_ik/pick_ik_tutorial.html
- Ruckig: synchronized velocity/acceleration/jerk-limited trajectory generation;
  used for MoveIt smoothing. It complements IK and is the relevant library for
  smoother command profiles. Integration alone is not a verified vibration fix.
  https://docs.ruckig.com/tutorial.html
  https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html
- Metal SDK gain reference:
  https://github.com/makermods-robotics/metal-python-ros/blob/humble/metal_sdk/native/config/motor_config.cpp

## User platform constraint (supersedes ROS-based integration suggestions)

All chosen dependencies must run standalone on macOS, Linux and Windows, with
no ROS runtime or packages. Standalone Ruckig is the preferred motion-generation
candidate: its upstream CI covers all three platforms and it provides Python
bindings. Use local state-to-state generation, not the optional intermediate-
waypoint cloud API. IKPy is a standalone pure-Python IK candidate if independent
IK comparison is required. Do not install ROS/MoveIt/TRAC-IK plugins.

https://github.com/pantor/ruckig
https://github.com/Phylliade/ikpy
