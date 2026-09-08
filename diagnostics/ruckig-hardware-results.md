# Powered Metal arm test — 2026-09-05

Implemented standalone Ruckig trajectory generation and disabled the follower's inferred
velocity feedforward. Real-arm measurements show substantially less oscillation on the
previously troublesome motion. This is evidence of improvement, not a guarantee that
all vibration is eliminated throughout the workspace.

## Repeated-path comparison

Tool target: x=0.350 m, y=-0.120 m, z=0.180 m, pitch=-80 degrees, from approximately
x=0.245 m, y=-0.040 m, z=0.180 m. Yellow block held throughout. Harness IK -> envelope
-> executor -> Metal driver. Both wrist and side cameras inspected after every move.

| Measurement | Before | After |
|---|---:|---:|
| Shoulder 2-6 Hz travel tracking-error RMS | 0.225 deg | 0.030 deg |
| Elbow 2-6 Hz travel tracking-error RMS | 0.712 deg | 0.040 deg |
| Shoulder overshoot beyond IK goal | 2.441 deg | 0.952 deg |
| Largest arm-joint range in final 5-second hold | — | 0.011 deg |
| FK tip z range in final hold | — | 0.071 mm |

The RMS values describe filtered encoder-minus-requested-command error, not camera
motion. The runs share requested start/end tool poses but differ in actual joint
state, grip setting and execution timing, so this is not a controlled experiment
isolating one cause. All software/control changes and factors are listed below.

Before log: `logs/episode-20260905-174738.jsonl`, move 5.
After log: `logs/episode-20260905-183234.jsonl`, move 5.
Plot/metrics: `diagnostics/ruckig-repeat-comparison.{png,json}`.

## Changes and probable causes

- Full motion uses local Ruckig 0.15.3 rest-to-rest, phase-synchronized trajectories.
  Speed 10 deg/s on this server, slow-zone 5 deg/s, acceleration 20 deg/s²,
  jerk 80 deg/s³. Sampled paths are envelope checked before sending.
- Removed feedback-clipped-position-derived velocity feedforward. Position writes
  explicitly request zero motor velocity. This avoids commanding residual velocity
  after arriving and removes an unintended feedback path through the lead cap.
- Existing endpoint trim fixes remain: no doubled arrival lag, bounded time-based
  integral, stable-window detection, jaw-only movements retain the held arm command.
- Gains above the MIT Kd wire maximum are normalized to 5 in config. This does not
  increase damping or change the protocol. Kp and lead caps are unchanged.
- Driver telemetry now records post-limit position requests and velocity fields,
  timestamped separately from the encoder sample, with gains. It does not claim to
  read packets back from the motor or prove the wire-level quantized values.
- The command-to-command jerk bounds describe planned trajectories. The driver lead
  clamp, motor controller, mechanical dynamics and settle trims remain additional
  influences on measured motion.

The custom damped-least-squares IK in `src/metal_arm_harness/ik.py` remains. It solves
the endpoint once per pose command. Ruckig controls how the arm travels between
joint states; it is not a replacement IK solver.

## Physical tests and recovery

First small inward move to (0.330,-0.120,0.180,-80) completed with yellow held,
followed by five-second encoder hold. A subsequent larger command was blocked
before trajectory playback when the gripper reported 72 C. The harness's 70 C
veto relieved gripper torque. We inspected both cameras, recorded a cooled hold,
and reduced the gripper target to 17.5 degrees around its ~20.8-degree contact,
measuring ~1.16 N m instead of 2.80 N m. Yellow remained held. Then inward/outward
pose tests completed with no further thermal aborts. Temperature matters during
long loaded holds; do not automatically reinstate full-close squeeze.

Current physical state at test end: stationary and powered, yellow held with reduced
squeeze, blue remains on table; cup-placement work remains paused for these tests.
One server owns the hardware, and all movements used `metal-arm-harness op`.

## Validation and practical limits

92 full regression tests passed; an additional resume test passed separately.
Targeted tests cover derivative bounds, synchronized arrival, floor/slow-zone limits,
actual driver clipping telemetry, no sends during checked restart and stale-state rejection.
No commit, push or PR performed. Most recent resume hardening is applied on next server
restart; live motion code is already loaded and tested. Hardware was tested on Linux;
macOS/Windows compatibility is a Ruckig capability, not a platform test performed here.

The arm sensors are sampled around 25 Hz. They cannot rule out higher-frequency
vibration or structural flex. FK is derived from encoders, not external metrology.
The endpoint still has a few millimetres of gravity/tracking offset despite quiet hold.

To reproduce offline comparisons:

```sh
python scripts/compare_motion.py \
  logs/episode-20260905-174738.jsonl:5 \
  logs/episode-20260905-183234.jsonl:5 \
  --output diagnostics/ruckig-repeat-comparison
```

To record on hardware through the already running server:

```sh
metal-arm-harness op monitor 5
metal-arm-harness op trace-tip X Y Z PITCH 5
```

Choose and approve each physical pose from fresh camera views before running it.

## inspect-robots solver question

The CaP-X plugin uses Pyroki IK. Its client sends `target_pose_wxyz_xyz` and the
previous full configuration to an `/ik` endpoint, warm-starts subsequent solves,
and returns the arm joint subset. Other plugins may use different policies;
inspect-robots itself does not mandate one universal IK solver.

Verified sources:
- https://github.com/robocurve/inspect-robots/blob/main/plugins/inspect-robots-capx/src/inspect_robots_capx/_servers.py
- https://github.com/pantor/ruckig
