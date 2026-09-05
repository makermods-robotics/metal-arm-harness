# Maker Arm integration

Maker and Metal share this harness. Select `--arm maker`; the changes are on
`main` and do not require switching branches in the harness or LeRobot.

**Current status: simulator motion and real read-only diagnostics work. Powered
Maker operation is intentionally unavailable until commissioning is complete.**
This is not yet ready for a real pick-and-place task.

## Read-only hardware session

```bash
.venv/bin/metal-arm-harness serve --arm maker --backend slcan \
  --port /dev/cu.usbmodem1101 --robot-id maker01 --cameras '' \
  --diagnostics-only --socket /tmp/maker-harness.sock
# In another terminal:
.venv/bin/metal-arm-harness op inspect --socket /tmp/maker-harness.sock
.venv/bin/metal-arm-harness op quit --socket /tmp/maker-harness.sock
```

Use the current port; re-check after reconnecting USB. One process owns the bus.
`--diagnostics-only` accepts only `inspect` and `quit`; it rejects `--armed`,
skips cameras and the table ritual, and cannot send motion or fault recovery.
`connect` opens LeRobot's bus with `handshake=False`; disconnect preserves torque.
Neither path calls the upstream follower's implicit torque-enable/calibration.

`inspect` serially requests fresh MIT fault status, `MECH_POS`, and `CAN_TIMEOUT`
for all seven motors. Missing replies are reported as `null`, never cached
positions or zero temperature. The fault query is `FF FF FF FF FF FF 00 FB`,
not the fault-clear packet with byte 6 equal to `FF`. Five-byte and padded
eight-byte fault replies are not encoder feedback. Read-only parameter results
are raw motor coordinates, not a verified normalized pose. Late duplicate CAN
replies cannot be distinguished perfectly because the protocol has no request
sequence numbers.

Hardware check on 2026-09-05, `/dev/cu.usbmodem1101`, through `serve` / `op inspect`:

- CAN IDs 1–7 all responded; every fault report was `00000000`.
- Every `MECH_POS` and `CAN_TIMEOUT` query timed out.
- No enable, STOP, fault-clear, zero, parameter-write, protocol-switch, or motion
  command was sent. The session was closed after inspection.
- Local log: `logs/maker-bringup/episode-20260905-012942.jsonl` (gitignored).

The operator confirmed the arm was physically in its resting zero pose during
this check. That reference was not changed or re-saved.

This independently reproduces the SDK's same-day observation. Fault status
proves connectivity, not working encoder access or watchdog support. It does
not identify the firmware version or prove an upgrade alone will fix it.

## Simulator

```bash
.venv/bin/metal-arm-harness serve --arm maker --backend sim --cameras '' \
  --armed --confirm-armed --socket /tmp/maker-sim.sock
.venv/bin/metal-arm-harness op status --socket /tmp/maker-sim.sock
.venv/bin/metal-arm-harness op goto shoulder_pan=5 --socket /tmp/maker-sim.sock
.venv/bin/metal-arm-harness op nudge up=0.01 --socket /tmp/maker-sim.sock
.venv/bin/metal-arm-harness op close --socket /tmp/maker-sim.sock
.venv/bin/metal-arm-harness op open --socket /tmp/maker-sim.sock
.venv/bin/metal-arm-harness op quit --socket /tmp/maker-sim.sock
```

The simulator uses the actual LeRobot Maker follower's limits, gains, sequential
writes, and relative-target clamp. Feedback tracks perfectly and its camera is
a cartoon, not a physically accurate simulation. This exercises the harness,
not loaded motor behavior. Maker's provisional CAD model supports FK, IK,
Cartesian nudges, and conservative floor checks in simulation.

## Maker conventions

CAN IDs 1–7 follow `shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_yaw
wrist_roll gripper`. Shoulder lift and elbow use RS02 (`O1` in LeRobot); the
others use RS00 (`O0`). Classic MIT CAN, 1 Mbps, 25 Hz harness control rate.

Positions are LeRobot motor degrees. LeRobot's zero is the folded resting pose
with open jaws. `open` targets -3 degrees; `close` targets -119.6 degrees. These
are inside the captured soft limits including the harness's 0.5-degree inset.
The generic operator reads endpoints from the arm's metadata. Do not use
Metal's `gripper 112` or `gripper 0` recipes. Maker's default lead cap is 2
degrees on every joint; Metal's bench-tuned overrides are not inherited.

`rest` refuses because no commanded rest target/path has been commissioned;
several Maker limits exclude zero. An arm physically resting at zero is useful
for establishing alignment but does not prove a commanded path back to zero
is safe. Nothing in the integration overwrites saved zeros or calibration.

The SDK CAD is Y-up. The adapter rotates it +90 degrees about X into Z-up;
its zero gripper points approximately -X. Cartesian `forward` follows the
projected tool axis instead of assuming Metal's shoulder-pan heading. The
SDK's grasp centroid differs from fingertip contact, and motor zero/sign
alignment remains provisional. Jaw floor bounds include the whole travel
range without an uncalibrated angle-to-gap interpolation. Provenance and
regeneration instructions are in `src/metal_arm_harness/assets/urdf/PROVENANCE.md`.

## Remaining work before powered tasks

1. Resolve fresh encoder/temperature feedback and watchdog support for the
   installed firmware. The older LeRobot driver reads via fault-clear and can
   keep stale feedback; those paths cannot serve as harness observations.
   The SDK's `Arm.connect()` / `refresh()` use STOP and can release torque;
   they are not read-only substitutes. Firmware changes are a separate step.
2. Implement and test the real adapter's enable, continuous idle hold, feedback
   watchdog, fault handling, and explicit supported-arm shutdown. The harness
   must not promise to keep holding after process exit when a motor-side CAN
   timeout can release an unsupported arm. Current real `enable_torque`,
   `read`, and `clear_faults` fail explicitly before changing motor state.
3. With the arm supported and cameras checked, validate motor signs/offsets
   against the CAD from the confirmed physical zero pose and additional poses.
   Establish a physical fingertip/contact model and commissioned rest target.
4. Measure Maker's own table height, then run bounded single-joint and
   Cartesian checks through `serve` / `op` before handling objects.

Table measurements now live under `~/.metal-arm-harness/tables/`, keyed by arm,
robot ID, and backend. Old `table.json` is left intact but not automatically
reused; remeasure existing Metal sessions once. This prevents Maker or the
simulator from overwriting/reusing another arm's floor.

## Upstream compatibility

The installed LeRobot Metal branch already supplies `RobstrideMotorsBus`.
The Maker follower classes are reused from the Maker branch when installed,
with a pinned Apache-2.0 compatibility copy for installations that have only
Metal. No sibling checkout is modified. Source commit and import-only changes
are documented in `src/metal_arm_harness/_vendor/maker_follower/PROVENANCE.md`.

Run `.venv/bin/python -m pytest tests`. Hardware reads are opt-in through the
operator session; the tests do not access the physical bus.

Validation: the full suite passed (74 tests at that point); the final Maker
module passed all 17 tests after three additional commissioning-gate/timeout
checks. A real Unix-socket simulator session successfully executed `goto`,
`nudge`, `close`, `open`, and `quit`. Changed Python files pass Ruff, and the
wheel build includes the Maker adapter, follower compatibility copy, URDF,
bounds, provenance, and licenses.
