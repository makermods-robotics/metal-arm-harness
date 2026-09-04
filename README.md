# metal-arm-harness

A small, auditable harness that lets **any agent** — a person at a terminal,
Claude Code, Codex, anything that can run a shell command and look at a JPEG —
control a real robot arm safely. Shipped with the MakerMods Metal arm,
arm-agnostic above the driver layer. No API keys, no model SDKs: the agent
reading this repository *is* the policy.

```bash
# 1. measure the table once (operator rests the gripper tip on it)
metal-arm-harness calibrate --port /dev/cu.usbmodemXXXX --watch

# 2. open one session: connect, set the floor, arm (torque on), answer commands
metal-arm-harness serve --port /dev/cu.usbmodemXXXX --robot-id metal01 \
    --cameras "overhead=2,front=0,wrist=1" --reuse-table --armed --confirm-armed &

# 3. drive it, one command at a time, looking at the frames between moves
metal-arm-harness op observe
metal-arm-harness op tip 0.34 -0.10 0.10 -80     # tool tip x y z (m), pitch (deg)
metal-arm-harness op nudge forward=0.02           # "a little forward"
metal-arm-harness op gripper 112 ; metal-arm-harness op gripper 45
metal-arm-harness op rest ; metal-arm-harness op quit
# or several at once, stopping at the first error:
metal-arm-harness ops "open" "tip 0.44 -0.12 0.034 -80" "close" "tip 0.44 -0.12 0.16 -80"
```

Every command returns the joint angles, the clearance above the table, the
tool-tip pose, and the paths of one fresh JPEG per camera. Every motion goes
through the safety envelope before a single CAN frame goes out. If you are
an agent: read `docs/OPERATING.md` next — it is the procedure, the bench
facts, and the mistakes already made for you.

No hardware? `--backend sim` runs everything against a bench simulator.

## The safety model

Every session begins with the **table-touch ritual**: the operator physically
places the gripper tip on the top of the table, and the tool tip's
forward-kinematics height at that pose is registered as the table surface —
measured on the real machine, not configured. Two samples must agree within
3 mm or the harness refuses to run. `calibrate --watch` needs no keyboard:
it waits until the gripper has been moved and then held still for 5 s.

From that measured floor, the envelope enforces, on every waypoint of every
move, before anything reaches the bus:

1. **Hard floor** — any waypoint whose lowest monitored point (elbow-to-wrist
   origins plus a conservatively long tool tip) would come within 10 mm of
   the table rejects the whole move. The arm cannot be commanded to press
   down on the table.
2. **Recovery, never descent** — if the arm already sits inside that margin
   (gravity sag, or the ritual itself), only moves that keep or raise the
   height are allowed: jaws may close, the arm may lift, nothing may go
   lower. Without this rule the envelope deadlocked on the bench.
3. **Slow zone** — within 3 inches (0.0762 m) of the table, speed drops to
   5 deg/s; free space runs at the full 20 deg/s. Both ends of every step are
   zone-tested, so there is no fast pose from which the arm can dive at the
   table. Gripper-only moves are not geometry and run at full speed.
4. **Joint soft limits** (vendor table, inset), a **35°-per-move excursion
   cap** on arm joints (the gripper is exempt), and runtime vetoes on
   **over-temperature** (70 °C) or non-finite feedback. The operator
   session plays bigger moves as consecutive approved legs.
5. Defense in depth: the LeRobot driver layer keeps its own soft-limit clamp
   and a per-command *lead cap* over the measured position (bench-tuned per
   joint: shoulder 5°, elbow 3°, gripper 8°, others 2°; it bounds torque, so
   it is also what keeps the gripper motor from stalling itself into a
   fault), enforced independently of everything above.

Torque is opt-in (`--armed`), gated behind an operator prompt
(`--confirm-armed` records that the operator said go out of band); unarmed
sessions are real sessions with the motor write removed. Ends of sessions
leave torque as it is — a holding arm keeps holding.

Rejections come back as readable errors with the reason spelled out; they
never take the session down.

## Motion control, learned on the bench

A P-controlled arm lags its command stream and sags under gravity. The
`Controller` (one primitive, `goto`) handles that so callers do not have to:

- unmentioned joints hold their last **commanded** value, not the sagged
  measured one, so error does not accumulate call after call;
- after the last waypoint the target is **settled with integral action**:
  each tick commands `goal + lead`, bounded and re-planned through the
  envelope, so a sagging joint ends up on target: ~0.2° left instead of ~1.3°;
- a gripper that stops short of its target while closing is **holding
  something** — reported, and its closed goal keeps riding along as the
  squeeze; one that stops while opening is reported as blocked;
- each new waypoint stream starts from the **last command sent**, and the
  settle inherits the lag the stream ended with, so torque is continuous
  and the arm no longer dips at the start of a move (the reply reports any
  counter-dip it measured);
- latched driver faults (Damiao status nibble, e.g. rotor overtemp after a
  gripper stall) are surfaced as a veto with the fix, and `op clear-faults`
  clears them; an over-temperature veto also relieves the hot motor;
- `tip` and `nudge` use a damped-least-squares **IK** on the URDF that solves
  in under a millisecond and stays in the current configuration family.

## Architecture

```
arms/base.py      the Arm + Kinematics contracts (degrees, named joints)
arms/metal.py     Metal over LeRobot's DamiaoMotorsBus/MetalFollower + a sim
arms/__init__.py  registry: build_arm("metal", ...) — new arms register here
kinematics.py     Pinocchio FK on the vendored, checksummed URDF (+ tool pose)
ik.py             tip position + pitch → joints, near the current pose
safety.py         the envelope: pure logic, fully unit-tested off-hardware
calibration.py    the table-touch ritual (keyboard or hands-free)
executor.py       plays approved waypoints at the control rate; settle
control.py        Controller.goto: commanded base, legs, settle, no-dip origin
operator.py       the session: Unix-socket server + `op` client + commands
cli.py            serve / op / ops / calibrate
docs/OPERATING.md the procedure for whoever (or whatever) is driving
```

The harness core never imports a concrete arm. To add one: implement `Arm`
(and `Kinematics`, without which the table guard is unavailable) and register
a factory — the envelope, controller, operator session, and CLI are unchanged.

The low level is deliberately reused, not rewritten: `DamiaoMotorsBus` and
`MetalFollower` come from lerobot's `arm/makermods-metal` branch, the CAN/MIT
stack proven under real teleoperation. `--robot-id` names the LeRobot
zero-pose calibration file
(`~/.cache/huggingface/lerobot/calibration/robots/metal_follower/<id>.json`)
that `--armed` requires.

## Development

```bash
uv venv && uv pip install -e . -e <lerobot checkout> pytest
.venv/bin/python -m pytest tests
```

The whole stack runs in CI conditions with no hardware: the sim arm tracks
commands perfectly, the operator session is exercised over a real Unix
socket, and IK is checked against the vendored URDF.
