# metal-arm-harness

A small, auditable harness that lets an LLM control a real robot arm —
shipped with the MakerMods Metal arm, arm-agnostic above the driver layer.

The model acts through four tools (`move_joints`, `look`, `done`, `give_up`),
sees labeled joint state, its clearance above the table, and camera frames
after every call, and everything it requests passes through a physical safety
envelope before a single CAN frame goes out.

```bash
# simulator, no hardware, no motion
metal-arm-harness run "pick up the blue cube" --backend sim

# real arm, dry run: real feedback, moves computed and logged, never sent
metal-arm-harness run "pick up the blue cube" --port /dev/cu.usbmodemXXXX

# real arm, torque on (operator-gated)
metal-arm-harness run "pick up the blue cube" --port /dev/cu.usbmodemXXXX --armed
```

Requires `ANTHROPIC_API_KEY`; default model is `claude-opus-5`.

## The safety model

Every session begins with the **table-touch ritual**: the operator physically
places the gripper tip on the top of the table, and the tool tip's
forward-kinematics height at that pose is registered as the table surface —
measured on the real machine, not configured. Two samples must agree within
3 mm or the harness refuses to run.

From that measured floor, the envelope enforces, on every waypoint of every
move, before anything reaches the bus:

1. **Hard floor** — any waypoint whose lowest monitored point (elbow-to-wrist
   origins plus a conservatively long tool tip) would come within 10 mm of
   the table rejects the whole move. The arm cannot be commanded to press
   down on the table.
2. **Slow zone** — within 3 inches (0.0762 m) of the table, speed drops to
   5 deg/s; free space runs at the full 20 deg/s. Both ends of every step are
   zone-tested, so there is no fast pose from which the arm can dive at the
   table.
3. **Joint soft limits** (vendor table, inset), a **35°-per-call excursion
   cap** (distance between camera looks stays bounded), and runtime vetoes on
   **over-temperature** (70 °C) or non-finite feedback.
4. Defense in depth: the LeRobot driver layer keeps its own soft-limit clamp
   and per-tick relative-target cap, enforced against *measured* position,
   independent of everything above.

Torque is opt-in (`--armed`), gated behind an operator prompt; unarmed runs
are real runs with the motor write removed. Ends of runs leave torque as it
is — a holding arm keeps holding.

Rejections are returned to the model as correctable tool errors with the
reason spelled out; they never crash the episode.

## Architecture

```
arms/base.py      the Arm + Kinematics contracts (degrees, named joints)
arms/metal.py     Metal over LeRobot's DamiaoMotorsBus/MetalFollower + a sim
arms/__init__.py  registry: build_arm("metal", ...) — new arms register here
kinematics.py     Pinocchio FK on the vendored, checksummed URDF
safety.py         the envelope: pure logic, fully unit-tested off-hardware
calibration.py    the table-touch ritual
executor.py       plays approved waypoints at the control rate
agent.py          the Anthropic tool-use loop (transport injectable)
cli.py            session order: connect → ritual → gate → run → close
```

The harness core never imports a concrete arm. To add one: implement `Arm`
(and `Kinematics`, without which the table guard is unavailable) and register
a factory — the agent, safety envelope, executor, and CLI are unchanged.

The low level is deliberately reused, not rewritten: `DamiaoMotorsBus` and
`MetalFollower` come from lerobot's `arm/makermods-metal` branch, the CAN/MIT
stack proven under real teleoperation.

## Development

```bash
uv venv && uv pip install -e . -e <lerobot checkout> pytest
.venv/bin/python -m pytest tests
```

The full agent loop runs in CI conditions with no hardware, no network, and
no API key: the sim arm tracks commands perfectly, and tests script the model
through `httpx.MockTransport`.
