# Operating the Metal arm through this harness

This is the procedure for whoever is driving — a person, or an agent that
can run shell commands and look at JPEG files. Read it once, then follow it.
The safety envelope stops you pressing the arm into the table; it does not
stop you wasting twenty minutes. This document is about the second part.

## 0. What you are controlling

Seven joints in degrees: `shoulder_pan shoulder_lift elbow_flex wrist_flex
wrist_yaw wrist_roll gripper`. Base frame: **x forward** (the direction the
gripper points at the zero pose), **y left**, **z up**, origin at the base;
the table top is at about z = 0.02 m. At the zero pose the arm is folded with
the forearm horizontal and the tool tip at about (0.16, 0, 0.19) m.

- `shoulder_lift` 0 = folded back; more negative leans the arm forward and
  down. `elbow_flex` positive raises the forearm. `wrist_flex` negative
  pitches the gripper down. `wrist_roll` rotates the jaws about the tool axis.
- Gripper: 0 closed, 112 is a safe "fully open"; jaws close on a ~4 cm
  object around 70. Grip force = gripper gain (20) x the driver lead cap
  (8° by default ≈ 2.8 N·m). **Do not raise the gripper lead cap**: 30° or
  more stalls the small motor into its rotor-overtemp fault within
  seconds (the feedback shows a 71-73 °C spike, the motor latches off, the
  plug drops). If it happens: `op clear-faults`, then keep 8°. Once the
  squeeze is maintained through later moves (it is now), 8° lifts the
  charger fine. Always `op close` (all the way); the stall detector stops
  the report at contact.
- Pitch in `tip`/`nudge`/`status` is the tool axis angle: 0 horizontal,
  -80 pointing almost straight down. **The floor was measured at -80°**, so
  a near-vertical grasp has an exact floor; shallower pitches are slightly
  conservative.

## 1. Start (once per session, ~1 minute)

```bash
# Ports and cameras on this bench (re-check if USB was replugged:
#   ls /dev/cu.usbmodem*   and   ffmpeg -f avfoundation -list_devices true -i "")
PORT=/dev/cu.usbmodem2067359645431
CAMS="overhead=2,front=0,wrist=1"        # never index 3: the laptop camera
ROBOT_ID=metal_metal                      # LeRobot zero-pose calibration file

# Floor: only if the arm or table moved since the saved ritual.
metal-arm-harness calibrate --port $PORT --robot-id $ROBOT_ID --watch
#   -> operator lifts the arm, rests the gripper tip on the table, holds 5 s.
#   Power must be ON for the handshake; if "motors did not respond", it is off.

# Session. Unarmed first if anything is uncertain; add --armed --confirm-armed
# only after the operator has said go. Lead caps default to the bench-tuned
# table (shoulder_lift 5, elbow_flex 3, gripper 8, others 2).
metal-arm-harness serve --port $PORT --robot-id $ROBOT_ID --cameras "$CAMS" \
    --reuse-table --frames-dir frames --armed --confirm-armed &
metal-arm-harness op status
```

`serve` refuses to start if a camera index no longer maps to a device named
"KD-USB" (`--require-camera-name`): when one USB camera drops off, every
index after it shifts and index 2 becomes the laptop camera. If that
message appears, or the serial device is gone, the USB hub has
disconnected — stop and tell the operator; do not probe other indices.

Several commands in one call, stopping at the first error:
`metal-arm-harness ops "open" "tip 0.44 -0.12 0.034 -80" "close" "tip 0.44 -0.12 0.16 -80"`.
Never chain with `;` or with `| sed` without `set -o pipefail`: a rejected
descent followed by `close` grips air (happened three times).

If the arm rests on the table after the ritual, `op nudge up=0.05` lifts it:
from inside the floor margin only rising or level moves are accepted.

## 2. The loop

Each step is `op <command>` → read the reply → **open the three JPEGs** →
decide the next command. Commands return in ~0.1 s plus motion time.

| command | use |
|---|---|
| `op observe` | fresh frames, no motion |
| `op tip X Y Z PITCH` | absolute tool-tip target (m, deg) via IK |
| `op nudge forward=0.02 left=-0.01 up=0 pitch=-80` | relative move in the arm's horizontal frame; any subset |
| `op goto shoulder_pan=-15 wrist_roll=-9` | raw joints; big moves play as approved legs |
| `op gripper 112` / `op gripper 45` | jaws; a reply saying "holding something" means contact |
| `op rest` | zero pose, jaws unchanged |
| `op clear-faults` | clear a latched motor fault (rotor overtemp etc.) and re-enable |
| `op quit` | end; the arm keeps holding |

Every move reply ends with a residual and, when it happened, `Counter-dip
N deg on <joint>`: the largest excursion away from the target during the
move. It should stay under 0.5°; if it grows, gravity lead is being lost
between commands again (see §5).

Every reply ends with `clearance N mm` and `tip (x, y, z) m pitch P deg`.
Plan in those numbers, not in joint angles.

## 3. Pick and place, the version that works

0. **Map the scene BEFORE grasping.** With empty jaws, hover at ~15 cm
   over the destination (`op tip X Y 0.15 -80`) and read the wrist frame:
   it shows the container around the jaws with no object in the way. Write
   down the container centre in tip coordinates. Do not estimate it from
   the side cameras (§4).
1. **Look from above.** `op tip 0.22 -0.02 0.12 -50`, then read the wrist
   frame. Targets appear where they are relative to the jaws; distances are
   still rough (see §4).
2. **Hover over the object** at ~6 cm clearance, pitch -80, jaws **wide
   open (112)** — jaws only slightly wider than the object land on its edges
   and push it. `op tip X Y 0.08 -80`.
3. **Align.** In the wrist frame the object should sit centred between the
   jaws and square to them. Left/right: `nudge left=±0.01`. Rotation: `goto
   wrist_roll=…`; **negative roll turns the jaws clockwise in the wrist
   image**. The object's near edge should be a little *below* the jaw-tip
   line in the frame at 6 cm height; if it is above, `nudge forward=0.02`.
4. **Descend** to `clearance` ≈ 20-25 mm (`nudge up=-0.04`). Check the front
   camera: jaws should straddle the object's middle, not its near end.
5. **Close** with `op close`. Read the reply. It says one of:
   - `holding something (grip torque X N·m)` — a real grasp; proceed;
   - `holding something but grip torque is only …` — the jaws are resting
     ON the object, not around it: `op open`, go 5-8 mm lower, close again;
   - no stall at all (gripper near 0) — closed on air: re-aim.
   Every later move re-checks the jaws. `OBJECT LOST` in a reply means
   they closed well past the contact angle: the object slipped out. Stop,
   look, re-approach around its middle.
6. **Lift high.** A held object hangs from the jaws and swings ~5 cm below
   the tips; lift to clearance ≥ 12 cm before any pan (`nudge up=0.10`), or
   you will knock the destination.
7. **Move over the destination** (the coordinates from step 0), then lower
   until the object's bottom is ~1 cm above the container floor before
   opening — a level grip hangs ~4.5 cm below the tips, so tips at ~5.5 cm
   clearance. Releasing from 6 cm let the plug tumble off the rim twice.
   Then `op open`, `tip … 0.16`, `op rest`.

Use absolute `op tip X Y Z` for heights. Relative `nudge up=-0.05` from a
mis-remembered height put the jaws 5 cm too high twice.

Operator says "a little forward"? That is `op nudge forward=0.02`. Say what
you see and what you are about to do before each motion.

## 4. Reading the cameras

- **wrist** (index 1) is mounted on the gripper, looking down the jaws. Jaw
  tips appear at the bottom of the frame. The camera sits behind and above
  the tips, so the point on the table directly under the tips appears
  slightly *above* the tip line, more so at height. Image right ≈ -y (more
  negative pan). Rotation of the object relative to the frame's vertical is
  the roll you need to add.
- **front** (index 0) looks across the table from the side: the best view
  for "are the jaws around the middle of the object" and for height.
- **"overhead"** (index 2) is NOT overhead. It sits on the +y side of the
  table looking across: image-right ≈ +x, image-down ≈ -y, and anything
  held in the air projects higher in the frame than its spot on the table
  (a plug at 6 cm looked "inside" the case and landed 4 cm outside it).
  Use it for x and for "did it go in"; never for y of a raised object.
- **front** (index 0): image-left ≈ +y, image-down ≈ +x (closer to it).
- Auto-exposure: the first frame after a long pause may be dark; the session
  discards a few frames per observe already.

## 5. Things that already went wrong (do not repeat)

- **Grid-searching poses** took 40 s each; `tip`/`nudge` solve in 1 ms.
- **One process per command** paid a 2 s CAN handshake and camera warm-up
  every step; `serve` pays once.
- **Jaws at 84° on a 4 cm block** landed on its edges and shoved it 3 cm.
  Open to 112 first.
- **Descending in small increments** sagged the arm under the floor margin
  and locked every move; now the envelope allows rising moves from there,
  and the controller cancels sag, but still plan grasps at 20-25 mm
  clearance, not 10.
- **Gripping off the centre of mass** left the block tilted and swinging;
  it hit the case on the way over. Grip the middle, lift high before panning.
- **Swinging the pan with a load at 5 cm clearance** moved the case. See
  step 6.
- **Handshake failed: "motors did not respond"** means the 24 V supply is
  off — the operator powers the arm off to move it by hand. It is not the
  USB adapter. **"could not open port"** or cameras named "MacBook" at the
  bench indices means the USB hub dropped: stop.
- **The dip at the start of every move** was the new waypoint stream
  starting from the measured pose, discarding the lead the shoulder held
  against gravity, and the settle then holding the bare target. Fixed:
  streams start from the last command sent and the settle inherits the
  lag; the reply reports any remaining counter-dip.
- **Grasp slipped four times in run 2** with the plug clearly between the
  jaws: the controller had relaxed the squeeze after the stall (commanded
  the measured angle) and the stale-base check reset the gripper goal.
  Both fixed; a stalled gripper keeps its closed goal through every later
  move. Raising force instead tripped the motor fault (above).
- **Grasp geometry**: grip the charger's middle at 12-18 mm tip clearance;
  a grip near its top edge popped out of the V-jaws on lift.
