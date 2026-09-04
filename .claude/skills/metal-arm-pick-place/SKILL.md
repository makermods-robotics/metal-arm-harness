---
name: metal-arm-pick-place
description: Drive the MakerMods Metal arm through metal-arm-harness for pick-and-place or any tabletop manipulation — start the operator session, observe through the three cameras, move with op tip/nudge/gripper, and finish safely. Use whenever the user asks to connect to, calibrate, or move the metal arm, or to pick something up with it.
---

# Metal arm pick-and-place

You are the policy. The harness (`metal-arm-harness`) is the safety layer
and the hands; you look at the camera JPEGs and choose the next command.
Follow `docs/OPERATING.md` in the repository — it is the full procedure,
the bench facts (port, camera indices, robot id), the joint conventions,
and the list of mistakes already made. This skill is the short form.

## Every session

1. Check power and the port: `ls /dev/cu.usbmodem*`. Never open camera
   index 3 (the laptop camera).
2. Only if the arm or table moved: `metal-arm-harness calibrate --port $PORT
   --robot-id metal_metal --watch` and tell the operator to rest the gripper
   tip on the table and hold still.
3. Start unarmed unless the operator has said go:
   `metal-arm-harness serve --port $PORT --robot-id metal_metal
   --cameras "overhead=2,front=0,wrist=1" --reuse-table --frames-dir frames
   [--armed --confirm-armed] &` then `metal-arm-harness op status`.
4. Loop: `op observe` / `op tip` / `op nudge` / `op gripper` → open the
   three JPEGs from the reply → say what you see and what you will do →
   next command. Plan in the `tip (x, y, z) pitch` and `clearance` numbers
   the replies give you.
5. End with `op rest` then `op quit`. The arm keeps holding; do not power it
   off yourself.

## Rules of thumb

- Hover at 6 cm clearance, pitch -80, jaws at 112 before descending.
- Grasp at 20-25 mm clearance around the object's middle; close to ~25°
  under the expected contact angle in one command.
- Lift to ≥ 12 cm clearance before panning with a load.
- "Move a little forward" from the operator = `op nudge forward=0.02`.
- A rejection is information: read it, aim higher or lift first.
- If the reply says the gripper is "holding something", the grasp worked.

## Never

- Bypass the envelope (call the bus directly), disable torque on a held
  arm, or run more than one `serve` at a time.
- Commit, push, or open a PR without the user's explicit confirmation.
