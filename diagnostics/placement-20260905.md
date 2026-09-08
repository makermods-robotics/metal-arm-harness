# Placement session, 2026-09-05

Yellow and blue were visually verified inside the right red cup in
`frames-ruckig/057_after_wrist.jpg` and again after lifting in frame 058.
Green had been placed in the left cup earlier in this task.

The user identified the height at frame 043 as the table and instructed us not
to go lower. The reported encoder-derived tip z was approximately 0.011 m.
Treat this as a minimum fingertip height for subsequent operation, with an
additional tracking allowance; the old saved floor (-0.006715 m) is not a
valid basis for descending below this user-specified limit. The running
server's floor configuration was not changed during this placement sequence.

Diagonal grasps pushed blue out. At the user's direction, wrist roll was
changed to -55 degrees to align the fingers with the block's flat faces.
Successful pick target: (0.285, -0.130, 0.024) m, pitch -80 degrees.
Grip contacted at 37.3 degrees; holding target 34 degrees reduced torque to
approximately 1.5 Nm. Vertical lift to z=0.130 m confirmed the grasp.

The right cup shifted during an earlier failed approach. Its original release
pose must not be reused. Successful blue release target at its new location:
(0.420, -0.310, 0.140) m, pitch -60 degrees, wrist roll -55 degrees,
gripper opened to 70 degrees. Yellow remained inside throughout verification.

Final parking target: (0.330, -0.160, 0.200) m, pitch -60 degrees.
Measured final tip: approximately (0.327, -0.161, 0.195) m, pitch -61 degrees.
Gripper empty, 70 degrees; wrist roll -55 degrees. Three-second read-only hold
recorded 75 samples. Session closed with `op quit`; arm remains powered and
holding, not in the zero-joint rest configuration.

Episode: `logs/episode-20260905-183234.jsonl`.
