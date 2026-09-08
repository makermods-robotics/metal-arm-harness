# Placement completed: blue left, green and yellow right

Latest user mapping supersedes the earlier green-left task. Hardware: Metal arm via the existing armed operator server, wrist and side cameras. All spatial movements used `op tip` (end-effector XYZ/pitch through harness IK); wrist roll and gripper targets were adjusted separately.

## Result and evidence

- Blue inside left red cup: `frames-reset-fixed/052_after_wrist.jpg`, confirmed again in frame 053. Left cup remained upright throughout subsequent work.
- Green inside right red cup: frame 069.
- Yellow and green together inside right red cup: frames 103–107. Side views confirm upright cups and no blocks remaining on the table.
- Empty arm parked clear at commanded `(0.350, -0.185, 0.200)` m, pitch `-65` degrees. Final encoder FK `(0.348, -0.185, 0.194)` m, pitch `-66` degrees. Arm remains powered, server armed and idle.
- Final read-only hold: 75 samples over 3.024 seconds; reported joint and FK tip peak-to-peak values were all zero at encoder resolution. This stationary observation does not establish vibration-free motion; some moves still reported 0.6–1.0 degree reverse excursions.
- Episode: `logs/episode-20260905-191108.jsonl`.

## Successful poses

Coordinates are commanded metres; angles are degrees. These are records of this scene, not targets to replay without checking fresh camera views.

| Block | Pickup XYZ | Pitch | Wrist roll | Gripper goal | Release XYZ | Release pitch |
|---|---|---:|---:|---:|---|---:|
| Blue | `(0.295, 0.000, 0.024)` | -80 | -20 | 10, relaxed to 16 | `(0.445, 0.105, 0.150)` | -75 |
| Green | `(0.174, -0.098, 0.026)` | -80 | -35 | 15 | `(0.415, -0.270, 0.150)` | -65 |
| Yellow | `(0.309, -0.174, 0.024)` | -75 | 40 | 14 | `(0.425, -0.270, 0.150)` | -70 |

Yellow required several retries. Earlier near-vertical grasps showed initial torque but slipped on short verification lifts. The final grasp captured an exposed end at the fingertips, with flat faces aligned, at pitch -75 and roll 40. Actual gripper position was about 19.3 degrees against goal 14; torque remained about 1.84 N m during successful lifts and transport. Release used gripper goal 70.

The user table reference remained z=0.011 m; server margin remained 0.010 m. Attempts at commanded z=0.021 and 0.022 were rejected by whole-path clearance checks and were not executed. Final yellow pickup used z=0.024. Joint-step rejections were handled by shorter intermediate IK poses without widening limits.

## Settling limit fix from the reset operation

In `executor.py`, clamp `target + proposed_lead` before planning the settling correction. The empty-waypoint fallback now receives the same bounded target used by the safety planner. Previously it could send a value slightly above the wrist-flex limit when the planner clamped that value and returned no waypoints, producing repeated limit corrections and blocking subsequent trajectories.

Regression: `test_settle_sag_at_soft_limit_never_sends_unclamped_empty_path` in `tests/test_control.py`. The 19 existing control tests passed; after fixing the new test's limit-property typo, the added test passed separately. The running server was restarted with the fix before this placement sequence; no recurrence of that limit lockout occurred during the sequence.
