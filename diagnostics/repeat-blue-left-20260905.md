# Recorded repeat placement

Completed: blue in left red cup; green and yellow in right red cup. User confirmed completion. Arm returned to harness rest, frame 165; measured tip (0.160, -0.001, 0.189) m, pitch -1 degree, empty gripper open at 69.9 degrees. Arm remains powered and armed.

All pickup and transport positions used harness tip IK with the existing table floor and motion limits. Direct rest and some long IK transitions were rejected before motion by the 35-degree step limit; shorter IK transitions followed by the built-in rest command succeeded without changing limits.

Commanded successful grasp poses (XYZ metres, pitch degrees):

| Block | XYZ | Pitch | Roll | Grip goal |
|---|---|---|---|---|
| Blue | .270, -.017, .028 | -80 | 45 | 10, relaxed to 32 after contact |
| Green | .205, -.100, .028 | -80 | 55 | 15 |
| Yellow | .260, -.210, .028 | -75 | 55 | 14 |

Yellow contact: actual gripper 18.6 degrees, torque 1.62 Nm. Short lift to .075 m verified grasp; vertical lift to .180 m before transport. Release at (.370, -.315, .150), pitch -65. Frames 157–159 confirm green and yellow inside right cup. Blue release at (.425, .085, .150), pitch -75; frame 130 confirms placement.

Videos are under recordings/:

- blue-left-repeat-20260905/: initial wrist.mp4 and side.mp4, approximately 101.7 seconds. Recorder stopped during the earlier pause; the complete blue placement was NOT captured. This gap was disclosed to the user.
- blue-left-repeat-20260905-part2/: wrist.mp4 and side.mp4, each 4914 frames at 5 fps, 1920x1080, duration 982.8 seconds, zero camera-fetch errors. Contains complete green and yellow placements. Detached recorder stopped cleanly via STOP file on user request during the return to rest. Final rest movement occurred after recording stopped.

ffprobe verified both part2 MP4 containers, frame counts, dimensions and durations. Timestamp JSONL files and summary.json accompany each camera pair. Per-move images and encoder logs remain available in frames-reset-fixed/ and logs/episode-20260905-191108.jsonl.
