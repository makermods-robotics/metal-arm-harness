# Metal arm session diagnostics

These files record the September 5, 2026 bench sessions. Recorded poses and
statements about the arm's state describe those sessions, not its current state.
Use fresh camera observations and the commissioned calibration before moving.

- `vibration-findings.md`: encoder evidence and control/driver hypotheses.
- `ruckig-hardware-results.md`: measured trajectory improvements, test limits,
  thermal recovery, and the distinction between IK and trajectory generation.
- `*.json` and `*.png`: derived metrics and plots from the local episode logs.
- `placement*.md` and `repeat-blue-left-20260905.md`: placement evidence and
  recording coverage, including the disclosed gaps in the early recordings.

Raw episode logs, camera frames, and videos remain local in the ignored `logs/`,
`frames*/`, and `recordings/` directories. Paths to those artifacts in these notes
are references to the original bench machine, not files shipped in this repository.
The plotting scripts require those source logs plus matplotlib; the comparison
script also requires scipy. The independent camera recorder requires OpenCV.

## Companion driver adjustment on the bench

The local LeRobot checkout also changed the Damiao reply collection timeout to
read `METAL_CAN_REPLY_TIMEOUT_S`, keeping the upstream 0.01-second default.
`lerobot-can-reply-timeout.patch` preserves that small dependency change against
LeRobot revision `85a3000d4c8b852e6e37f84d42b8d94ce7f47ae9` for reproducibility.
It is not automatically applied by installing this harness. It allows the Lapis
USB/CAN setup to use a longer reply window without changing other hosts' defaults.
Tests for this PR use the declared upstream dependency without this local patch.

The harness's Ruckig trajectory, position-only command, settling, telemetry, and
resume changes are implemented directly in `src/metal_arm_harness/`.
