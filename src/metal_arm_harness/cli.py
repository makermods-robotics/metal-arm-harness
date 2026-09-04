"""`metal-arm-harness run "task" [options]` — the one entry point.

Session order is fixed and safety-motivated:

1. connect (reads only; torque untouched),
2. the table-touch ritual — the operator puts the gripper tip on the table
   top and that measured height becomes the hard floor,
3. only if --armed: an operator gate, then torque on,
4. the agent runs the task,
5. close (a holding arm keeps holding).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from metal_arm_harness.agent import DEFAULT_MODEL, AgentConfig, ArmAgent
from metal_arm_harness.arms import build_arm
from metal_arm_harness.calibration import (
    load_calibration,
    run_table_ritual,
    save_calibration,
)
from metal_arm_harness.episode_log import EpisodeLog
from metal_arm_harness.safety import SafetyConfig, SafetyEnvelope

_DEFAULT_CALIBRATION = Path.home() / ".metal-arm-harness" / "table.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metal-arm-harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one task episode")
    run.add_argument("instruction", help="natural-language task for the arm")
    run.add_argument("--arm", default="metal", help="registered arm name (default: metal)")
    run.add_argument("--backend", default="slcan", choices=["slcan", "socketcan", "sim"])
    run.add_argument("--port", default=None, help="slcan serial device or socketcan interface")
    run.add_argument(
        "--cameras", default="overhead=0", help='e.g. "overhead=0,wrist=1"; "" for none'
    )
    run.add_argument("--armed", action="store_true", help="allow motor writes (default: dry run)")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--max-llm-calls", type=int, default=40)
    run.add_argument("--max-speed", type=float, default=20.0, help="deg/s ceiling in free space")
    run.add_argument("--slow-speed", type=float, default=5.0, help="deg/s ceiling near the table")
    run.add_argument("--log-dir", default="logs")
    run.add_argument(
        "--reuse-table",
        action="store_true",
        help=f"skip the table ritual and reuse {_DEFAULT_CALIBRATION} (only if nothing moved)",
    )
    run.add_argument(
        "--table-z",
        type=float,
        default=None,
        help="explicit table height in metres (sim/testing; skips the ritual)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set; the agent cannot call the model.\n"
            "fix: export ANTHROPIC_API_KEY=... and rerun",
            file=sys.stderr,
        )
        return 2

    safety_config = SafetyConfig(
        max_speed_deg_s=args.max_speed, slow_speed_deg_s=args.slow_speed
    )
    arm, kinematics = build_arm(
        args.arm,
        backend=args.backend,
        port=args.port,
        cameras=args.cameras,
        max_step_deg=args.max_speed / 25.0,
    )
    safety = SafetyEnvelope(safety_config, arm.info, kinematics)
    log = EpisodeLog(args.log_dir)

    try:
        arm.connect()

        # ── the hard floor ──────────────────────────────────────────────────
        if kinematics is not None:
            if args.table_z is not None:
                safety.set_floor(args.table_z)
                print(f"table height set explicitly: z={args.table_z:.4f} m")
            elif args.reuse_table:
                calibration = load_calibration(_DEFAULT_CALIBRATION)
                safety.set_floor(calibration.floor_z_m)
                print(
                    f"reusing table calibration z={calibration.floor_z_m:.4f} m "
                    f"from {_DEFAULT_CALIBRATION} — only valid if the arm and table "
                    "have not moved since"
                )
            elif args.backend == "sim":
                # No physical table to touch: put a synthetic one 10 cm below
                # the sim arm's lowest starting point.
                floor = kinematics.min_height_m(arm.read().positions_deg) - 0.10
                safety.set_floor(floor)
                print(f"sim: synthetic table at z={floor:.4f} m")
            else:
                calibration = run_table_ritual(arm, kinematics)
                save_calibration(calibration, _DEFAULT_CALIBRATION)
                safety.set_floor(calibration.floor_z_m)
            log.event("floor_set", floor_z_m=safety.floor_z_m)

        # ── torque gate ─────────────────────────────────────────────────────
        if args.armed:
            input(
                f"\nAbout to energize {len(arm.info.joints)} motors with firm gains. "
                "Clear the workspace and keep a hand on the power switch.\n"
                "Press ENTER to arm (Ctrl-C aborts)... "
            )
            arm.enable_torque()
        else:
            print("running UNARMED: moves are computed and logged, never sent")

        agent = ArmAgent(
            arm,
            safety,
            AgentConfig(model=args.model, max_llm_calls=args.max_llm_calls),
            armed=args.armed,
            log=log,
        )
        result = agent.run(args.instruction)
        print(
            f"\nepisode {result.status}: {result.detail}\n"
            f"  llm_calls={result.llm_calls} moves={result.moves} "
            f"rejections={result.rejections}"
        )
        if result.hindsight:
            print(f"  hindsight: {result.hindsight}")
        if log.path:
            print(f"  log: {log.path}")
        return 0 if result.status == "done" else 1
    finally:
        log.close()
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
