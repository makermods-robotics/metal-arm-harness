"""`metal-arm-harness serve|op|calibrate` — the command line.

`serve` + `op` are the operator session (one connection, many commands; see
`operator.py`): whoever runs `op` — a person, or an agent that has read this
repository — is the policy. `calibrate` performs the table-touch ritual on
its own, so the floor can be measured before anything else.

Session order is fixed and safety-motivated:

1. connect (reads only; torque untouched),
2. the table-touch ritual — the operator puts the gripper tip on the table
   top and that measured height becomes the hard floor,
3. only if --armed: an operator gate, then torque on,
4. commands are answered, each through the safety envelope,
5. close (a holding arm keeps holding).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from metal_arm_harness.arms import build_arm
from metal_arm_harness.calibration import (
    load_calibration,
    run_table_ritual,
    save_calibration,
    watch_for_touch,
)
from metal_arm_harness.control import Controller
from metal_arm_harness.episode_log import EpisodeLog
from metal_arm_harness.operator import (
    DEFAULT_SOCKET,
    OperatorError,
    OperatorSession,
    print_reply,
    send,
    serve,
)
from metal_arm_harness.safety import SafetyConfig, SafetyEnvelope

_DEFAULT_CALIBRATION = Path.home() / ".metal-arm-harness" / "table.json"


def _lead_cap(text: str) -> float | dict[str, float]:
    """`3` for every joint, or `shoulder_lift=5,elbow_flex=3` (others keep the default)."""
    if "=" not in text:
        return float(text)
    out: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        out[name.strip()] = float(value)
    return out


def _add_arm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", default="metal", help="registered arm name (default: metal)")
    parser.add_argument("--backend", default="slcan", choices=["slcan", "socketcan", "sim"])
    parser.add_argument("--port", default=None, help="slcan serial device or socketcan interface")
    parser.add_argument(
        "--robot-id",
        default="metal_arm",
        help=(
            "LeRobot robot id; selects the zero-pose calibration file "
            "~/.cache/huggingface/lerobot/calibration/robots/metal_follower/<id>.json "
            "that --armed requires (default: metal_arm)"
        ),
    )
    parser.add_argument(
        "--cameras", default="overhead=0", help='e.g. "overhead=0,wrist=1"; "" for none'
    )
    parser.add_argument(
        "--require-camera-name",
        default="KD-USB",
        help=(
            "refuse to open a camera index whose device name lacks this text (macOS, needs "
            "ffmpeg); guards against index shifts onto the laptop camera. '' disables"
        ),
    )
    parser.add_argument(
        "--lead-cap",
        type=_lead_cap,
        default=None,
        help=(
            "driver-layer cap on how far a command may lead the measured position, deg: "
            "a number for every joint, or overrides like shoulder_lift=5,gripper=8 on top of "
            "the bench-tuned defaults (shoulder_lift 5, elbow_flex 3, gripper 8, others 2)"
        ),
    )


def _add_floor_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reuse-table",
        action="store_true",
        help=f"skip the table ritual and reuse {_DEFAULT_CALIBRATION} (only if nothing moved)",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=None,
        help="explicit table height in metres (sim/testing; skips the ritual)",
    )


def _add_motion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--armed", action="store_true", help="allow motor writes (default: dry run)"
    )
    parser.add_argument("--max-speed", type=float, default=20.0, help="deg/s ceiling in free space")
    parser.add_argument(
        "--slow-speed", type=float, default=5.0, help="deg/s ceiling near the table"
    )
    parser.add_argument("--log-dir", default="logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metal-arm-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve", help="hold one connection open and answer `op` commands (operator as policy)"
    )
    _add_arm_options(serve)
    _add_motion_options(serve)
    _add_floor_options(serve)
    serve.add_argument(
        "--confirm-armed",
        action="store_true",
        help="skip the ENTER gate for --armed (the operator has confirmed out of band)",
    )
    serve.add_argument("--frames-dir", default="frames", help="where observe() saves JPEGs")
    serve.add_argument("--socket", default=str(DEFAULT_SOCKET))

    op = sub.add_parser("op", help="send one command to a running `serve` session")
    op.add_argument("op_command", help="observe|goto|tip|nudge|gripper|open|close|rest|status|quit")
    op.add_argument("op_args", nargs="*")
    op.add_argument("--socket", default=str(DEFAULT_SOCKET))

    ops = sub.add_parser(
        "ops",
        help=(
            'several op commands in order, e.g. ops "open" "tip 0.4 -0.1 0.034 -80" "close"; '
            "stops at the first error"
        ),
    )
    ops.add_argument("op_commands", nargs="+")
    ops.add_argument("--socket", default=str(DEFAULT_SOCKET))

    calibrate = sub.add_parser("calibrate", help="table-touch ritual only; saves the floor")
    _add_arm_options(calibrate)
    calibrate.add_argument(
        "--watch",
        action="store_true",
        help="no keyboard: wait until the gripper is moved and then held still 5 s",
    )
    return parser


def _set_floor(
    args: argparse.Namespace, arm: Any, kinematics: Any, safety: SafetyEnvelope, log: EpisodeLog
) -> None:
    """The hard floor, from an explicit value, a saved ritual, the sim, or the ritual."""
    if kinematics is None:
        return
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


def _torque_gate(arm: Any, armed: bool, confirmed: bool = False) -> None:
    if not armed:
        print("running UNARMED: moves are computed and logged, never sent")
        return
    if not confirmed:
        input(
            f"\nAbout to energize {len(arm.info.joints)} motors with firm gains. "
            "Clear the workspace and keep a hand on the power switch.\n"
            "Press ENTER to arm (Ctrl-C aborts)... "
        )
    arm.enable_torque()


def main_serve(args: argparse.Namespace) -> int:
    if args.cameras and args.require_camera_name:
        from metal_arm_harness.camera import check_device_names

        try:
            check_device_names(args.cameras, args.require_camera_name)
        except RuntimeError as error:
            print(f"{error}", file=sys.stderr)
            return 2
    safety_config = SafetyConfig(max_speed_deg_s=args.max_speed, slow_speed_deg_s=args.slow_speed)
    arm, kinematics = build_arm(
        args.arm,
        backend=args.backend,
        port=args.port,
        robot_id=args.robot_id,
        cameras=args.cameras,
        **({"lead_cap_deg": args.lead_cap} if args.lead_cap else {}),
    )
    safety = SafetyEnvelope(safety_config, arm.info, kinematics)
    log = EpisodeLog(args.log_dir)
    try:
        arm.connect()
        _set_floor(args, arm, kinematics, safety, log)
        _torque_gate(arm, args.armed, args.confirm_armed)
        session = OperatorSession(
            arm,
            kinematics,
            safety,
            Controller(arm, safety, armed=args.armed, log=log),
            frames_dir=Path(args.frames_dir),
            log=log,
        )
        print(session.handle("status", [])["text"])
        serve(session, Path(args.socket))
        return 0
    finally:
        log.close()
        arm.close()


def main_op(args: argparse.Namespace) -> int:
    try:
        reply = send(args.op_command, list(args.op_args), Path(args.socket))
    except OperatorError as error:
        print(f"{error}", file=sys.stderr)
        return 2
    return print_reply(reply)


def main_ops(args: argparse.Namespace) -> int:
    for index, line in enumerate(args.op_commands, 1):
        parts = line.split()
        if not parts:
            continue
        print(f"[{index}/{len(args.op_commands)}] {line}")
        try:
            reply = send(parts[0], parts[1:], Path(args.socket))
        except OperatorError as error:
            print(f"{error}", file=sys.stderr)
            return 2
        code = print_reply(reply)
        if code:
            print(f"stopped at command {index}: {line}", file=sys.stderr)
            return code
    return 0


def main_calibrate(args: argparse.Namespace) -> int:
    arm, kinematics = build_arm(
        args.arm,
        backend=args.backend,
        port=args.port,
        robot_id=args.robot_id,
        cameras="",
        **({"lead_cap_deg": args.lead_cap} if args.lead_cap else {}),
    )
    if kinematics is None:
        print("this arm has no kinematics; there is no floor to calibrate", file=sys.stderr)
        return 2
    try:
        arm.connect()
        ask = watch_for_touch(arm, kinematics) if args.watch else input
        calibration = run_table_ritual(arm, kinematics, ask=ask)
        save_calibration(calibration, _DEFAULT_CALIBRATION)
        print(f"saved {_DEFAULT_CALIBRATION}: {calibration.to_json()}")
        return 0
    finally:
        arm.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return main_serve(args)
    if args.command == "op":
        return main_op(args)
    if args.command == "ops":
        return main_ops(args)
    return main_calibrate(args)


if __name__ == "__main__":
    sys.exit(main())
