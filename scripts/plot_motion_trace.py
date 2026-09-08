"""Plot one logged move and its read-only endpoint dwell (no hardware access).

Usage: python scripts/plot_motion_trace.py logs/episode-....jsonl [--move-id N]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--move-id", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    rows = [row for row in rows if row.get("event") == "motion_sample"]
    move_id = args.move_id if args.move_id is not None else max(row["move_id"] for row in rows)
    rows = [row for row in rows if row["move_id"] == move_id]
    t = np.array([row["monotonic_s"] for row in rows])
    t -= t[0]
    measured = np.array([row["measured_deg"] for row in rows])
    command = np.array([row["command_deg"] for row in rows], dtype=float)
    goal = np.array([row["goal_deg"] for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for index, name in enumerate(rows[0]["joint_names"][:6]):
        ax = axes.flat[index]
        ax.plot(t, measured[:, index], label="Encoder")
        ax.plot(t, command[:, index], "--", label="Requested command (before driver limit)")
        ax.plot(t, goal[:, index], ":", label="IK / joint goal")
        ax.set(title=name, ylabel="degrees")
        ax.grid(alpha=0.2)
    ax = axes.flat[6]
    after = np.array([row["phase"] != "play" for row in rows])
    for index, name in enumerate(rows[0]["joint_names"][:6]):
        ax.plot(t[after], (measured - goal)[after, index], label=name)
    ax.set(title="Arm error after planned trajectory", ylabel="degrees")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.axhline(-0.5, color="gray", linestyle=":")
    ax.legend(fontsize=6)
    ax = axes.flat[-1]
    if all("tip_m" in row for row in rows):
        tip = np.array([row["tip_m"] for row in rows]) * 1000
        for index, name in enumerate("xyz"):
            ax.plot(t, tip[:, index] - tip[-1, index], label=name)
        ax.set(title="FK tip displacement from final sample", ylabel="mm")
        ax.legend()
    for phase, color in [("settle", "orange"), ("hold", "green")]:
        times = t[[row["phase"] == phase for row in rows]]
        if len(times):
            for ax in axes.flat:
                ax.axvspan(times[0], times[-1], alpha=0.1, color=color)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("seconds since first sample")
    fig.suptitle(f"Move {move_id}: orange = settling; green = read-only hold")
    fig.tight_layout()
    path = args.log.with_name(f"{args.log.stem}-move-{move_id}.png")
    fig.savefig(path, dpi=150)
    print(path)
    hold = [row for row in rows if row["phase"] == "hold"]
    if hold:
        summary = {
            "move_id": move_id,
            "hold_samples": len(hold),
            "hold_joint_peak_to_peak_deg": dict(
                zip(
                    rows[0]["joint_names"],
                    np.ptp([row["measured_deg"] for row in hold], axis=0).tolist(),
                    strict=True,
                )
            ),
        }
        if "tip_m" in hold[0]:
            summary["hold_tip_axis_peak_to_peak_mm"] = (
                np.ptp([row["tip_m"] for row in hold], axis=0) * 1000
            ).tolist()
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
