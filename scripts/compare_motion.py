"""Compare recorded trajectories without hardware access; do not bridge recording gaps.

python scripts/compare_motion.py LOG:MOVE_ID [LOG:MOVE_ID ...] --output diagnostics/comparison
Requires numpy, scipy, matplotlib. Encoder FK cannot measure structural flex.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("motion-comparison"))
    args = parser.parse_args()
    fig, axes = plt.subplots(4, len(args.traces), figsize=(7 * len(args.traces), 12), squeeze=False)
    reports = []
    for col, spec in enumerate(args.traces):
        path, move_id = spec.rsplit(":", 1)
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        rows = [
            r for r in rows if r.get("event") == "motion_sample" and r["move_id"] == int(move_id)
        ]
        # First contiguous recording only: exclude separately requested later monitors.
        ts = np.array([r["monotonic_s"] for r in rows])
        gaps = np.flatnonzero(np.diff(ts) > 0.5)
        if len(gaps):
            rows = rows[: gaps[0] + 1]
        t = np.array([r["monotonic_s"] for r in rows])
        t -= t[0]
        q = np.array([r["measured_deg"][:6] for r in rows])
        c = np.array([r["command_deg"][:6] for r in rows])
        g = np.array(rows[-1]["goal_deg"][:6])
        play = np.array([r["phase"] == "play" for r in rows])
        end = t[play][-1]
        report = {
            "log": path,
            "move_id": int(move_id),
            "samples": len(rows),
            "play_duration_s": float(end),
            "dt_ms_percentiles": np.percentile(np.diff(t) * 1000, [0, 50, 95, 100]).tolist(),
            "overshoot_deg": np.maximum(0, np.max((q - g) * np.sign(g - q[0]), axis=0)).tolist(),
        }
        for j, name in [(1, "shoulder"), (2, "elbow")]:
            ax = axes[j - 1, col]
            ax.plot(t, q[:, j], label=f"{name} encoder")
            ax.plot(t, c[:, j], "--", label="requested position")
            applied = [r.get("last_applied") for r in rows]
            valid = [
                i for i, a in enumerate(applied) if a and a["monotonic_s"] >= rows[0]["monotonic_s"]
            ]
            if valid:
                at = [applied[i]["monotonic_s"] - rows[0]["monotonic_s"] for i in valid]
                ax.plot(
                    at,
                    [applied[i]["position_deg"][j] for i in valid],
                    ":",
                    label="last driver position",
                )
            ax.axhline(g[j], ls=":", color="gray", label="IK goal")
            ax.set_ylabel("degrees")
            ax.legend(fontsize=7)
        dt = np.median(np.diff(t))
        uniform = np.arange(0, t[-1], dt)
        error = np.column_stack([np.interp(uniform, t, q[:, j] - c[:, j]) for j in range(6)])
        if len(uniform) > 40 and 0.5 / dt > 6:
            filtered = sosfiltfilt(
                butter(3, [2, 6], btype="bandpass", fs=1 / dt, output="sos"), error, axis=0
            )
            central = (uniform > 0.5) & (uniform < end - 0.5)
            report["travel_error_2_to_6_hz_rms_deg"] = (
                np.sqrt(np.mean(filtered[central] ** 2, axis=0)).tolist() if central.any() else None
            )
            for j, name in [(1, "shoulder"), (2, "elbow")]:
                axes[2, col].plot(uniform, filtered[:, j], label=name)
            axes[2, col].set_ylabel("2-6 Hz tracking error (deg)")
            axes[2, col].legend()
        hold = np.array([r["phase"] == "hold" for r in rows])
        if hold.any():
            report["hold_joint_range_deg"] = np.ptp(q[hold], axis=0).tolist()
            report["hold_tip_range_mm"] = (
                np.ptp([r["tip_m"] for r in rows if r["phase"] == "hold"], axis=0) * 1000
            ).tolist()
        for j, name in [(1, "shoulder"), (2, "elbow")]:
            axes[3, col].plot(t[1:], np.diff(c[:, j]) / np.diff(t), label=name)
        axes[3, col].set_ylabel("requested speed (deg/s)")
        axes[3, col].legend()
        for ax in axes[:, col]:
            ax.axvline(end, color="orange", label="trajectory end")
            ax.grid(alpha=0.2)
            ax.set_xlabel("seconds")
        axes[0, col].set_title(f"{Path(path).name} move {move_id}")
        reports.append(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    result = {
        "traces": reports,
        "limitations": [
            "Different paths are not controlled A/B tests.",
            "2-6 Hz RMS is a diagnostic band, not proof of a physical cause.",
            "Encoders at ~25 Hz cannot rule out higher frequency vibration or structural flex.",
            "last_applied is the previous driver send, timestamped separately; "
            "values precede wire quantization.",
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
