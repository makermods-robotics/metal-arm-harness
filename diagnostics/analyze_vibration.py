"""Read-only analysis of existing motion logs. Never connects to hardware."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = Path("logs/episode-20260905-174738.jsonl")
records = [json.loads(line) for line in path.read_text().splitlines()]
samples = [r for r in records if r["event"] == "motion_sample" and r["move_id"] == 5]
move = [r for r in samples if r["phase"] != "hold"]
hold = [r for r in samples if r["phase"] == "hold"]
t = np.array([r["monotonic_s"] for r in move])
t -= t[0]
q = np.array([r["measured_deg"][:6] for r in move])
c = np.array([r["command_deg"][:6] for r in move])
g = np.array(move[-1]["goal_deg"][:6])
names = move[0]["joint_names"][:6]
fig, axes = plt.subplots(3, 2, figsize=(13, 9))
for i, name in enumerate(names):
    axes[0, 0].plot(t, q[:, i] - q[0, i], label=name)
axes[0, 0].set(title="Recorded joint travel", xlabel="seconds", ylabel="degrees from start")
axes[0, 0].legend(fontsize=7)
for i in (1, 2, 3):
    axes[0, 1].plot(t, q[:, i] - g[i], label=names[i])
axes[0, 1].set(
    title="Near arrival: encoder minus IK joint goal",
    xlim=(2, 5.2),
    ylim=(-4, 4),
    xlabel="seconds",
    ylabel="degrees",
)
axes[0, 1].legend(fontsize=7)
v = np.diff(c, axis=0) / np.diff(t)[:, None]
for i in (0, 1, 2, 3):
    axes[1, 0].plot(t[1:], v[:, i], label=names[i])
axes[1, 0].set(
    title="Requested command speed (before driver clipping)", xlabel="seconds", ylabel="degrees/s"
)
axes[1, 0].legend(fontsize=7)
ht = np.array([r["monotonic_s"] for r in hold])
ht -= ht[0]
hq = np.array([r["measured_deg"][:6] for r in hold])
for i in range(6):
    axes[1, 1].plot(ht, hq[:, i] - hq[0, i], label=names[i])
axes[1, 1].set(
    title="Later 8-second hold: encoder change", xlabel="seconds from hold start", ylabel="degrees"
)
axes[1, 1].legend(fontsize=7)
xyz = np.array([r["tip_m"] for r in move]) * 1000
for i, name in enumerate("xyz"):
    axes[2, 0].plot(t, xyz[:, i] - xyz[-1, i], label=name)
axes[2, 0].set(title="FK tip displacement from final move sample", xlabel="seconds", ylabel="mm")
axes[2, 0].legend()
hxyz = np.array([r["tip_m"] for r in hold]) * 1000
for i, name in enumerate("xyz"):
    axes[2, 1].plot(ht, hxyz[:, i] - hxyz[0, i], label=name)
axes[2, 1].set(title="Later hold: FK tip change", xlabel="seconds from hold start", ylabel="mm")
axes[2, 1].legend()
for ax in axes.flat:
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("diagnostics/vibration-move-5.png", dpi=150)
play = [r for r in move if r["phase"] == "play"]
pt = np.array([r["monotonic_s"] for r in play])
summary = {
    "move_id": 5,
    "play_samples": len(play),
    "play_period_ms_min_median_max": (
        np.array([np.min(np.diff(pt)), np.median(np.diff(pt)), np.max(np.diff(pt))]) * 1000
    ).tolist(),
    "hold_samples": len(hold),
    "hold_period_ms_min_median_max": (
        np.array([np.min(np.diff(ht)), np.median(np.diff(ht)), np.max(np.diff(ht))]) * 1000
    ).tolist(),
    "hold_joint_range_deg": dict(zip(names, np.ptp(hq, axis=0).tolist(), strict=True)),
    "hold_tip_axis_range_mm": np.ptp(hxyz, axis=0).tolist(),
    "hold_command_arm_range_deg": np.ptp([r["command_deg"][:6] for r in hold], axis=0).tolist(),
    "measured_joint_overshoot_past_ik_goal_deg": dict(
        zip(names, np.maximum(0, np.max((q - g) * np.sign(g - q[0]), axis=0)).tolist(), strict=True)
    ),
    "limitations": [
        "Position samples are ~25 Hz, cannot exclude faster vibration or structural flex.",
        "Commands precede driver clipping and do not include velocity feedforward.",
        "Hold recording starts later than the move; "
        "the time gap is deliberately not plotted as observed data.",
    ],
}
Path("diagnostics/vibration-move-5.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
