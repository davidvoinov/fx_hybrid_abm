"""
Render scatter: peak amplitude (peak_abs_change_pct) vs recovery time (recovery_steps)
for the spread_resilience metric, separated by with-AMM / without-AMM, faceted by scenario.
Illustrates the "AMM improves amplitude but not recovery time" claim.
"""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

SRC = "output/resilience/resilience_priority_metric_points.csv"
OUT = "output/resilience/scatter_amplitude_vs_recovery_spread.png"
METRIC = "spread_resilience"
SCENARIO_LABEL = {
    "mm_withdrawal": "MM Withdrawal",
    "flash_crash": "Flash Crash",
    "dealer_liquidity_crisis": "Dealer Liquidity Crisis",
    "funding_liquidity_shock": "Funding Liquidity Shock",
    "high_vol_stress": "High-Vol Stress",
}

rows = []
with open(SRC) as f:
    for row in csv.DictReader(f):
        if row["metric_name"] != METRIC:
            continue
        try:
            peak = float(row["peak_abs_change_pct"])
            recov = float(row["recovery_steps"])
        except ValueError:
            continue
        if not np.isfinite(peak) or not np.isfinite(recov):
            continue
        rows.append({
            "preset": row["preset"],
            "amm": int(row["amm_enabled"]),
            "peak": peak,
            "recov": recov,
        })

presets = [p for p in SCENARIO_LABEL if any(r["preset"] == p for r in rows)]
fig, axes = plt.subplots(1, len(presets), figsize=(3.0 * len(presets), 3.8), sharey=False)
if len(presets) == 1:
    axes = [axes]

for ax, preset in zip(axes, presets):
    preset_rows = [r for r in rows if r["preset"] == preset]
    y_cap = np.percentile([r["recov"] for r in preset_rows], 97) if preset_rows else None
    for amm_val, color, label in [(0, "#d62728", "without AMM"), (1, "#1f77b4", "with AMM")]:
        sub = [r for r in preset_rows if r["amm"] == amm_val]
        xs = [r["peak"] for r in sub if y_cap is None or r["recov"] <= y_cap]
        ys = [r["recov"] for r in sub if y_cap is None or r["recov"] <= y_cap]
        ax.scatter(xs, ys, s=14, alpha=0.30, color=color, label=label, edgecolor="none")
        if xs:
            mx, my = np.mean(xs), np.mean(ys)
            sx_lo, sx_hi = np.percentile(xs, [25, 75])
            sy_lo, sy_hi = np.percentile(ys, [25, 75])
            ax.errorbar([mx], [my], xerr=[[mx - sx_lo], [sx_hi - mx]],
                        yerr=[[my - sy_lo], [sy_hi - my]],
                        fmt="o", color=color, markersize=9, markeredgecolor="black",
                        markeredgewidth=1.0, ecolor=color, elinewidth=1.5, capsize=3, zorder=6)
    ax.set_xscale("log")
    ax.set_title(SCENARIO_LABEL[preset], fontsize=9)
    ax.set_xlabel("Peak |spread change|, % (log)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)

axes[0].set_ylabel("Recovery time, ticks", fontsize=8)
axes[-1].legend(fontsize=7, loc="upper right", frameon=True)

fig.suptitle("AMM compresses post-shock amplitude but not recovery time (quoted-spread resilience; "
             "circles = group means with IQR; cloud = individual seeds)",
             fontsize=9.5, y=1.03)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
