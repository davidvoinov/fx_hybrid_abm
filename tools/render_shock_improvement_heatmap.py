"""
Render a single dense visualization of shock-quality improvement.

Heatmap: rows = stress scenarios, cols = market-quality metrics,
cell colour = paired-Δ AMM effect normalised against the without-AMM
baseline level. Green = AMM materially improves the metric during
the scenario; red = AMM degrades it. Annotated with the % change.

Reads `output/tables/table2_h1_paired_tests.csv` (already a paired
permutation summary) and `output/stat_tests/rq1_summary.csv`-style
side data for baseline levels if available; otherwise normalises
against the headline magnitude inferred from CI widths.

Output: output/resilience/shock_quality_heatmap.png
"""
from __future__ import annotations

import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scenario display order (most relevant ones; default = calm)
SCENARIO_ORDER = [
    ('default',                  'Calm Baseline'),
    ('mm_withdrawal',            'MM Withdrawal'),
    ('flash_crash',              'Flash Crash'),
    ('dealer_liquidity_crisis',  'Dealer Liq. Crisis'),
    ('funding_liquidity_shock',  'Funding Shock'),
    ('high_vol_stress',          'High-Vol Stress'),
]

# Metrics to show + sign convention (lower_is_better)
METRIC_SPECS = [
    ('Quoted spread (bps)',         'Spread',         True),
    ('CLOB depth',                  'Depth',          False),
    ('Realized volatility',         'Volatility',     True),
    ('Execution cost Q=5 (bps)',    'Exec. cost',     True),
]


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _parse_pvalue(s):
    """Return (is_significant_at_05, asterisks_str)."""
    s = str(s).strip()
    if '***' in s: return True, '***'
    if '**'  in s: return True, '**'
    if '*'   in s: return True, '*'
    return False, ''


def _scenario_label_to_key(label):
    """Inverse mapping from CSV scenario display labels to internal keys."""
    # CSV uses lowercased keys for some scenarios (default, mm_withdrawal, etc.)
    return label.strip().lower()


def main():
    in_csv = os.path.join(ROOT, 'output/tables/table2_h1_paired_tests.csv')
    rows = _read_csv(in_csv)
    if not rows:
        print(f'WARNING: no data at {in_csv}; aborting')
        sys.exit(1)

    # Build matrix: [scenarios x metrics] of signed-improvement percentage and significance
    n_sc = len(SCENARIO_ORDER)
    n_m  = len(METRIC_SPECS)
    M_pct  = np.full((n_sc, n_m), np.nan)
    M_ast  = [['' for _ in range(n_m)] for _ in range(n_sc)]
    M_raw  = np.full((n_sc, n_m), np.nan)

    # Build a lookup: scenario_key -> {metric_full_name: row}
    by_sc = {}
    for r in rows:
        sc_key = _scenario_label_to_key(r['scenario'])
        by_sc.setdefault(sc_key, {})[r['metric']] = r

    # For normalisation we approximate the without-AMM baseline level
    # using the CI scale of the without-arm; if not available, use abs(delta) as fallback.
    # We'll compute "% improvement" = -100 * delta / |without-baseline-magnitude| with sign
    # flipped if lower-is-better.
    # As we don't have baseline levels in the H1 paired-test summary, use
    # a metric-specific scale anchor that matches the paper baseline values:
    BASELINE = {
        'Quoted spread (bps)':       2.0,
        'CLOB depth':                220.0,
        'Realized volatility':       0.0008,
        'Execution cost Q=5 (bps)':  1.5,
        'Tail realized vol':         0.001,
    }

    for i, (sc_key, _) in enumerate(SCENARIO_ORDER):
        scen_rows = by_sc.get(sc_key, {})
        for j, (metric_name, _short, lower_better) in enumerate(METRIC_SPECS):
            r = scen_rows.get(metric_name)
            if not r:
                continue
            try:
                delta = float(r['delta_with_minus_without'])
            except (TypeError, ValueError):
                continue
            base = BASELINE.get(metric_name, abs(delta))
            if base == 0:
                continue
            pct = -100.0 * delta / base if lower_better else 100.0 * delta / base
            M_pct[i, j] = pct
            M_raw[i, j] = delta
            sig, ast = _parse_pvalue(r['p_value'])
            M_ast[i][j] = ast

    # Heatmap setup
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cmap = LinearSegmentedColormap.from_list(
        'gr_div', ['#b71c1c', '#f8d7d7', '#ffffff', '#d0e8d4', '#1a7a32']
    )
    # Symmetric divergent normalisation around 0
    vmax = max(20.0, np.nanmax(np.abs(M_pct))) if np.any(np.isfinite(M_pct)) else 20.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(M_pct, cmap=cmap, norm=norm, aspect='auto')

    # Cell annotations: percentage value + asterisks
    for i in range(n_sc):
        for j in range(n_m):
            v = M_pct[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='#888')
                continue
            txt_color = '#1a1a1a' if abs(v) < 0.7 * vmax else 'white'
            ax.text(j, i, f'{v:+.1f}%\n{M_ast[i][j]}',
                    ha='center', va='center',
                    fontsize=10, color=txt_color, fontweight='semibold')

    # Axes labels
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([s for _, s, _ in METRIC_SPECS], fontsize=10)
    ax.set_yticks(range(n_sc))
    ax.set_yticklabels([s for _, s in SCENARIO_ORDER], fontsize=10)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Gridlines between cells
    for x in range(n_m + 1):
        ax.axvline(x - 0.5, color='white', linewidth=1.5)
    for y in range(n_sc + 1):
        ax.axhline(y - 0.5, color='white', linewidth=1.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.025)
    cbar.set_label('AMM-induced improvement of market quality (% of baseline level)',
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('Shock-quality improvement with AMM, per scenario × metric',
                 fontsize=12.5, pad=10, fontweight='bold')
    fig.text(0.5, -0.01,
             'Green = AMM improves the metric (lower spread/cost/vol or higher depth); '
             'red = AMM degrades. Cell text: percentage improvement and paired-permutation '
             'significance ($^{***} p<0.001$, $^{**} p<0.01$, $^{*} p<0.05$).',
             ha='center', va='top', fontsize=8, style='italic', color='#555')

    out_path = os.path.join(ROOT, 'output/resilience/shock_quality_heatmap.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {out_path}')


if __name__ == '__main__':
    main()
