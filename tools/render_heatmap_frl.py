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
sys.path.insert(0, ROOT)
try:
    from AgentBasedModel.visualization.paper_style import use_paper_style
    use_paper_style()
except Exception:
    pass

# Scenario display order (most relevant ones; default = calm)
SCENARIO_ORDER = [
    ('default',                  'Calm Baseline'),
    ('mm_withdrawal',            'MM Withdrawal'),
    ('flash_crash',              'Flash Crash'),
    ('dealer_liquidity_crisis',  'Dealer Liq. Crisis'),
    ('funding_liquidity_shock',  'Funding Shock'),
    ('high_vol_stress',          'High-Vol Stress'),
]

# Metrics to show (rq1_tests.csv metric_name) + sign convention (lower_is_better)
METRIC_SPECS = [
    ('avg_clob_spread_bps',      'Spread',         True),
    ('avg_clob_depth',           'Depth',          False),
    ('avg_realized_volatility',  'Volatility',     True),
    ('avg_cost_clob_q5',         'Exec. cost',     True),
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


def _pval_ast(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def main():
    # Read the post-fix paired-test battery WITH levels, so the normalisation is
    # honest: % change relative to the actual without-AMM level (not a calm anchor).
    in_csv = os.path.join(ROOT, 'output/stat_tests/rq1_tests.csv')
    rows = _read_csv(in_csv)
    if not rows:
        print(f'WARNING: no data at {in_csv}; aborting')
        sys.exit(1)

    n_sc = len(SCENARIO_ORDER)
    n_m  = len(METRIC_SPECS)
    M_pct = np.full((n_sc, n_m), np.nan)
    M_ast = [['' for _ in range(n_m)] for _ in range(n_sc)]

    by_sc = {}
    for r in rows:
        by_sc.setdefault(r['scenario_key'], {})[r['metric_name']] = r

    for i, (sc_key, _) in enumerate(SCENARIO_ORDER):
        scen_rows = by_sc.get(sc_key, {})
        for j, (metric_name, _short, lower_better) in enumerate(METRIC_SPECS):
            r = scen_rows.get(metric_name)
            if not r:
                continue
            try:
                w = float(r['mean_with_amm']); o = float(r['mean_without_amm'])
            except (TypeError, ValueError):
                continue
            if o == 0:
                continue
            # honest % = reduction of the without-AMM level (gain, for higher-is-better)
            pct = (o - w) / abs(o) * 100.0 if lower_better else (w - o) / abs(o) * 100.0
            M_pct[i, j] = pct
            M_ast[i][j] = _pval_ast(r['permutation_p_value'])

    # Heatmap setup (FRL letter variant: compact, strict, grayscale).
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    # Grayscale: darker = larger AMM improvement. The sign of the effect is
    # carried by the per-cell text (+/-), so a sequential gray suffices.
    cmap = LinearSegmentedColormap.from_list(
        'grays', ['#ffffff', '#d9d9d9', '#969696', '#525252', '#1a1a1a']
    )
    finite = M_pct[np.isfinite(M_pct)]
    vmin = min(0.0, float(np.nanmin(finite))) if finite.size else 0.0
    vmax = max(20.0, float(np.nanmax(finite))) if finite.size else 20.0
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(M_pct, cmap=cmap, norm=norm, aspect='auto')

    # Cell annotations: percentage value + asterisks
    for i in range(n_sc):
        for j in range(n_m):
            v = M_pct[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='#999')
                continue
            txt_color = 'white' if norm(v) > 0.55 else '#111111'
            ax.text(j, i, f'{v:+.1f}%\n{M_ast[i][j]}',
                    ha='center', va='center',
                    fontsize=9.5, color=txt_color)

    # Axes labels
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([s for _, s, _ in METRIC_SPECS], fontsize=9.5)
    ax.set_yticks(range(n_sc))
    ax.set_yticklabels([s for _, s in SCENARIO_ORDER], fontsize=9.5)
    ax.tick_params(top=False, bottom=False, left=False, right=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Thin neutral separators between cells
    for x in range(n_m + 1):
        ax.axvline(x - 0.5, color='#bdbdbd', linewidth=0.6)
    for y in range(n_sc + 1):
        ax.axhline(y - 0.5, color='#bdbdbd', linewidth=0.6)

    # Slim colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02, aspect=22)
    cbar.set_label('AMM improvement (% of without-AMM level)', fontsize=8.5)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_linewidth(0.5)

    # No embedded title/footnote: the LaTeX \caption carries that text.

    out_path = os.path.join(ROOT, 'output/resilience/shock_quality_heatmap_frl.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {out_path}')


if __name__ == '__main__':
    main()
