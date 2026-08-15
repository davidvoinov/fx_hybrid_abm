"""Render one clean summary table: AMM-effect across 200 LHS compositions.

For each (scenario × outcome metric), compute and display:
  - % of compositions where the per-composition Δ is in the helping direction
  - % significantly helping (paired permutation p < 0.05 AND direction OK)
  - % NS (p >= 0.05)
  - % significantly hurting

Two flavours of the table:
  - All 200 LHS compositions
  - Filtered to n_mm >= 1 (the AMM-substitutable regime, n=190)

Usage:
    python tools/render_composition_summary.py
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HEADER_BG = '#37474f'
HEADER_FG = 'white'
ROW_ALT = '#f5f7fa'
HELP_DARK = '#1a9641'
HELP_LIGHT = '#c8e6c9'
HURT_DARK = '#d7191c'
HURT_LIGHT = '#ffcdd2'
NS_GRAY = '#eeeeee'
DPI = 200

OUTCOME_DIRECTION = {
    'avg_clob_spread_bps':       True,   # lower is better
    'avg_clob_depth':            False,  # higher is better
    'avg_realized_volatility':   True,
    'tail_realized_volatility':  True,
    'avg_cost_clob_q5':          True,
}

OUTCOME_LABELS = {
    'avg_clob_spread_bps':       'Quoted spread',
    'avg_clob_depth':            'CLOB depth',
    'avg_realized_volatility':   'Realized vol',
    'tail_realized_volatility':  'Tail realized vol',
    'avg_cost_clob_q5':          'Execution cost (Q=5)',
}

SCENARIO_LABELS = {
    'default':                   'Calm baseline',
    'dealer_liquidity_crisis':   'Dealer liq. crisis',
    'high_vol_stress':           'High-vol stress',
}


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _read_csv(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _summarize(rows: List[Dict], filter_min_nmm: int = 0, alpha: float = 0.05) -> List[Dict]:
    """Per (scenario × outcome), compute direction + significance breakdown."""
    summary = []
    scenarios = ['default', 'dealer_liquidity_crisis', 'high_vol_stress']
    outcomes = list(OUTCOME_DIRECTION.keys())

    for sc in scenarios:
        for o in outcomes:
            d_key = f'{sc}__{o}__delta'
            p_key = f'{sc}__{o}__pvalue'
            lower_better = OUTCOME_DIRECTION[o]
            items = []
            for r in rows:
                try:
                    mm = int(r.get('n_mm', -1))
                except (TypeError, ValueError):
                    continue
                if mm < filter_min_nmm:
                    continue
                d = _f(r.get(d_key))
                p = _f(r.get(p_key))
                if not (math.isfinite(d) and math.isfinite(p)):
                    continue
                items.append((d, p))
            if not items:
                continue
            total = len(items)
            direction_helps = sum(1 for d, _ in items
                                  if (d < 0 and lower_better) or (d > 0 and not lower_better))
            sig = [(d, p) for d, p in items if p < alpha]
            sig_helps = sum(1 for d, _ in sig
                            if (d < 0 and lower_better) or (d > 0 and not lower_better))
            sig_hurts = sum(1 for d, _ in sig
                            if (d > 0 and lower_better) or (d < 0 and not lower_better))
            ns = total - len(sig)
            summary.append({
                'scenario': SCENARIO_LABELS.get(sc, sc),
                'metric': OUTCOME_LABELS.get(o, o),
                'n_total': total,
                'pct_direction_helps': 100 * direction_helps / total,
                'pct_sig_helps': 100 * sig_helps / total,
                'pct_ns': 100 * ns / total,
                'pct_sig_hurts': 100 * sig_hurts / total,
                'lower_better': lower_better,
            })
    return summary


def _color_pct_helps(pct: float, lower_better: bool) -> str:
    """Color scale for % helping cells."""
    if pct >= 95: return HELP_DARK
    if pct >= 75: return HELP_LIGHT
    if pct >= 50: return '#e8f5e9'
    if pct >= 25: return '#fff4e5'
    return HURT_LIGHT


def _color_pct_hurts(pct: float) -> str:
    if pct >= 75: return HURT_DARK
    if pct >= 50: return HURT_LIGHT
    if pct >= 25: return '#fce4ec'
    return None


def _color_pct_ns(pct: float) -> str:
    if pct >= 50: return NS_GRAY
    if pct >= 25: return '#f5f5f5'
    return None


def render_table(summary: List[Dict], title: str, footnote: str, out_path: str) -> str:
    if not summary:
        return ''

    col_labels = ['Scenario', 'Metric', 'n', '% Δ in helping direction',
                  '% significantly helps (p<0.05)', '% NS', '% significantly hurts']
    col_widths = [2.6, 2.6, 0.7, 2.8, 3.2, 1.4, 2.6]

    n_rows = len(summary)
    fig_w = sum(col_widths) + 0.5
    fig_h = 0.55 * (n_rows + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, sum(col_widths))
    ax.set_ylim(0, n_rows + 1)
    ax.invert_yaxis()
    ax.axis('off')

    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Header
    for cx, w, label in zip(col_x, col_widths, col_labels):
        ax.add_patch(Rectangle((cx, 0), w, 1, facecolor=HEADER_BG, edgecolor='white', linewidth=1.2))
        ax.text(cx + w / 2, 0.5, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=HEADER_FG)

    # Body
    for r_idx, row in enumerate(summary):
        y = 1 + r_idx
        for c_idx, (cx, w) in enumerate(zip(col_x, col_widths)):
            base_bg = ROW_ALT if r_idx % 2 == 0 else 'white'
            cell_bg = base_bg
            text = ''
            text_color = '#1a1a1a'
            text_weight = 'normal'

            if c_idx == 0:
                text = row['scenario']
            elif c_idx == 1:
                text = row['metric']
            elif c_idx == 2:
                text = str(row['n_total'])
            elif c_idx == 3:
                pct = row['pct_direction_helps']
                cell_bg = _color_pct_helps(pct, row['lower_better']) or base_bg
                text = f'{pct:.1f}%'
                text_weight = 'bold' if pct >= 95 else 'normal'
                if pct >= 95:
                    text_color = 'white'
            elif c_idx == 4:
                pct = row['pct_sig_helps']
                cell_bg = _color_pct_helps(pct, row['lower_better']) or base_bg
                text = f'{pct:.1f}%'
                text_weight = 'bold' if pct >= 95 else 'normal'
                if pct >= 95:
                    text_color = 'white'
            elif c_idx == 5:
                pct = row['pct_ns']
                cell_bg = _color_pct_ns(pct) or base_bg
                text = f'{pct:.1f}%'
            elif c_idx == 6:
                pct = row['pct_sig_hurts']
                cell_bg = _color_pct_hurts(pct) or base_bg
                text = f'{pct:.1f}%'
                if pct >= 75:
                    text_color = 'white'
                    text_weight = 'bold'

            ax.add_patch(Rectangle((cx, y), w, 1, facecolor=cell_bg,
                                   edgecolor='#cfd8dc', linewidth=0.6))
            ax.text(cx + w / 2, y + 0.5, text, ha='center', va='center',
                    fontsize=10, color=text_color, fontweight=text_weight)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    if footnote:
        fig.text(0.5, 0.005, footnote, ha='center', va='bottom',
                 fontsize=9, style='italic', color='#555555')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in-csv', default='output/composition/analysis/amm_effect_by_point.csv')
    p.add_argument('--out-dir', default='output/tables/figures')
    args = p.parse_args()

    rows = _read_csv(args.in_csv)

    # All 200 compositions
    s_all = _summarize(rows, filter_min_nmm=0)
    p1 = render_table(
        s_all,
        title='AMM-effect across 200 LHS agent compositions (full sample, including n_mm = 0)',
        footnote=(
            'Per-composition paired permutation test (50 seeds × 5 000 sign-flips); α = 0.05. '
            '"Helping direction" = Δ has the empirically beneficial sign for that metric '
            '(spread / vol / cost ↓, depth ↑).'
        ),
        out_path=os.path.join(args.out_dir, 'composition_summary_all.png'),
    )
    print(f'  ✓ {p1}')

    # n_mm >= 1 only (190 compositions — viable dealer baseline regime)
    s_filt = _summarize(rows, filter_min_nmm=1)
    p2 = render_table(
        s_filt,
        title='AMM-effect across 190 LHS compositions with n_mm ≥ 1 (viable dealer baseline)',
        footnote=(
            'Filtered to compositions with at least one market-maker agent (the regime where AMM '
            'acts as a complement to a working CLOB). n_mm = 0 cases (n = 10) excluded as they '
            'represent degenerate markets where no dealer-side liquidity exists for AMM to support.'
        ),
        out_path=os.path.join(args.out_dir, 'composition_summary_nmm_ge1.png'),
    )
    print(f'  ✓ {p2}')


if __name__ == '__main__':
    main()
