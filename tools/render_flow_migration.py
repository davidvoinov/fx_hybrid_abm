"""Render the AMM customer-volume-migration table (Table 6).

The venue shares are ratios of summed successful routed-customer base volume;
they are not zero-filled means over clock ticks.

Uses the shared `tools._table_style.render_table` so the styling matches
all other paper tables exactly.

Produces:
  output/tables/table6_flow_migration.csv
  output/tables/figures/table6_flow_migration.png

Usage:
    python tools/render_flow_migration.py
"""
from __future__ import annotations

import csv
import math
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools._table_style import (
    render_table, HELP_GREEN, HURT_RED,
)

SCENARIO_ORDER = [
    ('mm_withdrawal',           'MM Withdrawal'),
    ('flash_crash',             'Flash Crash'),
    ('dealer_liquidity_crisis', 'Dealer Liq. Crisis'),
    ('funding_liquidity_shock', 'Funding Shock'),
    ('high_vol_stress',         'High-Vol Stress'),
]

INCREASE_STRONG = '#1a9641'
INCREASE_MED    = HELP_GREEN
INCREASE_SOFT   = '#e8f5e9'
DECREASE_STRONG = HURT_RED
DECREASE_SOFT   = '#fce4ec'


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _read_csv(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def build_rows(summary_path: str) -> List[Dict]:
    raw = _read_csv(summary_path)
    out = []
    for sc_key, sc_label in SCENARIO_ORDER:
        b = next((r for r in raw if r['scenario_key'] == sc_key and r['phase'] == 'before'), None)
        d = next((r for r in raw if r['scenario_key'] == sc_key and r['phase'] == 'during'), None)
        a = next((r for r in raw if r['scenario_key'] == sc_key and r['phase'] == 'after'), None)
        if not (b and d and a):
            continue
        amm_b = 100 * _f(b['amm_customer_volume_share'])
        amm_d = 100 * _f(d['amm_customer_volume_share'])
        amm_a = 100 * _f(a['amm_customer_volume_share'])
        mm_b  = 100 * _f(b['mm_share_active']); mm_d = 100 * _f(d['mm_share_active']); mm_a = 100 * _f(a['mm_share_active'])
        delta = amm_d - amm_b
        ratio = amm_d / amm_b if amm_b > 0 else float('nan')
        out.append({
            'scenario': sc_label,
            # raw numerics for downstream callers
            'amm_before_pct': amm_b, 'amm_during_pct': amm_d, 'amm_after_pct':  amm_a,
            'mm_active_before_pct': mm_b, 'mm_active_during_pct': mm_d, 'mm_active_after_pct': mm_a,
            'delta_amm_pp': delta, 'flow_migration_ratio': ratio,
            # also CLOB shares for completeness
            'clob_before_pct': 100*_f(b['clob_customer_volume_share']),
            'clob_during_pct': 100*_f(d['clob_customer_volume_share']),
            'clob_after_pct':  100*_f(a['clob_customer_volume_share']),
        })
    return out


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _color_flow(row, key, val):
    """Per-cell colour rules for the migration table."""
    try:
        v = float(val.split()[0].rstrip('%').rstrip('×').rstrip('p').rstrip('p'))
    except (ValueError, AttributeError):
        return None
    if not math.isfinite(v):
        return None
    if key == 'delta_amm_pp_str':
        if v >= 15: return INCREASE_STRONG
        if v >= 8:  return INCREASE_MED
        if v > 0:   return INCREASE_SOFT
        return None
    if key == 'flow_migration_ratio_str':
        if v >= 1.5: return INCREASE_STRONG
        if v >= 1.3: return INCREASE_MED
        if v > 1.0:  return INCREASE_SOFT
        return None
    if key == 'amm_during_pct_str':
        if v >= 40: return INCREASE_MED
        return None
    if key == 'mm_active_during_pct_str':
        if v <= 30: return DECREASE_STRONG
        if v <= 60: return DECREASE_SOFT
        return None
    return None


def render_png(rows: List[Dict], out_path: str) -> str:
    if not rows:
        return ''
    # Pre-format every numeric to a display string in a parallel set of keys
    display = []
    for r in rows:
        display.append({
            'scenario': r['scenario'],
            'amm_before_pct_str':   f"{r['amm_before_pct']:.1f}%",
            'amm_during_pct_str':   f"{r['amm_during_pct']:.1f}%",
            'amm_after_pct_str':    f"{r['amm_after_pct']:.1f}%",
            'delta_amm_pp_str':     f"{r['delta_amm_pp']:+.1f} pp",
            'flow_migration_ratio_str': f"{r['flow_migration_ratio']:.2f}×",
            'mm_active_during_pct_str': f"{r['mm_active_during_pct']:.1f}%",
            'mm_active_after_pct_str':  f"{r['mm_active_after_pct']:.1f}%",
        })

    col_labels = [
        'Scenario',
        'AMM %\nbefore', 'AMM %\nduring', 'AMM %\nafter',
        'Δ AMM\n(pp)', 'Migration\nratio',
        'MM active %\nduring', 'MM active %\nafter',
    ]
    col_keys = [
        'scenario',
        'amm_before_pct_str', 'amm_during_pct_str', 'amm_after_pct_str',
        'delta_amm_pp_str', 'flow_migration_ratio_str',
        'mm_active_during_pct_str', 'mm_active_after_pct_str',
    ]
    col_widths = [2.4, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4, 1.4]
    return render_table(
        display, col_labels, col_keys, col_widths,
        cell_color_fn=_color_flow,
        header_wrap_lines=2,
        title='Flow Migration from CLOB to AMM during Stress Scenarios',
        footnote=('AMM customer-volume share is the per-seed ratio of summed successful routed base volume; '
                  'MM-active share is a time average. Both are then averaged across 300 paired seeds, with-AMM branch. '
                  'Δ AMM = during − before (pp). Migration ratio = AMM_during / AMM_before.'),
        out_path=out_path,
    )


def main() -> None:
    rows = build_rows('output/stat_tests/h2_phase_summary.csv')
    csv_path = 'output/tables/table6_flow_migration.csv'
    write_csv(rows, csv_path)
    print(f'  ✓ {csv_path}')
    png_path = 'output/tables/figures/table6_flow_migration.png'
    render_png(rows, png_path)
    print(f'  ✓ {png_path}')


if __name__ == '__main__':
    main()
