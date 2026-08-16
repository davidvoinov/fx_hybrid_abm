"""Standalone plot regenerator.

Reads the CSVs produced by `tools/resilience_study.py` and `tools/stat_study.py`
and (re-)builds every individual plot in a clean per-type subfolder layout:

  output/resilience/scatter/{scenario}__{metric}.png
  output/resilience/km_curves/{scenario}__{metric}.png
  output/resilience/peak_impact/{metric}.png
  output/resilience/normalized_impact/{metric}.png
  output/resilience/recovery_boxplots/{metric}.png

  output/stat_tests/rq1/{metric}.png
  output/stat_tests/rq2/{metric}.png
  output/stat_tests/h2_phase/{metric}.png
  output/stat_tests/mm_behavior/heatmap.png

Use:
    python tools/regenerate_plots.py
    python tools/regenerate_plots.py --resilience-dir output/resilience --stat-dir output/stat_tests
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Make AgentBasedModel importable when running from the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')

from AgentBasedModel.visualization.paper_style import use_paper_style

use_paper_style()

from AgentBasedModel.visualization.individual_plots import (
    SCENARIOS,
    PRIORITY_METRICS,
    plot_recovery_scatter,
    plot_km_curve,
    plot_peak_impact_bars,
    plot_paired_delta_strip,
    plot_recovery_boxplot,
    plot_rq_effect,
    plot_h2_phase,
    plot_mm_behavior_heatmap,
    _scenario_label,
    _safe_float,
)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _read_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _slug(text: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in text.lower()).strip('_')


# ---------------------------------------------------------------------------
# Resilience: scatter & KM (per scenario × per metric)
# ---------------------------------------------------------------------------

def _group_priority_points(rows: List[dict]) -> Dict[Tuple[str, str], Dict[str, List[dict]]]:
    """Index priority-metric points by (panel_label, metric_name) → series_key → [points]."""
    out: Dict[Tuple[str, str], Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(r['panel_label'], r['metric_name'])][r['series_key']].append(r)
    return out


def _group_price_points(rows: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    """Index price scatter points by panel_label → series_key → [points]."""
    out: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r['panel_label']][r['series_key']].append(r)
    return out


def regenerate_resilience(resilience_dir: str) -> List[str]:
    paths_written: List[str] = []

    priority_points = _read_rows(os.path.join(resilience_dir, 'resilience_priority_metric_points.csv'))
    price_points = _read_rows(os.path.join(resilience_dir, 'resilience_scatter_points.csv'))

    if not priority_points and not price_points:
        print(f'  [WARN] No CSVs found in {resilience_dir}, skipping')
        return paths_written

    grouped = _group_priority_points(priority_points)
    price_grouped = _group_price_points(price_points)

    # 1. Scatter: per (scenario, metric) for spread/depth/cost/composite
    scatter_dir = os.path.join(resilience_dir, 'scatter')
    for (panel, metric_name), series in grouped.items():
        metric_label = dict(PRIORITY_METRICS).get(metric_name, metric_name)
        out_path = os.path.join(scatter_dir, f'{_slug(panel)}__{_slug(metric_name)}.png')
        plot_recovery_scatter(panel, metric_label,
                              series.get('with_amm', []), series.get('without_amm', []),
                              out_path)
        paths_written.append(out_path)

    # 1b. Scatter: per scenario for PRICE (separate from priority metrics)
    for panel, series in price_grouped.items():
        out_path = os.path.join(scatter_dir, f'{_slug(panel)}__price.png')
        plot_recovery_scatter(panel, 'Price',
                              series.get('with_amm', []), series.get('without_amm', []),
                              out_path)
        paths_written.append(out_path)

    # 2. KM: per (scenario, metric)
    km_dir = os.path.join(resilience_dir, 'km_curves')
    for (panel, metric_name), series in grouped.items():
        metric_label = dict(PRIORITY_METRICS).get(metric_name, metric_name)
        out_path = os.path.join(km_dir, f'{_slug(panel)}__{_slug(metric_name)}.png')
        plot_km_curve(panel, metric_label,
                      series.get('with_amm', []), series.get('without_amm', []),
                      out_path)
        paths_written.append(out_path)
    # KM for price
    for panel, series in price_grouped.items():
        out_path = os.path.join(km_dir, f'{_slug(panel)}__price.png')
        plot_km_curve(panel, 'Price',
                      series.get('with_amm', []), series.get('without_amm', []),
                      out_path)
        paths_written.append(out_path)

    # 3. Peak-impact bars per metric (across scenarios)
    peak_dir = os.path.join(resilience_dir, 'peak_impact')
    for metric_name, metric_label in PRIORITY_METRICS:
        per_scenario = []
        for sc_key, sc_label in SCENARIOS:
            panel_label = sc_label
            with_pts = grouped.get((panel_label, metric_name), {}).get('with_amm', [])
            without_pts = grouped.get((panel_label, metric_name), {}).get('without_amm', [])
            if not with_pts and not without_pts:
                continue
            per_scenario.append({
                'scenario_key': sc_key,
                'scenario_label': sc_label,
                'with_values': [_safe_float(p.get('peak_abs_change_pct')) for p in with_pts],
                'without_values': [_safe_float(p.get('peak_abs_change_pct')) for p in without_pts],
            })
        if not per_scenario:
            continue
        out_path = os.path.join(peak_dir, f'{_slug(metric_name)}.png')
        _PEAK_YLABEL = {
            'spread_resilience':         'Peak spread spike |%| from baseline',
            'depth_resilience':          'Peak depth drawdown |%| from baseline',
            'execution_cost_resilience': 'Peak execution-cost spike |%| from baseline',
        }
        plot_peak_impact_bars(metric_label, per_scenario, out_path,
                              field='peak_abs_change_pct',
                              y_label_override=_PEAK_YLABEL.get(metric_name))
        paths_written.append(out_path)

    # 4. Paired Δ for normalized_avg_impact per metric
    delta_dir = os.path.join(resilience_dir, 'normalized_impact')
    for metric_name, metric_label in PRIORITY_METRICS:
        per_scenario = []
        for sc_key, sc_label in SCENARIOS:
            panel_label = sc_label
            with_pts = grouped.get((panel_label, metric_name), {}).get('with_amm', [])
            without_pts = grouped.get((panel_label, metric_name), {}).get('without_amm', [])
            # pair by seed
            with_by_seed = {p['seed']: _safe_float(p.get('normalized_avg_impact')) for p in with_pts}
            without_by_seed = {p['seed']: _safe_float(p.get('normalized_avg_impact')) for p in without_pts}
            common = sorted(set(with_by_seed) & set(without_by_seed))
            paired = [(with_by_seed[s], without_by_seed[s]) for s in common]
            if not paired:
                continue
            per_scenario.append({
                'scenario_key': sc_key,
                'scenario_label': sc_label,
                'paired_diffs': paired,
            })
        if not per_scenario:
            continue
        out_path = os.path.join(delta_dir, f'{_slug(metric_name)}.png')
        plot_paired_delta_strip(metric_label, per_scenario, out_path,
                                field='normalized_avg_impact')
        paths_written.append(out_path)

    # 5. Recovery boxplots per metric
    box_dir = os.path.join(resilience_dir, 'recovery_boxplots')
    for metric_name, metric_label in PRIORITY_METRICS:
        per_scenario = []
        for sc_key, sc_label in SCENARIOS:
            panel_label = sc_label
            with_pts = grouped.get((panel_label, metric_name), {}).get('with_amm', [])
            without_pts = grouped.get((panel_label, metric_name), {}).get('without_amm', [])
            with_vals = [_safe_float(p.get('recovery_steps')) for p in with_pts
                         if (p.get('recovered') in (True, 'True', '1', 1))]
            without_vals = [_safe_float(p.get('recovery_steps')) for p in without_pts
                            if (p.get('recovered') in (True, 'True', '1', 1))]
            if not with_vals and not without_vals:
                continue
            per_scenario.append({
                'scenario_key': sc_key,
                'scenario_label': sc_label,
                'with_values': with_vals,
                'without_values': without_vals,
            })
        if not per_scenario:
            continue
        out_path = os.path.join(box_dir, f'{_slug(metric_name)}.png')
        plot_recovery_boxplot(metric_label, per_scenario, out_path)
        paths_written.append(out_path)

    return paths_written


# ---------------------------------------------------------------------------
# Stat tests: split dashboards into per-metric individual plots
# ---------------------------------------------------------------------------

# Metric direction map (for green/red coloring of effect plots)
HIGHER_IS_BETTER = {
    'avg_clob_depth', 'avg_systemic_liquidity',
    'post_shock_avg_clob_depth', 'post_shock_avg_systemic_liquidity',
    'post_shock_mm_share_active', 'avg_mm_share_active',
    'liquidity_composite_recovered',
}
LOWER_IS_BETTER = {
    'avg_clob_spread_bps', 'avg_realized_volatility', 'tail_realized_volatility',
    'avg_cost_clob_q5',
    'post_shock_avg_clob_spread_bps', 'post_shock_avg_realized_volatility',
    'post_shock_mm_share_defensive', 'post_shock_mm_share_endogenous', 'post_shock_mm_withdrawal_score',
    'avg_mm_share_defensive', 'avg_mm_share_endogenous', 'avg_mm_withdrawal_score',
    'price_recovery_steps', 'spread_recovery_steps', 'depth_recovery_steps',
    'liquidity_composite_recovery_steps',
}


def _metric_label_map() -> Dict[str, str]:
    return {
        'n_trades': 'Total Trades',
        'avg_clob_spread_bps': 'Average CLOB Spread (bps)',
        'avg_clob_depth': 'Average CLOB Depth',
        'avg_systemic_liquidity': 'Average Systemic Liquidity',
        'avg_realized_volatility': 'Average Realized Volatility',
        'tail_realized_volatility': 'Tail Realized Volatility',
        'avg_cost_clob_q5': 'Average CLOB Cost (Q=5)',
        'avg_flow_share_amm_total': 'AMM Customer Volume Share (Ratio of Sums)',
        'price_recovery_steps': 'Price Recovery Steps',
        'spread_recovery_steps': 'Spread Recovery Steps',
        'depth_recovery_steps': 'Depth Recovery Steps',
        'liquidity_composite_recovery_steps': 'Composite Recovery Steps',
        'liquidity_composite_recovered': 'Composite Recovery Indicator',
        'post_shock_avg_clob_spread_bps': 'Post-shock CLOB Spread (bps)',
        'post_shock_avg_clob_depth': 'Post-shock CLOB Depth',
        'post_shock_avg_systemic_liquidity': 'Post-shock Systemic Liquidity',
        'post_shock_avg_realized_volatility': 'Post-shock Realized Volatility',
        'post_shock_mm_share_active': 'Post-shock MM Active Share',
        'post_shock_mm_share_defensive': 'Post-shock MM Defensive Share',
        'post_shock_mm_share_endogenous': 'Post-shock MM Endogenous Withdrawal Share',
        'post_shock_mm_share_forced': 'Post-shock MM Forced Pause Share',
        'post_shock_mm_withdrawal_score': 'Post-shock MM Withdrawal Score',
        'avg_mm_share_active': 'Average MM Active Share',
        'avg_mm_share_defensive': 'Average MM Defensive Share',
        'avg_mm_share_endogenous': 'Average MM Endogenous Withdrawal Share',
        'avg_mm_share_forced': 'Average MM Forced Pause Share',
        'avg_mm_withdrawal_score': 'Average MM Withdrawal Score',
        'amm_customer_volume_share': 'AMM Customer Volume Share (Ratio of Sums)',
        'clob_customer_volume_share': 'CLOB Customer Volume Share (Ratio of Sums)',
        'amm_active_tick_flow_share': 'AMM Customer Share (Active Trade Ticks)',
        'clob_active_tick_flow_share': 'CLOB Customer Share (Active Trade Ticks)',
        'mm_share_endogenous': 'MM Endogenous Withdrawal Share',
        'mm_share_forced': 'MM Forced Pause Share',
        'mm_withdrawal_score': 'MM Withdrawal Score',
        'spill_corr': 'Liquidity Co-movement Correlation',
        'spill_beta_amm_to_clob_std': 'AMM→CLOB Spillover Beta',
        'spill_beta_clob_to_amm_std': 'CLOB→AMM Spillover Beta',
        'mm_share_active': 'MM Active Share',
        'mm_share_defensive': 'MM Defensive Share',
    }


def regenerate_stat_tests(stat_dir: str) -> List[str]:
    paths_written: List[str] = []
    label_map = _metric_label_map()

    rq1_rows = _read_rows(os.path.join(stat_dir, 'rq1_tests.csv'))
    rq2_rows = _read_rows(os.path.join(stat_dir, 'rq2_tests.csv'))
    rq_test_rows = _read_rows(os.path.join(stat_dir, 'rq_with_without_amm_tests.csv'))
    h2_summary_rows = _read_rows(os.path.join(stat_dir, 'h2_phase_summary.csv'))

    # 6. RQ1 effects: one PNG per metric
    rq1_dir = os.path.join(stat_dir, 'rq1')
    rq1_metrics = sorted({r['metric_name'] for r in rq1_rows})
    for m in rq1_metrics:
        rows = [r for r in rq1_rows if r['metric_name'] == m]
        # ensure scenario order
        scenario_order = {k: i for i, (k, _) in enumerate(SCENARIOS)}
        rows.sort(key=lambda r: scenario_order.get(r['scenario_key'], 99))
        out_path = os.path.join(rq1_dir, f'{_slug(m)}.png')
        plot_rq_effect('RQ1: AMM presence effect', label_map.get(m, m), m, rows, out_path,
                       higher_is_better_metrics=HIGHER_IS_BETTER,
                       lower_is_better_metrics=LOWER_IS_BETTER)
        paths_written.append(out_path)

    # 7. RQ2 effects: one PNG per metric
    rq2_dir = os.path.join(stat_dir, 'rq2')
    rq2_metrics = sorted({r['metric_name'] for r in rq2_rows})
    for m in rq2_metrics:
        rows = [r for r in rq2_rows if r['metric_name'] == m]
        scenario_order = {k: i for i, (k, _) in enumerate(SCENARIOS)}
        rows.sort(key=lambda r: scenario_order.get(r['scenario_key'], 99))
        out_path = os.path.join(rq2_dir, f'{_slug(m)}.png')
        plot_rq_effect('RQ2: AMM effect on stress dynamics', label_map.get(m, m), m, rows, out_path,
                       higher_is_better_metrics=HIGHER_IS_BETTER,
                       lower_is_better_metrics=LOWER_IS_BETTER)
        paths_written.append(out_path)

    # 8. H2 phase trajectories: one PNG per metric
    if h2_summary_rows:
        h2_dir = os.path.join(stat_dir, 'h2_phase')
        # Pivot: per scenario assemble {metric}_before/during/after
        per_scenario_pivot: Dict[str, dict] = {}
        for row in h2_summary_rows:
            sc_key = row['scenario_key']
            phase = row['phase']
            pivot = per_scenario_pivot.setdefault(sc_key, {
                'scenario_key': sc_key,
                'scenario_label': row['scenario_label'],
            })
            for k, v in row.items():
                if k in ('scenario_key', 'scenario_label', 'phase', 'n_seeds'):
                    continue
                pivot[f'{k}_{phase}'] = _safe_float(v)
        ordered = []
        scenario_order = {k: i for i, (k, _) in enumerate(SCENARIOS)}
        for sc_key in sorted(per_scenario_pivot, key=lambda k: scenario_order.get(k, 99)):
            ordered.append(per_scenario_pivot[sc_key])

        h2_metric_names = ['amm_customer_volume_share',
                           'clob_customer_volume_share',
                           'amm_active_tick_flow_share',
                           'clob_active_tick_flow_share',
                           'mm_share_active', 'mm_share_defensive',
                           'mm_share_endogenous', 'mm_share_forced',
                           'mm_withdrawal_score',
                           'spill_corr', 'spill_beta_amm_to_clob_std',
                           'spill_beta_clob_to_amm_std']
        for m in h2_metric_names:
            out_path = os.path.join(h2_dir, f'{_slug(m)}.png')
            plot_h2_phase(m, label_map.get(m, m), ordered, out_path)
            paths_written.append(out_path)

    # 9. MM behavior heatmap (single PNG)
    if rq_test_rows:
        out_path = os.path.join(stat_dir, 'mm_behavior', 'heatmap.png')
        plot_mm_behavior_heatmap(rq_test_rows, out_path)
        paths_written.append(out_path)

    return paths_written


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate individual plots from CSVs.')
    parser.add_argument('--resilience-dir', default='output/resilience',
                        help='Directory with resilience CSVs (default: output/resilience)')
    parser.add_argument('--stat-dir', default='output/stat_tests',
                        help='Directory with stat_tests CSVs (default: output/stat_tests)')
    args = parser.parse_args()

    print('Regenerating resilience plots...')
    rp = regenerate_resilience(args.resilience_dir)
    print(f'  → wrote {len(rp)} plots to {args.resilience_dir}/')
    for p in rp:
        rel = os.path.relpath(p, args.resilience_dir)
        print(f'      {rel}')

    print('Regenerating stat_tests plots...')
    sp = regenerate_stat_tests(args.stat_dir)
    print(f'  → wrote {len(sp)} plots to {args.stat_dir}/')
    for p in sp:
        rel = os.path.relpath(p, args.stat_dir)
        print(f'      {rel}')

    print(f'\nDone. {len(rp) + len(sp)} plots total.')


if __name__ == '__main__':
    main()
