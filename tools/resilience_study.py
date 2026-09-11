#!/usr/bin/env python3
"""Generate censor-aware multi-seed resilience study outputs."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AgentBasedModel.visualization.paper_style import use_paper_style

use_paper_style()

from main import _apply_preset_defaults, _auto_stress_around_shock, _seed_all, build_parser, build_sim
from AgentBasedModel.metrics.resilience import (
    composite_resilience_metrics,
    kaplan_meier_curve,
    price_resilience_metrics,
    series_resilience_metrics,
)
from AgentBasedModel.metrics.statistics import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    independent_permutation_test,
    paired_permutation_test,
)
# Plots are generated via tools.regenerate_plots after CSVs are written
# (see end of _run_study).


DEFAULT_PANELS = [
    ('mm_withdrawal', 'MM Withdrawal (Isolated)', '#1f77b4', 'pre_shock_baseline', 'Pre-shock baseline'),
    ('flash_crash', 'Flash Crash', '#2ca02c', 'pre_shock_baseline', 'Pre-shock baseline'),
    ('dealer_liquidity_crisis', 'Dealer Liquidity Crisis', '#ff7f0e', 'fair_value_gap', 'Fair-value gap'),
    ('funding_liquidity_shock', 'Funding Liquidity Shock', '#d62728', 'pre_shock_baseline', 'Pre-shock baseline'),
    ('high_vol_stress', 'High-Vol Stress', '#9467bd', 'pre_shock_baseline', 'Pre-stress baseline'),
]


def _apply_research_preset(args, preset: str):
    """Keep resilience panels pinned to the isolated MM-withdrawal scenario."""
    if preset == 'mm_withdrawal':
        args.clob_amm_interaction = 'none'

COMPARISON_SERIES = [
    ('with_amm', 'With AMM', True, None),
    ('without_amm', 'Without AMM', False, '#4c4c4c'),
]

PRIORITY_METRICS = [
    ('spread_resilience', 'Quoted Spread Resilience'),
    ('depth_resilience', 'Depth Resilience'),
    ('execution_cost_resilience', 'Execution Cost Resilience'),
    ('composite_resilience', 'Composite Resilience'),
]

STAT_ENDPOINTS = [
    ('recovered_indicator', 'Recovered Indicator', 'all', 'recovered', True),
    ('time_observed_steps', 'Observed Time', 'all', 'time_observed_steps', False),
    ('normalized_auc_abs', 'Normalized AUC', 'all', 'normalized_auc_abs', False),
    ('normalized_avg_impact', 'Normalized Average Impact', 'all', 'normalized_avg_impact', False),
    ('recovery_steps_recovered_only', 'Recovery Steps', 'recovered_only', 'recovery_steps', False),
]

RECOVERY_COMPARISON_ENDPOINTS = [
    ('recovered_indicator', 'Recovered Indicator', 'all', 'recovered', True),
    ('time_observed_steps', 'Observed Time', 'all', 'time_observed_steps', False),
    ('recovery_steps_both_recovered', 'Recovery Steps', 'both_recovered', 'recovery_steps', False),
    ('half_life_steps', 'Half-life', 'all', 'half_life_steps', False),
]


def _base_args_for_preset(preset: str, n_iter: int, min_post_shock_window: int,
                          forced_venue_choice_rule: str | None = None):
    parser = build_parser(default_venue_choice_rule=forced_venue_choice_rule or 'liquidity_aware')
    args = parser.parse_args([])
    args.preset = preset
    args.n_iter = n_iter
    _apply_preset_defaults(parser, args)
    _apply_research_preset(args, preset)
    if forced_venue_choice_rule is not None:
        args.venue_choice_rule = forced_venue_choice_rule
    if args.shock_iter is not None:
        args.n_iter = max(args.n_iter, int(args.shock_iter) + max(50, min_post_shock_window))
    _auto_stress_around_shock(args)
    return args


def _summary_row(label: str, target_mode: str, points: list, metric_name: str = '', metric_label: str = '') -> dict:
    km = kaplan_meier_curve(
        [point.get('time_observed_steps', float('nan')) for point in points],
        [bool(point.get('recovered', False)) for point in points],
    )
    row = {
        'panel_label': label,
        'target_mode': target_mode,
        'n_obs': len(points),
        'recovered_share': _mean_finite(
            1.0 if point.get('recovered', False) else 0.0 for point in points
        ),
        'km_median_steps': km['median_time'],
        'mean_observed_steps': _mean_finite(point.get('time_observed_steps', float('nan')) for point in points),
        'mean_recovery_steps': _mean_finite(point.get('recovery_steps', float('nan')) for point in points),
        'mean_normalized_avg_impact': _mean_finite(point.get('normalized_avg_impact', float('nan')) for point in points),
        'mean_normalized_auc_abs': _mean_finite(point.get('normalized_auc_abs', float('nan')) for point in points),
        'mean_initial_dislocation_pct': _mean_finite(point.get('initial_dislocation_pct', float('nan')) for point in points),
        'mean_reference_dislocation_step': _mean_finite(point.get('reference_dislocation_step', float('nan')) for point in points),
    }
    if metric_name:
        row['metric_name'] = metric_name
    if metric_label:
        row['metric_label'] = metric_label
    return row


def _stat_seed(base_seed: int, *parts: str) -> int:
    state = int(base_seed) % (2 ** 32)
    for part in parts:
        for ch in str(part):
            state = (state * 131 + ord(ch)) % (2 ** 32)
    return state


def _extract_endpoint_values(points: list, field: str,
                             sample_filter: str = 'all',
                             binary: bool = False) -> list[float]:
    values = []
    for point in points:
        if sample_filter == 'recovered_only' and not point.get('recovered', False):
            continue
        if binary:
            values.append(1.0 if point.get(field, False) else 0.0)
            continue
        value = point.get(field, float('nan'))
        if math.isfinite(value):
            values.append(float(value))
    return values


def _metric_groups(panel_specs) -> list[dict]:
    groups = [
        {
            'metric_name': 'price_resilience',
            'metric_label': 'Price Resilience',
            'panels': [
                {
                    'panel_label': spec['label'],
                    'series_key': series.get('series_key', ''),
                    'series_label': series.get('label', ''),
                    'amm_enabled': series.get('amm_enabled', ''),
                    'target_mode': spec.get('target_mode', 'pre_shock_baseline'),
                    'points': list(series.get('points', [])),
                }
                for spec in panel_specs
                for series in spec.get('series', [])
            ],
        }
    ]

    for metric_name, metric_label in PRIORITY_METRICS:
        panels = []
        for spec in panel_specs:
            for series in spec.get('series', []):
                metric_points = [
                    point for point in series.get('priority_metric_points', [])
                    if point.get('metric_name') == metric_name
                ]
                panels.append({
                    'panel_label': spec['label'],
                    'series_key': series.get('series_key', ''),
                    'series_label': series.get('label', ''),
                    'amm_enabled': series.get('amm_enabled', ''),
                    'target_mode': metric_points[0].get('analysis_target_mode', 'pre_shock_baseline') if metric_points else 'pre_shock_baseline',
                    'points': metric_points,
                })
        groups.append({
            'metric_name': metric_name,
            'metric_label': metric_label,
            'panels': panels,
        })
    return groups


def _resolve_cost_series(logger, cost_q: float):
    if not logger.clob_cost_curves:
        return [], float(cost_q)

    if cost_q in logger.clob_cost_curves:
        return logger.clob_cost_curves[cost_q], float(cost_q)

    available = sorted(logger.clob_cost_curves.keys(), key=lambda value: abs(value - cost_q))
    chosen_q = float(available[0])
    return logger.clob_cost_curves[chosen_q], chosen_q


def _priority_metric_points(price_metrics: dict, logger, shock_iter: int,
                            stable_window: int, avg_window: int,
                            tolerance_pct: float, retracement_fraction: float,
                            impact_window: int, horizon: int,
                            cost_q: float) -> list:
    depth_series = [depth.get('total', float('nan')) for depth in logger.clob_depth]
    cost_series, resolved_cost_q = _resolve_cost_series(logger, cost_q)

    spread_metrics = series_resilience_metrics(
        logger.clob_qspr,
        shock_iter,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
    )
    depth_metrics = series_resilience_metrics(
        depth_series,
        shock_iter,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
    )
    cost_metrics = series_resilience_metrics(
        cost_series,
        shock_iter,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
    )
    composite_metrics = composite_resilience_metrics(
        [price_metrics, spread_metrics, depth_metrics, cost_metrics],
        analysis_target_mode='joint_price_spread_depth_cost',
    )

    rows = []
    metric_rows = [
        ('spread_resilience', 'Quoted Spread Resilience', spread_metrics, ''),
        ('depth_resilience', 'Depth Resilience', depth_metrics, ''),
        ('execution_cost_resilience', 'Execution Cost Resilience', cost_metrics, f'clob_cost_q={resolved_cost_q:g}'),
        ('composite_resilience', 'Composite Resilience', composite_metrics, 'price+spread+depth+cost'),
    ]
    for metric_name, metric_label, metrics, note in metric_rows:
        row = dict(metrics)
        row['metric_name'] = metric_name
        row['metric_label'] = metric_label
        row['metric_note'] = note
        rows.append(row)
    return rows


def _event_iter_from_sim(sim) -> int | None:
    shock_iter = getattr(sim, 'shock_iter', None)
    if shock_iter is None:
        shock_iter = getattr(sim, '_shock_iter', None)
    if shock_iter is not None:
        return int(shock_iter)

    env = getattr(sim, 'env', None)
    stress_start = getattr(env, 'stress_start', None) if env is not None else None
    if stress_start is not None and stress_start >= 0:
        return int(stress_start)
    return None


def _collect_points(preset: str, seeds: int, base_seed: int,
                    n_iter: int, tolerance_pct: float,
                    stable_window: int, avg_window: int,
                    retracement_fraction: float,
                    impact_window: int,
                    min_post_shock_window: int,
                    target_mode: str,
                    cost_q: float,
                    progress_every: int = 0,
                    forced_venue_choice_rule: str | None = None,
                    enable_amm: bool = True,
                    series_key: str = 'with_amm') -> list:
    points = []
    priority_metric_points = []
    for offset in range(seeds):
        seed = base_seed + offset
        args = _base_args_for_preset(
            preset,
            n_iter,
            min_post_shock_window,
            forced_venue_choice_rule=forced_venue_choice_rule,
        )
        args.seed = seed
        args.enable_amm = 1 if enable_amm else 0
        if not enable_amm:
            args.amm_share_pct = 0
        _seed_all(seed)

        sim = build_sim(args)
        sim.simulate(args.n_iter, silent=True)

        event_iter = _event_iter_from_sim(sim)
        if event_iter is None:
            continue

        target_series = None
        if target_mode == 'fair_value_gap':
            target_series = getattr(sim.logger, 'fair_price_series', None)

        horizon = max(50, min(args.n_iter - event_iter, min_post_shock_window))

        metrics = price_resilience_metrics(
            sim.logger.clob_mid_series,
            shock_iter=event_iter,
            baseline_window=50,
            avg_window=avg_window,
            tolerance_pct=tolerance_pct,
            stable_window=stable_window,
            horizon=horizon,
            retracement_fraction=retracement_fraction,
            impact_window=impact_window,
            target_series=target_series,
            target_mode=target_mode,
        )
        points.append({
            'preset': preset,
            'seed': seed,
            'series_key': series_key,
            'amm_enabled': 1 if enable_amm else 0,
            'target_mode': metrics['analysis_target_mode'],
            'recovery_steps': metrics['recovery_steps'],
            'time_observed_steps': metrics['time_observed_steps'],
            'recovered': metrics['recovered'],
            'is_censored': metrics['is_censored'],
            'half_life_steps': metrics['half_life_steps'],
            'legacy_recovery_steps': metrics['legacy_recovery_steps'],
            'initial_dislocation_pct': metrics['initial_dislocation_pct'],
            'reference_dislocation_pct': metrics['reference_dislocation_pct'],
            'reference_dislocation_step': metrics['reference_dislocation_step'],
            'target_abs_deviation_pct': metrics['target_abs_deviation_pct'],
            'avg_change_pct': metrics['avg_change_pct'],
            'normalized_avg_impact': metrics['normalized_avg_impact'],
            'normalized_auc_abs': metrics['normalized_auc_abs'],
            'trough_change_pct': metrics['trough_change_pct'],
            'peak_abs_change_pct': metrics['peak_abs_change_pct'],
        })
        for metric_point in _priority_metric_points(
            metrics,
            sim.logger,
            event_iter,
            stable_window=stable_window,
            avg_window=avg_window,
            tolerance_pct=tolerance_pct,
            retracement_fraction=retracement_fraction,
            impact_window=impact_window,
            horizon=horizon,
            cost_q=cost_q,
        ):
            metric_row = dict(metric_point)
            metric_row['preset'] = preset
            metric_row['seed'] = seed
            metric_row['series_key'] = series_key
            metric_row['amm_enabled'] = 1 if enable_amm else 0
            priority_metric_points.append(metric_row)
        done = offset + 1
        if progress_every > 0 and (done == 1 or done % progress_every == 0 or done == seeds):
            print(f'[{preset}] {done}/{seeds} seeds complete', flush=True)
    return {
        'price_points': points,
        'priority_metric_points': priority_metric_points,
    }


def _save_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_scatter_points.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'series_key', 'amm_enabled', 'preset', 'seed', 'target_mode', 'recovered', 'is_censored',
                'recovery_steps', 'time_observed_steps', 'half_life_steps',
                'legacy_recovery_steps', 'initial_dislocation_pct', 'reference_dislocation_pct',
                'reference_dislocation_step', 'target_abs_deviation_pct',
                'avg_change_pct', 'normalized_avg_impact', 'normalized_auc_abs',
                'trough_change_pct', 'peak_abs_change_pct',
            ],
        )
        writer.writeheader()
        for spec in panel_specs:
            for point in spec['points']:
                row = dict(point)
                row['panel_label'] = spec['label']
                writer.writerow(row)
    return path


def _save_priority_metric_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_priority_metric_points.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'series_key', 'amm_enabled', 'preset', 'seed', 'metric_name', 'metric_label', 'metric_note',
                'analysis_target_mode', 'recovered', 'is_censored', 'recovery_steps',
                'time_observed_steps', 'half_life_steps', 'legacy_recovery_steps',
                'initial_dislocation_pct', 'reference_dislocation_pct', 'reference_dislocation_step',
                'target_abs_deviation_pct', 'avg_change_pct', 'normalized_avg_impact',
                'normalized_auc_abs', 'trough_change_pct', 'peak_abs_change_pct',
            ],
            extrasaction='ignore',
        )
        writer.writeheader()
        for spec in panel_specs:
            for point in spec['priority_metric_points']:
                row = dict(point)
                row['panel_label'] = spec['label']
                writer.writerow(row)
    return path


def _mean_finite(values) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float('nan')


def _median_finite(values) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float('nan')
    mid = len(clean) // 2
    if len(clean) % 2 == 1:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def _save_summary_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_panel_summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'series_key', 'series_label', 'amm_enabled', 'target_mode', 'n_obs', 'recovered_share', 'km_median_steps',
                'mean_observed_steps', 'mean_recovery_steps', 'mean_normalized_avg_impact',
                'mean_normalized_auc_abs', 'mean_initial_dislocation_pct',
                'mean_reference_dislocation_step',
            ],
        )
        writer.writeheader()
        for spec in panel_specs:
            for series in spec.get('series', []):
                row = _summary_row(
                    spec['label'],
                    spec.get('target_mode', 'pre_shock_baseline'),
                    list(series.get('points', [])),
                )
                row['series_key'] = series.get('series_key', '')
                row['series_label'] = series.get('label', '')
                row['amm_enabled'] = series.get('amm_enabled', '')
                writer.writerow(row)
    return path


def _save_priority_metric_summary_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_priority_metric_summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'series_key', 'series_label', 'amm_enabled', 'metric_name', 'metric_label', 'target_mode', 'n_obs',
                'recovered_share', 'km_median_steps', 'mean_observed_steps', 'mean_recovery_steps',
                'mean_normalized_avg_impact', 'mean_normalized_auc_abs',
                'mean_initial_dislocation_pct', 'mean_reference_dislocation_step',
            ],
        )
        writer.writeheader()
        for spec in panel_specs:
            for series in spec.get('series', []):
                for metric_name, metric_label in PRIORITY_METRICS:
                    points = [
                        point for point in series.get('priority_metric_points', [])
                        if point.get('metric_name') == metric_name
                    ]
                    if not points:
                        continue
                    row = _summary_row(
                        spec['label'],
                        points[0].get('analysis_target_mode', 'pre_shock_baseline'),
                        points,
                        metric_name=metric_name,
                        metric_label=metric_label,
                    )
                    row['series_key'] = series.get('series_key', '')
                    row['series_label'] = series.get('label', '')
                    row['amm_enabled'] = series.get('amm_enabled', '')
                    writer.writerow(row)
    return path


def _save_statistical_summary_csv(panel_specs, out_dir: str,
                                  bootstrap_reps: int,
                                  ci_level: float,
                                  base_seed: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_statistical_summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'series_key', 'series_label', 'amm_enabled', 'metric_name', 'metric_label', 'target_mode',
                'endpoint_name', 'endpoint_label', 'sample_filter', 'n_used',
                'mean', 'std', 'median', 'ci_level', 'ci_lo', 'ci_hi',
            ],
        )
        writer.writeheader()
        for group in _metric_groups(panel_specs):
            for panel in group['panels']:
                for endpoint_name, endpoint_label, sample_filter, field, binary in STAT_ENDPOINTS:
                    values = _extract_endpoint_values(panel['points'], field, sample_filter=sample_filter, binary=binary)
                    stats = bootstrap_mean_ci(
                        values,
                        n_boot=bootstrap_reps,
                        ci_level=ci_level,
                        seed=_stat_seed(base_seed, group['metric_name'], panel['panel_label'], endpoint_name),
                    )
                    writer.writerow({
                        'panel_label': panel['panel_label'],
                        'series_key': panel.get('series_key', ''),
                        'series_label': panel.get('series_label', ''),
                        'amm_enabled': panel.get('amm_enabled', ''),
                        'metric_name': group['metric_name'],
                        'metric_label': group['metric_label'],
                        'target_mode': panel['target_mode'],
                        'endpoint_name': endpoint_name,
                        'endpoint_label': endpoint_label,
                        'sample_filter': sample_filter,
                        'n_used': stats['n'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'median': stats['median'],
                        'ci_level': ci_level,
                        'ci_lo': stats['ci_lo'],
                        'ci_hi': stats['ci_hi'],
                    })
    return path


def _save_pairwise_tests_csv(panel_specs, out_dir: str,
                             bootstrap_reps: int,
                             ci_level: float,
                             base_seed: int,
                             permutation_reps: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_pairwise_tests.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'series_key', 'series_label', 'amm_enabled', 'metric_name', 'metric_label', 'endpoint_name', 'endpoint_label', 'sample_filter',
                'panel_a', 'panel_b', 'n_a', 'n_b', 'mean_a', 'mean_b', 'mean_diff_a_minus_b',
                'permutation_stat', 'permutation_p_value', 'ci_level', 'diff_ci_lo', 'diff_ci_hi',
            ],
        )
        writer.writeheader()
        for group in _metric_groups(panel_specs):
            panels_by_series = {}
            for panel in group['panels']:
                panels_by_series.setdefault(panel.get('series_key', ''), []).append(panel)
            for series_key, series_panels in panels_by_series.items():
                series_label = series_panels[0].get('series_label', '') if series_panels else ''
                amm_enabled = series_panels[0].get('amm_enabled', '') if series_panels else ''
                for endpoint_name, endpoint_label, sample_filter, field, binary in STAT_ENDPOINTS:
                    panel_series = []
                    for panel in series_panels:
                        values = _extract_endpoint_values(panel['points'], field, sample_filter=sample_filter, binary=binary)
                        panel_series.append((panel['panel_label'], values))
                    for (panel_a, sample_a), (panel_b, sample_b) in itertools.combinations(panel_series, 2):
                        permutation = independent_permutation_test(
                            sample_a,
                            sample_b,
                            n_perm=permutation_reps,
                            seed=_stat_seed(base_seed, 'independent_perm', group['metric_name'], series_key, endpoint_name, panel_a, panel_b),
                        )
                        boot = bootstrap_diff_ci(
                            sample_a,
                            sample_b,
                            n_boot=bootstrap_reps,
                            ci_level=ci_level,
                            seed=_stat_seed(base_seed, group['metric_name'], series_key, endpoint_name, panel_a, panel_b),
                        )
                        writer.writerow({
                            'series_key': series_key,
                            'series_label': series_label,
                            'amm_enabled': amm_enabled,
                            'metric_name': group['metric_name'],
                            'metric_label': group['metric_label'],
                            'endpoint_name': endpoint_name,
                            'endpoint_label': endpoint_label,
                            'sample_filter': sample_filter,
                            'panel_a': panel_a,
                            'panel_b': panel_b,
                            'n_a': permutation['n_a'],
                            'n_b': permutation['n_b'],
                            'mean_a': permutation['mean_a'],
                            'mean_b': permutation['mean_b'],
                            'mean_diff_a_minus_b': permutation['mean_diff'],
                            'permutation_stat': permutation['stat'],
                            'permutation_p_value': permutation['p_value'],
                            'ci_level': ci_level,
                            'diff_ci_lo': boot['ci_lo'],
                            'diff_ci_hi': boot['ci_hi'],
                        })
    return path


def _extract_paired_series_values(points_with: list, points_without: list,
                                  field: str,
                                  sample_filter: str = 'all',
                                  binary: bool = False) -> tuple[list[float], list[float]]:
    by_seed_with = {point['seed']: point for point in points_with}
    by_seed_without = {point['seed']: point for point in points_without}
    common_seeds = sorted(set(by_seed_with) & set(by_seed_without))

    sample_with = []
    sample_without = []
    for seed in common_seeds:
        point_with = by_seed_with[seed]
        point_without = by_seed_without[seed]
        if sample_filter == 'both_recovered' and (
            not point_with.get('recovered', False) or not point_without.get('recovered', False)
        ):
            continue
        if binary:
            sample_with.append(1.0 if point_with.get(field, False) else 0.0)
            sample_without.append(1.0 if point_without.get(field, False) else 0.0)
            continue
        value_with = point_with.get(field, float('nan'))
        value_without = point_without.get(field, float('nan'))
        if math.isfinite(value_with) and math.isfinite(value_without):
            sample_with.append(float(value_with))
            sample_without.append(float(value_without))
    return sample_with, sample_without


def _save_with_without_amm_comparison_csv(panel_specs, out_dir: str,
                                          bootstrap_reps: int,
                                          ci_level: float,
                                          base_seed: int,
                                          permutation_reps: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_comparison_summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'metric_name', 'metric_label', 'target_mode',
                'endpoint_name', 'endpoint_label', 'sample_filter', 'n_pairs',
                'mean_with_amm', 'mean_without_amm', 'mean_delta_with_minus_without',
                'median_delta_with_minus_without', 'share_positive_delta',
                'permutation_stat', 'permutation_p_value', 'ci_level', 'delta_ci_lo', 'delta_ci_hi',
            ],
        )
        writer.writeheader()
        for group in _metric_groups(panel_specs):
            grouped_panels = {}
            for panel in group['panels']:
                grouped_panels.setdefault(panel['panel_label'], {})[panel.get('series_key', '')] = panel
            for panel_label, panel_series in grouped_panels.items():
                with_panel = panel_series.get('with_amm')
                without_panel = panel_series.get('without_amm')
                if with_panel is None or without_panel is None:
                    continue
                for endpoint_name, endpoint_label, sample_filter, field, binary in RECOVERY_COMPARISON_ENDPOINTS:
                    sample_with, sample_without = _extract_paired_series_values(
                        with_panel['points'],
                        without_panel['points'],
                        field,
                        sample_filter=sample_filter,
                        binary=binary,
                    )
                    paired = paired_permutation_test(
                        sample_with,
                        sample_without,
                        n_perm=permutation_reps,
                        seed=_stat_seed(base_seed, 'paired_perm', panel_label, group['metric_name'], endpoint_name),
                    )
                    boot = bootstrap_paired_diff_ci(
                        sample_with,
                        sample_without,
                        n_boot=bootstrap_reps,
                        ci_level=ci_level,
                        seed=_stat_seed(base_seed, panel_label, group['metric_name'], endpoint_name),
                    )
                    deltas = [
                        value_with - value_without
                        for value_with, value_without in zip(sample_with, sample_without)
                        if math.isfinite(value_with) and math.isfinite(value_without)
                    ]
                    writer.writerow({
                        'panel_label': panel_label,
                        'metric_name': group['metric_name'],
                        'metric_label': group['metric_label'],
                        'target_mode': with_panel.get('target_mode', 'pre_shock_baseline'),
                        'endpoint_name': endpoint_name,
                        'endpoint_label': endpoint_label,
                        'sample_filter': sample_filter,
                        'n_pairs': paired['n'],
                        'mean_with_amm': paired['mean_a'],
                        'mean_without_amm': paired['mean_b'],
                        'mean_delta_with_minus_without': paired['mean_diff'],
                        'median_delta_with_minus_without': _median_finite(deltas),
                        'share_positive_delta': _mean_finite(1.0 if value > 0 else 0.0 for value in deltas),
                        'permutation_stat': paired['stat'],
                        'permutation_p_value': paired['p_value'],
                        'ci_level': ci_level,
                        'delta_ci_lo': boot['ci_lo'],
                        'delta_ci_hi': boot['ci_hi'],
                    })
    return path


def _save_spread_recovery_survival_panels(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    spread_panels = []
    for spec in panel_specs:
        spread_panels.append({
            'label': spec['label'],
            'target_label': spec.get('target_label', 'Pre-shock baseline'),
            'series': [
                {
                    'label': series.get('label', 'Series'),
                    'color': series.get('color', spec.get('color', '#1f77b4')),
                    'points': [
                        point for point in series.get('priority_metric_points', [])
                        if point.get('metric_name') == 'spread_resilience'
                    ],
                }
                for series in spec.get('series', [])
            ],
        })

    ncols = 2
    nrows = math.ceil(len(spread_panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(7, 4.3 * nrows)), sharex=False, sharey=True)
    axes_flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]

    for ax, spec in zip(axes_flat, spread_panels):
        stats_lines = []
        plotted_any = False
        for series in spec['series']:
            points = list(series.get('points', []))
            observed_steps = [point.get('time_observed_steps', float('nan')) for point in points]
            recovered_flags = [bool(point.get('recovered', False)) for point in points]
            km = kaplan_meier_curve(observed_steps, recovered_flags)
            if not km.get('times'):
                continue
            ax.step(
                km['times'],
                km['survival'],
                where='post',
                color=series.get('color', '#1f77b4'),
                lw=2.0,
                label=series.get('label', 'Series'),
            )
            plotted_any = True
            recovered_share = sum(recovered_flags) / max(len(recovered_flags), 1)
            stats_lines.append(
                f"{series.get('label', 'Series')}: rec={recovered_share:.0%}, KM med={km.get('median_time', float('nan')):.1f}"
            )

        ax.set_title(spec['label'], fontsize=12)
        ax.set_xlabel('Observed Steps Since Shock', fontsize=11)
        ax.set_ylabel('P(not yet spread-recovered)', fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color='#777777', ls='--', lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.02,
            0.96,
            f"Target: {spec['target_label']}",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.65),
        )
        if stats_lines:
            ax.text(
                0.98,
                0.96,
                '\n'.join(stats_lines),
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75),
            )
        if plotted_any:
            ax.legend(fontsize=8, loc='upper right')

    for ax in axes_flat[len(spread_panels):]:
        ax.axis('off')

    fig.suptitle(
        'Quoted Spread Recovery: Kaplan-Meier by Market Structure',
        fontsize=15,
        fontweight='bold',
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(out_dir, 'spread_recovery_survival_panels.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def _run_study(forced_venue_choice_rule: str | None = None,
               default_out_dir: str = 'output/resilience'):
    parser = argparse.ArgumentParser(
        description='Generate resilience scatter panels from multi-seed shock simulations.',
    )
    parser.add_argument('--seeds', type=int, default=300,
                        help='Number of sequential seeds per panel (default: 300)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Starting seed (default: 42)')
    parser.add_argument('--permutation-reps', type=int, default=10000,
                        help='Permutation repetitions for p-values (default: 10000)')
    parser.add_argument('--n-iter', type=int, default=900,
                        help='Simulation length for each run (default: 900)')
    parser.add_argument('--avg-window', type=int, default=20,
                        help='Window used for average post-shock price change (default: 20)')
    parser.add_argument('--tolerance-pct', type=float, default=50.0,
                        help='Reference fixed-band threshold in percent around baseline (default: 50.0 = [0.5x, 1.5x])')
    parser.add_argument('--retracement-fraction', type=float, default=0.8,
                        help='Fraction of initial dislocation that must be closed to count as recovery (default: 0.8)')
    parser.add_argument('--impact-window', type=int, default=10,
                        help='Early post-shock window used to anchor the peak dislocation (default: 10)')
    parser.add_argument('--min-post-shock-window', type=int, default=600,
                        help='Minimum number of post-shock steps observed for recovery analysis (default: 600)')
    parser.add_argument('--stable-window', type=int, default=10,
                        help='Consecutive periods required inside the recovery band (default: 10)')
    parser.add_argument('--progress-every', type=int, default=25,
                        help='Print progress every N seeds per panel (default: 25, 0 disables progress output)')
    parser.add_argument('--cost-q', type=float, default=10.0,
                        help='Representative CLOB trade size used for execution-cost resilience (default: 10)')
    parser.add_argument('--bootstrap-reps', type=int, default=2000,
                        help='Bootstrap repetitions used for confidence intervals (default: 2000)')
    parser.add_argument('--ci-level', type=float, default=0.95,
                        help='Confidence level for bootstrap intervals (default: 0.95)')
    parser.add_argument('--out-dir', default=default_out_dir,
                        help='Output directory for PNG and CSV files')
    args = parser.parse_args()

    panel_specs = []
    for preset, label, color, target_mode, target_label in DEFAULT_PANELS:
        print(f'Starting panel: {label} [{target_label}]', flush=True)
        series_specs = []
        for series_key, series_label, enable_amm, series_color in COMPARISON_SERIES:
            print(f'  -> collecting {series_label}', flush=True)
            panel_data = _collect_points(
                preset=preset,
                seeds=args.seeds,
                base_seed=args.base_seed,
                n_iter=args.n_iter,
                tolerance_pct=args.tolerance_pct,
                stable_window=args.stable_window,
                avg_window=args.avg_window,
                retracement_fraction=args.retracement_fraction,
                impact_window=args.impact_window,
                min_post_shock_window=args.min_post_shock_window,
                target_mode=target_mode,
                cost_q=args.cost_q,
                progress_every=args.progress_every,
                forced_venue_choice_rule=forced_venue_choice_rule,
                enable_amm=enable_amm,
                series_key=series_key,
            )
            series_specs.append({
                'series_key': series_key,
                'label': series_label,
                'amm_enabled': 1 if enable_amm else 0,
                'color': series_color or color,
                'points': panel_data['price_points'],
                'priority_metric_points': panel_data['priority_metric_points'],
            })
        panel_specs.append({
            'label': label,
            'color': color,
            'target_mode': target_mode,
            'target_label': target_label,
            'series': series_specs,
            'points': [point for series in series_specs for point in series['points']],
            'priority_metric_points': [
                point for series in series_specs for point in series['priority_metric_points']
            ],
        })

    csv_path = _save_csv(panel_specs, out_dir=args.out_dir)
    summary_path = _save_summary_csv(panel_specs, out_dir=args.out_dir)
    priority_csv_path = _save_priority_metric_csv(panel_specs, out_dir=args.out_dir)
    priority_summary_path = _save_priority_metric_summary_csv(panel_specs, out_dir=args.out_dir)
    stats_summary_path = _save_statistical_summary_csv(
        panel_specs,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        base_seed=args.base_seed,
    )
    pairwise_path = _save_pairwise_tests_csv(
        panel_specs,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        base_seed=args.base_seed,
        permutation_reps=args.permutation_reps,
    )
    comparison_path = _save_with_without_amm_comparison_csv(
        panel_specs,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        base_seed=args.base_seed,
        permutation_reps=args.permutation_reps,
    )

    print(f'Saved resilience point data to {csv_path}')
    print(f'Saved resilience panel summary to {summary_path}')
    print(f'Saved priority resilience metric data to {priority_csv_path}')
    print(f'Saved priority resilience metric summary to {priority_summary_path}')
    print(f'Saved statistical resilience summary to {stats_summary_path}')
    print(f'Saved pairwise statistical tests to {pairwise_path}')
    print(f'Saved with/without AMM comparison summary to {comparison_path}')

    # Generate individual plots in subfolders (scatter/, km_curves/, peak_impact/,
    # normalized_impact/, recovery_boxplots/) by reading the CSVs we just wrote.
    from tools.regenerate_plots import regenerate_resilience
    plot_paths = regenerate_resilience(args.out_dir)
    print(f'Generated {len(plot_paths)} individual plots under {args.out_dir}/')


def main():
    _run_study()


def main_fixed():
    _run_study(forced_venue_choice_rule='fixed_share',
               default_out_dir='output/resilience_fixed')


def main_aware():
    _run_study(forced_venue_choice_rule='liquidity_aware',
               default_out_dir='output/resilience_aware')


if __name__ == '__main__':
    main()