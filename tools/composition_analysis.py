"""Composition sweep — analysis & visualization.

Auto-detects all chunk CSVs in --results-dir, concatenates them, applies the
stability filter from `calibration/stability_criteria.json`, and produces:

  1) stability_map.csv         — per-point stability flag + per-scenario indicators
  2) amm_effect_by_point.csv   — per-point paired Δ (with-AMM − without-AMM)
                                 for each outcome × scenario
  3) Heatmap PNGs:
       - stability_map_{x}_{y}.png      (2D projection: stable vs unstable)
       - amm_effect_{outcome}_{x}_{y}.png  (2D projection of AMM-effect)

  4) regression_summary.csv    — OLS of Δ(outcome) on composition variables.

Usage:
    python tools/composition_analysis.py \
        --results-dir output/composition \
        --design output/composition/design.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

try:
    from AgentBasedModel.visualization.paper_style import use_paper_style
    use_paper_style()
except Exception:
    pass

COMPOSITION_KEYS = [
    'n_mm', 'n_fast_lp', 'n_latent_lp',
    'n_clob_fund', 'n_clob_chart', 'n_clob_univ',
    'n_fx_takers', 'n_fx_fund', 'n_retail', 'n_institutional', 'n_noise',
]

# Outcomes for AMM-effect map
OUTCOMES = [
    'avg_clob_spread_bps',
    'avg_clob_depth',
    'avg_realized_volatility',
    'avg_cost_clob_q5',
    'tail_realized_volatility',
]


def _load_chunks(results_dir: str) -> List[dict]:
    """Find all results_chunk_*.csv files and concatenate."""
    pattern = os.path.join(results_dir, 'results*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No results CSVs in {results_dir}')
    rows = []
    for fp in files:
        with open(fp, newline='') as f:
            for row in csv.DictReader(f):
                rows.append(row)
    print(f'  Loaded {len(rows)} rows from {len(files)} chunk file(s)')
    return rows


def _safe_float(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _load_criteria(path: str = 'calibration/stability_criteria.json') -> Dict:
    with open(path) as f:
        return json.load(f)


def _is_stable(row: Dict, criteria: List[Dict]) -> bool:
    """Apply per-criterion thresholds. Returns True iff ALL pass."""
    for c in criteria:
        value = _safe_float(row.get(c['metric']))
        if not math.isfinite(value):
            return False
        if 'min' in c and value < c['min']:
            return False
        if 'max' in c and value > c['max']:
            return False
    return True


def stability_map(rows: List[dict], criteria: List[Dict],
                  scenarios_to_eval: List[str]) -> Dict[int, Dict]:
    """For each point, evaluate stability per scenario × branch."""
    point_metrics: Dict[int, Dict] = defaultdict(lambda: {
        'composition': {},
        'stability_per_seed': defaultdict(list),  # (scenario, branch) -> list[bool]
        'metrics': defaultdict(lambda: defaultdict(list)),  # (scenario, branch) -> {metric -> [vals]}
    })
    for row in rows:
        pid = int(row['point_id'])
        scenario = row['scenario']
        branch = row['branch']
        if scenario not in scenarios_to_eval:
            continue
        if not point_metrics[pid]['composition']:
            point_metrics[pid]['composition'] = {k: int(row[k]) for k in COMPOSITION_KEYS}
        is_stab = _is_stable(row, criteria)
        point_metrics[pid]['stability_per_seed'][(scenario, branch)].append(is_stab)
        for o in OUTCOMES + ['avg_flow_share_amm_total', 'avg_systemic_liquidity', 'n_trades']:
            point_metrics[pid]['metrics'][(scenario, branch)][o].append(_safe_float(row.get(o)))
    return point_metrics


def _summarize_stability(point_metrics: Dict[int, Dict]) -> List[Dict]:
    """Per-point stability summary row."""
    summary = []
    for pid, info in sorted(point_metrics.items()):
        row = {'point_id': pid, **info['composition']}
        # Stability rates per (scenario, branch)
        for (sc, br), stabs in info['stability_per_seed'].items():
            n = len(stabs)
            n_stable = sum(1 for s in stabs if s)
            row[f'{sc}__{br}__stable_rate'] = n_stable / n if n > 0 else float('nan')
        # Overall: stable if >=50% seeds stable in 'default' for both branches
        with_d = info['stability_per_seed'].get(('default', 'with_amm'), [])
        without_d = info['stability_per_seed'].get(('default', 'without_amm'), [])
        rate_with = sum(with_d) / len(with_d) if with_d else 0.0
        rate_without = sum(without_d) / len(without_d) if without_d else 0.0
        row['overall_stable'] = bool(rate_with >= 0.5 and rate_without >= 0.5)
        row['overall_stable_rate'] = (rate_with + rate_without) / 2 if (with_d and without_d) else 0.0
        # Mean outcomes for default scenario, both branches
        for o in OUTCOMES + ['avg_flow_share_amm_total']:
            mw = info['metrics'].get(('default', 'with_amm'), {}).get(o, [])
            mo = info['metrics'].get(('default', 'without_amm'), {}).get(o, [])
            mw_clean = [x for x in mw if math.isfinite(x)]
            mo_clean = [x for x in mo if math.isfinite(x)]
            row[f'default__with_amm__{o}'] = float(np.mean(mw_clean)) if mw_clean else float('nan')
            row[f'default__without_amm__{o}'] = float(np.mean(mo_clean)) if mo_clean else float('nan')
        summary.append(row)
    return summary


def _paired_permutation_p(deltas: List[float], n_reps: int = 5000, seed: int = 0) -> float:
    """Two-sided paired permutation test on per-seed deltas.

    H0: mean Δ = 0 (random sign-flips have the same distribution as observed).
    """
    deltas = [d for d in deltas if math.isfinite(d)]
    if len(deltas) < 5:
        return float('nan')
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(deltas)))
    n = len(deltas)
    deltas_np = np.array(deltas)
    flips = rng.choice([-1.0, 1.0], size=(n_reps, n))
    perm_means = np.abs((flips * deltas_np).mean(axis=1))
    p = float((perm_means >= obs).sum() + 1) / float(n_reps + 1)
    return p


def _bootstrap_ci(deltas: List[float], n_reps: int = 2000, ci: float = 0.95,
                  seed: int = 0) -> Tuple[float, float]:
    """Bootstrap CI for mean Δ."""
    deltas = [d for d in deltas if math.isfinite(d)]
    if len(deltas) < 5:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    arr = np.array(deltas)
    n = len(arr)
    boots = np.empty(n_reps)
    for i in range(n_reps):
        boots[i] = rng.choice(arr, size=n, replace=True).mean()
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return (lo, hi)


def _compute_amm_effects(point_metrics: Dict[int, Dict],
                          include_significance: bool = True,
                          perm_reps: int = 5000,
                          boot_reps: int = 2000) -> List[Dict]:
    """For each (point, scenario, outcome) compute paired Δ statistics.

    Per-seed lists for with_amm and without_amm are aligned positionally (same
    seed order within each branch, by construction of the sweep runner), so
    paired Δ[s] = with[s] − without[s].
    """
    effects = []
    for pid, info in sorted(point_metrics.items()):
        row = {'point_id': pid, **info['composition']}
        for sc in {s for (s, _) in info['metrics']}:
            for o in OUTCOMES:
                mw = info['metrics'].get((sc, 'with_amm'), {}).get(o, [])
                mo = info['metrics'].get((sc, 'without_amm'), {}).get(o, [])
                # paired by index
                k = min(len(mw), len(mo))
                if k < 5:
                    continue
                pair_deltas = [mw[i] - mo[i]
                               for i in range(k)
                               if math.isfinite(mw[i]) and math.isfinite(mo[i])]
                if len(pair_deltas) < 5:
                    continue
                mean_w = float(np.mean([x for x in mw if math.isfinite(x)]))
                mean_o = float(np.mean([x for x in mo if math.isfinite(x)]))
                row[f'{sc}__{o}__with'] = mean_w
                row[f'{sc}__{o}__without'] = mean_o
                row[f'{sc}__{o}__delta'] = float(np.mean(pair_deltas))
                row[f'{sc}__{o}__n_pairs'] = len(pair_deltas)
                if include_significance:
                    # Deterministic seed per (point, scenario, outcome) for reproducibility
                    s_perm = abs(hash((pid, sc, o, 'perm'))) % (2 ** 32)
                    s_boot = abs(hash((pid, sc, o, 'boot'))) % (2 ** 32)
                    row[f'{sc}__{o}__pvalue'] = _paired_permutation_p(
                        pair_deltas, n_reps=perm_reps, seed=s_perm)
                    lo, hi = _bootstrap_ci(pair_deltas, n_reps=boot_reps, seed=s_boot)
                    row[f'{sc}__{o}__ci_lo'] = lo
                    row[f'{sc}__{o}__ci_hi'] = hi
        effects.append(row)
    return effects


def _write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        print(f'  [skip] no rows for {path}')
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = list(all_keys)
    # put point_id first
    if 'point_id' in fieldnames:
        fieldnames.remove('point_id')
        fieldnames = ['point_id'] + sorted(fieldnames)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'  Saved {len(rows)} rows → {path}')


def _hex_grid_scatter(xs, ys, vals, x_lab, y_lab, title, out_path,
                      cmap='RdBu_r', symmetric=True, vmin=None, vmax=None):
    """2D scatter plot of (x, y) with color=val. Used for both stability map
    (val ∈ {0,1}) and AMM-effect map (val continuous)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if symmetric and vals:
        finite_vals = [v for v in vals if math.isfinite(v)]
        if finite_vals:
            abs_max = max(abs(min(finite_vals)), abs(max(finite_vals)), 1e-9)
            vmin = vmin if vmin is not None else -abs_max
            vmax = vmax if vmax is not None else abs_max
    sc = ax.scatter(xs, ys, c=vals, s=85, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=ax)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'  PNG → {out_path}')


def _ols(X: np.ndarray, y: np.ndarray) -> Dict:
    """Simple OLS with t-stat reporting (no statsmodels dependency)."""
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    rss = float(np.sum(resid ** 2))
    df = max(1, n - k - 1)
    sigma2 = rss / df
    cov = sigma2 * np.linalg.pinv(Xb.T @ Xb)
    se = np.sqrt(np.diag(cov))
    tstat = beta / np.where(se > 0, se, 1.0)
    return {
        'beta': beta.tolist(),
        'se': se.tolist(),
        't_stat': tstat.tolist(),
        'n': n,
        'df': df,
    }


def _regression(effects: List[Dict], outcome: str, scenario: str) -> Dict:
    key = f'{scenario}__{outcome}__delta'
    xs, ys = [], []
    for r in effects:
        y = _safe_float(r.get(key))
        if not math.isfinite(y):
            continue
        x = [int(r[k]) for k in COMPOSITION_KEYS]
        xs.append(x)
        ys.append(y)
    if len(xs) < len(COMPOSITION_KEYS) + 2:
        return {'error': f'too few points ({len(xs)})'}
    X = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    return _ols(X, y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='output/composition')
    p.add_argument('--design', default='output/composition/design.csv')
    p.add_argument('--criteria', default='calibration/stability_criteria.json')
    p.add_argument('--out-dir', default='output/composition/analysis')
    p.add_argument('--projection-x', default='n_mm',
                   help='X-axis variable for 2D maps')
    p.add_argument('--projection-y', default='lp_total',
                   help='Y-axis variable for 2D maps (use "lp_total" or "book_total" for sums)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Loading results from {args.results_dir}...')
    rows = _load_chunks(args.results_dir)

    print(f'Loading stability criteria from {args.criteria}...')
    crit_payload = _load_criteria(args.criteria)
    criteria = crit_payload['criteria']

    # Discover all scenarios present in the data
    all_scenarios = sorted({r['scenario'] for r in rows})
    print(f'  Scenarios in data: {all_scenarios}')

    # 1) Per-point stability map (stability is judged from `default` scenario;
    # we still pass every scenario into the index so AMM-effect computations
    # cover all of them).
    print('Computing stability map...')
    pm = stability_map(rows, criteria, scenarios_to_eval=all_scenarios)
    summary_rows = _summarize_stability(pm)
    _write_csv(summary_rows, os.path.join(args.out_dir, 'stability_map.csv'))

    # 2) Per-point AMM effects across scenarios
    print('Computing AMM-effect deltas...')
    effects = _compute_amm_effects(pm)
    _write_csv(effects, os.path.join(args.out_dir, 'amm_effect_by_point.csv'))

    # 3) 2D heatmap-style scatter projections
    def _x_for(row):
        if args.projection_x == 'lp_total':
            return row.get('n_fast_lp', 0) + row.get('n_latent_lp', 0)
        if args.projection_x == 'book_total':
            return row.get('n_clob_fund', 0) + row.get('n_clob_chart', 0) + row.get('n_clob_univ', 0)
        return row.get(args.projection_x, 0)
    def _y_for(row):
        if args.projection_y == 'lp_total':
            return row.get('n_fast_lp', 0) + row.get('n_latent_lp', 0)
        if args.projection_y == 'book_total':
            return row.get('n_clob_fund', 0) + row.get('n_clob_chart', 0) + row.get('n_clob_univ', 0)
        return row.get(args.projection_y, 0)

    # Multi-panel stability diagnostic (replaces old binary scatter which was
    # uninformative — all 200 points pass, so the binary version had no signal).
    try:
        from tools.composition_plots import plot_stability_diagnostic
        plot_stability_diagnostic(
            summary_rows, criteria,
            os.path.join(args.out_dir,
                         f'stability_diagnostic__{args.projection_x}_x_{args.projection_y}.png'),
            x_var=args.projection_x, y_var=args.projection_y,
        )
    except ImportError as e:
        print(f'  [warn] could not generate stability diagnostic: {e}')

    # AMM effect maps (stable points only)
    stable_ids = {r['point_id'] for r in summary_rows if r['overall_stable']}
    scenarios_seen = set()
    for r in effects:
        for k in r.keys():
            if '__' in k and k.endswith('__delta'):
                scenarios_seen.add(k.split('__')[0])
    for outcome in OUTCOMES:
        for scenario in scenarios_seen:
            key = f'{scenario}__{outcome}__delta'
            xs_e, ys_e, vals = [], [], []
            for r in effects:
                if r['point_id'] not in stable_ids:
                    continue
                v = _safe_float(r.get(key))
                if not math.isfinite(v):
                    continue
                xs_e.append(_x_for(r))
                ys_e.append(_y_for(r))
                vals.append(v)
            if not vals:
                continue
            _hex_grid_scatter(
                xs_e, ys_e, vals,
                args.projection_x, args.projection_y,
                f'AMM effect on {outcome}\n(scenario={scenario}, stable points only)',
                os.path.join(args.out_dir, 'amm_effect',
                             f'{scenario}__{outcome}__{args.projection_x}_x_{args.projection_y}.png'),
            )

    # 4) Regression summary
    print('Running regressions...')
    reg_rows = []
    for scenario in all_scenarios:
        for outcome in OUTCOMES:
            res = _regression(effects, outcome, scenario)
            if 'error' in res:
                continue
            row = {'scenario': scenario, 'outcome': outcome, 'n': res['n'], 'df': res['df']}
            for i, key in enumerate(['intercept'] + COMPOSITION_KEYS):
                row[f'beta_{key}'] = res['beta'][i]
                row[f'tstat_{key}'] = res['t_stat'][i]
            reg_rows.append(row)
    _write_csv(reg_rows, os.path.join(args.out_dir, 'regression_summary.csv'))

    # 5) New visualization suite (threshold bars + projection panels + violins
    #    + improved heatmaps + importance summaries). All use a shared color
    #    scale per outcome across scenarios for proper magnitude comparison.
    print('Generating extended visualization suite...')
    try:
        from tools.composition_plots import (
            plot_threshold_bars, plot_projection_panel, plot_marginal_violins,
            plot_heatmap_v2, plot_importance_summary, global_scale_for_outcome,
            plot_fine_nmm_threshold, plot_fraction_significant,
        )
    except ImportError as e:
        print(f'  [warn] could not import composition_plots: {e}')
    else:
        stable_effects = [r for r in effects if int(r.get('point_id', -1)) in stable_ids]
        # Pre-compute shared scales per outcome across scenarios
        shared_scales: Dict[str, tuple] = {}
        for outcome in OUTCOMES:
            shared_scales[outcome] = global_scale_for_outcome(
                stable_effects, outcome, all_scenarios
            )

        for scenario in all_scenarios:
            for outcome in OUTCOMES:
                # (a) Threshold bars for n_mm and lp_total
                for cvar in ['n_mm', 'lp_total', 'book_total', 'taker_total']:
                    plot_threshold_bars(
                        stable_effects, cvar, scenario, outcome,
                        os.path.join(args.out_dir, 'thresholds',
                                     f'{scenario}__{outcome}__by_{cvar}.png'),
                    )
                # (a2) Fine n_mm threshold (isolates n_mm=0 from n_mm>=1 plateau)
                plot_fine_nmm_threshold(
                    stable_effects, scenario, outcome,
                    os.path.join(args.out_dir, 'thresholds_fine',
                                 f'{scenario}__{outcome}__by_exact_n_mm.png'),
                )
                # (a3) Per-composition significance: fraction helps/hurts/NS by n_mm
                plot_fraction_significant(
                    stable_effects, scenario, outcome,
                    os.path.join(args.out_dir, 'fraction_significant',
                                 f'{scenario}__{outcome}__by_n_mm.png'),
                )
                # (b) Projection panel
                plot_projection_panel(
                    stable_effects, scenario, outcome,
                    os.path.join(args.out_dir, 'projections',
                                 f'{scenario}__{outcome}__panel.png'),
                    global_vmin=shared_scales[outcome][0],
                    global_vmax=shared_scales[outcome][1],
                )
                # (c) Marginal violins
                plot_marginal_violins(
                    stable_effects, scenario, outcome,
                    os.path.join(args.out_dir, 'marginals',
                                 f'{scenario}__{outcome}__violins.png'),
                )
                # (d) Improved heatmap with fixed scale (n_mm × lp_total)
                vmin, vmax = shared_scales[outcome]
                plot_heatmap_v2(
                    stable_effects, scenario, outcome,
                    os.path.join(args.out_dir, 'heatmaps_v2',
                                 f'{scenario}__{outcome}__n_mm_x_lp_total.png'),
                    x_var='n_mm', y_var='lp_total',
                    vmin=vmin, vmax=vmax,
                )
                # (e) Importance summary from regression
                reg_row = next((r for r in reg_rows
                               if r['scenario'] == scenario and r['outcome'] == outcome), None)
                if reg_row:
                    plot_importance_summary(
                        reg_row, scenario, outcome,
                        os.path.join(args.out_dir, 'importance',
                                     f'{scenario}__{outcome}__importance.png'),
                    )
        print('  ✓ extended visualization suite done')

    n_stable = sum(1 for r in summary_rows if r['overall_stable'])
    print()
    print(f'=== Composition sweep analysis complete ===')
    print(f'Total LHS points analysed: {len(summary_rows)}')
    print(f'Stable points (default scenario, both branches ≥50% seed-stability): {n_stable} ({n_stable/len(summary_rows)*100:.1f}%)')
    print(f'Outputs in {args.out_dir}/')


if __name__ == '__main__':
    main()
