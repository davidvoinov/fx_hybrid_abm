"""Composition sensitivity sweep — main runner.

For each LHS design point in the input CSV, runs paired simulations
(with-AMM and without-AMM) across multiple seeds and scenarios. Writes one
row per (point_id, seed, scenario, branch) with stability indicators and
key outcome metrics.

Designed for SPLIT execution: pass --chunk-id N --total-chunks K to process
only design points where (point_id % K == N). Each machine's chunk produces
a separate output CSV. Merge later with composition_analysis.py.

Usage (single machine, full run):
    python tools/composition_sweep.py \
        --design output/composition/design.csv \
        --out output/composition/results.csv

Defaults: 50 seeds × 1000 iter × 3 scenarios (default + dealer_liquidity_crisis +
high_vol_stress) × 2 branches.

Usage (split into 4 chunks across 4 machines):
    # On PC 1:
    python tools/composition_sweep.py --chunk-id 0 --total-chunks 4 \
        --design output/composition/design.csv \
        --out output/composition/results_chunk_0_of_4.csv
    # On PC 2:
    python tools/composition_sweep.py --chunk-id 1 --total-chunks 4 \
        --design output/composition/design.csv \
        --out output/composition/results_chunk_1_of_4.csv
    # ...etc
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from main import build_parser, build_sim, _seed_all  # noqa: E402


COMPOSITION_KEYS = [
    'n_mm', 'n_fast_lp', 'n_latent_lp',
    'n_clob_fund', 'n_clob_chart', 'n_clob_univ',
    'n_fx_takers', 'n_fx_fund', 'n_retail', 'n_institutional', 'n_noise',
]

# Preset CLI flag for each scenario (these align with main.py REALISM_PRESETS).
SCENARIO_PRESET = {
    'default': None,                                      # no preset, calm
    'mm_withdrawal': 'mm_withdrawal',
    'flash_crash': 'flash_crash',
    'dealer_liquidity_crisis': 'dealer_liquidity_crisis',
    'funding_liquidity_shock': 'funding_liquidity_shock',
    'high_vol_stress': 'high_vol_stress',
}


def _load_design(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _row_for_chunk(rows: List[Dict], chunk_id: int, total_chunks: int) -> List[Dict]:
    if total_chunks <= 1:
        return rows
    return [r for r in rows if int(r['point_id']) % total_chunks == chunk_id]


def _build_args_for_point(parser, point: Dict, scenario: str, n_iter: int,
                          seed: int, with_amm: bool):
    """Construct args namespace for a single (point, scenario, seed, branch)."""
    base_args = parser.parse_args([])  # defaults from primary_model.json
    # Inject composition counts
    for key in COMPOSITION_KEYS:
        setattr(base_args, key, int(point[key]))
    # Override n_iter
    base_args.n_iter = n_iter
    # Scenario
    preset = SCENARIO_PRESET.get(scenario)
    base_args.preset = preset
    # With/without AMM
    base_args.enable_amm = 1 if with_amm else 0
    # Silence
    base_args.silent = True
    return base_args


def _mean_finite(values) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else float('nan')


def _tail_mean(values, window: int = 100) -> float:
    if not values:
        return float('nan')
    tail = list(values)[-window:]
    return _mean_finite(tail)


def _rolling_return_vol(mid_series, window: int = 20):
    out = []
    for i in range(len(mid_series)):
        if i < window:
            out.append(float('nan'))
            continue
        slice_ = mid_series[i - window: i]
        if not slice_ or slice_[0] is None or slice_[0] == 0:
            out.append(float('nan'))
            continue
        rets = []
        for j in range(1, len(slice_)):
            try:
                if slice_[j-1] > 0:
                    rets.append((slice_[j] - slice_[j-1]) / slice_[j-1])
            except Exception:
                continue
        if len(rets) < 2:
            out.append(float('nan'))
            continue
        mean_r = sum(rets) / len(rets)
        var_r = sum((x - mean_r) ** 2 for x in rets) / max(1, len(rets) - 1)
        out.append(var_r ** 0.5)
    return out


def _measure_stability(sim) -> Dict[str, float]:
    """Pull stability indicators from logger after simulate().

    Replicates the metric computation used by tests/stat_tests.py to ensure
    the stability filter operates on the same scales.
    """
    if sim.logger is None:
        return {}
    logger = sim.logger
    summary = logger.summary()
    depth_series = [d.get('total', float('nan')) for d in (logger.clob_depth or [])]
    realized_vol = _rolling_return_vol(logger.clob_mid_series, window=20)
    amm_flow_share = sum(
        float(summary.get(f'avg_flow_share_{venue}', 0.0))
        for venue in (logger.amm_cost_curves or {})
    )
    return {
        'avg_clob_spread_bps':      _mean_finite(logger.clob_qspr),
        'avg_clob_depth':           _mean_finite(depth_series),
        'avg_realized_volatility':  _mean_finite(realized_vol),
        'avg_systemic_liquidity':   _mean_finite(getattr(logger, 'systemic_liquidity_series', [])),
        'n_trades':                 float(summary.get('n_trades', 0)),
        'avg_flow_share_amm_total': amm_flow_share,
        'avg_cost_clob_q5':         float(summary.get('avg_cost_clob_Q5', float('nan'))),
        'tail_realized_volatility': _tail_mean(realized_vol),
    }


def _apply_preset_args(args, preset):
    """Apply scenario preset (calls main's preset_defaults logic)."""
    if preset is None:
        return
    args.preset = preset
    from main import _apply_preset_defaults, build_parser as _bp
    _apply_preset_defaults(_bp(), args)


def run_one(parser, point: Dict, scenario: str, seed: int, with_amm: bool,
            n_iter: int) -> Dict:
    """Run one simulation, return stability + outcome metrics."""
    args = _build_args_for_point(parser, point, scenario, n_iter, seed, with_amm)
    _apply_preset_args(args, SCENARIO_PRESET.get(scenario))
    _seed_all(seed)
    sim = build_sim(args)
    sim.simulate(n_iter, silent=True)
    metrics = _measure_stability(sim)
    return metrics


def main() -> None:
    parser_run = argparse.ArgumentParser(description="Composition sensitivity sweep.")
    parser_run.add_argument('--design', default='output/composition/design.csv')
    parser_run.add_argument('--seeds', type=int, default=50,
                            help='Seeds per (point, scenario, branch). Default 50.')
    parser_run.add_argument('--base-seed', type=int, default=42)
    parser_run.add_argument('--n-iter', type=int, default=1000)
    parser_run.add_argument('--scenarios', default='default,dealer_liquidity_crisis,high_vol_stress',
                            help='Comma-separated scenario names. Default covers calm + acute dealer crisis + sustained vol stress.')
    parser_run.add_argument('--chunk-id', type=int, default=0,
                            help='Index of this chunk (0 to total_chunks-1)')
    parser_run.add_argument('--total-chunks', type=int, default=1,
                            help='Number of parallel chunks total (1 = no split)')
    parser_run.add_argument('--out', default='output/composition/results.csv')
    parser_run.add_argument('--progress-every', type=int, default=1,
                            help='Print progress after every N points (default 1)')
    parser_run.add_argument('--resume', action='store_true',
                            help='If --out exists, skip (point, scenario, branch, seed) tuples already present and append. Otherwise overwrite from scratch.')
    args = parser_run.parse_args()

    if args.total_chunks < 1 or args.chunk_id < 0 or args.chunk_id >= args.total_chunks:
        raise ValueError(f'Bad chunk config: chunk_id={args.chunk_id}, total={args.total_chunks}')

    design = _load_design(args.design)
    my_points = _row_for_chunk(design, args.chunk_id, args.total_chunks)
    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    branches = [('with_amm', True), ('without_amm', False)]

    n_sims = len(my_points) * args.seeds * len(scenarios) * len(branches)
    print(f'=== Composition sweep ===')
    print(f'Chunk:       {args.chunk_id} of {args.total_chunks}')
    print(f'My points:   {len(my_points)} (out of {len(design)} total)')
    print(f'Seeds/point: {args.seeds}')
    print(f'Scenarios:   {len(scenarios)} -> {scenarios}')
    print(f'Branches:    {len(branches)}')
    print(f'Total sims:  {n_sims}')
    print(f'Output CSV:  {args.out}')
    print(f'==========================')
    print()

    base_main_parser = build_parser()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        'point_id', 'chunk_id', 'seed', 'scenario', 'branch',
        *COMPOSITION_KEYS,
        'avg_clob_spread_bps', 'avg_clob_depth', 'avg_realized_volatility',
        'avg_systemic_liquidity', 'n_trades', 'avg_flow_share_amm_total',
        'avg_cost_clob_q5', 'tail_realized_volatility',
        'wall_seconds',
    ]

    # Resume support: load any existing (point_id, scenario, branch, seed) tuples
    # from --out so we can skip them. Open file in append vs write mode accordingly.
    existing_keys: set = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline='') as f_in:
            reader = csv.DictReader(f_in)
            for r in reader:
                existing_keys.add((int(r['point_id']), r['scenario'],
                                   r['branch'], int(r['seed'])))
        print(f'[resume] Loaded {len(existing_keys)} existing rows from {args.out}')
        open_mode = 'a'
        write_header = False
    else:
        open_mode = 'w'
        write_header = True

    started_at = time.time()
    with open(args.out, open_mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        sims_done = 0
        sims_skipped = 0
        for p_idx, point in enumerate(my_points):
            point_id = int(point['point_id'])
            point_started = time.time()
            for scenario in scenarios:
                for branch_name, with_amm in branches:
                    for s_idx in range(args.seeds):
                        seed = args.base_seed + s_idx + 100_000 * point_id
                        if (point_id, scenario, branch_name, seed) in existing_keys:
                            sims_skipped += 1
                            continue
                        t0 = time.time()
                        try:
                            metrics = run_one(
                                base_main_parser, point, scenario, seed, with_amm,
                                args.n_iter,
                            )
                        except Exception as e:
                            metrics = {k: float('nan') for k in (
                                'avg_clob_spread_bps', 'avg_clob_depth',
                                'avg_realized_volatility', 'avg_systemic_liquidity',
                                'n_trades', 'avg_flow_share_amm_total',
                                'avg_cost_clob_q5', 'tail_realized_volatility',
                            )}
                            metrics['_error'] = str(e)[:120]
                        wall = time.time() - t0

                        row = {
                            'point_id': point_id,
                            'chunk_id': args.chunk_id,
                            'seed': seed,
                            'scenario': scenario,
                            'branch': branch_name,
                            **{k: int(point[k]) for k in COMPOSITION_KEYS},
                            **metrics,
                            'wall_seconds': round(wall, 3),
                        }
                        w.writerow(row)
                        f.flush()
                        sims_done += 1

            elapsed = time.time() - started_at
            point_elapsed = time.time() - point_started
            avg_per_sim = elapsed / max(1, sims_done)
            remaining = n_sims - sims_done - sims_skipped
            eta_seconds = avg_per_sim * remaining
            eta_h = eta_seconds / 3600.0
            if (p_idx + 1) % args.progress_every == 0:
                print(f'  point {p_idx+1}/{len(my_points)} (id={point_id}) done in '
                      f'{point_elapsed:.1f}s | new={sims_done} skipped={sims_skipped} '
                      f'total={sims_done+sims_skipped}/{n_sims} | '
                      f'elapsed {elapsed/60:.1f}m | ETA {eta_h:.2f}h')

    elapsed = time.time() - started_at
    print()
    print(f'=== Chunk {args.chunk_id} complete ===')
    print(f'Wrote {sims_done} new rows (skipped {sims_skipped} existing) in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)')
    print(f'Output: {args.out}')


if __name__ == '__main__':
    main()
