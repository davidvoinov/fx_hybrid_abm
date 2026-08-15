"""Find the LHS composition(s) that best match the calibrated baseline moments.

For each LHS point we already have realized outcomes from the default scenario
(with-AMM) in stability_map.csv. We compare those outcomes to the EUR/USD-style
calibration targets defined in calibration/primary_model_targets.json and rank
points by total normalized distance.

This is NOT a recommendation to replace the current baseline -- it is a
disclosure tool: "of 200 alternative compositions, the following N produce
moments closest to BIS/Ranaldo/Lo-Hall targets; compare to current baseline".

Outputs:
  calibration_match.csv    — per-point distance breakdown, sorted
  best_baselines.csv       — top 10 candidates with full agent counts
  baseline_comparison.csv  — comparison vs current baseline from primary_model.json

Usage:
    python tools/composition_calibration_match.py \
        --stability-map output/composition/analysis/stability_map.csv \
        --targets calibration/primary_model_targets.json \
        --primary calibration/primary_model.json \
        --out-dir output/composition/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COMPOSITION_KEYS = [
    'n_mm', 'n_fast_lp', 'n_latent_lp',
    'n_clob_fund', 'n_clob_chart', 'n_clob_univ',
    'n_fx_takers', 'n_fx_fund', 'n_retail', 'n_institutional', 'n_noise',
]

# Map calibration targets to realized columns in stability_map.csv
# (we use default scenario, with-AMM branch).
TARGET_TO_COL = {
    'quoted_spread_bps':  'default__with_amm__avg_clob_spread_bps',
    'near_touch_depth':   'default__with_amm__avg_clob_depth',
    'amm_volume_share':   'default__with_amm__avg_flow_share_amm_total',
    # cross_venue_basis_bps was withdrawn as a calibration target after its
    # cited source proved unrelated; it remains a runtime diagnostic only.
    # We use what's available; volatility & cost are diagnostics but not gating
    'avg_realized_volatility': 'default__with_amm__avg_realized_volatility',
}


def _safe_float(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _load_targets(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)['targets']


def _load_primary(path: str) -> Dict:
    with open(path) as f:
        return json.load(f).get('cli_defaults', {})


def _load_stability_map(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _target_distance(realized: float, target_def: Dict) -> float:
    """Return normalized distance: 0 if within accepted band, scaled excess otherwise."""
    if not math.isfinite(realized):
        return float('inf')
    band = target_def.get('accepted_error_band', {})
    rng = target_def.get('target_range')
    val = target_def.get('target_value')

    # Distance is excess beyond accepted band, normalized to band width.
    # Within target → 0; outside → grows linearly.
    if rng is not None:
        lo, hi = rng['low'], rng['high']
        abs_tol = band.get('absolute', 0.0)
        rel_tol = band.get('relative', 0.0)
        # Effective band
        eff_lo = lo - abs_tol - rel_tol * abs(lo)
        eff_hi = hi + abs_tol + rel_tol * abs(hi)
        if eff_lo <= realized <= eff_hi:
            return 0.0
        gap = (eff_lo - realized) if realized < eff_lo else (realized - eff_hi)
        denom = max(abs(eff_hi - eff_lo), 1e-9)
        return gap / denom
    elif val is not None:
        abs_tol = band.get('absolute', 0.0)
        rel_tol = band.get('relative', 0.0) * abs(val)
        tol = max(abs_tol, rel_tol, 1e-9)
        gap = abs(realized - val)
        if gap <= tol:
            return 0.0
        return (gap - tol) / tol
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stability-map', default='output/composition/analysis/stability_map.csv')
    p.add_argument('--targets', default='calibration/primary_model_targets.json')
    p.add_argument('--primary', default='calibration/primary_model.json')
    p.add_argument('--out-dir', default='output/composition/analysis')
    p.add_argument('--top-n', type=int, default=10, help='Report top N baselines')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Loading stability map: {args.stability_map}')
    points = _load_stability_map(args.stability_map)
    print(f'  {len(points)} LHS points')

    print(f'Loading targets: {args.targets}')
    all_targets = _load_targets(args.targets)
    # Keep only targets we can evaluate from stability_map columns
    usable_targets = [t for t in all_targets if t['observable'] in TARGET_TO_COL]
    print(f'  Using {len(usable_targets)} targets evaluable from default scenario: '
          f'{[t["observable"] for t in usable_targets]}')

    print(f'Loading primary baseline: {args.primary}')
    primary = _load_primary(args.primary)
    primary_composition = {k: int(primary.get(k, 0)) for k in COMPOSITION_KEYS}

    # Compute distances per point per target
    distance_rows = []
    for point in points:
        if point.get('overall_stable') not in ('True', 'true', '1', True):
            continue
        row = {'point_id': point['point_id']}
        for k in COMPOSITION_KEYS:
            row[k] = int(point.get(k, 0))
        total = 0.0
        n_eval = 0
        for t in usable_targets:
            col = TARGET_TO_COL[t['observable']]
            realized = _safe_float(point.get(col))
            d = _target_distance(realized, t)
            row[f'dist__{t["observable"]}'] = d
            row[f'realized__{t["observable"]}'] = realized
            total += d
            n_eval += 1
        row['total_distance'] = total
        row['n_targets_eval'] = n_eval
        distance_rows.append(row)

    distance_rows.sort(key=lambda r: r['total_distance'])

    # Write full sorted table
    out_full = os.path.join(args.out_dir, 'calibration_match.csv')
    with open(out_full, 'w', newline='') as f:
        if distance_rows:
            w = csv.DictWriter(f, fieldnames=list(distance_rows[0].keys()))
            w.writeheader()
            for r in distance_rows:
                w.writerow(r)
    print(f'  Wrote calibration distances → {out_full}')

    # Top-N
    out_best = os.path.join(args.out_dir, 'best_baselines.csv')
    with open(out_best, 'w', newline='') as f:
        if distance_rows:
            w = csv.DictWriter(f, fieldnames=list(distance_rows[0].keys()))
            w.writeheader()
            for r in distance_rows[:args.top_n]:
                w.writerow(r)
    print(f'  Wrote top-{args.top_n} → {out_best}')

    # Baseline comparison
    print('\n=== Comparison to current baseline ===')
    print(f'Current baseline from {os.path.basename(args.primary)}:')
    for k in COMPOSITION_KEYS:
        print(f'  {k:<20} = {primary_composition[k]}')

    print(f'\nTop-{min(5, args.top_n)} calibration-matching alternatives:')
    print(f'{"point_id":<10} {"total_dist":<12} ' + ' '.join(f'{k:<10}' for k in COMPOSITION_KEYS))
    for r in distance_rows[:5]:
        print(f'{r["point_id"]:<10} {r["total_distance"]:<12.4f} '
              + ' '.join(f'{r[k]:<10}' for k in COMPOSITION_KEYS))

    # Write structured comparison row
    out_cmp = os.path.join(args.out_dir, 'baseline_comparison.csv')
    with open(out_cmp, 'w', newline='') as f:
        fieldnames = ['source'] + COMPOSITION_KEYS + ['total_distance']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({'source': 'current_primary_baseline', **primary_composition, 'total_distance': ''})
        for r in distance_rows[:args.top_n]:
            row = {'source': f'lhs_point_{r["point_id"]}', 'total_distance': r['total_distance']}
            for k in COMPOSITION_KEYS:
                row[k] = r[k]
            w.writerow(row)
    print(f'\n  Wrote baseline comparison → {out_cmp}')

    n_perfect = sum(1 for r in distance_rows if r['total_distance'] == 0)
    print(f'\nSummary: {n_perfect} / {len(distance_rows)} stable points have distance=0 '
          f'(within all evaluable target bands). '
          f'Best non-trivial: point {distance_rows[0]["point_id"]} '
          f'with distance {distance_rows[0]["total_distance"]:.4f}.')


if __name__ == '__main__':
    main()
