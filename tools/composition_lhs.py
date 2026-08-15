"""Latin Hypercube Sampling for agent composition sensitivity sweep.

Generates N design points across the model's agent-composition space. Each
point is a full set of agent counts. Output is a single CSV that can be split
into chunks for parallel execution on multiple machines.

This sampler is intentionally NOT BIS-anchored. Per the research design, we
characterize the AMM-effect across the *space of plausible compositions* and
later filter to compositions where the model is stable. Empirical realism of
proportions is not the calibration target -- mechanism characterization is.

Usage:
    python tools/composition_lhs.py --n-points 200 --out output/composition/design.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Composition space bounds. Wide enough to span the regime from 'almost no
# dealers' to 'densely dealered', and from 'no LP layer' to 'very deep LP'.
# These are MODEL bounds, not empirical bounds.
COMPOSITION_BOUNDS: Dict[str, Tuple[int, int]] = {
    'n_mm':            (0, 10),
    'n_fast_lp':       (0, 20),
    'n_latent_lp':     (0, 12),
    'n_clob_fund':     (0, 6),
    'n_clob_chart':    (0, 5),
    'n_clob_univ':     (0, 4),
    'n_fx_takers':     (5, 30),
    'n_fx_fund':       (0, 12),
    'n_retail':        (0, 20),
    'n_institutional': (0, 8),
    'n_noise':         (5, 20),
}

# Minimum-viability constraints. Configurations that violate these are
# regenerated. The bounds above are wide on purpose; constraints keep
# samples in the 'at least pretend to be a market' region.
def _validate(counts: Dict[str, int]) -> bool:
    lp_total = counts['n_fast_lp'] + counts['n_latent_lp']
    book_total = counts['n_clob_fund'] + counts['n_clob_chart'] + counts['n_clob_univ']
    taker_total = counts['n_fx_takers'] + counts['n_fx_fund'] + counts['n_retail'] + counts['n_institutional']
    total = sum(counts.values())

    # Must have someone providing liquidity (either MM or LP)
    if counts['n_mm'] + lp_total < 1:
        return False
    # Must have someone taking liquidity
    if taker_total < 3:
        return False
    # At least some agents total (avoid degenerate near-empty markets)
    if total < 10:
        return False
    # Avoid degenerate "all-noise" configurations
    if counts['n_noise'] >= 0.6 * total:
        return False
    return True


def _sample_lhs(n_points: int, bounds: Dict[str, Tuple[int, int]], seed: int = 42) -> List[Dict[str, int]]:
    """Latin Hypercube Sampling in K dimensions, with rejection-based validation."""
    rng = np.random.default_rng(seed)
    dims = list(bounds.keys())
    K = len(dims)

    valid_points: List[Dict[str, int]] = []
    attempt = 0
    max_attempts = n_points * 20

    while len(valid_points) < n_points and attempt < max_attempts:
        # Re-generate the LHS in batches until we have enough valid points.
        # Each batch produces n_points candidates.
        candidates = []
        # LHS stratification: for each dim, take a uniform value from each of
        # n_points strata, then permute independently per dim.
        unit = (rng.random(size=(n_points, K)) + np.arange(n_points)[:, None]) / n_points
        for k, dim in enumerate(dims):
            rng.shuffle(unit[:, k])
        for i in range(n_points):
            counts = {}
            for k, dim in enumerate(dims):
                lo, hi = bounds[dim]
                counts[dim] = int(round(lo + unit[i, k] * (hi - lo)))
            candidates.append(counts)

        for counts in candidates:
            if _validate(counts):
                valid_points.append(counts)
                if len(valid_points) >= n_points:
                    break
        attempt += n_points

    if len(valid_points) < n_points:
        raise RuntimeError(
            f"Could only generate {len(valid_points)} valid LHS points after "
            f"{attempt} attempts. Try widening bounds or relaxing constraints."
        )
    return valid_points[:n_points]


def main() -> None:
    parser = argparse.ArgumentParser(description="LHS sampler for agent composition sweep.")
    parser.add_argument('--n-points', type=int, default=200,
                        help='Number of LHS design points (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='LHS seed for reproducibility (default: 42)')
    parser.add_argument('--out', default='output/composition/design.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f'Sampling {args.n_points} LHS points in {len(COMPOSITION_BOUNDS)}D composition space...')
    points = _sample_lhs(args.n_points, COMPOSITION_BOUNDS, seed=args.seed)

    fieldnames = ['point_id'] + list(COMPOSITION_BOUNDS.keys()) + ['total_agents', 'lp_total', 'book_total', 'taker_total']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, counts in enumerate(points):
            row = {'point_id': i, **counts}
            row['total_agents'] = sum(counts.values())
            row['lp_total'] = counts['n_fast_lp'] + counts['n_latent_lp']
            row['book_total'] = counts['n_clob_fund'] + counts['n_clob_chart'] + counts['n_clob_univ']
            row['taker_total'] = counts['n_fx_takers'] + counts['n_fx_fund'] + counts['n_retail'] + counts['n_institutional']
            w.writerow(row)

    print(f'Saved {len(points)} design points to {args.out}')
    print()

    # Quick summary of the sample
    import statistics
    for key in COMPOSITION_BOUNDS:
        vals = [p[key] for p in points]
        print(f'  {key:<18} min={min(vals):>3}  median={statistics.median(vals):>5.1f}  max={max(vals):>3}')


if __name__ == '__main__':
    main()
