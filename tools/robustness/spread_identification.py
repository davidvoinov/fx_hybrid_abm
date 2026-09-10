"""What sets the calm quoted spread, measured by moving one thing at a time.

The calm target on the mean quoted spread is an interval, [1.02, 1.80] basis
points, and the interval is wide: its width is three quarters of its lower
bound. A referee is entitled to ask whether a band that wide is a constraint at
all, or whether any parameterisation of this model would land inside it. The
question is answered by moving the parameters and reading the spread, and the
answer is in two parts. A band has bite if ordinary perturbations leave it,
which is a statement about the band. The spread is identified if the
perturbations that leave it are the ones that ought to, which is a statement
about the model.

Each parameter is moved on its own, by a half and by a double where it is
continuous and onto its neighbouring integers where it is a count, and the
calm scenario is rerun on a seed panel. Nothing here changes a stored result:
the file writes its own artifact and the model is put back as it was.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

TARGETS = os.path.join(ROOT, 'calibration', 'primary_model_targets.json')
OBSERVABLE = 'quoted_spread_mean_bps'
SCENARIO = 'baseline_primary'

# The quantities that could plausibly set the level of a quoted spread: the
# constant in each provider's own rule, the loading each puts on volatility,
# the size they show, the grid the quotes have to land on, the volatility of
# the price itself, and how many of each participant there are.
LADDER = {
    'mm_alpha0_base': ('halved', 'doubled'),
    'mm_alpha1': ('halved', 'doubled'),
    'mm_d0_base': ('halved', 'doubled'),
    'mm_stale_touch_ratio': ('halved', 'doubled'),
    'fast_lp_base_spread_bps': ('halved', 'doubled'),
    'fast_lp_vol_multiple': ('halved', 'doubled'),
    'fast_lp_base_withdraw_prob': ('halved', 'doubled'),
    'price_tick': ('halved', 'doubled'),
    'price_vol_scale': ('halved', 'doubled'),
    'n_mm': ('one fewer', 'one more'),
    'n_fast_lp': ('one fewer', 'one more'),
}
COUNTS = ('n_mm', 'n_fast_lp')


def _job(spec):
    name, value, seed = spec
    import calibration.runner as runner
    import main as main_module
    from calibration.fitter import CalibrationFitter

    with open(TARGETS, encoding='utf-8') as handle:
        payload = json.load(handle)
    scenario = [row for row in payload['calibration_scenarios']
                if row['name'] == SCENARIO][0]
    overrides = {} if name is None else {name: value}
    args = runner._make_scenario_args(scenario, overrides, seed=seed)
    if name is not None:
        # _make_scenario_args keeps a scenario preset where the candidate only
        # repeats the parser default, which would silently drop a perturbation
        # that happens to land on it. The calm scenario carries no preset, so
        # the value is set again here and the run is the one that was asked for.
        setattr(args, name, value)
    main_module._seed_all(seed)
    sim = main_module.build_sim(args)
    sim.simulate(args.n_iter, silent=True)
    metrics = CalibrationFitter(payload).realized_metrics(sim)
    return name, value, seed, float(metrics.get(OBSERVABLE, float('nan')))


def _median(values):
    finite = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    return statistics.median(finite) if finite else float('nan')


def _median_interval(values, draws=2000, seed=12345):
    """Bootstrap interval for a cell median.

    A perturbation is only evidence about a parameter where the shift it
    produces is larger than the spread of the panel it was measured on. At two
    seeds it was not: both halving and doubling the quoting increment moved the
    mean spread down, which is not a direction, it is noise. The interval is
    reported so that a reader can tell one from the other without rerunning
    the file.
    """
    finite = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if len(finite) < 2:
        return [float('nan'), float('nan')]
    rng = random.Random(seed)
    n = len(finite)
    medians = sorted(
        statistics.median(rng.choices(finite, k=n)) for _ in range(draws)
    )
    lo = medians[int(0.025 * draws)]
    hi = medians[min(draws - 1, int(0.975 * draws))]
    return [lo, hi]


def _target_band(target):
    """The acceptance band of a target, in whichever form it declares one.

    A pair states this target as a range on one branch and as a point with a
    tolerance on the other, and the ablation only needs the interval either
    of them denotes. Reading the range alone raised KeyError on the pair that
    carries the point, which stopped the ablation rather than measuring it.
    """
    band = target.get('target_range')
    if band:
        return float(band['low']), float(band['high'])
    centre = target.get('target_value')
    tol = target.get('accepted_error_band') or {}
    if centre is None:
        raise SystemExit(
            'target %s declares neither a range nor a value' % OBSERVABLE)
    centre = float(centre)
    if tol.get('absolute') is not None:
        width = float(tol['absolute'])
    elif tol.get('relative') is not None:
        width = abs(centre) * float(tol['relative'])
    else:
        width = 0.0
    return centre - width, centre + width


def run(seeds, workers):
    with open(TARGETS, encoding='utf-8') as handle:
        payload = json.load(handle)
    target = [row for row in payload['targets']
              if row['observable'] == OBSERVABLE][0]
    low, high = _target_band(target)

    import main as main_module
    parser = main_module.build_parser()
    base = {name: parser.get_default(name) for name in LADDER}

    specs = [(None, None, seed) for seed in seeds]
    for name in LADDER:
        centre = base[name]
        if centre is None:
            continue
        if name in COUNTS:
            moves = [int(centre) - 1, int(centre) + 1]
            moves = [m for m in moves if m >= 0]
        elif float(centre) == 0.0:
            # A multiplicative move on zero is no move. Nothing in this model
            # sets the calm spread from a parameter whose calibrated value is
            # zero, so the parameter is recorded as untestable and skipped
            # instead of being perturbed onto an arbitrary scale.
            moves = []
        else:
            moves = [0.5 * float(centre), 2.0 * float(centre)]
        for value in moves:
            specs.extend((name, value, seed) for seed in seeds)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, specs))

    grouped = {}
    for name, value, _seed, spread in results:
        grouped.setdefault((name, value), []).append(spread)

    baseline = _median(grouped[(None, None)])
    baseline_interval = _median_interval(grouped[(None, None)])
    rows = []
    for (name, value), spreads in grouped.items():
        if name is None:
            continue
        median = _median(spreads)
        interval = _median_interval(spreads)
        # A shift counts as resolved where the two intervals do not meet. This
        # is the weaker of the two readings of the panel and it is the one
        # reported, since the cells share seeds and a paired test would claim
        # more than the design supports.
        resolved = bool(
            math.isfinite(interval[0]) and math.isfinite(baseline_interval[0])
            and (interval[1] < baseline_interval[0]
                 or interval[0] > baseline_interval[1])
        )
        rows.append({
            'parameter': name,
            'calibrated_value': base[name],
            'perturbed_value': value,
            'direction': ('halved' if (name not in COUNTS
                                       and value < float(base[name]))
                          else 'one fewer' if value < base[name]
                          else 'doubled' if name not in COUNTS else 'one more'),
            'quoted_spread_mean_bps': median,
            'quoted_spread_mean_bps_ci': interval,
            'shift_bps': median - baseline,
            'shift_resolved': resolved,
            'inside_band': bool(low <= median <= high),
        })
    rows.sort(key=lambda row: -abs(row['shift_bps'] if
                                   math.isfinite(row['shift_bps']) else 0.0))

    outside = [row for row in rows if not row['inside_band']]
    resolved = [row for row in rows if row['shift_resolved']]
    untestable = [name for name in LADDER
                  if base[name] is not None and float(base[name] or 0.0) == 0.0
                  and name not in COUNTS]
    return {
        'observable': OBSERVABLE,
        'scenario': SCENARIO,
        'seeds': list(seeds),
        'target_range': {'low': low, 'high': high},
        'band_width_over_lower_bound': (high - low) / low,
        'baseline_quoted_spread_mean_bps': baseline,
        'baseline_quoted_spread_mean_bps_ci': baseline_interval,
        'perturbations': rows,
        'perturbations_outside_band': len(outside),
        'perturbations_with_resolved_shift': len(resolved),
        'perturbations_total': len(rows),
        'parameters_that_leave_the_band': sorted(
            {row['parameter'] for row in outside}
        ),
        'parameters_untestable_at_zero': untestable,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=12)
    parser.add_argument('--seed-start', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', default=os.path.join(
        ROOT, 'output', 'eurchf', 'spread_identification.json'))
    args = parser.parse_args()

    from tools.robustness.signatures import model_signature, measurement_signature
    report = run(range(args.seed_start, args.seed_start + args.seeds),
                 args.workers)
    digest = model_signature()
    report['provenance'] = {
        'scenario': SCENARIO,
        'observable': OBSERVABLE,
        'seed_start': args.seed_start,
        'seed_count': args.seeds,
        'simulation_model_signature': digest,
        'measurement_signature': measurement_signature(
            'spread_identification', digest, functions=(run, _job, _median),
            constants=(SCENARIO, OBSERVABLE, tuple(sorted(LADDER)), COUNTS),
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k != 'perturbations'}, indent=2))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
