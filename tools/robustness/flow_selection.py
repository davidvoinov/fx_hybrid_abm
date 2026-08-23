"""Is the flow that reaches the facility adversely selected.

H2 claims that order flow moves toward the facility as dealers retreat and
that what arrives under stress is adversely selected, so the facility takes on
loss making exposure in exactly the states where its availability is worth
most. The first limb follows from the routing rule and is not a finding. The
second is a measurement and had not been made.

The measurement is a markout. When a customer buys from the facility the
facility has sold, so its payoff over the next ``HORIZON`` periods is the
execution price less the reference price at the end of that horizon, and the
sign is reversed when the customer sells. A negative markout means the price
moved against the facility after it traded, which is what adverse selection
is. Reported per unit of base traded so the calm and the crisis windows are
comparable, and beside the book's own markout on the same seeds so that the
facility is read against the alternative and not against zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool as ProcPool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from tools.robustness.signatures import measurement_signature, model_signature

PRESET = 'dash_for_cash_2020'
N_ITER = 1000
WINDOW = 150
HORIZON = 30            # periods over which the markout is read


def _markouts(sim, lo, hi, prices):
    """Markout per unit of base, split by the venue that filled the order."""
    out = {'hfmm': [], 'clob': []}
    for trade in sim.logger.trade_log:
        if str(trade.get('execution_source', 'routed_customer')) != 'routed_customer':
            continue
        t = int(trade.get('t', -1))
        if not (lo <= t < hi) or t + HORIZON >= prices.size:
            continue
        quantity = max(0.0, float(trade.get('quantity', 0.0)))
        executed = float(trade.get('all_in_exec_price', float('nan')))
        later = float(prices[t + HORIZON])
        # The execution price is the denominator below, so it has to be
        # finite and positive and not merely not a nan, which is all the
        # earlier guard checked.
        if (quantity <= 0 or not np.isfinite(executed) or executed <= 0
                or not np.isfinite(later)):
            continue
        # The customer's side is the opposite of the facility's.
        side = 1.0 if str(trade.get('side', 'buy')) == 'buy' else -1.0
        payoff = side * (executed - later)
        venue = 'hfmm' if str(trade.get('venue', '')) == 'hfmm' else 'clob'
        out[venue].append(payoff / executed * 1e4)
    return out


def _run(task):
    preset, seed = task
    argv = ['--seed', str(seed), '--n-iter', str(N_ITER), '--silent',
            '--amm-lp-model', 'endogenous']
    if preset:
        argv = ['--preset', preset] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if preset:
        _apply_preset_defaults(parser, args)
    args.venue_choice_rule = _resolve_main_routing(args, argv)
    if preset:
        _auto_stress_around_shock(args)
    _seed_all(seed)
    sim = build_sim(args)
    sim.simulate(N_ITER, silent=True)

    shock = int(args.shock_iter) if preset else N_ITER // 2
    prices = np.asarray(sim.logger.fair_price_series, dtype=float)
    marks = _markouts(sim, shock, shock + WINDOW, prices)
    return {
        'state': preset or 'calm', 'seed': seed,
        'facility': float(np.mean(marks['hfmm'])) if marks['hfmm'] else float('nan'),
        'book': float(np.mean(marks['clob'])) if marks['clob'] else float('nan'),
        'facility_fills': len(marks['hfmm']), 'book_fills': len(marks['clob']),
    }


def _interval(values, seed=0, draws=10_000):
    values = np.asarray([v for v in values if v == v], dtype=float)
    if values.size < 3:
        return (float('nan'),) * 3
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(values, values.size, replace=True))
                     for _ in range(draws)])
    return (float(values.mean()), float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--seed-count', type=int, default=300)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--output', default='output/resilience/flow_selection.json')
    args = parser.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    tasks = [(preset, seed) for preset in (None, PRESET) for seed in seeds]
    with ProcPool(processes=args.workers) as pool:
        rows = pool.map(_run, tasks, chunksize=4)

    print(f'Markout at {HORIZON} periods, {len(seeds)} seeds, per unit of base '
          f'in basis points')
    print('negative means the price moved against the venue after it traded\n')
    head = (f"{'state':>20}{'facility':>26}{'book':>26}{'fills':>16}")
    print(head); print('-' * len(head))
    report = {}
    for state in ('calm', PRESET):
        subset = [r for r in rows if r['state'] == state]
        facility = _interval([r['facility'] for r in subset], seed=1)
        book = _interval([r['book'] for r in subset], seed=2)
        fills = int(np.median([r['facility_fills'] for r in subset]))
        report[state] = {
            'facility': {'mean': facility[0], 'ci': [facility[1], facility[2]]},
            'book': {'mean': book[0], 'ci': [book[1], book[2]]},
            'median_facility_fills': fills,
        }
        print(f'{state[:20]:>20}'
              f'{f"{facility[0]:+.3f} [{facility[1]:+.3f},{facility[2]:+.3f}]":>26}'
              f'{f"{book[0]:+.3f} [{book[1]:+.3f},{book[2]:+.3f}]":>26}'
              f'{fills:>16}')

    common = sorted({r['seed'] for r in rows if r['state'] == 'calm'}
                    & {r['seed'] for r in rows if r['state'] == PRESET})
    calm = {r['seed']: r for r in rows if r['state'] == 'calm'}
    crisis = {r['seed']: r for r in rows if r['state'] == PRESET}
    shift = _interval([crisis[s]['facility'] - calm[s]['facility'] for s in common],
                      seed=3)
    report['crisis_less_calm'] = {'mean': shift[0], 'ci': [shift[1], shift[2]],
                                  'n': len(common)}
    print(f'\nfacility markout, crisis less calm, paired within seed: '
          f'{shift[0]:+.3f} [{shift[1]:+.3f}, {shift[2]:+.3f}]')

    digest = model_signature()
    report['provenance'] = {
        'preset': PRESET, 'n_iter': N_ITER, 'window': WINDOW, 'horizon': HORIZON,
        'seed_start': args.seed_start, 'seed_count': args.seed_count,
        'simulation_model_signature': digest,
        'measurement_signature': measurement_signature(
            'flow_selection', digest, functions=(_run, _markouts, _interval),
            constants=(PRESET, N_ITER, WINDOW, HORIZON),
        ),
    }
    destination = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'written to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
