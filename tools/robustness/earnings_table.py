"""What the capital behind each arm earns in calm and loses in crisis.

One measurer, one run per seed and arm, both windows read off that run. The
table this produces was previously assembled from two sources: the crisis
column came from the welfare account and the percentage beside it divided a
median result by a median capital, while the figure quoted in the surrounding
prose came from the provider profit and loss tool, which takes the median of
the per-seed ratio instead. The two answers differ in the fourth decimal of a
percentage and there was nothing in the paper to say which one a reader was
looking at. Here every column of the table comes from the same rows.

A ratio of medians is not the median of a ratio, and the caption asks for the
second: what the capital committed at the opening of a window earns over that
window is a per-window quantity, so it is formed per seed and then summarised.

The capital base is the one the welfare account defines, the reserves in the
pool plus the wallets its providers hold beside it, because both are tied up
by the arrangement and neither is free to earn elsewhere. An order book arm
has no wallets and its base is the quoter's equity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.robustness.lp_pnl_corrected import CALM
from tools.robustness.signatures import measurement_signature, model_signature
from tools.robustness.welfare_accounting import (CRISIS, N_ITER, PRESET,
                                                 arm_window, run)

ARMS = ('reserve', 'reserve_frozen', 'dealer_of_last_resort', 'passive_book')
LABELS = {
    'reserve': 'Reserve priced pool',
    'reserve_frozen': 'Pool, capital held',
    'dealer_of_last_resort': 'Obliged quoter',
    'passive_book': 'Passive ladder',
}
_SIGNATURE = None


def measurement() -> str:
    global _SIGNATURE
    if _SIGNATURE is None:
        _SIGNATURE = measurement_signature(
            'earnings_table',
            model_signature(ROOT),
            functions=(run, arm_window, measure, summarise),
            constants=(PRESET, N_ITER, CALM, CRISIS, ARMS,
                       'earnings-one-measurer-v1'),
        )
    return _SIGNATURE


def _window_row(sim, shock, window) -> dict:
    w = arm_window(sim, shock, window)
    capital = w.facility_capital
    result = w.lp_operating_result
    ok = (math.isfinite(capital) and capital > 0.0 and math.isfinite(result))
    return {
        'result': float(result) if math.isfinite(result) else None,
        'capital': float(capital) if math.isfinite(capital) else None,
        'return': float(result / capital) if ok else None,
    }


def measure(seed: int, arm: str) -> dict:
    """One arm on one seed, both windows off one path."""
    sim, shock = run(seed, arm, windows=(CALM, CRISIS))
    calm = _window_row(sim, shock, CALM)
    crisis = _window_row(sim, shock, CRISIS)
    return {'arm': arm, 'seed': int(seed), 'calm': calm, 'crisis': crisis}


def _median(values):
    kept = [float(v) for v in values
            if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.median(kept) if kept else None


def summarise(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        mine = [r for r in rows if r['arm'] == arm]
        if not mine:
            continue
        earners = [r for r in mine
                   if r['calm']['return'] is not None
                   and r['calm']['return'] > 0.0]
        calm_return = _median([r['calm']['return'] for r in mine])
        crisis_return = _median([r['crisis']['return'] for r in mine])
        # How many calm windows one crisis window consumes, formed from the
        # two returns printed beside it so the row is consistent with itself.
        #
        # Taken instead as a median of the per-seed ratio it was a number
        # conditioned on the seeds that happened to earn, and on the primary
        # pair those were three of twenty four. The column then read as if the
        # arm covered itself in some number of calm windows while the median
        # seed earned nothing at all and no number of them covered anything.
        # It is absent exactly when the median seed earns nothing, which is
        # the statement the table means to make.
        windows = None
        if (calm_return is not None and crisis_return is not None
                and calm_return > 0.0 and crisis_return < 0.0):
            windows = -crisis_return / calm_return
        out[arm] = {
            'label': LABELS[arm],
            'seeds': len(mine),
            'calm_result': _median([r['calm']['result'] for r in mine]),
            'crisis_result': _median([r['crisis']['result'] for r in mine]),
            'calm_return': calm_return,
            'crisis_return': crisis_return,
            'calm_windows_per_crisis': windows,
            'seeds_earning_in_calm': len(earners),
        }
    return out


def _fmt(value, places, sign=True):
    if value is None:
        return '---'
    body = f'{abs(value):,.{places}f}'.replace(',', '{,}')
    mark = ('-' if value < 0 else '+') if sign else ('-' if value < 0 else '')
    return f'${mark}{body}$'


def latex(summary: dict) -> str:
    lines = []
    for arm in ARMS:
        row = summary.get(arm)
        if row is None:
            continue
        # No quantity of calm ever pays for one crisis when nothing is earned
        # while conditions hold, and the column says so in words.
        if row['calm_windows_per_crisis'] is None:
            per = 'never'
        else:
            per = f"${row['calm_windows_per_crisis']:,.0f}$".replace(',', '{,}')
        lines.append(
            f"{row['label']} & {_fmt(row['calm_result'], 3)} & "
            f"{_fmt(row['crisis_result'], 1, sign=False)} & "
            f"{_fmt(100.0 * row['calm_return'] if row['calm_return'] is not None else None, 5)} & "
            f"{_fmt(100.0 * row['crisis_return'] if row['crisis_return'] is not None else None, 4)} & "
            f"{per} \\\\")
    return '\n'.join(lines)


def _job(spec):
    return measure(*spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=300)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--output', default='output/resilience/earnings_table.json')
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f'unknown arm {unknown[0]}')
    specs = [(args.seed_start + i, arm)
             for arm in arms for i in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_job, specs, chunksize=2))
    else:
        rows = [_job(s) for s in specs]

    summary = summarise(rows)
    payload = {
        'model_signature': model_signature(ROOT),
        'measurement_signature': measurement(),
        'configuration': {
            'preset': PRESET, 'n_iter': N_ITER,
            'calm_window_relative_to_shock': list(CALM),
            'crisis_window_relative_to_shock': list(CRISIS),
            'seed_start': args.seed_start, 'seeds': args.seeds,
            'arms': arms,
            'capital_base': 'facility_capital_reserves_plus_provider_wallets',
            'statistic': 'median_of_per_seed_ratio',
        },
        'summary': summary,
        'rows': rows,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=1)

    body = latex(summary)
    with open(os.path.splitext(args.output)[0] + '.tex', 'w',
              encoding='utf-8') as handle:
        handle.write(body + '\n')
    print(body)
    for arm in arms:
        row = summary.get(arm)
        if row is None:
            continue
        print(f"  {arm:24s} сидов {row['seeds']:3d} "
              f"зарабатывают в спокойном {row['seeds_earning_in_calm']:3d}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
