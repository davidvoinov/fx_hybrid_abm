"""What retains a provider: the return elsewhere, or a transfer from a sponsor.

H4 says that sustainability turns on the outside option and not on the
severity of the episode. It cannot be tested on the crisis window. An outside
option is a rate per year and the window is a hundred and fifty seconds, so on
capital of about seven hundred thousand a hundred basis points per annum is
five hundredths of a quote unit against an operating loss of ten thousand;
the comparison is four to five orders of magnitude out and no calibration
changes that. What is comparable is the annual frame the welfare accounting
already defines, where the quantity a participation decision turns on is the
share of windows that may be in the crisis state before the provider stops
covering itself,

    s*(r, tau) = (C + tau_w - r_w) / (X + C),

with C the calm operating gain per window, X the crisis operating loss per
window, r_w the opportunity cost of the committed capital over one window and
tau_w a sponsor transfer over the same. All four are shares of pool value, so
the capital cancels out of everything but its own opportunity cost.

Two corrections are carried here against the break-even share already in the
record. That one is taken before the opportunity cost of the capital, which is
the whole of the channel H4 names, so it cannot move with the outside option
by construction. It also divides a calm window of one hundred seconds by the
sum of itself and a crisis window of one hundred and fifty, which compares
unequal windows; the rates below are per second and the windows enter through
their own lengths.

The severity of the episode enters through X alone, so the second half of the
claim is a second derivative in the same table: X is scaled and s* read again.
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
sys.path.insert(0, ROOT)

SECONDS_PER_YEAR = 252.0 * 24.0 * 60.0 * 60.0

OUTSIDE_OPTION_BPS = (0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 500.0)
SPONSOR_TRANSFER_BPS = (0.0, 10.0, 50.0)
# What the episode's loss is multiplied by, to read the same share against a
# milder and a more severe crisis than the one identified.
SEVERITY_SCALES = (0.5, 1.0, 2.0)
LP_MODEL = 'endogenous'


def _job(seed):
    from tools.robustness import lp_pnl_corrected as T
    record = T.measure(seed, None, LP_MODEL)
    return {
        'seed': int(seed),
        'calm_pct': record.get('total_calm'),
        'crisis_pct': record.get('total_crisis'),
    }


def _median(values):
    finite = [float(v) for v in values
              if v is not None and math.isfinite(float(v))]
    return statistics.median(finite) if finite else float('nan')


def _share(calm_rate, crisis_rate, outside_rate, transfer_rate,
           calm_seconds, crisis_seconds, severity):
    """Crisis-state share of windows at which the provider stops covering itself.

    Rates are per second and per unit of pool value. A window of the crisis
    contributes its loss and its opportunity cost; a window of calm
    contributes its gain, less the same opportunity cost, since the capital is
    committed in both. The transfer is paid in both as well.
    """
    gain = (calm_rate + transfer_rate - outside_rate) * calm_seconds
    loss = (severity * crisis_rate - transfer_rate + outside_rate) * crisis_seconds
    if not (math.isfinite(gain) and math.isfinite(loss)):
        return float('nan')
    if gain <= 0.0:
        # Calm no longer covers even itself, so no share of crisis windows is
        # small enough. This is a result and not a missing number.
        return 0.0
    if loss <= 0.0:
        return 1.0
    return gain / (gain + loss)


def run(seeds, workers):
    from tools.robustness import lp_pnl_corrected as T
    calm_seconds = float(T.CALM[1] - T.CALM[0])
    crisis_seconds = float(T.CRISIS[1] - T.CRISIS[0])

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_job, seeds))

    # Per second and as a share of pool value. The stored figures are per cent
    # of pool value over the whole of each window.
    calm_rate = _median([r['calm_pct'] for r in rows]) / 100.0 / calm_seconds
    crisis_rate = -_median([r['crisis_pct'] for r in rows]) / 100.0 / crisis_seconds

    cells = []
    for outside_bps in OUTSIDE_OPTION_BPS:
        outside_rate = outside_bps / 10_000.0 / SECONDS_PER_YEAR
        for transfer_bps in SPONSOR_TRANSFER_BPS:
            transfer_rate = transfer_bps / 10_000.0 / SECONDS_PER_YEAR
            for severity in SEVERITY_SCALES:
                cells.append({
                    'outside_option_bps_pa': outside_bps,
                    'sponsor_transfer_bps_pa': transfer_bps,
                    'episode_severity_scale': severity,
                    'break_even_crisis_share': _share(
                        calm_rate, crisis_rate, outside_rate, transfer_rate,
                        calm_seconds, crisis_seconds, severity),
                })

    def at(outside_bps, transfer_bps, severity):
        for cell in cells:
            if (cell['outside_option_bps_pa'] == outside_bps
                    and cell['sponsor_transfer_bps_pa'] == transfer_bps
                    and cell['episode_severity_scale'] == severity):
                return cell['break_even_crisis_share']
        return float('nan')

    # Where the share reaches zero is a closed form and not a bracket. The
    # calm gain is what the capital earns for standing there, so an outside
    # option above it ends the participation outright, whatever the episode
    # does. A transfer moves the ceiling one for one, which is the sense in
    # which a sponsor buys participation and not resilience.
    retention_ceiling = calm_rate * 10_000.0 * SECONDS_PER_YEAR
    ceilings = {f'{t:g}': retention_ceiling + t for t in SPONSOR_TRANSFER_BPS}

    base = at(0.0, 0.0, 1.0)
    # The two derivatives H4 compares. The first is the whole of its claim,
    # the second is what it denies. Both are reported as the proportional
    # change in the share against the largest move made on each margin.
    outside_span = at(max(OUTSIDE_OPTION_BPS), 0.0, 1.0)
    severity_span = at(0.0, 0.0, max(SEVERITY_SCALES))
    responses = {
        'break_even_share_at_baseline': base,
        'break_even_share_at_largest_outside_option': outside_span,
        'break_even_share_at_double_severity': severity_span,
        'proportional_change_from_outside_option': (
            (outside_span - base) / base if base else float('nan')),
        'proportional_change_from_severity': (
            (severity_span - base) / base if base else float('nan')),
    }

    return {
        'lp_model': LP_MODEL,
        'seeds': list(seeds),
        'calm_window_seconds': calm_seconds,
        'crisis_window_seconds': crisis_seconds,
        'calm_gain_rate_per_second': calm_rate,
        'crisis_loss_rate_per_second': crisis_rate,
        'calm_gain_pct_of_value_per_window': calm_rate * 100.0 * calm_seconds,
        'crisis_loss_pct_of_value_per_window': crisis_rate * 100.0 * crisis_seconds,
        'outside_option_bps_pa': list(OUTSIDE_OPTION_BPS),
        'sponsor_transfer_bps_pa': list(SPONSOR_TRANSFER_BPS),
        'episode_severity_scales': list(SEVERITY_SCALES),
        'cells': cells,
        'retention_ceiling_bps_pa': retention_ceiling,
        'retention_ceiling_bps_pa_by_transfer': ceilings,
        'h4_responses': responses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', default=os.path.join(
        ROOT, 'output', 'eurchf', 'participation_grid.json'))
    args = parser.parse_args()

    from tools.robustness.signatures import model_signature, measurement_signature
    report = run(range(args.seed_start, args.seed_start + args.seeds),
                 args.workers)
    digest = model_signature()
    report['provenance'] = {
        'seed_start': args.seed_start,
        'seed_count': args.seeds,
        'simulation_model_signature': digest,
        'measurement_signature': measurement_signature(
            'participation_grid', digest, functions=(run, _job, _share),
            constants=(LP_MODEL, OUTSIDE_OPTION_BPS, SPONSOR_TRANSFER_BPS,
                       SEVERITY_SCALES, SECONDS_PER_YEAR),
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'retention_ceiling_bps_pa':
                      report['retention_ceiling_bps_pa'],
                      **report['h4_responses']}, indent=2))
    print(f"wrote {args.output}")


if __name__ == '__main__':
    main()
