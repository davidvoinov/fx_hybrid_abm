"""Resource matched comparison of the facility arms.

Adding a facility to a dealer market moves three things at once, namely the
arrival of committed capital, the obligation to keep quoting and the rule by
which the venue prices. A comparison against a dealer only control pools all
three. These arms hold the first two fixed and vary the third, so the two
identifying contrasts are

    reserve less dealer of last resort   the pricing schedule
    dealer of last resort less control   standing availability

Matching the resources is the whole content of the design, so it is stated
here and not left to a default.

Capital. The arms are matched on the total economic capital the AMM side can
bring to the episode, which is the reserves inside the pool plus the wallets
its providers hold outside it, since a provider may commit those during the
window. Read at the opening of the crisis window over a hundred and twenty
development seeds this is 951,999 quote units, of which 670,671 sits inside
the pool and 281,328 in wallets. The quantity is nearly deterministic across
seeds, with a quartile range of seventeen units, so it is declared once here
instead of matched seed by seed.

The budget is given to the order book arms alone. Passing it to the reserve
arm as well would resize the pool to the figure that already includes the
wallets and then grant fresh wallets on top of it, so that arm runs at the
size the calibration gives it, which is what the figure above was read from.

Reallocation is not in this comparison. Funding the facility out of the
dealer sector caps it at what the sector holds, and each of the five dealers
carries 70,000 against a facility of 951,999, so the arm cannot be run at a
matched size at all. It is reported separately below at a size the sector can
fund, against a reserve arm of the same size, which holds the facility fixed
and varies only where its capital came from.

Price. The obliged quoter shows one spread at every size while the pool's
cost is convex in size, so a match at a single size is cheap at every other
one. The declared spread is the pool's round trip cost weighted by the
realised distribution of customer trade sizes over the crisis window, 3.469
basis points on the same development seeds. Matching at the median size alone
would have set it at 5.4, and the previous default of 12 was matched to a fee
of five basis points that the calibration no longer carries.

Neither number is fitted to any outcome below. Both are read off the reserve
arm before the comparison is run.

Outcome. The dislocation is the best executable round trip cost across both
venues at the median customer size, which is the outcome this paper defines.
Read instead as the order book's own quoted spread, an arm that posts into
that book pins the series with its own quote while an arm that is a separate
venue never enters it at all: measured that way the reserve priced pool
improved the peak by 2.7 basis points and the order book arms by 11 and 18,
and measured across both venues the same runs give 28.8 for the pool against
8.6 and 19.7. The order reverses, because the pool's price answers to its own
reserves and holds while the book blows out, which is the property the whole
design is about. The book only series is kept beside it as a diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool as ProcPool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from main import (load_primary_model_defaults, build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim, CRISIS_PRESET)
from AgentBasedModel.metrics.resilience import decay_rate_from_peak
from tools.robustness.signatures import measurement_signature, model_signature

PRESET = CRISIS_PRESET
N_ITER = 1000
WINDOW = 150
# The size the outcome is quoted at, which is the median customer trade, and
# the sizes the profile is read at beside it. A flat spread cannot match a
# convex schedule at every size: matched to the pool's cost weighted across the
# realised flow, the obliged quoter is dearer than the pool at the sizes most
# trades take and cheaper at the tail. Reporting the profile states where each
# arm wins instead of choosing one point and calling it the answer.
OUTCOME_SIZE = 2.0
OUTCOME_SIZES = (1.0, 2.0, 5.0, 20.0)
# H1 is a claim about amplitude and about speed, and the two have to be
# separated to be read. Speed is the periods each arm takes to bring a five
# period rolling median of the executable cost down to one common level, the
# halfway point between the control's own peak and its pre shock baseline on
# that seed. Two other definitions were tried and both entangle the two
# claims. Against a fixed absolute band the measure saturates, since the
# reserve priced arm is inside any such band from the first period and the
# order book arms never re-enter one. Against each arm's own amplitude it
# inverts, because an arm that flattens the peak has less of it to halve and
# its target sits nearer the baseline, so the arm that helped most reads as
# the slowest. A level taken from the control is the same bar for every arm.
RECOVERY_FRACTION = 0.5
RECOVERY_WINDOW = 5
# The capital is what the reserve priced pool actually holds once its
# providers have committed, read as opening capital plus wallets on seed 42,
# and the obliged quoter is endowed with the same so the arms are matched on
# resources. The spread is the pool's own round trip at the outcome size over
# the pre-shock periods, which is the price the quoter is matched to; it is
# read from the manifest, where its provenance is recorded, since a pair whose
# quoting increment and volatility differ prices its pool differently.
ARM_CAPITAL = 951_999.0
ARM_SPREAD_BPS = float(load_primary_model_defaults().get('arm_spread_bps', 3.469))
ARMS = ('none', 'reserve', 'reserve_frozen', 'dealer_of_last_resort',
        'passive_book')
# Sized to what the dealer sector can give up while still quoting, which is
# half the cash of each of the five dealers.
REALLOCATION_CAPITAL = 175_000.0
# The three add up: reserve less control is the sum of the three rows below,
# which is what makes the decomposition a decomposition and not a set of
# comparisons that happen to share a control.
CONTRASTS = (
    ('capital_flight', 'reserve', 'reserve_frozen'),
    ('pricing_schedule', 'reserve_frozen', 'dealer_of_last_resort'),
    ('committed_quoting', 'dealer_of_last_resort', 'none'),
)


def _run(task):
    arm, seed, capital = task
    argv = ['--preset', PRESET, '--facility-arm', arm,
            '--arm-spread-bps', str(ARM_SPREAD_BPS),
            '--seed', str(seed), '--n-iter', str(N_ITER), '--silent']
    if capital > 0:
        argv += ['--arm-capital', str(capital)]
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_preset_defaults(parser, args)
    args.venue_choice_rule = _resolve_main_routing(args, argv)
    _auto_stress_around_shock(args)
    _seed_all(seed)
    sim = build_sim(args)
    shock = int(args.shock_iter)

    # Stepped through the window so both venues can be quoted at the same
    # instant. Neither is a series the logger keeps, and the outcome is the
    # better of the two.
    def _executable(size=OUTCOME_SIZE):
        clob = sim.clob
        pool = list(sim.amm_pools.values())[0] if sim.amm_pools else None
        try:
            buy = float(clob.cost_bps(size, 'buy'))
            sell = float(clob.cost_bps(size, 'sell'))
        except Exception:
            return float('nan')
        if pool is not None:
            series = getattr(sim.logger, 'fair_price_series', []) or []
            reference = float(series[-1]) if series else float('nan')
            try:
                pool_buy = float(pool.quote_buy(size, reference)['cost_bps'])
                pool_sell = float(pool.quote_sell(size, reference)['cost_bps'])
                if math.isfinite(pool_buy) and math.isfinite(pool_sell):
                    return min(buy, pool_buy) + min(sell, pool_sell)
            except Exception:
                pass
        return buy + sell

    sim.simulate(200, silent=True)
    pre_window = []
    for _ in range(shock - 200):
        sim.simulate(1, silent=True)
        pre_window.append(_executable())
    window_series = []
    profile = {size: [] for size in OUTCOME_SIZES}
    for _ in range(WINDOW):
        sim.simulate(1, silent=True)
        window_series.append(_executable())
        for size in OUTCOME_SIZES:
            profile[size].append(_executable(size))
    remaining = N_ITER - shock - WINDOW
    if remaining > 0:
        sim.simulate(remaining, silent=True)

    lo, hi = shock, shock + WINDOW
    # A book too thin to fill the median trade quotes no price at all, which
    # the venue reports as an infinite cost. That is the worst resilience
    # outcome and not a large number, so availability is carried as its own
    # outcome and the cost outcomes are read where a price exists. A pool
    # quotes at every size by construction, so this separates the two things
    # a facility can fail to do.
    finite = lambda seq: np.asarray(
        [v for v in seq if v == v and math.isfinite(v)], dtype=float)
    pre = finite(pre_window)
    win = finite(window_series)
    quotable = [v for v in window_series if v == v]
    unavailable = (sum(1 for v in quotable if not math.isfinite(v))
                   / len(quotable)) if quotable else float('nan')
    base = float(np.median(pre)) if pre.size else float('nan')

    # Speed, beside amplitude. A facility that lowers the peak and leaves the
    # return to normal untouched is a different claim from one that does both.
    rolled = (np.convolve(win, np.ones(RECOVERY_WINDOW) / RECOVERY_WINDOW,
                          mode='valid') if win.size >= RECOVERY_WINDOW
              else np.asarray([], dtype=float))

    # Resilience in the standard sense has three parts: how far the market is
    # displaced, how fast the displacement decays, and how long it takes to
    # come back. Amplitude is the peak below and persistence the excess; this
    # is the rate, fitted to the deviation from its own peak, which does not
    # saturate the way a crossing time does when one arm never leaves the band.
    decay = half_life = fit_quality = float('nan')
    if math.isfinite(base) and base > 0 and win.size:
        deviation = [(v - base) / base * 100.0 for v in win]
        fitted = decay_rate_from_peak(deviation, 0, horizon=int(win.size))
        if isinstance(fitted, dict):
            decay = float(fitted.get('decay_rate_per_step') or float('nan'))
            half_life = float(fitted.get('decay_half_life_steps') or float('nan'))
            fit_quality = float(fitted.get('decay_r_squared') or float('nan'))

    book = np.asarray(sim.logger.clob_qspr, dtype=float)[lo:hi]
    book = book[np.isfinite(book)]

    env = next((t.env for t in sim.traders
                if getattr(t, 'env', None) is not None), None)
    capacity = np.asarray(getattr(env, 'dealer_capacity_history', []) or [],
                          dtype=float)[lo:hi]
    capacity = capacity[np.isfinite(capacity)]

    prices = np.asarray(sim.logger.fair_price_series, dtype=float)
    volume = cost = 0.0
    for trade in sim.logger.trade_log:
        if str(trade.get('execution_source', 'routed_customer')) != 'routed_customer':
            continue
        t = int(trade.get('t', -1))
        if not (lo <= t < hi) or t >= prices.size:
            continue
        quantity = max(0.0, float(trade.get('quantity', 0.0)))
        reference = float(trade.get('common_reference_price', prices[t]))
        executed = float(trade.get('all_in_exec_price', float('nan')))
        if quantity <= 0 or reference <= 0 or executed != executed:
            continue
        side = str(trade.get('side', 'buy'))
        bps = ((executed - reference) if side == 'buy'
               else (reference - executed)) / reference * 1e4
        volume += quantity
        cost += quantity * bps
    return {
        'arm': arm,
        'capital': capital,
        'seed': seed,
        'peak': float(np.max(win)) if win.size else float('nan'),
        'excess': float(np.sum(np.maximum(0.0, win - base))) if win.size else float('nan'),
        'book_only_peak': float(np.max(book)) if book.size else float('nan'),
        'unavailable_share': unavailable,
        'decay_rate': decay,
        'decay_half_life': half_life,
        'decay_fit_quality': fit_quality,
        # Realised cost is what customers actually paid, so it carries the
        # mix of sizes, venues and moments each arm produced: a difference
        # between arms is a difference in price and in composition together.
        # The quoted series is the same round trip at one declared size on
        # every period, so it prices the arms on one basket.
        'quoted_cost': float(np.mean(win)) if win.size else float('nan'),
        'base': base,
        'rolled': rolled.tolist(),
        'peak_by_size': {
            str(size): (float(np.max(finite(values))) if finite(values).size
                        else float('nan'))
            for size, values in profile.items()
        },
        'cost': cost / volume if volume > 0 else float('nan'),
        'withdrawal_peak': float(np.max(capacity)) if capacity.size else float('nan'),
        'full_evacuation': bool(capacity.size and np.max(capacity) >= 0.999),
    }


def _interval(values, seed=0, draws=10_000):
    values = np.asarray([v for v in values if v == v], dtype=float)
    if values.size < 3:
        return float('nan'), float('nan'), float('nan')
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
    parser.add_argument('--output', default='output/facility_arms.json')
    args = parser.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    # The order book arms carry the matched budget; the reserve arm runs at
    # the size the calibration gives it, and the control carries none.
    tasks = [(arm, seed, ARM_CAPITAL if arm in ('dealer_of_last_resort',
                                                'passive_book') else 0.0)
             for arm in ARMS for seed in seeds]
    tasks += [(arm, seed, REALLOCATION_CAPITAL)
              for arm in ('reserve', 'reallocation') for seed in seeds]
    with ProcPool(processes=args.workers) as pool:
        rows = pool.map(_run, tasks, chunksize=4)

    def _time_to(row, target):
        """Periods to bring the rolling median down to a given level."""
        series = np.asarray(row.get('rolled') or [], dtype=float)
        if not series.size or not math.isfinite(target):
            return float('nan')
        crest = int(np.argmax(series))
        hit = np.nonzero(series[crest:] <= target)[0]
        return float(hit[0]) if hit.size else float(series.size - crest)

    by_arm = {arm: {r['seed']: r for r in rows
                    if r['arm'] == arm and r['capital'] != REALLOCATION_CAPITAL}
              for arm in ARMS}
    funding = {arm: {r['seed']: r for r in rows
                     if r['arm'] == arm and r['capital'] == REALLOCATION_CAPITAL}
               for arm in ('reserve', 'reallocation')}
    control = by_arm['none']

    print(f'Facility arms, {len(seeds)} paired seeds on {PRESET}')
    print(f'capital {ARM_CAPITAL:,.0f} quote units, obliged quoter at '
          f'{ARM_SPREAD_BPS} bps\n')
    head = (f"{'arm':>24}{'evacuated':>11}{'no price at size':>18}"
            f"{'d peak [95% CI]':>26}{'d cost realised':>26}"
            f"{'d cost quoted':>26}")
    print(head)
    print('-' * len(head))
    report = {'arms': {}, 'contrasts': {}}
    for arm in ARMS:
        rowset = by_arm[arm]
        common = sorted(set(rowset) & set(control))
        evacuated = np.mean([rowset[s]['full_evacuation'] for s in common])
        withdrawal = float(np.nanmedian([rowset[s]['unavailable_share'] for s in common]))
        peak = _interval([rowset[s]['peak'] - control[s]['peak'] for s in common])
        cost = _interval([rowset[s]['cost'] - control[s]['cost'] for s in common], seed=1)
        quoted = _interval([rowset[s]['quoted_cost'] - control[s]['quoted_cost']
                            for s in common], seed=8)
        # Resilience decomposes into amplitude, decay and persistence, so each
        # arm carries all three against the control and not the peak alone.
        excess = _interval([rowset[s]['excess'] - control[s]['excess']
                            for s in common], seed=31)
        # A decay rate exists only where the displacement decays. Under a
        # permanent repricing the order book arms often do not return toward
        # their own pre-shock level inside the window, the exponential fit has
        # nothing to fit, and the seed drops out. Reporting the interval alone
        # then hides how few seeds are behind it, so the share that could be
        # fitted is carried beside it.
        _fitted = [rowset[s]['decay_rate'] - control[s]['decay_rate']
                   for s in common
                   if rowset[s]['decay_rate'] == rowset[s]['decay_rate']
                   and control[s]['decay_rate'] == control[s]['decay_rate']]
        decay = _interval([rowset[s]['decay_rate'] - control[s]['decay_rate']
                           for s in common], seed=32)
        report['arms'][arm] = {
            'full_evacuation_rate': float(evacuated),
            'unavailable_share_median': withdrawal,
            'delta_peak': {'mean': peak[0], 'ci': [peak[1], peak[2]]},
            'delta_excess': {'mean': excess[0], 'ci': [excess[1], excess[2]]},
            'delta_decay_rate': {'mean': decay[0], 'ci': [decay[1], decay[2]],
                                 'n_fitted': len(_fitted),
                                 'fitted_share': (len(_fitted) / len(common)
                                                  if common else float('nan'))},
            'delta_cost': {'mean': cost[0], 'ci': [cost[1], cost[2]]},
            'delta_quoted_cost': {'mean': quoted[0], 'ci': [quoted[1], quoted[2]]},
            'n': len(common),
        }
        print(f'{arm:>24}{evacuated:>10.1%}{withdrawal:>17.1%}'
              f'{f"{peak[0]:+.2f} [{peak[1]:+.2f},{peak[2]:+.2f}]":>26}'
              f'{f"{cost[0]:+.2f} [{cost[1]:+.2f},{cost[2]:+.2f}]":>26}'
              f'{f"{quoted[0]:+.2f} [{quoted[1]:+.2f},{quoted[2]:+.2f}]":>26}')

    print('\npeak executable cost against the control, by trade size')
    head = f"{'arm':>24}" + ''.join(f'{f"size {s:g}":>12}' for s in OUTCOME_SIZES)
    print(head); print('-' * len(head))
    report['peak_by_size'] = {}
    for arm in ARMS:
        rowset, line = by_arm[arm], f'{arm:>24}'
        common = sorted(set(rowset) & set(control))
        report['peak_by_size'][arm] = {}
        for size in OUTCOME_SIZES:
            key = str(size)
            diff = [rowset[s]['peak_by_size'].get(key, float('nan'))
                    - control[s]['peak_by_size'].get(key, float('nan'))
                    for s in common]
            stat = _interval(diff, seed=int(size) + 20)
            report['peak_by_size'][arm][key] = {'mean': stat[0],
                                                'ci': [stat[1], stat[2]]}
            line += f'{stat[0]:>12.2f}'
        print(line)

    def _bar(seed):
        """The common level, taken from the control on this seed."""
        row = control.get(seed)
        series = np.asarray((row or {}).get('rolled') or [], dtype=float)
        base_value = (row or {}).get('base', float('nan'))
        if not series.size or not math.isfinite(base_value):
            return float('nan')
        return base_value + RECOVERY_FRACTION * (float(np.max(series)) - base_value)

    print('\nidentifying contrasts, paired within seed')
    for name, left, right in CONTRASTS:
        a, b = by_arm[left], by_arm[right]
        common = sorted(set(a) & set(b))
        peak = _interval([a[s]['peak'] - b[s]['peak'] for s in common], seed=2)
        excess = _interval([a[s]['excess'] - b[s]['excess'] for s in common], seed=3)
        # What the customer pays is reported beside the dislocation, since the
        # two separate the arms differently: an obliged quoter narrows the peak
        # without changing the price of a trade, and the schedule shows up here.
        cost = _interval([a[s]['cost'] - b[s]['cost'] for s in common], seed=6)
        quoted = _interval([a[s]['quoted_cost'] - b[s]['quoted_cost']
                            for s in common], seed=9)
        speed = _interval([_time_to(a[s], _bar(s)) - _time_to(b[s], _bar(s))
                           for s in common], seed=7)
        _c_fitted = [s for s in common
                     if a[s]['decay_rate'] == a[s]['decay_rate']
                     and b[s]['decay_rate'] == b[s]['decay_rate']]
        decay = _interval([a[s]['decay_rate'] - b[s]['decay_rate']
                           for s in common], seed=8)
        report['contrasts'][name] = {
            'left': left, 'right': right, 'n': len(common),
            'delta_peak': {'mean': peak[0], 'ci': [peak[1], peak[2]]},
            'delta_excess': {'mean': excess[0], 'ci': [excess[1], excess[2]]},
            'delta_cost': {'mean': cost[0], 'ci': [cost[1], cost[2]]},
            'delta_recovery': {'mean': speed[0], 'ci': [speed[1], speed[2]]},
            'delta_decay_rate': {'mean': decay[0], 'ci': [decay[1], decay[2]]},
            'delta_quoted_cost': {'mean': quoted[0], 'ci': [quoted[1], quoted[2]]},
        }
        print(f'  {name:24s} {left} less {right}')
        print(f'{"":26s} peak   {peak[0]:+.2f} [{peak[1]:+.2f},{peak[2]:+.2f}]')
        print(f'{"":26s} excess {excess[0]:+.1f} [{excess[1]:+.1f},{excess[2]:+.1f}]')
        print(f'{"":26s} cost   {cost[0]:+.2f} [{cost[1]:+.2f},{cost[2]:+.2f}]'
              '  realised')
        print(f'{"":26s} cost   {quoted[0]:+.2f} [{quoted[1]:+.2f},{quoted[2]:+.2f}]'
              '  quoted at one size')
        print(f'{"":26s} speed  {speed[0]:+.1f} [{speed[1]:+.1f},{speed[2]:+.1f}]'
              ' periods to recover')
        # A decay rate needs a decay. Where a repricing is permanent the
        # dislocation does not come back toward the pre-shock level inside the
        # window, the fit has nothing to work on, and the few seeds that do
        # fit are the unrepresentative ones. Below half the panel the interval
        # is withheld and the coverage is stated in its place, so the row
        # cannot be read as a measurement it is not.
        _dec_n = len(_c_fitted)
        _dec_share = _dec_n / len(common) if common else 0.0
        report['contrasts'][name]['delta_decay_rate']['n_fitted'] = _dec_n
        report['contrasts'][name]['delta_decay_rate']['fitted_share'] = _dec_share
        if _dec_share < 0.5:
            report['contrasts'][name]['delta_decay_rate']['measurable'] = False
            print(f'{"":26s} decay  not measurable: the deviation decayed on '
                  f'{_dec_n} of {len(common)} seeds')
        else:
            report['contrasts'][name]['delta_decay_rate']['measurable'] = True
            print(f'{"":26s} decay  {decay[0]:+.4f} '
                  f'[{decay[1]:+.4f},{decay[2]:+.4f}] per period')

    print(f'\nfunding source, facility held at {REALLOCATION_CAPITAL:,.0f} '
          f'quote units in both')
    left, right = funding['reallocation'], funding['reserve']
    common = sorted(set(left) & set(right))
    for key, label, seed_ in (('peak', 'peak', 4), ('cost', 'cost', 5)):
        stat = _interval([left[s][key] - right[s][key] for s in common], seed=seed_)
        report['contrasts'][f'funding_source_{key}'] = {
            'left': 'reallocation', 'right': 'reserve', 'n': len(common),
            'capital': REALLOCATION_CAPITAL,
            'mean': stat[0], 'ci': [stat[1], stat[2]],
        }
        print(f'  taken from the dealers less added to the market, {label:5s}'
              f' {stat[0]:+.2f} [{stat[1]:+.2f},{stat[2]:+.2f}]')

    # The decomposition has to add up, or it is not one.
    total = _interval([by_arm['reserve'][s]['peak'] - control[s]['peak']
                       for s in sorted(set(by_arm['reserve']) & set(control))],
                      seed=12)
    parts = sum(report['contrasts'][name]['delta_peak']['mean']
                for name, _, _ in CONTRASTS)
    # The sum telescopes only while every contrast is read on the same seeds.
    # Where an arm is missing a seed the intermediate terms stop cancelling and
    # the three channels no longer account for the whole effect, which is a
    # failure of the decomposition and not a rounding matter, so it is checked
    # here and not left to a reader of the printed line.
    residual = parts - total[0]
    tolerance = max(1e-6, 1e-3 * abs(total[0]))
    print(f'\n  the three channels sum to {parts:+.2f} on the peak against a '
          f'total effect of {total[0]:+.2f}, residual {residual:+.2e}')
    report['decomposition_check'] = {'sum_of_channels': parts,
                                     'total_effect': total[0],
                                     'residual': residual,
                                     'tolerance': tolerance,
                                     'holds': abs(residual) <= tolerance}
    if abs(residual) > tolerance:
        raise AssertionError(
            f'the decomposition does not close: the channels sum to {parts:+.6f} '
            f'against a total of {total[0]:+.6f}, residual {residual:+.3e} '
            f'against a tolerance of {tolerance:.3e}. The contrasts are read on '
            f'different seed sets, so the intermediate arms do not cancel.')

    digest = model_signature()
    report['provenance'] = {
        'preset': PRESET, 'n_iter': N_ITER, 'window': WINDOW,
        'arm_capital': ARM_CAPITAL, 'arm_spread_bps': ARM_SPREAD_BPS,
        'reallocation_capital': REALLOCATION_CAPITAL,
        'seed_start': args.seed_start, 'seed_count': args.seed_count,
        'simulation_model_signature': digest,
        'measurement_signature': measurement_signature(
            'facility_arms', digest, functions=(_run, _interval),
            constants=(PRESET, N_ITER, WINDOW, OUTCOME_SIZE, ARM_CAPITAL,
                       ARM_SPREAD_BPS, REALLOCATION_CAPITAL,
                       RECOVERY_FRACTION, RECOVERY_WINDOW, ARMS),
        ),
    }
    # The per seed rows are kept so a contrast can be recomputed without
    # paying for the panel again.
    # The series were carried so the common level could be built; they are not
    # a result and would multiply the size of the report by two orders.
    for row in rows:
        row.pop('rolled', None)
    report['rows'] = rows
    destination = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nwritten to {args.output}')

    try:
        from AgentBasedModel.visualization.arms_plots import dashboard_arms
        drawn = dashboard_arms(report, out_dir=os.path.dirname(destination) or '.')
        if drawn:
            print(f'drawn to {os.path.relpath(drawn, ROOT)}')
    except Exception as exc:            # plotting must not lose a finished panel
        print(f'the figure could not be drawn: {exc}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
