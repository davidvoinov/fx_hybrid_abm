#!/usr/bin/env python3
"""Flow migration and dealer-state channels by phase, the source for Table 3.

    python3 tools/robustness/migration_table.py --seeds 40

The current table keeps endogenous withdrawal, a scenario-imposed pause and
the dealer's defensive state separate. Earlier output combined them and made a
scripted pause look like a dealer decision.

Phases follow the manuscript. For an event shock they are the window before
the shock tick, the window that begins at it, and the window after that. For a
regime scenario, which has no shock tick, they are read around the stress
window the preset declares.

The primary facility-share estimand is the ratio of summed successful routed
customer volume on the automated venue to summed successful routed customer
volume on every venue.  It is not a mean of per-second shares: quiet seconds
have no volume share and cannot be entered as zeros.  Customer trade-count and
active-tick shares are reported separately.  AMM arbitrage is also separate;
its AMM-leg volume is economically important to LP income but is not customer
routing demand.
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

# The label in the manuscript against the preset that produces it.
SCENARIOS = [
    ('Isolated dealer retreat', 'mm_withdrawal'),
    ('Flash crash', 'flash_crash'),
    ('Dealer liquidity crisis', 'dealer_liquidity_crisis'),
    ('Funding shock', 'funding_liquidity_shock'),
    ('High volatility stress', 'high_vol_stress'),
]

RAW = os.path.join(ROOT, 'output', 'resilience', 'raw_migration')
WINDOW = 100
PRE = 150


def _phases(shock, stress, n):
    """Before, during and after, in ticks."""
    if shock is not None and shock > 0:
        t0 = int(shock)
    elif stress:
        t0 = int(stress[0])
    else:
        t0 = n // 2
    return ((max(0, t0 - PRE), t0),
            (t0, min(n, t0 + WINDOW)),
            (min(n, t0 + WINDOW), min(n, t0 + WINDOW + PRE)))


def _phase_estimands(logger, lo, hi):
    """Return non-overlapping flow estimands for one phase."""
    amm_venues = [venue for venue in logger.flow_volume if venue != 'clob']
    customer_amm_volume = sum(
        logger.customer_volume(venue, lo, hi) for venue in amm_venues
    )
    customer_total_volume = sum(
        logger.customer_volume(venue, lo, hi)
        for venue in logger.flow_volume
    )
    arbitrage_volume = logger.arbitrage_volume_total(start=lo, end=hi)
    return {
        'share': logger.amm_customer_volume_share(lo, hi),
        'active_tick_share': logger.amm_active_tick_flow_share(lo, hi),
        'customer_trade_share': logger.amm_customer_trade_share(lo, hi),
        'customer_amm_volume_base': customer_amm_volume,
        'customer_total_volume_base': customer_total_volume,
        'arbitrage_volume_base': arbitrage_volume,
        'amm_execution_volume_base': customer_amm_volume + arbitrage_volume,
        'arbitrage_share_of_amm_execution': (
            logger.arbitrage_share_of_amm_execution(lo, hi)
        ),
    }


def measure(job):
    seed, preset, n_iter = job
    argv = ['--preset', preset, '--seed', str(seed),
            '--n-iter', str(n_iter), '--silent']
    p = build_parser()
    a = p.parse_args(argv)
    _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv)
    _auto_stress_around_shock(a)
    a.enable_amm = 1
    _seed_all(seed)
    sim = build_sim(a)
    sim.simulate(a.n_iter, silent=True)

    lg = sim.logger
    drawn = np.asarray(lg.mm_channel_shares.get('endogenous', []), dtype=float)
    forced = np.asarray(lg.mm_channel_shares.get('forced_pause', []), dtype=float)
    defen = np.asarray(lg.mm_channel_shares.get('defensive', []), dtype=float)
    n = len(lg.iterations)
    stress = None
    if getattr(a, 'stress_start', None) not in (None, -1):
        stress = (a.stress_start, getattr(a, 'stress_end', None))
    ph = _phases(getattr(a, 'shock_iter', None), stress, n)

    def m(arr, lo, hi):
        seg = arr[lo:hi]
        seg = seg[np.isfinite(seg)]
        return float(seg.mean()) if seg.size else float('nan')

    phase_rows = {
        name: _phase_estimands(lg, *window)
        for name, window in zip(('before', 'during', 'after'), ph)
    }

    return {
        'seed': seed, 'preset': preset,
        'model_signature': model_signature(ROOT),
        'signature': migration_signature(),
        **{
            f'{metric}_{phase}': value
            for phase, row in phase_rows.items()
            for metric, value in row.items()
        },
        'endogenous_during': m(drawn, *ph[1]),
        'endogenous_before': m(drawn, *ph[0]),
        'endogenous_w60': m(drawn, ph[1][0], min(n, ph[1][0] + 60)),
        'endogenous_peak': (float(np.nanmax(drawn[ph[1][0]:ph[1][1]]))
                            if drawn[ph[1][0]:ph[1][1]].size else float('nan')),
        'forced_during': m(forced, *ph[1]),
        'forced_w60': m(forced, ph[1][0], min(n, ph[1][0] + 60)),
        'forced_peak': (float(np.nanmax(forced[ph[1][0]:ph[1][1]]))
                        if forced[ph[1][0]:ph[1][1]].size else float('nan')),
        'defensive_during': m(defen, *ph[1]),
        'defensive_w60': m(defen, ph[1][0], min(n, ph[1][0] + 60)),
        'defensive_peak': (float(np.nanmax(defen[ph[1][0]:ph[1][1]]))
                           if defen[ph[1][0]:ph[1][1]].size else float('nan')),
        # Both the first sixty ticks and the full phase are retained because a
        # brief imposed pause and a persistent endogenous state are different
        # objects even when their full-window averages happen to coincide.
    }


_MIGRATION_SIGNATURE = None


def migration_signature():
    global _MIGRATION_SIGNATURE
    if _MIGRATION_SIGNATURE is None:
        _MIGRATION_SIGNATURE = measurement_signature(
            'migration_table',
            model_signature(ROOT),
            functions=(_phases, _phase_estimands, measure),
            constants=(tuple(SCENARIOS), WINDOW, PRE),
        )
    return _MIGRATION_SIGNATURE


def _path(preset):
    os.makedirs(RAW, exist_ok=True)
    return os.path.join(RAW, f'migration_{preset}.jsonl')


def load(preset, sig):
    have, stale = {}, 0
    p = _path(preset)
    if not os.path.exists(p):
        return have, stale
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get('signature') != sig:
            stale += 1
            continue
        have[int(r['seed'])] = r
    return have, stale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--seed-start', type=int, default=42)
    ap.add_argument('--n-iter', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--preset', choices=[preset for _, preset in SCENARIOS],
                    default=None, help='measure one scenario only')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    sig = migration_signature()
    wanted = list(range(args.seed_start, args.seed_start + args.seeds))
    done_new = 0

    scenarios = [row for row in SCENARIOS
                 if args.preset is None or row[1] == args.preset]
    for label, preset in scenarios:
        have, stale = load(preset, sig)
        todo = [s for s in wanted if s not in have]
        if args.chunk:
            todo = todo[:max(0, args.chunk - done_new)]
        if todo and not args.report_only:
            jobs = [(s, preset, args.n_iter) for s in todo]
            with open(_path(preset), 'a', encoding='utf-8') as fh:
                if args.workers > 1:
                    with ProcPool(args.workers) as pool:
                        for rec in pool.imap_unordered(measure, jobs):
                            fh.write(json.dumps(rec) + '\n')
                            fh.flush()
                            done_new += 1
                else:
                    for job in jobs:
                        fh.write(json.dumps(measure(job)) + '\n')
                        fh.flush()
                        done_new += 1
        if args.chunk and done_new >= args.chunk:
            break

    lines = [f'Model signature {model_signature(ROOT)}.',
             f'Migration measurement signature {sig}.',
             'Facility share by phase is the ratio of summed successful routed '
             'customer base volume on AMMs to that on all venues.',
             'Quiet seconds are not zeros. Customer trade-count share, '
             'active-tick share and AMM arbitrage volume are distinct estimands.',
             'Arbitrage is the successful AMM leg only and is excluded from '
             'the customer-routing share.',
             'Endogenous, forced-pause and defensive dealer channels are kept '
             'separate.',
             'One pool in the market, so the facility share is the hybrid pool '
             'alone.', '']
    head = (f'  {"scenario":24s}{"before":>8s}{"during":>8s}{"after":>8s}'
            f'{"delta pp":>10s}{"endo 60":>9s}{"forced 60":>11s}'
            f'{"def 60":>8s}{"def peak":>10s}{"seeds":>7s}')
    lines.append(head)
    for label, preset in scenarios:
        have, stale = load(preset, sig)
        recs = [have[s] for s in wanted if s in have]
        if not recs:
            lines.append(f'  {label:24s}{"nothing measured yet":>40s}')
            continue

        def col(k):
            # Records written before a column existed are simply absent, so a
            # cache from an earlier pass degrades to a blank cell rather than
            # forcing every seed to be measured again.
            v = np.asarray([r.get(k, float('nan')) for r in recs], dtype=float)
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float('nan')

        b, d, a_ = col('share_before'), col('share_during'), col('share_after')
        e60 = col('endogenous_w60')
        f60 = col('forced_w60')
        d60 = col('defensive_w60')
        dpk = col('defensive_peak')
        lines.append(f'  {label:24s}{100 * b:>7.0f}%{100 * d:>7.0f}%'
                     f'{100 * a_:>7.0f}%{100 * (d - b):>+9.1f}'
                     f'{100 * e60:>8.1f}%{100 * f60:>10.1f}%'
                     f'{100 * d60:>7.1f}%{100 * dpk:>9.1f}%'
                     f'{len(recs):>7d}')

    lines.append('')
    lines.append('  during-phase AMM estimands (means across seed-level ratios/volumes)')
    lines.append(
        f'  {"scenario":24s}{"customer vol":>14s}{"active tick":>14s}'
        f'{"trade count":>14s}{"arb base":>12s}{"arb/AMM":>11s}'
    )
    for label, preset in scenarios:
        have, _ = load(preset, sig)
        recs = [have[s] for s in wanted if s in have]
        if not recs:
            continue

        def during_col(key):
            values = np.asarray(
                [row.get(key, float('nan')) for row in recs], dtype=float
            )
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else float('nan')

        customer = during_col('share_during')
        active = during_col('active_tick_share_during')
        trades = during_col('customer_trade_share_during')
        arb_volume = during_col('arbitrage_volume_base_during')
        arb_share = during_col('arbitrage_share_of_amm_execution_during')
        lines.append(
            f'  {label:24s}{100 * customer:>13.1f}%'
            f'{100 * active:>13.1f}%{100 * trades:>13.1f}%'
            f'{arb_volume:>12.1f}{100 * arb_share:>10.1f}%'
        )
    lines.append('')
    lines.append('  endogenous withdrawal outside the shock window, as a control')
    for label, preset in scenarios:
        have, _ = load(preset, sig)
        recs = [have[s] for s in wanted if s in have]
        if recs:
            v = np.asarray([r['endogenous_before'] for r in recs], dtype=float)
            v = v[np.isfinite(v)]
            lines.append(f'    {label:24s}{100 * v.mean():>6.1f}%')

    text = '\n'.join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, 'w', encoding='utf-8').write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
