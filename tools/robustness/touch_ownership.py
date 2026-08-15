#!/usr/bin/env python3
"""Who actually stands at the best price, and who leaves when the shock lands.

    python3 tools/robustness/touch_ownership.py --seeds 12

The manuscript attributes the crisis spread spike to dealers withdrawing. That
attribution is only sound if dealers are the ones setting the best price to
begin with. After the price volatility was re anchored, a spot check on one
seed suggested they are not, so this measures it properly across seeds and
across the phases of the shock.

The book is read once per tick at the point where ``SimulatorInfo.capture``
runs, which is after the environment step and before the traders act. That is
a consistent sampling point rather than the only possible one, and every
number here is a share of ticks sampled that way.

What the tool is for is a decomposition, not a score. If the dealer classes
own the touch in calm and lose it in the crisis, the story in the manuscript
holds as written. If the touch belongs throughout to classes whose retreat is
a rule rather than a decision, then the endogenous withdrawal machinery is
not the mechanism behind the headline number and the text has to say so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from tools.robustness.lp_pnl_corrected import N_ITER, PRESET, run_signature

RAW = os.path.join(ROOT, 'output', 'resilience', 'raw_touch')


def _seed_record(seed, n_iter, band_bps):
    """Everything one seed contributes, already reduced to counters.

    Walking the book every tick is the expensive part, so a seed is measured
    once and its per phase tallies are stored. Without this the tool had to
    finish every seed inside one call, which is the one thing the sandbox
    cannot promise.
    """
    rows, depth, shock = run_seed(seed, n_iter, band_bps)
    ph = phases(rows, shock)
    rec = {'seed': seed, 'signature': run_signature(), 'phases': {}}
    for name, (lo, hi) in ph.items():
        c, t = tally(rows, lo, hi)
        d = Counter()
        for one in depth[lo:hi]:
            d.update(one)
        rec['phases'][name] = {
            'touch': dict(c), 'touch_total': t, 'depth': dict(d),
            'spreads': [r[2] for r in rows[lo:hi] if r[2] is not None],
        }
    lo, hi = ph['during']
    window = [(r[2] if r[2] is not None else -1, i)
              for i, r in enumerate(rows[lo:hi])]
    if window:
        _, idx = max(window)
        b, a_, _ = rows[lo + idx]
        rec['peak_owner'] = [b, a_]
    return rec


def _load(band_bps, sig):
    have, stale = {}, 0
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f'touch_band{band_bps:g}.jsonl')
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
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
    return have, stale, path

# Dealers proper. Everything else supplies liquidity by a rule, which is the
# distinction the measurement exists to draw.
DEALER = {'MarketMaker'}


def _owner(book):
    """The class standing at the front of one side of the book."""
    if book is None:
        return 'empty'
    first = getattr(book, 'first', None)
    if first is None:
        return 'empty'
    trader = getattr(first, 'trader', None)
    if trader is None:
        return 'background'
    return type(trader).__name__


def _depth_by_class(book_dict, mid, band_bps):
    """Resting quantity within a band of the mid, attributed to its owner.

    Standing at the front is not the same as carrying the book, and a dealer
    could be absent from the touch while supplying most of the size behind
    it. Without this the ownership table would invite exactly that mistake.
    """
    out = Counter()
    if not mid or mid <= 0:
        return out
    lo, hi = mid * (1 - band_bps / 1e4), mid * (1 + band_bps / 1e4)
    for side in ('bid', 'ask'):
        book = book_dict.get(side)
        if book is None:
            continue
        try:
            orders = list(book)
        except TypeError:
            continue
        for o in orders:
            price = float(getattr(o, 'price', 0.0) or 0.0)
            if not (lo <= price <= hi):
                continue
            trader = getattr(o, 'trader', None)
            name = 'background' if trader is None else type(trader).__name__
            out[name] += float(getattr(o, 'qty', 0.0) or 0.0)
    return out


def _install_probe(sim, out, depth_out, band_bps):
    """Record the touch each tick by wrapping the per tick capture."""
    info = sim.info
    original = info.capture

    def capture(*a, **kw):
        book = getattr(sim.exchange, 'order_book', {}) or {}
        bid, ask = _owner(book.get('bid')), _owner(book.get('ask'))
        sp, mid = None, None
        try:
            s = sim.exchange.spread()
            if s and s.get('bid') and s.get('ask'):
                mid = 0.5 * (s['bid'] + s['ask'])
                if mid > 0:
                    sp = 1e4 * (s['ask'] - s['bid']) / mid
        except Exception:
            sp, mid = None, None
        out.append((bid, ask, sp))
        depth_out.append(_depth_by_class(book, mid, band_bps))
        return original(*a, **kw)

    info.capture = capture
    return original


def run_seed(seed, n_iter, band_bps=25.0):
    argv = ['--preset', PRESET, '--seed', str(seed),
            '--n-iter', str(n_iter), '--silent']
    p = build_parser()
    a = p.parse_args(argv)
    _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv)
    _auto_stress_around_shock(a)
    a.enable_amm = 1
    a.clob_amm_interaction = 'competition'
    _seed_all(seed)
    sim = build_sim(a)
    rows, depth = [], []
    _install_probe(sim, rows, depth, band_bps)
    sim.simulate(a.n_iter, silent=True)
    return rows, depth, int(a.shock_iter)


def phases(rows, shock, pre=150, during=100, post=150):
    """The three windows the shock divides the run into."""
    n = len(rows)
    return {
        'before': (max(0, shock - pre), shock),
        'during': (shock, min(n, shock + during)),
        'after': (min(n, shock + during), min(n, shock + during + post)),
    }


def tally(rows, lo, hi):
    """Share of sampled ticks each class held a side of the touch."""
    c = Counter()
    total = 0
    for bid, ask, _ in rows[lo:hi]:
        for side in (bid, ask):
            c[side] += 1
            total += 1
    return c, total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--seed-start', type=int, default=42)
    ap.add_argument('--n-iter', type=int, default=N_ITER)
    ap.add_argument('--band-bps', type=float, default=25.0,
                    help='half width of the depth band around the mid')
    ap.add_argument('--chunk', type=int, default=0,
                    help='stop after this many new seeds')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    agg = {k: Counter() for k in ('before', 'during', 'after')}
    tot = {k: 0 for k in agg}
    spreads = {k: [] for k in agg}
    peak_owner = Counter()

    dagg = {k: Counter() for k in agg}
    sig = run_signature()
    have, stale, path = _load(args.band_bps, sig)
    if stale:
        print(f'{stale} stored seeds were made under a different model '
              f'and are ignored.')
    wanted = list(range(args.seed_start, args.seed_start + args.seeds))
    todo = [s for s in wanted if s not in have]
    if args.chunk:
        todo = todo[:args.chunk]
    if todo:
        print(f'measuring {len(todo)} seeds, {len(have)} already stored')
        with open(path, 'a', encoding='utf-8') as fh:
            for seed in todo:
                rec = _seed_record(seed, args.n_iter, args.band_bps)
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                have[seed] = rec

    recs = [have[s] for s in wanted if s in have]
    if not recs:
        print('no seeds measured yet')
        return 1
    for rec in recs:
        for name, blk in rec['phases'].items():
            agg[name].update(blk['touch'])
            tot[name] += blk['touch_total']
            dagg[name].update(blk['depth'])
            spreads[name] += blk['spreads']
        for who in rec.get('peak_owner', []):
            peak_owner[who] += 1

    lines = []

    def w(s=''):
        lines.append(s)

    w(f'Touch ownership around the {PRESET} shock, {len(recs)} seeds, '
      f'{args.n_iter} ticks each.')
    w('Shares are of sampled book sides, both sides counted, one sample a tick.')
    w()
    classes = sorted({k for c in agg.values() for k in c},
                     key=lambda k: -agg['before'][k])
    w(f'  {"class":22s}{"before":>10s}{"during":>10s}{"after":>10s}')
    for k in classes:
        row = ''.join(f'{(agg[p][k] / tot[p] if tot[p] else 0):>9.1%} '
                      for p in ('before', 'during', 'after'))
        mark = '  <- dealer' if k in DEALER else ''
        w(f'  {k:22s}{row}{mark}')
    w()
    dealer_share = {p: sum(agg[p][k] for k in DEALER) / tot[p] if tot[p] else 0
                    for p in agg}
    w(f'  dealers hold the touch  '
      f'{dealer_share["before"]:.1%} before, {dealer_share["during"]:.1%} during, '
      f'{dealer_share["after"]:.1%} after')
    w()
    w(f'  share of resting quantity within {args.band_bps:.0f} bps of the mid')
    dclasses = sorted({k for c in dagg.values() for k in c},
                      key=lambda k: -dagg['before'][k])
    w(f'  {"class":22s}{"before":>10s}{"during":>10s}{"after":>10s}')
    for k in dclasses:
        row = ''
        for p in ('before', 'during', 'after'):
            s = sum(dagg[p].values())
            row += f'{(dagg[p][k] / s if s else 0):>9.1%} '
        mark = '  <- dealer' if k in DEALER else ''
        w(f'  {k:22s}{row}{mark}')
    w()
    w('  median quoted spread, bps')
    for p in ('before', 'during', 'after'):
        v = float(np.median(spreads[p])) if spreads[p] else float('nan')
        w(f'    {p:8s}{v:>8.2f}')
    w()
    w('  standing at the touch on the worst spread tick of the crisis')
    tp = sum(peak_owner.values()) or 1
    for k, v in peak_owner.most_common():
        w(f'    {k:22s}{v / tp:>8.1%}')

    text = '\n'.join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
