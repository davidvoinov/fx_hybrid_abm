"""Measurements the figures need and the stored panels do not carry.

The arms panel, the welfare reports and the selection report hold the numbers
behind most of the figures. Five of them need series or sweeps that no panel
records, and they are gathered here so that every figure in the paper is drawn
from one artifact with one signature.
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
from tools.robustness.facility_arms import (ARM_CAPITAL, ARM_SPREAD_BPS,
                                            OUTCOME_SIZE, PRESET, WINDOW)
from tools.robustness.signatures import measurement_signature, model_signature

N_ITER = 1000
PATH_ARMS = ('none', 'reserve', 'reserve_frozen', 'dealer_of_last_resort',
             'passive_book')
PATH_OFFSETS = tuple(range(0, WINDOW, 5))
SIZE_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0)
AVAILABILITY_SIZES = (2.0, 20.0, 100.0, 300.0, 600.0)
FEE_GRID = (1.0, 2.0, 5.0, 10.0, 20.0)
SEVERITY_GRID = (0.4, 0.6, 0.8, 1.0, 1.2)


def _args(arm, seed, capital=0.0, fee_bps=None, severity=None, preset=PRESET):
    """Arguments for one run. ``preset=None`` gives the calm market."""
    argv = ['--facility-arm', arm,
            '--arm-spread-bps', repr(ARM_SPREAD_BPS),
            '--seed', str(seed), '--n-iter', str(N_ITER), '--silent',
            '--amm-lp-model', 'endogenous']
    if preset:
        argv = ['--preset', preset] + argv
    if capital > 0:
        argv += ['--arm-capital', repr(capital)]
    parser = build_parser()
    args = parser.parse_args(argv)
    if preset:
        _apply_preset_defaults(parser, args)
    args.venue_choice_rule = _resolve_main_routing(args, argv)
    if fee_bps is not None:
        args.hfmm_fee = float(fee_bps) / 1e4
    if severity is not None:
        for name in ('shock_pct', 'fundamental_shock_pct', 'order_flow_shock_qty',
                     'liquidity_shock_frac', 'funding_vol_shock_intensity'):
            setattr(args, name, float(getattr(args, name)) * float(severity))
    if preset:
        _auto_stress_around_shock(args)
    _seed_all(seed)
    return args


def _executable(sim, size):
    clob = sim.clob
    pool = list(sim.amm_pools.values())[0] if sim.amm_pools else None
    try:
        buy = float(clob.cost_bps(size, 'buy'))
        sell = float(clob.cost_bps(size, 'sell'))
    except Exception:
        return float('nan')
    best = buy + sell if np.isfinite(buy) and np.isfinite(sell) else float('inf')
    if pool is not None:
        series = getattr(sim.logger, 'fair_price_series', []) or []
        reference = float(series[-1]) if series else float('nan')
        try:
            pb = float(pool.quote_buy(size, reference)['cost_bps'])
            ps = float(pool.quote_sell(size, reference)['cost_bps'])
            if np.isfinite(pb) and np.isfinite(ps):
                best = min(best, buy + sell) if np.isfinite(buy) else best
                best = min(best, pb + ps)
        except Exception:
            pass
    return best


def _path(task):
    """Executable cost through the window, and availability by size."""
    arm, seed = task
    capital = ARM_CAPITAL if arm in ('dealer_of_last_resort', 'passive_book') else 0.0
    args = _args(arm, seed, capital)
    shock = int(args.shock_iter)
    sim = build_sim(args)
    sim.simulate(shock, silent=True)
    path, served = [], {s: 0 for s in AVAILABILITY_SIZES}
    for t in range(WINDOW):
        sim.simulate(1, silent=True)
        if t in PATH_OFFSETS:
            path.append(_executable(sim, OUTCOME_SIZE))
        for s in AVAILABILITY_SIZES:
            if np.isfinite(_executable(sim, s)):
                served[s] += 1
    return {'arm': arm, 'path': path,
            'served': {str(s): served[s] / WINDOW for s in AVAILABILITY_SIZES}}


def _providers(seed):
    """Reserves, wallets, departures and the pool's own price through the window."""
    args = _args('reserve', seed)
    shock = int(args.shock_iter)
    sim = build_sim(args)
    sim.simulate(shock, silent=True)
    pool = list(sim.amm_pools.values())[0]
    lps = [lp for pop in sim.lp_providers for lp in getattr(pop, 'providers', [])]
    reference = float(sim.logger.fair_price_series[-1])
    opening = float(pool.x) * reference + float(pool.y)
    rows = []
    for t in range(WINDOW):
        sim.simulate(1, silent=True)
        if t not in PATH_OFFSETS:
            continue
        reference = float(sim.logger.fair_price_series[-1])
        rows.append({
            'reserves': (float(pool.x) * reference + float(pool.y)) / opening,
            'wallets': sum(float(lp.wallet_base) * reference + float(lp.wallet_cash)
                           for lp in lps),
            'open': sum(1 for lp in lps if getattr(lp, 'active', True)),
            'cost': float(pool.quote_buy(OUTCOME_SIZE, reference)['cost_bps'])
                    + float(pool.quote_sell(OUTCOME_SIZE, reference)['cost_bps']),
        })
    return rows


def _size_curve(seed):
    """What each venue charges by size in the calm market."""
    args = _args('reserve', seed, preset=None)
    sim = build_sim(args)
    sim.simulate(N_ITER, silent=True)
    pool = list(sim.amm_pools.values())[0]
    clob = sim.clob
    reference = float(sim.logger.fair_price_series[-1])
    out = {}
    for size in SIZE_GRID:
        try:
            book = clob.cost_bps(size, 'buy') + clob.cost_bps(size, 'sell')
        except Exception:
            book = float('inf')
        try:
            venue = (pool.quote_buy(size, reference)['cost_bps']
                     + pool.quote_sell(size, reference)['cost_bps'])
        except Exception:
            venue = float('inf')
        out[str(size)] = {'book': float(book), 'pool': float(venue)}
    return out


def _frontier(task):
    """Provider result in calm and in crisis at one fee."""
    fee, seed = task
    from tools.robustness.lp_pnl_corrected import components
    out = {}
    for label, preset in (('calm', None), ('crisis', PRESET)):
        args = _args('reserve', seed, fee_bps=fee, preset=preset or None)
        sim = build_sim(args)
        sim.simulate(N_ITER, silent=True)
        shock = int(args.shock_iter) if preset else N_ITER // 2
        prices = np.asarray(sim.logger.fair_price_series, dtype=float)
        row = components(list(sim.amm_pools.values())[0], prices, shock, (0, WINDOW))
        out[label] = ((row[1] - row[0]) / row[2] * 100.0
                      if row and row[2] else float('nan'))
    return {'fee': fee, **out}


def _cascade(task):
    """Dislocation against the size of the shock, with the channel on and off."""
    severity, gain, seed = task
    args = _args('none', seed, severity=severity)
    args.dealer_cascade_gain = float(gain)
    sim = build_sim(args)
    shock = int(args.shock_iter)
    sim.simulate(shock, silent=True)
    # The executable round trip and not the book's own quoted spread. The
    # quoted spread is taken between whatever quotes remain, so a sector that
    # withdraws its widest dealers narrows it while the market is emptier: the
    # measured peak fell as severity rose past the point where the sector
    # reaches its withdrawal ceiling. Depth is what the cost at a size sees.
    peak = 0.0
    for _ in range(WINDOW):
        sim.simulate(1, silent=True)
        value = _executable(sim, OUTCOME_SIZE)
        if np.isfinite(value):
            peak = max(peak, value)
    return {'severity': severity, 'gain': gain, 'peak': float(peak)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=300)
    parser.add_argument('--frontier-seeds', type=int, default=60)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--output', default='output/figure_data.json')
    args = parser.parse_args(argv)
    seeds = list(range(42, 42 + args.seeds))
    report = {}

    with ProcPool(processes=args.workers) as pool:
        rows = pool.map(_path, [(a, s) for a in PATH_ARMS for s in seeds], chunksize=2)
        report['path'] = {
            arm: np.nanmedian(np.array([r['path'] for r in rows if r['arm'] == arm],
                                       dtype=float), axis=0).tolist()
            for arm in PATH_ARMS}
        report['availability'] = {
            arm: {k: float(np.mean([r['served'][k] for r in rows if r['arm'] == arm]))
                  for k in map(str, AVAILABILITY_SIZES)}
            for arm in PATH_ARMS}
        report['path_offsets'] = list(PATH_OFFSETS)

        provider = pool.map(_providers, seeds, chunksize=2)
        report['providers'] = {
            key: np.nanmedian(np.array([[r[key] for r in rows_] for rows_ in provider],
                                       dtype=float), axis=0).tolist()
            for key in ('reserves', 'wallets', 'open', 'cost')}

        curves = pool.map(_size_curve, seeds, chunksize=2)
        report['size_curve'] = {
            k: {v: float(np.nanmedian([c[k][v] for c in curves]))
                for v in ('book', 'pool')}
            for k in map(str, SIZE_GRID)}
        report['size_grid'] = list(SIZE_GRID)

        frontier = pool.map(_frontier, [(f, s) for f in FEE_GRID
                                        for s in seeds],
                            chunksize=2)
        report['frontier'] = {
            str(f): {k: float(np.nanmedian([r[k] for r in frontier if r['fee'] == f]))
                     for k in ('calm', 'crisis')}
            for f in FEE_GRID}

        cascade = pool.map(_cascade, [(v, g, s) for v in SEVERITY_GRID
                                      for g in (0.0, 4.0) for s in seeds],
                           chunksize=2)
        report['cascade'] = {
            str(v): {str(g): float(np.nanmedian(
                [r['peak'] for r in cascade
                 if r['severity'] == v and r['gain'] == g]))
                for g in (0.0, 4.0)}
            for v in SEVERITY_GRID}
        report['severity_grid'] = list(SEVERITY_GRID)

    digest = model_signature()
    report['provenance'] = {
        'preset': PRESET, 'n_iter': N_ITER, 'window': WINDOW,
        'seeds': args.seeds, 'simulation_model_signature': digest,
        'measurement_signature': measurement_signature(
            'figure_data', digest,
            functions=(_path, _providers, _size_curve, _frontier, _cascade),
            constants=(PRESET, N_ITER, WINDOW, SIZE_GRID, FEE_GRID,
                       SEVERITY_GRID, AVAILABILITY_SIZES)),
    }
    destination = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'written to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
