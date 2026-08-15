#!/usr/bin/env python3
"""Does the backstop survive without the liquidity floor?

The pool's liquidity providers follow a rule in which fee income raises
committed liquidity while volatility and funding cost reduce it, subject to a
floor that preserves a share of initial liquidity (core_liquidity_ratio, 0.75
for the HFMM in the shipped configuration). That floor hard codes the very
availability the headline result depends on, which is the substance of the
referee's objection that continued availability is assumed and not generated.

This script reruns the dealer liquidity crisis with the floor removed, holding
everything else fixed, and pairs the result against the stored baseline arms.

Usage
    python3 tools/robustness/lp_floor_experiment.py [--seeds N] [--floor F]
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
try:
    from main import _primary_run_label
except Exception:
    _primary_run_label = lambda a: 'dlc'

PRESET = 'dealer_liquidity_crisis'
N_ITER = 1000
BASE_SEED = 42


def run_one(seed: int, floor: float):
    """One paired run of the crisis arm with the LP floor set to `floor`."""
    argv = ['--preset', PRESET, '--seed', str(seed),
            '--n-iter', str(N_ITER), '--silent']
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv)
    _auto_stress_around_shock(a)
    a.run_label = _primary_run_label(a)
    a.enable_amm = 1
    a.clob_amm_interaction = 'competition'
    _seed_all(seed)
    sim = build_sim(a)
    # Override the floor after construction. update_liquidity reads the
    # attribute at call time, so no change to the model code is needed.
    for lp in (getattr(sim, 'lp_providers', None) or []):
        lp.core_liquidity_ratio = floor
    sim.simulate(a.n_iter, silent=True)
    log = sim.logger
    spr = np.asarray(log.clob_qspr, dtype=float)
    sh = int(a.shock_iter)
    liq_end, liq_min = [], []
    for lp in (getattr(sim, 'lp_providers', None) or []):
        try:
            x = np.asarray(lp.pool.x_history, dtype=float)
            y = np.asarray(lp.pool.y_history, dtype=float)
            # Geometric mean liquidity can stay flat while the pool is
            # severely imbalanced, so each reserve is tracked separately and
            # the binding side is the one that matters for availability.
            gx = x / max(x[0], 1e-12)
            gy = y / max(y[0], 1e-12)
            wx, wy = gx[sh:sh + 250], gy[sh:sh + 250]
            liq_min.append(float(min(np.nanmin(wx), np.nanmin(wy)))
                           if wx.size else np.nan)
            liq_end.append(float(min(gx[-1], gy[-1])))
        except Exception:
            liq_min.append(np.nan); liq_end.append(np.nan)
    return spr, sh, liq_end, liq_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=300)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--floor', type=float, default=0.0)
    ap.add_argument('--out', default='output/resilience/lp_floor_raw.npz')
    ap.add_argument('--cache', default='output/resilience/_lp_floor_cache')
    ap.add_argument('--budget', type=float, default=0.0,
                    help='stop after this many seconds, 0 means no limit')
    args = ap.parse_args()

    s0 = BASE_SEED + args.start
    seeds = list(range(s0, s0 + args.seeds))
    cache = os.path.join(ROOT, args.cache, 'floor_%g' % args.floor)
    os.makedirs(cache, exist_ok=True)

    todo = [sd for sd in seeds
            if not os.path.exists(os.path.join(cache, '%d.npz' % sd))]
    print('cached %d/%d, to run %d' % (len(seeds) - len(todo), len(seeds), len(todo)),
          flush=True)

    t0 = time.time()
    for k, sd in enumerate(todo, 1):
        if args.budget and (time.time() - t0) > args.budget:
            print('budget reached after %d seeds this pass' % (k - 1), flush=True)
            break
        spr, sh, liq, lmin = run_one(sd, args.floor)
        np.savez(os.path.join(cache, '%d.npz' % sd),
                 traj=spr[sh - 50:sh + 250], shock=sh,
                 end=np.asarray(liq, dtype=float),
                 min=np.asarray(lmin, dtype=float))
        if k % 5 == 0:
            el = time.time() - t0
            print('  %d done this pass, %.0fs, %.1fs per seed' % (k, el, el / k),
                  flush=True)

    trajs, ends, mins, shock = [], [], [], None
    for sd in seeds:
        f = os.path.join(cache, '%d.npz' % sd)
        if not os.path.exists(f):
            continue
        d = np.load(f)
        trajs.append(d['traj']); ends.append(d['end']); mins.append(d['min'])
        shock = int(d['shock'])
    print('aggregating %d/%d seeds' % (len(trajs), len(seeds)), flush=True)
    if not trajs:
        return

    m = min(len(t) for t in trajs)
    arr = np.vstack([t[:m] for t in trajs])
    outp = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    np.savez(outp, floor_traj=arr, shock=shock, floor=args.floor,
             n_seeds=len(trajs),
             end_liquidity=np.asarray(ends, dtype=float),
             min_liquidity=np.asarray(mins, dtype=float))
    print('saved %s shape=%s floor=%g' % (args.out, arr.shape, args.floor))
    ml = np.asarray(mins, dtype=float)
    if ml.size:
        print('binding reserve as a share of its initial level, median by pool')
        print('   trough in shock window:', np.round(np.nanmedian(ml, axis=0), 3))
        print('   worst seed            :', np.round(np.nanmin(ml, axis=0), 3))


if __name__ == '__main__':
    main()
