#!/usr/bin/env python3
"""Where the provider result changed, and which correction did it.

Four things were wrong with the published measurement at once, so knowing the
new figure is not the same as understanding it. Each correction is applied here
on top of the previous one over the same simulated paths, which attributes the
move to its source rather than leaving a single unexplained jump.

  v0  as published: a static endpoint benchmark holding the reserves seen at
      the start of the window, reserves paired with the price of the previous
      period, capital flows left in, hybrid function pool only
  v1  v0 with the reserve and price series aligned
  v2  v1 with the benchmark measured period by period against the reserves
      carried into that period, which is the realised counterpart of loss
      versus rebalancing
  v3  v2 with capital flows netted out, valued at the pool mid
  v4  v3 with the flows valued at the reference price, which is where every
      other term in the statement is marked

    python3 tools/robustness/lp_pnl_decompose.py --seeds 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from tools.robustness.lp_pnl_corrected import run, CALM, CRISIS


def variants(pool, price, shock, window):
    x = np.asarray(pool.x_history, dtype=float)
    y = np.asarray(pool.y_history, dtype=float)
    fr = np.asarray(pool.fee_revenue_history, dtype=float)
    dx = np.asarray(getattr(pool, 'flow_dx_history', []), dtype=float)
    dy = np.asarray(getattr(pool, 'flow_dy_history', []), dtype=float)
    n = min(len(x) - 1, len(y) - 1, len(fr) - 1, len(dx), len(dy), len(price))
    t0, t1 = shock + window[0], shock + window[1]
    if t0 < 1 or t1 >= n:
        return None
    out = {}

    # v0, exactly as published
    v0 = x[t0] * price[t0] + y[t0]
    p1 = price[t1]
    il = (x[t1] * p1 + y[t1]) - (x[t0] * p1 + y[t0])
    out['v0'] = 100.0 * (il + fr[t1] - fr[t0]) / v0

    # v1, series aligned
    v0a = x[t0 + 1] * price[t0] + y[t0 + 1]
    il = (x[t1 + 1] * p1 + y[t1 + 1]) - (x[t0 + 1] * p1 + y[t0 + 1])
    out['v1'] = 100.0 * (il + fr[t1 + 1] - fr[t0 + 1]) / v0a

    # v2 to v4, period by period, each adding one term
    for tag, flow_at in (('v2', None), ('v3', 'mid'), ('v4', 'ref')):
        lvr = fees = 0.0
        for i in range(t0 + 1, t1 + 1):
            p_now = price[i]
            lvr += (x[i] * p_now + y[i]) - (x[i + 1] * p_now + y[i + 1])
            if flow_at == 'ref':
                lvr += dx[i] * p_now + dy[i]
            elif flow_at == 'mid':
                mid = y[i] / x[i] if x[i] > 0 else p_now
                lvr += dx[i] * mid + dy[i]
            fees += fr[i + 1] - fr[i]
        out[tag] = 100.0 * (fees - lvr) / v0a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--fee-bps', type=float, default=5.0)
    a = ap.parse_args()

    rows = {}
    for sd in range(42, 42 + a.seeds):
        sim, sh = run(sd, a.fee_bps / 1e4)
        price = np.asarray(sim.logger.fair_price_series, dtype=float)
        for label, win in (('calm', CALM), ('crisis', CRISIS)):
            v = variants(sim.amm_pools['hfmm'], price, sh, win)
            if v:
                rows.setdefault(label, []).append(v)

    names = {'v0': 'as published',
             'v1': '+ series aligned',
             'v2': '+ rebalancing benchmark',
             'v3': '+ flows netted at pool mid',
             'v4': '+ flows at reference price'}
    print(f'Hybrid function pool, net result per cent, fee {a.fee_bps:g} bps, '
          f'{a.seeds} seeds, medians\n')
    print(f"  {'step':30s} {'calm':>9s} {'moved':>8s}   {'crisis':>9s} {'moved':>8s}")
    prev = {}
    for k in ('v0', 'v1', 'v2', 'v3', 'v4'):
        line = f'  {names[k]:30s}'
        for label in ('calm', 'crisis'):
            m = float(np.median([r[k] for r in rows[label]]))
            d = '' if k == 'v0' else f'{m - prev[label]:+8.4f}'
            line += f' {m:9.4f} {d:>8s}  '
            prev[label] = m
        print(line)
    print(f"\n  {'total move from published':30s}", end='')
    for label in ('calm', 'crisis'):
        a0 = float(np.median([r['v0'] for r in rows[label]]))
        a4 = float(np.median([r['v4'] for r in rows[label]]))
        print(f' {a4 - a0:+9.4f} {"":>8s}  ', end='')
    print()


if __name__ == '__main__':
    main()
