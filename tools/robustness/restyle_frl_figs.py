#!/usr/bin/env python3
"""Restyle the three FRL line/scatter figures from cached raw data (no re-run).

Reads the saved aggregates
  output/resilience/final_300seed_raw.npz   (spread trajectories, 300 seeds)
  output/resilience/cost_by_size_raw.npz    (cost-by-trade-size curves)
and regenerates, in the spare reference-figure style:
  output/resilience/dlc_spread_trajectory.png
  output/resilience/dlc_recovery_cdf.png
  output/resilience/dlc_cost_by_trade_size.png

Curves are identical to the originals; only the styling changes. Axis labels
are concise and placed at the ends of the axes; detailed description lives in
the paper text, not in the figure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AgentBasedModel.visualization.paper_style import (
    use_paper_style, end_labels, tidy_origin, COLOR_WITH_AMM, COLOR_NO_AMM)

use_paper_style()
RES = 'output/resilience'
SHOCK_IDX = 50  # shock sits at slice index 50 in the saved trajectories

WITH_LBL = 'With AMM (hybrid)'
WO_LBL = 'Without AMM (dealer CLOB only)'


def _save(fig, name):
    out = os.path.join(RES, name)
    fig.savefig(out, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def spread_trajectory():
    d = np.load(os.path.join(RES, 'final_300seed_raw.npz'))
    bt, ct = d['base_traj'], d['comp_traj']
    mb, mc = np.nanmean(bt, axis=0), np.nanmean(ct, axis=0)
    x = np.arange(len(mb)) - SHOCK_IDX
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axvline(0, color='#cccccc', ls='-', lw=0.8, zorder=0)
    ax.plot(x, mb, label=WO_LBL, color=COLOR_NO_AMM, lw=1.7)
    ax.plot(x, mc, label=WITH_LBL, color=COLOR_WITH_AMM, lw=1.7)
    ax.set_xlim(x.min(), x.max())
    ax.legend(loc='upper right')
    end_labels(ax, 'Ticks from shock', 'Spread (bps)')
    _save(fig, 'dlc_spread_trajectory.png')
    print('  peak with=%.2f without=%.2f' % (np.nanmax(mc), np.nanmax(mb)))


def recovery_cdf():
    d = np.load(os.path.join(RES, 'final_300seed_raw.npz'))
    bt, ct = d['base_traj'], d['comp_traj']
    s, HOLD, L = SHOCK_IDX, 10, 4.0
    H = min(bt.shape[1] - s, ct.shape[1] - s, 200)

    def rec(traj):
        out = []
        for row in traj:
            r = H
            for t in range(1, H - HOLD):
                if np.all(row[s + t:s + t + HOLD] <= L):
                    r = t
                    break
            out.append(r)
        return np.array(out)

    rw, ro = rec(ct), rec(bt)
    grid = np.arange(0, H)
    cdf_w = np.array([np.mean(rw <= t) for t in grid])
    cdf_o = np.array([np.mean(ro <= t) for t in grid])
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(grid, 100 * cdf_o, label=WO_LBL, color=COLOR_NO_AMM, lw=1.7)
    ax.plot(grid, 100 * cdf_w, label=WITH_LBL, color=COLOR_WITH_AMM, lw=1.7)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, H - 1)
    ax.legend(loc='lower right')
    end_labels(ax, 'Ticks since shock', 'Recovered (%)')
    tidy_origin(ax)
    _save(fig, 'dlc_recovery_cdf.png')
    print('  median recovery with=%.0f without=%.0f' % (np.median(rw), np.median(ro)))


def cost_by_size():
    d = np.load(os.path.join(RES, 'cost_by_size_raw.npz'))
    QS = d['QS']
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    # calm = mid gray, stress = near-black; CLOB solid/circle, HFMM dashed/square
    ax.plot(QS, d['cl_c'], 'o-', color=COLOR_NO_AMM, lw=1.5, label='CLOB, calm')
    ax.plot(QS, d['am_c'], 's--', color=COLOR_NO_AMM, lw=1.3, mfc='white', label='HFMM, calm')
    ax.plot(QS, d['cl_s'], 'o-', color=COLOR_WITH_AMM, lw=1.7, label='CLOB, stress')
    ax.plot(QS, d['am_s'], 's--', color=COLOR_WITH_AMM, lw=1.5, mfc='white', label='HFMM, stress')
    ax.set_xlim(QS.min(), QS.max())
    ax.legend(loc='upper left', ncol=2)
    end_labels(ax, 'Trade size (units)', 'Cost (bps)')
    _save(fig, 'dlc_cost_by_trade_size.png')


if __name__ == '__main__':
    spread_trajectory()
    recovery_cdf()
    cost_by_size()
