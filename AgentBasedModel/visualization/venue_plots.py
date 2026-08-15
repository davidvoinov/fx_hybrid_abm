"""
Visualization functions for multi-venue FX ABM results.

All functions accept a MetricsLogger instance and produce matplotlib figures.
Organised around two hypotheses:

    H1: Execution cost comparison CLOB vs AMM + AMM as added liquidity
    H2: Systemic linkage under stress — co-movement, liquidity flight to AMM
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from AgentBasedModel.metrics.logger import MetricsLogger
    from AgentBasedModel.venues.amm import CPMMPool, HFMMPool


# ===================================================================
#  H1 — Execution cost comparison  &  AMM adds liquidity
# ===================================================================

def plot_execution_cost_curves(logger: 'MetricsLogger',
                               figsize=(10, 6)):
    """
    Average all-in cost C_v(Q) across the whole simulation, per venue.
    Shows that AMM is structurally more expensive at all Q but still
    provides an executable alternative.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('H1 · Average Execution Cost  $C_v(Q)$ by Venue')
    ax.set_xlabel('Trade Size  $Q$  (base)')
    ax.set_ylabel('All-in cost (bps)')

    Q_grid = logger.Q_grid

    # CLOB
    means = _avg_finite(logger.clob_cost_curves, Q_grid)
    ax.plot(Q_grid, means, 'o-', label='CLOB', linewidth=2, color='#1f77b4')

    # AMMs
    colors = {'cpmm': '#ff7f0e', 'hfmm': '#2ca02c'}
    for name in logger.amm_cost_curves:
        means = _avg_finite(logger.amm_cost_curves[name], Q_grid)
        ax.plot(Q_grid, means, 's--', label=name.upper(), linewidth=2,
                color=colors.get(name, None))

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cost_decomposition(logger: 'MetricsLogger', Q: float = 5,
                            figsize=(10, 6)):
    """
    Stacked bar: slippage + fee breakdown for each venue (averaged).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'H1 · Cost Decomposition  (Q={Q}): Slippage vs Fee')

    venues = []
    slippages = []
    fees = []

    # CLOB
    s = logger.clob_cost_curves.get(Q, [])
    finite_s = [x for x in s if math.isfinite(x)]
    if finite_s:
        venues.append('CLOB')
        slippages.append(sum(finite_s) / len(finite_s))
        fees.append(0)

    # AMMs
    for name in logger.amm_slippage_curves:
        sl = logger.amm_slippage_curves[name].get(Q, [])
        fe = logger.amm_fee_curves[name].get(Q, [])
        finite_sl = [x for x in sl if math.isfinite(x)]
        finite_fe = [x for x in fe if math.isfinite(x)]
        if finite_sl:
            venues.append(name.upper())
            slippages.append(sum(finite_sl) / len(finite_sl))
            fees.append(sum(finite_fe) / len(finite_fe) if finite_fe else 0)

    x_pos = range(len(venues))
    ax.bar(x_pos, slippages, label='Slippage (price impact)', alpha=0.85,
           color='#4c72b0')
    ax.bar(x_pos, fees, bottom=slippages, label='LP Fee', alpha=0.85,
           color='#dd8452')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(venues)
    ax.set_ylabel('Cost (bps)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_total_market_depth(logger: 'MetricsLogger', rolling: int = 10,
                            figsize=(12, 5),
                            shock_iter: int = None):
    """
    Combined depth: CLOB depth (bid+ask qty) + AMM effective depth.
    Shows that AMM adds to total available liquidity even if more expensive.
    """
    fig, ax = plt.subplots(figsize=figsize)
    title = 'H1 · Total Market Depth: CLOB + AMM'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Depth (units)')

    n = len(logger.iterations)

    # CLOB depth
    clob_depths = []
    for d in logger.clob_depth:
        if isinstance(d, dict):
            clob_depths.append(d.get('bid', 0) + d.get('ask', 0))
        else:
            clob_depths.append(0)
    clob_r = _rolling_avg(clob_depths, rolling)
    ax.fill_between(range(n), 0, clob_r, alpha=0.4, label='CLOB depth',
                     color='#1f77b4')

    # AMM depths (stacked on top)
    amm_total = [0.0] * n
    for name in logger.amm_depth_series:
        vals = logger.amm_depth_series[name][:n]
        while len(vals) < n:
            vals.append(0)
        amm_total = [a + v for a, v in zip(amm_total, vals)]

    combined = [c + a for c, a in zip(clob_r, amm_total)]
    amm_r = _rolling_avg(amm_total, rolling)
    combined_r = [c + a for c, a in zip(clob_r, amm_r)]
    ax.fill_between(range(n), clob_r, combined_r, alpha=0.4,
                     label='AMM depth (added)', color='#2ca02c')

    _shade_stress(ax, logger)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===================================================================
#  H2 — Systemic linkage & liquidity flight under stress
# ===================================================================

def plot_cost_timeseries(logger: 'MetricsLogger', Q: float = 5,
                         rolling: int = 1, figsize=(12, 5)):
    """
    Time series of C_v(Q) for each venue — shows co-movement and
    stress-period divergence / convergence.
    """
    fig, ax = plt.subplots(figsize=figsize)
    title = f'H2 · Execution Cost Time Series  (Q={Q})'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (bps)')

    def _roll(s):
        if rolling <= 1:
            return list(range(len(s))), s
        out = []
        for i in range(len(s) - rolling + 1):
            window = [x for x in s[i:i + rolling] if math.isfinite(x)]
            out.append(sum(window) / len(window) if window else float('nan'))
        return list(range(rolling - 1, len(s))), out

    xs, ys = _roll(logger.cost_series('clob', Q))
    ax.plot(xs, ys, label='CLOB', linewidth=1.3, color='#1f77b4')

    colors = {'cpmm': '#ff7f0e', 'hfmm': '#2ca02c'}
    for name in logger.amm_cost_curves:
        xs, ys = _roll(logger.cost_series(name, Q))
        ax.plot(xs, ys, label=name.upper(), linewidth=1.3,
                color=colors.get(name, None))

    _shade_stress(ax, logger)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_flow_allocation(logger: 'MetricsLogger', rolling: int = 10,
                         figsize=(12, 5)):
    """
    Stacked area: zero-filled per-clock-tick customer venue use.

    Quiet ticks are plotted at zero for continuity.  This is a trajectory
    diagnostic, not an aggregate executed-volume share.
    """
    fig, ax = plt.subplots(figsize=figsize)
    title = 'H2 · Customer Venue Use per Clock Tick (quiet = 0)'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Customer Share per Tick')

    venues = list(logger.flow_volume.keys())
    if not venues:
        return

    n = len(logger.iterations)
    shares = {v: logger.flow_share(v) for v in venues}

    xs = list(range(n))
    bottoms = [0.0] * n
    venue_colors = {'clob': '#1f77b4', 'cpmm': '#ff7f0e', 'hfmm': '#2ca02c'}
    for v in venues:
        vals = _rolling_avg(shares[v], rolling)
        tops = [b + v_ for b, v_ in zip(bottoms, vals)]
        ax.fill_between(xs, bottoms, tops, label=v.upper(), alpha=0.6,
                         color=venue_colors.get(v, None))
        bottoms = tops

    _shade_stress(ax, logger)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clob_spread_vs_amm_cost(logger: 'MetricsLogger', Q: float = 5,
                                  rolling: int = 10, figsize=(12, 5)):
    """
    Dual-axis: CLOB quoted spread (bps) vs AMM all-in cost.
    If AMM becomes a refuge in stress, its cost should grow less than CLOB spread.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    title = f'H2 · CLOB Spread vs AMM Cost (Q={Q})'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax1.set_title(title)
    ax1.set_xlabel('Iteration')

    # CLOB spread
    qspr = _rolling_avg(logger.clob_qspr, rolling)
    ax1.plot(qspr, label='CLOB qspr (bps)', color='#1f77b4', linewidth=1.3)
    ax1.set_ylabel('CLOB Quoted Spread (bps)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # AMM cost on secondary axis
    ax2 = ax1.twinx()
    colors = {'cpmm': '#ff7f0e', 'hfmm': '#2ca02c'}
    for name in logger.amm_cost_curves:
        s = logger.cost_series(name, Q)
        s_r = _rolling_avg(s, rolling)
        ax2.plot(s_r, label=f'{name.upper()} cost (bps)',
                 color=colors.get(name, None), linewidth=1.3, linestyle='--')
    ax2.set_ylabel('AMM All-in Cost (bps)')
    ax2.tick_params(axis='y')

    _shade_stress(ax1, logger)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_commonality(logger: 'MetricsLogger', Q: float = 5,
                     window: int = 30, figsize=(12, 5)):
    """
    Rolling Pearson ρ between CLOB and AMM cost series.
    Higher ρ during stress → systemic linkage confirmed.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'H2 · Rolling Cost Correlation  (Q={Q}, window={window})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pearson ρ')

    clob_s = logger.cost_series('clob', Q)

    colors = {'cpmm': '#ff7f0e', 'hfmm': '#2ca02c'}
    for name in logger.amm_cost_curves:
        amm_s = logger.cost_series(name, Q)
        n = min(len(clob_s), len(amm_s))
        if n < window:
            continue
        rhos = []
        for i in range(window, n):
            w1 = clob_s[i - window:i]
            w2 = amm_s[i - window:i]
            pairs = [(a, b) for a, b in zip(w1, w2)
                     if math.isfinite(a) and math.isfinite(b)]
            if len(pairs) < 5:
                rhos.append(float('nan'))
                continue
            xs, ys = zip(*pairs)
            n_ = len(xs)
            mx, my = sum(xs) / n_, sum(ys) / n_
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n_
            sx = (sum((x - mx) ** 2 for x in xs) / n_) ** 0.5
            sy = (sum((y - my) ** 2 for y in ys) / n_) ** 0.5
            rhos.append(cov / (sx * sy) if sx > 0 and sy > 0 else 0)

        ax.plot(range(window, n), rhos, label=f'CLOB ↔ {name.upper()}',
                linewidth=1.3, color=colors.get(name, None))

    _shade_stress(ax, logger)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_amm_liquidity(logger: 'MetricsLogger', figsize=(12, 5)):
    """
    AMM pool liquidity L_t over time — does it survive stress?
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('H2 · AMM Liquidity Under Stress')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Liquidity Measure  $L_t$')

    colors = {'cpmm': '#ff7f0e', 'hfmm': '#2ca02c'}
    for name in logger.amm_L_series:
        ax.plot(logger.amm_L_series[name], label=name.upper(), linewidth=1.5,
                color=colors.get(name, None))

    _shade_stress(ax, logger)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stress_flow_migration(logger: 'MetricsLogger',
                                stress_start: int = None,
                                figsize=(10, 6)):
    """
    Bar chart: ratio-of-sums AMM customer volume share before vs stress.
    Direct test: does volume migrate to AMM when CLOB liquidity dries up?
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('H2 · AMM Customer Volume Share (Ratio of Sums)')

    venues = [v for v in logger.flow_volume if v != 'clob']
    n = len(logger.iterations)
    idx = min(stress_start, n) if stress_start is not None else n

    before_shares = {}
    stress_shares = {}
    for v in venues:
        before_shares[v] = logger.customer_volume_share(v, 0, idx)
        stress_shares[v] = logger.customer_volume_share(v, idx, n)

    x_pos = list(range(len(venues)))
    width = 0.35
    bars1 = ax.bar([x - width/2 for x in x_pos],
                    [before_shares[v] for v in venues],
                    width, label='Normal (before stress)', alpha=0.8,
                    color='#4c72b0')
    bars2 = ax.bar([x + width/2 for x in x_pos],
                    [stress_shares[v] for v in venues],
                    width, label='Stress period', alpha=0.8,
                    color='#c44e52')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([v.upper() for v in venues])
    ax.set_ylabel('Share of Total Volume')
    mx = max(max(before_shares.values(), default=0),
             max(stress_shares.values(), default=0))
    ax.set_ylim(0, mx * 1.4 if mx > 0 else 0.1)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1%}', xy=(bar.get_x() + bar.get_width()/2, h),
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1%}', xy=(bar.get_x() + bar.get_width()/2, h),
                    ha='center', va='bottom', fontsize=10)

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ===================================================================
#  Context / auxiliary plots
# ===================================================================

def plot_environment(logger: 'MetricsLogger', figsize=(12, 4)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_title('Volatility  $\\sigma_t$')
    ax1.plot(logger.sigma_series, color='tab:red')
    ax1.set_xlabel('Iteration')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Funding Cost  $c_t$')
    ax2.plot(logger.c_series, color='tab:blue')
    ax2.set_xlabel('Iteration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_fx_price(logger: 'MetricsLogger', rolling: int = 1,
                  figsize=(12, 4),
                  shock_iter: int = None):
    fig, ax = plt.subplots(figsize=figsize)
    title = 'FX Mid-Price (CLOB)'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    title += '  +  Raw Fair Price'
    ax.set_title(title)
    s = logger.clob_mid_series
    if rolling > 1:
        s = _rolling_avg(s, rolling)
    ax.plot(s, color='black', linewidth=1.25,
            label=f'CLOB mid [MA {rolling}]' if rolling > 1 else 'CLOB mid')
    fair = [
        value if value is not None and math.isfinite(value) else float('nan')
        for value in logger.fair_price_series
    ]
    if any(math.isfinite(value) for value in fair):
        ax.plot(fair, color='#c44e52', linewidth=1.0, ls='--', alpha=0.9,
                label='Fair price (raw)')
    if shock_iter is not None:
        ax.axvline(shock_iter, color='#8b0000', ls='--', lw=1.2, alpha=0.85,
                   label=f'Shock t={shock_iter}')
    _shade_stress(ax, logger)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Price')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clob_spread(logger: 'MetricsLogger', rolling: int = 5,
                     figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    title = 'CLOB Quoted Spread'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spread (bps)')

    s = _rolling_avg(logger.clob_qspr, rolling)
    ax.plot(s, color='black', linewidth=1)
    _shade_stress(ax, logger)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_amm_reserves(logger: 'MetricsLogger', pool_name: str = 'cpmm',
                      figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.set_title(f'{pool_name.upper()} Reserves')
    ax1.plot(logger.amm_x_series.get(pool_name, []), label='x (base)')
    ax1.set_ylabel('Base reserves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(logger.amm_y_series.get(pool_name, []), label='y (quote)',
             color='orange')
    ax2.set_ylabel('Quote reserves')
    ax2.set_xlabel('Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _shade_stress(ax1, logger)
    _shade_stress(ax2, logger)
    plt.tight_layout()
    plt.show()


def plot_volume_slippage_profile(logger: 'MetricsLogger',
                                 pool_name: str = 'cpmm',
                                 figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Volume-Slippage Profile — {pool_name.upper()}')
    ax.set_xlabel('Slippage threshold (bps)')
    ax.set_ylabel('Max trade size  $Q$')

    if pool_name not in logger.amm_vol_slip:
        ax.text(0.5, 0.5, f'No {pool_name.upper()} pool in this run',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='grey')
        plt.tight_layout()
        plt.show()
        return

    thresholds = logger.slippage_thresholds
    avg_Qs = []
    for th in thresholds:
        s = logger.amm_vol_slip[pool_name][th]
        avg_Qs.append(sum(s) / len(s) if s else 0)

    ax.plot(thresholds, avg_Qs, 'o-', linewidth=2, markersize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===================================================================
#  Helpers
# ===================================================================

def _avg_finite(curves: dict, Q_grid: list) -> list:
    """Average finite values from {Q: [series]} dict."""
    means = []
    for Q in Q_grid:
        s = curves.get(Q, [])
        finite = [x for x in s if math.isfinite(x)]
        means.append(sum(finite) / len(finite) if finite else float('nan'))
    return means


def _rolling_avg(s: list, w: int) -> list:
    """Simple rolling average."""
    if w <= 1:
        return list(s)
    return [sum(s[max(0, i - w + 1):i + 1]) / min(w, i + 1)
            for i in range(len(s))]


def _shade_stress(ax, logger: 'MetricsLogger', color='red', alpha=0.08):
    """Shade stress-regime intervals on an axis."""
    regimes = logger.regime_series
    in_stress = False
    start = 0
    for i, r in enumerate(regimes):
        if r == 'stress' and not in_stress:
            start = i
            in_stress = True
        elif r != 'stress' and in_stress:
            ax.axvspan(start, i, color=color, alpha=alpha)
            in_stress = False
    if in_stress:
        ax.axvspan(start, len(regimes), color=color, alpha=alpha)
