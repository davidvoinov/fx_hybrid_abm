"""
dashboards.py — Multi-panel dashboard figures for the FX ABM.

Each function produces a single high-resolution figure combining several
related plots. All dashboards are saved to *out_dir* as PNG files.

    dashboard_context()     — environment + price + spread overview
    dashboard_h1()          — H1: cost comparison & AMM adds liquidity
    dashboard_h2()          — H2: systemic linkage & liquidity flight
    generate_all_dashboards() — convenience wrapper
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if TYPE_CHECKING:
    from AgentBasedModel.metrics.logger import MetricsLogger

from AgentBasedModel.visualization.venue_plots import (
    _avg_finite, _rolling_avg, _shade_stress,
)
from AgentBasedModel.visualization import venue_plots as _vp

# Shared palette
_C = {
    'clob': '#1f77b4',
    'cpmm': '#ff7f0e',
    'hfmm': '#2ca02c',
    'stress': '#c44e52',
    'normal': '#4c72b0',
    'slip': '#4c72b0',
    'fee': '#dd8452',
}

DPI = 200


# ===================================================================
#  Dashboard 0 — Context overview
# ===================================================================

def dashboard_context(logger: 'MetricsLogger',
                                            out_dir: str = 'output',
                                            rolling: int = 5,
                                            shock_iter: Optional[int] = None) -> str:
    """
    3-panel context dashboard:
      top-left:  σ_t and c_t  (twin-axis)
            top-right: FX mid-price + raw fair price
      bottom:    CLOB quoted spread
    """
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    fig.suptitle('Market Context Dashboard', fontsize=16, fontweight='bold',
                 y=0.97)

    # ── σ_t / c_t ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Volatility  $\\sigma_t$  &  Funding Cost  $c_t$')
    l1 = ax1.plot(logger.sigma_series, color='tab:red', label='$\\sigma_t$',
                  linewidth=1.2)
    ax1.set_ylabel('$\\sigma_t$', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1b = ax1.twinx()
    l2 = ax1b.plot(logger.c_series, color='tab:blue', label='$c_t$',
                   linewidth=1.2)
    ax1b.set_ylabel('$c_t$', color='tab:blue')
    ax1b.tick_params(axis='y', labelcolor='tab:blue')
    lines = l1 + l2
    if shock_iter is not None:
        shock_line = ax1.axvline(shock_iter, color='#8b0000', ls='--',
                                 lw=1.2, alpha=0.85,
                                 label=f'Shock t={shock_iter}')
        lines = lines + [shock_line]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=10,
               loc='upper left')
    ax1.set_xlabel('Iteration')
    _shade_stress(ax1, logger)
    ax1.grid(True, alpha=0.25)

    # ── FX Price ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f'FX Mid-Price (CLOB)  [MA {rolling}]  +  Raw Fair Price')
    s = _rolling_avg(logger.clob_mid_series, rolling)
    ax2.plot(s, color='black', linewidth=1.35,
             label=f'CLOB mid [MA {rolling}]')
    fair = [
        value if value is not None and math.isfinite(value) else float('nan')
        for value in logger.fair_price_series
    ]
    if any(math.isfinite(value) for value in fair):
        ax2.plot(fair, color=_C['stress'], linewidth=1.0, ls='--', alpha=0.9,
                 label='Fair price (raw)')
    if shock_iter is not None:
        ax2.axvline(shock_iter, color='#8b0000', ls='--', lw=1.2, alpha=0.85,
                    label=f'Shock t={shock_iter}')
    _shade_stress(ax2, logger)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Price')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.25)

    # ── CLOB Spread ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title(f'CLOB Quoted Spread  [MA {rolling}]')
    s = _rolling_avg(logger.clob_qspr, rolling)
    ax3.plot(s, color='black', linewidth=1)
    if shock_iter is not None:
        ax3.axvline(shock_iter, color='#8b0000', ls='--', lw=1.2,
                    alpha=0.85)
    _shade_stress(ax3, logger)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Spread (bps)')
    ax3.grid(True, alpha=0.25)

    path = os.path.join(out_dir, 'dashboard_context.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Dashboard 1 — H1: cost comparison & AMM adds liquidity
# ===================================================================

def dashboard_h1(logger: 'MetricsLogger',
                 out_dir: str = 'output',
                 Q: float = 5,
                 rolling: int = 10,
                 shock_iter: Optional[int] = None) -> str:
    """
    4-panel H1 dashboard:
      top-left:   C_v(Q) cost curves
      top-right:  cost decomposition (slippage vs fee)
      bottom-left:  total market depth (CLOB + AMM stacked)
      bottom-right: summary KPI text
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    fig.suptitle('H1 · Execution Cost Comparison  &  AMM Adds Liquidity',
                 fontsize=16, fontweight='bold', y=0.97)

    Q_grid = logger.Q_grid

    # ── Cost curves ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Average Execution Cost  $C_v(Q)$')
    means = _avg_finite(logger.clob_cost_curves, Q_grid)
    ax1.plot(Q_grid, means, 'o-', label='CLOB', lw=2, color=_C['clob'])
    for name in logger.amm_cost_curves:
        m = _avg_finite(logger.amm_cost_curves[name], Q_grid)
        ax1.plot(Q_grid, m, 's--', label=name.upper(), lw=2,
                 color=_C.get(name))
    ax1.set_xlabel('Trade Size  $Q$')
    ax1.set_ylabel('Cost (bps)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)

    # ── Cost decomposition ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f'Cost Decomposition  (Q={Q})')

    venues, slippages, fees = [], [], []
    s = logger.clob_cost_curves.get(Q, [])
    fs = [x for x in s if math.isfinite(x)]
    if fs:
        venues.append('CLOB')
        slippages.append(sum(fs) / len(fs))
        fees.append(0)
    for name in logger.amm_slippage_curves:
        sl = logger.amm_slippage_curves[name].get(Q, [])
        fe = logger.amm_fee_curves[name].get(Q, [])
        fsl = [x for x in sl if math.isfinite(x)]
        ffe = [x for x in fe if math.isfinite(x)]
        if fsl:
            venues.append(name.upper())
            slippages.append(sum(fsl) / len(fsl))
            fees.append(sum(ffe) / len(ffe) if ffe else 0)

    xp = list(range(len(venues)))
    ax2.bar(xp, slippages, label='Slippage', alpha=0.85, color=_C['slip'])
    ax2.bar(xp, fees, bottom=slippages, label='LP Fee', alpha=0.85,
            color=_C['fee'])
    ax2.set_xticks(xp)
    ax2.set_xticklabels(venues)
    ax2.set_ylabel('Cost (bps)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25, axis='y')

    # ── Total market depth ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title(f'Total Market Depth: CLOB + AMM  [MA {rolling}]')
    n = len(logger.iterations)
    clob_d = []
    for d in logger.clob_depth:
        clob_d.append((d.get('bid', 0) + d.get('ask', 0))
                      if isinstance(d, dict) else 0)
    clob_r = _rolling_avg(clob_d, rolling)
    ax3.fill_between(range(n), 0, clob_r, alpha=0.4, label='CLOB',
                     color=_C['clob'])
    amm_total = [0.0] * n
    for name in logger.amm_depth_series:
        v = logger.amm_depth_series[name][:n]
        while len(v) < n:
            v.append(0)
        amm_total = [a + b for a, b in zip(amm_total, v)]
    amm_r = _rolling_avg(amm_total, rolling)
    combined = [c + a for c, a in zip(clob_r, amm_r)]
    ax3.fill_between(range(n), clob_r, combined, alpha=0.4,
                     label='AMM (added)', color=_C['hfmm'])
    _shade_stress(ax3, logger)
    ax3.legend(fontsize=10)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Depth (base units)')
    ax3.grid(True, alpha=0.25)

    # ── KPI summary panel ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary = logger.summary()
    venues_all = ['clob'] + list(logger.amm_cost_curves.keys())

    lines = []
    lines.append(f'Iterations: {summary.get("n_iterations", 0)}')
    lines.append(f'Total trades: {summary.get("n_trades", 0)}')
    lines.append('')
    lines.append('Avg Cost (bps):')
    hdr = f'  {"Q":>4s}'
    for v in venues_all:
        hdr += f'  {v.upper():>7s}'
    lines.append(hdr)
    for q in Q_grid:
        row = f'  {q:>4.0f}'
        for v in venues_all:
            val = summary.get(f'avg_cost_{v}_Q{q}', float('nan'))
            row += f'  {val:>7.1f}' if math.isfinite(val) else f'  {"N/A":>7s}'
        lines.append(row)
    lines.append('')
    lines.append('Flow Share:')
    amm_share = 0.0
    for v in venues_all:
        fs = summary.get(f'avg_flow_share_{v}', 0)
        lines.append(f'  {v.upper():>7s}: {fs:.1%}')
        if v != 'clob':
            amm_share += fs
    lines.append(f'\n→ AMM captures {amm_share:.1%} of volume')

    ax4.text(0.05, 0.95, '\n'.join(lines), transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                       alpha=0.8))

    path = os.path.join(out_dir, 'dashboard_h1.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Dashboard 2 — H2: systemic linkage & liquidity flight
# ===================================================================

def dashboard_h2(logger: 'MetricsLogger',
                 out_dir: str = 'output',
                 Q: float = 5,
                 rolling: int = 10,
                 corr_window: int = 30,
                 stress_start: Optional[int] = None,
                 shock_iter: Optional[int] = None) -> str:
    """
    6-panel H2 dashboard:
      row 0: cost time series (twin-axis)  |  CLOB spread vs AMM cost
      row 1: flow allocation   |  rolling correlation
      row 2: AMM liquidity     |  stress flow migration bars
    """
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    fig.suptitle('H2 · Systemic Linkage  &  Liquidity Flight Under Stress',
                 fontsize=16, fontweight='bold', y=0.97)

    def _roll(s, w=rolling):
        if w <= 1:
            return list(range(len(s))), s
        out = []
        for i in range(len(s) - w + 1):
            window = [x for x in s[i:i + w] if math.isfinite(x)]
            out.append(sum(window) / len(window) if window else float('nan'))
        return list(range(w - 1, len(s))), out

    # ── (0,0) Cost time series (twin-axis) ──────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f'Execution Cost Time Series  (Q={Q}, MA {rolling})')
    xs, ys = _roll(logger.cost_series('clob', Q))
    ax.plot(xs, ys, label='CLOB', lw=1.5, color=_C['clob'])
    ax.set_ylabel('CLOB Cost (bps)', color=_C['clob'])
    ax.tick_params(axis='y', labelcolor=_C['clob'])
    axr = ax.twinx()
    for name in logger.amm_cost_curves:
        xs2, ys2 = _roll(logger.cost_series(name, Q))
        axr.plot(xs2, ys2, label=name.upper(), lw=1.3, ls='--',
                 color=_C.get(name), alpha=0.7)
    axr.set_ylabel('AMM Cost (bps)')
    if shock_iter is not None:
        ax.axvline(shock_iter, color='red', ls='--', lw=1.2, alpha=0.7,
                   label=f'Shock t={shock_iter}')
    _shade_stress(ax, logger)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper left')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.25)

    # ── (0,1) CLOB spread vs AMM cost ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title(f'CLOB Spread vs AMM Cost  (Q={Q}, MA {rolling})')
    qspr = _rolling_avg(logger.clob_qspr, rolling)
    ax1.plot(qspr, label='CLOB qspr', color=_C['clob'], lw=1.3)
    ax1.set_ylabel('CLOB Spread (bps)', color=_C['clob'])
    ax1.tick_params(axis='y', labelcolor=_C['clob'])
    ax2 = ax1.twinx()
    for name in logger.amm_cost_curves:
        s = logger.cost_series(name, Q)
        sr = _rolling_avg(s, rolling)
        ax2.plot(sr, label=f'{name.upper()} cost', color=_C.get(name),
                 lw=1.3, ls='--')
    ax2.set_ylabel('AMM Cost (bps)')
    _shade_stress(ax1, logger)
    lines1, lb1 = ax1.get_legend_handles_labels()
    lines2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lb1 + lb2, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.25)

    # ── (1,0) Flow allocation ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(f'Customer Venue Use per Clock Tick (quiet = 0)  [MA {rolling}]')
    venues = list(logger.flow_volume.keys())
    n = len(logger.iterations)
    xs = list(range(n))
    bottoms = [0.0] * n
    for v in venues:
        vals = _rolling_avg(logger.flow_share(v), rolling)
        tops = [b + vi for b, vi in zip(bottoms, vals)]
        ax.fill_between(xs, bottoms, tops, label=v.upper(), alpha=0.6,
                         color=_C.get(v))
        bottoms = tops
    _shade_stress(ax, logger)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Customer Share per Tick')
    ax.grid(True, alpha=0.25)

    # ── (1,1) Rolling correlation ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title(f'Rolling Cost Correlation  (Q={Q}, w={corr_window})')
    clob_s = logger.cost_series('clob', Q)
    for name in logger.amm_cost_curves:
        amm_s = logger.cost_series(name, Q)
        nm = min(len(clob_s), len(amm_s))
        if nm < corr_window:
            continue
        rhos = []
        for i in range(corr_window, nm):
            w1 = clob_s[i - corr_window:i]
            w2 = amm_s[i - corr_window:i]
            pairs = [(a, b) for a, b in zip(w1, w2)
                     if math.isfinite(a) and math.isfinite(b)]
            if len(pairs) < 5:
                rhos.append(float('nan'))
                continue
            xv, yv = zip(*pairs)
            nn = len(xv)
            mx, my = sum(xv) / nn, sum(yv) / nn
            cov = sum((x - mx) * (y - my) for x, y in zip(xv, yv)) / nn
            sx = (sum((x - mx) ** 2 for x in xv) / nn) ** 0.5
            sy = (sum((y - my) ** 2 for y in yv) / nn) ** 0.5
            rhos.append(cov / (sx * sy) if sx > 0 and sy > 0 else 0)
        ax.plot(range(corr_window, nm), rhos, label=f'CLOB ↔ {name.upper()}',
                lw=1.3, color=_C.get(name))
    _shade_stress(ax, logger)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pearson ρ')
    ax.grid(True, alpha=0.25)

    # ── (2,0) AMM liquidity ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('AMM Liquidity  $L_t$  Under Stress')
    for name in logger.amm_L_series:
        ax.plot(logger.amm_L_series[name], label=name.upper(), lw=1.5,
                color=_C.get(name))
    _shade_stress(ax, logger)
    ax.legend(fontsize=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$L_t$')
    ax.grid(True, alpha=0.25)

    # ── (2,1) Stress flow migration ──────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('AMM Customer Volume Share (Ratio of Sums)')
    amm_venues = [v for v in logger.flow_volume if v != 'clob']
    idx = min(stress_start, n) if stress_start is not None else n
    before_sh, stress_sh = {}, {}
    for v in amm_venues:
        before_sh[v] = logger.customer_volume_share(v, 0, idx)
        stress_sh[v] = logger.customer_volume_share(v, idx, n)

    xp = list(range(len(amm_venues)))
    w = 0.35
    b1 = ax.bar([x - w / 2 for x in xp],
                [before_sh[v] for v in amm_venues],
                w, label='Normal', alpha=0.8, color=_C['normal'])
    b2 = ax.bar([x + w / 2 for x in xp],
                [stress_sh[v] for v in amm_venues],
                w, label='Stress', alpha=0.8, color=_C['stress'])
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f'{h:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    ha='center', va='bottom', fontsize=10)
    ax.set_xticks(xp)
    ax.set_xticklabels([v.upper() for v in amm_venues])
    ax.set_ylabel('Share of Total Volume')
    mx = max(max(before_sh.values(), default=0),
             max(stress_sh.values(), default=0))
    ax.set_ylim(0, mx * 1.45 if mx > 0 else 0.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis='y')

    # ── KPI text annotation ──────────────────────────────────────────
    kpi_lines = []
    for name in logger.amm_cost_curves:
        rho = logger.cost_correlation('clob', name, Q=Q)
        if stress_start is not None:
            ba = logger.commonality_before_after('clob', name, Q=Q,
                                                  stress_start=stress_start)
            b_s = f'{ba["before"]:.3f}' if math.isfinite(ba['before']) else 'N/A'
            a_s = f'{ba["after"]:.3f}' if math.isfinite(ba['after']) else 'N/A'
            kpi_lines.append(f'CLOB↔{name.upper()}: ρ={rho:.3f}  '
                             f'(before={b_s}, after={a_s})')
        else:
            kpi_lines.append(f'CLOB↔{name.upper()}: ρ={rho:.3f}')
    # CLOB spread normal vs stress
    qspr_raw = logger.clob_qspr
    if stress_start is not None:
        n_s = [s for s in qspr_raw[:idx] if math.isfinite(s)]
        s_s = [s for s in qspr_raw[idx:] if math.isfinite(s)]
        avg_n = sum(n_s) / len(n_s) if n_s else 0
        avg_s = sum(s_s) / len(s_s) if s_s else 0
        mult = f'{avg_s / avg_n:.1f}×' if avg_n > 0 else '—'
        kpi_lines.append(f'CLOB spread: {avg_n:.1f} → {avg_s:.1f} bps ({mult})')
    else:
        all_s = [s for s in qspr_raw if math.isfinite(s)]
        avg_all = sum(all_s) / len(all_s) if all_s else 0
        kpi_lines.append(f'CLOB spread: {avg_all:.1f} bps (avg)')

    fig.text(0.50, 0.01, '   |   '.join(kpi_lines),
             ha='center', va='bottom', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                       alpha=0.9))

    path = os.path.join(out_dir, 'dashboard_h2.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Dashboard 3 — Comparison: With AMM vs Without AMM
# ===================================================================

def _rolling_return_vol(mid_series, window=20):
    """Rolling standard deviation of log-returns."""
    import math as _m
    returns = []
    for i in range(1, len(mid_series)):
        p0, p1 = mid_series[i - 1], mid_series[i]
        if p0 > 0 and p1 > 0:
            returns.append(_m.log(p1 / p0))
        else:
            returns.append(0.0)
    vol = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns[start:i + 1]
        if len(chunk) < 2:
            vol.append(0.0)
        else:
            mu = sum(chunk) / len(chunk)
            var = sum((x - mu) ** 2 for x in chunk) / (len(chunk) - 1)
            vol.append(var ** 0.5)
    # Pad to same length as mid_series
    return [0.0] + vol


def _window_avg(values, start: int, end: int) -> float:
    finite = [value for value in values[start:end] if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float('nan')


def _window_median(values, start: int, end: int) -> float:
    finite = sorted(value for value in values[start:end] if math.isfinite(value))
    if not finite:
        return float('nan')
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _resolve_stress_end(logger: 'MetricsLogger', stress_start: int, n: int) -> int:
    regimes = list(getattr(logger, 'regime_series', []) or [])
    if len(regimes) >= n:
        end = stress_start
        while end < n and regimes[end] == 'stress':
            end += 1
        if end > stress_start:
            return end
    return min(n, stress_start + 100)


def _event_windows(logger: 'MetricsLogger',
                   n: int,
                   *,
                   shock_iter: Optional[int] = None,
                   stress_start: Optional[int] = None) -> dict:
    if shock_iter is not None:
        event_start = max(0, min(shock_iter, n))
        event_end = min(n, event_start + 20)
        return {
            'mode': 'shock',
            'has_event': event_start < n,
            'title': f'Local Comparison Around Shock t={event_start}',
            'subtitle': (
                f'Full pre=[0,{event_start})   '
                f'Local pre=[{max(0, event_start - 50)},{event_start})   '
                f'Shock=[{event_start},{event_end})'
            ),
            'event_label': 'Shock/Stress',
            'event_row_label': 'shock',
            'full_pre': (0, event_start),
            'local_pre': (max(0, event_start - 50), event_start),
            'event': (event_start, event_end),
        }

    if stress_start is not None:
        event_start = max(0, min(stress_start, n))
        event_end = _resolve_stress_end(logger, event_start, n)
        return {
            'mode': 'stress',
            'has_event': event_start < n,
            'title': f'Comparison Around Stress t={event_start}',
            'subtitle': (
                f'Full pre=[0,{event_start})   '
                f'Local pre=[{max(0, event_start - 50)},{event_start})   '
                f'Stress=[{event_start},{event_end})'
            ),
            'event_label': 'Shock/Stress',
            'event_row_label': 'stress',
            'full_pre': (0, event_start),
            'local_pre': (max(0, event_start - 50), event_start),
            'event': (event_start, event_end),
        }

    return {
        'mode': 'baseline',
        'has_event': False,
        'title': 'Market Quality Comparison',
        'subtitle': '=' * 42,
        'event_label': 'Shock/Stress',
        'event_row_label': 'event',
        'full_pre': (0, n),
        'local_pre': (0, n),
        'event': (0, n),
    }


def _fmt_metric(value: float, digits: int = 1) -> str:
    return f'{value:.{digits}f}' if math.isfinite(value) else 'N/A'


def dashboard_comparison(logger_amm: 'MetricsLogger',
                         logger_no_amm: 'MetricsLogger',
                         out_dir: str = 'output',
                         rolling: int = 10,
                         vol_window: int = 20,
                         stress_start: Optional[int] = None,
                         shock_iter: Optional[int] = None) -> str:
    """
    4-panel comparison dashboard — With AMM vs Without AMM:
      (0,0): CLOB Depth [MA]
      (0,1): Rolling Return Volatility
      (1,0): CLOB Quoted Spread [MA]
      (1,1): Market Quality Comparison table
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle('Comparison  ·  Market Quality With vs Without AMM',
                 fontsize=15, fontweight='bold', y=0.97)

    n_amm = len(logger_amm.iterations)
    n_no = len(logger_no_amm.iterations)
    n = min(n_amm, n_no)

    c_amm = '#2ca02c'    # green solid
    c_no = '#1f77b4'     # blue dashed

    def _mean(lst):
        f = [x for x in lst if math.isfinite(x)]
        return sum(f) / len(f) if f else float('nan')

    def _annotate_event(ax):
        if stress_start is not None:
            ax.axvspan(stress_start, min(350, n), alpha=0.08, color='red')
        if shock_iter is not None:
            ax.axvline(shock_iter, color='#8b0000', ls='--', lw=1.2,
                       alpha=0.85, label=f'Shock t={shock_iter}')

    # ── (0,0) CLOB Depth ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f'CLOB Depth  [MA {rolling}]')

    def _total_depth(logger):
        return [(d.get('bid', 0) + d.get('ask', 0)) if isinstance(d, dict) else 0
                for d in logger.clob_depth]

    d_amm = _rolling_avg(_total_depth(logger_amm)[:n], rolling)
    d_no = _rolling_avg(_total_depth(logger_no_amm)[:n], rolling)
    ax.plot(d_amm, color=c_amm, lw=1.5, label='With AMM')
    ax.plot(d_no, color=c_no, ls='--', lw=1.5, label='Without AMM')
    _annotate_event(ax)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Depth (bid + ask)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── (0,1) Rolling Return Volatility ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f'Rolling Return Volatility  (window={vol_window})')

    vol_amm = _rolling_return_vol(logger_amm.clob_mid_series[:n], vol_window)
    vol_no = _rolling_return_vol(logger_no_amm.clob_mid_series[:n], vol_window)
    ax.plot(vol_amm, color=c_amm, lw=1.5, label='With AMM')
    ax.plot(vol_no, color=c_no, ls='--', lw=1.5, label='Without AMM')
    _annotate_event(ax)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('σ(return)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── (1,0) CLOB Quoted Spread ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(f'CLOB Quoted Spread  [MA {rolling}]')

    s_amm = _rolling_avg(logger_amm.clob_qspr[:n], rolling)
    s_no = _rolling_avg(logger_no_amm.clob_qspr[:n], rolling)
    ax.plot(s_amm, color=c_amm, lw=1.5, label='With AMM')
    ax.plot(s_no, color=c_no, ls='--', lw=1.5, label='Without AMM')
    _annotate_event(ax)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spread (bps)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── (1,1) Market Quality Comparison Table ────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    depth_amm = _total_depth(logger_amm)
    depth_no = _total_depth(logger_no_amm)
    vol_amm_all = _rolling_return_vol(logger_amm.clob_mid_series, vol_window)
    vol_no_all = _rolling_return_vol(logger_no_amm.clob_mid_series, vol_window)
    event = _event_windows(logger_amm, n, shock_iter=shock_iter,
                           stress_start=stress_start)

    def _delta(a, b):
        if b == 0 or not math.isfinite(a) or not math.isfinite(b):
            return 'N/A'
        return f'{(a - b) / abs(b):+.1%}'

    if event['has_event']:
        full_pre = event['full_pre']
        local_pre = event['local_pre']
        event_window = event['event']
        row_suffix = event['event_row_label']
        rows = [
            ('Spread full pre (bps)',
             _fmt_metric(_window_avg(logger_amm.clob_qspr, *full_pre)),
             _fmt_metric(_window_avg(logger_no_amm.clob_qspr, *full_pre)),
             _delta(_window_avg(logger_amm.clob_qspr, *full_pre),
                _window_avg(logger_no_amm.clob_qspr, *full_pre))),
            ('Spread local pre (bps)',
             _fmt_metric(_window_avg(logger_amm.clob_qspr, *local_pre)),
             _fmt_metric(_window_avg(logger_no_amm.clob_qspr, *local_pre)),
             _delta(_window_avg(logger_amm.clob_qspr, *local_pre),
                _window_avg(logger_no_amm.clob_qspr, *local_pre))),
            (f'Spread {row_suffix} (bps)',
             _fmt_metric(_window_avg(logger_amm.clob_qspr, *event_window)),
             _fmt_metric(_window_avg(logger_no_amm.clob_qspr, *event_window)),
             _delta(_window_avg(logger_amm.clob_qspr, *event_window),
                _window_avg(logger_no_amm.clob_qspr, *event_window))),
            ('Depth full pre',
             _fmt_metric(_window_avg(depth_amm, *full_pre), digits=0),
             _fmt_metric(_window_avg(depth_no, *full_pre), digits=0),
             _delta(_window_avg(depth_amm, *full_pre),
                _window_avg(depth_no, *full_pre))),
            ('Depth local pre',
             _fmt_metric(_window_avg(depth_amm, *local_pre), digits=0),
             _fmt_metric(_window_avg(depth_no, *local_pre), digits=0),
             _delta(_window_avg(depth_amm, *local_pre),
                _window_avg(depth_no, *local_pre))),
            (f'Depth {row_suffix}',
             _fmt_metric(_window_avg(depth_amm, *event_window), digits=0),
             _fmt_metric(_window_avg(depth_no, *event_window), digits=0),
             _delta(_window_avg(depth_amm, *event_window),
                _window_avg(depth_no, *event_window))),
            ('Vol full pre',
             _fmt_metric(_window_avg(vol_amm_all, *full_pre), digits=4),
             _fmt_metric(_window_avg(vol_no_all, *full_pre), digits=4),
             _delta(_window_avg(vol_amm_all, *full_pre),
                _window_avg(vol_no_all, *full_pre))),
            ('Vol local pre',
             _fmt_metric(_window_avg(vol_amm_all, *local_pre), digits=4),
             _fmt_metric(_window_avg(vol_no_all, *local_pre), digits=4),
             _delta(_window_avg(vol_amm_all, *local_pre),
                _window_avg(vol_no_all, *local_pre))),
            (f'Vol {row_suffix}',
             _fmt_metric(_window_avg(vol_amm_all, *event_window), digits=4),
             _fmt_metric(_window_avg(vol_no_all, *event_window), digits=4),
             _delta(_window_avg(vol_amm_all, *event_window),
                _window_avg(vol_no_all, *event_window))),
        ]
        table_title    = event['title']
        table_subtitle = event['subtitle']
    else:
        avg_spr_amm = _mean(logger_amm.clob_qspr)
        avg_spr_no = _mean(logger_no_amm.clob_qspr)
        avg_dep_amm = _mean(depth_amm)
        avg_dep_no = _mean(depth_no)
        avg_vol_amm = _mean(vol_amm_all)
        avg_vol_no = _mean(vol_no_all)
        rows = [
            ('Avg Spread (bps)', f'{avg_spr_amm:.1f}', f'{avg_spr_no:.1f}',
             _delta(avg_spr_amm, avg_spr_no)),
            ('Avg CLOB Depth', f'{avg_dep_amm:.0f}', f'{avg_dep_no:.0f}',
             _delta(avg_dep_amm, avg_dep_no)),
            ('Avg Volatility', f'{avg_vol_amm:.4f}', f'{avg_vol_no:.4f}',
             _delta(avg_vol_amm, avg_vol_no)),
        ]
        table_title    = 'Market Quality Comparison'
        table_subtitle = ''

    # Title block on top of the panel (does not overlap the table).
    ax.text(0.5, 0.98, table_title, ha='center', va='top',
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    if table_subtitle:
        ax.text(0.5, 0.93, table_subtitle, ha='center', va='top',
                fontsize=8, fontfamily='monospace', color='#555',
                transform=ax.transAxes)

    # Table occupies the lower part of the axes — explicit bbox so it never
    # overlaps the title; wider Metric column, narrower numeric columns.
    bbox_top = 0.85 if table_subtitle else 0.90
    col_labels = ['Metric', 'AMM', 'No AMM', 'Δ']
    col_widths = [0.46, 0.16, 0.16, 0.22]
    tbl = ax.table(
        cellText=[[r[0], r[1], r[2], r[3]] for r in rows],
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, bbox_top],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10 if len(rows) > 4 else 11)
    # Left-align the first column (metric labels) for readability
    for i in range(0, len(rows) + 1):
        tbl[i, 0].set_text_props(ha='left')
        tbl[i, 0].PAD = 0.04
    # Header styling
    for j in range(4):
        tbl[0, j].set_facecolor('#37474f')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
        tbl[0, j].set_height(tbl[0, j].get_height() * 1.1)
    # Alt-row shading
    for i in range(1, len(rows) + 1):
        color = '#f5f7fa' if i % 2 == 0 else 'white'
        for j in range(4):
            tbl[i, j].set_facecolor(color)

    path = os.path.join(out_dir, 'dashboard_comparison.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Individual comparison plots (each panel from dashboard_comparison)
# ===================================================================

def save_comparison_individual(logger_amm: 'MetricsLogger',
                               logger_no_amm: 'MetricsLogger',
                               out_dir: str = 'output',
                               rolling: int = 10,
                               vol_window: int = 20,
                               stress_start: Optional[int] = None,
                               shock_iter: Optional[int] = None) -> list:
    """Save each panel of dashboard_comparison as a separate PNG."""
    os.makedirs(out_dir, exist_ok=True)
    paths: list = []

    n_amm = len(logger_amm.iterations)
    n_no = len(logger_no_amm.iterations)
    n = min(n_amm, n_no)

    c_amm = '#2ca02c'
    c_no = '#1f77b4'

    def _mean(lst):
        f = [x for x in lst if math.isfinite(x)]
        return sum(f) / len(f) if f else float('nan')

    def _total_depth(logger):
        return [(d.get('bid', 0) + d.get('ask', 0)) if isinstance(d, dict) else 0
                for d in logger.clob_depth]

    def _delta(a, b):
        if b == 0 or not math.isfinite(a) or not math.isfinite(b):
            return 'N/A'
        return f'{(a - b) / abs(b):+.1%}'

    def _annotate_event(ax):
        if stress_start is not None:
            ax.axvspan(stress_start, min(350, n), alpha=0.08, color='red')
        if shock_iter is not None:
            ax.axvline(shock_iter, color='#8b0000', ls='--', lw=1.2,
                       alpha=0.85, label=f'Shock t={shock_iter}')

    # ── 1. CLOB Depth ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'CLOB Depth  [MA {rolling}]', fontsize=13)
    d_amm = _rolling_avg(_total_depth(logger_amm)[:n], rolling)
    d_no = _rolling_avg(_total_depth(logger_no_amm)[:n], rolling)
    ax.plot(d_amm, color=c_amm, lw=1.5, label='With AMM')
    ax.plot(d_no, color=c_no, ls='--', lw=1.5, label='Without AMM')
    _annotate_event(ax)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Depth (bid + ask)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'cmp_clob_depth.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 2. Rolling Return Volatility ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'Rolling Return Volatility  (window={vol_window})', fontsize=13)
    vol_amm = _rolling_return_vol(logger_amm.clob_mid_series[:n], vol_window)
    vol_no = _rolling_return_vol(logger_no_amm.clob_mid_series[:n], vol_window)
    ax.plot(vol_amm, color=c_amm, lw=1.5, label='With AMM')
    ax.plot(vol_no, color=c_no, ls='--', lw=1.5, label='Without AMM')
    _annotate_event(ax)
    ax.set_xlabel('Iteration'); ax.set_ylabel('σ(return)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'cmp_rolling_volatility.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 3. CLOB Quoted Spread ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'CLOB Quoted Spread  [MA {rolling}]', fontsize=13)
    s_amm = _rolling_avg(logger_amm.clob_qspr[:n], rolling)
    s_no = _rolling_avg(logger_no_amm.clob_qspr[:n], rolling)
    ax.plot(s_amm, color=c_amm, lw=1.5, label='With AMM')
    ax.plot(s_no, color=c_no, ls='--', lw=1.5, label='Without AMM')
    _annotate_event(ax)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Spread (bps)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'cmp_quoted_spread.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 4. Market Quality Comparison Table ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('Market Quality Comparison: With vs Without AMM',
                 fontsize=14, fontweight='bold', pad=12)

    depth_amm = _total_depth(logger_amm)
    depth_no = _total_depth(logger_no_amm)
    vol_amm_all = _rolling_return_vol(logger_amm.clob_mid_series, vol_window)
    vol_no_all = _rolling_return_vol(logger_no_amm.clob_mid_series, vol_window)
    event = _event_windows(logger_amm, n, shock_iter=shock_iter,
                           stress_start=stress_start)

    if event['has_event']:
        full_pre = event['full_pre']
        local_pre = event['local_pre']
        event_window = event['event']
        row_suffix = event['event_row_label']
        rows = [
            ('Spread full pre (bps)',
             _fmt_metric(_window_avg(logger_amm.clob_qspr, *full_pre)),
             _fmt_metric(_window_avg(logger_no_amm.clob_qspr, *full_pre)),
             _delta(_window_avg(logger_amm.clob_qspr, *full_pre),
                    _window_avg(logger_no_amm.clob_qspr, *full_pre))),
            ('Spread local pre (bps)',
             _fmt_metric(_window_avg(logger_amm.clob_qspr, *local_pre)),
             _fmt_metric(_window_avg(logger_no_amm.clob_qspr, *local_pre)),
             _delta(_window_avg(logger_amm.clob_qspr, *local_pre),
                    _window_avg(logger_no_amm.clob_qspr, *local_pre))),
            (f'Spread {row_suffix} (bps)',
             _fmt_metric(_window_avg(logger_amm.clob_qspr, *event_window)),
             _fmt_metric(_window_avg(logger_no_amm.clob_qspr, *event_window)),
             _delta(_window_avg(logger_amm.clob_qspr, *event_window),
                    _window_avg(logger_no_amm.clob_qspr, *event_window))),
            ('Depth full pre',
             _fmt_metric(_window_avg(depth_amm, *full_pre), digits=0),
             _fmt_metric(_window_avg(depth_no, *full_pre), digits=0),
             _delta(_window_avg(depth_amm, *full_pre),
                    _window_avg(depth_no, *full_pre))),
            ('Depth local pre',
             _fmt_metric(_window_avg(depth_amm, *local_pre), digits=0),
             _fmt_metric(_window_avg(depth_no, *local_pre), digits=0),
             _delta(_window_avg(depth_amm, *local_pre),
                    _window_avg(depth_no, *local_pre))),
            (f'Depth {row_suffix}',
             _fmt_metric(_window_avg(depth_amm, *event_window), digits=0),
             _fmt_metric(_window_avg(depth_no, *event_window), digits=0),
             _delta(_window_avg(depth_amm, *event_window),
                    _window_avg(depth_no, *event_window))),
            ('Vol full pre',
             _fmt_metric(_window_avg(vol_amm_all, *full_pre), digits=4),
             _fmt_metric(_window_avg(vol_no_all, *full_pre), digits=4),
             _delta(_window_avg(vol_amm_all, *full_pre),
                    _window_avg(vol_no_all, *full_pre))),
            ('Vol local pre',
             _fmt_metric(_window_avg(vol_amm_all, *local_pre), digits=4),
             _fmt_metric(_window_avg(vol_no_all, *local_pre), digits=4),
             _delta(_window_avg(vol_amm_all, *local_pre),
                    _window_avg(vol_no_all, *local_pre))),
            (f'Vol {row_suffix}',
             _fmt_metric(_window_avg(vol_amm_all, *event_window), digits=4),
             _fmt_metric(_window_avg(vol_no_all, *event_window), digits=4),
             _delta(_window_avg(vol_amm_all, *event_window),
                    _window_avg(vol_no_all, *event_window))),
        ]
        ax.set_title(event['title'],
                 fontsize=14, fontweight='bold', pad=12)
        ax.text(0.5, 0.92, event['subtitle'],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9.5, fontfamily='monospace')
    else:
        avg_spr_amm = _mean(logger_amm.clob_qspr)
        avg_spr_no = _mean(logger_no_amm.clob_qspr)
        avg_dep_amm = _mean(depth_amm)
        avg_dep_no = _mean(depth_no)
        avg_vol_amm = _mean(vol_amm_all)
        avg_vol_no = _mean(vol_no_all)

        rows = [
            ('Avg Spread (bps)', f'{avg_spr_amm:.1f}', f'{avg_spr_no:.1f}',
             _delta(avg_spr_amm, avg_spr_no)),
            ('Avg CLOB Depth', f'{avg_dep_amm:.0f}', f'{avg_dep_no:.0f}',
             _delta(avg_dep_amm, avg_dep_no)),
            ('Avg Volatility', f'{avg_vol_amm:.4f}', f'{avg_vol_no:.4f}',
             _delta(avg_vol_amm, avg_vol_no)),
        ]

    col_labels = ['Metric', 'AMM', 'No AMM', 'Delta']
    tbl = ax.table(
        cellText=[[r[0], r[1], r[2], r[3]] for r in rows],
        colLabels=col_labels, loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11.5)
    tbl.scale(1.0, 1.7)
    try:
        tbl.auto_set_column_width(col=list(range(4)))
    except AttributeError:
        pass
    for j in range(4):
        tbl[0, j].set_facecolor('#4c72b0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(4):
            tbl[i, j].set_facecolor(color)
    plt.tight_layout()
    p = os.path.join(out_dir, 'cmp_quality_table.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    return paths


# ===================================================================
#  Standalone figures requested separately
# ===================================================================

def plot_market_quality_table(logger: 'MetricsLogger',
                              out_dir: str = 'output',
                              stress_start: Optional[int] = None,
                              shock_iter: Optional[int] = None) -> str:
    """
    Market-quality comparison table rendered as an image.
    Columns: Metric | Normal | Stress | Ratio.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(logger.iterations)
    event = _event_windows(logger, n, shock_iter=shock_iter,
                           stress_start=stress_start)

    # --- collect metrics ---
    def _mean(lst):
        f = [x for x in lst if math.isfinite(x)]
        return sum(f) / len(f) if f else float('nan')

    qspr = logger.clob_qspr
    depth_total = [(d.get('total', 0) if isinstance(d, dict) else 0)
                   for d in logger.clob_depth]

    rows = []
    if event['has_event']:
        full_pre = event['full_pre']
        local_pre = event['local_pre']
        event_window = event['event']

        def _ratio(event_value: float, local_value: float, digits: int = 2) -> str:
            if not (math.isfinite(event_value) and math.isfinite(local_value) and local_value != 0):
                return '—'
            return f'{event_value / local_value:.{digits}f}×'

        rows.append((
            'CLOB Quoted Spread (bps)',
            _fmt_metric(_window_avg(qspr, *full_pre)),
            _fmt_metric(_window_avg(qspr, *local_pre)),
            _fmt_metric(_window_avg(qspr, *event_window)),
            _ratio(_window_avg(qspr, *event_window), _window_avg(qspr, *local_pre)),
        ))
        rows.append((
            'CLOB Quoted Spread median',
            _fmt_metric(_window_median(qspr, *full_pre)),
            _fmt_metric(_window_median(qspr, *local_pre)),
            _fmt_metric(_window_median(qspr, *event_window)),
            _ratio(_window_median(qspr, *event_window), _window_median(qspr, *local_pre)),
        ))
        rows.append((
            'CLOB Depth (units)',
            _fmt_metric(_window_avg(depth_total, *full_pre), digits=0),
            _fmt_metric(_window_avg(depth_total, *local_pre), digits=0),
            _fmt_metric(_window_avg(depth_total, *event_window), digits=0),
            _ratio(_window_avg(depth_total, *event_window), _window_avg(depth_total, *local_pre)),
        ))

        for v in ['clob'] + list(logger.amm_cost_curves.keys()):
            s = logger.cost_series(v, 5)
            rows.append((
                f'{v.upper()} Cost Q=5 (bps)',
                _fmt_metric(_window_avg(s, *full_pre)),
                _fmt_metric(_window_avg(s, *local_pre)),
                _fmt_metric(_window_avg(s, *event_window)),
                _ratio(_window_avg(s, *event_window), _window_avg(s, *local_pre)),
            ))

        for v in logger.amm_cost_curves:
            full_share = logger.customer_volume_share(v, *full_pre)
            local_share = logger.customer_volume_share(v, *local_pre)
            event_share = logger.customer_volume_share(v, *event_window)
            rows.append((
                f'{v.upper()} Customer Volume Share (Ratio of Sums)',
                f'{full_share:.1%}' if math.isfinite(full_share) else 'N/A',
                f'{local_share:.1%}' if math.isfinite(local_share) else 'N/A',
                f'{event_share:.1%}' if math.isfinite(event_share) else 'N/A',
                _ratio(event_share, local_share, digits=1),
            ))

        clob_cost = logger.cost_series('clob', 5)
        for v in logger.amm_cost_curves:
            venue_cost = logger.cost_series(v, 5)
            full_rho = logger.series_correlation(clob_cost[full_pre[0]:full_pre[1]],
                                                 venue_cost[full_pre[0]:full_pre[1]])
            local_rho = logger.series_correlation(clob_cost[local_pre[0]:local_pre[1]],
                                                  venue_cost[local_pre[0]:local_pre[1]])
            event_rho = logger.series_correlation(clob_cost[event_window[0]:event_window[1]],
                                                  venue_cost[event_window[0]:event_window[1]])
            rows.append((
                f'ρ CLOB↔{v.upper()} (Q=5)',
                f'{full_rho:.3f}' if math.isfinite(full_rho) else 'N/A',
                f'{local_rho:.3f}' if math.isfinite(local_rho) else 'N/A',
                f'{event_rho:.3f}' if math.isfinite(event_rho) else 'N/A',
                '',
            ))
    else:
        avg_qspr = _mean(qspr)
        med_qspr = _window_median(qspr, 0, n)
        avg_depth = _mean(depth_total)
        med_depth = _window_median(depth_total, 0, n)
        rows.append((
            'CLOB Quoted Spread (bps)',
            _fmt_metric(avg_qspr),
            _fmt_metric(med_qspr),
            _fmt_metric(np.std([x for x in qspr if math.isfinite(x)]), digits=2),
        ))
        rows.append((
            'CLOB Depth (units)',
            _fmt_metric(avg_depth, digits=0),
            _fmt_metric(med_depth, digits=0),
            _fmt_metric(np.std([x for x in depth_total if math.isfinite(x)]), digits=1),
        ))

        for v in ['clob'] + list(logger.amm_cost_curves.keys()):
            s = logger.cost_series(v, 5)
            rows.append((
                f'{v.upper()} Cost Q=5 (bps)',
                _fmt_metric(_mean(s)),
                _fmt_metric(_window_median(s, 0, len(s))),
                _fmt_metric(np.std([x for x in s if math.isfinite(x)]), digits=2),
            ))

        for v in logger.amm_cost_curves:
            share = logger.customer_volume_share(v)
            rows.append((
                f'{v.upper()} Customer Volume Share (Ratio of Sums)',
                f'{share:.1%}' if math.isfinite(share) else 'N/A',
                '—',
                '—',
            ))

        for v in logger.amm_cost_curves:
            rho = logger.cost_correlation('clob', v, Q=5)
            rows.append((
                f'ρ CLOB↔{v.upper()} (Q=5)',
                f'{rho:.3f}' if math.isfinite(rho) else 'N/A',
                '—',
                '—',
            ))

    # --- render table ---
    has_event = event['has_event']
    if has_event:
        col_labels = ['Metric', 'Full pre', 'Local pre', event['event_label'], 'Event / local']
        cell_text = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
    else:
        col_labels = ['Metric', 'Average', 'Median', 'Std']
        cell_text = [[r[0], r[1], r[2], r[3]] for r in rows]

    fig, ax = plt.subplots(figsize=(13 if has_event else 10.5, 0.55 * len(rows) + 1.5))
    ax.axis('off')
    ax.set_title('Market Quality: Full pre vs Local pre vs Shock/Stress' if has_event
                 else 'Market Quality Summary',
                 fontsize=14, fontweight='bold', pad=12)
    if has_event:
        ax.text(0.5, 0.98, event['subtitle'],
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9.5, fontfamily='monospace')

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5 if has_event else 11)
    tbl.scale(1.0, 1.6)
    try:
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
    except AttributeError:
        pass

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#4c72b0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    path = os.path.join(out_dir, 'market_quality_table.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_clob_depth(logger: 'MetricsLogger',
                    out_dir: str = 'output',
                    rolling: int = 10) -> str:
    """CLOB order-book depth (bid + ask) over time."""
    os.makedirs(out_dir, exist_ok=True)

    n = len(logger.iterations)
    bid_d = [(d.get('bid', 0) if isinstance(d, dict) else 0)
             for d in logger.clob_depth]
    ask_d = [(d.get('ask', 0) if isinstance(d, dict) else 0)
             for d in logger.clob_depth]
    bid_r = _rolling_avg(bid_d, rolling)
    ask_r = _rolling_avg(ask_d, rolling)

    fig, ax = plt.subplots(figsize=(12, 5))
    title = 'CLOB Order-Book Depth (Bid + Ask)'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title, fontsize=13)
    ax.fill_between(range(n), 0, bid_r, alpha=0.5, label='Bid depth',
                     color='#2ca02c')
    ax.fill_between(range(n), 0, [-a for a in ask_r], alpha=0.5,
                     label='Ask depth', color='#d62728')
    ax.axhline(0, color='black', lw=0.6)
    _shade_stress(ax, logger)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Depth (units)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path = os.path.join(out_dir, 'clob_depth.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_quoted_spread(logger: 'MetricsLogger',
                       out_dir: str = 'output',
                       rolling: int = 5) -> str:
    """CLOB quoted spread with mean-line annotations for normal/stress."""
    os.makedirs(out_dir, exist_ok=True)

    qspr = logger.clob_qspr
    s = _rolling_avg(qspr, rolling)
    n = len(s)

    # Regime split
    regimes = logger.regime_series
    normal_vals = [v for v, r in zip(qspr, regimes) if r != 'stress' and math.isfinite(v)]
    stress_vals = [v for v, r in zip(qspr, regimes) if r == 'stress' and math.isfinite(v)]
    avg_n = sum(normal_vals) / len(normal_vals) if normal_vals else 0
    avg_s = sum(stress_vals) / len(stress_vals) if stress_vals else 0

    fig, ax = plt.subplots(figsize=(12, 5))
    title = 'CLOB Quoted Spread'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title, fontsize=13)
    ax.plot(s, color='black', linewidth=1)
    _shade_stress(ax, logger)

    # Mean lines
    ax.axhline(avg_n, color=_C['normal'], ls='--', lw=1.2,
               label=f'Normal mean = {avg_n:.1f} bps')
    if stress_vals:
        ax.axhline(avg_s, color=_C['stress'], ls='--', lw=1.2,
                   label=f'Stress mean = {avg_s:.1f} bps')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spread (bps)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path = os.path.join(out_dir, 'quoted_spread.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_h1_summary_table(logger: 'MetricsLogger',
                          out_dir: str = 'output') -> str:
    """
    H1 summary table as image: avg cost per venue per Q + flow shares + trade count.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = logger.summary()
    Q_grid = logger.Q_grid
    venues = ['clob'] + list(logger.amm_cost_curves.keys())

    # --- Cost rows ---
    col_labels = ['Q'] + [v.upper() for v in venues]
    cost_rows = []
    for q in Q_grid:
        row = [str(int(q))]
        for v in venues:
            val = summary.get(f'avg_cost_{v}_Q{q}', float('nan'))
            row.append(f'{val:.1f}' if math.isfinite(val) else 'N/A')
        cost_rows.append(row)

    # --- Flow share row ---
    flow_row = ['Flow %']
    for v in venues:
        fs = summary.get(f'avg_flow_share_{v}', 0)
        flow_row.append(f'{fs:.1%}')

    # --- Meta ---
    n_iter = summary.get('n_iterations', 0)
    n_trades = summary.get('n_trades', 0)

    all_rows = cost_rows + [flow_row]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(all_rows) + 2.2))
    ax.axis('off')
    ax.set_title(f'H1 · Average Execution Cost (bps) & Flow Share\n'
                 f'Iterations: {n_iter}  |  Trades: {n_trades}',
                 fontsize=13, fontweight='bold', pad=14)

    tbl = ax.table(cellText=all_rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.7)

    n_cols = len(col_labels)
    # Header
    for j in range(n_cols):
        tbl[0, j].set_facecolor('#4c72b0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Flow share row highlight
    flow_idx = len(cost_rows) + 1  # +1 for header
    for j in range(n_cols):
        tbl[flow_idx, j].set_facecolor('#e8f0fe')
        tbl[flow_idx, j].set_text_props(fontweight='bold')
    # Alternate cost rows
    for i in range(1, len(cost_rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(n_cols):
            tbl[i, j].set_facecolor(color)

    path = os.path.join(out_dir, 'h1_summary_table.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Convenience wrapper
# ===================================================================

def generate_all_dashboards(logger: 'MetricsLogger',
                            out_dir: str = 'output',
                            stress_start: Optional[int] = None,
                            Q: float = 5,
                            rolling: int = 10,
                            logger_no_amm: 'MetricsLogger' = None,
                            shock_iter: Optional[int] = None) -> list:
    """Generate all dashboards, return list of file paths."""
    os.makedirs(out_dir, exist_ok=True)

    paths = [
        dashboard_context(logger, out_dir=out_dir, rolling=rolling,
                  shock_iter=shock_iter),
        dashboard_h1(logger, out_dir=out_dir, Q=Q, rolling=rolling,
                     shock_iter=shock_iter),
        dashboard_h2(logger, out_dir=out_dir, Q=Q, rolling=rolling,
                     stress_start=stress_start, shock_iter=shock_iter),
    ]
    if logger_no_amm is not None:
        paths.append(dashboard_comparison(
            logger, logger_no_amm,
            out_dir=out_dir, rolling=rolling, stress_start=stress_start,
            shock_iter=shock_iter,
        ))
    return paths


# ===================================================================
#  Individual plot saver
# ===================================================================

def save_all_individual_plots(logger: 'MetricsLogger',
                              out_dir: str = 'output',
                              stress_start: Optional[int] = None,
                              Q: float = 5,
                              rolling: int = 10,
                              logger_no_amm: 'MetricsLogger' = None,
                              shock_iter: Optional[int] = None) -> list:
    """Call every venue_plots function, intercept plt.show → savefig."""
    os.makedirs(out_dir, exist_ok=True)
    saved: list = []

    # Map: (function, kwargs, filename)
    plot_calls = [
        # Context
        (_vp.plot_environment,              {},                                      'environment'),
        (_vp.plot_fx_price,                 {'rolling': rolling, 'shock_iter': shock_iter}, 'fx_price'),
        (_vp.plot_clob_spread,              {'rolling': rolling},                    'clob_spread'),
        # H1
        (_vp.plot_execution_cost_curves,    {},                                      'h1_cost_curves'),
        (_vp.plot_cost_decomposition,       {'Q': Q},                                'h1_cost_decomposition'),
        (_vp.plot_total_market_depth,       {'rolling': rolling, 'shock_iter': shock_iter}, 'h1_total_depth'),
        # H2
        (_vp.plot_cost_timeseries,          {'Q': Q, 'rolling': rolling},            'h2_cost_timeseries'),
        (_vp.plot_flow_allocation,          {'rolling': rolling},                    'h2_flow_allocation'),
        (_vp.plot_clob_spread_vs_amm_cost,  {'Q': Q, 'rolling': rolling},           'h2_spread_vs_amm'),
        (_vp.plot_commonality,              {'Q': Q, 'window': 30},                 'h2_commonality'),
        (_vp.plot_amm_liquidity,            {},                                      'h2_amm_liquidity'),
        (_vp.plot_stress_flow_migration,    {'stress_start': stress_start},          'h2_stress_migration'),
        # Auxiliary
        (_vp.plot_amm_reserves,             {'pool_name': 'cpmm'},                  'aux_cpmm_reserves'),
        (_vp.plot_amm_reserves,             {'pool_name': 'hfmm'},                  'aux_hfmm_reserves'),
        (_vp.plot_volume_slippage_profile,  {'pool_name': 'cpmm'},                  'aux_cpmm_vol_slip'),
        (_vp.plot_volume_slippage_profile,  {'pool_name': 'hfmm'},                  'aux_hfmm_vol_slip'),
    ]

    # Direct-save plots (they handle savefig internally)
    direct_saves = [
        (plot_market_quality_table, {'stress_start': stress_start, 'shock_iter': shock_iter}),
        (plot_clob_depth,           {'rolling': rolling}),
        (plot_quoted_spread,        {'rolling': rolling}),
        (plot_h1_summary_table,     {}),
    ]
    for fn, kw in direct_saves:
        try:
            p = fn(logger, out_dir=out_dir, **kw)
            saved.append(p)
        except Exception as exc:
            print(f'  ⚠ {fn.__name__}: {exc}')

    # Temporarily replace plt.show so the plot functions save instead
    _real_show = plt.show

    for fn, kwargs, fname in plot_calls:
        # Intercept show: grab current figure, save, close
        path = os.path.join(out_dir, f'{fname}.png')

        def _save_show():
            fig = plt.gcf()
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            plt.close(fig)

        plt.show = _save_show  # type: ignore[assignment]
        try:
            fn(logger, **kwargs)
        except Exception as exc:
            print(f'  ⚠ {fname}: {exc}')
            continue
        finally:
            plt.show = _real_show  # type: ignore[assignment]

        saved.append(path)

    # ── Comparison individual plots (With vs Without AMM) ────────────
    if logger_no_amm is not None:
        cmp_paths = save_comparison_individual(
            logger, logger_no_amm,
            out_dir=out_dir, rolling=rolling, stress_start=stress_start,
            shock_iter=shock_iter,
        )
        saved.extend(cmp_paths)

    return saved
