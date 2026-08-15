"""
shock_dashboard.py — CLOB-only vs AMM-only: 20 % price-shock dashboard.

One combined dashboard with 6 panels (3 rows × 2 cols) comparing
liquidity metrics and recovery dynamics after a sudden −20 % price shock.

    Left column  = CLOB-only market
    Right column = AMM-only  market

Row 0: Mid-Price (shows the shock + recovery trajectory)
Row 1: Liquidity metric (CLOB spread | AMM execution cost)
Row 2: Market depth / reserves (CLOB depth | AMM liquidity L_t)

Also exposes ``save_shock_individual_plots`` for 6 stand-alone PNGs.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if TYPE_CHECKING:
    from AgentBasedModel.metrics.logger import MetricsLogger

from AgentBasedModel.visualization.venue_plots import (
    _avg_finite, _rolling_avg, _shade_stress,
)

DPI = 200

_C_CLOB = '#1f77b4'
_C_AMM_CPMM = '#ff7f0e'
_C_AMM_HFMM = '#2ca02c'
_C_SHOCK = '#c44e52'


# ── helpers ──────────────────────────────────────────────────────────

def _mean(lst):
    f = [x for x in lst if math.isfinite(x)]
    return sum(f) / len(f) if f else float('nan')


def _shade_shock(ax, shock_iter: int, color=_C_SHOCK, alpha=0.12, label='Shock'):
    """Draw a vertical line + narrow band at the shock iteration."""
    ax.axvline(shock_iter, color=color, ls='--', lw=1.4, alpha=0.7, label=label)


# =====================================================================
#  Main comparison dashboard
# =====================================================================

def dashboard_shock_comparison(
    logger_clob: 'MetricsLogger',
    logger_amm: 'MetricsLogger',
    shock_iter: int = 200,
    out_dir: str = 'output',
    rolling: int = 5,
    Q: float = 5,
) -> str:
    """
    6-panel dashboard: CLOB-only (left) vs AMM-only (right) under a
    −20 % price shock at *shock_iter*.

    Returns path to saved PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.30)
    fig.suptitle(
        'Price Shock −20 %  ·  CLOB-Only vs AMM-Only  —  Liquidity & Recovery',
        fontsize=16, fontweight='bold', y=0.97,
    )

    n_c = len(logger_clob.iterations)
    n_a = len(logger_amm.iterations)

    pool_names = list(logger_amm.amm_cost_curves.keys())

    # ── Row 0: Mid-Price ─────────────────────────────────────────────

    # (0,0) CLOB price
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f'CLOB-Only: Mid-Price  [MA {rolling}]', fontsize=12)
    p_clob = _rolling_avg(logger_clob.clob_mid_series, rolling)
    ax.plot(p_clob, color=_C_CLOB, lw=1.2)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Price')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)

    # (0,1) AMM price(s)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f'AMM-Only: Mid-Prices  [MA {rolling}]', fontsize=12)
    # CLOB reference (thin)
    ref = _rolling_avg(logger_amm.clob_mid_series, rolling)
    ax.plot(ref, color='grey', lw=1, ls=':', alpha=0.5, label='CLOB ref')
    for name in pool_names:
        s = logger_amm.amm_mid_series.get(name, [])
        if s:
            ax.plot(_rolling_avg(s, rolling), lw=1.2,
                    label=name.upper(),
                    color=_C_AMM_CPMM if name == 'cpmm' else _C_AMM_HFMM)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Price')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)

    # ── Row 1: Liquidity metric ──────────────────────────────────────

    # (1,0) CLOB quoted spread
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(f'CLOB-Only: Quoted Spread  [MA {rolling}]', fontsize=12)
    spr = _rolling_avg(logger_clob.clob_qspr, rolling)
    ax.plot(spr, color=_C_CLOB, lw=1.2)
    _shade_shock(ax, shock_iter)
    # mean lines: pre / post
    pre = logger_clob.clob_qspr[:shock_iter]
    post = logger_clob.clob_qspr[shock_iter:]
    m_pre = _mean(pre)
    m_post = _mean(post)
    ax.axhline(m_pre, color='#4c72b0', ls='--', lw=1, alpha=0.6,
               label=f'Pre-shock mean {m_pre:.1f}')
    ax.axhline(m_post, color=_C_SHOCK, ls='--', lw=1, alpha=0.6,
               label=f'Post-shock mean {m_post:.1f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spread (bps)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # (1,1) AMM all-in cost C(Q)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title(f'AMM-Only: Execution Cost  C(Q={int(Q)})  [MA {rolling}]',
                 fontsize=12)
    for name in pool_names:
        costs = logger_amm.cost_series(name, Q)
        if costs:
            ax.plot(_rolling_avg(costs, rolling), lw=1.2,
                    label=name.upper(),
                    color=_C_AMM_CPMM if name == 'cpmm' else _C_AMM_HFMM)
    _shade_shock(ax, shock_iter)
    # aggregate pre / post cost (first pool, for caption)
    if pool_names:
        first = pool_names[0]
        cs = logger_amm.cost_series(first, Q)
        m_pre_a = _mean(cs[:shock_iter]) if cs else float('nan')
        m_post_a = _mean(cs[shock_iter:]) if cs else float('nan')
        ax.axhline(m_pre_a, color='#4c72b0', ls='--', lw=1, alpha=0.5,
                   label=f'Pre mean {m_pre_a:.1f}')
        ax.axhline(m_post_a, color=_C_SHOCK, ls='--', lw=1, alpha=0.5,
                   label=f'Post mean {m_post_a:.1f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (bps)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Row 2: Depth / Reserves ──────────────────────────────────────

    # (2,0) CLOB order-book depth
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title(f'CLOB-Only: Order-Book Depth  [MA {rolling}]', fontsize=12)
    depth_total = [
        (d.get('bid', 0) + d.get('ask', 0)) if isinstance(d, dict) else 0
        for d in logger_clob.clob_depth
    ]
    dr = _rolling_avg(depth_total, rolling)
    ax.plot(dr, color=_C_CLOB, lw=1.2)
    _shade_shock(ax, shock_iter)
    m_pre_d = _mean(depth_total[:shock_iter])
    m_post_d = _mean(depth_total[shock_iter:])
    ax.axhline(m_pre_d, color='#4c72b0', ls='--', lw=1, alpha=0.5,
               label=f'Pre mean {m_pre_d:.0f}')
    ax.axhline(m_post_d, color=_C_SHOCK, ls='--', lw=1, alpha=0.5,
               label=f'Post mean {m_post_d:.0f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Depth (bid + ask)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # (2,1) AMM liquidity L_t
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('AMM-Only: Pool Liquidity  $L_t$', fontsize=12)
    for name in pool_names:
        ls = logger_amm.amm_L_series.get(name, [])
        if ls:
            ax.plot(ls, lw=1.2, label=name.upper(),
                    color=_C_AMM_CPMM if name == 'cpmm' else _C_AMM_HFMM)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$L_t$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Bottom KPI annotation ────────────────────────────────────────
    kpi = []
    kpi.append(f'CLOB spread: {m_pre:.1f} → {m_post:.1f} bps '
               f'({m_post/m_pre:.1f}× shock)' if m_pre > 0 else 'CLOB spread: N/A')
    if pool_names:
        first = pool_names[0]
        cs = logger_amm.cost_series(first, Q)
        mp = _mean(cs[:shock_iter]) if cs else float('nan')
        ma = _mean(cs[shock_iter:]) if cs else float('nan')
        if math.isfinite(mp) and mp > 0:
            kpi.append(f'{first.upper()} cost Q={int(Q)}: '
                       f'{mp:.1f} → {ma:.1f} bps ({ma/mp:.1f}×)')
    kpi.append(f'CLOB depth: {m_pre_d:.0f} → {m_post_d:.0f}')

    fig.text(
        0.50, 0.01, '   |   '.join(kpi),
        ha='center', va='bottom', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7', alpha=0.9),
    )

    path = os.path.join(out_dir, 'dashboard_shock_clob_vs_amm.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# =====================================================================
#  Individual panels as separate PNGs
# =====================================================================

def save_shock_individual_plots(
    logger_clob: 'MetricsLogger',
    logger_amm: 'MetricsLogger',
    shock_iter: int = 200,
    out_dir: str = 'output',
    rolling: int = 5,
    Q: float = 5,
) -> list:
    """Save each of the 6 dashboard panels as a standalone PNG."""
    os.makedirs(out_dir, exist_ok=True)
    paths: list = []
    pool_names = list(logger_amm.amm_cost_curves.keys())

    # ── 1. CLOB price ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'CLOB-Only: Mid-Price After −20 % Shock  [MA {rolling}]', fontsize=13)
    ax.plot(_rolling_avg(logger_clob.clob_mid_series, rolling),
            color=_C_CLOB, lw=1.2)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Price')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'shock_clob_price.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 2. AMM price ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'AMM-Only: Mid-Prices After −20 % Shock  [MA {rolling}]', fontsize=13)
    ref = _rolling_avg(logger_amm.clob_mid_series, rolling)
    ax.plot(ref, color='grey', lw=1, ls=':', alpha=0.5, label='CLOB ref')
    for name in pool_names:
        s = logger_amm.amm_mid_series.get(name, [])
        if s:
            ax.plot(_rolling_avg(s, rolling), lw=1.2, label=name.upper(),
                    color=_C_AMM_CPMM if name == 'cpmm' else _C_AMM_HFMM)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Price')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'shock_amm_price.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 3. CLOB spread ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'CLOB-Only: Quoted Spread After Shock  [MA {rolling}]', fontsize=13)
    ax.plot(_rolling_avg(logger_clob.clob_qspr, rolling), color=_C_CLOB, lw=1.2)
    _shade_shock(ax, shock_iter)
    pre = logger_clob.clob_qspr[:shock_iter]
    post = logger_clob.clob_qspr[shock_iter:]
    ax.axhline(_mean(pre), color='#4c72b0', ls='--', lw=1, alpha=0.6,
               label=f'Pre mean {_mean(pre):.1f}')
    ax.axhline(_mean(post), color=_C_SHOCK, ls='--', lw=1, alpha=0.6,
               label=f'Post mean {_mean(post):.1f}')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Spread (bps)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'shock_clob_spread.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 4. AMM cost ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'AMM-Only: Execution Cost C(Q={int(Q)}) After Shock  [MA {rolling}]',
                 fontsize=13)
    for name in pool_names:
        costs = logger_amm.cost_series(name, Q)
        if costs:
            ax.plot(_rolling_avg(costs, rolling), lw=1.2, label=name.upper(),
                    color=_C_AMM_CPMM if name == 'cpmm' else _C_AMM_HFMM)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Cost (bps)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'shock_amm_cost.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 5. CLOB depth ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'CLOB-Only: Order-Book Depth After Shock  [MA {rolling}]', fontsize=13)
    depth = [(d.get('bid', 0) + d.get('ask', 0)) if isinstance(d, dict) else 0
             for d in logger_clob.clob_depth]
    ax.plot(_rolling_avg(depth, rolling), color=_C_CLOB, lw=1.2)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Depth (bid + ask)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'shock_clob_depth.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ── 6. AMM liquidity ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('AMM-Only: Pool Liquidity $L_t$ After Shock', fontsize=13)
    for name in pool_names:
        ls = logger_amm.amm_L_series.get(name, [])
        if ls:
            ax.plot(ls, lw=1.2, label=name.upper(),
                    color=_C_AMM_CPMM if name == 'cpmm' else _C_AMM_HFMM)
    _shade_shock(ax, shock_iter)
    ax.set_xlabel('Iteration'); ax.set_ylabel('$L_t$')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, 'shock_amm_liquidity.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight'); plt.close(fig); paths.append(p)

    return paths

# =====================================================================
#  Price movement & recovery — dedicated plot
# =====================================================================

def _find_recovery_iter(series, shock_iter, threshold_pct=2.0):
    """
    Find the first iteration after *shock_iter* where the price returns
    within *threshold_pct* % of the pre-shock mean and stays there for
    at least 10 consecutive periods.  Returns None if not recovered.
    """
    pre_mean = _mean(series[:shock_iter])
    if math.isnan(pre_mean) or pre_mean == 0:
        return None
    band = pre_mean * threshold_pct / 100.0
    run = 0
    for i in range(shock_iter, len(series)):
        if abs(series[i] - pre_mean) <= band:
            run += 1
            if run >= 10:
                return i - 9       # first iter of the stable run
        else:
            run = 0
    return None


def plot_price_recovery(
    logger_clob: 'MetricsLogger',
    logger_amm: 'MetricsLogger',
    shock_iter: int = 200,
    out_dir: str = 'output',
    rolling: int = 5,
) -> str:
    """
    Two-row figure focused purely on price movement & recovery.

    Row 0 — Absolute prices: both CLOB and AMM on the same axes.
    Row 1 — Normalised deviation from pre-shock mean (%).

    Annotations: shock point, trough, recovery iteration, recovery time.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── data prep ────────────────────────────────────────────────────
    clob_raw = list(logger_clob.clob_mid_series)
    clob_sm = _rolling_avg(clob_raw, rolling)

    pool_names = list(logger_amm.amm_cost_curves.keys())
    # Pick first AMM pool for the combined comparison
    amm_name = pool_names[0] if pool_names else None
    amm_raw = list(logger_amm.amm_mid_series.get(amm_name, [])) if amm_name else []
    amm_sm = _rolling_avg(amm_raw, rolling) if amm_raw else []

    # Pre-shock reference levels
    pre_clob = _mean(clob_raw[:shock_iter])
    pre_amm = _mean(amm_raw[:shock_iter]) if amm_raw else float('nan')

    # Normalised deviations (%)
    def _pct_dev(series, ref):
        if math.isnan(ref) or ref == 0:
            return []
        return [100.0 * (v - ref) / ref for v in series]

    clob_dev = _pct_dev(clob_sm, pre_clob)
    amm_dev = _pct_dev(amm_sm, pre_amm) if amm_sm else []

    # Recovery iterations
    rec_clob = _find_recovery_iter(clob_raw, shock_iter)
    rec_amm = _find_recovery_iter(amm_raw, shock_iter) if amm_raw else None

    # Trough (min price in post-shock window)
    def _trough(series, si):
        post = series[si:min(si + 100, len(series))]
        if not post:
            return si, float('nan')
        idx = int(np.argmin(post))
        return si + idx, post[idx]

    tr_clob_i, tr_clob_v = _trough(clob_sm, shock_iter)
    tr_amm_i, tr_amm_v = _trough(amm_sm, shock_iter) if amm_sm else (shock_iter, float('nan'))

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={'height_ratios': [1.2, 1],
                                          'hspace': 0.15})
    fig.suptitle(
        'Price Movement & Recovery  ·  CLOB-Only vs AMM-Only  '
        '(−20 % Shock)',
        fontsize=15, fontweight='bold', y=0.97,
    )

    # ---- Row 0: absolute prices ------------------------------------
    ax = axes[0]
    ax.set_title(f'Absolute Mid-Price  [MA {rolling}]', fontsize=12, pad=8)
    ax.plot(clob_sm, color=_C_CLOB, lw=1.5, label='CLOB mid')
    if amm_sm:
        ax.plot(amm_sm, color=_C_AMM_CPMM, lw=1.5,
                label=f'{amm_name.upper()} mid')

    # Pre-shock reference
    ax.axhline(pre_clob, color=_C_CLOB, ls=':', lw=1, alpha=0.45,
               label=f'CLOB pre-shock {pre_clob:.2f}')
    if math.isfinite(pre_amm):
        ax.axhline(pre_amm, color=_C_AMM_CPMM, ls=':', lw=1, alpha=0.45,
                   label=f'AMM pre-shock  {pre_amm:.2f}')

    _shade_shock(ax, shock_iter)

    # Trough markers
    ax.annotate(f'Trough {tr_clob_v:.1f}', xy=(tr_clob_i, tr_clob_v),
                xytext=(tr_clob_i + 15, tr_clob_v - 2),
                fontsize=9, color=_C_CLOB,
                arrowprops=dict(arrowstyle='->', color=_C_CLOB, lw=1.2))
    if amm_sm and math.isfinite(tr_amm_v):
        ax.annotate(f'Trough {tr_amm_v:.1f}', xy=(tr_amm_i, tr_amm_v),
                    xytext=(tr_amm_i + 15, tr_amm_v + 2),
                    fontsize=9, color=_C_AMM_CPMM,
                    arrowprops=dict(arrowstyle='->', color=_C_AMM_CPMM, lw=1.2))

    # Recovery markers
    if rec_clob is not None:
        ax.axvline(rec_clob, color=_C_CLOB, ls='-.', lw=1, alpha=0.55)
        ax.annotate(f'CLOB recovers\niter {rec_clob} (+{rec_clob - shock_iter})',
                    xy=(rec_clob, pre_clob), fontsize=8, color=_C_CLOB,
                    xytext=(rec_clob + 8, pre_clob + 1.5),
                    arrowprops=dict(arrowstyle='->', color=_C_CLOB, lw=0.8))
    if rec_amm is not None:
        ax.axvline(rec_amm, color=_C_AMM_CPMM, ls='-.', lw=1, alpha=0.55)
        ax.annotate(f'AMM recovers\niter {rec_amm} (+{rec_amm - shock_iter})',
                    xy=(rec_amm, pre_amm), fontsize=8, color=_C_AMM_CPMM,
                    xytext=(rec_amm + 8, pre_amm - 2.5),
                    arrowprops=dict(arrowstyle='->', color=_C_AMM_CPMM, lw=0.8))

    ax.set_ylabel('Price', fontsize=11)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.25)

    # ---- Row 1: normalised deviation % -----------------------------
    ax = axes[1]
    ax.set_title('Deviation from Pre-Shock Mean  (%)', fontsize=12, pad=8)
    if clob_dev:
        ax.plot(clob_dev, color=_C_CLOB, lw=1.5, label='CLOB')
    if amm_dev:
        ax.plot(amm_dev, color=_C_AMM_CPMM, lw=1.5,
                label=f'{amm_name.upper()}')

    ax.axhline(0, color='grey', lw=0.8, ls='-', alpha=0.5)
    ax.axhspan(-2, 2, color='green', alpha=0.06, label='±2 % band')
    _shade_shock(ax, shock_iter)

    # Shade post-shock window
    ax.axvspan(shock_iter, min(shock_iter + 100, max(len(clob_dev), len(amm_dev))),
               color=_C_SHOCK, alpha=0.04)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Deviation  (%)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)

    # ── Bottom KPI box ───────────────────────────────────────────────
    kpi_parts = [f'Pre-shock CLOB: {pre_clob:.2f}']
    if math.isfinite(pre_amm):
        kpi_parts.append(f'Pre-shock AMM: {pre_amm:.2f}')
    kpi_parts.append(f'CLOB trough: {tr_clob_v:.2f} ({100*(tr_clob_v-pre_clob)/pre_clob:+.1f} %)')
    if math.isfinite(tr_amm_v) and math.isfinite(pre_amm) and pre_amm:
        kpi_parts.append(f'AMM trough: {tr_amm_v:.2f} ({100*(tr_amm_v-pre_amm)/pre_amm:+.1f} %)')
    if rec_clob is not None:
        kpi_parts.append(f'CLOB recovery: +{rec_clob - shock_iter} iters')
    else:
        kpi_parts.append('CLOB: not recovered')
    if rec_amm is not None:
        kpi_parts.append(f'AMM recovery: +{rec_amm - shock_iter} iters')
    elif amm_raw:
        kpi_parts.append('AMM: not recovered')

    fig.text(
        0.50, 0.01, '   |   '.join(kpi_parts),
        ha='center', va='bottom', fontsize=9.5, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7', alpha=0.9),
    )

    path = os.path.join(out_dir, 'price_recovery_comparison.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path