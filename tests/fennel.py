#!/usr/bin/env python3
"""
Fennell Tests -- Transaction Cost Analysis: AMM vs CLOB
=======================================================

Full replication of James Fennell's honours thesis (UTS, 2024):
    "Are Automated Market Makers the Future of Foreign Exchange?"

Five components:
  A. Analytical replication -- breakeven liquidity, AMM vs CLOB costs
     using Fennell formulas and Melvin (2020) CLOB benchmarks
  B. Partial adoption sweep -- turnover 5->100%, recalculate V0
  C. Fee optimization -- endogenous V0(f), U-curve, optimal fee
  D. HFMM extension -- StableSwap (our contribution, not in Fennell)
  E. Simulation-based test -- calibrated pools at breakeven V0

Run:
    python tests/fennel.py

Output -> output/fennell/
"""

import sys, os, math, copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'fennell')
os.makedirs(OUT_DIR, exist_ok=True)

C_CLOB = '#2196F3'
C_CPMM = '#FF9800'
C_HFMM = '#4CAF50'
C_GRAY = '#9E9E9E'

# =====================================================================
#  0. FENNELL DATA -- from thesis tables and Melvin et al. (2020)
# =====================================================================

FENNELL_DATA = {
    # -- G10 --
    'USD/AUD': dict(
        asc_pct=-0.000530, turnover_m=381, v0_m=5269,
        clob_retail=1.090, clob_wholesale=3.520,
        amm_retail=33.797, amm_wholesale=68.103,
        opt_fee_bps=19.768, opt_retail=39.497, group='G10'),
    'USD/EUR': dict(
        asc_pct=-0.000269, turnover_m=1705, v0_m=23888,
        clob_retail=0.470, clob_wholesale=1.290,
        amm_retail=30.837, amm_wholesale=38.379,
        opt_fee_bps=7.513, opt_retail=15.021, group='G10'),
    'USD/GBP': dict(
        asc_pct=-0.000421, turnover_m=714, v0_m=9923,
        clob_retail=0.900, clob_wholesale=2.510,
        amm_retail=32.016, amm_wholesale=50.196,
        opt_fee_bps=13.328, opt_retail=26.639, group='G10'),
    'USD/JPY': dict(
        asc_pct=-0.000344, turnover_m=1013, v0_m=14144,
        clob_retail=0.530, clob_wholesale=1.680,
        amm_retail=31.414, amm_wholesale=44.160,
        opt_fee_bps=10.484, opt_retail=20.956, group='G10'),
    'USD/CHF': dict(
        asc_pct=-0.000254, turnover_m=293, v0_m=4105,
        clob_retail=1.070, clob_wholesale=4.240,
        amm_retail=34.875, amm_wholesale=78.962,
        opt_fee_bps=17.857, opt_retail=35.683, group='G10'),
    'USD/CAD': dict(
        asc_pct=-0.000243, turnover_m=410, v0_m=5751,
        clob_retail=0.960, clob_wholesale=3.100,
        amm_retail=33.479, amm_wholesale=64.900,
        opt_fee_bps=14.912, opt_retail=29.802, group='G10'),
    'USD/NZD': dict(
        asc_pct=-0.000502, turnover_m=99, v0_m=1374,
        clob_retail=1.620, clob_wholesale=6.410,
        amm_retail=34.574, amm_wholesale=177.678,
        opt_fee_bps=38.110, opt_retail=76.076, group='G10'),
    'USD/SEK': dict(
        asc_pct=-0.000538, turnover_m=93, v0_m=1291,
        clob_retail=2.140, clob_wholesale=8.530,
        amm_retail=45.513, amm_wholesale=187.324,
        opt_fee_bps=40.309, opt_retail=80.457, group='G10'),
    'USD/NOK': dict(
        asc_pct=-0.000809, turnover_m=81, v0_m=1102,
        clob_retail=2.810, clob_wholesale=12.310,
        amm_retail=48.179, amm_wholesale=214.812,
        opt_fee_bps=50.449, opt_retail=100.647, group='G10'),
    # -- Exotics --
    'USD/CNY': dict(
        asc_pct=-0.000101, turnover_m=494, v0_m=6980,
        clob_retail=0.430, clob_wholesale=1.940,
        amm_retail=32.866, amm_wholesale=58.738,
        opt_fee_bps=None, opt_retail=None, group='Exotic'),
    'USD/SGD': dict(
        asc_pct=-0.000091, turnover_m=169, v0_m=2389,
        clob_retail=0.990, clob_wholesale=4.580,
        amm_retail=38.379, amm_wholesale=114.418,
        opt_fee_bps=None, opt_retail=None, group='Exotic'),
    'USD/MXN': dict(
        asc_pct=-0.000773, turnover_m=103, v0_m=1410,
        clob_retail=2.300, clob_wholesale=10.530,
        amm_retail=44.196, amm_wholesale=173.799,
        opt_fee_bps=None, opt_retail=None, group='Exotic'),
    'USD/TRY': dict(
        asc_pct=-0.002236, turnover_m=23, v0_m=303,
        clob_retail=3.140, clob_wholesale=14.220,
        amm_retail=96.382, amm_wholesale=736.003,
        opt_fee_bps=None, opt_retail=None, group='Exotic'),
    'USD/PLN': dict(
        asc_pct=-0.000567, turnover_m=33, v0_m=460,
        clob_retail=3.800, clob_wholesale=13.960,
        amm_retail=73.621, amm_wholesale=484.036,
        opt_fee_bps=None, opt_retail=None, group='Exotic'),
    'USD/ZAR': dict(
        asc_pct=-0.001143, turnover_m=64, v0_m=864,
        clob_retail=4.920, clob_wholesale=21.550,
        amm_retail=53.187, amm_wholesale=266.806,
        opt_fee_bps=None, opt_retail=None, group='Exotic'),
}

RETAIL_USD_M = 1.0
WHOLESALE_USD_M = 25.0
UNISWAP_FEE = 0.003
RF_ANNUAL = 0.0408
TRADING_DAYS = 252
OPP_COST_DAILY = RF_ANNUAL / TRADING_DAYS


# =====================================================================
#  1. ADVERSE SELECTION COST (ASC)
# =====================================================================

def asc_from_return(R_T):
    """Fennell Eq. 4: ASC_T = sqrt(R_T) - 0.5 * (1 + R_T)"""
    if R_T <= 0:
        return -0.5
    return math.sqrt(R_T) - 0.5 * (1.0 + R_T)


def asc_from_sigma(sigma_daily):
    """Expected daily |ASC| from GBM volatility: |ASC| ~ sigma^2 / 8."""
    return sigma_daily ** 2 / 8.0


def asc_from_returns_series(returns, window=35):
    """Compute ASC from daily returns using Fennell rolling-window method."""
    n = len(returns)
    if n < window:
        sigma = np.std(returns)
        return asc_from_sigma(sigma)

    asc_windows = []
    for i in range(n - window + 1):
        cum_ret = np.prod(1.0 + returns[i:i + window])
        asc_val = asc_from_return(cum_ret)
        asc_windows.append(asc_val)

    avg_asc_per_window = np.mean(asc_windows)
    daily_asc = avg_asc_per_window / window
    return abs(daily_asc)


# =====================================================================
#  2. BREAKEVEN LIQUIDITY (Fennell Eq. 8)
# =====================================================================

def breakeven_v0(fee, daily_turnover, asc_daily,
                 opp_cost_daily=OPP_COST_DAILY):
    """V0 = f * Q_T / (opp_cost + |ASC|)"""
    denom = opp_cost_daily + asc_daily
    if denom <= 0:
        return float('inf')
    return fee * daily_turnover / denom


def _implied_asc(d):
    """ASC from Fennell Table 1 (σ²/8 estimate from returns)."""
    return abs(d['asc_pct'])


# =====================================================================
#  3. CPMM PRICE IMPACT -- Fennell Eq. 12
# =====================================================================

def cpmm_price_impact_bps(v0, trade_size):
    """PI = T / (V0/2 - T) * 10000"""
    half = v0 / 2.0
    if trade_size >= half:
        return float('inf')
    return trade_size / (half - trade_size) * 10_000


def cpmm_total_cost_bps(v0, trade_size, fee=UNISWAP_FEE):
    """TC = PI + fee_bps (Fennell Eq. 13)"""
    pi = cpmm_price_impact_bps(v0, trade_size)
    return pi + fee * 10_000


# =====================================================================
#  4. HFMM (StableSwap) PRICE IMPACT -- our extension
# =====================================================================

def hfmm_cost_bps(v0, trade_size, fee, A=50.0, rate=1.0):
    """HFMM cost at breakeven V0, using HFMMPool.quote_buy/sell."""
    half_usd = v0 / 2.0
    x = half_usd / rate
    y = half_usd
    if x <= 0 or y <= 0:
        return dict(cost_bps=float('inf'), slippage_bps=float('inf'),
                    fee_bps=fee * 10_000)

    pool = HFMMPool(x=x, y=y, A=A, fee=fee, rate=rate)
    Q = trade_size / rate

    if Q >= x * 0.99:
        return dict(cost_bps=float('inf'), slippage_bps=float('inf'),
                    fee_bps=fee * 10_000)

    q_buy = pool.quote_buy(Q)
    q_sell = pool.quote_sell(Q)
    avg_cost = (q_buy['cost_bps'] + q_sell['cost_bps']) / 2
    avg_slip = (q_buy['slippage_bps'] + q_sell['slippage_bps']) / 2

    return dict(cost_bps=avg_cost, slippage_bps=avg_slip,
                fee_bps=fee * 10_000)


# =====================================================================
#  A. ANALYTICAL FENNELL REPLICATION
# =====================================================================

def run_analytical_replication():
    """Replicate Fennell Tables 1 & 2: AMM vs CLOB costs using published V0."""
    print("\n" + "=" * 90)
    print("  A. ANALYTICAL FENNELL REPLICATION -- AMM vs CLOB (Fennell Tables 1 & 2)")
    print("=" * 90)

    rows = []
    for group in ['G10', 'Exotic']:
        pairs = {k: v for k, v in FENNELL_DATA.items() if v['group'] == group}

        print(f"\n  --- {group} Currencies ---")
        hdr = (f"  {'Pair':<10} | {'CLOB R':>7} {'CLOB W':>7} |"
               f" {'AMM R':>7} {'AMM W':>7} |"
               f" {'Fennell R':>9} {'Fennell W':>9} |"
               f" {'HFMM R':>7} {'HFMM W':>7}")
        print(hdr)
        print("  " + "-" * 90)

        for pair, d in pairs.items():
            v0 = d['v0_m']

            amm_r = cpmm_total_cost_bps(v0, RETAIL_USD_M)
            amm_w = cpmm_total_cost_bps(v0, WHOLESALE_USD_M)

            hfmm_r = hfmm_cost_bps(v0, RETAIL_USD_M, fee=UNISWAP_FEE, A=50)
            hfmm_w = hfmm_cost_bps(v0, WHOLESALE_USD_M, fee=UNISWAP_FEE, A=50)

            print(f"  {pair:<10} | {d['clob_retail']:>7.3f} {d['clob_wholesale']:>7.3f} |"
                  f" {amm_r:>7.2f} {amm_w:>7.1f} |"
                  f" {d['amm_retail']:>9.3f} {d['amm_wholesale']:>9.3f} |"
                  f" {hfmm_r['cost_bps']:>7.2f} {hfmm_w['cost_bps']:>7.1f}")

            rows.append(dict(
                pair=pair, group=group,
                clob_retail=d['clob_retail'], clob_wholesale=d['clob_wholesale'],
                cpmm_retail=amm_r, cpmm_wholesale=amm_w,
                fennell_retail=d['amm_retail'], fennell_wholesale=d['amm_wholesale'],
                hfmm_retail=hfmm_r['cost_bps'], hfmm_wholesale=hfmm_w['cost_bps'],
                v0_m=v0, turnover_m=d['turnover_m'],
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'analytical_replication.csv'), index=False)

    _plot_amm_vs_clob_bars(df, 'G10', 'retail')
    _plot_amm_vs_clob_bars(df, 'G10', 'wholesale')
    _plot_amm_vs_clob_bars(df, 'Exotic', 'retail')
    _plot_amm_vs_clob_bars(df, 'Exotic', 'wholesale')

    return df


def _plot_amm_vs_clob_bars(df, group, size):
    """Bar chart comparing CLOB, CPMM, HFMM costs (Fennell Fig 7-10)."""
    sub = df[df['group'] == group].copy()
    pairs = sub['pair'].values
    n = len(pairs)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    ax.bar(x - w, sub[f'clob_{size}'], w, color=C_CLOB, label='CLOB (Melvin)')
    ax.bar(x, sub[f'cpmm_{size}'], w, color=C_CPMM, label='CPMM')
    ax.bar(x + w, sub[f'hfmm_{size}'], w, color=C_HFMM, label='HFMM')

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Transaction Cost (bps)', fontsize=11)
    trade_label = f"Retail (${RETAIL_USD_M:.0f}M)" if size == 'retail' \
        else f"Wholesale (${WHOLESALE_USD_M:.0f}M)"
    ax.set_title(f'{group} {trade_label} - AMM vs CLOB', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    fname = f'amm_vs_clob_{group.lower()}_{size}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


# =====================================================================
#  B. PARTIAL ADOPTION (Fennell Section 6)
# =====================================================================

def run_partial_adoption():
    """Sweep adoption rate 5->100%, recalculate V0, compute AMM cost."""
    print("\n" + "=" * 90)
    print("  B. PARTIAL ADOPTION -- Cost vs Adoption Rate (Fennell Figs 11-14)")
    print("=" * 90)

    adoption_pcts = np.arange(5, 101, 5)
    rows = []

    for pair, d in FENNELL_DATA.items():
        eff_asc = _implied_asc(d)

        for adopt_pct in adoption_pcts:
            frac = adopt_pct / 100.0
            partial_turnover = d['turnover_m'] * frac
            v0_partial = breakeven_v0(UNISWAP_FEE, partial_turnover,
                                      eff_asc, OPP_COST_DAILY)

            amm_r = cpmm_total_cost_bps(v0_partial, RETAIL_USD_M)
            diff_r = amm_r - d['clob_retail']

            if WHOLESALE_USD_M < v0_partial / 2:
                amm_w = cpmm_total_cost_bps(v0_partial, WHOLESALE_USD_M)
                diff_w = amm_w - d['clob_wholesale']
            else:
                amm_w = float('inf')
                diff_w = float('inf')

            rows.append(dict(
                pair=pair, group=d['group'],
                adoption_pct=adopt_pct,
                v0_partial=v0_partial,
                amm_retail=amm_r, amm_wholesale=amm_w,
                diff_retail=diff_r, diff_wholesale=diff_w,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'partial_adoption.csv'), index=False)

    _plot_adoption_curves(df, 'G10', 'retail')
    _plot_adoption_curves(df, 'G10', 'wholesale')
    _plot_adoption_curves(df, 'Exotic', 'retail')
    _plot_adoption_curves(df, 'Exotic', 'wholesale')

    # Summary
    print(f"\n  {'Pair':<10} | {'10%':>8} {'25%':>8} {'50%':>8} {'100%':>8}  (retail AMM cost diff, bps)")
    print("  " + "-" * 55)
    for pair in FENNELL_DATA:
        sub = df[df['pair'] == pair]
        vals = []
        for pct in [10, 25, 50, 100]:
            row = sub[sub['adoption_pct'] == pct]
            if len(row) > 0 and row['diff_retail'].values[0] < 1e6:
                vals.append(f"{row['diff_retail'].values[0]:>8.1f}")
            else:
                vals.append(f"{'inf':>8}")
        print(f"  {pair:<10} | {' '.join(vals)}")

    return df


def _plot_adoption_curves(df, group, size):
    """Plot adoption % vs cost difference (Fennell Figs 11-14)."""
    sub = df[(df['group'] == group)].copy()
    pairs = sub['pair'].unique()

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10

    for i, pair in enumerate(pairs):
        p_sub = sub[sub['pair'] == pair]
        vals = p_sub[f'diff_{size}'].values.copy()
        adopt = p_sub['adoption_pct'].values
        mask = np.isfinite(vals)
        if mask.sum() < 2:
            continue
        ax.plot(adopt[mask], vals[mask], color=cmap(i % 10),
                marker='o', markersize=3, linewidth=1.5,
                label=pair.replace('USD/', ''))

    ax.set_xlabel('Adoption Rate (%)', fontsize=12)
    ax.set_ylabel('AMM - CLOB Cost Difference (bps)', fontsize=12)
    trade_label = 'Retail' if size == 'retail' else 'Wholesale'
    ax.set_title(f'{group} {trade_label} - Partial Adoption (Fennell Fig)',
                 fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f'adoption_{group.lower()}_{size}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


# =====================================================================
#  C. FEE OPTIMIZATION with endogenous V0 (Fennell Section 7)
# =====================================================================

def run_fee_optimization():
    """Sweep fee 0.5->80 bps, recompute V0(f), find f* minimizing TC."""
    print("\n" + "=" * 90)
    print("  C. FEE OPTIMIZATION - Endogenous V0 (Fennell Fig 19 & Table 5)")
    print("=" * 90)

    fee_grid_bps = np.arange(0.5, 80, 0.5)
    fee_grid = fee_grid_bps / 10_000

    rows = []
    opt_results = {}

    for pair, d in FENNELL_DATA.items():
        eff_asc = _implied_asc(d)

        best_tc = float('inf')
        best_fee_bps = 30.0

        for f, f_bps in zip(fee_grid, fee_grid_bps):
            v0 = breakeven_v0(f, d['turnover_m'], eff_asc)
            tc = cpmm_total_cost_bps(v0, RETAIL_USD_M, fee=f)

            rows.append(dict(pair=pair, group=d['group'],
                             fee_bps=f_bps, v0_m=v0, tc_retail=tc))

            if tc < best_tc:
                best_tc = tc
                best_fee_bps = f_bps

        opt_results[pair] = dict(
            opt_fee_bps=best_fee_bps, opt_tc=best_tc,
            fennell_opt_fee=d.get('opt_fee_bps'),
            fennell_opt_tc=d.get('opt_retail'),
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'fee_optimization.csv'), index=False)

    # Print Table 5 comparison
    print(f"\n  {'Pair':<10} | {'Our f*':>7} {'Our TC*':>8} |"
          f" {'Fennell f*':>10} {'Fennell TC*':>11} | {'CLOB R':>7}")
    print("  " + "-" * 70)
    for pair in FENNELL_DATA:
        o = opt_results[pair]
        d = FENNELL_DATA[pair]
        f_str = f"{o['fennell_opt_fee']:>10.1f}" if o['fennell_opt_fee'] else f"{'N/A':>10}"
        t_str = f"{o['fennell_opt_tc']:>11.1f}" if o['fennell_opt_tc'] else f"{'N/A':>11}"
        print(f"  {pair:<10} | {o['opt_fee_bps']:>7.1f} {o['opt_tc']:>8.1f} |"
              f" {f_str} {t_str} | {d['clob_retail']:>7.3f}")

    _plot_fee_optimization(df)

    return df, opt_results


def _plot_fee_optimization(df):
    """Plot fee vs TC (U-curves) for selected currencies."""
    selected = ['USD/EUR', 'USD/GBP', 'USD/AUD', 'USD/NOK', 'USD/TRY']
    selected = [p for p in selected if p in df['pair'].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for i, pair in enumerate(selected):
        sub = df[df['pair'] == pair]
        valid = sub[sub['tc_retail'] < 500]
        if len(valid) == 0:
            continue
        ax.plot(valid['fee_bps'], valid['tc_retail'],
                color=cmap(i), linewidth=2,
                label=pair.replace('USD/', ''))
        best = valid.loc[valid['tc_retail'].idxmin()]
        ax.plot(best['fee_bps'], best['tc_retail'], 'o',
                color=cmap(i), markersize=8)

    ax.set_xlabel('Fixed Fee (bps)', fontsize=12)
    ax.set_ylabel('Transaction Cost (bps)', fontsize=12)
    ax.set_title('Fee Optimization - TC vs Fixed Fee Level (Fennell Fig 19)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, min(300, ax.get_ylim()[1]))
    fig.tight_layout()

    fig.savefig(os.path.join(OUT_DIR, 'fee_optimization_ucurve.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  fee_optimization_ucurve.png")


# =====================================================================
#  D. HFMM EXTENSION -- compare StableSwap to CPMM at breakeven V0
# =====================================================================

def run_hfmm_extension():
    """Compare CPMM vs HFMM at same breakeven V0 for multiple A values."""
    print("\n" + "=" * 90)
    print("  D. HFMM EXTENSION - StableSwap vs CPMM at Breakeven V0")
    print("=" * 90)

    A_values = [10, 50, 100, 500]
    rows = []

    a_cols = "  ".join([f"A={a:>3}" for a in A_values])
    print(f"\n  {'Pair':<10} | {'CLOB R':>7} | {'CPMM':>7} | {a_cols}   (retail bps)")
    print("  " + "-" * 75)

    for pair, d in FENNELL_DATA.items():
        v0 = d['v0_m']
        cpmm_tc = cpmm_total_cost_bps(v0, RETAIL_USD_M)

        hfmm_costs = []
        for A in A_values:
            res = hfmm_cost_bps(v0, RETAIL_USD_M, fee=UNISWAP_FEE, A=A)
            hfmm_costs.append(res['cost_bps'])
            rows.append(dict(pair=pair, group=d['group'],
                             A=A, v0_m=v0,
                             cpmm_retail=cpmm_tc,
                             hfmm_retail=res['cost_bps'],
                             hfmm_slippage=res['slippage_bps'],
                             clob_retail=d['clob_retail']))

        a_strs = "  ".join([f"{c:>5.2f}" for c in hfmm_costs])
        print(f"  {pair:<10} | {d['clob_retail']:>7.3f} | {cpmm_tc:>7.2f} | {a_strs}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'hfmm_extension.csv'), index=False)

    _plot_hfmm_comparison(df, A_values)

    return df


def _plot_hfmm_comparison(df, A_values):
    """Bar chart: CLOB, CPMM, and HFMM at different A values."""
    g10 = df[df['group'] == 'G10'].copy()
    pairs = g10['pair'].unique()
    n = len(pairs)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))
    n_bars = 2 + len(A_values)
    w = 0.8 / n_bars
    x = np.arange(n)

    clob_vals = [g10[g10['pair'] == p].iloc[0]['clob_retail'] for p in pairs]
    ax.bar(x - w * (n_bars - 1) / 2, clob_vals, w, color=C_CLOB, label='CLOB')

    cpmm_vals = [g10[g10['pair'] == p].iloc[0]['cpmm_retail'] for p in pairs]
    ax.bar(x - w * (n_bars - 1) / 2 + w, cpmm_vals, w, color=C_CPMM, label='CPMM')

    greens = ['#C8E6C9', '#81C784', '#4CAF50', '#2E7D32']
    for j, A in enumerate(A_values):
        vals = []
        for p in pairs:
            row = g10[(g10['pair'] == p) & (g10['A'] == A)]
            vals.append(row['hfmm_retail'].values[0] if len(row) > 0 else 0)
        ax.bar(x - w * (n_bars - 1) / 2 + w * (2 + j), vals, w,
               color=greens[j % len(greens)], label=f'HFMM A={A}')

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('USD/', '') for p in pairs],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Retail Cost (bps)', fontsize=11)
    ax.set_title('G10 Retail: CLOB vs CPMM vs HFMM at Various A', fontsize=13)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    fig.savefig(os.path.join(OUT_DIR, 'hfmm_comparison_g10.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  hfmm_comparison_g10.png")


# =====================================================================
#  E. SIMULATION-BASED TEST -- calibrated pools at breakeven V0
# =====================================================================

def _build_clob(mid, half_spread_bps, depth_per_level, n_levels=10):
    """Synthetic CLOB centered at mid with given spread and depth."""
    ex = ExchangeAgent.__new__(ExchangeAgent)
    ex.name = "FennellCLOB"
    ex.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
    ex.dividend_book = [mid] * 5
    ex.risk_free = 0.0
    ex.transaction_cost = 0.0

    tick = mid * half_spread_bps / 10_000.0
    for i in range(n_levels):
        ask_px = round(mid + tick * (i + 1), 6)
        bid_px = round(mid - tick * (i + 1), 6)
        qty = depth_per_level * (1.0 + 0.3 * i)
        ex.order_book['ask'].append(Order(ask_px, qty, 'ask', None))
        ex.order_book['bid'].push(Order(bid_px, qty, 'bid', None))

    return CLOBVenue(ex)


def run_simulation_test(pair='USD/EUR', n_steps=500,
                        trades_per_bucket=60, seed=42):
    """
    Simulation-based Fennell test, calibrated to a real currency pair.
    Pool reserves from breakeven V0, trade sizes in USD, CLOB from Melvin.
    """
    d = FENNELL_DATA[pair]
    v0 = d['v0_m']
    rate = 1.0

    eff_asc = _implied_asc(d)
    sigma_daily = math.sqrt(8.0 * eff_asc)

    clob_half_spread_bps = d['clob_retail']
    clob_depth = v0 / 50.0

    Q_buckets_usd = [1.0, 5.0, 10.0, 25.0]

    best_f = _find_optimal_fee(d, eff_asc)

    fee_regimes = [
        ('fee=0', 0.0, 0.0, 0.0),
        ('30 bps (Uniswap)', 0.0, UNISWAP_FEE, UNISWAP_FEE),
        ('Optimized', 0.0, best_f, best_f * 0.5),
    ]

    print(f"\n{'=' * 90}")
    print(f"  E. SIMULATION TEST - {pair}")
    print(f"{'=' * 90}")
    print(f"  V0 = ${v0:,.0f}M  |  sigma = {sigma_daily:.4f}"
          f"  |  CLOB spread = {clob_half_spread_bps:.3f} bps"
          f"  |  Opt fee = {best_f * 10000:.1f} bps")

    rng = np.random.RandomState(seed)
    prices = np.zeros(n_steps + 1)
    prices[0] = rate
    for t in range(n_steps):
        eps = rng.randn()
        prices[t + 1] = prices[t] * math.exp(
            -0.5 * sigma_daily ** 2 + sigma_daily * eps)

    import random as py_random
    py_random.seed(seed)
    orders = []
    for Q_usd in Q_buckets_usd:
        for i in range(trades_per_bucket):
            side = 'buy' if i % 2 == 0 else 'sell'
            orders.append({'side': side, 'quantity': Q_usd, 'bucket': Q_usd})
    py_random.shuffle(orders)

    all_records = []

    # Assign each order to a step (spread evenly across n_steps)
    for order in orders:
        order['step'] = rng.randint(0, n_steps)

    for regime_name, clob_fee, cpmm_fee, hfmm_fee in fee_regimes:
        half_v0 = v0 / 2.0

        # Pre-group orders by step
        step_orders = {}
        for order in orders:
            step_orders.setdefault(order['step'], []).append(order)

        for t in range(n_steps):
            P_mid = prices[t]

            # Fresh pools each step at breakeven V0, centered at P_mid
            # This matches the analytical model: pool is always at
            # equilibrium depth. Trades affect cost but not state.
            cpmm = CPMMPool(x=half_v0 / P_mid, y=half_v0, fee=cpmm_fee)
            hfmm_pool = HFMMPool(x=half_v0 / P_mid, y=half_v0,
                            A=50.0, fee=hfmm_fee, rate=P_mid)

            clob = _build_clob(P_mid, clob_half_spread_bps,
                               clob_depth, n_levels=10)

            for order in step_orders.get(t, []):
                side = order['side']
                Q_usd = order['quantity']
                bucket = order['bucket']
                Q = Q_usd / P_mid  # use current price for sizing

                # CLOB
                cq = clob.quote_buy(Q) if side == 'buy' else clob.quote_sell(Q)
                pe = cq['exec_price']
                if pe < 1e6:
                    sgn = 1.0 if side == 'buy' else -1.0
                    slip = sgn * (pe - P_mid) / P_mid * 10_000
                    cost = slip + clob_fee
                else:
                    slip = cost = float('inf')
                all_records.append(dict(
                    regime=regime_name, step=t, side=side,
                    Q_usd=Q_usd, bucket=bucket,
                    venue='CLOB', P_mid=P_mid,
                    cost_bps=cost, slippage_bps=slip, fee_bps=clob_fee))

                # CPMM
                aq = cpmm.quote_buy(Q, S_t=P_mid) if side == 'buy' \
                    else cpmm.quote_sell(Q, S_t=P_mid)
                all_records.append(dict(
                    regime=regime_name, step=t, side=side,
                    Q_usd=Q_usd, bucket=bucket,
                    venue='CPMM', P_mid=P_mid,
                    cost_bps=aq['cost_bps'], slippage_bps=aq['slippage_bps'],
                    fee_bps=aq['fee_bps']))

                # HFMM
                hq = hfmm_pool.quote_buy(Q, S_t=P_mid) if side == 'buy' \
                    else hfmm_pool.quote_sell(Q, S_t=P_mid)
                all_records.append(dict(
                    regime=regime_name, step=t, side=side,
                    Q_usd=Q_usd, bucket=bucket,
                    venue='HFMM', P_mid=P_mid,
                    cost_bps=hq['cost_bps'], slippage_bps=hq['slippage_bps'],
                    fee_bps=hq['fee_bps']))

    df = pd.DataFrame(all_records)
    df = df[df['cost_bps'] < 1e6].copy()
    df.to_csv(os.path.join(OUT_DIR, f'simulation_{pair.replace("/", "")}.csv'),
              index=False)

    # Aggregate & print
    for regime in df['regime'].unique():
        rsub = df[df['regime'] == regime]
        agg = rsub.groupby(['venue', 'bucket']).agg(
            avg_cost=('cost_bps', 'mean'),
            count=('cost_bps', 'count'),
        ).reset_index()

        print(f"\n  -- {regime} " + "-" * (70 - len(regime)))
        print(f"  {'$M':>5} | {'CLOB':>10} | {'CPMM':>10} | {'HFMM':>10}")
        print("  " + "-" * 45)
        for b in sorted(agg['bucket'].unique()):
            parts = []
            for v in ['CLOB', 'CPMM', 'HFMM']:
                row = agg[(agg['venue'] == v) & (agg['bucket'] == b)]
                if len(row) > 0:
                    parts.append(f"{row['avg_cost'].values[0]:>10.2f}")
                else:
                    parts.append(f"{'N/A':>10}")
            print(f"  {b:>5.0f} | {' | '.join(parts)}")

    _plot_simulation_curves(df, pair)

    return df


def _find_optimal_fee(d, eff_asc):
    """Find fee minimizing retail TC for a currency pair."""
    best_f = UNISWAP_FEE
    best_tc = float('inf')
    for f_bps in np.arange(0.5, 80, 0.5):
        f = f_bps / 10_000
        v0 = breakeven_v0(f, d['turnover_m'], eff_asc)
        tc = cpmm_total_cost_bps(v0, RETAIL_USD_M, fee=f)
        if tc < best_tc:
            best_tc = tc
            best_f = f
    return best_f


def _plot_simulation_curves(df, pair):
    """Cost curves from simulation by regime."""
    regimes = df['regime'].unique()
    n = len(regimes)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        rsub = df[df['regime'] == regime]
        agg = rsub.groupby(['venue', 'bucket'])['cost_bps'].mean().reset_index()

        for venue, color, marker in [('CLOB', C_CLOB, 'o'),
                                      ('CPMM', C_CPMM, 's'),
                                      ('HFMM', C_HFMM, '^')]:
            v = agg[agg['venue'] == venue].sort_values('bucket')
            if len(v) > 0:
                ax.plot(v['bucket'], v['cost_bps'], color=color,
                        marker=marker, linewidth=2, markersize=6,
                        label=venue)

        ax.set_xlabel('Trade Size ($M)')
        ax.set_title(regime, fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Avg Cost (bps)')
    fig.suptitle(f'Simulation: {pair} - Cost by Trade Size',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    fname = f'simulation_{pair.replace("/", "")}_curves.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


# =====================================================================
#  F. COMPETITIVE FRONTIER -- Can HFMM compete with CLOB?
# =====================================================================
#
#  Three mechanisms analyzed:
#    F1. Joint (fee, A) optimization with concentrated ASC model
#    F2. Dynamic fees -- regime-adaptive, median < mean
#    F3. Oracle/MEV reduction -- lower effective ASC
#
#  Key insight: HFMM at high A has ~0 slippage, so TC ≈ fee.
#  To compete with CLOB, fee must drop below CLOB spread.
#  But LP breakeven: V0 = f * Q_T / (opp + ASC_eff).
#  At low f with high ASC_eff -> small V0 -> more slippage.
#  The question: can concentrated liquidity + smart fee mechanics
#  close the gap?
# =====================================================================


def _hfmm_breakeven_v0(fee, turnover, base_asc, A, beta,
                       opp=OPP_COST_DAILY):
    """
    Breakeven V0 for HFMM LP accounting for concentrated ASC.

    LP in a StableSwap pool with amplification A has concentrated
    liquidity near the peg. For small price moves, impermanent loss
    scales roughly as A^beta relative to CPMM:
        ASC_eff = ASC_base * A^beta

    beta values:
        0   = ASC unaffected by concentration (optimistic)
        0.5 = sqrt scaling (moderate, empirical estimate)
        1.0 = linear scaling (conservative)
    """
    asc_eff = base_asc * (A ** beta)
    denom = opp + asc_eff
    if denom <= 0:
        return float('inf')
    return fee * turnover / denom


def run_competitive_frontier():
    """
    F. Competitive frontier analysis: three mechanisms to make
    HFMM competitive with CLOB.
    """
    print("\n" + "=" * 90)
    print("  F. COMPETITIVE FRONTIER -- Can HFMM compete with CLOB?")
    print("=" * 90)

    selected_pairs = ['USD/EUR', 'USD/GBP', 'USD/AUD', 'USD/JPY',
                      'USD/NOK', 'USD/TRY']
    selected_pairs = [p for p in selected_pairs if p in FENNELL_DATA]

    df_f1 = _run_f1_joint_optimization(selected_pairs)
    df_f2 = _run_f2_dynamic_fees(selected_pairs)
    df_f3 = _run_f3_oracle_reduction(selected_pairs)

    _plot_competitive_summary(df_f1, df_f2, df_f3, selected_pairs)

    return df_f1, df_f2, df_f3


# --- F1: Joint (fee, A) optimization ---

def _run_f1_joint_optimization(selected_pairs):
    """
    Sweep (fee, A) grid. For each combination, compute HFMM TC using
    endogenous V0 with concentrated ASC model.

    Three beta scenarios:
        beta=0   : optimistic (ASC unaffected by A)
        beta=0.5 : moderate
        beta=1.0 : conservative (ASC proportional to A)
    """
    print("\n  --- F1: Joint (fee, A) Optimization ---")

    fee_grid_bps = np.arange(0.1, 31.0, 0.1)
    A_grid = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
    betas = [0.0, 0.5, 1.0]

    rows = []

    for pair in selected_pairs:
        d = FENNELL_DATA[pair]
        base_asc = _implied_asc(d)

        for beta in betas:
            best_tc = float('inf')
            best_fee = 30.0
            best_A = 50

            for A in A_grid:
                for f_bps in fee_grid_bps:
                    f = f_bps / 10_000
                    v0 = _hfmm_breakeven_v0(f, d['turnover_m'],
                                            base_asc, A, beta)

                    if v0 <= 0 or RETAIL_USD_M >= v0 / 2:
                        continue

                    res = hfmm_cost_bps(v0, RETAIL_USD_M, fee=f, A=A)
                    tc = res['cost_bps']

                    if tc < best_tc:
                        best_tc = tc
                        best_fee = f_bps
                        best_A = A

                    rows.append(dict(
                        pair=pair, beta=beta, A=A,
                        fee_bps=f_bps, v0_m=v0,
                        tc_bps=tc, clob_bps=d['clob_retail'],
                        competitive=(tc <= d['clob_retail']),
                    ))

            competitive = "YES" if best_tc <= d['clob_retail'] else "no"
            print(f"  {pair:<10} beta={beta:.1f} | f*={best_fee:>5.1f} bps"
                  f"  A*={best_A:>5}  TC*={best_tc:>7.2f} bps"
                  f"  CLOB={d['clob_retail']:>6.3f} bps  [{competitive}]")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'competitive_f1_joint.csv'), index=False)

    # Plot heatmaps for EUR
    _plot_f1_heatmap(df, 'USD/EUR')

    return df


def _plot_f1_heatmap(df, pair):
    """Heatmap: TC(fee, A) for a given pair at different beta values."""
    sub = df[df['pair'] == pair]
    betas = sorted(sub['beta'].unique())
    n = len(betas)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    clob = FENNELL_DATA[pair]['clob_retail']

    for ax, beta in zip(axes, betas):
        bsub = sub[sub['beta'] == beta]
        pivot = bsub.pivot_table(index='A', columns='fee_bps',
                                 values='tc_bps', aggfunc='first')
        # Subsample columns for readability
        cols = [c for c in pivot.columns if c % 1.0 == 0 and c <= 30]
        if len(cols) > 30:
            cols = cols[::2]
        pivot_show = pivot[cols] if cols else pivot

        im = ax.imshow(pivot_show.values, aspect='auto',
                       cmap='RdYlGn_r',
                       vmin=0, vmax=max(50, clob * 5),
                       origin='lower')

        # Mark CLOB threshold
        ax.set_title(f'β={beta:.1f}  (CLOB={clob:.2f} bps)', fontsize=11)

        yticks = range(len(pivot_show.index))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(a)) for a in pivot_show.index],
                           fontsize=7)
        ax.set_ylabel('A (amplification)')

        n_xticks = min(10, len(cols))
        step = max(1, len(cols) // n_xticks)
        xtick_pos = list(range(0, len(cols), step))
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels([f'{cols[i]:.0f}' for i in xtick_pos],
                           fontsize=7, rotation=45)
        ax.set_xlabel('Fee (bps)')

    fig.colorbar(im, ax=axes, label='TC (bps)', shrink=0.8)
    fig.suptitle(f'{pair} - HFMM Total Cost by (Fee, A)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    fname = f'competitive_f1_heatmap_{pair.replace("/", "")}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


# --- F2: Dynamic fees ---

def _run_f2_dynamic_fees(selected_pairs):
    """
    Dynamic fee model: fee adapts to realized volatility.

    Model:
        - Proportion p_stable of trading days have low vol (σ < σ_avg)
        - Fee in stable: f_base
        - Fee in volatile: f_base × multiplier
        - LP revenue = E[fee] × Q_T = f_base × w_avg × Q_T
        - Median trader pays f_base (stable regime > 50% of time)

    Key insight: E[fee] > median(fee). LP earns average, trader
    typically pays median. This wedge is the mechanism.

    V0 = E[fee] × Q_T / (opp + ASC)   <- LP perspective
    TC_trader = f_base + slippage       <- trader in stable period
    """
    print("\n  --- F2: Dynamic Fees (Regime-Adaptive) ---")

    # Regime parameters
    p_stable = 0.70       # 70% of time is stable
    vol_mult = 3.0        # fee scales 3x in volatile periods
    w_avg = p_stable * 1.0 + (1 - p_stable) * vol_mult  # = 1.9

    base_fee_grid_bps = np.arange(0.1, 15.1, 0.1)
    A = 200  # high A for near-zero slippage

    rows = []

    print(f"\n  Model: p_stable={p_stable:.0%}, vol_mult={vol_mult:.0f}x"
          f" -> E[fee]/f_base = {w_avg:.2f}")
    print(f"  A = {A} (fixed, high concentration)")
    print(f"\n  {'Pair':<10} | {'f_base*':>7} {'TC_stable*':>10} {'TC_volatile':>11}"
          f" | {'CLOB':>6} | {'Competitive?':>12}")
    print("  " + "-" * 75)

    for pair in selected_pairs:
        d = FENNELL_DATA[pair]
        base_asc = _implied_asc(d)

        best_tc_stable = float('inf')
        best_f_base = 0
        best_tc_volatile = 0

        for f_base_bps in base_fee_grid_bps:
            f_base = f_base_bps / 10_000
            f_avg = f_base * w_avg

            # V0 from LP perspective (average fee revenue)
            v0 = breakeven_v0(f_avg, d['turnover_m'], base_asc)

            if v0 <= 0 or RETAIL_USD_M >= v0 / 2:
                continue

            # Trader cost in stable period (pays f_base)
            res_stable = hfmm_cost_bps(v0, RETAIL_USD_M, fee=f_base, A=A)
            # Trader cost in volatile period (pays f_base × vol_mult)
            f_vol = f_base * vol_mult
            res_volatile = hfmm_cost_bps(v0, RETAIL_USD_M, fee=f_vol, A=A)

            tc_s = res_stable['cost_bps']
            tc_v = res_volatile['cost_bps']

            rows.append(dict(
                pair=pair, f_base_bps=f_base_bps,
                f_avg_bps=f_base_bps * w_avg,
                v0_m=v0, A=A,
                tc_stable=tc_s, tc_volatile=tc_v,
                tc_weighted=p_stable * tc_s + (1 - p_stable) * tc_v,
                clob_bps=d['clob_retail'],
                competitive_stable=(tc_s <= d['clob_retail']),
            ))

            if tc_s < best_tc_stable:
                best_tc_stable = tc_s
                best_f_base = f_base_bps
                best_tc_volatile = tc_v

        comp = "YES" if best_tc_stable <= d['clob_retail'] else "no"
        print(f"  {pair:<10} | {best_f_base:>7.1f} {best_tc_stable:>10.2f}"
              f" {best_tc_volatile:>11.2f}"
              f" | {d['clob_retail']:>6.3f} | {comp:>12}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'competitive_f2_dynamic.csv'), index=False)

    _plot_f2_dynamic(df, selected_pairs)

    return df


def _plot_f2_dynamic(df, selected_pairs):
    """Plot dynamic fee curves: stable vs volatile cost by base fee."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.tab10
    for i, pair in enumerate(selected_pairs[:6]):
        sub = df[df['pair'] == pair]
        if len(sub) == 0:
            continue
        clob = sub['clob_bps'].values[0]
        label = pair.replace('USD/', '')

        axes[0].plot(sub['f_base_bps'], sub['tc_stable'],
                     color=cmap(i), linewidth=2, label=label)
        axes[0].axhline(clob, color=cmap(i), linestyle=':', alpha=0.4)

        axes[1].plot(sub['f_base_bps'], sub['tc_volatile'],
                     color=cmap(i), linewidth=2, label=label)
        axes[1].axhline(clob, color=cmap(i), linestyle=':', alpha=0.4)

    for ax, title in zip(axes, ['Stable Period (70%)', 'Volatile Period (30%)']):
        ax.set_xlabel('Base Fee (bps)', fontsize=11)
        ax.set_ylabel('TC (bps)', fontsize=11)
        ax.set_title(f'Dynamic Fees - {title}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 30)

    fig.tight_layout()
    fname = 'competitive_f2_dynamic.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


# --- F3: Oracle / MEV reduction ---

def _run_f3_oracle_reduction(selected_pairs):
    """
    If oracle feeds reduce effective ASC by factor kappa:
        ASC_eff = kappa * ASC_base,  kappa in (0, 1]

    This models:
        - Chainlink/Pyth oracle -> pool pre-adjusts before arb
        - MEV-capture mechanisms (MEV-Share, OFA)
        - Just-in-time liquidity
        - Batch auctions reducing information leakage

    At kappa -> 0: V0 -> f * Q_T / opp (only opportunity cost)
    Find kappa_threshold: the ASC reduction needed for HFMM < CLOB.
    """
    print("\n  --- F3: Oracle/MEV ASC Reduction ---")

    kappa_grid = np.concatenate([
        np.arange(0.01, 0.10, 0.01),
        np.arange(0.10, 1.01, 0.05),
    ])
    A = 200
    fee_bps = 0.5  # very low fee

    rows = []

    print(f"\n  Fixed: A={A}, fee={fee_bps} bps")
    print(f"\n  {'Pair':<10} | {'κ_threshold':>11} | {'V0 at κ_thr':>11}"
          f" | {'TC at κ=0.01':>12} | {'CLOB':>6}")
    print("  " + "-" * 70)

    for pair in selected_pairs:
        d = FENNELL_DATA[pair]
        base_asc = _implied_asc(d)
        f = fee_bps / 10_000

        kappa_thr = None
        v0_at_thr = None

        for kappa in kappa_grid:
            asc_eff = kappa * base_asc
            v0 = breakeven_v0(f, d['turnover_m'], asc_eff)

            if v0 <= 0 or RETAIL_USD_M >= v0 / 2:
                tc = float('inf')
            else:
                res = hfmm_cost_bps(v0, RETAIL_USD_M, fee=f, A=A)
                tc = res['cost_bps']

            rows.append(dict(
                pair=pair, kappa=round(kappa, 3),
                asc_eff=asc_eff, v0_m=v0, tc_bps=tc,
                fee_bps=fee_bps, A=A,
                clob_bps=d['clob_retail'],
                competitive=(tc <= d['clob_retail']),
            ))

            if kappa_thr is None and tc <= d['clob_retail']:
                kappa_thr = kappa
                v0_at_thr = v0

        # TC at most aggressive kappa
        tc_best = rows[-1]['tc_bps'] if rows else float('inf')
        # Actually get kappa=0.01
        best_row = [r for r in rows if r['pair'] == pair
                    and abs(r['kappa'] - 0.01) < 0.005]
        tc_01 = best_row[0]['tc_bps'] if best_row else float('inf')

        k_str = f"{kappa_thr:.2f}" if kappa_thr else "N/A"
        v_str = f"${v0_at_thr:,.0f}M" if v0_at_thr else "N/A"
        print(f"  {pair:<10} | {k_str:>11} | {v_str:>11}"
              f" | {tc_01:>12.2f} | {d['clob_retail']:>6.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'competitive_f3_oracle.csv'), index=False)

    _plot_f3_oracle(df, selected_pairs)

    # Also do a joint sweep: kappa × fee for EUR
    _run_f3_joint_kappa_fee(selected_pairs[:3])

    return df


def _run_f3_joint_kappa_fee(pairs):
    """Joint kappa × fee sweep for selected pairs."""
    print("\n  F3b: Joint κ × fee sweep")

    kappa_grid = np.arange(0.01, 0.51, 0.01)
    fee_grid_bps = np.arange(0.1, 5.1, 0.1)
    A = 200

    rows = []
    for pair in pairs:
        d = FENNELL_DATA[pair]
        base_asc = _implied_asc(d)

        best_tc = float('inf')
        best_k = 1.0
        best_f = 5.0

        for kappa in kappa_grid:
            for f_bps in fee_grid_bps:
                f = f_bps / 10_000
                asc_eff = kappa * base_asc
                v0 = breakeven_v0(f, d['turnover_m'], asc_eff)

                if v0 <= 0 or RETAIL_USD_M >= v0 / 2:
                    continue

                res = hfmm_cost_bps(v0, RETAIL_USD_M, fee=f, A=A)
                tc = res['cost_bps']

                rows.append(dict(pair=pair, kappa=round(kappa, 2),
                                 fee_bps=f_bps, tc_bps=tc,
                                 clob_bps=d['clob_retail']))

                if tc < best_tc:
                    best_tc = tc
                    best_k = kappa
                    best_f = f_bps

        comp = "YES" if best_tc <= d['clob_retail'] else "no"
        print(f"  {pair:<10} κ*={best_k:.2f} f*={best_f:.1f} bps"
              f" -> TC*={best_tc:.2f} bps  CLOB={d['clob_retail']:.3f} [{comp}]")

    if rows:
        df_joint = pd.DataFrame(rows)
        df_joint.to_csv(os.path.join(OUT_DIR, 'competitive_f3_joint.csv'),
                        index=False)

        # Heatmap for EUR
        eur = df_joint[df_joint['pair'] == pairs[0]]
        if len(eur) > 0:
            _plot_f3_heatmap(eur, pairs[0])


def _plot_f3_oracle(df, selected_pairs):
    """Plot TC vs kappa for selected pairs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for i, pair in enumerate(selected_pairs):
        sub = df[df['pair'] == pair].copy()
        sub = sub[sub['tc_bps'] < 100]
        if len(sub) == 0:
            continue
        clob = sub['clob_bps'].values[0]
        ax.plot(sub['kappa'], sub['tc_bps'], color=cmap(i),
                linewidth=2, label=pair.replace('USD/', ''))
        ax.axhline(clob, color=cmap(i), linestyle=':', alpha=0.3)

    ax.set_xlabel('κ (ASC reduction factor)', fontsize=12)
    ax.set_ylabel('HFMM Total Cost (bps)', fontsize=12)
    ax.set_title('Oracle/MEV Reduction: TC vs ASC Factor κ\n'
                 f'(A={df["A"].values[0]}, fee={df["fee_bps"].values[0]} bps,'
                 f' dotted = CLOB cost)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    fig.tight_layout()

    fname = 'competitive_f3_oracle.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


def _plot_f3_heatmap(df_eur, pair):
    """Heatmap: TC(kappa, fee) for a given pair."""
    pivot = df_eur.pivot_table(index='kappa', columns='fee_bps',
                               values='tc_bps', aggfunc='first')

    clob = df_eur['clob_bps'].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=max(5, clob * 3),
                   origin='lower', interpolation='nearest')

    # Axes
    n_yticks = min(10, len(pivot.index))
    ystep = max(1, len(pivot.index) // n_yticks)
    ytick_pos = list(range(0, len(pivot.index), ystep))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f'{pivot.index[i]:.2f}' for i in ytick_pos],
                       fontsize=8)
    ax.set_ylabel('κ (ASC reduction)')

    n_xticks = min(10, len(pivot.columns))
    xstep = max(1, len(pivot.columns) // n_xticks)
    xtick_pos = list(range(0, len(pivot.columns), xstep))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f'{pivot.columns[i]:.1f}' for i in xtick_pos],
                       fontsize=8)
    ax.set_xlabel('Fee (bps)')

    # Contour at CLOB level
    try:
        cs = ax.contour(pivot.values, levels=[clob], colors='white',
                        linewidths=2, linestyles='--')
        ax.clabel(cs, fmt=f'CLOB={clob:.2f}', fontsize=9, colors='white')
    except Exception:
        pass

    fig.colorbar(im, ax=ax, label='HFMM TC (bps)')
    ax.set_title(f'{pair} - HFMM Cost: κ × Fee  (A=200)\n'
                 f'White contour = CLOB breakeven ({clob:.3f} bps)',
                 fontsize=12)
    fig.tight_layout()

    fname = f'competitive_f3_heatmap_{pair.replace("/", "")}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


def _plot_competitive_summary(df_f1, df_f2, df_f3, selected_pairs):
    """Summary bar chart: best achievable TC under each mechanism vs CLOB."""
    summary_rows = []

    for pair in selected_pairs:
        d = FENNELL_DATA[pair]

        # F1 best (optimistic, beta=0)
        f1 = df_f1[(df_f1['pair'] == pair) & (df_f1['beta'] == 0.0)]
        f1_best = f1['tc_bps'].min() if len(f1) > 0 else float('inf')

        # F1 moderate (beta=0.5)
        f1m = df_f1[(df_f1['pair'] == pair) & (df_f1['beta'] == 0.5)]
        f1m_best = f1m['tc_bps'].min() if len(f1m) > 0 else float('inf')

        # F2 best (stable period cost)
        f2 = df_f2[df_f2['pair'] == pair]
        f2_best = f2['tc_stable'].min() if len(f2) > 0 else float('inf')

        # F3 best (most aggressive kappa)
        f3 = df_f3[df_f3['pair'] == pair]
        f3_best = f3['tc_bps'].min() if len(f3) > 0 else float('inf')

        # CPMM optimal from section C
        v0_c = breakeven_v0(0.003, d['turnover_m'], _implied_asc(d))
        cpmm_30 = cpmm_total_cost_bps(v0_c, RETAIL_USD_M)

        summary_rows.append(dict(
            pair=pair, clob=d['clob_retail'],
            cpmm_30bps=cpmm_30,
            f1_optimistic=f1_best, f1_moderate=f1m_best,
            f2_dynamic=f2_best, f3_oracle=f3_best,
        ))

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUT_DIR, 'competitive_summary.csv'),
                  index=False)

    # Print summary table
    print("\n" + "=" * 90)
    print("  COMPETITIVE SUMMARY -- Best achievable TC (retail, bps)")
    print("=" * 90)
    print(f"  {'Pair':<10} | {'CLOB':>6} | {'CPMM':>6} |"
          f" {'F1 opt':>7} {'F1 mod':>7} | {'F2 dyn':>7} | {'F3 orc':>7}")
    print("  " + "-" * 72)
    for _, r in df_sum.iterrows():
        markers = []
        if r['f1_optimistic'] <= r['clob']:
            markers.append('F1o')
        if r['f2_dynamic'] <= r['clob']:
            markers.append('F2')
        if r['f3_oracle'] <= r['clob']:
            markers.append('F3')
        tag = " <-- " + "+".join(markers) if markers else ""
        print(f"  {r['pair']:<10} | {r['clob']:>6.3f} | {r['cpmm_30bps']:>6.1f} |"
              f" {r['f1_optimistic']:>7.2f} {r['f1_moderate']:>7.2f} |"
              f" {r['f2_dynamic']:>7.2f} | {r['f3_oracle']:>7.2f}{tag}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    pairs = df_sum['pair'].values
    x = np.arange(len(pairs))
    w = 0.13

    bars = [
        ('clob',          'CLOB',            C_CLOB),
        ('cpmm_30bps',    'CPMM 30bps',      C_CPMM),
        ('f1_optimistic', 'F1: (f,A)* β=0',  '#66BB6A'),
        ('f1_moderate',   'F1: (f,A)* β=0.5', '#2E7D32'),
        ('f2_dynamic',    'F2: Dynamic fee',  '#AB47BC'),
        ('f3_oracle',     'F3: Oracle κ=0.01','#EF5350'),
    ]

    for j, (col, lbl, clr) in enumerate(bars):
        vals = df_sum[col].values.copy()
        vals = np.clip(vals, 0, 60)
        ax.bar(x + (j - len(bars) / 2) * w, vals, w,
               color=clr, label=lbl, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('USD/', '') for p in pairs],
                       fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Transaction Cost (bps)', fontsize=11)
    ax.set_title('Competitive Frontier: Can HFMM Match CLOB?', fontsize=13)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 60)
    fig.tight_layout()

    fname = 'competitive_summary.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {fname}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 90)
    print("  FENNELL TESTS - Full Replication + HFMM Extension")
    print("  James Fennell, 'Are AMMs the Future of FX?', UTS 2024")
    print("=" * 90)

    # A. Analytical replication with Fennell published data
    df_anal = run_analytical_replication()

    # B. Partial adoption sweep
    df_adopt = run_partial_adoption()

    # C. Fee optimization with endogenous V0
    df_fee, opt_results = run_fee_optimization()

    # D. HFMM extension (StableSwap comparison)
    df_hfmm = run_hfmm_extension()

    # E. Simulation-based test for EUR/USD (most liquid)
    df_sim_eur = run_simulation_test('USD/EUR', n_steps=500,
                                     trades_per_bucket=60)

    # E2. Simulation for an exotic pair
    df_sim_try = run_simulation_test('USD/TRY', n_steps=500,
                                     trades_per_bucket=40)

    # F. Competitive frontier -- can HFMM match CLOB?
    df_f1, df_f2, df_f3 = run_competitive_frontier()

    # Summary
    print("\n" + "=" * 90)
    print("  DONE - All results saved to output/fennell/")
    print("=" * 90)
    print("  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if not f.startswith('.'):
            print(f"    {f}")
    print()


if __name__ == '__main__':
    main()
