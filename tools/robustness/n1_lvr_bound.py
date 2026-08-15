#!/usr/bin/env python3
"""N1 (panel #2): back-of-envelope LP loss-versus-rebalancing (LVR) bound for the
automated venue over the dealer-liquidity-crisis window, against fee income.

The backstop benefits takers (tighter spreads); its cost falls on the pool's
liquidity providers as LVR, which peaks exactly in the DLC (arbitrage re-anchors
the lagged pool mid most aggressively then). We bound that cost with the canonical
constant-product CFMM rate of Milionis-Moallemi-Roughgarden-Zhang (2022):

    LVR rate per tick  =  (sigma_t^2 / 8) * V_t        V_t = pool value (quote units)

and compare cumulative LVR over the shock window to cumulative fee income at the
5 bps pool fee (fee_t = f * AMM volume_t). This is an order-of-magnitude check to
discipline the welfare language, not a precise LP P&L; StableSwap amplification
redistributes LVR (lower far from peg, higher near peg) around this benchmark.
Reported: implied break-even fee (LVR / AMM volume, bps) vs the 5 bps charged."""
import sys, numpy as np, statistics as st
sys.path.insert(0, __import__('os').path.abspath(
    __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')))
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from AgentBasedModel.venues.amm import HFMMPool, _hfmm_mid_price, _hfmm_get_y
try: from main import _primary_run_label
except Exception: _primary_run_label = lambda a: 'dlc'

A_HFMM = 18.0   # benchmark amplification (calibration/primary_model.json)

def pool_lvr_rate(xt, yt, rate, sigma):
    """Pool-specific Milionis LVR rate (1/2) sigma^2 p^2 |dx/dp| for the StableSwap
    curve at the logged reserves, via finite difference along the invariant.
    Falls back to the constant-product rate if the curve solve fails (off-peg)."""
    try:
        p = HFMMPool(xt, yt, A=A_HFMM, fee=FEE, rate=rate)
        p0 = p.mid_price(); D = p.D
        dx = xt * 1e-4
        xn2 = (xt + dx) * rate
        yn2 = _hfmm_get_y(xn2, D, A_HFMM)
        p1 = _hfmm_mid_price(xn2, yn2, A_HFMM, D) * rate
        if not np.isfinite(p1) or p1 == p0: raise ValueError
        dxdp = abs(dx / (p1 - p0))
        return 0.5 * sigma ** 2 * p0 ** 2 * dxdp
    except Exception:
        V = xt * rate + yt
        return sigma ** 2 / 8.0 * V

PRESET = 'dealer_liquidity_crisis'; N_ITER = 1000; SEEDS = list(range(42, 42 + 24))
W = 100           # shock-window length (ticks) over which LVR/fees accumulate
FEE = 5e-4        # 5 bps pool fee
# The simulated price process scales its innovations by this factor, so the
# volatility that actually drives the reserves is PRICE_VOL_SCALE * sigma.
# Feeding the unscaled sigma into a variance based loss overstates it by the
# square of the reciprocal, which is more than eight fold at the shipped value.
PRICE_VOL_SCALE = 0.35


def run(seed):
    argv = ['--preset', PRESET, '--seed', str(seed), '--n-iter', str(N_ITER), '--silent']
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv); _auto_stress_around_shock(a)
    a.run_label = _primary_run_label(a); a.enable_amm = 1
    a.clob_amm_interaction = 'competition'
    _seed_all(seed); sim = build_sim(a); sim.simulate(a.n_iter, silent=True)
    return sim.logger, int(a.shock_iter)


def amm_key(log):
    # prefer the hybrid pool; fall back to whatever AMM series exist
    keys = list(log.amm_x_series.keys())
    for pref in ('HFMM', 'hfmm', 'StableSwap', 'stableswap'):
        if pref in keys: return pref
    return keys[0] if keys else None


def window_lvr_fees(log, sh):
    sig = np.asarray(log.sigma_series, dtype=float)
    fair = np.asarray(log.fair_price_series, dtype=float)   # S_t = governance peg rate
    k = amm_key(log)
    if k is None: return None
    x = np.asarray(log.amm_x_series[k], dtype=float)
    y = np.asarray(log.amm_y_series[k], dtype=float)
    mid = np.asarray(log.amm_mid_series[k], dtype=float)
    fv = None
    for vk in ('amm', 'AMM', k):
        if vk in log.flow_volume:
            fv = np.asarray(log.flow_volume[vk], dtype=float); break
    if fv is None: fv = np.zeros_like(x)
    n = min(len(sig), len(x), len(y), len(mid), len(fv), len(fair))
    lo, hi = sh, min(sh + W, n)
    # Effective volatility of the simulated process, not the nominal sigma.
    s = PRICE_VOL_SCALE * sig[lo:hi]
    V = x[lo:hi] * mid[lo:hi] + y[lo:hi]; vol = fv[lo:hi]
    lvr_cp = float(np.sum((s ** 2) / 8.0 * V))        # constant-product reference, quote units
    # pool-specific StableSwap LVR per tick from the actual curve
    lvr_pool = 0.0
    for i in range(lo, hi):
        rate = fair[i] if (np.isfinite(fair[i]) and fair[i] > 0) else mid[i]
        lvr_pool += pool_lvr_rate(x[i], y[i], rate, PRICE_VOL_SCALE * sig[i])
    # Flow volume is recorded as a traded base quantity, while the loss above is
    # in quote units. Multiplying by the price puts the two on the same footing.
    # Without it the fee is understated by the price level, a hundredfold here.
    fees = float(np.sum(FEE * vol * mid[lo:hi]))
    amm_vol = float(np.sum(vol))
    Vbar = float(np.nanmean(V)); sbar = float(np.nanmean(s))
    return lvr_cp, lvr_pool, fees, amm_vol, Vbar, sbar


rows = []
for sd in SEEDS:
    log, sh = run(sd)
    r = window_lvr_fees(log, sh)
    if r is None: continue
    lvr_cp, lvr_pool, fees, amm_vol, Vbar, sbar = r
    cover = (fees / lvr_pool) if lvr_pool > 0 else float('nan')
    rows.append((lvr_cp, lvr_pool, fees, amm_vol, Vbar, sbar, cover))
    print(f'seed {sd}: LVRcp={lvr_cp:8.1f} LVRpool={lvr_pool:9.1f} fees={fees:6.2f} '
          f'ammVol={amm_vol:8.1f} Vbar={Vbar:9.1f} sig={sbar:.4f} cover={cover:.4f}',
          flush=True)

if rows:
    a = np.array(rows)
    med_cp = float(np.median(a[:, 0])); med_pool = float(np.median(a[:, 1]))
    med_fee = float(np.median(a[:, 2])); med_V = float(np.median(a[:, 4]))
    cp_pct = med_cp / med_V * 100.0              # constant-product LVR as % pool value
    pool_pct = med_pool / med_V * 100.0          # StableSwap LVR as % pool value
    cov_cp = med_fee / med_cp * 100.0            # fee as % of CP LVR
    cov_pool = med_fee / med_pool * 100.0        # fee as % of pool-specific LVR
    ampl = med_pool / med_cp                      # amplification of LVR vs constant product
    out = [
        f'N1 LVR bound vs fee income, DLC shock window (W={W} ticks, {len(rows)} seeds, fee={FEE*1e4:.0f}bps).',
        'Two LVR measures (Milionis et al. 2022):',
        '  (a) constant-product reference  : (sigma^2/8) * pool_value per tick.',
        f'  (b) pool-specific StableSwap   : (1/2) sigma^2 p^2 |dx/dp| from the actual A={A_HFMM:.0f} curve.',
        'Reported as a share of pool value (scale-robust: pool is sized for depth, window',
        'turnover is small relative to reserves, so a per-trade break-even fee is uninformative).',
        '',
        f'  mean pool value V (median, notional)        : {med_V:.0f}',
        f'  AMM window volume (median, notional)        : {np.median(a[:,3]):.0f}',
        f'  mean sigma in window (median)               : {np.median(a[:,5]):.4f}',
        '',
        f'  (a) constant-product LVR  (% pool value)    : {cp_pct:.2f}%   fee covers {cov_cp:.2f}%',
        f'  (b) StableSwap pool LVR   (% pool value)    : {pool_pct:.2f}%   fee covers {cov_pool:.3f}%',
        f'  amplification of LVR (pool / constant-prod) : {ampl:.0f}x',
        '',
        f'  => The amplified (A={A_HFMM:.0f}) StableSwap curve concentrates liquidity near the peg, so the',
        f'     actual LP loss-versus-rebalancing over the crisis window is ~{pool_pct:.0f}% of pool value,',
        f'     about {ampl:.0f}x the constant-product reference and far above the ~{cov_pool:.2f}% offset by the',
        f'     {FEE*1e4:.0f}bps fee. The backstop transfers value from liquidity providers to takers precisely',
        '     in the crisis it cushions, so welfare/policy claims must be conditional on LP compensation.',
    ]
    txt = '\n'.join(out)
    print('\n' + txt)
    open('output/resilience/robustness_n1_lvr_bound.txt', 'w').write(txt + '\n')
    print('\nsaved output/resilience/robustness_n1_lvr_bound.txt')
