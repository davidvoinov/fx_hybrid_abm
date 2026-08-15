#!/usr/bin/env python3
"""N1b (300 seeds): realised liquidity provider result for the automated
backstop, calm against crisis, over fees of 5, 10 and 20 basis points, with
bootstrap confidence intervals on the median.

The measurement itself lives in ``lp_pnl_corrected`` and is imported rather
than repeated. The version that used to sit here compared terminal reserves
against a static endpoint benchmark, left capital flows in the figure, paired
each price with the reserves of the previous period and looked only at the
hybrid function pool. Every published number it produced was wrong, so the
function was removed rather than adjusted. What this file still contributes is
the harness, three hundred paired seeds evaluated across cores, and the peak
spread reduction against the arm with no automated venue.

Windows are calm = [shock - 150, shock - 50] and crisis = [shock, shock + 100].
Results are a percentage of the value in the pool at the start of the window."""
import sys, os, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __import__('os').path.abspath(
    __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')))
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label = lambda a: 'dlc'

from tools.robustness.lp_pnl_corrected import components

PRESET = 'dealer_liquidity_crisis'; N_ITER = 1000
SEEDS = list(range(42, 42 + 300))
FEES_BPS = [5, 10, 20]
CALM = (-150, -50); CRISIS = (0, 100)
H = 250
N_WORKERS = min(10, os.cpu_count() or 4)


def run(seed, amm, fee_frac):
    argv = ['--preset', PRESET, '--seed', str(seed), '--n-iter', str(N_ITER), '--silent',
            '--hfmm-fee', str(fee_frac)]
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv); _auto_stress_around_shock(a)
    a.run_label = _primary_run_label(a); a.enable_amm = 1 if amm else 0
    if not amm: a.amm_share_pct = 0
    a.clob_amm_interaction = 'competition'
    _seed_all(seed); sim = build_sim(a); sim.simulate(a.n_iter, silent=True)
    return sim, int(a.shock_iter)


def peak(sim, sh):
    s = np.asarray(sim.logger.clob_qspr, dtype=float)
    return float(np.nanmax(s[sh:sh + H])) if s.size >= sh + H else float('nan')


def lp_pnl_window(pools, p, sh, win):
    """Loss, fees and net across every pool in the arm, as percentages.

    Summing the two results before dividing weights each pool by the capital
    actually in it. Averaging the two percentages would weight them equally
    however the capital happens to be split.
    """
    lvr = fees = v0 = 0.0
    for pool in pools:
        c = components(pool, p, sh, win)
        if c is None:
            return None
        lvr += c[0]; fees += c[1]; v0 += c[2]
    if v0 <= 0:
        return None
    return 100 * lvr / v0, 100 * fees / v0, 100 * (fees - lvr) / v0


def eval_seed(seed):
    """One seed: no-AMM peak once, then with-AMM P&L + peak per fee.
    Returns {bps: (calm_net, crisis_net, peak_reduction)}; None entries on bad window."""
    sim0, sh0 = run(seed, False, FEES_BPS[0] / 1e4)
    pk_wo = peak(sim0, sh0)
    out = {}
    for bps in FEES_BPS:
        sim, sh = run(seed, True, bps / 1e4)
        pools = list(sim.amm_pools.values())
        fp = np.asarray(sim.logger.fair_price_series, dtype=float)
        rc = lp_pnl_window(pools, fp, sh, CALM)
        rd = lp_pnl_window(pools, fp, sh, CRISIS)
        pk_with = peak(sim, sh)
        pkred = (100 * (pk_wo - pk_with) / pk_wo
                 if np.isfinite(pk_with) and np.isfinite(pk_wo) and pk_wo > 0 else np.nan)
        out[bps] = (rc[2] if rc else np.nan,        # calm net
                    rd[2] if rd else np.nan,         # crisis net
                    rc[0] if rc else np.nan,         # calm loss
                    rc[1] if rc else np.nan,         # calm fee
                    rd[0] if rd else np.nan,         # crisis loss
                    rd[1] if rd else np.nan,         # crisis fee
                    pkred)
    return out


def boot_ci(v, fn=np.median, n=5000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    bs = fn(v[idx], axis=1)
    return float(fn(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == '__main__':
    print(f'N1b 300-seed LP P&L: {len(SEEDS)} seeds x {len(FEES_BPS)} fees, '
          f'{N_WORKERS} workers...', flush=True)
    with Pool(N_WORKERS) as pool:
        results = pool.map(eval_seed, SEEDS)

    lines = [f'N1b realised provider result, both pools, {len(SEEDS)} seeds, '
             'bootstrap 95 per cent interval on the median.',
             'Net = fee income less loss, per cent of the value in the pools at '
             'the start of the window.',
             'Loss is measured period by period against the reserves carried '
             'into that period, with capital flows netted out at the reference '
             'price. It is the realised counterpart of loss versus '
             'rebalancing under a capped arbitrage, not the frictionless bound.',
             'Backstop efficacy = peak spread reduction against the arm with no '
             'automated venue.', '']
    for bps in FEES_BPS:
        calm_net = [r[bps][0] for r in results]
        cri_net  = [r[bps][1] for r in results]
        calm_il  = [r[bps][2] for r in results]
        calm_fee = [r[bps][3] for r in results]
        cri_il   = [r[bps][4] for r in results]
        cri_fee  = [r[bps][5] for r in results]
        pkred    = [r[bps][6] for r in results]
        cn = boot_ci(calm_net, seed=bps); rn = boot_ci(cri_net, seed=bps + 1)
        pk = boot_ci(pkred, seed=bps + 2)
        nseed = sum(np.isfinite(cri_net))
        lines.append(f'--- fee = {bps} bps  (n={nseed} valid seeds) ---')
        lines.append(f'  CALM  : loss={np.nanmedian(calm_il):+6.3f}%  fees={np.nanmedian(calm_fee):+6.3f}%  '
                     f'net={cn[0]:+6.3f}%  [95% CI {cn[1]:+.3f}, {cn[2]:+.3f}]')
        lines.append(f'  CRISIS: loss={np.nanmedian(cri_il):+6.3f}%  fees={np.nanmedian(cri_fee):+6.3f}%  '
                     f'net={rn[0]:+6.3f}%  [95% CI {rn[1]:+.3f}, {rn[2]:+.3f}]')
        lines.append(f'  backstop efficacy (peak-spread reduction vs no-AMM): '
                     f'{pk[0]:.1f}%  [95% CI {pk[1]:.1f}, {pk[2]:.1f}]')
        lines.append('')
    txt = '\n'.join(lines)
    print('\n' + txt, flush=True)
    open('output/resilience/robustness_n1b_lp_pnl_300.txt', 'w').write(txt + '\n')
    print('saved output/resilience/robustness_n1b_lp_pnl_300.txt', flush=True)
