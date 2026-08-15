#!/usr/bin/env python3
"""N1b: realised liquidity provider result for the automated backstop, calm
against crisis, over fees of 5, 10 and 20 basis points, on a short seed list.

The theoretical bound in ``n1_lvr_bound.py`` is frictionless. Arbitrage is
capped here, so what providers actually lose is the realised counterpart, and
what they earn is the fee income the pool skims. Fees are held apart from the
reserves rather than compounded into them, so there is no double count.

The measurement lives in ``lp_pnl_corrected`` and is imported rather than
repeated. The version that used to sit here compared terminal reserves against
a static endpoint benchmark, left capital flows in the figure, paired each
price with the reserves of the previous period and looked only at the hybrid
function pool. Every published number it produced was wrong, so the function
was removed rather than adjusted.

Each fee also carries the peak spread reduction against the arm with no
automated venue, since a higher fee routes less flow to the pool and weakens
the very result the pool delivers, which is the trade off the paper is about.

Windows are calm = [shock - 150, shock - 50] and crisis = [shock, shock + 100].
Results are a percentage of the value in the pools at the start of the window.

The full three hundred seed version is ``n1b_lp_pnl_300.py``."""
import sys, numpy as np, statistics as st
sys.path.insert(0, __import__('os').path.abspath(
    __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')))
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from tools.robustness.lp_pnl_corrected import components
try: from main import _primary_run_label
except Exception: _primary_run_label = lambda a: 'dlc'

PRESET = 'dealer_liquidity_crisis'; N_ITER = 1000; SEEDS = list(range(42, 42 + 20))
FEES_BPS = [5, 10, 20]
CALM = (-150, -50); CRISIS = (0, 100)
H = 250


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


lines = ['N1b realised provider result, both pools, %d seeds. Net = fee income'
         % len(SEEDS),
         'less loss, per cent of the value in the pools at the start of the window.',
         'Loss is measured period by period against the reserves carried into that',
         'period, with capital flows netted out at the reference price. Backstop',
         'efficacy = peak spread reduction against the arm with no automated venue.', '']

for bps in FEES_BPS:
    fee_frac = bps / 1e4
    calm_il, calm_fee, calm_net = [], [], []
    cri_il, cri_fee, cri_net = [], [], []
    pk_red = []
    for sd in SEEDS:
        sim, sh = run(sd, True, fee_frac)
        pools = list(sim.amm_pools.values())
        fp = np.asarray(sim.logger.fair_price_series, dtype=float)
        rc = lp_pnl_window(pools, fp, sh, CALM)
        rd = lp_pnl_window(pools, fp, sh, CRISIS)
        if rc: calm_il.append(rc[0]); calm_fee.append(rc[1]); calm_net.append(rc[2])
        if rd: cri_il.append(rd[0]); cri_fee.append(rd[1]); cri_net.append(rd[2])
        pk_with = peak(sim, sh)
        sim0, sh0 = run(sd, False, fee_frac)
        pk_wo = peak(sim0, sh0)
        if np.isfinite(pk_with) and np.isfinite(pk_wo) and pk_wo > 0:
            pk_red.append(100 * (pk_wo - pk_with) / pk_wo)
    med = lambda v: float(np.median(v)) if v else float('nan')
    lines.append(f'--- fee = {bps} bps ---')
    lines.append(f'  CALM    : IL={med(calm_il):+6.3f}%  fees={med(calm_fee):+6.3f}%  net={med(calm_net):+6.3f}%  (of pool value / 100 ticks)')
    lines.append(f'  CRISIS  : IL={med(cri_il):+6.3f}%  fees={med(cri_fee):+6.3f}%  net={med(cri_net):+6.3f}%  (of pool value / 100 ticks)')
    lines.append(f'  backstop efficacy (peak-spread reduction vs no-AMM): {med(pk_red):.1f}%')
    lines.append('')
    print('\n'.join(lines[-5:]), flush=True)

txt = '\n'.join(lines)
open('output/resilience/robustness_n1b_lp_pnl.txt', 'w').write(txt + '\n')
print('\nsaved output/resilience/robustness_n1b_lp_pnl.txt')
