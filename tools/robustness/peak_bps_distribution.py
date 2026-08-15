#!/usr/bin/env python3
"""Per-seed peak spread in bps (DLC, competition mode = headline), with vs without AMM.
Reports mean/median/p90 and multiple-over-calm, to replace the '% over calm' framing."""
import os
import sys, numpy as np, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
from main import (build_parser,_apply_preset_defaults,_resolve_main_routing,
                  _auto_stress_around_shock,_seed_all,build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label=lambda a:'dlc'
PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+120)); H=250
def series(seed, amm):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1 if amm else 0
    if not amm: a.amm_share_pct=0
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    return np.asarray(sim.logger.clob_qspr,dtype=float), int(a.shock_iter)
def stats(amm):
    peaks=[]; bases=[]; mult=[]
    for sd in SEEDS:
        s,sh=series(sd,amm)
        if s.size<sh+H: continue
        base=float(np.nanmean(s[sh-45:sh-5])); pk=float(np.nanmax(s[sh:sh+H]))
        peaks.append(pk); bases.append(base); mult.append(pk/base if base>0 else np.nan)
    peaks=np.array(peaks); mult=np.array(mult)
    return dict(base=float(np.mean(bases)), pk_mean=float(np.mean(peaks)),
               pk_med=float(np.median(peaks)), pk_p90=float(np.percentile(peaks,90)),
               x_mean=float(np.nanmean(mult)), x_med=float(np.nanmedian(mult)))
for amm,lbl in [(True,'WITH AMM'),(False,'WITHOUT AMM')]:
    d=stats(amm)
    print(f'{lbl:12s}: calm={d["base"]:.2f}bps  peak mean={d["pk_mean"]:.1f} median={d["pk_med"]:.1f} p90={d["pk_p90"]:.1f} bps | x-over-calm mean={d["x_mean"]:.1f} median={d["x_med"]:.1f}')
