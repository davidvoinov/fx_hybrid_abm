#!/usr/bin/env python3
"""Mean quoted-spread trajectory around the DLC shock, with vs without AMM."""
import os
import sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AgentBasedModel.visualization.paper_style import use_paper_style
use_paper_style()
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
try:
    from main import _primary_run_label
except Exception:
    _primary_run_label = lambda a: 'dlc'

PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+120))

def run(seed, enable_amm):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1 if enable_amm else 0
    if not enable_amm: a.amm_share_pct=0
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    return np.asarray(sim.info.spreads if hasattr(sim,'info') and getattr(sim,'info',None) else [], dtype=float), int(a.shock_iter)

# collect quoted spread series via logger
def series(seed, enable_amm):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1 if enable_amm else 0
    if not enable_amm: a.amm_share_pct=0
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    log=sim.logger
    s=getattr(log,'clob_qspr',None)
    return np.asarray(s,dtype=float), int(a.shock_iter)

def collect(enable_amm):
    arrs=[]; shock=350
    for sd in SEEDS:
        s,shock=series(sd,enable_amm)
        if s.size: arrs.append(s)
    L=min(len(a) for a in arrs); M=np.vstack([a[:L] for a in arrs])
    return np.nanmean(M,axis=0), shock

mean_w,shock=collect(True); mean_o,_=collect(False)
lo=max(0,shock-50); hi=min(len(mean_w),shock+250)
x=np.arange(lo,hi)-shock
fig,ax=plt.subplots(figsize=(6.2,3.4))
ax.plot(x,mean_o[lo:hi],label='Without AMM (dealer CLOB only)',color='#c0392b',lw=1.8)
ax.plot(x,mean_w[lo:hi],label='With AMM (hybrid)',color='#2c3e50',lw=1.8)
ax.axvline(0,color='grey',ls=':',lw=1)
ax.set_xlabel('Event time relative to shock (ticks)')
ax.set_ylabel('Mean quoted spread (bps)')
ax.legend(frameon=False,fontsize=9)
fig.tight_layout()
out='output/resilience/dlc_spread_trajectory.png'
fig.savefig(out,dpi=200,bbox_inches='tight')
print('saved',out,'peak_with=%.2f peak_without=%.2f'%(mean_w[lo:hi].max(),mean_o[lo:hi].max()))
