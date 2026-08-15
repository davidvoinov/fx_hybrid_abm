#!/usr/bin/env python3
"""H3 trade-size: all-in execution cost (bps) vs trade size for CLOB vs HFMM,
snapshotted at calm and at the stress peak of the DLC, averaged over seeds.
Shows the AMM cannot beat the dealer on price in calm, yet becomes cheaper for
small/medium trades under stress, while the largest trades stay expensive."""
import os
import sys, numpy as np, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AgentBasedModel.visualization.paper_style import use_paper_style
use_paper_style()
from main import (build_parser,_apply_preset_defaults,_resolve_main_routing,
                  _auto_stress_around_shock,_seed_all,build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label=lambda a:'dlc'
PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+100))
QS=[5,10,25,50,100]

def _args(seed):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1; a.clob_amm_interaction='competition'
    return a
def shock_of(seed): return int(_args(seed).shock_iter)
def sim_to(seed, tick):
    a=_args(seed); _seed_all(seed); sim=build_sim(a); sim.simulate(tick,silent=True); return sim

def costs(sim):
    clob=getattr(sim,'clob',None) or sim.exchange   # venue wrapper has cost_bps
    try: mid=clob.mid_price()
    except Exception: mid=None
    pool=sim.amm_pools.get('hfmm') or (list(sim.amm_pools.values())[0] if sim.amm_pools else None)
    cl=[]; am=[]
    for Q in QS:
        try: c=0.5*(clob.cost_bps(Q,'buy')+clob.cost_bps(Q,'sell'))
        except Exception: c=float('nan')
        cl.append(c)
        try:
            qb=pool.quote_buy(Q,S_t=mid)['cost_bps']; qs=pool.quote_sell(Q,S_t=mid)['cost_bps']
            am.append(0.5*(qb+qs))
        except Exception: am.append(float('nan'))
    return cl,am

def gather(when):
    CL=[]; AM=[]
    for sd in SEEDS:
        sh=shock_of(sd)
        tick = sh-20 if when=='calm' else sh+3
        s2=sim_to(sd, tick)
        cl,am=costs(s2); CL.append(cl); AM.append(am)
    CL=np.array(CL,float); AM=np.array(AM,float)
    return np.nanmedian(CL,0), np.nanmedian(AM,0)

print('calm ...',flush=True); cl_c,am_c=gather('calm')
print('stress ...',flush=True); cl_s,am_s=gather('stress')
print('\nQ        CLOB_calm AMM_calm | CLOB_stress AMM_stress  (median bps)')
for i,Q in enumerate(QS):
    print(f'{Q:4d}  {cl_c[i]:9.2f} {am_c[i]:8.2f} | {cl_s[i]:11.2f} {am_s[i]:10.2f}')
np.savez('output/resilience/cost_by_size_raw.npz',QS=QS,cl_c=cl_c,am_c=am_c,cl_s=cl_s,am_s=am_s)

fig,ax=plt.subplots(figsize=(6.2,3.6))
ax.plot(QS,cl_c,'o-',color='#c0392b',lw=1.6,label='CLOB, calm')
ax.plot(QS,am_c,'s--',color='#c0392b',lw=1.4,alpha=0.6,label='HFMM, calm')
ax.plot(QS,cl_s,'o-',color='#2c3e50',lw=1.8,label='CLOB, stress')
ax.plot(QS,am_s,'s--',color='#2c3e50',lw=1.6,label='HFMM, stress')
ax.set_xlabel('Trade size (model units, $\\approx$ EUR mn)'); ax.set_ylabel('All-in execution cost (bps)')
ax.legend(frameon=False,fontsize=8,ncol=2); fig.tight_layout()
fig.savefig('output/resilience/dlc_cost_by_trade_size.png',dpi=200,bbox_inches='tight')
print('saved dlc_cost_by_trade_size.png')
