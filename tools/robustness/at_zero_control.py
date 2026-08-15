#!/usr/bin/env python3
"""DLC control with A_t OFF (clob_amm_interaction='none'): does the AMM
severity/recovery benefit survive without the circular outside-option term?
Compares with vs without AMM, paired, and contrasts with competition mode."""
import os
import sys, numpy as np, random, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
from main import (build_parser,_apply_preset_defaults,_resolve_main_routing,
                  _auto_stress_around_shock,_seed_all,build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label=lambda a:'dlc'

PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+120))
HORIZON=250; HOLD=10; COMMON_L=4.0

def series(seed, amm, mode):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1 if amm else 0
    if not amm: a.amm_share_pct=0
    a.clob_amm_interaction=mode                    # force A_t mode
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    return np.asarray(sim.logger.clob_qspr,dtype=float), int(a.shock_iter)

def peak(s,shock,base): return float(np.nanmax(s[shock:shock+HORIZON]))
def disloc_pct(s,shock,base): return (peak(s,shock,base)-base)/base*100.0 if base>0 else float('nan')
def auc(s,shock,base):
    end=min(len(s),shock+200); return float(np.sum(np.clip(s[shock:end]-base,0,None)))
def below(s,shock,L):
    for t in range(shock+1,min(len(s),shock+HORIZON)-HOLD):
        if np.all(s[t:t+HOLD]<=L): return t-shock
    return HORIZON

def collect(amm, mode):
    out={}
    for sd in SEEDS:
        s,shock=series(sd,amm,mode)
        if s.size<shock+HORIZON: continue
        base=float(np.nanmean(s[shock-45:shock-5]))
        out[sd]=dict(peak=peak(s,shock,base),disloc=disloc_pct(s,shock,base),
                     auc=auc(s,shock,base),below=below(s,shock,COMMON_L),base=base)
    return out

def perm(d):
    md=st.mean(d); random.seed(0); R=20000; obs=abs(md); c=0
    for _ in range(R):
        if abs(sum(x if random.random()<.5 else -x for x in d)/len(d))>=obs: c+=1
    return md,(c+1)/(R+1)

print('=== DLC in NONE mode (A_t OFF): with vs without AMM ===')
W=collect(True,'none'); O=collect(False,'none')
seeds=sorted(set(W)&set(O)); print('paired seeds:',len(seeds))
print(f'calm baseline  withAMM={st.mean(W[s]["base"] for s in seeds):.2f}  noAMM={st.mean(O[s]["base"] for s in seeds):.2f} bps')
for key,lbl,unit in [('peak','peak mean spread','bps'),('disloc','peak dislocation (% of baseline)','%'),
                     ('auc','cumulative excess spread (AUC)','bps*ticks'),('below',f'fall below {COMMON_L}bps','ticks')]:
    mw=st.mean(W[s][key] for s in seeds); mo=st.mean(O[s][key] for s in seeds)
    d=[W[s][key]-O[s][key] for s in seeds]; md,p=perm(d)
    rel=(mw-mo)/abs(mo)*100 if mo else float('nan')
    print(f'  {lbl:34s}: withAMM={mw:8.1f} noAMM={mo:8.1f} {unit:9s} rel={rel:+5.1f}% p={p:.4g}')
print('(competition-mode reference: peak dislocation -18%, AUC -34%)')
