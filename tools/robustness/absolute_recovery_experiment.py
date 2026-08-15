#!/usr/bin/env python3
"""Absolute-recovery comparison (DLC), paired by seed, with vs without AMM.
Three baseline-fair notions, contrasted with the peak-anchored metric."""
import os
import sys, numpy as np, random, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
from main import (build_parser,_apply_preset_defaults,_resolve_main_routing,
                  _auto_stress_around_shock,_seed_all,build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label=lambda a:'dlc'

PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+120))
HORIZON=250; HOLD=10; TAU=1.0; COMMON_L=4.0   # bps

def series(seed, amm):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1 if amm else 0
    if not amm: a.amm_share_pct=0
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    return np.asarray(sim.logger.clob_qspr,dtype=float), int(a.shock_iter)

def recover_to_baseline(s, shock, base):
    for t in range(shock+1, min(len(s),shock+HORIZON)-HOLD):
        if np.all(s[t:t+HOLD] <= base+TAU): return t-shock
    return HORIZON
def fall_below(s, shock, L):
    for t in range(shock+1, min(len(s),shock+HORIZON)-HOLD):
        if np.all(s[t:t+HOLD] <= L): return t-shock
    return HORIZON
def excess_auc(s, shock, base):
    end=min(len(s),shock+200)
    return float(np.sum(np.clip(s[shock:end]-base,0,None)))

def collect(amm):
    out={}
    for sd in SEEDS:
        s,shock=series(sd,amm)
        if s.size<shock+HORIZON: continue
        base=float(np.nanmean(s[shock-45:shock-5]))
        out[sd]=dict(rec=recover_to_baseline(s,shock,base),
                     below=fall_below(s,shock,COMMON_L),
                     auc=excess_auc(s,shock,base), base=base)
    return out

def main():
    # The experiment used to run at import. Any tool that merely loaded the
    # module, pytest collection among them, started a full multi seed run and
    # never returned. It is now behind an entry point, and the file no longer
    # matches the test discovery pattern either.
    W=collect(True); O=collect(False)
    seeds=sorted(set(W)&set(O))
    def perm(d):
        md=st.mean(d); random.seed(0); R=20000; obs=abs(md); c=0
        for _ in range(R):
            if abs(sum(x if random.random()<.5 else -x for x in d)/len(d))>=obs: c+=1
        return md,(c+1)/(R+1)
    print(f'paired seeds: {len(seeds)}')
    print(f'calm baseline  withAMM={st.mean(W[s]["base"] for s in seeds):.2f}  noAMM={st.mean(O[s]["base"] for s in seeds):.2f} bps')
    for key,lbl,unit in [('rec',f'recover within +/-{TAU}bps of own baseline','ticks'),
                         ('below',f'fall below common {COMMON_L}bps','ticks'),
                         ('auc','cumulative excess spread (AUC)','bps*ticks')]:
        mw=st.mean(W[s][key] for s in seeds); mo=st.mean(O[s][key] for s in seeds)
        d=[W[s][key]-O[s][key] for s in seeds]; md,p=perm(d)
        verdict='WITH faster/less' if md<0 else 'WITH slower/more'
        print(f'{lbl:42s}: withAMM={mw:7.1f} noAMM={mo:7.1f}  delta(W-O)={md:+7.1f} {unit}  p={p:.4g}  [{verdict}]')
    print('(peak-anchored resilience metric for reference: withAMM=51 noAMM=41 ticks, WITH slower)')


if __name__ == '__main__':
    main()
