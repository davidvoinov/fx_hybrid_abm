#!/usr/bin/env python3
"""AMM-share sensitivity (panel MAJOR #5) + bootstrap CIs for Table 1.
DLC, liquidity-aware routing, prior AMM share in {10,30,50}%, vs the
share-independent no-AMM baseline. Reports peak spread (bps), AUC, and
recovery, paired by seed, with permutation p and bootstrap 95% CIs."""
import os
import sys, numpy as np, random, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
from main import (build_parser,_apply_preset_defaults,_resolve_main_routing,
                  _auto_stress_around_shock,_seed_all,build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label=lambda a:'dlc'
PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+120)); H=250; HOLD=10; L=4.0

def series(seed, amm, share=None):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv)   # stays liquidity_aware
    _auto_stress_around_shock(a); a.run_label=_primary_run_label(a)
    a.enable_amm=1 if amm else 0
    if not amm: a.amm_share_pct=0
    elif share is not None: a.amm_share_pct=share
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    return np.asarray(sim.logger.clob_qspr,dtype=float), int(a.shock_iter)
def feats(s,sh):
    base=float(np.nanmean(s[sh-45:sh-5])); pk=float(np.nanmax(s[sh:sh+H]))
    end=min(len(s),sh+200); auc=float(np.sum(np.clip(s[sh:end]-base,0,None)))
    bel=H
    for t in range(sh+1,min(len(s),sh+H)-HOLD):
        if np.all(s[t:t+HOLD]<=L): bel=t-sh; break
    return pk,auc,bel
def collect(amm,share=None):
    out={}
    for sd in SEEDS:
        s,sh=series(sd,amm,share)
        if s.size<sh+H: continue
        out[sd]=feats(s,sh)
    return out
def boot_ci(d,reps=2000):
    random.seed(1); n=len(d); xs=sorted(sum(d[random.randrange(n)] for _ in range(n))/n for _ in range(reps))
    return xs[int(.025*reps)], xs[int(.975*reps)]
def perm(d,reps=20000):
    md=st.mean(d); random.seed(0); obs=abs(md); c=0
    for _ in range(reps):
        if abs(sum(x if random.random()<.5 else -x for x in d)/len(d))>=obs: c+=1
    return md,(c+1)/(reps+1)

O=collect(False)  # no-AMM baseline (share-independent)
print('AMM-share sensitivity (DLC, 120 paired seeds). Baseline = no AMM.')
for share in (10,30,50):
    W=collect(True,share); seeds=sorted(set(W)&set(O))
    for i,(lbl,unit) in enumerate([('peak spread','bps'),('AUC','bps*ticks'),('recover<4bps','ticks')]):
        wv=[W[s][i] for s in seeds]; ov=[O[s][i] for s in seeds]
        d=[a-b for a,b in zip(wv,ov)]; md,p=perm(d); lo,hi=boot_ci(d)
        mw=st.median(wv); mo=st.median(ov); rel=(st.mean(wv)-st.mean(ov))/abs(st.mean(ov))*100
        print(f'  share={share:2d}%  {lbl:12s}: withAMM(med)={mw:7.1f} noAMM(med)={mo:7.1f} {unit:9s} '
              f'meanDelta={md:+7.1f} [95%CI {lo:+.1f},{hi:+.1f}] rel={rel:+5.1f}% p={p:.4g}')
