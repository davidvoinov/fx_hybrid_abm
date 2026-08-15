#!/usr/bin/env python3
"""Final 300-seed pass for the DLC headline: baseline / competition / A_t=0.
Locks Table 1 (peak bps, AUC, common-band recovery + bootstrap CI), regenerates
the 300-seed mean spread trajectory, and captures H2 dynamics (active-tick AMM
customer-volume share + withdrawn share over the event window). Quiet seconds
are undefined, not zero. Saves raw aggregates to .npz before plotting."""
import os
import sys, numpy as np, random, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AgentBasedModel.visualization.paper_style import use_paper_style
use_paper_style()
from main import (build_parser,_apply_preset_defaults,_resolve_main_routing,
                  _auto_stress_around_shock,_seed_all,build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label=lambda a:'dlc'
PRESET='dealer_liquidity_crisis'; N_ITER=1000; SEEDS=list(range(42,42+300))
H=250; HOLD=10; L=4.0

def run(seed, amm, mode):
    argv=['--preset',PRESET,'--seed',str(seed),'--n-iter',str(N_ITER),'--silent']
    p=build_parser(); a=p.parse_args(argv); _apply_preset_defaults(p,a)
    a.venue_choice_rule=_resolve_main_routing(a,argv); _auto_stress_around_shock(a)
    a.run_label=_primary_run_label(a); a.enable_amm=1 if amm else 0
    if not amm: a.amm_share_pct=0
    a.clob_amm_interaction=mode
    _seed_all(seed); sim=build_sim(a); sim.simulate(a.n_iter,silent=True)
    log=sim.logger; sh=int(a.shock_iter)
    spr=np.asarray(log.clob_qspr,dtype=float)
    try: amm_sh=np.asarray(log.amm_active_tick_customer_volume_share_series(),dtype=float)
    except Exception: amm_sh=np.array([])
    try: wd=np.asarray(log.mm_channel_shares['endogenous'],dtype=float)
    except Exception: wd=np.array([])
    return spr, sh, amm_sh, wd

def feats(s,sh):
    base=float(np.nanmean(s[sh-45:sh-5]))
    win=s[sh:sh+H]; pk=float(np.nanmax(win)); pk_step=sh+int(np.nanargmax(win))
    end=min(len(s),sh+200); auc=float(np.sum(np.clip(s[sh:end]-base,0,None)))
    # (1) common absolute band: spread <= L bps, held HOLD ticks (primary)
    rec=H
    for t in range(sh+1,min(len(s),sh+H)-HOLD):
        if np.all(s[t:t+HOLD]<=L): rec=t-sh; break
    # (2) return to own pre-shock baseline within +/-1 bps, held HOLD ticks
    rec_own=H
    for t in range(sh+1,min(len(s),sh+H)-HOLD):
        if np.all(s[t:t+HOLD]<=base+1.0): rec_own=t-sh; break
    # (3) peak-anchored 80% retracement: spread <= base + 0.2*(pk-base) after the
    #     peak, held HOLD ticks. The arm with the smaller peak faces a STRICTER
    #     absolute finish line -- the normalisation artifact panel #2 flags.
    thr=base+0.2*(pk-base); rec_peak=H
    for t in range(pk_step+1,min(len(s),sh+H)-HOLD):
        if np.all(s[t:t+HOLD]<=thr): rec_peak=t-sh; break
    return base,pk,auc,rec,rec_own,rec_peak

def collect(amm,mode,keep_series=False):
    out={'base':{},'pk':{},'auc':{},'rec':{},'rec_own':{},'rec_peak':{}}
    trajs=[]; amms=[]; wds=[]; shock=350
    for sd in SEEDS:
        s,sh,ash,wd=run(sd,amm,mode); shock=sh
        if s.size<sh+H: continue
        b,pk,auc,rec,rec_own,rec_peak=feats(s,sh)
        out['base'][sd]=b; out['pk'][sd]=pk; out['auc'][sd]=auc; out['rec'][sd]=rec
        out['rec_own'][sd]=rec_own; out['rec_peak'][sd]=rec_peak
        if keep_series:
            lo,hi=sh-50,sh+H
            trajs.append(s[lo:hi])
            if ash.size>=hi: amms.append(ash[lo:hi])
            if wd.size>=hi: wds.append(wd[lo:hi])
    return out,trajs,amms,wds,shock

print('running baseline (no AMM) ...',flush=True)
B,btr,_,_,shock=collect(False,'none',keep_series=True)
print('running competition (A_t on) ...',flush=True)
C,ctr,cam,cwd,_=collect(True,'competition',keep_series=True)
print('running none (A_t off) ...',flush=True)
N,_,_,_,_=collect(True,'none')

def boot(d,reps=2000):
    random.seed(1); n=len(d); xs=sorted(sum(d[random.randrange(n)] for _ in range(n))/n for _ in range(reps))
    return xs[int(.025*reps)],xs[int(.975*reps)]
def perm(d,reps=20000):
    md=st.mean(d); random.seed(0); obs=abs(md); c=sum(1 for _ in range(reps)
        if abs(sum(x if random.random()<.5 else -x for x in d)/len(d))>=obs); return md,(c+1)/(reps+1)
def cmp(A,key,base=B):
    seeds=sorted(set(A[key])&set(base[key])); w=[A[key][s] for s in seeds]; o=[base[key][s] for s in seeds]
    d=[a-b for a,b in zip(w,o)]; md,p=perm(d); lo,hi=boot(d)
    return st.median(w),st.median(o),md,lo,hi,(st.mean(w)-st.mean(o))/abs(st.mean(o))*100,p,len(seeds)

# save raw aggregates first (crash-safe)
np.savez('output/resilience/final_300seed_raw.npz',
    base_traj=np.array([t[:min(map(len,btr))] for t in btr]) if btr else np.array([]),
    comp_traj=np.array([t[:min(map(len,ctr))] for t in ctr]) if ctr else np.array([]),
    comp_amm_active_tick_customer_share=(
        np.array([t[:min(map(len,cam))] for t in cam]) if cam else np.array([])
    ),
    comp_wd=np.array([t[:min(map(len,cwd))] for t in cwd]) if cwd else np.array([]), shock=shock)

print('\n=== TABLE 1 (DLC, 300 seeds, competition vs baseline) ===')
for key,lbl in [('pk','peak spread (bps)'),('auc','cumulative excess (bps*ticks)'),
                ('rec','recovery common-band <4bps (ticks)'),
                ('rec_own','recovery to own baseline (ticks)'),
                ('rec_peak','recovery peak-anchored 80% (ticks)'),
                ('base','calm spread (bps)')]:
    mw,mo,md,lo,hi,rel,p,n=cmp(C,key)
    print(f'  {lbl:38s}: with={mw:7.1f} without={mo:7.1f} delta={md:+7.1f}[{lo:+.1f},{hi:+.1f}] rel={rel:+5.1f}% p={p:.4g} n={n}')
print('=== A_t=0 control (none vs baseline) ===')
for key,lbl in [('pk','peak spread (bps)'),('auc','cumulative excess')]:
    mw,mo,md,lo,hi,rel,p,n=cmp(N,key)
    print(f'  {lbl:32s}: with={mw:7.1f} without={mo:7.1f} rel={rel:+5.1f}% p={p:.4g}')

# --- plots ---
def meancurve(arrs):
    Lm=min(len(a) for a in arrs); M=np.vstack([a[:Lm] for a in arrs]); return np.nanmean(M,axis=0)
x0=shock-(shock-50)
# trajectory
mb=meancurve(btr); mc=meancurve(ctr); x=np.arange(len(mb))-50
fig,ax=plt.subplots(figsize=(6.2,3.4))
ax.plot(x,mb,label='Without AMM (dealer CLOB only)',color='#c0392b',lw=1.8)
ax.plot(x,mc,label='With AMM (hybrid)',color='#2c3e50',lw=1.8)
ax.axvline(0,color='grey',ls=':',lw=1); ax.set_xlabel('Event time relative to shock (ticks)')
ax.set_ylabel('Mean quoted spread (bps)'); ax.legend(frameon=False,fontsize=9); fig.tight_layout()
fig.savefig('output/resilience/dlc_spread_trajectory.png',dpi=200,bbox_inches='tight')
print('saved dlc_spread_trajectory.png (300 seeds)')
# H2 dynamics
if cam and cwd:
    ma=meancurve(cam); mw=meancurve(cwd); xa=np.arange(len(ma))-50
    fig,ax=plt.subplots(figsize=(6.2,3.4))
    ax.plot(xa,100*ma,label='AMM customer share, active trade ticks (%)',color='#2c3e50',lw=1.8)
    ax.plot(xa[:len(mw)],100*mw,label='Dealers withdrawn (%)',color='#e67e22',lw=1.8)
    ax.axvline(0,color='grey',ls=':',lw=1); ax.set_xlabel('Event time relative to shock (ticks)')
    ax.set_ylabel('Share (%)'); ax.legend(frameon=False,fontsize=9); fig.tight_layout()
    fig.savefig('output/resilience/dlc_h2_migration.png',dpi=200,bbox_inches='tight')
    print('saved dlc_h2_migration.png')
