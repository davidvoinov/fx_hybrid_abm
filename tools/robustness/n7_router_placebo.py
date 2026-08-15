#!/usr/bin/env python3
"""N7 (panel #2): router-vs-venue placebo decomposition.

Devil's-advocate alt-explanation: the severity reduction might be the work of the
intelligent router (which reassigns flow to whichever venue stays open), not a
property of the AMM pool's slippage curve. Placebo test: replace liquidity-aware
routing with a 'dumb' fixed-share split (always-open venue, no intelligence) and
ask whether the benefit survives.

  arm A  baseline          : no AMM (dealer CLOB only)
  arm B  liquidity_aware   : AMM + intelligent routing   (full mechanism)
  arm C  fixed_share        : AMM + dumb fixed split      (placebo)

If C still beats baseline, the pool slippage curve / always-on availability is
load-bearing; (B - C) isolates the router's marginal contribution.
120 paired seeds, competition coupling, common-band recovery at 4.0 bps."""
import os
import sys, numpy as np, random, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label = lambda a: 'dlc'

PRESET = 'dealer_liquidity_crisis'; N_ITER = 1000; SEEDS = list(range(42, 42 + 120))
H = 250; HOLD = 10; L = 4.0


def run(seed, amm, route_rule):
    argv = ['--preset', PRESET, '--seed', str(seed), '--n-iter', str(N_ITER), '--silent']
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv); _auto_stress_around_shock(a)
    a.run_label = _primary_run_label(a)
    a.enable_amm = 1 if amm else 0
    if not amm: a.amm_share_pct = 0
    a.clob_amm_interaction = 'competition'
    if amm: a.venue_choice_rule = route_rule        # override AFTER preset defaults
    _seed_all(seed); sim = build_sim(a); sim.simulate(a.n_iter, silent=True)
    log = sim.logger; sh = int(a.shock_iter)
    return np.asarray(log.clob_qspr, dtype=float), sh


def feats(s, sh):
    base = float(np.nanmean(s[sh - 45:sh - 5])); pk = float(np.nanmax(s[sh:sh + H]))
    end = min(len(s), sh + 200); auc = float(np.sum(np.clip(s[sh:end] - base, 0, None)))
    rec = H
    for t in range(sh + 1, min(len(s), sh + H) - HOLD):
        if np.all(s[t:t + HOLD] <= L): rec = t - sh; break
    return base, pk, auc, rec


def collect(amm, route_rule):
    out = {'base': {}, 'pk': {}, 'auc': {}, 'rec': {}}
    for sd in SEEDS:
        s, sh = run(sd, amm, route_rule)
        if s.size < sh + H: continue
        b, pk, auc, rec = feats(s, sh)
        out['base'][sd] = b; out['pk'][sd] = pk; out['auc'][sd] = auc; out['rec'][sd] = rec
    return out


def boot(d, reps=2000):
    random.seed(1); n = len(d)
    xs = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return xs[int(.025 * reps)], xs[int(.975 * reps)]


def perm(d, reps=20000):
    md = st.mean(d); random.seed(0); obs = abs(md)
    c = sum(1 for _ in range(reps)
            if abs(sum(x if random.random() < .5 else -x for x in d) / len(d)) >= obs)
    return md, (c + 1) / (reps + 1)


def cmp(A, base, key):
    seeds = sorted(set(A[key]) & set(base[key]))
    w = [A[key][s] for s in seeds]; o = [base[key][s] for s in seeds]
    d = [a - b for a, b in zip(w, o)]; md, p = perm(d); lo, hi = boot(d)
    rel = (st.mean(w) - st.mean(o)) / abs(st.mean(o)) * 100
    return st.median(w), st.median(o), md, lo, hi, rel, p, len(seeds)


print('running baseline (no AMM) ...', flush=True);          B = collect(False, None)
print('running liquidity_aware (AMM + smart router) ...', flush=True); LA = collect(True, 'liquidity_aware')
print('running fixed_share (AMM + dumb split, PLACEBO) ...', flush=True); FS = collect(True, 'fixed_share')

lines = ['N7 router-vs-venue placebo (DLC, 120 paired seeds, competition coupling).',
         'arm B = AMM + liquidity_aware routing; arm C = AMM + fixed_share (dumb) split.',
         'If C still beats baseline, the pool slippage curve is load-bearing, not the router.', '']
for label, A in [('liquidity_aware (full mechanism)', LA), ('fixed_share (placebo)', FS)]:
    lines.append(f'--- {label} vs baseline ---')
    for key, lbl in [('pk', 'peak spread (bps)'), ('auc', 'cumulative excess (bps*ticks)'),
                     ('rec', 'recovery to <4bps (ticks)')]:
        mw, mo, md, lo, hi, rel, p, n = cmp(A, B, key)
        lines.append(f'  {lbl:30s}: with={mw:7.1f} without={mo:7.1f} '
                     f'delta={md:+7.1f}[{lo:+.1f},{hi:+.1f}] rel={rel:+5.1f}% p={p:.4g} n={n}')
    lines.append('')

# router marginal contribution = liquidity_aware effect - fixed_share effect (peak)
seeds = sorted(set(LA['pk']) & set(FS['pk']))
dmarg = [LA['pk'][s] - FS['pk'][s] for s in seeds]
md, p = perm(dmarg); lo, hi = boot(dmarg)
lines.append(f"router marginal on peak (LA - FS): delta={md:+.2f}[{lo:+.2f},{hi:+.2f}] bps "
             f"p={p:.4g} n={len(seeds)}  (near zero => availability, not intelligence, drives it)")

out = '\n'.join(lines)
print('\n' + out)
open('output/resilience/robustness_n7_router_placebo.txt', 'w').write(out + '\n')
print('saved output/resilience/robustness_n7_router_placebo.txt')
