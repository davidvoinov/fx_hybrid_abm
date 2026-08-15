#!/usr/bin/env python3
"""N8: stale/noisy-oracle robustness for the HFMM backstop (DLC).

The headline pegs the pool rate to the latent fair price S_t, idealizing the
oracle. A deployed FX-AMM oracle would lag and be noisy, worst in stress -- the
same regime where the backstop is supposed to help. This experiment degrades
*only* the pool's oracle (the AMMArbitrageur's reference price, which drives both
the arbitrage target and the curve re-peg), leaving the rest of the market on
true fair value, and asks whether the peak-spread reduction survives.

Degraded oracle:  o_t = (1-a_t) S_t + a_t o_{t-1},  then x (1 + eps),  eps ~ N(0, eta_t)
  a_t   = min(0.99, a0 + ka * stress)     staleness persistence (lag)
  eta_t = (n0 + kn * stress) bps           multiplicative noise sd
  stress = max(sigma/sigma_low - 1, 3 (1 - systemic_liquidity))   widens in crisis

Levels:  ideal (off, reproduces headline) / mild / severe.
For each seed: no-AMM peak once (oracle-independent) + with-AMM peak per level.
Peak-spread reduction = (peak_wo - peak_with) / peak_wo, paired by seed/shock."""
import os
import sys, os, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from AgentBasedModel.agents import AMMArbitrageur
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label = lambda a: 'dlc'

PRESET = 'dealer_liquidity_crisis'; N_ITER = 1000
SEEDS = list(range(42, 42 + 120))
H = 250
N_WORKERS = min(10, os.cpu_count() or 4)

# degradation presets: a0 lag, ka lag-stress gain, n0 noise bps, kn noise-stress gain
LEVELS = {
    'ideal':  None,
    'mild':   dict(a0=0.50, ka=0.30, n0=2.0, kn=5.0),
    'severe': dict(a0=0.70, ka=0.50, n0=5.0, kn=20.0),
}

# ---- monkeypatch the oracle (applied at import time so spawn-workers inherit it)
_ORIG_REF = AMMArbitrageur._reference_price


def _degraded_ref(self):
    true = _ORIG_REF(self)
    cfg = getattr(self, '_degrade_cfg', None)
    if true is None or cfg is None:
        return true
    env = self.env
    stress = 0.0
    s0 = getattr(env, 'sigma_low', None) if env is not None else None
    if s0 and s0 > 0:
        stress = max(0.0, env.sigma / s0 - 1.0)
    illiq = max(0.0, 1.0 - getattr(env, 'systemic_liquidity', 1.0)) if env is not None else 0.0
    stress = max(stress, 3.0 * illiq)
    a_t = min(0.99, cfg['a0'] + cfg['ka'] * stress)
    eta = cfg['n0'] + cfg['kn'] * stress
    prev = getattr(self, '_orc_prev', None)
    stale = true if prev is None else (1.0 - a_t) * true + a_t * prev
    self._orc_prev = stale
    eps = float(np.random.normal(0.0, eta / 1e4))
    return stale * (1.0 + eps)


AMMArbitrageur._reference_price = _degraded_ref


def run(seed, amm, cfg):
    argv = ['--preset', PRESET, '--seed', str(seed), '--n-iter', str(N_ITER), '--silent']
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv); _auto_stress_around_shock(a)
    a.run_label = _primary_run_label(a); a.enable_amm = 1 if amm else 0
    if not amm: a.amm_share_pct = 0
    a.clob_amm_interaction = 'competition'
    _seed_all(seed); sim = build_sim(a);
    arb = getattr(sim, 'arbitrageur', None)
    if arb is not None and cfg is not None:
        arb._degrade_cfg = cfg
    sim.simulate(a.n_iter, silent=True)
    return sim, int(a.shock_iter)


def peak(sim, sh):
    s = np.asarray(sim.logger.clob_qspr, dtype=float)
    return float(np.nanmax(s[sh:sh + H])) if s.size >= sh + H else float('nan')


def eval_seed(seed):
    sim0, sh0 = run(seed, False, None)
    pk_wo = peak(sim0, sh0)
    out = {}
    for name, cfg in LEVELS.items():
        sim, sh = run(seed, True, cfg)
        pk_with = peak(sim, sh)
        red = (100 * (pk_wo - pk_with) / pk_wo
               if np.isfinite(pk_with) and np.isfinite(pk_wo) and pk_wo > 0 else np.nan)
        out[name] = (pk_with, red)
    out['_wo'] = pk_wo
    return out


def boot_ci(v, n=5000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    bs = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_p(red, n=10000, seed=0):
    """Sign-flip permutation p-value: H0 reduction == 0 (paired diffs are peak_wo-peak_with)."""
    v = np.asarray([x for x in red if np.isfinite(x)], dtype=float)
    if v.size == 0: return np.nan
    obs = abs(np.mean(v))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, v.size))
    null = np.abs((signs * np.abs(v)).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n + 1))


if __name__ == '__main__':
    print(f'N8 stale-oracle: {len(SEEDS)} seeds x {len(LEVELS)} levels, {N_WORKERS} workers...',
          flush=True)
    with Pool(N_WORKERS) as pool:
        results = pool.map(eval_seed, SEEDS)

    pk_wo = float(np.nanmedian([r['_wo'] for r in results]))
    lines = [f'N8 stale/noisy-oracle robustness (HFMM), DLC, {len(SEEDS)} paired seeds.',
             'Peak-spread reduction = (peak_wo - peak_with)/peak_wo vs the no-AMM arm.',
             'Only the pool oracle is degraded; the rest of the market sees true fair value.',
             f'Median no-AMM peak spread = {pk_wo:.1f} bps.', '']
    for name in LEVELS:
        reds = [r[name][1] for r in results]
        peaks = [r[name][0] for r in results]
        m, lo, hi = boot_ci(reds, seed=hash(name) % 999)
        pmed = float(np.nanmedian(peaks))
        pval = perm_p(reds, seed=hash(name) % 997)
        tag = '(ideal oracle, control)' if name == 'ideal' else ''
        lines.append(f'--- {name:6s} {tag}')
        lines.append(f'    median with-AMM peak = {pmed:5.1f} bps   '
                     f'peak-spread reduction = {m:5.1f}%  [95% CI {lo:.1f}, {hi:.1f}]  p={pval:.4f}')
        lines.append('')
    txt = '\n'.join(lines)
    print('\n' + txt, flush=True)
    open('output/resilience/robustness_n8_stale_oracle.txt', 'w').write(txt + '\n')
    print('saved output/resilience/robustness_n8_stale_oracle.txt', flush=True)
