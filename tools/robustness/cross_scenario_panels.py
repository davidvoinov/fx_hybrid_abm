#!/usr/bin/env python3
"""Cross-scenario generality panels (post-fix re-run).
For each of the five stress scenarios, runs N paired seeds (with vs without the
automated venue) and captures, per seed, the post-shock spread severity (peak
excess and cumulative excess over the pre-shock baseline) plus the with-AMM
order-flow migration trajectory. Produces:
  - dlc_resilience_heatmap.png   : scenarios x {peak spread, cumulative excess,
                                   composite}, coloured by with-AMM reduction (%)
  - scenario_migration_curves.png: active-tick AMM customer-share trajectories
                                   across scenarios (quiet ticks are undefined)
  - dlc_h2_migration.png         : DLC withdrawn-share + AMM-share (post-fix, clean)
Severity is the post-event peak/cumulative deviation from the pre-event baseline;
reduction is (without - with)/without. The [0,600] window is bit-identical to the
1000-tick headline run for the same seed, so these panels match Table 1. The
AMM share is hfmm+cpmm (there is no single 'amm' venue key).

Usage:  python3 tools/robustness/cross_scenario_panels.py [N_SEEDS]
"""
import os
import sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from AgentBasedModel.visualization.paper_style import use_paper_style
use_paper_style()
from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
try: from main import _primary_run_label
except Exception: _primary_run_label = lambda a: 'run'

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_ITER = 600
SEEDS = list(range(42, 42 + N_SEEDS))
PRESETS = [('mm_withdrawal',          'MM\nWithdrawal'),
           ('flash_crash',            'Flash\nCrash'),
           ('dealer_liquidity_crisis','Dealer Liq.\nCrisis'),
           ('funding_liquidity_shock','Funding\nShock'),
           ('high_vol_stress',        'High-Vol\nStress')]

def windows(preset, sh):
    if preset == 'high_vol_stress':
        return 300, (250, 295), (300, 500)
    return sh, (sh - 45, sh - 5), (sh, sh + 200)

PRE_PAD, POST_PAD = 50, 200

def build(seed, preset, amm):
    argv = ['--preset', preset, '--seed', str(seed), '--n-iter', str(N_ITER), '--silent']
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv); _auto_stress_around_shock(a)
    a.run_label = _primary_run_label(a)
    a.enable_amm = 1 if amm else 0
    if not amm: a.amm_share_pct = 0
    _seed_all(seed); sim = build_sim(a); sim.simulate(a.n_iter, silent=True)
    return sim, int(getattr(a, 'shock_iter', 350) or 350)

def series(sim):
    log = sim.logger
    spr = np.asarray(log.clob_qspr, float)
    try:
        amm = np.asarray(
            log.amm_active_tick_customer_volume_share_series(), dtype=float
        )
    except Exception: amm = np.array([])
    try: wd = np.asarray(log.mm_channel_shares['endogenous'], float)
    except Exception: wd = np.array([])
    return spr, amm, wd

def sev_peak(s, bw, pw):
    if s.size < pw[1]: return np.nan
    base = np.nanmedian(s[bw[0]:bw[1]])
    return np.nanmax(s[pw[0]:pw[1]]) - base if np.isfinite(base) else np.nan

def sev_auc(s, bw, pw):
    if s.size < pw[1]: return np.nan
    base = np.nanmedian(s[bw[0]:bw[1]])
    return float(np.nansum(np.clip(s[pw[0]:pw[1]] - base, 0, None))) if np.isfinite(base) else np.nan

heat = np.full((len(PRESETS), 3), np.nan)   # cols: peak spread, cumulative excess, composite
mig = {}; dlc_wd = None; dlc_amm = None
labels = [lbl for _, lbl in PRESETS]

for r, (preset, lbl) in enumerate(PRESETS):
    pk = ([], []); au = ([], [])     # (without, with)
    amm_traj = []; wd_traj = []
    for sd in SEEDS:
        try:
            sim_w, sh = build(sd, preset, True)
            sim_o, _ = build(sd, preset, False)
        except Exception as e:
            print(f'  [{preset} seed {sd}] FAILED: {e}', flush=True); continue
        ev, bw, pw = windows(preset, sh)
        spr_w, amm_w, wd_w = series(sim_w); spr_o, _, _ = series(sim_o)
        pk[1].append(sev_peak(spr_w, bw, pw)); pk[0].append(sev_peak(spr_o, bw, pw))
        au[1].append(sev_auc(spr_w, bw, pw)); au[0].append(sev_auc(spr_o, bw, pw))
        if amm_w.size >= ev + POST_PAD: amm_traj.append(amm_w[ev - PRE_PAD: ev + POST_PAD])
        if wd_w.size >= ev + POST_PAD:  wd_traj.append(wd_w[ev - PRE_PAD: ev + POST_PAD])
    for ci, (wo_l, wi_l) in enumerate([pk, au]):
        wo = np.nanmedian(wo_l); wi = np.nanmedian(wi_l)
        heat[r, ci] = 100 * (wo - wi) / wo if (np.isfinite(wo) and wo > 1e-9) else np.nan
    heat[r, 2] = np.nanmean(heat[r, :2])
    if amm_traj:
        m = min(len(t) for t in amm_traj); mig[lbl] = 100 * np.nanmean(np.vstack([t[:m] for t in amm_traj]), 0)
    if preset == 'dealer_liquidity_crisis':
        if amm_traj:
            m = min(len(t) for t in amm_traj); dlc_amm = 100 * np.nanmean(np.vstack([t[:m] for t in amm_traj]), 0)
        if wd_traj:
            m = min(len(t) for t in wd_traj); dlc_wd = 100 * np.nanmean(np.vstack([t[:m] for t in wd_traj]), 0)
    print(f'[{preset}] peak={heat[r,0]:.1f}%  AUC={heat[r,1]:.1f}%  composite={heat[r,2]:.1f}%  '
          f'(n_mig={len(amm_traj)})', flush=True)

np.savez('output/resilience/cross_scenario_raw.npz', heat=heat, labels=labels,
         **{f'mig_{i}': v for i, v in enumerate(mig.values())},
         mig_labels=list(mig.keys()), dlc_amm=dlc_amm if dlc_amm is not None else [],
         dlc_wd=dlc_wd if dlc_wd is not None else [], n_seeds=N_SEEDS)

# ---------- (1) heatmap ----------
metrics = ['Peak\nspread', 'Cumulative\nexcess', 'Composite']
fig, ax = plt.subplots(figsize=(4.8, 3.4))
vmax = max(5, np.nanmax(heat)); vmin = min(0, np.nanmin(heat))
im = ax.imshow(heat, cmap='RdYlGn', norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(metrics, fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
for i in range(len(labels)):
    for j in range(3):
        v = heat[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=9,
                    fontweight='bold' if j == 2 else 'normal')
ax.set_title('With-AMM spread-severity reduction', fontsize=9.5)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('Reduction (%)', fontsize=8)
fig.tight_layout()
fig.savefig('output/resilience/dlc_resilience_heatmap.png', dpi=200, bbox_inches='tight')
print('saved dlc_resilience_heatmap.png')

# ---------- (2) migration curves ----------
fig, ax = plt.subplots(figsize=(6.0, 3.4))
x = np.arange(-PRE_PAD, POST_PAD); cols = ['#7f8c8d', '#2980b9', '#c0392b', '#27ae60', '#8e44ad']
for (lbl, y), c in zip(mig.items(), cols):
    ax.plot(x[:len(y)], y, lw=1.8, color=c, label=lbl.replace('\n', ' '))
ax.axvline(0, color='grey', ls=':', lw=1)
ax.set_xlabel('Event time relative to shock (ticks)')
ax.set_ylabel('AMM customer share, active trade ticks (%)')
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('output/resilience/scenario_migration_curves.png', dpi=200, bbox_inches='tight')
print('saved scenario_migration_curves.png')

# ---------- (3) clean DLC migration ----------
if dlc_amm is not None and dlc_wd is not None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(np.arange(-PRE_PAD, -PRE_PAD + len(dlc_amm)), dlc_amm,
            color='#2c3e50', lw=1.8,
            label='AMM customer share, active trade ticks (%)')
    ax.plot(np.arange(-PRE_PAD, -PRE_PAD + len(dlc_wd)), dlc_wd, color='#e67e22', lw=1.8, label='Dealers withdrawn (%)')
    ax.axvline(0, color='grey', ls=':', lw=1)
    ax.set_xlabel('Event time relative to shock (ticks)'); ax.set_ylabel('Share (%)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('output/resilience/dlc_h2_migration.png', dpi=200, bbox_inches='tight')
    print('saved dlc_h2_migration.png (post-fix)')
print('DONE')
