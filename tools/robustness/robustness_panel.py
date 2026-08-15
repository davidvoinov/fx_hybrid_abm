#!/usr/bin/env python3
"""Robustness verification panel for the DLC headline (post-fix data).
Three sub-panels in one figure:
 (a) per-seed peak quoted-spread distribution, with vs without the AMM (300-seed
     trajectories from final_300seed_raw.npz);
 (b) peak-spread reduction across the calm-state AMM-share prior (10/30/50%),
     showing the effect is flat in the prior (robustness_amm_share_sweep);
 (c) peak-spread reduction with the dealer A_t outside-option channel on vs off,
     showing the effect survives the circularity control (final_300seed pass).
All inputs are the locked post-fix results; no new simulation is run here.
"""
import os
import sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AgentBasedModel.visualization.paper_style import use_paper_style
use_paper_style()

NPZ = 'output/resilience/final_300seed_raw.npz'
d = np.load(NPZ)
base = d['base_traj']; comp = d['comp_traj']; shock_idx = 50  # window starts shock-50
# per-seed post-shock peak spread (bps)
pk_no = np.nanmax(base[:, shock_idx:], axis=1)
pk_yes = np.nanmax(comp[:, shock_idx:], axis=1)
pk_no = pk_no[np.isfinite(pk_no)]; pk_yes = pk_yes[np.isfinite(pk_yes)]
med_no, med_yes = np.median(pk_no), np.median(pk_yes)

# locked summaries (see output/resilience/robustness_amm_share_sweep.txt, final_300seed_results.txt)
share_x = [10, 30, 50]
share_red = [33.9, 29.4, 31.9]          # % peak-spread reduction by calm AMM-share prior
at_labels = ['A$_t$ on\n(competition)', 'A$_t$ off\n(none)']
at_red = [34.0, 31.8]                    # % peak-spread reduction, 300-seed

C_NO, C_YES = '#c0392b', '#2c3e50'
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))

# (a) peak distribution
parts = ax[0].violinplot([pk_no, pk_yes], positions=[0, 1], showmedians=True, widths=0.8)
for i, b in enumerate(parts['bodies']):
    b.set_facecolor([C_NO, C_YES][i]); b.set_alpha(0.45); b.set_edgecolor([C_NO, C_YES][i])
for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    if key in parts: parts[key].set_color('#333333'); parts[key].set_linewidth(1.0)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['Without\nAMM', 'With\nAMM'])
ax[0].set_ylabel('Peak quoted spread (bps)')
ax[0].annotate(f'med {med_no:.0f}', (0, med_no), textcoords='offset points', xytext=(10, 0),
               fontsize=8, color=C_NO, va='center')
ax[0].annotate(f'med {med_yes:.0f}', (1, med_yes), textcoords='offset points', xytext=(10, 0),
               fontsize=8, color=C_YES, va='center')
ax[0].set_title('(a) Peak-spread distribution', fontsize=9)

# (b) share-prior sweep
ax[1].bar([str(s) + '%' for s in share_x], share_red, color=C_YES, alpha=0.85, width=0.6)
ax[1].axhline(np.mean(share_red), color='#888', ls='--', lw=1)
ax[1].set_ylim(0, 45); ax[1].set_ylabel('Peak-spread reduction (%)')
ax[1].set_xlabel('Calm-state AMM-share prior')
for i, v in enumerate(share_red):
    ax[1].text(i, v + 1, f'{v:.0f}', ha='center', fontsize=8, color=C_YES)
ax[1].set_title('(b) Insensitive to share prior', fontsize=9)

# (c) A_t channel on/off
ax[2].bar(at_labels, at_red, color=['#2c3e50', '#e67e22'], alpha=0.85, width=0.6)
ax[2].set_ylim(0, 45); ax[2].set_ylabel('Peak-spread reduction (%)')
for i, v in enumerate(at_red):
    ax[2].text(i, v + 1, f'{v:.0f}', ha='center', fontsize=8)
ax[2].set_title('(c) Survives A$_t$=0 control', fontsize=9)

for a in ax:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.tight_layout(w_pad=1.4)
out = 'output/resilience/dlc_robustness_panel.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'peak median without/with = {med_no:.1f} / {med_yes:.1f} bps  (n={len(pk_no)}/{len(pk_yes)})')
print(f'saved {out}')
