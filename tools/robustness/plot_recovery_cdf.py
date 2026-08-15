#!/usr/bin/env python3
"""Recovery visualization: fraction of seeds whose spread has returned to the
common 4 bps band (held 10 ticks) by each tick after the shock, AMM vs no-AMM.
A left-shifted curve = faster recovery. Consumes final_300seed_raw.npz."""
import os
import sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from AgentBasedModel.visualization.paper_style import use_paper_style; use_paper_style()
except Exception: pass

d = np.load('output/resilience/final_300seed_raw.npz')
bt, ct = d['base_traj'], d['comp_traj']            # [seeds x ticks], shock at slice idx 50
s = 50; HOLD = 10; L = 4.0; H = min(bt.shape[1]-s, ct.shape[1]-s, 200)

def rec_ticks(traj):
    out = []
    for row in traj:
        r = H
        for t in range(1, H-HOLD):
            if np.all(row[s+t:s+t+HOLD] <= L): r = t; break
        out.append(r)
    return np.array(out)

rw = rec_ticks(ct); ro = rec_ticks(bt)
grid = np.arange(0, H)
cdf_w = np.array([np.mean(rw <= t) for t in grid])
cdf_o = np.array([np.mean(ro <= t) for t in grid])

fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.plot(grid, 100*cdf_o, label='Without AMM (dealer CLOB only)', color='#c0392b', lw=1.9)
ax.plot(grid, 100*cdf_w, label='With AMM (hybrid)', color='#2c3e50', lw=1.9)
ax.axhline(50, color='grey', ls=':', lw=0.8)
ax.set_xlabel('Ticks since shock'); ax.set_ylabel('Share of seeds recovered (%)')
ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=9, loc='lower right')
fig.tight_layout()
fig.savefig('output/resilience/dlc_recovery_cdf.png', dpi=200, bbox_inches='tight')
print('median recovery: with=%.0f without=%.0f ticks' % (np.median(rw), np.median(ro)))
print('saved output/resilience/dlc_recovery_cdf.png')
