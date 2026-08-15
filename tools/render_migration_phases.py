#!/usr/bin/env python3
"""cp-Figure-8 style: AMM flow share across the three phases (before / during /
after the shock window) per stress scenario. Reads the ready phase summary
output/tables/table6_flow_migration.csv (no simulation re-run)."""
import os
import csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AgentBasedModel.visualization.paper_style import use_paper_style
use_paper_style()

rows = list(csv.DictReader(open('output/tables/table6_flow_migration.csv')))
scen = [r['scenario'] for r in rows]
before = [float(r['amm_before_pct']) for r in rows]
during = [float(r['amm_during_pct']) for r in rows]
after = [float(r['amm_after_pct']) for r in rows]

x = np.arange(len(scen)); w = 0.26
fig, ax = plt.subplots(figsize=(6.6, 3.4))
b1 = ax.bar(x - w, before, w, label='Before', color='#9bb8d3')
b2 = ax.bar(x,      during, w, label='During', color='#c0392b')
b3 = ax.bar(x + w, after,  w, label='After',  color='#7f8c8d')
for bars in (b1, b2, b3):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7.2)
ax.set_xticks(x); ax.set_xticklabels([s.replace(' ', '\n', 1) for s in scen], fontsize=8)
ax.set_ylabel('AMM share of executed flow (%)')
ax.set_ylim(0, max(during) + 10)
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
out = 'output/resilience/migration_phases.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('before:', [f'{v:.0f}' for v in before])
print('during:', [f'{v:.0f}' for v in during])
print('after :', [f'{v:.0f}' for v in after])
print('saved', out)
