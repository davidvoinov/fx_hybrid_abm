"""The comparison this model is about, drawn from the runner's own report.

The plots this package carried were built around a market with a facility
against the same market without one. That comparison pools three things the
facility brings at once, namely the capital it commits, the obligation to keep
quoting and the rule by which it prices, so the resource matched arms replaced
it. These read the arms runner's report and draw what it identifies.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib.pyplot as plt

from AgentBasedModel.visualization.paper_style import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_NEUTRAL,
    tidy_origin,
)

ARM_LABELS = {
    'none': 'dealer only',
    'reserve': 'reserve priced pool',
    'dealer_of_last_resort': 'obliged quoter',
    'passive_book': 'passive ladder',
}
CONTRAST_LABELS = {
    'standing_availability': 'standing availability',
    'pricing_schedule': 'pricing schedule',
}


def _load(report: str | dict) -> dict:
    if isinstance(report, dict):
        return report
    with open(report, encoding='utf-8') as handle:
        return json.load(handle)


def _bars(ax, rows, title, xlabel):
    """One horizontal bar per row, with the interval drawn on it.

    Sign carries meaning here: a negative change is an improvement for the
    market, so the colour follows the sign and not the ordering.
    """
    names = [name for name, _, _, _ in rows]
    means = [mean for _, mean, _, _ in rows]
    positions = range(len(rows))
    colours = [COLOR_GOOD if mean < 0 else COLOR_BAD for mean in means]
    ax.barh(list(positions), means, color=colours, height=0.55, alpha=0.85)
    for y, (_, mean, low, high) in zip(positions, rows):
        ax.plot([low, high], [y, y], color=COLOR_NEUTRAL, linewidth=1.4)
        ax.plot([low, low], [y - 0.12, y + 0.12], color=COLOR_NEUTRAL, linewidth=1.4)
        ax.plot([high, high], [y - 0.12, y + 0.12], color=COLOR_NEUTRAL, linewidth=1.4)
    ax.axvline(0.0, color='black', linewidth=0.8)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left')
    tidy_origin(ax)


def dashboard_arms(report: str | dict = 'output/facility_arms.json',
                   out_dir: str = 'output',
                   filename: str = 'facility_arms.png',
                   figsize=(11, 7)) -> Optional[str]:
    """Draw each arm against the control and the two identifying contrasts."""
    payload = _load(report)
    arms = payload.get('arms') or {}
    contrasts = payload.get('contrasts') or {}
    if not arms:
        return None

    def rows(source, key, labels):
        out = []
        for name, entry in source.items():
            block = entry.get(key)
            if not isinstance(block, dict):
                continue
            mean = block.get('mean')
            interval = block.get('ci') or [float('nan')] * 2
            if mean is None or mean != mean or name == 'none':
                continue
            out.append((labels.get(name, name), mean, interval[0], interval[1]))
        return out

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    _bars(axes[0][0], rows(arms, 'delta_peak', ARM_LABELS),
          'Peak executable cost against the dealer only control',
          'change in basis points')
    _bars(axes[0][1], rows(arms, 'delta_cost', ARM_LABELS),
          'What customers paid, against the same control',
          'change in basis points')
    _bars(axes[1][0], rows(contrasts, 'delta_peak', CONTRAST_LABELS),
          'Peak, decomposed', 'change in basis points')
    _bars(axes[1][1], rows(contrasts, 'delta_cost', CONTRAST_LABELS),
          'Customer cost, decomposed', 'change in basis points')

    provenance = payload.get('provenance') or {}
    seeds = provenance.get('seed_count')
    preset = provenance.get('preset', '')
    fig.suptitle(
        f'Resource matched facility arms — {preset}, {seeds} paired seeds',
        x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
