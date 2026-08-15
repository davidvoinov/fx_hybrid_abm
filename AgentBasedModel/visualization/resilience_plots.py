from __future__ import annotations

import math
import os
from typing import Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt

from AgentBasedModel.metrics.resilience import kaplan_meier_curve
from AgentBasedModel.visualization.paper_style import (
    use_paper_style,
    COLOR_NO_AMM,
)

use_paper_style()

DPI = 300
NO_AMM_COLOR = COLOR_NO_AMM


def _finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def _mean(values: Iterable[float]) -> float:
    clean = _finite(values)
    return sum(clean) / len(clean) if clean else float('nan')


def _median(values: Iterable[float]) -> float:
    clean = sorted(_finite(values))
    if not clean:
        return float('nan')
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _jitter_repeated(xs: List[float], width: float = 0.18) -> List[float]:
    """Spread repeated x-values slightly so dense vertical stacks remain visible."""
    positions = {}
    for idx, value in enumerate(xs):
        positions.setdefault(value, []).append(idx)

    out = list(xs)
    for value, indices in positions.items():
        if len(indices) <= 1:
            continue
        center = 0.5 * (len(indices) - 1)
        for offset, index in enumerate(indices):
            out[index] = value + (offset - center) * width
    return out


def _series_specs(spec: Mapping) -> List[Mapping]:
    series = list(spec.get('series', []))
    if series:
        return series
    return [{
        'series_key': spec.get('series_key', 'default'),
        'label': spec.get('series_label', 'Scenario'),
        'color': spec.get('color', '#1f77b4'),
        'points': list(spec.get('points', [])),
    }]


def plot_resilience_scatter_panels(panel_specs: Sequence[Mapping],
                                   out_dir: str = 'output/resilience',
                                   filename: str = 'resilience_scatter_panels.png') -> str:
    """Render 2x2 resilience panels with recovered and censored observations separated."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False, sharey=False)
    axes_flat = list(axes.flat)

    for ax, spec in zip(axes_flat, panel_specs):
        label = spec.get('label', 'Scenario')
        target_label = spec.get('target_label', 'Pre-shock baseline')

        all_y = []
        all_recovery_x = []
        stats_lines = []
        plotted_any = False

        for series in _series_specs(spec):
            series_label = series.get('label', 'Series')
            color = series.get('color', NO_AMM_COLOR)
            points = list(series.get('points', []))

            recovered_points = []
            censored_points = []
            for point in points:
                x_val = point.get('time_observed_steps', point.get('recovery_steps', float('nan')))
                y_val = point.get('normalized_avg_impact', float('nan'))
                if not (math.isfinite(x_val) and math.isfinite(y_val)):
                    continue
                if point.get('recovered', False):
                    recovered_points.append((x_val, y_val))
                else:
                    censored_points.append((x_val, y_val))

            rec_x = [x_val for x_val, _ in recovered_points]
            rec_y = [y_val for _, y_val in recovered_points]
            cen_x = [x_val for x_val, _ in censored_points]
            cen_y = [y_val for _, y_val in censored_points]
            series_y = rec_y + cen_y

            all_y.extend(series_y)
            all_recovery_x.extend(rec_x)

            if rec_x:
                ax.scatter(
                    _jitter_repeated(rec_x), rec_y,
                    s=30, alpha=0.9, color=color,
                    edgecolors='white', linewidths=0.35,
                    label=f'{series_label} recovered',
                )
                plotted_any = True
            if cen_x:
                ax.scatter(
                    _jitter_repeated(cen_x), cen_y,
                    s=40, alpha=0.9, facecolors='none', edgecolors=color,
                    linewidths=1.1, marker='^',
                    label=f'{series_label} censored',
                )
                plotted_any = True

            if rec_x or cen_x:
                recovered_share = len(rec_x) / max(len(rec_x) + len(cen_x), 1)
                median_recovery = _median(rec_x)
                stats_lines.append(
                    f'{series_label}: n={len(rec_x) + len(cen_x)}, '
                    f'rec={recovered_share:.0%}, '
                    f'med={median_recovery:.1f}, '
                    f'impact={_mean(series_y):+.2f}x'
                )

        ax.axhline(0.0, color='black', lw=1.0)

        mean_y = _mean(all_y)
        if math.isfinite(mean_y):
            ax.axhline(mean_y, color='#b03a2e', ls='--', lw=1.0, alpha=0.75)

        median_recovery = _median(all_recovery_x)
        if math.isfinite(median_recovery):
            ax.axvline(median_recovery, color='#444444', ls=':', lw=1.0, alpha=0.6)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Observed Time to 80% Recovery (steps)', fontsize=11)
        ax.set_ylabel('Normalized Average Impact', fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.text(0.02, 0.96, f'Target: {target_label}', transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.65))

        if stats_lines:
            text = '\n'.join(stats_lines)
            ax.text(0.98, 0.04, text, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75))
        if plotted_any:
            ax.legend(fontsize=8, loc='upper right')

    for ax in axes_flat[len(panel_specs):]:
        ax.axis('off')

    fig.suptitle('Resilience Study: Normalized Impact vs Recovery Time',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_kaplan_meier_panels(panel_specs: Sequence[Mapping],
                             out_dir: str = 'output/resilience',
                             filename: str = 'recovery_survival_panels.png') -> str:
    """Render Kaplan-Meier panels for time-to-recovery under right censoring."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False, sharey=True)
    axes_flat = list(axes.flat)

    for ax, spec in zip(axes_flat, panel_specs):
        label = spec.get('label', 'Scenario')
        color = spec.get('color', '#1f77b4')
        target_label = spec.get('target_label', 'Pre-shock baseline')
        points = list(spec.get('points', []))
        observed_steps = [point.get('time_observed_steps', float('nan')) for point in points]
        recovered_flags = [bool(point.get('recovered', False)) for point in points]
        km = kaplan_meier_curve(observed_steps, recovered_flags)

        ax.step(km['times'], km['survival'], where='post', color=color, lw=2.0)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Observed Steps Since Shock', fontsize=11)
        ax.set_ylabel('P(not yet recovered)', fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
        ax.text(0.02, 0.96, f'Target: {target_label}', transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.65))

        median_time = km.get('median_time', float('nan'))
        if math.isfinite(median_time):
            ax.axvline(median_time, color='#444444', ls=':', lw=1.0, alpha=0.6)
            ax.axhline(0.5, color='#777777', ls='--', lw=0.8, alpha=0.5)

        recovered_share = sum(recovered_flags) / max(len(recovered_flags), 1)
        text = (
            f'n = {km["n_obs"]}\n'
            f'recovered = {recovered_share:.0%}\n'
            f'KM median = {median_time:.1f}'
        )
        ax.text(0.98, 0.96, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75))

    for ax in axes_flat[len(panel_specs):]:
        ax.axis('off')

    fig.suptitle('Kaplan-Meier Recovery Curves', fontsize=15, fontweight='bold', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path