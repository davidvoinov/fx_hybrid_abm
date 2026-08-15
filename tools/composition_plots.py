"""Visualization helpers for the composition sensitivity sweep.

Five plot types tuned for the sweep output:

  1. plot_threshold_bars        — bar plot of AMM-effect by composition bin,
                                  with SE error bars and threshold annotation
  2. plot_projection_panel      — 3x2 grid of heatmaps over 6 variable pairs
                                  for the same scenario × outcome
  3. plot_marginal_violins      — violins of AMM-effect distribution per
                                  composition variable bin
  4. plot_heatmap_v2            — improved heatmap with fixed color scale,
                                  threshold annotations, sample-size legend
  5. plot_importance_summary    — bar chart of |t-stat| from regression,
                                  sorted, with sign coloring

All plots follow consistent semantics:
  - Green = AMM beneficial (per outcome direction)
  - Red   = AMM adverse
  - Outcome-aware color: spread/cost/vol → green when Δ<0; depth → green when Δ>0
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Outcome direction: True = lower is better (spread, vol, cost)
#                    False = higher is better (depth, amm_share)
OUTCOME_DIRECTION = {
    'avg_clob_spread_bps':       True,
    'avg_clob_depth':            False,
    'avg_realized_volatility':   True,
    'tail_realized_volatility':  True,
    'avg_cost_clob_q5':          True,
}

OUTCOME_LABELS = {
    'avg_clob_spread_bps':       'Quoted spread (bps)',
    'avg_clob_depth':            'CLOB depth (units)',
    'avg_realized_volatility':   'Realized volatility',
    'tail_realized_volatility':  'Tail realized volatility',
    'avg_cost_clob_q5':          'Execution cost Q=5 (bps)',
}

SCENARIO_LABELS = {
    'default':                   'Calm baseline',
    'dealer_liquidity_crisis':   'Dealer liquidity crisis',
    'high_vol_stress':           'High-vol stress regime',
    'mm_withdrawal':             'MM withdrawal (isolated)',
    'flash_crash':               'Flash crash',
    'funding_liquidity_shock':   'Funding liquidity shock',
}

# Variable pairs used by the projection panel
PROJECTION_PAIRS = [
    ('n_mm', 'lp_total'),
    ('n_mm', 'book_total'),
    ('n_mm', 'taker_total'),
    ('n_mm', 'n_noise'),
    ('lp_total', 'book_total'),
    ('lp_total', 'taker_total'),
]

COMPOSITION_VARS = [
    'n_mm', 'n_fast_lp', 'n_latent_lp',
    'n_clob_fund', 'n_clob_chart', 'n_clob_univ',
    'n_fx_takers', 'n_fx_fund', 'n_retail', 'n_institutional', 'n_noise',
]

DERIVED_VARS = ['lp_total', 'book_total', 'taker_total']


def _outcome_is_good(outcome: str, delta: float) -> bool:
    """True if the Δ value indicates AMM is helping for this outcome."""
    direction = OUTCOME_DIRECTION.get(outcome, True)
    return (delta < 0) if direction else (delta > 0)


def _color_for_delta(outcome: str, delta: float) -> str:
    if not math.isfinite(delta):
        return '#888888'
    return '#1a9641' if _outcome_is_good(outcome, delta) else '#d7191c'


def _derive(row: Dict, var: str) -> float:
    """Get either a base composition var or a derived total."""
    if var == 'lp_total':
        return row.get('n_fast_lp', 0) + row.get('n_latent_lp', 0)
    if var == 'book_total':
        return row.get('n_clob_fund', 0) + row.get('n_clob_chart', 0) + row.get('n_clob_univ', 0)
    if var == 'taker_total':
        return (row.get('n_fx_takers', 0) + row.get('n_fx_fund', 0)
                + row.get('n_retail', 0) + row.get('n_institutional', 0))
    return row.get(var, 0)


def _safe_float(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return path


def _bin_var(var: str, vals: Sequence[float], n_bins: int = 5) -> List[Tuple[str, Tuple[float, float]]]:
    """Return list of (bin_label, (lo, hi)) tuples spanning the data range."""
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return []
    lo, hi = min(finite), max(finite)
    if lo == hi:
        return [(f'{int(lo)}', (lo, hi))]
    # Make integer-friendly bins
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    bins = []
    for i in range(n_bins):
        e0, e1 = edges[i], edges[i + 1]
        label = f'{int(round(e0))}-{int(round(e1))}'
        bins.append((label, (e0, e1)))
    return bins


# ---------------------------------------------------------------------------
# 1. Threshold bar plot
# ---------------------------------------------------------------------------

def plot_threshold_bars(effects: List[Dict],
                        composition_var: str,
                        scenario: str,
                        outcome: str,
                        out_path: str,
                        n_bins: int = 5) -> str:
    """For each bin of `composition_var`, show mean Δ ± SE with sign-coloring."""
    delta_key = f'{scenario}__{outcome}__delta'
    rows = [(r, _derive(r, composition_var), _safe_float(r.get(delta_key))) for r in effects]
    rows = [(r, v, d) for r, v, d in rows if math.isfinite(d) and math.isfinite(v)]
    if not rows:
        return ''

    vals = [v for _, v, _ in rows]
    bins = _bin_var(composition_var, vals, n_bins=n_bins)
    if not bins:
        return ''

    bin_data = []
    for label, (lo, hi) in bins:
        bucket = [d for _, v, d in rows if lo <= v <= hi]
        if not bucket:
            bin_data.append((label, float('nan'), float('nan'), 0))
            continue
        mean = float(np.mean(bucket))
        se = float(np.std(bucket, ddof=1) / math.sqrt(len(bucket))) if len(bucket) > 1 else 0.0
        bin_data.append((label, mean, se, len(bucket)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(bin_data))
    means = [b[1] for b in bin_data]
    ses = [b[2] for b in bin_data]
    colors = [_color_for_delta(outcome, m) for m in means]

    bars = ax.bar(xs, means, yerr=ses, color=colors, edgecolor='black',
                  linewidth=0.6, capsize=4,
                  error_kw=dict(ecolor='#222', elinewidth=1.2))

    # Annotate sample size & value
    for i, (label, mean, se, n) in enumerate(bin_data):
        if not math.isfinite(mean):
            continue
        ax.text(i, mean + (se if mean >= 0 else -se) * 1.2,
                f'{mean:+.3g}\n(n={n})',
                ha='center', va='bottom' if mean >= 0 else 'top',
                fontsize=9, fontweight='bold')

    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bin_data])
    ax.set_xlabel(composition_var.replace('_', ' '))
    ax.set_ylabel(f'Δ {OUTCOME_LABELS.get(outcome, outcome)}\n(with AMM − without AMM)')
    ax.set_title(f'AMM effect on {OUTCOME_LABELS.get(outcome, outcome)} by {composition_var}\nScenario: {SCENARIO_LABELS.get(scenario, scenario)}',
                 fontsize=12, fontweight='bold')

    # Legend explaining colors
    direction_text = 'Lower is better' if OUTCOME_DIRECTION.get(outcome, True) else 'Higher is better'
    handles = [
        Patch(facecolor='#1a9641', edgecolor='black', label=f'AMM helps ({direction_text})'),
        Patch(facecolor='#d7191c', edgecolor='black', label='AMM hurts'),
    ]
    ax.legend(handles=handles, loc='best', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Multi-projection panel
# ---------------------------------------------------------------------------

def plot_projection_panel(effects: List[Dict],
                          scenario: str,
                          outcome: str,
                          out_path: str,
                          var_pairs: Sequence[Tuple[str, str]] = PROJECTION_PAIRS,
                          global_vmin: Optional[float] = None,
                          global_vmax: Optional[float] = None,
                          stable_ids: Optional[set] = None) -> str:
    """3x2 grid of heatmap-style scatters over multiple variable pairs."""
    delta_key = f'{scenario}__{outcome}__delta'
    if stable_ids is not None:
        eff_filt = [r for r in effects if r.get('point_id') in stable_ids
                    or int(r.get('point_id', -1)) in stable_ids]
    else:
        eff_filt = effects

    # Compute global scale across all pairs if not provided
    all_vals = [_safe_float(r.get(delta_key)) for r in eff_filt]
    all_vals = [v for v in all_vals if math.isfinite(v)]
    if not all_vals:
        return ''
    if global_vmax is None:
        abs_max = max(abs(min(all_vals)), abs(max(all_vals)))
        global_vmin, global_vmax = -abs_max, abs_max

    # Use red-white-green for AMM-effect interpretation
    # Negative = green (AMM helps for spread/cost/vol); positive = red
    # For depth: invert the colormap
    if OUTCOME_DIRECTION.get(outcome, True):
        cmap = 'RdYlGn_r'  # red for positive Δ (bad), green for negative Δ (good)
    else:
        cmap = 'RdYlGn'    # green for positive (good for depth), red for negative

    rows = 3
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    axes_flat = axes.flatten()

    for ax, (x_var, y_var) in zip(axes_flat, var_pairs):
        xs, ys, vals = [], [], []
        for r in eff_filt:
            v = _safe_float(r.get(delta_key))
            if not math.isfinite(v):
                continue
            xs.append(_derive(r, x_var))
            ys.append(_derive(r, y_var))
            vals.append(v)
        if not vals:
            ax.set_axis_off()
            continue
        sc = ax.scatter(xs, ys, c=vals, s=120, cmap=cmap,
                        vmin=global_vmin, vmax=global_vmax,
                        edgecolors='black', linewidths=0.6, alpha=0.85)
        ax.set_xlabel(x_var.replace('_', ' '))
        ax.set_ylabel(y_var.replace('_', ' '))
        ax.grid(linestyle='--', alpha=0.3)
        # Annotate threshold for n_mm if it's the x or y axis
        if x_var == 'n_mm':
            ax.axvline(2.5, color='black', lw=1.2, ls=':', alpha=0.6)
            ax.text(2.5, ax.get_ylim()[1] * 0.97, ' n_mm=3 threshold',
                    fontsize=8, va='top', ha='left', color='black')
        if y_var == 'n_mm':
            ax.axhline(2.5, color='black', lw=1.2, ls=':', alpha=0.6)

    # Single colorbar
    cbar = fig.colorbar(sc, ax=axes_flat, shrink=0.55, pad=0.02, location='right',
                        aspect=40)
    direction_str = 'lower=better' if OUTCOME_DIRECTION.get(outcome, True) else 'higher=better'
    cbar.set_label(f'Δ {OUTCOME_LABELS.get(outcome, outcome)} ({direction_str})',
                   fontsize=10)

    fig.suptitle(f'AMM effect on {OUTCOME_LABELS.get(outcome, outcome)} across composition projections\n'
                 f'Scenario: {SCENARIO_LABELS.get(scenario, scenario)}  |  n={len(eff_filt)} points',
                 fontsize=13, fontweight='bold', y=0.995)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Marginal violins
# ---------------------------------------------------------------------------

def plot_marginal_violins(effects: List[Dict],
                          scenario: str,
                          outcome: str,
                          out_path: str,
                          comp_vars: Sequence[str] = COMPOSITION_VARS) -> str:
    """For each composition variable, violin of Δ binned by that variable."""
    delta_key = f'{scenario}__{outcome}__delta'

    n_vars = len(comp_vars)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.0 * rows + 1))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, var in zip(axes_flat, comp_vars):
        vals = [_derive(r, var) for r in effects]
        deltas = [_safe_float(r.get(delta_key)) for r in effects]
        paired = [(v, d) for v, d in zip(vals, deltas) if math.isfinite(d) and math.isfinite(v)]
        if not paired:
            ax.set_axis_off()
            continue
        # 4 quartile bins
        var_vals = sorted(set(v for v, _ in paired))
        if len(var_vals) <= 1:
            ax.set_axis_off()
            continue
        n_bins = min(4, len(var_vals))
        edges = np.linspace(min(var_vals), max(var_vals) + 1e-9, n_bins + 1)
        bin_data = []
        bin_labels = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            bucket = [d for v, d in paired if lo <= v <= hi]
            if bucket:
                bin_data.append(bucket)
                bin_labels.append(f'{int(round(lo))}-{int(round(hi))}')
        if not bin_data:
            ax.set_axis_off()
            continue

        positions = list(range(1, len(bin_data) + 1))
        parts = ax.violinplot(bin_data, positions=positions, showmeans=True,
                              showmedians=False, widths=0.7)
        # Color violins by mean direction
        for body, vals_in_bin in zip(parts['bodies'], bin_data):
            mean = float(np.mean(vals_in_bin))
            body.set_facecolor(_color_for_delta(outcome, mean))
            body.set_alpha(0.6)
            body.set_edgecolor('black')

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_title(var.replace('_', ' '), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.25)

    # Turn off unused axes
    for ax in axes_flat[n_vars:]:
        ax.set_axis_off()

    fig.suptitle(f'Marginal AMM-effect distribution per composition variable\n'
                 f'Scenario: {SCENARIO_LABELS.get(scenario, scenario)}  |  '
                 f'Outcome: {OUTCOME_LABELS.get(outcome, outcome)}',
                 fontsize=12, fontweight='bold', y=1.005)
    fig.text(0.5, 0.001, 'Bars within each panel = quartile bins of the variable. '
             'Color: green = AMM helps for this metric, red = AMM hurts.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Improved heatmap (single pair, with fixed scale + annotations)
# ---------------------------------------------------------------------------

def plot_heatmap_v2(effects: List[Dict],
                    scenario: str,
                    outcome: str,
                    out_path: str,
                    x_var: str = 'n_mm',
                    y_var: str = 'lp_total',
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    stable_ids: Optional[set] = None,
                    n_grid: int = 12) -> str:
    """Heatmap with interpolated binning grid + fixed colorscale + n_mm threshold line."""
    delta_key = f'{scenario}__{outcome}__delta'
    rows = []
    for r in effects:
        if stable_ids is not None and int(r.get('point_id', -1)) not in stable_ids:
            continue
        d = _safe_float(r.get(delta_key))
        if not math.isfinite(d):
            continue
        rows.append((_derive(r, x_var), _derive(r, y_var), d))
    if not rows:
        return ''

    xs = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    vals = np.array([r[2] for r in rows])

    # Determine fixed color scale
    if vmin is None or vmax is None:
        abs_max = max(abs(vals.min()), abs(vals.max()))
        vmin, vmax = -abs_max, abs_max

    # Outcome-aware colormap
    if OUTCOME_DIRECTION.get(outcome, True):
        cmap = 'RdYlGn_r'  # red for positive (bad), green for negative (good)
    else:
        cmap = 'RdYlGn'

    fig, ax = plt.subplots(figsize=(9, 7))

    # Use 2D histogram with mean-value-per-cell
    x_edges = np.linspace(xs.min() - 0.5, xs.max() + 0.5, n_grid + 1)
    y_edges = np.linspace(ys.min() - 0.5, ys.max() + 0.5, n_grid + 1)

    sum_grid, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=vals)
    cnt_grid, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    mean_grid = np.where(cnt_grid > 0, sum_grid / np.maximum(cnt_grid, 1), np.nan)

    im = ax.pcolormesh(x_edges, y_edges, mean_grid.T, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='flat', edgecolors='white', linewidth=0.3)

    # Overlay original points as small dots for transparency
    ax.scatter(xs, ys, c=vals, s=18, cmap=cmap, vmin=vmin, vmax=vmax,
               edgecolors='black', linewidths=0.4, alpha=0.7)

    # n_mm = 3 threshold line if applicable
    if x_var == 'n_mm':
        ax.axvline(2.5, color='black', lw=1.5, ls=':')
        ax.text(2.7, ax.get_ylim()[1] * 0.97,
                'n_mm = 3 threshold\n(AMM substitution viable →)',
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    if y_var == 'n_mm':
        ax.axhline(2.5, color='black', lw=1.5, ls=':')

    direction_str = 'lower=better' if OUTCOME_DIRECTION.get(outcome, True) else 'higher=better'
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f'Δ {OUTCOME_LABELS.get(outcome, outcome)} ({direction_str})', fontsize=10)

    ax.set_xlabel(x_var.replace('_', ' '))
    ax.set_ylabel(y_var.replace('_', ' '))
    ax.set_title(f'AMM effect on {OUTCOME_LABELS.get(outcome, outcome)}\n'
                 f'Scenario: {SCENARIO_LABELS.get(scenario, scenario)}  |  n={len(rows)} points  |  bins={n_grid}×{n_grid}',
                 fontsize=12, fontweight='bold')
    ax.grid(linestyle='--', alpha=0.25)

    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Fine n_mm threshold plot (isolates n_mm=0 from the rest)
# ---------------------------------------------------------------------------

def plot_fine_nmm_threshold(effects: List[Dict],
                            scenario: str,
                            outcome: str,
                            out_path: str,
                            mm_var: str = 'n_mm') -> str:
    """Bar plot of mean Δ per EXACT n_mm value (0, 1, 2, ..., 10).

    Isolates the n_mm=0 catastrophic point from the n_mm>=1 plateau, which
    the coarse bins (0-2, 3-4, ...) hid in the original threshold plot.
    """
    delta_key = f'{scenario}__{outcome}__delta'
    by_mm: Dict[int, List[float]] = {}
    for r in effects:
        m = int(_derive(r, mm_var))
        d = _safe_float(r.get(delta_key))
        if not math.isfinite(d):
            continue
        by_mm.setdefault(m, []).append(d)
    if not by_mm:
        return ''
    xs = sorted(by_mm)
    means = [float(np.mean(by_mm[m])) for m in xs]
    ses = [float(np.std(by_mm[m], ddof=1) / math.sqrt(len(by_mm[m])))
           if len(by_mm[m]) > 1 else 0.0 for m in xs]
    ns = [len(by_mm[m]) for m in xs]
    colors = [_color_for_delta(outcome, m) for m in means]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(xs, means, yerr=ses, color=colors, edgecolor='black',
                  linewidth=0.6, capsize=4,
                  error_kw=dict(ecolor='#222', elinewidth=1.2))
    for x, m, se, n in zip(xs, means, ses, ns):
        if not math.isfinite(m):
            continue
        sign = +1 if m >= 0 else -1
        ax.text(x, m + sign * (se + 0.04 * max(map(abs, means))),
                f'{m:+.2f}\n(n={n})',
                ha='center', va='bottom' if m >= 0 else 'top',
                fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(xs)
    ax.set_xlabel(mm_var.replace('_', ' '))
    ax.set_ylabel(f'Δ {OUTCOME_LABELS.get(outcome, outcome)}\n(with AMM − without AMM)')
    direction_text = 'Lower is better' if OUTCOME_DIRECTION.get(outcome, True) else 'Higher is better'
    ax.set_title(f'AMM effect on {OUTCOME_LABELS.get(outcome, outcome)} by exact {mm_var}\n'
                 f'Scenario: {SCENARIO_LABELS.get(scenario, scenario)}',
                 fontsize=12, fontweight='bold')
    handles = [
        Patch(facecolor='#1a9641', edgecolor='black', label=f'AMM helps ({direction_text})'),
        Patch(facecolor='#d7191c', edgecolor='black', label='AMM hurts'),
    ]
    ax.legend(handles=handles, loc='best', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Fraction-significant plot (per n_mm bin, % of compositions where
#    Δ is significantly < 0 / > 0 / NS at p<0.05)
# ---------------------------------------------------------------------------

def plot_fraction_significant(effects: List[Dict],
                              scenario: str,
                              outcome: str,
                              out_path: str,
                              mm_var: str = 'n_mm',
                              alpha: float = 0.05) -> str:
    """Stacked bar by exact n_mm value: % of compositions where AMM
    significantly helps / hurts / NS.

    Uses the per-composition pvalue + delta sign stored in effects.
    """
    delta_key = f'{scenario}__{outcome}__delta'
    p_key = f'{scenario}__{outcome}__pvalue'
    direction = OUTCOME_DIRECTION.get(outcome, True)

    by_mm: Dict[int, List[Tuple[float, float]]] = {}
    for r in effects:
        d = _safe_float(r.get(delta_key))
        p = _safe_float(r.get(p_key))
        if not (math.isfinite(d) and math.isfinite(p)):
            continue
        m = int(_derive(r, mm_var))
        by_mm.setdefault(m, []).append((d, p))
    if not by_mm:
        return ''

    xs = sorted(by_mm)
    helps_pct, hurts_pct, ns_pct = [], [], []
    ns_list = []
    for m in xs:
        items = by_mm[m]
        total = len(items)
        ns_list.append(total)
        n_help = sum(1 for d, p in items
                     if p < alpha and ((d < 0 and direction) or (d > 0 and not direction)))
        n_hurt = sum(1 for d, p in items
                     if p < alpha and ((d > 0 and direction) or (d < 0 and not direction)))
        n_ns = total - n_help - n_hurt
        helps_pct.append(100 * n_help / total)
        hurts_pct.append(100 * n_hurt / total)
        ns_pct.append(100 * n_ns / total)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.7
    p1 = ax.bar(xs, helps_pct, width, color='#1a9641', edgecolor='black',
                label='AMM helps (p<0.05)')
    p2 = ax.bar(xs, ns_pct, width, bottom=helps_pct, color='#cccccc',
                edgecolor='black', label='NS (p≥0.05)')
    bottom2 = [a + b for a, b in zip(helps_pct, ns_pct)]
    p3 = ax.bar(xs, hurts_pct, width, bottom=bottom2, color='#d7191c',
                edgecolor='black', label='AMM hurts (p<0.05)')

    for x, h, n_total in zip(xs, helps_pct, ns_list):
        ax.text(x, 102, f'n={n_total}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel(mm_var.replace('_', ' '))
    ax.set_ylabel(f'% of compositions (paired permutation test, α={alpha})')
    ax.set_ylim(0, 115)
    ax.set_title(f'Per-composition significance of AMM effect on {OUTCOME_LABELS.get(outcome, outcome)}\n'
                 f'Scenario: {SCENARIO_LABELS.get(scenario, scenario)}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. SHAP-style importance summary
# ---------------------------------------------------------------------------

def plot_importance_summary(regression_row: Dict,
                            scenario: str,
                            outcome: str,
                            out_path: str) -> str:
    """Bar chart of |t-stat| from regression, sorted by magnitude, signed coloring."""
    items = []
    for k, v in regression_row.items():
        if not k.startswith('tstat_') or k == 'tstat_intercept':
            continue
        try:
            t = float(v)
            if not math.isfinite(t):
                continue
            beta_key = 'beta_' + k.replace('tstat_', '')
            b = float(regression_row.get(beta_key, 0.0))
            var = k.replace('tstat_', '')
            items.append((var, b, t))
        except (TypeError, ValueError):
            continue

    items.sort(key=lambda x: abs(x[2]), reverse=True)
    if not items:
        return ''

    labels = [it[0] for it in items]
    abs_t = [abs(it[2]) for it in items]
    signs = [it[2] for it in items]
    # Color: green if β implies AMM-helping direction, red otherwise
    direction = OUTCOME_DIRECTION.get(outcome, True)
    colors = []
    for var, b, t in items:
        # In a Δ-regression, β > 0 means "more of this var → larger Δ"
        # For lower-is-better outcomes (spread, vol, cost), AMM helps more when Δ more negative.
        # So a NEGATIVE β on a Δ-regression means "more of var → AMM helps MORE" → that's a stable
        # interpretation only if β is significant. We just color by significance + sign.
        helps = (b < 0) if direction else (b > 0)
        if abs(t) < 1.96:
            colors.append('#888888')
        else:
            colors.append('#1a9641' if helps else '#d7191c')

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels) + 1.5)))
    ys = list(range(len(labels)))
    ax.barh(ys, abs_t, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(1.96, color='black', lw=1, ls=':', alpha=0.7)
    ax.text(2.0, len(labels) - 0.5, '|t|=1.96\n(p=0.05)', fontsize=8, va='top')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('|t-statistic|')
    ax.set_title(f'Predictor importance for Δ {OUTCOME_LABELS.get(outcome, outcome)}\n'
                 f'Scenario: {SCENARIO_LABELS.get(scenario, scenario)}',
                 fontsize=12, fontweight='bold')

    # Annotate with sign of β
    for i, (var, b, t) in enumerate(items):
        sign = '↓ helps' if (b < 0 and direction) or (b > 0 and not direction) else '↑ hurts'
        if abs(t) < 1.96:
            sign = '(NS)'
        ax.text(abs(t) + 0.1, i, f'β={b:+.3g} {sign}',
                fontsize=8, va='center')

    handles = [
        Patch(facecolor='#1a9641', edgecolor='black', label='Significant: AMM helps with more of this var'),
        Patch(facecolor='#d7191c', edgecolor='black', label='Significant: AMM hurts with more of this var'),
        Patch(facecolor='#888888', edgecolor='black', label='Not significant (|t|<1.96)'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 8. Stability + coverage diagnostic (4-panel)
# ---------------------------------------------------------------------------

def plot_stability_diagnostic(summary_rows: List[Dict],
                              criteria: List[Dict],
                              out_path: str,
                              x_var: str = 'n_mm',
                              y_var: str = 'lp_total') -> str:
    """Replace the uninformative binary stability map with a 4-panel diagnostic.

    Panels:
      (A) LHS coverage density (2D histogram of points per cell)
      (B) Realized CLOB spread per composition (continuous)
      (C) Realized CLOB depth per composition (continuous)
      (D) Stability margin: min normalized distance to criterion failure

    A composition row passes a criterion C with metric value v iff:
        C.min <= v <= C.max
    Normalized margin = min over all criteria of:
        (v - C.min) / (C.max - C.min) for closed bounds, or
        (v - C.min) / max(|v|, 1) for one-sided lower bound, etc.
    Values > 0 mean within bands; values close to 0 mean near edge.
    """
    if not summary_rows:
        return ''

    xs = [_derive(r, x_var) if x_var in ('lp_total', 'book_total', 'taker_total') else int(r.get(x_var, 0))
          for r in summary_rows]
    ys = [_derive(r, y_var) if y_var in ('lp_total', 'book_total', 'taker_total') else int(r.get(y_var, 0))
          for r in summary_rows]

    # Pull realized metrics from default__with_amm cols
    spreads = [_safe_float(r.get('default__with_amm__avg_clob_spread_bps')) for r in summary_rows]
    depths = [_safe_float(r.get('default__with_amm__avg_clob_depth')) for r in summary_rows]

    # Stability margin: for each criterion, compute (v - lower)/range or (upper - v)/range,
    # take min across criteria. Smaller value = closer to failing.
    def _margin(row):
        margin = float('inf')
        for c in criteria:
            metric = c['metric']
            # try with_amm version first
            v = _safe_float(row.get(f'default__with_amm__{metric}'))
            if not math.isfinite(v):
                v = _safe_float(row.get(metric))
            if not math.isfinite(v):
                continue
            lo = c.get('min', float('-inf'))
            hi = c.get('max', float('inf'))
            # Normalize: distance to nearest bound, scaled to bound range
            if math.isfinite(lo) and math.isfinite(hi):
                rng = hi - lo
                d = min(v - lo, hi - v) / max(rng, 1e-9)
            elif math.isfinite(lo):
                d = (v - lo) / max(abs(lo), abs(v), 1.0)
            elif math.isfinite(hi):
                d = (hi - v) / max(abs(hi), abs(v), 1.0)
            else:
                continue
            margin = min(margin, d)
        return margin if math.isfinite(margin) else float('nan')

    margins = [_margin(r) for r in summary_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (axA, axB), (axC, axD) = axes

    # Common grid for hex/scatter
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

    # ---- A: Coverage density (2D histogram) ----
    h, xedges, yedges = np.histogram2d(xs, ys, bins=[12, 12], range=[[x_min, x_max], [y_min, y_max]])
    imA = axA.pcolormesh(xedges, yedges, h.T, cmap='Blues', shading='flat',
                          edgecolors='white', linewidth=0.3)
    axA.scatter(xs, ys, s=25, color='#1a4b8c', edgecolors='white', linewidths=0.4, alpha=0.85)
    cbA = fig.colorbar(imA, ax=axA, shrink=0.85)
    cbA.set_label('# LHS points in cell', fontsize=9)
    axA.set_xlabel(x_var.replace('_', ' '))
    axA.set_ylabel(y_var.replace('_', ' '))
    axA.set_title('(A) LHS coverage density\n'
                  f'200 sampled compositions in {x_var}×{y_var} space',
                  fontsize=11, fontweight='bold')
    axA.grid(linestyle='--', alpha=0.25)

    # ---- B: Realized spread ----
    finite_sp = [v for v in spreads if math.isfinite(v)]
    if finite_sp:
        scB = axB.scatter(xs, ys, c=spreads, s=120, cmap='YlOrRd',
                          vmin=min(finite_sp), vmax=max(finite_sp),
                          edgecolors='black', linewidths=0.5, alpha=0.9)
        cbB = fig.colorbar(scB, ax=axB, shrink=0.85)
        cbB.set_label('Avg CLOB spread (bps)', fontsize=9)
    axB.set_xlabel(x_var.replace('_', ' '))
    axB.set_ylabel(y_var.replace('_', ' '))
    axB.set_title('(B) Realized CLOB spread per composition\n'
                  '(with-AMM branch, default scenario)',
                  fontsize=11, fontweight='bold')
    axB.grid(linestyle='--', alpha=0.25)

    # ---- C: Realized depth ----
    finite_d = [v for v in depths if math.isfinite(v)]
    if finite_d:
        scC = axC.scatter(xs, ys, c=depths, s=120, cmap='YlGn',
                          vmin=min(finite_d), vmax=max(finite_d),
                          edgecolors='black', linewidths=0.5, alpha=0.9)
        cbC = fig.colorbar(scC, ax=axC, shrink=0.85)
        cbC.set_label('Avg CLOB depth (units)', fontsize=9)
    axC.set_xlabel(x_var.replace('_', ' '))
    axC.set_ylabel(y_var.replace('_', ' '))
    axC.set_title('(C) Realized CLOB depth per composition',
                  fontsize=11, fontweight='bold')
    axC.grid(linestyle='--', alpha=0.25)

    # ---- D: Stability margin ----
    finite_m = [m for m in margins if math.isfinite(m)]
    if finite_m:
        vmin_m = max(0, min(finite_m))  # never below 0 since all pass
        vmax_m = max(finite_m)
        scD = axD.scatter(xs, ys, c=margins, s=120, cmap='RdYlGn',
                          vmin=vmin_m, vmax=vmax_m,
                          edgecolors='black', linewidths=0.5, alpha=0.9)
        cbD = fig.colorbar(scD, ax=axD, shrink=0.85)
        cbD.set_label('Min stability margin (norm. distance to nearest threshold)',
                      fontsize=9)
    axD.set_xlabel(x_var.replace('_', ' '))
    axD.set_ylabel(y_var.replace('_', ' '))
    axD.set_title('(D) Per-composition stability margin\n'
                  'Lower (red) = closer to a criterion edge; higher (green) = comfortable',
                  fontsize=11, fontweight='bold')
    axD.grid(linestyle='--', alpha=0.25)

    # Overall title
    n_total = len(summary_rows)
    n_stable = sum(1 for r in summary_rows if str(r.get('overall_stable', '')).lower() in ('true', '1'))
    fig.suptitle(f'Composition sweep stability diagnostic — {n_stable}/{n_total} points pass all criteria '
                 f'({100*n_stable/n_total:.0f}%)',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Color-consistent scale helper (across scenarios for the same outcome)
# ---------------------------------------------------------------------------

def global_scale_for_outcome(all_effects: List[Dict],
                              outcome: str,
                              scenarios: Sequence[str]) -> Tuple[float, float]:
    """Compute symmetric vmin/vmax across all scenarios for one outcome."""
    vals = []
    for sc in scenarios:
        key = f'{sc}__{outcome}__delta'
        for r in all_effects:
            v = _safe_float(r.get(key))
            if math.isfinite(v):
                vals.append(v)
    if not vals:
        return (-1.0, 1.0)
    abs_max = max(abs(min(vals)), abs(max(vals)))
    return (-abs_max, abs_max)
