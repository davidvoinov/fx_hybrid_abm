"""Every figure the manuscript carries, drawn from the stored panels.

One module, one style, one source of numbers. Each function reads an artifact
written by a runner and returns the path it drew to, so a figure in the paper
can always be traced to the panel and the signature behind it.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from AgentBasedModel.visualization.paper_style import (
    COLOR_BAD,
    COLOR_CONTROL,
    COLOR_GOOD,
    COLOR_NEUTRAL,
    COLOR_TREATMENT,
    PAPER_PALETTE,
    use_paper_style,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGURES = os.path.join(ROOT, 'EconMod', 'figures')

ARM_LABEL = {
    'none': 'dealer only',
    'reserve': 'reserve priced pool',
    'reserve_frozen': 'pool, capital held',
    'dealer_of_last_resort': 'obliged quoter',
    'passive_book': 'passive ladder',
}
CHANNEL_LABEL = {
    'committed_quoting': 'committed quoting',
    'pricing_schedule': 'pricing schedule',
    'capital_flight': 'flight of provider capital',
}
# Okabe and Ito, which separates for the colour blind and keeps its ordering
# in grayscale. One assignment for every figure, so a reader who learns the
# arms on one plot reads them on the rest without looking again at a legend.
CONTROL = '#666666'      # neutral grey, the dealer only market
POOL = '#0072B2'         # blue, the reserve priced pool
POOL_FROZEN = '#56B4E9'  # sky blue, the same pool with its capital held
QUOTER = '#D55E00'       # vermillion, the obliged quoter
LADDER = '#009E73'       # green, the passive ladder
GOOD = '#009E73'
BAD = '#D55E00'
RULE = '#333333'

ARM_COLOUR = {
    'none': CONTROL,
    'reserve': POOL,
    'reserve_frozen': POOL_FROZEN,
    'dealer_of_last_resort': QUOTER,
    'passive_book': LADDER,
}
ARM_DASH = {
    'none': (4, 2),
    'reserve': (),
    'reserve_frozen': (1, 1.4),
    'dealer_of_last_resort': (6, 2, 1, 2),
    'passive_book': (2, 1.5),
}
ARM_MARKER = {
    'none': 's', 'reserve': 'o', 'reserve_frozen': 'D',
    'dealer_of_last_resort': '^', 'passive_book': 'v',
}


def _frame(ax) -> None:
    """Two spines, not four. The box adds ink and carries no information."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)
# Single column Elsevier CAS: the text block is about 6.5 inches wide.
WIDE = (6.6, 2.7)
TALL = (6.8, 3.5)


def _load(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as handle:
        return json.load(handle)


def _save(fig, filename):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def _interval_bars(ax, rows, xlabel, colour_by_sign=True):
    """Horizontal bars with the interval drawn on each."""
    positions = list(range(len(rows)))
    means = [mean for _, mean, _, _ in rows]
    if colour_by_sign:
        colours = [GOOD if m < 0 else BAD for m in means]
    else:
        colours = [POOL] * len(rows)
    ax.barh(positions, means, color=colours, height=0.55, alpha=0.85)
    for y, (_, mean, low, high) in zip(positions, rows):
        ax.plot([low, high], [y, y], color=RULE, linewidth=1.2)
        for edge in (low, high):
            ax.plot([edge, edge], [y - 0.11, y + 0.11],
                    color=RULE, linewidth=1.2)
    ax.axvline(0.0, color='black', linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels([name for name, _, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    _frame(ax)


# ── 1. executability by size ─────────────────────────────────────────────
def fig_availability(data='output/figure_data.json') -> str:
    payload = _load(data)
    served = payload['availability']
    sizes = sorted({float(k) for arm in served.values() for k in arm})
    arms = [a for a in ARM_LABEL if a in served]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        heights = [served[arm][str(s)] * 100.0 for s in sizes]
        ax.bar([x + i * width for x in range(len(sizes))], heights,
               width=width, label=ARM_LABEL[arm], color=ARM_COLOUR[arm],
               edgecolor='white', linewidth=0.6)
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(sizes))])
    ax.set_xticklabels([f'{s:g}' for s in sizes])
    ax.set_xlabel('trade size, base units')
    ax.set_ylabel('per cent of crisis periods')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    _frame(ax)
    return _save(fig, 'availability_by_size.pdf')


# ── 2. the decomposition ─────────────────────────────────────────────────
def fig_decomposition(data='output/facility_arms.json') -> str:
    payload = _load(data)
    contrasts = payload['contrasts']
    order = ['committed_quoting', 'pricing_schedule', 'capital_flight']
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, key, title in (
            (axes[0], 'delta_peak', 'Peak executable cost'),
            (axes[1], 'delta_cost', 'What customers paid')):
        rows = []
        for name in order:
            block = contrasts.get(name, {}).get(key)
            if not block:
                continue
            rows.append((CHANNEL_LABEL[name], block['mean'],
                         block['ci'][0], block['ci'][1]))
        _interval_bars(ax, rows, 'change in basis points')
        ax.set_title(title, loc='left', fontsize=9)
    axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    return _save(fig, 'decomposition.pdf')


# ── 3. the two cost measures ─────────────────────────────────────────────
def fig_cost_measures(data='output/facility_arms.json') -> str:
    payload = _load(data)
    arms = [a for a in ARM_LABEL if a in payload['arms'] and a != 'none']
    realised = [payload['arms'][a]['delta_cost']['mean'] for a in arms]
    quoted = [payload['arms'][a]['delta_quoted_cost']['mean'] for a in arms]
    positions = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=WIDE)
    ax.barh(positions - 0.19, realised, height=0.36, color=POOL,
            alpha=0.9, label='paid on the trades that happened')
    ax.barh(positions + 0.19, quoted, height=0.36, color=POOL_FROZEN,
            alpha=0.9, label='quoted at one size, every period')
    ax.axvline(0.0, color='black', linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels([ARM_LABEL[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel('change against the dealer only control, basis points')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    _frame(ax)
    return _save(fig, 'cost_measures.pdf')


# ── 4. the path of the dislocation ───────────────────────────────────────
def fig_dislocation_path(data='output/figure_data.json') -> str:
    payload = _load(data)
    offsets = payload['path_offsets']
    fig, ax = plt.subplots(figsize=WIDE)
    for arm, series in payload['path'].items():
        line, = ax.plot(offsets, series, label=ARM_LABEL.get(arm, arm),
                        color=ARM_COLOUR.get(arm, COLOR_NEUTRAL), linewidth=1.6)
        dashes = ARM_DASH.get(arm, ())
        if dashes:
            line.set_dashes(dashes)
    ax.set_xlabel('periods after the shock')
    ax.set_ylabel('round trip cost, basis points')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _frame(ax)
    return _save(fig, 'dislocation_path.pdf')


# ── 5. what happens to the providers ─────────────────────────────────────
def fig_providers(data='output/figure_data.json') -> str:
    payload = _load(data)
    offsets = payload['path_offsets']
    block = payload['providers']
    fig, axes = plt.subplots(2, 2, figsize=TALL)
    panels = (
        (axes[0][0], [v * 100 for v in block['reserves']],
         'Reserves left in the pool', 'per cent of opening'),
        (axes[0][1], block['wallets'],
         'Capital pulled into provider wallets', 'quote units'),
        (axes[1][0], block['open'],
         'Providers still committed', 'count'),
        (axes[1][1], block['cost'],
         "The pool's own price", 'round trip, basis points'),
    )
    for ax, series, title, ylabel in panels:
        ax.plot(offsets, series, color=POOL, linewidth=1.6)
        ax.set_title(title, loc='left', fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel('periods after the shock', fontsize=8)
        if ylabel == 'count':
            # Providers are counted, so a tick at four and a half means nothing.
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _frame(ax)
    fig.tight_layout()
    return _save(fig, 'provider_fate.pdf')


# ── 6. adverse selection ─────────────────────────────────────────────────
def fig_markout(data='output/resilience/flow_selection.json') -> str:
    """Markout by venue and state, on a panel each.

    The two states differ by two orders: a venue earns about a basis point per
    unit while conditions hold and loses sixty when they break. Drawn on one
    scale the calm state is a line on the axis, so each state carries its own.
    """
    payload = _load(data)
    states = [k for k in payload if k not in ('crisis_less_calm', 'provenance')]
    titles = {'calm': 'Calm market'}
    venues = (('facility', 'the reserve priced pool', POOL),
              ('book', 'the order book', CONTROL))
    fig, axes = plt.subplots(1, len(states), figsize=(6.4, 2.9))
    for ax, state in zip(np.atleast_1d(axes), states):
        for x, (key, label, colour) in enumerate(venues):
            block = payload[state][key]
            ax.bar([x], [block['mean']], width=0.55, color=colour, alpha=0.9)
            ax.plot([x, x], block['ci'], color=COLOR_NEUTRAL, linewidth=1.2)
            for edge in block['ci']:
                ax.plot([x - 0.09, x + 0.09], [edge, edge],
                        color=COLOR_NEUTRAL, linewidth=1.2)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(venues)))
        ax.set_xticklabels([label for _, label, _ in venues], fontsize=8)
        ax.set_title(titles.get(state, 'Crisis window'), loc='left', fontsize=9)
        ax.set_ylabel('markout per unit, basis points', fontsize=8)
        _frame(ax)
    fig.tight_layout()
    return _save(fig, 'markout.pdf')


# ── 7. cost by size, and where the book stops ────────────────────────────
def fig_size_curve(data='output/figure_data.json') -> str:
    payload = _load(data)
    grid = payload['size_grid']
    curve = payload['size_curve']
    book = [curve[str(s)]['book'] for s in grid]
    pool = [curve[str(s)]['pool'] for s in grid]
    finite = [s for s, b in zip(grid, book) if np.isfinite(b)]
    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot([s for s, b in zip(grid, book) if np.isfinite(b)],
            [b for b in book if np.isfinite(b)],
            color=CONTROL, linewidth=1.6, label='the order book')
    ax.plot(grid, pool, color=POOL, linewidth=1.6,
            label='the reserve priced pool')
    if finite and len(finite) < len(grid):
        edge = max(finite)
        ax.axvline(edge, color=BAD, linewidth=1.0, linestyle=':')
        ax.annotate('the book fills nothing beyond here',
                    xy=(edge, ax.get_ylim()[1] * 0.62),
                    xytext=(-8, 0), textcoords='offset points',
                    fontsize=8, color=BAD, va='center', ha='right')
    ax.set_xscale('log')
    ax.set_xlabel('trade size, base units')
    ax.set_ylabel('round trip cost, basis points')
    ax.legend(frameon=False, fontsize=8)
    _frame(ax)
    return _save(fig, 'size_curve.pdf')


# ── 8. the fee frontier ──────────────────────────────────────────────────
def fig_fee_frontier(data='output/figure_data.json') -> str:
    payload = _load(data)
    frontier = payload['frontier']
    fees = sorted(float(k) for k in frontier)
    calm = [frontier[str(f)]['calm'] for f in fees]
    crisis = [frontier[str(f)]['crisis'] for f in fees]
    breakeven = [(c / (c - k) * 100.0) if (c > 0 and k < 0) else 0.0
                 for c, k in zip(calm, crisis)]
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    axes[0].plot(fees, crisis, color=BAD, linewidth=1.6, marker='o',
                 markersize=3.5, label='crisis window')
    axes[0].plot(fees, calm, color=GOOD, linewidth=1.6, marker='o',
                 markersize=3.5, label='calm window')
    axes[0].axhline(0.0, color='black', linewidth=0.8)
    axes[0].set_xlabel('fee, basis points')
    axes[0].set_ylabel('result, per cent of capital')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title('What the capital earns', loc='left', fontsize=9)
    _frame(axes[0])
    axes[1].plot(fees, breakeven, color=POOL, linewidth=1.6,
                 marker='o', markersize=3.5)
    axes[1].set_xlabel('fee, basis points')
    axes[1].set_ylabel('per cent of windows')
    axes[1].set_title('Crisis frequency it could carry', loc='left', fontsize=9)
    _frame(axes[1])
    fig.tight_layout()
    return _save(fig, 'fee_frontier.pdf')

def fig_cascade(data='output/figure_data.json') -> str:
    payload = _load(data)
    grid = payload['severity_grid']
    block = payload['cascade']
    off = [block[str(v)]['0.0'] for v in grid]
    on = [block[str(v)]['4.0'] for v in grid]
    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(grid, off, color=CONTROL, linewidth=1.6, marker='s',
            markersize=3.5, label='channel off')
    ax.plot(grid, on, color=QUOTER, linewidth=1.6, marker='o',
            markersize=3.5, label='channel on')
    ax.set_xlabel('severity of the shock, multiple of the calibrated one')
    ax.set_ylabel('peak round trip, basis points')
    ax.legend(frameon=False, fontsize=8)
    _frame(ax)
    return _save(fig, 'capacity_threshold.pdf')


# ── 11. what each arm returns for what it consumes ───────────────────────
def fig_efficiency(pattern='output/resilience/welfare_{}.json') -> str:
    """Customer benefit against provider loss, one point per arm.

    The question H5 asks is not whether an arrangement is worth having but
    which of them returns most for what it consumes, and a table of two
    columns hides the ratio the question is about.
    """
    arms = ('reserve', 'reserve_frozen', 'dealer_of_last_resort', 'passive_book')
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    points = []
    for arm in arms:
        try:
            block = _load(pattern.format(arm))['summary']['medians']
        except (OSError, KeyError):
            continue
        points.append((arm, block['matched_notional_user_benefit_bps'],
                       -block['lp_operating_result']))
    # The limits are fixed before anything is drawn into the plane. The rays
    # run far past the markers, so autoscaling on them would move the frame
    # out from under any label already placed against it.
    xmax = max(b for _, b, _ in points) * 1.30
    ymax = max(l for _, _, l in points) * 1.45
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)

    # Iso cost rays: every point on one of them buys a basis point at the
    # same price, so the reader can order the arms without arithmetic. Each
    # is labelled where it leaves the frame, clear of the markers.
    for ratio in (100, 200, 300):
        if ratio * xmax <= ymax:
            edge, offset, ha, va = (xmax, ratio * xmax), (-3, 3), 'right', 'bottom'
        else:
            edge, offset, ha, va = (ymax / ratio, ymax), (3, -3), 'left', 'top'
        ax.plot([0.0, edge[0]], [0.0, edge[1]], color=RULE, linewidth=0.6,
                linestyle=':', zorder=1)
        ax.annotate(f'{ratio} per bp', edge, xytext=offset,
                    textcoords='offset points', fontsize=7, color=RULE,
                    ha=ha, va=va)

    # The arms fall into close pairs. Within a pair the upper label is thrown
    # up and the lower one down, so the two never meet in the middle.
    def _offset(n):
        _, bx, ly = points[n]
        rest = [(abs(bx - b) / xmax + abs(ly - l) / ymax, l)
                for m, (_, b, l) in enumerate(points) if m != n]
        return (8, 5) if ly >= min(rest)[1] else (8, -11)

    for n, (arm, benefit, loss) in enumerate(points):
        ax.scatter(benefit, loss, s=70, color=ARM_COLOUR[arm],
                   marker=ARM_MARKER[arm], zorder=3, label=ARM_LABEL[arm])
        ax.annotate(f'{loss / benefit:,.0f} per bp', (benefit, loss),
                    xytext=_offset(n), textcoords='offset points', fontsize=8,
                    color=ARM_COLOUR[arm], zorder=4)
    ax.set_xlabel('customer benefit, basis points')
    ax.set_ylabel('provider loss, quote units')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _frame(ax)
    return _save(fig, 'efficiency.pdf')


def fig_resilience(data='output/facility_arms.json') -> str:
    """Amplitude, decay and persistence, which is what resilience decomposes into.

    A market is resilient when a shock displaces it little, when the
    displacement decays quickly and when it does not linger. An arrangement can
    do well on one and badly on another, so the three are drawn beside each
    other and not summarised into one number.
    """
    payload = _load(data)
    arms = [a for a in ARM_LABEL if a in payload['arms'] and a != 'none']
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True)
    panels = ((axes[0], 'delta_peak', 'Amplitude', 'peak, bps'),
              (axes[1], 'delta_decay_rate', 'Decay', 'rate per period'),
              (axes[2], 'delta_excess', 'Persistence', 'cumulative excess'))
    for ax, key, title, xlabel in panels:
        rows = []
        for arm in arms:
            block = payload['arms'].get(arm, {}).get(key)
            if not isinstance(block, dict) or block['mean'] != block['mean']:
                continue
            rows.append((ARM_LABEL[arm], block['mean'],
                         block['ci'][0], block['ci'][1]))
        if not rows:
            continue
        # A faster decay is an improvement, so its sign is turned to match.
        if key == 'delta_decay_rate':
            rows = [(n, -m, -h, -l) for n, m, l, h in rows]
        _interval_bars(ax, rows, xlabel)
        ax.set_title(title, loc='left', fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.tight_layout()
    return _save(fig, 'resilience.pdf')


def draw_all() -> list:
    """Draw every figure, reporting the ones whose data is not there yet."""
    use_paper_style()
    drawn = []
    for maker in ALL:
        try:
            drawn.append(maker())
        except Exception as exc:
            print(f'  {maker.__name__}: {exc}')
    return drawn


if __name__ == '__main__':
    for path in draw_all():
        print(os.path.relpath(path, ROOT))


# ── 12. the second pair ──────────────────────────────────────────────────
def fig_pair_comparison(main='output/resilience/welfare_{}.json',
                        branch='output/eurchf/welfare_{}.json') -> str:
    """What the arms return and what they consume, on two pairs.

    The ordering the main pair reports is not a property of the designs. It
    turns on the size of the dislocation, and the second pair carries one an
    order of magnitude larger, so the panel puts the two beside each other
    and asserts neither as the result.
    """
    arms = ('reserve', 'dealer_of_last_resort', 'passive_book')
    labels = ['reserve priced\npool', 'obliged\nquoter', 'passive\nladder']
    series = []
    for pattern in (main, branch):
        benefit, ratio = [], []
        for arm in arms:
            block = _load(pattern.format(arm))['summary']['medians']
            b = float(block['matched_notional_user_benefit_bps'])
            loss = -float(block['lp_operating_result'])
            benefit.append(b)
            ratio.append(loss / b if b else float('nan'))
        series.append((benefit, ratio))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    x = np.arange(len(arms))
    width = 0.36
    names = ('EUR/USD, 2020', 'EUR/CHF, 2015')
    colours = (CONTROL, POOL)
    for ax, index, title, ylabel in (
            (axes[0], 0, 'What customers gained', 'benefit, basis points'),
            (axes[1], 1, 'What it cost to give it', 'provider loss per basis point')):
        for n, ((benefit, ratio), name, colour) in enumerate(
                zip(series, names, colours)):
            values = (benefit, ratio)[index]
            ax.bar(x + (n - 0.5) * width, values, width=width,
                   color=colour, alpha=0.9, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, loc='left', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        _frame(ax)
    # The right panel spans thirty to one, which no linear axis shows.
    axes[1].set_yscale('log')
    # The tallest bar of the left panel is the first group, so the legend
    # goes to the panel that has room for it.
    axes[1].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    return _save(fig, 'pair_comparison.pdf')


# Every figure the manuscript prints. The list used to sit above the last
# function in this file and therefore could not name it, so the panel that
# compares the two pairs was drawn once by hand and never redrawn: it was the
# one figure in the paper that no run of the figure command could refresh, and
# it would have gone stale without saying so.
ALL = (fig_availability, fig_decomposition, fig_resilience, fig_cost_measures,
       fig_dislocation_path, fig_providers, fig_markout, fig_size_curve,
       fig_fee_frontier, fig_cascade, fig_efficiency, fig_pair_comparison)
