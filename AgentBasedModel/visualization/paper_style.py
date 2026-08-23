"""Shared journal-style matplotlib configuration.

Single entry point :func:`use_paper_style` applies a spare, print-first style
modelled on the example reference figure: a white plotting area, no gridlines,
a sans-serif face, short outward ticks, only the left and bottom spines, and a
restrained grayscale palette that survives black-and-white printing. Axis
labels are placed at the *ends* of the axes (the y-label sits horizontally above
the axis, the x-label at the right below it) via the :func:`end_labels` helper.

Call :func:`use_paper_style` once at the top of any script before figures are
created, then :func:`end_labels` on each axis to position the labels.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


_APPLIED = False


# Restrained, grayscale-legible palette. The first two entries are the standing
# semantic pair (hero series in near-black, baseline in mid-gray); the rest are
# darker/lighter grays for occasional extra series.
PAPER_PALETTE: Sequence[str] = (
    '#1a1a1a',  # near-black   -- primary series / "with AMM"
    '#9e9e9e',  # mid gray     -- baseline / "without AMM"
    '#5a5a5a',  # dark gray
    '#bdbdbd',  # light gray
    '#3a3a3a',  # charcoal
    '#787878',  # gray
)

# Semantic colors that survive grayscale printing.
# The series under study and the control it is read against. These were
# named for a comparison of a market with a facility against the same
# market without one, which the resource matched arms replaced.
COLOR_TREATMENT = PAPER_PALETTE[0]  # near-black, the series under study
COLOR_CONTROL = PAPER_PALETTE[1]    # mid gray, the control
COLOR_GOOD = '#2f6b4f'              # muted green (improvement)
COLOR_BAD = '#8c3b2f'               # muted brick (degradation)
COLOR_NEUTRAL = PAPER_PALETTE[2]


def use_paper_style(*, base_size: float = 10.0,
                    figure_dpi: int = 150,
                    save_dpi: int = 300,
                    palette: Optional[Iterable[str]] = None) -> None:
    """Activate the paper style. Idempotent; safe to call repeatedly.

    Parameters
    ----------
    base_size : float
        Base font size for tick labels; titles/labels scale relative to it.
    figure_dpi : int
        Screen dpi (affects ``plt.show`` and inline previews).
    save_dpi : int
        DPI used when ``savefig`` is called without an explicit ``dpi``.
    palette : iterable of str, optional
        Overrides the default categorical color cycle.
    """
    global _APPLIED

    # Start from a clean baseline so prior style choices do not leak through.
    plt.style.use('default')

    colors = list(palette) if palette is not None else list(PAPER_PALETTE)

    mpl.rcParams.update({
        # Typography -- plain sans-serif, matching the reference figure.
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial',
                            'Liberation Sans'],
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': True,

        # Sizes.
        'font.size': base_size,
        'axes.titlesize': base_size + 1.0,
        'axes.titleweight': 'regular',
        'axes.titlepad': 6.0,
        'axes.labelsize': base_size,
        'xtick.labelsize': base_size - 1.0,
        'ytick.labelsize': base_size - 1.0,
        'legend.fontsize': base_size - 1.5,
        'figure.titlesize': base_size + 2.0,

        # Frame -- full rectangular box (all four spines), in near-black.
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#1a1a1a',
        'axes.labelcolor': '#000000',
        'axes.titlecolor': '#000000',
        'text.color': '#000000',
        'xtick.color': '#1a1a1a',
        'ytick.color': '#1a1a1a',
        'xtick.labelcolor': '#000000',
        'ytick.labelcolor': '#000000',

        # Short outward tick marks, as in the reference figure.
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.top': False,
        'ytick.right': False,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.major.pad': 4.0,
        'ytick.major.pad': 4.0,

        # No grid.
        'axes.grid': False,
        'axes.axisbelow': True,

        # Lines & markers.
        'lines.linewidth': 1.6,
        'lines.markersize': 4.0,
        'lines.solid_capstyle': 'round',
        'patch.linewidth': 0.8,

        # Legend -- frameless, unobtrusive.
        'legend.frameon': False,
        'legend.handlelength': 1.8,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.0,
        'legend.borderpad': 0.3,

        # Output -- white throughout.
        'figure.dpi': figure_dpi,
        'savefig.dpi': save_dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.constrained_layout.use': False,

        # Categorical color cycle.
        'axes.prop_cycle': mpl.cycler(color=colors),
    })

    _APPLIED = True


def end_labels(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               *, xy: float = -0.085, yy: float = 1.04) -> None:
    """Place axis labels at the ends of the axes, reference-figure style.

    The x-label is right-aligned below the right end of the x-axis; the y-label
    is written horizontally, left-aligned, just above the top of the y-axis.

    Parameters
    ----------
    ax : matplotlib axis
    xlabel, ylabel : str, optional
        Label text. If ``None`` the corresponding label is left untouched.
    xy : float
        Vertical offset (axes fraction) of the x-label below the axis.
    yy : float
        Vertical position (axes fraction) of the y-label above the axis.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_label_coords(1.0, xy)
        ax.xaxis.label.set_horizontalalignment('right')
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=0)
        ax.yaxis.set_label_coords(0.0, yy)
        ax.yaxis.label.set_horizontalalignment('left')
        ax.yaxis.label.set_verticalalignment('bottom')


def tidy_origin(ax) -> None:
    """Suppress the duplicate ``0`` at the origin.

    When both axes begin at zero, the x-axis ``0`` and the y-axis ``0`` print
    on top of each other at the bottom-left corner. This hides the x-axis ``0``
    label in that case, leaving a single zero.
    """
    ax.figure.canvas.draw()
    x0, _ = ax.get_xlim()
    y0, _ = ax.get_ylim()
    if abs(x0) < 1e-9 and abs(y0) < 1e-9:
        for lab in ax.get_xticklabels():
            if lab.get_text() in ('0', '0.0'):
                lab.set_visible(False)


def is_applied() -> bool:
    return _APPLIED
