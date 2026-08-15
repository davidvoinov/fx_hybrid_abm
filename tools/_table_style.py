"""
Unified styling for all paper tables rendered as PNGs.

Single source of truth for table colours, fonts, dimensions, and the
rendering helper. Importable by all `tools/render_*.py` scripts so the
visual style across Tables 1-6 is exactly consistent.

Usage:
    from tools._table_style import (
        render_table, HELP_GREEN, HURT_RED, NS_GRAY,
        HEADER_BG, ROW_ALT, DPI,
    )
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Palette (single source of truth) --------------------------------------
HELP_GREEN   = '#c8e6c9'  # row/cell tinted green = AMM helps / target passes
HURT_RED     = '#ffcdd2'  # row/cell tinted red   = AMM hurts / target fails
NS_GRAY      = '#eeeeee'  # row/cell tinted gray  = not significant / no verdict
HEADER_BG    = '#37474f'  # header row background
HEADER_FG    = 'white'    # header row text
ROW_ALT      = '#f5f7fa'  # even-row alt background
ROW_BORDER   = '#cfd8dc'  # cell border
TEXT_DARK    = '#1a1a1a'
TEXT_MUTED   = '#555555'
DPI          = 200

# --- Typography -------------------------------------------------------------
HEADER_FONTSIZE  = 10
CELL_FONTSIZE    = 9
TITLE_FONTSIZE   = 13
FOOTNOTE_FONTSIZE = 8
TITLE_WEIGHT     = 'bold'
HEADER_WEIGHT    = 'bold'

# --- Layout -----------------------------------------------------------------
ROW_HEIGHT_IN    = 0.55          # vertical inches per row
EXTRA_MARGIN_IN  = 0.6           # extra inches for title + footnote
COL_PADDING_IN   = 1.0           # extra horizontal inches beyond sum of col widths
HEADER_LINE_W    = 1.2
CELL_LINE_W      = 0.6


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_table(rows: List[Dict],
                 col_labels: List[str],
                 col_keys: List[str],
                 col_widths: List[float],
                 *,
                 cell_color_fn: Optional[Callable[[Dict, str, str], Optional[str]]] = None,
                 row_filter: Optional[Callable[[Dict], bool]] = None,
                 title: str = '',
                 footnote: str = '',
                 out_path: str = '',
                 header_wrap_lines: int = 1) -> str:
    """Render a single table to PNG using the project's unified style.

    Parameters
    ----------
    rows: list of dicts (one row each)
    col_labels: human-readable header labels (newlines allowed for multi-line)
    col_keys:   keys into `rows[i]` to extract cell text
    col_widths: relative widths (inches) per column
    cell_color_fn: optional fn(row, col_key, raw_value) -> hex color or None
    row_filter:    optional fn(row) -> bool; rows where it returns False are dropped
    title:    suptitle text (rendered bold above the table)
    footnote: italic muted text below the table
    out_path: absolute or relative output PNG path
    header_wrap_lines: max number of header lines (controls header row height)

    Returns the out_path on success, '' on empty input.
    """
    if row_filter is not None:
        rows = [r for r in rows if row_filter(r)]
    n_rows = len(rows)
    n_cols = len(col_keys)
    if n_rows == 0:
        return ''

    fig_w = sum(col_widths) + COL_PADDING_IN
    header_h = max(1.0, 0.35 * header_wrap_lines + 0.45)
    fig_h = ROW_HEIGHT_IN * (n_rows + 3) + EXTRA_MARGIN_IN
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, sum(col_widths))
    ax.set_ylim(0, n_rows + header_h)
    ax.invert_yaxis()
    ax.axis('off')

    # Column x-positions (left edges)
    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Header
    for cx, w, label in zip(col_x, col_widths, col_labels):
        ax.add_patch(Rectangle((cx, 0), w, header_h,
                               facecolor=HEADER_BG, edgecolor='white',
                               linewidth=HEADER_LINE_W))
        ax.text(cx + w / 2, header_h / 2, label,
                ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight=HEADER_WEIGHT,
                color=HEADER_FG)

    # Data rows
    for r_idx, row in enumerate(rows):
        y = header_h + r_idx
        base_bg = ROW_ALT if r_idx % 2 == 0 else 'white'
        for cx, w, key in zip(col_x, col_widths, col_keys):
            raw = row.get(key, '')
            cell_bg = cell_color_fn(row, key, raw) if cell_color_fn else None
            facecolor = cell_bg if cell_bg else base_bg
            ax.add_patch(Rectangle((cx, y), w, 1,
                                   facecolor=facecolor,
                                   edgecolor=ROW_BORDER,
                                   linewidth=CELL_LINE_W))
            ax.text(cx + w / 2, y + 0.5, str(raw),
                    ha='center', va='center',
                    fontsize=CELL_FONTSIZE, color=TEXT_DARK)

    # Title + footnote
    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight=TITLE_WEIGHT, y=0.99)
    if footnote:
        fig.text(0.5, 0.005, footnote, ha='center', va='bottom',
                 fontsize=FOOTNOTE_FONTSIZE, style='italic', color=TEXT_MUTED)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path
