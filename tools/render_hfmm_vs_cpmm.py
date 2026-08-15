"""Self-contained HFMM-vs-CPMM comparison figure + LaTeX cost table.

Generates, purely from the pool classes at the paper's calibrated parameters
(HFMM A=18, fee=5 bps; CPMM standard 30 bps benchmark fee), an instantaneous
"static pool at parity" comparison:

  output/main_aware/hfmm_vs_cpmm_curves.png   — two panels:
     (a) all-in execution cost (bps, log-y) vs trade size Q
     (b) pure bonding-curve slippage (bps) vs Q — isolates the invariant effect

and prints a LaTeX tabular (CPMM vs HFMM, slippage / all-in / ratio) to stdout
for pasting into the paper. Both the figure and the table come from the SAME
numbers so they cannot disagree.

Run: python tools/render_hfmm_vs_cpmm.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AgentBasedModel.venues.amm import CPMMPool, HFMMPool  # noqa: E402

PRICE = 100.0
RESERVES = 1000.0          # model units; 1 unit ~ EUR 1mn base notional
CPMM_FEE = 0.002           # 20 bps — matches calibration/primary_model.json
HFMM_FEE = 0.0005          # 5 bps — calibrated (Table 1)
HFMM_A = 18.0              # calibrated amplification (Table 1)
Q_GRID = [1, 5, 10, 20, 50, 100]

# Representative dealer CLOB book (calibration-consistent): quoted spread 2 bps
# (half-spread 1 bp), one-sided near-touch depth ~240 units laddered 1 bp apart.
# clob_cost_bps() below reproduces CLOBVenue.quote_buy's all-in cost
# (half-spread + book-walk, zero venue fee) — see AgentBasedModel/venues/clob.py.
CLOB_HALF_SPREAD_BPS = 1.0
CLOB_TICK_BPS = 1.0
CLOB_DEPTH_PER_LEVEL = 60.0
CLOB_N_LEVELS = 60

OUT_PNG = "output/main_aware/hfmm_vs_cpmm_curves.png"

CPMM_COLOR = "#d1751f"
HFMM_COLOR = "#2c7a3f"
CLOB_COLOR = "#3b5b92"


def clob_cost_bps(Q: float) -> float:
    """All-in CLOB execution cost (bps) for a market buy of size Q, walking a
    static calibrated book. Matches CLOBVenue.quote_buy: cost = 1e4*(vwap-mid)/mid."""
    mid = PRICE
    remaining, spent = float(Q), 0.0
    for k in range(CLOB_N_LEVELS):
        lvl_bps = CLOB_HALF_SPREAD_BPS + k * CLOB_TICK_BPS
        price = mid * (1.0 + lvl_bps / 10_000.0)
        fill = min(remaining, CLOB_DEPTH_PER_LEVEL)
        spent += fill * price
        remaining -= fill
        if remaining <= 1e-12:
            break
    if remaining > 1e-9:
        return float("nan")
    vwap = spent / Q
    return 10_000.0 * (vwap - mid) / mid


def _fresh_pools():
    cpmm = CPMMPool(x=RESERVES, y=RESERVES * PRICE, fee=CPMM_FEE)
    hfmm = HFMMPool(x=RESERVES, y=RESERVES * PRICE, A=HFMM_A, fee=HFMM_FEE, rate=PRICE)
    return cpmm, hfmm


def compute_rows():
    cpmm, hfmm = _fresh_pools()
    rows = []
    for Q in Q_GRID:
        c = cpmm.quote_buy(Q, S_t=PRICE)
        h = hfmm.quote_buy(Q, S_t=PRICE)
        ratio = c["cost_bps"] / h["cost_bps"] if h["cost_bps"] > 0 else float("nan")
        rows.append({
            "Q": Q,
            "cpmm_slip": c["slippage_bps"], "cpmm_all": c["cost_bps"],
            "hfmm_slip": h["slippage_bps"], "hfmm_all": h["cost_bps"],
            "clob_cost": clob_cost_bps(Q),
            "ratio": ratio,
        })
    return rows


def render_figure(rows):
    try:
        from AgentBasedModel.visualization.paper_style import use_paper_style
        use_paper_style()
    except Exception:
        pass

    Qs = [r["Q"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # Panel (a): pure bonding-curve slippage (AMM invariant only), linear
    ax1.plot(Qs, [r["cpmm_slip"] for r in rows], 's--', color=CPMM_COLOR, label="CPMM slippage")
    ax1.plot(Qs, [r["hfmm_slip"] for r in rows], 's-', color=HFMM_COLOR, label="HFMM slippage")
    ax1.set_xlabel("Trade size $Q$ (base units)")
    ax1.set_ylabel("Bonding-curve slippage (bps)")
    ax1.set_title("(a) Pure bonding-curve slippage (AMM invariant only)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, alpha=0.3)

    # Panel (b): all-in execution cost across venues (incl. dealer CLOB), log-y
    ax2.plot(Qs, [r["cpmm_all"] for r in rows], 's--', color=CPMM_COLOR, label="CPMM (20 bps fee)")
    ax2.plot(Qs, [r["hfmm_all"] for r in rows], 's-', color=HFMM_COLOR, label="HFMM ($A=18$, 5 bps fee)")
    ax2.plot(Qs, [r["clob_cost"] for r in rows], 'o-', color=CLOB_COLOR, label="Dealer CLOB (2 bps spread)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Trade size $Q$ (base units)")
    ax2.set_ylabel("All-in execution cost (bps, log scale)")
    ax2.set_title("(b) All-in execution cost across venues")
    ax2.legend(loc="upper left", frameon=False)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


def print_latex_table(rows):
    print("\n% --- paste into cp.tex (cost table) ---")
    for r in rows:
        print(f"{r['Q']:>3} & {r['cpmm_slip']:>7.1f} & {r['cpmm_all']:>7.1f} & "
              f"{r['hfmm_slip']:>6.2f} & {r['hfmm_all']:>6.2f} & "
              f"{r['ratio']:>4.1f}$\\times$ \\\\")


if __name__ == "__main__":
    rows = compute_rows()
    render_figure(rows)
    print_latex_table(rows)
