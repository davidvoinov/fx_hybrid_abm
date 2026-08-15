from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from AgentBasedModel.simulator.simulator import Simulator


SEED = 42
N_ITER = 120


def build_snapshot() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)

    sim = Simulator.default_fx(
        n_noise=6,
        n_fx_takers=8,
        n_fx_fund=3,
        n_retail=6,
        n_institutional=2,
        cpmm_reserves=500,
        hfmm_reserves=500,
        stress_start=40,
        stress_end=90,
        price=100,
        amm_share_pct=25.0,
        deterministic=True,
    )
    sim.simulate(N_ITER, silent=True)

    logger = sim.logger
    summary = sim.logger.summary()
    bins = logger.bin_summary()

    def pick_cost(venue: str, q: int) -> float:
        return float(summary.get(f"avg_cost_{venue}_Q{q}", float("nan")))

    def pick_flow(venue: str) -> float:
        return float(summary.get(f"avg_flow_share_{venue}", 0.0))

    def pick_bin(bucket: str, venue: str) -> dict:
        data = bins.get(bucket, {}).get(venue, {})
        return {
            "avg_cost_bps": float(data.get("avg_cost_bps", float("nan"))),
            "total_volume": float(data.get("total_volume", 0.0)),
        }

    snapshot = {
        "meta": {
            "seed": SEED,
            "n_iter": N_ITER,
        },
        "metrics": {
            "n_iterations": int(summary.get("n_iterations", 0)),
            "n_trades": int(summary.get("n_trades", 0)),
            "avg_cost_bps": {
                "clob_Q1": pick_cost("clob", 1),
                "clob_Q5": pick_cost("clob", 5),
                "clob_Q20": pick_cost("clob", 20),
                "cpmm_Q5": pick_cost("cpmm", 5),
                "hfmm_Q5": pick_cost("hfmm", 5),
            },
            "avg_flow_share": {
                "clob": pick_flow("clob"),
                "cpmm": pick_flow("cpmm"),
                "hfmm": pick_flow("hfmm"),
            },
            "cost_correlation_Q5": {
                "clob_cpmm": float(logger.cost_correlation("clob", "cpmm", Q=5)),
                "clob_hfmm": float(logger.cost_correlation("clob", "hfmm", Q=5)),
            },
            "theta_bins": {
                "small": {
                    "clob": pick_bin("small", "clob"),
                    "cpmm": pick_bin("small", "cpmm"),
                    "hfmm": pick_bin("small", "hfmm"),
                },
                "medium": {
                    "clob": pick_bin("medium", "clob"),
                    "cpmm": pick_bin("medium", "cpmm"),
                    "hfmm": pick_bin("medium", "hfmm"),
                },
                "large": {
                    "clob": pick_bin("large", "clob"),
                    "cpmm": pick_bin("large", "cpmm"),
                    "hfmm": pick_bin("large", "hfmm"),
                },
            },
        },
    }
    return snapshot


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    snapshot = sanitize_for_json(build_snapshot())
    text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)

    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
        print(f"Baseline written to: {args.write}")
        return

    print(text)


if __name__ == "__main__":
    main()
