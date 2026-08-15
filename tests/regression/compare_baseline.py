from __future__ import annotations

import json
import math
from pathlib import Path

from tests.regression.collect_metrics import build_snapshot


BASELINE_PATH = Path("tests/baselines/fx_regression_baseline.json")


ABS_TOLERANCES = {
    "metrics.n_iterations": 0.0,
    "metrics.n_trades": 0.0,
    "metrics.avg_cost_bps.clob_Q1": 0.5,
    "metrics.avg_cost_bps.clob_Q5": 1.0,
    "metrics.avg_cost_bps.clob_Q20": 2.0,
    "metrics.avg_cost_bps.cpmm_Q5": 1.0,
    "metrics.avg_cost_bps.hfmm_Q5": 1.0,
    "metrics.avg_flow_share.clob": 0.03,
    "metrics.avg_flow_share.cpmm": 0.03,
    "metrics.avg_flow_share.hfmm": 0.03,
    "metrics.cost_correlation_Q5.clob_cpmm": 0.08,
    "metrics.cost_correlation_Q5.clob_hfmm": 0.08,
}


def flatten(data: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, full))
        elif isinstance(value, (int, float)):
            out[full] = float(value)
    return out


def tolerance_for(path: str) -> float:
    if path in ABS_TOLERANCES:
        return ABS_TOLERANCES[path]
    if "theta_bins" in path and path.endswith("total_volume"):
        return 15.0
    if "theta_bins" in path and path.endswith("avg_cost_bps"):
        return 2.0
    return 1e-6


def main() -> None:
    if not BASELINE_PATH.exists():
        raise SystemExit(
            f"Baseline file not found: {BASELINE_PATH}. "
            "Generate it with: python -m tests.regression.collect_metrics --write tests/baselines/fx_regression_baseline.json"
        )

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = build_snapshot()

    bflat = flatten(baseline)
    cflat = flatten(current)

    failures: list[str] = []

    for key, bval in bflat.items():
        if key.startswith("meta."):
            continue
        cval = cflat.get(key)
        if cval is None:
            failures.append(f"missing metric: {key}")
            continue

        if math.isnan(bval) and math.isnan(cval):
            continue

        tol = tolerance_for(key)
        delta = abs(cval - bval)
        if delta > tol:
            failures.append(
                f"{key}: current={cval:.6f}, baseline={bval:.6f}, "
                f"delta={delta:.6f} > tol={tol:.6f}"
            )

    if failures:
        print("Regression check FAILED:")
        for line in failures:
            print(f" - {line}")
        raise SystemExit(1)

    print("Regression check passed against baseline.")


if __name__ == "__main__":
    main()
