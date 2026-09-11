from __future__ import annotations

import json
import math
from pathlib import Path

from tests.regression.collect_metrics import build_snapshot


BASELINE_PATH = Path("tests/baselines/fx_regression_baseline.json")


# One seeded run is bit for bit reproducible on the same tree, so the only
# difference this check has to tolerate is floating point dust. A per metric
# table of absolute allowances instead, running to two basis points on
# quantities whose own scale is a fraction of one, admits differences that are
# not dust. A detector set that loose passes a doubling of the large trade cost
# without a word, which is not the job the module is named for.
TOLERANCE = 1e-6


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
    """One allowance, and it is for arithmetic and not for behaviour."""
    del path
    return TOLERANCE


def compare() -> list[str]:
    """Differences between the current model and the stored baseline."""
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

    return failures


def test_the_model_matches_its_regression_baseline():
    """The baseline exists to catch a change nobody meant to make.

    It was never collected: this module carried a ``main`` and no test, so
    pytest reported "no tests collected" for the whole regression directory
    and the baseline drifted for as long as the model did. A change detector
    that nobody runs detects nothing, and by the time it was read by hand it
    disagreed with the model on nearly every metric, which says nothing about
    any single change.

    Regenerate explicitly after an intended change, with
    ``python -m tests.regression.collect_metrics --write
    tests/baselines/fx_regression_baseline.json``, so that the next
    unintended one still shows up.
    """
    failures = compare()
    assert not failures, (
        f"{len(failures)} metric(s) drifted from the baseline:\n  "
        + "\n  ".join(failures[:20])
    )


def main() -> None:
    failures = compare()
    if failures:
        print("Regression check FAILED:")
        for line in failures:
            print(f" - {line}")
        raise SystemExit(1)

    print("Regression check passed against baseline.")


if __name__ == "__main__":
    main()
