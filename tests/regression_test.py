"""Collect the regression baseline check into the suite.

The check itself lives in ``tests/regression/compare_baseline.py``, which also
runs as a script. That module is named for what it does and not for the
collector, so pytest never picked it up: the whole ``tests/regression``
directory reported "no tests collected" and the baseline drifted for as long
as the model did. By the time anyone read it by hand it disagreed with the
model on nearly every metric, which is the one state in which a change
detector says nothing at all.

This module exists so the check is collected under the same ``*_test.py``
convention as the rest of the suite.
"""
from __future__ import annotations

from tests.regression.compare_baseline import compare


def test_the_model_matches_its_regression_baseline():
    """Catch a change to the simulated trajectory that nobody meant to make.

    This is not an acceptance target and carries no economic claim. It is a
    fingerprint of one seeded run, and its only job is to make an unintended
    change visible on the commit that causes it.

    After an intended change, regenerate it explicitly:

        python -m tests.regression.collect_metrics \\
            --write tests/baselines/fx_regression_baseline.json
    """
    failures = compare()
    assert not failures, (
        f'{len(failures)} metric(s) drifted from the regression baseline. '
        'If the change was intended, regenerate the baseline; if not, this is '
        'the commit that changed the trajectory.\n  '
        + '\n  '.join(failures[:20])
    )
