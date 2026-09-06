#!/usr/bin/env python3
"""Tests for the measurement machinery behind the provider economics.

    python3 tests/pnl_tooling_test.py

The mutation audit covers the liquidity rule thoroughly and does not touch
this file, so the statistics reported in the paper rested on code that nothing
checked. Four things had already gone wrong here before these tests existed.
The fee sweep moved one venue while the table called the column a pool fee. An
interval for a ratio was assembled from the endpoints of two separate
intervals. A cache keyed on the seed alone would hand back a figure produced by
a model that no longer existed. And a run reported the seeds it wanted rather
than the seeds it had.
"""
from __future__ import annotations

import functools
import os
import pytest
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.robustness import lp_pnl_corrected as T

W = 78
PASS, FAIL = "✓ PASS", "✗ FAIL"
total_pass = total_fail = 0
_failures = []


def _asserting(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        start = len(_failures)
        fn(*a, **k)
        new = _failures[start:]
        assert not new, (f"{len(new)} check(s) failed in {fn.__name__}: "
                         + "; ".join(new))
    return wrapper


def subsection(title):
    print(f"\n  ── {title} {'─' * max(0, W - len(title) - 6)}")


def check_bool(label, cond, detail=""):
    global total_pass, total_fail
    if cond:
        total_pass += 1
        print(f"    {PASS}  {label}")
    else:
        total_fail += 1
        _failures.append(label)
        print(f"    {FAIL}  {label}  {detail}")


def check_close(label, a, b, tol=1e-9):
    check_bool(label, abs(a - b) <= tol, f"got {a!r}, expected {b!r}")


@_asserting
def test_a_uniform_fee_reaches_both_venues():
    """A frontier over the fee has to move every venue it claims to."""
    subsection("the fee under test is applied to both pools")

    sim, _ = T.run(42, 30e-4, 'rule')
    fees = {n: p.fee for n, p in sim.amm_pools.items()}
    check_bool("every venue in the market carries the fee under test",
               all(abs(f - 30e-4) < 1e-12 for f in fees.values()), f"{fees}")

    # The calibrated market now runs one pool, so the original defect, a fee
    # reaching one venue and not the other, cannot show up there any more.
    # The regression is kept alive by putting the second pool back, which is
    # also the configuration the resource matched comparison uses.
    sim2, _ = T.run(42, 30e-4, 'rule', enable_cpmm=True)
    fees2 = {n: p.fee for n, p in sim2.amm_pools.items()}
    check_bool("there is more than one venue to get it wrong with",
               len(fees2) >= 2, f"{list(fees2)}")
    check_bool("both venues carry it when both are in the market",
               all(abs(f - 30e-4) < 1e-12 for f in fees2.values()), f"{fees2}")


@_asserting
def test_the_window_contains_the_step_it_names():
    """The tick the shock lands on has to be inside the crisis window.

    The window is declared as an offset pair against the shock, so ``(0, 100)``
    has to mean the hundred steps beginning at the shock and not the hundred
    beginning after it. The earlier form opened its capital at the end of the
    shock step and started accumulating on the step after, which put the
    repricing and the first arbitrage against the pool outside the measurement
    entirely. On the calibrated crisis that single step carried about four
    fifths of a basis point of the loss, and excluding it understated the
    crisis result by roughly a factor of two.

    The path below is synthetic and its answer is known by construction. One
    step loses exactly ten units of quote value and nothing else happens, so a
    window that names that step must report ten and a window that names the
    following one must report zero.
    """
    subsection("the crisis window contains the step it names")

    class Path:
        """A pool whose whole history is written by hand."""

        def __init__(self, xs, ys):
            self.x_history = list(xs)
            self.y_history = list(ys)
            n = len(xs) - 1
            self.fee_base_history = [0.0] * n
            self.fee_quote_history = [0.0] * n
            self.flow_dx_history = [0.0] * n
            self.flow_dy_history = [0.0] * n

    # Reserves are flat except for step 5, where ten units of quote leave the
    # pool without any capital flow, which is a pure loss of ten.
    xs = [100.0] * 11
    ys = [1000.0] * 6 + [990.0] * 5
    price = np.ones(11)
    pool = Path(xs, ys)

    # The value carried into step 5 is 100 + 1000 = 1100.
    r = T.components(pool, price, shock=5, window=(0, 3))
    check_bool("a window opening on the loss step is measurable", r is not None)
    if r is not None:
        lvr, fees, v0 = r
        check_close("it sees the whole loss", lvr, 10.0, tol=1e-9)
        check_close("and opens on the capital carried in", v0, 1100.0, tol=1e-9)

    r2 = T.components(pool, price, shock=6, window=(0, 3))
    check_bool("a window opening after it is measurable", r2 is not None)
    if r2 is not None:
        lvr2, _, v02 = r2
        check_close("it sees no loss", lvr2, 0.0, tol=1e-9)
        check_close("and opens on the reduced capital", v02, 1090.0, tol=1e-9)


@_asserting
def test_the_rule_table_agrees_with_the_code():
    """A table that documents the code has to be checked against the code.

    The manuscript prints the coefficients of the liquidity supply rule as a
    table. Nothing tied it to the module, so when the per period cap was tied
    to a day of position turnover the table went on printing the value it had
    before, which was larger by a factor of more than four thousand. The
    numbers in that table are as load bearing as the ones in a results table
    and they are checked here the same way.
    """
    subsection("the liquidity supply rule table matches the module")

    import re
    from AgentBasedModel.simulator.simulator import calibrated_default

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    article = os.path.join(root, 'EconMod', 'article', 'econmod.tex')
    if not os.path.exists(article):
        pytest.skip('manuscript not present beside the model')
    tex = open(article, encoding='utf-8').read()
    i = tex.find('\\label{tab:lprule}')
    body = tex[i:tex.find('\\end{tabular}', i)] if i >= 0 else ''
    check_bool("the table is present", bool(body))

    # The cap to check is the one the primary runtime carries, which comes from
    # the manifest and drives the endogenous population. This check used to read
    # the default on the reduced form AMMProvider, a class the primary model
    # does not instantiate, so it held the table to a number no run ever used.
    cap = calibrated_default('amm_lp_max_adj', 0.0023873085271651773)
    # Printed as a fraction and not a decimal, so the check looks for the
    # denominator the module actually carries.
    denom = round(1.0 / cap)
    check_bool("the cap in the table is the cap in the module",
               f'1/{denom}' in body.replace(' ', ''),
               f'module cap {cap!r}, denominator {denom}')

    # The curvature of the amplified curve is derived, not stored, so the
    # table has to carry the derivation and not a number that once matched it.
    check_bool("the amplified curvature is printed as a derivation",
               '1/8+A/4' in body.replace(' ', '').replace('$', ''),
               body[:200])
    a = float(re.search(r'A/4=([0-9.]+)', body.replace(' ', '')).group(1))
    from AgentBasedModel.simulator.simulator import calibrated_default
    want = 0.125 + 0.25 * float(calibrated_default('hfmm_A', 18.0))
    check_close("and its value matches the calibrated amplification", a, want,
                tol=1e-9)


@_asserting
def test_the_baseline_keeps_the_calibrated_fees():
    """The arm every other result is measured on must not be overwritten."""
    subsection("the baseline arm is left as calibrated")

    import json
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(root, 'calibration', 'primary_model.json'),
              encoding='utf-8') as fh:
        cal = json.load(fh)['cli_defaults']
    sim, _ = T.run(42, None, 'rule')
    check_close("the hybrid pool keeps its calibrated fee",
                sim.amm_pools['hfmm'].fee, cal['hfmm_fee'], tol=1e-12)
    check_bool("the calibrated market runs the hybrid pool alone",
               set(sim.amm_pools) == {'hfmm'}, f"{sorted(sim.amm_pools)}")

    # With both pools in the market the baseline still has to leave each on
    # its own calibrated fee, and the two differ, so conflating them shows.
    sim2, _ = T.run(42, None, 'rule', enable_cpmm=True)
    check_close("the constant product pool keeps its calibrated fee",
                sim2.amm_pools['cpmm'].fee, cal['cpmm_fee'], tol=1e-12)
    check_close("the hybrid pool keeps its calibrated fee alongside it",
                sim2.amm_pools['hfmm'].fee, cal['hfmm_fee'], tol=1e-12)
    check_bool("and the two differ, so conflating them would show",
               abs(cal['cpmm_fee'] - cal['hfmm_fee']) > 1e-12)


@_asserting
def test_the_bootstrap_keeps_seeds_paired():
    """A seed has to contribute its calm and its crisis together."""
    subsection("the break even interval is paired")

    # Two clusters of seeds. Within a cluster calm and crisis move together, so
    # an unpaired resample would mix a calm from one cluster with a crisis from
    # the other and report a spread the data does not contain.
    recs = ([{'total_calm': 0.002, 'total_crisis': -0.06} for _ in range(50)]
            + [{'total_calm': 0.004, 'total_crisis': -0.12} for _ in range(50)])
    pt, lo, hi, n = T.paired_bootstrap_breakeven(recs, n=4000, seed=1)
    check_bool("every seed was used", n == 100, f"{n}")
    # Both clusters give the same ratio, so a paired interval must be tight.
    check_close("the point estimate is the common ratio", pt,
                0.003 / (0.003 + 0.09), tol=1e-9)
    check_bool("and the paired interval is degenerate, as the data implies",
               hi - lo < 1e-6, f"[{lo}, {hi}]")

    # A genuinely dispersed sample must produce a genuinely wide interval, or
    # the test above would pass on a function that returns a constant.
    rng = np.random.default_rng(0)
    spread = [{'total_calm': float(c), 'total_crisis': float(d)}
              for c, d in zip(rng.uniform(0.001, 0.01, 60),
                              rng.uniform(-0.12, -0.02, 60))]
    _, lo2, hi2, _ = T.paired_bootstrap_breakeven(spread, n=4000, seed=2)
    check_bool("a dispersed sample gives a wide interval", hi2 - lo2 > 1e-3,
               f"[{lo2}, {hi2}]")


@_asserting
def test_the_break_even_handles_the_degenerate_states():
    """A facility that earns nothing in calm covers nothing."""
    subsection("the degenerate cases of the break even share")

    loss_always = [{'total_calm': -0.001, 'total_crisis': -0.05}] * 20
    pt, _, _, _ = T.paired_bootstrap_breakeven(loss_always, n=200)
    check_close("no calm earnings means no crisis state is covered", pt, 0.0)

    never_loses = [{'total_calm': 0.002, 'total_crisis': 0.001}] * 20
    pt2, _, _, _ = T.paired_bootstrap_breakeven(never_loses, n=200)
    check_close("no crisis loss means every window is covered", pt2, 1.0)

    zero_calm = [{'total_calm': 0.0, 'total_crisis': -0.05}] * 20
    pt3, _, _, _ = T.paired_bootstrap_breakeven(zero_calm, n=200)
    check_close("breaking even in calm covers nothing either", pt3, 0.0)

    missing = [{'total_calm': float('nan'), 'total_crisis': -0.05}] * 5
    pt4, _, _, n4 = T.paired_bootstrap_breakeven(missing, n=200)
    check_bool("a window that could not be measured is dropped, not counted",
               n4 == 0, f"{n4}")


@_asserting
def test_the_cache_refuses_results_from_another_model():
    """A seed number is not enough to identify a measurement."""
    subsection("the cache is keyed on the model, not only the seed")

    import json
    import tempfile
    real_dir = T.RAW_DIR
    with tempfile.TemporaryDirectory() as tmp:
        T.RAW_DIR = tmp
        try:
            T.append_raw(5.0, 'rule', {
                'seed': 1,
                'total_calm': 0.1,
                'total_calm_loss': 0.0,
                'total_calm_fees': 0.1,
                'total_crisis': -0.1,
                'total_crisis_loss': 0.1,
                'total_crisis_fees': 0.0,
            })
            have, stale = T.load_raw(5.0, 'rule')
            check_bool("a record written now is readable now",
                       set(have) == {1} and stale == 0, f"{have}, {stale}")

            # A record from a different model must not be handed back.
            with open(T.raw_path(5.0, 'rule'), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps({'seed': 2, 'total_calm': 0.1,
                                     'total_crisis': -0.1,
                                     'run_signature': 'not-this-model'}) + '\n')
            have2, stale2 = T.load_raw(5.0, 'rule')
            check_bool("a record from another model is ignored",
                       set(have2) == {1}, f"{have2}")
            check_bool("and the run is told how many were ignored", stale2 == 1,
                       f"{stale2}")

            # A record with no signature at all is from before this existed.
            with open(T.raw_path(5.0, 'rule'), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps({'seed': 3, 'total_calm': 0.1,
                                     'total_crisis': -0.1}) + '\n')
            have3, stale3 = T.load_raw(5.0, 'rule')
            check_bool("an unsigned record is ignored too", set(have3) == {1},
                       f"{have3}")
            check_bool("and counted", stale3 == 2, f"{stale3}")

            # A current signature in the wrong economic cell is still stale.
            wrong_cell = {
                'seed': 4, 'total_calm': 0.1, 'total_crisis': -0.1,
                'run_signature': T.run_signature(),
                'raw_schema_version': T._RAW_SCHEMA_VERSION,
                'fee_bps': 10.0, 'fee_mode': 'uniform', 'lp_model': 'rule',
            }
            with open(T.raw_path(5.0, 'rule'), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(wrong_cell) + '\n')
            have4, stale4 = T.load_raw(5.0, 'rule')
            check_bool("a current-signature record from another fee cell is ignored",
                       set(have4) == {1}, f"{have4}")
            check_bool("and counted as invalid", stale4 == 3, f"{stale4}")

            wrong_model = dict(wrong_cell, seed=5, fee_bps=5.0,
                               lp_model='endogenous')
            wrong_mode = dict(wrong_cell, seed=6, fee_bps=5.0,
                              fee_mode='calibrated')
            unsigned_schema = dict(wrong_cell, seed=7, fee_bps=5.0)
            unsigned_schema.pop('raw_schema_version')
            malformed = '{not json}\n'
            with open(T.raw_path(5.0, 'rule'), 'a', encoding='utf-8') as fh:
                for row in (wrong_model, wrong_mode, unsigned_schema):
                    fh.write(json.dumps(row) + '\n')
                fh.write(malformed)
            have5, stale5 = T.load_raw(5.0, 'rule')
            check_bool("wrong model, mode, schema and malformed rows are ignored",
                       set(have5) == {1}, f"{have5}")
            check_bool("every rejected row is counted", stale5 == 7, f"{stale5}")

            corrupt_result = dict(have5[1], seed=9, total_calm='not-a-number')
            with open(T.raw_path(5.0, 'rule'), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(corrupt_result) + '\n')
            have6, stale6 = T.load_raw(5.0, 'rule')
            check_bool("a wrong-typed result is rejected", set(have6) == {1},
                       f"{have6}")
            check_bool("the wrong-typed result is counted", stale6 == 8,
                       f"{stale6}")

            try:
                T.append_raw(5.0, 'rule', {
                    'seed': 8, 'fee_bps': 10.0,
                    'total_calm': 0.1, 'total_crisis': -0.1,
                })
                append_rejected = False
            except ValueError:
                append_rejected = True
            check_bool("append refuses a record labelled for another cell",
                       append_rejected)
        finally:
            T.RAW_DIR = real_dir

    sig = T.run_signature()
    check_bool("the signature is stable within a process",
               sig == T.run_signature(), sig)
    check_bool("and it is a digest, not a placeholder",
               isinstance(sig, str) and len(sig) == 16, f"{sig!r}")


@_asserting
def test_the_baseline_and_the_sweep_do_not_share_a_cache():
    """Two different configurations must not write to one file."""
    subsection("the baseline is stored apart from the sweep")

    check_bool("the baseline has its own file",
               T.raw_path(None, 'rule') != T.raw_path(5.0, 'rule'),
               T.raw_path(None, 'rule'))
    check_bool("and the two provider models do not share one either",
               T.raw_path(5.0, 'rule') != T.raw_path(5.0, 'endogenous'))


@_asserting
def test_a_report_counts_the_seeds_it_has():
    """A table has to say how many seeds are behind it."""
    subsection("the seed count is measured, not assumed")

    import tempfile
    real_dir = T.RAW_DIR
    with tempfile.TemporaryDirectory() as tmp:
        T.RAW_DIR = tmp
        try:
            for sd in range(7):
                T.append_raw(5.0, 'rule',
                             {'seed': sd, 'total_calm': 0.002,
                              'total_crisis': -0.05,
                              'total_calm_loss': 0.0, 'total_calm_fees': 0.002,
                              'total_crisis_loss': 0.05,
                              'total_crisis_fees': 0.0})
            # The same seed written twice must not be counted twice.
            T.append_raw(5.0, 'rule',
                         {'seed': 3, 'total_calm': 0.003, 'total_crisis': -0.05,
                          'total_calm_loss': 0.0, 'total_calm_fees': 0.003,
                          'total_crisis_loss': 0.05, 'total_crisis_fees': 0.0})
            have, duplicate_count = T.load_raw(5.0, 'rule')
            check_bool("a repeated seed invalidates that seed", len(have) == 6,
                       f"{len(have)}")
            check_bool("the duplicate is counted and neither row is selected",
                       duplicate_count == 1
                       and 3 not in have,
                       f"{duplicate_count}, {sorted(have)}")
            _, _, _, n = T.paired_bootstrap_breakeven(list(have.values()), n=200)
            check_bool("and the interval excludes the conflicted seed", n == 6,
                       f"{n}")
        finally:
            T.RAW_DIR = real_dir


@_asserting
def test_the_manuscript_matches_the_stored_runs():
    """A number in the paper has to be a number that was measured.

    The manuscript is not part of this repository, which carries the model
    alone, so this check is skipped where the article is absent. It still runs
    wherever the two are checked out beside each other, which is where a
    mismatch between a printed figure and a stored run can actually occur.
    """
    subsection("the manuscript is checked against the runs")

    import importlib.util
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    article = os.path.join(root, 'EconMod', 'article', 'econmod.tex')
    if not os.path.exists(article):
        pytest.skip('manuscript not present beside the model')
    spec = importlib.util.spec_from_file_location(
        'verify_article_claims', os.path.join(root, 'tools',
                                              'verify_article_claims.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = mod.claims()
    check_bool("there are stored measurements to check against", bool(rows),
               "run the sweep first")
    if not rows:
        return
    text = open(mod.ARTICLE, encoding='utf-8').read()
    missing = []
    for label, value, places, where in rows:
        shown = f'{value:.{places}f}'
        body = mod._table_body(text, where)
        # Every way the manuscript is allowed to write the number, the tool's
        # own list. A figure in the thousands is typeset with a separator, and
        # a check that knows only the bare digits fails on presentation and
        # not on content; this one carried its own shorter list and did.
        if not any(w in body for w in mod.spellings(value, places)):
            missing.append(f'{label}={shown}')
    check_bool("every checked figure appears in its own table", not missing,
               "; ".join(missing))
    # When current-signature provider reports exist, point estimates alone
    # would leave the interval endpoints unguarded. If the model changed, the
    # only acceptable alternative is to withdraw every old P&L number until a
    # new run exists; stale reports must never be treated as current claims.
    sigs = mod.signatures()
    pnl_current = (sigs['baseline'] == sigs['current'] == sigs['sweep'])
    # What a reader of the typeset paper meets. Read against the source, this
    # check passed on a marker sitting in a LaTeX comment above a table that
    # went on presenting the figures as live results.
    shown_text = mod.visible_text(text)
    provenance = mod.pair_provenance()
    if pnl_current:
        check_bool("the check covers the intervals and not only the points",
                   len(rows) >= 30, f"{len(rows)}")
    elif provenance['reports_are_from_another_pair']:
        # A branch calibrated to one pair cannot reproduce another pair's
        # results and is not meant to. That is not staleness, and withdrawing
        # the primary pair's results from a paper that reports both pairs
        # would remove the comparison the paper is making. What it does owe
        # the reader is the label, in the text and not in a comment.
        check_bool("results from the other pair's branch are labelled in the text",
                   mod.PAIR_PROVENANCE_MARKER in shown_text,
                   "the manuscript presents the primary pair's stored runs "
                   "without saying in the text that they come from that branch")
    else:
        check_bool("stale provider results are explicitly withdrawn",
                   mod.WITHDRAWN_MARKER in shown_text,
                   "the manuscript still presents stale P&L")
    counts = mod.seed_counts()
    check_bool("every current stored report carries three hundred seeds",
               all(v == 300 for v in counts.values()), f"{counts}")
    check_bool("current reports are used, or labelled, or withdrawn",
               pnl_current
               or (provenance['reports_are_from_another_pair']
                   and mod.PAIR_PROVENANCE_MARKER in shown_text)
               or mod.WITHDRAWN_MARKER in shown_text,
               f"{sigs} | {provenance}")
    build = mod.build_state()
    check_bool("the typeset manuscript exists and is newer than its source",
               build['pdf'] and build['fresh'], f"{build}")
    check_bool("the bibliography has no comment inside an entry",
               build['comments_inside_entries'] == 0,
               f"{build['comments_inside_entries']}")


@_asserting
def test_a_stale_baseline_cache_is_reported_and_not_a_crash():
    """The path that says the cache went stale must survive saying it."""
    subsection("a stale baseline cache is handled")

    import json
    import tempfile
    real = T.RAW_DIR
    with tempfile.TemporaryDirectory() as tmp:
        T.RAW_DIR = tmp
        try:
            with open(T.raw_path(None, 'rule'), 'w', encoding='utf-8') as fh:
                fh.write(json.dumps({'seed': 1, 'total_calm': 0.1,
                                     'total_crisis': -0.1,
                                     'run_signature': 'other'}) + '\n')
            have, stale = T.load_raw(None, 'rule')
            check_bool("the stale record is not returned", not have, f"{have}")
            check_bool("and it is counted", stale == 1, f"{stale}")
            # Naming a fee setting must work when there is no number to name.
            check_bool("the baseline has a name", T.fee_label(None) == 'baseline',
                       T.fee_label(None))
            check_bool("and a fee has one too", T.fee_label(5.0) == '5 bps',
                       T.fee_label(5.0))
            T.report([None], 'rule')
            check_bool("reporting a stale baseline does not raise", True)
        except TypeError as exc:
            check_bool("reporting a stale baseline does not raise", False, str(exc))
        finally:
            T.RAW_DIR = real


@_asserting
def test_the_interval_does_not_depend_on_the_order_of_records():
    """An interval is a property of the measurements, not of the file."""
    subsection("the bootstrap is order independent")

    import random as _r
    recs = [{'seed': i, 'total_calm': 0.001 + 0.0001 * (i % 7),
             'total_crisis': -0.05 - 0.001 * (i % 5)} for i in range(60)]
    a = T.paired_bootstrap_breakeven(recs)
    shuffled = list(recs)
    _r.Random(3).shuffle(shuffled)
    b = T.paired_bootstrap_breakeven(shuffled)
    check_bool("shuffling the records leaves the interval alone", a == b,
               f"{a} against {b}")


@_asserting
def test_the_signature_tracks_the_model_and_ignores_the_prose():
    """A caption must not throw away a thousand measurements."""
    subsection("the signature covers what a measurement depends on")

    files = T._signature_files()
    check_bool("the whole model package is covered",
               sum(1 for f in files if f.startswith('AgentBasedModel')) > 10,
               f"{len(files)}")
    for needed in ('main.py', os.path.join('calibration', 'primary_model.json')):
        check_bool(f"{needed} is covered", needed in files, f"{files[:4]}")
    for needed in (os.path.join('AgentBasedModel', 'utils', 'orders.py'),
                   os.path.join('AgentBasedModel', 'venues', 'amm.py')):
        check_bool(f"{needed} is covered", needed in files, "not in the digest")
    check_bool("the measuring functions are named",
               set(T._MEASURING) >= {'run', 'components', 'measure'},
               f"{T._MEASURING}")
    a = {'cli_defaults': {'n_mm': 5},
         'calibration_notes': {'book': 'first wording'}}
    b = {'cli_defaults': {'n_mm': 5},
         'calibration_notes': {'book': 'revised wording'}}
    c = {'cli_defaults': {'n_mm': 6},
         'calibration_notes': {'book': 'revised wording'}}
    check_bool("calibration prose is excluded from the signature",
               T._calibration_semantics(a) == T._calibration_semantics(b),
               f"{T._calibration_semantics(a)}")
    check_bool("runtime calibration remains in the signature",
               T._calibration_semantics(a) != T._calibration_semantics(c),
               f"{T._calibration_semantics(c)}")


@_asserting
def test_survival_cache_keys_every_economic_run_setting():
    subsection("LP survival cache includes the whole run specification")

    from tools.robustness import lp_survival as S

    base = {
        'outside_option': 1.3319e-9, 'subsidy_rate': 0.0,
        'loss_rebate_fraction': 0.0,
        'exit_patience': 25, 'entry_patience': 40,
        'kappa': 0.35, 'response_scale': 1e-6,
        'max_adj': 0.05, 'ewma_alpha': 0.10,
        'entry_margin': 0.25, 'n_iter': 1000,
    }
    alternatives = {
        'outside_option': 2e-9, 'subsidy_rate': 1e-5,
        'loss_rebate_fraction': 1.0,
        'exit_patience': 26, 'entry_patience': 41,
        'kappa': 0.20, 'response_scale': 2e-6,
        'max_adj': 0.02, 'ewma_alpha': 0.05,
        'entry_margin': 0.50, 'n_iter': 900,
    }
    path = S._raw_path(base)
    for key, value in alternatives.items():
        changed = {**base, key: value}
        check_bool(f"changing {key} changes the cache key",
                   S._raw_path(changed) != path,
                   f"{S._raw_path(changed)}")


@_asserting
def test_welfare_accounting_does_not_count_transfers_as_social_gains():
    subsection("fees and subsidies are incidence, not free welfare")

    from tools.robustness.welfare_accounting import ArmWindow, account_pair

    off = ArmWindow(executed_notional=1000.0, taker_execution_cost=9.0)
    on = ArmWindow(executed_notional=1000.0, taker_execution_cost=4.0,
                   lp_opening_capital=2000.0, lp_lvr=7.0, lp_fees=5.0,
                   sponsor_transfer=3.0, sponsor_rate_transfer=1.0,
                   sponsor_loss_rebate=2.0)
    base = account_pair(on, off, annual_capital_rate=0.0)
    rich_subsidy = account_pair(
        ArmWindow(**{**on.__dict__, 'sponsor_transfer': 300.0}),
        off, annual_capital_rate=0.0)

    check_close("matched-notional user benefit is the cost reduction",
                base['matched_notional_user_benefit'], 5.0, tol=1e-12)
    check_close("provider operating result is fees less LVR",
                base['lp_operating_result'], -2.0, tol=1e-12)
    check_bool("the subsidy receipt cancels the fiscal debit",
               base['subsidy_transfer_cancels'])
    check_close("the unconditional rate transfer is disclosed separately",
                base['rate_transfer_paid'], 1.0, tol=1e-12)
    check_close("the state-contingent loss rebate is disclosed separately",
                base['loss_rebate_paid'], 2.0, tol=1e-12)
    check_close("a larger subsidy does not manufacture user-side welfare",
                rich_subsidy['user_side_net_benefit'],
                base['user_side_net_benefit'], tol=1e-12)
    check_bool("the module refuses to label the partial ledger total welfare",
               base['total_welfare_identified'] is False)


@_asserting
def test_welfare_comparison_uses_common_notional():
    subsection("different realised volume is not silently priced as welfare")

    from tools.robustness.welfare_accounting import ArmWindow, account_pair

    # Both arms have their costs stated on their own realised volumes.  Only
    # the 800 of notional common to the two is valued; the extra 400 in the AMM
    # arm remains an unpriced quantity because no demand utility is modelled.
    on = ArmWindow(1200.0, 6.0)
    off = ArmWindow(800.0, 8.0)
    row = account_pair(on, off, annual_capital_rate=0.0)
    check_close("common notional is the smaller realised exposure",
                row['common_executed_notional'], 800.0, tol=1e-12)
    check_close("the cost-rate improvement is applied only to that exposure",
                row['matched_notional_user_benefit'], 4.0, tol=1e-12)
    check_close("the remaining volume difference is exposed separately",
                row['incremental_executed_notional'], 400.0, tol=1e-12)


@_asserting
def test_welfare_comparison_matches_inside_fixed_size_buckets():
    subsection("trade-size composition is not mistaken for welfare")

    from tools.robustness.welfare_accounting import ArmWindow, account_pair

    # Aggregate rates would compare 10/1000 with 20/1000 and report a gain.
    # Within each fixed size bucket both arms have exactly the same rate, so
    # the composition-controlled benefit must be zero.
    on = ArmWindow(
        1000.0, 10.0,
        execution_buckets={
            'q_le_5': {'notional': 900.0, 'cost': 9.0},
            'q_gt_20': {'notional': 100.0, 'cost': 1.0},
        })
    off = ArmWindow(
        1000.0, 20.0,
        execution_buckets={
            'q_le_5': {'notional': 100.0, 'cost': 1.0},
            'q_gt_20': {'notional': 900.0, 'cost': 9.0},
        })
    row = account_pair(on, off, annual_capital_rate=0.0)
    check_close("bucket-matched benefit is zero", row['matched_notional_user_benefit'],
                0.0, tol=1e-12)
    check_close("only common volume within buckets is priced",
                row['common_executed_notional'], 200.0, tol=1e-12)
    check_bool("the report states the matching method",
               row['user_benefit_method'] == 'fixed_absolute_size_buckets')


@_asserting
def test_welfare_bootstrap_is_seed_order_invariant():
    subsection("welfare intervals retain paired seed observations")

    from tools.robustness.welfare_accounting import aggregate

    rows = [
        {'seed': seed, 'accounting': {
            'matched_notional_user_benefit': value,
            'matched_notional_user_benefit_bps': value,
        }}
        for seed, value in enumerate((-2.0, -1.0, 1.0, 5.0), start=42)
    ]
    a = aggregate(rows, bootstrap_draws=2000, bootstrap_seed=7)
    # Sorting by seed inside the metric is unnecessary: resampling a vector is
    # exchangeable. The fixed RNG must nevertheless give the same interval
    # after a pure reordering of the paired rows.
    b = aggregate(list(reversed(rows)), bootstrap_draws=2000, bootstrap_seed=7)
    ia = a['bootstrap_95_intervals_on_median']['matched_notional_user_benefit']
    ib = b['bootstrap_95_intervals_on_median']['matched_notional_user_benefit']
    check_bool("row order leaves the interval unchanged", ia == ib,
               f"{ia} against {ib}")


def _run(fn):
    try:
        fn()
    except AssertionError:
        pass


def main():
    print(f"\n{'=' * W}\n  PROVIDER ECONOMICS TOOLING — Unit Tests\n{'=' * W}")
    for f in (test_a_uniform_fee_reaches_both_venues,
              test_the_window_contains_the_step_it_names,
              test_the_rule_table_agrees_with_the_code,
              test_the_baseline_keeps_the_calibrated_fees,
              test_the_bootstrap_keeps_seeds_paired,
              test_the_break_even_handles_the_degenerate_states,
              test_the_cache_refuses_results_from_another_model,
              test_the_baseline_and_the_sweep_do_not_share_a_cache,
              test_a_report_counts_the_seeds_it_has,
              test_the_manuscript_matches_the_stored_runs,
              test_a_stale_baseline_cache_is_reported_and_not_a_crash,
              test_the_interval_does_not_depend_on_the_order_of_records,
              test_the_signature_tracks_the_model_and_ignores_the_prose,
              test_survival_cache_keys_every_economic_run_setting,
              test_welfare_accounting_does_not_count_transfers_as_social_gains,
              test_welfare_comparison_uses_common_notional,
              test_welfare_comparison_matches_inside_fixed_size_buckets,
              test_welfare_bootstrap_is_seed_order_invariant):
        _run(f)
    print(f"\n{'=' * W}\n  passed {total_pass}, failed {total_fail}\n{'=' * W}")
    return 1 if total_fail else 0


if __name__ == '__main__':
    sys.exit(main())
