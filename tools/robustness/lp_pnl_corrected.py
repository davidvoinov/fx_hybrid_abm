#!/usr/bin/env python3
"""Liquidity provider profit and loss, corrected.

The published figure came from ``n1b_lp_pnl_300.py``, which compared terminal
reserves against a static endpoint benchmark holding the reserves observed at
the start of the window. Four things were wrong with that.

  * The provider adds and removes liquidity every period, so the terminal
    reserves contain capital flows that the static benchmark does not. Taking
    ten per cent of the liquidity out at an unchanged price read as roughly
    minus ten per cent of profit and loss even though nothing had been earned
    or lost. Here every flow is recorded by the pool and netted out, so the
    figure measures investment result alone.
  * The reserve histories carry one more entry than the price series, because
    they open with the state before the first step. Pairing ``x_history[t]``
    with ``fair_price[t]`` therefore compared reserves with the price of the
    previous period. The correct pairing is ``x_history[t + 1]``.
  * The caption described a hold and rebalance benchmark while the code held a
    fixed basket to the endpoint. The loss is measured here period by period
    against the reserves carried into that period, which is the realised
    counterpart of loss versus rebalancing.
  * Only the hybrid function pool was measured, although the treatment arm
    contains a constant product pool as well. Both are reported, and so is the
    combined result of holding the two, which is what a provider present in the
    whole treatment arm actually earns.
  * Capital flows were first recorded as a single value struck at the pool mid,
    while every other term here is marked at the reference price. A pool sitting
    off peg made the two disagree, and the gap read as a gain or a loss that
    nobody had made. On a synthetic path with no investment result at all, a
    pool quoting a mid of one against a reference price of two reported a loss
    of 8.71 per cent. The pool now records the base and quote amounts that
    moved, and both are marked here at the reference price.

    python3 tools/robustness/lp_pnl_corrected.py --seeds 300
    python3 tools/robustness/lp_pnl_corrected.py --self-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from tools.robustness.signatures import (
    calibration_semantics,
    measurement_signature,
    model_signature,
    model_signature_files,
)

PRESET = 'dealer_liquidity_crisis'
N_ITER = 1000
CALM = (-160, -60)          # window offsets relative to the shock
CRISIS = (0, 100)


RAW_DIR = os.path.join(ROOT, 'output', 'resilience', 'raw')

# Files whose contents decide what a measured seed means. A cached result is
# only reusable if every one of them is byte for byte what it was when the
# result was produced. Without this the cache is a trap. Changing the loss
# curvature, the windows, the preset or the profit and loss formula leaves the
# seed numbers untouched, so a later run sees a seed it already has and
# silently reports a figure from a model that no longer exists.
# Naming the files by hand is how the list came to be short of the mark. The
# preset that selects the scenario, the order book, the order objects and the
# venue wrappers all decide what a seed produces, and none of them was listed.
# The signature therefore walks the model package and adds the entry point and
# the calibration alongside it, so a file has to be deleted from the project to
# fall out of the digest.
_SIGNATURE_ROOTS = (
    os.path.join('AgentBasedModel'),
)
_SIGNATURE_EXTRA = (
    'main.py',
    os.path.join('calibration', 'primary_model.json'),
)
# Only the functions that decide what a stored record contains. Hashing this
# whole file instead would throw away every measurement whenever a caption or a
# format string changed, which is a good way to make people stop trusting the
# check and reach for a flag that turns it off.
_MEASURING = ('run', '_series', 'components', 'measure')
_RAW_SCHEMA_VERSION = 1


def _signature_files():
    return model_signature_files(ROOT)
_SIG = None


def _calibration_semantics(data):
    """Only runtime calibration values belong in a simulation signature.

    Source lists and calibration notes are provenance, not model inputs.  A
    copy edit there must not invalidate hundreds of otherwise identical
    seeded simulations; changing any CLI default still must.
    """
    return calibration_semantics(data)


def run_signature():
    """Digest of the simulation model and the P&L measurement definition."""
    global _SIG
    if _SIG is None:
        _SIG = measurement_signature(
            'lp_pnl_corrected',
            model_signature(ROOT),
            functions=tuple(globals().get(name) for name in _MEASURING
                            if globals().get(name) is not None),
            constants=(PRESET, N_ITER, CALM, CRISIS, _RAW_SCHEMA_VERSION),
        )
    return _SIG


def fee_label(bps):
    """How a fee setting is named in messages and file names."""
    return 'baseline' if bps is None else f'{bps:g} bps'


def raw_path(bps, lp_model):
    signature = run_signature()
    if bps is None:
        name = f'lp_pnl_{lp_model}_baseline_{signature}.jsonl'
    else:
        name = f'lp_pnl_{lp_model}_fee{bps:g}_{signature}.jsonl'
    return os.path.join(RAW_DIR, name)


def _raw_cell(bps, lp_model):
    """Exact cache identity for one fee/provider-model cell."""
    if lp_model not in {'rule', 'endogenous'}:
        raise ValueError(f'unknown LP model {lp_model!r}')
    return {
        'raw_schema_version': _RAW_SCHEMA_VERSION,
        'fee_bps': None if bps is None else float(bps),
        'fee_mode': 'calibrated' if bps is None else 'uniform',
        'lp_model': lp_model,
    }


def _raw_record_matches_cell(rec, expected):
    if not isinstance(rec, dict):
        return False
    if rec.get('raw_schema_version') != expected['raw_schema_version']:
        return False
    if rec.get('fee_mode') != expected['fee_mode']:
        return False
    if rec.get('lp_model') != expected['lp_model']:
        return False
    observed, wanted = rec.get('fee_bps'), expected['fee_bps']
    if wanted is None:
        if observed is not None:
            return False
    else:
        try:
            if not np.isfinite(float(observed)) or float(observed) != wanted:
                return False
        except (TypeError, ValueError):
            return False
    seed = rec.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        return False
    if seed < 0:
        return False
    for phase in ('calm', 'crisis'):
        keys = (
            f'total_{phase}',
            f'total_{phase}_loss',
            f'total_{phase}_fees',
        )
        if not all(key in rec for key in keys):
            return False
        values = [rec[key] for key in keys]
        if all(value is None for value in values):
            continue
        if any(value is None or isinstance(value, bool) for value in values):
            return False
        try:
            net, loss, fees = (float(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            return False
        if (not all(np.isfinite(value) for value in (net, loss, fees))
                or not np.isclose(net, fees - loss,
                                  rtol=1e-12, atol=1e-10)):
            return False
    return True


def load_raw(bps, lp_model, strict=True):
    """Seeds measured under the current model, keyed by seed.

    Records carrying a different signature are ignored rather than trusted,
    and the count of what was skipped is returned so a run can say out loud
    that its cache went stale.
    """
    path = raw_path(bps, lp_model)
    out, invalid = {}, 0
    conflicted_seeds = set()
    if not os.path.exists(path):
        return (out, invalid) if strict else out
    want = run_signature()
    expected = _raw_cell(bps, lp_model)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                invalid += 1
                continue
            if (rec.get('run_signature') != want
                    or not _raw_record_matches_cell(rec, expected)):
                invalid += 1
                continue
            seed = int(rec['seed'])
            if seed in out or seed in conflicted_seeds:
                # Parallel/restarted jobs should never produce two valid rows
                # for one exact cell.  Selecting either observation would make
                # publication evidence depend on append order, so the seed is
                # unusable until the cache is repaired.
                invalid += 1
                out.pop(seed, None)
                conflicted_seeds.add(seed)
                continue
            out[seed] = rec
    return (out, invalid) if strict else out


def append_raw(bps, lp_model, rec):
    rec = dict(rec)
    expected = _raw_cell(bps, lp_model)
    for key, value in expected.items():
        if key in rec and rec[key] != value:
            raise ValueError(
                f'raw record {key}={rec[key]!r} does not match cell {value!r}'
            )
    rec.update(expected)
    rec['run_signature'] = run_signature()
    if not _raw_record_matches_cell(rec, expected):
        raise ValueError('raw record has no valid nonnegative integer seed')
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(raw_path(bps, lp_model), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')


def paired_bootstrap_breakeven(recs, n=20000, seed=0):
    """Operating break even crisis state share, with a paired interval.

    The share is the fraction of windows that may be in the crisis state
    before the calm earnings stop covering the crisis losses,

        s* = calm / (calm + |crisis|),

    with both figures as medians across seeds. It is a share of windows and
    not a count of episodes in a year, and turning one into the other needs a
    window length and a year length that this model does not yet fix. It is
    also before the opportunity cost of the capital, which the welfare
    accounting carries separately, so it is a break even for the operating
    result alone.

    The interval resamples whole seeds, so a seed contributes its calm and its
    crisis observation together. Combining the endpoints of two separately
    computed intervals, which is what an earlier version reported, is not an
    interval for the ratio at all.
    """
    # Sorted by seed, so the draw does not depend on the order records happen
    # to sit in the file. Parallel workers append as they finish, and a file
    # rewritten in a different order gave different interval endpoints for the
    # same measurements, which is not a property an interval may have.
    recs = sorted(recs, key=lambda r: r.get('seed', 0))
    calm = np.array([r['total_calm'] for r in recs], dtype=float)
    crisis = np.array([r['total_crisis'] for r in recs], dtype=float)
    ok = np.isfinite(calm) & np.isfinite(crisis)
    calm, crisis = calm[ok], crisis[ok]
    if calm.size == 0:
        return float('nan'), float('nan'), float('nan'), 0

    def share(c, d):
        g = np.median(c)
        loss = -np.median(d)
        if not np.isfinite(g) or not np.isfinite(loss):
            return float('nan')
        if g <= 0:
            return 0.0                      # no calm earnings, nothing is covered
        if loss <= 0:
            return 1.0                      # no crisis loss, every window is fine
        return g / (g + loss)

    point = share(calm, crisis)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, calm.size, size=(n, calm.size))
    draws = np.array([share(calm[i], crisis[i]) for i in idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float('nan'), float('nan'), calm.size
    return (point, float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), int(calm.size))


def boot_ci(v, n=5000, seed=0):
    """Median and a bootstrap 95 per cent interval around it.

    The caller passes the values already ordered by seed, so the interval is a
    property of the measurements and not of the order a parallel run happened
    to write them in.
    """
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    bs = np.median(v[rng.integers(0, v.size, size=(n, v.size))], axis=1)
    return (float(np.median(v)), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def run(seed: int, fee, lp_model: str = 'rule', enable_cpmm=None):
    """One path, either at the calibrated fees or at a common fee on both.

    Passing ``None`` leaves both venues on the fees the model is calibrated
    with, which is twenty basis points on the constant product pool and five
    on the hybrid one. That configuration is the treatment arm every other
    result in the paper is measured on, so the provider economics of the
    facility as it is actually specified has to be read there.

    Passing a number puts that fee on both venues. That is a policy
    counterfactual over the fee and not the baseline, and an earlier version
    conflated the two by moving the hybrid fee alone while calling the column
    a pool fee.
    """
    argv = ['--preset', PRESET, '--seed', str(seed),
            '--n-iter', str(N_ITER), '--silent',
            '--amm-lp-model', lp_model]
    # The constant product pool is out of the market by calibration. It can be
    # put back for the resource matched comparison and for the regression that
    # catches a fee reaching one venue and not the other.
    if enable_cpmm is not None:
        argv += ['--enable-cpmm', '1' if enable_cpmm else '0']
    p = build_parser(); a = p.parse_args(argv); _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv)
    _auto_stress_around_shock(a)
    a.enable_amm = 1
    a.clob_amm_interaction = 'competition'
    if fee is not None:
        # Every automated venue in the market carries the fee under test. An
        # earlier version moved the hybrid fee alone, so a row labelled five
        # basis points was really twenty on one pool and five on the other.
        a.hfmm_fee = fee
        a.cpmm_fee = fee
    _seed_all(seed)
    sim = build_sim(a)
    sim.simulate(a.n_iter, silent=True)
    return sim, int(a.shock_iter)


def _series(pool, price):
    """Aligned reserve, fee and flow series, trimmed to a common length.

    ``price[i]`` is the reference price of step ``i`` and pairs with
    ``x_history[i + 1]``, the reserves left at the end of that step, because
    the reserve histories open with the state before the first step. The flow
    and fee records follow the same convention, so ``flow_dx_history[i]`` is
    the base that entered or left between ``x_history[i]`` and
    ``x_history[i + 1]``, and ``fee_base_history[i]`` is the base charged in
    fees over the same step.
    """
    x = np.asarray(pool.x_history, dtype=float)
    y = np.asarray(pool.y_history, dtype=float)
    # Fees are read as the native amounts of the period and valued here at the
    # same reference price as everything else. The cumulative counter on the
    # pool is struck at the price each trade executed at, so reading its
    # differences put one price into the fee term and another into the
    # reserves, and the figure disagreed with what the providers were paid.
    fb = np.asarray(getattr(pool, 'fee_base_history', []), dtype=float)
    fq = np.asarray(getattr(pool, 'fee_quote_history', []), dtype=float)
    dx = np.asarray(getattr(pool, 'flow_dx_history', []), dtype=float)
    dy = np.asarray(getattr(pool, 'flow_dy_history', []), dtype=float)
    n = min(len(x) - 1, len(y) - 1, len(fb), len(fq), len(dx), len(dy), len(price))
    return x, y, fb, fq, dx, dy, n


def components(pool, price, shock, window):
    """Loss, fees and net over a window, in units of the quote currency."""
    x, y, fb, fq, dx, dy, n = _series(pool, price)
    t0, t1 = shock + window[0], shock + window[1]
    if t0 < 0 or t1 >= n:
        return None

    # The window names the steps it covers, so ``(0, 100)`` is the hundred
    # steps beginning at the shock. An earlier form opened the capital at the
    # end of the shock step and began accumulating on the step after it, which
    # left the repricing and the first arbitrage against the pool outside the
    # measurement. On the calibrated crisis that one step carried close to half
    # the loss, and the reported crisis result was understated by about a
    # factor of two.
    #
    # ``x[i]`` is the reserve carried into step ``i`` and ``price[i]`` is that
    # step's reference price, so the opening value uses the same pair the loop
    # uses on its first iteration.
    v0 = x[t0] * price[t0] + y[t0]
    # A venue that has already wound down holds nothing at the start of the
    # window, so there is no capital whose return could be measured. Reporting
    # a percentage of nothing produced a silent NaN that then propagated into
    # the medians. The window is declared unmeasurable instead and counted.
    if v0 <= 1e-9:
        return None
    lvr = fees = 0.0
    for i in range(t0, t1):
        p_now = price[i]
        hold = x[i] * p_now + y[i]              # reserves carried in, marked now
        pool_v = x[i + 1] * p_now + y[i + 1]    # reserves carried out
        # Capital committed during the period inflates the pool without being
        # a gain, and capital withdrawn deflates it without being a loss. Both
        # are marked at the same reference price as the reserves they moved.
        flow = dx[i] * p_now + dy[i]
        lvr += hold - pool_v + flow
        fees += fb[i] * p_now + fq[i]
    return lvr, fees, v0


def pnl(pool, price, shock, window):
    """Net provider result over a window, as a percentage of opening value."""
    r = components(pool, price, shock, window)
    if r is None:
        return None
    lvr, fees, v0 = r
    net = fees - lvr
    return 100.0 * lvr / v0, 100.0 * fees / v0, 100.0 * net / v0


def self_check(verbose=True):
    """The identity the whole measurement rests on, checked on a real pool.

    A pool that only receives and returns capital has made nothing and lost
    nothing, whatever the reference price is doing, so the measured result must
    be zero. Marking the flows at the pool mid instead of the reference price
    fails this at any price away from the mid, which is the defect this file
    was rewritten to remove.
    """
    from AgentBasedModel.venues.amm import CPMMPool, HFMMPool

    ok = True
    for make in (lambda: HFMMPool(x=1000.0, y=1000.0, A=18.0, fee=5e-4),
                 lambda: CPMMPool(x=1000.0, y=1000.0, fee=5e-4)):
        for p_ref in (1.0, 2.0, 0.5):
            pool = make()
            pool.record_state()
            for t in range(24):
                pool.remove_liquidity(0.02) if t % 2 else pool.add_liquidity(0.03)
                pool.record_state()
            price = np.full(len(pool.x_history), p_ref)
            r = pnl(pool, price, shock=2, window=(0, 15))
            name = type(pool).__name__
            bad = r is None or abs(r[2]) > 1e-6
            ok = ok and not bad
            got = 'no window' if r is None else f'{r[2]:+.6f}%'
            if verbose:
                print(f"    {name:9s} reference price {p_ref:<4g} net {got:>12s}  "
                      f"{'FAIL' if bad else 'ok'}")
    return ok


def _measure_one(job):
    seed, bps, lp_model = job
    return measure(seed, bps, lp_model)


def measure(seed, bps, lp_model):
    """One seed at one fee setting, reduced to the numbers the table needs."""
    sim, sh = run(seed, None if bps is None else bps / 1e4, lp_model)
    price = np.asarray(sim.logger.fair_price_series, dtype=float)
    rec = {'seed': int(seed),
           'fee_bps': None if bps is None else float(bps),
           'fee_mode': 'calibrated' if bps is None else 'uniform',
           'lp_model': lp_model}
    for label, win in (('calm', CALM), ('crisis', CRISIS)):
        agg = [0.0, 0.0, 0.0]
        complete = True
        for name, pool in sim.amm_pools.items():
            c = components(pool, price, sh, win)
            if c is None:
                complete = False
                rec[f'{name}_{label}'] = None
                continue
            lvr, fees, v0 = c
            rec[f'{name}_{label}'] = 100.0 * (fees - lvr) / v0
            rec[f'{name}_{label}_loss'] = 100.0 * lvr / v0
            rec[f'{name}_{label}_fees'] = 100.0 * fees / v0
            agg[0] += lvr; agg[1] += fees; agg[2] += v0
        # Summing before dividing weights each pool by the capital in it.
        rec[f'total_{label}'] = (100.0 * (agg[1] - agg[0]) / agg[2]
                                 if complete and agg[2] > 0 else None)
        rec[f'total_{label}_loss'] = (100.0 * agg[0] / agg[2]
                                      if complete and agg[2] > 0 else None)
        rec[f'total_{label}_fees'] = (100.0 * agg[1] / agg[2]
                                      if complete and agg[2] > 0 else None)
    return rec


def report(fees, lp_model, out=None, seeds=None):
    lines = [f'Model signature {model_signature(ROOT)}.',
             f'P&L measurement signature {run_signature()}.',
             'Realised provider result, per cent of the value committed at the '
             'start of the window, capital flows netted out at the reference '
             'price.',
             'Every automated venue in the market carries the fee under '
             'test. The portfolio row is the venues held together, '
             'weighted by the capital in each, and it coincides with the '
             'single pool row while the market runs one facility.',
             'Intervals are a bootstrap 95 per cent interval on the median '
             'across seeds.', '']
    for bps in fees:
        raw, stale = load_raw(bps, lp_model)
        # A report has to describe the seeds it was asked about. Aggregating
        # whatever happens to be cached made the count depend on the history of
        # the directory instead of on the request.
        recs = [r for k, r in sorted(raw.items())
                if seeds is None or k in seeds]
        outside = len(raw) - len(recs)
        if outside:
            lines.append(f'  {fee_label(bps)}, {outside} cached seeds fall '
                         f'outside the range asked for and are not used')
        if stale:
            lines.append(f'  {fee_label(bps)}, {stale} cached seeds were '
                         f'produced by a different model and are ignored')
        if not recs:
            lines.append(f'  {fee_label(bps)}, provider model {lp_model}, '
                         f'nothing measured yet')
            continue
        label_fee = ('the calibrated fee, 5 bps on the hybrid pool, which is '
                     'the facility the calibrated market runs' if bps is None
                     else f'{fee_label(bps)} on every venue in the market')
        lines.append(f'  {label_fee}, provider model {lp_model}, '
                     f'{len(recs)} seeds')
        lines.append(f"    {'pool':9s} {'window':7s} {'loss %':>9s} "
                     f"{'fees %':>9s} {'net %':>9s} {'95% interval on net':>22s}"
                     f"  seeds")
        order = {'total': 2}
        keys = sorted({k.rsplit('_', 1)[0] for r in recs for k in r
                       if k.endswith('_calm') or k.endswith('_crisis')})
        for label in ('calm', 'crisis'):
            for name in sorted({k for k in keys},
                               key=lambda n: (order.get(n, 0), n)):
                vals = [r.get(f'{name}_{label}') for r in recs]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                if not vals:
                    lines.append(f'    {name:9s} {label:7s} no capital at the '
                                 f'start of the window on any seed')
                    continue
                loss = [r.get(f'{name}_{label}_loss') for r in recs]
                fee_ = [r.get(f'{name}_{label}_fees') for r in recs]
                loss = [v for v in loss if v is not None and np.isfinite(v)]
                fee_ = [v for v in fee_ if v is not None and np.isfinite(v)]
                m, lo, hi = boot_ci(vals, seed=0 if bps is None else int(bps * 100))
                lines.append(
                    f'    {name:9s} {label:7s} {np.median(loss):9.4f} '
                    f'{np.median(fee_):9.4f} {m:9.4f} '
                    f'{"[" + f"{lo:+.4f}, {hi:+.4f}" + "]":>22s}  {len(vals)}')
        usable = [r for r in recs
                  if r.get('total_calm') is not None
                  and r.get('total_crisis') is not None]
        pt, lo, hi, n = paired_bootstrap_breakeven(usable)
        lines.append(f'    operating break even crisis state share, before the '
                     f'cost of capital')
        lines.append(f'      {pt:.4%}  [{lo:.4%}, {hi:.4%}]  paired over {n} seeds')
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
        print(f'saved {out}')


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _bootstrap_summary(values, draws, seed):
    values = [float(value) for value in values
              if value is not None and np.isfinite(value)]
    if not values:
        return {'n': 0, 'median': None, 'bootstrap_95_interval': None}
    if int(draws) <= 0:
        return {
            'n': len(values),
            'median': float(np.median(values)),
            'bootstrap_95_interval': None,
        }
    median, lo, hi = boot_ci(values, n=int(draws), seed=int(seed))
    return {
        'n': len(values),
        'median': _finite_or_none(median),
        'bootstrap_95_interval': (
            [_finite_or_none(lo), _finite_or_none(hi)]
            if np.isfinite(lo) and np.isfinite(hi) else None
        ),
    }


def aggregate_json(fees, lp_model, seeds=None, *, seed_start=None,
                   requested_seeds=None, bootstrap_draws=20000,
                   bootstrap_seed=0):
    """Machine-readable counterpart of :func:`report`.

    Completeness is about requested seeds, not whether a window happens to
    contain capital. A closed pool makes that window economically
    unmeasurable and is counted as such; it does not make the cache incomplete.
    No sign or profitability target is imposed. The accounting check asks only
    whether every stored net return equals fees minus realised loss.
    """
    wanted = None if seeds is None else set(int(seed) for seed in seeds)
    requested_count = None if wanted is None else len(wanted)
    settings = []
    overall_complete = True if wanted is not None else None
    stat_index = 0

    for bps in fees:
        raw, stale = load_raw(bps, lp_model)
        recs = [record for seed, record in sorted(raw.items())
                if wanted is None or seed in wanted]
        available = {int(record['seed']) for record in recs}
        seed_complete = None if wanted is None else available == wanted
        raw_provenance_clean = int(stale) == 0
        complete = (
            None if seed_complete is None
            else bool(seed_complete and raw_provenance_clean)
        )
        if overall_complete is not None:
            overall_complete = bool(overall_complete and complete)

        bases = sorted({
            key.rsplit('_', 1)[0]
            for record in recs for key in record
            if key.endswith('_calm') or key.endswith('_crisis')
        }, key=lambda name: (name == 'total', name))
        windows = {}
        identities = []
        for label in ('calm', 'crisis'):
            windows[label] = {}
            for name in bases:
                net_key = f'{name}_{label}'
                loss_key = f'{name}_{label}_loss'
                fee_key = f'{name}_{label}_fees'
                net = [record.get(net_key) for record in recs]
                loss = [record.get(loss_key) for record in recs]
                income = [record.get(fee_key) for record in recs]
                metric_rows = {}
                for metric, values in (('loss_pct', loss),
                                       ('fees_pct', income),
                                       ('net_pct', net)):
                    metric_rows[metric] = _bootstrap_summary(
                        values, bootstrap_draws, bootstrap_seed + stat_index
                    )
                    stat_index += 1
                for record in recs:
                    n_v = record.get(net_key)
                    l_v = record.get(loss_key)
                    f_v = record.get(fee_key)
                    if n_v is None or l_v is None or f_v is None:
                        continue
                    triple = (float(n_v), float(l_v), float(f_v))
                    if not all(np.isfinite(value) for value in triple):
                        continue
                    identities.append(abs(triple[0] - (triple[2] - triple[1])))
                measurable = metric_rows['net_pct']['n']
                windows[label][name] = {
                    **metric_rows,
                    'requested_or_available_seed_count': len(recs),
                    'measurable_seed_count': measurable,
                    'unmeasurable_seed_count': len(recs) - measurable,
                }

        usable = [record for record in recs
                  if record.get('total_calm') is not None
                  and record.get('total_crisis') is not None]
        point, lo, hi, paired_n = paired_bootstrap_breakeven(
            usable, n=int(bootstrap_draws), seed=int(bootstrap_seed)
        )
        max_identity_error = max(identities) if identities else None
        settings.append({
            'fee_bps': None if bps is None else float(bps),
            'fee_label': fee_label(bps),
            'fee_mode': 'calibrated' if bps is None else 'uniform_all_amm_venues',
            'raw_cache_path': os.path.relpath(raw_path(bps, lp_model), ROOT),
            'requested_seed_count': requested_count,
            'available_seed_count': len(recs),
            'available_seed_start': min(available) if available else None,
            'available_seed_end': max(available) if available else None,
            'stale_records_ignored': int(stale),
            'invalid_or_stale_records_ignored': int(stale),
            'raw_provenance_clean': raw_provenance_clean,
            'complete': complete,
            'windows': windows,
            'accounting_sign_convention': {
                'loss_positive_is_provider_cost': True,
                'fees_positive_is_provider_income': True,
                'net_formula': 'fees_minus_loss',
                'identity_observations': len(identities),
                'max_abs_identity_error_pct': _finite_or_none(max_identity_error),
                'identity_holds': bool(
                    identities and max_identity_error <= 1e-10
                ),
            },
            'operating_break_even_crisis_state_share': {
                'point': _finite_or_none(point),
                'paired_bootstrap_95_interval': (
                    [_finite_or_none(lo), _finite_or_none(hi)]
                    if np.isfinite(lo) and np.isfinite(hi) else None
                ),
                'paired_seed_count': int(paired_n),
                'before_capital_opportunity_cost': True,
            },
        })

    invalid_or_stale_total = sum(
        int(row['invalid_or_stale_records_ignored']) for row in settings
    )
    return {
        'model_signature': model_signature(ROOT),
        'measurement_signature': run_signature(),
        'report_signature': aggregate_report_signature(),
        'aggregate_schema_version': 1,
        'configuration': {
            'preset': PRESET,
            'n_iter': N_ITER,
            'lp_model': lp_model,
            'fee_grid_bps': [None if fee is None else float(fee) for fee in fees],
            'calm_window_relative_to_shock': list(CALM),
            'crisis_window_relative_to_shock': list(CRISIS),
            'seed_start': seed_start,
            'requested_seeds_per_fee': requested_seeds,
            'requested_seed_list': (
                sorted(wanted) if wanted is not None else None
            ),
            'bootstrap_draws': int(bootstrap_draws),
            'bootstrap_seed': int(bootstrap_seed),
        },
        'complete': overall_complete,
        'invalid_or_stale_records_ignored': invalid_or_stale_total,
        'raw_provenance_clean': bool(
            settings and invalid_or_stale_total == 0
            and all(row['raw_provenance_clean'] for row in settings)
        ),
        'capital_flow_zero_result_self_check_passed': bool(self_check(verbose=False)),
        'profitability_acceptance_target': None,
        'interpretation': (
            'Accounting/sign and sampling report. Positive profit is an outcome, '
            'not an acceptance criterion.'
        ),
        'fee_settings': settings,
    }


_AGGREGATE_REPORT_SIGNATURE = None


def aggregate_report_signature():
    """Digest of the transformation from current raw P&L records to JSON."""
    global _AGGREGATE_REPORT_SIGNATURE
    if _AGGREGATE_REPORT_SIGNATURE is None:
        _AGGREGATE_REPORT_SIGNATURE = measurement_signature(
            'lp_pnl_corrected_aggregate',
            run_signature(),
            functions=(
                _bootstrap_summary,
                paired_bootstrap_breakeven,
                aggregate_json,
                self_check,
            ),
            constants=(1, CALM, CRISIS, 'fees_minus_loss'),
        )
    return _AGGREGATE_REPORT_SIGNATURE


def write_json_aggregate(path, fees, lp_model, seeds=None, **kwargs):
    payload = aggregate_json(fees, lp_model, seeds, **kwargs)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'saved {path}')
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=300,
                    help='how many seeds the target is, counted from --seed-start')
    ap.add_argument('--seed-start', type=int, default=42)
    ap.add_argument('--chunk', type=int, default=0,
                    help='stop after this many new seeds, so a long sweep can be '
                         'accumulated across several runs')
    ap.add_argument('--fee-bps', type=float, nargs='*', default=[5.0],
                    help='uniform fee on both venues, one run per value')
    ap.add_argument('--baseline', action='store_true',
                    help='measure the calibrated arm instead, 20 bps on the '
                         'constant product pool and 5 on the hybrid one')
    ap.add_argument('--include-baseline', action='store_true',
                    help='prepend the calibrated arm to the requested uniform '
                         'fee grid')
    ap.add_argument('--lp-model', choices=['rule', 'endogenous'], default='rule')
    ap.add_argument('--out', default=None)
    ap.add_argument('--json-out', default=None,
                    help='optional machine-readable aggregate; the text report '
                         'is unchanged')
    ap.add_argument('--bootstrap-draws', type=int, default=20000)
    ap.add_argument('--bootstrap-seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--all-seeds', action='store_true',
                    help='report every cached seed instead of only the range '
                         'asked for')
    ap.add_argument('--report-only', action='store_true',
                    help='aggregate what has already been measured and stop')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()

    if args.self_check:
        print('Capital flows netted out on a path with no investment result\n')
        return 0 if self_check() else 1

    fees = ([None] if args.baseline else
            ([None] if args.include_baseline else []) + args.fee_bps)
    want = (None if args.all_seeds
            else set(range(args.seed_start, args.seed_start + args.seeds)))
    if not args.report_only:
        wanted = list(range(args.seed_start, args.seed_start + args.seeds))
        done_new = 0
        for bps in fees:
            have, stale = load_raw(bps, args.lp_model)
            if stale:
                print(f'  {stale} cached seeds at {fee_label(bps)} do not match '
                      f'the current model and will be measured again')
            todo = [sd for sd in wanted if sd not in have]
            if args.chunk:
                todo = todo[:max(0, args.chunk - done_new)]
            if not todo:
                continue
            jobs = [(sd, bps, args.lp_model) for sd in todo]
            if args.workers > 1:
                with Pool(args.workers) as pool:
                    for rec in pool.imap_unordered(_measure_one, jobs):
                        append_raw(bps, args.lp_model, rec)
                        done_new += 1
            else:
                for job in jobs:
                    append_raw(bps, args.lp_model, _measure_one(job))
                    done_new += 1
            if args.chunk and done_new >= args.chunk:
                print(f'stopped after {done_new} new seeds this run')
                report(fees, args.lp_model, args.out, want)
                if args.json_out:
                    payload = write_json_aggregate(
                        args.json_out, fees, args.lp_model, want,
                        seed_start=args.seed_start,
                        requested_seeds=args.seeds,
                        bootstrap_draws=args.bootstrap_draws,
                        bootstrap_seed=args.bootstrap_seed,
                    )
                    return 0 if payload['complete'] else 2
                return 0
    report(fees, args.lp_model, args.out, want)
    if args.json_out:
        payload = write_json_aggregate(
            args.json_out, fees, args.lp_model, want,
            seed_start=args.seed_start,
            requested_seeds=args.seeds,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        return 0 if payload['complete'] else 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
