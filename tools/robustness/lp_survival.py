#!/usr/bin/env python3
"""Does the facility survive to the crisis when providers may leave.

    python3 tools/robustness/lp_survival.py --seeds 60

Legacy provider economics were measured on the rule provider, which carries a
floor and therefore cannot be abandoned. The primary runtime now uses the
endogenous population, and this tool audits the resulting prior question. Is
the pool still open when the shock lands, and is it still open while the shock
is being absorbed.

Nothing here measures profit. ``lp_pnl_corrected.py`` does that and refuses to
report a crisis figure when the pool held no capital at the start of the
window, which is exactly the state this tool is built to detect and count.

The outside option defaults to the cost of capital already anchored in the
calibration file, ``funding_rate_scale``, about three per cent a year at one
second to the tick. It is not a free parameter and is not fitted here. The
behavioural parameters are another matter. Exit and entry patience, the EWMA
speed, adjustment cap and the dispersion of the outside option have no
counterpart in \\FX{} data, so they are swept and not calibrated, and
the full configuration is part of the cache key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from multiprocessing import Pool as ProcPool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from tools.robustness.lp_pnl_corrected import CALM, CRISIS, N_ITER, PRESET
from tools.robustness.signatures import measurement_signature, model_signature

RAW_DIR = os.path.join(ROOT, 'output', 'resilience', 'raw_survival')

# Raw survival rows are a publication input, not an informal cache.  A schema
# version makes old/partial rows fail closed when the measurement changes.
_RAW_SCHEMA_VERSION = 1
_CANONICAL_POOL_NAMES = ('hfmm',)
_RAW_RECORD_FIELDS = frozenset({
    'raw_schema_version', 'seed', 'shock', 'pools', 'model_signature',
    'signature', 'configuration',
})
_CONFIG_FLOAT_FIELDS = frozenset({
    'outside_option', 'subsidy_rate', 'loss_rebate_fraction', 'kappa',
    'response_scale', 'max_adj', 'ewma_alpha', 'entry_margin',
})
_CONFIG_INT_FIELDS = frozenset({'exit_patience', 'entry_patience', 'n_iter'})
_POOL_BOOL_FIELDS = frozenset({
    'open_at_shock', 'open_after_initial_shock_step', 'open_through_crisis',
    'gross_event_identity_holds', 'claim_supply_identity_end_holds',
})
_POOL_INT_FIELDS = frozenset({
    'exit_events_total', 'new_entry_events_total', 'reentry_events_total',
    'new_entry_events_pre_shock', 'new_entry_events_crisis',
    'new_entry_events_post_crisis', 'reentry_events_pre_shock',
    'reentry_events_crisis', 'reentry_events_post_crisis',
    'new_entry_risk_set_providers', 'new_entry_right_censored_providers',
    'reentry_risk_set_episodes', 'reentry_completed_episodes',
    'reentry_right_censored_episodes', 'reentry_right_censored_providers',
    'max_entry_streak',
})
_POOL_NULLABLE_INT_FIELDS = frozenset({
    'first_close', 'first_new_entry_tick', 'first_reentry_tick',
})
_POOL_FLOAT_FIELDS = frozenset({
    'ticks_closed_pre', 'ticks_closed_crisis', 'ticks_closed_total',
    'active_pre_shock', 'active_min_crisis', 'active_end',
    'deployed_pre_shock', 'deployed_end', 'deployed_share_pre_shock',
    'deployed_share_end', 'supply_ratio_pre_shock',
    'supply_ratio_min_crisis', 'supply_ratio_end', 'exits_in_crisis',
    'net_active_drawdown_in_crisis', 'gross_event_identity_max_abs_error',
    'claim_supply_identity_end_abs_error', 'max_entry_progress',
    'entry_progress_at_horizon', 'rho_calm_median', 'rho_over_option',
    'x_ratio_min_crisis',
})
_POOL_ROW_FIELDS = frozenset({'pool'}).union(
    _POOL_BOOL_FIELDS,
    _POOL_INT_FIELDS,
    _POOL_NULLABLE_INT_FIELDS,
    _POOL_FLOAT_FIELDS,
)


def _calibrated_lp_defaults():
    """Runtime LP defaults from the canonical calibration file.

    Reading them here instead of repeating literals keeps the survival audit
    on the same economic specification as the primary runtime.
    """
    path = os.path.join(ROOT, 'calibration', 'primary_model.json')
    try:
        with open(path, encoding='utf-8') as fh:
            cal = json.load(fh)
    except OSError:
        cal = {}
    defaults = cal.get('cli_defaults', {}) if isinstance(cal, dict) else {}
    return {
        'outside_option': float(defaults.get('amm_lp_outside_option', 1.3319e-9)),
        'subsidy_rate': float(defaults.get('amm_lp_subsidy_rate', 0.0)),
        'loss_rebate_fraction': float(defaults.get(
            'amm_lp_loss_rebate_fraction', 0.0)),
        'exit_patience': int(defaults.get('amm_lp_exit_patience', 290)),
        'entry_patience': int(defaults.get('amm_lp_entry_patience', 290)),
        'kappa': float(defaults.get('amm_lp_kappa', 0.35)),
        'response_scale': float(defaults.get('amm_lp_response_scale', 1e-6)),
        'max_adj': float(defaults.get('amm_lp_max_adj', 0.0023873085271651773)),
        'ewma_alpha': float(defaults.get('amm_lp_ewma_alpha', 0.02)),
        'entry_margin': float(defaults.get('amm_lp_entry_margin', 0.25)),
    }


def _run_config(args) -> dict:
    defaults = _calibrated_lp_defaults()
    return {
        'outside_option': float(defaults['outside_option'] if args.outside_option is None
                                else args.outside_option),
        'subsidy_rate': float(defaults['subsidy_rate'] if args.subsidy_rate is None
                              else args.subsidy_rate),
        'loss_rebate_fraction': float(
            defaults['loss_rebate_fraction'] if args.loss_rebate is None
            else args.loss_rebate),
        'exit_patience': int(defaults['exit_patience'] if args.exit_patience is None
                             else args.exit_patience),
        'entry_patience': int(defaults['entry_patience'] if args.entry_patience is None
                              else args.entry_patience),
        'kappa': float(defaults['kappa'] if args.kappa is None else args.kappa),
        'response_scale': float(defaults['response_scale']
                                if args.response_scale is None
                                else args.response_scale),
        'max_adj': float(defaults['max_adj'] if args.max_adjustment is None
                         else args.max_adjustment),
        'ewma_alpha': float(defaults['ewma_alpha'] if args.ewma_alpha is None
                            else args.ewma_alpha),
        'entry_margin': float(defaults['entry_margin'] if args.entry_margin is None
                              else args.entry_margin),
        'n_iter': int(args.n_iter),
    }


def _simulate(seed, config):
    """One seeded run of the dealer liquidity crisis with a free provider."""
    argv = ['--preset', PRESET, '--seed', str(seed),
            '--n-iter', str(config['n_iter']), '--silent',
            '--amm-lp-model', 'endogenous',
            '--amm-lp-outside-option', repr(config['outside_option']),
            '--amm-lp-subsidy-rate', repr(config['subsidy_rate']),
            '--amm-lp-loss-rebate', repr(config['loss_rebate_fraction']),
            '--amm-lp-exit-patience', str(config['exit_patience']),
            '--amm-lp-entry-patience', str(config['entry_patience']),
            '--amm-lp-kappa', repr(config['kappa']),
            '--amm-lp-response-scale', repr(config['response_scale']),
            '--amm-lp-max-adjustment', repr(config['max_adj']),
            '--amm-lp-ewma-alpha', repr(config['ewma_alpha']),
            '--amm-lp-entry-margin', repr(config['entry_margin'])]
    p = build_parser()
    a = p.parse_args(argv)
    _apply_preset_defaults(p, a)
    a.venue_choice_rule = _resolve_main_routing(a, argv)
    _auto_stress_around_shock(a)
    a.enable_amm = 1
    a.clob_amm_interaction = 'competition'
    _seed_all(seed)
    sim = build_sim(a)
    sim.simulate(a.n_iter, silent=True)
    return sim, int(a.shock_iter)


def _read_population(lp, shock, option):
    """What one provider population did around the shock, as plain numbers."""
    h = lp.history
    closed = np.asarray(h['closed'], dtype=float)
    active = np.asarray(h['active'], dtype=float)
    rho = np.asarray(h['rho'], dtype=float)
    supply = np.asarray(h.get('supply', []), dtype=float)
    n = closed.size
    if n <= shock:
        return None

    def aligned(key):
        values = np.asarray(h.get(key, []), dtype=float)
        return values if values.size == n else np.zeros(n, dtype=float)

    entries = aligned('entries_gross')
    reentries = aligned('reentries_gross')
    exits = aligned('exits_gross')
    deployed = aligned('deployed')
    deployed_share = aligned('deployed_share')
    entry_progress = aligned('entry_progress_max')

    def first_event(values):
        hits = np.flatnonzero(values > 0)
        return int(hits[0]) if hits.size else None

    c0, c1 = max(0, shock + CALM[0]), max(1, shock + CALM[1])
    k0, k1 = shock + CRISIS[0], min(n, shock + CRISIS[1])
    calm_rho = rho[c0:c1]
    crisis_closed = closed[k0:k1]
    crisis_active = active[k0:k1]
    crisis_supply = supply[k0:k1] if supply.size == n else np.asarray([])

    shut = np.flatnonzero(closed > 0)
    pre_index = max(0, shock - 1)
    pre = float(active[pre_index])
    supply0 = float(supply[0]) if supply.size == n and supply[0] > 0.0 else float('nan')

    providers = list(getattr(lp, 'providers', []))
    n_incumbents = int(getattr(lp, 'n_incumbents', len(providers)))
    expected_active = (n_incumbents + np.cumsum(entries)
                       + np.cumsum(reentries) - np.cumsum(exits))
    event_identity_error = float(np.max(np.abs(active - expected_active)))
    final_claims = sum(
        max(0.0, float(getattr(provider, 'tokens', 0.0)))
        + max(0.0, float(getattr(provider, 'pending_burn', 0.0)))
        for provider in providers
    )
    final_supply = float(supply[-1]) if supply.size == n else float('nan')
    claim_supply_error = (
        abs(final_claims - final_supply)
        if math.isfinite(final_supply) else float('nan')
    )
    claim_supply_tolerance = 1e-9 * max(1.0, abs(final_supply)) \
        if math.isfinite(final_supply) else float('nan')
    potential = providers[n_incumbents:]
    new_entry_risk = len(potential)
    new_entry_censored = sum(int(getattr(provider, 'entry_count', 0)) == 0
                             for provider in potential)
    reentry_risk_episodes = sum(
        int(getattr(provider, 'exit_count', 0)) for provider in providers
    )
    reentry_completed_episodes = sum(
        int(getattr(provider, 'reentry_count', 0)) for provider in providers
    )
    reentry_censored_episodes = sum(
        max(0, int(getattr(provider, 'exit_count', 0))
            - int(getattr(provider, 'reentry_count', 0)))
        for provider in providers
    )
    reentry_censored = sum(
        int(getattr(provider, 'exit_count', 0))
        > int(getattr(provider, 'reentry_count', 0))
        for provider in providers
    )
    entry_population = potential if potential else providers
    max_entry_streak = max(
        (int(getattr(provider, 'max_entry_streak', 0))
         for provider in entry_population),
        default=0,
    )
    max_entry_progress = max(
        (min(1.0, float(getattr(provider, 'max_entry_streak', 0))
         / max(1, int(getattr(provider, 'entry_patience', 1))))
         for provider in entry_population),
        default=0.0,
    )

    x = np.asarray(getattr(lp.pool, 'x_history', []), dtype=float)
    x_ratio = float(np.min(x[k0:k1 + 1]) / x[0]) if x.size > k1 and x[0] else float('nan')

    return {
        'pool': type(lp.pool).__name__.replace('Pool', '').lower(),
        # Availability at the instant the shock lands is the state carried into
        # the shock step. ``closed[shock]`` is already the state after that
        # step and would put a one tick look ahead into the label.
        'open_at_shock': bool(closed[pre_index] == 0),
        'open_after_initial_shock_step': bool(closed[shock] == 0),
        'open_through_crisis': bool(closed[pre_index] == 0
                                    and crisis_closed.sum() == 0),
        'first_close': int(shut[0]) if shut.size else None,
        'ticks_closed_pre': float(closed[:shock].sum()),
        'ticks_closed_crisis': float(crisis_closed.sum()),
        'ticks_closed_total': float(closed.sum()),
        'active_pre_shock': pre,
        'active_min_crisis': float(crisis_active.min()) if crisis_active.size else float('nan'),
        'active_end': float(active[-1]),
        'deployed_pre_shock': float(deployed[pre_index]),
        'deployed_end': float(deployed[-1]),
        'deployed_share_pre_shock': float(deployed_share[pre_index]),
        'deployed_share_end': float(deployed_share[-1]),
        'supply_ratio_pre_shock': (
            float(supply[pre_index] / supply0) if math.isfinite(supply0) else float('nan')
        ),
        'supply_ratio_min_crisis': (
            float(np.min(crisis_supply) / supply0)
            if crisis_supply.size and math.isfinite(supply0) else float('nan')
        ),
        'supply_ratio_end': (
            float(supply[-1] / supply0) if math.isfinite(supply0) else float('nan')
        ),
        # Gross events do not net simultaneous entries against exits.  The old
        # pre-minus-min-active proxy remains as a separately named drawdown.
        'exits_in_crisis': float(exits[k0:k1].sum()),
        'net_active_drawdown_in_crisis': (
            pre - (float(crisis_active.min()) if crisis_active.size else pre)
        ),
        'exit_events_total': int(exits.sum()),
        'new_entry_events_total': int(entries.sum()),
        'reentry_events_total': int(reentries.sum()),
        'new_entry_events_pre_shock': int(entries[:shock].sum()),
        'new_entry_events_crisis': int(entries[k0:k1].sum()),
        'new_entry_events_post_crisis': int(entries[k1:].sum()),
        'reentry_events_pre_shock': int(reentries[:shock].sum()),
        'reentry_events_crisis': int(reentries[k0:k1].sum()),
        'reentry_events_post_crisis': int(reentries[k1:].sum()),
        'first_new_entry_tick': first_event(entries),
        'first_reentry_tick': first_event(reentries),
        'new_entry_risk_set_providers': new_entry_risk,
        'new_entry_right_censored_providers': new_entry_censored,
        'reentry_risk_set_episodes': reentry_risk_episodes,
        'reentry_completed_episodes': reentry_completed_episodes,
        'reentry_right_censored_episodes': reentry_censored_episodes,
        'reentry_right_censored_providers': reentry_censored,
        'gross_event_identity_holds': bool(event_identity_error <= 1e-9),
        'gross_event_identity_max_abs_error': event_identity_error,
        'claim_supply_identity_end_holds': bool(
            math.isfinite(claim_supply_error)
            and claim_supply_error <= claim_supply_tolerance
        ),
        'claim_supply_identity_end_abs_error': claim_supply_error,
        'max_entry_streak': max_entry_streak,
        'max_entry_progress': max_entry_progress,
        'entry_progress_at_horizon': float(entry_progress[-1]),
        'rho_calm_median': float(np.median(calm_rho)) if calm_rho.size else float('nan'),
        # How far the calm earnings sit above the alternative. Below one and
        # the provider is giving something up by staying.
        'rho_over_option': (float(np.median(calm_rho)) / option) if option else float('nan'),
        'x_ratio_min_crisis': x_ratio,
    }


def measure(job):
    """One seed, both pools."""
    seed, config = job
    sim, shock = _simulate(seed, config)
    pools = []
    for lp in getattr(sim, 'lp_providers', []):
        rec = _read_population(lp, shock, config['outside_option'])
        if rec is not None:
            pools.append(rec)
    return {'raw_schema_version': _RAW_SCHEMA_VERSION,
            'seed': seed, 'shock': shock, 'pools': pools,
            'model_signature': model_signature(ROOT),
            'signature': survival_signature(), 'configuration': config}


_SURVIVAL_SIGNATURE = None


def survival_signature():
    global _SURVIVAL_SIGNATURE
    if _SURVIVAL_SIGNATURE is None:
        _SURVIVAL_SIGNATURE = measurement_signature(
            'lp_survival',
            model_signature(ROOT),
            functions=(_simulate, _read_population, measure),
            constants=(PRESET, N_ITER, CALM, CRISIS, _RAW_SCHEMA_VERSION,
                       _CANONICAL_POOL_NAMES),
        )
    return _SURVIVAL_SIGNATURE


def _raw_path(config, signature=None):
    """Signature-scoped append-only cache for one economic configuration."""
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'))
    tag = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    signature = survival_signature() if signature is None else str(signature)
    return os.path.join(RAW_DIR, f'survival_{tag}_{signature}.jsonl')


def _configuration_matches(observed, expected):
    """Whether a raw row carries the exact typed survival configuration."""
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    required = _CONFIG_FLOAT_FIELDS | _CONFIG_INT_FIELDS
    if set(observed) != required or set(expected) != required:
        return False
    for key in _CONFIG_FLOAT_FIELDS:
        value = observed[key]
        if type(value) is not float or not math.isfinite(value):
            return False
        expected_value = expected[key]
        if type(expected_value) is not float or value != expected_value:
            return False
    for key in _CONFIG_INT_FIELDS:
        value = observed[key]
        if type(value) is not int or type(expected[key]) is not int:
            return False
        if value != expected[key]:
            return False
    return observed['exit_patience'] > 0 \
        and observed['entry_patience'] > 0 \
        and observed['n_iter'] > 0


def _pool_row_is_canonical(row, pool_name, n_iter):
    """Validate one raw pool row without constructing a simulation."""
    if not isinstance(row, dict) or set(row) != _POOL_ROW_FIELDS:
        return False
    if type(row['pool']) is not str or row['pool'] != pool_name:
        return False
    if any(type(row[key]) is not bool for key in _POOL_BOOL_FIELDS):
        return False
    if any(type(row[key]) is not int or row[key] < 0
           for key in _POOL_INT_FIELDS):
        return False
    for key in _POOL_NULLABLE_INT_FIELDS:
        value = row[key]
        if value is not None and (
                type(value) is not int or value < 0 or value >= n_iter):
            return False
    if any(type(row[key]) is not float or not math.isfinite(row[key])
           for key in _POOL_FLOAT_FIELDS):
        return False
    if row['gross_event_identity_max_abs_error'] < 0.0:
        return False
    if row['claim_supply_identity_end_abs_error'] < 0.0:
        return False
    if row['new_entry_right_censored_providers'] \
            > row['new_entry_risk_set_providers']:
        return False
    if row['reentry_completed_episodes'] \
            + row['reentry_right_censored_episodes'] \
            != row['reentry_risk_set_episodes']:
        return False
    if row['new_entry_events_pre_shock'] \
            + row['new_entry_events_crisis'] \
            + row['new_entry_events_post_crisis'] \
            != row['new_entry_events_total']:
        return False
    if row['reentry_events_pre_shock'] \
            + row['reentry_events_crisis'] \
            + row['reentry_events_post_crisis'] \
            != row['reentry_events_total']:
        return False
    if row['open_through_crisis'] and not row['open_at_shock']:
        return False
    return True


def _raw_record_is_canonical(rec, signature, config, expected_model_signature):
    """Fail-closed validation for one cached survival measurement."""
    if not isinstance(rec, dict) or set(rec) != _RAW_RECORD_FIELDS:
        return False
    if type(rec['raw_schema_version']) is not int \
            or rec['raw_schema_version'] != _RAW_SCHEMA_VERSION:
        return False
    if type(rec['seed']) is not int or rec['seed'] < 0:
        return False
    if type(rec['shock']) is not int \
            or rec['shock'] < 0 or rec['shock'] >= config['n_iter']:
        return False
    if type(rec['signature']) is not str or rec['signature'] != signature:
        return False
    if type(rec['model_signature']) is not str \
            or rec['model_signature'] != expected_model_signature:
        return False
    if not _configuration_matches(rec['configuration'], config):
        return False
    pools = rec['pools']
    if not isinstance(pools, list) or len(pools) != len(_CANONICAL_POOL_NAMES):
        return False
    names = [row.get('pool') if isinstance(row, dict) else None for row in pools]
    if names != list(_CANONICAL_POOL_NAMES) or len(names) != len(set(names)):
        return False
    return all(_pool_row_is_canonical(row, name, config['n_iter'])
               for row, name in zip(pools, _CANONICAL_POOL_NAMES))


def load_raw(path, signature, config):
    """Valid current-model seeds and the count of rejected raw rows."""
    have, invalid = {}, 0
    if not os.path.exists(path):
        return have, invalid
    expected_model_signature = model_signature(ROOT)
    duplicate_seeds = set()
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid += 1
                continue
            if not _raw_record_is_canonical(
                    rec, signature, config, expected_model_signature):
                invalid += 1
                continue
            seed = rec['seed']
            if seed in duplicate_seeds:
                invalid += 1
                continue
            if seed in have:
                # Neither row is trusted once a cache contains two ostensibly
                # canonical measurements for the same seed.
                have.pop(seed)
                duplicate_seeds.add(seed)
                invalid += 2
                continue
            have[seed] = rec
    return have, invalid


def append_raw(rec, signature=None):
    """Append only a canonical row to its signature-scoped cache."""
    if not isinstance(rec, dict) or not isinstance(rec.get('configuration'), dict):
        raise ValueError('refusing malformed survival raw record')
    signature = survival_signature() if signature is None else str(signature)
    config = rec['configuration']
    if not _raw_record_is_canonical(
            rec, signature, config, model_signature(ROOT)):
        raise ValueError('refusing malformed survival raw record')
    path = _raw_path(config, signature)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(rec, allow_nan=False) + '\n')
    return path


def _wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0 or successes < 0 or successes > trials:
        return [None, None]
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    ) / denominator
    return [centre - half_width, centre + half_width]


def summarize(recs, config):
    """Machine-readable survival facts, including sampling uncertainty."""
    by_pool = {}
    for rec in recs:
        for row in rec.get('pools', []):
            by_pool.setdefault(row['pool'], []).append(row)

    pools = {}
    for name, rows in sorted(by_pool.items()):
        n = len(rows)
        open_at_shock = sum(bool(row['open_at_shock']) for row in rows)
        open_after_step = sum(bool(row.get('open_after_initial_shock_step',
                                           row['open_at_shock'])) for row in rows)
        open_through = sum(bool(row['open_through_crisis']) for row in rows)
        any_close = sum(row.get('first_close') is not None for row in rows)
        any_entry = sum(int(row.get('new_entry_events_total', 0)) > 0 for row in rows)
        any_reentry = sum(int(row.get('reentry_events_total', 0)) > 0 for row in rows)
        any_exit = sum(int(row.get('exit_events_total', 0)) > 0 for row in rows)
        entry_events = sum(int(row.get('new_entry_events_total', 0)) for row in rows)
        reentry_events = sum(int(row.get('reentry_events_total', 0)) for row in rows)
        exit_events = sum(int(row.get('exit_events_total', 0)) for row in rows)
        entry_risk = sum(int(row.get('new_entry_risk_set_providers', 0)) for row in rows)
        entry_censored = sum(
            int(row.get('new_entry_right_censored_providers', 0)) for row in rows
        )
        reentry_censored = sum(
            int(row.get('reentry_right_censored_providers', 0)) for row in rows
        )
        reentry_risk_episodes = sum(
            int(row.get('reentry_risk_set_episodes',
                        row.get('exit_events_total', 0)))
            for row in rows
        )
        reentry_completed_episodes = sum(
            int(row.get('reentry_completed_episodes',
                        row.get('reentry_events_total', 0)))
            for row in rows
        )
        reentry_censored_episodes = sum(
            int(row.get('reentry_right_censored_episodes',
                        max(0,
                            int(row.get('exit_events_total', 0))
                            - int(row.get('reentry_events_total', 0)))))
            for row in rows
        )
        event_identity_ok = sum(row.get('gross_event_identity_holds') is True
                                for row in rows)
        claim_identity_ok = sum(row.get('claim_supply_identity_end_holds') is True
                                for row in rows)

        def median(key):
            values = [float(row[key]) for row in rows
                      if row.get(key) is not None
                      and math.isfinite(float(row[key]))]
            return float(np.median(values)) if values else None

        pools[name] = {
            'n_seeds': n,
            'open_at_shock_count': open_at_shock,
            'open_at_shock_rate': open_at_shock / n if n else None,
            'open_at_shock_rate_wilson_95': _wilson_interval(open_at_shock, n),
            'open_after_initial_shock_step_count': open_after_step,
            'open_after_initial_shock_step_rate': open_after_step / n if n else None,
            'open_after_initial_shock_step_rate_wilson_95': (
                _wilson_interval(open_after_step, n)
            ),
            'open_through_crisis_count': open_through,
            'open_through_crisis_rate': open_through / n if n else None,
            'open_through_crisis_rate_wilson_95': _wilson_interval(open_through, n),
            'any_closure_count': any_close,
            'any_closure_rate': any_close / n if n else None,
            'any_closure_rate_wilson_95': _wilson_interval(any_close, n),
            'seeds_with_any_exit_count': any_exit,
            'seeds_with_any_exit_rate': any_exit / n if n else None,
            'seeds_with_any_exit_rate_wilson_95': _wilson_interval(any_exit, n),
            'seeds_with_new_entry_count': any_entry,
            'seeds_with_new_entry_rate': any_entry / n if n else None,
            'seeds_with_new_entry_rate_wilson_95': _wilson_interval(any_entry, n),
            'seeds_with_reentry_count': any_reentry,
            'seeds_with_reentry_rate': any_reentry / n if n else None,
            'seeds_with_reentry_rate_wilson_95': _wilson_interval(any_reentry, n),
            'gross_exit_events': exit_events,
            'gross_new_entry_events': entry_events,
            'gross_reentry_events': reentry_events,
            'new_entry_risk_set_providers': entry_risk,
            'new_entry_right_censored_providers': entry_censored,
            'new_entry_provider_activation_rate': (
                (entry_risk - entry_censored) / entry_risk if entry_risk else None
            ),
            'new_entry_provider_activation_rate_wilson_95': (
                _wilson_interval(entry_risk - entry_censored, entry_risk)
            ),
            'reentry_risk_set_episodes': reentry_risk_episodes,
            'reentry_completed_episodes': reentry_completed_episodes,
            'reentry_right_censored_episodes': reentry_censored_episodes,
            'reentry_episode_activation_rate': (
                reentry_completed_episodes / reentry_risk_episodes
                if reentry_risk_episodes else None
            ),
            'reentry_episode_activation_rate_wilson_95': (
                _wilson_interval(reentry_completed_episodes,
                                 reentry_risk_episodes)
            ),
            'reentry_right_censored_providers': reentry_censored,
            'gross_event_identity_pass_count': event_identity_ok,
            'gross_event_identity_pass_rate': event_identity_ok / n if n else None,
            'claim_supply_identity_end_pass_count': claim_identity_ok,
            'claim_supply_identity_end_pass_rate': claim_identity_ok / n if n else None,
            'entry_activation_observed': bool(entry_events or reentry_events),
            'entry_activation_status': (
                'observed'
                if entry_events or reentry_events
                else ('right_censored_no_activation_by_horizon'
                      if (entry_censored or reentry_censored
                          or reentry_censored_episodes)
                      else 'no_population_at_risk')
            ),
            'median_first_close_tick': median('first_close'),
            'median_ticks_closed_pre_shock': median('ticks_closed_pre'),
            'median_ticks_closed_crisis': median('ticks_closed_crisis'),
            'median_exits_in_crisis': median('exits_in_crisis'),
            'median_active_pre_shock': median('active_pre_shock'),
            'median_active_min_crisis': median('active_min_crisis'),
            'median_active_end': median('active_end'),
            'median_deployed_pre_shock': median('deployed_pre_shock'),
            'median_deployed_end': median('deployed_end'),
            'median_deployed_share_pre_shock': median('deployed_share_pre_shock'),
            'median_deployed_share_end': median('deployed_share_end'),
            'median_supply_ratio_pre_shock': median('supply_ratio_pre_shock'),
            'median_supply_ratio_min_crisis': median('supply_ratio_min_crisis'),
            'median_supply_ratio_end': median('supply_ratio_end'),
            'median_max_entry_streak': median('max_entry_streak'),
            'median_max_entry_progress': median('max_entry_progress'),
            'max_gross_event_identity_abs_error': (
                max(float(row.get('gross_event_identity_max_abs_error', 0.0))
                    for row in rows) if rows else None
            ),
            'max_claim_supply_identity_end_abs_error': (
                max(float(row.get('claim_supply_identity_end_abs_error', 0.0))
                    for row in rows) if rows else None
            ),
            'median_min_reserve_ratio_crisis': median('x_ratio_min_crisis'),
            'median_calm_return_over_option': median('rho_over_option'),
        }
    return {
        'n_seed_records': len(recs),
        'configuration': dict(config),
        'pools': pools,
        'interpretation': (
            'Survival and participation robustness only; not profitability '
            'or welfare, and not an externally calibrated LP-behaviour target. '
            'Entry non-activation is reported as right-censored whenever the '
            'simulation ends with providers still at risk; it is not silently '
            'interpreted as proof that entry cannot occur.'
        ),
    }


_SURVIVAL_REPORT_SCHEMA_VERSION = 2
_SURVIVAL_REPORT_SIGNATURE = None


def _survival_json_payload(recs, config, measurement_sig, report_sig,
                           seed_start, wanted, invalid):
    invalid = int(invalid)
    seed_complete = len(recs) == len(wanted)
    raw_provenance_clean = invalid == 0
    return {
        'model_signature': model_signature(ROOT),
        'measurement_signature': measurement_sig,
        'report_signature': report_sig,
        'report_schema_version': _SURVIVAL_REPORT_SCHEMA_VERSION,
        'requested_seed_start': int(seed_start),
        'requested_seeds': len(wanted),
        'requested_seed_list': list(wanted),
        'available_seeds': len(recs),
        'invalid_or_stale_records_ignored': invalid,
        'raw_provenance_clean': raw_provenance_clean,
        'complete': bool(seed_complete and raw_provenance_clean),
        'summary': summarize(recs, config),
    }


def survival_report_signature():
    """Digest of the aggregation applied to current raw survival records."""
    global _SURVIVAL_REPORT_SIGNATURE
    if _SURVIVAL_REPORT_SIGNATURE is None:
        _SURVIVAL_REPORT_SIGNATURE = measurement_signature(
            'lp_survival_report',
            survival_signature(),
            functions=(
                _raw_path,
                _raw_record_is_canonical,
                load_raw,
                append_raw,
                summarize,
                _survival_json_payload,
            ),
            constants=(
                'survival-summary-v2', _SURVIVAL_REPORT_SCHEMA_VERSION,
            ),
        )
    return _SURVIVAL_REPORT_SIGNATURE


def report(recs, config, out=None):
    """What the seeds say, in the order a reader would ask."""
    lines = []

    def w(s=''):
        lines.append(s)

    w(f'Model signature {model_signature(ROOT)}.')
    w(f'Survival measurement signature {survival_signature()}.')
    w('Endogenous provider population, dealer liquidity crisis preset.')
    w(f"Outside option {config['outside_option']:.4e} per tick, "
      f"subsidy {config['subsidy_rate']:.4e} per tick, loss rebate "
      f"{config['loss_rebate_fraction']:.1%}.")
    w(f"Exit/entry patience {config['exit_patience']}/{config['entry_patience']}, "
      f"kappa {config['kappa']:.4g}, response scale "
      f"{config['response_scale']:.4g}, max adjustment {config['max_adj']:.4g}, "
      f"EWMA alpha {config['ewma_alpha']:.4g}, entry margin "
      f"{config['entry_margin']:.4g}, n_iter {config['n_iter']}.")
    w(f'{len(recs)} seeds.')
    w()

    by_pool = {}
    for rec in recs:
        for p in rec['pools']:
            by_pool.setdefault(p['pool'], []).append(p)

    head = (f'  {"pool":6s}{"open at shock":>15s}{"open through":>14s}'
            f'{"exits in":>10s}{"min x/x0":>10s}{"rho/option":>12s}')
    w(head)
    w(f'  {"":6s}{"":>15s}{"the crisis":>14s}{"crisis":>10s}{"":>10s}{"in calm":>12s}')
    for pool in sorted(by_pool):
        rows = by_pool[pool]
        n = len(rows)
        open_shock = sum(r['open_at_shock'] for r in rows) / n
        open_through = sum(r['open_through_crisis'] for r in rows) / n
        exits = float(np.median([r['exits_in_crisis'] for r in rows]))
        xmin = float(np.nanmedian([r['x_ratio_min_crisis'] for r in rows]))
        ratio = float(np.nanmedian([r['rho_over_option'] for r in rows]))
        w(f'  {pool:6s}{open_shock:>14.0%}{open_through:>14.0%}'
          f'{exits:>10.2f}{xmin:>10.3f}{ratio:>12,.0f}')
    w()

    for pool in sorted(by_pool):
        rows = by_pool[pool]
        closed_pre = [r for r in rows if not r['open_at_shock']]
        w(f'  {pool}')
        w(f'    seeds shut when the shock landed      {len(closed_pre)} of {len(rows)}')
        firsts = [r['first_close'] for r in rows if r['first_close'] is not None]
        if firsts:
            w(f'    seeds that closed at any point       {len(firsts)} of {len(rows)}, '
              f'median first closure at tick {int(np.median(firsts))}')
        else:
            w('    seeds that closed at any point       0')
        w(f'    median ticks closed in the crisis    '
          f'{float(np.median([r["ticks_closed_crisis"] for r in rows])):.0f} of '
          f'{CRISIS[1] - CRISIS[0]}')
        w(f'    median providers active before shock '
          f'{float(np.median([r["active_pre_shock"] for r in rows])):.1f}')
        w(f'    median providers active at the end   '
          f'{float(np.median([r["active_end"] for r in rows])):.1f}')
        w(f'    gross entry/re-entry/exit events     '
          f'{sum(int(r.get("new_entry_events_total", 0)) for r in rows)}/'
          f'{sum(int(r.get("reentry_events_total", 0)) for r in rows)}/'
          f'{sum(int(r.get("exit_events_total", 0)) for r in rows)}')
        event_failures = sum(
            r.get('gross_event_identity_holds') is not True for r in rows
        )
        claim_failures = sum(
            r.get('claim_supply_identity_end_holds') is not True for r in rows
        )
        w(f'    accounting identity failures        '
          f'{event_failures} event / {claim_failures} claim-supply')
        censored = sum(int(r.get('new_entry_right_censored_providers', 0))
                       for r in rows)
        risk = sum(int(r.get('new_entry_risk_set_providers', 0)) for r in rows)
        reentry_risk = sum(int(r.get('reentry_risk_set_episodes',
                                     r.get('exit_events_total', 0)))
                           for r in rows)
        reentry_censored = sum(int(r.get(
            'reentry_right_censored_episodes',
            max(0, int(r.get('exit_events_total', 0))
                - int(r.get('reentry_events_total', 0))),
        )) for r in rows)
        activation = (sum(int(r.get('new_entry_events_total', 0))
                          for r in rows)
                      + sum(int(r.get('reentry_events_total', 0))
                            for r in rows))
        status = ('observed' if activation else
                  ('right-censored at the horizon'
                   if censored or reentry_censored else 'no risk set'))
        w(f'    entry activation status              {status}; '
          f'{censored} of {risk} potential entrants and '
          f'{reentry_censored} of {reentry_risk} exit episodes censored')
        end_supply = [r.get('supply_ratio_end') for r in rows]
        end_supply = [float(value) for value in end_supply
                      if value is not None and math.isfinite(float(value))]
        if end_supply:
            w(f'    median supply / initial at the end  '
              f'{float(np.median(end_supply)):.3f}')
        w()

    text = '\n'.join(lines)
    print(text)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--seed-start', type=int, default=42)
    ap.add_argument('--outside-option', type=float, default=None,
                    help='per tick, default is funding_rate_scale')
    ap.add_argument('--subsidy-rate', type=float, default=None)
    ap.add_argument('--loss-rebate', type=float, default=None)
    ap.add_argument('--exit-patience', type=int, default=None)
    ap.add_argument('--entry-patience', type=int, default=None)
    ap.add_argument('--kappa', type=float, default=None)
    ap.add_argument('--response-scale', type=float, default=None)
    ap.add_argument('--max-adjustment', type=float, default=None)
    ap.add_argument('--ewma-alpha', type=float, default=None)
    ap.add_argument('--entry-margin', type=float, default=None)
    ap.add_argument('--n-iter', type=int, default=N_ITER)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=0,
                    help='stop after this many new seeds, for a capped call')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--json-out', default=None,
                    help='optional machine-readable survival summary')
    args = ap.parse_args(argv)

    config = _run_config(args)
    sig = survival_signature()
    path = _raw_path(config, sig)
    have, invalid = load_raw(path, sig, config)
    if invalid:
        print(f'{invalid} stored records failed the current raw schema or '
              f'cache identity and are ignored.')

    wanted = list(range(args.seed_start, args.seed_start + args.seeds))
    todo = [s for s in wanted if s not in have]
    if args.chunk:
        todo = todo[:args.chunk]

    if todo and not args.report_only:
        jobs = [(s, config) for s in todo]
        print(f'measuring {len(todo)} seeds, {len(have)} already stored')
        with open(path, 'a', encoding='utf-8') as fh:
            if args.workers > 1:
                with ProcPool(args.workers) as pool:
                    for rec in pool.imap_unordered(measure, jobs):
                        if not _raw_record_is_canonical(
                                rec, sig, config, model_signature(ROOT)):
                            raise RuntimeError(
                                'new survival measurement failed its raw schema'
                            )
                        fh.write(json.dumps(rec, allow_nan=False) + '\n')
                        fh.flush()
                        have[rec['seed']] = rec
            else:
                for job in jobs:
                    rec = measure(job)
                    if not _raw_record_is_canonical(
                            rec, sig, config, model_signature(ROOT)):
                        raise RuntimeError(
                            'new survival measurement failed its raw schema'
                        )
                    fh.write(json.dumps(rec, allow_nan=False) + '\n')
                    fh.flush()
                    have[rec['seed']] = rec

    # Re-read the append-only cache so an unexpected duplicate or malformed
    # concurrent write cannot be hidden by the in-memory rows just measured.
    have, invalid = load_raw(path, sig, config)
    recs = [have[s] for s in wanted if s in have]
    if not recs:
        print('no seeds measured yet')
        return 1
    complete = len(recs) == len(wanted) and invalid == 0
    if not complete:
        print(f'incomplete: {len(recs)} of {len(wanted)} requested seeds available')
    report(recs, config, args.out)
    if args.json_out:
        payload = _survival_json_payload(
            recs, config, sig, survival_report_signature(),
            args.seed_start, wanted, invalid,
        )
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
    # A deliberately capped chunk is a valid incremental run, but it is not a
    # completed acceptance result.  Returning two prevents automation from
    # publishing a partial panel as final while leaving the raw cache intact
    # for the next invocation.
    return 0 if complete else 2


if __name__ == '__main__':
    sys.exit(main())
