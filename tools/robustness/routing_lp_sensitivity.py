#!/usr/bin/env python3
"""Precommitted OAT sensitivity of routing, arbitrage and LP economics.

The design is deliberately not the full factorial of the eight three-/two-
level controls below.  It changes one parameter at a time around the primary
liquidity-aware routing specification:

* routing cost scale: 2, 4, 8 bps;
* maximum prior weight: 0, 0.18;
* perceived-cost noise: 0, 1.5, 3 bps;
* HFMM reserves: 0.5, 1, 2 times the primary reserve setting.
* arbitrageur quote and base wallets: 0.25, 1, 4 times pool reserves;
* arbitrage trade cap: 0.025, 0.05, 0.10 of pool base reserves;
* maximum one-tick correction: 5, 10, 20 per cent.

The centre point is measured once, producing sixteen specifications rather
than a full factorial.  The panel is descriptive robustness, not a calibration search: every
declared row and every requested seed must be present before ``complete`` can
be true.  Customer volume, customer trade count, active-tick allocation and
arbitrage are separate estimands.  AMM--CLOB basis and the arbitrageur's
prefunded native-asset budgets are reported explicitly because arbitrage is a
large share of AMM execution in the crisis state.  Provider operating result
nets capital flows at the common reference price, using the same accounting
identity as ``lp_pnl_corrected.py``.

Example::

    python3 tools/robustness/routing_lp_sensitivity.py --seeds 60 --workers 4
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from multiprocessing import Pool as ProcPool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim)
from tools.robustness.lp_pnl_corrected import (
    CALM, CRISIS, N_ITER, PRESET,
    _series as pnl_series,
    components as pnl_components,
)
from tools.robustness.signatures import measurement_signature, model_signature


RAW_DIR = os.path.join(ROOT, 'output', 'resilience',
                       'raw_routing_lp_sensitivity')

BASELINE = {
    'routing_cost_scale_bps': 4.0,
    'routing_prior_mix_cap': 0.18,
    'cost_noise_std': 1.5,
    'hfmm_reserve_factor': 1.0,
    # These are the actual dealer_liquidity_crisis values after parser
    # defaults, the calibrated runtime and the preset have all been applied.
    # In particular the preset overrides the generic 0.20 trade cap to 0.05.
    'amm_arb_cash_buffer_ratio': 1.0,
    'amm_arb_base_buffer_ratio': 1.0,
    'arb_trade_fraction_cap': 0.05,
    'arb_max_correction_pct': 10.0,
}

PRECOMMIT_PROTOCOL = {
    'protocol_version': 'routing-lp-arb-oat-v2',
    'design': 'one_parameter_at_a_time_around_actual_primary_runtime_baseline',
    'not_full_factorial': True,
    'scenario': PRESET,
    'default_n_iter': N_ITER,
    'phase_windows_relative_to_shock': {
        'calm': list(CALM),
        'crisis': list(CRISIS),
    },
    'baseline': dict(BASELINE),
    'declared_levels': {
        'routing_cost_scale_bps': [2.0, 4.0, 8.0],
        'routing_prior_mix_cap': [0.0, 0.18],
        'cost_noise_std': [0.0, 1.5, 3.0],
        'hfmm_reserve_factor': [0.5, 1.0, 2.0],
        'amm_arb_cash_buffer_ratio': [0.25, 1.0, 4.0],
        'amm_arb_base_buffer_ratio': [0.25, 1.0, 4.0],
        'arb_trade_fraction_cap': [0.025, 0.05, 0.10],
        'arb_max_correction_pct': [5.0, 10.0, 20.0],
    },
    'expected_unique_specifications': 16,
    'seed_policy': (
        'all consecutive seeds requested on the command line; no seed may be '
        'dropped from a completed panel'
    ),
    'parameter_selection_forbidden': True,
    'manuscript_text_out_of_scope': True,
}

PHASES = ('calm', 'crisis')

PHASE_METRICS = (
    'amm_customer_volume_share',
    'amm_customer_trade_share',
    'amm_active_tick_flow_share',
    'customer_amm_volume_base',
    'customer_total_volume_base',
    'amm_arbitrage_volume_base',
    'amm_execution_volume_base',
    'amm_arbitrage_share_of_amm_execution',
    'amm_clob_abs_basis_mean_bps',
    'amm_clob_abs_basis_time_p95_bps',
    'arb_alignment_attempt_count',
    'arb_execution_count',
    'arb_cash_budget_binding_count',
    'arb_base_budget_binding_count',
    'arb_trade_fraction_binding_count',
    'arb_correction_binding_count',
    'arb_cash_exhausted_attempt_count',
    'arb_base_exhausted_attempt_count',
    'lp_operating_result_pct',
    'lp_loss_pct',
    'lp_fee_pct',
    'lp_opening_capital_quote',
    'lp_operating_result_measurable',
    'lp_available_tick_share',
    'lp_active_provider_mean',
)

RUN_METRICS = (
    'lp_population_count',
    'lp_open_at_shock_share',
    'lp_open_through_crisis_share',
    'arb_wallet_initial_cash_quote',
    'arb_wallet_final_cash_quote',
    'arb_wallet_initial_base',
    'arb_wallet_final_base',
    'arb_wallet_cash_remaining_share',
    'arb_wallet_base_remaining_share',
    'arb_wallet_cash_exhausted',
    'arb_wallet_base_exhausted',
    'arb_alignment_attempt_count',
    'arb_execution_count',
    'arb_cash_budget_binding_count',
    'arb_base_budget_binding_count',
    'arb_trade_fraction_binding_count',
    'arb_correction_binding_count',
    'arb_cash_exhausted_attempt_count',
    'arb_base_exhausted_attempt_count',
)

_RAW_SCHEMA_VERSION = 2

# The runtime cell is deliberately exact.  If any of these primary-market
# values changes, the model signature changes as well, but accepting the new
# value here still requires an explicit review and not silently moving the
# centre of a precommitted OAT.
_PRIMARY_AMM_SHARE_PCT = 22.0
_PRIMARY_HFMM_RESERVES_BASE = 3400.0

_ARB_EVENT_METRICS = (
    'arb_alignment_attempt_count',
    'arb_execution_count',
    'arb_cash_budget_binding_count',
    'arb_base_budget_binding_count',
    'arb_trade_fraction_binding_count',
    'arb_correction_binding_count',
    'arb_cash_exhausted_attempt_count',
    'arb_base_exhausted_attempt_count',
)


def _specifications(n_iter: int = N_ITER):
    """Return the frozen sixteen-row OAT design."""
    baseline = dict(BASELINE, n_iter=int(n_iter))
    rows = [('baseline', baseline)]

    def add(label, **change):
        config = dict(baseline)
        config.update(change)
        rows.append((label, config))

    add('routing_cost_scale_2bps', routing_cost_scale_bps=2.0)
    add('routing_cost_scale_8bps', routing_cost_scale_bps=8.0)
    add('routing_prior_mix_cap_0', routing_prior_mix_cap=0.0)
    add('cost_noise_0bps', cost_noise_std=0.0)
    add('cost_noise_3bps', cost_noise_std=3.0)
    add('hfmm_reserves_0.5x', hfmm_reserve_factor=0.5)
    add('hfmm_reserves_2x', hfmm_reserve_factor=2.0)
    add('arb_cash_buffer_0.25x', amm_arb_cash_buffer_ratio=0.25)
    add('arb_cash_buffer_4x', amm_arb_cash_buffer_ratio=4.0)
    add('arb_base_buffer_0.25x', amm_arb_base_buffer_ratio=0.25)
    add('arb_base_buffer_4x', amm_arb_base_buffer_ratio=4.0)
    add('arb_trade_fraction_0.025', arb_trade_fraction_cap=0.025)
    add('arb_trade_fraction_0.10', arb_trade_fraction_cap=0.10)
    add('arb_max_correction_5pct', arb_max_correction_pct=5.0)
    add('arb_max_correction_20pct', arb_max_correction_pct=20.0)
    _validate_specifications(rows)
    return rows


def _validate_specifications(rows) -> None:
    """Fail closed if the implemented design drifts from the precommitment."""
    expected_n = int(PRECOMMIT_PROTOCOL['expected_unique_specifications'])
    if len(rows) != expected_n:
        raise ValueError(f'expected {expected_n} OAT rows, found {len(rows)}')
    labels = [label for label, _ in rows]
    if len(set(labels)) != len(labels) or labels[0] != 'baseline':
        raise ValueError('OAT labels must be unique and start with baseline')

    baseline = rows[0][1]
    varied_fields = tuple(BASELINE)
    observed_levels = {field: set() for field in varied_fields}
    for idx, (_label, config) in enumerate(rows):
        for field in varied_fields:
            observed_levels[field].add(float(config[field]))
        if idx:
            changed = [
                field for field in varied_fields
                if float(config[field]) != float(baseline[field])
            ]
            if len(changed) != 1:
                raise ValueError('every non-baseline row must change one field')

    for field, declared in PRECOMMIT_PROTOCOL['declared_levels'].items():
        if observed_levels[field] != {float(value) for value in declared}:
            raise ValueError(
                f'{field} levels {sorted(observed_levels[field])} do not '
                f'match precommitment {declared}'
            )


def protocol_signature() -> str:
    payload = json.dumps(
        PRECOMMIT_PROTOCOL, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


def _phase_bounds(shock: int, n_obs: int, phase: str):
    offsets = {'calm': CALM, 'crisis': CRISIS}[phase]
    lo = int(shock + offsets[0])
    hi = int(shock + offsets[1])
    if lo < 0 or hi <= lo or hi > int(n_obs):
        return None
    return lo, hi


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _assert_primary_runtime_baseline(args) -> dict:
    """Bind the OAT centre to the effective primary runtime, not defaults.

    A parser default, a calibrated value and a preset override can all name
    the same argument.  The experiment is precommitted around the value that
    actually reaches ``build_sim``.  Refuse to run if that centre drifts.
    """
    observed = {
        field: float(getattr(args, field))
        for field in BASELINE
        if field != 'hfmm_reserve_factor'
    }
    expected = {
        field: float(value)
        for field, value in BASELINE.items()
        if field != 'hfmm_reserve_factor'
    }
    mismatches = {
        field: (observed[field], expected[field])
        for field in expected
        if not math.isclose(
            observed[field], expected[field], rel_tol=0.0, abs_tol=1e-15
        )
    }
    if mismatches:
        raise ValueError(
            f'effective {PRESET} runtime no longer matches the frozen OAT '
            f'baseline: {mismatches}'
        )
    return dict(observed, hfmm_reserve_factor=1.0)


def _customer_and_arb_metrics(logger, lo: int, hi: int) -> dict:
    amm_customer = sum(
        logger.customer_volume(venue, lo, hi)
        for venue in logger.flow_volume if venue != 'clob'
    )
    total_customer = sum(
        logger.customer_volume(venue, lo, hi)
        for venue in logger.flow_volume
    )
    arb = logger.arbitrage_volume_total(start=lo, end=hi)
    return {
        'amm_customer_volume_share': _finite_or_none(
            logger.amm_customer_volume_share(lo, hi)
        ),
        'amm_customer_trade_share': _finite_or_none(
            logger.amm_customer_trade_share(lo, hi)
        ),
        'amm_active_tick_flow_share': _finite_or_none(
            logger.amm_active_tick_flow_share(lo, hi)
        ),
        'customer_amm_volume_base': float(amm_customer),
        'customer_total_volume_base': float(total_customer),
        'amm_arbitrage_volume_base': float(arb),
        'amm_execution_volume_base': float(amm_customer + arb),
        'amm_arbitrage_share_of_amm_execution': _finite_or_none(
            logger.arbitrage_share_of_amm_execution(lo, hi)
        ),
    }


def _basis_metrics(logger, lo: int, hi: int) -> dict:
    """Time-weighted absolute AMM--CLOB basis over one phase.

    The logger supplies the maximum absolute basis across AMM venues at each
    tick.  The mean and the 95th time percentile therefore diagnose both
    persistent and tail price-alignment failures without treating missing
    prices as zero basis.
    """
    series = np.asarray(logger.max_venue_basis_series(), dtype=float)
    values = series[int(lo):int(hi)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'amm_clob_abs_basis_mean_bps': None,
            'amm_clob_abs_basis_time_p95_bps': None,
        }
    return {
        'amm_clob_abs_basis_mean_bps': float(np.mean(values)),
        'amm_clob_abs_basis_time_p95_bps': float(
            np.percentile(values, 95.0)
        ),
    }


def _instrument_arbitrageur(arbitrageur) -> dict:
    """Attach non-invasive capacity telemetry to one arbitrageur instance.

    The wrappers do not change prices, quantities, RNG calls or budgets.  They
    observe the limit already computed by the production methods and count a
    constraint as binding only when the executed quantity reaches that limit.
    Zero-wallet attempts are counted separately because they produce no trade.
    """
    telemetry = {
        'arb_wallet_initial_cash_quote': None,
        'arb_wallet_final_cash_quote': None,
        'arb_wallet_initial_base': None,
        'arb_wallet_final_base': None,
        'arb_wallet_cash_remaining_share': None,
        'arb_wallet_base_remaining_share': None,
        'arb_wallet_cash_exhausted': None,
        'arb_wallet_base_exhausted': None,
        'arb_alignment_attempt_count': 0.0,
        'arb_execution_count': 0.0,
        'arb_cash_budget_binding_count': 0.0,
        'arb_base_budget_binding_count': 0.0,
        'arb_trade_fraction_binding_count': 0.0,
        'arb_correction_binding_count': 0.0,
        'arb_cash_exhausted_attempt_count': 0.0,
        'arb_base_exhausted_attempt_count': 0.0,
        '_event_ticks': {metric: [] for metric in _ARB_EVENT_METRICS},
    }
    if arbitrageur is None:
        return telemetry

    initial_cash = max(0.0, float(arbitrageur.cash))
    initial_base = max(0.0, float(arbitrageur.assets))
    telemetry['arb_wallet_initial_cash_quote'] = initial_cash
    telemetry['arb_wallet_initial_base'] = initial_base

    original_affordable = arbitrageur._max_affordable_buy_qty
    original_one_pool = arbitrageur._arb_one_pool
    affordability = {'requested': None, 'allowed': None}

    def record_event(metric):
        telemetry[metric] += 1.0
        tick = getattr(getattr(arbitrageur, 'env', None), '_t', None)
        # ``MarketEnvironment.step`` increments ``_t`` before the simulator
        # records logger index zero.  Convert to the logger's zero-based index
        # so these events slice against the exact same phase bounds as prices,
        # flow and LP accounting.
        telemetry['_event_ticks'][metric].append(
            max(0, int(tick) - 1) if tick is not None else None
        )

    def tracked_affordable(pool, requested):
        allowed = original_affordable(pool, requested)
        affordability['requested'] = float(requested)
        affordability['allowed'] = float(allowed)
        return allowed

    def tracked_one_pool(pool, reference_price):
        affordability['requested'] = None
        affordability['allowed'] = None
        direction = None
        trade_limit = None
        correction_limited = False
        eligible = False
        assets_before = max(0.0, float(arbitrageur.assets))
        try:
            if arbitrageur._reserve_healthy(pool):
                current = float(pool.mid_price())
                target = float(reference_price)
                if (current > 0.0 and target > 0.0
                        and math.isfinite(current) and math.isfinite(target)):
                    deviation = (target - current) / current
                    correction_limited = (
                        abs(deviation) > float(arbitrageur.max_correction_pct)
                    )
                    capped = (
                        current * (1.0 + math.copysign(
                            float(arbitrageur.max_correction_pct), deviation
                        ))
                        if correction_limited else target
                    )
                    # Match the production pool's profitability gate exactly.
                    # A price difference inside twice the fee is not an
                    # alignment attempt: even an unconstrained arbitrageur
                    # would decline it.  Counting such a no-trade as wallet
                    # exhaustion would mechanically make capital look binding
                    # on almost every quiet tick.
                    fee = float(pool.fee)
                    outside_fee_band = (
                        math.isfinite(capped) and capped > 0.0
                        and math.isfinite(fee) and fee >= 0.0
                        and abs(current - capped) / capped >= 2.0 * fee
                    )
                    if outside_fee_band:
                        if current < capped:
                            direction = 'buy'
                            eligible = True
                        elif current > capped:
                            direction = 'sell'
                            eligible = True
                        trade_limit = max(
                            0.0,
                            float(pool.x)
                            * float(arbitrageur.trade_fraction_cap),
                        )
        except (AttributeError, TypeError, ValueError, ArithmeticError):
            eligible = False

        before_volume = float(arbitrageur.cumulative_traded_base)
        original_one_pool(pool, reference_price)
        executed = max(
            0.0,
            float(arbitrageur.cumulative_traded_base) - before_volume,
        )
        if not eligible or trade_limit is None:
            return

        record_event('arb_alignment_attempt_count')
        if correction_limited:
            record_event('arb_correction_binding_count')
        if executed > 1e-9:
            record_event('arb_execution_count')

        limit_kind = 'trade'
        applied_limit = trade_limit
        if direction == 'buy':
            requested = affordability['requested']
            allowed = affordability['allowed']
            if requested is not None and allowed is not None:
                applied_limit = max(0.0, allowed)
                if allowed < requested - max(1e-10, abs(requested) * 1e-10):
                    limit_kind = 'cash'
                if applied_limit <= 1e-9:
                    record_event('arb_cash_exhausted_attempt_count')
        elif direction == 'sell':
            applied_limit = min(trade_limit, assets_before)
            if assets_before < trade_limit - max(1e-10, abs(trade_limit) * 1e-10):
                limit_kind = 'base'
            if applied_limit <= 1e-9:
                record_event('arb_base_exhausted_attempt_count')

        tolerance = max(1e-8, abs(applied_limit) * 1e-8)
        if executed > 1e-9 and abs(executed - applied_limit) <= tolerance:
            key = {
                'cash': 'arb_cash_budget_binding_count',
                'base': 'arb_base_budget_binding_count',
                'trade': 'arb_trade_fraction_binding_count',
            }[limit_kind]
            record_event(key)

    arbitrageur._max_affordable_buy_qty = tracked_affordable
    arbitrageur._arb_one_pool = tracked_one_pool
    return telemetry


def _finalize_arbitrage_telemetry(arbitrageur, telemetry: dict) -> dict:
    """Add terminal native-wallet levels and scale-free exhaustion fields."""
    if arbitrageur is None:
        return telemetry
    initial_cash = float(telemetry['arb_wallet_initial_cash_quote'])
    initial_base = float(telemetry['arb_wallet_initial_base'])
    final_cash = max(0.0, float(arbitrageur.cash))
    final_base = max(0.0, float(arbitrageur.assets))
    cash_eps = max(1e-9, initial_cash * 1e-12)
    base_eps = max(1e-9, initial_base * 1e-12)
    telemetry.update({
        'arb_wallet_final_cash_quote': final_cash,
        'arb_wallet_final_base': final_base,
        'arb_wallet_cash_remaining_share': (
            final_cash / initial_cash if initial_cash > 0.0 else None
        ),
        'arb_wallet_base_remaining_share': (
            final_base / initial_base if initial_base > 0.0 else None
        ),
        'arb_wallet_cash_exhausted': float(final_cash <= cash_eps),
        'arb_wallet_base_exhausted': float(final_base <= base_eps),
    })
    return telemetry


def _arbitrage_phase_metrics(telemetry: dict, lo: int, hi: int) -> dict:
    """Count capacity events whose simulator tick lies in ``[lo, hi)``."""
    ticks_by_metric = telemetry.get('_event_ticks', {}) or {}
    return {
        metric: float(sum(
            tick is not None and int(lo) <= int(tick) < int(hi)
            for tick in ticks_by_metric.get(metric, [])
        ))
        for metric in _ARB_EVENT_METRICS
    }


def _lp_operating_metrics(sim, shock: int, phase: str) -> dict:
    """Capital-flow-neutral aggregate LP result for one phase."""
    price = np.asarray(sim.logger.fair_price_series, dtype=float)
    window = {'calm': CALM, 'crisis': CRISIS}[phase]
    aggregate_loss = 0.0
    aggregate_fees = 0.0
    aggregate_capital = 0.0
    complete = bool(sim.amm_pools)
    for pool in sim.amm_pools.values():
        row = pnl_components(pool, price, shock, window)
        if row is None:
            complete = False
            continue
        loss, fees, capital = row
        aggregate_loss += float(loss)
        aggregate_fees += float(fees)
        aggregate_capital += float(capital)

    measurable = bool(complete and aggregate_capital > 0.0)
    if not measurable:
        return {
            'lp_operating_result_pct': None,
            'lp_loss_pct': None,
            'lp_fee_pct': None,
            'lp_opening_capital_quote': None,
            'lp_operating_result_measurable': 0.0,
        }
    return {
        'lp_operating_result_pct': (
            100.0 * (aggregate_fees - aggregate_loss) / aggregate_capital
        ),
        'lp_loss_pct': 100.0 * aggregate_loss / aggregate_capital,
        'lp_fee_pct': 100.0 * aggregate_fees / aggregate_capital,
        'lp_opening_capital_quote': aggregate_capital,
        'lp_operating_result_measurable': 1.0,
    }


def _lp_availability_metrics(sim, shock: int, lo: int, hi: int) -> dict:
    """Availability across provider populations and ticks, without look-ahead."""
    populations = list(getattr(sim, 'lp_providers', []) or [])
    tick_available = []
    active_counts = []
    open_at_shock = []
    open_through_crisis = []
    crisis_bounds = _phase_bounds(shock, len(sim.logger.iterations), 'crisis')
    for population in populations:
        history = getattr(population, 'history', {}) or {}
        closed = np.asarray(history.get('closed', []), dtype=float)
        active = np.asarray(history.get('active', []), dtype=float)
        if closed.size < hi:
            continue
        tick_available.extend((closed[lo:hi] <= 0.0).astype(float).tolist())
        if active.size >= hi:
            active_counts.extend(active[lo:hi].tolist())
        pre_index = max(0, int(shock) - 1)
        if closed.size > pre_index:
            open_at_shock.append(float(closed[pre_index] <= 0.0))
        if crisis_bounds is not None and closed.size >= crisis_bounds[1]:
            c0, c1 = crisis_bounds
            open_through_crisis.append(float(
                closed[pre_index] <= 0.0 and np.all(closed[c0:c1] <= 0.0)
            ))
    return {
        'lp_available_tick_share': (
            float(np.mean(tick_available)) if tick_available else None
        ),
        'lp_active_provider_mean': (
            float(np.mean(active_counts)) if active_counts else None
        ),
        'lp_population_count': float(len(populations)),
        'lp_open_at_shock_share': (
            float(np.mean(open_at_shock)) if open_at_shock else None
        ),
        'lp_open_through_crisis_share': (
            float(np.mean(open_through_crisis))
            if open_through_crisis else None
        ),
    }


def _simulate(seed: int, config: dict):
    argv = [
        '--preset', PRESET,
        '--seed', str(seed),
        '--n-iter', str(config['n_iter']),
        '--silent',
        '--amm-lp-model', 'endogenous',
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_preset_defaults(parser, args)
    args.venue_choice_rule = _resolve_main_routing(args, argv)
    _auto_stress_around_shock(args)
    primary_runtime = _assert_primary_runtime_baseline(args)
    args.enable_amm = 1
    args.clob_amm_interaction = 'competition'
    args.routing_cost_scale_bps = float(config['routing_cost_scale_bps'])
    args.routing_prior_mix_cap = float(config['routing_prior_mix_cap'])
    args.cost_noise_std = float(config['cost_noise_std'])
    baseline_hfmm_reserves = float(args.hfmm_reserves)
    args.hfmm_reserves = (
        baseline_hfmm_reserves * float(config['hfmm_reserve_factor'])
    )
    args.amm_arb_cash_buffer_ratio = float(
        config['amm_arb_cash_buffer_ratio']
    )
    args.amm_arb_base_buffer_ratio = float(
        config['amm_arb_base_buffer_ratio']
    )
    args.arb_trade_fraction_cap = float(config['arb_trade_fraction_cap'])
    args.arb_max_correction_pct = float(config['arb_max_correction_pct'])
    _seed_all(seed)
    sim = build_sim(args)
    arb_telemetry = _instrument_arbitrageur(sim.arbitrageur)
    sim.simulate(args.n_iter, silent=True)
    _finalize_arbitrage_telemetry(sim.arbitrageur, arb_telemetry)
    runtime = {
        'venue_choice_rule': args.venue_choice_rule,
        'amm_share_pct': float(args.amm_share_pct),
        'routing_cost_scale_bps': float(args.routing_cost_scale_bps),
        'routing_prior_mix_cap': float(args.routing_prior_mix_cap),
        'cost_noise_std': float(args.cost_noise_std),
        'hfmm_reserves_primary_base': baseline_hfmm_reserves,
        'hfmm_reserves_run': float(args.hfmm_reserves),
        'hfmm_reserve_factor': float(config['hfmm_reserve_factor']),
        'amm_arb_cash_buffer_ratio': float(args.amm_arb_cash_buffer_ratio),
        'amm_arb_base_buffer_ratio': float(args.amm_arb_base_buffer_ratio),
        'arb_trade_fraction_cap': float(args.arb_trade_fraction_cap),
        'arb_max_correction_pct': float(args.arb_max_correction_pct),
        'primary_runtime_baseline': primary_runtime,
        'amm_lp_model': args.amm_lp_model,
        'amm_pool_names': sorted(sim.amm_pools),
    }
    return sim, int(args.shock_iter), runtime, arb_telemetry


def measure(job):
    """Measure one precommitted specification and seed."""
    seed, config = job
    sim, shock, runtime, arb_telemetry = _simulate(int(seed), config)
    record = {
        'raw_schema_version': _RAW_SCHEMA_VERSION,
        'seed': int(seed),
        'configuration': dict(config),
        'runtime': runtime,
        'model_signature': model_signature(ROOT),
        'measurement_signature': sensitivity_signature(),
        'protocol_signature': protocol_signature(),
    }
    run_availability = None
    for phase in PHASES:
        bounds = _phase_bounds(shock, len(sim.logger.iterations), phase)
        if bounds is None:
            for metric in PHASE_METRICS:
                record[f'{metric}_{phase}'] = None
            continue
        lo, hi = bounds
        metrics = _customer_and_arb_metrics(sim.logger, lo, hi)
        metrics.update(_basis_metrics(sim.logger, lo, hi))
        metrics.update(_arbitrage_phase_metrics(arb_telemetry, lo, hi))
        metrics.update(_lp_operating_metrics(sim, shock, phase))
        availability = _lp_availability_metrics(sim, shock, lo, hi)
        metrics.update({
            'lp_available_tick_share': availability['lp_available_tick_share'],
            'lp_active_provider_mean': availability['lp_active_provider_mean'],
        })
        run_availability = availability
        for metric in PHASE_METRICS:
            record[f'{metric}_{phase}'] = metrics.get(metric)
    for metric in RUN_METRICS:
        if metric in arb_telemetry:
            record[metric] = arb_telemetry[metric]
        else:
            record[metric] = (
                run_availability.get(metric)
                if run_availability is not None else None
            )
    return record


_SENSITIVITY_SIGNATURE = None


def sensitivity_signature() -> str:
    global _SENSITIVITY_SIGNATURE
    if _SENSITIVITY_SIGNATURE is None:
        _SENSITIVITY_SIGNATURE = measurement_signature(
            'routing_lp_sensitivity',
            model_signature(ROOT),
            functions=(
                _specifications,
                _assert_primary_runtime_baseline,
                _phase_bounds,
                _customer_and_arb_metrics,
                _basis_metrics,
                _instrument_arbitrageur,
                _finalize_arbitrage_telemetry,
                _arbitrage_phase_metrics,
                _lp_operating_metrics,
                _lp_availability_metrics,
                _simulate,
                measure,
                pnl_series,
                pnl_components,
            ),
            constants=(
                protocol_signature(),
                PRESET, N_ITER, CALM, CRISIS,
                PHASES, PHASE_METRICS, RUN_METRICS, _ARB_EVENT_METRICS,
                _RAW_SCHEMA_VERSION,
            ),
        )
    return _SENSITIVITY_SIGNATURE


def _raw_path(config: dict, raw_dir: str = RAW_DIR) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'))
    tag = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return os.path.join(raw_dir, f'routing_lp_{tag}.jsonl')

def _number_or_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _real_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_number(value) -> bool:
    return _real_number(value) and float(value) >= 0.0


def _unit_interval(value) -> bool:
    return _real_number(value) and 0.0 <= float(value) <= 1.0


def _event_count(value) -> bool:
    return (
        _nonnegative_number(value)
        and float(value).is_integer()
    )


def _close(lhs, rhs, *, atol=1e-9, rtol=1e-12) -> bool:
    return (
        _real_number(lhs) and _real_number(rhs)
        and math.isclose(float(lhs), float(rhs), rel_tol=rtol, abs_tol=atol)
    )


def _configuration_is_declared(config: dict) -> bool:
    expected_keys = set(BASELINE) | {'n_iter'}
    if not isinstance(config, dict) or set(config) != expected_keys:
        return False
    n_iter = config.get('n_iter')
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter <= 0:
        return False
    if not all(_real_number(config.get(field)) for field in BASELINE):
        return False
    return any(
        config == declared
        for _label, declared in _specifications(n_iter)
    )


def _expected_runtime_configuration(config: dict) -> dict:
    """Exact effective runtime identity for one declared OAT cell."""
    primary = {field: float(value) for field, value in BASELINE.items()}
    factor = float(config['hfmm_reserve_factor'])
    return {
        'venue_choice_rule': 'liquidity_aware',
        'amm_share_pct': _PRIMARY_AMM_SHARE_PCT,
        'routing_cost_scale_bps': float(config['routing_cost_scale_bps']),
        'routing_prior_mix_cap': float(config['routing_prior_mix_cap']),
        'cost_noise_std': float(config['cost_noise_std']),
        'hfmm_reserves_primary_base': _PRIMARY_HFMM_RESERVES_BASE,
        'hfmm_reserves_run': _PRIMARY_HFMM_RESERVES_BASE * factor,
        'hfmm_reserve_factor': factor,
        'amm_arb_cash_buffer_ratio': float(
            config['amm_arb_cash_buffer_ratio']
        ),
        'amm_arb_base_buffer_ratio': float(
            config['amm_arb_base_buffer_ratio']
        ),
        'arb_trade_fraction_cap': float(config['arb_trade_fraction_cap']),
        'arb_max_correction_pct': float(config['arb_max_correction_pct']),
        'primary_runtime_baseline': primary,
        'amm_lp_model': 'endogenous',
        'amm_pool_names': ['hfmm'],
    }


def _runtime_matches_config(runtime: dict, config: dict) -> bool:
    if not isinstance(runtime, dict) or not _configuration_is_declared(config):
        return False
    expected = _expected_runtime_configuration(config)
    if set(runtime) != set(expected):
        return False
    numeric_fields = {
        'amm_share_pct', 'routing_cost_scale_bps',
        'routing_prior_mix_cap', 'cost_noise_std',
        'hfmm_reserves_primary_base', 'hfmm_reserves_run',
        'hfmm_reserve_factor', 'amm_arb_cash_buffer_ratio',
        'amm_arb_base_buffer_ratio', 'arb_trade_fraction_cap',
        'arb_max_correction_pct',
    }
    for field in numeric_fields:
        if not _real_number(runtime.get(field)) or not _close(
            runtime[field], expected[field], atol=1e-12, rtol=0.0
        ):
            return False
    primary = runtime.get('primary_runtime_baseline')
    if not isinstance(primary, dict) or set(primary) != set(BASELINE):
        return False
    if not all(
        _real_number(primary.get(field))
        and _close(primary[field], BASELINE[field], atol=1e-12, rtol=0.0)
        for field in BASELINE
    ):
        return False
    return all(
        runtime.get(field) == expected[field]
        for field in ('venue_choice_rule', 'amm_lp_model', 'amm_pool_names')
    )


def _phase_record_semantics(record: dict, phase: str) -> bool:
    def value(metric):
        return record.get(f'{metric}_{phase}')

    customer_amm = value('customer_amm_volume_base')
    customer_total = value('customer_total_volume_base')
    arbitrage = value('amm_arbitrage_volume_base')
    execution = value('amm_execution_volume_base')
    volumes = (customer_amm, customer_total, arbitrage, execution)
    if not all(_nonnegative_number(item) for item in volumes):
        return False
    if float(customer_amm) > float(customer_total) + 1e-9:
        return False
    if not _close(execution, float(customer_amm) + float(arbitrage)):
        return False

    customer_shares = (
        value('amm_customer_volume_share'),
        value('amm_customer_trade_share'),
        value('amm_active_tick_flow_share'),
    )
    if float(customer_total) > 0.0:
        if not all(_unit_interval(item) for item in customer_shares):
            return False
        if not _close(
            customer_shares[0], float(customer_amm) / float(customer_total),
            atol=1e-12, rtol=1e-12,
        ):
            return False
    elif not (float(customer_amm) == 0.0
              and all(item is None for item in customer_shares)):
        return False

    arb_share = value('amm_arbitrage_share_of_amm_execution')
    if float(execution) > 0.0:
        if (not _unit_interval(arb_share)
                or not _close(
                    arb_share, float(arbitrage) / float(execution),
                    atol=1e-12, rtol=1e-12,
                )):
            return False
    elif not (float(arbitrage) == 0.0 and arb_share is None):
        return False

    if not all(_nonnegative_number(value(metric)) for metric in (
        'amm_clob_abs_basis_mean_bps',
        'amm_clob_abs_basis_time_p95_bps',
    )):
        return False

    events = {metric: value(metric) for metric in _ARB_EVENT_METRICS}
    if not all(_event_count(item) for item in events.values()):
        return False
    attempts = float(events['arb_alignment_attempt_count'])
    executions = float(events['arb_execution_count'])
    if executions > attempts:
        return False
    if float(events['arb_correction_binding_count']) > attempts:
        return False
    if sum(float(events[metric]) for metric in (
        'arb_cash_budget_binding_count', 'arb_base_budget_binding_count',
        'arb_trade_fraction_binding_count',
    )) > executions:
        return False
    if sum(float(events[metric]) for metric in (
        'arb_cash_exhausted_attempt_count',
        'arb_base_exhausted_attempt_count',
    )) > attempts:
        return False

    measurable = value('lp_operating_result_measurable')
    if not _event_count(measurable) or float(measurable) not in (0.0, 1.0):
        return False
    pnl_fields = (
        value('lp_operating_result_pct'), value('lp_loss_pct'),
        value('lp_fee_pct'), value('lp_opening_capital_quote'),
    )
    if float(measurable) == 1.0:
        net, loss, fee, opening = pnl_fields
        if (not all(_real_number(item) for item in pnl_fields)
                or float(fee) < 0.0 or float(opening) <= 0.0
                or not _close(net, float(fee) - float(loss), atol=1e-9)):
            return False
    elif any(item is not None for item in pnl_fields):
        return False

    return (
        _unit_interval(value('lp_available_tick_share'))
        and _nonnegative_number(value('lp_active_provider_mean'))
    )


def _run_record_semantics(record: dict) -> bool:
    population = record.get('lp_population_count')
    if not _event_count(population) or float(population) != 1.0:
        return False
    if not all(_unit_interval(record.get(field)) for field in (
        'lp_open_at_shock_share', 'lp_open_through_crisis_share',
    )):
        return False

    for asset in ('cash', 'base'):
        initial = record.get(f'arb_wallet_initial_{asset}' + (
            '_quote' if asset == 'cash' else ''
        ))
        final = record.get(f'arb_wallet_final_{asset}' + (
            '_quote' if asset == 'cash' else ''
        ))
        remaining = record.get(f'arb_wallet_{asset}_remaining_share')
        exhausted = record.get(f'arb_wallet_{asset}_exhausted')
        if (not _real_number(initial) or float(initial) <= 0.0
                or not _nonnegative_number(final)
                or not _nonnegative_number(remaining)
                or not _close(remaining, float(final) / float(initial))
                or not _event_count(exhausted)
                or float(exhausted) not in (0.0, 1.0)):
            return False
        epsilon = max(1e-9, float(initial) * 1e-12)
        if float(exhausted) != float(float(final) <= epsilon):
            return False

    totals = {metric: record.get(metric) for metric in _ARB_EVENT_METRICS}
    if not all(_event_count(item) for item in totals.values()):
        return False
    attempts = float(totals['arb_alignment_attempt_count'])
    executions = float(totals['arb_execution_count'])
    if executions > attempts:
        return False
    if float(totals['arb_correction_binding_count']) > attempts:
        return False
    if sum(float(totals[metric]) for metric in (
        'arb_cash_budget_binding_count', 'arb_base_budget_binding_count',
        'arb_trade_fraction_binding_count',
    )) > executions:
        return False
    if sum(float(totals[metric]) for metric in (
        'arb_cash_exhausted_attempt_count',
        'arb_base_exhausted_attempt_count',
    )) > attempts:
        return False

    for metric in _ARB_EVENT_METRICS:
        phase_sum = sum(float(record[f'{metric}_{phase}']) for phase in PHASES)
        if phase_sum > float(totals[metric]):
            return False
    return True


def _raw_record_matches(record: dict, config: dict, signature: str) -> bool:
    """Validate the complete identity and finite-number schema of one row."""
    if not isinstance(record, dict):
        return False
    seed = record.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        return False
    expected_fields = {
        'raw_schema_version', 'seed', 'configuration', 'runtime',
        'model_signature', 'measurement_signature', 'protocol_signature',
        *(f'{metric}_{phase}'
          for phase in PHASES for metric in PHASE_METRICS),
        *RUN_METRICS,
    }
    if (
        set(record) != expected_fields
        or not _configuration_is_declared(config)
        or record.get('raw_schema_version') != _RAW_SCHEMA_VERSION
        or record.get('model_signature') != model_signature(ROOT)
        or record.get('measurement_signature') != signature
        or record.get('protocol_signature') != protocol_signature()
        or record.get('configuration') != config
        or not _runtime_matches_config(record.get('runtime'), config)
    ):
        return False
    return (
        all(_phase_record_semantics(record, phase) for phase in PHASES)
        and _run_record_semantics(record)
    )


def load_raw(config: dict, signature: str, raw_dir: str = RAW_DIR):
    """Load valid records; duplicate current rows invalidate their seed.

    Later-wins semantics make the selected result depend on append order and
    can conceal two concurrent writers or nondeterministic reruns.  Once a
    seed occurs twice in the same current cell, omit it until the cache is
    repaired, even if the duplicate payloads happen to be identical.
    """
    have, stale = {}, 0
    duplicate_seeds = set()
    path = _raw_path(config, raw_dir)
    if not os.path.exists(path):
        return have, stale
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                stale += 1
                continue
            if not _raw_record_matches(record, config, signature):
                stale += 1
                continue
            seed = int(record['seed'])
            if seed in have or seed in duplicate_seeds:
                stale += 1
                have.pop(seed, None)
                duplicate_seeds.add(seed)
                continue
            have[seed] = record
    return have, stale


def append_raw(record: dict, raw_dir: str = RAW_DIR) -> None:
    config = record.get('configuration') if isinstance(record, dict) else None
    if not isinstance(config, dict) or not _raw_record_matches(
        record, config, sensitivity_signature()
    ):
        raise ValueError('refusing malformed or wrongly identified routing row')
    path = _raw_path(record['configuration'], raw_dir)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')


def _describe(values) -> dict:
    finite = np.asarray([
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ], dtype=float)
    if finite.size == 0:
        return {
            'n_finite': 0,
            'mean': None,
            'median': None,
            'p10': None,
            'p90': None,
        }
    return {
        'n_finite': int(finite.size),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'p10': float(np.percentile(finite, 10.0)),
        'p90': float(np.percentile(finite, 90.0)),
    }


def _describe_arbitrage_share(records, field: str, numerator_field: str,
                              denominator_field: str) -> dict:
    """Describe the arbitrage ratio without inventing zero-volume shares.

    ``None`` is the correct estimand when AMM execution volume is zero.  The
    explicit counters let downstream gates distinguish that identified
    zero-denominator case from a missing or otherwise invalid observation.
    The ratio is reconciled to its reported numerator and denominator for
    every observation.  This prevents a reporting layer from silently
    replacing an undefined ratio by zero or emitting a numerically plausible
    ratio backed by inconsistent volume totals.
    """
    finite = []
    n_zero_denominator = 0
    n_invalid = 0
    n_observations = 0
    for record in records:
        n_observations += 1
        value = record.get(field)
        numerator = record.get(numerator_field)
        denominator = record.get(denominator_field)
        try:
            numerator = float(numerator)
            denominator = float(denominator)
        except (TypeError, ValueError):
            n_invalid += 1
            continue

        volumes_valid = (
            math.isfinite(numerator)
            and math.isfinite(denominator)
            and numerator >= 0.0
            and denominator >= 0.0
            and numerator <= denominator
        )
        if not volumes_valid:
            n_invalid += 1
            continue

        value_is_finite = False
        if value is not None:
            try:
                value = float(value)
                value_is_finite = math.isfinite(value)
            except (TypeError, ValueError):
                value_is_finite = False

        if denominator > 0.0:
            expected = numerator / denominator
            if (value_is_finite and 0.0 <= value <= 1.0
                    and math.isclose(
                        value, expected, rel_tol=1e-12, abs_tol=1e-12
                    )):
                finite.append(value)
            else:
                n_invalid += 1
        elif numerator == 0.0 and value is None:
            n_zero_denominator += 1
        else:
            n_invalid += 1

    summary = _describe(finite)
    summary.update({
        'n_zero_denominator': int(n_zero_denominator),
        'n_invalid': int(n_invalid),
        'n_observations': int(n_observations),
    })
    return summary


def summarize_records(records) -> dict:
    fields = [
        f'{metric}_{phase}' for phase in PHASES for metric in PHASE_METRICS
    ] + list(RUN_METRICS)
    summary = {
        field: _describe(record.get(field) for record in records)
        for field in fields
    }
    for phase in PHASES:
        field = f'amm_arbitrage_share_of_amm_execution_{phase}'
        numerator_field = f'amm_arbitrage_volume_base_{phase}'
        denominator_field = f'amm_execution_volume_base_{phase}'
        summary[field] = _describe_arbitrage_share(
            records, field, numerator_field, denominator_field
        )
    return summary


def paired_deltas(records, baseline_records) -> dict:
    """Within-seed sensitivity deltas, specification minus baseline."""
    baseline = {int(record['seed']): record for record in baseline_records}
    fields = (
        'amm_customer_volume_share_calm',
        'amm_customer_volume_share_crisis',
        'amm_customer_trade_share_calm',
        'amm_customer_trade_share_crisis',
        'amm_active_tick_flow_share_calm',
        'amm_active_tick_flow_share_crisis',
        'amm_arbitrage_share_of_amm_execution_crisis',
        'amm_clob_abs_basis_mean_bps_calm',
        'amm_clob_abs_basis_mean_bps_crisis',
        'amm_clob_abs_basis_time_p95_bps_calm',
        'amm_clob_abs_basis_time_p95_bps_crisis',
        'arb_alignment_attempt_count_calm',
        'arb_alignment_attempt_count_crisis',
        'arb_execution_count_calm',
        'arb_execution_count_crisis',
        'arb_cash_budget_binding_count_calm',
        'arb_cash_budget_binding_count_crisis',
        'arb_base_budget_binding_count_calm',
        'arb_base_budget_binding_count_crisis',
        'arb_trade_fraction_binding_count_calm',
        'arb_trade_fraction_binding_count_crisis',
        'arb_correction_binding_count_calm',
        'arb_correction_binding_count_crisis',
        'arb_cash_exhausted_attempt_count_calm',
        'arb_cash_exhausted_attempt_count_crisis',
        'arb_base_exhausted_attempt_count_calm',
        'arb_base_exhausted_attempt_count_crisis',
        'lp_operating_result_pct_calm',
        'lp_operating_result_pct_crisis',
        'lp_available_tick_share_calm',
        'lp_available_tick_share_crisis',
        'lp_open_at_shock_share',
        'lp_open_through_crisis_share',
        'arb_wallet_cash_remaining_share',
        'arb_wallet_base_remaining_share',
        'arb_wallet_cash_exhausted',
        'arb_wallet_base_exhausted',
        'arb_cash_budget_binding_count',
        'arb_base_budget_binding_count',
        'arb_trade_fraction_binding_count',
        'arb_correction_binding_count',
    )
    out = {}
    for field in fields:
        deltas = []
        for record in records:
            other = baseline.get(int(record['seed']))
            if other is None:
                continue
            lhs, rhs = record.get(field), other.get(field)
            if lhs is None or rhs is None:
                continue
            lhs, rhs = float(lhs), float(rhs)
            if math.isfinite(lhs) and math.isfinite(rhs):
                deltas.append(lhs - rhs)
        out[field] = _describe(deltas)
    return out


def assemble_report_payload(*, specifications, loaded, missing_by_label,
                            stale_by_label, wanted, seed_start: int,
                            n_iter: int, measurement_digest: str,
                            report_digest: str) -> dict:
    """Pure final transformation from validated seed rows to the OAT report.

    Keeping this assembly in a named function makes every publication flag,
    seed-manifest field and summary transformation part of the report signature.
    The command-line wrapper is then only orchestration and file I/O.
    """
    baseline_records = loaded['baseline']
    results = []
    for label, config in specifications:
        records = loaded[label]
        results.append({
            'label': label,
            'configuration': config,
            'runtime_configuration': (
                records[0].get('runtime') if records else None
            ),
            'available_records': len(records),
            'missing_records': len(missing_by_label[label]),
            'stale_records_ignored': stale_by_label[label],
            'summary': summarize_records(records),
            'paired_delta_vs_baseline': paired_deltas(
                records, baseline_records
            ),
        })

    expected_records = len(specifications) * len(wanted)
    available_records = sum(len(records) for records in loaded.values())
    complete = all(not missing for missing in missing_by_label.values())
    protocol_conformant = int(n_iter) == int(N_ITER)
    raw_provenance_clean = all(
        int(count) == 0 for count in stale_by_label.values()
    )
    return {
        'model_signature': model_signature(ROOT),
        'measurement_signature': measurement_digest,
        'report_signature': report_digest,
        'protocol_signature': protocol_signature(),
        'protocol': PRECOMMIT_PROTOCOL,
        'requested_seed_start': int(seed_start),
        'requested_seeds_per_specification': len(wanted),
        'requested_seeds': list(wanted),
        'n_iter': int(n_iter),
        'expected_specifications': len(specifications),
        'expected_records': expected_records,
        'available_records': available_records,
        'complete': bool(complete),
        'raw_provenance_clean': bool(raw_provenance_clean),
        'precommit_protocol_conformant': bool(protocol_conformant),
        'publication_ready': bool(
            complete and protocol_conformant and raw_provenance_clean
        ),
        'missing_seeds_by_specification': missing_by_label,
        'results': results,
    }


_REPORT_SIGNATURE = None


def report_signature() -> str:
    """Digest of the transformation from valid seed records to the panel."""
    global _REPORT_SIGNATURE
    if _REPORT_SIGNATURE is None:
        _REPORT_SIGNATURE = measurement_signature(
            'routing_lp_sensitivity_report',
            sensitivity_signature(),
            functions=(
                _number_or_none,
                _runtime_matches_config,
                _raw_record_matches,
                load_raw,
                _describe,
                _describe_arbitrage_share,
                summarize_records,
                paired_deltas,
                assemble_report_payload,
            ),
            constants=(
                protocol_signature(), PHASES, PHASE_METRICS, RUN_METRICS,
                _ARB_EVENT_METRICS, _RAW_SCHEMA_VERSION,
            ),
        )
    return _REPORT_SIGNATURE


def _measure_job(job):
    return measure(job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=60)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--n-iter', type=int, default=N_ITER)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=0,
                        help='measure at most this many missing records, then stop')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--raw-dir', default=RAW_DIR)
    parser.add_argument(
        '--output', default='output/resilience/routing_lp_sensitivity_60.json'
    )
    args = parser.parse_args(argv)
    if args.seeds <= 0 or args.n_iter <= 0 or args.workers <= 0:
        parser.error('seeds, n-iter and workers must be positive')

    specifications = _specifications(args.n_iter)
    signature = sensitivity_signature()
    wanted = list(range(args.seed_start, args.seed_start + args.seeds))
    pending = []
    stale_by_label = {}
    for label, config in specifications:
        have, stale = load_raw(config, signature, args.raw_dir)
        stale_by_label[label] = stale
        if not args.report_only:
            pending.extend(
                (seed, config) for seed in wanted if seed not in have
            )
    if args.chunk > 0:
        pending = pending[:args.chunk]

    if pending:
        if args.workers > 1:
            with ProcPool(args.workers) as pool:
                for record in pool.imap_unordered(_measure_job, pending):
                    append_raw(record, args.raw_dir)
        else:
            for job in pending:
                append_raw(measure(job), args.raw_dir)

    loaded = {}
    missing_by_label = {}
    for label, config in specifications:
        have, stale = load_raw(config, signature, args.raw_dir)
        records = [have[seed] for seed in wanted if seed in have]
        loaded[label] = records
        missing = [seed for seed in wanted if seed not in have]
        missing_by_label[label] = missing
        stale_by_label[label] = stale

    payload = assemble_report_payload(
        specifications=specifications,
        loaded=loaded,
        missing_by_label=missing_by_label,
        stale_by_label=stale_by_label,
        wanted=wanted,
        seed_start=args.seed_start,
        n_iter=args.n_iter,
        measurement_digest=signature,
        report_digest=report_signature(),
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({
        'model_signature': payload['model_signature'],
        'measurement_signature': signature,
        'report_signature': payload['report_signature'],
        'protocol_signature': payload['protocol_signature'],
        'specifications': len(specifications),
        'expected_records': payload['expected_records'],
        'available_records': payload['available_records'],
        'complete': payload['complete'],
        'raw_provenance_clean': payload['raw_provenance_clean'],
        'precommit_protocol_conformant': (
            payload['precommit_protocol_conformant']
        ),
        'publication_ready': payload['publication_ready'],
        'output': args.output,
    }, indent=2))
    # Completeness alone is not a publication result when the horizon departs
    # from the precommitted protocol.  Keep the artifact for diagnosis, but make
    # automation fail closed in the same way as the acceptance validator.
    return 0 if payload['publication_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
