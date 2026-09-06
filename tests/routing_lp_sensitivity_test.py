import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from AgentBasedModel.agents.agents import AMMArbitrageur
from AgentBasedModel.metrics.logger import MetricsLogger
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
from tools.robustness import routing_lp_sensitivity as R


def test_precommitted_design_is_sixteen_actual_centered_oat_rows():
    rows = R._specifications(1000)
    assert len(rows) == 16
    assert rows[0][0] == 'baseline'
    assert len({label for label, _ in rows}) == 16
    assert rows[0][1]['arb_trade_fraction_cap'] == 0.05
    assert rows[0][1]['arb_max_correction_pct'] == 10.0
    assert rows[0][1]['amm_arb_cash_buffer_ratio'] == 1.0
    assert rows[0][1]['amm_arb_base_buffer_ratio'] == 1.0

    baseline = rows[0][1]
    varied = tuple(R.BASELINE)
    for _label, config in rows[1:]:
        changed = [field for field in varied if config[field] != baseline[field]]
        assert len(changed) == 1

    for field, declared in R.PRECOMMIT_PROTOCOL['declared_levels'].items():
        observed = {float(config[field]) for _label, config in rows}
        assert observed == {float(value) for value in declared}


def test_runtime_baseline_guard_fails_if_effective_preset_drifts():
    args = SimpleNamespace(**{
        field: value for field, value in R.BASELINE.items()
        if field != 'hfmm_reserve_factor'
    })
    observed = R._assert_primary_runtime_baseline(args)
    assert observed == R.BASELINE

    args.arb_trade_fraction_cap = 0.20
    with pytest.raises(ValueError, match='frozen OAT baseline'):
        R._assert_primary_runtime_baseline(args)


def test_runner_keeps_customer_active_tick_and_arbitrage_estimands_separate():
    logger = MetricsLogger()
    logger.iterations = [0, 1, 2]
    logger.flow_volume = {
        'clob': [1.0, 0.0, 90.0],
        'hfmm': [9.0, 0.0, 10.0],
    }
    logger.flow_count = {
        'clob': [1, 0, 4],
        'hfmm': [1, 0, 1],
    }
    logger.arbitrage_volume = {'hfmm': [0.0, 0.0, 81.0]}
    logger.arbitrage_count = {'hfmm': [0, 0, 1]}

    row = R._customer_and_arb_metrics(logger, 0, 3)
    assert math.isclose(row['amm_customer_volume_share'], 19.0 / 110.0)
    assert math.isclose(row['amm_customer_trade_share'], 2.0 / 7.0)
    assert math.isclose(row['amm_active_tick_flow_share'], 0.5)
    assert row['customer_amm_volume_base'] == 19.0
    assert row['customer_total_volume_base'] == 110.0
    assert row['amm_arbitrage_volume_base'] == 81.0
    assert row['amm_execution_volume_base'] == 100.0
    assert math.isclose(row['amm_arbitrage_share_of_amm_execution'], 0.81)


def test_zero_amm_execution_preserves_undefined_arbitrage_share():
    logger = MetricsLogger()
    logger.iterations = [0, 1]
    logger.flow_volume = {
        'clob': [10.0, 20.0],
        'hfmm': [0.0, 0.0],
    }
    logger.flow_count = {
        'clob': [1, 1],
        'hfmm': [0, 0],
    }
    logger.arbitrage_volume = {'hfmm': [0.0, 0.0]}
    logger.arbitrage_count = {'hfmm': [0, 0]}

    row = R._customer_and_arb_metrics(logger, 0, 2)
    assert row['amm_execution_volume_base'] == 0.0
    assert row['amm_arbitrage_share_of_amm_execution'] is None


def test_basis_metrics_report_time_mean_and_tail_without_nan_imputation():
    logger = MetricsLogger()
    logger.clob_mid_series = [100.0] * 6
    logger.amm_mid_series = {
        'hfmm': [100.0, 101.0, float('nan'), 103.0, 104.0, 105.0],
    }
    row = R._basis_metrics(logger, 1, 6)
    expected = np.asarray([100.0, 300.0, 400.0, 500.0])
    assert math.isclose(row['amm_clob_abs_basis_mean_bps'], expected.mean())
    assert math.isclose(
        row['amm_clob_abs_basis_time_p95_bps'],
        np.percentile(expected, 95.0),
    )


class _FixedCLOB:
    def __init__(self, price=100.0):
        self.price = float(price)

    def mid_price(self):
        return self.price

    def quoted_spread_bps(self):
        return 1.0


def test_arbitrage_capacity_telemetry_observes_limits_without_changing_trade():
    pool = CPMMPool(x=1_000.0, y=80_000.0, fee=0.0005)
    arb = AMMArbitrageur(
        _FixedCLOB(), {'cpmm': pool}, cash=100.0, assets=0.0,
        trade_fraction_cap=1.0, max_correction_pct=50.0,
    )
    before = (pool.x, pool.y, arb.cash, arb.assets)
    telemetry = R._instrument_arbitrageur(arb)
    arb.arbitrage()
    R._finalize_arbitrage_telemetry(arb, telemetry)

    assert (pool.x, pool.y, arb.cash, arb.assets) != before
    assert telemetry['arb_alignment_attempt_count'] == 1.0
    assert telemetry['arb_execution_count'] == 1.0
    assert telemetry['arb_cash_budget_binding_count'] == 1.0
    assert telemetry['arb_trade_fraction_binding_count'] == 0.0
    assert telemetry['arb_wallet_final_cash_quote'] >= 0.0
    assert telemetry['arb_wallet_cash_exhausted'] == 1.0

    sell_pool = CPMMPool(x=1_000.0, y=120_000.0, fee=0.0005)
    unfunded = AMMArbitrageur(
        _FixedCLOB(), {'cpmm': sell_pool}, cash=1_000.0, assets=0.0,
        trade_fraction_cap=1.0, max_correction_pct=50.0,
    )
    sell_telemetry = R._instrument_arbitrageur(unfunded)
    unfunded.arbitrage()
    R._finalize_arbitrage_telemetry(unfunded, sell_telemetry)
    assert sell_telemetry['arb_execution_count'] == 0.0
    assert sell_telemetry['arb_base_exhausted_attempt_count'] == 1.0
    assert sell_telemetry['arb_wallet_base_exhausted'] == 1.0


def test_in_fee_band_is_not_misclassified_as_wallet_exhaustion():
    # A one-basis-point gap is inside this pool's ten-basis-point round-trip
    # fee band.  Production declines it even with unlimited inventory, so a
    # zero base wallet is not the reason no trade occurs.
    pool = CPMMPool(x=1_000.0, y=100_010.0, fee=0.0005)
    arb = AMMArbitrageur(
        _FixedCLOB(), {'cpmm': pool}, cash=100_000.0, assets=0.0,
        trade_fraction_cap=1.0, max_correction_pct=50.0,
    )
    telemetry = R._instrument_arbitrageur(arb)
    arb.arbitrage()
    R._finalize_arbitrage_telemetry(arb, telemetry)

    assert arb.cumulative_traded_base == 0.0
    assert all(telemetry[metric] == 0.0 for metric in R._ARB_EVENT_METRICS)
    # Terminal inventory is still truthfully reported as empty; it is the
    # causal exhausted-attempt counter that must remain zero.
    assert telemetry['arb_wallet_base_exhausted'] == 1.0


def test_arbitrage_capacity_events_are_sliced_into_economic_phases():
    telemetry = {
        '_event_ticks': {
            metric: [] for metric in R._ARB_EVENT_METRICS
        }
    }
    telemetry['_event_ticks']['arb_alignment_attempt_count'] = [249, 250, 349, 350]
    telemetry['_event_ticks']['arb_execution_count'] = [250, 349]
    telemetry['_event_ticks']['arb_cash_budget_binding_count'] = [300, 700]

    calm = R._arbitrage_phase_metrics(telemetry, 250, 350)
    crisis = R._arbitrage_phase_metrics(telemetry, 350, 450)
    assert calm['arb_alignment_attempt_count'] == 2.0
    assert calm['arb_execution_count'] == 2.0
    assert calm['arb_cash_budget_binding_count'] == 1.0
    assert crisis['arb_alignment_attempt_count'] == 1.0
    assert crisis['arb_execution_count'] == 0.0
    assert crisis['arb_cash_budget_binding_count'] == 0.0


def test_arbitrage_share_summary_separates_zero_denominator_from_invalid():
    field = 'amm_arbitrage_share_of_amm_execution_calm'
    numerator = 'amm_arbitrage_volume_base_calm'
    denominator = 'amm_execution_volume_base_calm'
    clean = R._describe_arbitrage_share([
        {field: 0.75, numerator: 30.0, denominator: 40.0},
        {field: None, numerator: 0.0, denominator: 0.0},
    ], field, numerator, denominator)
    assert clean['n_finite'] == 1
    assert clean['n_zero_denominator'] == 1
    assert clean['n_invalid'] == 0
    assert clean['n_observations'] == 2
    assert clean['mean'] == 0.75

    invalid = R._describe_arbitrage_share([
        # A zero denominator must have a zero numerator and exactly None.
        {field: 0.0, numerator: 0.0, denominator: 0.0},
        {field: None, numerator: 1.0, denominator: 0.0},
        {field: float('nan'), numerator: 0.0, denominator: 0.0},
        # Positive denominators require a finite, reconciled ratio.
        {field: None, numerator: 1.0, denominator: 5.0},
        {field: float('nan'), numerator: 1.0, denominator: 5.0},
        {field: 1.1, numerator: 1.0, denominator: 5.0},
        {field: 0.75, numerator: 20.0, denominator: 40.0},
        # Both volume components must be finite, nonnegative and ordered.
        {field: 0.5, numerator: -1.0, denominator: 5.0},
        {field: 0.5, numerator: 1.0, denominator: -5.0},
        {field: 0.5, numerator: 6.0, denominator: 5.0},
        {field: 0.5, numerator: float('nan'), denominator: 5.0},
        {field: 0.5, numerator: 1.0, denominator: float('inf')},
        {field: None, numerator: 0.0, denominator: None},
    ], field, numerator, denominator)
    assert invalid['n_finite'] == 0
    assert invalid['n_zero_denominator'] == 0
    assert invalid['n_invalid'] == 13
    assert invalid['n_observations'] == 13


def test_lp_operating_result_nets_pure_capital_flows_to_zero():
    # The shock sits far enough into the run for the calm window to fit behind
    # it and the crisis window in front. Both windows are read from the module,
    # so a fixture that pins its own offsets stops testing the measurement the
    # moment either window moves; this one is placed against them.
    pool = HFMMPool(x=1000.0, y=100000.0, A=18.0, fee=0.0005)
    pool.record_state()
    shock = -R.CALM[0] + 40
    ticks = shock + R.CRISIS[1] + 40
    for tick in range(ticks):
        if tick % 2:
            pool.remove_liquidity(0.001)
        else:
            pool.add_liquidity(0.001)
        pool.record_state()
    sim = SimpleNamespace(
        amm_pools={'hfmm': pool},
        logger=SimpleNamespace(fair_price_series=np.full(ticks, 100.0)),
    )

    calm = R._lp_operating_metrics(sim, shock=shock, phase='calm')
    crisis = R._lp_operating_metrics(sim, shock=shock, phase='crisis')
    assert calm['lp_operating_result_measurable'] == 1.0
    assert crisis['lp_operating_result_measurable'] == 1.0
    assert abs(calm['lp_operating_result_pct']) < 1e-9
    assert abs(crisis['lp_operating_result_pct']) < 1e-9


def _valid_raw_record(config, seed=42):
    # Spelled out and not taken from the module, so that a change to the
    # recorded runtime identity has to be restated here before the cache test
    # goes green again.
    runtime = {
        'venue_choice_rule': 'liquidity_aware',
        'amm_share_pct': 22.0,
        'routing_cost_scale_bps': float(config['routing_cost_scale_bps']),
        'routing_prior_mix_cap': float(config['routing_prior_mix_cap']),
        'cost_noise_std': float(config['cost_noise_std']),
        'hfmm_reserves_primary_base': 3400.0,
        'hfmm_reserves_run': 3400.0 * float(config['hfmm_reserve_factor']),
        'hfmm_reserve_factor': float(config['hfmm_reserve_factor']),
        'amm_arb_cash_buffer_ratio': float(config['amm_arb_cash_buffer_ratio']),
        'amm_arb_base_buffer_ratio': float(config['amm_arb_base_buffer_ratio']),
        'arb_trade_fraction_cap': float(config['arb_trade_fraction_cap']),
        'arb_max_correction_pct': float(config['arb_max_correction_pct']),
        'primary_runtime_baseline': {
            field: float(value) for field, value in R.BASELINE.items()
        },
        'amm_lp_model': 'endogenous',
        'amm_pool_names': ['hfmm'],
    }
    record = {
        'raw_schema_version': R._RAW_SCHEMA_VERSION,
        'seed': int(seed),
        'configuration': dict(config),
        'runtime': runtime,
        'model_signature': R.model_signature(R.ROOT),
        'measurement_signature': R.sensitivity_signature(),
        'protocol_signature': R.protocol_signature(),
    }
    # A row of zeros is not a valid row: the loader checks the accounting
    # identities between volumes, shares, event counts and the provider's
    # result, and an all-zero record violates several of them at once.  The
    # fixture therefore carries one internally consistent measurement.
    phase_values = {
        'customer_total_volume_base': 1000.0,
        'customer_amm_volume_base': 200.0,
        'amm_customer_volume_share': 0.2,
        'amm_customer_trade_share': 0.25,
        'amm_active_tick_flow_share': 0.3,
        'amm_arbitrage_volume_base': 50.0,
        'amm_execution_volume_base': 250.0,
        'amm_arbitrage_share_of_amm_execution': 0.2,
        'amm_clob_abs_basis_mean_bps': 1.5,
        'amm_clob_abs_basis_time_p95_bps': 4.0,
        'arb_alignment_attempt_count': 10.0,
        'arb_execution_count': 6.0,
        'arb_cash_budget_binding_count': 1.0,
        'arb_base_budget_binding_count': 1.0,
        'arb_trade_fraction_binding_count': 1.0,
        'arb_correction_binding_count': 2.0,
        'arb_cash_exhausted_attempt_count': 1.0,
        'arb_base_exhausted_attempt_count': 1.0,
        'lp_operating_result_measurable': 1.0,
        'lp_fee_pct': 0.3,
        'lp_loss_pct': 0.1,
        'lp_operating_result_pct': 0.2,
        'lp_opening_capital_quote': 340000.0,
        'lp_available_tick_share': 0.95,
        'lp_active_provider_mean': 1.0,
    }
    assert set(phase_values) == set(R.PHASE_METRICS)
    for phase in R.PHASES:
        for metric, value in phase_values.items():
            record[f'{metric}_{phase}'] = value
    # Run totals hold the whole trajectory, so each event count is at least
    # the sum of its two phases.
    record.update({
        'lp_population_count': 1.0,
        'lp_open_at_shock_share': 1.0,
        'lp_open_through_crisis_share': 1.0,
        'arb_wallet_initial_cash_quote': 100000.0,
        'arb_wallet_final_cash_quote': 40000.0,
        'arb_wallet_cash_remaining_share': 0.4,
        'arb_wallet_cash_exhausted': 0.0,
        'arb_wallet_initial_base': 1000.0,
        'arb_wallet_final_base': 400.0,
        'arb_wallet_base_remaining_share': 0.4,
        'arb_wallet_base_exhausted': 0.0,
        'arb_alignment_attempt_count': 20.0,
        'arb_execution_count': 12.0,
        'arb_cash_budget_binding_count': 2.0,
        'arb_base_budget_binding_count': 2.0,
        'arb_trade_fraction_binding_count': 2.0,
        'arb_correction_binding_count': 4.0,
        'arb_cash_exhausted_attempt_count': 2.0,
        'arb_base_exhausted_attempt_count': 2.0,
    })
    assert set(R.RUN_METRICS) <= set(record)
    return record


def test_cache_rejects_stale_records_and_report_requires_every_seed(tmp_path):
    config = R._specifications(500)[0][1]
    signature = R.sensitivity_signature()
    record = _valid_raw_record(config)
    R.append_raw(record, str(tmp_path / 'raw'))
    stale = dict(record, seed=43, measurement_signature='stale')
    with open(R._raw_path(config, str(tmp_path / 'raw')), 'a') as handle:
        handle.write(json.dumps(stale) + '\n')

    have, ignored = R.load_raw(config, signature, str(tmp_path / 'raw'))
    assert sorted(have) == [42]
    assert ignored == 1

    with pytest.raises(ValueError, match='malformed'):
        R.append_raw({'seed': 44, 'configuration': config},
                     str(tmp_path / 'raw'))

    duplicate_dir = str(tmp_path / 'duplicate')
    R.append_raw(record, duplicate_dir)
    R.append_raw(record, duplicate_dir)
    duplicate_have, duplicate_invalid = R.load_raw(
        config, signature, duplicate_dir
    )
    assert duplicate_have == {}
    assert duplicate_invalid == 1

    output = tmp_path / 'report.json'
    status = R.main([
        '--seeds', '2',
        '--seed-start', '42',
        '--n-iter', '500',
        '--workers', '1',
        '--report-only',
        '--raw-dir', str(tmp_path / 'empty'),
        '--output', str(output),
    ])
    payload = json.loads(output.read_text())
    assert status == 2
    assert payload['complete'] is False
    assert payload['precommit_protocol_conformant'] is False
    assert payload['publication_ready'] is False
    assert payload['raw_provenance_clean'] is True
    specification_count = len(R._specifications(500))
    assert payload['expected_specifications'] == specification_count
    assert payload['report_signature'] == R.report_signature()
    assert payload['expected_records'] == specification_count * 2
    assert payload['available_records'] == 0
    assert all(
        len(missing) == 2
        for missing in payload['missing_seeds_by_specification'].values()
    )
