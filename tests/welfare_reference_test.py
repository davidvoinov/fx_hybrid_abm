import math

from AgentBasedModel.agents.agents import Trader
from tools.robustness.welfare_accounting import (
    ArmWindow,
    _common_reference_cost_bps,
    account_pair,
    aggregate,
    arm_window,
)
from AgentBasedModel.metrics.logger import MetricsLogger
from types import SimpleNamespace

from tools.robustness.lp_survival import _read_population, summarize
from tools.robustness.lp_sensitivity import _specifications
from tools.robustness import lp_pnl_corrected as pnl_tool
from tools.robustness import routing_lp_sensitivity as routing_tool
from tools.robustness import lp_survival as survival_tool
from tools.robustness import lp_sensitivity as sensitivity_tool
from tools.robustness import welfare_accounting as welfare_tool
from tools.robustness import lp_acceptance as acceptance_tool
from tools.robustness.lp_acceptance import (
    acceptance_code_sha256,
    validate as validate_lp_acceptance,
)
from tools.robustness.signatures import model_signature
from calibration import runner as calibration_runner
import copy
import hashlib
import json
import os
import pytest


def test_trade_record_exposes_one_all_in_execution_price():
    trader = object.__new__(Trader)
    trader.id = 1
    trader.type = 'test'
    clob = trader._make_trade_record(
        'clob', 'buy', 2.0,
        {'exec_price': 100.0, 'fee_bps': 5.0, 'cost_bps': 7.0,
         'requested_qty': 2.0},
        {},
    )
    amm = trader._make_trade_record(
        'hfmm', 'buy', 2.0,
        {'exec_price': 100.05, 'fee_bps': 5.0, 'cost_bps': 7.0,
         'requested_qty': 2.0},
        {},
    )

    assert abs(clob['all_in_exec_price'] - 100.05) < 1e-12
    assert abs(amm['all_in_exec_price'] - 100.05) < 1e-12
    assert clob['local_cost_bps'] == 7.0


def test_welfare_uses_common_pretrade_mid_for_buys_and_sells():
    assert abs(_common_reference_cost_bps(
        {'side': 'buy', 'all_in_exec_price': 101.0,
         'common_reference_price': 100.0}, 90.0
    ) - 100.0) < 1e-12
    assert abs(_common_reference_cost_bps(
        {'side': 'sell', 'all_in_exec_price': 99.0,
         'common_reference_price': 100.0}, 90.0
    ) - 100.0) < 1e-12


def test_price_improvement_is_not_silently_floored_at_zero():
    assert _common_reference_cost_bps(
        {'side': 'buy', 'all_in_exec_price': 99.5}, 100.0
    ) < 0.0


def test_subsidy_is_a_private_transfer_and_cancels_from_the_partial_account():
    off = ArmWindow(1000.0, 9.0)
    low = account_pair(
        ArmWindow(1000.0, 4.0, lp_opening_capital=2000.0,
                  lp_lvr=7.0, lp_fees=5.0, sponsor_transfer=3.0),
        off, annual_capital_rate=0.0,
    )
    high = account_pair(
        ArmWindow(1000.0, 4.0, lp_opening_capital=2000.0,
                  lp_lvr=7.0, lp_fees=5.0, sponsor_transfer=300.0),
        off, annual_capital_rate=0.0,
    )

    assert high['provider_private_result_after_subsidy_and_capital'] \
        - low['provider_private_result_after_subsidy_and_capital'] == 297.0
    assert high['sponsor_fiscal_result'] - low['sponsor_fiscal_result'] == -297.0
    assert high['provider_plus_sponsor_result'] == low['provider_plus_sponsor_result']
    assert high['matched_notional_user_benefit'] == low['matched_notional_user_benefit']
    assert low['subsidy_transfer_cancels'] and high['subsidy_transfer_cancels']
    summary = aggregate([{'accounting': low}, {'accounting': high}],
                        bootstrap_draws=0)
    assert summary['all_subsidy_transfers_cancel'] is True
    assert summary['subsidy_transfer_cancellation_failures'] == 0


def test_an_unmeasurable_provider_window_is_missing_and_not_zero():
    """A seed that carries no provider result must not vote in the median.

    A pool that had already wound down before the window opened has no
    capital whose return could be measured.  Entering that seed as an exact
    zero put a value into the middle of the distribution that no measurement
    supports, and with enough such seeds the median of a genuinely negative
    result is dragged to zero.
    """
    off = ArmWindow(1000.0, 9.0)
    measured = [
        account_pair(
            ArmWindow(1000.0, 4.0, lp_opening_capital=2000.0,
                      lp_lvr=lvr, lp_fees=0.0, sponsor_transfer=0.0),
            off, annual_capital_rate=0.0,
        )
        for lvr in (30.0, 40.0, 50.0)
    ]
    unmeasurable = account_pair(
        ArmWindow(1000.0, 4.0, lp_opening_capital=None,
                  lp_lvr=None, lp_fees=None, sponsor_transfer=0.0),
        off, annual_capital_rate=0.0,
    )

    assert unmeasurable['lp_window_measurable'] is False
    assert math.isnan(unmeasurable['lp_operating_result'])
    assert math.isnan(unmeasurable['provider_plus_sponsor_result'])
    assert math.isnan(unmeasurable['capital_opportunity_cost'])
    # An identity that could not be evaluated has not been broken.
    assert unmeasurable['subsidy_transfer_cancels'] is None
    assert all(row['subsidy_transfer_cancels'] for row in measured)

    rows = [{'accounting': row} for row in measured + [unmeasurable] * 3]
    summary = aggregate(rows, bootstrap_draws=0)

    assert summary['n_seeds'] == 6
    assert summary['unmeasurable_lp_windows'] == 3
    assert summary['finite_counts']['lp_operating_result'] == 3
    # The median of the three measured seeds, not a median dragged towards
    # zero by the three that measured nothing.
    assert summary['medians']['lp_operating_result'] == -40.0
    assert summary['subsidy_transfer_cancellation_failures'] == 0
    assert summary['all_subsidy_transfers_cancel'] is True


def test_welfare_excludes_arbitrage_from_customer_benefit_but_reports_its_flow():
    logger = MetricsLogger()
    logger.iterations = [0, 1, 2]
    logger.fair_price_series = [100.0, 100.0, 100.0]
    logger.trade_log = [
        {
            't': 1, 'execution_source': 'routed_customer', 'trader_type': 'Retail',
            'venue': 'hfmm', 'side': 'buy', 'quantity': 2.0,
            'all_in_exec_price': 101.0, 'common_reference_price': 100.0,
        },
        {
            # Contaminate the customer log on purpose. The source guard must
            # keep this much larger arbitrage leg out of the user estimand.
            't': 1, 'execution_source': 'arbitrage',
            'trader_type': 'AMMArbitrageur', 'venue': 'hfmm', 'side': 'buy',
            'quantity': 50.0, 'all_in_exec_price': 80.0,
            'common_reference_price': 100.0,
        },
    ]
    logger.flow_volume = {'clob': [1.0, 0.0, 90.0],
                          'hfmm': [9.0, 0.0, 10.0]}
    logger.arbitrage_volume = {'hfmm': [0.0, 0.0, 81.0]}
    logger.mm_channel_shares['endogenous'] = [0.0, 0.0, 0.0]
    sim = SimpleNamespace(logger=logger, amm_pools={}, lp_providers=[])

    arm = arm_window(sim, shock=0, window=(0, 3))
    assert arm.executed_notional == 200.0
    assert abs(arm.taker_execution_cost - 2.0) < 1e-12
    assert arm.amm_customer_volume_base == 19.0
    assert arm.arbitrage_volume_base == 81.0
    assert arm.arbitrage_share_of_amm_execution == 0.81

    accounting = account_pair(arm, ArmWindow(200.0, 4.0),
                              annual_capital_rate=0.0)
    assert accounting['arbitrage_execution_in_user_benefit'] is False
    assert accounting['arbitrage_surplus_identified'] is False
    assert accounting['with_arbitrage_volume_base'] == 81.0
    assert accounting['matched_notional_user_benefit'] == 2.0
    assert accounting['total_welfare_identified'] is False


def test_survival_summary_reports_binomial_uncertainty():
    rows = []
    for seed in range(10):
        rows.append({
            'seed': seed,
            'pools': [{
                'pool': 'hfmm',
                'open_at_shock': seed != 0,
                'open_through_crisis': seed not in (0, 1),
                'first_close': 700 if seed in (0, 1) else None,
                'ticks_closed_pre': 0.0,
                'ticks_closed_crisis': 0.0,
                'exits_in_crisis': 1.0,
                'active_pre_shock': 5.0,
                'active_min_crisis': 4.0,
                'active_end': 3.0,
                'x_ratio_min_crisis': 0.9,
                'rho_over_option': 1.1,
                'new_entry_events_total': 1 if seed == 2 else 0,
                'reentry_events_total': 1 if seed == 3 else 0,
                'exit_events_total': 2,
                'new_entry_risk_set_providers': 5,
                'new_entry_right_censored_providers': 4 if seed == 2 else 5,
                'reentry_risk_set_episodes': 2,
                'reentry_completed_episodes': 1 if seed == 3 else 0,
                'reentry_right_censored_episodes': 1 if seed == 3 else 2,
                'reentry_right_censored_providers': 1,
            }],
        })
    row = summarize(rows, {'n_iter': 1000})['pools']['hfmm']

    assert row['open_at_shock_count'] == 9
    assert row['open_through_crisis_count'] == 8
    assert row['any_closure_count'] == 2
    assert row['gross_new_entry_events'] == 1
    assert row['gross_reentry_events'] == 1
    assert row['new_entry_right_censored_providers'] == 49
    assert row['reentry_risk_set_episodes'] == 20
    assert row['reentry_right_censored_episodes'] == 19
    assert row['reentry_episode_activation_rate'] == 0.05
    assert row['entry_activation_status'] == 'observed'
    assert row['open_at_shock_rate_wilson_95'][0] < 0.9
    assert row['open_at_shock_rate_wilson_95'][1] > 0.9


def test_survival_reads_the_state_carried_into_the_shock():
    history = {
        'closed': [0, 0, 1, 0, 0],
        'active': [5, 5, 4, 4, 4],
        'rho': [0.0] * 5,
        'supply': [100.0, 100.0, 90.0, 90.0, 90.0],
        'entries_gross': [0] * 5,
        'reentries_gross': [0] * 5,
        'exits_gross': [0, 0, 1, 0, 0],
        'deployed': [5, 5, 4, 4, 4],
        'deployed_share': [1.0, 1.0, 0.9, 0.9, 0.9],
        'entry_progress_max': [0.0] * 5,
    }
    pool = SimpleNamespace(x_history=[100.0] * 6)
    population = SimpleNamespace(
        history=history, pool=pool, providers=[], n_incumbents=5,
    )
    row = _read_population(population, shock=2, option=1e-9)

    assert row['open_at_shock'] is True
    assert row['open_after_initial_shock_step'] is False
    assert row['open_through_crisis'] is False
    assert row['exits_in_crisis'] == 1.0
    assert row['gross_event_identity_holds'] is True


def test_no_entry_is_explicitly_right_censored():
    recs = [{
        'seed': 42,
        'pools': [{
            'pool': 'hfmm', 'open_at_shock': True,
            'open_through_crisis': True, 'first_close': None,
            'new_entry_events_total': 0, 'reentry_events_total': 0,
            'exit_events_total': 1, 'new_entry_risk_set_providers': 5,
            'new_entry_right_censored_providers': 5,
            'reentry_risk_set_episodes': 1,
            'reentry_completed_episodes': 0,
            'reentry_right_censored_episodes': 1,
            'reentry_right_censored_providers': 1,
        }],
    }]
    row = summarize(recs, {'n_iter': 1000})['pools']['hfmm']

    assert row['entry_activation_observed'] is False
    assert row['entry_activation_status'] == 'right_censored_no_activation_by_horizon'
    assert row['new_entry_provider_activation_rate'] == 0.0
    assert row['reentry_episode_activation_rate'] == 0.0


def test_lp_sensitivity_has_oat_controls_and_a_complete_patience_ewma_grid():
    specs = _specifications(1000)
    baseline = dict(specs[0][1])
    assert len(specs) == 15
    patience_ewma_cells = set()
    for label, config in specs[1:]:
        changed = {key for key in config if config[key] != baseline[key]}
        if label.startswith('patience_'):
            assert changed == {'exit_patience', 'entry_patience', 'max_adj'}
            patience = config['exit_patience']
            assert abs(config['max_adj'] - (1.0 - 0.5 ** (1.0 / patience))) < 1e-15
        elif label.startswith('joint_patience_'):
            assert changed == {
                'exit_patience', 'entry_patience', 'max_adj', 'ewma_alpha',
            }
            patience = config['exit_patience']
            assert config['entry_patience'] == patience
            assert abs(config['max_adj'] - (1.0 - 0.5 ** (1.0 / patience))) < 1e-15
            patience_ewma_cells.add((patience, config['ewma_alpha']))
        else:
            assert len(changed) == 1

    assert patience_ewma_cells == {
        (145, 0.01), (145, 0.05), (580, 0.01), (580, 0.05),
    }


def test_pnl_json_aggregate_reports_completeness_cis_and_accounting_identity(
        monkeypatch):
    records = {
        42: {
            'seed': 42,
            'total_calm': 0.3, 'total_calm_loss': 0.2,
            'total_calm_fees': 0.5,
            'total_crisis': -0.8, 'total_crisis_loss': 1.0,
            'total_crisis_fees': 0.2,
        },
        43: {
            'seed': 43,
            'total_calm': 0.1, 'total_calm_loss': 0.3,
            'total_calm_fees': 0.4,
            'total_crisis': None, 'total_crisis_loss': None,
            'total_crisis_fees': None,
        },
    }
    monkeypatch.setattr(pnl_tool, 'load_raw',
                        lambda fee, model: (records, 3))
    payload = pnl_tool.aggregate_json(
        [5.0], 'endogenous', {42, 43}, seed_start=42,
        requested_seeds=2, bootstrap_draws=0,
    )

    setting = payload['fee_settings'][0]
    assert payload['complete'] is False
    assert payload['raw_provenance_clean'] is False
    assert payload['invalid_or_stale_records_ignored'] == 3
    assert payload['profitability_acceptance_target'] is None
    assert setting['complete'] is False
    assert setting['raw_provenance_clean'] is False
    assert setting['stale_records_ignored'] == 3
    assert setting['accounting_sign_convention']['identity_holds'] is True
    assert setting['windows']['calm']['total']['net_pct']['median'] == 0.2
    assert setting['windows']['crisis']['total']['measurable_seed_count'] == 1
    assert setting['windows']['crisis']['total']['unmeasurable_seed_count'] == 1

    incomplete = pnl_tool.aggregate_json(
        [5.0], 'endogenous', {42, 43, 44}, seed_start=42,
        requested_seeds=3, bootstrap_draws=0,
    )
    assert incomplete['complete'] is False


def test_lp_acceptance_is_an_integrity_gate_not_an_outcome_sign_filter():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    signature = model_signature(root)
    targets = json.loads(open(
        os.path.join(root, 'calibration', 'primary_model_targets.json'),
        encoding='utf-8',
    ).read())
    protocol = json.loads(open(
        os.path.join(root, 'calibration', 'final_protocol.json'),
        encoding='utf-8',
    ).read())
    # Direct validation accepts an injected protocol so that a one-seed
    # synthetic fixture can exercise every branch cheaply.  The CLI rejects
    # any non-canonical path; the canonical protocol separately commits 300
    # development and 300 untouched holdout seeds.
    protocol['phases']['development'].update(seed_start=42, seed_count=1)
    protocol['phases']['holdout'].update(seed_start=10042, seed_count=1)
    lp_design = protocol['lp_design']
    for name in (
        'development_robustness_panel', 'entry_identification_arm',
        'loss_rebate_identification_arm',
    ):
        lp_design[name].update(seed_start=42, seed_count=1)
    for name in ('final_inference_panel', 'pnl_panel',
                 'welfare_incidence_panels'):
        lp_design[name].update(seed_start=10042, seed_count=1)
    lp_design['routing_sensitivity'] = {
        'design': routing_tool.PRECOMMIT_PROTOCOL['design'],
        'expected_specifications': (
            routing_tool.PRECOMMIT_PROTOCOL['expected_unique_specifications']
        ),
        **copy.deepcopy(routing_tool.PRECOMMIT_PROTOCOL['declared_levels']),
    }
    protocol['frozen_hashes'] = {
        'target_matrix_sha256': calibration_runner._target_matrix_sha256(targets),
        'primary_runtime_sha256': calibration_runner._primary_runtime_sha256(),
        'simulation_model_signature': signature,
        'book_acceptance_signature': (
            calibration_runner.book_acceptance_signature(targets)
        ),
    }
    protocol['lp_frozen_signatures'] = {
        'survival_measurement_signature': survival_tool.survival_signature(),
        'survival_report_signature': survival_tool.survival_report_signature(),
        'sensitivity_report_signature': (
            sensitivity_tool.sensitivity_report_signature()
        ),
        'routing_measurement_signature': routing_tool.sensitivity_signature(),
        'routing_report_signature': routing_tool.report_signature(),
        'routing_protocol_signature': routing_tool.protocol_signature(),
        'pnl_measurement_signature': pnl_tool.run_signature(),
        'pnl_report_signature': pnl_tool.aggregate_report_signature(),
        'welfare_measurement_signature': (
            welfare_tool.welfare_measurement_signature()
        ),
        'welfare_report_signature': welfare_tool.welfare_report_signature(),
        'lp_acceptance_code_sha256': acceptance_code_sha256(),
    }
    original_routing_report_signature = 'synthetic-original-routing-report'
    original_acceptance_sha256 = '0' * 64
    protocol['lp_frozen_signatures']['routing_report_signature'] = (
        original_routing_report_signature
    )
    protocol['lp_frozen_signatures']['lp_acceptance_code_sha256'] = (
        original_acceptance_sha256
    )
    protocol_sha = hashlib.sha256(b'synthetic-protocol').hexdigest()
    book_holdout = {
        'provenance': {
            'protocol_sha256': protocol_sha,
            'protocol_version': protocol['protocol_version'],
            'protocol_phase': 'holdout',
            'seed_commitment_exact_match': True,
            'simulation_model_signature': signature,
            'book_acceptance_signature': protocol['frozen_hashes'][
                'book_acceptance_signature'
            ],
            'primary_runtime_sha256': protocol['frozen_hashes'][
                'primary_runtime_sha256'
            ],
            'target_matrix_sha256': protocol['frozen_hashes'][
                'target_matrix_sha256'
            ],
            'aggregation_mode': 'fresh_simulation',
        },
        'panel': {
            'seeds': [10042], 'n_seeds': 1,
            'mechanism_checks': {'synthetic_complete': True},
        },
        'summary': {
            'status': 'pass', 'gating_failures': 0,
            'mechanism_failures': 0,
        },
    }
    book_development_sha = hashlib.sha256(
        b'synthetic-development-artifact'
    ).hexdigest()
    book_development = copy.deepcopy(book_holdout)
    book_development['provenance']['protocol_phase'] = 'development'
    book_development['panel']['seeds'] = [42]
    book_holdout['provenance'].update({
        'development_gate_verified': True,
        'development_gate_sha256': book_development_sha,
    })
    pool_summary = {
        'n_seeds': 1,
        'gross_event_identity_pass_rate': 1.0,
        'claim_supply_identity_end_pass_rate': 1.0,
        'entry_activation_status': 'right_censored_no_activation_by_horizon',
        'entry_activation_observed': False,
        'gross_new_entry_events': 0,
        'gross_reentry_events': 0,
        'gross_exit_events': 2,
        'open_through_crisis_count': 0,
        'new_entry_risk_set_providers': 5,
        'new_entry_provider_activation_rate': 0.0,
        'median_max_entry_progress': 0.5,
    }
    primary_lp_config = survival_tool._calibrated_lp_defaults()
    primary_lp_config['n_iter'] = survival_tool.N_ITER
    primary = {
        'model_signature': signature,
        'measurement_signature': survival_tool.survival_signature(),
        'report_signature': survival_tool.survival_report_signature(),
        'complete': True, 'available_seeds': 1, 'requested_seeds': 1,
        'invalid_or_stale_records_ignored': 0,
        'raw_provenance_clean': True,
        'requested_seed_start': 10042,
        'requested_seed_list': [10042],
        'summary': {'n_seed_records': 1,
                    'configuration': dict(primary_lp_config),
                    'pools': {'hfmm': dict(pool_summary)}},
    }
    funded_pool = dict(pool_summary)
    funded_pool.update(entry_activation_status='observed',
                       entry_activation_observed=True,
                       gross_new_entry_events=1)
    funded = {
        'model_signature': signature,
        'measurement_signature': survival_tool.survival_signature(),
        'report_signature': survival_tool.survival_report_signature(),
        'complete': True, 'available_seeds': 1, 'requested_seeds': 1,
        'invalid_or_stale_records_ignored': 0,
        'raw_provenance_clean': True,
        'requested_seed_start': 42,
        'requested_seed_list': [42],
        'summary': {'n_seed_records': 1,
                    'configuration': {
                        **primary_lp_config,
                        'subsidy_rate': 1e-5,
                    },
                    'pools': {'hfmm': funded_pool}},
    }
    rebate_pool = dict(pool_summary)
    rebate_pool.update(gross_exit_events=1, open_through_crisis_count=1)
    rebate = {
        'model_signature': signature,
        'measurement_signature': survival_tool.survival_signature(),
        'report_signature': survival_tool.survival_report_signature(),
        'complete': True, 'available_seeds': 1, 'requested_seeds': 1,
        'invalid_or_stale_records_ignored': 0,
        'raw_provenance_clean': True,
        'requested_seed_start': 42,
        'requested_seed_list': [42],
        'summary': {'n_seed_records': 1,
                    'configuration': {
                        **primary_lp_config,
                        'loss_rebate_fraction': 1.0,
                    },
                    'pools': {'hfmm': rebate_pool}},
    }
    sensitivity_specs = dict(sensitivity_tool._specifications(
        survival_tool.N_ITER
    ))
    sensitivity = {
        'model_signature': signature,
        'measurement_signature': survival_tool.survival_signature(),
        'report_signature': sensitivity_tool.sensitivity_report_signature(),
        'complete': True,
        'invalid_or_stale_records_ignored': 0,
        'raw_provenance_clean': True,
        'design': 'oat_plus_joint_patience_by_ewma_around_primary_lp_specification',
        'requested_seed_start': 42,
        'requested_seeds_per_specification': 1,
        'requested_seeds': [42],
        'results': [
            {'label': label, 'available_seeds': 1,
             'stale_records_ignored': 0,
             'invalid_or_stale_records_ignored': 0,
             'raw_provenance_clean': True,
             'summary': {
                 'n_seed_records': 1,
                 'configuration': config,
                 'pools': {'hfmm': {
                     **pool_summary,
                     'gross_new_entry_events': (
                         1 if label == 'entry_margin_0' else 0
                     ),
                 }},
             }}
            for label, config in sensitivity_specs.items()
        ],
    }
    routing_phase_metrics = routing_tool.PHASE_METRICS
    routing_summary = {
        f'{metric}_{phase}': {
            'n_finite': 1, 'mean': 1.0, 'median': 1.0,
            'p10': 1.0, 'p90': 1.0,
        }
        for phase in routing_tool.PHASES
        for metric in routing_phase_metrics
    }
    routing_summary.update({
        metric: {
            'n_finite': 1, 'mean': 1.0, 'median': 1.0,
            'p10': 1.0, 'p90': 1.0,
        }
        for metric in routing_tool.RUN_METRICS
    })
    for phase in routing_tool.PHASES:
        routing_summary[
            f'amm_arbitrage_share_of_amm_execution_{phase}'
        ] = {
            'n_finite': 0,
            'n_zero_denominator': 1,
            'n_invalid': 0,
            'n_observations': 1,
            'mean': None,
            'median': None,
            'p10': None,
            'p90': None,
        }
    routing_specifications = routing_tool._specifications(routing_tool.N_ITER)
    routing_specification_count = len(routing_specifications)
    routing = {
        'model_signature': signature,
        'measurement_signature': routing_tool.sensitivity_signature(),
        'report_signature': routing_tool.report_signature(),
        'protocol_signature': routing_tool.protocol_signature(),
        'protocol': routing_tool.PRECOMMIT_PROTOCOL,
        'requested_seed_start': 42,
        'requested_seeds_per_specification': 1,
        'requested_seeds': [42],
        'n_iter': routing_tool.N_ITER,
        'expected_specifications': routing_specification_count,
        'expected_records': routing_specification_count,
        'available_records': routing_specification_count,
        'complete': True,
        'raw_provenance_clean': True,
        'precommit_protocol_conformant': True,
        'publication_ready': True,
        'missing_seeds_by_specification': {
            label: []
            for label, _config in routing_specifications
        },
        'results': [
            {
                'label': label, 'available_records': 1,
                'missing_records': 0, 'stale_records_ignored': 0,
                'configuration': config,
                'summary': routing_summary,
            }
            for label, config in routing_specifications
        ],
    }
    pnl_setting = {
        'complete': True,
        'stale_records_ignored': 0,
        'invalid_or_stale_records_ignored': 0,
        'raw_provenance_clean': True,
        'requested_seed_count': 1,
        'available_seed_count': 1,
        'available_seed_start': 10042,
        'available_seed_end': 10042,
        'accounting_sign_convention': {
            'identity_holds': True, 'net_formula': 'fees_minus_loss',
        },
        'windows': {
            label: {'total': {'net_pct': {
                # A negative result must not fail an integrity-only gate.
                'n': 1, 'median': -5.0,
                'bootstrap_95_interval': [-7.0, -3.0],
            }}}
            for label in ('calm', 'crisis')
        },
    }
    fee_grid = [None, 1.0, 2.5, 5.0, 10.0, 20.0]
    fee_settings = []
    for fee in fee_grid:
        fee_settings.append({
            **copy.deepcopy(pnl_setting),
            'fee_bps': fee,
            'fee_mode': ('calibrated' if fee is None
                         else 'uniform_all_amm_venues'),
        })
    pnl = {
        'model_signature': signature,
        'measurement_signature': pnl_tool.run_signature(),
        'report_signature': pnl_tool.aggregate_report_signature(),
        'complete': True, 'capital_flow_zero_result_self_check_passed': True,
        'invalid_or_stale_records_ignored': 0,
        'raw_provenance_clean': True,
        'profitability_acceptance_target': None,
        'configuration': {
            'preset': pnl_tool.PRESET,
            'n_iter': pnl_tool.N_ITER,
            'lp_model': 'endogenous',
            'seed_start': 10042,
            'requested_seeds_per_fee': 1,
            'requested_seed_list': [10042],
            'fee_grid_bps': fee_grid,
            'bootstrap_draws': 20000,
            'bootstrap_seed': 0,
        },
        'fee_settings': fee_settings,
    }
    accounting = {key: 0.0 for key in welfare_tool.SUMMARY_METRICS}
    accounting.update({
        'subsidy_transfer_cancels': True,
        'user_benefit_uses_routed_customer_fills_only': True,
        'arbitrage_execution_in_user_benefit': False,
        'arbitrage_surplus_identified': False,
        'total_welfare_identified': False,
        'common_executed_notional': 100.0,
        'with_execution_cost_bps': 1.0,
        'without_execution_cost_bps': 2.0,
        'matched_notional_user_benefit': 0.01,
        'matched_notional_user_benefit_bps': 1.0,
        'lp_operating_result': -5.0,
        'lp_operating_return': -0.005,
        'capital_opportunity_cost': 0.1,
        'with_amm_customer_volume_base': 10.0,
        'with_arbitrage_volume_base': 40.0,
        'with_amm_execution_volume_base': 50.0,
        'with_arbitrage_share_of_amm_execution': 0.8,
    })
    def welfare_arm(subsidy_rate, loss_rebate):
        return {
        'model_signature': signature,
        'measurement_signature': welfare_tool.welfare_measurement_signature(),
        'report_signature': welfare_tool.welfare_report_signature(),
        'configuration': {
            'seed_start': 10042,
            'seeds': 1,
            'lp_model': 'endogenous',
            'capital_rate': 0.029,
            'subsidy_rate': subsidy_rate,
            'loss_rebate': loss_rebate,
            'bootstrap_draws': 20000,
            'bootstrap_seed': 0,
        },
        'summary': welfare_tool.aggregate(
            [{'accounting': accounting}], bootstrap_draws=20000,
            bootstrap_seed=0,
        ),
        'rows': [{
            'seed': 10042,
            'accounting': accounting,
            'with_amm': {
                'executed_notional': 100.0,
                'taker_execution_cost': 0.01,
                'lp_opening_capital': 1000.0,
                'lp_lvr': 6.0,
                'lp_fees': 1.0,
                'amm_customer_volume_base': 10.0,
                'arbitrage_volume_base': 40.0,
                'arbitrage_share_of_amm_execution': 0.8,
            },
            'without_amm': {
                'executed_notional': 100.0,
                'taker_execution_cost': 0.02,
            },
        }],
    }
    welfare = [welfare_arm(0.0, 0.0), welfare_arm(1.3319e-9, 1.0)]

    original_routing = copy.deepcopy(routing)
    original_routing['report_signature'] = original_routing_report_signature
    base_summary_fields = {'n_finite', 'mean', 'median', 'p10', 'p90'}
    for row in original_routing['results']:
        for phase in routing_tool.PHASES:
            field = f'amm_arbitrage_share_of_amm_execution_{phase}'
            row['summary'][field] = {
                key: value for key, value in row['summary'][field].items()
                if key in base_summary_fields
            }
    failed_acceptance = {
        'model_signature': signature,
        'lp_acceptance_code_sha256': original_acceptance_sha256,
        'protocol_version': protocol['protocol_version'],
        'protocol_sha256': protocol_sha,
        'passed': False,
        'checks': [{
            'name': (
                'routing_customer_volume_count_active_tick_and_arb_are_separate'
            ),
            'passed': False,
            'detail': '',
        }],
    }
    artifact_sha256 = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in acceptance_tool.AMENDMENT_ARTIFACT_PATHS
    }
    artifact_sha256['book_development'] = book_development_sha
    amendment = {
        'amendment_version': acceptance_tool.AMENDMENT_VERSION,
        'amendment_type': acceptance_tool.AMENDMENT_TYPE,
        'base_protocol': {
            'path': 'calibration/final_protocol.json',
            'version': protocol['protocol_version'],
            'sha256': protocol_sha,
        },
        'change_control': copy.deepcopy(
            acceptance_tool.AMENDMENT_CHANGE_CONTROL
        ),
        'correction': copy.deepcopy(acceptance_tool.AMENDMENT_CORRECTION),
        'original_frozen_signatures': copy.deepcopy(
            protocol['lp_frozen_signatures']
        ),
        'effective_frozen_signatures': (
            acceptance_tool._current_lp_frozen_signatures()
        ),
        'artifact_bindings': {
            name: {'path': path, 'sha256': artifact_sha256[name]}
            for name, path in acceptance_tool.AMENDMENT_ARTIFACT_PATHS.items()
        },
    }
    amendment_sha256 = hashlib.sha256(
        json.dumps(amendment, sort_keys=True).encode()
    ).hexdigest()
    amendment_args = {
        'amendment': amendment,
        'amendment_sha256': amendment_sha256,
        'artifact_sha256': artifact_sha256,
        'original_routing': original_routing,
        'failed_acceptance': failed_acceptance,
    }

    result = validate_lp_acceptance(
        protocol, protocol_sha, book_development, book_development_sha,
        book_holdout, primary, funded, rebate,
        sensitivity, routing, pnl, welfare,
        **amendment_args,
    )
    assert result['passed'] is True, [
        row for row in result['checks'] if not row['passed']
    ]
    assert 'positive_lp_profit' in result['explicit_noncriteria']
    assert result['amendment_version'] == acceptance_tool.AMENDMENT_VERSION
    assert result['amendment_sha256'] == amendment_sha256

    direct_protocol = copy.deepcopy(protocol)
    direct_protocol['protocol_version'] = acceptance_tool.DIRECT_PROTOCOL_VERSION
    direct_paths = {
        name: (
            f'output/final/{name}.json' if name.startswith('book_')
            else f'output/resilience/{name}.json'
        )
        for name in acceptance_tool.DIRECT_ARTIFACT_PATH_KEYS
    }
    direct_protocol['artifact_paths'] = direct_paths
    direct_protocol['phases']['development']['output'] = direct_paths[
        'book_development'
    ]
    direct_protocol['phases']['holdout']['output'] = direct_paths['book_holdout']
    direct_protocol['lp_frozen_signatures'] = (
        acceptance_tool._current_lp_frozen_signatures()
    )
    direct_sha = hashlib.sha256(b'synthetic-direct-protocol').hexdigest()
    direct_development = copy.deepcopy(book_development)
    direct_holdout = copy.deepcopy(book_holdout)
    for artifact in (direct_development, direct_holdout):
        artifact['provenance']['protocol_version'] = (
            acceptance_tool.DIRECT_PROTOCOL_VERSION
        )
        artifact['provenance']['protocol_sha256'] = direct_sha
    direct_artifact_sha = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in acceptance_tool.DIRECT_INPUT_ARTIFACT_KEYS
    }
    direct_artifact_sha['book_development'] = book_development_sha
    direct_result = validate_lp_acceptance(
        direct_protocol, direct_sha,
        direct_development, book_development_sha, direct_holdout,
        primary, funded, rebate, sensitivity, routing, pnl, welfare,
        artifact_sha256=direct_artifact_sha,
    )
    assert direct_result['passed'] is True
    assert direct_result['provenance_mode'] == 'direct_frozen_protocol'
    assert direct_result['amendment_version'] is None

    def direct_failure(*, primary_payload=primary,
                       sensitivity_payload=sensitivity,
                       routing_payload=routing, pnl_payload=pnl):
        return validate_lp_acceptance(
            direct_protocol, direct_sha,
            direct_development, book_development_sha, direct_holdout,
            primary_payload, funded, rebate, sensitivity_payload,
            routing_payload, pnl_payload, welfare,
            artifact_sha256=direct_artifact_sha,
        )

    dirty_survival = copy.deepcopy(primary)
    dirty_survival.pop('raw_provenance_clean')
    failed = direct_failure(primary_payload=dirty_survival)
    assert any(
        row['name'] == 'survival_artifacts_have_clean_raw_provenance'
        and not row['passed'] for row in failed['checks']
    )

    dirty_sensitivity = copy.deepcopy(sensitivity)
    dirty_sensitivity['invalid_or_stale_records_ignored'] = 1
    dirty_sensitivity['raw_provenance_clean'] = False
    failed = direct_failure(sensitivity_payload=dirty_sensitivity)
    assert any(
        row['name'] == 'sensitivity_has_clean_raw_provenance'
        and not row['passed'] for row in failed['checks']
    )

    dirty_routing = copy.deepcopy(routing)
    dirty_routing['raw_provenance_clean'] = False
    failed = direct_failure(routing_payload=dirty_routing)
    assert any(
        row['name'] == 'routing_lp_sensitivity_has_clean_raw_provenance'
        and not row['passed'] for row in failed['checks']
    )

    dirty_pnl = copy.deepcopy(pnl)
    dirty_pnl['fee_settings'][0]['invalid_or_stale_records_ignored'] = 1
    dirty_pnl['fee_settings'][0]['raw_provenance_clean'] = False
    failed = direct_failure(pnl_payload=dirty_pnl)
    assert any(
        row['name'] == 'pnl_has_clean_raw_provenance'
        and not row['passed'] for row in failed['checks']
    )

    wrong_base = copy.deepcopy(amendment_args)
    wrong_base['amendment'] = copy.deepcopy(amendment)
    wrong_base['amendment']['base_protocol']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='exact base protocol'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, routing, pnl, welfare, **wrong_base,
        )

    changed_thresholds = copy.deepcopy(amendment_args)
    changed_thresholds['amendment'] = copy.deepcopy(amendment)
    changed_thresholds['amendment']['change_control'][
        'acceptance_thresholds_changed'
    ] = True
    with pytest.raises(ValueError, match='change-control'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, routing, pnl, welfare, **changed_thresholds,
        )

    wrong_artifact = copy.deepcopy(amendment_args)
    wrong_artifact['amendment'] = copy.deepcopy(amendment)
    wrong_artifact['amendment']['artifact_bindings']['pnl']['sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='hash mismatch for pnl'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, routing, pnl, welfare, **wrong_artifact,
        )

    changed_measurement = copy.deepcopy(amendment_args)
    changed_measurement['original_routing']['results'][0]['configuration'][
        'routing_cost_scale_bps'
    ] = 99.0
    with pytest.raises(ValueError, match='changes more than'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, routing, pnl, welfare, **changed_measurement,
        )

    broken_welfare = copy.deepcopy(welfare)
    broken_welfare[0]['summary']['all_subsidy_transfers_cancel'] = False
    failed = validate_lp_acceptance(
        protocol, protocol_sha, book_development, book_development_sha,
        book_holdout, primary, funded, rebate,
        sensitivity, routing, pnl, broken_welfare,
        **amendment_args,
    )
    assert failed['passed'] is False
    assert any(row['name'] == 'welfare_subsidy_transfer_identities_hold'
               and not row['passed'] for row in failed['checks'])

    stale_welfare_report = copy.deepcopy(welfare)
    stale_welfare_report[0].pop('report_signature')
    failed = validate_lp_acceptance(
        protocol, protocol_sha, book_development, book_development_sha,
        book_holdout, primary, funded, rebate,
        sensitivity, routing, pnl, stale_welfare_report,
        **amendment_args,
    )
    assert failed['passed'] is False
    assert any(
        row['name'] == 'welfare_uses_current_measurement_and_report_signatures'
        and not row['passed']
        for row in failed['checks']
    )

    missing_core_value = copy.deepcopy(welfare)
    del missing_core_value[0]['rows'][0]['accounting'][
        'common_executed_notional'
    ]
    failed = validate_lp_acceptance(
        protocol, protocol_sha, book_development, book_development_sha,
        book_holdout, primary, funded, rebate,
        sensitivity, routing, pnl, missing_core_value,
        **amendment_args,
    )
    assert failed['passed'] is False
    assert any(
        row['name'] == 'welfare_core_row_estimands_are_finite_and_reconciled'
        and not row['passed']
        for row in failed['checks']
    )

    wrong_summary_count = copy.deepcopy(welfare)
    wrong_summary_count[0]['summary']['finite_counts'][
        'matched_notional_user_benefit'
    ] = 0
    failed = validate_lp_acceptance(
        protocol, protocol_sha, book_development, book_development_sha,
        book_holdout, primary, funded, rebate,
        sensitivity, routing, pnl, wrong_summary_count,
        **amendment_args,
    )
    assert failed['passed'] is False
    assert any(
        row['name']
        == 'welfare_core_summary_counts_medians_and_intervals_are_complete'
        and not row['passed']
        for row in failed['checks']
    )

    missing_active_tick = copy.deepcopy(routing)
    del missing_active_tick['results'][0]['summary'][
        'amm_active_tick_flow_share_calm'
    ]
    with pytest.raises(ValueError, match='changes more than'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, missing_active_tick, pnl, welfare,
            **amendment_args,
        )

    invalid_zero_denominator = copy.deepcopy(routing)
    invalid_summary = invalid_zero_denominator['results'][0]['summary'][
        'amm_arbitrage_share_of_amm_execution_calm'
    ]
    invalid_summary.update(
        n_finite=0, n_zero_denominator=0, n_invalid=1,
        n_observations=1,
    )
    with pytest.raises(ValueError, match='changes more than'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, invalid_zero_denominator, pnl, welfare,
            **amendment_args,
        )

    wrong_cell = copy.deepcopy(routing)
    wrong_cell['results'][0]['configuration']['routing_cost_scale_bps'] = 99.0
    with pytest.raises(ValueError, match='changes more than'):
        validate_lp_acceptance(
            protocol, protocol_sha, book_development, book_development_sha,
            book_holdout, primary, funded, rebate,
            sensitivity, wrong_cell, pnl, welfare,
            **amendment_args,
        )
