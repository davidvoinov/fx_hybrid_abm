import hashlib
import json
from pathlib import Path

import pytest

from calibration.fitter import (
    CalibrationFitter,
    _kaplan_meier_median,
    _lifecycle_summary,
)
from calibration.runner import (
    MAX_DEALER_LIFETIME_ATOM_SHARE,
    MAX_NONBANK_LIFETIME_ATOM_SHARE,
    MAX_SAME_TICK_SCHEDULED_END_SHARE,
    MAX_UNCATEGORIZED_LIFECYCLE_SHARE,
    MIN_COMPLETED_LIFECYCLE_EVENTS,
    MIN_TWO_SIDED_BOOK_RATE,
    _mechanism_audit,
    _verify_development_artifact,
    _wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]


def _passing_book_integrity(n):
    return {
        'book_two_sided_rate': [1.0] * n,
        'dealer_lifecycle_max_atom_share': [0.02] * n,
        'nonbank_lifecycle_max_atom_share': [0.28] * n,
        'dealer_lifecycle_uncategorized_share': [0.0] * n,
        'nonbank_lifecycle_uncategorized_share': [0.0] * n,
        'dealer_lifecycle_max_same_tick_scheduled_end_share': [0.02] * n,
        'nonbank_lifecycle_max_same_tick_scheduled_end_share': [0.02] * n,
        'dealer_lifecycle_completed_count': [500.0] * n,
        'nonbank_lifecycle_completed_count': [800.0] * n,
        'dealer_order_lifetime_median_seconds': [290.0] * n,
        'nonbank_order_lifetime_median_seconds': [3.0] * n,
    }


def test_wilson_limits_require_a_material_300_seed_stability_margin():
    ebs_fail = _wilson_interval(280, 300)
    ebs_pass = _wilson_interval(281, 300)
    calm = _wilson_interval(5, 300)
    crisis = _wilson_interval(288, 300)

    assert ebs_fail[0] < 0.90
    assert ebs_pass[0] > 0.90
    assert calm[1] < 0.05
    assert crisis[0] > 0.93


def test_mechanism_audit_accepts_rare_calm_and_common_crisis_activation():
    ebs = [True] * 281 + [False] * 19
    calm = [0.2] * 5 + [0.0] * 295
    crisis = [0.4] * 288 + [0.0] * 12
    forced = [0.0] * 300

    audit = _mechanism_audit(
        ebs, calm, crisis, forced,
        book_integrity=_passing_book_integrity(300),
    )

    assert audit['calm_activation_count'] == 5
    assert audit['crisis_activation_count'] == 288
    assert all(audit['mechanism_checks'].values())


def test_small_panel_cannot_certify_a_low_false_positive_rate():
    audit = _mechanism_audit(
        [True] * 8,
        [0.0] * 8,
        [0.4] * 8,
        [0.0] * 8,
        book_integrity=_passing_book_integrity(8),
    )

    assert audit['calm_activation_rate'] == 0.0
    assert not audit['mechanism_checks'][
        'calm_activation_upper_95pct_at_most_5pct'
    ]


def test_book_integrity_gates_reject_bulk_lifecycle_mechanics():
    book = _passing_book_integrity(300)
    book['book_two_sided_rate'][17] = 0.95
    book['dealer_lifecycle_max_atom_share'][23] = 0.90
    book['nonbank_lifecycle_uncategorized_share'][41] = 0.01
    book['dealer_lifecycle_max_same_tick_scheduled_end_share'][52] = 0.40
    audit = _mechanism_audit(
        [True] * 281 + [False] * 19,
        [0.2] * 5 + [0.0] * 295,
        [0.4] * 288 + [0.0] * 12,
        [0.0] * 300,
        book_integrity=book,
    )
    checks = audit['mechanism_checks']

    assert not checks['book_two_sided_rate_at_least_99_9pct']
    assert not checks['dealer_lifecycle_atom_share_at_most_25pct']
    assert not checks['nonbank_lifecycle_reasons_fully_categorized']
    assert not checks['dealer_scheduled_ends_not_synchronized']
    assert checks['nonbank_lifecycle_atom_share_at_most_35pct']


def test_protocol_book_threshold_names_match_acceptance_code():
    protocol = json.loads((ROOT / 'calibration' / 'final_protocol.json').read_text())
    assert protocol['book_integrity_thresholds'] == {
        'minimum_two_sided_book_rate': MIN_TWO_SIDED_BOOK_RATE,
        'maximum_dealer_completed_lifetime_atom_share': (
            MAX_DEALER_LIFETIME_ATOM_SHARE
        ),
        'maximum_nonbank_completed_lifetime_atom_share': (
            MAX_NONBANK_LIFETIME_ATOM_SHARE
        ),
        'maximum_uncategorized_lifecycle_share_per_class': (
            MAX_UNCATEGORIZED_LIFECYCLE_SHARE
        ),
        'maximum_same_tick_scheduled_end_share_per_class': (
            MAX_SAME_TICK_SCHEDULED_END_SHARE
        ),
        'minimum_completed_lifecycle_events_per_seed_per_class': (
            MIN_COMPLETED_LIFECYCLE_EVENTS
        ),
    }


def test_v2_protocol_preserves_failed_development_and_commits_fresh_holdout():
    protocol = json.loads((ROOT / 'calibration' / 'final_protocol.json').read_text())
    development = protocol['phases']['development']
    holdout = protocol['phases']['holdout']
    revisions = {row['version']: row for row in protocol['revision_history']}

    assert protocol['protocol_version'] == 'book-lp-final-v2'
    assert development['seed_count'] == holdout['seed_count'] == 300
    assert set(range(development['seed_start'],
                     development['seed_start'] + development['seed_count'])).isdisjoint(
        range(holdout['seed_start'], holdout['seed_start'] + holdout['seed_count'])
    )
    assert development['output'] != revisions['book-lp-final-v1'][
        'development_output'
    ]
    assert revisions['book-lp-final-v1']['result'] == 'failed_development_gate'
    assert revisions['book-lp-final-v1']['holdout_opened'] is False
    assert revisions['book-lp-final-v2']['thresholds_changed'] is False
    assert revisions['book-lp-final-v2']['holdout_opened_at_freeze'] is False


def test_holdout_authorization_requires_the_exact_passing_development(tmp_path):
    protocol = json.loads((ROOT / 'calibration' / 'final_protocol.json').read_text())
    protocol['phases']['development'].update(
        seed_start=42, seed_count=1,
        output=str(tmp_path / 'development.json'),
    )
    protocol['phases']['holdout'].update(seed_start=10042, seed_count=1)
    protocol_raw = json.dumps(protocol, sort_keys=True).encode()
    frozen = protocol['frozen_hashes']
    report = {
        'provenance': {
            'protocol_version': protocol['protocol_version'],
            'protocol_phase': 'development',
            'protocol_sha256': hashlib.sha256(protocol_raw).hexdigest(),
            'seed_commitment_exact_match': True,
            'aggregation_mode': 'fresh_simulation',
            'simulation_model_signature': frozen['simulation_model_signature'],
            'book_acceptance_signature': frozen['book_acceptance_signature'],
            'primary_runtime_sha256': frozen['primary_runtime_sha256'],
            'target_matrix_sha256': frozen['target_matrix_sha256'],
        },
        'panel': {
            'seeds': [42], 'n_seeds': 1,
            'mechanism_checks': {'all_precommitted_checks': True},
        },
        'summary': {
            'status': 'pass', 'gating_failures': 0,
            'mechanism_failures': 0,
        },
    }
    path = Path(protocol['phases']['development']['output'])
    path.write_text(json.dumps(report))
    targets = json.loads(
        (ROOT / 'calibration' / 'primary_model_targets.json').read_text()
    )

    assert len(_verify_development_artifact(
        protocol, targets, protocol_raw
    )) == 64

    report['summary']['status'] = 'fail'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='does not pass'):
        _verify_development_artifact(protocol, targets, protocol_raw)


def test_lifecycle_estimator_keeps_live_orders_as_right_censored_exposure():
    rows = [
        {'lifetime': value, 'censored': False, 'reason': 'fill',
         'ended_tick': value}
        for value in (1.0, 2.0, 3.0, 4.0)
    ] + [
        {'lifetime': 4.0, 'censored': True, 'reason': 'censored',
         'ended_tick': None}
        for _ in range(4)
    ]

    summary = _lifecycle_summary(rows)
    assert summary['completed_median'] == 2.5
    assert _kaplan_meier_median(rows) == 4.0
    assert summary['km_median'] == 4.0
    assert summary['completed_count'] == 4.0
    assert summary['live_censored_count'] == 4.0


def test_unidentified_observables_are_reported_without_pass_fail_labels():
    targets = json.loads(
        (ROOT / 'calibration' / 'primary_model_targets.json').read_text()
    )
    fitter = CalibrationFitter(targets)
    diagnostics = {
        row['observable']: row
        for row in targets['targets']
        if row.get('reported_only')
    }
    expected = {
        'near_touch_depth',
        'impact_curve',
        'dealer_withdrawal_share',
        'funding_liquidity_stress_propagation',
        'amm_volume_share',
        'executed_volume_per_second',
    }

    assert expected.issubset(diagnostics)
    for observable in expected:
        target = diagnostics[observable]
        assert 'target_value' not in target
        assert 'target_range' not in target
        report = fitter.evaluate_metrics(
            {observable: 123.0},
            scenario_name=target['evaluation_scenario'],
        )
        row = next(item for item in report['targets']
                   if item['observable'] == observable)
        assert row['status'] == 'reported_only'
        assert report['summary']['evaluated_targets'] == 0
