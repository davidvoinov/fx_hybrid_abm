"""Mutation tests for publication-signature dependency closure."""
from __future__ import annotations

import json
import hashlib

from calibration import runner as book
from tools.robustness import lp_acceptance as acceptance
from tools.robustness import lp_pnl_corrected as pnl
from tools.robustness import lp_sensitivity as sensitivity
from tools.robustness import lp_survival as survival
from tools.robustness import routing_lp_sensitivity as routing
from tools.robustness import welfare_accounting as welfare
from tools.robustness.signatures import model_signature


def _reset(module, *names):
    for name in names:
        setattr(module, name, None)


def test_book_signature_tracks_a_transitive_protocol_helper(monkeypatch):
    targets = json.loads(
        (book.PROJECT_ROOT / 'calibration' / 'primary_model_targets.json').read_text()
    )
    before = book.book_acceptance_signature(targets)
    with monkeypatch.context() as patch:
        patch.setattr(book, '_protocol_seed_list', book._finite_median)
        after = book.book_acceptance_signature(targets)
    assert after != before

    with monkeypatch.context() as patch:
        patch.setattr(book, 'ACTIVATION_EPSILON', book.ACTIVATION_EPSILON * 10.0)
        after_constant_change = book.book_acceptance_signature(targets)
    assert after_constant_change != before


def test_survival_and_sensitivity_signatures_track_nested_helpers(monkeypatch):
    _reset(survival, '_SURVIVAL_REPORT_SIGNATURE')
    before_report = survival.survival_report_signature()
    with monkeypatch.context() as patch:
        patch.setattr(survival, '_wilson_interval', survival._raw_path)
        _reset(survival, '_SURVIVAL_REPORT_SIGNATURE')
        after_report = survival.survival_report_signature()
    _reset(survival, '_SURVIVAL_REPORT_SIGNATURE')
    assert after_report != before_report

    _reset(sensitivity, '_REPORT_SIGNATURE')
    before_panel = sensitivity.sensitivity_report_signature()
    with monkeypatch.context() as patch:
        # This helper is reached as ``S._calibrated_lp_defaults()`` rather
        # than as a direct global of the sensitivity module.
        patch.setattr(survival, '_calibrated_lp_defaults', survival._raw_path)
        _reset(sensitivity, '_REPORT_SIGNATURE')
        after_panel = sensitivity.sensitivity_report_signature()
    _reset(sensitivity, '_REPORT_SIGNATURE')
    assert after_panel != before_panel


def test_routing_and_pnl_signatures_track_nested_helpers(monkeypatch):
    _reset(routing, '_SENSITIVITY_SIGNATURE')
    before_routing = routing.sensitivity_signature()
    with monkeypatch.context() as patch:
        patch.setattr(routing, '_validate_specifications', routing._finite_or_none)
        _reset(routing, '_SENSITIVITY_SIGNATURE')
        after_routing = routing.sensitivity_signature()
    _reset(routing, '_SENSITIVITY_SIGNATURE')
    assert after_routing != before_routing

    _reset(routing, '_SENSITIVITY_SIGNATURE', '_REPORT_SIGNATURE')
    before_measurement = routing.sensitivity_signature()
    before_report = routing.report_signature()
    with monkeypatch.context() as patch:
        patch.setattr(routing, 'PHASES', tuple(reversed(routing.PHASES)))
        _reset(routing, '_SENSITIVITY_SIGNATURE', '_REPORT_SIGNATURE')
        after_measurement = routing.sensitivity_signature()
        after_report = routing.report_signature()
    _reset(routing, '_SENSITIVITY_SIGNATURE', '_REPORT_SIGNATURE')
    assert after_measurement != before_measurement
    assert after_report != before_report

    _reset(routing, '_REPORT_SIGNATURE')
    before_report_assembly = routing.report_signature()
    with monkeypatch.context() as patch:
        patch.setattr(
            routing,
            'assemble_report_payload',
            routing.summarize_records,
        )
        _reset(routing, '_REPORT_SIGNATURE')
        after_report_assembly = routing.report_signature()
    _reset(routing, '_REPORT_SIGNATURE')
    assert after_report_assembly != before_report_assembly

    _reset(pnl, '_AGGREGATE_REPORT_SIGNATURE')
    before_pnl = pnl.aggregate_report_signature()
    with monkeypatch.context() as patch:
        patch.setattr(pnl, '_finite_or_none', pnl.fee_label)
        _reset(pnl, '_AGGREGATE_REPORT_SIGNATURE')
        after_pnl = pnl.aggregate_report_signature()
    _reset(pnl, '_AGGREGATE_REPORT_SIGNATURE')
    assert after_pnl != before_pnl

    with monkeypatch.context() as patch:
        patch.setattr(pnl, '_SIG', 'a' * 16)
        path_a = pnl.raw_path(5.0, 'endogenous')
        patch.setattr(pnl, '_SIG', 'b' * 16)
        path_b = pnl.raw_path(5.0, 'endogenous')
    assert path_a != path_b


def test_welfare_separates_raw_measurement_from_report_dependencies(monkeypatch):
    _reset(welfare, '_WELFARE_SIGNATURE', '_WELFARE_REPORT_SIGNATURE')
    measurement_before = welfare.welfare_measurement_signature()
    report_before = welfare.welfare_report_signature()

    with monkeypatch.context() as patch:
        patch.setattr(welfare, 'aggregate', welfare.withdrawal_window)
        _reset(welfare, '_WELFARE_SIGNATURE', '_WELFARE_REPORT_SIGNATURE')
        measurement_after_report_change = welfare.welfare_measurement_signature()
        report_after = welfare.welfare_report_signature()
    _reset(welfare, '_WELFARE_SIGNATURE', '_WELFARE_REPORT_SIGNATURE')

    assert measurement_after_report_change == measurement_before
    assert report_after != report_before

    with monkeypatch.context() as patch:
        # ``components`` is imported from the P&L module and was previously
        # absent from the welfare provenance graph.
        patch.setattr(welfare, 'components', welfare.withdrawal_window)
        _reset(welfare, '_WELFARE_SIGNATURE')
        measurement_after_import_change = welfare.welfare_measurement_signature()
    _reset(welfare, '_WELFARE_SIGNATURE', '_WELFARE_REPORT_SIGNATURE')
    assert measurement_after_import_change != measurement_before


def test_acceptance_sha_tracks_every_downstream_signature(monkeypatch):
    before = acceptance.acceptance_code_sha256()
    with monkeypatch.context() as patch:
        patch.setattr(acceptance, 'welfare_report_signature', lambda: '0' * 16)
        after = acceptance.acceptance_code_sha256()
    assert after != before


def test_v3_direct_protocol_uses_one_exact_artifact_manifest_and_no_amendment():
    paths = {
        name: (
            f'output/final/{name}.json' if name.startswith('book_')
            else f'output/resilience/{name}.json'
        )
        for name in acceptance.DIRECT_ARTIFACT_PATH_KEYS
    }
    protocol = {
        'protocol_version': acceptance.DIRECT_PROTOCOL_VERSION,
        'phases': {
            'development': {'output': paths['book_development']},
            'holdout': {'output': paths['book_holdout']},
        },
        'artifact_paths': paths,
        'lp_frozen_signatures': acceptance._current_lp_frozen_signatures(),
    }
    digests = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in acceptance.DIRECT_INPUT_ARTIFACT_KEYS
    }
    acceptance._verify_direct_protocol(protocol, digests)

    missing = json.loads(json.dumps(protocol))
    missing['artifact_paths'].pop('pnl')
    try:
        acceptance._verify_direct_protocol(missing, digests)
        missing_rejected = False
    except ValueError:
        missing_rejected = True
    assert missing_rejected

    stale = json.loads(json.dumps(protocol))
    stale['lp_frozen_signatures']['pnl_report_signature'] = 'stale'
    try:
        acceptance._verify_direct_protocol(stale, digests)
        stale_rejected = False
    except ValueError:
        stale_rejected = True
    assert stale_rejected


def test_model_signature_covers_every_project_model_module(tmp_path):
    roots = []
    for value in ('return 1\n', 'return 2\n'):
        root = tmp_path / str(len(roots))
        (root / 'AgentBasedModel').mkdir(parents=True)
        (root / 'calibration').mkdir()
        (root / 'AgentBasedModel' / 'trajectory.py').write_text(
            'def trajectory():\n    ' + value
        )
        (root / 'main.py').write_text('from AgentBasedModel import trajectory\n')
        (root / 'calibration' / 'primary_model.json').write_text(
            json.dumps({'cli_defaults': {'n_mm': 5}, 'notes': 'ignored'})
        )
        roots.append(root)

    assert model_signature(roots[0]) != model_signature(roots[1])
