#!/usr/bin/env python3
"""Narrow publication gate for the endogenous LP block.

This validator checks completeness, accounting identities, mechanism
activation and estimand scope.  It deliberately does *not* require positive LP
profit, a positive user-benefit estimate, or a preferred survival rate: those
are empirical outcomes, not software/calibration integrity conditions.

Example
-------
    python3 tools/robustness/lp_acceptance.py \
      --protocol calibration/final_protocol.json \
      --amendment calibration/lp_amendment_v2a.json \
      --book-development-json output/final/book_development_v2_300.json \
      --book-holdout-json output/final/book_holdout_v2_300.json \
      --survival-json output/resilience/lp_survival_primary_endo_300.json \
      --funded-survival-json output/resilience/lp_survival_entry_validation_60.json \
      --rebate-survival-json output/resilience/lp_survival_rebate_validation_60.json \
      --sensitivity-json output/resilience/lp_sensitivity_60.json \
      --routing-sensitivity-json output/resilience/routing_lp_sensitivity_60.json \
      --pnl-json output/resilience/lp_pnl_endogenous_300.json \
      --welfare-json output/resilience/welfare_primary_endo_300.json \
                     output/resilience/welfare_loss_guarantee_300.json \
      --output output/resilience/lp_acceptance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calibration.runner import _protocol_seed_list, _verify_protocol
from tools.robustness.signatures import model_signature
from tools.robustness.lp_survival import (
    N_ITER as SURVIVAL_N_ITER,
    _calibrated_lp_defaults as calibrated_lp_defaults,
    survival_report_signature,
    survival_signature,
)
from tools.robustness.lp_sensitivity import (
    _specifications as lp_specifications,
    sensitivity_report_signature,
)
from tools.robustness.lp_pnl_corrected import (
    N_ITER as PNL_N_ITER,
    PRESET as LP_PRESET,
    aggregate_report_signature as pnl_report_signature,
    run_signature as pnl_measurement_signature,
)
from tools.robustness.welfare_accounting import (
    N_ITER as WELFARE_N_ITER,
    SUMMARY_METRICS as WELFARE_SUMMARY_METRICS,
    welfare_measurement_signature,
    welfare_report_signature,
)
from tools.robustness.routing_lp_sensitivity import (
    N_ITER as ROUTING_N_ITER,
    PHASE_METRICS as ROUTING_PHASE_METRICS,
    PHASES as ROUTING_PHASES,
    PRECOMMIT_PROTOCOL as ROUTING_PROTOCOL,
    RUN_METRICS as ROUTING_RUN_METRICS,
    _specifications as routing_specifications,
    protocol_signature as routing_protocol_signature,
    report_signature as routing_report_signature,
    sensitivity_signature as routing_measurement_signature,
)


CANONICAL_PROTOCOL = os.path.join(ROOT, 'calibration', 'final_protocol.json')
CANONICAL_AMENDMENT = os.path.join(ROOT, 'calibration', 'lp_amendment_v2a.json')
TARGET_MATRIX = os.path.join(ROOT, 'calibration', 'primary_model_targets.json')

DIRECT_PROTOCOL_VERSION = 'book-lp-final-v3'
DIRECT_ARTIFACT_PATH_KEYS = {
    'book_development',
    'book_holdout',
    'primary_survival',
    'funded_survival',
    'rebate_survival',
    'sensitivity',
    'routing_sensitivity',
    'pnl',
    'welfare_private_primary',
    'welfare_full_loss_guarantee_upper_bound',
    'acceptance',
}
DIRECT_INPUT_ARTIFACT_KEYS = DIRECT_ARTIFACT_PATH_KEYS - {'acceptance'}

AMENDMENT_VERSION = 'lp-amendment-v2a'
AMENDMENT_TYPE = 'post_run_non_discretionary_zero_denominator_correction'
AMENDMENT_ARTIFACT_PATHS = {
    'book_development': 'output/final/book_development_v2_300.json',
    'book_holdout': 'output/final/book_holdout_v2_300.json',
    'primary_survival': 'output/resilience/lp_survival_final_holdout_300.json',
    'funded_survival': 'output/resilience/lp_survival_entry_validation_60.json',
    'rebate_survival': 'output/resilience/lp_survival_rebate_validation_60.json',
    'sensitivity': 'output/resilience/lp_sensitivity_final_60.json',
    'routing_original': (
        'output/resilience/routing_lp_sensitivity_pre_amendment_failed_60.json'
    ),
    'routing_effective': (
        'output/resilience/routing_lp_sensitivity_final_60.json'
    ),
    'pnl': 'output/resilience/lp_pnl_final_holdout_300.json',
    'welfare_private_primary': (
        'output/resilience/welfare_primary_final_holdout_300.json'
    ),
    'welfare_full_loss_guarantee_upper_bound': (
        'output/resilience/welfare_loss_guarantee_final_holdout_300.json'
    ),
    'failed_acceptance': (
        'output/resilience/lp_acceptance_pre_amendment_failed.json'
    ),
}
AMENDMENT_CHANGE_CONTROL = {
    'timing': 'post_run',
    'discretion': 'non_discretionary',
    'reason': 'zero_denominator_domain_correction',
    'model_changed': False,
    'acceptance_thresholds_changed': False,
    'raw_measurements_changed': False,
    'estimand_numerators_changed': False,
    'estimand_denominators_changed': False,
    'seed_or_configuration_design_changed': False,
    'permitted_changes': [
        'routing_report_zero_denominator_classification',
        'lp_acceptance_zero_denominator_support_check',
    ],
}
AMENDMENT_CORRECTION = {
    'estimand': 'amm_arbitrage_share_of_amm_execution',
    'numerator': 'amm_arbitrage_volume_base',
    'denominator': 'amm_execution_volume_base',
    'zero_denominator_rule': (
        'denominator_equals_zero_and_numerator_equals_zero_requires_null_ratio'
    ),
    'positive_denominator_rule': (
        'finite_ratio_equals_numerator_divided_by_denominator'
    ),
    'acceptance_identity': (
        'n_finite_plus_n_zero_denominator_equals_n_observations_equals_requested'
        '_and_n_invalid_equals_zero'
    ),
}


def acceptance_code_sha256() -> str:
    """Bind acceptance to this file and every downstream signature graph."""
    downstream = (
        survival_signature(),
        survival_report_signature(),
        sensitivity_report_signature(),
        routing_measurement_signature(),
        routing_report_signature(),
        routing_protocol_signature(),
        pnl_measurement_signature(),
        pnl_report_signature(),
        welfare_measurement_signature(),
        welfare_report_signature(),
    )
    digest = hashlib.sha256()
    digest.update(b'lp-acceptance-transitive-v1\0')
    with open(__file__, 'rb') as handle:
        digest.update(handle.read())
    for value in downstream:
        digest.update(b'\0')
        digest.update(value.encode('ascii'))
    return digest.hexdigest()


def _consecutive_seeds(row: dict, label: str) -> list[int]:
    try:
        start = int(row['seed_start'])
        count = int(row['seed_count'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{label} has no valid seed_start/seed_count') from exc
    if count <= 0:
        raise ValueError(f'{label} seed_count must be positive')
    return list(range(start, start + count))


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _load(path: str) -> dict:
    with open(path, encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return value


def _load_with_sha256(path: str) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return value, hashlib.sha256(raw).hexdigest()


def _current_lp_frozen_signatures() -> dict[str, str]:
    return {
        'survival_measurement_signature': survival_signature(),
        'survival_report_signature': survival_report_signature(),
        'sensitivity_report_signature': sensitivity_report_signature(),
        'routing_measurement_signature': routing_measurement_signature(),
        'routing_report_signature': routing_report_signature(),
        'routing_protocol_signature': routing_protocol_signature(),
        'pnl_measurement_signature': pnl_measurement_signature(),
        'pnl_report_signature': pnl_report_signature(),
        'welfare_measurement_signature': welfare_measurement_signature(),
        'welfare_report_signature': welfare_report_signature(),
        'lp_acceptance_code_sha256': acceptance_code_sha256(),
    }


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f'{label} must have exact keys {sorted(expected)}; got {actual}'
        )


def _direct_artifact_paths(protocol: dict) -> dict[str, str]:
    """Validate and return the v3 single-source artifact manifest."""
    paths = protocol.get('artifact_paths')
    _require_exact_keys(paths, DIRECT_ARTIFACT_PATH_KEYS, 'artifact_paths')
    normalized = {}
    for name, value in paths.items():
        if not isinstance(value, str) or not value or '\\' in value:
            raise ValueError(f'artifact path {name} must be a POSIX relative path')
        candidate = Path(value)
        if (candidate.is_absolute() or '..' in candidate.parts
                or candidate.suffix != '.json'):
            raise ValueError(f'artifact path {name} is not a safe relative JSON path')
        normalized_value = candidate.as_posix()
        expected_prefix = 'output/final/' if name.startswith('book_') \
            else 'output/resilience/'
        if not normalized_value.startswith(expected_prefix):
            raise ValueError(
                f'artifact path {name} must be below {expected_prefix}'
            )
        normalized[name] = normalized_value
    if len(set(normalized.values())) != len(normalized):
        raise ValueError('artifact_paths must contain unique paths')
    phases = protocol.get('phases') or {}
    if ((phases.get('development') or {}).get('output')
            != normalized['book_development']
            or (phases.get('holdout') or {}).get('output')
            != normalized['book_holdout']):
        raise ValueError('book phase outputs must equal artifact_paths entries')
    return normalized


def _verify_direct_protocol(protocol: dict,
                            artifact_sha256: dict[str, str]) -> None:
    """Verify v3 provenance without inheriting the historical v2 amendment."""
    if protocol.get('protocol_version') != DIRECT_PROTOCOL_VERSION:
        raise ValueError('direct LP acceptance requires the v3 protocol')
    _direct_artifact_paths(protocol)
    if protocol.get('lp_frozen_signatures') != _current_lp_frozen_signatures():
        raise ValueError('v3 protocol does not freeze current LP signatures')
    if set(artifact_sha256) != DIRECT_INPUT_ARTIFACT_KEYS:
        raise ValueError('v3 artifact SHA manifest has incomplete or extra keys')
    if not all(_valid_sha256(value) for value in artifact_sha256.values()):
        raise ValueError('v3 artifact SHA manifest contains a malformed digest')


def _routing_reports_differ_only_by_zero_denominator_correction(
        original: dict, effective: dict) -> bool:
    """Prove that the amended report changes no measured routing quantity."""
    original = json.loads(json.dumps(original))
    effective = json.loads(json.dumps(effective))
    original.pop('report_signature', None)
    effective.pop('report_signature', None)
    original_results = original.pop('results', None)
    effective_results = effective.pop('results', None)
    if original != effective:
        return False
    if not isinstance(original_results, list) or not isinstance(
            effective_results, list):
        return False
    if len(original_results) != len(effective_results):
        return False

    allowed_summary_keys = {
        'n_finite', 'mean', 'median', 'p10', 'p90',
        'n_zero_denominator', 'n_invalid', 'n_observations',
    }
    base_summary_keys = {'n_finite', 'mean', 'median', 'p10', 'p90'}
    for original_row, effective_row in zip(original_results, effective_results):
        original_row = dict(original_row)
        effective_row = dict(effective_row)
        original_summary = original_row.pop('summary', None)
        effective_summary = effective_row.pop('summary', None)
        if original_row != effective_row:
            return False
        if not isinstance(original_summary, dict) or not isinstance(
                effective_summary, dict):
            return False
        for phase in ROUTING_PHASES:
            field = f'amm_arbitrage_share_of_amm_execution_{phase}'
            old = original_summary.pop(field, None)
            new = effective_summary.pop(field, None)
            if not isinstance(old, dict) or not isinstance(new, dict):
                return False
            if set(old) != base_summary_keys or set(new) != allowed_summary_keys:
                return False
            if any(old.get(key) != new.get(key) for key in base_summary_keys):
                return False
            requested = int(effective.get(
                'requested_seeds_per_specification', -1
            ))
            if (
                int(new.get('n_finite', -1))
                + int(new.get('n_zero_denominator', -1)) != requested
                or int(new.get('n_invalid', -1)) != 0
                or int(new.get('n_observations', -1)) != requested
            ):
                return False
        if original_summary != effective_summary:
            return False
    return True


def _verify_amendment(
        amendment: dict, amendment_sha256: str, protocol: dict,
        protocol_sha256: str, artifact_sha256: dict[str, str],
        original_routing: dict, effective_routing: dict,
        failed_acceptance: dict) -> None:
    """Validate the narrow post-run amendment and its immutable evidence."""
    _require_exact_keys(amendment, {
        'amendment_version', 'amendment_type', 'base_protocol',
        'change_control', 'correction', 'original_frozen_signatures',
        'effective_frozen_signatures', 'artifact_bindings',
    }, 'LP amendment')
    if amendment.get('amendment_version') != AMENDMENT_VERSION:
        raise ValueError('unexpected LP amendment version')
    if amendment.get('amendment_type') != AMENDMENT_TYPE:
        raise ValueError('unexpected LP amendment type')
    if not _valid_sha256(amendment_sha256):
        raise ValueError('LP amendment has no valid SHA-256')

    base = amendment.get('base_protocol')
    _require_exact_keys(base, {'path', 'version', 'sha256'}, 'base_protocol')
    if base != {
        'path': 'calibration/final_protocol.json',
        'version': protocol.get('protocol_version'),
        'sha256': protocol_sha256,
    }:
        raise ValueError('LP amendment does not bind the exact base protocol')
    if not _valid_sha256(base.get('sha256')):
        raise ValueError('base protocol SHA-256 is malformed')
    if amendment.get('change_control') != AMENDMENT_CHANGE_CONTROL:
        raise ValueError('LP amendment change-control declaration is not exact')
    if amendment.get('correction') != AMENDMENT_CORRECTION:
        raise ValueError('LP amendment zero-denominator rule is not exact')

    original_signatures = protocol.get('lp_frozen_signatures') or {}
    if amendment.get('original_frozen_signatures') != original_signatures:
        raise ValueError('LP amendment does not preserve original signatures')
    current_signatures = _current_lp_frozen_signatures()
    if amendment.get('effective_frozen_signatures') != current_signatures:
        raise ValueError('LP amendment does not freeze current effective code')
    for name, original_value in original_signatures.items():
        if name not in {'routing_report_signature',
                        'lp_acceptance_code_sha256'}:
            if current_signatures.get(name) != original_value:
                raise ValueError(
                    f'LP amendment illegally changes frozen signature {name}'
                )
    if (
        current_signatures['routing_report_signature']
        == original_signatures.get('routing_report_signature')
        or current_signatures['lp_acceptance_code_sha256']
        == original_signatures.get('lp_acceptance_code_sha256')
    ):
        raise ValueError('LP amendment does not identify both permitted changes')

    bindings = amendment.get('artifact_bindings')
    _require_exact_keys(
        bindings, set(AMENDMENT_ARTIFACT_PATHS), 'artifact_bindings'
    )
    if set(artifact_sha256) != set(AMENDMENT_ARTIFACT_PATHS):
        raise ValueError('internal LP artifact manifest is incomplete')
    for name, expected_path in AMENDMENT_ARTIFACT_PATHS.items():
        binding = bindings.get(name)
        _require_exact_keys(binding, {'path', 'sha256'}, f'artifact {name}')
        if binding.get('path') != expected_path:
            raise ValueError(f'LP amendment uses a non-canonical path for {name}')
        if not _valid_sha256(binding.get('sha256')):
            raise ValueError(f'LP amendment has malformed SHA-256 for {name}')
        if binding.get('sha256') != artifact_sha256.get(name):
            raise ValueError(f'LP amendment hash mismatch for {name}')

    frozen_model = (protocol.get('frozen_hashes') or {}).get(
        'simulation_model_signature'
    )
    if not (
        original_routing.get('model_signature') == frozen_model
        == effective_routing.get('model_signature')
        and original_routing.get('measurement_signature')
        == original_signatures.get('routing_measurement_signature')
        == effective_routing.get('measurement_signature')
        and original_routing.get('protocol_signature')
        == original_signatures.get('routing_protocol_signature')
        == effective_routing.get('protocol_signature')
        and original_routing.get('report_signature')
        == original_signatures.get('routing_report_signature')
        and effective_routing.get('report_signature')
        == current_signatures.get('routing_report_signature')
    ):
        raise ValueError('routing evidence does not preserve model/raw protocol')
    if not _routing_reports_differ_only_by_zero_denominator_correction(
            original_routing, effective_routing):
        raise ValueError(
            'routing amendment changes more than zero-denominator reporting'
        )

    failed_checks = failed_acceptance.get('checks') or []
    failed_names = [
        row.get('name') for row in failed_checks if row.get('passed') is False
    ]
    if not (
        failed_acceptance.get('passed') is False
        and failed_acceptance.get('protocol_version')
        == protocol.get('protocol_version')
        and failed_acceptance.get('protocol_sha256') == protocol_sha256
        and failed_acceptance.get('model_signature') == frozen_model
        and failed_acceptance.get('lp_acceptance_code_sha256')
        == original_signatures.get('lp_acceptance_code_sha256')
        and failed_names
        == ['routing_customer_volume_count_active_tick_and_arb_are_separate']
    ):
        raise ValueError('failed acceptance evidence is not the exact prior gate')


def validate(protocol: dict, protocol_sha256: str,
             book_development: dict, book_development_sha256: str,
             book_holdout: dict,
             primary_survival: dict, funded_survival: dict,
             rebate_survival: dict, sensitivity: dict,
             routing_sensitivity: dict, pnl: dict,
             welfare: list[dict], *, artifact_sha256: dict[str, str],
             amendment: Optional[dict] = None,
             amendment_sha256: Optional[str] = None,
             original_routing: Optional[dict] = None,
             failed_acceptance: Optional[dict] = None) -> dict:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: Any, detail: str = '') -> None:
        checks.append({
            'name': name,
            'passed': bool(condition),
            'detail': detail,
        })

    def exact_zero(value: Any) -> bool:
        return type(value) is int and value == 0

    def clean_raw_provenance(payload: dict) -> bool:
        return (
            payload.get('raw_provenance_clean') is True
            and exact_zero(payload.get('invalid_or_stale_records_ignored'))
        )

    with open(TARGET_MATRIX, encoding='utf-8') as handle:
        target_payload = json.load(handle)
    # This is deliberately not a best-effort warning.  A changed runtime,
    # target matrix or acceptance definition invalidates the frozen protocol
    # and therefore every purported final artifact until the protocol is
    # explicitly rotated.
    book_development_seeds = _verify_protocol(
        protocol, target_payload, 'development'
    )
    book_holdout_seeds = _verify_protocol(protocol, target_payload, 'holdout')
    lp_design = protocol.get('lp_design') or {}
    final_row = lp_design.get('final_inference_panel') or {}
    development_row = lp_design.get('development_robustness_panel') or {}
    entry_row = lp_design.get('entry_identification_arm') or {}
    rebate_row = lp_design.get('loss_rebate_identification_arm') or {}
    pnl_row = lp_design.get('pnl_panel') or {}
    welfare_row = lp_design.get('welfare_incidence_panels') or {}
    final_seeds = _consecutive_seeds(final_row, 'LP final inference panel')
    development_seeds = _consecutive_seeds(
        development_row, 'LP development robustness panel'
    )
    entry_seeds = _consecutive_seeds(entry_row, 'LP entry identification arm')
    rebate_seeds = _consecutive_seeds(
        rebate_row, 'LP loss-rebate identification arm'
    )
    pnl_seeds = _consecutive_seeds(pnl_row, 'LP P&L panel')
    welfare_seeds = _consecutive_seeds(welfare_row, 'LP welfare panels')

    current_signature = model_signature(ROOT)
    frozen_lp = protocol.get('lp_frozen_signatures') or {}
    current_lp = _current_lp_frozen_signatures()
    if book_development_sha256 != artifact_sha256.get('book_development'):
        raise ValueError('book development SHA disagrees with artifact manifest')
    direct_v3 = protocol.get('protocol_version') == DIRECT_PROTOCOL_VERSION
    if direct_v3:
        if any(value is not None for value in (
                amendment, amendment_sha256, original_routing,
                failed_acceptance)):
            raise ValueError('v3 direct acceptance forbids a historical amendment')
        _verify_direct_protocol(protocol, artifact_sha256)
        check(
            'protocol_directly_freezes_current_lp_signatures',
            frozen_lp == current_lp,
        )
    else:
        if not all(value is not None for value in (
                amendment, amendment_sha256, original_routing,
                failed_acceptance)):
            raise ValueError('pre-v3 acceptance requires its historical amendment')
        _verify_amendment(
            amendment, amendment_sha256, protocol, protocol_sha256,
            artifact_sha256, original_routing, routing_sensitivity,
            failed_acceptance,
        )
        check(
            'base_protocol_signatures_are_preserved_by_the_amendment',
            amendment.get('original_frozen_signatures') == frozen_lp,
        )
        check(
            'amendment_freezes_current_effective_lp_reporting_and_acceptance_code',
            amendment.get('effective_frozen_signatures') == current_lp,
        )
    artifacts = [primary_survival, funded_survival, rebate_survival, sensitivity,
                 routing_sensitivity, pnl, *welfare]
    artifact_signatures = [str(item.get('model_signature', '')) for item in artifacts]
    check(
        'all_artifacts_use_current_model_signature',
        bool(artifact_signatures)
        and all(value == current_signature for value in artifact_signatures),
        f'current={current_signature}; artifacts={artifact_signatures}',
    )
    check(
        'protocol_lp_final_panel_is_the_book_holdout_panel',
        final_seeds == book_holdout_seeds
        and pnl_seeds == book_holdout_seeds
        and welfare_seeds == book_holdout_seeds,
    )
    check(
        'protocol_lp_development_panels_are_disjoint_from_holdout',
        development_seeds == entry_seeds == rebate_seeds
        and not set(development_seeds).intersection(book_holdout_seeds)
        and not set(book_development_seeds).intersection(book_holdout_seeds),
    )
    check(
        'protocol_horizons_match_current_runners',
        int(final_row.get('n_iter', -1)) == SURVIVAL_N_ITER
        and int(development_row.get('n_iter', -1)) == SURVIVAL_N_ITER
        and int(entry_row.get('n_iter', -1)) == SURVIVAL_N_ITER
        and int(rebate_row.get('n_iter', -1)) == SURVIVAL_N_ITER
        and int(pnl_row.get('n_iter', -1)) == PNL_N_ITER
        and int(welfare_row.get('n_iter', -1)) == WELFARE_N_ITER,
    )
    check(
        'protocol_uses_the_current_lp_crisis_scenario',
        lp_design.get('scenario') == LP_PRESET,
    )
    check(
        'protocol_declares_every_identification_control_and_claim_limit',
        set(lp_design.get('mechanism_controls') or ()) == {
            'zero_support_private_sustainability',
            'posted_rate_entry_activation',
            'loss_rebate_exit_response',
            'entry_margin_separation',
        }
        and (lp_design.get('claim_limits') or {}).get(
            'behavioural_parameters_externally_calibrated') is False
        and (lp_design.get('claim_limits') or {}).get(
            'total_welfare_identified') is False
        and (lp_design.get('claim_limits') or {}).get(
            'customer_routing_and_arbitrage_flow_are_separate_estimands')
        is True,
    )

    development_provenance = book_development.get('provenance', {}) or {}
    development_panel = book_development.get('panel', {}) or {}
    development_summary = book_development.get('summary', {}) or {}
    development_mechanisms = development_panel.get('mechanism_checks', {}) or {}
    frozen = protocol.get('frozen_hashes', {}) or {}
    check(
        'book_development_is_bound_to_the_exact_protocol',
        development_provenance.get('protocol_sha256') == protocol_sha256
        and development_provenance.get('protocol_version')
        == protocol.get('protocol_version')
        and development_provenance.get('protocol_phase') == 'development'
        and development_provenance.get('seed_commitment_exact_match') is True,
    )
    check(
        'book_development_uses_current_frozen_signatures',
        development_provenance.get('simulation_model_signature')
        == current_signature
        == frozen.get('simulation_model_signature')
        and development_provenance.get('book_acceptance_signature')
        == frozen.get('book_acceptance_signature')
        and development_provenance.get('primary_runtime_sha256')
        == frozen.get('primary_runtime_sha256')
        and development_provenance.get('target_matrix_sha256')
        == frozen.get('target_matrix_sha256')
        and development_provenance.get('aggregation_mode')
        == 'fresh_simulation',
    )
    check(
        'book_development_has_exact_seed_manifest',
        development_panel.get('seeds') == book_development_seeds
        and int(development_panel.get('n_seeds', -1))
        == len(book_development_seeds),
    )
    check(
        'book_development_passes_before_holdout_is_opened',
        development_summary.get('status') == 'pass'
        and int(development_summary.get('gating_failures', -1)) == 0
        and int(development_summary.get('mechanism_failures', -1)) == 0
        and bool(development_mechanisms)
        and all(value is True for value in development_mechanisms.values()),
    )

    book_provenance = book_holdout.get('provenance', {}) or {}
    book_panel = book_holdout.get('panel', {}) or {}
    book_summary = book_holdout.get('summary', {}) or {}
    book_mechanisms = book_panel.get('mechanism_checks', {}) or {}
    check(
        'book_holdout_is_bound_to_the_exact_protocol',
        bool(protocol_sha256)
        and book_provenance.get('protocol_sha256') == protocol_sha256
        and book_provenance.get('protocol_version') == protocol.get('protocol_version')
        and book_provenance.get('protocol_phase') == 'holdout'
        and book_provenance.get('seed_commitment_exact_match') is True,
    )
    check(
        'book_holdout_was_authorized_by_the_exact_passing_development_artifact',
        bool(book_development_sha256)
        and book_provenance.get('development_gate_verified') is True
        and book_provenance.get('development_gate_sha256')
        == book_development_sha256,
    )
    check(
        'book_holdout_uses_current_frozen_signatures',
        book_provenance.get('simulation_model_signature') == current_signature
        and book_provenance.get('simulation_model_signature')
        == frozen.get('simulation_model_signature')
        and book_provenance.get('book_acceptance_signature')
        == frozen.get('book_acceptance_signature')
        and book_provenance.get('primary_runtime_sha256')
        == frozen.get('primary_runtime_sha256')
        and book_provenance.get('target_matrix_sha256')
        == frozen.get('target_matrix_sha256')
        and book_provenance.get('aggregation_mode') == 'fresh_simulation',
    )
    check(
        'book_holdout_has_exact_seed_manifest',
        book_panel.get('seeds') == book_holdout_seeds
        and int(book_panel.get('n_seeds', -1)) == len(book_holdout_seeds),
    )
    check(
        'book_holdout_passes_all_book_and_mechanism_gates',
        book_summary.get('status') == 'pass'
        and int(book_summary.get('gating_failures', -1)) == 0
        and int(book_summary.get('mechanism_failures', -1)) == 0
        and bool(book_mechanisms)
        and all(value is True for value in book_mechanisms.values()),
    )

    check(
        'survival_artifacts_use_current_measurement_and_report_signatures',
        all(item.get('measurement_signature') == survival_signature()
            and item.get('report_signature') == survival_report_signature()
            for item in (primary_survival, funded_survival, rebate_survival)),
    )
    check(
        'survival_artifacts_have_clean_raw_provenance',
        all(clean_raw_provenance(item) for item in (
            primary_survival, funded_survival, rebate_survival
        )),
    )
    check(
        'sensitivity_uses_current_measurement_and_report_signatures',
        sensitivity.get('measurement_signature') == survival_signature()
        and sensitivity.get('report_signature') == sensitivity_report_signature(),
    )
    check(
        'sensitivity_has_clean_raw_provenance',
        clean_raw_provenance(sensitivity)
        and all(
            row.get('raw_provenance_clean') is True
            and exact_zero(row.get('invalid_or_stale_records_ignored'))
            and exact_zero(row.get('stale_records_ignored'))
            for row in (sensitivity.get('results') or [])
        ),
    )
    check(
        'pnl_uses_current_measurement_and_report_signatures',
        pnl.get('measurement_signature') == pnl_measurement_signature()
        and pnl.get('report_signature') == pnl_report_signature(),
    )
    check(
        'welfare_uses_current_measurement_and_report_signatures',
        bool(welfare) and all(
            item.get('measurement_signature') == welfare_measurement_signature()
            and item.get('report_signature') == welfare_report_signature()
            for item in welfare
        ),
    )

    def survival_pools(payload: dict) -> dict:
        return payload.get('summary', {}).get('pools', {}) or {}

    primary_pools = survival_pools(primary_survival)
    check('primary_survival_complete', primary_survival.get('complete') is True,
          f"available={primary_survival.get('available_seeds')}, "
          f"requested={primary_survival.get('requested_seeds')}")
    primary_config = primary_survival.get('summary', {}).get('configuration', {})
    check(
        'primary_survival_has_exact_holdout_seed_manifest',
        int(primary_survival.get('requested_seed_start', -1)) == final_seeds[0]
        and int(primary_survival.get('requested_seeds', -1)) == len(final_seeds)
        and primary_survival.get('requested_seed_list') == final_seeds
        and int(primary_survival.get('available_seeds', -1)) == len(final_seeds),
    )
    expected_primary_config = calibrated_lp_defaults()
    expected_primary_config['n_iter'] = int(final_row['n_iter'])
    check(
        'primary_survival_is_exact_zero_support_endogenous_specification',
        primary_config == expected_primary_config,
    )
    check(
        'primary_survival_summary_counts_every_holdout_seed',
        int(primary_survival.get('summary', {}).get('n_seed_records', -1))
        == len(final_seeds)
        and bool(primary_pools)
        and all(int(row.get('n_seeds', -1)) == len(final_seeds)
                for row in primary_pools.values()),
    )
    check('primary_survival_has_pool_results', bool(primary_pools))
    check(
        'primary_gross_event_identities_hold',
        bool(primary_pools) and all(
            row.get('gross_event_identity_pass_rate') == 1.0
            for row in primary_pools.values()
        ),
    )
    check(
        'primary_claim_supply_identities_hold',
        bool(primary_pools) and all(
            row.get('claim_supply_identity_end_pass_rate') == 1.0
            for row in primary_pools.values()
        ),
    )
    allowed_activation = {
        'observed', 'right_censored_no_activation_by_horizon',
        'no_population_at_risk',
    }
    check(
        'primary_entry_nonactivation_is_not_silently_interpreted',
        bool(primary_pools) and all(
            row.get('entry_activation_status') in allowed_activation
            for row in primary_pools.values()
        ),
    )
    check(
        'primary_has_an_entry_population_at_risk',
        bool(primary_pools) and all(
            int(row.get('new_entry_risk_set_providers', 0) or 0) > 0
            for row in primary_pools.values()
        ),
    )

    funded_pools = survival_pools(funded_survival)
    funded_config = funded_survival.get('summary', {}).get('configuration', {})
    expected_funded_config = dict(expected_primary_config)
    expected_funded_config.update(
        n_iter=int(entry_row['n_iter']),
        subsidy_rate=float(entry_row.get('subsidy_rate_per_tick')),
        loss_rebate_fraction=0.0,
    )
    check('funded_activation_arm_complete', funded_survival.get('complete') is True)
    check(
        'funded_activation_arm_has_exact_development_seed_manifest',
        int(funded_survival.get('requested_seed_start', -1)) == entry_seeds[0]
        and int(funded_survival.get('requested_seeds', -1)) == len(entry_seeds)
        and funded_survival.get('requested_seed_list') == entry_seeds
        and int(funded_survival.get('available_seeds', -1)) == len(entry_seeds)
        and int(funded_config.get('n_iter', -1)) == int(entry_row['n_iter']),
    )
    check('funded_activation_arm_has_exact_posted_rate_support',
          funded_config == expected_funded_config)
    check(
        'funded_activation_summary_counts_every_development_seed',
        int(funded_survival.get('summary', {}).get('n_seed_records', -1))
        == len(entry_seeds)
        and bool(funded_pools)
        and all(int(row.get('n_seeds', -1)) == len(entry_seeds)
                for row in funded_pools.values()),
    )
    check(
        'full_simulator_observes_funded_entry_or_reentry',
        bool(funded_pools) and any(
            bool(row.get('entry_activation_observed'))
            and (int(row.get('gross_new_entry_events', 0))
                 + int(row.get('gross_reentry_events', 0)) > 0)
            for row in funded_pools.values()
        ),
        'This is a mechanism-identification arm, not a positive-profit target.',
    )
    check(
        'funded_arm_accounting_identities_hold',
        bool(funded_pools) and all(
            row.get('gross_event_identity_pass_rate') == 1.0
            and row.get('claim_supply_identity_end_pass_rate') == 1.0
            for row in funded_pools.values()
        ),
    )

    rebate_pools = survival_pools(rebate_survival)
    rebate_config = rebate_survival.get('summary', {}).get('configuration', {})
    expected_rebate_config = dict(expected_primary_config)
    expected_rebate_config.update(
        n_iter=int(rebate_row['n_iter']),
        subsidy_rate=float(rebate_row.get('subsidy_rate_per_tick', 0.0)),
        loss_rebate_fraction=float(rebate_row.get('loss_rebate_fraction')),
    )
    check('loss_rebate_identification_arm_complete',
          rebate_survival.get('complete') is True)
    check(
        'loss_rebate_arm_has_exact_development_seed_manifest',
        int(rebate_survival.get('requested_seed_start', -1)) == rebate_seeds[0]
        and int(rebate_survival.get('requested_seeds', -1)) == len(rebate_seeds)
        and rebate_survival.get('requested_seed_list') == rebate_seeds
        and int(rebate_survival.get('available_seeds', -1)) == len(rebate_seeds)
        and int(rebate_config.get('n_iter', -1)) == int(rebate_row['n_iter']),
    )
    check('loss_rebate_arm_is_an_isolated_full_loss_rebate',
          rebate_config == expected_rebate_config)
    check(
        'loss_rebate_summary_counts_every_development_seed',
        int(rebate_survival.get('summary', {}).get('n_seed_records', -1))
        == len(rebate_seeds)
        and bool(rebate_pools)
        and all(int(row.get('n_seeds', -1)) == len(rebate_seeds)
                for row in rebate_pools.values()),
    )
    check(
        'loss_rebate_arm_accounting_identities_hold',
        bool(rebate_pools) and all(
            row.get('gross_event_identity_pass_rate') == 1.0
            and row.get('claim_supply_identity_end_pass_rate') == 1.0
            for row in rebate_pools.values()
        ),
    )

    results = sensitivity.get('results', []) or []
    labels = {str(row.get('label')) for row in results}
    expected_joint = {
        'joint_patience_145_ewma_0.01',
        'joint_patience_145_ewma_0.05',
        'joint_patience_580_ewma_0.01',
        'joint_patience_580_ewma_0.05',
    }
    expected_labels = expected_joint | {
        'baseline', 'patience_145', 'patience_580',
        'ewma_alpha_0.01', 'ewma_alpha_0.05',
        'response_scale_0.5x', 'response_scale_2x',
        'outside_option_0.5x', 'outside_option_2x',
        'entry_margin_0', 'entry_margin_0.5',
    }
    requested_per_spec = int(
        sensitivity.get('requested_seeds_per_specification', 0) or 0
    )
    check('sensitivity_panel_complete', sensitivity.get('complete') is True)
    check(
        'sensitivity_has_exact_development_seed_manifest',
        int(sensitivity.get('requested_seed_start', -1)) == development_seeds[0]
        and requested_per_spec == len(development_seeds)
        and sensitivity.get('requested_seeds') == development_seeds,
    )
    check(
        'sensitivity_design_declares_joint_grid',
        sensitivity.get('design')
        == 'oat_plus_joint_patience_by_ewma_around_primary_lp_specification',
    )
    check('joint_patience_ewma_corners_complete', expected_joint <= labels,
          f'missing={sorted(expected_joint - labels)}')
    check('sensitivity_panel_has_exact_declared_cells', labels == expected_labels,
          f'missing={sorted(expected_labels - labels)}; '
          f'extra={sorted(labels - expected_labels)}')
    check(
        'every_sensitivity_cell_has_requested_seeds',
        bool(results) and requested_per_spec > 0 and all(
            int(row.get('available_seeds', -1)) == requested_per_spec
            and int(row.get('summary', {}).get('n_seed_records', -1))
            == requested_per_spec
            and all(
                int(pool.get('n_seeds', -1)) == requested_per_spec
                for pool in (row.get('summary', {}).get('pools', {}) or {}).values()
            )
            for row in results
        ),
    )
    expected_sensitivity_rows = dict(lp_specifications(SURVIVAL_N_ITER))
    check(
        'sensitivity_cells_use_exact_current_configurations',
        labels == set(expected_sensitivity_rows)
        and all(
            (row.get('summary', {}).get('configuration', {}) or {})
            == expected_sensitivity_rows.get(str(row.get('label')))
            for row in results
        ),
    )
    sensitivity_pool_rows = [
        pool
        for result in results
        for pool in (result.get('summary', {}).get('pools', {}) or {}).values()
    ]
    check(
        'sensitivity_accounting_identities_hold',
        bool(sensitivity_pool_rows) and all(
            row.get('gross_event_identity_pass_rate') == 1.0
            and row.get('claim_supply_identity_end_pass_rate') == 1.0
            for row in sensitivity_pool_rows
        ),
    )

    sensitivity_by_label = {
        str(row.get('label')): row for row in results
    }
    baseline_pools = (
        sensitivity_by_label.get('baseline', {}).get('summary', {}).get('pools', {})
        or {}
    )
    margin_zero_pools = (
        sensitivity_by_label.get('entry_margin_0', {}).get('summary', {})
        .get('pools', {}) or {}
    )
    margin_high_pools = (
        sensitivity_by_label.get('entry_margin_0.5', {}).get('summary', {})
        .get('pools', {}) or {}
    )
    entry_margin_metrics = (
        'gross_new_entry_events',
        'gross_reentry_events',
        'new_entry_provider_activation_rate',
        'median_max_entry_progress',
    )
    common_margin_pools = set(margin_zero_pools).intersection(margin_high_pools)
    entry_margin_measurable = bool(common_margin_pools) and all(
        _finite_number(margin_zero_pools[pool].get(metric))
        and _finite_number(margin_high_pools[pool].get(metric))
        for pool in common_margin_pools for metric in entry_margin_metrics
    )
    entry_margin_effect = bool(entry_margin_measurable and any(
        abs(float(margin_zero_pools[pool][metric])
            - float(margin_high_pools[pool][metric])) > 1e-15
        for pool in common_margin_pools for metric in entry_margin_metrics
    ))
    entry_margin_status = (
        'observed_effect' if entry_margin_effect
        else ('measured_nonbinding_in_precommitted_panel'
              if entry_margin_measurable else 'not_measurable')
    )
    check(
        'entry_margin_control_effect_is_measured_or_explicitly_nonbinding',
        entry_margin_measurable,
        f'status={entry_margin_status}',
    )

    common_rebate_pools = set(baseline_pools).intersection(rebate_pools)
    rebate_response_measurable = bool(common_rebate_pools) and all(
        _finite_number(baseline_pools[pool].get('gross_exit_events'))
        and _finite_number(rebate_pools[pool].get('gross_exit_events'))
        and _finite_number(baseline_pools[pool].get('open_through_crisis_count'))
        and _finite_number(rebate_pools[pool].get('open_through_crisis_count'))
        for pool in common_rebate_pools
    )
    rebate_weak_direction = bool(rebate_response_measurable and all(
        float(rebate_pools[pool]['gross_exit_events'])
        <= float(baseline_pools[pool]['gross_exit_events'])
        and float(rebate_pools[pool]['open_through_crisis_count'])
        >= float(baseline_pools[pool]['open_through_crisis_count'])
        for pool in common_rebate_pools
    ))
    rebate_strict_response = bool(rebate_response_measurable and any(
        float(rebate_pools[pool]['gross_exit_events'])
        < float(baseline_pools[pool]['gross_exit_events'])
        or float(rebate_pools[pool]['open_through_crisis_count'])
        > float(baseline_pools[pool]['open_through_crisis_count'])
        for pool in common_rebate_pools
    ))
    check(
        'full_loss_rebate_weakly_reduces_exit_and_has_a_strict_response',
        rebate_weak_direction and rebate_strict_response,
        'Paired development seeds; no profitability or welfare sign is imposed.',
    )

    routing_results = routing_sensitivity.get('results', []) or []
    expected_routing_rows = dict(routing_specifications(ROUTING_N_ITER))
    expected_routing_labels = set(expected_routing_rows)
    expected_routing_count = len(expected_routing_rows)
    observed_routing_labels = {
        str(row.get('label')) for row in routing_results
    }
    routing_requested = int(
        routing_sensitivity.get('requested_seeds_per_specification', 0) or 0
    )
    routing_seed_start = int(
        routing_sensitivity.get('requested_seed_start', 0) or 0
    )
    routing_seeds = routing_sensitivity.get('requested_seeds', []) or []
    routing_missing = (
        routing_sensitivity.get('missing_seeds_by_specification', {}) or {}
    )
    check('routing_lp_sensitivity_uses_current_measurement_signature',
          routing_sensitivity.get('measurement_signature')
          == routing_measurement_signature())
    check('routing_lp_sensitivity_uses_current_report_signature',
          routing_sensitivity.get('report_signature')
          == routing_report_signature())
    check('routing_lp_sensitivity_uses_current_protocol_signature',
          routing_sensitivity.get('protocol_signature')
          == routing_protocol_signature())
    check('routing_lp_sensitivity_protocol_is_exactly_precommitted',
          routing_sensitivity.get('protocol') == ROUTING_PROTOCOL
          and routing_sensitivity.get('precommit_protocol_conformant') is True
          and int(routing_sensitivity.get('n_iter', -1)) == ROUTING_N_ITER)
    check('routing_lp_sensitivity_complete_and_publication_ready',
          routing_sensitivity.get('complete') is True
          and routing_sensitivity.get('publication_ready') is True)
    check(
        'routing_lp_sensitivity_has_clean_raw_provenance',
        routing_sensitivity.get('raw_provenance_clean') is True
        and bool(routing_results)
        and all(exact_zero(row.get('stale_records_ignored'))
                for row in routing_results),
    )
    check('routing_lp_sensitivity_has_exact_precommitted_oat_cells',
          expected_routing_count
          == int(ROUTING_PROTOCOL.get('expected_unique_specifications', -1))
          and int(routing_sensitivity.get('expected_specifications', -1))
          == expected_routing_count
          and len(routing_results) == expected_routing_count
          and observed_routing_labels == expected_routing_labels
          and all(
              row.get('configuration')
              == expected_routing_rows.get(str(row.get('label')))
              for row in routing_results
          ),
          f'missing={sorted(expected_routing_labels - observed_routing_labels)}; '
          f'extra={sorted(observed_routing_labels - expected_routing_labels)}')
    check('routing_lp_sensitivity_declares_exact_requested_seed_set',
          routing_requested > 0
          and routing_seeds
          == list(range(routing_seed_start,
                        routing_seed_start + routing_requested))
          and routing_seeds == development_seeds
          and set(routing_missing) == expected_routing_labels
          and all(not routing_missing[label]
                  for label in expected_routing_labels))
    routing_protocol_row = lp_design.get('routing_sensitivity') or {}
    expected_routing_protocol_fields = {
        'design', 'expected_specifications',
        *ROUTING_PROTOCOL['declared_levels'],
    }
    check(
        'routing_oat_design_matches_final_protocol',
        set(routing_protocol_row) == expected_routing_protocol_fields
        and int(routing_protocol_row.get('expected_specifications', -1))
        == expected_routing_count
        and routing_protocol_row.get('design')
        == ROUTING_PROTOCOL.get('design')
        and all(
            [float(value) for value in routing_protocol_row.get(field, [])]
            == [float(value) for value in values]
            for field, values in ROUTING_PROTOCOL['declared_levels'].items()
        ),
    )
    check('routing_lp_sensitivity_has_every_requested_seed_in_every_cell',
          routing_requested > 0
          and int(routing_sensitivity.get('expected_records', -1))
          == expected_routing_count * routing_requested
          and int(routing_sensitivity.get('available_records', -2))
          == expected_routing_count * routing_requested
          and all(int(row.get('available_records', -1)) == routing_requested
                  and int(row.get('missing_records', -1)) == 0
                  for row in routing_results))

    routing_flow_metrics = (
        'amm_customer_volume_share',
        'amm_customer_trade_share',
        'amm_active_tick_flow_share',
        'customer_amm_volume_base',
        'customer_total_volume_base',
        'amm_arbitrage_volume_base',
        'amm_execution_volume_base',
    )
    routing_availability_metrics = (
        'lp_available_tick_share',
        'lp_active_provider_mean',
    )

    def all_cells_finite(metrics, phases=ROUTING_PHASES):
        return bool(routing_results) and all(
            int((row.get('summary', {}).get(f'{metric}_{phase}', {}) or {})
                .get('n_finite', -1)) == routing_requested
            and all(_finite_number(
                (row.get('summary', {}).get(f'{metric}_{phase}', {}) or {})
                .get(stat)
            ) for stat in ('mean', 'median', 'p10', 'p90'))
            for row in routing_results
            for phase in phases
            for metric in metrics
        )

    routing_regular_phase_metrics = tuple(
        metric for metric in ROUTING_PHASE_METRICS
        if metric != 'amm_arbitrage_share_of_amm_execution'
    )
    check(
        'routing_all_predeclared_phase_estimands_are_finite',
        all_cells_finite(routing_regular_phase_metrics),
    )
    check(
        'routing_all_predeclared_run_telemetry_is_finite',
        bool(routing_results) and all(
            int((row.get('summary', {}).get(metric, {}) or {}).get(
                'n_finite', -1
            )) == routing_requested
            and all(_finite_number(
                (row.get('summary', {}).get(metric, {}) or {}).get(stat)
            ) for stat in ('mean', 'median', 'p10', 'p90'))
            for row in routing_results
            for metric in ROUTING_RUN_METRICS
        ),
    )

    check('routing_customer_volume_count_active_tick_and_arb_are_separate',
          all_cells_finite(routing_flow_metrics))
    check(
        'routing_arbitrage_share_zero_denominators_are_declared',
        bool(routing_results) and routing_requested > 0 and all(
            int(summary.get('n_finite', -1))
            + int(summary.get('n_zero_denominator', -1))
            == routing_requested
            and int(summary.get('n_invalid', -1)) == 0
            and int(summary.get('n_observations', -1)) == routing_requested
            for row in routing_results
            for phase in ROUTING_PHASES
            for summary in [
                row.get('summary', {}).get(
                    f'amm_arbitrage_share_of_amm_execution_{phase}', {}
                ) or {}
            ]
        ),
        'Undefined shares are allowed only when aggregate AMM execution is '
        'exactly zero; they must never be imputed as zero.',
    )
    check('routing_lp_operating_result_is_measurable_in_both_phases',
          all_cells_finite(('lp_operating_result_pct',))
          and all_cells_finite(('lp_operating_result_measurable',))
          and all(
              (row.get('summary', {}).get(
                  f'lp_operating_result_measurable_{phase}', {}) or {}).get('p10')
              == 1.0
              and (row.get('summary', {}).get(
                  f'lp_operating_result_measurable_{phase}', {}) or {}).get('p90')
              == 1.0
              for row in routing_results for phase in ROUTING_PHASES
          ))
    run_availability_metrics = (
        'lp_population_count',
        'lp_open_at_shock_share',
        'lp_open_through_crisis_share',
    )
    check('routing_lp_availability_telemetry_is_complete',
          all_cells_finite(routing_availability_metrics)
          and bool(routing_results)
          and all(
              int((row.get('summary', {}).get(metric, {}) or {}).get(
                  'n_finite', -1)) == routing_requested
              for row in routing_results
              for metric in run_availability_metrics
          ))

    fee_settings = pnl.get('fee_settings', []) or []
    pnl_config = pnl.get('configuration', {}) or {}
    expected_fee_grid = [
        None if value == 'calibrated' else float(value)
        for value in pnl_row.get('fee_bps', [])
    ]
    observed_fee_grid = [row.get('fee_bps') for row in fee_settings]
    check('pnl_seed_grid_complete', pnl.get('complete') is True)
    check(
        'pnl_has_clean_raw_provenance',
        clean_raw_provenance(pnl)
        and bool(fee_settings)
        and all(
            row.get('raw_provenance_clean') is True
            and exact_zero(row.get('invalid_or_stale_records_ignored'))
            and exact_zero(row.get('stale_records_ignored'))
            for row in fee_settings
        ),
    )
    check(
        'pnl_has_exact_holdout_seed_manifest',
        int(pnl_config.get('seed_start', -1)) == pnl_seeds[0]
        and int(pnl_config.get('requested_seeds_per_fee', -1)) == len(pnl_seeds)
        and pnl_config.get('requested_seed_list') == pnl_seeds
        and int(pnl_config.get('n_iter', -1)) == int(pnl_row['n_iter']),
    )
    check(
        'pnl_contains_exact_calibrated_and_precommitted_fee_grid',
        pnl_config.get('fee_grid_bps') == expected_fee_grid
        and observed_fee_grid == expected_fee_grid
        and len(fee_settings) == len(expected_fee_grid)
        and sum(row.get('fee_mode') == 'calibrated' for row in fee_settings) == 1
        and all(
            row.get('fee_mode') == (
                'calibrated' if fee is None else 'uniform_all_amm_venues'
            )
            for row, fee in zip(fee_settings, expected_fee_grid)
        ),
    )
    check(
        'pnl_uses_exact_endogenous_model_and_bootstrap_protocol',
        pnl_config.get('preset') == LP_PRESET
        and pnl_config.get('lp_model') == pnl_row.get('lp_model') == 'endogenous'
        and int(pnl_config.get('bootstrap_draws', -1))
        == int(pnl_row.get('bootstrap_draws', -2))
        and int(pnl_config.get('bootstrap_seed', -1))
        == int(pnl_row.get('bootstrap_seed', 0)),
    )
    check('pnl_capital_flow_self_check_passes',
          pnl.get('capital_flow_zero_result_self_check_passed') is True)
    check('pnl_has_no_profitability_acceptance_target',
          pnl.get('profitability_acceptance_target') is None)
    check(
        'pnl_accounting_and_sign_convention_hold',
        bool(fee_settings) and all(
            row.get('complete') is True
            and int(row.get('requested_seed_count', -1)) == len(pnl_seeds)
            and int(row.get('available_seed_count', -1)) == len(pnl_seeds)
            and int(row.get('available_seed_start', -1)) == pnl_seeds[0]
            and int(row.get('available_seed_end', -1)) == pnl_seeds[-1]
            and row.get('accounting_sign_convention', {}).get('identity_holds') is True
            and row.get('accounting_sign_convention', {}).get('net_formula')
            == 'fees_minus_loss'
            for row in fee_settings
        ),
    )
    pnl_windows_measurable = True
    pnl_intervals_present = True
    for setting in fee_settings:
        windows = setting.get('windows', {}) or {}
        for label in ('calm', 'crisis'):
            total = (windows.get(label, {}) or {}).get('total', {})
            net = total.get('net_pct', {}) or {}
            pnl_windows_measurable &= int(net.get('n', 0) or 0) > 0
            if int(pnl.get('configuration', {}).get('bootstrap_draws', 0) or 0) > 0:
                pnl_intervals_present &= net.get('bootstrap_95_interval') is not None
    check('pnl_calm_and_crisis_windows_have_measurable_capital',
          pnl_windows_measurable)
    check('pnl_bootstrap_intervals_present', pnl_intervals_present)

    welfare_nonempty = bool(welfare)
    welfare_complete = True
    welfare_scope = True
    transfer_identity = True
    arbitrage_excluded = True
    telemetry_present = True
    welfare_rows_finite = True
    welfare_summaries_complete = True

    def valid_welfare_row(row: dict) -> bool:
        accounting = row.get('accounting', {}) or {}
        with_amm = row.get('with_amm', {}) or {}
        without_amm = row.get('without_amm', {}) or {}
        finite_accounting = (
            'common_executed_notional',
            'with_execution_cost_bps',
            'without_execution_cost_bps',
            'matched_notional_user_benefit',
            'matched_notional_user_benefit_bps',
            'lp_operating_result',
            'lp_operating_return',
            'capital_opportunity_cost',
            'with_amm_customer_volume_base',
            'with_arbitrage_volume_base',
            'with_amm_execution_volume_base',
            'with_arbitrage_share_of_amm_execution',
        )
        finite_with = (
            'executed_notional', 'taker_execution_cost',
            'lp_opening_capital', 'lp_lvr', 'lp_fees',
            'amm_customer_volume_base', 'arbitrage_volume_base',
        )
        finite_without = ('executed_notional', 'taker_execution_cost')
        if (not all(_finite_number(accounting.get(key))
                    for key in finite_accounting)
                or not all(_finite_number(with_amm.get(key))
                           for key in finite_with)
                or not all(_finite_number(without_amm.get(key))
                           for key in finite_without)):
            return False
        common = float(accounting['common_executed_notional'])
        opening = float(with_amm['lp_opening_capital'])
        customer = float(accounting['with_amm_customer_volume_base'])
        arbitrage = float(accounting['with_arbitrage_volume_base'])
        execution = float(accounting['with_amm_execution_volume_base'])
        share = float(accounting['with_arbitrage_share_of_amm_execution'])
        return bool(
            common > 0.0
            and opening > 0.0
            and float(with_amm['executed_notional']) > 0.0
            and float(without_amm['executed_notional']) > 0.0
            and customer >= 0.0
            and arbitrage >= 0.0
            and execution > 0.0
            and 0.0 <= share <= 1.0
            and math.isclose(execution, customer + arbitrage,
                             rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(share, arbitrage / execution,
                             rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(
                customer, float(with_amm['amm_customer_volume_base']),
                rel_tol=1e-12, abs_tol=1e-9,
            )
            and math.isclose(
                arbitrage, float(with_amm['arbitrage_volume_base']),
                rel_tol=1e-12, abs_tol=1e-9,
            )
            and float(accounting['capital_opportunity_cost']) >= 0.0
        )

    def valid_welfare_summary(summary: dict, n_rows: int,
                              bootstrap_draws: int) -> bool:
        medians = summary.get('medians', {}) or {}
        counts = summary.get('finite_counts', {}) or {}
        intervals = summary.get('bootstrap_95_intervals_on_median', {}) or {}
        expected = set(WELFARE_SUMMARY_METRICS)
        if not (set(medians) == set(counts) == set(intervals) == expected):
            return False
        for metric in WELFARE_SUMMARY_METRICS:
            if (int(counts.get(metric, -1)) != n_rows
                    or not _finite_number(medians.get(metric))):
                return False
            interval = intervals.get(metric)
            if bootstrap_draws > 0:
                if (not isinstance(interval, list) or len(interval) != 2
                        or not all(_finite_number(value) for value in interval)
                        or float(interval[0]) > float(interval[1])):
                    return False
            elif interval is not None:
                return False
        return True

    expected_welfare_arms = {
        (
            float(row.get('subsidy_rate')),
            float(row.get('loss_rebate')),
        ): str(row.get('name'))
        for row in welfare_row.get('arms', [])
    }
    observed_welfare_arms = {}
    for payload in welfare:
        summary = payload.get('summary', {}) or {}
        config = payload.get('configuration', {}) or {}
        rows = payload.get('rows', []) or []
        arm_key = (
            float(config.get('subsidy_rate', float('nan'))),
            float(config.get('loss_rebate', float('nan'))),
        )
        observed_welfare_arms[arm_key] = observed_welfare_arms.get(arm_key, 0) + 1
        welfare_complete &= (
            int(summary.get('n_seeds', -1)) == int(config.get('seeds', -2))
            and len(rows) == int(config.get('seeds', -2))
            and int(config.get('seed_start', -1)) == welfare_seeds[0]
            and int(config.get('seeds', -1)) == len(welfare_seeds)
            and sorted(int(row.get('seed', -1)) for row in rows) == welfare_seeds
            and config.get('lp_model') == welfare_row.get('lp_model', 'endogenous')
            and float(config.get('capital_rate', float('nan')))
            == float(welfare_row.get('annual_capital_rate', 0.029))
            and int(config.get('bootstrap_draws', -1))
            == int(welfare_row.get('bootstrap_draws', -2))
            and int(config.get('bootstrap_seed', -1))
            == int(welfare_row.get('bootstrap_seed', 0))
            and int(summary.get('bootstrap_draws', -1))
            == int(welfare_row.get('bootstrap_draws', -2))
        )
        scope = summary.get('estimand_scope', {}) or {}
        welfare_scope &= (
            summary.get('total_welfare_identified') is False
            and scope.get('customer_sample') == 'successful_routed_customer_fills_only'
            and scope.get('total_welfare_identified') is False
        )
        transfer_identity &= (
            summary.get('all_subsidy_transfers_cancel') is True
            and int(summary.get('subsidy_transfer_cancellation_failures', -1)) == 0
            and all(row.get('accounting', {}).get('subsidy_transfer_cancels') is True
                    for row in rows)
        )
        arbitrage_excluded &= all(
            row.get('accounting', {}).get(
                'user_benefit_uses_routed_customer_fills_only') is True
            and row.get('accounting', {}).get(
                'arbitrage_execution_in_user_benefit') is False
            and row.get('accounting', {}).get('arbitrage_surplus_identified') is False
            and row.get('accounting', {}).get('total_welfare_identified') is False
            for row in rows
        )
        telemetry_present &= all(
            'arbitrage_volume_base' in row.get('with_amm', {})
            and 'arbitrage_share_of_amm_execution' in row.get('with_amm', {})
            for row in rows
        )
        welfare_rows_finite &= bool(rows) and all(
            valid_welfare_row(row) for row in rows
        )
        welfare_summaries_complete &= valid_welfare_summary(
            summary,
            len(rows),
            int(config.get('bootstrap_draws', 0) or 0),
        )
    check(
        'welfare_has_exact_two_precommitted_incidence_arms',
        len(welfare) == len(expected_welfare_arms) == 2
        and set(observed_welfare_arms) == set(expected_welfare_arms)
        and all(count == 1 for count in observed_welfare_arms.values()),
    )
    check('welfare_seed_panels_complete', welfare_nonempty and welfare_complete)
    check('welfare_estimand_scope_is_partial_customer_execution',
          welfare_nonempty and welfare_scope)
    check('welfare_subsidy_transfer_identities_hold',
          welfare_nonempty and transfer_identity)
    check('arbitrage_is_excluded_from_customer_benefit',
          welfare_nonempty and arbitrage_excluded)
    check('arbitrage_execution_telemetry_is_present',
          welfare_nonempty and telemetry_present)
    check('welfare_core_row_estimands_are_finite_and_reconciled',
          welfare_nonempty and welfare_rows_finite)
    check('welfare_core_summary_counts_medians_and_intervals_are_complete',
          welfare_nonempty and welfare_summaries_complete)

    # The absence of these checks is itself intentional and machine-readable.
    noncriteria = [
        'positive_lp_profit',
        'positive_user_benefit',
        'positive_total_welfare',
        'a_prespecified_survival_rate_without_an_external_target',
    ]
    passed = all(row['passed'] for row in checks)
    return {
        'model_signature': current_signature,
        'lp_acceptance_code_sha256': acceptance_code_sha256(),
        'provenance_mode': (
            'direct_frozen_protocol' if direct_v3 else 'historical_amendment'
        ),
        'amendment_version': (
            None if amendment is None else amendment.get('amendment_version')
        ),
        'amendment_sha256': amendment_sha256,
        'protocol_version': protocol.get('protocol_version'),
        'protocol_sha256': protocol_sha256,
        'artifact_sha256': dict(sorted(artifact_sha256.items())),
        'passed': passed,
        'checks_passed': sum(row['passed'] for row in checks),
        'checks_total': len(checks),
        'checks': checks,
        'explicit_noncriteria': noncriteria,
        'mechanism_control_diagnostics': {
            'entry_margin_status': entry_margin_status,
            'entry_margin_effect_observed': entry_margin_effect,
            'loss_rebate_response_measurable': rebate_response_measurable,
            'loss_rebate_weak_direction_holds': rebate_weak_direction,
            'loss_rebate_strict_response_observed': rebate_strict_response,
        },
        'interpretation': (
            'Publication-integrity gate for LP mechanics, accounting, sampling '
            'completeness and estimand scope; not an outcome-sign filter.'
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--amendment')
    parser.add_argument('--book-development-json')
    parser.add_argument('--book-holdout-json')
    parser.add_argument('--survival-json')
    parser.add_argument('--funded-survival-json')
    parser.add_argument('--rebate-survival-json')
    parser.add_argument('--sensitivity-json')
    parser.add_argument('--routing-sensitivity-json')
    parser.add_argument('--pnl-json')
    parser.add_argument('--welfare-json', nargs='+')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)

    try:
        if Path(args.protocol).resolve() != Path(CANONICAL_PROTOCOL).resolve():
            raise ValueError(
                'publication gate requires the canonical '
                'calibration/final_protocol.json'
            )
        with open(args.protocol, 'rb') as handle:
            protocol_raw = handle.read()
        protocol = json.loads(protocol_raw)
        welfare_names = (
            'welfare_private_primary',
            'welfare_full_loss_guarantee_upper_bound',
        )
        direct_v3 = protocol.get('protocol_version') == DIRECT_PROTOCOL_VERSION
        manual_inputs = (
            args.book_development_json, args.book_holdout_json,
            args.survival_json, args.funded_survival_json,
            args.rebate_survival_json, args.sensitivity_json,
            args.routing_sensitivity_json, args.pnl_json,
            args.welfare_json,
        )
        if direct_v3:
            if args.amendment is not None:
                raise ValueError('v3 protocol forbids --amendment')
            if any(value is not None for value in manual_inputs):
                raise ValueError(
                    'v3 artifact paths come only from protocol artifact_paths'
                )
            manifest = _direct_artifact_paths(protocol)
            supplied_paths = {
                name: str(Path(ROOT) / manifest[name])
                for name in DIRECT_INPUT_ARTIFACT_KEYS
            }
            declared_output = str(Path(ROOT) / manifest['acceptance'])
            if (args.output is not None
                    and Path(args.output).resolve() != Path(declared_output).resolve()):
                raise ValueError(
                    f'v3 acceptance output must use {declared_output}'
                )
            args.output = declared_output
            amendment = None
            amendment_sha256 = None
        else:
            if args.amendment is None:
                raise ValueError('pre-v3 protocol requires --amendment')
            if Path(args.amendment).resolve() != Path(CANONICAL_AMENDMENT).resolve():
                raise ValueError(
                    'pre-v3 publication gate requires the canonical '
                    'calibration/lp_amendment_v2a.json'
                )
            if any(value is None for value in manual_inputs):
                raise ValueError('pre-v3 publication gate requires every artifact')
            if len(args.welfare_json) != len(welfare_names):
                raise ValueError('exactly two ordered welfare artifacts are required')
            amendment, amendment_sha256 = _load_with_sha256(args.amendment)
            supplied_paths = {
                'book_development': args.book_development_json,
                'book_holdout': args.book_holdout_json,
                'primary_survival': args.survival_json,
                'funded_survival': args.funded_survival_json,
                'rebate_survival': args.rebate_survival_json,
                'sensitivity': args.sensitivity_json,
                'routing_effective': args.routing_sensitivity_json,
                'pnl': args.pnl_json,
                **dict(zip(welfare_names, args.welfare_json)),
            }
            for name, supplied in supplied_paths.items():
                declared = Path(ROOT) / AMENDMENT_ARTIFACT_PATHS[name]
                if Path(supplied).resolve() != declared.resolve():
                    raise ValueError(
                        f'{name} artifact must use the canonical path: '
                        f'{declared}'
                    )
        loaded = {}
        artifact_sha256 = {}
        for name, supplied in supplied_paths.items():
            loaded[name], artifact_sha256[name] = _load_with_sha256(supplied)
        if direct_v3:
            routing_key = 'routing_sensitivity'
            original_routing = None
            failed_acceptance = None
        else:
            routing_key = 'routing_effective'
            for name in ('routing_original', 'failed_acceptance'):
                evidence_path = Path(ROOT) / AMENDMENT_ARTIFACT_PATHS[name]
                loaded[name], artifact_sha256[name] = _load_with_sha256(
                    str(evidence_path)
                )
            original_routing = loaded['routing_original']
            failed_acceptance = loaded['failed_acceptance']
        result = validate(
            protocol,
            hashlib.sha256(protocol_raw).hexdigest(),
            loaded['book_development'],
            artifact_sha256['book_development'],
            loaded['book_holdout'],
            loaded['primary_survival'],
            loaded['funded_survival'],
            loaded['rebate_survival'],
            loaded['sensitivity'],
            loaded[routing_key],
            loaded['pnl'],
            [loaded[name] for name in welfare_names],
            amendment=amendment,
            amendment_sha256=amendment_sha256,
            artifact_sha256=artifact_sha256,
            original_routing=original_routing,
            failed_acceptance=failed_acceptance,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'LP acceptance input error: {exc}', file=sys.stderr)
        return 2

    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
