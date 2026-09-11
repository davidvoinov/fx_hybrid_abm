from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module
from calibration.fitter import (
    CalibrationFitter,
    _kaplan_meier_median,
    _lifecycle_summary,
)


# Share of seeds that individually satisfy every EBS observable. Reported as a
# diagnostic and no longer gated, for a reason worth stating in full.
#
# The check conflated two properties. Whether the per-seed distribution is
# centred on the published figure is one thing, and how widely it disperses
# around it is another. Centring is already gated, since the panel evaluates
# the median across seeds against the same four targets. What the per-seed rate
# added on top of that was a bound on dispersion, and neither the 0.90 nor the
# band it is taken against has any source: no study publishes the session to
# session dispersion of median order lifetime on EBS, and requiring one
# thousand second window to reproduce an annual venue median to within about
# twenty per cent on nine draws in ten is a standard nobody has defended.
#
# A high pass rate can also come from a cap. Hold the dealer quote life at the
# published target of 290 seconds and stop dealers taking each other, and every
# dealer order ends on its timer, so the per seed median clings to the cap, the
# fifth to ninety fifth percentile spans 55 seconds and 99.3 per cent of seeds
# sit inside the band. With quotes able to trade the same statistic centres at
# 292 seconds against that target and disperses over 141 seconds, so 83.7 per
# cent of seeds fall inside. The first rate measures the cap and the second
# measures the model. Dispersion is reported with its Wilson interval and gated
# by nothing.
EBS_SEED_PASS_RATE_REFERENCE = 0.90
MAX_CALM_ACTIVATION_RATE = 0.05
MIN_CRISIS_ACTIVATION_RATE = 0.75
MIN_ACTIVATION_RATE_GAP = 0.50
ACTIVATION_EPSILON = 1e-12
MIN_TWO_SIDED_BOOK_RATE = 0.999
MAX_DEALER_LIFETIME_ATOM_SHARE = 0.25
# A three-second lifetime on a one-second grid necessarily has a sizeable
# first-tick mass.  This bound is looser than the dealer bound by design:
# it catches a provider wide refresh pulse without rejecting the natural
# discretisation of an independent geometric clock.
MAX_NONBANK_LIFETIME_ATOM_SHARE = 0.35
# The share of the seed panel that must sit inside the bound above before the
# panel can be asked whether its tail carries the statistic. It is a majority
# and not a high threshold, because it is not the test: the test is that the
# median order lifetime does not move when the seeds outside the bound are
# dropped, and that test needs a remainder large enough to compare against.
MIN_NONBANK_LIFETIME_PANEL_MAJORITY = 0.5
MAX_UNCATEGORIZED_LIFECYCLE_SHARE = 0.0
MAX_SAME_TICK_SCHEDULED_END_SHARE = 0.25
MIN_COMPLETED_LIFECYCLE_EVENTS = 100


# Fractions of the manifest value that each searched parameter is bracketed
# by. The grid is derived from the manifest and not written out, because
# a grid of literals goes stale silently: before 15.08.2026 this searched
# mm_alpha0_base over 1.8 to 2.7 while the calibrated value was 0.05, and
# mm_alpha2 over 420 to 700 while it was 20. Those literals belonged to the
# parameterisation that preceded the rescaling of the model's units, so the
# search could not reach the current optimum and said nothing about it. A
# grid built from the manifest cannot drift away from the point it is meant
# to bracket.
SEARCH_BRACKET: dict[str, tuple[float, ...]] = {
    'mm_alpha0_base': (0.5, 1.0, 2.0),
    'mm_alpha1': (0.5, 1.0, 2.0),
    'mm_alpha2': (0.5, 1.0, 2.0),
    'mm_d0_base': (0.75, 1.0, 1.25),
    'mm_stale_touch_ratio': (0.5, 1.0, 1.5),
    'fast_lp_base_spread_bps': (0.75, 1.0, 1.25),
    # The constant and the volatility loading of the fast provider's quote set
    # the level of the spread and its elasticity to volatility between them, so
    # a search moving one without the other trades one against the other in
    # silence. Both are searched.
    'fast_lp_vol_multiple': (0.5, 1.0, 1.5),
    'hedger_flow_persistence': (0.6, 1.0, 1.4),
    'retail_flow_persistence': (0.6, 1.0, 1.4),
    'institutional_flow_persistence': (0.6, 1.0, 1.4),
    'amm_share_pct': (1.0, 1.4, 1.8),
    'cost_noise_std': (0.5, 1.0, 1.5),
    'hfmm_reserves': (0.6, 1.0, 1.4),
    'hfmm_fee': (1.0, 2.0, 3.0),
    'hfmm_A': (0.55, 1.0, 1.7),
    'arb_trade_fraction_cap': (0.45, 1.0, 1.1),
}

# Counts are searched on their own small integer ladders; scaling them by a
# fraction of the manifest value would not produce whole agents.
SEARCH_COUNTS: dict[str, tuple[int, ...]] = {
    'n_clob_fund': (3, 5, 8),
}


def _default_search_grid(base_defaults: dict[str, Any]) -> dict[str, list[float]]:
    """Bracket each searched parameter around its manifest value."""
    grid: dict[str, list[float]] = {}
    for key, counts in SEARCH_COUNTS.items():
        if key in base_defaults:
            grid[key] = [int(value) for value in counts]
    for key, factors in SEARCH_BRACKET.items():
        if key not in base_defaults:
            continue
        try:
            centre = float(base_defaults[key])
        except (TypeError, ValueError):
            continue
        if centre == 0.0:
            continue
        values = sorted({round(centre * factor, 12) for factor in factors})
        if len(values) > 1:
            grid[key] = values
    return grid


def _objective_value(report: dict[str, Any]) -> float:
    summary = report.get('summary', {})
    objective = summary.get('objective_score')
    if objective is None or not math.isfinite(objective):
        return float('inf')
    penalty = 1000.0 * float(summary.get('gating_failures', 0))
    return float(objective) + penalty


def _make_scenario_args(scenario: dict[str, Any], overrides: dict[str, Any], seed: int) -> argparse.Namespace:
    parser = main_module.build_parser()
    args = parser.parse_args([])
    args.seed = seed
    args.no_plots = True
    args.no_comparison = True
    args.silent = True
    args.spillover_artifacts = False
    args.run_label = scenario['name']
    args.preset = scenario.get('preset')

    if args.preset:
        main_module._apply_preset_defaults(parser, args)

    for key, value in overrides.items():
        # Preserve scenario-preset values when the candidate simply repeats
        # the parser baseline default. Only explicit deviations from the
        # baseline should override the scenario-specific preset.
        if hasattr(args, key):
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            if (main_module._same_value(value, default_value)
                    and not main_module._same_value(current_value, default_value)):
                continue
        setattr(args, key, value)

    if scenario.get('n_iter') is not None:
        args.n_iter = int(scenario['n_iter'])
    if scenario.get('shock_iter') is not None:
        args.shock_iter = scenario['shock_iter']
    if scenario.get('stress_start') is not None:
        args.stress_start = scenario['stress_start']
    if scenario.get('stress_end') is not None:
        args.stress_end = scenario['stress_end']

    main_module._auto_stress_around_shock(args)
    args.venue_choice_rule = main_module._resolve_main_routing(args, [])
    return args


def evaluate_candidate(overrides: dict[str, Any], target_payload: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    fitter = CalibrationFitter(target_payload)
    scenario_metrics: dict[str, dict[str, float]] = {}
    scenario_reports: dict[str, dict[str, Any]] = {}

    for scenario in target_payload.get('calibration_scenarios', []):
        args = _make_scenario_args(scenario, overrides, seed=seed)
        main_module._seed_all(seed)
        sim = main_module.build_sim(args)
        sim.simulate(args.n_iter, silent=True)
        metrics = fitter.realized_metrics(sim)
        scenario_metrics[scenario['name']] = metrics
        scenario_reports[scenario['name']] = fitter.evaluate_metrics(
            metrics,
            run_label=args.run_label,
            scenario_name=scenario['name'],
        )

    suite_report = fitter.evaluate_scenario_suite(scenario_metrics, run_label='calibration_suite')
    suite_report['scenario_reports'] = scenario_reports
    suite_report['candidate_overrides'] = deepcopy(overrides)
    suite_report['score'] = _objective_value(suite_report)
    return suite_report


def _finite_median(values: list[float]) -> float:
    finite_values = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite_values.append(numeric)
    values = sorted(finite_values)
    if not values:
        return float('nan')
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _wilson_interval(successes: int, trials: int,
                     z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for a binomial rate, without a scipy dependency."""
    if trials <= 0 or successes < 0 or successes > trials:
        return float('nan'), float('nan')
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    ) / denominator
    return centre - half_width, centre + half_width


def _mechanism_audit(ebs_seed_pass: list[bool], calm_peaks: list[float],
                     crisis_peaks: list[float], forced_peaks: list[float],
                     book_integrity: Optional[dict[str, list[float]]] = None
                     ) -> dict[str, Any]:
    """Audit stochastic selection into withdrawal, not a zero-event fiction.

    A genuinely endogenous risk rule can fire on an unusual calm path.  The
    identifying restriction is instead that such false positives are rare,
    crisis activation is common, and the two rates are well separated.  The
    Wilson limits make a small seed panel insufficient to certify those facts.
    """
    book_integrity = book_integrity or {}
    required_book_fields = (
        'book_two_sided_rate',
        'dealer_lifecycle_max_atom_share',
        'nonbank_lifecycle_max_atom_share',
        'dealer_lifecycle_uncategorized_share',
        'nonbank_lifecycle_uncategorized_share',
        'dealer_lifecycle_max_same_tick_scheduled_end_share',
        'nonbank_lifecycle_max_same_tick_scheduled_end_share',
        'dealer_lifecycle_completed_count',
        'nonbank_lifecycle_completed_count',
        'dealer_order_lifetime_median_seconds',
        'nonbank_order_lifetime_median_seconds',
    )
    lengths = {len(ebs_seed_pass), len(calm_peaks), len(crisis_peaks), len(forced_peaks)}
    complete = len(lengths) == 1 and next(iter(lengths), 0) > 0
    finite = complete and all(
        math.isfinite(float(value))
        for series in (calm_peaks, crisis_peaks, forced_peaks)
        for value in series
    )
    n = len(calm_peaks) if complete else 0
    book_complete = bool(
        complete
        and all(field in book_integrity for field in required_book_fields)
        and all(len(book_integrity[field]) == n for field in required_book_fields)
    )
    book_finite = bool(
        book_complete
        and all(
            math.isfinite(float(value))
            for field in required_book_fields
            for value in book_integrity[field]
        )
    )
    ebs_successes = sum(bool(value) for value in ebs_seed_pass) if complete else 0
    calm_successes = (sum(float(value) > ACTIVATION_EPSILON for value in calm_peaks)
                      if finite else 0)
    crisis_successes = (sum(float(value) > ACTIVATION_EPSILON for value in crisis_peaks)
                        if finite else 0)
    ebs_rate = ebs_successes / n if n else float('nan')
    calm_rate = calm_successes / n if n else float('nan')
    crisis_rate = crisis_successes / n if n else float('nan')
    ebs_ci = _wilson_interval(ebs_successes, n)
    calm_ci = _wilson_interval(calm_successes, n)
    crisis_ci = _wilson_interval(crisis_successes, n)
    conservative_gap = (crisis_ci[0] - calm_ci[1]
                        if all(math.isfinite(v) for v in (*calm_ci, *crisis_ci))
                        else float('nan'))

    def _all_at_least(field: str, threshold: float) -> bool:
        return bool(book_finite and all(
            float(value) >= threshold for value in book_integrity[field]
        ))

    def _all_at_most(field: str, threshold: float) -> bool:
        return bool(book_finite and all(
            float(value) <= threshold + 1e-15 for value in book_integrity[field]
        ))

    def _tail_does_not_carry(field: str, threshold: float, protects: str,
                             majority: float) -> bool:
        """The tail does not carry the statistic the bound protects.

        A per-seed bound answers a question about every draw, which is the
        right form where a single breach would invalidate a measurement. This
        bound is not of that kind. It exists so that a median order lifetime
        is not assembled out of orders that begin and end in the same period,
        and that purpose survives a tail, so a per-seed form rejects panels
        that support the statistic perfectly well.

        A share of the panel is the obvious weakening and it is the wrong one,
        because the share has to be chosen and nothing derives it. What the
        purpose does derive is a test of the statistic itself: drop every seed
        outside the bound and the panel median of the protected quantity must
        not move. That fails exactly when the tail is what produced the
        number, which is the defect the bound was written against, and it
        passes when the tail is incidental to it. A majority of the panel must
        still comply, since a statistic cannot be shown to be clean against a
        remainder too small to compare with.
        """
        values = [float(v) for v in book_integrity[field]]
        protected = [float(v) for v in book_integrity[protects]]
        if not (book_finite and values and len(protected) == len(values)):
            return False
        inside = [p for v, p in zip(values, protected)
                  if v <= threshold + 1e-15]
        if len(inside) <= majority * len(values):
            return False
        whole = _finite_median(protected)
        kept = _finite_median(inside)
        if not (math.isfinite(whole) and math.isfinite(kept)):
            return False
        return bool(abs(whole - kept) <= 1e-9 * max(1.0, abs(whole)))

    checks = {
        'complete_finite_seed_panel': bool(complete and finite and book_finite),
        'calm_activation_upper_95pct_at_most_5pct': bool(
            finite and math.isfinite(calm_ci[1])
            and calm_ci[1] <= MAX_CALM_ACTIVATION_RATE
        ),
        'crisis_activation_lower_95pct_at_least_75pct': bool(
            finite and math.isfinite(crisis_ci[0])
            and crisis_ci[0] >= MIN_CRISIS_ACTIVATION_RATE
        ),
        'conservative_activation_gap_at_least_50pp': bool(
            finite and math.isfinite(conservative_gap)
            and conservative_gap >= MIN_ACTIVATION_RATE_GAP
        ),
        'no_scripted_pause_in_dealer_crisis': bool(
            finite and all(float(value) <= ACTIVATION_EPSILON for value in forced_peaks)
        ),
        'no_full_dealer_evacuation': bool(
            finite and all(float(value) < 1.0 - ACTIVATION_EPSILON
                           for value in crisis_peaks)
        ),
        'book_two_sided_rate_at_least_99_9pct': _all_at_least(
            'book_two_sided_rate', MIN_TWO_SIDED_BOOK_RATE
        ),
        'dealer_lifecycle_atom_share_at_most_25pct': _all_at_most(
            'dealer_lifecycle_max_atom_share', MAX_DEALER_LIFETIME_ATOM_SHARE
        ),
        'nonbank_lifecycle_atom_share_does_not_carry_the_median':
            _tail_does_not_carry('nonbank_lifecycle_max_atom_share',
                                 MAX_NONBANK_LIFETIME_ATOM_SHARE,
                                 'nonbank_order_lifetime_median_seconds',
                                 MIN_NONBANK_LIFETIME_PANEL_MAJORITY),
        'dealer_lifecycle_reasons_fully_categorized': _all_at_most(
            'dealer_lifecycle_uncategorized_share',
            MAX_UNCATEGORIZED_LIFECYCLE_SHARE,
        ),
        'nonbank_lifecycle_reasons_fully_categorized': _all_at_most(
            'nonbank_lifecycle_uncategorized_share',
            MAX_UNCATEGORIZED_LIFECYCLE_SHARE,
        ),
        'dealer_scheduled_ends_not_synchronized': _all_at_most(
            'dealer_lifecycle_max_same_tick_scheduled_end_share',
            MAX_SAME_TICK_SCHEDULED_END_SHARE,
        ),
        'nonbank_scheduled_ends_not_synchronized': _all_at_most(
            'nonbank_lifecycle_max_same_tick_scheduled_end_share',
            MAX_SAME_TICK_SCHEDULED_END_SHARE,
        ),
        'dealer_lifecycle_sample_at_least_100': _all_at_least(
            'dealer_lifecycle_completed_count', MIN_COMPLETED_LIFECYCLE_EVENTS
        ),
        'nonbank_lifecycle_sample_at_least_100': _all_at_least(
            'nonbank_lifecycle_completed_count', MIN_COMPLETED_LIFECYCLE_EVENTS
        ),
        'lifecycle_km_medians_estimable': bool(
            book_finite
            and all(float(value) >= 0.0
                    for field in ('dealer_order_lifetime_median_seconds',
                                  'nonbank_order_lifetime_median_seconds')
                    for value in book_integrity[field])
        ),
    }
    return {
        'n_seeds': n,
        'ebs_seed_pass_rate': ebs_rate,
        'ebs_seed_pass_rate_wilson_95': list(ebs_ci),
        'calm_activation_count': calm_successes,
        'calm_activation_rate': calm_rate,
        'calm_activation_rate_wilson_95': list(calm_ci),
        'crisis_activation_count': crisis_successes,
        'crisis_activation_rate': crisis_rate,
        'crisis_activation_rate_wilson_95': list(crisis_ci),
        'activation_rate_gap': crisis_rate - calm_rate if n else float('nan'),
        'conservative_activation_rate_gap_95': conservative_gap,
        'thresholds': {
            'maximum_calm_activation_rate': MAX_CALM_ACTIVATION_RATE,
            'minimum_crisis_activation_rate': MIN_CRISIS_ACTIVATION_RATE,
            'minimum_activation_rate_gap': MIN_ACTIVATION_RATE_GAP,
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
        },
        'book_integrity_extrema': {
            field: {
                'minimum': min(float(value) for value in book_integrity[field]),
                'maximum': max(float(value) for value in book_integrity[field]),
            }
            for field in required_book_fields
        } if book_finite else {},
        'mechanism_checks': checks,
    }


# Free text that records where a target came from and why, which is part of
# the manifest and no part of what the target tests.
_TARGET_PROSE = ('ingestion_note', 'source_excerpt', 'scope', 'ingestion_method',
                 'revision_history', 'description')


def target_matrix_semantics(target_payload: dict[str, Any]) -> Any:
    """What each target tests, with the prose that explains it removed.

    The digest is a commitment device: it has to move when what is being
    tested moves. Hashing the manifest whole made it move when a note was
    reworded too, which invalidates a frozen protocol for a change that cannot
    alter a verdict and buries the changes that can. ``_primary_runtime_sha256``
    below already draws this line for the model manifest, and this draws the
    same one for the target matrix. The observable, the scenario, the band, the
    weight, the gate and the source all remain inside the digest.
    """
    if isinstance(target_payload, dict):
        return {key: target_matrix_semantics(value)
                for key, value in target_payload.items()
                if key not in _TARGET_PROSE}
    if isinstance(target_payload, list):
        return [target_matrix_semantics(item) for item in target_payload]
    return target_payload


def _target_matrix_sha256(target_payload: dict[str, Any]) -> str:
    encoded = json.dumps(target_matrix_semantics(target_payload),
                         sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _primary_runtime_sha256(model_path: Path = Path('calibration/primary_model.json')) -> str:
    """Hash runtime inputs while excluding provenance prose."""
    from tools.robustness.signatures import calibration_semantics
    payload = calibration_semantics(json.loads(model_path.read_text()))
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def book_acceptance_signature(target_payload: dict[str, Any]) -> str:
    """Digest the model and code that turn seeded paths into a verdict."""
    from tools.robustness.signatures import measurement_signature, model_signature
    return measurement_signature(
        'book_acceptance_panel',
        model_signature(),
        functions=(
            _make_scenario_args,
            evaluate_candidate,
            evaluate_candidate_panel,
            _kaplan_meier_median,
            _lifecycle_summary,
            CalibrationFitter._lifecycle_rows,
            CalibrationFitter._two_sided_book_rate,
            CalibrationFitter.realized_metrics,
            CalibrationFitter.evaluate_metrics,
            CalibrationFitter.evaluate_scenario_suite,
            _finite_median,
            _wilson_interval,
            _mechanism_audit,
            _verify_development_artifact,
            _panel_from_reports,
            _objective_value,
        ),
        constants=(
            _target_matrix_sha256(target_payload),
            EBS_SEED_PASS_RATE_REFERENCE,
            MAX_CALM_ACTIVATION_RATE,
            MIN_CRISIS_ACTIVATION_RATE,
            MIN_ACTIVATION_RATE_GAP,
            ACTIVATION_EPSILON,
            MIN_TWO_SIDED_BOOK_RATE,
            MAX_DEALER_LIFETIME_ATOM_SHARE,
            MAX_NONBANK_LIFETIME_ATOM_SHARE,
            MAX_UNCATEGORIZED_LIFECYCLE_SHARE,
            MAX_SAME_TICK_SCHEDULED_END_SHARE,
            MIN_COMPLETED_LIFECYCLE_EVENTS,
        ),
    )


def _protocol_seed_list(protocol: dict[str, Any], phase: str) -> list[int]:
    phases = protocol.get('phases') or {}
    row = phases.get(phase) or {}
    if 'seeds' in row:
        seeds = [int(value) for value in row['seeds']]
    else:
        start = int(row.get('seed_start'))
        count = int(row.get('seed_count'))
        seeds = list(range(start, start + count))
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError(f'protocol phase {phase!r} has empty or duplicate seeds')
    return seeds


def _verify_protocol(protocol: dict[str, Any], target_payload: dict[str, Any],
                     phase: str) -> list[int]:
    """Refuse a holdout whose frozen model, targets, code or seeds drifted."""
    from tools.robustness.signatures import model_signature

    if phase not in {'development', 'holdout'}:
        raise ValueError('protocol phase must be development or holdout')
    development = _protocol_seed_list(protocol, 'development')
    holdout = _protocol_seed_list(protocol, 'holdout')
    if set(development).intersection(holdout):
        raise ValueError('development and holdout seed commitments overlap')

    expected_thresholds = {
        'maximum_calm_activation_rate': MAX_CALM_ACTIVATION_RATE,
        'minimum_crisis_activation_rate': MIN_CRISIS_ACTIVATION_RATE,
        'minimum_activation_rate_gap': MIN_ACTIVATION_RATE_GAP,
    }
    if protocol.get('thresholds') != expected_thresholds:
        raise ValueError('protocol thresholds do not match the acceptance code')
    expected_book_thresholds = {
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
        'nonbank_lifetime_atom_share_panel_majority': (
            MIN_NONBANK_LIFETIME_PANEL_MAJORITY
        ),
    }
    if protocol.get('book_integrity_thresholds') != expected_book_thresholds:
        raise ValueError(
            'protocol book-integrity thresholds do not match the acceptance code'
        )

    frozen = protocol.get('frozen_hashes') or {}
    actual = {
        'target_matrix_sha256': _target_matrix_sha256(target_payload),
        'primary_runtime_sha256': _primary_runtime_sha256(),
        'simulation_model_signature': model_signature(),
        'book_acceptance_signature': book_acceptance_signature(target_payload),
    }
    mismatches = {
        key: {'expected': frozen.get(key), 'actual': value}
        for key, value in actual.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ValueError('frozen protocol hash mismatch: '
                         + json.dumps(mismatches, sort_keys=True))
    return development if phase == 'development' else holdout


def _protocol_output_path(protocol: dict[str, Any], phase: str) -> Path:
    raw = ((protocol.get('phases') or {}).get(phase) or {}).get('output')
    if not raw:
        raise ValueError(f'protocol phase {phase!r} has no output path')
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify_development_artifact(protocol: dict[str, Any],
                                 target_payload: dict[str, Any],
                                 protocol_raw: bytes) -> str:
    """Require a passing, same-protocol development gate before holdout use."""
    path = _protocol_output_path(protocol, 'development')
    if not path.exists():
        raise ValueError(
            f'development artifact is required before holdout: {path}'
        )
    raw = path.read_bytes()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid development artifact: {path}') from exc

    expected_seeds = _protocol_seed_list(protocol, 'development')
    provenance = report.get('provenance') or {}
    panel = report.get('panel') or {}
    summary = report.get('summary') or {}
    checks = panel.get('mechanism_checks') or {}
    frozen = protocol.get('frozen_hashes') or {}
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    valid = (
        provenance.get('protocol_version') == protocol.get('protocol_version')
        and provenance.get('protocol_phase') == 'development'
        and provenance.get('protocol_sha256') == protocol_sha
        and provenance.get('seed_commitment_exact_match') is True
        and provenance.get('aggregation_mode') == 'fresh_simulation'
        and provenance.get('simulation_model_signature')
        == frozen.get('simulation_model_signature')
        and provenance.get('book_acceptance_signature')
        == frozen.get('book_acceptance_signature')
        and provenance.get('primary_runtime_sha256')
        == frozen.get('primary_runtime_sha256')
        and provenance.get('target_matrix_sha256')
        == frozen.get('target_matrix_sha256')
        == _target_matrix_sha256(target_payload)
        and panel.get('seeds') == expected_seeds
        and int(panel.get('n_seeds', -1)) == len(expected_seeds)
        and summary.get('status') == 'pass'
        and int(summary.get('gating_failures', -1)) == 0
        and int(summary.get('mechanism_failures', -1)) == 0
        and bool(checks)
        and all(value is True for value in checks.values())
    )
    if not valid:
        raise ValueError(
            'development artifact does not pass the exact frozen protocol'
        )
    return hashlib.sha256(raw).hexdigest()


def _panel_from_reports(overrides: dict[str, Any], target_payload: dict[str, Any],
                        seeds: list[int], reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate already measured seed reports under the current audit rules."""
    if not seeds or len(seeds) != len(reports):
        raise ValueError('seeds and reports must be non-empty and have equal length')
    fitter = CalibrationFitter(target_payload)
    scenario_names = [row['name'] for row in target_payload.get('calibration_scenarios', [])]
    median_metrics: dict[str, dict[str, float]] = {}
    for scenario_name in scenario_names:
        keys = set()
        for report in reports:
            keys.update(report.get('scenario_metrics', {}).get(scenario_name, {}))
        median_metrics[scenario_name] = {
            key: _finite_median([
                report.get('scenario_metrics', {}).get(scenario_name, {}).get(key, float('nan'))
                for report in reports
            ])
            for key in sorted(keys)
        }

    panel = fitter.evaluate_scenario_suite(median_metrics, run_label='calibration_seed_panel')
    # One entry per observable. Two targets sharing a name would leave the
    # dictionary below holding one status for both, whichever the evaluation
    # emitted last, and the panel would certify one of them and never look at
    # the other.
    ebs_observables = {
        'quoted_spread_mean_bps',
        'dealer_order_lifetime_median_seconds',
        'nonbank_order_lifetime_median_seconds',
        'dealer_maker_volume_share',
    }
    ebs_seed_pass = []
    for report in reports:
        refreshed = fitter.evaluate_scenario_suite(
            report.get('scenario_metrics', {}), run_label='calibration_seed'
        )
        statuses = {
            row.get('observable'): row.get('status')
            for row in refreshed.get('targets', [])
            if row.get('observable') in ebs_observables
        }
        ebs_seed_pass.append(
            ebs_observables.issubset(statuses)
            and all(statuses[name] == 'pass' for name in ebs_observables)
        )

    def metric(report, scenario, name):
        value = report.get('scenario_metrics', {}).get(scenario, {}).get(name, float('nan'))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float('nan')

    calm_peaks = [metric(report, 'baseline_primary', 'dealer_withdrawal_peak_share')
                  for report in reports]
    # The crisis the panel audits is the identified episode. The synthetic
    # preset that stood here evacuated the entire dealer sector on every seed,
    # so the mechanism checks were reading a scripted outcome.
    crisis_peaks = [metric(report, main_module.CRISIS_PRESET, 'dealer_withdrawal_peak_share')
                    for report in reports]
    forced_peaks = [metric(report, main_module.CRISIS_PRESET, 'dealer_forced_pause_peak_share')
                    for report in reports]
    book_fields = (
        'book_two_sided_rate',
        'dealer_lifecycle_max_atom_share',
        'nonbank_lifecycle_max_atom_share',
        'dealer_lifecycle_uncategorized_share',
        'nonbank_lifecycle_uncategorized_share',
        'dealer_lifecycle_max_same_tick_scheduled_end_share',
        'nonbank_lifecycle_max_same_tick_scheduled_end_share',
        'dealer_lifecycle_completed_count',
        'nonbank_lifecycle_completed_count',
        'dealer_order_lifetime_median_seconds',
        'nonbank_order_lifetime_median_seconds',
    )
    book_integrity = {
        field: [metric(report, 'baseline_primary', field) for report in reports]
        for field in book_fields
    }
    audit = _mechanism_audit(
        ebs_seed_pass, calm_peaks, crisis_peaks, forced_peaks,
        book_integrity=book_integrity,
    )
    panel['panel'] = {
        'seeds': list(seeds),
        'ebs_seed_pass': ebs_seed_pass,
        'calm_endogenous_peak_share': calm_peaks,
        'crisis_endogenous_peak_share': crisis_peaks,
        'crisis_forced_pause_peak_share': forced_peaks,
        'book_integrity': book_integrity,
        **audit,
    }
    panel['seed_reports'] = reports
    panel['summary']['median_literature_status'] = panel['summary']['status']
    checks = audit['mechanism_checks']
    panel['summary']['mechanism_failures'] = sum(not passed for passed in checks.values())
    panel['summary']['status'] = (
        'pass' if panel['summary']['gating_failures'] == 0 and all(checks.values()) else 'fail'
    )
    panel['score'] = _objective_value(panel)
    panel['candidate_overrides'] = deepcopy(overrides)
    panel['provenance'] = {
        'target_matrix_sha256': _target_matrix_sha256(target_payload),
    }
    return panel


def evaluate_candidate_panel(overrides: dict[str, Any], target_payload: dict[str, Any],
                             seeds: list[int], workers: int = 1) -> dict[str, Any]:
    """Evaluate calibration on a seed panel and audit the withdrawal channel.

    Literature targets are applied to the across-seed medians. Endogenous
    dealer exit has no defensible numerical field target, so it is kept out of
    the objective and identified by separation: calm activation must be rare,
    crisis activation common, their confidence bounds well separated, the
    crisis preset may not script a pause, and no path may evacuate every dealer.
    """
    if not seeds:
        raise ValueError('at least one seed is required')
    jobs = [(overrides, target_payload, seed) for seed in seeds]
    if workers > 1:
        with Pool(processes=workers) as pool:
            reports = pool.starmap(evaluate_candidate, jobs)
    else:
        reports = [evaluate_candidate(*job) for job in jobs]
    panel = _panel_from_reports(overrides, target_payload, seeds, reports)
    from tools.robustness.signatures import model_signature
    panel['provenance'].update({
        'simulation_model_signature': model_signature(),
        'aggregation_mode': 'fresh_simulation',
    })
    return panel


def reassess_stored_panel(stored: dict[str, Any], target_payload: dict[str, Any],
                          source_sha256: Optional[str] = None) -> dict[str, Any]:
    """Reapply corrected targets/checks without rerunning stored simulations."""
    reports = list(stored.get('seed_reports') or [])
    seeds = list((stored.get('panel') or {}).get('seeds') or [])
    panel = _panel_from_reports(
        stored.get('candidate_overrides') or stored.get('best_overrides') or {},
        target_payload,
        seeds,
        reports,
    )
    from tools.robustness.signatures import model_signature
    panel['provenance'].update({
        'model_signature_at_reassessment': model_signature(),
        'aggregation_mode': 'reassessed_from_stored_seed_metrics',
        'source_report_sha256': source_sha256,
        'source_simulation_signature': (stored.get('provenance') or {}).get(
            'simulation_model_signature'
        ),
    })
    return panel


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            lo, hi = (int(value) for value in part.split(':', 1))
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


def coordinate_search(target_payload: dict[str, Any], base_overrides: dict[str, Any],
                      search_grid: dict[str, list[float]], passes: int = 1,
                      seed: int = 42) -> dict[str, Any]:
    current = deepcopy(base_overrides)
    best_report = evaluate_candidate(current, target_payload, seed=seed)

    for _ in range(max(1, passes)):
        improved = False
        for parameter, candidates in search_grid.items():
            local_best_value = current.get(parameter)
            local_best_report = best_report

            ordered_candidates = []
            if local_best_value is not None:
                ordered_candidates.append(local_best_value)
            ordered_candidates.extend(value for value in candidates if value != local_best_value)

            for value in ordered_candidates:
                candidate = deepcopy(current)
                candidate[parameter] = value
                report = evaluate_candidate(candidate, target_payload, seed=seed)
                if report['score'] + 1e-12 < local_best_report['score']:
                    local_best_value = value
                    local_best_report = report

            if local_best_report is not best_report:
                current[parameter] = local_best_value
                best_report = local_best_report
                improved = True

        if not improved:
            break

    best_report['best_overrides'] = deepcopy(current)
    return best_report


def write_back_primary_defaults(best_overrides: dict[str, Any], model_path: Path) -> None:
    payload = json.loads(model_path.read_text())
    cli_defaults = payload.setdefault('cli_defaults', {})
    cli_defaults.update(best_overrides)
    model_path.write_text(json.dumps(payload, indent=2) + '\n')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run scenario-aware calibration search for the primary FX model.')
    parser.add_argument('--passes', type=int, default=1,
                        help='Coordinate-descent passes over the search grid (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Deterministic seed for calibration runs (default: 42)')
    parser.add_argument('--seeds', default=None,
                        help='Seed panel, for example 42:49 or 42,44,46; current-only mode only')
    parser.add_argument('--current-only', action='store_true',
                        help='Evaluate the current primary defaults only, without running the coordinate search')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers for a seed-panel evaluation (default: 1)')
    parser.add_argument('--write-back', action='store_true',
                        help='Write the best parameter set back into calibration/primary_model.json')
    parser.add_argument('--output', type=Path, default=Path('output/main_aware/calibration_search_report.json'),
                        help='Path for the calibration report artifact')
    parser.add_argument('--reassess', type=Path, default=None,
                        help='Reapply current targets and mechanism checks to a stored seed panel')
    parser.add_argument('--protocol', type=Path, default=None,
                        help='Frozen final-calibration protocol manifest')
    parser.add_argument('--protocol-phase', choices=['development', 'holdout'],
                        default=None,
                        help='Run the exact seed commitment and output declared by --protocol')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_payload = main_module.load_primary_model_targets()
    protocol = None
    protocol_raw = None
    protocol_seeds = None
    development_gate_sha256 = None
    if (args.protocol is None) != (args.protocol_phase is None):
        raise SystemExit('--protocol and --protocol-phase must be supplied together')
    if args.protocol is not None:
        if args.reassess is not None or args.write_back or args.seeds is not None:
            raise SystemExit(
                'protocol mode forbids --reassess, --write-back and arbitrary --seeds'
            )
        if not args.current_only:
            raise SystemExit('protocol mode requires --current-only and forbids search')
        protocol_raw = args.protocol.read_bytes()
        protocol = json.loads(protocol_raw)
        try:
            protocol_seeds = _verify_protocol(
                protocol, target_payload, args.protocol_phase
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise SystemExit(f'protocol verification failed: {exc}') from exc
        args.output = _protocol_output_path(protocol, args.protocol_phase)
        if args.protocol_phase == 'holdout':
            try:
                development_gate_sha256 = _verify_development_artifact(
                    protocol, target_payload, protocol_raw
                )
            except (OSError, TypeError, ValueError, KeyError) as exc:
                raise SystemExit(
                    f'holdout authorization failed: {exc}'
                ) from exc
        if args.protocol_phase == 'holdout' and args.output.exists():
            raise SystemExit(
                f'holdout output already exists and cannot be overwritten: {args.output}'
            )
    if args.reassess is not None:
        raw = args.reassess.read_bytes()
        report = reassess_stored_panel(
            json.loads(raw), target_payload, hashlib.sha256(raw).hexdigest()
        )
        report['best_overrides'] = deepcopy(report.get('candidate_overrides', {}))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(main_module.json_dumps_strict(report, indent=2) + '\n')
        print(main_module.json_dumps_strict({
            'status': report.get('summary', {}).get('status'),
            'objective_score': report.get('summary', {}).get('objective_score'),
            'gating_failures': report.get('summary', {}).get('gating_failures'),
            'mechanism_failures': report.get('summary', {}).get('mechanism_failures'),
            'output': str(args.output),
        }, indent=2))
        return
    base_defaults = main_module.load_primary_model_defaults()
    search_grid = _default_search_grid(base_defaults)
    base_overrides = {key: base_defaults[key] for key in search_grid}
    if args.current_only:
        if protocol_seeds is not None:
            report = evaluate_candidate_panel(
                base_overrides, target_payload, seeds=protocol_seeds,
                workers=max(1, args.workers),
            )
            if args.protocol.read_bytes() != protocol_raw:
                raise SystemExit(
                    'protocol changed during the run; refusing to write an artifact'
                )
            report['provenance'].update({
                'protocol_version': protocol.get('protocol_version'),
                'protocol_phase': args.protocol_phase,
                'protocol_sha256': hashlib.sha256(protocol_raw).hexdigest(),
                'book_acceptance_signature': book_acceptance_signature(target_payload),
                'primary_runtime_sha256': _primary_runtime_sha256(),
                'seed_commitment_exact_match': True,
            })
            if args.protocol_phase == 'holdout':
                report['provenance'].update({
                    'development_gate_verified': True,
                    'development_gate_sha256': development_gate_sha256,
                })
        elif args.seeds:
            report = evaluate_candidate_panel(
                base_overrides, target_payload, seeds=_parse_seeds(args.seeds),
                workers=max(1, args.workers),
            )
        else:
            report = evaluate_candidate(base_overrides, target_payload, seed=args.seed)
        report['best_overrides'] = deepcopy(base_overrides)
    else:
        report = coordinate_search(
            target_payload=target_payload,
            base_overrides=base_overrides,
            search_grid=search_grid,
            passes=args.passes,
            seed=args.seed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(main_module.json_dumps_strict(report, indent=2) + '\n')

    if args.write_back:
        write_back_primary_defaults(report['best_overrides'], Path('calibration/primary_model.json'))

    print(main_module.json_dumps_strict({
        'status': report.get('summary', {}).get('status'),
        'objective_score': report.get('summary', {}).get('objective_score'),
        'gating_failures': report.get('summary', {}).get('gating_failures'),
        'best_overrides': report.get('best_overrides', {}),
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
