"""main.py — the entry point of the multi-venue FX agent based model.

The model is configured by calibration/primary_model.json, which is the single
source of every default. Each command names a runner that carries its own seed
commitment and its own provenance, and none of them reimplements another:

    python -m main help                    the commands and what each produces
    python -m main run --preset baseline   one simulation, its summary and plots
    python -m main accept                  the panel against the frozen targets
    python -m main arms                    the resource matched facility arms
    python -m main welfare --arm reserve   the welfare account for one arm
    python -m main selection               the markout of the flow each venue fills
    python -m main figures                 the figures of the article
    python -m main calibrate               the coordinate search
    python -m main config                  where each calibrated value came from

An episode is named with --preset on the run command. The declared ones are
baseline, mm_withdrawal, flash_crash, dealer_liquidity_crisis,
funding_liquidity_shock, dash_for_cash_2020, dealer_capacity_contagion and
high_vol_stress; the article rests on dash_for_cash_2020. Every model parameter
is exposed as a flag on run, and a flag given explicitly overrides the episode
it belongs to:

    python -m main run --preset dash_for_cash_2020 --seed 42
    python -m main run --shock-iter 350 --shock-mode realism --fundamental-shock-pct -12

The entry point takes no comparison against the same market without a facility.
Adding one moves committed capital, the obligation to keep quoting and the
pricing rule at once, and a single switch pools the three; `arms` separates
them and is the sanctioned comparison. Routing follows the liquidity aware rule
by default, with --venue-choice-rule fixed_share available as the placebo that
holds the facility open while taking the routing away from it.
"""

import argparse
import json
import math
import os
import re
import sys
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from AgentBasedModel.simulator.simulator import Simulator, calibrated_default
from AgentBasedModel.visualization.dashboards import (
    generate_all_dashboards,
    save_all_individual_plots,
)
from calibration.fitter import build_acceptance_report


PRIMARY_MODEL_PATH = Path(__file__).resolve().parent / "calibration" / "primary_model.json"
PRIMARY_TARGETS_PATH = Path(__file__).resolve().parent / "calibration" / "primary_model_targets.json"


# What CRISIS_PRESET holds when the manifest could not be read. It is not the
# name of any episode, so the module imports and every command that would have
# run an episode stops on an unknown preset. The fallback used to be the name
# of a real episode of the primary pair, which on this branch is the one
# outcome the function below says it refuses: the wrong episode measured
# against the wrong crisis targets, silently.
UNNAMED_EPISODE = '__no_episode_declared__'


def _crisis_scenario() -> str:
    """The episode this calibration studies, named once in the manifest.

    A manifest that cannot be read at all leaves a sentinel, since the model
    has to be importable without one, and the sentinel names no episode, so
    nothing runs on it. A manifest that reads but does not name the episode is
    a different matter: on a branch calibrated to another pair that would run
    the wrong episode against the wrong crisis targets and say nothing, so it
    is refused outright.
    """
    try:
        spec = load_primary_model_spec()
    except (OSError, TypeError, ValueError):
        return UNNAMED_EPISODE
    try:
        return str(spec['crisis_scenario'])
    except (KeyError, TypeError, ValueError):
        raise KeyError(
            'calibration/primary_model.json declares no crisis_scenario. The '
            'episode every crisis measurement is taken on has to be named '
            'there, since a default would run one pair\'s episode against '
            "another pair's targets without saying so.")



def load_primary_model_spec(config_path: Optional[Path] = None) -> dict:
    path = PRIMARY_MODEL_PATH if config_path is None else Path(config_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def load_primary_model_defaults(config_path: Optional[Path] = None) -> dict:
    payload = load_primary_model_spec(config_path)
    return payload.get("cli_defaults", {})


def load_primary_model_targets(path: Optional[Path] = None) -> dict:
    target_path = PRIMARY_TARGETS_PATH if path is None else Path(path)
    if not target_path.exists():
        return {}
    try:
        with target_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _infer_acceptance_scenario(args: argparse.Namespace) -> str:
    if args.preset == 'baseline':
        return 'baseline_primary'
    if args.preset in REALISM_PRESETS:
        return args.preset
    if args.shock_iter is not None:
        return 'custom_shock'
    if args.stress_start >= 0:
        return 'custom_stress'
    return 'baseline_primary'


def _apply_primary_model_parser_defaults(parser: argparse.ArgumentParser):
    defaults = load_primary_model_defaults()
    if not defaults:
        return
    for action in parser._actions:
        if action.dest in defaults:
            action.default = defaults[action.dest]


_HELP_DEFAULT_PATTERN = re.compile(r'\(default:\s*([^)]+)\)')


def _format_action_default_for_help(action: argparse.Action, original_fragment: str) -> str:
    value = action.default
    fragment = original_fragment.strip().lower()

    if isinstance(value, bool):
        return 'on' if value else 'off'
    if value is None:
        return 'off' if 'off' in fragment else 'None'

    if isinstance(value, float):
        numeric = f'{value:g}'
    else:
        numeric = str(value)

    if '= off' in fragment:
        return f'{numeric} = off' if isinstance(value, (int, float)) and float(value) < 0 else numeric
    if '=' in fragment and 'bps' in fragment and isinstance(value, (int, float)):
        return f'{numeric} = {float(value) * 10_000:g} bps'
    if 'bps' in fragment:
        return f'{numeric} bps'
    if fragment.endswith('x'):
        return f'{numeric}x'
    return numeric


def _sync_parser_help_defaults(parser: argparse.ArgumentParser):
    for action in parser._actions:
        help_text = getattr(action, 'help', None)
        if not help_text or '(default:' not in help_text:
            continue

        match = _HELP_DEFAULT_PATTERN.search(help_text)
        if not match:
            continue

        default_text = _format_action_default_for_help(action, match.group(1))
        action.help = _HELP_DEFAULT_PATTERN.sub(f'(default: {default_text})', help_text, count=1)


def _same_value(lhs, rhs) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1e-9)
    return lhs == rhs


def _primary_model_overrides(args: argparse.Namespace) -> list[dict]:
    defaults = load_primary_model_defaults()
    overrides = []
    for key, expected in defaults.items():
        if not hasattr(args, key):
            continue
        actual = getattr(args, key)
        if not _same_value(actual, expected):
            overrides.append({
                'field': key,
                'primary_value': expected,
                'run_value': actual,
            })
    return overrides


def _primary_run_label(args: argparse.Namespace) -> str:
    overrides = _primary_model_overrides(args)
    if not overrides:
        return 'primary'
    short = ', '.join(item['field'] for item in overrides[:4])
    if len(overrides) > 4:
        short += f', +{len(overrides) - 4} more'
    return f'ablation(primary - {short})'


def json_dumps_strict(payload, **kwargs) -> str:
    """Serialise reports as standards-compliant JSON.

    Calibration metrics legitimately contain unavailable observations. The
    default Python encoder writes those as ``NaN``, which is not part of the
    JSON standard and cannot be read by strict downstream consumers. Keep the
    in-memory NaN semantics, but publish unavailable values as JSON ``null``.
    """
    def sanitise(value):
        if isinstance(value, dict):
            return {key: sanitise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitise(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    return json.dumps(sanitise(payload), allow_nan=False, **kwargs)


def save_primary_model_artifacts(args: argparse.Namespace, out_dir: str,
                                 acceptance_report: Optional[dict] = None):
    os.makedirs(out_dir, exist_ok=True)
    spec = load_primary_model_spec()
    targets = load_primary_model_targets()
    overrides = _primary_model_overrides(args)
    run_manifest = {
        'run_label': _primary_run_label(args),
        'generated_at_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'primary_model_version': spec.get('version'),
        'primary_model_name': spec.get('model_name'),
        'pair_class': spec.get('pair_class'),
        'session_scope': spec.get('session_scope'),
        'calibration_spec_file': PRIMARY_TARGETS_PATH.name if targets else None,
        'overrides_from_primary': overrides,
        'acceptance_summary': acceptance_report.get('summary') if acceptance_report else None,
    }
    (Path(out_dir) / 'primary_model_manifest.json').write_text(
        json_dumps_strict(spec, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    if targets:
        (Path(out_dir) / 'primary_model_targets.json').write_text(
            json_dumps_strict(targets, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    if acceptance_report:
        (Path(out_dir) / 'primary_acceptance_report.json').write_text(
            json_dumps_strict(acceptance_report, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    (Path(out_dir) / 'run_manifest.json').write_text(
        json_dumps_strict(run_manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


# ── Reproducibility: seed BOTH RNGs ─────────────────────────────
def _seed_all(seed: int):
    import random as _rnd
    _rnd.seed(seed)
    np.random.seed(seed)


REALISM_PRESETS = {
    "baseline": dict(
        shock_mode="realism",
        clob_amm_interaction="competition",
    ),
    "mm_withdrawal": dict(
        shock_iter=350,
        shock_mode="realism",
        clob_amm_interaction="none",
        # Isolated dealer-withdrawal event: dealers pull quotes under one-sided
        # flow, but AMM presence does not directly change CLOB MM risk-taking.
        # This keeps the withdrawal trigger exogenous to AMM presence so the
        # scenario can study whether AMMs stabilize the market after MM retreat.
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=135.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.54,
        force_mm_pause=False,
        funding_vol_shock_intensity=0.35,
        arb_trade_fraction_cap=0.08,
        # Recovery is slower than the original soft preset, but materially
        # faster than the full dealer-liquidity-crisis scenario.
        reprice_prob_recovery=0.045,
        anchor_strength_recovery=0.022,
        bg_target_ratio_recovery=0.085,
        toxic_flow_decay=0.83,
        liquidity_shock_decay=0.87,
        mm_withdraw_threshold=0.70,
        mm_reentry_threshold=0.40,
        mm_loss_threshold_bps=50.0,
        mm_min_withdraw_ticks=4,
        mm_reentry_ticks=2,
        mm_withdraw_confirmation_ticks=2,
    ),
    "flash_crash": dict(
        shock_iter=350,
        shock_mode="realism",
        clob_amm_interaction="none",
        # Pure microstructure dislocation: aggressive sell sweep + sudden
        # quote withdrawal, but without a large permanent fair-value shift.
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=320.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.90,
        force_mm_pause=True,
        funding_vol_shock_intensity=0.55,
        # Flash crashes overshoot sharply, then partially mean-revert faster
        # than a dealer balance-sheet crisis once liquidity reappears.
        arb_trade_fraction_cap=0.12,
        reprice_prob_recovery=0.070,
        anchor_strength_recovery=0.040,
        bg_target_ratio_recovery=0.085,
        toxic_flow_decay=0.80,
        liquidity_shock_decay=0.84,
    ),
    "dealer_liquidity_crisis": dict(
        shock_iter=350,
        shock_mode="realism",
        clob_amm_interaction="competition",
        # Fundamental dislocation: constrained dealers cannot absorb flow
        # → price discovery breaks (BIS WP 1073, Huang et al.)
        fundamental_shock_pct=-3.0,
        # Large directional sweep typical of institutional panic selling
        order_flow_shock_qty=200.0,
        order_flow_shock_side="sell",
        # Deep cancellation wave: dealers pull quotes (Mancini et al.)
        liquidity_shock_frac=0.60,
        force_mm_pause=False,
        # Volatility & funding spike (Bollerslev & Melvin)
        funding_vol_shock_intensity=0.80,
        # Restrict arbitrageur capacity during stress
        arb_trade_fraction_cap=0.05,
        # Slow recovery: dealer constraints persist (Lo & Hall resiliency)
        reprice_prob_recovery=0.015,
        anchor_strength_recovery=0.008,
        bg_target_ratio_recovery=0.025,
        toxic_flow_decay=0.93,
        liquidity_shock_decay=0.95,
        # Dealer exits are generated by the common inventory/P&L rule with
        # heterogeneous risk capacities; the preset does not impose a pause
        # or target a literature-free dealer-exit percentage. Re-entry stays
        # sticky so the recovery is gradual.
        mm_withdraw_threshold=0.70,
        mm_reentry_threshold=0.40,
        mm_withdraw_confirmation_ticks=2,
    ),
    "funding_liquidity_shock": dict(
        shock_iter=350,
        shock_mode="realism",
        clob_amm_interaction="competition",
        # Macro funding squeeze without a permanent fair-value repricing:
        # spreads widen and liquidity thins, but the latent anchor stays put.
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=140.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.35,
        funding_vol_shock_intensity=1.00,
        arb_trade_fraction_cap=0.06,
        # Slower than a flash crash, faster than a dealer balance-sheet crisis.
        reprice_prob_recovery=0.040,
        anchor_strength_recovery=0.018,
        bg_target_ratio_recovery=0.060,
        toxic_flow_decay=0.90,
        liquidity_shock_decay=0.91,
    ),
    "dash_for_cash_2020": dict(
        shock_iter=350,
        shock_mode="realism",
        clob_amm_interaction="competition",
        # March 2020, in character and not in matched time. A run is a
        # thousand seconds and this stress persists for hundreds, while the
        # episode itself unfolded over weeks: what is taken from it is the
        # composition of the shock and the order of the responses, so that
        # the stress and the dealer reaction overlap as they did then.
        # The defining feature for a major pair was an acute dollar
        # funding squeeze against binding dealer balance sheets, with the spot
        # rate itself moving far less than funding conditions did, so funding
        # carries the episode and the fundamental displacement is modest.
        fundamental_shock_pct=-1.2,
        order_flow_shock_qty=190.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.55,
        funding_vol_shock_intensity=1.00,
        force_mm_pause=False,
        arb_trade_fraction_cap=0.05,
        # The episode ran for weeks, so its stress has to outlast the time a
        # dealer takes to act on it. At the decay the synthetic presets carry,
        # a half life of about 13 seconds, the stress is gone before the sector
        # responds: measured on the crisis window, impaired capacity rises past
        # its threshold only after friction has fallen to 0.064, so anything
        # keyed on capacity has nothing left to work on. At a half life near
        # 230 seconds the two overlap and friction while capacity is impaired
        # is 0.718.
        reprice_prob_recovery=0.015,
        anchor_strength_recovery=0.008,
        bg_target_ratio_recovery=0.025,
        toxic_flow_decay=0.996,
        liquidity_shock_decay=0.997,
        # The volatility and funding cost limb, on the same clock as the two
        # above. Left undeclared it ran at an eleven second half life while
        # they ran at hundreds, so anything keyed on volatility saw a spike
        # where the episode has a plateau.
        stress_overlay_decay=0.997,
        mm_withdraw_threshold=0.70,
        mm_reentry_threshold=0.40,
        mm_withdraw_confirmation_ticks=2,
    ),
    "dealer_capacity_contagion": dict(
        shock_iter=350,
        shock_mode="realism",
        clob_amm_interaction="competition",
        # The displacement is small by design and the dislocation is meant to
        # come from the capacity channel, so this scenario is informative only
        # against the same run with the cascade gain set to zero.
        fundamental_shock_pct=-1.0,
        order_flow_shock_qty=150.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.42,
        funding_vol_shock_intensity=0.50,
        force_mm_pause=False,
        arb_trade_fraction_cap=0.06,
        # Stress outlasts the dealer reaction here for the same reason it does
        # in the episode above, since a channel keyed on dealer capacity is
        # inert wherever the two do not overlap.
        reprice_prob_recovery=0.030,
        anchor_strength_recovery=0.015,
        bg_target_ratio_recovery=0.050,
        toxic_flow_decay=0.996,
        liquidity_shock_decay=0.997,
        # The volatility and funding cost limb, on the same clock as the two
        # above. Left undeclared it ran at an eleven second half life while
        # they ran at hundreds, so anything keyed on volatility saw a spike
        # where the episode has a plateau.
        stress_overlay_decay=0.997,
        mm_withdraw_threshold=0.70,
        mm_reentry_threshold=0.40,
        mm_withdraw_confirmation_ticks=2,
    ),
    "high_vol_stress": dict(
        shock_mode="realism",
        clob_amm_interaction="none",
        stress_start=300,
        stress_end=500,
        sigma_low=0.01,
        sigma_high=0.06,
        c_low=0.002,
        c_high=0.015,
    ),
}


# One family remains. The bundles that named a fixed split between the
# venues went with the counterfactual they served.
PRESETS = dict(REALISM_PRESETS)

# Read once at import so a runner cannot disagree with the manifest.
CRISIS_PRESET = _crisis_scenario()


def _format_preset_help() -> str:
    return ("Named parameter bundle (overridden by explicit flags).\n"
            "Episodes: " + ', '.join(REALISM_PRESETS.keys()))


# ── Auto-generate stress around shock (realism) ─────────────────
def _auto_stress_around_shock(args: argparse.Namespace):
    """
    If shock is set and the user explicitly asks for coupled
    regime-stress, auto-generate a realistic post-shock stress window.

    Pure shocks already trigger an endogenous microstructure aftermath
    inside MarketEnvironment.apply_shock().  This helper is only for a
    second, slower exogenous regime-stress layer.
    """
    if args.shock_iter is None:
        return
    if getattr(args, 'no_shock_stress', False):
        return
    if not getattr(args, 'shock_regime_stress', False):
        return
    if args.stress_start >= 0:
        return  # already configured

    shock_mag = abs(args.shock_pct)
    recovery_tail = max(100, int(shock_mag * 10))
    args.stress_start = max(0, args.shock_iter)
    args.stress_end = min(args.n_iter, args.shock_iter + recovery_tail)

    sigma_mult = max(3.0, 1.0 + shock_mag / 5.0)
    args.sigma_high = round(args.sigma_low * sigma_mult, 4)

    c_mult = max(2.0, 1.0 + shock_mag / 8.0)
    args.c_high = round(args.c_low * c_mult, 4)


def _shock_iter_from_sim(sim: Simulator):
    shock_iter = getattr(sim, 'shock_iter', None)
    if shock_iter is None:
        shock_iter = getattr(sim, '_shock_iter', None)
    return shock_iter


def _linkage_split_point(sim: Simulator):
    """Prefer explicit regime stress onset; otherwise split on the shock tick."""
    stress_start = getattr(sim.env, 'stress_start', None) if getattr(sim, 'env', None) else None
    if stress_start is not None and stress_start >= 0:
        return int(stress_start), 'Stress', 'Normal', 'Stress'

    shock_iter = _shock_iter_from_sim(sim)
    if shock_iter is not None and shock_iter >= 0:
        return int(shock_iter), 'Shock', 'Pre-shock', 'Post-shock'

    return None, None, None, None


def _finite_avg(values):
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else float('nan')


def _shock_window_slices(n: int, shock_iter: int):
    windows = [
        ('Pre', max(0, shock_iter - 50), max(0, shock_iter)),
        ('t0..t0+5', shock_iter, min(n, shock_iter + 5)),
        ('t0+5..t0+20', min(n, shock_iter + 5), min(n, shock_iter + 20)),
        ('t0+20..t0+100', min(n, shock_iter + 20), min(n, shock_iter + 100)),
    ]
    return [(label, start, end) for label, start, end in windows if end > start]


def _window_average(series, start: int, end: int):
    return _finite_avg(series[start:end])


def _rolling_normalization_time(series, *, shock_iter: int,
                                baseline: float,
                                direction: str,
                                rel_tol: float,
                                abs_tol: float = 0.0,
                                window: int = 5,
                                horizon: int = 100):
    # Both callers unpack a pair, so an unmeasurable baseline has to return
    # one too. Returning a bare nan here raised a TypeError at the call site
    # instead of reporting that the quantity could not be measured.
    if not math.isfinite(baseline):
        return float('nan'), float('nan')

    values = pd.Series(
        [x if math.isfinite(x) else np.nan for x in series],
        dtype='float64',
    )
    post = values.iloc[shock_iter:min(len(values), shock_iter + horizon)].reset_index(drop=True)
    rolled = post.rolling(window, min_periods=window).median()
    if direction == 'upper':
        target = max(baseline * (1.0 + rel_tol), baseline + abs_tol)
        recovered = rolled <= target
    else:
        target = baseline * (1.0 - rel_tol)
        recovered = rolled >= target

    hits = recovered[recovered].index.tolist()
    if not hits:
        return float('inf'), target
    return max(0, hits[0]), target


def _series_trough(series, *, shock_iter: int, direction: str, horizon: int = 100):
    end = min(len(series), shock_iter + horizon)
    post = [x for x in series[shock_iter:end]]
    if not post:
        return float('nan'), float('nan')

    best_idx = None
    best_val = None
    for idx, value in enumerate(post):
        if not math.isfinite(value):
            continue
        if best_val is None:
            best_val = value
            best_idx = idx
            continue
        if direction == 'upper' and value > best_val:
            best_val = value
            best_idx = idx
        if direction == 'lower' and value < best_val:
            best_val = value
            best_idx = idx
    if best_idx is None:
        return float('nan'), float('nan')
    return best_val, float(best_idx)


def _replenishment_speed(series, *, shock_iter: int, direction: str):
    start = shock_iter + 5
    end = shock_iter + 20
    if start >= len(series):
        return float('nan')
    start_val = _window_average(series, shock_iter, min(len(series), start))
    end_val = _window_average(series, min(len(series), start), min(len(series), end))
    if not (math.isfinite(start_val) and math.isfinite(end_val)):
        return float('nan')
    periods = max(1, min(len(series), end) - min(len(series), start))
    if direction == 'upper':
        return (start_val - end_val) / periods
    return (end_val - start_val) / periods


def _apply_preset_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if not args.preset:
        return

    preset_vals = PRESETS[args.preset]
    for k, v in preset_vals.items():
        cli_key = k.replace("-", "_")
        if cli_key in vars(args):
            default_val = parser.get_default(cli_key)
            current_val = getattr(args, cli_key)
            if current_val == default_val:
                setattr(args, cli_key, v)
        else:
            setattr(args, cli_key, v)


def _median_finite(values):
    clean = [x for x in values if math.isfinite(x)]
    if not clean:
        return float('nan')
    return float(statistics.median(clean))


def _mean_finite(values):
    clean = [x for x in values if math.isfinite(x)]
    if not clean:
        return float('nan')
    return float(sum(clean) / len(clean))


def _std_finite(values):
    clean = [x for x in values if math.isfinite(x)]
    if len(clean) < 2:
        return 0.0 if clean else float('nan')
    return float(statistics.pstdev(clean))


def _window_bounds(n: int, start: int, end: int):
    lo = max(0, min(n, start))
    hi = max(lo, min(n, end))
    return lo, hi


def _shock_metric_snapshot(sim: Simulator):
    logger = sim.logger
    shock_iter = _shock_iter_from_sim(sim)
    if shock_iter is None:
        return None

    n = len(logger.iterations)
    pre_start, pre_end = _window_bounds(n, shock_iter - 30, shock_iter)
    shock_start, shock_end = _window_bounds(n, shock_iter, shock_iter + 5)
    post_start, post_end = _window_bounds(n, shock_iter + 5, shock_iter + 20)

    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    basis_series = logger.max_venue_basis_series()

    pre_spread = _median_finite(logger.clob_qspr[pre_start:pre_end])
    shock_spread = _median_finite(logger.clob_qspr[shock_start:shock_end])
    post_spread = _median_finite(logger.clob_qspr[post_start:post_end])
    pre_depth = _median_finite(depth_series[pre_start:pre_end])
    shock_depth = _median_finite(depth_series[shock_start:shock_end])
    post_depth = _median_finite(depth_series[post_start:post_end])
    shock_liq = _median_finite(logger.systemic_liquidity_series[shock_start:shock_end])
    post_liq = _median_finite(logger.systemic_liquidity_series[post_start:post_end])
    max_basis = _mean_finite(basis_series[shock_start:post_end])

    depth_ratio = float('nan')
    if math.isfinite(pre_depth) and pre_depth > 0 and math.isfinite(shock_depth):
        depth_ratio = shock_depth / pre_depth

    spread_ratio = float('nan')
    if math.isfinite(pre_spread) and pre_spread > 0 and math.isfinite(shock_spread):
        spread_ratio = shock_spread / pre_spread

    recovery_times = []
    metrics = [
        (logger.clob_qspr, 'upper', 0.50, 10.0),
        (depth_series, 'lower', 0.25, 0.0),
        (logger.systemic_liquidity_series, 'lower', 0.15, 0.0),
        (basis_series, 'upper', 0.25, 5.0),
    ]
    for series, direction, rel_tol, abs_tol in metrics:
        baseline = _median_finite(series[pre_start:pre_end])
        rt, _target = _rolling_normalization_time(
            series,
            shock_iter=shock_iter,
            baseline=baseline,
            direction=direction,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            window=5,
            horizon=100,
        )
        if math.isfinite(rt):
            recovery_times.append(rt)
        elif rt == float('inf'):
            recovery_times.append(rt)

    system_recovery = float('inf') if any(rt == float('inf') for rt in recovery_times) else (max(recovery_times) if recovery_times else float('nan'))

    return {
        'mode': 'shock',
        'pre_spread_bps': pre_spread,
        'shock_spread_bps': shock_spread,
        'post_spread_bps': post_spread,
        'spread_ratio': spread_ratio,
        'pre_depth': pre_depth,
        'shock_depth': shock_depth,
        'post_depth': post_depth,
        'depth_ratio': depth_ratio,
        'shock_systemic_liquidity': shock_liq,
        'post_systemic_liquidity': post_liq,
        'avg_basis_bps_0_20': max_basis,
        'system_recovery_ticks': system_recovery,
    }


def _stress_metric_snapshot(sim: Simulator):
    logger = sim.logger
    stress_start = getattr(sim.env, 'stress_start', None) if sim.env is not None else None
    if stress_start is None:
        return None

    n = len(logger.iterations)
    pre_start, pre_end = _window_bounds(n, stress_start - 50, stress_start)
    stress_start_i, stress_end_i = _window_bounds(n, stress_start, min(n, stress_start + 100))
    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]

    pre_spread = _mean_finite(logger.clob_qspr[pre_start:pre_end])
    stress_spread = _mean_finite(logger.clob_qspr[stress_start_i:stress_end_i])
    pre_depth = _mean_finite(depth_series[pre_start:pre_end])
    stress_depth = _mean_finite(depth_series[stress_start_i:stress_end_i])
    pre_liq = _mean_finite(logger.systemic_liquidity_series[pre_start:pre_end])
    stress_liq = _mean_finite(logger.systemic_liquidity_series[stress_start_i:stress_end_i])
    pre_clob_share = logger.customer_volume_share('clob', pre_start, pre_end)
    stress_clob_share = logger.customer_volume_share(
        'clob', stress_start_i, stress_end_i
    )

    spread_ratio = float('nan')
    if math.isfinite(pre_spread) and pre_spread > 0 and math.isfinite(stress_spread):
        spread_ratio = stress_spread / pre_spread

    depth_ratio = float('nan')
    if math.isfinite(pre_depth) and pre_depth > 0 and math.isfinite(stress_depth):
        depth_ratio = stress_depth / pre_depth

    return {
        'mode': 'stress',
        'pre_spread_bps': pre_spread,
        'stress_spread_bps': stress_spread,
        'spread_ratio': spread_ratio,
        'pre_depth': pre_depth,
        'stress_depth': stress_depth,
        'depth_ratio': depth_ratio,
        'pre_systemic_liquidity': pre_liq,
        'stress_systemic_liquidity': stress_liq,
        'pre_clob_customer_volume_share': pre_clob_share,
        'stress_clob_customer_volume_share': stress_clob_share,
    }


def _baseline_metric_snapshot(sim: Simulator):
    logger = sim.logger
    n = len(logger.iterations)
    tail_start, tail_end = _window_bounds(n, max(0, n - 100), n)
    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    return {
        'mode': 'baseline',
        'avg_spread_bps': _mean_finite(logger.clob_qspr[tail_start:tail_end]),
        'avg_depth': _mean_finite(depth_series[tail_start:tail_end]),
        'avg_systemic_liquidity': _mean_finite(logger.systemic_liquidity_series[tail_start:tail_end]),
    }


def _robustness_snapshot(sim: Simulator):
    shock_snapshot = _shock_metric_snapshot(sim)
    if shock_snapshot is not None:
        return shock_snapshot
    stress_snapshot = _stress_metric_snapshot(sim)
    if stress_snapshot is not None:
        return stress_snapshot
    return _baseline_metric_snapshot(sim)


def _fmt_stat(mean_value: float, std_value: float, *, precision: int = 2, pct: bool = False):
    if not math.isfinite(mean_value):
        return 'N/A'
    suffix = '%' if pct else ''
    if not math.isfinite(std_value):
        return f'{mean_value:.{precision}f}{suffix}'
    return f'{mean_value:.{precision}f} ± {std_value:.{precision}f}{suffix}'


def print_robustness_summary(args: argparse.Namespace):
    seeds = max(2, int(args.robustness_seeds))
    base_seed = args.robustness_base_seed
    if base_seed is None:
        base_seed = args.seed if args.seed is not None else 42

    snapshots = []
    for offset in range(seeds):
        seed = base_seed + offset
        _seed_all(seed)
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        sim = build_sim(run_args)
        sim.simulate(run_args.n_iter, silent=True)
        snapshot = _robustness_snapshot(sim)
        snapshot['seed'] = seed
        snapshots.append(snapshot)

    mode = snapshots[0]['mode'] if snapshots else 'baseline'
    metrics = sorted(k for k in snapshots[0].keys() if k not in {'mode', 'seed'}) if snapshots else []

    W = 65
    print("\n" + "-" * W)
    print("  CROSS SEED DISPERSION")
    print("-" * W)
    print(f"  Seeds: {base_seed}..{base_seed + seeds - 1}  (n={seeds})")
    print(f"  Scenario type: {mode}")
    print("\n  Metric means ± cross-seed std:")

    for metric in metrics:
        values = [snap[metric] for snap in snapshots]
        avg = _mean_finite(values)
        std = _std_finite(values)
        print(f"    {metric:<28s} {_fmt_stat(avg, std):>18s}")

    if mode == 'shock':
        stable = [snap for snap in snapshots if math.isfinite(snap['spread_ratio']) and math.isfinite(snap['depth_ratio'])]
        if stable:
            spread_ok = sum(snap['spread_ratio'] > 1.5 for snap in stable)
            depth_ok = sum(snap['depth_ratio'] < 0.8 for snap in stable)
            liq_ok = sum(snap['post_systemic_liquidity'] >= snap['shock_systemic_liquidity'] for snap in stable)
            print("\n  Quick checks:")
            print(f"    spread widens after shock: {spread_ok}/{len(stable)} seeds")
            print(f"    depth drops on impact:     {depth_ok}/{len(stable)} seeds")
            print(f"    liquidity rebounds by t+20:{liq_ok}/{len(stable)} seeds")
    elif mode == 'stress':
        stable = [snap for snap in snapshots if math.isfinite(snap['spread_ratio']) and math.isfinite(snap['depth_ratio'])]
        if stable:
            spread_ok = sum(snap['spread_ratio'] > 1.1 for snap in stable)
            depth_ok = sum(snap['depth_ratio'] < 0.95 for snap in stable)
            print("\n  Quick checks:")
            print(f"    spread wider in stress:    {spread_ok}/{len(stable)} seeds")
            print(f"    depth lower in stress:     {depth_ok}/{len(stable)} seeds")

    print("-" * W)


def build_parser(default_venue_choice_rule: str = "liquidity_aware") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-venue FX Agent-Based Model — interactive demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    g = p.add_argument_group("General")
    g.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                    help=_format_preset_help())
    g.add_argument("--n-iter", type=int, default=1000,
                   help="Number of simulation iterations (default: 1000)")
    g.add_argument("--price", type=float, default=100.0,
                   help="Initial FX mid-price (default: 100)")
    g.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    g.add_argument("--silent", action="store_true",
                   help="Suppress progress bar")
    g.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    g.add_argument("--no-summary", action="store_true",
                   help="Skip text summary")
    g.add_argument("--robustness-check", action="store_true",
                   help="Run a short multi-seed sanity check after the main simulation")
    g.add_argument("--robustness-seeds", type=int, default=5,
                   help="Number of sequential seeds used in robustness check (default: 5)")
    g.add_argument("--robustness-base-seed", type=int, default=None,
                   help="First seed for robustness check (default: --seed or 42)")
    g.add_argument("--spillover-artifacts", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Save spillover diagnostics (CSV + PNG) after each main run (default: on)")
    g.add_argument("--spillover-lag", type=int, default=1,
                   help="Lag used in directional spillover regressions (default: 1)")
    g.add_argument("--spillover-roll-window", type=int, default=30,
                   help="Rolling window for liquidity-change correlation in spillover chart (default: 30)")

    g = p.add_argument_group(
        "Balance sheets / solvency",
        "Shared feasibility and financing layer applied to CLOB and AMM execution.\n"
        "  Negative cash pays funding carry.\n"
        "  Short inventory pays an additional borrow carry.\n"
        "  Maintenance breaches trigger forced deleveraging or default."
    )
    g.add_argument("--maintenance-margin-ratio", type=float, default=0.06,
                   help="Maintenance margin requirement as a share of gross exposure (default: 0.06)")
    g.add_argument("--liquidation-fraction", type=float, default=0.75,
                   help="Fraction of position targeted in one forced deleveraging step (default: 0.75)")
    g.add_argument("--borrow-spread-multiplier", type=float, default=0.6,
                   help="Multiplier on funding cost applied to negative cash balances (default: 0.6)")
    g.add_argument("--short-borrow-spread-multiplier", type=float, default=0.8,
                   help="Multiplier on funding cost applied to short inventory notional (default: 0.8)")

    g = p.add_argument_group(
        "CLOB agents",
        "Traders that populate the central limit order book.\n"
        "  Noise traders place random limit/market/cancel orders.\n"
        "  FastRecyclerLP recycle short-lived near-mid liquidity.\n"
        "  Fundamentalist place DCF-based limit orders.\n"
        "  Market Maker quotes both sides with σ/c-dependent spread."
    )
    g.add_argument("--n-noise", type=int, default=12,
                   help="CLOB noise/liquidity traders (default: 12)")
    g.add_argument("--n-mm", type=int, default=5,
                   help="Number of CLOB Market Makers (default: 5, 0=off)")
    g.add_argument("--n-fast-lp", type=int, default=10,
                   help="Fast replenishing LPs on the CLOB (default: 10)")
    g.add_argument("--n-clob-fund", type=int, default=2,
                   help="Fundamentalist book agents on CLOB (default: 2)")
    g.add_argument("--clob-fund-observation-noise", type=float,
                   default=float(calibrated_default('clob_fund_observation_noise', 1.0)),
                   help="Dispersion of a book fundamentalist's reading of the "
                        "latent value, as a multiple of the price volatility "
                        "of one period. Zero makes the trader an oracle")
    g.add_argument("--clob-fund-quote-offset", type=float,
                   default=float(calibrated_default('clob_fund_quote_offset', 0.5)),
                   help="How far a book fundamentalist rests from its own "
                        "reading, as a multiple of the price volatility of "
                        "one period, floored at one tick")
    g.add_argument("--enable-clob-mm", type=int, choices=[0, 1], default=1,
                   help="Enable CLOB Market Maker: 1=yes, 0=no (default: 1)")
    g.add_argument("--clob-std", type=float, default=2.0,
                   help="Std of initial order-book price distribution (default: 2.0)")
    g.add_argument("--clob-volume", type=int, default=1000,
                   help="Initial number of orders in book (default: 1000)")
    g.add_argument("--price-tick", type=float,
                   default=float(calibrated_default('price_tick', 0.005)),
                   help="Price grid the book quotes on, in price units. At a "
                        "reference price of 100 a tick of 0.005 is half a "
                        "basis point. The realised quoted spread cannot be "
                        "finer than one tick, so this is a declared model "
                        "input and not an implementation detail")
    g.add_argument("--clob-anchor-strength", type=float, default=0.35,
                   help="Fraction of the fair-price gap closed by the CLOB per tick (default: 0.35)")
    g.add_argument("--clob-anchor-threshold-bps", type=float, default=5.0,
                   help="Ignore tiny fair-price gaps below this level when anchoring the CLOB (default: 5 bps)")
    g.add_argument("--clob-near-mid-target-ratio", type=float, default=1.0,
                   help="Multiplier for seeded near-mid background liquidity on the CLOB (default: 1.0x)")
    g.add_argument("--clob-support-max-share", type=float, default=0.30,
                   help="Max anonymous support-layer depth near mid as a share of trader-owned near-mid depth (default: 0.30)")
    g.add_argument("--clob-amm-interaction", choices=["none", "competition", "toxicity"],
                   default="competition",
                   help="How AMM conditions affect CLOB MM risk-taking: none, competition, or toxicity (default comes from primary model)")
    g.add_argument("--clob-amm-spread-impact-bps", type=float, default=3.0,
                   help="Sensitivity of MM spread-risk relief/penalty from AMM conditions (default: 3.0)")
    g.add_argument("--clob-amm-depth-impact", type=float, default=60.0,
                   help="Sensitivity of MM inventory/depth relief from AMM conditions (default: 60.0)")
    g.add_argument("--mm-withdraw-threshold", type=float, default=0.7,
                   help="Withdrawal score threshold of the least risk-tolerant bank dealer")
    g.add_argument("--mm-withdraw-threshold-step", type=float,
                   default=float(calibrated_default('mm_withdraw_threshold_step', 0.3)),
                   help="Risk-capacity increment between successive heterogeneous bank dealers")
    g.add_argument("--mm-reentry-threshold", type=float, default=0.4,
                   help="Re-entry score threshold of the least risk-tolerant bank dealer")
    g.add_argument("--mm-loss-threshold-bps", type=float, default=50.0,
                   help="EWMA mark-to-market loss in bps that maps to a unit withdrawal-loss score (default: 20)")
    g.add_argument("--mm-min-withdraw-ticks", type=int, default=4,
                   help="Minimum time endogenous MM stays withdrawn once it retreats (default: 4)")
    g.add_argument("--mm-reentry-ticks", type=int, default=3,
                   help="Ticks spent in the reentering state before returning to normal quoting (default: 3)")
    g.add_argument("--mm-withdraw-confirmation-ticks", type=int,
                   default=int(calibrated_default('mm_withdraw_confirmation_ticks', 2)),
                   help="Consecutive above-threshold observations required for full dealer withdrawal")
    g.add_argument("--mm-alpha0-base", type=float, default=2.1,
                   help="Base MM quoted-spread intercept in bps (default: 2.1)")
    g.add_argument("--mm-alpha0-step", type=float, default=0.3,
                   help="Increment added to alpha0 for each additional MM (default: 0.3)")
    g.add_argument("--mm-alpha1", type=float, default=8.0,
                   help="MM spread sensitivity to volatility sigma (default: 320)")
    g.add_argument("--mm-alpha2", type=float, default=700.0,
                   help="MM spread sensitivity to funding cost c (default: 560)")
    g.add_argument("--mm-alpha3", type=float, default=35.0,
                   help="MM spread sensitivity to order-flow imbalance (default: 35)")
    g.add_argument("--mm-d0-base", type=float, default=40.0,
                   help="Base MM depth intercept before clob-liq scaling. The "
                        "effective value comes from calibration/primary_model.json "
                        "(currently 40)")
    g.add_argument("--mm-d0-step", type=float, default=5.0,
                   help="Depth intercept decrement for each additional MM (default: 5)")
    g.add_argument("--mm-d1", type=float, default=560.0,
                   help="MM depth sensitivity to volatility sigma (default: 620)")
    g.add_argument("--mm-d2", type=float, default=360.0,
                   help="MM depth sensitivity to funding cost c (default: 420)")
    g.add_argument("--mm-d3", type=float, default=12.0,
                   help="MM depth sensitivity to order-flow imbalance (default: 18)")

    g = p.add_argument_group(
        "FX liquidity takers",
        "Agents that route orders between CLOB and AMM pools.\n"
        "  Noise     — random direction, random size [q_min, q_max].\n"
        "  Fundament — trades when |mid − fair_rate| > γ.\n"
        "  Retail    — small orders, high frequency.\n"
        "  Institut. — large orders, low frequency."
    )
    g.add_argument("--n-fx-takers", type=int, default=15,
                   help="Noise takers with venue routing (default: 15)")
    g.add_argument("--n-fx-fund", type=int, default=5,
                   help="Fundamentalist traders (default: 5)")
    g.add_argument("--n-retail", type=int, default=10,
                   help="Retail noise traders (default: 10)")
    g.add_argument("--n-institutional", type=int, default=3,
                   help="Institutional traders (default: 3)")
    g.add_argument("--hedger-flow-persistence", type=float, default=0.24,
                   help="Directional persistence for hedger/noise takers (default: 0.10)")
    g.add_argument("--retail-flow-persistence", type=float, default=0.42,
                   help="Directional persistence for retail-toxic takers (default: 0.28)")
    g.add_argument("--institutional-flow-persistence", type=float, default=0.24,
                   help="Directional persistence for real-money takers (default: 0.18)")

    g = p.add_argument_group(
        "Flow allocation  (CLOB ↔ AMM split)",
        "Step 1 — AMM vs CLOB: coin flip with P(AMM) = amm-share/100.\n"
        "Step 2 — within AMM, CPMM vs HFMM: logit softmax with β_AMM,\n"
        "         adjusted by CPMM bias and cost noise.\n"
        "Optional: liquidity_aware makes the top-level venue choice\n"
        "          respond to cost, depth, and venue price alignment."
    )
    g.add_argument("--amm-share", type=float, default=22.0,
                   dest="amm_share_pct",
                   help="AMM routing prior in %% (0–100, default: 22). Under liquidity_aware this is a weak prior, not a target share; explicit use opts into legacy fixed_share")
    g.add_argument("--venue-choice-rule", choices=["fixed_share", "liquidity_aware"],
                   default=default_venue_choice_rule,
                   help="Top-level routing regime: liquidity_aware is the default; fixed_share preserves the legacy two-step AMM/CLOB split")
    g.add_argument("--deterministic", action="store_true",
                   help="Use deterministic argmin venue choice (overrides any stochastic routing rule)")
    g.add_argument("--beta-amm", type=float, default=0.05,
                   help="Logit sensitivity for CPMM vs HFMM (default: 0.05)")
    g.add_argument("--cpmm-bias", type=float, default=0.0,
                   dest="cpmm_bias_bps",
                   help="CPMM non-monetary cost discount in bps (default: 5)")
    g.add_argument("--cost-noise", type=float, default=1.5,
                   dest="cost_noise_std",
                   help="Std of cost estimation noise in bps (default: 1.5)")
    g.add_argument("--routing-cost-scale", type=float, default=4.0,
                   dest="routing_cost_scale_bps",
                   help="Cost-gap scale of liquidity-aware routing in bps (design parameter; default: 1)")
    g.add_argument("--routing-prior-mix-cap", type=float, default=0.18,
                   dest="routing_prior_mix_cap",
                   help="Maximum weight on the AMM routing prior (design parameter; default: 0.18)")
    g.add_argument("--routing-basis-scale", type=float, default=50.0,
                   dest="routing_basis_scale_bps",
                   help="AMM/CLOB basis scale in the alignment score, bps (default: 50)")
    g.add_argument("--routing-clob-depth-multiple", type=float, default=10.0,
                   help="CLOB depth normalizer per unit requested (default: 10)")
    g.add_argument("--routing-amm-depth-multiple", type=float, default=8.0,
                   help="AMM depth normalizer per unit requested (default: 8)")

    g = p.add_argument_group(
        "Liquidity levels",
        "Global multipliers that scale depth on each side.\n"
        "  clob-liq × → n_noise, n_fast_lp, clob_volume, MM depth.\n"
        "  amm-liq  × → CPMM / HFMM reserves."
    )
    g.add_argument("--clob-liq", type=float, default=1.0,
                   help="CLOB liquidity multiplier (default: 1.0)")
    g.add_argument("--amm-liq", type=float, default=1.0,
                   help="AMM liquidity multiplier (default: 1.0)")
    g.add_argument("--match-initial-depth", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Optional calibration aid: match initial live CLOB near-mid depth to aggregate AMM effective depth (default: off)")
    g.add_argument("--enable-amm", type=int, choices=[0, 1], default=1,
                   help="Enable AMM pools: 1=yes, 0=no (default: 1)")
    g.add_argument("--fast-lp-base-withdraw-prob", type=float,
                   default=float(calibrated_default('fast_lp_base_withdraw_prob', 0.10)),
                   help="Probability that the fast automated provider does not quote at all in a period when conditions are calm")
    g.add_argument("--facility-arm",
                   choices=['reserve', 'reserve_frozen', 'dealer_of_last_resort',
                            'passive_book', 'reallocation', 'none'],
                   default='reserve', dest="facility_arm",
                   help="Which arm of the resource matched comparison to run. "
                        "Every arm carries the same committed capital and the "
                        "same inventory capacity and differs only in how it "
                        "prices. reserve_frozen is the same pool with its "
                        "providers unable to resize, enter or leave, which "
                        "separates the flight of provider capital from the "
                        "schedule. reallocation funds the facility out of the "
                        "dealer sector, holding total market capital fixed.")
    g.add_argument("--arm-spread-bps", type=float,
                   default=calibrated_default('arm_spread_bps', 3.469),
                   dest="arm_spread_bps",
                   help="Fixed bid ask spread the obliged quoter shows, in "
                        "basis points, matched to the round trip cost of the "
                        "reserve priced pool weighted by the realised "
                        "distribution of customer trade sizes. The former "
                        "default of 12 was matched to a five basis point fee "
                        "the calibration no longer carries.")
    g.add_argument("--arm-capital", type=float, default=0.0,
                   dest="arm_capital",
                   help="Committed capital of the facility in quote units, "
                        "shared by every arm. Zero leaves each arm at its own "
                        "natural size, which is not a matched comparison.")
    g.add_argument("--mm-core-threshold", type=float,
                   default=float(calibrated_default('mm_core_threshold', 0.0)),
                   dest="mm_core_threshold",
                   help="Withdrawal threshold of the most robust dealer, which "
                        "sits above the ladder the others occupy and makes a "
                        "full evacuation of the sector impossible. Zero leaves "
                        "the ladder linear.")
    g.add_argument("--mm-client-flow-intensity", type=float,
                   default=float(calibrated_default('mm_client_flow_intensity', 0.0)),
                   dest="mm_client_flow_intensity",
                   help="Size of the private client flow each dealer internalises "
                        "per period. Most customer volume in spot FX is "
                        "internalised bilaterally and only the residual reaches "
                        "the interdealer book, so a dealer's position comes "
                        "mainly from a franchise that is its own. Zero removes "
                        "the franchise.")
    g.add_argument("--mm-client-flow-persistence", type=float,
                   default=float(calibrated_default('mm_client_flow_persistence', 0.85)),
                   dest="mm_client_flow_persistence",
                   help="Persistence of a dealer's own client flow.")
    g.add_argument("--mm-softlimit", type=float,
                   default=float(calibrated_default('mm_softlimit', 100.0)),
                   dest="mm_softlimit",
                   help="Inventory beyond which a dealer stands down. It was a "
                        "constructor default of 100 and absent from the manifest "
                        "while the inventory term alone drives the withdrawal "
                        "score, so it set when a dealer leaves.")
    g.add_argument("--dealer-cascade-gain", type=float,
                   default=float(calibrated_default('dealer_cascade_gain', 0.0)),
                   dest="dealer_cascade_gain",
                   help="Strength of the feedback from impaired dealer capacity "
                        "back into systemic liquidity. Zero reproduces the "
                        "earlier model, in which the emptiness of the book made "
                        "conditions no worse.")
    g.add_argument("--dealer-capacity-threshold", type=float,
                   default=float(calibrated_default('dealer_capacity_threshold', 0.5)),
                   dest="dealer_capacity_threshold",
                   help="Share of dealer quoting capacity that must be impaired "
                        "before the feedback acts at all, after BIS Working "
                        "Paper 1138.")
    g.add_argument("--fast-lp-vol-multiple", type=float,
                   default=float(calibrated_default('fast_lp_vol_multiple', 1.0)),
                   dest="fast_lp_vol_multiple",
                   help="Loading on price volatility in the quote of the fast "
                        "automated provider. It compensates the provider for "
                        "leaving a quote exposed for its whole life, so it is "
                        "what makes the quoted spread respond to volatility at "
                        "all. It was a constructor default of 1.0 and absent "
                        "from the manifest until 16.08.2026.")
    g.add_argument("--fast-lp-stress-abstention", type=float,
                   default=float(calibrated_default('fast_lp_stress_abstention', 0.10)),
                   help="Most the fast automated provider abstains from "
                        "quoting at maximum stress. A non-bank on a primary "
                        "venue retreats by about a tenth of its making and "
                        "charges the rest of the risk in the width, so this "
                        "is a small number and not a withdrawal")
    g.add_argument("--fx-flow-intensity-scale", type=float,
                   default=float(calibrated_default('fx_flow_intensity_scale', 1.0)),
                   help="Common scale on ordinary FX taker arrival probabilities; stress sweeps are unchanged")
    g.add_argument("--common-flow-response", type=float,
                   default=float(calibrated_default('common_flow_response', 0.05)),
                   help="Directional response to lagged market-wide order-flow imbalance")
    g.add_argument("--fast-lp-base-spread-bps", type=float,
                   default=float(calibrated_default('fast_lp_base_spread_bps', 1.6)),
                   help="Base half turn spread of the fast automated provider, in bps. It is this class, not the dealer, that holds the touch, so it sets the quoted spread of the market")
    g.add_argument("--fast-lp-quote-life", type=int,
                   default=int(calibrated_default('fast_lp_quote_life', 3)),
                   help="Maximum resting life of a fast non-bank quote in one-second ticks")
    g.add_argument("--fast-lp-base-qty", type=int,
                   default=int(calibrated_default('fast_lp_base_qty', 1)),
                   help="Normal displayed quantity per fast non-bank quote level")
    g.add_argument("--fast-lp-levels", type=int,
                   default=int(calibrated_default('fast_lp_levels', 1)),
                   help="Number of price levels displayed by each fast non-bank provider")
    g.add_argument("--mm-revenue-horizon", type=int,
                   default=int(calibrated_default('mm_revenue_horizon', 300)),
                   help="Horizon in ticks over which the dealer accumulates the trading revenue that enters its withdrawal score")
    g.add_argument("--mm-stale-touch-ratio", type=float,
                   default=float(calibrated_default('mm_stale_touch_ratio', 0.06)),
                   help="Fraction of the spread the dealer would quote now, inside which a resting quote is withdrawn instead of being left to become the best price in the market")
    g.add_argument("--mm-replacement-gain", type=float,
                   default=float(calibrated_default('mm_replacement_gain', 0.0)),
                   dest="mm_replacement_gain",
                   help="How much of the depth a departing provider leaves "
                        "behind the dealers that stay pick up. Zero is the "
                        "model without the channel.")
    g.add_argument("--mm-inv-skew-bps", type=float,
                   default=float(calibrated_default('mm_inv_skew_bps', 0.3)),
                   help="How far the dealer shifts its quoted mid against its own position, in bps per unit of inventory. It sets how fast inventory mean reverts, and the anchor is the median half life of an FX dealer position")
    g.add_argument("--mm-quote-life", type=int,
                   default=int(calibrated_default('mm_quote_life', 290)),
                   help="Maximum resting life of a bank-dealer quote, in "
                        "one-second ticks (default: 290, a heuristic cap based "
                        "on the reported EBS median bank-order life, not a "
                        "matched median). Fills, withdrawal, or "
                        "a reference-price move beyond the refresh tolerance "
                        "can end a realised order earlier")
    g.add_argument("--mm-quote-refresh-tol-bps", type=float,
                   default=float(calibrated_default('mm_quote_refresh_tol_bps', 20.0)),
                   help="Reference-price movement that causes the bank dealer to cancel and replace its resting ladder")
    g.add_argument("--mm-n-levels", type=int,
                   default=int(calibrated_default('mm_n_levels', 10)),
                   help="Number of price levels in each bank dealer's resting ladder")
    g.add_argument("--mm-level-step-ticks", type=float,
                   default=float(calibrated_default('mm_level_step_ticks', 2.0)),
                   help="Distance between successive bank-dealer levels in exchange ticks")
    g.add_argument("--enable-cpmm", type=int, choices=[0, 1],
                   default=int(bool(calibrated_default('enable_cpmm', False))),
                   help="Put the constant product pool in the market beside "
                        "the hybrid one. Off by default. The pool is retained "
                        "as the unamplified arm of the resource matched "
                        "comparison, which is run one pool at a time, and as "
                        "the anchor for the rebalancing curvature measurement")

    g = p.add_argument_group(
        "AMM pool parameters",
        "Configure CPMM (Uniswap-like) and HFMM (Curve-like) pools.\n"
        "  CPMM: x·y = k,  fee = cpmm-fee.\n"
        "  HFMM: StableSwap invariant,  fee = hfmm-fee, amplification = A."
    )
    g.add_argument("--cpmm-reserves", type=float, default=3600.0,
                   help="CPMM base-currency reserves (default: 1000)")
    g.add_argument("--hfmm-reserves", type=float, default=3400.0,
                   help="HFMM base-currency reserves (default: 1000)")
    g.add_argument("--cpmm-fee", type=float, default=0.002,
                   help="CPMM swap fee as fraction (default: 0.003 = 30 bps)")
    g.add_argument("--hfmm-fee", type=float, default=0.0005,
                   help="HFMM swap fee as fraction (default: 0.001 = 10 bps)")
    g.add_argument("--hfmm-A", type=float, default=18.0,
                   help="HFMM amplification coefficient A (default: 10)")
    g.add_argument("--dynamic-fee", action="store_true",
                   help="Enable dynamic AMM fees that scale with volatility")
    g.add_argument("--amm-lp-wallet-cash-buffer", type=float, default=0.20,
                   dest="amm_lp_wallet_cash_buffer_ratio",
                   help="LP external quote-wallet buffer as a fraction of deployed AMM quote inventory (default: 0.20)")
    g.add_argument("--amm-lp-wallet-base-buffer", type=float, default=0.20,
                   dest="amm_lp_wallet_base_buffer_ratio",
                   help="LP external base-wallet buffer as a fraction of deployed AMM base inventory (default: 0.20)")
    g.add_argument("--price-vol-scale", type=float, default=0.0012859,
                   dest="price_vol_scale",
                   help="Volatility of the latent price per tick, as a multiple of "
                        "the stress index sigma. The default maps one tick to one "
                        "second of EUR/USD at six per cent a year (default: 0.0012859)")
    g.add_argument("--amm-lp-model", choices=["rule", "endogenous"],
                   default=calibrated_default('amm_lp_model', 'endogenous'),
                   dest="amm_lp_model",
                   help="Who supplies the automated venue. 'rule' adjusts reserves "
                        "by a fixed rule and can never leave. 'endogenous' compares "
                        "its own result against an outside option, exits when the "
                        "pool stops paying and re-enters when it starts again "
                        "(default: endogenous)")
    g.add_argument("--amm-lp-n-providers", type=int,
                   default=int(calibrated_default('amm_lp_n_providers', 5)),
                   dest="amm_lp_n_providers",
                   help="Incumbent providers per pool under the endogenous model (default: 5)")
    g.add_argument("--amm-lp-n-entrants", type=int,
                   default=int(calibrated_default('amm_lp_n_entrants', 5)),
                   dest="amm_lp_n_entrants",
                   help="Potential entrants per pool under the endogenous model (default: 5)")
    g.add_argument("--amm-lp-outside-option", type=float,
                   default=float(calibrated_default('amm_lp_outside_option', 1.3319e-9)),
                   dest="amm_lp_outside_option",
                   help="Return per period a provider can earn elsewhere (default: 0.0)")
    g.add_argument("--amm-lp-option-dispersion", type=float,
                   default=float(calibrated_default('amm_lp_option_dispersion', 0.6)),
                   dest="amm_lp_option_dispersion",
                   help="Spread of the outside option across providers (default: 0.6)")
    g.add_argument("--amm-lp-kappa", type=float,
                   default=float(calibrated_default('amm_lp_kappa', 0.35)),
                   dest="amm_lp_kappa",
                   help="How strongly a provider responds to the gap against its "
                        "outside option (default: 0.35)")
    g.add_argument("--amm-lp-response-scale", type=float,
                   default=float(calibrated_default(
                       'amm_lp_response_scale', 1e-6)),
                   dest="amm_lp_response_scale",
                   help="Per-period return scale used to normalise the LP response; "
                        "a behavioural design parameter (default: 1e-6)")
    g.add_argument("--amm-lp-max-adjustment", type=float,
                   default=float(calibrated_default('amm_lp_max_adj', 0.0023873085271651773)),
                   dest="amm_lp_max_adj",
                   help="Maximum fraction of one provider's position changed in "
                        "one period after warm-up (default: 0.0023873)")
    g.add_argument("--amm-lp-ewma-alpha", type=float,
                   default=float(calibrated_default('amm_lp_ewma_alpha', 0.02)),
                   dest="amm_lp_ewma_alpha",
                   help="EWMA weight on the latest per-period provider return; "
                        "also determines a three-horizon warm-up (default: 0.02)")
    g.add_argument("--amm-lp-exit-patience", type=int,
                   default=int(calibrated_default('amm_lp_exit_patience', 290)),
                   dest="amm_lp_exit_patience",
                   help="Periods below the outside option before a provider leaves (default: 290)")
    g.add_argument("--amm-lp-entry-patience", type=int,
                   default=int(calibrated_default('amm_lp_entry_patience', 290)),
                   dest="amm_lp_entry_patience",
                   help="Periods above the outside option before a provider returns (default: 290)")
    g.add_argument("--amm-lp-entry-margin", type=float,
                   default=float(calibrated_default('amm_lp_entry_margin', 0.25)),
                   dest="amm_lp_entry_margin",
                   help="Required excess-return margin for entry, relative to the "
                        "participation signal scale (default: 0.25)")
    g.add_argument("--amm-lp-wallet-ratio", type=float,
                   default=float(calibrated_default('amm_lp_wallet_ratio', 0.20)),
                   dest="amm_lp_wallet_ratio",
                   help="Native base and quote buffers held outside the pool, each "
                        "as a fraction of the corresponding committed reserve "
                        "(default: 0.20)")
    g.add_argument("--amm-lp-withdraw-skew", type=float,
                   default=float(calibrated_default('amm_lp_withdraw_skew', 0.0)),
                   dest="amm_lp_withdraw_skew",
                   help="Tilt of a redemption toward the scarcer reserve, 0 is pro "
                        "rata and 1 is fully tilted (default: 0.0)")
    g.add_argument("--amm-lp-subsidy-rate", type=float,
                   default=float(calibrated_default('amm_lp_subsidy_rate', 0.0)),
                   dest="amm_lp_subsidy_rate",
                   help="Per period payment from a sponsor to the providers, as a "
                        "fraction of pool value (default: 0.0)")
    g.add_argument("--amm-lp-loss-rebate", type=float,
                   default=float(calibrated_default(
                       'amm_lp_loss_rebate_fraction', 0.0)),
                   dest="amm_lp_loss_rebate_fraction",
                   help="Fraction of a realised negative LP operating payoff "
                        "reimbursed by the sponsor, bounded to [0, 1] "
                        "(default: 0.0)")
    g.add_argument("--amm-arb-cash-buffer", type=float, default=0.15,
                   dest="amm_arb_cash_buffer_ratio",
                   help="Arbitrageur quote-wallet buffer as a fraction of aggregate AMM quote reserves (default: 0.15)")
    g.add_argument("--amm-arb-base-buffer", type=float, default=0.15,
                   dest="amm_arb_base_buffer_ratio",
                   help="Arbitrageur base inventory buffer as a fraction of aggregate AMM base reserves (default: 0.15)")

    g = p.add_argument_group(
        "Stress regime",
        "Gradual volatility / funding-cost ramp over [stress-start, stress-end].\n"
        "  σ: sigma-low → sigma-high\n"
        "  c: c-low     → c-high\n"
        "Set --stress-start -1 to disable.\n\n"
        "NOTE: a shock already includes an endogenous short-run liquidity\n"
        "      aftermath. Use --shock-regime-stress only if you also want\n"
        "      a slower exogenous regime-stress window."
    )
    g.add_argument("--stress-start", type=int, default=-1,
                   help="Iteration when stress begins (default: -1 = off)")
    g.add_argument("--stress-end", type=int, default=-1,
                   help="Iteration when stress ends (default: -1 = off)")
    g.add_argument("--sigma-low", type=float, default=0.01,
                   help="Volatility in normal regime (default: 0.01)")
    g.add_argument("--sigma-high", type=float, default=0.05,
                   help="Volatility in stress regime (default: 0.05)")
    g.add_argument("--c-low", type=float, default=0.002,
                   help="Funding cost in normal regime (default: 0.002)")
    g.add_argument("--c-high", type=float, default=0.020,
                   help="Funding cost in stress regime (default: 0.020)")

    g = p.add_argument_group(
        "Exogenous price shock",
        "Shock entry point shared by two implementations.\n"
        "  research: unified shock_pct with synchronous cross-venue reset.\n"
        "  realism:  decomposed event shock with separate fair-value, flow,\n"
        "            liquidity, and funding/volatility components.\n"
        "  Optional: --shock-regime-stress adds a slower regime-stress layer."
    )
    g.add_argument("--shock-iter", type=int, default=None,
                   help="Iteration of exogenous price shock (default: off)")
    g.add_argument("--shock-mode", choices=["research", "realism"], default="realism",
                   help="Shock implementation: research keeps unified shock_pct; realism uses decomposed event shocks")
    g.add_argument("--shock-pct", type=float, default=-20.0,
                   help="Shock magnitude in %% (default: -20)")
    g.add_argument("--shock-regime-stress", action="store_true",
                   help="Couple the shock to an extended exogenous regime-stress window")
    g.add_argument("--no-shock-stress", action="store_true",
                   help="Deprecated alias. Pure shock is now the default.")

    g = p.add_argument_group(
        "Realism shock components",
        "Used only when --shock-mode realism.\n"
        "  fundamental: latent fair value jump\n"
        "  order-flow:  large CLOB sweep market order\n"
        "  liquidity:   cancellations + MM withdrawal + slower quote replenishment\n"
        "  funding-vol: σ and c spike with gradual decay"
    )
    g.add_argument("--fundamental-shock-pct", type=float, default=0.0,
                   help="Latent fair-value jump in %% for realism mode")
    g.add_argument("--order-flow-shock-qty", type=float, default=0.0,
                   help="Sweep market-order size on the CLOB in base units for realism mode")
    g.add_argument("--order-flow-shock-side", choices=["auto", "buy", "sell"], default="auto",
                   help="Direction of the realism-mode sweep order (default: auto from shock sign)")
    g.add_argument("--liquidity-shock-frac", type=float, default=0.0,
                   help="Fraction of resting orders cancelled in realism mode (0-1)")
    g.add_argument("--force-mm-pause", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Explicitly impose a dealer quote pause during the liquidity shock; off by default so dealer exit remains endogenous")
    g.add_argument("--forced-pause-ticks", type=int, default=None,
                   help="Length of an explicitly forced dealer pause; default derives from shock intensity")
    g.add_argument("--funding-vol-shock-intensity", type=float, default=0.0,
                   help="Intensity of sigma/funding jump in realism mode (1.0 ~= normal stress jump)")
    g.add_argument("--arb-max-correction-pct", type=float, default=10.0,
                   help="Max AMM price correction per arbitrage step in %% (default: 10)")
    g.add_argument("--arb-trade-fraction-cap", type=float, default=0.45,
                   help="Max fraction of AMM base reserves the arbitrageur can trade per step (default: 0.20)")

    _apply_primary_model_parser_defaults(p)
    _sync_parser_help_defaults(p)
    return p


def build_sim(args: argparse.Namespace) -> Simulator:
    stress_start = args.stress_start if args.stress_start >= 0 else None
    stress_end = args.stress_end if stress_start is not None else None

    return Simulator.default_fx(
        n_noise=args.n_noise,
        n_mm=args.n_mm if args.enable_clob_mm else 0,
        n_fast_lp=args.n_fast_lp,
        n_clob_fund=args.n_clob_fund,
        clob_fund_observation_noise=args.clob_fund_observation_noise,
        clob_fund_quote_offset=args.clob_fund_quote_offset,
        n_fx_takers=args.n_fx_takers,
        n_fx_fund=args.n_fx_fund,
        n_retail=args.n_retail,
        n_institutional=args.n_institutional,
        clob_std=args.clob_std,
        clob_volume=args.clob_volume,
        price_tick=args.price_tick,
        clob_liq=args.clob_liq,
        clob_anchor_strength=args.clob_anchor_strength,
        clob_anchor_threshold_bps=args.clob_anchor_threshold_bps,
        clob_background_target_ratio=args.clob_near_mid_target_ratio,
        clob_support_max_share=args.clob_support_max_share,
        clob_amm_interaction=args.clob_amm_interaction,
        clob_amm_spread_impact_bps=args.clob_amm_spread_impact_bps,
        clob_amm_depth_impact=args.clob_amm_depth_impact,
        mm_alpha0_base=args.mm_alpha0_base,
        mm_alpha0_step=args.mm_alpha0_step,
        mm_alpha1=args.mm_alpha1,
        mm_alpha2=args.mm_alpha2,
        mm_alpha3=args.mm_alpha3,
        mm_d0_base=args.mm_d0_base,
        mm_d0_step=args.mm_d0_step,
        mm_d1=args.mm_d1,
        mm_d2=args.mm_d2,
        mm_d3=args.mm_d3,
        hedger_flow_persistence=args.hedger_flow_persistence,
        retail_flow_persistence=args.retail_flow_persistence,
        institutional_flow_persistence=args.institutional_flow_persistence,
        common_flow_response=args.common_flow_response,
        fx_flow_intensity_scale=args.fx_flow_intensity_scale,
        mm_withdraw_threshold=args.mm_withdraw_threshold,
        mm_withdraw_threshold_step=args.mm_withdraw_threshold_step,
        mm_reentry_threshold=args.mm_reentry_threshold,
        mm_loss_threshold_bps=args.mm_loss_threshold_bps,
        mm_quote_life=args.mm_quote_life,
        mm_quote_refresh_tol_bps=args.mm_quote_refresh_tol_bps,
        mm_n_levels=args.mm_n_levels,
        mm_level_step_ticks=args.mm_level_step_ticks,
        mm_inv_skew_bps=args.mm_inv_skew_bps,
        mm_replacement_gain=args.mm_replacement_gain,
        mm_revenue_horizon=args.mm_revenue_horizon,
        mm_stale_touch_ratio=args.mm_stale_touch_ratio,
        fast_lp_base_spread_bps=args.fast_lp_base_spread_bps,
        fast_lp_quote_life=args.fast_lp_quote_life,
        fast_lp_base_qty=args.fast_lp_base_qty,
        fast_lp_levels=args.fast_lp_levels,
        fast_lp_base_withdraw_prob=args.fast_lp_base_withdraw_prob,
        fast_lp_stress_abstention=args.fast_lp_stress_abstention,
        fast_lp_vol_multiple=args.fast_lp_vol_multiple,
        mm_softlimit=args.mm_softlimit,
        mm_client_flow_intensity=args.mm_client_flow_intensity,
        mm_core_threshold=args.mm_core_threshold,
        facility_arm=args.facility_arm,
        arm_capital=args.arm_capital,
        arm_spread_bps=args.arm_spread_bps,
        mm_client_flow_persistence=args.mm_client_flow_persistence,
        dealer_cascade_gain=args.dealer_cascade_gain,
        dealer_capacity_threshold=args.dealer_capacity_threshold,
        mm_min_withdraw_ticks=args.mm_min_withdraw_ticks,
        mm_reentry_ticks=args.mm_reentry_ticks,
        mm_withdraw_confirmation_ticks=args.mm_withdraw_confirmation_ticks,
        maintenance_margin_ratio=args.maintenance_margin_ratio,
        liquidation_fraction=args.liquidation_fraction,
        borrow_spread_multiplier=args.borrow_spread_multiplier,
        short_borrow_spread_multiplier=args.short_borrow_spread_multiplier,
        enable_amm=bool(args.enable_amm),
        enable_cpmm=bool(args.enable_cpmm),
        amm_liq=args.amm_liq,
        match_initial_depth=args.match_initial_depth,
        cpmm_reserves=args.cpmm_reserves,
        hfmm_reserves=args.hfmm_reserves,
        cpmm_fee=args.cpmm_fee,
        hfmm_fee=args.hfmm_fee,
        hfmm_A=args.hfmm_A,
        amm_share_pct=args.amm_share_pct,
        venue_choice_rule=args.venue_choice_rule,
        deterministic=args.deterministic,
        beta_amm=args.beta_amm,
        cpmm_bias_bps=args.cpmm_bias_bps,
        cost_noise_std=args.cost_noise_std,
        routing_cost_scale_bps=args.routing_cost_scale_bps,
        routing_prior_mix_cap=args.routing_prior_mix_cap,
        routing_basis_scale_bps=args.routing_basis_scale_bps,
        routing_clob_depth_multiple=args.routing_clob_depth_multiple,
        routing_amm_depth_multiple=args.routing_amm_depth_multiple,
        price=args.price,
        stress_start=stress_start,
        stress_end=stress_end,
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
        c_low=args.c_low,
        c_high=args.c_high,
        shock_iter=args.shock_iter,
        shock_pct=args.shock_pct,
        shock_mode=args.shock_mode,
        fundamental_shock_pct=args.fundamental_shock_pct,
        order_flow_shock_qty=args.order_flow_shock_qty,
        order_flow_shock_side=args.order_flow_shock_side,
        liquidity_shock_frac=args.liquidity_shock_frac,
        force_mm_pause=args.force_mm_pause,
        forced_pause_ticks=args.forced_pause_ticks,
        funding_vol_shock_intensity=args.funding_vol_shock_intensity,
        arb_max_correction_pct=args.arb_max_correction_pct,
        arb_trade_fraction_cap=args.arb_trade_fraction_cap,
        amm_lp_wallet_cash_buffer_ratio=args.amm_lp_wallet_cash_buffer_ratio,
        amm_lp_wallet_base_buffer_ratio=args.amm_lp_wallet_base_buffer_ratio,
        price_vol_scale=getattr(args, 'price_vol_scale', 0.0012859),
        amm_lp_model=getattr(args, 'amm_lp_model', 'endogenous'),
        amm_lp_n_providers=getattr(args, 'amm_lp_n_providers', 5),
        amm_lp_n_entrants=getattr(args, 'amm_lp_n_entrants', 5),
        amm_lp_outside_option=getattr(args, 'amm_lp_outside_option', 0.0),
        amm_lp_option_dispersion=getattr(args, 'amm_lp_option_dispersion', 0.6),
        amm_lp_kappa=getattr(args, 'amm_lp_kappa', 0.35),
        amm_lp_response_scale=getattr(args, 'amm_lp_response_scale', 1e-6),
        amm_lp_max_adj=getattr(args, 'amm_lp_max_adj', 0.0023873085271651773),
        amm_lp_ewma_alpha=getattr(args, 'amm_lp_ewma_alpha', 0.02),
        amm_lp_exit_patience=getattr(args, 'amm_lp_exit_patience', 290),
        amm_lp_entry_patience=getattr(args, 'amm_lp_entry_patience', 290),
        amm_lp_entry_margin=getattr(args, 'amm_lp_entry_margin', 0.25),
        amm_lp_wallet_ratio=getattr(args, 'amm_lp_wallet_ratio', 0.20),
        amm_lp_withdraw_skew=getattr(args, 'amm_lp_withdraw_skew', 0.0),
        amm_lp_subsidy_rate=getattr(args, 'amm_lp_subsidy_rate', 0.0),
        amm_lp_loss_rebate_fraction=getattr(
            args, 'amm_lp_loss_rebate_fraction', 0.0),
        amm_arb_cash_buffer_ratio=args.amm_arb_cash_buffer_ratio,
        amm_arb_base_buffer_ratio=args.amm_arb_base_buffer_ratio,
        dynamic_fee=args.dynamic_fee,
        reprice_prob_recovery=getattr(args, 'reprice_prob_recovery', None),
        anchor_strength_recovery=getattr(args, 'anchor_strength_recovery', None),
        bg_target_ratio_recovery=getattr(args, 'bg_target_ratio_recovery', None),
        toxic_flow_decay=getattr(args, 'toxic_flow_decay', None),
        liquidity_shock_decay=getattr(args, 'liquidity_shock_decay', None),
        stress_overlay_decay=getattr(args, 'stress_overlay_decay', None),
    )


def print_config(args: argparse.Namespace):
    W = 65
    print("\n" + "=" * W)
    print("  FX ABM — SIMULATION CONFIGURATION")
    print("=" * W)

    def row(label, value, unit=""):
        print(f"    {label:<32s} {str(value):>12s} {unit}")

    print("\n  General")
    row("Iterations", str(args.n_iter))
    row("Initial price", f"{args.price:.1f}")
    row("Random seed", str(args.seed) if args.seed is not None else "random")
    row("Robustness check", "ON" if args.robustness_check else "OFF")
    if args.robustness_check:
        row("Robustness seeds", str(args.robustness_seeds))
        row("Robustness base seed",
            str(args.robustness_base_seed) if args.robustness_base_seed is not None else "auto")
    if args.preset:
        row("Preset applied", args.preset)
        row("Preset", "declared episode" if args.preset in PRESETS
             else "not a declared episode")
    row("Primary config", PRIMARY_MODEL_PATH.name if PRIMARY_MODEL_PATH.exists() else "none")
    row("Run label", getattr(args, 'run_label', 'primary'))

    eff_mm = args.n_mm if args.enable_clob_mm else 0
    shadow = bool(args.enable_amm) and eff_mm == 0
    print("\n  CLOB agents")
    row("Noise traders", str(args.n_noise))
    row("Market Makers", str(eff_mm))
    row("FastRecyclerLP", str(args.n_fast_lp))
    row("Fundamentalists (book)", str(args.n_clob_fund))
    row("CLOB mode", "Shadow (synthetic)" if shadow else "Live order-book")
    row("Order-book volume", str(args.clob_volume))
    row("Price std", f"{args.clob_std:.1f}")
    row("CLOB liquidity multiplier", f"{args.clob_liq:.1f}", "×")
    row("CLOB anchor strength", f"{args.clob_anchor_strength:.2f}")
    row("CLOB anchor threshold", f"{args.clob_anchor_threshold_bps:.1f}", "bps")
    row("Near-mid target", f"{args.clob_near_mid_target_ratio:.1f}", "×")
    row("Support max share", f"{args.clob_support_max_share:.2f}", "×")
    row("AMM interaction", args.clob_amm_interaction)
    row("AMM spread impact", f"{args.clob_amm_spread_impact_bps:.1f}", "bps")
    row("AMM depth impact", f"{args.clob_amm_depth_impact:.1f}")
    row("MM alpha0 base", f"{args.mm_alpha0_base:.2f}", "bps")
    row("MM alpha1 / alpha2", f"{args.mm_alpha1:.0f}/{args.mm_alpha2:.0f}")
    row("MM alpha3", f"{args.mm_alpha3:.0f}")
    row("MM d0 base", f"{args.mm_d0_base:.1f}")
    row("MM d1 / d2 / d3", f"{args.mm_d1:.0f}/{args.mm_d2:.0f}/{args.mm_d3:.0f}")

    print("\n  Balance sheets / solvency")
    row("Maint. margin ratio", f"{args.maintenance_margin_ratio:.3f}")
    row("Liquidation fraction", f"{args.liquidation_fraction:.2f}")
    row("Borrow carry multiplier", f"{args.borrow_spread_multiplier:.2f}")
    row("Short carry multiplier", f"{args.short_borrow_spread_multiplier:.2f}")

    print("\n  FX liquidity takers")
    row("Noise takers", str(args.n_fx_takers))
    row("Fundamentalists", str(args.n_fx_fund))
    row("Retail", str(args.n_retail))
    row("Institutional", str(args.n_institutional))
    total = args.n_fx_takers + args.n_fx_fund + args.n_retail + args.n_institutional
    row("Total takers", str(total))
    row("Hedger persistence", f"{args.hedger_flow_persistence:.2f}")
    row("Retail persistence", f"{args.retail_flow_persistence:.2f}")
    row("Institutional persistence", f"{args.institutional_flow_persistence:.2f}")

    print("\n  Flow allocation")
    row("AMM routing prior", f"{args.amm_share_pct:.0f}", "%")
    route_mode = "argmin" if args.deterministic else args.venue_choice_rule
    row("Venue choice mode", route_mode)
    row("beta_AMM (intra-AMM)", f"{args.beta_amm:.3f}")
    row("CPMM bias", f"{args.cpmm_bias_bps:.1f}", "bps")
    row("Cost noise sigma", f"{args.cost_noise_std:.1f}", "bps")
    row("Routing cost scale", f"{args.routing_cost_scale_bps:.1f}", "bps")
    row("Routing prior mix cap", f"{args.routing_prior_mix_cap:.2f}")
    row("Routing basis scale", f"{args.routing_basis_scale_bps:.1f}", "bps")

    print("\n  AMM pools")
    row("AMM enabled", "YES" if args.enable_amm else "NO")
    if args.enable_amm:
        row("Initial depth match", "ON" if args.match_initial_depth else "OFF")
        row("CPMM reserves (base)", f"{args.cpmm_reserves:.0f}")
        row("HFMM reserves (base)", f"{args.hfmm_reserves:.0f}")
        row("CPMM fee", f"{args.cpmm_fee * 10_000:.0f}", "bps")
        row("HFMM fee", f"{args.hfmm_fee * 10_000:.0f}", "bps")
        row("HFMM amplification A", f"{args.hfmm_A:.0f}")
        row("Dynamic fees", "ON" if args.dynamic_fee else "OFF")
        row("AMM liquidity multiplier", f"{args.amm_liq:.1f}", "×")
        row("LP wallet cash buffer", f"{args.amm_lp_wallet_cash_buffer_ratio:.2f}", "×")
        row("LP wallet base buffer", f"{args.amm_lp_wallet_base_buffer_ratio:.2f}", "×")
        row("Arb wallet cash buffer", f"{args.amm_arb_cash_buffer_ratio:.2f}", "×")
        row("Arb wallet base buffer", f"{args.amm_arb_base_buffer_ratio:.2f}", "×")

    print("\n  Stress regime")
    if args.stress_start >= 0:
        auto_tag = ""
        if (args.shock_iter is not None
                and getattr(args, 'shock_regime_stress', False)
                and not getattr(args, 'no_shock_stress', False)):
            auto_tag = " (auto from shock)"
        row("Window", f"[{args.stress_start}, {args.stress_end}]{auto_tag}")
        row("Volatility σ", f"{args.sigma_low:.4f} → {args.sigma_high:.4f}")
        row("Funding cost c", f"{args.c_low:.4f} → {args.c_high:.4f}")
    else:
        row("Status", "OFF")

    print("\n  Exogenous price shock")
    if args.shock_iter is not None:
        row("Iteration", str(args.shock_iter))
        row("Shock mode", args.shock_mode)
        if args.shock_mode == 'research':
            row("Magnitude", f"{args.shock_pct:+.0f}", "%")
            row("Hits CLOB orders", "YES")
            row("Hits AMM reserves", "YES" if args.enable_amm else "N/A")
            row("Endogenous aftermath", "ON")
        else:
            row("Fundamental shock", f"{args.fundamental_shock_pct:+.1f}", "%")
            row("Order-flow shock", f"{args.order_flow_shock_qty:.0f}", "base")
            row("Order-flow side", args.order_flow_shock_side)
            row("Liquidity shock", f"{args.liquidity_shock_frac:.2f}")
            row("Funding/vol shock", f"{args.funding_vol_shock_intensity:.2f}")
            row("AMM reserve rebalance", "OFF")
            row("Arb max correction", f"{args.arb_max_correction_pct:.1f}", "%")
            row("Arb trade cap", f"{args.arb_trade_fraction_cap:.2f}")
        row("Regime stress layer", "ON"
            if (getattr(args, 'shock_regime_stress', False)
                and not getattr(args, 'no_shock_stress', False))
            else "OFF")
    else:
        row("Status", "OFF")

    print("=" * W + "\n")


def print_summary(sim: Simulator, acceptance_report: Optional[dict] = None):
    logger = sim.logger
    summary = logger.summary()

    W = 65
    print("\n" + "=" * W)
    print("  FX ABM — SIMULATION RESULTS")
    print("=" * W)
    print(f"  Iterations:   {summary.get('n_iterations', 0)}")
    print(f"  Total trades: {summary.get('n_trades', 0)}")

    print("\n" + "-" * W)
    print("  EXECUTION COST AND VENUE INTERACTION")
    print("-" * W)

    Q_values = [1, 2, 5, 10, 20, 50]
    venues = ['clob'] + list(logger.amm_cost_curves.keys())

    print("\n  Average All-in Cost (bps):")
    print(f"  {'Q':>6s}", end='')
    for v in venues:
        print(f"  {v.upper():>8s}", end='')
    print()
    for Q in Q_values:
        print(f"  {Q:>6.0f}", end='')
        for v in venues:
            val = summary.get(f'avg_cost_{v}_Q{Q}', float('nan'))
            if math.isfinite(val):
                print(f"  {val:>8.1f}", end='')
            else:
                print(f"  {'N/A':>8s}", end='')
        print()

    print("\n  Successful Customer Volume Share (ratio of sums):")
    for v in venues:
        val = summary.get(f'avg_flow_share_{v}', 0)
        print(f"    {v.upper():>6s}: {val:.1%}")
    amm_share = sum(summary.get(f'avg_flow_share_{v}', 0)
                    for v in venues if v != 'clob')
    print(f"\n  -> AMM captures {amm_share:.1%} of successful routed customer volume.")

    split_iter, split_title, phase_before, phase_after = _linkage_split_point(sim)
    print("\n" + "-" * W)
    print("  SYSTEMIC LINKAGE UNDER STRESS")
    print("-" * W)

    if not logger.amm_cost_curves:
        print("  (no AMM pools — skipped)")
        print("=" * W + "\n")
        return

    print("\n  Cost Correlation (CLOB <-> AMM, Q=5):")
    for name in logger.amm_cost_curves:
        rho = logger.cost_correlation('clob', name, Q=5)
        print(f"    CLOB <-> {name.upper()}: rho = {rho:.3f}")

    clob_depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    print("\n  Liquidity Commonality (near-mid depth / shared factor):")
    clob_factor = logger.series_correlation(clob_depth_series, logger.systemic_liquidity_series)
    print(f"    CLOB depth <-> factor: rho = {clob_factor:.3f}")
    for name in logger.amm_cost_curves:
        amm_depth = logger.amm_depth_series.get(name, [])
        depth_rho = logger.series_correlation(clob_depth_series, amm_depth)
        factor_rho = logger.series_correlation(amm_depth, logger.systemic_liquidity_series)
        print(f"    CLOB depth <-> {name.upper()} depth: rho = {depth_rho:.3f}"
              f"  |  {name.upper()} depth <-> factor: rho = {factor_rho:.3f}")

    if split_iter is not None:
        n = len(logger.iterations)
        idx = min(split_iter, n)

        if split_title == 'Shock':
            windows = _shock_window_slices(n, idx)

            print(f"\n  Cost Correlation By Shock Window (Q=5, t={split_iter}):")
            for name in logger.amm_cost_curves:
                parts = []
                clob_cost = logger.cost_series('clob', 5)
                amm_cost = logger.cost_series(name, 5)
                for label, start, end in windows:
                    rho = logger.series_correlation(clob_cost[start:end], amm_cost[start:end])
                    rho_s = f"{rho:.3f}" if math.isfinite(rho) else "N/A"
                    parts.append(f"{label}={rho_s}")
                print(f"    CLOB <-> {name.upper()}: " + "  ".join(parts))

            print("\n  CLOB Quoted Spread By Local Window:")
            for label, start, end in windows:
                avg_qspr = _window_average(logger.clob_qspr, start, end)
                avg_s = f"{avg_qspr:.1f}" if math.isfinite(avg_qspr) else "N/A"
                print(f"    {label:<10s} {avg_s:>8s} bps")

            venues_all = ['clob'] + list(logger.amm_cost_curves.keys())
            print("\n  Execution Cost By Local Window (Q=5, bps):")
            print(f"  {'Window':<12s}", end='')
            for venue in venues_all:
                print(f"  {venue.upper():>8s}", end='')
            print()
            for label, start, end in windows:
                print(f"  {label:<12s}", end='')
                for venue in venues_all:
                    avg_cost = _window_average(logger.cost_series(venue, 5), start, end)
                    avg_s = f"{avg_cost:.1f}" if math.isfinite(avg_cost) else "N/A"
                    print(f"  {avg_s:>8s}", end='')
                print()

            print("\n  AMM Customer Volume Share By Local Window (ratio of sums):")
            for name in logger.amm_cost_curves:
                parts = []
                for label, start, end in windows:
                    avg_share = logger.customer_volume_share(name, start, end)
                    share_s = f"{avg_share:.1%}" if math.isfinite(avg_share) else "N/A"
                    parts.append(f"{label}={share_s}")
                print(f"    {name.upper()}: " + "  ".join(parts))

            print("\n  Venue Basis By Local Window (abs bps):")
            max_basis = logger.max_venue_basis_series()
            parts = []
            for label, start, end in windows:
                avg_basis = _window_average(max_basis, start, end)
                basis_s = f"{avg_basis:.1f}" if math.isfinite(avg_basis) else "N/A"
                parts.append(f"{label}={basis_s}")
            print(f"    MAX |AMM - CLOB|: " + "  ".join(parts))
            for name in logger.amm_cost_curves:
                basis = [abs(x) if math.isfinite(x) else float('nan') for x in logger.venue_basis_series(name)]
                parts = []
                for label, start, end in windows:
                    avg_basis = _window_average(basis, start, end)
                    basis_s = f"{avg_basis:.1f}" if math.isfinite(avg_basis) else "N/A"
                    parts.append(f"{label}={basis_s}")
                print(f"    {name.upper():>6s}: " + "  ".join(parts))
        else:
            print(f"\n  Correlation Before / After {split_title} (t={split_iter}):")
            for name in logger.amm_cost_curves:
                ba = logger.commonality_before_after('clob', name, Q=5,
                                                     stress_start=split_iter)
                b_s = f"{ba['before']:.3f}" if math.isfinite(ba['before']) else "N/A"
                a_s = f"{ba['after']:.3f}" if math.isfinite(ba['after']) else "N/A"
                print(f"    CLOB <-> {name.upper()}: before={b_s}, after={a_s}")

            print(f"\n  Depth Correlation Before / After {split_title}:")
            for name in logger.amm_cost_curves:
                ba = logger.series_commonality_before_after(
                    clob_depth_series,
                    logger.amm_depth_series.get(name, []),
                    split_iter,
                )
                b_s = f"{ba['before']:.3f}" if math.isfinite(ba['before']) else "N/A"
                a_s = f"{ba['after']:.3f}" if math.isfinite(ba['after']) else "N/A"
                print(f"    CLOB <-> {name.upper()}: before={b_s}, after={a_s}")

            print(f"\n  AMM Customer Volume Share (ratio of sums) — "
                  f"{phase_before} vs {phase_after}:")
            for name in logger.amm_cost_curves:
                avg_b = logger.customer_volume_share(name, 0, idx)
                avg_a = logger.customer_volume_share(name, idx, len(logger.iterations))
                delta = avg_a - avg_b
                print(f"    {name.upper()}: {phase_before}={avg_b:.1%}  "
                      f"{phase_after}={avg_a:.1%}  Delta={delta:+.1%}")

            qspr = logger.clob_qspr
            normal_s = [s for s in qspr[:idx] if math.isfinite(s)]
            stress_s = [s for s in qspr[idx:] if math.isfinite(s)]
            avg_n = sum(normal_s) / len(normal_s) if normal_s else 0
            avg_st = sum(stress_s) / len(stress_s) if stress_s else 0
            print("\n  CLOB Quoted Spread:")
            print(f"    {phase_before}: {avg_n:.1f} bps")
            if avg_n > 0:
                print(f"    {phase_after}: {avg_st:.1f} bps  ({avg_st/avg_n:.1f}x wider)")

            Q_rep = [1, 5, 10, 50]
            venues_all = ['clob'] + list(logger.amm_cost_curves.keys())
            print(f"\n  Execution Cost — {phase_before} vs {phase_after} (bps):")
            print(f"  {'Q':>4s}", end='')
            for v in venues_all:
                print(f"  {v.upper()+'-N':>8s} {v.upper()+'-S':>8s} {'×':>5s}", end='')
            print()
            for Q in Q_rep:
                print(f"  {Q:>4.0f}", end='')
                for v in venues_all:
                    cs = logger.cost_series(v, Q)
                    nrm = [x for x in cs[:idx] if math.isfinite(x)]
                    strs = [x for x in cs[idx:] if math.isfinite(x)]
                    avg_nrm = sum(nrm) / len(nrm) if nrm else float('nan')
                    avg_str = sum(strs) / len(strs) if strs else float('nan')
                    ratio = avg_str / avg_nrm if avg_nrm and avg_nrm > 0 else float('nan')
                    n_s = f"{avg_nrm:.1f}" if math.isfinite(avg_nrm) else "N/A"
                    s_s = f"{avg_str:.1f}" if math.isfinite(avg_str) else "N/A"
                    r_s = f"{ratio:.1f}" if math.isfinite(ratio) else "-"
                    print(f"  {n_s:>8s} {s_s:>8s} {r_s:>5s}", end='')
                print()

    # ---- Recovery time after shock ------------------------------------
    shock_iter = _shock_iter_from_sim(sim)

    if shock_iter is not None:
        print("\n" + "-" * W)
        print("  POST SHOCK RECOVERY TIME")
        print("-" * W)

        trades = pd.DataFrame(logger.trade_log) if logger.trade_log else pd.DataFrame()

        WINDOW = 5
        MAX_WARMUP = 50
        HORIZON = 100

        baseline_end = shock_iter
        if (sim.env is not None
                and getattr(sim.env, 'stress_start', None) is not None):
            baseline_end = min(baseline_end, sim.env.stress_start)
        warmup = min(MAX_WARMUP, max(0, baseline_end - 30))

        def _baseline(series):
            clean = [x for x in series[warmup:baseline_end] if np.isfinite(x)]
            if len(clean) < WINDOW:
                clean = [x for x in series[:baseline_end] if np.isfinite(x)]
            if len(clean) < WINDOW:
                return float('nan')
            return float(np.median(clean))

        def _rolling_recovery(series, *, direction: str,
                              rel_tol: float, abs_tol: float = 0.0,
                              window: int = WINDOW):
            baseline = _baseline(series)
            if not np.isfinite(baseline):
                return float('nan'), float('nan'), float('nan')

            values = pd.Series(
                [x if np.isfinite(x) else np.nan for x in series],
                dtype='float64',
            )
            post_values = values.iloc[shock_iter:].reset_index(drop=True)
            post = post_values.rolling(window, min_periods=window).median()
            if len(post) == 0:
                return float('nan'), baseline, float('nan')

            if direction == 'upper':
                target = max(baseline * (1.0 + rel_tol), baseline + abs_tol)
                recovered = post <= target
            else:
                target = baseline * (1.0 - rel_tol)
                recovered = post >= target

            hits = recovered[recovered].index.tolist()
            if not hits:
                return float('inf'), baseline, target
            return max(0, hits[0]), baseline, target

        def _trade_cost_series(venue: str = None, fallback=None):
            if fallback is None:
                fallback = []
            if trades.empty or 'cost_bps' not in trades.columns:
                return fallback

            subset = trades
            if venue is not None:
                subset = trades[trades['venue'] == venue]
            if subset.empty:
                return fallback

            med = subset.groupby('t')['cost_bps'].median()
            out = [float('nan')] * len(logger.iterations)
            for t, value in med.items():
                idx = int(t)
                if 0 <= idx < len(out):
                    out[idx] = float(value)

            finite = sum(np.isfinite(value) for value in out)
            return out if finite >= WINDOW else fallback

        def _avg_amm_cost_series(Q: float = 5):
            out = []
            for i in range(len(logger.iterations)):
                vals = []
                for name in logger.amm_cost_curves:
                    series = logger.amm_cost_curves[name].get(Q, [])
                    if i < len(series) and np.isfinite(series[i]):
                        vals.append(series[i])
                out.append(sum(vals) / len(vals) if vals else float('nan'))
            return out

        clob_depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
        clob_cost_series = _trade_cost_series(
            'clob',
            fallback=logger.cost_series('clob', 5),
        )
        max_basis_series = logger.max_venue_basis_series()
        metrics = [
            ('CLOB trade cost', clob_cost_series, 'upper', 0.50, 10.0),
            ('CLOB quoted spread', logger.clob_qspr, 'upper', 0.50, 10.0),
            ('CLOB near-mid depth', clob_depth_series, 'lower', 0.25, 0.0),
            ('Systemic liquidity', logger.systemic_liquidity_series, 'lower', 0.15, 0.0),
            ('Venue basis |AMM-CLOB|', max_basis_series, 'upper', 0.25, 5.0),
        ]

        if logger.amm_cost_curves:
            amm_depth_series = [
                sum(logger.amm_depth_series[name][i] for name in logger.amm_depth_series)
                for i in range(len(logger.iterations))
            ]
            metrics.extend([
                ('AMM quote cost (Q=5)', _avg_amm_cost_series(5), 'upper', 0.50, 10.0),
                ('AMM effective depth', amm_depth_series, 'lower', 0.25, 0.0),
            ])

        def _fmt(v):
            if v != v:
                return 'N/A'
            if v == float('inf'):
                return 'never'
            return f'{v:.0f}'

        baseline_mid = _window_average(logger.clob_mid_series, warmup, baseline_end)
        price_drawdown = []
        for mid in logger.clob_mid_series:
            if math.isfinite(mid) and math.isfinite(baseline_mid) and baseline_mid > 0:
                price_drawdown.append(10_000.0 * (mid - baseline_mid) / baseline_mid)
            else:
                price_drawdown.append(float('nan'))
        trough_bps, trough_time = _series_trough(
            price_drawdown,
            shock_iter=shock_iter,
            direction='lower',
            horizon=HORIZON,
        )

        print(f"\n  Event windows: [t0,t0+5), [t0+5,t0+20), [t0+20,t0+100)")
        print("  Recovery is evaluated locally around the event, not on the full post-shock tail.")
        print("\n  Impact trough:")
        trough_s = f"{trough_bps:.1f}" if math.isfinite(trough_bps) else "N/A"
        t_s = f"+{trough_time:.0f}" if math.isfinite(trough_time) else "N/A"
        print(f"    CLOB mid drawdown: {trough_s} bps at {t_s}")
        basis_trough, basis_time = _series_trough(
            max_basis_series,
            shock_iter=shock_iter,
            direction='upper',
            horizon=HORIZON,
        )
        basis_s = f"{basis_trough:.1f}" if math.isfinite(basis_trough) else "N/A"
        basis_t = f"+{basis_time:.0f}" if math.isfinite(basis_time) else "N/A"
        print(f"    Max venue basis:   {basis_s} bps at {basis_t}")

        print("\n  Replenishment speed (per tick, t0+5 -> t0+20):")
        clob_repl = _replenishment_speed(clob_depth_series, shock_iter=shock_iter, direction='lower')
        sysliq_repl = _replenishment_speed(logger.systemic_liquidity_series, shock_iter=shock_iter, direction='lower')
        basis_repl = _replenishment_speed(max_basis_series, shock_iter=shock_iter, direction='upper')
        print(f"    CLOB depth:        {clob_repl:.2f}" if math.isfinite(clob_repl) else "    CLOB depth:        N/A")
        print(f"    Systemic liquidity:{sysliq_repl:.4f}" if math.isfinite(sysliq_repl) else "    Systemic liquidity:N/A")
        print(f"    Basis compression: {basis_repl:.2f} bps" if math.isfinite(basis_repl) else "    Basis compression: N/A")

        def _bl_fmt(v):
            return f'{v:.1f}' if np.isfinite(v) else 'N/A'

        def _target_fmt(v):
            return f'{v:.1f}' if np.isfinite(v) else 'N/A'

        print(f"\n  {'Metric':<24s}  {'Baseline':>10s}  {'Target':>10s}  {'Normalize':>10s}")
        print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*10}")

        recovery_times = []
        for label, series, direction, rel_tol, abs_tol in metrics:
            baseline = _baseline(series)
            rt, target = _rolling_normalization_time(
                series,
                shock_iter=shock_iter,
                baseline=baseline,
                direction=direction,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                window=WINDOW,
                horizon=HORIZON,
            )
            if np.isfinite(rt) or rt == float('inf'):
                recovery_times.append(rt)
            print(f"  {label:<24s}  {_bl_fmt(baseline):>10s}  "
                  f"{_target_fmt(target):>10s}  {_fmt(rt):>10s}")

        if recovery_times:
            system_recovery = float('inf') if any(rt == float('inf') for rt in recovery_times) else max(recovery_times)
            print(f"\n  {'System-wide max':<24s}  {'—':>10s}  {'—':>10s}  {_fmt(system_recovery):>10s}")

    if acceptance_report and acceptance_report.get('summary'):
        acc = acceptance_report['summary']

        def _fmt_acceptance_value(value):
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return f"{float(value):.3g}"
            return "N/A"

        print("\n" + "-" * W)
        print("  PRIMARY MODEL ACCEPTANCE")
        print("-" * W)
        # A run that measured no target has produced no evidence, which is the
        # ordinary case for a short exploratory one, and calling that a failed
        # panel reads as a verdict the run never reached. The verdict against
        # the frozen matrix belongs to ``accept``, on its own seed commitment.
        evaluated = int(acc.get('evaluated_targets', 0) or 0)
        if evaluated == 0:
            print("  Status:       not evaluated on this run")
            print(f"  Measurable:   0 of {len(acceptance_report.get('targets', []))}"
                  " targets; the verdict is the business of `accept`")
        else:
            print(f"  Status:       {acc.get('status', 'fail').upper()}")
            print(f"  Passed:       {acc.get('passed_targets', 0)}/{evaluated}")
            print("  Objective:    "
                  f"{_fmt_acceptance_value(acc.get('objective_score', float('nan')))}")
        for item in acceptance_report.get('targets', []):
            if item.get('status') == 'not_evaluable':
                continue
            print(
                f"    {item['observable']:<32s} {item['status'].upper():>4s}  "
                f"realized={_fmt_acceptance_value(item.get('realized_value'))}  "
                f"target={_fmt_acceptance_value(item.get('target'))}"
            )

    print("=" * W + "\n")


def generate_all_plots(sim: Simulator,
                       out_dir: str = 'output/main_aware'):
    import glob as _glob
    logger = sim.logger
    stress_start = sim.env.stress_start if sim.env else None
    shock_iter = getattr(sim, 'shock_iter', None)

    # Remove stale plots from previous runs.
    for old_png in _glob.glob(os.path.join(out_dir, '*.png')):
        os.remove(old_png)

    # Save dashboards + all individual plots to the selected output directory.
    generate_all_dashboards(
        logger,
        out_dir=out_dir,
        stress_start=stress_start,
        Q=5,
        rolling=10,
        shock_iter=shock_iter,
    )
    save_all_individual_plots(
        logger,
        out_dir=out_dir,
        stress_start=stress_start,
        Q=5,
        rolling=10,
        shock_iter=shock_iter,
    )


def _spillover_safe_series(values):
    out = []
    for value in values:
        out.append(float(value) if (value is not None and math.isfinite(value)) else float('nan'))
    return out


def _spillover_rolling_corr(x, y, window: int):
    sx = pd.Series(x, dtype='float64')
    sy = pd.Series(y, dtype='float64')
    return sx.rolling(window=window, min_periods=window).corr(sy).tolist()


def _spillover_standardized_beta(y, x, beta: float, lag: int):
    """Standardized lagged-beta: beta * std(x_{t-lag}) / std(y_t)."""
    if not math.isfinite(beta):
        return float('nan')
    lag = max(1, int(lag))
    if len(y) <= lag or len(x) <= lag:
        return float('nan')

    ys = y[lag:]
    xs = x[:-lag]
    pairs = [(xi, yi) for xi, yi in zip(xs, ys) if math.isfinite(xi) and math.isfinite(yi)]
    if len(pairs) < 3:
        return float('nan')

    xs_f, ys_f = zip(*pairs)
    sx = float(np.std(xs_f))
    sy = float(np.std(ys_f))
    if not (math.isfinite(sx) and math.isfinite(sy) and sy > 0):
        return float('nan')
    return float(beta) * (sx / sy)


def _spillover_window_mean(series, start: int, end: int) -> float:
    lo = max(0, int(start))
    hi = max(lo, min(len(series), int(end)))
    finite = [float(value) for value in series[lo:hi] if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float('nan')


def _spillover_phase_windows(sim: Simulator, n_obs: int) -> list[dict]:
    event_iter = _shock_iter_from_sim(sim)
    env = getattr(sim, 'env', None)
    stress_end = getattr(env, 'stress_end', None) if env is not None else None

    windows = [
        {'phase': 'full', 'label': 'Full sample', 'start': 0, 'end': n_obs},
    ]
    if event_iter is None:
        return windows

    event_iter = max(0, min(int(event_iter), n_obs))
    before_start = max(0, event_iter - 50)
    if stress_end is not None and stress_end > event_iter:
        during_end = max(event_iter, min(n_obs, int(stress_end)))
        after_end = min(n_obs, during_end + 100)
    else:
        during_end = min(n_obs, event_iter + 20)
        after_end = min(n_obs, event_iter + 100)

    windows.extend([
        {'phase': 'before', 'label': 'Before', 'start': before_start, 'end': event_iter},
        {'phase': 'during', 'label': 'During', 'start': event_iter, 'end': during_end},
        {'phase': 'after', 'label': 'After', 'start': during_end, 'end': after_end},
    ])
    return [window for window in windows if window['end'] > window['start']]


def _spillover_flow_share_series(logger) -> tuple[list[float], list[float]]:
    n_obs = len(logger.iterations)
    if not getattr(logger, 'flow_volume', None):
        return [float('nan')] * n_obs, [float('nan')] * n_obs
    return (
        logger.active_tick_customer_volume_share_series('clob'),
        logger.amm_active_tick_customer_volume_share_series(),
    )


def _spillover_slice_metrics(logger, d_clob, d_amm, lag: int, start: int, end: int) -> dict:
    diff_start = max(0, int(start))
    diff_end = max(diff_start, min(len(d_clob), max(int(start), int(end) - 1)))
    slice_clob = d_clob[diff_start:diff_end]
    slice_amm = d_amm[diff_start:diff_end]
    corr = logger.series_correlation(slice_clob, slice_amm)
    amm_to_clob = logger._lag_regression(slice_clob, slice_amm, lag=lag)
    clob_to_amm = logger._lag_regression(slice_amm, slice_clob, lag=lag)
    return {
        'corr_dliq_clob_amm': corr,
        'amm_to_clob_beta_std': _spillover_standardized_beta(
            slice_clob,
            slice_amm,
            amm_to_clob.get('beta', float('nan')),
            lag,
        ),
        'clob_to_amm_beta_std': _spillover_standardized_beta(
            slice_amm,
            slice_clob,
            clob_to_amm.get('beta', float('nan')),
            lag,
        ),
        'amm_to_clob_beta_raw': amm_to_clob.get('beta', float('nan')),
        'clob_to_amm_beta_raw': clob_to_amm.get('beta', float('nan')),
    }


def save_spillover_artifacts(sim: Simulator,
                             out_dir: str,
                             lag: int = 1,
                             rolling_window: int = 30,
                             prefix: str = 'spillover_main'):
    """Save spillover diagnostics for every main-model run."""
    import matplotlib.pyplot as plt

    logger = sim.logger
    if len(logger.iterations) < 3:
        print('Spillover artifacts: skipped (too few observations).')
        return

    if not hasattr(logger, 'liquidity_spillover_metrics'):
        print('Spillover artifacts: skipped (logger has no spillover metrics API).')
        return

    os.makedirs(out_dir, exist_ok=True)

    lag = max(1, int(lag))
    rolling_window = max(5, int(rolling_window))
    split_at = _shock_iter_from_sim(sim)

    if hasattr(logger, 'clob_total_depth_series'):
        clob_depth = logger.clob_total_depth_series()
    else:
        clob_depth = [d.get('total', float('nan')) for d in logger.clob_depth]

    if hasattr(logger, 'amm_total_depth_series'):
        amm_depth = logger.amm_total_depth_series()
    else:
        amm_depth = [0.0 for _ in range(len(logger.iterations))]

    d_clob = logger._log_diff(clob_depth)
    d_amm = logger._log_diff(amm_depth)
    roll_corr = _spillover_rolling_corr(d_clob, d_amm, rolling_window)
    spill = logger.liquidity_spillover_metrics(lag=lag, split_at=split_at)
    clob_share, amm_share = _spillover_flow_share_series(logger)
    phase_windows = _spillover_phase_windows(sim, len(clob_depth))

    split_idx = None
    if split_at is not None:
        split_idx = max(0, min(int(split_at) - 1, min(len(d_clob), len(d_amm))))

    beta_std = {
        'full': {
            'amm_to_clob': _spillover_standardized_beta(
                d_clob, d_amm, spill['amm_to_clob'].get('beta', float('nan')), lag,
            ),
            'clob_to_amm': _spillover_standardized_beta(
                d_amm, d_clob, spill['clob_to_amm'].get('beta', float('nan')), lag,
            ),
        },
        'before': {'amm_to_clob': float('nan'), 'clob_to_amm': float('nan')},
        'after': {'amm_to_clob': float('nan'), 'clob_to_amm': float('nan')},
    }

    if split_idx is not None:
        before = spill.get('before_after', {}).get('before', {})
        after = spill.get('before_after', {}).get('after', {})
        beta_std['before']['amm_to_clob'] = _spillover_standardized_beta(
            d_clob[:split_idx], d_amm[:split_idx],
            before.get('amm_to_clob', {}).get('beta', float('nan')), lag,
        )
        beta_std['before']['clob_to_amm'] = _spillover_standardized_beta(
            d_amm[:split_idx], d_clob[:split_idx],
            before.get('clob_to_amm', {}).get('beta', float('nan')), lag,
        )
        beta_std['after']['amm_to_clob'] = _spillover_standardized_beta(
            d_clob[split_idx:], d_amm[split_idx:],
            after.get('amm_to_clob', {}).get('beta', float('nan')), lag,
        )
        beta_std['after']['clob_to_amm'] = _spillover_standardized_beta(
            d_amm[split_idx:], d_clob[split_idx:],
            after.get('clob_to_amm', {}).get('beta', float('nan')), lag,
        )

    pre_end = min(len(clob_depth), split_at) if split_at is not None else len(clob_depth)
    pre_clob = [x for x in clob_depth[:pre_end] if math.isfinite(x)]
    pre_amm = [x for x in amm_depth[:pre_end] if math.isfinite(x)]
    base_clob = (sum(pre_clob) / len(pre_clob)) if pre_clob else 1.0
    base_amm = (sum(pre_amm) / len(pre_amm)) if pre_amm else 1.0
    clob_idx = [x / base_clob if math.isfinite(x) and base_clob > 0 else float('nan') for x in clob_depth]
    amm_idx = [x / base_amm if math.isfinite(x) and base_amm > 0 else float('nan') for x in amm_depth]

    summary_rows = []
    for window in phase_windows:
        row = {
            'phase': window['phase'],
            'phase_label': window['label'],
            'start': window['start'],
            'end': window['end'],
            'clob_active_tick_customer_share': _spillover_window_mean(
                clob_share, window['start'], window['end']
            ),
            'amm_active_tick_customer_share': _spillover_window_mean(
                amm_share, window['start'], window['end']
            ),
            'avg_clob_depth_index': _spillover_window_mean(clob_idx, window['start'], window['end']),
            'avg_amm_depth_index': _spillover_window_mean(amm_idx, window['start'], window['end']),
        }
        row.update(_spillover_slice_metrics(logger, d_clob, d_amm, lag, window['start'], window['end']))
        summary_rows.append(row)

    df = pd.DataFrame({
        't': list(range(len(clob_depth))),
        'clob_depth_total': _spillover_safe_series(clob_depth),
        'amm_depth_total': _spillover_safe_series(amm_depth),
        'clob_depth_index': _spillover_safe_series(clob_idx),
        'amm_depth_index': _spillover_safe_series(amm_idx),
        'clob_active_tick_customer_share': _spillover_safe_series(clob_share),
        'amm_active_tick_customer_share': _spillover_safe_series(amm_share),
    })
    if len(d_clob):
        df_d = pd.DataFrame({
            't': list(range(1, len(clob_depth))),
            'dlog_clob_depth': _spillover_safe_series(d_clob),
            'dlog_amm_depth': _spillover_safe_series(d_amm),
            f'rolling_corr_w{rolling_window}': _spillover_safe_series(roll_corr),
        })
        df = df.merge(df_d, on='t', how='left')

    csv_path = os.path.join(out_dir, f'{prefix}.csv')
    df.to_csv(csv_path, index=False)

    summary_path = os.path.join(out_dir, f'{prefix}_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = list(axes.flat)

    x0 = np.arange(len(clob_idx))
    axes[0].plot(x0, clob_idx, label='CLOB depth index', color='#1f77b4', lw=2)
    axes[0].plot(x0, amm_idx, label='AMM depth index', color='#2ca02c', lw=2)
    if split_at is not None:
        axes[0].axvline(split_at, color='#d62728', ls='--', lw=1.5, label='shock')
    axes[0].set_title('Liquidity Levels: CLOB vs AMM (indexed to pre-shock mean)')
    axes[0].set_ylabel('Index')
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9)

    x1 = np.arange(1, len(clob_depth))
    axes[1].plot(x1, _spillover_safe_series(roll_corr), color='#9467bd', lw=2)
    axes[1].axhline(0.0, color='#333333', ls=':', lw=1.0)
    if split_at is not None:
        axes[1].axvline(split_at, color='#d62728', ls='--', lw=1.5)
    axes[1].set_title(f'Rolling correlation of liquidity changes (window={rolling_window})')
    axes[1].set_ylabel('corr')
    axes[1].grid(alpha=0.3)

    axes[2].plot(x0, _spillover_safe_series(clob_share), color='#1f77b4', lw=2,
                 label='CLOB customer share')
    axes[2].plot(x0, _spillover_safe_series(amm_share), color='#2ca02c', lw=2,
                 label='AMM customer share')
    if split_at is not None:
        axes[2].axvline(split_at, color='#d62728', ls='--', lw=1.5)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel('share of customer volume')
    axes[2].set_title('Venue shares on active customer-trade ticks')
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=9)

    phase_labels = [row['phase_label'] for row in summary_rows if row['phase'] != 'full']
    phase_rows = [row for row in summary_rows if row['phase'] != 'full'] or summary_rows
    pos = np.arange(len(phase_rows))
    wbar = 0.35
    axes[3].bar(
        pos - 0.5 * wbar,
        [row.get('amm_to_clob_beta_std', float('nan')) for row in phase_rows],
        width=wbar,
        color='#1f77b4',
        label='AMM→CLOB',
    )
    axes[3].bar(
        pos + 0.5 * wbar,
        [row.get('clob_to_amm_beta_std', float('nan')) for row in phase_rows],
        width=wbar,
        color='#2ca02c',
        label='CLOB→AMM',
    )
    axes[3].axhline(0.0, color='#333333', ls=':', lw=1.0)
    axes[3].set_xticks(pos)
    axes[3].set_xticklabels(phase_labels, rotation=15)
    axes[3].set_ylabel('standardized beta')
    axes[3].set_title(f'Directional spillovers by phase (lag={lag})')
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc='upper right', fontsize=9)

    fig.suptitle('Spillover diagnostics (main run)', fontsize=12)

    png_path = os.path.join(out_dir, f'{prefix}.png')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    table_fig, table_ax = plt.subplots(figsize=(11, max(2.8, 0.75 * len(summary_rows) + 1.8)))
    table_ax.axis('off')
    table_ax.set_title('Spillover summary by phase', fontsize=13, fontweight='bold', pad=12)
    table_rows = []
    for row in summary_rows:
        window_label = f"[{row['start']},{row['end']})"
        table_rows.append([
            row['phase_label'],
            window_label,
            f"{100.0 * row['amm_active_tick_customer_share']:.1f}%"
            if math.isfinite(row['amm_active_tick_customer_share']) else 'N/A',
            f"{100.0 * row['clob_active_tick_customer_share']:.1f}%"
            if math.isfinite(row['clob_active_tick_customer_share']) else 'N/A',
            f"{row['corr_dliq_clob_amm']:.3f}" if math.isfinite(row['corr_dliq_clob_amm']) else 'N/A',
            f"{row['amm_to_clob_beta_std']:.3f}" if math.isfinite(row['amm_to_clob_beta_std']) else 'N/A',
            f"{row['clob_to_amm_beta_std']:.3f}" if math.isfinite(row['clob_to_amm_beta_std']) else 'N/A',
        ])
    table = table_ax.table(
        cellText=table_rows,
        colLabels=['Phase', 'Window', 'AMM active-tick share',
                   'CLOB active-tick share', 'dliq corr', 'AMM→CLOB β',
                   'CLOB→AMM β'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    try:
        table.auto_set_column_width(col=list(range(7)))
    except AttributeError:
        pass
    for col_idx in range(7):
        table[0, col_idx].set_facecolor('#4c72b0')
        table[0, col_idx].set_text_props(color='white', fontweight='bold')
    table_path = os.path.join(out_dir, f'{prefix}_table.png')
    table_fig.savefig(table_path, dpi=180, bbox_inches='tight')
    plt.close(table_fig)

    print('\nSpillover artifacts saved:')
    print(f'  CSV -> {csv_path}')
    print(f'  Summary CSV -> {summary_path}')
    print(f'  PNG -> {png_path}')
    print(f'  Table PNG -> {table_path}')
    print(f'  Flow shares: CLOB={spill.get("avg_flow_share_clob", float("nan")):.3f} '
          f'AMM={spill.get("avg_flow_share_amm", float("nan")):.3f}')
    print('  Raw beta full: '
          f'AMM->CLOB={spill["amm_to_clob"].get("beta", float("nan")):.6g}, '
          f'CLOB->AMM={spill["clob_to_amm"].get("beta", float("nan")):.6g}')
    print('  Std beta full: '
          f'AMM->CLOB={beta_std["full"]["amm_to_clob"]:.6g}, '
          f'CLOB->AMM={beta_std["full"]["clob_to_amm"]:.6g}')
    print('  Bootstrap full (95% CI, p_boot): '
        f'AMM->CLOB=[{spill["amm_to_clob"].get("beta_boot_lo", float("nan")):.6g}, '
        f'{spill["amm_to_clob"].get("beta_boot_hi", float("nan")):.6g}], '
        f'p={spill["amm_to_clob"].get("p_boot", float("nan")):.4f}; '
        f'CLOB->AMM=[{spill["clob_to_amm"].get("beta_boot_lo", float("nan")):.6g}, '
        f'{spill["clob_to_amm"].get("beta_boot_hi", float("nan")):.6g}], '
        f'p={spill["clob_to_amm"].get("p_boot", float("nan")):.4f}')


def _cli_flag_present(flag: str, argv: list[str]) -> bool:
    return any(arg == flag or arg.startswith(f'{flag}=') for arg in argv)


def _resolve_main_routing(args: argparse.Namespace, argv: list[str]) -> str:
    if _cli_flag_present('--venue-choice-rule', argv):
        return args.venue_choice_rule
    if _cli_flag_present('--amm-share', argv):
        return 'fixed_share'
    return args.venue_choice_rule


def _main_output_dir(venue_choice_rule: str) -> str:
    if venue_choice_rule == 'fixed_share':
        return 'output/main_fixed'
    return 'output/main_aware'


def _run_main(argv: Optional[list[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Apply preset defaults (CLI overrides preset)
    _apply_preset_defaults(parser, args)

    args.venue_choice_rule = _resolve_main_routing(args, argv)
    plot_out_dir = _main_output_dir(args.venue_choice_rule)

    # 2. Auto-generate stress around shock (AFTER preset applied)
    _auto_stress_around_shock(args)
    args.run_label = _primary_run_label(args)

    # 3. Seed both RNGs
    if args.seed is not None:
        _seed_all(args.seed)

    print_config(args)

    sim = build_sim(args)
    sim.simulate(args.n_iter, silent=args.silent)

    target_payload = load_primary_model_targets()
    acceptance_report = None
    acceptance_scenario = _infer_acceptance_scenario(args)
    if target_payload.get('targets'):
        acceptance_report = build_acceptance_report(
            sim,
            target_payload=target_payload,
            run_label=args.run_label,
            scenario_name=acceptance_scenario,
        )

    if not args.no_summary:
        print_summary(sim, acceptance_report=acceptance_report)

    if args.robustness_check:
        print_robustness_summary(args)

    # The counterfactual is not a switch on this run. Adding a facility moves
    # committed capital, the obligation to keep quoting and the pricing rule
    # at once, and a comparison against the same market without it pools the
    # three. The resource matched arms of ``arms`` separate them, and that is
    # the sanctioned comparison.
    if not args.no_plots:
        generate_all_plots(sim, out_dir=plot_out_dir)

    if args.spillover_artifacts and not args.no_plots:
        save_spillover_artifacts(
            sim,
            out_dir=plot_out_dir,
            lag=args.spillover_lag,
            rolling_window=args.spillover_roll_window,
        )

    save_primary_model_artifacts(args, plot_out_dir, acceptance_report=acceptance_report)


COMMANDS = {
    'run': 'one simulation of a scenario, with its summary and plots',
    'accept': 'the acceptance panel against the frozen target matrix',
    'arms': 'the resource matched comparison of the facility arms',
    'welfare': 'the welfare account for one arm against the dealer only control',
    'selection': 'the markout of the flow each venue fills, calm against crisis',
    'identify': 'what moves the calm quoted spread, one parameter at a time',
    'participation': 'the crisis share a provider covers, by outside option',
    'figures': 'the figures of the article, redrawn from the current artifacts',
    'calibrate': 'the coordinate search over the declared parameters',
    'config': 'the calibrated configuration and where each value came from',
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f'  {name.ljust(width)}  {text}' for name, text in COMMANDS.items()]
    return ('usage: python -m main <command> [options]\n\n'
            'The model is configured by calibration/primary_model.json, which is\n'
            'the single source of every default below. A command takes the options\n'
            'of the runner it names; pass --help after the command to see them.\n\n'
            + '\n'.join(lines) + '\n')


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch to the workflow named by the first argument.

    The entry point used to be one run with a hundred and sixty switches and a
    built in comparison against the same market without a facility. Both the
    calibration and the comparison are now separate runners with their own
    seed commitments and their own provenance, so the entry point names them
    instead of reimplementing them.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(_usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f'unknown command: {command}\n', file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    if command == 'run':
        _run_main(rest)
        return 0
    if command == 'config':
        parser = build_parser()
        args = parser.parse_args(rest)
        _apply_preset_defaults(parser, args)
        args.venue_choice_rule = _resolve_main_routing(args, rest)
        print_config(args)
        return 0
    if command == 'figures':
        from AgentBasedModel.visualization.paper_figures import ALL, draw_all
        drawn = draw_all()
        for path in drawn:
            print(os.path.relpath(path))
        # A figure that cannot be drawn leaves the article carrying the one the
        # previous run left behind, which is worse than carrying none, so a
        # partial set is reported as a failure and not as a warning.
        missing = len(ALL) - len(drawn)
        if missing:
            print(f'{missing} of {len(ALL)} figures were not drawn',
                  file=sys.stderr)
            return 1
        return 0

    def _delegate(entry, name, extra=()):
        """Run a module entry point that reads sys.argv, with our arguments."""
        saved, sys.argv = sys.argv, [name] + list(extra) + rest
        try:
            return int(entry() or 0)
        finally:
            sys.argv = saved

    if command == 'accept':
        from calibration.runner import main as run_panel
        # The panel evaluates the current defaults; the search is `calibrate`.
        extra = () if '--current-only' in rest else ('--current-only',)
        return _delegate(run_panel, 'calibration.runner', extra)
    if command == 'calibrate':
        from calibration.runner import main as run_search
        return _delegate(run_search, 'calibration.runner')
    if command == 'welfare':
        from tools.robustness.welfare_accounting import main as run_welfare
        return _delegate(run_welfare, 'welfare_accounting')
    if command == 'arms':
        from tools.robustness.facility_arms import main as run_arms
        return int(run_arms(rest) or 0)
    if command == 'selection':
        from tools.robustness.flow_selection import main as run_selection
        return int(run_selection(rest) or 0)
    if command == 'identify':
        from tools.robustness.spread_identification import main as run_identify
        return _delegate(run_identify, 'spread_identification')
    if command == 'participation':
        from tools.robustness.participation_grid import main as run_grid
        return _delegate(run_grid, 'participation_grid')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
