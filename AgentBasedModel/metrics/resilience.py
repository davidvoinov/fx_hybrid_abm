from __future__ import annotations

import math
import statistics
from typing import Iterable, List, Optional, Sequence


def _finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def _median_finite(values: Iterable[float]) -> float:
    clean = _finite(values)
    return float(statistics.median(clean)) if clean else float('nan')


def _mean_finite(values: Iterable[float]) -> float:
    clean = _finite(values)
    return float(sum(clean) / len(clean)) if clean else float('nan')


def baseline_level(series: List[float], shock_iter: int,
                   lookback: int = 50) -> float:
    """Robust pre-shock baseline from the last *lookback* observations."""
    if shock_iter <= 0:
        return float('nan')

    start = max(0, shock_iter - lookback)
    baseline = _median_finite(series[start:shock_iter])
    if math.isfinite(baseline):
        return baseline
    return _median_finite(series[:shock_iter])


def pct_deviation_series(series: List[float], baseline: float) -> List[float]:
    """Percent deviation from baseline for each observation."""
    if not math.isfinite(baseline) or baseline == 0:
        return [float('nan')] * len(series)

    out: List[float] = []
    for value in series:
        if math.isfinite(value):
            out.append(100.0 * (value - baseline) / baseline)
        else:
            out.append(float('nan'))
    return out


def pct_gap_to_target_series(series: Sequence[float],
                             target_series: Sequence[float]) -> List[float]:
    """Percent gap between an observed series and a time-varying target series."""
    out: List[float] = []
    n = max(len(series), len(target_series))
    for idx in range(n):
        value = series[idx] if idx < len(series) else float('nan')
        target = target_series[idx] if idx < len(target_series) else float('nan')
        if math.isfinite(value) and math.isfinite(target) and target != 0:
            out.append(100.0 * (value - target) / target)
        else:
            out.append(float('nan'))
    return out


def initial_dislocation_pct(dev_series: List[float], shock_iter: int,
                            impact_window: int = 10) -> dict:
    """Early peak post-shock deviation used to normalize recovery statistics."""
    if shock_iter >= len(dev_series):
        return {
            'dislocation_pct': float('nan'),
            'dislocation_step': float('nan'),
        }

    end = min(len(dev_series), shock_iter + max(1, impact_window))
    window = [
        (idx, value)
        for idx, value in enumerate(dev_series[shock_iter:end], start=shock_iter)
        if math.isfinite(value)
    ]
    if not window:
        return {
            'dislocation_pct': float('nan'),
            'dislocation_step': float('nan'),
        }

    peak_idx, peak_value = max(window, key=lambda item: abs(item[1]))
    return {
        'dislocation_pct': float(peak_value),
        'dislocation_step': float(peak_idx - shock_iter),
    }


def recovery_steps_pct(dev_series: List[float], shock_iter: int,
                       tolerance_pct: float = 1.0,
                       stable_window: int = 10,
                       horizon: Optional[int] = None) -> float:
    """Legacy fixed-band recovery time kept for backward compatibility."""
    if shock_iter >= len(dev_series):
        return float('nan')

    end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + horizon)
    if end - shock_iter < stable_window:
        return float('nan')

    for idx in range(shock_iter, end - stable_window + 1):
        window = dev_series[idx:idx + stable_window]
        if len(window) < stable_window:
            break
        if all(math.isfinite(value) and abs(value) <= tolerance_pct for value in window):
            return float(idx - shock_iter)

    return float(end - shock_iter)


def recovery_steps_target_band(series: Sequence[float], shock_iter: int,
                               stable_window: int = 10,
                               horizon: Optional[int] = None,
                               target_series: Optional[Sequence[float]] = None,
                               lower_ratio: float = 0.5,
                               upper_ratio: float = 1.5) -> dict:
    """Recovery time until a series re-enters a target band and stays there.

    The target band is defined as ``[lower_ratio * target, upper_ratio * target]``.
    When ``target_series`` is omitted, the pre-shock baseline is used as a
    constant target.
    """
    if shock_iter >= len(series):
        return {
            'recovery_steps': float('nan'),
            'time_observed_steps': float('nan'),
            'recovered': False,
            'is_censored': True,
            'target_lower_ratio': float(lower_ratio),
            'target_upper_ratio': float(upper_ratio),
            'target_abs_deviation_pct': float('nan'),
        }

    lower_ratio = float(lower_ratio)
    upper_ratio = float(upper_ratio)
    if lower_ratio > upper_ratio:
        lower_ratio, upper_ratio = upper_ratio, lower_ratio

    end = len(series) if horizon is None else min(len(series), shock_iter + horizon)
    observed_steps = max(0, end - shock_iter)
    if observed_steps < stable_window:
        return {
            'recovery_steps': float('nan'),
            'time_observed_steps': float(observed_steps),
            'recovered': False,
            'is_censored': True,
            'target_lower_ratio': lower_ratio,
            'target_upper_ratio': upper_ratio,
            'target_abs_deviation_pct': 100.0 * max(abs(1.0 - lower_ratio), abs(upper_ratio - 1.0)),
        }

    scalar_target = None
    if target_series is None:
        scalar_target = baseline_level(list(series), shock_iter, lookback=50)

    for idx in range(shock_iter, end - stable_window + 1):
        window_ok = True
        for win_idx in range(idx, idx + stable_window):
            value = series[win_idx]
            target = scalar_target if target_series is None else target_series[win_idx]
            if not (math.isfinite(value) and math.isfinite(target) and target > 0):
                window_ok = False
                break
            lower_bound = lower_ratio * target
            upper_bound = upper_ratio * target
            if value < lower_bound or value > upper_bound:
                window_ok = False
                break
        if window_ok:
            steps = float(idx - shock_iter)
            return {
                'recovery_steps': steps,
                'time_observed_steps': steps,
                'recovered': True,
                'is_censored': False,
                'target_lower_ratio': lower_ratio,
                'target_upper_ratio': upper_ratio,
                'target_abs_deviation_pct': 100.0 * max(abs(1.0 - lower_ratio), abs(upper_ratio - 1.0)),
            }

    return {
        'recovery_steps': float('nan'),
        'time_observed_steps': float(observed_steps),
        'recovered': False,
        'is_censored': True,
        'target_lower_ratio': lower_ratio,
        'target_upper_ratio': upper_ratio,
        'target_abs_deviation_pct': 100.0 * max(abs(1.0 - lower_ratio), abs(upper_ratio - 1.0)),
    }


def recovery_steps_retracement(dev_series: List[float], shock_iter: int,
                               retracement_fraction: float = 0.8,
                               stable_window: int = 10,
                               horizon: Optional[int] = None,
                               impact_window: int = 10) -> dict:
    """Recovery time defined as a sustained closure of a fraction of the initial dislocation."""
    if shock_iter >= len(dev_series):
        return {
            'recovery_steps': float('nan'),
            'time_observed_steps': float('nan'),
            'recovered': False,
            'is_censored': True,
            'target_abs_deviation_pct': float('nan'),
            'initial_dislocation_pct': float('nan'),
        }

    end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + horizon)
    observed_steps = max(0, end - shock_iter)
    dislocation = initial_dislocation_pct(dev_series, shock_iter, impact_window=impact_window)
    d0 = dislocation['dislocation_pct']
    anchor_step = dislocation['dislocation_step']
    if not math.isfinite(d0):
        return {
            'recovery_steps': float('nan'),
            'time_observed_steps': float(observed_steps),
            'recovered': False,
            'is_censored': True,
            'target_abs_deviation_pct': float('nan'),
            'initial_dislocation_pct': float('nan'),
            'reference_dislocation_step': float('nan'),
        }

    retracement_fraction = min(max(retracement_fraction, 0.0), 1.0)
    d0_abs = abs(d0)
    if d0_abs == 0.0:
        return {
            'recovery_steps': 0.0,
            'time_observed_steps': 0.0,
            'recovered': True,
            'is_censored': False,
            'target_abs_deviation_pct': 0.0,
            'initial_dislocation_pct': 0.0,
            'reference_dislocation_step': anchor_step,
        }

    target_abs = (1.0 - retracement_fraction) * d0_abs
    if observed_steps < stable_window:
        return {
            'recovery_steps': float('nan'),
            'time_observed_steps': float(observed_steps),
            'recovered': False,
            'is_censored': True,
            'target_abs_deviation_pct': target_abs,
            'initial_dislocation_pct': d0,
            'reference_dislocation_step': anchor_step,
        }

    start_idx = shock_iter + int(anchor_step) if math.isfinite(anchor_step) else shock_iter
    for idx in range(start_idx, end - stable_window + 1):
        window = dev_series[idx:idx + stable_window]
        if len(window) < stable_window:
            break
        if all(math.isfinite(value) and abs(value) <= target_abs for value in window):
            steps = float(idx - shock_iter)
            return {
                'recovery_steps': steps,
                'time_observed_steps': steps,
                'recovered': True,
                'is_censored': False,
                'target_abs_deviation_pct': target_abs,
                'initial_dislocation_pct': d0,
                'reference_dislocation_step': anchor_step,
            }

    return {
        'recovery_steps': float('nan'),
        'time_observed_steps': float(observed_steps),
        'recovered': False,
        'is_censored': True,
        'target_abs_deviation_pct': target_abs,
        'initial_dislocation_pct': d0,
        'reference_dislocation_step': anchor_step,
    }


def half_life_steps_pct(dev_series: List[float], shock_iter: int,
                        stable_window: int = 3,
                        horizon: Optional[int] = None) -> float:
    """Time until absolute deviation halves relative to the largest post-shock dislocation."""
    if shock_iter >= len(dev_series):
        return float('nan')

    end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + horizon)
    post = dev_series[shock_iter:end]
    finite_post = [(idx, value) for idx, value in enumerate(post) if math.isfinite(value)]
    if not finite_post:
        return float('nan')

    peak_idx, peak_value = max(finite_post, key=lambda item: abs(item[1]))
    target = 0.5 * abs(peak_value)
    origin = shock_iter + peak_idx

    for idx in range(origin, end - stable_window + 1):
        window = dev_series[idx:idx + stable_window]
        if len(window) < stable_window:
            break
        if all(math.isfinite(value) and abs(value) <= target for value in window):
            return float(idx - shock_iter)

    return float(end - shock_iter)


def trough_change_pct(dev_series: List[float], shock_iter: int,
                      horizon: Optional[int] = None) -> float:
    """Most negative post-shock deviation."""
    if shock_iter >= len(dev_series):
        return float('nan')

    end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + horizon)
    clean = _finite(dev_series[shock_iter:end])
    return min(clean) if clean else float('nan')


def peak_abs_change_pct(dev_series: List[float], shock_iter: int,
                        horizon: Optional[int] = None) -> float:
    """Largest absolute post-shock deviation in percent."""
    if shock_iter >= len(dev_series):
        return float('nan')

    end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + horizon)
    clean = _finite(dev_series[shock_iter:end])
    return max((abs(value) for value in clean), default=float('nan'))


def normalized_avg_impact(dev_series: List[float], shock_iter: int,
                          initial_dislocation: float,
                          avg_window: int = 20,
                          horizon: Optional[int] = None) -> float:
    """Signed average post-shock deviation normalized by the initial dislocation magnitude."""
    scale = abs(initial_dislocation)
    if not math.isfinite(scale) or scale == 0.0:
        return float('nan')

    avg_end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + min(avg_window, horizon))
    if avg_end <= shock_iter:
        avg_end = min(len(dev_series), shock_iter + avg_window)
    avg_change = _mean_finite(dev_series[shock_iter:avg_end])
    if not math.isfinite(avg_change):
        return float('nan')
    return avg_change / scale


def normalized_auc_abs(dev_series: List[float], shock_iter: int,
                       initial_dislocation: float,
                       horizon: Optional[int] = None) -> float:
    """Mean absolute post-shock deviation normalized by the initial dislocation magnitude."""
    scale = abs(initial_dislocation)
    if not math.isfinite(scale) or scale == 0.0:
        return float('nan')

    end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + horizon)
    clean = _finite(abs(value) for value in dev_series[shock_iter:end])
    return _mean_finite(clean) / scale if clean else float('nan')


def series_resilience_metrics(series: Sequence[float], shock_iter: int,
                              baseline_window: int = 50,
                              avg_window: int = 20,
                              tolerance_pct: float = 50.0,
                              stable_window: int = 10,
                              horizon: Optional[int] = None,
                              retracement_fraction: float = 0.8,
                              impact_window: int = 10,
                              analysis_target_mode: str = 'pre_shock_baseline') -> dict:
    """Generic resilience metrics for any scalar post-shock series."""
    baseline = baseline_level(list(series), shock_iter, lookback=baseline_window)
    dev_series = pct_deviation_series(list(series), baseline)

    dislocation = initial_dislocation_pct(dev_series, shock_iter, impact_window=impact_window)
    initial_dislocation = dislocation['dislocation_pct']
    band_recovery = recovery_steps_target_band(
        series,
        shock_iter,
        stable_window=stable_window,
        horizon=horizon,
        lower_ratio=0.5,
        upper_ratio=1.5,
    )
    retracement_recovery = recovery_steps_retracement(
        dev_series,
        shock_iter,
        retracement_fraction=retracement_fraction,
        stable_window=stable_window,
        horizon=horizon,
        impact_window=impact_window,
    )

    avg_end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + min(avg_window, horizon))
    if avg_end <= shock_iter:
        avg_end = min(len(dev_series), shock_iter + avg_window)

    return {
        'baseline_level': baseline,
        'shock_change_pct': dev_series[shock_iter] if shock_iter < len(dev_series) else float('nan'),
        'initial_dislocation_pct': initial_dislocation,
        'reference_dislocation_pct': initial_dislocation,
        'reference_dislocation_step': dislocation['dislocation_step'],
        'analysis_target_mode': analysis_target_mode,
        'avg_change_pct': _mean_finite(dev_series[shock_iter:avg_end]),
        'normalized_avg_impact': normalized_avg_impact(
            dev_series, shock_iter, initial_dislocation, avg_window=avg_window, horizon=horizon,
        ),
        'normalized_auc_abs': normalized_auc_abs(
            dev_series, shock_iter, initial_dislocation, horizon=horizon,
        ),
        'trough_change_pct': trough_change_pct(dev_series, shock_iter, horizon=horizon),
        'peak_abs_change_pct': peak_abs_change_pct(dev_series, shock_iter, horizon=horizon),
        'half_life_steps': half_life_steps_pct(
            dev_series, shock_iter, stable_window=max(3, stable_window // 3), horizon=horizon,
        ),
        'legacy_recovery_steps': band_recovery['recovery_steps'],
        'recovery_steps_retracement': retracement_recovery['recovery_steps'],
        'time_observed_steps_retracement': retracement_recovery['time_observed_steps'],
        'recovered_retracement': retracement_recovery['recovered'],
        'is_censored_retracement': retracement_recovery['is_censored'],
        'recovery_steps_fixed_band_pct': recovery_steps_pct(
            dev_series, shock_iter, tolerance_pct=tolerance_pct,
            stable_window=stable_window, horizon=horizon,
        ),
        'recovery_steps': retracement_recovery['recovery_steps'],
        'time_observed_steps': retracement_recovery['time_observed_steps'],
        'recovered': retracement_recovery['recovered'],
        'is_censored': retracement_recovery['is_censored'],
        'target_abs_deviation_pct': retracement_recovery['target_abs_deviation_pct'],
        'recovery_steps_band': band_recovery['recovery_steps'],
        'recovered_band': band_recovery['recovered'],
        'is_censored_band': band_recovery['is_censored'],
        'target_lower_ratio': band_recovery['target_lower_ratio'],
        'target_upper_ratio': band_recovery['target_upper_ratio'],
        'deviation_series': dev_series,
    }


def composite_resilience_metrics(component_metrics: Sequence[dict],
                                 analysis_target_mode: str = 'composite') -> dict:
    """Combine multiple recovery criteria into a joint event-time metric."""
    valid = [metric for metric in component_metrics if metric]
    if not valid:
        return {
            'analysis_target_mode': analysis_target_mode,
            'recovery_steps': float('nan'),
            'time_observed_steps': float('nan'),
            'recovered': False,
            'is_censored': True,
            'half_life_steps': float('nan'),
            'legacy_recovery_steps': float('nan'),
            'initial_dislocation_pct': float('nan'),
            'reference_dislocation_pct': float('nan'),
            'reference_dislocation_step': float('nan'),
            'target_abs_deviation_pct': float('nan'),
            'avg_change_pct': float('nan'),
            'normalized_avg_impact': float('nan'),
            'normalized_auc_abs': float('nan'),
            'trough_change_pct': float('nan'),
            'peak_abs_change_pct': float('nan'),
        }

    observed_steps = _finite(metric.get('time_observed_steps', float('nan')) for metric in valid)
    recovery_steps = _finite(metric.get('recovery_steps', float('nan')) for metric in valid)
    initial_dislocations = _finite(metric.get('initial_dislocation_pct', float('nan')) for metric in valid)
    reference_steps = _finite(metric.get('reference_dislocation_step', float('nan')) for metric in valid)
    targets = _finite(metric.get('target_abs_deviation_pct', float('nan')) for metric in valid)
    avg_changes = _finite(metric.get('avg_change_pct', float('nan')) for metric in valid)
    avg_impacts = _finite(metric.get('normalized_avg_impact', float('nan')) for metric in valid)
    auc_values = _finite(metric.get('normalized_auc_abs', float('nan')) for metric in valid)
    trough_values = _finite(metric.get('trough_change_pct', float('nan')) for metric in valid)
    peak_values = _finite(metric.get('peak_abs_change_pct', float('nan')) for metric in valid)
    half_lives = _finite(metric.get('half_life_steps', float('nan')) for metric in valid)
    legacy_steps = _finite(metric.get('legacy_recovery_steps', float('nan')) for metric in valid)

    recovered = all(bool(metric.get('recovered', False)) for metric in valid)
    composite_recovery = max(recovery_steps) if recovered and recovery_steps else float('nan')
    composite_observed = min(observed_steps) if observed_steps else float('nan')

    return {
        'analysis_target_mode': analysis_target_mode,
        'recovery_steps': composite_recovery,
        'time_observed_steps': composite_recovery if recovered and math.isfinite(composite_recovery) else composite_observed,
        'recovered': recovered,
        'is_censored': not recovered,
        'half_life_steps': max(half_lives) if half_lives else float('nan'),
        'legacy_recovery_steps': max(legacy_steps) if legacy_steps else float('nan'),
        'initial_dislocation_pct': max(initial_dislocations, key=abs) if initial_dislocations else float('nan'),
        'reference_dislocation_pct': max(initial_dislocations, key=abs) if initial_dislocations else float('nan'),
        'reference_dislocation_step': max(reference_steps) if reference_steps else float('nan'),
        'target_abs_deviation_pct': max(targets) if targets else float('nan'),
        'avg_change_pct': _mean_finite(avg_changes),
        'normalized_avg_impact': _mean_finite(avg_impacts),
        'normalized_auc_abs': _mean_finite(auc_values),
        'trough_change_pct': min(trough_values) if trough_values else float('nan'),
        'peak_abs_change_pct': max(peak_values) if peak_values else float('nan'),
    }


def kaplan_meier_curve(observed_steps: Sequence[float], recovered_flags: Sequence[bool]) -> dict:
    """Kaplan-Meier survival curve for time-to-recovery with right censoring."""
    sample = []
    for time_step, recovered in zip(observed_steps, recovered_flags):
        if math.isfinite(time_step) and time_step >= 0:
            sample.append((float(time_step), bool(recovered)))

    if not sample:
        return {
            'times': [0.0],
            'survival': [1.0],
            'median_time': float('nan'),
            'n_obs': 0,
        }

    unique_event_times = sorted({time_step for time_step, recovered in sample if recovered})
    times = [0.0]
    survival = [1.0]
    current_survival = 1.0
    median_time = float('nan')

    for time_step in unique_event_times:
        at_risk = sum(1 for observed, _ in sample if observed >= time_step)
        events = sum(1 for observed, recovered in sample if recovered and observed == time_step)
        if at_risk <= 0:
            continue
        current_survival *= (1.0 - events / at_risk)
        times.append(time_step)
        survival.append(current_survival)
        if not math.isfinite(median_time) and current_survival <= 0.5:
            median_time = time_step

    return {
        'times': times,
        'survival': survival,
        'median_time': median_time,
        'n_obs': len(sample),
    }


def price_resilience_metrics(series: List[float], shock_iter: int,
                             baseline_window: int = 50,
                             avg_window: int = 20,
                             tolerance_pct: float = 50.0,
                             stable_window: int = 10,
                             horizon: Optional[int] = None,
                             retracement_fraction: float = 0.8,
                             impact_window: int = 10,
                             target_series: Optional[Sequence[float]] = None,
                             target_mode: str = 'pre_shock_baseline') -> dict:
    """Price-centric resilience metrics for post-shock recovery studies.

    Price recovery remains anchored to relative-dislocation closure rather than a
    very wide level band. The manuscript-style 0.5x–1.5x norm band is used for
    liquidity metrics, while price keeps a stricter event-study notion of
    recovery to avoid trivial immediate recovery after large price shocks.
    """
    resolved_target_mode = target_mode
    if target_mode == 'fair_value_gap' and target_series is not None:
        baseline = baseline_level(series, shock_iter, lookback=baseline_window)
        dev_series = pct_gap_to_target_series(series, target_series)
        band_recovery = recovery_steps_target_band(
            series,
            shock_iter,
            stable_window=stable_window,
            horizon=horizon,
            target_series=target_series,
            lower_ratio=0.5,
            upper_ratio=1.5,
        )
    else:
        resolved_target_mode = 'pre_shock_baseline'
        baseline = baseline_level(series, shock_iter, lookback=baseline_window)
        dev_series = pct_deviation_series(series, baseline)
        band_recovery = recovery_steps_target_band(
            series,
            shock_iter,
            stable_window=stable_window,
            horizon=horizon,
            lower_ratio=0.5,
            upper_ratio=1.5,
        )

    dislocation = initial_dislocation_pct(dev_series, shock_iter, impact_window=impact_window)
    initial_dislocation = dislocation['dislocation_pct']
    retracement_recovery = recovery_steps_retracement(
        dev_series,
        shock_iter,
        retracement_fraction=retracement_fraction,
        stable_window=stable_window,
        horizon=horizon,
        impact_window=impact_window,
    )

    avg_end = len(dev_series) if horizon is None else min(len(dev_series), shock_iter + min(avg_window, horizon))
    if avg_end <= shock_iter:
        avg_end = min(len(dev_series), shock_iter + avg_window)

    return {
        'baseline_price': baseline,
        'shock_change_pct': dev_series[shock_iter] if shock_iter < len(dev_series) else float('nan'),
        'initial_dislocation_pct': initial_dislocation,
        'reference_dislocation_pct': initial_dislocation,
        'reference_dislocation_step': dislocation['dislocation_step'],
        'analysis_target_mode': resolved_target_mode,
        'avg_change_pct': _mean_finite(dev_series[shock_iter:avg_end]),
        'normalized_avg_impact': normalized_avg_impact(
            dev_series, shock_iter, initial_dislocation, avg_window=avg_window, horizon=horizon,
        ),
        'normalized_auc_abs': normalized_auc_abs(
            dev_series, shock_iter, initial_dislocation, horizon=horizon,
        ),
        'trough_change_pct': trough_change_pct(dev_series, shock_iter, horizon=horizon),
        'peak_abs_change_pct': peak_abs_change_pct(dev_series, shock_iter, horizon=horizon),
        'half_life_steps': half_life_steps_pct(
            dev_series, shock_iter, stable_window=max(3, stable_window // 3), horizon=horizon,
        ),
        'legacy_recovery_steps': band_recovery['recovery_steps'],
        'recovery_steps_retracement': retracement_recovery['recovery_steps'],
        'time_observed_steps_retracement': retracement_recovery['time_observed_steps'],
        'recovered_retracement': retracement_recovery['recovered'],
        'is_censored_retracement': retracement_recovery['is_censored'],
        'recovery_steps_fixed_band_pct': recovery_steps_pct(
            dev_series, shock_iter, tolerance_pct=tolerance_pct,
            stable_window=stable_window, horizon=horizon,
        ),
        'recovery_steps': retracement_recovery['recovery_steps'],
        'time_observed_steps': retracement_recovery['time_observed_steps'],
        'recovered': retracement_recovery['recovered'],
        'is_censored': retracement_recovery['is_censored'],
        'target_abs_deviation_pct': retracement_recovery['target_abs_deviation_pct'],
        'target_lower_ratio': band_recovery['target_lower_ratio'],
        'target_upper_ratio': band_recovery['target_upper_ratio'],
        'deviation_series': dev_series,
    }