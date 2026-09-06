from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional

from AgentBasedModel.metrics.resilience import (
    baseline_level,
    half_life_steps_pct,
    pct_deviation_series,
)


def _finite_mean(series: list[float]) -> float:
    finite = [float(value) for value in series if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float('nan')


def _finite_median(series: list[float]) -> float:
    finite = sorted(float(value) for value in series if math.isfinite(value))
    if not finite:
        return float('nan')
    mid = len(finite) // 2
    if len(finite) % 2 == 1:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _full_withdrawal_ticks(series, tolerance: float = 1e-9) -> float:
    """Ticks on which every dealer is withdrawn through the endogenous channel."""
    count = 0
    for value in series or ():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v >= 1.0 - tolerance:
            count += 1
    return float(count)


def _finite_max(series: list[float]) -> float:
    finite = [float(value) for value in series if math.isfinite(value)]
    return max(finite) if finite else float('nan')


_LIFECYCLE_TERMINAL_REASONS = frozenset({
    'fill',
    'scheduled_cancel',
    'stale_reprice',
    'provider_withdrawal',
    'endogenous_withdrawal',
    'forced_pause',
    'default',
    'dealer_cancel',
    'provider_cancel',
    'no_reference',
    'scenario_cancel_wave',
    'cancel',
})


def _kaplan_meier_median(rows: list[dict]) -> float:
    """Product-limit median for terminal and right-censored order lives.

    An order still resting at the last simulated tick has survived for at
    least its observed exposure. Dropping these administratively censored
    observations biases a multi-minute dealer lifetime downward in a
    1,000-second window. Events are processed before censoring at a tied time,
    as in the standard Kaplan--Meier convention.
    """
    observations: list[tuple[float, bool]] = []
    for row in rows:
        try:
            lifetime = float(row.get('lifetime'))
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(lifetime) or lifetime < 0.0:
            continue
        observations.append((lifetime, not bool(row.get('censored', False))))
    if not observations:
        return float('nan')

    grouped: dict[float, list[int]] = {}
    for lifetime, is_event in observations:
        counts = grouped.setdefault(lifetime, [0, 0])
        counts[0 if is_event else 1] += 1

    at_risk = len(observations)
    survival = 1.0
    for lifetime in sorted(grouped):
        events, censored = grouped[lifetime]
        if at_risk <= 0:
            break
        if events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5 + 1e-15:
                return float(lifetime)
        at_risk -= events + censored
    return float('nan')


def _lifecycle_summary(rows: list[dict]) -> dict[str, float]:
    """Scalar lifecycle diagnostics suitable for a seed-panel report."""
    completed = []
    live = []
    for row in rows:
        try:
            lifetime = float(row.get('lifetime'))
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(lifetime) or lifetime < 0.0:
            continue
        (live if bool(row.get('censored', False)) else completed).append(row)

    completed_lives = [float(row['lifetime']) for row in completed]
    atom_counts = Counter(completed_lives)
    max_atom_share = (
        max(atom_counts.values()) / len(completed_lives)
        if completed_lives else float('nan')
    )
    uncategorized = sum(
        str(row.get('reason', '')) not in _LIFECYCLE_TERMINAL_REASONS
        for row in completed
    )
    uncategorized_share = (
        uncategorized / len(completed) if completed else float('nan')
    )
    scheduled = [row for row in completed
                 if row.get('reason') == 'scheduled_cancel']
    same_tick_counts = Counter(
        row.get('ended_tick') for row in scheduled
        if row.get('ended_tick') is not None
    )
    same_tick_share = (
        max(same_tick_counts.values()) / len(scheduled)
        if scheduled and same_tick_counts else float('nan')
    )
    return {
        'km_median': _kaplan_meier_median(rows),
        'completed_median': _finite_median(completed_lives),
        'completed_count': float(len(completed)),
        'live_censored_count': float(len(live)),
        'censored_share': (
            len(live) / (len(completed) + len(live))
            if completed or live else float('nan')
        ),
        'max_completed_lifetime_atom_share': max_atom_share,
        'uncategorized_terminal_share': uncategorized_share,
        'max_same_tick_scheduled_end_share': same_tick_share,
    }


def _window_mean(series: list[float], start: int, end: int) -> float:
    if not series:
        return float('nan')
    lo = max(0, int(start))
    hi = max(lo, min(len(series), int(end)))
    return _finite_mean(series[lo:hi])


def _lag1_autocorr(series: list[float]) -> float:
    pairs = [
        (float(prev), float(curr))
        for prev, curr in zip(series[:-1], series[1:])
        if math.isfinite(prev) and math.isfinite(curr)
    ]
    if len(pairs) < 3:
        return float('nan')

    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    vy = sum((y - my) ** 2 for y in ys) / len(ys)
    if vx <= 0 or vy <= 0:
        return float('nan')
    return cov / math.sqrt(vx * vy)


def _signed_nonzero_series(series: list[float], zero_tol: float = 1e-12) -> list[float]:
    out: list[float] = []
    for value in series:
        if not math.isfinite(value):
            continue
        if abs(float(value)) <= zero_tol:
            continue
        out.append(1.0 if float(value) > 0 else -1.0)
    return out


def _run_lengths(sign_series: list[float]) -> list[float]:
    if not sign_series:
        return []

    runs: list[float] = []
    current = sign_series[0]
    run_length = 1
    for sign in sign_series[1:]:
        if sign == current:
            run_length += 1
            continue
        runs.append(float(run_length))
        current = sign
        run_length = 1
    runs.append(float(run_length))
    return runs


def _signed_flow_diagnostics(series: list[float]) -> dict[str, float]:
    sign_series = _signed_nonzero_series(series)
    run_lengths = _run_lengths(sign_series)
    return {
        'lag1_sign_autocorr': _lag1_autocorr(sign_series),
        'mean_run_length': _finite_mean(run_lengths),
    }


def _nearest_trade_size(grid: list[float], target_q: float) -> float:
    if not grid:
        return float(target_q)
    return min(grid, key=lambda candidate: abs(float(candidate) - float(target_q)))



# Periods after the shock over which the dealer response is read. It was 60,
# which is shorter than the response it measures: with the inventory half life
# at 65 seconds and withdrawal requiring two consecutive above-threshold
# observations, impaired capacity reaches 0.56 only by the fortieth period and
# peaks around the eightieth. Measured on the identified March 2020 episode over
# 300 seeds, a 60 period window recorded no withdrawal at all on 29 per cent of
# seeds where dealers did withdraw later, so crisis activation read 0.707 while
# the sector responded on essentially every path.
DEALER_RESPONSE_WINDOW = 150


def _price_discovery_half_life(logger, shock_iter) -> float:
    """Periods for the book's distance from the latent value to halve.

    A market discovers a price; it is not handed one. This measures whether
    the traded mid actually converges on the latent value after it moves,
    and it exists because nothing else in the target set does. A book that
    stood two hundred and eighty basis points away from the fundamental for
    a hundred periods passed every other acceptance target, which made the
    whole set blind to the one thing a price formation model has to get
    right.
    """
    if shock_iter is None:
        return float('nan')
    mid = list(getattr(logger, 'clob_mid_series', []) or [])
    fair = list(getattr(logger, 'fair_price_series', []) or [])
    n = min(len(mid), len(fair))
    start = int(shock_iter)
    if n <= start + 1:
        return float('nan')

    def gap(i):
        m, f = mid[i], fair[i]
        if m is None or f is None or not (m == m) or not (f == f) or f <= 0:
            return float('nan')
        return abs(m - f) / f * 1e4

    opening = gap(start)
    if not (opening == opening) or opening <= 0.0:
        return float('nan')
    for i in range(start, n):
        g = gap(i)
        if g == g and g <= opening / 2.0:
            return float(i - start)
    # Never halved inside the run; report the window and not a silent nan
    # so that the failure is visible as a number.
    return float(n - start)


class CalibrationFitter:
    """Scaffold for evaluating one simulation run against literature targets."""

    def __init__(self, target_payload: Optional[dict] = None):
        self.target_payload = target_payload or {}
        self.targets = list(self.target_payload.get('targets', []))

    @staticmethod
    def _applies_to_scenario(target: dict[str, Any], scenario_name: Optional[str]) -> bool:
        target_scenario = target.get('evaluation_scenario')
        if target_scenario is None:
            return True
        return target_scenario == scenario_name

    def target_metric_defaults(self, scenario_name: Optional[str] = None) -> dict[str, float]:
        defaults: dict[str, float] = {}
        for target in self.targets:
            if not self._applies_to_scenario(target, scenario_name):
                continue
            observable = target.get('observable')
            if not observable:
                continue
            if 'target_value' in target:
                defaults[observable] = float(target['target_value'])
                continue
            target_range = target.get('target_range') or {}
            if 'low' in target_range and 'high' in target_range:
                defaults[observable] = 0.5 * (float(target_range['low']) + float(target_range['high']))
                continue
            # A one-sided range states a bound and no centre, so the bound is
            # the only value a default can honestly take.
            if 'low' in target_range:
                defaults[observable] = float(target_range['low'])
                continue
            if 'high' in target_range:
                defaults[observable] = float(target_range['high'])
                continue
            if target.get('qualitative_bounds'):
                defaults[observable] = 1.0
        return defaults

    @staticmethod
    def _shock_iter(sim) -> Optional[int]:
        shock_iter = getattr(sim, 'shock_iter', None)
        if shock_iter is not None:
            return int(shock_iter)

        env = getattr(sim, 'env', None)
        if env is None:
            return None
        stress_start = getattr(env, 'stress_start', None)
        return int(stress_start) if stress_start is not None else None

    @staticmethod
    def _funding_liquidity_propagation(logger) -> float:
        """Magnitude of funding to spread comovement. Diagnostic only.

        No published statistic measures this. The field estimates are monthly
        and observational, while this is a tick frequency correlation inside a
        scenario built to shock funding, so only the number is reported.
        """
        return abs(float(logger.series_correlation(logger.c_series, logger.clob_qspr)))

    @staticmethod
    def _funding_spread_comovement_sign(logger) -> float:
        """Direction of the funding channel: +1 if tighter funding widens spreads.

        Banti and Phylaktis show lower repo availability raising FX transaction
        costs, and Mancini, Ranaldo and Wrampelmeyer show a higher TED spread
        and a higher VIX going with lower FX liquidity, so the direction is well
        established in the field.

        This was a gate until its power was measured. It is not one now, because
        the model cannot fail it. At mm_alpha2 of zero the measured correlation
        is +0.80, and at -200, which is ten times the calibrated magnitude with
        the opposite sign, it is +0.18 and still passing. Funding stress in this
        scenario arrives alongside higher volatility and a cancellation wave,
        and those widen the spread by themselves, so the sign of this
        correlation says nothing about the funding coefficient. Reported as a
        diagnostic.
        """
        corr = float(logger.series_correlation(logger.c_series, logger.clob_qspr))
        if not math.isfinite(corr) or abs(corr) < 1e-9:
            return 0.0
        return 1.0 if corr > 0 else -1.0

    @staticmethod
    def _amm_volume_share(logger) -> float:
        """Successful routed-customer AMM volume divided by all such volume.

        A previous implementation averaged per-tick shares after assigning a
        zero to every second without a trade.  With sparse EBS-calibrated
        arrivals that measured AMM-using seconds, not executed-volume share.
        """
        return float(logger.amm_customer_volume_share())

    @staticmethod
    def _amm_customer_trade_share(logger) -> float:
        return float(logger.amm_customer_trade_share())

    @staticmethod
    def _amm_active_tick_share(logger) -> float:
        return float(logger.amm_active_tick_flow_share())

    @staticmethod
    def _amm_arbitrage_volume(logger) -> float:
        return float(logger.arbitrage_volume_total())

    @staticmethod
    def _amm_arbitrage_share(logger) -> float:
        return float(logger.arbitrage_share_of_amm_execution())

    @staticmethod
    def _lifecycle_rows(sim, owner_type: str) -> list[dict]:
        """Collect one deduplicated owner class, including live exposures."""
        owners = []
        seen = set()
        candidates = [getattr(sim, 'mm', None)] + list(getattr(sim, 'book_agents', []) or [])
        for trader in candidates:
            if trader is None or id(trader) in seen:
                continue
            seen.add(id(trader))
            if getattr(trader, 'type', None) == owner_type:
                owners.append(trader)

        rows: list[dict] = []
        for owner in owners:
            observations = getattr(owner, 'lifecycle_observations', None)
            if callable(observations):
                rows.extend(observations(include_live=True))
                continue
            # A legacy object remains measurable but is meant to fail the
            # reason-coverage gate instead of masquerading as full telemetry.
            rows.extend({
                'lifetime': float(age),
                'censored': False,
                'reason': 'legacy_unclassified',
                'ended_tick': None,
            } for age in getattr(owner, 'completed_order_lifetimes', []))
        return rows

    @staticmethod
    def _two_sided_book_rate(logger) -> float:
        valid = []
        for row in getattr(logger, 'clob_depth', []):
            if not isinstance(row, dict):
                continue
            try:
                bid = float(row.get('bid'))
                ask = float(row.get('ask'))
            except (TypeError, ValueError):
                continue
            if math.isfinite(bid) and math.isfinite(ask):
                valid.append(bid > 1e-12 and ask > 1e-12)
        return sum(valid) / len(valid) if valid else float('nan')

    @staticmethod
    def _dealer_maker_volume_share(logger, start=None, stop=None) -> float:
        """Bank-dealer share of executed passive CLOB volume.

        Both targets on this observable come from one table, and that table
        reports two days: the period before the event and the event day. The
        model's counterpart of that division is the repricing, so the episode
        run supplies both figures, one from each side of it. Left whole-run,
        the episode's own figure averaged the two together and a gate written
        on the event day was answered mostly by the days before it.

        The response window is not used here, although every other crisis
        observable takes it. Those observables describe the dislocation and
        the window is what the dislocation is; this one is a day aggregate in
        the source and has no row in that table to match a hundred and fifty
        seconds of it. It is reported on the response window as well, and the
        two are far apart: over the dislocation the model's non-bank provider
        supplies almost nothing, and over the day it supplies rather more than
        the market did.
        """
        by_owner = getattr(logger, 'clob_maker_volume', {}) or {}

        def window(series):
            values = list(series or ())
            if start is None:
                return values
            return values[int(start):None if stop is None else int(stop)]

        total = sum(sum(float(v) for v in window(series))
                    for series in by_owner.values())
        dealer = sum(float(v) for v in window(by_owner.get('Market Maker', [])))
        return dealer / total if total > 0.0 else float('nan')

    @staticmethod
    def _executed_volume_per_second(logger) -> float:
        """Executed model base units per one-second iteration, all venues."""
        n = max(1, len(getattr(logger, 'iterations', [])))
        total = sum(
            sum(float(v) for v in series)
            for series in getattr(logger, 'flow_volume', {}).values()
        )
        return total / n

    @staticmethod
    def _hfmm_basis_series(sim) -> list[float]:
        """Time series of |HFMM mid - CLOB mid| in bps.

        HFMM is the economically relevant automated venue per the paper
        (§3.2); CPMM is a benchmark protocol whose basis is naturally
        wider because of higher curvature. Conflating the two via a
        max-of-pools statistic distorts the FX-side reading.
        """
        logger = getattr(sim, 'logger', None)
        if logger is None or 'hfmm' not in logger.amm_mid_series:
            return []
        clob_series = logger.clob_mid_series
        hfmm_series = logger.amm_mid_series['hfmm']
        out: list[float] = []
        for clob_mid, pool_mid in zip(clob_series, hfmm_series):
            if not (math.isfinite(clob_mid) and math.isfinite(pool_mid) and clob_mid > 0):
                continue
            out.append(abs(10_000.0 * (pool_mid - clob_mid) / clob_mid))
        return out

    @staticmethod
    def _parameter_sanity(sim) -> float:
        pools = getattr(sim, 'amm_pools', {}) or {}
        if not pools:
            return 1.0

        checks: list[bool] = []
        # Use median HFMM basis (matches paper claim of "average CLOB-HFMM
        # basis") and not the time-mean of the max across pools, which
        # is sensitive to outliers and conflates CPMM with HFMM.
        hfmm_basis = CalibrationFitter._hfmm_basis_series(sim)
        median_hfmm_basis = _finite_median(hfmm_basis) if hfmm_basis else float('nan')
        if math.isfinite(median_hfmm_basis):
            checks.append(median_hfmm_basis <= 25.0)

        for pool in pools.values():
            # The reserve a run is configured with, and not the reserve it
            # ends on. A pool the model allows to be drawn down on one side
            # reports a terminal reserve near zero for a reason the design
            # intends, which read as a parameter failure on every seed and
            # made this diagnostic say nothing about the parameters.
            opening = getattr(pool, 'x_history', None)
            reserve = float(opening[0]) if opening else float(pool.x)
            checks.append(100.0 <= reserve <= 50_000.0)
            # One band per curve. The two were applied cumulatively, so a
            # hybrid pool had to satisfy the constant product band as well as
            # its own, leaving an effective floor of five basis points that
            # the hybrid band was written to relax.
            if hasattr(pool, 'A'):
                checks.append(2.0 <= float(pool.A) <= 100.0)
                checks.append(0.0001 <= float(pool.fee) <= 0.005)
            else:
                checks.append(0.0005 <= float(pool.fee) <= 0.01)

        return 1.0 if checks and all(checks) else 0.0

    def realized_metrics(self, sim) -> dict[str, float]:
        logger = sim.logger
        dealer_lifecycle = _lifecycle_summary(
            self._lifecycle_rows(sim, 'Market Maker')
        )
        nonbank_lifecycle = _lifecycle_summary(
            self._lifecycle_rows(sim, 'FastRecyclerLP')
        )
        shock_iter = self._shock_iter(sim)
        clob_flow_series = list(
            getattr(logger, 'clob_flow_imbalance_series', logger.flow_imbalance_series)
        )
        flow_diagnostics = _signed_flow_diagnostics(clob_flow_series)

        impact_trade_size = 10.0
        impact_series = logger.clob_impact_curves.get(
            _nearest_trade_size(list(logger.clob_impact_curves.keys()), impact_trade_size),
            [],
        )

        recovery_half_life = float('nan')
        dealer_withdrawal_share = float('nan')
        dealer_forced_pause_share = float('nan')
        dealer_withdrawal_peak_share = _finite_max(
            logger.mm_channel_shares.get('endogenous', [])
        )
        dealer_full_withdrawal_ticks = _full_withdrawal_ticks(
            logger.mm_channel_shares.get('endogenous', [])
        )
        dealer_forced_pause_peak_share = _finite_max(
            logger.mm_channel_shares.get('forced_pause', [])
        )
        dealer_maker_volume_share = self._dealer_maker_volume_share(logger)
        dealer_maker_volume_share_whole_run = dealer_maker_volume_share
        dealer_maker_volume_share_pre_event = float('nan')
        dealer_maker_volume_share_response = float('nan')
        if shock_iter is not None and shock_iter < len(logger.iterations):
            qspr_baseline = baseline_level(list(logger.clob_qspr), shock_iter, lookback=50)
            qspr_dev = pct_deviation_series(list(logger.clob_qspr), qspr_baseline)
            recovery_half_life = half_life_steps_pct(qspr_dev, shock_iter, stable_window=3, horizon=120)
            # The endogenous channel, not the union. mm_state_shares
            # ['withdrawn'] counts a dealer the scenario is holding out of
            # the book together with one that left on its own reading of its
            # own position, and every scenario that produces a large number
            # here is a scenario that scripts a pause, so the union measures
            # the preset and not the mechanism.
            dealer_withdrawal_share = _window_mean(
                logger.mm_channel_shares.get('endogenous', []),
                shock_iter,
                shock_iter + DEALER_RESPONSE_WINDOW,
            )
            dealer_forced_pause_share = _window_mean(
                logger.mm_channel_shares.get('forced_pause', []),
                shock_iter,
                shock_iter + DEALER_RESPONSE_WINDOW,
            )
            dealer_withdrawal_peak_share = _finite_max(
                logger.mm_channel_shares.get('endogenous', [])
                [shock_iter:shock_iter + DEALER_RESPONSE_WINDOW]
            )
            dealer_full_withdrawal_ticks = _full_withdrawal_ticks(
                logger.mm_channel_shares.get('endogenous', [])
                [shock_iter:shock_iter + DEALER_RESPONSE_WINDOW]
            )
            dealer_forced_pause_peak_share = _finite_max(
                logger.mm_channel_shares.get('forced_pause', [])
                [shock_iter:shock_iter + DEALER_RESPONSE_WINDOW]
            )
            dealer_maker_volume_share = self._dealer_maker_volume_share(
                logger, shock_iter, None
            )
            dealer_maker_volume_share_pre_event = (
                self._dealer_maker_volume_share(logger, 0, shock_iter)
            )
            dealer_maker_volume_share_response = (
                self._dealer_maker_volume_share(
                    logger, shock_iter, shock_iter + DEALER_RESPONSE_WINDOW
                )
            )

        return {
            'price_discovery_half_life_ticks': _price_discovery_half_life(
                logger, shock_iter
            ),
            'dealer_forced_pause_share': dealer_forced_pause_share,
            'dealer_forced_pause_peak_share': dealer_forced_pause_peak_share,
            'dealer_withdrawal_peak_share': dealer_withdrawal_peak_share,
            # How long the sector stands fully withdrawn, and not merely
            # whether it ever does. A peak of one says the book emptied; only
            # the duration says whether it emptied for a moment or stayed
            # empty, and those are different markets.
            'dealer_full_withdrawal_ticks': dealer_full_withdrawal_ticks,
            # The target statistic is estimated with live end-of-window orders
            # as right-censored exposures.  The completed-only median remains
            # beside it as an audit diagnostic so the correction is visible.
            'dealer_order_lifetime_median_seconds': dealer_lifecycle['km_median'],
            'dealer_order_lifetime_completed_median_seconds': dealer_lifecycle['completed_median'],
            'dealer_lifecycle_completed_count': dealer_lifecycle['completed_count'],
            'dealer_lifecycle_live_censored_count': dealer_lifecycle['live_censored_count'],
            'dealer_lifecycle_censored_share': dealer_lifecycle['censored_share'],
            'dealer_lifecycle_max_atom_share': dealer_lifecycle[
                'max_completed_lifetime_atom_share'
            ],
            'dealer_lifecycle_uncategorized_share': dealer_lifecycle[
                'uncategorized_terminal_share'
            ],
            'dealer_lifecycle_max_same_tick_scheduled_end_share': dealer_lifecycle[
                'max_same_tick_scheduled_end_share'
            ],
            'nonbank_order_lifetime_median_seconds': nonbank_lifecycle['km_median'],
            'nonbank_order_lifetime_completed_median_seconds': nonbank_lifecycle['completed_median'],
            'nonbank_lifecycle_completed_count': nonbank_lifecycle['completed_count'],
            'nonbank_lifecycle_live_censored_count': nonbank_lifecycle['live_censored_count'],
            'nonbank_lifecycle_censored_share': nonbank_lifecycle['censored_share'],
            'nonbank_lifecycle_max_atom_share': nonbank_lifecycle[
                'max_completed_lifetime_atom_share'
            ],
            'nonbank_lifecycle_uncategorized_share': nonbank_lifecycle[
                'uncategorized_terminal_share'
            ],
            'nonbank_lifecycle_max_same_tick_scheduled_end_share': nonbank_lifecycle[
                'max_same_tick_scheduled_end_share'
            ],
            'book_two_sided_rate': self._two_sided_book_rate(logger),
            # From the repricing to the end of the run, which is the event
            # day of the source. On a run with no shock it is the whole run.
            'dealer_maker_volume_share': dealer_maker_volume_share,
            # Up to the repricing, which is the source's pre-event period.
            # The episode run carries both rows of that table, measured the
            # same way on the same seed, and the calm target is read here.
            'dealer_maker_volume_share_pre_event': dealer_maker_volume_share_pre_event,
            # The two scopes that are not gated stay beside the two that are,
            # so that the choice of window is legible in the artifact and not
            # only in the code that wrote it.
            'dealer_maker_volume_share_response_window': dealer_maker_volume_share_response,
            'dealer_maker_volume_share_whole_run': dealer_maker_volume_share_whole_run,
            'executed_volume_per_second': self._executed_volume_per_second(logger),
            'quoted_spread_bps': _finite_median(list(logger.clob_qspr)),
            # Reported alongside the median because the two are not
            # interchangeable here. The median sits on the minimum
            # representable spread, one tick, so it has little power to
            # discriminate from below, and the venue figure the band is
            # drawn from is an average across the whole day.
            'quoted_spread_mean_bps': _finite_mean(list(logger.clob_qspr)),
            'near_mid_depth_thin_side': _finite_mean(
                logger.clob_thin_side_depth_series()
            ),
            'order_flow_autocorrelation': flow_diagnostics['lag1_sign_autocorr'],
            'order_flow_mean_run_length': flow_diagnostics['mean_run_length'],
            'impact_curve': _finite_mean(impact_series),
            'recovery_half_life_ticks': recovery_half_life,
            'dealer_withdrawal_share': dealer_withdrawal_share,
            'funding_liquidity_stress_propagation': self._funding_liquidity_propagation(logger),
            'funding_spread_comovement_sign': self._funding_spread_comovement_sign(logger),
            # Median HFMM-CLOB basis and not the time-mean of the
            # max across pools, which is dominated by CPMM tail episodes.
            'cross_venue_basis_bps': _finite_median(self._hfmm_basis_series(sim)),
            'amm_volume_share': self._amm_volume_share(logger),
            'amm_customer_trade_share': self._amm_customer_trade_share(logger),
            'amm_active_tick_flow_share': self._amm_active_tick_share(logger),
            'amm_arbitrage_volume_base': self._amm_arbitrage_volume(logger),
            'amm_arbitrage_share_of_amm_execution': self._amm_arbitrage_share(logger),
            'fx_amm_parameter_sanity': self._parameter_sanity(sim),
        }

    @staticmethod
    def _band_threshold(target: dict[str, Any], reference_value: float) -> float:
        band = target.get('accepted_error_band') or {}
        if 'absolute' in band:
            return max(float(band['absolute']), 1e-9)
        if 'relative' in band:
            scale = abs(reference_value) if reference_value != 0 else 1.0
            return max(scale * float(band['relative']), 1e-9)
        return 0.0

    def evaluate_metrics(self, metrics: dict[str, float], run_label: Optional[str] = None,
                         scenario_name: Optional[str] = None) -> dict:
        report_targets = []
        evaluated_targets = 0
        passed_targets = 0
        gating_failures = 0
        objective_terms = []

        for target in self.targets:
            if not self._applies_to_scenario(target, scenario_name):
                continue
            observable = target.get('observable')
            if not observable:
                continue

            raw_realized = metrics.get(observable, float('nan'))
            try:
                realized = float(raw_realized)
            except (TypeError, ValueError):
                realized = float('nan')
            gating = bool(target.get('gating', True))
            target_range = target.get('target_range') or {}
            target_value = target.get('target_value')
            if 'low' in target_range or 'high' in target_range:
                # A source may bound one side and say nothing about the other.
                # Where a study measures a quantity the model reports on a
                # different basis, a bound can be defensible in one direction
                # while equality is not, and forcing a second bound onto it
                # would assert a precision the source does not carry.
                low = float(target_range.get('low', -math.inf))
                high = float(target_range.get('high', math.inf))
                if math.isfinite(realized):
                    reference_value = min(max(float(realized), low), high)
                elif math.isfinite(low) and math.isfinite(high):
                    reference_value = 0.5 * (low + high)
                else:
                    reference_value = low if math.isfinite(low) else high
                inside_range = math.isfinite(realized) and low <= float(realized) <= high
                target_display = (f'[{low}, {high}]' if math.isfinite(low) and math.isfinite(high)
                                  else (f'>= {low}' if math.isfinite(low) else f'<= {high}'))
            elif target_value is not None:
                reference_value = float(target_value)
                inside_range = False
                target_display = reference_value
            elif target.get('qualitative_bounds'):
                reference_value = 1.0
                inside_range = bool(math.isfinite(realized) and float(realized) >= 1.0 - 1e-9)
                target_display = 'sanity_bounds'
            else:
                reference_value = float('nan')
                inside_range = False
                target_display = 'n/a'

            entry = {
                'observable': observable,
                'units': target.get('units'),
                'source': target.get('source'),
                'confidence': target.get('confidence'),
                'match_type': target.get('match_type'),
                'evaluation_scenario': target.get('evaluation_scenario'),
                'gating': gating,
                'reported_only': bool(target.get('reported_only', False)),
                'realized_value': float(realized) if math.isfinite(realized) else float('nan'),
                'target': target_display,
                'accepted_error_band': target.get('accepted_error_band'),
                'source_excerpt': target.get('source_excerpt'),
            }

            if not math.isfinite(realized):
                entry['status'] = 'not_evaluable'
                report_targets.append(entry)
                continue

            # Some observables are useful diagnostics but have no externally
            # identified numerical counterpart in the cited evidence.  A
            # zero objective weight was not enough: the old report still
            # labelled them pass/fail against arbitrary or circular numbers,
            # which made a non-target look like failed validation.  Keep the
            # measurement visible without manufacturing an acceptance test.
            if entry['reported_only']:
                entry['status'] = 'reported_only'
                report_targets.append(entry)
                continue

            threshold = self._band_threshold(target, reference_value)
            distance = 0.0 if inside_range else abs(float(realized) - float(reference_value))
            passed = inside_range or threshold == 0.0 and distance == 0.0 or distance <= threshold

            # For a range target the tolerance is applied outside the range, so
            # the band that actually decides the verdict is wider than the one
            # the report displayed. A basis of 1.47 was passing a stated range
            # of three to twenty five because a tolerance of twelve put the
            # effective floor below zero. The effective band is reported
            # alongside the stated one so nothing can look like it cleared a
            # range it did not, and a floor that cannot be negative is not
            # allowed to become negative.
            if ('low' in target_range or 'high' in target_range) and threshold > 0:
                # A target may declare which side of its range the tolerance is
                # allowed to relax. Where the lower bound carries the meaning of
                # the target, as with a wedge that is expected to be tight but
                # not zero, letting the tolerance reach through it deletes the
                # claim the target was making.
                applies = str(target.get('tolerance_applies', 'both')).lower()
                eff_low = float(target_range.get('low', -math.inf))
                eff_high = float(target_range.get('high', math.inf))
                if applies in ('both', 'lower') and math.isfinite(eff_low):
                    eff_low -= threshold
                    if float(target_range['low']) >= 0.0:
                        eff_low = max(0.0, eff_low)
                if applies in ('both', 'upper') and math.isfinite(eff_high):
                    eff_high += threshold
                entry['effective_band'] = (
                    f'[{eff_low:g}, {eff_high:g}]'
                    if math.isfinite(eff_low) and math.isfinite(eff_high)
                    else (f'>= {eff_low:g}' if math.isfinite(eff_low) else f'<= {eff_high:g}'))
                entry['tolerance_applies'] = applies
                entry['inside_stated_range'] = bool(inside_range)
                passed = inside_range or (eff_low <= float(realized) <= eff_high)
            normalized_error = distance / threshold if threshold > 0 else 0.0

            entry['status'] = 'pass' if passed else 'fail'
            entry['distance_to_target'] = distance
            entry['normalized_error'] = normalized_error
            evaluated_targets += 1
            if passed:
                passed_targets += 1
            elif gating:
                gating_failures += 1
            weight = max(0.0, float(target.get('objective_weight', 1.0)))
            if weight > 0.0:
                objective_terms.append(weight * normalized_error ** 2)
            report_targets.append(entry)

        objective_score = sum(objective_terms) / len(objective_terms) if objective_terms else float('nan')
        summary = {
            'run_label': run_label,
            'scenario_name': scenario_name,
            'evaluated_targets': evaluated_targets,
            'passed_targets': passed_targets,
            'failed_targets': max(0, evaluated_targets - passed_targets),
            'gating_failures': gating_failures,
            'objective_score': objective_score,
            'status': 'pass' if evaluated_targets > 0 and gating_failures == 0 else 'fail',
        }
        return {'summary': summary, 'targets': report_targets}

    def evaluate_simulation(self, sim, run_label: Optional[str] = None,
                            scenario_name: Optional[str] = None) -> dict:
        metrics = self.realized_metrics(sim)
        report = self.evaluate_metrics(metrics, run_label=run_label, scenario_name=scenario_name)
        report['realized_metrics'] = metrics
        return report

    @staticmethod
    def _attach_crisis_multiples(suite_metrics: dict[str, dict[str, float]]) -> None:
        """Express a stressed reading against the calm one beside it.

        A source that reports a spread in both states pins their ratio more
        firmly than either level, because the level carries the conversion
        between whatever the study measured and what the model reports while
        the ratio cancels it. Where a study gives both, the ratio is the
        honest target and the level is a weaker one.
        """
        calm = suite_metrics.get('baseline_primary') or {}
        base = calm.get('quoted_spread_mean_bps', float('nan'))
        if not (isinstance(base, (int, float)) and math.isfinite(base) and base > 0.0):
            return
        for name, block in suite_metrics.items():
            if name == 'baseline_primary' or not isinstance(block, dict):
                continue
            here = block.get('quoted_spread_mean_bps', float('nan'))
            if isinstance(here, (int, float)) and math.isfinite(here):
                block['crisis_spread_multiple'] = float(here) / float(base)

    def evaluate_scenario_suite(self, suite_metrics: dict[str, dict[str, float]],
                                run_label: Optional[str] = None) -> dict:
        self._attach_crisis_multiples(suite_metrics)
        report_targets = []
        evaluated_targets = 0
        passed_targets = 0
        gating_failures = 0
        objective_terms = []

        for target in self.targets:
            scenario_name = target.get('evaluation_scenario')
            metrics = suite_metrics.get(scenario_name or 'baseline_primary')
            if metrics is None:
                continue

            observable = target.get('observable')
            if not observable:
                continue

            single = self.evaluate_metrics(
                {observable: metrics.get(observable, float('nan'))},
                run_label=run_label,
                scenario_name=scenario_name,
            )
            for item in single.get('targets', []):
                if item.get('observable') != observable:
                    continue
                report_targets.append(item)
                if item.get('status') in ('not_evaluable', 'reported_only'):
                    break
                evaluated_targets += 1
                if item.get('status') == 'pass':
                    passed_targets += 1
                elif item.get('gating', True):
                    gating_failures += 1
                weight = max(0.0, float(target.get('objective_weight', 1.0)))
                if weight > 0.0:
                    objective_terms.append(weight * item.get('normalized_error', 0.0) ** 2)
                break

        objective_score = sum(objective_terms) / len(objective_terms) if objective_terms else float('nan')
        return {
            'summary': {
                'run_label': run_label,
                'scenario_name': 'suite',
                'evaluated_targets': evaluated_targets,
                'passed_targets': passed_targets,
                'failed_targets': max(0, evaluated_targets - passed_targets),
                'gating_failures': gating_failures,
                'objective_score': objective_score,
                'status': 'pass' if evaluated_targets > 0 and gating_failures == 0 else 'fail',
            },
            'targets': report_targets,
            'scenario_metrics': suite_metrics,
        }


def build_acceptance_report(sim, target_payload: Optional[dict] = None,
                            run_label: Optional[str] = None,
                            scenario_name: Optional[str] = None) -> dict:
    fitter = CalibrationFitter(target_payload)
    return fitter.evaluate_simulation(sim, run_label=run_label, scenario_name=scenario_name)
