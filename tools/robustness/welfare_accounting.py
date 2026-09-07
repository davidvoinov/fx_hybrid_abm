#!/usr/bin/env python3
"""Auditable LP economics and partial-equilibrium welfare accounting.

The model does not contain a utility or value-of-trade schedule.  It therefore
cannot identify total welfare or the value of requests that do not execute.
This module reports, by design, only the narrower quantities the model *does*
identify, and keeps transfers separate from resource costs:

* fees move value from takers to liquidity providers;
* a subsidy moves quote currency from the sponsor to providers;
* LVR moves value from providers to informed traders/arbitrageurs;
* the outside option of committed capital is a real opportunity cost;
* a matched-notional execution-cost reduction, measured only on successful
  routed-customer fills from actual all-in execution prices against the common
  pre-trade CLOB mid, is a user-side benefit proxy, not a complete
  social-surplus estimate;
* AMM arbitrage execution is reported separately.  It is not customer demand,
  and neither its volume nor an unobserved arbitrage surplus enters the
  user-benefit proxy.

Subsidies are shown as transfers and cancel between providers and the sponsor.
Fees do enter the user's private execution cost and the LP's private income;
they may not be counted as a social gain. A paper may call the matched-cost
number a partial-equilibrium user benefit, but must not call it total welfare
without adding and identifying trader utility, dealer/counterparty incidence,
and the value of unexecuted demand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool as ProcessPool
from typing import Iterable, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import (build_parser, _apply_preset_defaults, _resolve_main_routing,
                  _auto_stress_around_shock, _seed_all, build_sim, CRISIS_PRESET)
from tools.robustness.lp_pnl_corrected import components
from tools.robustness.signatures import measurement_signature, model_signature

PRESET = CRISIS_PRESET
N_ITER = 1000
# The same window the resilience comparison uses. At a hundred periods this
# was also shorter than the fastest exit a provider can complete, since the
# least patient one needs a hundred and forty five, so a provider result read
# over it could not register a departure the episode caused.
CRISIS = (0, 150)
# Matched to the total economic capital of the AMM side, reserves plus the
# wallets its providers hold outside the pool, and to the pool's round trip
# cost weighted by the realised distribution of customer trade sizes. Both are
# read off the reserve arm before any comparison, in facility_arms.py.
# Read from the arms runner and not copied. The two had to agree on what a
# matched arm is endowed with and on the price it is matched to, and nothing
# made them; a branch that recalibrates the pair moves both.
from tools.robustness.facility_arms import ARM_CAPITAL, ARM_SPREAD_BPS  # noqa: E402
POOL_ARMS = ('reserve', 'reserve_frozen', 'reallocation')
ABSOLUTE_SIZE_BOUNDS = (5.0, 20.0)
# The calibration maps one tick to one second and its funding anchor to 252
# full FX trading days, not to every calendar second of the year.
SECONDS_PER_YEAR = 252.0 * 24.0 * 60.0 * 60.0
ESTIMAND_SCOPE = {
    'name': 'matched_routed_customer_execution_cost_partial_equilibrium',
    'customer_sample': 'successful_routed_customer_fills_only',
    'matching': 'common_notional_within_fixed_absolute_size_buckets',
    'arbitrage_treatment': 'separate_execution_telemetry_excluded_from_user_benefit',
    'unexecuted_demand_treatment': 'reported_as_unpriced_quantity_difference',
    'total_welfare_identified': False,
}

SUMMARY_METRICS = (
    'with_execution_cost_bps',
    'without_execution_cost_bps',
    'matched_notional_user_benefit',
    'matched_notional_user_benefit_bps',
    'with_amm_customer_volume_base',
    'with_arbitrage_volume_base',
    'with_amm_execution_volume_base',
    'with_arbitrage_share_of_amm_execution',
    'capital_opportunity_cost',
    'lp_operating_result',
    'lp_operating_return',
    'subsidy_paid_share_of_opening_capital',
    'rate_transfer_paid',
    'loss_rebate_paid',
    'loss_rebate_share_of_opening_capital',
    'provider_private_result_after_subsidy_and_capital',
    'sponsor_fiscal_result',
    'identified_user_benefit_less_capital_cost',
    'user_side_net_benefit',
    'dealer_withdrawal_change',
)


@dataclass(frozen=True)
class ArmWindow:
    executed_notional: float
    taker_execution_cost: float
    # ``None`` means the provider window could not be measured on this seed,
    # which is a different statement from a measured result of zero.  A pool
    # that had already wound down before the window opened used to be entered
    # as an exact zero, and a median taken over a mixture of measured results
    # and those zeros is pulled towards zero by the seeds that carry no
    # information at all.
    lp_opening_capital: Optional[float] = 0.0
    # Capital the providers hold outside the pool at the opening of the window.
    # It is theirs and it is idle, so it carries the same opportunity cost as
    # the capital inside, and an accounting that charged only the reserves
    # understated what the arrangement ties up. At the calibrated size it is
    # about two fifths again of the capital in the pool.
    lp_wallet_capital: Optional[float] = 0.0
    lp_lvr: Optional[float] = 0.0
    lp_fees: Optional[float] = 0.0
    # An order book arm holds the facility's capital in an obliged quoter, so
    # there is no pool to read reserves and rebalancing loss from. Its capital
    # is the quoter's equity at the opening of the window and its operating
    # result is what that equity did over the window. Without these the whole
    # provider side of the account was undefined for those arms, so a welfare
    # comparison could only be run against a pool.
    book_facility_capital: Optional[float] = None
    book_facility_result: Optional[float] = None
    sponsor_transfer: float = 0.0
    dealer_withdrawal_mean: float = 0.0
    sponsor_rate_transfer: float = 0.0
    sponsor_loss_rebate: float = 0.0
    execution_buckets: dict = field(default_factory=dict)
    amm_customer_volume_base: float = 0.0
    arbitrage_volume_base: float = 0.0
    arbitrage_share_of_amm_execution: Optional[float] = None

    @property
    def execution_cost_rate(self) -> float:
        if self.executed_notional <= 0.0:
            return float('nan')
        return self.taker_execution_cost / self.executed_notional

    @property
    def lp_window_measurable(self) -> bool:
        if self.book_facility_capital is not None:
            return self.book_facility_result is not None
        return None not in (self.lp_opening_capital, self.lp_lvr, self.lp_fees)

    @property
    def facility_capital(self) -> float:
        """Capital the facility ties up over the window, wherever it sits."""
        if self.book_facility_capital is not None:
            return max(0.0, float(self.book_facility_capital))
        if self.lp_opening_capital is None:
            return float('nan')
        return (max(0.0, float(self.lp_opening_capital))
                + max(0.0, float(self.lp_wallet_capital or 0.0)))

    @property
    def lp_operating_result(self) -> float:
        if not self.lp_window_measurable:
            return float('nan')
        if self.book_facility_capital is not None:
            return float(self.book_facility_result)
        return self.lp_fees - self.lp_lvr


def account_pair(with_amm: ArmWindow, without_amm: ArmWindow,
                 annual_capital_rate: float = 0.029,
                 window_seconds: float = 100.0,
                 operator_resource_cost: float = 0.0) -> dict:
    """Reconcile one paired window without treating transfers as welfare.

    Execution costs are compared on the notional common to both arms.  This
    avoids mechanically calling a high-volume arm worse merely because more
    trades execute.  The residual difference in volume is exposed but not
    priced because the model has no value-of-trade schedule.
    """
    common_notional = 0.0
    user_benefit = 0.0
    matched_buckets = {}
    bucket_names = sorted(set(with_amm.execution_buckets)
                          | set(without_amm.execution_buckets))
    for bucket in bucket_names:
        on = with_amm.execution_buckets.get(bucket, {})
        off = without_amm.execution_buckets.get(bucket, {})
        n_on = max(0.0, float(on.get('notional', 0.0)))
        n_off = max(0.0, float(off.get('notional', 0.0)))
        c_on = float(on.get('cost', 0.0))
        c_off = float(off.get('cost', 0.0))
        common = min(n_on, n_off)
        if common <= 0.0 or n_on <= 0.0 or n_off <= 0.0:
            continue
        r_on, r_off = c_on / n_on, c_off / n_off
        if not (math.isfinite(r_on) and math.isfinite(r_off)):
            continue
        benefit = common * (r_off - r_on)
        common_notional += common
        user_benefit += benefit
        matched_buckets[bucket] = {
            'common_notional': common,
            'with_cost_rate': r_on,
            'without_cost_rate': r_off,
            'user_benefit': benefit,
        }

    r_with = with_amm.execution_cost_rate
    r_without = without_amm.execution_cost_rate
    benefit_method = 'fixed_absolute_size_buckets'
    if not matched_buckets:
        benefit_method = 'aggregate_common_notional_fallback'
        common_notional = max(0.0, min(with_amm.executed_notional,
                                       without_amm.executed_notional))
        if not (math.isfinite(r_with) and math.isfinite(r_without)):
            user_benefit = float('nan')
        else:
            user_benefit = common_notional * (r_without - r_with)
    elif not math.isfinite(user_benefit):
        user_benefit = float('nan')

    # An unmeasurable provider window makes every quantity that depends on it
    # unmeasurable too.  Carrying it as a zero would put a seed that says
    # nothing into the middle of the distribution of seeds that do.
    lp_measurable = with_amm.lp_window_measurable
    if lp_measurable:
        committed = with_amm.facility_capital
        capital_cost = (committed
                        * max(0.0, float(annual_capital_rate))
                        * max(0.0, float(window_seconds)) / SECONDS_PER_YEAR)
        private_lp = (with_amm.lp_operating_result + with_amm.sponsor_transfer
                      - capital_cost)
        sponsor = -with_amm.sponsor_transfer
        provider_plus_sponsor = private_lp + sponsor
    else:
        capital_cost = float('nan')
        private_lp = float('nan')
        sponsor = float('nan')
        provider_plus_sponsor = float('nan')
    user_side_net = (user_benefit - capital_cost - operator_resource_cost
                     if math.isfinite(user_benefit) and lp_measurable
                     else float('nan'))

    opening = with_amm.facility_capital if lp_measurable else 0.0
    return {
        'estimand_scope': dict(ESTIMAND_SCOPE),
        'user_benefit_uses_routed_customer_fills_only': True,
        'arbitrage_execution_in_user_benefit': False,
        'arbitrage_surplus_identified': False,
        'arbitrage_surplus_quote': None,
        'common_executed_notional': common_notional,
        'user_benefit_method': benefit_method,
        'matched_size_buckets': matched_buckets,
        'incremental_executed_notional': (with_amm.executed_notional
                                          - without_amm.executed_notional),
        'with_routed_customer_executed_notional': with_amm.executed_notional,
        'without_routed_customer_executed_notional': without_amm.executed_notional,
        'with_amm_customer_volume_base': with_amm.amm_customer_volume_base,
        'with_arbitrage_volume_base': with_amm.arbitrage_volume_base,
        'with_amm_execution_volume_base': (
            with_amm.amm_customer_volume_base + with_amm.arbitrage_volume_base
        ),
        'with_arbitrage_share_of_amm_execution': (
            with_amm.arbitrage_share_of_amm_execution
        ),
        'with_execution_cost_rate': r_with,
        'without_execution_cost_rate': r_without,
        'with_execution_cost_bps': r_with * 10_000.0 if math.isfinite(r_with) else float('nan'),
        'without_execution_cost_bps': (r_without * 10_000.0
                                       if math.isfinite(r_without) else float('nan')),
        'matched_notional_user_benefit': user_benefit,
        'matched_notional_user_benefit_bps': (
            user_benefit / common_notional * 10_000.0
            if common_notional > 0.0 and math.isfinite(user_benefit)
            else float('nan')
        ),
        'lp_window_measurable': lp_measurable,
        'lp_lvr': with_amm.lp_lvr if lp_measurable else None,
        'lp_fee_income': with_amm.lp_fees if lp_measurable else None,
        'lp_operating_result': with_amm.lp_operating_result,
        'lp_operating_return': (with_amm.lp_operating_result / opening
                                if opening > 0.0 else float('nan')),
        'subsidy_paid_share_of_opening_capital': (
            with_amm.sponsor_transfer / opening if opening > 0.0 else float('nan')
        ),
        'rate_transfer_paid': with_amm.sponsor_rate_transfer,
        'loss_rebate_paid': with_amm.sponsor_loss_rebate,
        'loss_rebate_share_of_opening_capital': (
            with_amm.sponsor_loss_rebate / opening
            if opening > 0.0 else float('nan')
        ),
        'capital_opportunity_cost': capital_cost,
        'operator_resource_cost': float(operator_resource_cost),
        'provider_private_result_after_subsidy_and_capital': private_lp,
        'sponsor_fiscal_result': sponsor,
        'provider_plus_sponsor_result': provider_plus_sponsor,
        # ``None`` and not ``False`` when there is no measured window: an
        # identity that could not be evaluated has not been violated, and
        # counting it as a violation would report a data gap as an accounting
        # error.
        'subsidy_transfer_cancels': (
            bool(abs(
                provider_plus_sponsor
                - (with_amm.lp_operating_result - capital_cost)
            ) <= 1e-9 * max(1.0, abs(provider_plus_sponsor)))
            if lp_measurable else None
        ),
        # This combines an identified user-side monetary effect with the real
        # opportunity cost of AMM capital. It is deliberately not summed with
        # LP P&L, because unmatched notional and dealer/counterparty legs keep
        # the model from closing a social-welfare ledger.
        'identified_user_benefit_less_capital_cost': user_side_net,
        'user_side_net_benefit': user_side_net,
        'dealer_withdrawal_change': (with_amm.dealer_withdrawal_mean
                                     - without_amm.dealer_withdrawal_mean),
        'total_welfare_identified': False,
        'unidentified_terms': [
            'gross value and urgency of executed demand',
            'value of unexecuted demand',
            'value and exact composition of demand unmatched across arms',
            'counterparty incidence of spread and price-impact revenue',
            'counterparty to LVR/arbitrage gains',
            'window-specific arbitrage surplus (execution volume is observed)',
        ],
    }



def annual_frame(user_benefit_per_crisis_window: float,
                 provider_crisis_loss_fraction: float,
                 provider_calm_gain_fraction: float,
                 committed_capital: float,
                 window_seconds: float = float(CRISIS[1] - CRISIS[0]),
                 risk_free_rate: float = 0.029,
                 idle_capital_fraction: float = 0.0,
                 operator_resource_cost: float = 0.0) -> dict:
    """Put the taker gain and the provider cost on one annual footing.

    The two sides of this facility are measured in different units, over
    different populations and on different horizons, and comparing a
    percentage reduction in spread with a percentage loss of pool value is
    not meaningful. Expressing both as expected annual quantities is what
    makes them commensurable.

    With ``s`` the share of windows in the crisis state and ``N`` the number
    of windows of this length in a year,

        C(s) = N V [ s L_c - (1 - s) L_0 ] + r_f V k
        B(s) = N s SUM_Q vbar_Q dC(Q)

    Three things an earlier version of this accounting got wrong are kept
    right here. The calm result enters the cost with a minus sign, because
    it is a gain and reduces what has to be found from elsewhere. The
    volume enters the benefit once: writing a share of volume beside a
    volume weights the same quantity twice. And the opportunity cost sits
    outside the factor ``N``, so ``N`` does not cancel between the two sides
    and a comparison of levels requires it.

    Two break even shares come out of this. The private one is where the
    provider's own calm earnings cover its own crisis losses, and the social
    one is where the taker gain covers the whole cost including the
    opportunity cost of the capital. H5 is the claim that the social share
    is the lower of the two; it is returned here as a measured gap rather
    than asserted.

    What this does not settle is whether a crisis window drawn from this
    model may be scaled to a year at all. The model produces one shock per
    run and says nothing about how often such shocks arrive or how long they
    last in a market. That is an economic judgement about extrapolation and
    not a missing unit, so ``s`` is a free variable here and no value of it
    is supplied.
    """
    window = max(1e-9, float(window_seconds))
    windows_per_year = SECONDS_PER_YEAR / window
    capital = max(0.0, float(committed_capital))
    loss = float(provider_crisis_loss_fraction)
    gain = float(provider_calm_gain_fraction)
    benefit_per_window = float(user_benefit_per_crisis_window)
    idle_cost = (float(risk_free_rate) * capital
                 * max(0.0, float(idle_capital_fraction)))

    def private_cost(share: float) -> float:
        """What the owner of the capital bears. Transfers count here."""
        share = min(1.0, max(0.0, float(share)))
        return (windows_per_year * capital * (share * loss - (1.0 - share) * gain)
                + idle_cost + float(operator_resource_cost))

    def social_cost(share: float) -> float:
        """What the economy bears. Transfers do not count here.

        The provider's loss against rebalancing is not consumed by anybody:
        it is paid to whoever traded against the pool, so counting it as a
        social cost charges the economy for a payment it makes to itself. It
        was that treatment which made the facility look never worthwhile:
        against the whole provider loss the taker gain was smaller by four
        orders of magnitude, and against the resources actually consumed it
        is larger than them.

        The counterparties are overwhelmingly arbitrageurs. Measured over the
        crisis window on sixty seeds, arbitrage is a median 98 per cent of the
        volume executed against the pool, reported here as
        ``arbitrage_share_of_amm_execution``. That is a share of volume and
        not a decomposition of the loss itself: the trade log records routed
        customer fills only, so the loss cannot be split by counterparty from
        it, and no such split is claimed. An earlier version of this module
        carried a literal 0.69 as the arbitrage share of the loss with no
        measurement behind it anywhere in the repository. It has been removed
        and not carried forward, and nothing in this accounting depended
        on it, since the whole of the loss is a transfer either way.

        What society does give up is the return the committed capital would
        have earned elsewhere, plus whatever it costs to run the thing.
        """
        del share
        return idle_cost + float(operator_resource_cost)

    cost = private_cost

    def benefit(share: float) -> float:
        share = min(1.0, max(0.0, float(share)))
        return windows_per_year * share * benefit_per_window

    # Private break even: the provider's own calm earnings cover its own
    # crisis losses. Independent of N, which cancels.
    denominator = loss + gain
    private = gain / denominator if denominator > 0.0 else float('nan')

    # Social break even: B(s) = C(s). Both sides are linear in s, so this
    # solves in closed form; a non-positive slope difference means the two
    # never meet and the facility is either always or never worthwhile.
    # B(s) = C_social(s). The social cost does not vary with s, so this is
    # simply the share at which the benefit covers the resources consumed.
    social_slope = windows_per_year * benefit_per_window
    social = (social_cost(0.0) / social_slope
              if social_slope > 1e-18 else float('nan'))
    # A root outside the unit interval is not a missing number, it is the
    # statement that the two lines do not cross among admissible crisis
    # shares. Distinguishing that from an unevaluated quantity matters,
    # because one is a result and the other is a gap in the measurement.
    verdict = 'crosses_within_unit_interval'
    if not (social == social):
        verdict = 'no_benefit_measured'
    elif not (0.0 <= social <= 1.0):
        verdict = 'never_worthwhile_to_society'
        social = float('nan')

    return {
        'windows_per_year': windows_per_year,
        'window_seconds': window,
        'committed_capital': capital,
        'user_benefit_per_crisis_window': benefit_per_window,
        'provider_crisis_loss_fraction': loss,
        'provider_calm_gain_fraction': gain,
        'idle_capital_cost_per_year': idle_cost,
        'private_break_even_share': private,
        'social_break_even_share': social,
        'social_is_lower_than_private': (
            bool(social < private) if (social == social and private == private)
            else None
        ),
        'break_even_gap': (
            float(private - social) if (social == social and private == private)
            else float('nan')
        ),
        'private_cost_at': {str(x): private_cost(x) for x in (0.001, 0.01, 0.05)},
        'social_cost_per_year': social_cost(0.0),
        'provider_loss_is_a_transfer': True,
        'benefit_at': {str(x): benefit(x) for x in (0.001, 0.01, 0.05)},
        'social_break_even_verdict': verdict,
        'crisis_share_is_not_identified_by_this_model': True,
    }


def _window_bounds(shock: int, window: tuple[int, int], n: int) -> tuple[int, int]:
    lo = max(0, shock + int(window[0]))
    hi = min(n, shock + int(window[1]))
    return lo, max(lo, hi)


def _absolute_size_bucket(quantity: float) -> str:
    if quantity <= ABSOLUTE_SIZE_BOUNDS[0]:
        return 'q_le_5'
    if quantity <= ABSOLUTE_SIZE_BOUNDS[1]:
        return 'q_5_to_20'
    return 'q_gt_20'


def _common_reference_cost_bps(trade: dict, fallback_reference: float) -> float:
    """Signed taker cost from an actual all-in fill and one common benchmark.

    Positive is costly to the taker; negative is price improvement. Legacy
    synthetic fixtures without an execution price retain their explicit cost
    field, but every newly simulated trade carries ``all_in_exec_price``.
    """
    try:
        execution_price = float(trade['all_in_exec_price'])
        reference = float(trade.get('common_reference_price', fallback_reference))
    except (KeyError, TypeError, ValueError):
        return float(trade.get('cost_bps', float('nan')))
    if not (math.isfinite(execution_price) and math.isfinite(reference)
            and execution_price > 0.0 and reference > 0.0):
        return float('nan')
    side = str(trade.get('side', '')).lower()
    if side == 'buy':
        return 10_000.0 * (execution_price - reference) / reference
    if side == 'sell':
        return 10_000.0 * (reference - execution_price) / reference
    return float('nan')


def execution_window(sim, shock: int,
                     window: tuple[int, int] = CRISIS) -> tuple[float, float, dict]:
    """Routed-customer notional and all-in cost in quote currency.

    ``MetricsLogger.trade_log`` is contractually the successful routed-customer
    log; arbitrage legs live in ``arbitrage_volume``.  The explicit source
    guard also prevents a future combined log (or a synthetic fixture) from
    silently broadening this estimand.  Legacy customer records predate the
    source tag and remain admissible unless they identify an arbitrageur.
    """
    prices = list(sim.logger.fair_price_series)
    lo, hi = _window_bounds(shock, window, len(prices))
    notional = 0.0
    cost = 0.0
    buckets = {}
    for trade in sim.logger.trade_log:
        source = trade.get('execution_source')
        if source is not None and str(source) != 'routed_customer':
            continue
        if str(trade.get('trader_type', '')).lower() == 'ammarbitrageur':
            continue
        t = int(trade.get('t', -1))
        if t < lo or t >= hi or t >= len(prices):
            continue
        q = max(0.0, float(trade.get('quantity', 0.0)))
        fallback_price = float(prices[t])
        p = float(trade.get('common_reference_price', fallback_price))
        bps = _common_reference_cost_bps(trade, fallback_price)
        if not (math.isfinite(q) and math.isfinite(p) and math.isfinite(bps)):
            continue
        trade_notional = q * p
        notional += trade_notional
        trade_cost = trade_notional * bps / 10_000.0
        cost += trade_cost
        bucket = _absolute_size_bucket(q)
        row = buckets.setdefault(bucket, {'notional': 0.0, 'cost': 0.0})
        row['notional'] += trade_notional
        row['cost'] += trade_cost
    return notional, cost, buckets


def execution_composition_window(sim, shock: int,
                                 window: tuple[int, int] = CRISIS
                                 ) -> tuple[float, float, Optional[float]]:
    """Customer and arbitrage AMM-leg base volume, kept as distinct flows."""
    prices = list(sim.logger.fair_price_series)
    lo, hi = _window_bounds(shock, window, len(prices))
    customer = sum(
        sum(float(value) for value in series[lo:hi])
        for venue, series in getattr(sim.logger, 'flow_volume', {}).items()
        if venue != 'clob'
    )
    if hasattr(sim.logger, 'arbitrage_volume_total'):
        arbitrage = float(sim.logger.arbitrage_volume_total(start=lo, end=hi))
    else:
        arbitrage = sum(
            sum(float(value) for value in series[lo:hi])
            for series in getattr(sim.logger, 'arbitrage_volume', {}).values()
        )
    customer = max(0.0, float(customer))
    arbitrage = max(0.0, float(arbitrage))
    total = customer + arbitrage
    share = arbitrage / total if total > 0.0 else None
    return customer, arbitrage, share


def lp_window(sim, shock: int,
              window: tuple[int, int] = CRISIS
              ) -> Optional[tuple[float, float, float]]:
    """Opening capital, realised LVR and native fees across all pools.

    ``None`` when a pool is present but its window cannot be measured, which
    happens when the pool held nothing at the opening of the window or the
    window runs past the end of the run.  Skipping such a pool and returning
    the sum over the rest reported an unmeasurable seed as a measured zero,
    and those zeros then entered the medians.  An arm with no facility at all
    is a different case: its provider capital is measured, and it is zero.
    """
    price = np.asarray(sim.logger.fair_price_series, dtype=float)
    opening = lvr = fees = 0.0
    for pool in sim.amm_pools.values():
        row = components(pool, price, shock, window)
        if row is None:
            return None
        loss_i, fees_i, opening_i = row
        lvr += loss_i
        fees += fees_i
        opening += opening_i
    return float(opening), float(lvr), float(fees)


def sponsor_window(sim, shock: int,
                   window: tuple[int, int] = CRISIS) -> tuple[float, float, float]:
    total = rate = rebate = 0.0
    for population in getattr(sim, 'lp_providers', []) or []:
        history = getattr(population, 'history', {})
        series = list(history.get('subsidy', []))
        lo, hi = _window_bounds(shock, window, len(series))
        total += sum(float(value) for value in series[lo:hi]
                     if math.isfinite(float(value)))
        rate_series = list(history.get('rate_subsidy', []))
        r_lo, r_hi = _window_bounds(shock, window, len(rate_series))
        rate += sum(float(value) for value in rate_series[r_lo:r_hi]
                    if math.isfinite(float(value)))
        rebate_series = list(history.get('loss_rebate', []))
        b_lo, b_hi = _window_bounds(shock, window, len(rebate_series))
        rebate += sum(float(value) for value in rebate_series[b_lo:b_hi]
                      if math.isfinite(float(value)))
    return total, rate, rebate


def withdrawal_window(sim, shock: int,
                      window: tuple[int, int] = CRISIS) -> float:
    series = list(sim.logger.mm_channel_shares.get('endogenous', []))
    lo, hi = _window_bounds(shock, window, len(series))
    finite = [float(value) for value in series[lo:hi]
              if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def _staged_window(sim, window: tuple[int, int]) -> Optional[dict]:
    """What the staging captured at the opening of this window.

    ``None`` when the run staged nothing at all, which is how a simulation
    built by hand in a test reaches these readers. A run that staged some
    windows and not the one asked for is a different case and raises, since
    answering it from another window would mark a facility at the wrong
    instant and report the number as if it were the right one.
    """
    staged = getattr(sim, 'facility_windows', None)
    if staged is None:
        return getattr(sim, 'facility_window', None)
    key = (int(window[0]), int(window[1]))
    if key not in staged:
        raise KeyError(f'the run staged no facility reading for window {key}')
    return staged[key]


def lp_wallet_window(sim, shock: int,
                     window: tuple[int, int] = CRISIS) -> Optional[float]:
    """Value of the provider wallets at the opening of the window.

    Read at the same instant as the reserves, and in the same currency, so the
    two can be added into one capital base.
    """
    captured = _staged_window(sim, window)
    if captured is not None:
        return captured.get('opening_wallets')
    providers = [lp for pop in getattr(sim, 'lp_providers', []) or []
                 for lp in getattr(pop, 'providers', [])]
    return 0.0 if not providers else None


def arm_window(sim, shock: int,
               window: tuple[int, int] = CRISIS) -> ArmWindow:
    notional, cost, buckets = execution_window(sim, shock, window)
    customer_amm, arbitrage, arbitrage_share = execution_composition_window(
        sim, shock, window
    )
    lp = lp_window(sim, shock, window)
    opening, lvr, fees = lp if lp is not None else (None, None, None)
    wallets = lp_wallet_window(sim, shock, window) if lp is not None else None
    captured = _staged_window(sim, window) or {}
    book_open = captured.get('opening_capital')
    book_result = captured.get('operating_result')
    sponsor, rate, rebate = sponsor_window(sim, shock, window)
    return ArmWindow(
        executed_notional=notional,
        taker_execution_cost=cost,
        lp_opening_capital=opening,
        lp_wallet_capital=wallets,
        book_facility_capital=book_open,
        book_facility_result=book_result,
        lp_lvr=lvr,
        lp_fees=fees,
        sponsor_transfer=sponsor,
        dealer_withdrawal_mean=withdrawal_window(sim, shock, window),
        sponsor_rate_transfer=rate,
        sponsor_loss_rebate=rebate,
        execution_buckets=buckets,
        amm_customer_volume_base=customer_amm,
        arbitrage_volume_base=arbitrage,
        arbitrage_share_of_amm_execution=arbitrage_share,
    )


def run(seed: int, arm, lp_model: str = 'endogenous',
        subsidy_rate: float = 0.0, loss_rebate_fraction: float = 0.0,
        response_scale: Optional[float] = None,
        outside_option: Optional[float] = None,
        windows: tuple[tuple[int, int], ...] = (CRISIS,)):
    """One arm of the comparison on one seed.

    ``arm`` names the facility, so that the welfare account carries the same
    decomposition the resilience comparison does. Read as a flag it collapsed
    to the presence of a pool, and every quantity below then attributed to the
    facility what its capital, its obligation to quote and its schedule had
    produced between them. A bare boolean is still accepted and means the
    reserve priced pool or the dealer only control.
    """
    if isinstance(arm, bool):
        arm = 'reserve' if arm else 'none'
    arm = str(arm)
    argv = ['--preset', PRESET, '--seed', str(seed), '--n-iter', str(N_ITER),
            '--silent', '--amm-lp-model', lp_model,
            '--facility-arm', arm, '--arm-spread-bps', repr(ARM_SPREAD_BPS),
            '--amm-lp-subsidy-rate', repr(float(subsidy_rate)),
            '--amm-lp-loss-rebate', repr(float(loss_rebate_fraction))]
    if arm in ('dealer_of_last_resort', 'passive_book'):
        argv.extend(['--arm-capital', repr(ARM_CAPITAL)])
    # The return a provider can earn away from the pool. It is zero on this
    # calibration because both policy legs were at or below zero, and that
    # removes the opportunity-cost channel from the participation margin
    # instead of merely shrinking it. The override exists so the size of what
    # was removed can be measured and not merely asserted.
    if outside_option is not None:
        argv.extend(['--amm-lp-outside-option', repr(float(outside_option))])
    if response_scale is not None:
        argv.extend(['--amm-lp-response-scale', repr(float(response_scale))])
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_preset_defaults(parser, args)
    args.venue_choice_rule = _resolve_main_routing(args, argv)
    _auto_stress_around_shock(args)
    carries_pool = arm in POOL_ARMS
    args.enable_amm = 1 if carries_pool else 0
    if not carries_pool:
        args.amm_share_pct = 0.0
    args.clob_amm_interaction = 'competition' if carries_pool else 'none'
    _seed_all(seed)
    sim = build_sim(args)
    shock = int(args.shock_iter)
    facility = next((t for t in sim.traders
                     if getattr(t, 'is_facility_arm', False)), None)
    providers = [lp for pop in getattr(sim, 'lp_providers', []) or []
                 for lp in getattr(pop, 'providers', [])]

    def _fair():
        series = getattr(sim.logger, 'fair_price_series', []) or []
        return float(series[-1]) if series else float('nan')

    # Staged, because both the wallets a provider holds and the position a
    # quoter carries are attributes of the agent at the current tick and not
    # series. Read after a run taken in one step they report the end of the
    # year: the wallets came back four times their size at the opening of the
    # window, and the capital cost with them.
    #
    # Windows are staged in one pass and in order. A calm window and a crisis
    # window read off the same path are two states of one market, which is
    # what setting them beside each other asserts. Read off two runs they were
    # two markets, and the order book arms had no calm reading at all, since
    # the staging here is the only place their equity is marked.
    captured = {}
    tick = 0
    for window in sorted(windows, key=lambda w: int(w[0])):
        opening_tick = max(0, shock + int(window[0]))
        span = max(0, int(window[1]) - int(window[0]))
        if opening_tick > tick:
            sim.simulate(opening_tick - tick, silent=True)
            tick = opening_tick
        elif opening_tick < tick:
            raise ValueError('windows must not overlap and must be ordered')
        opening_price = _fair()
        opening_wallets = sum(float(lp.wallet_base) * opening_price
                              + float(lp.wallet_cash) for lp in providers)
        opening_capital = (float(facility.cash)
                           + float(facility.assets) * opening_price
                           if facility is not None else None)

        # The quoter's result is accumulated the way the pool's is, period by
        # period against a hold and rebalance benchmark, so that the two arms
        # are measured on one construction. Marked instead at the price the
        # market opened the year on, a position carried through a one per cent
        # decline showed no loss at all and both order book arms reported a
        # profit.
        result = 0.0
        if facility is not None:
            cash, assets = float(facility.cash), float(facility.assets)
            for _ in range(span):
                sim.simulate(1, silent=True)
                price = _fair()
                new_cash = float(facility.cash)
                new_assets = float(facility.assets)
                if math.isfinite(price):
                    result += (new_cash - cash) + (new_assets - assets) * price
                cash, assets = new_cash, new_assets
        else:
            sim.simulate(span, silent=True)
        tick += span
        captured[(int(window[0]), int(window[1]))] = {
            'opening_capital': opening_capital,
            'operating_result': result if facility is not None else None,
            'opening_wallets': opening_wallets if providers else 0.0,
        }

    remaining = int(args.n_iter) - tick
    if remaining > 0:
        sim.simulate(remaining, silent=True)
    sim.facility_windows = captured
    sim.facility_window = captured.get((int(CRISIS[0]), int(CRISIS[1])))
    return sim, shock


def measure(seed: int, annual_capital_rate: float = 0.029,
            lp_model: str = 'endogenous', subsidy_rate: float = 0.0,
            loss_rebate_fraction: float = 0.0,
            response_scale: Optional[float] = None,
            arm: str = 'reserve',
            outside_option: Optional[float] = None) -> dict:
    """One arm against the dealer only control, on one seed.

    The control is the same market in both, so the difference is the arm.
    """
    without, shock0 = run(seed, 'none', lp_model, subsidy_rate,
                          loss_rebate_fraction, response_scale, outside_option)
    with_arm, shock1 = run(seed, arm, lp_model, subsidy_rate,
                           loss_rebate_fraction, response_scale, outside_option)
    off = arm_window(without, shock0)
    on = arm_window(with_arm, shock1)
    result = account_pair(on, off, annual_capital_rate=annual_capital_rate,
                          window_seconds=float(CRISIS[1] - CRISIS[0]))
    return {'seed': int(seed), 'arm': arm, 'with_amm': asdict(on),
            'without_amm': asdict(off), 'accounting': result}


def aggregate(rows: Iterable[dict], bootstrap_draws: int = 5000,
              bootstrap_seed: int = 0) -> dict:
    rows = list(rows)
    # A seed whose provider window could not be measured has no identity to
    # check, so it is counted apart and not as a failure of the identity.
    transfer_failures = sum(
        row.get('accounting', {}).get('subsidy_transfer_cancels') is False
        for row in rows
    )
    unmeasurable_lp_windows = sum(
        not bool(row.get('accounting', {}).get('lp_window_measurable', True))
        for row in rows
    )
    medians = {}
    intervals = {}
    finite_counts = {}
    rng = np.random.default_rng(int(bootstrap_seed))
    for key in SUMMARY_METRICS:
        values = []
        for row in rows:
            raw = row['accounting'].get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        values.sort()
        finite_counts[key] = len(values)
        medians[key] = float(np.median(values)) if values else None
        if values and bootstrap_draws > 0:
            array = np.asarray(values, dtype=float)
            draws = rng.choice(array, size=(int(bootstrap_draws), array.size),
                               replace=True)
            boot = np.median(draws, axis=1)
            intervals[key] = [float(np.quantile(boot, 0.025)),
                              float(np.quantile(boot, 0.975))]
        else:
            intervals[key] = None
    return {
        'n_seeds': len(rows),
        'medians': medians,
        'finite_counts': finite_counts,
        'bootstrap_95_intervals_on_median': intervals,
        'bootstrap_draws': int(bootstrap_draws),
        'bootstrap_seed': int(bootstrap_seed),
        'subsidy_transfer_cancellation_failures': transfer_failures,
        'all_subsidy_transfers_cancel': bool(rows and transfer_failures == 0),
        # Reported and not absorbed: a median over fewer seeds than the
        # run carries is a different claim from a median over all of them.
        'unmeasurable_lp_windows': unmeasurable_lp_windows,
        'total_welfare_identified': False,
        'estimand_scope': dict(ESTIMAND_SCOPE),
        'interpretation': (
            'Partial-equilibrium execution-cost benefit on matched routed-customer '
            'fills and LP/fiscal incidence. Arbitrage execution is telemetry only; '
            'this is not total social welfare.'
        ),
    }


_WELFARE_SIGNATURE = None
_WELFARE_REPORT_SIGNATURE = None


def welfare_measurement_signature():
    """Digest of paired seed simulation and row-level accounting only."""
    global _WELFARE_SIGNATURE
    if _WELFARE_SIGNATURE is None:
        _WELFARE_SIGNATURE = measurement_signature(
            'welfare_accounting',
            model_signature(ROOT),
            functions=(
                _common_reference_cost_bps,
                _staged_window,
                run,
                execution_window,
                execution_composition_window,
                lp_window,
                sponsor_window,
                withdrawal_window,
                arm_window,
                account_pair,
                measure,
            ),
            constants=(PRESET, N_ITER, CRISIS, ABSOLUTE_SIZE_BOUNDS,
                       SECONDS_PER_YEAR, tuple(sorted(ESTIMAND_SCOPE.items()))),
        )
    return _WELFARE_SIGNATURE


def welfare_report_signature():
    """Digest of the summary applied to current-signature welfare rows."""
    global _WELFARE_REPORT_SIGNATURE
    if _WELFARE_REPORT_SIGNATURE is None:
        _WELFARE_REPORT_SIGNATURE = measurement_signature(
            'welfare_accounting_report',
            welfare_measurement_signature(),
            functions=(aggregate,),
            constants=('welfare-partial-equilibrium-summary-v1',),
        )
    return _WELFARE_REPORT_SIGNATURE


def self_check() -> bool:
    # Named, so that inserting a field cannot silently re-map the values. The
    # positional form survived one such insertion by arithmetic coincidence,
    # the shifted fees and loss still differing by the same two units, and a
    # check that passes for the wrong reason is worse than one that fails.
    on = ArmWindow(executed_notional=1000.0, taker_execution_cost=4.0,
                   lp_opening_capital=2000.0, lp_lvr=7.0, lp_fees=5.0,
                   sponsor_transfer=3.0, dealer_withdrawal_mean=0.2,
                   sponsor_rate_transfer=1.0, sponsor_loss_rebate=2.0)
    off = ArmWindow(executed_notional=1000.0, taker_execution_cost=9.0,
                    dealer_withdrawal_mean=0.1)
    a = account_pair(on, off, annual_capital_rate=0.0, window_seconds=100.0)
    b = account_pair(ArmWindow(**{**asdict(on), 'sponsor_transfer': 300.0}), off,
                     annual_capital_rate=0.0, window_seconds=100.0)
    return bool(
        abs(a['matched_notional_user_benefit'] - 5.0) < 1e-12
        and abs(a['lp_operating_result'] + 2.0) < 1e-12
        and a['subsidy_transfer_cancels']
        and b['subsidy_transfer_cancels']
        and abs(a['user_side_net_benefit'] - b['user_side_net_benefit']) < 1e-12
        and not a['total_welfare_identified']
    )


def _measure_job(job):
    return measure(*job)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--capital-rate', type=float, default=0.029)
    parser.add_argument('--lp-model', choices=['rule', 'endogenous'],
                        default='endogenous')
    parser.add_argument('--subsidy-rate', type=float, default=0.0,
                        help='quote transfer per tick as a fraction of pool NAV')
    parser.add_argument('--loss-rebate', type=float, default=0.0,
                        help='fraction of a negative LP operating payoff reimbursed')
    parser.add_argument('--outside-option', type=float, default=None,
                        help='provider outside option per tick; the manifest '
                             'value is used when this is not given')
    parser.add_argument('--response-scale', type=float, default=None,
                        help='override the calibrated LP return-response scale')
    parser.add_argument('--output', default='output/resilience/welfare_accounting.json')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--arm', default='reserve',
                        choices=['reserve', 'reserve_frozen',
                                 'dealer_of_last_resort',
                                 'passive_book', 'reallocation'],
                        help='facility measured against the dealer only '
                             'control, so the welfare account carries the same '
                             'decomposition the resilience comparison does')
    args = parser.parse_args()
    if args.self_check:
        ok = self_check()
        print('welfare accounting self-check:', 'pass' if ok else 'fail')
        return 0 if ok else 1
    jobs = [(seed, args.capital_rate, args.lp_model, args.subsidy_rate,
             args.loss_rebate, args.response_scale, args.arm,
             args.outside_option)
            for seed in range(args.seed_start, args.seed_start + args.seeds)]
    if args.workers > 1:
        with ProcessPool(args.workers) as pool:
            rows = list(pool.imap(_measure_job, jobs))
    else:
        rows = [_measure_job(job) for job in jobs]
    payload = {
        'model_signature': model_signature(ROOT),
        'measurement_signature': welfare_measurement_signature(),
        'report_signature': welfare_report_signature(),
        'configuration': {**vars(args),
                          'absolute_size_bounds': list(ABSOLUTE_SIZE_BOUNDS),
                          'execution_cost_benchmark': 'pretrade_clob_mid',
                          'execution_price_basis': 'actual_all_in_fill',
                          'estimand_scope': dict(ESTIMAND_SCOPE)},
        'summary': aggregate(rows, args.bootstrap_draws, args.bootstrap_seed),
        'rows': rows,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(payload['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
