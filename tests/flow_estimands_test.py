import math
from types import SimpleNamespace

from AgentBasedModel.agents.agents import AMMArbitrageur, Trader
from AgentBasedModel.metrics.logger import MetricsLogger
from AgentBasedModel.venues.amm import CPMMPool
from calibration.fitter import CalibrationFitter
from tools.stat_study import _phase_customer_flow_estimands
from tools.robustness.migration_table import _phase_estimands


def _sparse_flow_logger():
    logger = MetricsLogger()
    logger.iterations = [0, 1, 2]
    # Active ticks have very different total sizes and AMM shares.  The quiet
    # middle tick makes the old zero-filled time mean a third, even though the
    # ratio of summed AMM to market volume is 19 / 110.
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
    return logger


def test_customer_volume_trade_and_active_tick_shares_are_distinct():
    logger = _sparse_flow_logger()

    assert math.isclose(logger.amm_customer_volume_share(), 19.0 / 110.0)
    assert math.isclose(logger.amm_active_tick_flow_share(), 0.5)
    assert math.isclose(logger.amm_customer_trade_share(), 2.0 / 7.0)
    assert math.isclose(sum(logger.flow_share('hfmm')) / 3.0, 1.0 / 3.0)
    amm_series = logger.amm_active_tick_customer_volume_share_series()
    clob_series = logger.active_tick_customer_volume_share_series('clob')
    assert math.isclose(amm_series[0], 0.9)
    assert math.isnan(amm_series[1])
    assert math.isclose(amm_series[2], 0.1)
    assert math.isclose(clob_series[0] + amm_series[0], 1.0)
    assert math.isnan(clob_series[1])
    assert math.isclose(clob_series[2] + amm_series[2], 1.0)

    summary = logger.summary()
    assert math.isclose(summary['avg_flow_share_hfmm'], 19.0 / 110.0)
    assert math.isclose(
        summary['avg_flow_share_clob'] + summary['avg_flow_share_hfmm'], 1.0
    )
    assert math.isclose(summary['active_tick_flow_share_hfmm'], 0.5)
    assert math.isclose(summary['customer_trade_share_hfmm'], 2.0 / 7.0)
    assert math.isclose(summary['zero_filled_tick_flow_share_hfmm'], 1.0 / 3.0)


def test_calibration_and_migration_use_ratio_of_summed_customer_volume():
    logger = _sparse_flow_logger()

    assert math.isclose(
        CalibrationFitter._amm_volume_share(logger),
        19.0 / 110.0,
    )
    phase = _phase_estimands(logger, 0, 3)
    assert math.isclose(phase['share'], 19.0 / 110.0)
    assert math.isclose(phase['active_tick_share'], 0.5)
    assert math.isclose(phase['customer_trade_share'], 2.0 / 7.0)

    stat_phase = _phase_customer_flow_estimands(logger, 0, 3)
    assert math.isclose(
        stat_phase['amm_customer_volume_share'], 19.0 / 110.0
    )
    assert math.isclose(
        stat_phase['clob_customer_volume_share'], 91.0 / 110.0
    )
    assert math.isclose(stat_phase['amm_active_tick_flow_share'], 0.5)
    assert math.isclose(stat_phase['clob_active_tick_flow_share'], 0.5)


def test_arbitrage_is_separate_from_customer_routing_volume():
    logger = _sparse_flow_logger()

    assert logger.arbitrage_volume_total() == 81.0
    assert logger.amm_execution_volume() == 100.0
    assert math.isclose(logger.arbitrage_share_of_amm_execution(), 0.81)
    # Customer routing is unchanged by the additional arbitrage volume.
    assert math.isclose(logger.amm_customer_volume_share(), 19.0 / 110.0)


def test_arbitrageur_publishes_successful_period_amm_legs():
    class MockCLOB:
        def mid_price(self):
            return 100.0

        def quoted_spread_bps(self):
            return 10.0

    pool = CPMMPool(x=1000.0, y=80000.0, fee=0.003)
    arbitrageur = AMMArbitrageur(
        MockCLOB(),
        {'cpmm': pool},
        max_correction_pct=50.0,
        trade_fraction_cap=0.10,
        cash=1_000_000.0,
        assets=1_000.0,
    )

    arbitrageur.arbitrage()

    assert len(arbitrageur.period_trades) == 1
    trade = arbitrageur.period_trades[0]
    assert trade['execution_source'] == 'amm_arbitrage'
    assert trade['venue'] == 'cpmm'
    assert trade['quantity'] > 0.0
    assert abs(trade['signed_quantity']) == trade['quantity']
    assert trade['side'] in {'buy', 'sell'}


def _snapshot(logger, attempts, trades=()):
    """Drive one period of the logger's routing accounting."""
    env = SimpleNamespace(
        sigma=0.01, funding_cost=0.001, is_stress=lambda: False,
        session_name='London', systemic_liquidity=1.0,
        order_flow_imbalance=0.0, clob_order_flow_imbalance=0.0,
        fair_price=100.0,
    )
    logger.iterations.append(len(logger.iterations))
    index = len(logger.iterations) - 1
    routed, refused, reasons = {}, {}, {}
    for attempt in attempts:
        venue = attempt['venue']
        requested = float(attempt['requested_quantity'])
        executed = float(attempt['executed_quantity'])
        routed[venue] = routed.get(venue, 0.0) + requested
        unfilled = max(0.0, requested - executed)
        if attempt.get('refusal_reason') is None and unfilled <= 0.0:
            continue
        refused[venue] = refused.get(venue, 0.0) + unfilled
        key = attempt.get('refusal_reason') or 'partial_fill'
        reasons.setdefault(venue, {})[key] = (
            reasons.get(venue, {}).get(key, 0.0) + unfilled
        )
    for venue in set(routed) | set(logger.routed_volume):
        logger._ensure_routed_venue(venue)
        logger._append_at(logger.routed_volume[venue], index,
                          routed.get(venue, 0.0))
        logger._append_at(logger.refused_volume[venue], index,
                          refused.get(venue, 0.0))
        volumes = logger.refusal_reason_volume[venue]
        for reason in set(volumes) | set(reasons.get(venue, {})):
            volumes.setdefault(reason, [])
            logger._append_at(volumes[reason], index,
                              reasons.get(venue, {}).get(reason, 0.0))
    del env, trades


def test_routed_share_is_immune_to_a_customer_side_inventory_limit():
    """H2 has to be measured on demand, not on what the demand could do.

    In a crisis nearly every request on the book fails because the seller has
    hit its own short limit.  That collapses the denominator of the executed
    share and lifts the facility's share without one order changing venue.
    The routed share is built on the routing decision itself and does not
    move; the refusal split says why the executed share did.
    """
    logger = MetricsLogger()
    logger.iterations = []

    # Calm: nine tenths of demand goes to the book and all of it trades.
    _snapshot(logger, [
        {'venue': 'clob', 'requested_quantity': 90.0,
         'executed_quantity': 90.0, 'refusal_reason': None},
        {'venue': 'hfmm', 'requested_quantity': 10.0,
         'executed_quantity': 10.0, 'refusal_reason': None},
    ])
    calm_routed = logger.amm_routed_volume_share(0, 1)
    assert math.isclose(calm_routed, 0.1)

    # Crisis: the identical routing decision, but the book's sellers are at
    # their inventory limit, so almost nothing they send can execute.
    _snapshot(logger, [
        {'venue': 'clob', 'requested_quantity': 90.0,
         'executed_quantity': 0.9, 'refusal_reason': 'no_assets'},
        {'venue': 'hfmm', 'requested_quantity': 10.0,
         'executed_quantity': 10.0, 'refusal_reason': None},
    ])

    assert math.isclose(logger.amm_routed_volume_share(1, 2), 0.1)
    assert math.isclose(logger.refused_volume_share('clob', 1, 2),
                        89.1 / 90.0)
    assert logger.refused_volume_share('hfmm', 1, 2) == 0.0
    reasons = logger.refusal_reason_shares('clob', 1, 2)
    assert math.isclose(reasons['no_assets'], 1.0)
    # The reason is on the customer's side of the trade, so it says nothing
    # about the book having become unavailable.
    assert 'no_liquidity' not in reasons

    # Over the whole run the routing decision never moved.
    assert math.isclose(logger.amm_routed_volume_share(), 0.1)


def test_refusal_reasons_separate_the_venue_from_the_customer():
    logger = MetricsLogger()
    logger.iterations = []
    _snapshot(logger, [
        {'venue': 'clob', 'requested_quantity': 40.0,
         'executed_quantity': 0.0, 'refusal_reason': 'no_liquidity'},
        {'venue': 'clob', 'requested_quantity': 60.0,
         'executed_quantity': 0.0, 'refusal_reason': 'no_assets'},
    ])

    shares = logger.refusal_reason_shares('clob')
    assert math.isclose(shares['no_liquidity'], 0.4)
    assert math.isclose(shares['no_assets'], 0.6)
    assert math.isclose(sum(shares.values()), 1.0)
    assert math.isclose(logger.refused_volume_share('clob'), 1.0)


def test_customer_trade_records_carry_an_explicit_execution_source():
    trader = SimpleNamespace(id=7, type='Hedger')
    record = Trader._make_trade_record(
        trader,
        venue='clob',
        side='buy',
        Q=2.0,
        result={
            'exec_price': 100.0,
            'fee_bps': 0.25,
            'cost_bps': 0.5,
            'requested_qty': 2.0,
        },
        cls_info={'theta': 0.01, 'size_bucket': 'small'},
    )

    assert record['execution_source'] == 'routed_customer'
