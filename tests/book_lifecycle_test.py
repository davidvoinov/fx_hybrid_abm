import random
import statistics

import pytest

from AgentBasedModel.agents.agents import (
    ExchangeAgent,
    FastRecyclerLP,
    MarketMaker,
    _draw_geometric_lifetime,
)
from AgentBasedModel.environment.processes import MarketEnvironment
from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.utils import Order


def _fast_provider(seed=7, median=3):
    random.seed(seed)
    exchange = ExchangeAgent(price=100.0, std=2.0, volume=20, price_tick=0.005)
    env = MarketEnvironment(price=100.0, price_vol_scale=0.0012859)
    provider = FastRecyclerLP(
        exchange, cash=20_000.0, env=env, levels=1, ttl=median,
        base_qty=1, base_withdraw_prob=0.0,
    )
    provider.max_cash_borrow = 25_000.0
    provider.max_short_assets = 75.0
    return exchange, env, provider


def _dealer(seed=11, median=290):
    random.seed(seed)
    exchange = ExchangeAgent(price=100.0, std=2.0, volume=20, price_tick=0.005)
    env = MarketEnvironment(price=100.0, price_vol_scale=0.0012859)
    dealer = MarketMaker(
        exchange, cash=70_000.0, env=env,
        alpha0=0.1, alpha1=1.0, alpha2=50.0, alpha3=0.0,
        d0=40.0, d1=0.0, d2=0.0, d3=0.0,
        n_levels=5, quote_life=median, quote_refresh_tol_bps=20.0,
        venue_interaction_mode='none',
    )
    dealer.max_cash_borrow = 80_000.0
    dealer.max_short_assets = 250.0
    return exchange, env, dealer


def test_geometric_clock_uses_the_configured_median_without_a_point_mass():
    # Use one stream for the distribution; constructing one stream per draw
    # would repeat the first variate on purpose.
    rng_short = random.Random(17)
    short = [_draw_geometric_lifetime(rng_short, 3) for _ in range(20_000)]
    rng_dealer = random.Random(23)
    dealer = [_draw_geometric_lifetime(rng_dealer, 290) for _ in range(20_000)]

    assert statistics.median(short) == 3
    assert 275 <= statistics.median(dealer) <= 305
    assert len(set(short)) > 10
    assert len(set(dealer)) > 500
    assert short.count(3) / len(short) < 0.20
    assert dealer.count(290) / len(dealer) < 0.01


def test_fast_provider_replaces_one_expiry_without_cancelling_the_survivor():
    exchange, env, provider = _fast_provider()
    env._t = 1
    provider.call()
    assert len(provider.orders) == 2

    survivor = provider.orders[0]
    expiring = provider.orders[1]
    survivor.ttl = 50
    expiring.ttl = 0
    env._t = 2
    exchange.expire_orders(fair_price=env.fair_price)

    assert survivor in provider.orders
    assert expiring not in provider.orders
    assert provider.order_lifecycle_events[-1]['reason'] == 'scheduled_cancel'

    provider.call()
    assert survivor in provider.orders
    assert len(provider.orders) == 2
    replacement = next(order for order in provider.orders if order is not survivor)
    assert replacement._created_tick == 2
    assert replacement._scheduled_lifetime >= 1


def test_lifecycle_observations_include_live_right_censoring_without_mutation():
    _, env, provider = _fast_provider()
    env._t = 4
    order = provider._stamp_quote(
        Order(99.99, 2.0, 'bid', provider), reference_mid=100.0,
    )
    provider.orders.append(order)
    env._t = 9

    first = provider.lifecycle_observations()
    second = provider.lifecycle_observations()
    assert first == second
    assert len(first) == 1
    assert first[0]['ended_tick'] is None
    assert first[0]['lifetime'] == 5.0
    assert first[0]['reason'] == 'censored'
    assert first[0]['censored'] is True
    assert not provider.completed_order_lifetimes

    provider.record_order_end(order, reason='stale_reprice')
    provider.record_order_end(order, reason='fill')
    completed = provider.lifecycle_observations(include_live=False)
    assert len(completed) == 1
    assert completed[0]['reason'] == 'stale_reprice'
    assert completed[0]['lifetime'] == 5.0


def test_dealer_ladder_has_independent_order_clocks():
    _, env, dealer = _dealer()
    env._t = 1
    dealer.call()
    scheduled = [order._scheduled_lifetime for order in dealer.orders]

    assert len(scheduled) == 2 * dealer.n_levels
    assert len(set(scheduled)) >= 5
    assert sum(value == dealer.quote_life for value in scheduled) <= 1


def test_dealer_expiry_does_not_trigger_a_bulk_ladder_refresh():
    exchange, env, dealer = _dealer()
    env._t = 1
    dealer.call()
    survivor = dealer.orders[0]
    expiring = dealer.orders[-1]
    survivor.ttl = 500
    expiring.ttl = 0

    env._t = 2
    exchange.expire_orders(fair_price=env.fair_price)
    before_ids = {order.order_id for order in dealer.orders}
    assert survivor.order_id in before_ids
    assert expiring.order_id not in before_ids

    dealer.call()
    after_ids = {order.order_id for order in dealer.orders}
    assert survivor.order_id in after_ids
    assert len(before_ids - after_ids) == 0
    assert len(after_ids - before_ids) == 1
    assert dealer.order_lifecycle_events[-1]['reason'] == 'scheduled_cancel'


def test_dealer_bulk_removals_have_distinct_lifecycle_reasons():
    _, env, dealer = _dealer()
    env._t = 1
    dealer.call()
    forced_ids = {order.order_id for order in dealer.orders}
    env.mm_pause_ticks = 2
    dealer.cancel_all_quotes()

    forced = [row for row in dealer.order_lifecycle_events
              if row['order_id'] in forced_ids]
    assert forced
    assert {row['reason'] for row in forced} == {'forced_pause'}
    assert dealer.mm_state == 'active'

    env.mm_pause_ticks = 0
    env._t = 2
    dealer.call()
    endogenous_ids = {order.order_id for order in dealer.orders}
    dealer.cancel_all_quotes(reason='endogenous_withdrawal')

    endogenous = [row for row in dealer.order_lifecycle_events
                  if row['order_id'] in endogenous_ids]
    assert endogenous
    assert {row['reason'] for row in endogenous} == {'endogenous_withdrawal'}


def test_dealer_withdrawal_requires_persistence_and_resets_after_a_drop():
    _, _, dealer = _dealer()
    dealer.withdrawal_threshold = 0.7
    dealer.reentry_threshold = 0.4
    dealer.withdrawal_confirmation_ticks = 2
    dealer._update_mtm_signals = lambda mid: None
    scores = iter((0.8, 0.6, 0.8, 0.8))

    def components(mid):
        return {'score': next(scores), 'liquidity_factor': 1.0}

    dealer._withdrawal_components = components
    dealer._update_endogenous_state(100.0)
    assert dealer.mm_state == 'defensive'
    assert dealer._withdrawal_confirmation_count == 1

    dealer._update_endogenous_state(100.0)
    assert dealer.mm_state == 'defensive'
    assert dealer._withdrawal_confirmation_count == 0

    dealer._update_endogenous_state(100.0)
    assert dealer.mm_state == 'defensive'
    assert dealer._withdrawal_confirmation_count == 1

    dealer._update_endogenous_state(100.0)
    assert dealer.mm_state == 'withdrawn'
    assert dealer._withdrawal_confirmation_count == 2


def test_dealer_scheduler_is_seeded_exchangeable_and_deduplicated():
    dealers = [_dealer(seed=100 + index)[2] for index in range(5)]
    # Include the compatibility handle in book_agents on purpose. The
    # scheduler must still call it only once. Changing which dealer occupies
    # that handle must not create a deterministic first-call privilege.
    def make_sim():
        return Simulator(
            traders=list(dealers),
            book_agents=[dealers[0], *dealers[1:]],
            market_maker=dealers[0],
            dealer_scheduler_rng=random.Random(907),
        )

    first = []
    reproduced = []
    first_sim = make_sim()
    reproduced_sim = make_sim()
    random.seed(19)
    for tick in range(250):
        scheduled = first_sim._scheduled_market_makers()
        assert len(scheduled) == len(dealers)
        assert {id(dealer) for dealer in scheduled} == {
            id(dealer) for dealer in dealers
        }
        first.append(dealers.index(scheduled[0]))

        # Consume an unrelated and tick-varying number of process-global
        # draws and mutate the public book-agent order just as step 3 does.
        # The private stream and immutable roster must reproduce the same
        # dealer schedule in the paired simulator.
        for _ in range(tick % 11):
            random.random()
        random.shuffle(first_sim.book_agents)
        scheduled_again = reproduced_sim._scheduled_market_makers()
        reproduced.append(dealers.index(scheduled_again[0]))

    assert first == reproduced
    assert set(first) == set(range(len(dealers)))
    counts = [first.count(index) for index in range(len(dealers))]
    assert max(counts) - min(counts) < 30
    assert first.count(0) < len(first)


def test_dealer_schedule_is_paired_across_amm_treatment_arms():
    def schedule(enable_amm):
        random.seed(42)
        sim = Simulator.default_fx(enable_amm=enable_amm)
        roster = list(sim._market_maker_roster)
        return [
            [roster.index(dealer)
             for dealer in sim._scheduled_market_makers()]
            for _ in range(100)
        ]

    assert schedule(True) == schedule(False)


def test_allocating_dealer_scheduler_does_not_shift_shared_agent_rng():
    random.seed(314159)
    expected_next_draw = random.random()

    random.seed(314159)
    Simulator()
    assert random.random() == expected_next_draw


def _minimal_paired_fx_sim(**overrides):
    config = dict(
        n_noise=0,
        n_mm=1,
        n_fast_lp=0,
        n_clob_fund=0,
        n_fx_takers=1,
        n_fx_fund=0,
        n_retail=0,
        n_institutional=0,
        clob_volume=20,
        enable_cpmm=False,
        amm_lp_model='endogenous',
        fx_flow_intensity_scale=1.0,
    )
    config.update(overrides)
    random.seed(20260813)
    return Simulator.default_fx(**config)


def _customer_intent_and_route_path(trader, periods=1000):
    """Draw demand and route it without mutating balances or venue reserves.

    Quote execution is omitted by design, since the invariant under test is that
    arrival, side and requested size are common innovations while routing is
    allowed to respond to the treatment or OAT parameter.
    """
    demand, routes = [], []
    for _ in range(periods):
        intent = trader._draw_customer_intent()
        demand.append(intent)
        if intent is None:
            continue
        side, quantity = intent
        routes.append(trader.choose_venue(quantity, side))
    return demand, routes


def test_amm_construction_does_not_shift_shared_or_customer_rngs():
    without = _minimal_paired_fx_sim(enable_amm=False)
    without_global = random.getstate()
    with_amm = _minimal_paired_fx_sim(enable_amm=True)
    with_global = random.getstate()

    assert without_global == with_global
    assert (without._fx_scheduler_rng.getstate()
            == with_amm._fx_scheduler_rng.getstate())
    assert (without.fx_traders[0]._flow_rng.getstate()
            == with_amm.fx_traders[0]._flow_rng.getstate())
    assert (without.fx_traders[0]._routing_rng.getstate()
            == with_amm.fx_traders[0]._routing_rng.getstate())


def test_customer_demand_is_paired_while_treatment_and_routing_may_differ():
    without = _minimal_paired_fx_sim(
        enable_amm=False, venue_choice_rule='fixed_share', amm_share_pct=100.0,
    )
    with_amm = _minimal_paired_fx_sim(
        enable_amm=True, venue_choice_rule='fixed_share', amm_share_pct=100.0,
    )
    demand_without, routes_without = _customer_intent_and_route_path(
        without.fx_traders[0]
    )
    demand_with, routes_with = _customer_intent_and_route_path(
        with_amm.fx_traders[0]
    )
    assert demand_without == demand_with
    assert routes_without != routes_with
    assert set(routes_without) == {'clob'}
    assert set(routes_with) == {'hfmm'}

    scale_2 = _minimal_paired_fx_sim(
        enable_amm=True,
        venue_choice_rule='liquidity_aware',
        routing_cost_scale_bps=2.0,
    )
    scale_8 = _minimal_paired_fx_sim(
        enable_amm=True,
        venue_choice_rule='liquidity_aware',
        routing_cost_scale_bps=8.0,
    )
    demand_2, routes_2 = _customer_intent_and_route_path(
        scale_2.fx_traders[0]
    )
    demand_8, routes_8 = _customer_intent_and_route_path(
        scale_8.fx_traders[0]
    )
    assert demand_2 == demand_8
    assert routes_2 != routes_8


def test_multilevel_dealer_fill_is_not_replenished_twice():
    exchange, env, dealer = _dealer()
    env._t = 1
    dealer.call()
    initial_bid_qty = sum(order.qty for order in dealer.orders
                          if order.order_type == 'bid')
    removed = [order for order in dealer.orders
               if order.order_type == 'bid'
               and order._quote_level in {0, 1}]
    assert {order._quote_level for order in removed} == {0, 1}
    taken = sum(order.qty for order in removed)

    # Emulate the post-match state passed to the fill-replenishment hook.  The
    # settlement path itself is covered by the balance-sheet tests; here the
    # invariant is that one removed unit creates exactly one replacement unit.
    for order in removed:
        dealer.orders.remove(order)
        exchange.order_book['bid'].remove(order)
        dealer.release_order_reservation(order)
        order.qty = 0.0
        dealer.record_order_end(order, reason='fill')

    assert dealer.on_trade(bid_taken=taken)
    after_replenishment = sum(order.qty for order in dealer.orders
                              if order.order_type == 'bid')
    assert after_replenishment == pytest.approx(initial_bid_qty)
    assert {order._quote_level for order in dealer.orders
            if order.order_type == 'bid'} == set(range(dealer.n_levels))

    env._t = 2
    dealer.call()
    after_next_call = sum(order.qty for order in dealer.orders
                          if order.order_type == 'bid')
    assert after_next_call == pytest.approx(after_replenishment)
