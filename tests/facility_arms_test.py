"""The arms differ in how they price and in nothing else.

Each check below corresponds to a defect that made the comparison measure
something other than the facility: an obliged quoter counted as an incumbent
dealer, an arm holding more capital than the pool it was matched to, and an
episode whose limbs ran on three different clocks.
"""
import math

import pytest

import main as M
from AgentBasedModel.environment.processes import _MAX_STRESS_DECAY


def _args(argv):
    parser = M.build_parser()
    args = parser.parse_args(argv)
    M._apply_preset_defaults(parser, args)
    args.venue_choice_rule = M._resolve_main_routing(args, argv)
    M._auto_stress_around_shock(args)
    return args


def _build(arm, capital=0.0, seed=42):
    argv = ['--preset', 'dash_for_cash_2020', '--facility-arm', arm,
            '--seed', str(seed), '--n-iter', '400', '--silent']
    if capital > 0:
        argv += ['--arm-capital', str(capital)]
    args = _args(argv)
    M._seed_all(seed)
    return M.build_sim(args), args


@pytest.mark.parametrize('arm', ['dealer_of_last_resort', 'passive_book'])
def test_the_obliged_quoter_is_not_counted_as_an_incumbent_dealer(arm):
    """An arm that never withdraws must not dilute the withdrawal measure.

    The quoter an arm adds is a MarketMaker by class. Counted in the dealer
    sector it lowered the measured share of withdrawn capacity for that arm
    alone and weakened the cascade that reads it, so the arms differed in the
    measurement as well as in the facility.
    """
    sim, _ = _build(arm, capital=200_000.0)
    dealers = [t for t in sim.traders if type(t).__name__ == 'MarketMaker']
    facility = [d for d in dealers if getattr(d, 'is_facility_arm', False)]
    incumbents = [d for d in dealers if not getattr(d, 'is_facility_arm', False)]
    assert len(facility) == 1, 'the arm should add exactly one obliged quoter'
    assert incumbents, 'the incumbent dealers should still be present'
    assert len(dealers) == len(incumbents) + 1

    observed = []
    original = sim.env.observe_dealer_sector

    def _capture(sector):
        observed.append(list(sector or ()))
        return original(sector)

    sim.env.observe_dealer_sector = _capture
    sim.simulate(30, silent=True)
    assert observed, 'the dealer sector should be observed every period'
    for sector in observed:
        assert all(not getattr(d, 'is_facility_arm', False) for d in sector), (
            'the obliged quoter reached the incumbent dealer aggregate')


def test_the_obliged_quoter_carries_the_same_capital_and_the_same_exposure():
    """The same capital as the pool, held in the same two currencies.

    An arm compared against a reserve priced pool has to carry the same
    exposure to the pair, or the contrast between them is the pricing schedule
    plus a currency position the pool has and it does not. Endowed all in cash
    the arm lost nothing on the base leg through a one per cent decline while
    the pool lost on all of it. Neither line is granted: holding base to sell
    and quote to buy with, it needs no credit to quote either side.
    """
    budget = 200_000.0
    sim, args = _build('dealer_of_last_resort', capital=budget)
    facility = next(t for t in sim.traders
                    if getattr(t, 'is_facility_arm', False))
    price = float(args.price)
    reserve = budget / (2.0 * price)
    assert float(getattr(facility, 'max_cash_borrow', 0.0)) == 0.0
    assert float(getattr(facility, 'max_short_assets', 0.0)) == 0.0
    assert float(facility.cash) == pytest.approx(budget / 2.0, rel=0.02)
    assert float(facility.assets) == pytest.approx(reserve, rel=0.02)
    committed = float(facility.cash) + float(facility.assets) * price
    assert committed == pytest.approx(budget, rel=0.02), (
        f'the arm should commit the budget it is given, holding {committed}')
    assert float(facility.softlimit) == pytest.approx(reserve, rel=0.02)


def test_the_endowment_does_not_read_as_an_exposure():
    """A matched endowment must not shrink the quote it is meant to fund.

    A dealer's neutral position here is zero, so half the budget held in base
    put the arm at its own limit and the depth rule subtracted a tenth of the
    endowment from a base depth of a hundred and twenty, flooring the quote at
    three units on a balance sheet of nearly a million.
    """
    sim, args = _build('dealer_of_last_resort', capital=951_999.0)
    facility = next(t for t in sim.traders
                    if getattr(t, 'is_facility_arm', False))
    assert float(facility.assets) > 0.0, 'the arm should hold the base leg'
    assert float(facility.risk_inventory) == 0.0, (
        'its endowment is the position it treats as flat')
    depth = float(facility._target_depth(float(args.price)))
    assert depth > 10.0 * float(facility.d_min), (
        f'the arm quotes {depth} against a floor of {facility.d_min}')
    assert depth == pytest.approx(float(facility.d0), rel=0.05)


def test_every_limb_of_a_sustained_episode_outlasts_the_dysfunction_counter():
    """One episode, one duration, on all three limbs.

    Toxic flow was cleared outright when the dysfunction counter reached zero
    at about thirty seconds, so an episode declaring a half life of a hundred
    and seventy three ran that limb for thirty whatever it declared, while the
    liquidity and volatility limbs beside it ran for hundreds.
    """
    sim, args = _build('reserve')
    shock = int(args.shock_iter)
    sim.simulate(shock, silent=True)
    env = sim.env

    sim.simulate(20, silent=True)
    early = (abs(env.toxic_flow_bias), env._liquidity_shock, env._shock_sigma_overlay)
    assert all(v > 0 for v in early), 'every limb should be live inside the window'

    # Past the dysfunction counter, which is the point the limb was cleared at.
    while env.shock_ticks_remaining > 0:
        sim.simulate(1, silent=True)
    sim.simulate(30, silent=True)
    assert env.shock_ticks_remaining == 0
    late = (abs(env.toxic_flow_bias), env._liquidity_shock, env._shock_sigma_overlay)
    for name, value in zip(('toxic flow', 'liquidity', 'volatility'), late):
        assert value > 0.0, f'the {name} limb was cleared by the counter'
    assert late[0] > 0.5 * early[0], (
        'toxic flow should decay on its declared clock, not collapse')


def test_a_declared_episode_duration_survives_the_clamp():
    """The ceiling has to admit the durations the episodes declare.

    Both sustained episodes declare 0.997 per tick, a half life of two hundred
    and thirty one seconds. A ceiling of 0.99 truncated that to sixty nine and
    did so silently, so the episodes ran at a third of the duration they stated.
    """
    args = _args(['--preset', 'dash_for_cash_2020', '--seed', '42',
                  '--n-iter', '400', '--silent'])
    for name in ('liquidity_shock_decay', 'toxic_flow_decay',
                 'stress_overlay_decay'):
        declared = float(getattr(args, name))
        assert declared <= _MAX_STRESS_DECAY, (
            f'{name} declares {declared} above the ceiling {_MAX_STRESS_DECAY}')

    M._seed_all(42)
    sim = M.build_sim(args)
    sim.simulate(int(args.shock_iter) + 5, silent=True)
    env = sim.env
    half_life = lambda d: math.log(0.5) / math.log(d)
    assert half_life(env._liquidity_shock_decay) > 200.0
    assert half_life(env._shock_decay) > 200.0
    assert half_life(env._toxic_flow_decay) > 150.0


def test_the_obliged_quoter_holds_its_spread_through_the_episode():
    """The arm shows one spread in every state, which is its whole content.

    Zeroing the volatility, funding and flow terms left the multiplicative
    liquidity loading in place, so an arm declared at three and a half basis
    points quoted between nine and twelve through the crisis and set the
    market's spread late in the window at three times its stated level. An arm
    that widens with the market cannot isolate standing availability from the
    dealer behaviour it is compared against.
    """
    sim, args = _build('dealer_of_last_resort', capital=951_999.0)
    facility = next(t for t in sim.traders
                    if getattr(t, 'is_facility_arm', False))
    assert facility.state_independent_quote is True
    declared = float(args.arm_spread_bps)
    sim.simulate(int(args.shock_iter), silent=True)
    quoted = []
    depths = []
    for _ in range(150):
        sim.simulate(1, silent=True)
        quoted.append(float(facility._target_spread_bps(float(args.price))))
        depths.append(float(facility._target_depth(float(args.price))))
    assert max(quoted) == pytest.approx(declared, rel=1e-6)
    assert min(quoted) == pytest.approx(declared, rel=1e-6)
    # Depth still answers to the position the arm has taken, as the pool's
    # schedule answers to the reserves it has left.
    assert depths[-1] < depths[0]


def test_an_incumbent_dealer_still_widens_with_the_market():
    """The exemption is the arm's alone and must not reach the dealers."""
    sim, args = _build('none')
    dealers = [t for t in sim.traders if type(t).__name__ == 'MarketMaker']
    assert dealers
    assert all(d.state_independent_quote is False for d in dealers)
    calm = [float(d._target_spread_bps(float(args.price))) for d in dealers]
    sim.simulate(int(args.shock_iter) + 30, silent=True)
    stressed = [float(d._target_spread_bps(float(args.price))) for d in dealers]
    assert max(stressed) > max(calm), (
        'incumbent dealers should still widen when the market does')


def test_the_obliged_quoter_reprices_every_period():
    """Both arms must reprice at the same frequency, or the contrast is that.

    The incumbent refresh tolerance of twenty basis points froze the arm in a
    market whose spread is under one: its offer stood unchanged through the
    second half of the window while the mid fell eleven basis points, so an
    arm declared at three and a half basis points showed twenty three. The
    pool reprices every period by construction, so the contrast between them
    measured how often each repriced instead of how each one prices.
    """
    sim, args = _build('dealer_of_last_resort', capital=951_999.0)
    facility = next(t for t in sim.traders
                    if getattr(t, 'is_facility_arm', False))
    assert float(facility.quote_refresh_tol_bps) == 0.0
    declared = float(args.arm_spread_bps)
    sim.simulate(int(args.shock_iter), silent=True)
    implied = []
    for _ in range(150):
        sim.simulate(1, silent=True)
        orders = list(getattr(facility, 'orders', []) or [])
        bids = [float(o.price) for o in orders if str(o.order_type) == 'bid']
        asks = [float(o.price) for o in orders if str(o.order_type) == 'ask']
        mid = sim.clob.mid_price()
        if bids and asks and mid:
            implied.append((min(asks) - max(bids)) / mid * 1e4)
    assert implied, 'the arm should be quoting on both sides'
    assert max(implied) < 2.0 * declared, (
        f'the arm shows up to {max(implied):.2f} bps against a declared '
        f'{declared}, which is a stale quote and not a wide one')


def test_the_agent_table_describes_the_model_that_runs():
    """Every class the paper tabulates must exist in the calibrated model.

    Table 1 was inherited from the general purpose model this one was built
    from. It tabulated chartists, universalists and a latent provider, whose
    populations the calibration sets to zero, and omitted the dealers, the
    provider population and the arbitrageur that the results are about. The
    paper therefore described a market that no run contained.
    """
    import json
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    article = os.path.join(root, 'EconMod', 'article', 'econmod.tex')
    if not os.path.exists(article):
        pytest.skip('the manuscript is not part of this checkout')
    text = open(article, encoding='utf-8').read()
    table = text[text.index(r'\label{tab:agent_rules}'):]
    table = table[:table.index(r'\end{tabular}')]

    defaults = json.load(open(
        os.path.join(root, 'calibration', 'primary_model.json'),
        encoding='utf-8'))['cli_defaults']
    # These populations were tabulated with a count of zero, so the paper
    # described trend following, strategy switching and a conditional provider
    # that no run contained. They are gone from the model and must stay gone
    # from the table.
    for klass in ('Chartist', 'Universalist', 'LatentLP'):
        assert klass not in table, f'{klass} is retired but still tabulated'
    for name in ('n_clob_chart', 'n_clob_univ', 'n_latent_lp'):
        assert name not in defaults, f'{name} is retired but still calibrated'
    # A population the calibration switches off must not be tabulated as part
    # of the model. n_noise stands at zero and is not tabulated, which is the
    # state this checks for.
    empty = {name for name, value in defaults.items()
             if name.startswith('n_') and value == 0}
    for name in empty:
        assert name not in table, (
            f'{name} is calibrated to zero and still appears in the table')

    for klass in ('MarketMaker', 'FastRecyclerLP', 'Fundamentalist'):
        assert klass in table, f'{klass} runs in every simulation but is not tabulated'

    # The counts in the table are the counts the model runs.
    tabulated = set(re.findall(r'\$(\d+)(?:\{,\}\d+)?\$ &', table))
    for name in ('n_mm', 'n_clob_fund', 'n_fast_lp', 'n_fx_fund'):
        assert str(defaults[name]) in tabulated, (
            f'{name} is {defaults[name]} in the calibration and does not '
            f'appear among the tabulated counts {sorted(tabulated)}')
