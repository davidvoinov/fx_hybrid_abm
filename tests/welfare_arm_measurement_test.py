"""What the welfare account reads, and when it reads it.

Each check below corresponds to a quantity that was measured at the wrong
instant or against the wrong benchmark, and each of those turned a loss into
a profit or a capital base into four of them.
"""
import math

import pytest

from tools.robustness.welfare_accounting import (
    ARM_CAPITAL, CRISIS, measure, run,
)

SEED = 42


def test_provider_wallets_are_read_at_the_opening_of_the_window():
    """Wallets are an attribute of the provider, not a series.

    Read after a run taken in one step they report the end of the year. At the
    calibrated size the wallets stand at about 231,000 when the window opens
    and about 952,000 when the run ends, so the capital the arrangement was
    charged for was four times the capital it tied up.
    """
    sim, shock = run(SEED, 'reserve')
    captured = sim.facility_window['opening_wallets']
    providers = [lp for pop in sim.lp_providers for lp in pop.providers]
    prices = list(sim.logger.fair_price_series)
    closing = sum(float(lp.wallet_base) * float(prices[-1])
                  + float(lp.wallet_cash) for lp in providers)
    assert captured > 0.0
    assert closing > 0.0
    assert captured < 0.5 * closing, (
        f'the captured wallets {captured:,.0f} look like the end of the run '
        f'{closing:,.0f} and not the opening of the window')


@pytest.mark.parametrize('arm', ['dealer_of_last_resort', 'passive_book'])
def test_the_order_book_facility_is_marked_against_a_moving_price(arm):
    """A position carried through a decline cannot show no loss.

    Marked at the price the market opened the year on, the inventory an
    obliged quoter accumulates is never revalued, and both order book arms
    reported a profit through an episode that repriced the pair by about one
    per cent.
    """
    sim, shock = run(SEED, arm)
    captured = sim.facility_window
    facility = next(t for t in sim.traders
                    if getattr(t, 'is_facility_arm', False))
    prices = list(sim.logger.fair_price_series)
    opening_price = float(prices[shock + CRISIS[0]])
    closing_price = float(prices[shock + CRISIS[1]])
    assert closing_price < opening_price, 'the episode should reprice the pair'
    assert facility.assets > 0, 'the quoter should carry inventory'

    # The result has to be the trading payoff against a hold and rebalance
    # benchmark, which is what the pool's result is measured against.
    result = captured['operating_result']
    assert result is not None and math.isfinite(result)
    naive = (float(facility.cash) + float(facility.assets) * 100.0
             - captured['opening_capital'])
    assert not math.isclose(result, naive, rel_tol=1e-6), (
        'the result still looks like equity marked at a fixed price')
    assert result < 0.0, (
        f'an inventory carried through the decline should not show a profit, '
        f'got {result:+.2f}')


@pytest.mark.parametrize('arm', ['dealer_of_last_resort', 'passive_book'])
def test_the_obliged_quoter_stays_out_of_the_dealer_state_shares(arm):
    """The withdrawal denominator has to be the same sector in every arm.

    The quoter belongs in the roster, since it quotes and needs its place in
    the queue, but counted among the incumbents it added a sixth dealer that
    never withdraws to the denominator of every state share.
    """
    sim, _ = run(SEED, arm)
    roster = sim._market_maker_roster
    assert sum(1 for d in roster if getattr(d, 'is_facility_arm', False)) == 1
    assert len(roster) == 6, 'the quoter should still be scheduled'
    assert sim._market_maker_state_summary()['n_market_makers'] == 5, (
        'the quoter reached the incumbent dealer state shares')


def test_the_arms_are_charged_for_comparable_capital():
    """Matched arms must present the same capital base to the account."""
    pool = measure(SEED, arm='reserve')['with_amm']
    book = measure(SEED, arm='dealer_of_last_resort')['with_amm']
    pool_capital = pool['lp_opening_capital'] + pool['lp_wallet_capital']
    book_capital = book['book_facility_capital']
    assert pool_capital == pytest.approx(book_capital, rel=0.05), (
        f'pool holds {pool_capital:,.0f} against the book arm {book_capital:,.0f}')
