import math

import pytest

from AgentBasedModel.agents.agents import AMMArbitrageur
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool


class _FixedCLOB:
    def __init__(self, price=100.0):
        self.price = float(price)

    def mid_price(self):
        return self.price

    def quoted_spread_bps(self):
        return 1.0


def _pool(kind, price):
    if kind == 'cpmm':
        return CPMMPool(x=1_000.0, y=1_000.0 * price, fee=0.0005)
    return HFMMPool(
        x=1_000.0, y=1_000.0 * price, A=18.0, fee=0.0005,
        rate=100.0,
    )


@pytest.mark.parametrize('kind', ['cpmm', 'hfmm'])
def test_arbitrage_sell_requires_prefunded_base(kind):
    pool = _pool(kind, 120.0)
    before = (pool.x, pool.y)
    arb = AMMArbitrageur(
        _FixedCLOB(), {'pool': pool}, cash=1_000_000.0, assets=0.0,
        max_correction_pct=50.0, trade_fraction_cap=1.0,
    )

    arb.arbitrage()

    assert (pool.x, pool.y) == before
    assert arb.assets == 0.0
    assert arb.period_trades == []


@pytest.mark.parametrize('kind', ['cpmm', 'hfmm'])
def test_arbitrage_sell_exhausts_but_never_exceeds_base_wallet(kind):
    pool = _pool(kind, 120.0)
    initial_assets = 7.0
    arb = AMMArbitrageur(
        _FixedCLOB(), {'pool': pool}, cash=0.0, assets=initial_assets,
        max_correction_pct=50.0, trade_fraction_cap=1.0,
    )

    arb.arbitrage()

    sold = sum(
        trade['quantity'] for trade in arb.period_trades
        if trade['signed_quantity'] < 0.0
    )
    assert sold <= initial_assets + 1e-9
    assert arb.assets >= 0.0
    assert math.isclose(sold + arb.assets, initial_assets, abs_tol=1e-9)
    state = (pool.x, pool.y, arb.cash, arb.assets)
    arb.arbitrage()
    assert (pool.x, pool.y, arb.cash, arb.assets) == state
    assert arb.period_trades == []


@pytest.mark.parametrize('kind', ['cpmm', 'hfmm'])
def test_arbitrage_buy_exhausts_but_never_exceeds_cash_wallet(kind):
    pool = _pool(kind, 80.0)
    initial_cash = 500.0
    arb = AMMArbitrageur(
        _FixedCLOB(), {'pool': pool}, cash=initial_cash, assets=0.0,
        max_correction_pct=50.0, trade_fraction_cap=1.0,
    )

    before_pool_quote = pool.y + pool.fee_quote
    arb.arbitrage()
    after_pool_quote = pool.y + pool.fee_quote

    assert arb.cash >= 0.0
    assert arb.cash <= initial_cash
    assert after_pool_quote - before_pool_quote <= initial_cash + 1e-7
    assert arb.assets >= 0.0


def test_arbitrage_wallet_is_shared_across_pools():
    pools = {
        'first': _pool('cpmm', 120.0),
        'second': _pool('cpmm', 120.0),
    }
    initial_assets = 10.0
    arb = AMMArbitrageur(
        _FixedCLOB(), pools, cash=0.0, assets=initial_assets,
        max_correction_pct=50.0, trade_fraction_cap=1.0,
        routing='all',
    )

    arb.arbitrage()

    sold = sum(
        trade['quantity'] for trade in arb.period_trades
        if trade['signed_quantity'] < 0.0
    )
    assert sold <= initial_assets + 1e-9
    assert arb.assets >= 0.0
    assert math.isclose(sold + arb.assets, initial_assets, abs_tol=1e-9)


def test_base_buffer_ratio_is_an_effective_capacity_parameter():
    empty_pool = _pool('cpmm', 120.0)
    funded_pool = _pool('cpmm', 120.0)
    empty = AMMArbitrageur(
        _FixedCLOB(), {'pool': empty_pool}, cash_buffer_ratio=0.0,
        asset_buffer_ratio=0.0, max_correction_pct=50.0,
        trade_fraction_cap=1.0,
    )
    funded = AMMArbitrageur(
        _FixedCLOB(), {'pool': funded_pool}, cash_buffer_ratio=0.0,
        asset_buffer_ratio=0.01, max_correction_pct=50.0,
        trade_fraction_cap=1.0,
    )

    empty.arbitrage()
    funded.arbitrage()

    assert empty.period_trades == []
    assert sum(t['quantity'] for t in funded.period_trades) > 0.0
    assert funded.assets >= 0.0
