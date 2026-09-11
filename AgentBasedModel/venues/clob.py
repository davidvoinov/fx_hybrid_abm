"""
CLOB venue wrapper — computes execution cost metrics over ExchangeAgent's
order book without modifying it (for quoting) and delegates execution.

Also provides ``ShadowCLOB`` — a synthetic, non-depleting book that
models hypothetical CLOB costs from an exogenous fair price S_t and
environment parameters (σ_t, c_t).  Use it in AMM-only scenarios where
a live CLOB is not viable but you still need cost benchmarks.

Metrics:
    - quoted spread (qspr)
    - effective spread (espr)
    - price impact
    - all-in execution cost  C_CLOB(Q)
    - depth by price level
    - Amihud illiquidity measure
"""

from __future__ import annotations
import math as _math
from typing import TYPE_CHECKING, Optional, List, Dict

if TYPE_CHECKING:
    from AgentBasedModel.agents.agents import ExchangeAgent
    from AgentBasedModel.environment.processes import MarketEnvironment


class CLOBVenue:
    """
    Read-only analytics layer on top of an :class:`ExchangeAgent` order book.

    Parameters
    ----------
    exchange : ExchangeAgent
        The underlying CLOB.
    env : MarketEnvironment, optional
        Stored for potential use by downstream analytics.
        Note: ``mid_price()`` always returns the book mid
        ``(best_bid + best_ask) / 2``, not ``env.fair_price``.
    f_broker : float
        Broker fee in **bps** (default 0).
    f_venue : float
        Exchange/venue fee in **bps** (default 0).
    """

    def __init__(self, exchange: 'ExchangeAgent',
                 env: 'MarketEnvironment' = None,
                 f_broker: float = 0.0, f_venue: float = 0.0):
        self.exchange = exchange
        self.env = env
        self.f_broker = f_broker
        self.f_venue = f_venue

    # ---- basic prices ----------------------------------------------------

    def mid_price(self) -> float:
        sp = self.exchange.spread()
        if sp is not None:
            return (sp['bid'] + sp['ask']) / 2.0

        bid_book = self.exchange.order_book.get('bid')
        ask_book = self.exchange.order_book.get('ask')
        if bid_book:
            return float(bid_book.first.price)
        if ask_book:
            return float(ask_book.first.price)

        cached_mid = getattr(self.exchange, '_last_book_mid_price', None)
        if cached_mid is not None and cached_mid > 0:
            return float(cached_mid)

        return 100.0

    def spread(self) -> Optional[dict]:
        return self.exchange.spread()

    def quoted_spread_bps(self) -> float:
        """qspr_t = 10 000 × (ask − bid) / mid"""
        sp = self.exchange.spread()
        if sp is None:
            return float('inf')
        mid = self.mid_price()
        if mid <= 0:
            return float('inf')
        return 10_000.0 * (sp['ask'] - sp['bid']) / mid

    # ---- walk the book (read-only) ---------------------------------------

    def _walk_book(self, Q: float, side: str = 'buy'):
        """
        Walk the order book for *Q* units (base) without executing.

        Returns (vwap, levels_consumed, remaining_qty).
        *levels_consumed* is a list of (price, fill_qty) tuples.
        """
        book_side = 'ask' if side == 'buy' else 'bid'
        total_cost = 0.0
        remaining = Q
        levels = []

        for order in self.exchange.order_book[book_side]:
            if remaining <= 1e-12:
                break
            fill = min(remaining, order.qty)
            total_cost += fill * order.price
            remaining -= fill
            levels.append((order.price, fill))

        if remaining > 0:
            return None, levels, remaining

        vwap = total_cost / Q
        return vwap, levels, 0.0

    # ---- cost quoting ----------------------------------------------------

    def quote_buy(self, Q: float) -> dict:
        """
        Quote buying *Q* base from the ask side.  Returns dict:
            exec_price, half_spread_bps, espr_bps, impact_bps,
            fee_bps, cost_bps
        """
        return self._quote(Q, 'buy')

    def quote_sell(self, Q: float) -> dict:
        """Quote selling *Q* base into the bid side."""
        return self._quote(Q, 'sell')

    def _quote(self, Q: float, side: str) -> dict:
        if Q <= 0:
            return self._zero_quote()

        mid = self.mid_price()
        vwap, levels, remaining = self._walk_book(Q, side)

        if vwap is None:
            return self._inf_quote()

        # Half effective spread (one-sided cost vs mid)
        if side == 'buy':
            half_spr = 10_000.0 * (vwap - mid) / mid
        else:
            half_spr = 10_000.0 * (mid - vwap) / mid

        # Effective spread (two-sided, standard definition)
        espr = 2.0 * half_spr

        # Price impact: estimate new mid after hypothetical execution
        impact_bps = self._estimate_impact_bps(Q, side, levels)

        fee_bps = self.f_broker + self.f_venue

        # All-in cost for venue comparison: half-spread + fee
        # (impact is logged separately as an externality metric)
        cost_bps = half_spr + fee_bps

        return dict(exec_price=vwap, half_spread_bps=half_spr,
                    espr_bps=espr, impact_bps=impact_bps,
                    fee_bps=fee_bps, cost_bps=cost_bps)

    def _estimate_impact_bps(self, Q: float, side: str,
                             levels: list) -> float:
        """
        Estimate post-trade mid shift.
        impact = 10 000 × (S_{t+} − S_t) / S_t  (signed for buy).
        """
        mid = self.mid_price()
        sp = self.exchange.spread()
        if sp is None:
            return 0.0

        book_side = 'ask' if side == 'buy' else 'bid'
        consumed_qty = sum(qty for _, qty in levels)
        remaining_in_levels = 0.0

        # Walk the book again to find new best price after consumption
        cursor_qty = 0.0
        new_best = None
        for order in self.exchange.order_book[book_side]:
            cursor_qty += order.qty
            if cursor_qty > consumed_qty:
                # This order is partially (or fully) remaining
                new_best = order.price
                break

        if new_best is None:
            # Book exhausted on this side
            return 0.0

        if side == 'buy':
            new_mid = (sp['bid'] + new_best) / 2.0
        else:
            new_mid = (new_best + sp['ask']) / 2.0

        return 10_000.0 * (new_mid - mid) / mid

    # ---- depth -----------------------------------------------------------

    def depth(self, n_levels: int = 5) -> dict:
        """
        Aggregate volume at the top *n_levels* distinct price levels on
        each side of the book.
        """
        result = {'bid': [], 'ask': []}
        for book_side in ('bid', 'ask'):
            price_vol: Dict[float, float] = {}
            for order in self.exchange.order_book[book_side]:
                price_vol.setdefault(order.price, 0.0)
                price_vol[order.price] += order.qty
            items = list(price_vol.items())[:n_levels]
            result[book_side] = items
        return result

    def total_depth(self, bps_from_mid: float = 100) -> dict:
        """Total volume within *bps_from_mid* of mid."""
        return self.total_depth_around(self.mid_price(), bps_from_mid)

    def total_depth_around(self, centre: float, bps: float = 100) -> dict:
        """Total volume within *bps* of an arbitrary centre.

        Depth has to be measured around the price the question is about. The
        book mid and the fair price separate after a fundamental shock, and a
        book that is two sided around the stale mid can have nothing at all on
        one side of the new fair price. Measuring around one centre while
        restoring around the other reported depletion that was a change of
        coordinates, and hid depletion that was real.
        """
        if centre is None or centre <= 0:
            return {'bid': 0.0, 'ask': 0.0, 'total': 0.0}
        lo = centre * (1 - bps / 10_000)
        hi = centre * (1 + bps / 10_000)
        # Both bounds, on both sides. Applying the far bound alone is harmless
        # while the centre is the book mid, because no bid can sit above it in
        # an uncrossed book. Around the fair price it is not, since after the
        # fundamental value falls the bids left behind sit far above the new
        # centre and would still count as depth beside it, so a corridor with
        # nothing in it reports size and the repair never fires.
        bid_vol = sum(o.qty for o in self.exchange.order_book['bid']
                      if lo <= o.price <= hi)
        ask_vol = sum(o.qty for o in self.exchange.order_book['ask']
                      if lo <= o.price <= hi)
        return {'bid': bid_vol, 'ask': ask_vol, 'total': bid_vol + ask_vol}

    # ---- Amihud illiquidity ----------------------------------------------

    @staticmethod
    def amihud(returns: List[float], volumes: List[float]) -> float:
        """
        Amihud illiquidity = mean(|r_t| / Vol_t).
        """
        if not returns or not volumes:
            return 0.0
        vals = []
        for r, v in zip(returns, volumes):
            if v > 0:
                vals.append(abs(r) / v)
        return sum(vals) / len(vals) if vals else 0.0

    # ---- helpers ---------------------------------------------------------

    def cost_bps(self, Q: float, side: str = 'buy') -> float:
        """Convenience: return scalar all-in cost in bps."""
        if side == 'buy':
            return self.quote_buy(Q)['cost_bps']
        return self.quote_sell(Q)['cost_bps']

    def _zero_quote(self):
        return dict(exec_price=self.mid_price(), half_spread_bps=0,
                    espr_bps=0, impact_bps=0, fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), half_spread_bps=float('inf'),
                    espr_bps=float('inf'), impact_bps=float('inf'),
                    fee_bps=self.f_broker + self.f_venue,
                    cost_bps=float('inf'))


# =====================================================================
#  ShadowCLOB — synthetic, non-depleting book for AMM-only scenarios
# =====================================================================

class ShadowCLOB:
    """
    Synthetic CLOB that never depletes.

    Instead of a live order book it models hypothetical CLOB metrics
    from an exogenous fair price S_t and environment-dependent spread /
    depth parameters:

        half_spread(Q) = ½ qspr_0 + λ Q        (linear impact model)
        qspr_0 = α₀ + α₁ σ_t + α₂ c_t          (static MM model)
        depth  = D₀ / (1 + κ σ_t)                (depth shrinks in stress)

    The object is a *drop-in replacement* for :class:`CLOBVenue` in the
    logger / arb / metric code — same method signatures.

    Parameters
    ----------
    env : MarketEnvironment
        Source of σ_t, c_t, and S_t (``fair_price``).
    alpha0, alpha1, alpha2 : float
        Spread model coefficients (same meaning as MarketMaker params).
    base_depth : float
        Depth in base units under normal conditions.
    impact_lambda : float
        Linear impact coefficient (bps per unit Q traded).
    """

    def __init__(self, env: 'MarketEnvironment', *,
                 alpha0: float = 1.5,
                 alpha1: float = 300.0,
                 alpha2: float = 500.0,
                 base_depth: float = 500.0,
                 impact_lambda: float = 0.15):
        self.env = env
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.base_depth = base_depth
        self.impact_lambda = impact_lambda

    # ---- basic prices ----------------------------------------------------

    def mid_price(self) -> float:
        """S_t from the environment (never fails)."""
        p = self.env.fair_price
        return p if p is not None else 100.0

    def spread(self) -> Optional[dict]:
        mid = self.mid_price()
        hs = self._half_spread_bps(0) / 10_000.0 * mid
        return {'bid': mid - hs, 'ask': mid + hs}

    def quoted_spread_bps(self) -> float:
        return 2.0 * self._half_spread_bps(0)

    # ---- cost quoting (synthetic) ----------------------------------------

    def quote_buy(self, Q: float) -> dict:
        return self._quote(Q, 'buy')

    def quote_sell(self, Q: float) -> dict:
        return self._quote(Q, 'sell')

    def _quote(self, Q: float, side: str) -> dict:
        if Q <= 0:
            return self._zero_quote()
        mid = self.mid_price()
        hs = self._half_spread_bps(Q)
        impact = self.impact_lambda * Q
        fee_bps = 0.0
        cost_bps = hs + fee_bps

        if side == 'buy':
            exec_price = mid * (1.0 + hs / 10_000.0)
        else:
            exec_price = mid * (1.0 - hs / 10_000.0)

        return dict(exec_price=exec_price,
                    half_spread_bps=hs,
                    espr_bps=2.0 * hs,
                    impact_bps=impact,
                    fee_bps=fee_bps,
                    cost_bps=cost_bps)

    # ---- depth -----------------------------------------------------------

    def total_depth(self, bps_from_mid: float = 100) -> dict:
        d = self._depth()
        return {'bid': d / 2, 'ask': d / 2, 'total': d}

    def depth(self, n_levels: int = 5) -> dict:
        """Synthetic depth — single level at best bid/ask."""
        sp = self.spread()
        d = self._depth() / 2
        return {
            'bid': [(sp['bid'], d)],
            'ask': [(sp['ask'], d)],
        }

    # ---- convenience -----------------------------------------------------

    def cost_bps(self, Q: float, side: str = 'buy') -> float:
        if side == 'buy':
            return self.quote_buy(Q)['cost_bps']
        return self.quote_sell(Q)['cost_bps']

    @staticmethod
    def amihud(returns: List[float], volumes: List[float]) -> float:
        if not returns or not volumes:
            return 0.0
        vals = [abs(r) / v for r, v in zip(returns, volumes) if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    # ---- internal --------------------------------------------------------

    def _half_spread_bps(self, Q: float) -> float:
        """Half quoted spread (bps) for size Q, given current σ, c."""
        sigma = self.env.sigma
        c = self.env.funding_cost
        base = self.alpha0 + self.alpha1 * sigma + self.alpha2 * c
        return max(0.5, base + self.impact_lambda * Q)

    def _depth(self) -> float:
        """Total depth in base units, regime-dependent.

        Depth shrinks with σ; calibrated to Mancini et al. (2009)
        observation of dramatic FX liquidity evaporation during crisis.
        """
        sigma = self.env.sigma
        return self.base_depth / (1.0 + 80.0 * sigma)

    def _zero_quote(self):
        mid = self.mid_price()
        return dict(exec_price=mid, half_spread_bps=0,
                    espr_bps=0, impact_bps=0, fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), half_spread_bps=float('inf'),
                    espr_bps=float('inf'), impact_bps=float('inf'),
                    fee_bps=0, cost_bps=float('inf'))
