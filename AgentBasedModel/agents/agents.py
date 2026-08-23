from __future__ import annotations

from AgentBasedModel.utils import Order, OrderList
from AgentBasedModel.utils.math import exp, mean
import random
import math as _math
from typing import TYPE_CHECKING, Optional, List, Dict, Union

if TYPE_CHECKING:
    from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
    from AgentBasedModel.venues.clob import CLOBVenue
    from AgentBasedModel.environment.processes import MarketEnvironment
    from AgentBasedModel.venues.amm import classify_trade_size as _classify_trade_size


def _draw_geometric_lifetime(rng: random.Random, median_ticks: float) -> int:
    """Draw an integer quote life whose distribution has the given median.

    A fixed three- or 290-tick cancellation date turns a field median into a
    point mass chosen by the modeller.  The least structured alternative is a
    constant per-tick cancellation hazard.  ``median_ticks`` parameterises
    that hazard; fills and stale-price amendments may still end the order
    sooner, so it is not itself the realised lifetime statistic.
    """
    median = max(1.0, float(median_ticks))
    lower_hazard = 1.0 - 0.5 ** (1.0 / median)
    # At the lower boundary F(median)=0.5 exactly, so finite samples flip
    # between the intended integer median and the next tick.  Use the middle
    # of the hazard interval for which the requested integer is the unique
    # discrete median.  For long dealer lives the adjustment is negligible;
    # for a three-second non-bank life it prevents a purely numerical 3/4
    # coin flip.
    if median > 1.0:
        upper_hazard = 1.0 - 0.5 ** (1.0 / (median - 1.0))
    else:
        upper_hazard = 1.0
    hazard = 0.5 * (lower_hazard + upper_hazard)
    u = min(1.0 - 1e-15, max(0.0, rng.random()))
    return max(1, int(_math.ceil(_math.log1p(-u) / _math.log1p(-hazard))))


def _stamp_lifecycle_order(owner, order: Order, median_ticks: float,
                           reference_mid: Optional[float] = None,
                           quote_level: Optional[int] = None) -> Order:
    """Give a new quote its own clock and the fields needed for an audit."""
    env = getattr(owner, 'env', None)
    created = int(getattr(env, '_t', 0)) if env is not None else 0
    rng = getattr(owner, '_lifecycle_rng', None)
    if rng is None:
        # One draw from the globally seeded stream gives every firm a stable,
        # independent lifecycle stream without coupling later draws to agent
        # shuffle order.
        rng = random.Random(random.getrandbits(64))
        owner._lifecycle_rng = rng
    scheduled = _draw_geometric_lifetime(rng, median_ticks)
    order._created_tick = created
    order._initial_qty = float(getattr(order, 'qty', 0.0))
    order._scheduled_lifetime = scheduled
    order.ttl = scheduled
    if reference_mid is not None:
        order._quote_reference_mid = float(reference_mid)
    if quote_level is not None:
        order._quote_level = int(quote_level)
    return order


def _record_lifecycle_end(owner, order: Order, reason: Optional[str] = None) -> None:
    """Record one terminal event, retaining the legacy lifetime series."""
    if getattr(order, '_lifetime_recorded', False):
        return
    created = getattr(order, '_created_tick', None)
    env = getattr(owner, 'env', None)
    if created is None or env is None:
        return
    ended = int(getattr(env, '_t', created))
    lifetime = float(max(0, ended - int(created)))
    if reason is None:
        if float(getattr(order, 'qty', 0.0)) <= 1e-12:
            reason = 'fill'
        elif getattr(order, 'ttl', None) is not None and order.ttl <= 0:
            reason = 'scheduled_cancel'
        else:
            reason = 'cancel'
    initial = float(getattr(order, '_initial_qty', getattr(order, 'qty', 0.0)))
    remaining = max(0.0, float(getattr(order, 'qty', 0.0)))
    events = getattr(owner, 'order_lifecycle_events', None)
    if events is None:
        events = []
        owner.order_lifecycle_events = events
    events.append({
        'order_id': int(getattr(order, 'order_id', -1)),
        'owner_type': str(getattr(owner, 'type', type(owner).__name__)),
        'side': str(getattr(order, 'order_type', 'unknown')),
        'created_tick': int(created),
        'ended_tick': ended,
        'lifetime': lifetime,
        'reason': str(reason),
        'initial_qty': initial,
        'remaining_qty': remaining,
        'filled_qty': max(0.0, initial - remaining),
        'scheduled_lifetime': getattr(order, '_scheduled_lifetime', None),
        'censored': False,
    })
    owner.completed_order_lifetimes.append(lifetime)
    order._lifetime_recorded = True


def _lifecycle_observations(owner, include_live: bool = True) -> List[dict]:
    """Return completed events plus non-mutating right-censored exposures."""
    rows = [dict(row) for row in getattr(owner, 'order_lifecycle_events', [])]
    if not include_live:
        return rows
    env = getattr(owner, 'env', None)
    now = int(getattr(env, '_t', 0)) if env is not None else 0
    seen = {row.get('order_id') for row in rows}
    for order in list(getattr(owner, 'orders', []) or []):
        if getattr(order, '_lifetime_recorded', False):
            continue
        created = getattr(order, '_created_tick', None)
        order_id = int(getattr(order, 'order_id', -1))
        if created is None or order_id in seen:
            continue
        initial = float(getattr(order, '_initial_qty', getattr(order, 'qty', 0.0)))
        remaining = max(0.0, float(getattr(order, 'qty', 0.0)))
        rows.append({
            'order_id': order_id,
            'owner_type': str(getattr(owner, 'type', type(owner).__name__)),
            'side': str(getattr(order, 'order_type', 'unknown')),
            'created_tick': int(created),
            'ended_tick': None,
            'lifetime': float(max(0, now - int(created))),
            'reason': 'censored',
            'initial_qty': initial,
            'remaining_qty': remaining,
            'filled_qty': max(0.0, initial - remaining),
            'scheduled_lifetime': getattr(order, '_scheduled_lifetime', None),
            'censored': True,
        })
    return rows


class BackgroundLiquidityAccount:
    """Explicit external-sector balance sheet behind anonymous seed orders."""

    def __init__(self):
        self.type = 'BackgroundLiquidity'
        self.cash = 0.0
        self.assets = 0.0
        self.executed_volume = 0.0

    def apply_fill(self, order: Order, fill_qty: float, fill_price: float,
                   t_cost: float, is_buy: bool, qty_before: float):
        gross = float(fill_price) * float(fill_qty)
        if is_buy:
            self.cash -= gross * (1.0 + t_cost)
            self.assets += fill_qty
        else:
            self.cash += gross * (1.0 - t_cost)
            self.assets -= fill_qty
        self.executed_volume += float(fill_qty)


class ExchangeAgent:
    """
    ExchangeAgent implements automatic orders handling within the order book. It supports limit orders,
    market orders, cancel orders, returns current spread prices and volumes.
    """
    id = 0

    def __init__(self, price: Union[float, int] = 100, std: Union[float, int] = 25, volume: int = 1000,
                 rf: float = 5e-4, transaction_cost: float = 0,
                 price_tick: float = 0.005,
                 background_corridor_bps: float = 25.0,
                 background_target_ratio: float = 1.0,
                 background_max_share_of_trader_depth: float = 1.0,
                 anchor_strength: float = 1.0,
                 anchor_threshold_bps: float = 0.0):
        """
        Initialization parameters
        :param price: stock initial price
        :param std: standard deviation of order prices in book
        :param volume: number of orders in book
        :param rf: risk-free rate (interest rate for cash holdings of agents)
        :param transaction_cost: cost that is paid on each successful deal
        """
        self.name = f'ExchangeAgent{self.id}'
        ExchangeAgent.id += 1

        self.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
        self.dividend_book = list()  # list of future dividends
        self.risk_free = rf
        self.transaction_cost = transaction_cost
        self._seed_price = float(price)
        self._seed_std = float(std)
        self._seed_volume = int(volume)
        self._price_tick = max(1e-4, float(price_tick))
        self._seed_qty_lo = max(1, volume // 500)
        self._seed_qty_hi = max(5, volume // 100)
        self._background_corridor_bps = max(5.0, float(background_corridor_bps))
        self._background_target_ratio = max(1.0, float(background_target_ratio))
        # Keep calm near-mid background support from collapsing with transient trader-depth decay.
        self._background_floor_ratio = 0.71
        # How far into the corridor the backstop opens, as a fraction
        # of the corridor. It stands behind the book's own quotes so
        # that it cannot set the touch.
        self._background_standoff_ratio = 0.5
        self._background_max_share_of_trader_depth = max(0.0, float(background_max_share_of_trader_depth))
        self._anchor_strength = max(0.0, min(1.0, float(anchor_strength)))
        self._anchor_threshold_bps = max(0.0, float(anchor_threshold_bps))
        self._last_book_mid_price = float(price)
        self.background_account = BackgroundLiquidityAccount()
        self._fill_book(price, std, volume, rf * price)
        background_qty = self._background_qty_near_mid(
            float(price), self._background_corridor_bps
        )
        self._background_target_qty = {
            side: max(1.0, qty * self._background_target_ratio)
            for side, qty in background_qty.items()
        }

    def _background_order(self, price: float, qty: float, side: str) -> Order:
        order = Order(price, qty, side, None)
        if not hasattr(self, 'background_account'):
            # Some legacy tests construct an exchange via ``__new__``. Keep
            # even those books conservative and not silently returning to
            # unaccounted fills.
            self.background_account = BackgroundLiquidityAccount()
        order.settlement_account = self.background_account
        return order

    def generate_dividend(self):
        """
        Generate time series on future dividends.
        """
        # Generate future dividend
        d = self.dividend_book[-1] * self._next_dividend()
        self.dividend_book.append(max(d, 0))  # dividend > 0
        self.dividend_book.pop(0)

    def expire_orders(self, fair_price: float = None):
        """Remove only truly expiring orders.

        TTL-tagged orders are short-lived by construction and should roll
        off the book naturally. The seeded anonymous book, however,
        represents persistent background liquidity; removing it simply
        because fair value drifted was collapsing the default CLOB depth.
        """

        for side in ('bid', 'ask'):
            expired = []
            for order in self.order_book[side]:
                # TTL-based expiry
                if order.ttl is not None:
                    if order.ttl <= 0:
                        expired.append(order)
                    else:
                        order.ttl -= 1
            for order in expired:
                if order.trader is not None and hasattr(order.trader, 'orders'):
                    if hasattr(order.trader, 'record_order_end'):
                        order.trader.record_order_end(order, reason='scheduled_cancel')
                    try:
                        order.trader.orders.remove(order)
                    except ValueError:
                        pass
                    if hasattr(order.trader, 'last_quoted'):
                        order.trader.last_quoted = bool(order.trader.orders)
                    if hasattr(order.trader, 'release_order_reservation'):
                        order.trader.release_order_reservation(order)
                self.order_book[side].remove(order)

    def rebuild_order_book(self):
        """Restore linked-list ordering without amending owned quotes.

        Background orders are an exchange-side scaffold and may be moved when
        a re-anchor leaves them crossed.  An order submitted by an agent must
        instead keep its price until that agent cancels/replaces it; changing
        it here preserves an invalid queue timestamp and desynchronises its
        cash reservation from its price.
        """
        def _sort_side(side):
            orders = list(self.order_book[side])
            orders.sort(key=lambda order: order.price, reverse=(side == 'bid'))

            rebuilt = OrderList(side)
            for order in orders:
                order.left = None
                order.right = None
                rebuilt.append(order)
            self.order_book[side] = rebuilt

        for side in ('bid', 'ask'):
            _sort_side(side)

        if self.order_book['bid'] and self.order_book['ask']:
            best_bid = self.order_book['bid'].first.price
            best_ask = self.order_book['ask'].first.price

            if best_bid >= best_ask:
                tick = self._tick_size()
                owned_bids = [o.price for o in self.order_book['bid'] if o.trader is not None]
                owned_asks = [o.price for o in self.order_book['ask'] if o.trader is not None]

                if owned_bids and owned_asks:
                    anchor = 0.5 * (max(owned_bids) + min(owned_asks))
                else:
                    anchor = 0.5 * (best_bid + best_ask)
                target_bid = self.floor_price(anchor - 0.5 * tick)
                target_ask = self.ceil_price(anchor + 0.5 * tick)
                if owned_asks:
                    target_bid = min(target_bid, self.floor_price(min(owned_asks) - tick))
                if owned_bids:
                    target_ask = max(target_ask, self.ceil_price(max(owned_bids) + tick))

                changed = False
                for order in self.order_book['bid']:
                    if order.trader is None and order.price > target_bid:
                        order.price = target_bid
                        changed = True
                for order in self.order_book['ask']:
                    if order.trader is None and order.price < target_ask:
                        order.price = target_ask
                        changed = True
                if changed:
                    _sort_side('bid')
                    _sort_side('ask')

                if (self.order_book['bid'] and self.order_book['ask']
                        and self.order_book['bid'].first.price >= self.order_book['ask'].first.price):
                    # Normal order entry matches crossing owned quotes, so this
                    # can only be caused by another in-place mutation.  Keep a
                    # visible invariant counter and not silently rewriting
                    # an owner's order to conceal it.
                    self.crossed_owned_book_events = getattr(
                        self, 'crossed_owned_book_events', 0
                    ) + 1

    def _tick_size(self) -> float:
        return max(1e-4, float(getattr(self, '_price_tick', 0.01)))

    def round_price(self, price: float) -> float:
        tick = self._tick_size()
        return round(round(float(price) / tick) * tick, 3)

    def floor_price(self, price: float) -> float:
        tick = self._tick_size()
        return round(_math.floor(float(price) / tick + 1e-12) * tick, 3)

    def ceil_price(self, price: float) -> float:
        tick = self._tick_size()
        return round(_math.ceil(float(price) / tick - 1e-12) * tick, 3)

    def _background_qty_near_mid(self, reference_price: float, corridor_bps: float) -> dict:
        if reference_price <= 0:
            return {'bid': 0.0, 'ask': 0.0}

        lo = reference_price * (1.0 - corridor_bps / 10_000.0)
        hi = reference_price * (1.0 + corridor_bps / 10_000.0)
        return {
            'bid': sum(
                order.qty for order in self.order_book['bid']
                if order.trader is None and lo <= order.price <= hi
            ),
            'ask': sum(
                order.qty for order in self.order_book['ask']
                if order.trader is None and lo <= order.price <= hi
            ),
        }

    def _trader_qty_near_mid(self, reference_price: float, corridor_bps: float) -> dict:
        if reference_price <= 0:
            return {'bid': 0.0, 'ask': 0.0}

        lo = reference_price * (1.0 - corridor_bps / 10_000.0)
        hi = reference_price * (1.0 + corridor_bps / 10_000.0)
        return {
            'bid': sum(
                order.qty for order in self.order_book['bid']
                if order.trader is not None and lo <= order.price <= hi
            ),
            'ask': sum(
                order.qty for order in self.order_book['ask']
                if order.trader is not None and lo <= order.price <= hi
            ),
        }

    def _draw_background_price(self, side: str, reference_price: float,
                               corridor_bps: float,
                               aggressiveness: float = 1.0) -> float:
        """Price for one anonymous backstop order.

        The offset opens at a fraction of the corridor and not at one
        tick.  Anchoring it to the grid made this scaffold the tightest
        quote in the book: it took the best price whenever the book's own
        participants were anywhere but the minimum increment, and since the
        best price is a minimum over many draws it settled on the floor, so
        the realised spread of the whole market was the price grid plus a
        thin tail instead of anything a participant decided.  Measured
        across grids from two basis points down to nine hundredths, the book
        sat at exactly one tick in about nine periods in ten.

        A backstop exists so that a side of the book is not empty.  It is
        not a participant competing for the touch, and it should not be the
        thing that sets the price of a market, so it stands well behind
        whoever is quoting and only shows where nobody else does.
        """
        tick = self._tick_size()
        max_offset = max(tick, reference_price * corridor_bps / 10_000.0)
        aggressiveness = max(0.25, min(1.0, float(aggressiveness)))
        widen_mult = 1.0 + 2.5 * (1.0 - aggressiveness)
        min_offset = max(tick * widen_mult,
                         max_offset * self._background_standoff_ratio)
        scale = max(tick, max_offset / (3.0 * aggressiveness))
        offset = min(max_offset, min_offset + random.expovariate(1.0 / scale))

        if side == 'bid':
            return self.round_price(reference_price - offset)
        return self.round_price(reference_price + offset)

    def _draw_background_price_outside_corridor(self, side: str,
                                                reference_price: float,
                                                corridor_bps: float,
                                                aggressiveness: float = 1.0) -> float:
        tick = self._tick_size()
        aggressiveness = max(0.25, min(1.0, float(aggressiveness)))
        widen_mult = 1.0 + 1.5 * (1.0 - aggressiveness)
        min_offset = max(tick, reference_price * corridor_bps / 10_000.0 + tick * widen_mult)
        scale = max(tick, min_offset / (2.0 * aggressiveness))
        offset = min_offset + random.expovariate(1.0 / scale)

        if side == 'bid':
            return self.round_price(reference_price - offset)
        return self.round_price(reference_price + offset)

    def set_background_depth_target(self, reference_price: float,
                                    total_target_qty: float,
                                    corridor_bps: float = None,
                                    rebalance: bool = True,
                                    respect_trader_cap: bool = True):
        if reference_price is None or reference_price <= 0:
            return

        corridor_bps = corridor_bps or self._background_corridor_bps
        per_side = max(1.0, float(total_target_qty) / 2.0)
        self._background_target_qty = {'bid': per_side, 'ask': per_side}

        if rebalance:
            self.rebalance_background_liquidity(
                reference_price,
                corridor_bps=corridor_bps,
                target_ratio=1.0,
                respect_trader_cap=respect_trader_cap,
            )

    def rebalance_background_liquidity(self, reference_price: float,
                                       corridor_bps: float = None,
                                       target_ratio: float = 1.0,
                                       respect_trader_cap: bool = True):
        if reference_price is None or reference_price <= 0:
            return

        corridor_bps = corridor_bps or self._background_corridor_bps
        effective_aggressiveness = max(0.25, min(1.0, float(target_ratio)))
        trader_near_qty = self._trader_qty_near_mid(reference_price, corridor_bps)
        targets = {}
        for side in ('bid', 'ask'):
            scaled_target = max(1.0, self._background_target_qty.get(side, 0.0) * target_ratio)
            if respect_trader_cap:
                # The floor fades out as the book's own participants arrive.
                # It used to survive at a fixed fraction of the target
                # whatever they supplied, which made this scaffold a standing
                # presence and not a backstop: it held about half of the
                # best prices through a crisis and after it, while owning no
                # capital, bearing no inventory limit and being unable to
                # withdraw. It exists for a book that is empty, and an empty
                # book is what it is now measured against.
                trader_qty = max(0.0, trader_near_qty.get(side, 0.0))
                coverage = min(1.0, trader_qty / max(scaled_target, 1e-9))
                floor = scaled_target * self._background_floor_ratio * (1.0 - coverage)
                scaled_target = min(
                    scaled_target,
                    floor
                    + self._background_max_share_of_trader_depth * trader_qty
                )
            targets[side] = scaled_target
        changed = False

        lo = reference_price * (1.0 - corridor_bps / 10_000.0)
        hi = reference_price * (1.0 + corridor_bps / 10_000.0)

        for side in ('bid', 'ask'):
            near_qty = 0.0
            near_orders = []
            far_orders = []

            for order in self.order_book[side]:
                if order.trader is not None:
                    continue

                # A corridor is a bounded interval regardless of side.  With
                # only the far-side bound, a bid left above a newly lowered
                # fair value (or an ask below a newly raised one) was counted
                # as depth beside the new fair value.  The repair then saw no
                # gap to fill and could leave that side exactly empty.
                wrong_side = (
                    (side == 'bid' and order.price >= reference_price)
                    or (side == 'ask' and order.price <= reference_price)
                )
                in_corridor = (not wrong_side) and lo <= order.price <= hi
                if in_corridor:
                    near_qty += order.qty
                    near_orders.append(order)
                else:
                    # A large fair-value move can strand anonymous bids above
                    # the new reference (or asks below it).  Leaving the
                    # unused part there after a partial move makes the book
                    # crossed; rebuild_order_book then shifts the freshly
                    # repaired levels away from the corridor again.  Re-home
                    # such stale support on its economically valid side before
                    # using it as the reservoir for the near-mid refill.
                    if wrong_side:
                        order.price = self._draw_background_price_outside_corridor(
                            side,
                            reference_price,
                            corridor_bps,
                            aggressiveness=effective_aggressiveness,
                        )
                        changed = True
                    far_orders.append(order)

            near_orders.sort(key=lambda order: abs(order.price - reference_price), reverse=True)
            far_orders.sort(key=lambda order: abs(order.price - reference_price), reverse=True)

            while near_qty > targets[side] and near_orders:
                order = near_orders.pop(0)
                excess = near_qty - targets[side]
                move_qty = min(order.qty, excess)
                if move_qty <= 0:
                    break
                if move_qty + 1e-9 < order.qty:
                    order.qty -= move_qty
                    moved = self._background_order(
                        self._draw_background_price_outside_corridor(
                            side,
                            reference_price,
                            corridor_bps,
                            aggressiveness=effective_aggressiveness,
                        ),
                        move_qty,
                        side,
                    )
                    self.order_book[side].append(moved)
                else:
                    order.price = self._draw_background_price_outside_corridor(
                        side,
                        reference_price,
                        corridor_bps,
                        aggressiveness=effective_aggressiveness,
                    )
                near_qty -= move_qty
                changed = True

            while near_qty < targets[side] and far_orders:
                order = far_orders.pop(0)
                gap = targets[side] - near_qty
                move_qty = min(order.qty, gap)
                if move_qty <= 0:
                    break
                if move_qty + 1e-9 < order.qty:
                    order.qty -= move_qty
                    moved = self._background_order(
                        self._draw_background_price(
                            side,
                            reference_price,
                            corridor_bps,
                            aggressiveness=effective_aggressiveness,
                        ),
                        move_qty,
                        side,
                    )
                    self.order_book[side].append(moved)
                else:
                    order.price = self._draw_background_price(
                        side,
                        reference_price,
                        corridor_bps,
                        aggressiveness=effective_aggressiveness,
                    )
                near_qty += move_qty
                changed = True

            while near_qty < targets[side]:
                gap = targets[side] - near_qty
                qty = min(random.randint(self._seed_qty_lo, self._seed_qty_hi), gap)
                if qty <= 0:
                    break
                price = self._draw_background_price(
                    side,
                    reference_price,
                    corridor_bps,
                    aggressiveness=effective_aggressiveness,
                )
                self.order_book[side].append(self._background_order(price, qty, side))
                near_qty += qty
                changed = True

        if changed:
            self.rebuild_order_book()

    def recenter_book(self, fair_price: float, reprice_prob: float = 0.6,
                      reprice_noise_bps: float = 0.5,
                      anchor_strength: float = None,
                      min_reprice_gap_bps: float = None,
                      background_target_ratio: float = 1.0):
        """Probabilistically shift resting orders toward *fair_price*.

        The book closes only a fraction of the gap to the anchor on each
        tick, creating a realistic lag between latent fair value and the
        observed CLOB mid.

        Each anonymous background order reprices with probability
        *reprice_prob*, creating a natural lag between fair-price moves and
        book adjustment. Trader-owned orders are left to their owners: an
        exchange-side in-place amendment both bypasses the agent's lifecycle
        rule and incorrectly preserves price-time priority. Repriced
        background orders receive a small noise term to prevent artificial
        clustering.

        Parameters
        ----------
        fair_price : float
            Target mid-price (from GBM or post-shock).
        reprice_prob : float
            Per-background-order probability of adjusting this tick (0–1).
        reprice_noise_bps : float
            Std of Gaussian noise added to repriced orders (in bps
            of *fair_price*).
        anchor_strength : float or None
            Fraction of the gap between the current book mid and
            *fair_price* closed this tick.  If None, uses the exchange
            default.
        min_reprice_gap_bps : float or None
            Ignore tiny deviations below this threshold.  If None,
            uses the exchange default.
        """
        if fair_price is None or fair_price <= 0:
            return
        sp = self.spread()
        if sp is None:
            # After a shock cancel-wave one side can be empty. Re-seed the
            # anonymous background around fair value so the book can recover.
            self.rebalance_background_liquidity(fair_price, target_ratio=background_target_ratio)
            return
        book_mid = (sp['bid'] + sp['ask']) / 2.0
        anchor_strength = self._anchor_strength if anchor_strength is None else max(0.0, min(1.0, anchor_strength))
        min_reprice_gap_bps = (
            self._anchor_threshold_bps
            if min_reprice_gap_bps is None else max(0.0, min_reprice_gap_bps)
        )
        gap_bps = abs(fair_price - book_mid) / max(book_mid, 1e-9) * 10_000.0
        if anchor_strength <= 0.0:
            self.rebalance_background_liquidity(book_mid, target_ratio=background_target_ratio)
            return
        if gap_bps < min_reprice_gap_bps:
            self.rebalance_background_liquidity(book_mid, target_ratio=background_target_ratio)
            return
        target_mid = book_mid + (fair_price - book_mid) * anchor_strength
        dp = target_mid - book_mid
        noise_std = max(target_mid, 1e-9) * reprice_noise_bps / 10_000.0
        # The standoff has to survive repricing. It is applied where a
        # backstop price is drawn, but this path does not draw one: it shifts
        # an existing order by a delta. After a large move in the fundamental
        # the shifted orders landed wherever the arithmetic put them, which
        # included the touch, and the backstop went back to setting the best
        # price in the crisis it is supposed to stand behind.
        # Some legacy tests build an exchange via ``__new__``, so these are
        # read defensively in the same way the order factory reads its own.
        standoff = max(
            self._tick_size(),
            target_mid
            * getattr(self, '_background_corridor_bps', 25.0) / 10_000.0
            * getattr(self, '_background_standoff_ratio', 0.5),
        )
        repriced = False
        for side in ('bid', 'ask'):
            for order in self.order_book[side]:
                if order.trader is None and random.random() < reprice_prob:
                    noise = random.gauss(0, noise_std) if noise_std > 0 else 0.0
                    price = order.price + dp + noise
                    if side == 'bid':
                        price = min(price, target_mid - standoff)
                    else:
                        price = max(price, target_mid + standoff)
                    order.price = self.round_price(price)
                    repriced = True

        if repriced:
            self.rebuild_order_book()
            self.rebalance_background_liquidity(target_mid, target_ratio=background_target_ratio)

    def cancel_wave(self, cancel_frac: float = 0.5, near_touch: bool = True):
        """Cancel a fraction of resting orders (liquidity crisis).

        Parameters
        ----------
        cancel_frac : float
            Fraction of orders to cancel (0–1).
        near_touch : bool
            If True, priority is given to orders near the top of book
            (most aggressive), which is realistic — market-makers and
            aggressive limit orders withdraw first in a crisis.
        """
        for side in ('bid', 'ask'):
            orders = list(self.order_book[side])
            if not orders:
                continue
            n_cancel = max(1, int(len(orders) * cancel_frac))
            if near_touch:
                # Cancel from the top (most aggressive) first
                to_cancel = orders[:n_cancel]
            else:
                to_cancel = random.sample(orders, min(n_cancel, len(orders)))
            for order in to_cancel:
                if order.trader is not None and hasattr(order.trader, 'orders'):
                    if hasattr(order.trader, 'record_order_end'):
                        order.trader.record_order_end(
                            order, reason='scenario_cancel_wave'
                        )
                    try:
                        order.trader.orders.remove(order)
                    except ValueError:
                        pass
                    if hasattr(order.trader, 'release_order_reservation'):
                        order.trader.release_order_reservation(order)
                self.order_book[side].remove(order)

    def _fill_book(self, price, std, volume, div: float = 0.05):
        """
        Fill order book with random orders. Fill dividend book with n future dividends.
        """
        # Order book — qty range scales with volume for deeper books
        qty_lo = max(1, volume // 500)
        qty_hi = max(5, volume // 100)
        prices1 = [self.round_price(random.normalvariate(price - std, std)) for _ in range(volume // 2)]
        prices2 = [self.round_price(random.normalvariate(price + std, std)) for _ in range(volume // 2)]
        quantities = [random.randint(qty_lo, qty_hi) for _ in range(volume)]

        for (p, q) in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = self._background_order(self.round_price(p), q, 'ask')
                self.order_book['ask'].append(order)
            else:
                order = self._background_order(self.round_price(p), q, 'bid')
                self.order_book['bid'].push(order)

        # Dividend book
        for i in range(100):
            self.dividend_book.append(max(div, 0))  # dividend > 0
            div *= self._next_dividend()

    def _clear_book(self):
        """
        Clears glass from orders with 0 qty.

        complexity O(n)

        :return: void
        """
        self.order_book['bid'] = OrderList.from_list([order for order in self.order_book['bid'] if order.qty > 0])
        self.order_book['ask'] = OrderList.from_list([order for order in self.order_book['ask'] if order.qty > 0])

    def spread(self) -> Optional[dict]:
        """
        :return: {'bid': float, 'ask': float}
        """
        if self.order_book['bid'] and self.order_book['ask']:
            bid = self.order_book['bid'].first.price
            ask = self.order_book['ask'].first.price
            self._last_book_mid_price = 0.5 * (bid + ask)
            return {'bid': bid, 'ask': ask}
        return None

    def spread_volume(self) -> Optional[dict]:
        """
        :return: {'bid': float, 'ask': float}
        """
        if self.order_book['bid'] and self.order_book['ask']:
            return {'bid': self.order_book['bid'].first.qty, 'ask': self.order_book['ask'].first.qty}
        return None

    def price(self) -> Optional[float]:
        spread = self.spread()
        if spread:
            return round((spread['bid'] + spread['ask']) / 2, 1)
        raise Exception(f'Price cannot be determined, since no orders either bid or ask')

    def dividend(self, access: int = None) -> Union[list, float]:
        """
        Returns current dividend payment value. If called by a trader, returns n future dividends
        given information access.
        """
        if access is None:
            return self.dividend_book[0]
        return self.dividend_book[:access]

    @classmethod
    def _next_dividend(cls, std=5e-3):
        return exp(random.normalvariate(0, std))

    def limit_order(self, order: Order):
        """
        Executes limit order, fulfilling orders if on other side of spread

        :return: void
        """
        if order.trader is not None and hasattr(order.trader, 'prepare_limit_order'):
            if not order.trader.prepare_limit_order(order, self.transaction_cost):
                return False

        sp = self.spread()
        if sp is None:
            # Book empty — insert directly (no matching possible)
            if order.order_type == 'bid':
                self.order_book['bid'].insert(order)
            elif order.order_type == 'ask':
                self.order_book['ask'].insert(order)
            return True

        bid, ask = sp.values()
        t_cost = self.transaction_cost
        if not bid or not ask:
            return False

        if order.order_type == 'bid':
            if order.price >= ask:
                order = self.order_book['ask'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['bid'].insert(order)
            return True

        elif order.order_type == 'ask':
            if order.price <= bid:
                order = self.order_book['bid'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['ask'].insert(order)
            return True
        return False

    def market_order(self, order: Order) -> Order:
        """
        Executes market order, fulfilling orders on the other side of spread

        :return: Order
        """
        t_cost = self.transaction_cost
        if order.order_type == 'bid':
            order = self.order_book['ask'].fulfill(order, t_cost)
        elif order.order_type == 'ask':
            order = self.order_book['bid'].fulfill(order, t_cost)
        return order

    def cancel_order(self, order: Order):
        """
        Cancel order from order book

        :return: void
        """
        if order.order_type == 'bid':
            self.order_book['bid'].remove(order)
        elif order.order_type == 'ask':
            self.order_book['ask'].remove(order)
        if order.trader is not None and hasattr(order.trader, 'release_order_reservation'):
            order.trader.release_order_reservation(order)


class Trader:
    id = 0

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 venue_choice_rule: str = 'fixed_share',
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 routing_cost_scale_bps: float = 4.0,
                 routing_prior_mix_cap: float = 0.18,
                 routing_basis_scale_bps: float = 50.0,
                 routing_clob_depth_multiple: float = 10.0,
                 routing_amm_depth_multiple: float = 8.0,
                 routing_rng: Optional[random.Random] = None,
                 max_cash_borrow: float = 0.0,
                 max_short_assets: float = 0.0):
        """
        Trader that is activated on call to perform action.

        :param market: link to exchange agent
        :param cash: trader's cash available
        :param assets: trader's number of shares hold
        :param clob: CLOBVenue wrapper (enables multi-venue mode)
        :param amm_pools: dict of AMM pools (enables multi-venue mode)
        :param env: MarketEnvironment for σ_t, c_t
        :param amm_share_pct: AMM prior (0–100); a hard probability only under fixed_share
        :param venue_choice_rule: routing regime, either fixed_share or liquidity_aware
        :param deterministic_venue: if True, use argmin instead of logit
        :param beta_amm: logit sensitivity for intra-AMM choice (CPMM vs HFMM)
        :param cpmm_bias_bps: non-monetary utility discount applied to CPMM cost (bps)
        :param cost_noise_std: std of Gaussian noise added to AMM cost estimates (bps)
        :param routing_cost_scale_bps: cost-gap scale in the liquidity-aware logit
        :param routing_prior_mix_cap: maximum weight placed on ``amm_share_pct``
        :param routing_basis_scale_bps: basis scale of the AMM alignment penalty
        :param routing_clob_depth_multiple: CLOB depth needed per requested unit
        :param routing_amm_depth_multiple: AMM depth needed per requested unit
        """
        self.type = 'Unknown'
        self.name = f'Trader{self.id}'
        self.id = Trader.id
        Trader.id += 1

        self.market = market
        self.orders = list()

        self.cash = cash
        self.assets = assets
        self.max_cash_borrow = max(0.0, float(max_cash_borrow))
        self.max_short_assets = max(0.0, float(max_short_assets))
        self.reserved_cash = 0.0
        self.reserved_assets = 0.0
        self.maintenance_margin_ratio = 0.0
        self.liquidation_fraction = 0.5
        self.borrow_spread_multiplier = 1.0
        self.short_borrow_spread_multiplier = 1.25
        self.defaulted = False
        self.default_reason: Optional[str] = None
        self.last_financing_charge = 0.0
        self.cumulative_financing_charge = 0.0

        # Multi-venue (optional)
        self.clob = clob
        self.amm_pools = amm_pools or {}
        self.env = env
        self.amm_share_pct = max(0.0, min(100.0, amm_share_pct))
        self.venue_choice_rule = (
            venue_choice_rule if venue_choice_rule in {'fixed_share', 'liquidity_aware'}
            else 'fixed_share'
        )
        self.deterministic_venue = deterministic_venue
        self.beta_amm = beta_amm
        self.cpmm_bias_bps = cpmm_bias_bps
        self.cost_noise_std = cost_noise_std
        self.routing_cost_scale_bps = max(1e-9, float(routing_cost_scale_bps))
        self.routing_prior_mix_cap = max(0.0, min(1.0, float(routing_prior_mix_cap)))
        self.routing_basis_scale_bps = max(1e-9, float(routing_basis_scale_bps))
        self.routing_clob_depth_multiple = max(1e-9, float(routing_clob_depth_multiple))
        self.routing_amm_depth_multiple = max(1e-9, float(routing_amm_depth_multiple))
        # Routing is an endogenous treatment channel.  Its random draws must
        # not advance the stream that generates customer arrival, direction
        # and size in paired AMM/no-AMM or routing-sensitivity comparisons.
        # Direct/legacy callers retain the historical module-level stream;
        # ``Simulator.default_fx`` supplies a dedicated seeded stream.
        self._routing_rng = routing_rng if routing_rng is not None else random
        self.trades: List[dict] = []
        # The routed request this customer made on the current tick, whether
        # or not it filled.  Read and cleared by the simulation loop.
        self.last_routing_attempt: Optional[dict] = None

    @property
    def multi_venue(self) -> bool:
        """True if trader operates in FX multi-venue mode (even CLOB-only)."""
        return self.clob is not None

    def __str__(self) -> str:
        return f'{self.name} ({self.type})'

    def equity(self, reference_price: Optional[float] = None):
        price = reference_price
        if price is None:
            price = self.market.price() if self.market.price() is not None else 0
        return self.cash + self.assets * price

    def gross_exposure(self, reference_price: Optional[float] = None) -> float:
        price = reference_price
        if price is None:
            price = self.market.price() if self.market.price() is not None else 0.0
        price = max(0.0, float(price))
        borrowed_cash = max(0.0, -self.cash)
        return borrowed_cash + abs(self.assets) * price

    def maintenance_margin_required(self, reference_price: Optional[float] = None) -> float:
        return max(0.0, self.maintenance_margin_ratio) * self.gross_exposure(reference_price)

    def has_margin_breach(self, reference_price: Optional[float] = None) -> bool:
        return self.equity(reference_price) + 1e-9 < self.maintenance_margin_required(reference_price)

    def cancel_all_resting_orders(self):
        for order in self.orders.copy():
            try:
                self._cancel_order(order)
            except Exception:
                pass

    def apply_financing_charge(self, reference_price: Optional[float], funding_cost: float) -> float:
        if self.defaulted:
            self.last_financing_charge = 0.0
            return 0.0

        ref_price = max(0.0, float(reference_price or 0.0))
        base_rate = max(0.0, float(funding_cost))
        borrowed_cash = max(0.0, -self.cash)
        short_notional = max(0.0, -self.assets) * ref_price
        charge = borrowed_cash * base_rate * max(0.0, self.borrow_spread_multiplier)
        charge += short_notional * base_rate * max(0.0, self.short_borrow_spread_multiplier)
        if charge > 0.0:
            self.cash -= charge
            self.cumulative_financing_charge += charge
        self.last_financing_charge = charge
        return charge

    def _mark_default(self, reason: str):
        self.cancel_all_resting_orders()
        self.defaulted = True
        self.default_reason = reason

    def enforce_balance_sheet_discipline(self, reference_price: Optional[float] = None) -> dict:
        if self.defaulted:
            return {'status': 'defaulted', 'reason': self.default_reason}

        ref_price = reference_price
        if ref_price is None:
            ref_price = self.market.price() if self.market.price() is not None else 0.0
        ref_price = max(0.0, float(ref_price))

        hard_cash_breach = self.cash < -self.max_cash_borrow - 1e-9
        hard_short_breach = self.assets < -self.max_short_assets - 1e-9
        margin_breach = self.has_margin_breach(ref_price)
        if not (hard_cash_breach or hard_short_breach or margin_breach):
            return {'status': 'ok'}

        self.cancel_all_resting_orders()

        best_bid = None
        best_ask = None
        try:
            if self.market.order_book['bid']:
                best_bid = self.market.order_book['bid'].first.price
            if self.market.order_book['ask']:
                best_ask = self.market.order_book['ask'].first.price
        except Exception:
            pass

        if self.assets > 0.0 and (hard_cash_breach or margin_breach):
            cash_target = -0.8 * self.max_cash_borrow
            cash_gap = max(0.0, cash_target - self.cash)
            bid_ref = max(1e-9, float(best_bid if best_bid is not None else ref_price or 1.0))
            qty_for_cash = int(_math.ceil(cash_gap / bid_ref)) if cash_gap > 0 else 0
            qty_for_margin = int(_math.ceil(max(0.0, self.assets) * max(0.0, self.liquidation_fraction))) if margin_breach else 0
            qty_to_sell = max(qty_for_cash, qty_for_margin)
            if qty_to_sell > 0:
                self._execute_clob(qty_to_sell, 'sell')

        if self.assets < 0.0 and (hard_short_breach or margin_breach):
            short_excess = max(0.0, abs(self.assets) - 0.8 * self.max_short_assets)
            qty_for_short = int(_math.ceil(short_excess)) if short_excess > 0 else 0
            qty_for_margin = int(_math.ceil(abs(self.assets) * max(0.0, self.liquidation_fraction))) if margin_breach else 0
            qty_to_buy = max(qty_for_short, qty_for_margin)
            if qty_to_buy > 0:
                self._execute_clob(qty_to_buy, 'buy')

        if self.cash < -self.max_cash_borrow - 1e-9:
            self._mark_default('cash_limit_breach')
        elif self.assets < -self.max_short_assets - 1e-9:
            self._mark_default('short_limit_breach')
        elif self.has_margin_breach(ref_price):
            self._mark_default('maintenance_margin_breach')

        return {
            'status': 'defaulted' if self.defaulted else 'stabilized',
            'reason': self.default_reason,
            'cash': self.cash,
            'assets': self.assets,
        }

    def available_cash(self) -> float:
        return self.cash + self.max_cash_borrow - self.reserved_cash

    def available_assets(self) -> float:
        return self.assets + self.max_short_assets - self.reserved_assets

    def prepare_limit_order(self, order: Order, t_cost: float) -> bool:
        if order.trader is not self:
            return False
        if getattr(order, 'reserved_cash', 0.0) > 0.0 or getattr(order, 'reserved_assets', 0.0) > 0.0:
            return True
        if order.qty <= 0:
            return False
        if order.order_type == 'bid':
            reserve = max(0.0, order.price * order.qty * (1.0 + t_cost))
            if reserve > self.available_cash() + 1e-9:
                return False
            order.reserved_cash = reserve
            self.reserved_cash += reserve
            return True
        reserve = float(order.qty)
        if reserve > self.available_assets() + 1e-9:
            return False
        order.reserved_assets = reserve
        self.reserved_assets += reserve
        return True

    def release_order_reservation(self, order: Order):
        cash_release = min(getattr(order, 'reserved_cash', 0.0), self.reserved_cash)
        asset_release = min(getattr(order, 'reserved_assets', 0.0), self.reserved_assets)
        if cash_release > 0.0:
            self.reserved_cash -= cash_release
        if asset_release > 0.0:
            self.reserved_assets -= asset_release
        order.reserved_cash = 0.0
        order.reserved_assets = 0.0

    def adjust_order_price(self, order: Order, new_price: float, t_cost: float) -> bool:
        if order.trader is not self:
            return False
        if order.order_type != 'bid':
            order.price = new_price
            return True
        new_reserve = max(0.0, new_price * order.qty * (1.0 + t_cost))
        delta = new_reserve - getattr(order, 'reserved_cash', 0.0)
        if delta > self.available_cash() + 1e-9:
            return False
        self.reserved_cash += delta
        order.reserved_cash = new_reserve
        order.price = new_price
        return True

    def apply_fill(self, order: Order, fill_qty: float, fill_price: float,
                   t_cost: float, is_buy: bool, qty_before: float):
        if qty_before > 0.0 and getattr(order, 'reserved_cash', 0.0) > 0.0:
            release = min(order.reserved_cash, order.reserved_cash * fill_qty / qty_before)
            order.reserved_cash -= release
            self.reserved_cash = max(0.0, self.reserved_cash - release)
        if qty_before > 0.0 and getattr(order, 'reserved_assets', 0.0) > 0.0:
            release = min(order.reserved_assets, order.reserved_assets * fill_qty / qty_before)
            order.reserved_assets -= release
            self.reserved_assets = max(0.0, self.reserved_assets - release)

        gross = fill_price * fill_qty
        if is_buy:
            self.cash -= gross * (1.0 + t_cost)
            self.assets += fill_qty
        else:
            self.cash += gross * (1.0 - t_cost)
            self.assets -= fill_qty

        # A fully executed order no longer rests and must not remain in the
        # owner's local order list.  Keeping the zero-quantity shell made the
        # dealer believe it still had a quote set and prevented replenishment
        # after the first hit, while fast providers were allowed to repost.
        if order.qty <= 1e-12:
            if hasattr(self, 'record_order_end'):
                self.record_order_end(order, reason='fill')
            try:
                self.orders.remove(order)
            except ValueError:
                pass

    def _clob_venue(self):
        if self.clob is not None:
            return self.clob
        from AgentBasedModel.venues.clob import CLOBVenue
        return CLOBVenue(self.market)

    def _affordable_limit_buy_qty(self, quantity: float, price: float) -> int:
        unit_cost = price * (1.0 + self.market.transaction_cost)
        if unit_cost <= 0:
            return 0
        return max(0, min(int(round(quantity)), int(self.available_cash() // unit_cost)))

    def _sellable_limit_qty(self, quantity: float) -> int:
        return max(0, min(int(round(quantity)), int(self.available_assets())))

    def _affordable_market_buy_qty(self, quantity: float) -> int:
        target = max(0, int(round(quantity)))
        if target <= 0:
            return 0
        venue = self._clob_venue()
        lo, hi, best = 0, target, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid <= 0:
                lo = 1
                continue
            quote = venue.quote_buy(mid)
            cost = quote.get('exec_price', float('inf')) * mid * (1.0 + self.market.transaction_cost)
            feasible = quote.get('cost_bps', float('inf')) != float('inf') and cost <= self.available_cash() + 1e-9
            if feasible:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _sellable_market_qty(self, quantity: float) -> int:
        target = max(0, int(round(quantity)))
        return max(0, min(target, int(self.available_assets())))

    def _clip_amm_buy_qty(self, pool, quantity: float) -> float:
        target = max(0.0, float(quantity))
        if target <= 0.0:
            return 0.0
        available_cash = self.available_cash()
        if available_cash <= 0.0:
            return 0.0
        hi = min(target, max(0.0, pool.x * 0.949))
        if hi <= 0.0:
            return 0.0
        hi_quote = pool.quote_buy(hi)
        if hi_quote.get('cost_bps', float('inf')) != float('inf') and hi_quote.get('delta_y', float('inf')) <= available_cash:
            return hi
        lo = 0.0
        best = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mid <= 0.0:
                break
            quote = pool.quote_buy(mid)
            feasible = quote.get('cost_bps', float('inf')) != float('inf') and quote.get('delta_y', float('inf')) <= available_cash
            if feasible:
                best = mid
                lo = mid
            else:
                hi = mid
        return best

    def _clip_amm_sell_qty(self, quantity: float) -> float:
        return max(0.0, min(float(quantity), self.available_assets()))

    def _buy_limit(self, quantity, price):
        if self.defaulted:
            return 0
        quote_price = self.market.round_price(price) if hasattr(self.market, 'round_price') else round(price, 2)
        quantity = self._affordable_limit_buy_qty(quantity, quote_price)
        if quantity <= 0:
            return 0
        order = Order(quote_price, quantity, 'bid', self)
        if self.env is not None:
            order._created_tick = int(getattr(self.env, '_t', 0))
        accepted = self.market.limit_order(order)
        if accepted and order.qty > 0:
            self.orders.append(order)
        return order.qty

    def _sell_limit(self, quantity, price):
        if self.defaulted:
            return 0
        quantity = self._sellable_limit_qty(quantity)
        if quantity <= 0:
            return 0
        quote_price = self.market.round_price(price) if hasattr(self.market, 'round_price') else round(price, 2)
        order = Order(quote_price, quantity, 'ask', self)
        if self.env is not None:
            order._created_tick = int(getattr(self.env, '_t', 0))
        accepted = self.market.limit_order(order)
        if accepted and order.qty > 0:
            self.orders.append(order)
        return order.qty

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if self.defaulted:
            return 0
        quantity = self._affordable_market_buy_qty(quantity)
        if quantity <= 0:
            return 0
        if not self.market.order_book['ask']:
            return 0
        order = Order(self.market.order_book['ask'].last.price, round(quantity), 'bid', self)
        order = self.market.market_order(order)
        self._last_clob_fills = list(order.fills)
        return order.qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if self.defaulted:
            return 0
        quantity = self._sellable_market_qty(quantity)
        if quantity <= 0:
            return 0
        if not self.market.order_book['bid']:
            return 0
        order = Order(self.market.order_book['bid'].last.price, round(quantity), 'ask', self)
        order = self.market.market_order(order)
        self._last_clob_fills = list(order.fills)
        return order.qty

    def _cancel_order(self, order: Order, reason: str = 'cancel'):
        if hasattr(self, 'record_order_end'):
            self.record_order_end(order, reason=reason)
        self.market.cancel_order(order)
        self.orders.remove(order)

    # ---- multi-venue routing (active only when amm_pools is set) ---------

    def _routing_reference_price(self) -> float:
        """Public routing benchmark used for cross-venue AMM quotes."""
        try:
            clob_mid = self.clob.mid_price()
            if _math.isfinite(clob_mid) and clob_mid > 0:
                return float(clob_mid)
        except Exception:
            pass

        if self.env is not None:
            try:
                fair_price = self.env.fair_price
                if fair_price is not None and _math.isfinite(fair_price) and fair_price > 0:
                    return float(fair_price)
            except Exception:
                pass

        return float('nan')

    def estimate_costs(self, Q: float, side: str = 'buy') -> Dict[str, float]:
        """Return {venue_name: internal execution cost_bps} for quantity Q."""
        costs: Dict[str, float] = {}
        benchmark_price = self._routing_reference_price()
        try:
            costs['clob'] = self.clob.cost_bps(Q, side)
        except Exception:
            costs['clob'] = float('inf')
        for name, pool in self.amm_pools.items():
            try:
                # Route AMMs against the public market benchmark so stale but
                # executable AMM quotes can attract flow during CLOB stress.
                if _math.isfinite(benchmark_price) and benchmark_price > 0:
                    q = (
                        pool.quote_buy(Q, S_t=benchmark_price)
                        if side == 'buy' else pool.quote_sell(Q, S_t=benchmark_price)
                    )
                else:
                    q = pool.quote_buy(Q) if side == 'buy' else pool.quote_sell(Q)
                costs[name] = q['cost_bps']
            except Exception:
                costs[name] = float('inf')
        return costs

    def _perceived_costs(self, costs: Dict[str, float]) -> Dict[str, float]:
        """Apply venue-specific bias and estimation noise to routing costs."""
        perceived: Dict[str, float] = {}
        for name, cost in costs.items():
            adjusted = cost
            if name == 'cpmm' and cost != float('inf'):
                adjusted -= self.cpmm_bias_bps
            if name != 'clob' and self.cost_noise_std > 0 and cost != float('inf'):
                adjusted += self._routing_rng.gauss(0, self.cost_noise_std)
            perceived[name] = adjusted
        return perceived

    def _liquidity_aware_pick(self, costs: Dict[str, float], Q: float) -> str:
        """Route via an endogenous CLOB-vs-AMM group choice, then pick an AMM pool."""
        amm_names = [name for name in costs if name != 'clob']
        if not amm_names:
            return 'clob'

        perceived = self._perceived_costs(costs)
        try:
            ref_mid = self.clob.mid_price()
        except Exception:
            ref_mid = float('nan')

        try:
            depth_row = self.clob.total_depth(25)
            clob_depth = min(depth_row['bid'], depth_row['ask'])
        except Exception:
            clob_depth = 0.0

        pool_depths: Dict[str, float] = {}
        pool_alignment: Dict[str, float] = {}
        for name in amm_names:
            pool = self.amm_pools[name]
            try:
                depth = pool.effective_depth(ref_mid) if _math.isfinite(ref_mid) else pool.effective_depth()
            except Exception:
                depth = 0.0
            pool_depths[name] = max(0.0, depth)

            alignment_score = 1.0
            try:
                pool_mid = pool.mid_price()
                if _math.isfinite(ref_mid) and ref_mid > 0 and _math.isfinite(pool_mid):
                    basis_bps = abs(pool_mid - ref_mid) / ref_mid * 10_000.0
                    # Cross-venue basis is a strong signal of stale/mis-priced
                    # AMM quote. With a 50 bps half-life and 0.10 floor a 100
                    # bps basis cuts AMM weight by ~3x, while a 1000 bps drift
                    # essentially removes it from contention.
                    alignment_score = max(
                        0.10,
                        1.0 / (1.0 + basis_bps / self.routing_basis_scale_bps),
                    )
            except Exception:
                alignment_score = 1.0
            pool_alignment[name] = alignment_score

        finite_amm = [name for name in amm_names if _math.isfinite(perceived[name]) and perceived[name] != float('inf')]
        best_amm_name = min(finite_amm, key=lambda name: perceived[name]) if finite_amm else amm_names[0]

        group_costs = {
            'clob': perceived.get('clob', float('inf')),
            'amm': perceived.get(best_amm_name, float('inf')),
        }
        finite_group_costs = [cost for cost in group_costs.values() if _math.isfinite(cost) and cost != float('inf')]
        best_group_cost = min(finite_group_costs) if finite_group_costs else 0.0

        amm_target = max(0.0, min(1.0, self.amm_share_pct / 100.0))
        cost_span = (
            max(finite_group_costs) - min(finite_group_costs)
            if len(finite_group_costs) >= 2 else 0.0
        )
        liquidity_factor = getattr(self.env, 'systemic_liquidity', 1.0) if self.env is not None else 1.0
        venue_basis_bps = getattr(self.env, 'venue_basis_bps', 0.0) if self.env is not None else 0.0
        arbitrage_capacity = getattr(self.env, 'arbitrage_capacity', 1.0) if self.env is not None else 1.0
        # ``amm_share_pct`` is a weak prior by design under this routing
        # regime, not a quota.  Its strength is exposed because the old
        # hard-coded 0.18 cap could otherwise decide AMM flow and LP fee income
        # without appearing in a robustness specification.
        prior_mix = self.routing_prior_mix_cap / (1.0 + cost_span / 10.0)
        prior_mix *= 0.75 + 0.25 * max(0.0, min(1.0, liquidity_factor))
        prior_mix /= 1.0 + max(0.0, venue_basis_bps) / 25.0
        prior_mix = max(0.0, min(self.routing_prior_mix_cap, prior_mix))

        neutral_priors = {'clob': 0.5, 'amm': 0.5}
        target_priors = {'clob': 1.0 - amm_target, 'amm': amm_target}
        prior_scores = {}
        for name in ('clob', 'amm'):
            eff_prior = (1.0 - prior_mix) * neutral_priors[name] + prior_mix * target_priors[name]
            prior_scores[name] = eff_prior / neutral_priors[name]

        amm_depth = sum(pool_depths.values())
        amm_alignment = pool_alignment.get(best_amm_name, 1.0)
        group_depths = {'clob': clob_depth, 'amm': amm_depth}
        group_alignment = {'clob': 1.0, 'amm': amm_alignment}
        amm_state_score = 1.0
        if self.env is not None:
            liquidity_component = 0.80 + 0.20 * max(0.0, min(1.0, liquidity_factor))
            arbitrage_component = 0.65 + 0.35 * max(0.0, min(1.0, arbitrage_capacity))
            # Arbitrage capacity already embeds venue basis, so penalizing the
            # same basis again here would suppress realistic flight-to-AMM.
            amm_state_score = liquidity_component * arbitrage_component
            amm_state_score = max(0.35, min(1.0, amm_state_score))
        group_state_scores = {'clob': 1.0, 'amm': amm_state_score}

        group_scores: Dict[str, float] = {}
        for name, cost in group_costs.items():
            if cost == float('inf') or not _math.isfinite(cost):
                group_scores[name] = 0.0
                continue
            rel_cost = max(0.0, cost - best_group_cost)
            # This elasticity is a design parameter, not an EBS calibration
            # moment.  Keeping it explicit makes the LP-income and flow-share
            # sensitivity auditable.
            cost_score = _math.exp(-rel_cost / self.routing_cost_scale_bps)
            depth = group_depths[name]
            depth_multiple = (
                self.routing_clob_depth_multiple
                if name == 'clob' else self.routing_amm_depth_multiple
            )
            depth_score = min(max(depth, 0.0) / max(depth_multiple * Q, 1.0), 2.0)
            liquidity_score = 0.4 + 0.6 * min(depth_score, 1.0)
            group_scores[name] = (
                prior_scores[name]
                * cost_score
                * liquidity_score
                * group_alignment[name]
                * group_state_scores[name]
            )

        if self._weighted_pick(group_scores, rng=self._routing_rng) == 'clob':
            return 'clob'

        amm_scores: Dict[str, float] = {}
        best_amm_cost = min(
            perceived[name] for name in amm_names
            if _math.isfinite(perceived[name]) and perceived[name] != float('inf')
        ) if finite_amm else 0.0
        for name in amm_names:
            cost = perceived[name]
            if cost == float('inf') or not _math.isfinite(cost):
                amm_scores[name] = 0.0
                continue
            rel_cost = max(0.0, cost - best_amm_cost)
            cost_score = _math.exp(-rel_cost / self.routing_cost_scale_bps)
            depth_score = min(
                pool_depths.get(name, 0.0)
                / max(self.routing_amm_depth_multiple * Q, 1.0),
                2.0,
            )
            liquidity_score = 0.4 + 0.6 * min(depth_score, 1.0)
            amm_scores[name] = cost_score * liquidity_score * pool_alignment.get(name, 1.0)

        return self._weighted_pick(amm_scores, rng=self._routing_rng)

    def choose_venue(self, Q: float, side: str = 'buy') -> str:
        """Venue selection under a fixed-share or liquidity-aware routing rule.

        fixed_share:
            Step 1 — CLOB vs AMM with fixed probability ``amm_share_pct``.
            Step 2 — within AMM, CPMM vs HFMM (softmax with β_amm),
                using bias-adjusted and noise-perturbed costs.

        liquidity_aware:
            Route probabilistically across all venues using perceived cost,
            executable depth, price alignment, and a soft prior derived from
            ``amm_share_pct``.

        Falls back to CLOB when there are no AMM pools.
        """
        costs = self.estimate_costs(Q, side)

        # Deterministic shortcut
        if self.deterministic_venue:
            return min(costs, key=costs.get)

        # Identify AMM venues
        amm_names = [n for n in costs if n != 'clob']

        # If no AMM pools → always CLOB
        if not amm_names:
            return 'clob'

        if self.venue_choice_rule == 'liquidity_aware':
            return self._liquidity_aware_pick(costs, Q)

        # ── Step 1: CLOB vs AMM (direct probability) ─────────────────
        if self._routing_rng.random() * 100.0 >= self.amm_share_pct:
            return 'clob'

        # ── Step 2: within AMM — CPMM vs HFMM ────────────────────────
        amm_raw_costs = {n: costs[n] for n in amm_names}
        adj_costs = {n: self._perceived_costs({n: c})[n] for n, c in amm_raw_costs.items()}

        return self._softmax_pick(
            adj_costs, self.beta_amm, rng=self._routing_rng
        )

    # ---- helper ----------------------------------------------------------

    @staticmethod
    def _softmax_pick(costs: Dict[str, float], beta: float,
                      rng=None) -> str:
        """Pick a key from *costs* dict via logit softmax with sensitivity *beta*."""
        rng = rng if rng is not None else random
        items = list(costs.items())
        finite = [c for _, c in items if c != float('inf')]
        if not finite:
            return items[0][0]
        min_c = min(finite)
        weights = []
        for name, c in items:
            if c == float('inf'):
                weights.append(0.0)
            else:
                weights.append(_math.exp(-beta * (c - min_c)))
        total = sum(weights)
        if total == 0:
            return items[0][0]
        r = rng.random() * total
        cumulative = 0.0
        for (name, _), w in zip(items, weights):
            cumulative += w
            if r <= cumulative:
                return name
        return items[-1][0]

    @staticmethod
    def _weighted_pick(scores: Dict[str, float], rng=None) -> str:
        """Pick a key from positive score weights."""
        rng = rng if rng is not None else random
        items = list(scores.items())
        total = sum(max(0.0, score) for _, score in items)
        if total <= 0.0:
            return items[0][0]
        r = rng.random() * total
        cumulative = 0.0
        for name, score in items:
            cumulative += max(0.0, score)
            if r <= cumulative:
                return name
        return items[-1][0]

    def _classify(self, Q: float) -> dict:
        """
        Compute θ = Q/R relative to max AMM effective depth.
        """
        from AgentBasedModel.venues.amm import classify_trade_size
        try:
            mid = self.clob.mid_price()
        except Exception:
            return dict(theta=0.0, size_bucket='medium', R=1.0)
        R = 1.0
        ref_pool = None
        for pool in self.amm_pools.values():
            d = pool.effective_depth(mid)
            if d > R:
                R = d
                ref_pool = pool
        if ref_pool is not None:
            return classify_trade_size(Q, ref_pool, mid)
        theta = Q / R
        bucket = 'small' if theta <= 0.01 else ('medium' if theta <= 0.05 else 'large')
        return dict(theta=theta, size_bucket=bucket, R=R)

    def _execute_on_venue(self, venue: str, Q: float, side: str) -> dict:
        """Execute trade on chosen venue. Returns cost dict."""
        if self.defaulted:
            venue_obj = self._clob_venue() if venue == 'clob' else self.amm_pools.get(venue)
            result = venue_obj._inf_quote() if venue_obj is not None and hasattr(venue_obj, '_inf_quote') else {'cost_bps': float('inf')}
            result['executed_qty'] = 0.0
            result['requested_qty'] = Q
            result['refusal_reason'] = 'defaulted'
            return result
        if venue == 'clob':
            return self._execute_clob(Q, side)
        pool = self.amm_pools[venue]
        if side == 'buy':
            exec_q = self._clip_amm_buy_qty(pool, Q)
            if exec_q <= 0.0:
                result = pool._inf_quote()
                result['executed_qty'] = 0.0
                result['requested_qty'] = Q
                # Why the request died decides whether it says anything about
                # the venue.  A buyer out of cash is a fact about the buyer.
                result['refusal_reason'] = (
                    'no_cash' if self.available_cash() <= 0.0
                    else 'no_liquidity'
                )
                return result
            result = pool.execute_buy(exec_q)
            # Settle against what the pool says it did, not against what was
            # asked for. A venue can refuse, and a closed one always does, so
            # taking the request on trust moves the trader's balances against
            # a trade that never happened.
            filled = float(result.get('executed_qty', exec_q))
            self.cash -= result.get('delta_y', 0.0)
            self.assets += filled
        else:
            exec_q = self._clip_amm_sell_qty(Q)
            if exec_q <= 0.0:
                result = pool._inf_quote()
                result['executed_qty'] = 0.0
                result['requested_qty'] = Q
                # The seller has hit its own inventory or short limit. This is
                # the single largest source of unfilled demand in a crisis and
                # it has nothing to do with either venue's state.
                result['refusal_reason'] = 'no_assets'
                return result
            result = pool.execute_sell(exec_q)
            filled = float(result.get('executed_qty', exec_q))
            self.cash += result.get('delta_y', 0.0)
            self.assets -= filled
        result['executed_qty'] = filled
        result['requested_qty'] = Q
        if filled <= 0.0:
            # The pool was asked for a size it had agreed to quote and still
            # returned nothing, which is a statement about the venue.
            result['refusal_reason'] = 'venue_refused'
        return result

    def _execute_clob(self, Q: float, side: str) -> dict:
        """Market order on CLOB with cost quoting."""
        venue = self._clob_venue()
        self._last_clob_fills = []
        try:
            if self.defaulted:
                quote_info = venue._inf_quote()
                quote_info['executed_qty'] = 0
                quote_info['requested_qty'] = Q
                quote_info['refusal_reason'] = 'defaulted'
                return quote_info
            if side == 'buy':
                exec_q = self._affordable_market_buy_qty(Q)
                if exec_q <= 0:
                    quote_info = venue._inf_quote()
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    # The search returns nothing both when the buyer cannot
                    # afford one unit and when the book cannot price one, and
                    # those are different findings, so they are separated by
                    # asking the book for the smallest size it could fill.
                    quote_info['refusal_reason'] = (
                        'no_liquidity'
                        if venue.quote_buy(1).get('cost_bps', float('inf')) == float('inf')
                        else 'no_cash'
                    )
                    return quote_info
                quote_info = venue.quote_buy(exec_q)
                if quote_info['cost_bps'] == float('inf'):
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    quote_info['refusal_reason'] = 'no_liquidity'
                    return quote_info
                unfilled = self._buy_market(exec_q)
                quote_info['executed_qty'] = max(0, exec_q - unfilled)
                quote_info['maker_fills'] = list(self._last_clob_fills)
                quote_info['requested_qty'] = Q
                if quote_info['executed_qty'] <= 0:
                    quote_info['refusal_reason'] = 'no_liquidity'
                return quote_info
            else:
                exec_q = self._sellable_market_qty(Q)
                if exec_q <= 0:
                    quote_info = venue._inf_quote()
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    quote_info['refusal_reason'] = 'no_assets'
                    return quote_info
                quote_info = venue.quote_sell(exec_q)
                if quote_info['cost_bps'] == float('inf'):
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    quote_info['refusal_reason'] = 'no_liquidity'
                    return quote_info
                unfilled = self._sell_market(exec_q)
                quote_info['executed_qty'] = max(0, exec_q - unfilled)
                quote_info['maker_fills'] = list(self._last_clob_fills)
                quote_info['requested_qty'] = Q
                if quote_info['executed_qty'] <= 0:
                    quote_info['refusal_reason'] = 'no_liquidity'
                return quote_info
        except Exception:
            quote_info = venue._inf_quote()
            quote_info['executed_qty'] = 0
            quote_info['requested_qty'] = Q
            quote_info['refusal_reason'] = 'error'
            return quote_info

    def _make_routing_attempt(self, venue: str, side: str, Q: float,
                              result: dict) -> dict:
        """One routed customer request, filled or not.

        ``requested`` is the demand that arrived at the venue, ``executed``
        the part of it that traded, and ``reason`` names why the rest did not.
        A filled request carries no reason.
        """
        requested = float(result.get('requested_qty', Q))
        executed = float(result.get('executed_qty', 0.0) or 0.0)
        return dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side,
            requested_quantity=max(0.0, requested),
            executed_quantity=max(0.0, executed),
            refusal_reason=(
                None if executed > 0.0
                else str(result.get('refusal_reason', 'unspecified'))
            ),
        )

    def _make_trade_record(self, venue: str, side: str, Q: float,
                           result: dict, cls_info: dict) -> dict:
        """Build a trade record dict for logging."""
        exec_price = float(result.get('exec_price', float('nan')))
        fee_bps = float(result.get('fee_bps', 0.0) or 0.0)
        # AMM execution prices already include the swap fee: a buy records
        # gross quote paid and a sell records net quote received. CLOB VWAP
        # excludes any explicit venue/broker charge, so convert it to the same
        # all-in object here. Welfare can then use one common fair-price
        # benchmark instead of comparing venue-specific local mids.
        if venue == 'clob' and _math.isfinite(exec_price):
            fee_fraction = fee_bps / 10_000.0
            all_in_exec_price = (
                exec_price * (1.0 + fee_fraction)
                if side == 'buy'
                else exec_price * max(0.0, 1.0 - fee_fraction)
            )
        else:
            all_in_exec_price = exec_price
        return dict(
            trader_id=self.id, trader_type=self.type,
            execution_source='routed_customer',
            venue=venue, side=side, quantity=Q,
            requested_quantity=result.get('requested_qty', Q),
            cost_bps=result.get('cost_bps', 0),
            local_cost_bps=result.get('cost_bps', 0),
            exec_price=exec_price,
            all_in_exec_price=all_in_exec_price,
            common_reference_price=result.get('common_reference_price', float('nan')),
            fee_bps=fee_bps,
            theta=cls_info.get('theta', 0),
            size_bucket=cls_info.get('size_bucket', 'medium'),
            maker_fills=list(result.get('maker_fills', [])),
        )


class Random(Trader):
    """
    Random creates noisy orders to recreate trading in real environment.

    When *amm_pools* / *clob* are provided (multi-venue mode), the agent
    becomes a stochastic market-order taker that routes via cost estimation
    (replaces former FXNoiseTaker).  Use *label* to tag sub-populations
    (e.g. 'Retail', 'Institutional').
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 # multi-venue params (forwarded to Trader)
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 venue_choice_rule: str = 'fixed_share',
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 routing_cost_scale_bps: float = 4.0,
                 routing_prior_mix_cap: float = 0.18,
                 routing_basis_scale_bps: float = 50.0,
                 routing_clob_depth_multiple: float = 10.0,
                 routing_amm_depth_multiple: float = 8.0,
                 routing_rng: Optional[random.Random] = None,
                 # FX-noise-taker params (only used in multi-venue mode)
                 trade_prob: float = 0.3,
                 q_min: int = 1,
                 q_max: int = 5,
                 label: str = 'Random',
                 flow_role: Optional[str] = None,
                 flow_persistence: float = 0.25,
                 common_flow_response: float = 0.05,
                 session_sensitivity: float = 1.0,
                 flow_rng: Optional[random.Random] = None):
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, venue_choice_rule=venue_choice_rule,
                         deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std,
                         routing_cost_scale_bps=routing_cost_scale_bps,
                         routing_prior_mix_cap=routing_prior_mix_cap,
                         routing_basis_scale_bps=routing_basis_scale_bps,
                         routing_clob_depth_multiple=routing_clob_depth_multiple,
                         routing_amm_depth_multiple=routing_amm_depth_multiple,
                         routing_rng=routing_rng)
        self.type = label if self.multi_venue else 'Random'
        self.trade_prob = trade_prob
        self.q_min = q_min
        self.q_max = q_max
        self.flow_role = flow_role or (label if self.multi_venue else 'Random')
        self.flow_persistence = max(0.0, min(1.0, float(flow_persistence)))
        self.common_flow_response = max(0.0, min(0.45, float(common_flow_response)))
        self.session_sensitivity = max(0.0, float(session_sensitivity))
        # This stream contains the primitive customer-demand innovations only.
        # Venue construction and routing may consume a different number of
        # draws across counterfactual arms without changing these uniforms.
        self._flow_rng = flow_rng if flow_rng is not None else random
        self._last_fx_side: Optional[str] = None
        self._flow_program_side: Optional[str] = None
        self._flow_program_remaining: int = 0
        self._flow_program_qty_scale: float = 1.0
        self._flow_program_trade_prob_mult: float = 1.0
        self._flow_slice_qty_scale: float = 1.0

    def _clear_flow_program(self):
        self._flow_program_side = None
        self._flow_program_remaining = 0
        self._flow_program_qty_scale = 1.0
        self._flow_program_trade_prob_mult = 1.0

    def _flow_program_profile(self) -> Optional[dict]:
        profile = None
        if self.flow_role == 'RetailToxic':
            profile = dict(start_prob=0.02 + 0.10 * self.flow_persistence,
                           run_lo=1, run_hi=3,
                           qty_scale=0.95,
                           trade_prob_mult=1.08)
        elif self.flow_role == 'RealMoney':
            profile = dict(start_prob=0.03 + 0.12 * self.flow_persistence,
                           run_lo=2, run_hi=4,
                           qty_scale=0.72,
                           trade_prob_mult=1.15)
        elif self.flow_role == 'Hedger':
            profile = dict(start_prob=0.01 + 0.04 * self.flow_persistence,
                           run_lo=1, run_hi=2,
                           qty_scale=0.92,
                           trade_prob_mult=1.04)

        if profile is None or self.env is None:
            return profile

        session_name = getattr(self.env, 'session_name', '')
        if self.flow_role == 'RealMoney' and session_name in {'London', 'Overlap', 'NewYork'}:
            profile['start_prob'] *= 1.10
        elif self.flow_role == 'RetailToxic' and session_name == 'Overlap':
            profile['start_prob'] *= 1.08
        elif self.flow_role == 'Hedger' and session_name == 'Asia':
            profile['start_prob'] *= 1.05

        profile['start_prob'] = max(0.0, min(0.45, profile['start_prob']))
        return profile

    def _continue_flow_program(self) -> Optional[tuple[str, float]]:
        if self._flow_program_side is None or self._flow_program_remaining <= 0:
            self._clear_flow_program()
            return None

        side = self._flow_program_side
        qty_scale = self._flow_program_qty_scale
        self._flow_program_remaining -= 1
        self._last_fx_side = side
        if self._flow_program_remaining <= 0:
            self._clear_flow_program()
        return side, qty_scale

    def _maybe_start_flow_program(self, side: str) -> float:
        profile = self._flow_program_profile()
        if profile is None or self._flow_rng.random() >= profile['start_prob']:
            return 1.0

        total_slices = self._flow_rng.randint(
            profile['run_lo'], profile['run_hi']
        )
        if total_slices > 1:
            self._flow_program_side = side
            self._flow_program_remaining = total_slices - 1
            self._flow_program_qty_scale = profile['qty_scale']
            self._flow_program_trade_prob_mult = profile['trade_prob_mult']
        else:
            self._clear_flow_program()
        return profile['qty_scale']

    def _session_trade_probability(self) -> float:
        prob = self.trade_prob
        if self._flow_program_side is not None and self._flow_program_remaining > 0:
            prob *= self._flow_program_trade_prob_mult
        if self.env is not None:
            session_mult = getattr(self.env, 'session_flow_multiplier', 1.0)
            prob *= max(0.5, 1.0 + self.session_sensitivity * (session_mult - 1.0))
            session_name = getattr(self.env, 'session_name', '')
            if self.flow_role == 'RetailToxic' and session_name == 'Overlap':
                prob *= 1.10
            elif self.flow_role == 'RealMoney' and session_name in {'London', 'Overlap', 'NewYork'}:
                prob *= 1.12
            elif self.flow_role == 'Hedger' and session_name == 'Asia':
                prob *= 0.95
        return max(0.0, min(0.98, prob))

    def _session_quantity_bounds(self) -> tuple[int, int]:
        lo = self.q_min
        hi = self.q_max
        if self.env is not None:
            session_mult = getattr(self.env, 'session_flow_multiplier', 1.0)
            hi = max(lo, int(round(hi * max(0.75, session_mult))))
            if self.flow_role == 'RealMoney':
                hi = max(lo, int(round(hi * 1.15)))
            elif self.flow_role == 'RetailToxic':
                hi = max(lo, int(round(hi * 0.9)))
        slice_scale = max(0.60, min(1.25, float(getattr(self, '_flow_slice_qty_scale', 1.0))))
        lo = max(1, int(round(lo * max(0.70, slice_scale))))
        hi = max(lo, int(round(hi * slice_scale)))
        return max(1, lo), max(max(1, lo), hi)

    def _draw_flow_side(self) -> str:
        self._flow_slice_qty_scale = 1.0
        program_side = self._continue_flow_program()
        if program_side is not None:
            side, qty_scale = program_side
            self._flow_slice_qty_scale = qty_scale
            return side

        bias = self.env.toxic_flow_bias if self.env else 0.0
        p_buy = 0.5 + bias
        # Previous-period market-wide imbalance is public information. A
        # small response captures parent-order/common-client-flow clustering
        # across participants; relying only on each agent's last trade erases
        # persistence when arrivals are sparse enough to match EBS volume.
        if self.env is not None:
            p_buy += self.common_flow_response * float(
                getattr(self.env, 'order_flow_imbalance', 0.0)
            )
        persist_shift = 0.0
        if self._last_fx_side is not None:
            role_scale = 1.0
            if self.flow_role == 'Hedger':
                role_scale = 0.35
            elif self.flow_role == 'RetailToxic':
                role_scale = 1.15
            elif self.flow_role == 'RealMoney':
                role_scale = 0.90
            persist_shift = 0.18 * self.flow_persistence * role_scale
            if self._last_fx_side == 'buy':
                p_buy += persist_shift
            else:
                p_buy -= persist_shift
        if self.flow_role == 'Hedger' and self._last_fx_side is not None:
            mean_reversion_shift = 0.35 * persist_shift
            if self._last_fx_side == 'buy':
                p_buy -= mean_reversion_shift
            else:
                p_buy += mean_reversion_shift
        p_buy = max(0.05, min(0.95, p_buy))
        side = 'buy' if self._flow_rng.random() < p_buy else 'sell'
        self._last_fx_side = side
        self._flow_slice_qty_scale = self._maybe_start_flow_program(side)
        return side

    @staticmethod
    def draw_delta(std: Union[float, int] = 2.5):
        lamb = 1 / std
        return random.expovariate(lamb)

    @staticmethod
    def draw_price(order_type, spread: dict, std: Union[float, int] = 2.5,
                   sigma: float = None, sigma_low: float = None) -> float:
        """
        Draw price for limit order of Noise Agent.

        The inside-spread probability scales inversely with σ/σ_low,
        modelling the empirical tendency of passive liquidity to
        withdraw from the top of book during volatile markets.

        1) p_inside (35 % in normal, lower in stress) — uniform inside spread
        2) (1 − p_inside) — out of spread with exponential delta
        """
        # Regime-aware inside-spread probability
        p_inside = 0.35
        eff_std = std
        if sigma is not None and sigma_low is not None and sigma_low > 0:
            stress_ratio = sigma / sigma_low  # 1.0 normal, 4.0+ stress
            p_inside = max(0.05, 0.35 / stress_ratio)
            # Widen draw_delta in stress: orders placed further from touch
            eff_std = std * max(1.0, stress_ratio * 0.7)

        random_state = random.random()

        # Within the spread
        if random_state < p_inside:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = Random.draw_delta(eff_std)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    @staticmethod
    def draw_quantity(a=1, b=5) -> float:
        """
        Draw random quantity to buy from uniform distribution.

        :param a: minimal quantity
        :param b: maximal quantity
        :return: quantity for order
        """
        return random.randint(a, b)

    def _draw_customer_intent(self) -> Optional[tuple[str, int]]:
        """Draw the primitive arrival, side and requested quantity.

        This is completed before venue choice by design.  Paired research
        arms can therefore share one demand innovation while routing and
        execution remain endogenous outcomes of the venue configuration.
        """
        if (self.defaulted
                or self._flow_rng.random()
                > self._session_trade_probability()):
            return None
        side = self._draw_flow_side()
        q_lo, q_hi = self._session_quantity_bounds()
        return side, self._flow_rng.randint(q_lo, q_hi)

    def call(self):
        # ---------- multi-venue mode (FX noise taker) ---------------------
        if self.multi_venue:
            # Cleared on every call so that a tick on which this customer had
            # no demand cannot be read as a repeat of its previous request.
            self.last_routing_attempt = None
            intent = self._draw_customer_intent()
            if intent is None:
                return None
            side, Q = intent
            venue = self.choose_venue(Q, side)
            try:
                common_reference_price = float(self._clob_venue().mid_price())
            except Exception:
                common_reference_price = float('nan')
            result = self._execute_on_venue(venue, Q, side)
            result['common_reference_price'] = common_reference_price
            exec_q = result.get('executed_qty', 0)
            # The request is recorded whether or not it filled. Measuring the
            # venue's share on executed volume alone divides by a denominator
            # that collapses in a crisis for reasons that have nothing to do
            # with either venue, and reads the collapse as a migration of
            # demand.
            self.last_routing_attempt = self._make_routing_attempt(
                venue, side, Q, result
            )
            if exec_q <= 0:
                return None
            cls_info = self._classify(exec_q)
            rec = self._make_trade_record(venue, side, exec_q, result, cls_info)
            self.trades.append(rec)
            return rec

        # ---------- classic single-venue mode -----------------------------
        spread = self.market.spread()
        if spread is None:
            return

        mid = (spread['bid'] + spread['ask']) / 2.0

        # Use fair_price as reference for staleness when available
        ref_mid = mid
        if self.env is not None:
            fp = getattr(self.env, 'fair_price', None)
            if fp is not None and fp > 0:
                fair_weight = 0.25 * getattr(self.env, 'systemic_liquidity', 1.0)
                ref_mid = mid + fair_weight * (fp - mid)

        # ── Re-center stale orders ──────────────────────────────────
        # If any resting order is >200 bps from reference mid, cancel it
        # and immediately replace with a fresh limit near the spread.
        stale = None
        for order in self.orders:
            dist_bps = abs(order.price - ref_mid) / ref_mid * 10_000
            if dist_bps > 200:
                stale = order
                break
        if stale is not None:
            side = stale.order_type
            self._cancel_order(stale)
            _sig = self.env.sigma if self.env else None
            _sig_lo = self.env.sigma_low if self.env else None
            price = self.draw_price(side, spread, sigma=_sig, sigma_low=_sig_lo)
            quantity = self.draw_quantity()
            quote_price = self.market.round_price(price) if hasattr(self.market, 'round_price') else round(price, 2)
            order = Order(quote_price, round(quantity), side, self)
            accepted = self.market.limit_order(order)
            if accepted and order.qty > 0:
                self.orders.append(order)
            return  # one re-center per tick is enough

        # ── Normal action selection (Fennell distribution) ──────────
        order_type = 'bid' if random.random() > 0.5 else 'ask'

        random_state = random.random()
        # Market order (15%)
        if random_state > .85:
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order (35%)
        elif random_state > .50:
            _sig = self.env.sigma if self.env else None
            _sig_lo = self.env.sigma_low if self.env else None
            price = self.draw_price(order_type, spread, sigma=_sig, sigma_low=_sig_lo)
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            elif order_type == 'ask':
                self._sell_limit(quantity, price)

        # Cancellation order (35%)
        elif random_state < .35:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class RestingQuoteProvider:
    """Machinery shared by providers whose quotes rest in the book.

    Two different events used to be handled by one method, and merging them
    caused two separate defects.

    The first event is the tick. Once a second the provider ages its quotes,
    decides whether they are stale enough to replace, and may choose not to
    be in the book at all. The second event is a fill. A resting quote that
    is hit is gone, and the provider reposts what was taken. These happen at
    different rates: at a one second tick a non bank market maker on the
    primary venue has a published order life of about three seconds, so it
    ages its quote once per tick, but it may be hit any number of times
    within that tick and reposts each time.

    Running the tick logic on every fill made the second event consume the
    first one's clock, so an order life of two ticks became an order life of
    two trades, and it also let a provider that had not been hit at all
    cancel and rebuild its whole quote set out of nothing. The first turned
    the time contract into an order count, the second manufactured liquidity.
    ``on_trade`` therefore reposts only the side that was actually consumed,
    at the geometry the provider last chose, and never touches the age.
    """

    def _cancel_all(self, reason: str = 'provider_cancel'):
        for order in self.orders.copy():
            try:
                self._cancel_order(order, reason=reason)
            except Exception:
                pass
        self.orders.clear()

    def _stamp_quote(self, order: Order, reference_mid: Optional[float] = None,
                     quote_level: Optional[int] = None) -> Order:
        """Attach the model clock used by completed-order lifetime metrics."""
        if getattr(self, '_stochastic_order_lifecycle', False):
            return _stamp_lifecycle_order(
                self, order, self.ttl,
                reference_mid=reference_mid, quote_level=quote_level,
            )
        env = getattr(self, 'env', None)
        if env is not None:
            order._created_tick = int(getattr(env, '_t', 0))
            order._initial_qty = float(getattr(order, 'qty', 0.0))
        return order

    def record_order_end(self, order: Order, reason: Optional[str] = None):
        """Record one completed resting-quote lifetime exactly once."""
        _record_lifecycle_end(self, order, reason=reason)

    def lifecycle_observations(self, include_live: bool = True) -> List[dict]:
        return _lifecycle_observations(self, include_live=include_live)

    def _cancel_stale_orders(self, mid: float, tol_bps: float) -> int:
        """Cancel only quotes whose own reference has become stale."""
        cancelled = 0
        if mid <= 0:
            return cancelled
        for order in self.orders.copy():
            if getattr(order, 'qty', 0.0) <= 0.0:
                continue
            reference = getattr(order, '_quote_reference_mid', None)
            if reference is None:
                # Legacy/provider quotes pre-dating per-order references use
                # their price as the conservative fallback.
                reference = float(getattr(order, 'price', mid))
            if reference <= 0:
                continue
            if abs(mid - reference) / reference * 1e4 <= tol_bps:
                continue
            try:
                self._cancel_order(order, reason='stale_reprice')
                cancelled += 1
            except Exception:
                pass
        return cancelled

    def _quotes_still_good(self, mid: float, tol_bps: float) -> bool:
        """Whether the standing quotes are close enough to leave alone.

        Every provider in this book used to cancel its whole quote set at the
        top of the period and rebuild it, so the near book was torn down and
        reconstructed once a second and its depth was the joint realisation of
        a dozen independent draws. That is what produced the tail in the
        quoted spread: not a shock and not a withdrawal, but the fact that
        nothing rested. Published order lives on the primary venue are around
        three seconds for a non bank and far longer for a bank, so a quote is
        meant to survive the period that made it.
        """
        live = [o for o in self.orders if getattr(o, 'qty', 0) > 0]
        if not live:
            return False
        if mid <= 0:
            return False
        for o in live:
            if abs(o.price - mid) / mid * 1e4 > tol_bps:
                return False
        return True

    def _remember_geometry(self, half_spread: float, qty: int, levels: int,
                           level_step: float):
        """Store the quote shape so a fill can be answered without re-deciding."""
        self._geometry = {'half_spread': float(half_spread), 'qty': int(qty),
                          'levels': int(levels), 'level_step': float(level_step)}

    def on_trade(self, bid_taken: float = 0.0, ask_taken: float = 0.0) -> bool:
        """Replace the size a fill removed. Returns whether anything was posted.

        The caller measures how much of this provider's resting quantity went,
        and on which side, so what is replaced is what was lost. Deciding
        instead from the shape of the remaining book let a provider filled on
        its first level while still showing a second one conclude that nothing
        had happened to it, and let a provider that had never posted a side
        create one on somebody else's execution. By design the age is
        untouched: being hit is not the passage of time.
        """
        geo = getattr(self, '_geometry', None)
        if geo is None or not getattr(self, 'last_quoted', False):
            return False
        wants = [(side, qty) for side, qty in (('bid', bid_taken), ('ask', ask_taken))
                 if qty > 1e-12]
        if not wants:
            return False
        spread = self.market.spread()
        if spread is None:
            return False
        mid = 0.5 * (spread['bid'] + spread['ask'])
        if mid <= 0:
            return False
        tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01
        half_spread = max(0.5 * tick, geo['half_spread'])
        posted = False
        for side, want in wants:
            remaining = float(want)
            for level in range(geo['levels']):
                if remaining <= 1e-12:
                    break
                # Split the amount actually taken across the available quote
                # levels.  The remembered base quantity is only the normal
                # shape of one level; it is not a cap on a fill that may have
                # consumed several randomly sized levels.
                levels_left = max(1, geo['levels'] - level)
                qty = min(remaining, max(1.0, _math.ceil(remaining / levels_left)))
                if qty <= 0:
                    break
                offset = half_spread * (1.0 + geo['level_step'] * level)
                raw = mid - offset if side == 'bid' else mid + offset
                price = (self.market.round_price(raw)
                         if hasattr(self.market, 'round_price') else round(raw, 2))
                if side == 'bid' and price >= spread['ask']:
                    price = (self.market.round_price(spread['ask'] - tick)
                             if hasattr(self.market, 'round_price')
                             else round(spread['ask'] - tick, 2))
                if side == 'ask' and price <= spread['bid']:
                    price = (self.market.round_price(spread['bid'] + tick)
                             if hasattr(self.market, 'round_price')
                             else round(spread['bid'] + tick, 2))
                if price <= 0:
                    break
                order = self._stamp_quote(
                    Order(price, qty, side, self, ttl=self.ttl),
                    reference_mid=mid, quote_level=level,
                )
                if self.market.limit_order(order) and order.qty > 0:
                    self.orders.append(order)
                    posted = True
                    # Only accepted resting quantity replaces the fill.  A
                    # rejected order must not consume the remaining amount;
                    # a later, cheaper/dearer level may still be feasible.
                    remaining -= order.qty
        if posted:
            self._replenishments = getattr(self, '_replenishments', 0) + 1
        return posted


class FastRecyclerLP(RestingQuoteProvider, Random):
    """Fast electronic LP that rapidly replenishes near-mid depth.

    The class represents short-lived, low-inventory liquidity provision.
    Quotes have independent cancellation clocks, cluster close to the mid,
    and become less aggressive when toxicity or funding stress rises.
    """

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 env: MarketEnvironment = None,
                 levels: int = 2,
                 ttl: int = 2,
                 base_qty: int = 2,
                 max_qty: int = 5,
                 base_spread_bps: float = 1.6,
                 base_withdraw_prob: float = 0.10,
                 # The most this provider abstains from quoting when the
                 # stress index is at its maximum. It is a retreat of a tenth
                 # of its making and not a withdrawal, which is what the
                 # venue evidence shows a non bank doing under stress.
                 stress_abstention: float = 0.10,
                 refresh_tol_bps: float = 2.0,
                 vol_multiple: float = 1.0,
                 **kwargs):
        super().__init__(market, cash, assets, env=env, label='FastRecyclerLP', **kwargs)
        self.type = 'FastRecyclerLP'
        self.levels = max(1, int(levels))
        self.ttl = max(1, int(ttl))
        self.base_qty = max(1, int(base_qty))
        self.max_qty = max(self.base_qty, int(max_qty))
        self.base_spread_bps = float(base_spread_bps)
        self.base_withdraw_prob = max(0.0, float(base_withdraw_prob))
        self.stress_abstention = max(0.0, float(stress_abstention))
        self.refresh_tol_bps = max(0.0, float(refresh_tol_bps))
        # ``ttl`` is the median of an independent per-order cancellation
        # clock, not a provider-wide deterministic refresh date.  A private
        # stream keeps providers asynchronous and keeps later lifecycle draws
        # independent of the order in which the simulator shuffles agents.
        self._stochastic_order_lifecycle = True
        self._lifecycle_rng = random.Random(random.getrandbits(64))
        self.vol_multiple = float(vol_multiple)
        self.last_quoted = False
        self._geometry = None
        self._replenishments = 0
        self.completed_order_lifetimes: List[float] = []
        self.order_lifecycle_events: List[dict] = []

    def _reference_mid(self, spread: dict) -> float:
        book_mid = 0.5 * (spread['bid'] + spread['ask'])
        if self.env is None:
            return book_mid
        fair_price = getattr(self.env, 'fair_price', None)
        if fair_price is None or fair_price <= 0:
            return book_mid
        fair_weight = 0.20 + 0.25 * getattr(self.env, 'systemic_liquidity', 1.0)
        fair_weight = max(0.15, min(0.45, fair_weight))
        return book_mid + fair_weight * (fair_price - book_mid)

    def call(self):
        spread = self.market.spread()
        if spread is None:
            self._cancel_all(reason='no_reference')
            self.last_quoted = False
            return

        mid = self._reference_mid(spread)
        if mid <= 0:
            self._cancel_all(reason='no_reference')
            self.last_quoted = False
            return

        # Expiry is exchange-side and per order.  Here only quotes whose own
        # reference price is stale are amended; a fresh replacement never
        # inherits the residual age of the quote set it joined.
        self._cancel_stale_orders(mid, self.refresh_tol_bps)
        live = [o for o in self.orders if getattr(o, 'qty', 0.0) > 0.0]
        present = {
            side: {int(getattr(o, '_quote_level', 0)) for o in live
                   if getattr(o, 'order_type', None) == side}
            for side in ('bid', 'ask')
        }
        missing = {
            side: [level for level in range(self.levels)
                   if level not in present[side]]
            for side in ('bid', 'ask')
        }
        if not missing['bid'] and not missing['ask']:
            self.last_quoted = True
            return

        sigma = getattr(self.env, 'sigma', 0.01) if self.env is not None else 0.01
        funding = getattr(self.env, 'funding_cost', 0.0) if self.env is not None else 0.0
        liquidity = getattr(self.env, 'systemic_liquidity', 1.0) if self.env is not None else 1.0
        toxic_bias = abs(getattr(self.env, 'toxic_flow_bias', 0.0)) if self.env is not None else 0.0

        # The constant term is the probability of not quoting at all in a
        # period when nothing is wrong. It was a tenth, so each provider went
        # silent about once every ten seconds in calm conditions, and with a
        # two tick order life that opened one sided gaps in the near book.
        # A non bank market maker on a primary venue refreshes continuously
        # and manages risk by widening and not by disappearing, so the
        # calm rate of abstention belongs at or near zero.
        #
        # The stress terms used to carry the retreat, and unbounded they
        # reached the cap of 0.85 in a crisis, which took this provider out
        # of the book entirely: measured over the crisis window it held 0.2%
        # of the touch against 41.5% before the shock, and 0.0% afterwards.
        # The evidence says otherwise. On the ECB contact group figures for
        # EBS a non bank's make:take moved only from 33:67 to 30:70 through a
        # stress event, a retreat of about a tenth of its making, while a
        # bank's moved from 55:45 to 70:30. The stress index therefore scales
        # a small abstention and not the whole quote set, and the risk it
        # represents is charged in the width below, which is where a provider
        # that stays actually puts it.
        stress_index = min(1.0, 0.35 * toxic_bias + 0.25 * (1.0 - liquidity))
        withdraw_prob = max(0.0, self.base_withdraw_prob
                            + self.stress_abstention * stress_index)
        if random.random() < min(0.85, withdraw_prob):
            self._cancel_all(reason='provider_withdrawal')
            self.last_quoted = False
            return

        # A quote left in the book for ``ttl`` ticks is exposed to the price
        # moving away from it, and the compensation for that is the volatility
        # of the price over the life of the quote. Charging in units of the
        # stress index instead made this provider quote 3.4 basis points in a
        # market moving 35 basis points a tick, and quote the same 3.4 in a
        # market moving 0.13, so it was never pricing the risk it bore.
        vol_bps = getattr(self.env, 'price_sigma_bps', 1e4 * 0.35 * sigma) \
            if self.env is not None else 1e4 * 0.35 * sigma
        total_spread_bps = (self.base_spread_bps
                            + self.vol_multiple * vol_bps * _math.sqrt(self.ttl)
                            + 120.0 * funding)
        total_spread_bps *= 1.0 + 0.60 * toxic_bias + 0.35 * (1.0 - liquidity)
        tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01
        # The floor is half a tick, so the tightest quotable market is one
        # tick wide. It used to be a hundredth of a price unit, which at a
        # numeraire of one hundred is a whole basis point of half spread and
        # therefore a two basis point market, whatever the rule above
        # computed. At the calibrated coefficients the rule produced 2.02
        # basis points and the floor sat just underneath, so it never showed
        # itself, and the quoted spread of this model was a constant rather
        # than a behaviour. Lowering any quoting coefficient moved nothing.
        # The constraint that actually exists is the price grid, and it is a
        # grid, not an absolute amount of quote currency.
        half_spread = max(0.5 * tick, mid * total_spread_bps / 20_000.0)
        self._remember_geometry(half_spread,
                                max(1, min(self.max_qty, self.base_qty)),
                                self.levels, 0.45)

        for side in ('bid', 'ask'):
            # A provider quoting around its own reference can find that
            # reference on the far side of somebody else's. After a
            # repricing this one follows the fundamental further than the
            # dealer does, so in a crisis its quote lands through the stale
            # book. Clamping it to one increment behind the touch is what
            # made the crisis spread narrower than the calm one: the
            # provider had widened, from one and a quarter basis points to
            # nearly five, and the clamp put it on the grid anyway. The book
            # then showed a tight spread made of quotes that would have
            # traded against each other had either been allowed to.
            #
            # It trades instead, taking only the size shown, and does not
            # also post that side this period.
            if missing[side]:
                # Re-read the book: a trade on the side handled first has
                # already changed it, and deciding the second side against
                # the pre-loop snapshot could send a second market order on
                # a comparison that is no longer true.
                live = self.market.spread()
                if live is None:
                    break
                tightest = half_spread * (1.0 + 0.45 * min(missing[side]))
                raw = mid - tightest if side == 'bid' else mid + tightest
                crosses = ((side == 'bid' and raw >= live['ask'])
                           or (side == 'ask' and raw <= live['bid']))
                if crosses:
                    against = 'ask' if side == 'bid' else 'bid'
                    book = self.market.order_book.get(against)
                    best = getattr(book, 'first', None) if book is not None else None
                    shown = 0.0
                    if best is not None:
                        top = float(getattr(best, 'price', 0.0) or 0.0)
                        for resting in book:
                            if abs(float(getattr(resting, 'price', 0.0) or 0.0) - top) > 1e-12:
                                break
                            if getattr(resting, 'trader', None) is self:
                                continue
                            shown += float(getattr(resting, 'qty', 0.0) or 0.0)
                    take = int(min(shown, self.max_qty))
                    if take > 0:
                        # Having traded, it does not also post this side.
                        if side == 'bid':
                            self._buy_market(take)
                        else:
                            self._sell_market(take)
                        continue
                    # Nothing could be taken, so the crossing is notional.
                    # Falling through leaves the level loop to place the
                    # quote behind the touch, which is a narrower quote than
                    # intended but still a quote; skipping would withdraw
                    # the provider from a side it meant to show.

            for level in missing[side]:
                level_mult = 1.0 + 0.45 * level
                offset = half_spread * level_mult
                qty_scale = max(0.5, liquidity) * (1.0 - 0.25 * level)
                qty = max(1, min(
                    self.max_qty,
                    int(round(self.base_qty * qty_scale + random.random())),
                ))
                raw = mid - offset if side == 'bid' else mid + offset
                price = (self.market.round_price(raw)
                         if hasattr(self.market, 'round_price') else round(raw, 2))
                if side == 'bid' and price >= spread['ask']:
                    price = (self.market.round_price(spread['ask'] - tick)
                             if hasattr(self.market, 'round_price')
                             else round(spread['ask'] - tick, 2))
                if side == 'ask' and price <= spread['bid']:
                    price = (self.market.round_price(spread['bid'] + tick)
                             if hasattr(self.market, 'round_price')
                             else round(spread['bid'] + tick, 2))
                order = self._stamp_quote(
                    Order(price, qty, side, self),
                    reference_mid=mid, quote_level=level,
                )
                if self.market.limit_order(order) and order.qty > 0:
                    self.orders.append(order)

        self.last_quoted = bool(self.orders)


class Fundamentalist(Trader):
    """
    Fundamentalist traders strictly believe in the information they receive. If they find an ask
    order with a price lower or a bid order with a price higher than their estimated present
    value, i.e. E(V|Ij,k), they accept the limit order, otherwise they put a new limit order
    between the former best bid and best ask prices.

    In multi-venue mode (amm_pools provided), the agent trades when the CLOB
    mid-price deviates from *fundamental_rate* — replaces former FXFundamentalist.
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0, access: int = 1,
                 # multi-venue params
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 venue_choice_rule: str = 'fixed_share',
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 routing_cost_scale_bps: float = 4.0,
                 routing_prior_mix_cap: float = 0.18,
                 routing_basis_scale_bps: float = 50.0,
                 routing_clob_depth_multiple: float = 10.0,
                 routing_amm_depth_multiple: float = 8.0,
                 routing_rng: Optional[random.Random] = None,
                 # FX-fundamentalist params (used only in multi-venue mode)
                 fundamental_rate: float = 100.0,
                 fx_gamma: float = 5e-3,
                 fx_q_max: int = 10,
                 flow_role: str = 'LeveragedDirectional',
                 # Book-side price discovery parameters. Both are
                 # dimensionless multiples of the price volatility of one
                 # period, so they carry no currency and no price level of
                 # their own and need not be restated when either changes.
                 observation_noise_multiple: float = 1.0,
                 quote_offset_multiple: float = 0.5):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        :param fundamental_rate: exogenous fair FX rate (multi-venue mode)
        :param fx_gamma: min mispricing to trigger FX trade
        :param fx_q_max: max FX trade size
        """
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, venue_choice_rule=venue_choice_rule,
                         deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std,
                         routing_cost_scale_bps=routing_cost_scale_bps,
                         routing_prior_mix_cap=routing_prior_mix_cap,
                         routing_basis_scale_bps=routing_basis_scale_bps,
                         routing_clob_depth_multiple=routing_clob_depth_multiple,
                         routing_amm_depth_multiple=routing_amm_depth_multiple,
                         routing_rng=routing_rng)
        self.type = 'Fundamentalist'
        self.access = access
        self.fundamental_rate = fundamental_rate
        self.fx_gamma = fx_gamma
        self.fx_q_max = fx_q_max
        self.flow_role = flow_role
        self.observation_noise_multiple = max(0.0, float(observation_noise_multiple))
        self.quote_offset_multiple = max(0.0, float(quote_offset_multiple))
        # How far its own reading may move away from a resting order before
        # that order stops representing an opinion and is withdrawn.
        self.valuation_refresh_tol_bps = 25.0

    def _observed_value(self) -> Optional[float]:
        """This trader's own noisy reading of the latent value.

        Nobody observes the fundamental. A trader that reads it exactly is
        an oracle, and a market made of oracles does not discover a price,
        it copies one. The reading is therefore drawn around the latent
        value with a dispersion proportional to the volatility of one
        period, so that the market aggregates several disagreeing opinions
        and the price is the outcome of that disagreement.
        """
        if self.env is None:
            return None
        fair_price = getattr(self.env, 'fair_price', None)
        if fair_price is None or fair_price <= 0:
            return None
        if self.observation_noise_multiple <= 0.0:
            return float(fair_price)
        sigma = abs(float(getattr(self.env, 'price_sigma', 0.0) or 0.0))
        if sigma <= 0.0:
            return float(fair_price)
        return float(fair_price) * (
            1.0 + random.gauss(0.0, self.observation_noise_multiple * sigma)
        )

    @staticmethod
    def evaluate(dividends: list, risk_free: float):
        """
        Evaluate stock using constant dividend model.
        """
        divs = dividends  # expected value of future dividends
        r = risk_free  # risk-free rate

        perp = divs[-1] / r / (1 + r)**(len(divs) - 1)  # perpetual payments
        known = sum([divs[i] / (1 + r)**(i + 1) for i in range(len(divs) - 1)]) if len(divs) > 1 else 0
        return known + perp

    @staticmethod
    def draw_quantity(pf, p, gamma: float = 5e-3):
        q = round(abs(pf - p) / p / gamma)
        return min(q, 5)

    def call(self):
        # ---------- multi-venue mode (FX fundamentalist) ------------------
        if self.multi_venue:
            self.last_routing_attempt = None
            if self.defaulted:
                return None
            try:
                mid = self.clob.mid_price()
            except Exception:
                return None
            fair = self.env.fair_price if (self.env is not None and hasattr(self.env, 'fair_price')) else self.fundamental_rate
            session_mult = getattr(self.env, 'session_flow_multiplier', 1.0) if self.env is not None else 1.0
            gamma = self.fx_gamma / max(0.75, session_mult)
            mispricing = (fair - mid) / mid
            if abs(mispricing) < gamma:
                return None
            side = 'buy' if mispricing > 0 else 'sell'
            Q = min(round(abs(mispricing) / gamma), self.fx_q_max)
            if self.flow_role == 'LeveragedDirectional':
                Q = min(self.fx_q_max, max(1, int(round(Q * max(1.0, session_mult)))))
            if Q <= 0:
                return None
            venue = self.choose_venue(Q, side)
            try:
                common_reference_price = float(self._clob_venue().mid_price())
            except Exception:
                common_reference_price = float('nan')
            result = self._execute_on_venue(venue, Q, side)
            result['common_reference_price'] = common_reference_price
            exec_q = result.get('executed_qty', 0)
            self.last_routing_attempt = self._make_routing_attempt(
                venue, side, Q, result
            )
            if exec_q <= 0:
                return None
            cls_info = self._classify(exec_q)
            rec = self._make_trade_record(venue, side, exec_q, result, cls_info)
            self.trades.append(rec)
            return rec

        # ---------- classic single-venue mode --------------
        # FX-aware: when an env.fair_price is available we use it as the
        # fundamental anchor (the FX paper's Fundamentalist trades against
        # the latent fair value, not against a dividend stream). The
        # dividend DCF is preserved only as a legacy fallback for stock-
        # market scenarios with no env attached. This lets the same class
        # participate as a CLOB-side *limit-order provider* anchored on
        # env.fair_price (placing passive bids/offers around fair) — a
        # complementary role to the FX-mode Fundamentalist, which acts as
        # a *liquidity taker* via choose_venue + market-order execution.
        # The reading is this trader's own and by design is not the
        # latent value itself; see ``_observed_value``.
        pf = self._observed_value()
        if pf is None:
            pf = round(self.evaluate(self.market.dividend(self.access),
                                     self.market.risk_free), 1)
        p = self.market.price()
        spread = self.market.spread()
        t_cost = self.market.transaction_cost

        if spread is None:
            return

        # A resting valuation has to be withdrawn when the valuation moves
        # away from it. Without this the trader placed an order against its
        # reading before a shock and left it there: measured across the
        # crisis window its quotes stood two to three hundred basis points
        # from the mid for hundreds of periods, which is not an opinion
        # about value but an order nobody cancelled. The book's depth was
        # counting them.
        stale = [
            order for order in self.orders
            if getattr(order, 'qty', 0.0) > 0.0
            and abs(float(order.price) - pf) / max(pf, 1e-9) * 1e4
            > self.valuation_refresh_tol_bps
        ]
        for order in stale:
            try:
                self._cancel_order(order, reason='stale_valuation')
            except Exception:
                pass

        random_state = random.random()
        qty = Fundamentalist.draw_quantity(pf, p)
        if not qty:
            return

        # How far from its own valuation this trader is willing to rest.
        # It used to be an exponential draw with a mean of two and a half
        # price units, which at a numeraire of one hundred is two hundred
        # and fifty basis points: an order that could never be at the touch
        # of a market whose spread is under one. The offer therefore stood
        # so far away that the whole of price discovery ran through the
        # market orders in the branches below, and this class never set a
        # price. The offset is now a multiple of the volatility of one
        # period and is floored at one tick, so a resting valuation is a
        # quote and not a gesture.
        def offset() -> float:
            sigma = abs(float(getattr(self.env, 'price_sigma', 0.0) or 0.0)) \
                if self.env is not None else 0.0
            tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01
            return max(tick, pf * self.quote_offset_multiple * sigma
                       + abs(random.gauss(0.0, tick)))

        # Rounded onto the venue's own grid and not to a tenth of a
        # price unit, which at this numeraire was a ten basis point sieve
        # and coarser than the spread being measured.
        ask_t = self.market.round_price(spread['ask'] * (1 + t_cost))
        bid_t = self.market.round_price(spread['bid'] * (1 - t_cost))
        if random_state > .45:
            random_state = random.random()
            if pf >= ask_t:
                if random_state > .5:
                    self._buy_market(qty)
                else:
                    self._sell_limit(qty, (pf + offset()) * (1 + t_cost))
            elif pf <= bid_t:
                if random_state > .5:
                    self._sell_market(qty)
                else:
                    self._buy_limit(qty, (pf - offset()) * (1 - t_cost))
            elif ask_t > pf > bid_t:
                if random_state > .5:
                    self._buy_limit(qty, (pf - offset()) * (1 - t_cost))
                else:
                    self._sell_limit(qty, (pf + offset()) * (1 + t_cost))
        else:
            if self.orders:
                self._cancel_order(self.orders[0])


class MarketMaker(Trader):
    """
    MarketMaker creates limit orders on both sides of the spread trying to gain on
    spread between bid and ask prices, and maintain its assets to cash ratio in balance.

    When *env* is provided (multi-venue mode), the quoted spread and depth become
    functions of exogenous σ_t and c_t (Brunnermeier–Pedersen model) — replaces
    former FXMarketMaker.
    """

    def __init__(self, market: ExchangeAgent, cash: float, assets: int = 0, softlimit: int = 100,
                 # multi-venue / FX params
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 # Brunnermeier-Pedersen params (used when env is set)
                 alpha0: float = 1.5, alpha1: float = 300.0, alpha2: float = 500.0,
                 alpha3: float = 50.0,
                 d0: float = 50.0, d1: float = 750.0, d2: float = 500.0,
                 d3: float = 30.0,
                 d_min: float = 3.0, n_levels: int = 5,
                 level_step_ticks: float = 2.0,
                 inv_skew_bps: float = 0.3,
                 client_flow_intensity: float = 0.0,
                 client_flow_persistence: float = 0.85,
                 venue_interaction_mode: str = 'competition',
                 amm_spread_impact_bps: float = 3.0,
                 amm_depth_impact: float = 60.0,
                 # When 0/None the reference quote size is derived from
                 # d0/n_levels (the dealer's typical per-level slice).
                 amm_reference_q: float = 0.0,
                 withdrawal_threshold: float = 0.7,
                 reentry_threshold: float = 0.4,
                 # Median of an independent per-order cancellation clock, in
                 # one-second ticks.  It must not be implemented as a hard cap:
                 # doing so put most completed dealer orders at exactly 290 and
                 # made the realised median an identity and not an outcome.
                 quote_life: int = 290,
                 quote_refresh_tol_bps: float = 20.0,
                 revenue_horizon: int = 300,
                 # Fraction of the spread it would quote now, inside
                 # which a resting quote is withdrawn and not left
                 # to become the best price in the market.
                 stale_touch_ratio: float = 0.06,
                 loss_threshold_bps: float = 50.0,
                 min_withdraw_ticks: int = 4,
                 reentry_ticks: int = 3,
                 withdrawal_confirmation_ticks: int = 2):
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std)
        self.type = 'Market Maker'
        self.softlimit = softlimit
        # Position this dealer treats as flat. Zero for an incumbent, which is
        # what every risk term below assumed. A facility arm is endowed with
        # half its capital in the base currency so that it carries the same
        # exposure as the pool it is compared against, and without a reference
        # that endowment reads as an exposure: the depth rule subtracted a
        # tenth of it and the position opened at the soft limit.
        # Most customer volume in spot FX is internalised bilaterally against a
        # dealer's own client base, and only the residual reaches the
        # interdealer book \citep{bis2025}. A dealer's position is therefore
        # dominated by a franchise that is its own, while this model gave every
        # dealer the same book to trade against. Measured over a thousand
        # periods the pairwise correlation of dealer inventories was 0.65 on
        # average and 0.95 at its highest, which is what brings the sector to
        # its withdrawal thresholds together.
        #
        # Each dealer draws a private signed client flow with its own
        # persistence and its own stream, so the franchises are independent.
        # An intensity of zero reproduces the model without them.
        self.client_flow_intensity = max(0.0, float(client_flow_intensity))
        self.client_flow_persistence = min(0.99, max(0.0, float(client_flow_persistence)))
        self._client_flow = 0.0
        # Created on first use. Drawing the seed in the constructor would take
        # a number from the global stream and shift every later draw in the
        # model, so a franchise of zero intensity would not be the model
        # without franchises.
        self._client_rng = None

        self.inventory_reference = 0.0
        self.ul = softlimit
        self.ll = -softlimit

        # Position a stood-down dealer is content to carry, and the most of
        # its limit it will show in one period while working out of the
        # rest. Both are fractions of its own limit, so neither carries a
        # currency or a price level.
        self.unwind_floor_ratio = 0.25
        # A resting quote is withdrawn once the price has drifted to
        # within this fraction of the spread the dealer would quote
        # now. It keeps a stale order from becoming the best price.
        self.stale_touch_ratio = max(0.0, float(stale_touch_ratio))
        self.unwind_rate = 0.05
        self.panic = False
        # Brunnermeier-Pedersen coefficients
        # A facility arm quotes off its own position and knows nothing about
        # market stress, which is what the reserve priced pool does: its price
        # is a function of its reserves and of nothing else. Left on, the
        # multiplicative liquidity loading below widened an arm declared to
        # show a fixed spread of three and a half basis points to between nine
        # and twelve, so an arm built to isolate standing availability was in
        # fact a dealer that widened threefold in stress.
        self.state_independent_quote = False
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3    # OFI-based spread widening
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3            # OFI-based depth reduction
        self.d_min = d_min
        self.n_levels = max(1, n_levels)
        self.level_step_ticks = max(0.5, float(level_step_ticks))
        self.inv_skew_bps = inv_skew_bps  # inventory skew coefficient
        self.venue_interaction_mode = venue_interaction_mode
        self.amm_spread_impact_bps = max(0.0, amm_spread_impact_bps)
        self.amm_depth_impact = max(0.0, amm_depth_impact)
        # Scale the AMM probe size to the dealer's typical per-level quote
        # so the support score reflects pricing for trades of the dealer's
        # native size, not an arbitrary 5-unit hard-coded probe. Per-level
        # quote ~ d0/n_levels (touch slice in the inverse pyramid).
        derived_q = max(1.0, float(d0) / max(1, int(n_levels)))
        self.amm_reference_q = max(1.0, float(amm_reference_q if amm_reference_q else derived_q))
        self.withdrawal_threshold = max(0.1, float(withdrawal_threshold))
        self.reentry_threshold = max(0.0, min(self.withdrawal_threshold, float(reentry_threshold)))
        self.loss_threshold_bps = max(1.0, float(loss_threshold_bps))
        # Rolling mark-to-market P&L over a horizon. This includes realised
        # spread earnings and the revaluation of inventory carried into a
        # price move. The instantaneous rate it replaces was a smoothed per-tick loss, which
        # has no horizon and therefore no threshold that can be anchored: at
        # one second to the tick a threshold of twenty basis points meant a
        # dealer losing a fifth of a per cent of its capital every second.
        self._revenue_horizon = max(2, int(revenue_horizon))
        self._wealth_hist: List[float] = []
        self._capital_base: float = 0.0
        self.quote_life = max(1, int(quote_life))
        self.completed_order_lifetimes: List[float] = []
        self.order_lifecycle_events: List[dict] = []
        self._lifecycle_rng = random.Random(random.getrandbits(64))
        # How far the market may move before a resting quote is amended. Set
        # near the width the dealer itself quotes, so it replaces a price the
        # market has walked away from and leaves one that is still competitive.
        self.quote_refresh_tol_bps = max(0.0, float(quote_refresh_tol_bps))
        self._quote_mid = None
        self.min_withdraw_ticks = max(1, int(min_withdraw_ticks))
        self.reentry_ticks = max(1, int(reentry_ticks))
        # Leaving a primary venue is a discrete, costly decision.  One noisy
        # OFI/inventory observation makes the dealer defensive immediately,
        # but full withdrawal requires persistence.  Without this confirmation
        # the most risk-sensitive dealer occasionally left a calm book for one
        # isolated tick even though its loss was essentially zero and every
        # market-state stress component was inactive.
        self.withdrawal_confirmation_ticks = max(
            1, int(withdrawal_confirmation_ticks)
        )
        self._withdrawal_confirmation_count = 0

        # Inventory & order-flow tracking
        # The net position is read from ``Trader.assets`` and is not carried
        # separately. There used to be two: the withdrawal score read
        # ``assets``, which every fill updates the moment it happens, while
        # the quote skew and the one sided thinning read a private counter
        # that was only refreshed on the dealer's next call by inspecting its
        # own old orders. During a scenario pause that call is skipped and the
        # orders are cancelled, so the fills were never counted and the two
        # drifted apart exactly when the shock was doing its work: one dealer
        # carried assets of 18.8 against a private counter still reading 3.
        # The dealer then skewed and thinned the wrong side.
        self._ofi_window: List[float] = []  # recent order-flow imbalance
        self._ofi_maxlen: int = 20
        self.mm_state: str = 'active'
        self.mm_state_timer: int = 0
        self.mm_withdrawal_score: float = 0.0
        # Set by the simulator when the scenario writes a pause, cleared as
        # soon as the dealer quotes again. It separates a pause imposed on the
        # dealer from a withdrawal the dealer chose.
        self.mm_forced_pause: bool = False
        self.mm_state_details: Dict[str, float] = {}
        self._mtm_wealth_prev: Optional[float] = None
        self._mtm_pnl_bps_ewma: float = 0.0
        self._loss_bps_ewma: float = 0.0
        self._signal_decay: float = 0.85

    def _recent_ofi(self) -> float:
        """Average recent order-flow imbalance (positive = buy pressure)."""
        if not self._ofi_window:
            return 0.0
        return sum(self._ofi_window) / len(self._ofi_window)

    def _update_ofi(self):
        """Blend book imbalance with realised taker flow imbalance."""
        sv = self.market.spread_volume()
        if sv is not None:
            book_ofi = (sv['bid'] - sv['ask']) / max(1, sv['bid'] + sv['ask'])
        else:
            book_ofi = 0.0
        flow_ofi = 0.0
        if self.env is not None:
            flow_ofi = getattr(
                self.env,
                'clob_order_flow_imbalance',
                getattr(self.env, 'order_flow_imbalance', 0.0),
            )
        ofi = 0.4 * book_ofi + 0.6 * flow_ofi
        self._ofi_window.append(ofi)
        if len(self._ofi_window) > self._ofi_maxlen:
            self._ofi_window.pop(0)

    def _liquidity_factor(self) -> float:
        if self.env is None:
            return 1.0
        return max(0.2, min(1.0, getattr(self.env, 'systemic_liquidity', 1.0)))

    def _reference_mid(self, spread: Optional[dict], fair_mid: Optional[float]) -> Optional[float]:
        """Dealer-internal reference mid as a weighted average of the
        observable book mid and the latent fair price.

        Real FX dealers run internal pricing engines that incorporate
        macro / cross-rate signals beyond the visible CLOB (Brunnermeier &
        Pedersen 2009; Karnaukh, Ranaldo, Soderlind 2015), but the weight
        on the latent anchor is small in calm markets (the book is the
        primary information source) and shrinks further in stress (when
        the book signal is the most reliable thing the dealer has).

        Cap reduced from 0.35 to 0.20 to keep dealer behaviour driven by
        observable order-flow and not the exogenous fair-price series.
        """
        book_mid = None
        if spread is not None:
            book_mid = (spread['bid'] + spread['ask']) / 2.0

        if book_mid is None:
            return fair_mid
        if fair_mid is None or fair_mid <= 0:
            return book_mid

        sigma_base = getattr(self.env, 'sigma_low', None)
        stress_scale = 1.0
        if sigma_base is not None and sigma_base > 0:
            stress_scale = max(1.0, self.env.sigma / sigma_base)

        # The weight was additionally divided by the size of the gap it is
        # meant to close, so the dealer trusted the latent value least at the
        # moment it disagreed with the book most. After a three per cent
        # repricing the three suppressors multiplied out to 0.0037 and the
        # weight sat on its floor of 0.02: the dealer could see the price had
        # moved three hundred basis points and was instructed to believe the
        # book. A large disagreement is the strongest evidence the book is
        # stale, not a reason to discount the evidence, so that term is gone.
        #
        # The other two are kept. A noisier estimate deserves less weight,
        # which is what dividing by the stress scale does, and a dealer with
        # a damaged liquidity factor is in no position to lead a price. The
        # cap of a fifth is also kept: the latent value is not something a
        # dealer observes, and a dealer weighting it heavily is copying an
        # oracle instead of intermediating. The book carries its own
        # participants who price off the fundamental directly, and price
        # discovery is properly their work.
        liquidity_factor = self._liquidity_factor()
        fair_weight = 0.20 * liquidity_factor / stress_scale
        fair_weight = max(0.02, min(0.20, fair_weight))
        return book_mid + fair_weight * (fair_mid - book_mid)

    def _amm_support_score(self, mid: float) -> float:
        """Competitive quality of AMM liquidity beside the CLOB dealer.

        The score is high only when AMM liquidity is simultaneously deep,
        cheap for a representative trade, and internally consistent in price.
        This avoids giving the CLOB an automatic bonus just because an AMM
        exists in the market.
        """
        if not self.amm_pools or mid <= 0:
            return 0.0

        scores = []
        pool_mids = []
        stress_scale = 1.0
        sigma_base = getattr(self.env, 'sigma_low', None)
        if sigma_base is not None and sigma_base > 0:
            stress_scale = max(1.0, self.env.sigma / sigma_base)

        for pool in self.amm_pools.values():
            try:
                depth = max(0.0, pool.effective_depth(mid))
                buy_cost = pool.quote_buy(self.amm_reference_q, S_t=mid)['cost_bps']
                sell_cost = pool.quote_sell(self.amm_reference_q, S_t=mid)['cost_bps']
                pool_mid = pool.mid_price()
                avg_cost = 0.5 * (buy_cost + sell_cost)
                if not (_math.isfinite(avg_cost) and _math.isfinite(pool_mid) and pool_mid > 0):
                    continue

                alignment_bps = abs(pool_mid - mid) / mid * 10_000.0
                depth_score = min(depth / max(20.0 * self.amm_reference_q, 1.0), 2.0)
                cost_score = 1.0 / (1.0 + avg_cost / 25.0)
                alignment_score = 1.0 / (1.0 + alignment_bps / 20.0)
                scores.append(depth_score * cost_score * alignment_score)
                pool_mids.append(pool_mid)
            except Exception:
                pass

        if not scores:
            return 0.0

        support = sum(scores) / len(scores)
        if len(pool_mids) > 1:
            dispersion_bps = (max(pool_mids) - min(pool_mids)) / mid * 10_000.0
            support *= 1.0 / (1.0 + dispersion_bps / 15.0)

        return max(0.0, min(1.0, support / stress_scale))

    def _venue_interaction_state(self, mid: float) -> dict:
        """Translate AMM presence into dealer risk modifiers.

        Three modes:
          * 'none'        — AMM presence does not affect dealer quoting.
                            Use for experiments isolating dealer mechanics.
          * 'toxicity'    — AMM cream-skims small benign flow, leaving the
                            CLOB dealer with a more toxic mix (Lehar &
                            Parlour 2024; consistent with Aoyagi & Ito
                            2024 informational asymmetry channel). Spreads
                            widen and depth thins.
          * 'competition' — AMM presence makes dealer quoting marginally
                            more conservative on the inventory channel,
                            but does NOT mechanically tighten spreads or
                            inflate depth. Net effect is small and one-
                            sided (no liquidity 'subsidy' from AMM
                            existing). Earlier versions of this code
                            applied a symmetric tightening + depth boost
                            that is not in the paper and not supported by
                            the FX coexistence literature; that subsidy
                            has been removed.
        """
        liquidity_factor = self._liquidity_factor()
        state = {
            'ofi_scale': 1.0,
            'inventory_scale': 1.0,
            'liquidity_factor': liquidity_factor,
        }
        if self.venue_interaction_mode == 'none':
            return state

        support = self._amm_support_score(mid)
        if support <= 0.0:
            return state

        spread_load = min(0.35, self.amm_spread_impact_bps / 20.0) * support
        depth_load = min(0.35, self.amm_depth_impact / 150.0) * support

        if self.venue_interaction_mode == 'toxicity':
            # Higher OFI sensitivity; inventory penalty grows; effective
            # liquidity factor falls. Dealer becomes more defensive in
            # the presence of cream-skimming AMM.
            state['ofi_scale'] = 1.0 + 0.60 * spread_load
            state['inventory_scale'] = 1.0 + 0.75 * depth_load
            state['liquidity_factor'] = max(0.2, liquidity_factor * (1.0 - 0.40 * depth_load))
            return state

        # 'competition' mode (legacy default) — neutral on spread/OFI,
        # mild risk-management discipline on inventory. No subsidy.
        state['inventory_scale'] = 1.0 + 0.25 * depth_load
        return state

    def _target_spread_bps(self, mid: float) -> float:
        """Quoted spread in bps, paper Eq. (1):
            s* = alpha0 + alpha1*sigma + alpha2*c + alpha3*|OFI|.

        The systemic-liquidity factor enters multiplicatively below
        (paper supplement Eq. (10) with the ell-tilde modifier), not as
        an additive penalty, so a faded liquidity regime widens spreads
        proportionally instead of tacking on a fixed bps penalty.
        """
        venue_state = self._venue_interaction_state(mid)
        effective_ofi = abs(self._recent_ofi()) * venue_state['ofi_scale']
        # ``alpha1`` is a multiple of the volatility of the price per tick,
        # expressed in basis points, not a coefficient on the stress index. The
        # index says how stressed the market is, not how far the price moves,
        # and a spread has to compensate for the second.
        vol_bps = getattr(self.env, 'price_sigma_bps', 1e4 * 0.35 * self.env.sigma)
        base = self.alpha0 + self.alpha1 * vol_bps + self.alpha2 * self.env.funding_cost
        base += self.alpha3 * effective_ofi
        # Multiplicative liquidity loading (ell-tilde in Eq. (10)). At
        # ell = 1 (calm) this is a no-op; at ell = 0.4 (deep stress)
        # spreads widen by ~50%. Bounded to avoid runaway widening.
        liquidity_factor = max(0.2, min(1.0, venue_state['liquidity_factor']))
        if not self.state_independent_quote:
            base /= liquidity_factor
        return max(0.5, base)

    def _target_depth(self, mid: float) -> float:
        """Depth as f(σ, c, |OFI|, |inventory|), floored at d_min."""
        venue_state = self._venue_interaction_state(mid)
        effective_ofi = abs(self._recent_ofi()) * venue_state['ofi_scale']
        inventory_penalty = (0.1 * abs(self.risk_inventory)
                             * venue_state['inventory_scale'])
        d = self.d0 - self.d1 * self.env.sigma - self.d2 * self.env.funding_cost
        d -= self.d3 * effective_ofi
        d -= inventory_penalty
        if not self.state_independent_quote:
            d *= venue_state['liquidity_factor']
        return max(d, self.d_min)

    def set_inventory_reference(self, reference: float) -> None:
        """Declare the position this dealer treats as flat, and re-centre."""
        self.inventory_reference = float(reference)
        self.ul = self.inventory_reference + float(self.softlimit)
        self.ll = self.inventory_reference - float(self.softlimit)

    @property
    def risk_inventory(self) -> float:
        """Position measured from the one this dealer treats as flat."""
        return float(self.assets) - float(getattr(self, 'inventory_reference', 0.0))

    @property
    def inventory(self) -> float:
        """Net position, read from the single source that every fill moves."""
        return float(self.assets)

    @inventory.setter
    def inventory(self, value: float):
        # Kept so that tests and tools that set a position directly still
        # work. Writing the position writes the assets it stands for.
        self.assets = float(value)

    def _stamp_dealer_order(self, order: Order, reference_mid: float,
                            quote_level: int = 0) -> Order:
        return _stamp_lifecycle_order(
            self, order, self.quote_life,
            reference_mid=reference_mid, quote_level=quote_level,
        )

    def _submit_dealer_order(self, side: str, quantity: float, price: float,
                             reference_mid: float, quote_level: int = 0) -> bool:
        """Submit one quote and attach an independent lifecycle if it rests."""
        before = {id(order) for order in self.orders}
        if side == 'bid':
            posted = self._buy_limit(quantity, price)
        else:
            posted = self._sell_limit(quantity, price)
        if not posted:
            return False
        for order in self.orders:
            if id(order) not in before:
                self._stamp_dealer_order(
                    order, reference_mid=reference_mid,
                    quote_level=quote_level,
                )
        return True

    def record_order_end(self, order: Order, reason: Optional[str] = None):
        """Record one completed dealer-order lifetime exactly once."""
        _record_lifecycle_end(self, order, reason=reason)

    def lifecycle_observations(self, include_live: bool = True) -> List[dict]:
        return _lifecycle_observations(self, include_live=include_live)

    def _cancel_stale_dealer_orders(self, mid: float) -> int:
        """Withdraw quotes the market has moved away from, or onto.

        The drift tolerance alone is a symmetric rule, and the two
        directions are not symmetric for a dealer.  A quote the price has
        moved *away* from is harmless: it rests deep in the book, which is
        where a bank's liquidity is supposed to sit and why its orders live
        as long as they do.  A quote the price has moved *onto* is a
        different matter, because it is now the best price in the market at
        a level the dealer would not choose to show.

        With a tolerance of twenty basis points in a market whose spread is
        under one, that is what set the touch: the dealer's best quote sat
        on the minimum increment three periods in four, not because it had
        decided to quote there but because the mid had drifted onto an old
        order.  The realised spread of the market was then the price grid.
        Tightening the tolerance fixed the spread and destroyed the order
        lifetime, which is a published statistic, because it amended the
        deep ladder too.  Only the near side is amended here, so the touch
        is chosen and the ladder still ages.
        """
        cancelled = 0
        half_spread = mid * self._target_spread_bps(mid) / 10_000.0 / 2.0
        keep_out = self.stale_touch_ratio * half_spread
        for order in self.orders.copy():
            if getattr(order, 'qty', 0.0) <= 0.0:
                continue
            reference = getattr(order, '_quote_reference_mid', None)
            if reference is None or reference <= 0.0:
                continue
            move_bps = abs(mid - reference) / reference * 1e4
            drifted_inside = abs(float(order.price) - mid) < keep_out
            if move_bps <= self.quote_refresh_tol_bps and not drifted_inside:
                continue
            try:
                self._cancel_order(order, reason='stale_reprice')
                cancelled += 1
            except Exception:
                pass
        return cancelled

    def apply_fill(self, order: Order, fill_qty: float, fill_price: float,
                   t_cost: float, is_buy: bool, qty_before: float):
        super().apply_fill(order, fill_qty, fill_price, t_cost, is_buy, qty_before)
        if order.qty <= 1e-12:
            self.record_order_end(order, reason='fill')

    def on_trade(self, bid_taken: float = 0.0, ask_taken: float = 0.0) -> bool:
        """Replace dealer size consumed within the current one-second period.

        The replacement is a new order with new queue priority.  Quotes that
        were not filled retain their original price and age, so execution is
        not confused with a clock tick or an exchange-side amendment.
        """
        if (self.defaulted or self.mm_forced_pause
                or self.mm_state in {'withdrawn', 'reentering'}):
            return False
        wants = [(side, float(qty)) for side, qty in (
            ('bid', bid_taken), ('ask', ask_taken)
        ) if qty > 1e-12]
        if not wants:
            return False
        sp = self.market.spread()
        if sp is None:
            return False
        mid = self._reference_mid(sp, getattr(self.env, 'fair_price', None))
        if mid is None or mid <= 0:
            return False
        quote_state = self._state_quote_adjustments()
        if not quote_state['can_quote']:
            return False
        half_spread = max(
            0.5 * self.market._tick_size(),
            float(getattr(self, '_dealer_half_spread', 0.5 * self.market._tick_size())),
        )
        eff_mid = mid + self._inventory_skew(mid)
        tick = self.market._tick_size()
        # A sweep can remove more than one structural level.  Reposting the
        # entire swept quantity at level zero and then letting ``call`` rebuild
        # the other missing levels restores the same fill twice.  Allocate the
        # replacement across the levels that actually disappeared; a partial
        # fill whose level still rests is a genuine top-up at the touch.
        present = {
            side: {int(getattr(order, '_quote_level', 0))
                   for order in self.orders
                   if getattr(order, 'order_type', None) == side
                   and getattr(order, 'qty', 0.0) > 0.0}
            for side in ('bid', 'ask')
        }
        posted = False
        for side, qty in wants:
            levels = [level for level in range(self.n_levels)
                      if level not in present[side]] or [0]
            target_units = max(0, int(round(qty)))
            base_units, extra_units = divmod(target_units, len(levels))
            for position, level in enumerate(levels):
                replacement_qty = base_units + (1 if position < extra_units else 0)
                if replacement_qty <= 0:
                    continue
                offset = half_spread + level * self.level_step_ticks * tick
                if side == 'bid':
                    price = self.market.round_price(eff_mid - offset)
                    if price >= sp['ask']:
                        price = self.market.floor_price(sp['ask'] - tick)
                else:
                    price = self.market.round_price(eff_mid + offset)
                    if price <= sp['bid']:
                        price = self.market.ceil_price(sp['bid'] + tick)
                accepted = self._submit_dealer_order(
                    side, replacement_qty, price,
                    reference_mid=mid, quote_level=level,
                )
                posted = accepted or posted
        return posted

    def _inventory_skew(self, mid: float) -> float:
        """Skew mid-price away from inventory risk (bps → price offset)."""
        return -self.inv_skew_bps * self.risk_inventory * mid / 10_000.0

    def cancel_all_quotes(self, reason: Optional[str] = None):
        if reason is None:
            if getattr(self, 'defaulted', False):
                reason = 'default'
            elif (getattr(self, 'mm_forced_pause', False)
                  or int(getattr(getattr(self, 'env', None),
                                 'mm_pause_ticks', 0) or 0) > 0):
                reason = 'forced_pause'
            elif getattr(self, 'mm_state', 'active') == 'withdrawn':
                reason = 'endogenous_withdrawal'
            else:
                reason = 'dealer_cancel'
        for order in self.orders.copy():
            try:
                self._cancel_order(order, reason=reason)
            except Exception:
                pass
        self.orders.clear()

    def _mark_to_market(self, mid: float) -> float:
        return self.cash + self.assets * mid

    def _update_mtm_signals(self, mid: float):
        wealth = self._mark_to_market(mid)
        if self._mtm_wealth_prev is None:
            self._mtm_wealth_prev = wealth
            return

        base = max(abs(self._mtm_wealth_prev), 1e-9)
        pnl_bps = 10_000.0 * (wealth - self._mtm_wealth_prev) / base
        decay = self._signal_decay
        self._mtm_pnl_bps_ewma = decay * self._mtm_pnl_bps_ewma + (1.0 - decay) * pnl_bps
        self._loss_bps_ewma = decay * self._loss_bps_ewma + (1.0 - decay) * max(0.0, -pnl_bps)
        self._mtm_wealth_prev = wealth
        if self._capital_base <= 0.0:
            self._capital_base = max(abs(wealth), 1e-9)
        self._wealth_hist.append(wealth)
        if len(self._wealth_hist) > self._revenue_horizon + 1:
            del self._wealth_hist[0]

    def _withdrawal_components(self, mid: float) -> Dict[str, float]:
        sigma_base = max(getattr(self.env, 'sigma_low', self.env.sigma), 1e-6)
        funding_base = max(getattr(self.env, 'c_low', self.env.funding_cost), 1e-6)
        liquidity_factor = self._liquidity_factor()

        sigma_score = min(2.5, max(0.0, self.env.sigma / sigma_base - 1.0))
        funding_score = min(2.5, max(0.0, self.env.funding_cost / funding_base - 1.0))
        # Use Trader.assets as the single source of truth. Trader.assets
        # is updated on every fill via apply_fill; the legacy mm.inventory
        # field is kept in sync for backward compatibility but not used
        # for risk scoring.
        inventory_abs = abs(float(self.risk_inventory))
        # Normalised by the limit itself and not by one and a half times
        # it. Kirilenko and co-authors describe a market maker that supplies
        # liquidity up to a level of inventory and then stands down, so the
        # term has to reach unity at the limit and not fifty per cent beyond
        # it. Under the old denominator a dealer sitting on half its limit
        # scored a third, and the term could only bite once the limit had
        # already been breached.
        inventory_ratio = min(1.5, inventory_abs / max(float(self.softlimit), 1.0))
        # Risk trigger reads the env's single-EWMA CLOB order-flow
        # imbalance directly. self._recent_ofi() adds a second 20-tick
        # smoothing on top of the env EWMA (used for quoting), which
        # diluted a one-sided sweep by ~11x so the dealer could not
        # "feel" acute directional flow. Quoting still uses _recent_ofi
        # (calm-spread calibration depends on it).
        env_ofi = getattr(self.env, 'clob_order_flow_imbalance',
                          getattr(self.env, 'order_flow_imbalance', 0.0))
        ofi_score = min(2.0, abs(env_ofi))
        liquidity_damage = min(1.5, max(0.0, 1.0 - liquidity_factor))
        amm_support_score = 0.0
        if self.venue_interaction_mode != 'none':
            amm_support_score = min(1.5, self._amm_support_score(mid))
        # Losses enter as rolling mark-to-market P&L over a horizon, expressed
        # in basis points of the capital the dealer started with, so the
        # threshold is a fraction of capital and not a rate per second.
        pnl_window_bps = 0.0
        if len(self._wealth_hist) >= 2 and self._capital_base > 0.0:
            pnl_window_bps = 10_000.0 * (self._wealth_hist[-1]
                                         - self._wealth_hist[0]) / self._capital_base
        loss_score = min(3.0, max(0.0, -pnl_window_bps) / self.loss_threshold_bps)
        # The empirical ordering of causes puts the dealer's own book first.
        # Kirilenko and co-authors find that a market maker supplies liquidity
        # up to a level of inventory and then stands down, and Comerton-Forde
        # and co-authors find spreads widening with large positions and with
        # poor trading results, the effects being nonlinear. The earlier
        # weights had the two own book terms carrying under a tenth of the
        # threshold between them while exogenous stress indices and a shock
        # dummy carried the rest, so the dealer withdrew in response to the
        # weather and not to its own position.
        # There is no shock-window dummy here, by design. A scenario may
        # move prices, funding or flow, but a dealer leaves only if those
        # events show up in its inventory/P&L or in observable current market
        # conditions. With a unit withdrawal threshold, reaching either the
        # inventory limit or the rolling-loss tolerance is sufficient to step
        # away; the smaller market-state terms can make the dealer defensive
        # earlier but do not encode knowledge that a scenario is active.
        #
        # AMM support remains a reported diagnostic and affects quote
        # competition, but it is not an additive withdrawal trigger.  The
        # support score saturates in calm markets and fades under stress, so a
        # positive term here was effectively a calm-only intercept: it made a
        # healthy automated venue push a bank out while becoming *less*
        # important exactly when dealer balance-sheet stress arrived.  That is
        # neither a dealer loss nor an inventory constraint.
        score = (
            1.00 * inventory_ratio
            + 1.00 * loss_score
            + 0.10 * ofi_score
            + 0.15 * liquidity_damage
            + 0.04 * sigma_score
            + 0.05 * funding_score
        )
        return {
            'score': score,
            'loss_score': loss_score,
            'sigma_score': sigma_score,
            'funding_score': funding_score,
            'inventory_ratio': inventory_ratio,
            'ofi_score': ofi_score,
            'liquidity_factor': liquidity_factor,
            'amm_support_score': amm_support_score,
            # Compatibility alias for stored development diagnostics written
            # before v2. It is not an outside-option or withdrawal-score term.
            'outside_option_score': amm_support_score,
            'loss_bps_ewma': self._loss_bps_ewma,
            'mtm_pnl_bps_ewma': self._mtm_pnl_bps_ewma,
            'pnl_window_bps': pnl_window_bps,
        }

    def _update_endogenous_state(self, mid: float) -> str:
        if self.env is None:
            self.mm_state = 'active'
            self.mm_state_timer = 0
            self._withdrawal_confirmation_count = 0
            self.mm_withdrawal_score = 0.0
            self.mm_state_details = {
                'score': 0.0,
                'liquidity_factor': self._liquidity_factor(),
                'withdrawal_confirmation_count': 0,
            }
            return self.mm_state

        self._update_mtm_signals(mid)
        details = self._withdrawal_components(mid)
        score = details['score']
        defensive_threshold = max(self.reentry_threshold, 0.60 * self.withdrawal_threshold)
        withdraw_release_threshold = max(defensive_threshold, 0.85 * self.withdrawal_threshold)
        liquidity_ok = details['liquidity_factor'] >= 0.55

        self.mm_withdrawal_score = score
        self.mm_state_details = details

        if self.mm_state == 'withdrawn':
            self._withdrawal_confirmation_count = 0
            if self.mm_state_timer > 0:
                self.mm_state_timer -= 1
            if self.mm_state_timer <= 0 and score <= withdraw_release_threshold and liquidity_ok:
                self.mm_state = 'reentering'
                self.mm_state_timer = self.reentry_ticks
            self.mm_state_details['withdrawal_confirmation_count'] = 0
            return self.mm_state

        if self.mm_state == 'reentering':
            if score >= self.withdrawal_threshold:
                self._withdrawal_confirmation_count += 1
                if (self._withdrawal_confirmation_count
                        >= self.withdrawal_confirmation_ticks):
                    self.mm_state = 'withdrawn'
                    self.mm_state_timer = self.min_withdraw_ticks
                self.mm_state_details['withdrawal_confirmation_count'] = (
                    self._withdrawal_confirmation_count
                )
                return self.mm_state

            self._withdrawal_confirmation_count = 0
            if self.mm_state_timer > 0:
                self.mm_state_timer -= 1
            if self.mm_state_timer <= 0:
                self.mm_state = 'active' if score < 0.5 * self.reentry_threshold else 'defensive'
            self.mm_state_details['withdrawal_confirmation_count'] = 0
            return self.mm_state

        if score >= self.withdrawal_threshold:
            self._withdrawal_confirmation_count += 1
            if (self._withdrawal_confirmation_count
                    >= self.withdrawal_confirmation_ticks):
                self.mm_state = 'withdrawn'
                self.mm_state_timer = self.min_withdraw_ticks
            else:
                # The risk signal is actionable on the first observation: the
                # dealer widens and thins immediately while awaiting evidence
                # that justifies the fixed cost of leaving the venue.
                self.mm_state = 'defensive'
                self.mm_state_timer = 0
        elif score >= defensive_threshold:
            self._withdrawal_confirmation_count = 0
            self.mm_state = 'defensive'
            self.mm_state_timer = 0
        elif self.mm_state == 'defensive' and score > self.reentry_threshold:
            self._withdrawal_confirmation_count = 0
            self.mm_state = 'defensive'
        else:
            self._withdrawal_confirmation_count = 0
            self.mm_state = 'active'
            self.mm_state_timer = 0
        self.mm_state_details['withdrawal_confirmation_count'] = (
            self._withdrawal_confirmation_count
        )
        return self.mm_state

    def _work_out_inventory(self, mid: float):
        """A dealer that has stopped making a market still has a position.

        Withdrawal used to be absorbing: standing down cancelled the ladder,
        and with no quotes the dealer could not trade, so its inventory was
        frozen at whatever the crisis left it holding.  Since the inventory
        ratio is the largest term in the withdrawal score, a dealer that
        left with a full book could never score its way back, and measured
        over the window after the shock none of them did: zero per cent
        active across five hundred and fifty periods.

        What it posts is one sided and reduce only.  This is a dealer working
        out of a position, not one making a market: there is no second side
        and the size never exceeds the excess being unwound.

        This is only reached once the dealer has stood down, and standing
        down is itself the statement that the position is no longer one it
        wants to carry, so it does not simply wait to be filled. Resting
        alone left the position frozen: after a crisis the touch belongs to
        the fast non-bank provider, and a stood-down dealer's passive
        interest sat unfilled for as long as it was measured, with inventory
        held at four fifths of the limit for a hundred and fifty periods.

        It takes what is shown to it and rests the remainder.  A dealer
        reducing risk pays the spread on the size that is actually there;
        it does not sweep a thin book to get done, both because the price
        it would realise gets worse with every level and because the size
        beyond the touch is not on offer at the touch price.
        """
        inventory = float(self.risk_inventory)
        limit = max(1.0, float(self.softlimit))
        excess = abs(inventory) - self.unwind_floor_ratio * limit
        if excess <= 0.0 or mid <= 0:
            return
        qty = max(1, int(min(excess, self.unwind_rate * limit)))
        tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01
        side = 'bid' if inventory > 0 else 'ask'
        book = self.market.order_book.get(side)
        shown = 0.0
        best = getattr(book, 'first', None) if book is not None else None
        if best is not None:
            price = float(getattr(best, 'price', 0.0) or 0.0)
            for order in book:
                if abs(float(getattr(order, 'price', 0.0) or 0.0) - price) > 1e-12:
                    break
                if getattr(order, 'trader', None) is self:
                    continue
                shown += float(getattr(order, 'qty', 0.0) or 0.0)
        crossed = int(min(qty, shown))
        rested = qty - crossed
        # The remainder joins the market instead of undercutting it. Resting
        # it one tick from the mid made a reduce-only order the best price in
        # the book, so a dealer that had stood down was setting the touch on
        # the minimum increment, and the crisis spread collapsed onto the
        # grid instead of widening. It is shown no tighter than the price
        # already on that side, and no tighter than the dealer's own quote.
        own = mid * self._target_spread_bps(mid) / 10_000.0 / 2.0
        sp = self.market.spread()
        if inventory > 0:
            floor_px = mid + max(own, tick)
            if sp is not None:
                floor_px = max(floor_px, float(sp['ask']))
            if crossed > 0:
                self._sell_market(crossed)
            if rested > 0:
                self._sell_limit(rested, self.market.round_price(floor_px))
        else:
            cap_px = mid - max(own, tick)
            if sp is not None:
                cap_px = min(cap_px, float(sp['bid']))
            if crossed > 0:
                self._buy_market(crossed)
            if rested > 0:
                self._buy_limit(rested, self.market.round_price(cap_px))

    def _state_quote_adjustments(self) -> Dict[str, float | bool]:
        if self.mm_state == 'withdrawn':
            return {'can_quote': False, 'spread_mult': float('inf'), 'depth_mult': 0.0}

        if self.mm_state == 'reentering':
            progress = 1.0 - self.mm_state_timer / max(1, self.reentry_ticks)
            return {
                'can_quote': True,
                'spread_mult': max(1.05, 1.30 - 0.20 * progress),
                'depth_mult': min(0.80, 0.35 + 0.35 * progress),
            }

        if self.mm_state == 'defensive':
            score_ratio = min(1.5, self.mm_withdrawal_score / max(self.withdrawal_threshold, 1e-9))
            return {
                'can_quote': True,
                'spread_mult': 1.0 + 0.30 * score_ratio,
                'depth_mult': max(0.25, 1.0 - 0.40 * score_ratio),
            }

        return {'can_quote': True, 'spread_mult': 1.0, 'depth_mult': 1.0}

    def _absorb_client_flow(self) -> None:
        """Take one period of the dealer's own internalised client flow.

        The trade settles at the reference price and moves cash and inventory
        together, so the franchise transfers a position to the dealer without
        creating or destroying value.
        """
        if self.client_flow_intensity <= 0.0:
            return
        ref = getattr(self.env, 'fair_price', None) if self.env is not None else None
        if ref is None or not _math.isfinite(ref) or ref <= 0:
            return
        if self._client_rng is None:
            self._client_rng = random.Random(random.getrandbits(64))
        shock = self._client_rng.gauss(0.0, 1.0)
        self._client_flow = (self.client_flow_persistence * self._client_flow
                             + (1.0 - self.client_flow_persistence) * shock)
        qty = self.client_flow_intensity * self._client_flow
        if abs(qty) < 1e-12:
            return
        # A client buying from the dealer leaves the dealer short.
        self.assets -= qty
        self.cash += qty * float(ref)

    def call(self):
        # ---------- multi-venue mode (σ/c-dependent MM) -------------------
        if self.env is not None:
            self._absorb_client_flow()
            # Update OFI tracking
            self._update_ofi()

            # Track inventory from fills on previous orders:
            # orders that were partially filled have reduced qty.
            # The remaining quantity is written back, because a quote that
            # rests for more than one period would otherwise have its whole
            # cumulative fill counted again on every period it survives.
            for order in self.orders:
                if hasattr(order, '_mm_init_qty'):
                    filled = order._mm_init_qty - order.qty
                    if filled > 0:
                        # The position itself is already in ``assets``, which
                        # apply_fill moved at the moment of the fill. Only the
                        # tag is refreshed here so a resting quote is not
                        # counted twice.
                        order._mm_init_qty = order.qty

            sp = self.market.spread()
            fair_mid = getattr(self.env, 'fair_price', None)

            if sp is None and (fair_mid is None or fair_mid <= 0):
                return

            mid = self._reference_mid(sp, fair_mid)
            if mid is None or mid <= 0:
                return

            self._update_endogenous_state(mid)
            quote_state = self._state_quote_adjustments()
            if not quote_state['can_quote']:
                # A genuine state withdrawal still removes the ladder.  It is
                # distinct from scheduled lifecycle expiry in the telemetry.
                self.cancel_all_quotes(reason='endogenous_withdrawal')
                self._work_out_inventory(mid)
                return

            # Each order expires on its own exchange-side clock.  A stale
            # order is amended individually against the reference used when
            # that order was written; there is no 290-tick bulk refresh pulse.
            self._cancel_stale_dealer_orders(mid)
            live = [o for o in self.orders if getattr(o, 'qty', 0.0) > 0.0]
            present = {
                side: {int(getattr(o, '_quote_level', 0)) for o in live
                       if getattr(o, 'order_type', None) == side}
                for side in ('bid', 'ask')
            }
            missing = {
                side: [level for level in range(self.n_levels)
                       if level not in present[side]]
                for side in ('bid', 'ask')
            }
            if not missing['bid'] and not missing['ask']:
                return
            self._quote_mid = mid

            # Apply inventory skew: shift effective mid away from risk
            skew = self._inventory_skew(mid)
            eff_mid = mid + skew

            half_spread = (
                mid * self._target_spread_bps(mid) * quote_state['spread_mult'] / 10_000.0 / 2.0
            )
            self._dealer_half_spread = half_spread
            total_depth = max(1, int(self._target_depth(mid) * quote_state['depth_mult']))

            # Inventory-based one-sided thinning: reduce depth on the
            # overexposed side to encourage inventory mean-reversion
            deviation = self.risk_inventory
            inv_ratio = min(1.0, abs(deviation) / max(1, self.softlimit))
            bid_depth_frac = 1.0 - 0.5 * inv_ratio if deviation > 0 else 1.0
            ask_depth_frac = 1.0 - 0.5 * inv_ratio if deviation < 0 else 1.0

            # Inverse pyramid: most depth at tight spread, tapering out
            weights = list(range(self.n_levels, 0, -1))
            w_sum = sum(weights)
            tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01
            for i in range(self.n_levels):
                base_qty = max(1, int(total_depth * weights[i] / w_sum))
                # One competitive slice supplies the touch; the remaining bank
                # orders populate a deeper ladder.  Scaling every level by the
                # tiny touch spread collapsed all five orders onto one price
                # grid point, so every bank order was executed in seconds and
                # the published bank-order lifetime could not be represented.
                offset = half_spread + i * self.level_step_ticks * tick
                bid_qty = max(1, int(base_qty * bid_depth_frac))
                ask_qty = max(1, int(base_qty * ask_depth_frac))
                # Dealer quotes live on the same EBS-style price grid as every
                # other order.  Rounding to two decimals on a 0.005 grid made
                # tight bid and ask quotes collapse to the same price, execute
                # against each other, and disappear instead of supplying the
                # touch.  Directional rounding also guarantees a non-crossing
                # pair at the minimum one-tick spread.
                bid_price = (self.market.round_price(eff_mid - offset)
                             if hasattr(self.market, 'round_price')
                             else round(eff_mid - offset, 2))
                ask_price = (self.market.round_price(eff_mid + offset)
                             if hasattr(self.market, 'round_price')
                             else round(eff_mid + offset, 2))
                if bid_price >= ask_price:
                    bid_price = (self.market.floor_price(eff_mid - 0.5 * tick)
                                 if hasattr(self.market, 'floor_price')
                                 else round(eff_mid - 0.5 * tick, 2))
                    ask_price = (self.market.ceil_price(eff_mid + 0.5 * tick)
                                 if hasattr(self.market, 'ceil_price')
                                 else round(eff_mid + 0.5 * tick, 2))
                # A quote that would cross the book is not a quote, it is a
                # trade the dealer has decided to make and then declined to
                # make. Clamping it to one increment behind the touch was
                # what kept the crisis spread on the grid: after a repricing
                # the book lags the fundamental, the dealer follows the
                # fundamental further than the book has, and its intended
                # quote lands through the stale side. That happened on
                # twelve per cent of calm periods and thirty five per cent
                # of crisis periods, and on each of them the dealer was
                # placed at the touch by the clamp and not by any view
                # of its own, so the quoted spread of the market in a crisis
                # was set by an anti-crossing rule.
                #
                # A dealer holding that view sells to the stale bid instead.
                # It takes only the size that is shown, for the same reason
                # the inventory unwind does: the price beyond the touch is
                # not on offer at the touch. Having traded, it does not also
                # post on that side this period.
                place_bid = place_ask = True
                if i == 0 and sp is not None:
                    crossed_side = None
                    if ask_price <= sp['bid']:
                        crossed_side = 'bid'
                    elif bid_price >= sp['ask']:
                        crossed_side = 'ask'
                    if crossed_side is not None:
                        shown = 0.0
                        book = self.market.order_book.get(crossed_side)
                        best = getattr(book, 'first', None) if book is not None else None
                        if best is not None:
                            top = float(getattr(best, 'price', 0.0) or 0.0)
                            for resting in book:
                                if abs(float(getattr(resting, 'price', 0.0) or 0.0) - top) > 1e-12:
                                    break
                                if getattr(resting, 'trader', None) is self:
                                    continue
                                shown += float(getattr(resting, 'qty', 0.0) or 0.0)
                        take = int(min(shown, max(bid_qty, ask_qty)))
                        if take > 0:
                            if crossed_side == 'bid':
                                self._sell_market(take)
                                # Only the crossing side is withheld. This
                                # loop body places both sides, so skipping it
                                # outright also cancelled the level-0 bid,
                                # which crossed nothing: the dealer withdrew
                                # from the side it still wanted to show, on
                                # exactly the periods crossing is common.
                                place_ask = False
                            else:
                                self._buy_market(take)
                                place_bid = False

                # One side of the book can be empty after a sweep, and then
                # there is no opposing quote to stay behind. The dealer still
                # has its own reference and quotes around that; reading the
                # absent side raised a TypeError that ended the run.
                if sp is not None and bid_price >= sp['ask']:
                    bid_price = (self.market.floor_price(sp['ask'] - tick)
                                 if hasattr(self.market, 'floor_price')
                                 else round(sp['ask'] - tick, 2))
                if sp is not None and ask_price <= sp['bid']:
                    ask_price = (self.market.ceil_price(sp['bid'] + tick)
                                 if hasattr(self.market, 'ceil_price')
                                 else round(sp['bid'] + tick, 2))
                if bid_price >= ask_price:
                    continue
                if place_bid and i in missing['bid']:
                    self._submit_dealer_order(
                        'bid', bid_qty, bid_price,
                        reference_mid=mid, quote_level=i,
                    )
                if place_ask and i in missing['ask']:
                    self._submit_dealer_order(
                        'ask', ask_qty, ask_price,
                        reference_mid=mid, quote_level=i,
                    )
            # Tag orders with initial qty for inventory tracking next tick
            for order in self.orders:
                order._mm_init_qty = order.qty
            return

        # ---------- classic single-venue mode (inventory balance) ---------
        # Clear previous orders
        self.cancel_all_quotes()

        spread = self.market.spread()

        # Calculate bid and ask volume
        bid_volume = max(0., self.ul - 1 - self.assets)
        ask_volume = max(0., self.assets - self.ll - 1)

        # If in panic state we only either sell or buy commodities
        if not bid_volume or not ask_volume:
            self.panic = True
            self._buy_market((self.ul + self.ll) / 2 - self.assets) if ask_volume is None else None
            self._sell_market(self.assets - (self.ul + self.ll) / 2) if bid_volume is None else None
        else:
            self.panic = False
            base_offset = -((spread['ask'] - spread['bid']) * (self.assets / self.softlimit))  # Price offset
            self._buy_limit(bid_volume, spread['bid'] - base_offset - .1)  # BID
            self._sell_limit(ask_volume, spread['ask'] + base_offset + .1)  # ASK


# ---------------------------------------------------------------------------
# AMM-only agents (genuinely new concepts with no CLOB-only analogue)
# ---------------------------------------------------------------------------

class AMMProvider:
    """
    Liquidity Provider for AMM pools.

    Rule (§5.3):
        L_{t+1} = L_t + φ₁ Π^fee_t − φ₂ σ_t − φ₃ c_t

    Produces non-monotonic response to volatility:
    at moderate σ, fee income dominates → L grows;
    at high σ, risk terms dominate → L shrinks.
    """

    def __init__(self, pool: CPMMPool | HFMMPool,
                 env: MarketEnvironment,
                 phi1: float = 0.5,
                 phi2: float = 2.0,
                 phi3: float = 1.0,
                 # The largest share of its committed capital a provider can
                 # move in one period. At one second to the tick the previous
                 # value of five per cent meant a provider able to reallocate
                 # its entire position in twenty seconds, which is a rate no
                 # one supplies liquidity at. The default now lets a provider
                 # turn its position over in one trading day, which is the
                 # horizon on which such capital is actually reallocated. It
                 # does not bind at the calibrated scale, where the untruncated
                 # response is of the order of ten to the minus eight per
                 # period, and that is worth knowing instead of hiding.
                 max_adj: float = 1.0 / 86400.0,
                 core_liquidity_ratio: float = 0.65,
                 wallet_cash: Optional[float] = None,
                 wallet_base: Optional[float] = None,
                 wallet_cash_buffer_ratio: float = 0.20,
                 wallet_base_buffer_ratio: float = 0.20):
        self.type = 'AMMProvider'
        self.pool = pool
        self.env = env
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3
        self.max_adj = max_adj
        self.core_liquidity_ratio = max(0.0, min(1.0, core_liquidity_ratio))
        self._initial_liquidity = max(self.pool.liquidity_measure(), 1e-9)
        self.deployed_cash = max(0.0, float(self.pool.y))
        self.deployed_base = max(0.0, float(self.pool.x))
        self.wallet_cash = (
            max(0.0, float(wallet_cash))
            if wallet_cash is not None
            else self.deployed_cash * max(0.0, float(wallet_cash_buffer_ratio))
        )
        self.wallet_base = (
            max(0.0, float(wallet_base))
            if wallet_base is not None
            else self.deployed_base * max(0.0, float(wallet_base_buffer_ratio))
        )
        # Fee income actually received, kept in the currency it was charged
        # in so that the running total can be checked against the pool.
        self.fees_received_base = 0.0
        self.fees_received_quote = 0.0
        ref_price = self._reference_price()
        self._prev_pool_value = self._pool_mark_to_market(ref_price)

    def _reference_price(self) -> float:
        ref_price = getattr(self.env, 'fair_price', None)
        if ref_price is None or not _math.isfinite(ref_price) or ref_price <= 0:
            ref_price = self.pool.mid_price()
        return max(float(ref_price), 1e-9)

    def _pool_mark_to_market(self, reference_price: float) -> float:
        return max(1e-9, self.pool.x * reference_price + self.pool.y)

    def _max_affordable_add_fraction(self) -> float:
        base_capacity = self.wallet_base / max(self.pool.x, 1e-9)
        cash_capacity = self.wallet_cash / max(self.pool.y, 1e-9)
        return max(0.0, min(base_capacity, cash_capacity))

    def _deploy_liquidity(self, fraction: float) -> float:
        fraction = max(0.0, min(float(fraction), self._max_affordable_add_fraction()))
        if fraction <= 0:
            return 0.0
        base_added = self.pool.x * fraction
        cash_added = self.pool.y * fraction
        self.wallet_base -= base_added
        self.wallet_cash -= cash_added
        self.deployed_base += base_added
        self.deployed_cash += cash_added
        self.pool.add_liquidity(fraction)
        return fraction

    def _withdraw_liquidity(self, fraction: float) -> float:
        fraction = max(0.0, min(float(fraction), 0.5))
        if fraction <= 0:
            return 0.0
        base_removed = self.pool.x * fraction
        cash_removed = self.pool.y * fraction
        self.pool.remove_liquidity(fraction)
        self.deployed_base = max(0.0, self.deployed_base - base_removed)
        self.deployed_cash = max(0.0, self.deployed_cash - cash_removed)
        self.wallet_base += base_removed
        self.wallet_cash += cash_removed
        return fraction

    def update_liquidity(self):
        """Adjust pool liquidity per the paper's LP rule.

        Paper text (§3 / Supplement):
            L_{t+1} = L_t + phi1 * Pi_fee - phi2 * sigma - phi3 * c

        The fractional change is delta_L / L, capped at +/- max_adj per
        period. A core-liquidity floor is preserved as a numerical safety
        device (so a long stress streak does not trivially zero out a
        pool); the floor is parameterised by core_liquidity_ratio and is
        the only addition relative to the headline equation. Earlier
        loss/adverse-selection/stabilising terms have been removed so the
        implementation matches the paper.
        """
        L = self.pool.liquidity_measure()
        if L <= 0:
            return

        ref_price = self._reference_price()
        # Two smoothed statistics used to be accumulated here, a mark to
        # market result and an adverse selection score. The rule they once fed
        # was removed so that the implementation would match the equation in
        # the paper, and the state was left behind, computed every period and
        # read by nothing. Carrying it made the provider look as though it
        # weighed things it does not weigh.

        # Fee income has to be claimed from the pool and credited to the
        # provider before it can be acted on. The pool withholds a sell fee in
        # base and a buy fee in quote, and reading the pool's quote valued
        # counter instead left the money sitting in the venue with no owner
        # while the rule responded to it, and while the profit and loss
        # statement counted it as earned. On a sell of four hundred units into
        # a pool of a thousand by a thousand the counter said 13.77, the base
        # actually withheld was worth 20.00 and the provider received nothing.
        claimed_base, claimed_quote = 0.0, 0.0
        try:
            claimed_base, claimed_quote = self.pool.claim_fees()
        except AttributeError:
            pass
        self.wallet_base += claimed_base
        self.wallet_cash += claimed_quote
        self.fees_received_base += claimed_base
        self.fees_received_quote += claimed_quote
        fee_income = claimed_base * ref_price + claimed_quote

        # Every term below is a rate per period, so the fractional change is
        # dimensionless and does not depend on the currency the pair happens to
        # be quoted in. The earlier form added a fee in quote units to a
        # volatility and a funding cost, then divided the sum by a liquidity
        # measure carrying units of its own, the square root of the reserve
        # product for the constant product pool and the invariant for the
        # hybrid one. Restating the same market with a price of a hundred
        # and not one changed the constant product response to a fee by a
        # factor of twenty six and reversed its sign, so the mechanism was
        # reading the numeraire and not the economics.
        #
        #     dL / L = phi1 * fee income / pool value
        #              - phi2 * (volatility of the price per period) squared
        #              - phi3 * funding rate per period
        #
        # The volatility term takes the volatility of the price and not the
        # dimensionless stress index, so the three terms share one time scale
        # as well as one set of units, and it enters squared. What a provider
        # loses to rebalancing over a period grows with the square of the price
        # move, at a rate of one eighth of the variance for a constant product
        # pool in \\citet{milionis2022} and faster on an amplified curve, so a
        # term linear in the volatility is not a loss rate at all. It only
        # looked defensible while the index had no time scale and the
        # coefficient could absorb anything.
        pool_value = max(self._pool_mark_to_market(ref_price), 1e-9)
        fee_return = fee_income / pool_value
        vol = float(getattr(self.env, 'price_sigma', self.env.sigma))
        c = float(getattr(self.env, 'funding_rate', self.env.funding_cost))
        frac = self.phi1 * fee_return - self.phi2 * vol * vol - self.phi3 * c

        # Core-liquidity floor: prevent a long adverse streak from draining
        # the pool below a configurable share of its initial liquidity.
        core_floor = self._initial_liquidity * self.core_liquidity_ratio
        if L < core_floor:
            refill = min(self.max_adj, max(0.0, (core_floor - L) / max(L, 1e-9)))
            frac = max(frac, refill)
        elif frac < 0 and L * (1.0 + frac) < core_floor:
            frac = core_floor / max(L, 1e-9) - 1.0

        frac = max(-self.max_adj, min(self.max_adj, frac))
        if frac > 0:
            self._deploy_liquidity(frac)
        elif frac < 0:
            self._withdraw_liquidity(abs(frac))


class AMMArbitrageur:
    """
    Arbitrageur that trades on AMM pools to align their mid-price with
    a reference price S_t.

    Reference price cascade
    -----------------------
    1. **env.fair_price** — exogenous GBM price (always authoritative
       when available; used in AMM-only mode).
    2. **CLOB mid** — if fair_price is absent and book is healthy.
    3. **Blend** — weighted average of CLOB and AMM consensus when
       spread is moderately wide.
    4. **AMM consensus** — pure average of pool mid-prices as last
       resort.

    Routing modes
    -------------
    * **all** (default) — arbitrage every mispriced pool independently.
    * **best** — rank pools by profitability (deviation − fee), trade
      only the most profitable pool each iteration.

    Robustness features
    -------------------
    * **Max correction cap**: per-iteration price correction is capped
      at ``max_correction_pct`` to prevent reserve drainage from a
      single erratic tick.
    * **Reserve-health guard**: arbitrage is skipped for a pool whose
      reserve value-ratio is dangerously out of balance.
    * **HFMM rate update**: after arbitrage the HFMM ``rate`` is
      re-centered when the mid-price drifts > 1 % from the peg.
    """

    def __init__(self, clob,
                 amm_pools: Dict[str, CPMMPool | HFMMPool],
                 env: MarketEnvironment = None,
                 max_spread_bps: float = 500.0,
                 max_correction_pct: float = 10.0,
                 routing: str = 'all',
                 trade_fraction_cap: float = 0.20,
                 cash: Optional[float] = None,
                 assets: Optional[float] = None,
                 cash_buffer_ratio: float = 0.15,
                 asset_buffer_ratio: float = 0.15):
        self.clob = clob
        self.amm_pools = amm_pools
        self.env = env
        self.type = 'AMMArbitrageur'
        self.max_spread_bps = max_spread_bps
        self.max_correction_pct = max_correction_pct / 100.0
        self.routing = routing
        self.trade_fraction_cap = max(0.0, min(1.0, trade_fraction_cap))
        aggregate_base = sum(max(0.0, float(pool.x)) for pool in amm_pools.values())
        aggregate_cash = sum(max(0.0, float(pool.y)) for pool in amm_pools.values())
        self.cash = (
            max(0.0, float(cash))
            if cash is not None
            else aggregate_cash * max(0.0, float(cash_buffer_ratio))
        )
        self.assets = (
            max(0.0, float(assets))
            if assets is not None
            else aggregate_base * max(0.0, float(asset_buffer_ratio))
        )
        self.cumulative_traded_base = 0.0
        self.cumulative_pnl_quote = 0.0
        # Successful AMM legs from the most recent simulator period.  They
        # are telemetry only: customer routing and AMM arbitrage are distinct
        # economic sources of volume and must not share a denominator by
        # accident.  The list is reset on every ``arbitrage`` call.
        self.period_trades: List[dict] = []

    # ---- helpers ---------------------------------------------------------

    def _amm_consensus(self) -> Optional[float]:
        """Average AMM mid-price only when pools are internally consistent."""
        mids = []
        for p in self.amm_pools.values():
            try:
                mid = p.mid_price()
                if _math.isfinite(mid) and mid > 0:
                    mids.append(mid)
            except Exception:
                pass
        if not mids:
            return None
        if len(mids) == 1:
            return mids[0]

        mids = sorted(mids)
        mid_idx = len(mids) // 2
        if len(mids) % 2 == 0:
            center = 0.5 * (mids[mid_idx - 1] + mids[mid_idx])
        else:
            center = mids[mid_idx]
        dispersion_bps = 10_000.0 * (mids[-1] - mids[0]) / max(center, 1e-9)
        if dispersion_bps > 250.0:
            return None
        return sum(mids) / len(mids)

    def _blend_with_fair(self, observed: Optional[float]) -> Optional[float]:
        """Treat env.fair_price as a latent anchor and not a command."""
        fair = None
        if self.env is not None:
            fair = self.env.fair_price

        if fair is None:
            return observed
        if observed is None:
            return fair

        sigma_base = getattr(self.env, 'sigma_low', None) if self.env is not None else None
        stress_scale = 1.0
        if sigma_base is not None and sigma_base > 0 and self.env is not None:
            stress_scale = max(1.0, self.env.sigma / sigma_base)

        liquidity_factor = getattr(self.env, 'systemic_liquidity', 1.0)
        gap_bps = abs(fair - observed) / max(observed, 1e-9) * 10_000.0
        fair_weight = 0.35 * liquidity_factor / stress_scale
        fair_weight /= 1.0 + gap_bps / 75.0
        fair_weight = max(0.05, min(0.35, fair_weight))
        return observed + fair_weight * (fair - observed)

    def _reference_price(self) -> Optional[float]:
        """
        Determine the best available reference price.

        Priority for arbitrage in major FX pairs follows Mancini, Ranaldo &
        Wrampelmeyer (JF 2013) and Chaboud et al. (RFS 2014): when the
        primary CLOB venue has a finite quoted spread, CLOB *is* the
        reference. AMM consensus only enters as the CLOB book degrades.
        env.fair_price acts as a weak latent anchor — never as a command —
        so the arbitrageur cannot legitimise a drifting AMM by blending it
        into the target.

        Cascade:
          1. CLOB mid alone, when book is healthy (qspr <= max_spread_bps).
          2. As qspr widens, weight shifts from CLOB to AMM consensus.
          3. Pure AMM consensus only when CLOB is degenerate.
          4. Pure fair_price as the last resort.
        """
        fair = self.env.fair_price if self.env is not None else None

        S_clob = None
        try:
            S_clob = self.clob.mid_price()
        except Exception:
            pass

        S_amm = self._amm_consensus()

        if S_clob is None and S_amm is None:
            return fair
        if S_clob is None:
            return self._blend_with_fair(S_amm)
        if S_amm is None:
            return self._blend_with_fair(S_clob)

        # Sanity: if CLOB is implausible relative to AMM (>50 % off),
        # fall back to AMM consensus (likely CLOB book corruption).
        clob_dev = abs(S_clob - S_amm) / S_amm if S_amm > 0 else 0
        if clob_dev > 0.5:
            return self._blend_with_fair(S_amm)

        try:
            spread_bps = self.clob.quoted_spread_bps()
        except Exception:
            spread_bps = float('inf')

        if not _math.isfinite(spread_bps) or spread_bps > 2000:
            return self._blend_with_fair(S_amm)

        # Healthy CLOB: anchor on CLOB. AMM gets zero weight in the target,
        # so the arbitrageur pulls AMM mid back to CLOB and not to a
        # blended midpoint that legitimises pool drift.
        if spread_bps <= self.max_spread_bps:
            return self._blend_with_fair(S_clob)

        # Stressed CLOB: linearly transition from CLOB-anchored to
        # AMM-anchored as qspr widens between max_spread_bps and 2000 bps.
        w = max(0.0, 1.0 - (spread_bps - self.max_spread_bps)
                / (2000 - self.max_spread_bps))
        market_ref = w * S_clob + (1.0 - w) * S_amm
        return self._blend_with_fair(market_ref)

    @staticmethod
    def _reserve_healthy(pool, threshold: float = 0.1) -> bool:
        """Check that pool reserves are not dangerously one-sided."""
        try:
            mp = pool.mid_price()
            if mp <= 0 or pool.x <= 0:
                return False
            value_ratio = pool.y / (pool.x * mp)
            return threshold < value_ratio < (1.0 / threshold)
        except Exception:
            return False

    def _max_affordable_buy_qty(self, pool, max_trade_qty: float) -> float:
        if max_trade_qty <= 0 or self.cash <= 0:
            return 0.0

        lo = 0.0
        hi = max_trade_qty
        for _ in range(60):
            mid = (lo + hi) / 2.0
            quote = pool.quote_buy(mid)
            if quote['cost_bps'] == float('inf') or quote['delta_y'] > self.cash:
                hi = mid
            else:
                lo = mid
        return lo

    def _settle_trade(self, pool, traded_qty: float, before_y: float, reference_price: float):
        # The arbitrageur is prefunded in both native assets. The affordability
        # checks in ``_arb_one_pool`` cap each leg by the remaining wallet, and
        # settlement persists that wallet across pools and periods. Tiny
        # negative round-off is clipped so a fully exhausted budget cannot turn
        # into an unintended credit line on the next call.
        if traded_qty > 0:
            cash_spent = max(0.0, (pool.y - before_y) / max(1.0 - pool.fee, 1e-9))
            self.assets += traded_qty
            self.cash = max(0.0, self.cash - cash_spent)
            self.cumulative_pnl_quote += traded_qty * reference_price - cash_spent
            self.cumulative_traded_base += traded_qty
        elif traded_qty < 0:
            sell_qty = abs(traded_qty)
            cash_received = max(0.0, before_y - pool.y)
            self.assets = max(0.0, self.assets - sell_qty)
            self.cash += cash_received
            self.cumulative_pnl_quote += cash_received - sell_qty * reference_price
            self.cumulative_traded_base += sell_qty

    # ---- main entry point ------------------------------------------------

    def arbitrage(self):
        """For each AMM pool, align its price to S_t if profitable."""
        self.period_trades = []
        S_t = self._reference_price()
        if S_t is None:
            return

        if self.routing == 'best':
            self._arbitrage_best(S_t)
        else:
            self._arbitrage_all(S_t)

    def _arbitrage_all(self, S_t):
        """Original mode: trade every mispriced pool."""
        for name, pool in self.amm_pools.items():
            self._arb_one_pool(pool, S_t)

    def _arbitrage_best(self, S_t):
        """Route to the most profitable pool only."""
        candidates = []
        for name, pool in self.amm_pools.items():
            if not self._reserve_healthy(pool):
                if hasattr(pool, 'update_rate'):
                    pool.update_rate()
                continue
            try:
                current = pool.mid_price()
                if current <= 0:
                    continue
                dev_bps = abs(S_t - current) / current * 10_000
                fee_bps = pool.fee * 10_000
                profit_bps = dev_bps - fee_bps * 2
                if profit_bps > 0:
                    candidates.append((profit_bps, name, pool))
            except Exception:
                pass

        if not candidates:
            return
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, best_pool = candidates[0]
        self._arb_one_pool(best_pool, S_t)

    def _arb_one_pool(self, pool, S_t):
        """Arbitrage a single pool toward S_t."""
        try:
            if not self._reserve_healthy(pool):
                # Only re-peg the HFMM curve to the external reference when
                # the pool is in poor health, never to its own drifted mid.
                if hasattr(pool, 'update_rate'):
                    pool.update_rate(target_rate=S_t)
                return

            current = pool.mid_price()
            if current <= 0:
                return
            dev = (S_t - current) / current
            if abs(dev) > self.max_correction_pct:
                S_capped = current * (1.0 + _math.copysign(
                    self.max_correction_pct, dev))
            else:
                S_capped = S_t

            max_trade_qty = max(0.0, pool.x * self.trade_fraction_cap)
            if current < S_capped:
                max_trade_qty = self._max_affordable_buy_qty(pool, max_trade_qty)
            elif current > S_capped:
                # Selling base is symmetric to buying it: the configured base
                # buffer is a real budget, not a decorative parameter. The old
                # branch capped buys by cash but allowed unlimited short sales,
                # silently introducing an external lender on one side only.
                max_trade_qty = min(max_trade_qty, max(0.0, self.assets))

            if max_trade_qty <= 1e-9:
                return

            before_y = pool.y
            traded_qty = pool.arbitrage_to_target(S_capped, max_trade_qty=max_trade_qty)
            if abs(traded_qty) <= 1e-9:
                return
            self._settle_trade(pool, traded_qty, before_y, S_capped)
            pool_name = next(
                (name for name, candidate in self.amm_pools.items()
                 if candidate is pool),
                getattr(pool, 'pool_name', 'amm'),
            )
            self.period_trades.append({
                'execution_source': 'amm_arbitrage',
                'venue': pool_name,
                'quantity': abs(float(traded_qty)),
                'signed_quantity': float(traded_qty),
                'side': 'buy' if traded_qty > 0.0 else 'sell',
                'reference_price': float(S_capped),
            })

            # Re-peg the HFMM curve to the *external* reference and not
            # the pool's drifted mid. This keeps the StableSwap amplification
            # benefit anchored on the true equilibrium price (S_t) instead of
            # silently legitimising pool drift.
            if hasattr(pool, 'update_rate'):
                pool.update_rate(target_rate=S_t)
        except Exception:
            pass
