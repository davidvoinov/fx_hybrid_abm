from __future__ import annotations

from AgentBasedModel.agents import ExchangeAgent, Universalist, Chartist, Fundamentalist, MarketMaker, AMMProvider, AMMArbitrageur, Random, FastRecyclerLP, LatentLP
from AgentBasedModel.utils.math import mean, std, difference, rolling
import random
from typing import Optional, List, Dict, Any
from tqdm import tqdm

import hashlib as _hashlib
import json as _json
import math as _math
import os as _os

# The calibrated values live in one file and every entry point has to read them
# from there. The factory used to carry its own literals in the signature, so a
# script calling ``Simulator.default_fx`` directly built a different model from
# the one the command line builds, with no warning and no way to notice.
_CALIBRATION_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    'calibration', 'primary_model.json')


def _venue_seed(name: str) -> int:
    """A per venue stream that is stable across processes.

    The venue name is hashed with a fixed digest rather than with the built in
    ``hash``, which is salted per process. Mixing in a draw from the global
    generator keeps distinct venues on distinct streams while tying the whole
    thing to the seed the caller has already set.
    """
    digest = _hashlib.sha256(str(name).encode('utf-8')).digest()
    stable = int.from_bytes(digest[:4], 'big')
    return (stable ^ random.getrandbits(32)) & 0xFFFFFFFF


def _isolated_venue_rng(name: str) -> random.Random:
    """Create a named RNG without advancing the shared agent stream.

    A dedicated scheduler should isolate dealer priority from other draws, but
    allocating that stream must not itself shift every customer, lifecycle and
    routing variate in the calibrated model. The caller has already seeded the
    shared generator; snapshotting around ``_venue_seed`` derives a stable
    named stream while preserving the pre-existing global path.
    """
    state = random.getstate()
    try:
        return random.Random(_venue_seed(name))
    finally:
        random.setstate(state)


_CAL = None


def calibrated_default(name, fallback=None):
    """Read a calibrated default, caching the file after the first read."""
    global _CAL
    if _CAL is None:
        try:
            with open(_CALIBRATION_PATH, encoding='utf-8') as fh:
                _CAL = _json.load(fh).get('cli_defaults', {}) or {}
        except Exception:
            _CAL = {}
    return _CAL.get(name, fallback)


class Simulator:
    """
    Simulator is responsible for launching agents' actions and executing scenarios.

    Operates in two modes determined automatically:

    **Classic mode** (no AMM / CLOB wrapper supplied):
        Legacy stock-market ABM — same behaviour as before.

    **Multi-venue mode** (``clob``, ``amm_pools`` supplied):
        CLOB + AMM FX simulation with venue routing, LP agents,
        arbitrageurs, MetricsLogger, and MarketEnvironment.
    """

    # Weight of a fully capitalised facility in the recovery support sum,
    # named here rather than buried in the expression because it is the one
    # number in that sum which decides whether the recovery leg of the
    # comparison is a test or a restatement of its own inputs.
    VENUE_RECOVERY_WEIGHT = 0.15

    def __init__(self,
                 exchange: ExchangeAgent = None,
                 traders: list = None,
                 events: list = None,
                 # ---- multi-venue extras (all optional) ----
                 clob: Any = None,
                 amm_pools: Dict[str, Any] = None,
                 env: Any = None,
                 fx_traders: List = None,
                 book_agents: List = None,
                 market_maker: Any = None,
                 lp_providers: List[AMMProvider] = None,
                 arbitrageur: AMMArbitrageur = None,
                 logger: Any = None,
                 dealer_scheduler_rng: Any = None,
                 fx_scheduler_rng: Any = None,
                 shock_iter: Optional[int] = None,
                 shock_pct: float = -20.0,
                 shock_mode: str = 'research',
                 realism_shock_config: Optional[Dict[str, Any]] = None,
                 ):
        self.exchange = exchange
        self.events = [event.link(self) for event in events] if events else None
        self.traders = traders  # used in classic mode (+ SimulatorInfo)

        # Price shock (multi-venue mode)
        self.shock_iter = shock_iter
        self.shock_pct = shock_pct
        self.shock_mode = shock_mode
        self.realism_shock_config = realism_shock_config or {}
        # Whether providers may requote between taker orders inside a period.
        self.substep_replenish = True

        # Multi-venue
        self.clob = clob
        self.amm_pools = amm_pools or {}
        # Depth each pool opened with, captured on first use rather than here
        # so that it does not depend on the order in which the factory builds
        # the venue and matches the pool the run actually starts from.
        self._venue_opening_depth: Dict[str, float] = {}
        self.env = env
        self.fx_traders = fx_traders or []
        self.book_agents = book_agents or []
        self.mm = market_maker
        dealer_candidates = ([self.mm] if self.mm is not None else []) + [
            trader for trader in self.book_agents
            if isinstance(trader, MarketMaker)
        ]
        dealer_seen = set()
        dealer_roster = []
        for dealer in dealer_candidates:
            marker = id(dealer)
            if marker in dealer_seen:
                continue
            dealer_seen.add(marker)
            dealer_roster.append(dealer)
        self._market_maker_roster = tuple(dealer_roster)
        # Dealer queue priority has its own seeded stream. Unrelated noise,
        # routing or AMM draws must not silently change which bank dealer is
        # first among equal-price quotes on a given tick.
        self._dealer_scheduler_rng = (
            dealer_scheduler_rng
            if dealer_scheduler_rng is not None
            else _isolated_venue_rng('dealer_scheduler')
        )
        # Customer order within a tick is part of the paired demand schedule,
        # not an AMM-side innovation.  Keeping its shuffle private prevents a
        # treatment-specific fill/replenishment draw from changing the next
        # period's customer sequence.
        self._fx_scheduler_rng = (
            fx_scheduler_rng
            if fx_scheduler_rng is not None
            else _isolated_venue_rng('fx_scheduler')
        )
        self.lp_providers = lp_providers or []
        self.arbitrageur = arbitrageur
        self.logger = logger

        # SimulatorInfo — only created when classic traders list is given
        if self.traders:
            self.info = SimulatorInfo(self.exchange, self.traders)
        else:
            self.info = None

    def _risk_managed_traders(self):
        seen = set()
        traders = []
        for trader in list(self.book_agents) + list(self.fx_traders) + ([self.mm] if self.mm is not None else []):
            if trader is None:
                continue
            trader_id = getattr(trader, 'id', id(trader))
            if trader_id in seen:
                continue
            seen.add(trader_id)
            traders.append(trader)
        return traders

    def _apply_balance_sheet_controls(self):
        if self.exchange is None:
            return
        reference_price = None
        try:
            if self.clob is not None:
                reference_price = self.clob.mid_price()
        except Exception:
            reference_price = None
        if reference_price is None:
            try:
                reference_price = self.exchange.price()
            except Exception:
                reference_price = None
        funding_rate = 0.0
        if self.env is not None:
            # ``funding_cost`` is a dimensionless stress state used by quote
            # and withdrawal rules.  Financing a balance sheet needs the rate
            # on the model's one-second clock instead.  Passing the stress
            # state here charged roughly 20 bps per second in calm conditions
            # rather than about 1.3e-9 per second.
            funding_rate = max(0.0, float(getattr(self.env, 'funding_rate', 0.0)))

        for trader in self._risk_managed_traders():
            if getattr(trader, 'defaulted', False):
                continue
            try:
                trader.apply_financing_charge(reference_price, funding_rate)
                trader.enforce_balance_sheet_discipline(reference_price)
            except Exception:
                pass

    @property
    def multi_venue(self) -> bool:
        return self.clob is not None and (bool(self.amm_pools) or bool(self.fx_traders))

    def _market_makers(self) -> List[MarketMaker]:
        if not self.traders:
            return []
        return [tr for tr in self.traders if isinstance(tr, MarketMaker)]

    def _scheduled_market_makers(self) -> List[MarketMaker]:
        """Return every CLOB dealer once, in a seeded random call order.

        ``self.mm`` is only a compatibility handle for the first dealer made
        by :meth:`default_fx`; it is not a more senior economic agent. Calling
        that object before the other dealers gave it deterministic price-time
        priority whenever quotes tied. In the calibrated heterogeneous pack
        the same object also has the lowest withdrawal threshold and greatest
        depth, so the scheduling artefact concentrated inventory and calm
        withdrawals on dealer zero.

        Every public entry point seeds the process-wide generator before this
        simulator's dedicated dealer-scheduling stream is created. A fresh
        shuffle on each tick therefore keeps runs reproducible and invariant
        to unrelated agent RNG consumption while making queue priority
        exchangeable across dealer identities. Object identity only guards
        against a manually constructed simulator putting ``self.mm`` in
        ``book_agents`` as well.
        """
        # ``book_agents`` is shuffled in place below. Starting from that
        # mutable order would make the next dealer permutation depend on the
        # AMM/routing arm even with a private RNG, so always copy the immutable
        # construction-time roster.
        dealers = list(self._market_maker_roster)
        self._dealer_scheduler_rng.shuffle(dealers)
        return dealers

    def _market_maker_state_summary(self) -> Dict[str, Any]:
        market_makers = self._market_makers()
        states = ('active', 'defensive', 'withdrawn', 'reentering')
        counts = {state: 0 for state in states}
        if not market_makers:
            return {
                'n_market_makers': 0,
                'counts': counts,
                'shares': {state: 0.0 for state in states},
                'avg_withdrawal_score': float('nan'),
                'avg_loss_bps_ewma': float('nan'),
                'avg_inventory_ratio': float('nan'),
                'avg_withdrawal_confirmation_count': float('nan'),
                'withdrawal_confirmation_pending_share': float('nan'),
            }

        # Four channels take a dealer out of the book and they are not the
        # same object. A pause written by the scenario, a transition the
        # dealer's own score produced, a default or margin liquidation, and a
        # dealer that stayed but quotes defensively. Reporting only the union
        # of the first two let a scripted pause stand as evidence for the
        # endogenous mechanism.
        channels = {'forced_pause': 0, 'endogenous': 0, 'defaulted': 0,
                    'defensive': 0}
        for mm in market_makers:
            state = getattr(mm, 'mm_state', 'active')
            counts[state] = counts.get(state, 0) + 1
            if getattr(mm, 'defaulted', False):
                channels['defaulted'] += 1
            elif getattr(mm, 'mm_forced_pause', False):
                channels['forced_pause'] += 1
            elif state == 'withdrawn':
                channels['endogenous'] += 1
            elif state == 'defensive':
                channels['defensive'] += 1

        n_market_makers = len(market_makers)
        return {
            'n_market_makers': n_market_makers,
            'counts': counts,
            'channels': channels,
            'channel_shares': {k: v / n_market_makers for k, v in channels.items()},
            'shares': {state: counts.get(state, 0) / n_market_makers for state in states},
            'avg_withdrawal_score': sum(getattr(mm, 'mm_withdrawal_score', 0.0) for mm in market_makers) / n_market_makers,
            'avg_loss_bps_ewma': sum(getattr(mm, '_loss_bps_ewma', 0.0) for mm in market_makers) / n_market_makers,
            'avg_withdrawal_confirmation_count': (
                sum(getattr(mm, '_withdrawal_confirmation_count', 0)
                    for mm in market_makers) / n_market_makers
            ),
            'withdrawal_confirmation_pending_share': (
                sum(getattr(mm, '_withdrawal_confirmation_count', 0) > 0
                    and getattr(mm, 'mm_state', 'active') != 'withdrawn'
                    for mm in market_makers) / n_market_makers
            ),
            'avg_inventory_ratio': (
                sum(max(abs(getattr(mm, 'inventory', 0.0)), abs(float(getattr(mm, 'assets', 0.0)))) /
                    max(float(getattr(mm, 'softlimit', 1.0)), 1.0) for mm in market_makers)
                / n_market_makers
            ),
        }

    def _pool_depth(self, pool) -> float:
        try:
            depth = float(pool.effective_depth())
        except Exception:
            return 0.0
        return depth if _math.isfinite(depth) and depth > 0.0 else 0.0

    def _venue_support_share(self) -> Optional[float]:
        """How much of its opening depth the facility is still standing up.

        ``None`` when there is no facility in the market, which is what makes
        the paired comparison run at all.  Otherwise the mean over pools of a
        number in [0, 1]: one while the facility is at or above the depth it
        opened with, less as its reserves are drawn down, zero once it closes.

        Depth rather than outstanding provider capital, because the two move
        on different clocks.  A provider's participation decision runs on the
        smoothing and patience horizons, which are far longer than the window
        the recovery of the market is measured over; measured on capital, the
        facility's contribution is therefore constant inside that window and
        the recovery result would be fixed by construction again.  Reserves
        respond within it, falling to about four fifths of opening depth at
        the trough of the calibrated crisis, so the term varies while the
        recovery it is supposed to explain is happening.

        The reference is the depth the pool opened with: fixed by the
        configuration, the same in every seed, and needing no smoothing
        parameter of its own.  Calm drift carries the pool a few per cent
        above it, which the cap at one absorbs, so drift cannot manufacture
        support the facility is not supplying.
        """
        if not self.amm_pools:
            return None

        shares = []
        for name, pool in self.amm_pools.items():
            if getattr(pool, 'closed', False):
                shares.append(0.0)
                continue
            opening = self._venue_opening_depth.get(name)
            if opening is None:
                opening = self._pool_depth(pool)
                self._venue_opening_depth[name] = opening
            if opening <= 0.0:
                shares.append(1.0 if self._pool_depth(pool) > 0.0 else 0.0)
                continue
            shares.append(max(0.0, min(1.0, self._pool_depth(pool) / opening)))
        return sum(shares) / len(shares)

    def _estimate_recovery_support(self) -> float:
        """Aggregate standing liquidity behind the market's return to normal.

        Every provider that can hold the near side of the book after a shock
        belongs here.  The facility was previously left out, so the one
        provider in the market that never withdraws contributed nothing, and
        the recovery leg of the comparison could not have come out any way
        other than the way it did.

        The facility enters through its state and not through its presence.
        A closed or drawn down pool supports less, and how much depth it
        still holds during a crisis is an outcome of the run rather than an
        input to it, so the sign of the recovery result is not fixed in
        advance by this term.  The weight sits between the two book
        providers: the facility quotes continuously rather than
        intermittently, which argues for the upper end of their band, while
        it holds no view on the fundamental, which argues against exceeding
        it.
        """
        fast_agents = [tr for tr in self.book_agents if isinstance(tr, FastRecyclerLP)]
        latent_agents = [tr for tr in self.book_agents if isinstance(tr, LatentLP)]

        fast_total = len(fast_agents)
        latent_total = len(latent_agents)
        fast_active = sum(1 for tr in fast_agents if getattr(tr, 'last_quoted', False))
        latent_active = sum(1 for tr in latent_agents if getattr(tr, 'last_quoted', False))
        market_makers = self._market_makers()
        mm_active = 0.0
        if market_makers:
            mm_active = sum(1 for mm in market_makers if getattr(mm, 'mm_state', 'active') != 'withdrawn') / len(market_makers)

        static_support = 0.85 + 0.02 * fast_total + 0.03 * latent_total
        dynamic_support = 0.0
        if fast_total > 0:
            dynamic_support += 0.14 * (fast_active / fast_total)
        if latent_total > 0:
            dynamic_support += 0.18 * (latent_active / latent_total)
        dynamic_support += 0.08 * mm_active
        venue_share = self._venue_support_share()
        if venue_share is not None:
            dynamic_support += self.VENUE_RECOVERY_WEIGHT * venue_share

        return max(0.75, min(1.65, static_support + dynamic_support))

    def _sync_recovery_support(self):
        if self.env is None or not hasattr(self.env, 'set_recovery_support'):
            return
        self.env.set_recovery_support(self._estimate_recovery_support())

    def _restore_clob_near_mid_liquidity(self):
        if self.env is None or self.exchange is None or self.clob is None:
            return

        fair_price = getattr(self.env, 'fair_price', None)
        if fair_price is None or fair_price <= 0:
            return

        # The corridor is measured around the same centre the restoration
        # writes to. Checking around the book mid and restoring around the
        # fair price meant that after a fundamental move the book could be two
        # sided around a stale mid while one side of the new fair price was
        # empty, and the check would pass on a book that needed the repair.
        try:
            near_mid_depth = self.clob.total_depth_around(fair_price, 25)
        except Exception:
            return

        # Each side is checked on its own. The aggregate was enough to keep
        # the restoration from firing when one side had been swept and the
        # other still held size, which is exactly the state that needs it.
        if (near_mid_depth.get('bid', 0.0) > 1e-9
                and near_mid_depth.get('ask', 0.0) > 1e-9):
            return

        self.exchange.rebalance_background_liquidity(
            fair_price,
            corridor_bps=25.0,
            target_ratio=max(0.5, float(getattr(self.env, 'systemic_liquidity', 1.0))),
            respect_trader_cap=False,
        )

    # ------------------------------------------------------------------
    # Classic helpers
    # ------------------------------------------------------------------

    def _payments(self):
        for trader in self.traders:
            trader.cash += trader.assets * self.exchange.dividend()
            trader.cash += trader.cash * self.exchange.risk_free

    @staticmethod
    def _rebalance_pool_to_multiplier(pool, multiplier: float):
        import math as _math
        from AgentBasedModel.venues.amm import _hfmm_get_D

        sqrt_m = _math.sqrt(max(multiplier, 1e-6))
        old_x = pool.x
        old_y = pool.y
        pool.x = old_x / sqrt_m
        pool.y = old_y * sqrt_m
        if hasattr(pool, 'k'):
            pool.k = pool.x * pool.y
        if hasattr(pool, '_sync_norm'):
            pool._sync_norm()
        if hasattr(pool, 'D') and hasattr(pool, '_xn'):
            pool.D = _hfmm_get_D(pool._xn, pool._yn, pool.A)
        if hasattr(pool, 'rate'):
            pool.rate = pool.y / pool.x

    def _apply_research_shock(self):
        from itertools import chain as _chain

        multiplier = 1.0 + self.shock_pct / 100.0

        if self.env is not None:
            self.env.apply_shock(self.shock_pct)

        if self.exchange is not None:
            try:
                mid = self.exchange.price()
            except Exception:
                mid = 100.0
            dp = mid * (self.shock_pct / 100.0)
            for order in _chain(*self.exchange.order_book.values()):
                order.price = round(order.price + dp, 2)
            self.exchange.rebuild_order_book()

            cancel_frac = self.env.cancel_wave_frac if self.env else 0.5
            self.exchange.cancel_wave(cancel_frac, near_touch=True)

            for tr in self.book_agents[:3]:
                try:
                    tr.call()
                except Exception:
                    pass

        for pool in self.amm_pools.values():
            self._rebalance_pool_to_multiplier(pool, multiplier)

    def _resolve_realism_shock_side(self, config: Dict[str, Any]) -> str:
        side = config.get('order_flow_side', 'auto')
        if side in ('buy', 'sell'):
            return side

        signed_move = config.get('fundamental_pct', 0.0)
        if signed_move == 0:
            signed_move = self.shock_pct
        return 'sell' if signed_move < 0 else 'buy'

    def _execute_order_flow_sweep(self, side: str, quantity: float):
        from AgentBasedModel.agents.agents import Trader
        from AgentBasedModel.utils.orders import Order

        if self.exchange is None or quantity <= 0:
            return
        if side == 'buy' and not self.exchange.order_book['ask']:
            return
        if side == 'sell' and not self.exchange.order_book['bid']:
            return

        pseudo_trader = Trader(self.exchange, cash=1e9, assets=int(1e6))
        if side == 'buy':
            price = self.exchange.order_book['ask'].last.price
            order = Order(price, round(quantity), 'bid', pseudo_trader)
        else:
            price = self.exchange.order_book['bid'].last.price
            order = Order(price, round(quantity), 'ask', pseudo_trader)
        self.exchange.market_order(order)

    def _apply_realism_shock(self):
        config = self.realism_shock_config or {}
        fundamental_pct = float(config.get('fundamental_pct', 0.0) or 0.0)
        order_flow_qty = float(config.get('order_flow_qty', 0.0) or 0.0)
        liquidity_frac = max(0.0, min(1.0, float(config.get('liquidity_frac', 0.0) or 0.0)))
        funding_vol_intensity = max(0.0, float(config.get('funding_vol_intensity', 0.0) or 0.0))
        order_flow_side = self._resolve_realism_shock_side(config)
        direction = 1.0 if order_flow_side == 'buy' else -1.0

        if self.env is not None:
            if abs(fundamental_pct) > 0:
                self.env.apply_fundamental_shock(fundamental_pct, anchor_weight=1.0)
            if funding_vol_intensity > 0:
                self.env.apply_funding_volatility_shock(funding_vol_intensity)

            flow_intensity = 0.0
            if order_flow_qty > 0 and self.exchange is not None:
                try:
                    near_mid_depth = self.clob.total_depth(25)['total']
                except Exception:
                    near_mid_depth = 0.0
                if near_mid_depth > 0:
                    flow_intensity = min(2.0, order_flow_qty / near_mid_depth)

            liquidity_intensity = max(liquidity_frac, flow_intensity)
            if liquidity_intensity > 0:
                cancel_frac = max(liquidity_frac, min(0.85, 0.10 + 0.25 * liquidity_intensity))
                # Optional slow-decay parameters from config
                decay_kwargs = {}
                for key in ('reprice_prob_recovery', 'anchor_strength_recovery',
                            'bg_target_ratio_recovery', 'toxic_flow_decay',
                            'liquidity_shock_decay'):
                    val = config.get(key)
                    if val is not None:
                        decay_kwargs[key] = float(val)
                self.env.apply_liquidity_shock(
                    cancel_frac=cancel_frac,
                    direction=direction,
                    intensity=liquidity_intensity,
                    force_mm_pause=bool(config.get('force_mm_pause', False)),
                    forced_pause_ticks=config.get('forced_pause_ticks'),
                    **decay_kwargs,
                )

        if order_flow_qty > 0:
            self._execute_order_flow_sweep(order_flow_side, order_flow_qty)

        if self.exchange is not None:
            cancel_frac = self.env.cancel_wave_frac if self.env is not None else liquidity_frac
            if cancel_frac > 0:
                self.exchange.cancel_wave(cancel_frac, near_touch=True)
            self.exchange.rebuild_order_book()

            for tr in self.book_agents[:2]:
                try:
                    tr.call()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Unified simulate
    # ------------------------------------------------------------------

    def simulate(self, n_iter: int, silent: bool = False) -> Simulator:
        if self.multi_venue:
            return self._simulate_multi(n_iter, silent)
        return self._simulate_classic(n_iter, silent)

    # ---- classic single-venue loop --------------------------------------

    def _simulate_classic(self, n_iter: int, silent: bool) -> Simulator:
        for it in tqdm(range(n_iter), desc='Simulation', disable=silent):
            # Scenario events
            if self.events:
                for event in self.events:
                    event.call(it)

            # Capture info
            if self.info is not None:
                self.info.capture()

            # Behaviour changes (Universalist / Chartist)
            for trader in self.traders:
                if type(trader) == Universalist:
                    trader.change_strategy(self.info)
                elif type(trader) == Chartist:
                    trader.change_sentiment(self.info)

            # Call traders
            random.shuffle(self.traders)
            for trader in self.traders:
                trader.call()

            # Payments & dividends
            self._payments()
            self.exchange.generate_dividend()

        return self

    # ---- multi-venue loop -----------------------------------------------

    def _simulate_multi(self, n_iter: int, silent: bool) -> Simulator:
        for t in tqdm(range(n_iter), desc='Simulation', disable=silent):

            # 0. Price shock — hits ALL venues simultaneously
            if self.shock_iter is not None and t == self.shock_iter:
                if self.shock_mode == 'realism':
                    self._apply_realism_shock()
                else:
                    self._apply_research_shock()

            # 1. Environment step → σ_t, c_t, S_t
            _sigma_prev = self.env.sigma if self.env is not None else None
            if self.env is not None:
                self._sync_recovery_support()
                self.env.step()
                # Re-centre book on new S_t — use reduced probability
                # during shock aftermath for natural price discovery lag
                fp = self.env.fair_price
                if fp is not None:
                    base_rp = self.env.reprice_prob_override if self.env.reprice_prob_override is not None else 0.6
                    liquidity_factor = max(0.2, min(1.0, self.env.systemic_liquidity))
                    venue_basis_bps = max(0.0, getattr(self.env, 'venue_basis_bps', 0.0))
                    arbitrage_capacity = max(0.0, min(1.0, getattr(self.env, 'arbitrage_capacity', 1.0)))
                    basis_damp = 1.0 / (1.0 + venue_basis_bps / 35.0)
                    rp = base_rp * (0.55 + 0.45 * liquidity_factor) * basis_damp
                    rp = max(0.05, min(base_rp, rp))

                    base_anchor_strength = getattr(self.exchange, '_anchor_strength', 0.25)
                    anchor_strength = base_anchor_strength * (0.45 + 0.55 * arbitrage_capacity) * basis_damp
                    anchor_strength = max(0.02, min(base_anchor_strength, anchor_strength))

                    background_target_ratio = max(0.35, self.env.systemic_liquidity)
                    if self.shock_mode == 'realism' and self.env.shock_ticks_remaining > 0:
                        if self.env.anchor_strength_override is not None:
                            anchor_strength = min(anchor_strength, self.env.anchor_strength_override * basis_damp)
                        if self.env.background_target_ratio_override is not None:
                            background_target_ratio = min(background_target_ratio, self.env.background_target_ratio_override)
                    self.exchange.recenter_book(
                        fp,
                        reprice_prob=rp,
                        anchor_strength=anchor_strength,
                        background_target_ratio=background_target_ratio,
                    )

            # 1a. Dynamic fees — scale AMM fees with lagged σ
            if self.env is not None:
                for pool in self.amm_pools.values():
                    if hasattr(pool, 'update_fee'):
                        pool.update_fee(self.env.sigma, self.env.sigma_low,
                                        sigma_prev=_sigma_prev)

            # 1b. Pre-trade arbitrage removed: the post-trade arbitrage
            # pass at step 7 already aligns AMM prices once per tick, and
            # major-pair FX arbitrage (Mancini, Ranaldo, Wrampelmeyer 2013)
            # converges within a single quote interval in normal conditions.
            # Running it twice per tick effectively doubled the per-tick
            # correction and made the documented arb_trade_fraction_cap
            # less meaningful as a robustness control.

            # 1c. SimulatorInfo capture + behaviour changes
            if self.info is not None:
                try:
                    self.info.capture()
                    for tr in self.book_agents:
                        if type(tr) == Universalist:
                            tr.change_strategy(self.info)
                        elif type(tr) == Chartist:
                            tr.change_sentiment(self.info)
                except Exception:
                    pass

            # 2. CLOB market-maker quotes (paused during shock aftermath).
            # ``self.mm`` is a compatibility handle, not a priority class:
            # every dealer participates in the same seeded per-tick shuffle.
            mm_paused = (self.env is not None and self.env.mm_pause_ticks > 0)
            scheduled_market_makers = self._scheduled_market_makers()

            # The flag belongs to the scenario window and is cleared the tick
            # the window closes, by whoever set it. It used to be cleared by
            # the dealer itself, on the first tick it managed to quote again,
            # which is several ticks later because its own state machine has
            # to walk back from withdrawn through re-entering. A five tick
            # scripted pause was therefore recorded as twelve to seventeen
            # ticks of forced pause, and every tick of that overhang was a
            # tick of the dealer's own decision being filed under the
            # scenario's.
            #
            # The scenario pause is an overlay and never owns ``mm_state``.
            # Keeping the state machine untouched matters when a dealer was
            # already withdrawn or re-entering before the pause: resetting it
            # to active would erase the minimum-withdrawal and hysteresis
            # timers.  Once the overlay closes the dealer is called normally
            # below and continues its own transition from the preserved state.
            if not mm_paused:
                for _mm in scheduled_market_makers:
                    if getattr(_mm, 'mm_forced_pause', False):
                        _mm.mm_forced_pause = False

            for _mm in scheduled_market_makers:
                if mm_paused:
                    # The scenario takes the quotes down. This is imposed by
                    # the shock and is not a decision of the dealer, so it is
                    # flagged separately and no longer hides inside the
                    # withdrawn share. Reporting the two together made a
                    # scripted pause look like evidence for an endogenous
                    # mechanism.
                    _mm.cancel_all_quotes()
                    _mm.mm_forced_pause = True
                else:
                    _mm.call()

            # 3. Non-dealer CLOB book agents (diverse limit-order providers).
            # Dealers were all scheduled together above; leaving the four
            # non-primary objects in this pass would call them twice.
            random.shuffle(self.book_agents)
            for tr in self.book_agents:
                if isinstance(tr, MarketMaker):
                    continue
                if getattr(tr, 'defaulted', False):
                    continue
                try:
                    tr.call()
                except Exception:
                    pass

            # 4. (Dividends skipped in FX mode — no FX-side use.)

            # 5. Reset AMM period fees
            for pool in self.amm_pools.values():
                pool.reset_period_fees()

            # 6. FX liquidity takers route orders.
            #
            # The providers are given a chance to react between orders rather
            # than after all of them. Running the whole taker batch against a
            # book frozen since the start of the period made two ordinary
            # institutional orders arrive as one block against liquidity that
            # could not step aside or step back in, which is what emptied one
            # side of the book and left the other untouched. At one second to
            # the tick a non bank market maker has on the order of a million
            # opportunities to requote between two trades, so denying it even
            # one is the less realistic choice. A bank dealer may replace
            # quantity that was actually filled here, but its untouched
            # orders retain their price, queue priority and multi-minute
            # lifecycle. Completed-order lifetime is therefore not confused
            # with reaction time to an execution.
            #
            # What the provider does here is a fill reaction, not a tick. It
            # reposts the side that was consumed and leaves the age of its
            # quote alone. Calling the tick method instead let every trade
            # spend a tick of quote life, so a two second order life became a
            # two trade order life, and it let providers that had not been hit
            # rebuild their whole quote set, which manufactured depth. A swap
            # against the automated venue takes no limit order out of this
            # book, so it triggers nothing.
            period_trades: List[dict] = []
            # Every routed request of the period, filled or not. Kept apart
            # from the trades so that nothing downstream of the executed flow
            # changes, and so that unmet demand has somewhere to be counted.
            period_routing: List[dict] = []
            self._fx_scheduler_rng.shuffle(self.fx_traders)
            replenishing_providers = [tr for tr in self.book_agents
                                      if isinstance(tr, (FastRecyclerLP, LatentLP, MarketMaker))]
            if self.mm is not None:
                replenishing_providers.append(self.mm)
            for tr in self.fx_traders:
                if getattr(tr, 'defaulted', False):
                    continue
                # Who reposts is decided by who was actually filled. Asking
                # instead whether a provider is currently showing both sides
                # let one that had been hit on an earlier trade, or had never
                # posted a side, repost on somebody else's execution, which is
                # size appearing without a cause. The resting quantity is read
                # before and after the taker acts and only the providers whose
                # own size fell are given the chance to replace it.
                watch = replenishing_providers if (self.substep_replenish
                                                    and replenishing_providers) else []

                def _resting(lp):
                    bid = ask = 0.0
                    for o in lp.orders:
                        if o.qty <= 0:
                            continue
                        if o.order_type == 'bid':
                            bid += o.qty
                        else:
                            ask += o.qty
                    return bid, ask

                before = {id(lp): _resting(lp) for lp in watch}
                result = tr.call()
                attempt = getattr(tr, 'last_routing_attempt', None)
                if attempt is not None:
                    period_routing.append(attempt)
                if result is not None:
                    period_trades.append(result)
                    if watch and result.get('venue') == 'clob':
                        # Each provider is told how much of its own size went,
                        # and on which side, so it replaces what it lost rather
                        # than deciding from the shape of its remaining book. A
                        # provider filled on its first level while still showing
                        # a second one used to conclude that nothing had
                        # happened to it.
                        taken = []
                        for lp in watch:
                            if getattr(lp, 'defaulted', False):
                                continue
                            b0, a0 = before.get(id(lp), (0.0, 0.0))
                            b1, a1 = _resting(lp)
                            db, da = b0 - b1, a0 - a1
                            if db > 1e-12 or da > 1e-12:
                                taken.append((lp, max(0.0, db), max(0.0, da)))
                        random.shuffle(taken)
                        for lp, db, da in taken:
                            lp.on_trade(db, da)

            if self.env is not None:
                self.env.observe_order_flow(period_trades)

            # 7. Arbitrageurs align AMM prices
            if self.arbitrageur is not None:
                self.arbitrageur.arbitrage()

            if self.env is not None and self.amm_pools:
                self.env.observe_venue_conditions(
                    self.clob,
                    self.amm_pools,
                    arbitrageur=self.arbitrageur,
                )

            # 7a. Expire stale orders (TTL-based lifecycle)
            fp = self.env.fair_price if self.env is not None else None
            self.exchange.expire_orders(fair_price=fp)
            self._sync_recovery_support()

            # 8. LP agents adjust AMM liquidity
            for lp in self.lp_providers:
                lp.update_liquidity()

            # 9. Record AMM state
            for pool in self.amm_pools.values():
                pool.record_state()

            # 9a. Funding carry, maintenance checks, and explicit default handling
            self._apply_balance_sheet_controls()

            # Restore minimal anonymous support if the book is technically
            # non-empty but no liquidity remains near fair value.
            self._restore_clob_near_mid_liquidity()

            # 10. Metrics snapshot
            if self.logger is not None:
                self.logger.snapshot(t, self.clob, self.amm_pools,
                                    self.env, period_trades,
                                    mm_state_summary=self._market_maker_state_summary(),
                                    arbitrage_trades=(
                                        getattr(self.arbitrageur, 'period_trades', [])
                                        if self.arbitrageur is not None else []
                                    ),
                                    routing_attempts=period_routing)

        return self

    # ------------------------------------------------------------------
    # Convenience factory for multi-venue mode
    # ------------------------------------------------------------------

    @classmethod
    def default_fx(cls,
                   n_noise: int = calibrated_default('n_noise', 12),
                   # Top-tier EUR/USD intermediation is supplied by ~5-8
                   # heterogeneous dealers (BIS WP 1073; Triennial 2022).
                   # Default of 5 reflects the major-pair tier-1 set (JPM,
                   # Deutsche, Citi, UBS, HSBC) with stepped alpha0 / d0
                   # for aggressiveness heterogeneity.
                   n_mm: int = calibrated_default('n_mm', 5),
                   n_fast_lp: int = calibrated_default('n_fast_lp', 10),
                   n_latent_lp: int = calibrated_default('n_latent_lp', 6),
                   # CLOB-side fundamental / momentum book agents. With
                   # the FX-aware Fundamentalist branch (env.fair_price
                   # anchor) these participate as *limit-order liquidity
                   # providers* complementary to the FX-taker fundament-
                   # alists (n_fx_fund) which consume liquidity via market
                   # orders. Heterogeneous strategy mix is consistent with
                   # the literature on FX participant types: macro funds /
                   # CTAs (Fundamentalist), trend followers (Chartist),
                   # and regime-switching hedge funds (Universalist). Kept
                   # small (2/1/1 = 4 extra agents) so the calibrated
                   # dealer pack remains the primary liquidity supplier.
                   n_clob_fund: int = calibrated_default('n_clob_fund', 2),
                   # Dispersion of a book fundamentalist's reading of the
                   # latent value, and how far from that reading it rests,
                   # both as multiples of the price volatility of one period.
                   clob_fund_observation_noise: float = calibrated_default(
                       'clob_fund_observation_noise', 1.0),
                   clob_fund_quote_offset: float = calibrated_default(
                       'clob_fund_quote_offset', 0.5),
                   n_clob_chart: int = calibrated_default('n_clob_chart', 1),
                   n_clob_univ: int = calibrated_default('n_clob_univ', 1),
                   n_fx_takers: int = calibrated_default('n_fx_takers', 15),
                   n_fx_fund: int = calibrated_default('n_fx_fund', 5),
                   n_retail: int = calibrated_default('n_retail', 10),
                   n_institutional: int = calibrated_default('n_institutional', 3),
                   clob_std: float = calibrated_default('clob_std', 2.0),
                   clob_volume: int = calibrated_default('clob_volume', 1000),
                   # The price grid the book quotes on. It was a default buried
                   # in the exchange constructor and absent from the manifest,
                   # while being the single largest determinant of the realised
                   # quoted spread, so it is a declared model input here.
                   price_tick: float = calibrated_default('price_tick', 0.005),
                   cpmm_reserves: float = calibrated_default('cpmm_reserves', 1000.0),
                   hfmm_reserves: float = calibrated_default('hfmm_reserves', 1000.0),
                   hfmm_A: float = calibrated_default('hfmm_A', 10.0),
                   cpmm_fee: float = calibrated_default('cpmm_fee', 0.003),
                   hfmm_fee: float = calibrated_default('hfmm_fee', 0.001),
                   stress_start: Optional[int] = calibrated_default('stress_start', None),
                   stress_end: Optional[int] = calibrated_default('stress_end', None),
                   sigma_low: float = calibrated_default('sigma_low', 0.01),
                   sigma_high: float = calibrated_default('sigma_high', 0.05),
                   # ``sigma`` is the stress index that drives dealer spreads,
                   # withdrawal and provider behaviour, and it is calibrated as
                   # such. The volatility of the latent price is a separate
                   # quantity and is set here. The two were previously tied at
                   # a ratio of 0.35, which put 35 basis points of price
                   # movement into every tick. Against a quoted spread of two
                   # basis points and a pool fee of five that is not a market
                   # any liquidity provider survives, and the mismatch was the
                   # reason the endogenous pool wound down before the shock it
                   # exists to absorb. The default now maps one tick to one
                   # second of EUR/USD at six per cent a year, which is the
                   # realised figure for 2026 and sits inside the five to ten
                   # per cent range of a calm year. Stress raises sigma
                   # fivefold and so carries the price to thirty per cent.
                   price_vol_scale: float = calibrated_default('price_vol_scale', 0.0012859),
                   # The cost of committed capital as a rate per tick. About
                   # three per cent a year across the two currencies of the
                   # pair, at one second to the tick.
                   funding_rate_scale: float = calibrated_default(
                       'funding_rate_scale', 1.3319e-9),
                   c_low: float = calibrated_default('c_low', 0.001),
                   c_high: float = calibrated_default('c_high', 0.02),
                   price: float = calibrated_default('price', 100.0),
                   # Weak routing prior under liquidity-aware choice.  No
                   # source identifies a numerical AMM share for an EBS-like
                   # FX market, so this is a disclosed design input rather
                   # than a calibrated volume quota.
                   amm_share_pct: float = calibrated_default('amm_share_pct', 22.0),
                   venue_choice_rule: str = calibrated_default('venue_choice_rule', 'fixed_share'),
                   deterministic: bool = False,
                   # ── intra-AMM venue selection ────────────────
                   beta_amm: float = calibrated_default('beta_amm', 0.05),
                   cpmm_bias_bps: float = calibrated_default('cpmm_bias_bps', 5.0),
                   cost_noise_std: float = calibrated_default('cost_noise_std', 1.5),
                   routing_cost_scale_bps: float = calibrated_default(
                       'routing_cost_scale_bps', 4.0),
                   routing_prior_mix_cap: float = calibrated_default(
                       'routing_prior_mix_cap', 0.18),
                   routing_basis_scale_bps: float = calibrated_default(
                       'routing_basis_scale_bps', 50.0),
                   routing_clob_depth_multiple: float = calibrated_default(
                       'routing_clob_depth_multiple', 10.0),
                   routing_amm_depth_multiple: float = calibrated_default(
                       'routing_amm_depth_multiple', 8.0),
                   # ── venue configuration ──────────────────────
                   enable_amm: bool = calibrated_default('enable_amm', True),
                   enable_cpmm: bool = calibrated_default('enable_cpmm', False),
                   clob_liq: float = calibrated_default('clob_liq', 1.0),
                   clob_anchor_strength: float = calibrated_default('clob_anchor_strength', 0.35),
                   clob_anchor_threshold_bps: float = calibrated_default('clob_anchor_threshold_bps', 5.0),
                   # Anonymous background liquidity is a *backstop* for the
                   # CLOB book, not a replacement for trader-owned depth.
                   # With 5 MMs the trader pack meets the touch-depth
                   # anchor on its own; the background layer is capped at
                   # 0.30 * trader near-mid depth so it does not over-
                   # supply liquidity in calm conditions.
                   clob_background_target_ratio: float = 1.0,
                   clob_support_max_share: float = calibrated_default('clob_support_max_share', 0.30),
                   clob_amm_interaction: str = calibrated_default('clob_amm_interaction', 'competition'),
                   clob_amm_spread_impact_bps: float = calibrated_default('clob_amm_spread_impact_bps', 3.0),
                   clob_amm_depth_impact: float = calibrated_default('clob_amm_depth_impact', 60.0),
                   mm_alpha0_base: float = calibrated_default('mm_alpha0_base', 2.1),
                   mm_alpha0_step: float = calibrated_default('mm_alpha0_step', 0.3),
                   mm_alpha1: float = calibrated_default('mm_alpha1', 8.0),
                   mm_alpha2: float = calibrated_default('mm_alpha2', 700.0),
                   mm_alpha3: float = calibrated_default('mm_alpha3', 35.0),
                   # 5 MMs * mm_d0_base ~ 200 displayed depth per side,
                   # plus FastRecyclerLP and a capped anonymous support
                   # layer, lands near the 220 base/side touch-depth
                   # anchor (Lo & Hall L1).
                   mm_d0_base: float = calibrated_default('mm_d0_base', 40.0),
                   mm_d0_step: float = calibrated_default('mm_d0_step', 5.0),
                   mm_d1: float = calibrated_default('mm_d1', 560.0),
                   mm_d2: float = calibrated_default('mm_d2', 360.0),
                   mm_d3: float = calibrated_default('mm_d3', 12.0),
                   hedger_flow_persistence: float = calibrated_default('hedger_flow_persistence', 0.10),
                   retail_flow_persistence: float = calibrated_default('retail_flow_persistence', 0.28),
                   institutional_flow_persistence: float = calibrated_default('institutional_flow_persistence', 0.18),
                   common_flow_response: float = calibrated_default('common_flow_response', 0.05),
                   fx_flow_intensity_scale: float = calibrated_default('fx_flow_intensity_scale', 1.0),
                   mm_withdraw_threshold: float = calibrated_default('mm_withdraw_threshold', 0.7),
                   mm_withdraw_threshold_step: float = calibrated_default('mm_withdraw_threshold_step', 0.3),
                   mm_reentry_threshold: float = calibrated_default('mm_reentry_threshold', 0.4),
                   mm_loss_threshold_bps: float = calibrated_default('mm_loss_threshold_bps', 50.0),
                   mm_quote_life: int = calibrated_default('mm_quote_life', 290),
                   mm_quote_refresh_tol_bps: float = calibrated_default('mm_quote_refresh_tol_bps', 20.0),
                   mm_n_levels: int = calibrated_default('mm_n_levels', 10),
                   mm_level_step_ticks: float = calibrated_default('mm_level_step_ticks', 2.0),
                   mm_inv_skew_bps: float = calibrated_default('mm_inv_skew_bps', 0.3),
                   mm_revenue_horizon: int = calibrated_default('mm_revenue_horizon', 300),
                   fast_lp_base_spread_bps: float = calibrated_default('fast_lp_base_spread_bps', 1.6),
                   fast_lp_quote_life: int = calibrated_default('fast_lp_quote_life', 3),
                   fast_lp_base_qty: int = calibrated_default('fast_lp_base_qty', 1),
                   fast_lp_levels: int = calibrated_default('fast_lp_levels', 1),
                   fast_lp_base_withdraw_prob: float = calibrated_default('fast_lp_base_withdraw_prob', 0.10),
                   fast_lp_stress_abstention: float = calibrated_default('fast_lp_stress_abstention', 0.10),
                   mm_min_withdraw_ticks: int = calibrated_default('mm_min_withdraw_ticks', 4),
                   mm_reentry_ticks: int = calibrated_default('mm_reentry_ticks', 3),
                   mm_withdraw_confirmation_ticks: int = calibrated_default(
                       'mm_withdraw_confirmation_ticks', 2),
                   maintenance_margin_ratio: float = calibrated_default('maintenance_margin_ratio', 0.06),
                   liquidation_fraction: float = calibrated_default('liquidation_fraction', 0.75),
                   borrow_spread_multiplier: float = calibrated_default('borrow_spread_multiplier', 0.6),
                   short_borrow_spread_multiplier: float = calibrated_default('short_borrow_spread_multiplier', 0.8),
                   amm_liq: float = calibrated_default('amm_liq', 1.0),
                   match_initial_depth: bool = calibrated_default('match_initial_depth', False),
                   shock_iter: Optional[int] = calibrated_default('shock_iter', None),
                   shock_pct: float = calibrated_default('shock_pct', -20.0),
                   shock_mode: str = calibrated_default('shock_mode', 'research'),
                   fundamental_shock_pct: float = calibrated_default('fundamental_shock_pct', 0.0),
                   order_flow_shock_qty: float = calibrated_default('order_flow_shock_qty', 0.0),
                   order_flow_shock_side: str = calibrated_default('order_flow_shock_side', 'auto'),
                   liquidity_shock_frac: float = calibrated_default('liquidity_shock_frac', 0.0),
                   force_mm_pause: bool = calibrated_default('force_mm_pause', False),
                   forced_pause_ticks: Optional[int] = calibrated_default('forced_pause_ticks', None),
                   funding_vol_shock_intensity: float = calibrated_default('funding_vol_shock_intensity', 0.0),
                   arb_max_correction_pct: float = calibrated_default('arb_max_correction_pct', 10.0),
                   arb_trade_fraction_cap: float = calibrated_default('arb_trade_fraction_cap', 0.20),
                   amm_lp_wallet_cash_buffer_ratio: float = calibrated_default('amm_lp_wallet_cash_buffer_ratio', 0.20),
                   amm_lp_wallet_base_buffer_ratio: float = calibrated_default('amm_lp_wallet_base_buffer_ratio', 0.20),
                   # ── who supplies the automated venue ──────────
                   # 'rule' is the original provider, which adjusts reserves
                   # by a fixed rule and can neither leave nor be replaced.
                   # 'endogenous' is a population that compares its own
                   # result against an outside option, exits when the pool
                   # stops paying, and re-enters when it starts again, so
                   # liquidity supply is a decision rather than a setting.
                   amm_lp_model: str = calibrated_default('amm_lp_model', 'endogenous'),
                   amm_lp_n_providers: int = calibrated_default('amm_lp_n_providers', 5),
                   amm_lp_n_entrants: int = calibrated_default('amm_lp_n_entrants', 5),
                   amm_lp_outside_option: float = calibrated_default(
                       'amm_lp_outside_option', 1.3319e-9),
                   amm_lp_option_dispersion: float = calibrated_default(
                       'amm_lp_option_dispersion', 0.6),
                   amm_lp_kappa: float = calibrated_default('amm_lp_kappa', 0.35),
                   amm_lp_response_scale: float = calibrated_default(
                       'amm_lp_response_scale', 1e-6),
                   amm_lp_max_adj: float = calibrated_default(
                       'amm_lp_max_adj', 0.0023873085271651773),
                   amm_lp_ewma_alpha: float = calibrated_default(
                       'amm_lp_ewma_alpha', 0.02),
                   amm_lp_exit_patience: int = calibrated_default(
                       'amm_lp_exit_patience', 290),
                   amm_lp_entry_patience: int = calibrated_default(
                       'amm_lp_entry_patience', 290),
                   amm_lp_entry_margin: float = calibrated_default(
                       'amm_lp_entry_margin', 0.25),
                   amm_lp_wallet_ratio: float = calibrated_default(
                       'amm_lp_wallet_ratio', 0.20),
                   amm_lp_withdraw_skew: float = calibrated_default(
                       'amm_lp_withdraw_skew', 0.0),
                   amm_lp_subsidy_rate: float = calibrated_default(
                       'amm_lp_subsidy_rate', 0.0),
                   amm_lp_loss_rebate_fraction: float = calibrated_default(
                       'amm_lp_loss_rebate_fraction', 0.0),
                   # Prefunded arbitrage wallet. The one-pool defaults are one
                   # times the corresponding reserves: deliberately large
                   # enough not to bind ordinary alignment, while keeping both
                   # trade directions resource backed and auditable.
                   amm_arb_cash_buffer_ratio: float = calibrated_default('amm_arb_cash_buffer_ratio', 1.0),
                   amm_arb_base_buffer_ratio: float = calibrated_default('amm_arb_base_buffer_ratio', 1.0),
                   dynamic_fee: bool = calibrated_default('dynamic_fee', False),
                   # ── liquidity shock decay rates ──────────────
                   reprice_prob_recovery: Optional[float] = None,
                   anchor_strength_recovery: Optional[float] = None,
                   bg_target_ratio_recovery: Optional[float] = None,
                   toxic_flow_decay: Optional[float] = None,
                   liquidity_shock_decay: Optional[float] = None,
                   ) -> Simulator:
        """
        Build a ready-to-run multi-venue FX simulator.

        Venue configuration
        -------------------
        n_mm : int
            Number of Market Makers on the CLOB (0 = no MM).
        enable_amm : bool
            If False, no AMM pools / LP / arbitrageur are created.
        clob_liq : float  (0 … ∞, default 1.0)
            Multiplier that scales CLOB-side liquidity:
            ``clob_volume``, ``n_noise``, ``n_fast_lp``, ``n_latent_lp``,
            and MM depth params ``d0`` are all multiplied by this factor.
        amm_liq : float  (0 … ∞, default 1.0)
            Multiplier that scales AMM-side liquidity:
            ``cpmm_reserves`` and ``hfmm_reserves`` are multiplied
            by this factor.
        match_initial_depth : bool, default False
            Optional calibration aid. If enabled, rebalance the live CLOB
            at t=0 so its near-mid depth is comparable to aggregate AMM
            effective depth. Disabled by default to avoid imposing a
            like-for-like market structure before trading starts.

        Flow allocation
        ---------------
        amm_share_pct : float (0–100, default 25)
            Target probability (%) of routing a trade to AMM vs CLOB.
            E.g. 25 means ~25% AMM, ~75% CLOB.
        venue_choice_rule : str
            Routing regime. ``fixed_share`` preserves the current two-step
            top-level AMM/CLOB split, while ``liquidity_aware`` makes
            routing sensitive to cost, depth, and venue price alignment.

        Intra-AMM venue selection
        -------------------------
        beta_amm : float
            Logit sensitivity for CPMM vs HFMM choice (lower → more
            uniform; higher → cost-driven).  Default 0.3.
        cpmm_bias_bps : float
            Non-monetary utility discount subtracted from perceived
            CPMM cost (bps).  Models convenience / accessibility.
        cost_noise_std : float
            Std of Gaussian noise added to AMM cost estimates (bps).
            Models imperfect information about pool costs.

        Convenience shortcuts
        ---------------------
        * **CLOB-only**:  ``enable_amm=False``
        * **AMM-only**:   ``n_mm=0, clob_liq=0.1``
          In this mode a **ShadowCLOB** replaces the live book: the
          CLOB never depletes and arb/metrics use the exogenous
          GBM fair price S_t.
        * **70/30 CLOB-heavy**: ``clob_liq=1.4, amm_liq=0.6``
        """
        from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
        from AgentBasedModel.venues.clob import CLOBVenue, ShadowCLOB
        from AgentBasedModel.environment.processes import MarketEnvironment
        from AgentBasedModel.metrics.logger import MetricsLogger

        # ── Decide CLOB mode ────────────────────────────────────────
        # "Shadow" mode: no live MM, thin CLOB → use ShadowCLOB for
        # metrics and arb reference, keep tiny live book for backward
        # compatibility with agents that still need exchange.price().
        shadow_clob = (enable_amm and n_mm == 0)

        # ── Apply liquidity scaling ──────────────────────────────────
        eff_clob_volume = max(10, int(clob_volume * clob_liq))
        # Zero is a meaningful calibration choice: the FX taker population
        # already supplies demand, while forcing one legacy mixed Random
        # agent into the book adds an unidentified passive supplier.
        eff_n_noise = max(0, int(n_noise * clob_liq))
        eff_n_fast_lp = max(0, int(n_fast_lp * clob_liq))
        eff_n_latent_lp = max(0, int(n_latent_lp * clob_liq))
        eff_d0 = 75.0 * clob_liq
        eff_cpmm_res = cpmm_reserves * amm_liq
        eff_hfmm_res = hfmm_reserves * amm_liq

        # Exchange (CLOB) — always present (reference price + classic agents)
        exchange = ExchangeAgent(price=price, std=clob_std,
                     volume=eff_clob_volume,
                     price_tick=price_tick,
                     background_target_ratio=clob_background_target_ratio,
                     background_max_share_of_trader_depth=clob_support_max_share,
                     anchor_strength=clob_anchor_strength,
                     anchor_threshold_bps=clob_anchor_threshold_bps)

        # Environment — always pass price so GBM S_t is available.
        # Its stochastic path runs on a stream of its own, drawn here before
        # anything venue specific is built, so that a market with the facility
        # and a market without it see the same volatility, funding and price
        # realisation on the same seed. Sharing the global generator made the
        # two arms diverge from the second tick and put a difference of
        # realisations inside every paired comparison.
        env_rng = random.Random(_venue_seed('environment'))
        # Equal-price dealer priority is also a paired-design primitive. Draw
        # its private stream before any AMM-specific object is constructed, so
        # the same seed produces the same dealer permutation with and without
        # the facility. Routing or LP draws cannot perturb it afterwards.
        dealer_scheduler_rng = _isolated_venue_rng('dealer_scheduler')
        fx_scheduler_rng = _isolated_venue_rng('fx_scheduler')
        env = MarketEnvironment(
            rng=env_rng,
            sigma_low=sigma_low, sigma_high=sigma_high,
            c_low=c_low, c_high=c_high,
            stress_start=stress_start, stress_end=stress_end,
            mode='piecewise',
            price=price,
            price_vol_scale=price_vol_scale,
            funding_rate_scale=funding_rate_scale,
        )

        # CLOB wrapper: live or shadow
        if shadow_clob:
            clob = ShadowCLOB(env)
        else:
            clob = CLOBVenue(exchange, env=env)

        # ── AMM pools (optional) ────────────────────────────────────
        amm_pools: Dict[str, Any] = {}
        lp_providers: List = []
        arb = None

        if enable_amm:
            hfmm = HFMMPool(x=eff_hfmm_res,
                            y=eff_hfmm_res * price,
                            A=hfmm_A, fee=hfmm_fee, rate=price,
                            dynamic_fee=dynamic_fee)
            amm_pools = {'hfmm': hfmm}
            # The constant product pool is no longer a second venue of this
            # market. Its curvature is wrong for a pair that trades in a
            # narrow band, it carried a small share of the flow, and holding
            # two facilities at once made the question of how much capital
            # the facility commits harder to state than it needed to be. It
            # is kept in two places instead. It anchors the measurement of
            # the rebalancing curvature, where reproducing the frictionless
            # one eighth is what makes that measurement credible, and it
            # serves as the unamplified arm of the resource matched
            # comparison, which is run one pool at a time so that the two
            # curves do not compete for the same flow.
            if enable_cpmm:
                amm_pools['cpmm'] = CPMMPool(x=eff_cpmm_res,
                                             y=eff_cpmm_res * price,
                                             fee=cpmm_fee,
                                             dynamic_fee=dynamic_fee)
            if str(amm_lp_model).lower() in ('endogenous', 'endo'):
                from AgentBasedModel.agents.lp_endogenous import LPPopulation
                lp_providers = [
                    LPPopulation(
                        pool, env,
                        n_providers=amm_lp_n_providers,
                        n_entrants=amm_lp_n_entrants,
                        outside_option=amm_lp_outside_option,
                        option_dispersion=amm_lp_option_dispersion,
                        kappa=amm_lp_kappa,
                        response_scale=amm_lp_response_scale,
                        max_adj=amm_lp_max_adj,
                        ewma_alpha=amm_lp_ewma_alpha,
                        exit_patience=amm_lp_exit_patience,
                        entry_patience=amm_lp_entry_patience,
                        entry_margin=amm_lp_entry_margin,
                        wallet_ratio=amm_lp_wallet_ratio,
                        withdraw_skew=amm_lp_withdraw_skew,
                        subsidy_rate=amm_lp_subsidy_rate,
                        loss_rebate_fraction=amm_lp_loss_rebate_fraction,
                        # Each pool draws its own stream, but allocating it
                        # must not advance the shared stream used to construct
                        # the CLOB and customer populations.  Otherwise an
                        # AMM/no-AMM pair starts from different customer
                        # arrival, side and size innovations on the same seed.
                        rng=_isolated_venue_rng(f'lp_population:{name}'))
                    for name, pool in amm_pools.items()
                ]
            else:
                # phi1 is the share of the fee return a provider commits back,
                # phi2 the curvature of the rebalancing loss and phi3 the
                # weight on the cost of the capital. All three are
                # dimensionless and multiply rates per period.
                #
                # For a constant product pool the curvature is one eighth of
                # the variance, the frictionless value in \citet{milionis2022}.
                # The amplified curve rebalances more for the same price move,
                # so its curvature is larger. Measured on the curve itself by
                # walking the pool to a nearby price and comparing against
                # holding the reserves, it is one eighth plus a quarter of the
                # amplification, exact to four decimal places over
                # amplifications from one to forty. The same measurement
                # returns exactly one eighth on the constant product curve,
                # which is what makes the method credible. An earlier version
                # carried a flat 2.25 that was assumed and not derived, and at
                # the calibrated amplification of eighteen it understated the
                # loss by more than half.
                #
                # Built by iterating the pools that are actually present. The
                # earlier form named both pools directly, so removing one from
                # the market left this branch referring to a pool that had not
                # been constructed.
                _rule_cfg = {
                    'cpmm': (calibrated_default('amm_lp_phi1_cpmm', 1.0),
                             calibrated_default('amm_lp_phi2_cpmm', 0.125),
                             calibrated_default('amm_lp_phi3_cpmm', 1.0),
                             0.45),
                    'hfmm': (calibrated_default('amm_lp_phi1_hfmm', 1.0),
                             0.125 + 0.25 * hfmm_A,
                             calibrated_default('amm_lp_phi3_hfmm', 1.0),
                             0.75),
                }
                lp_providers = []
                for name, pool in amm_pools.items():
                    phi1, phi2, phi3, core = _rule_cfg[name]
                    lp_providers.append(AMMProvider(
                        pool, env,
                        phi1=phi1, phi2=phi2, phi3=phi3,
                        core_liquidity_ratio=core,
                        wallet_cash_buffer_ratio=amm_lp_wallet_cash_buffer_ratio,
                        wallet_base_buffer_ratio=amm_lp_wallet_base_buffer_ratio))
            arb = AMMArbitrageur(
                clob,
                amm_pools,
                env=env,
                max_correction_pct=arb_max_correction_pct,
                trade_fraction_cap=arb_trade_fraction_cap,
                cash_buffer_ratio=amm_arb_cash_buffer_ratio,
                asset_buffer_ratio=amm_arb_base_buffer_ratio,
            )

        if not shadow_clob:
            exchange.rebalance_background_liquidity(price, target_ratio=1.0)

        if enable_amm and not shadow_clob and match_initial_depth and amm_pools:
            ref_mid = clob.mid_price()
            aggregate_amm_depth = sum(
                max(0.0, pool.effective_depth(ref_mid))
                for pool in amm_pools.values()
            )
            if aggregate_amm_depth > 0:
                exchange.set_background_depth_target(
                    ref_mid,
                    aggregate_amm_depth,
                    respect_trader_cap=False,
                )

        # ── CLOB book agents (diverse limit-order providers) ─────
        def _set_risk_limits(trader, cash_borrow: float, short_assets: float):
            trader.max_cash_borrow = max(0.0, float(cash_borrow))
            trader.max_short_assets = max(0.0, float(short_assets))

        def _set_risk_policy(trader):
            trader.maintenance_margin_ratio = max(0.0, float(maintenance_margin_ratio))
            trader.liquidation_fraction = max(0.0, min(1.0, float(liquidation_fraction)))
            trader.borrow_spread_multiplier = max(0.0, float(borrow_spread_multiplier))
            trader.short_borrow_spread_multiplier = max(0.0, float(short_borrow_spread_multiplier))

        book_agents = [Random(exchange, cash=1e4, env=env)
                       for _ in range(eff_n_noise)]
        for _ in range(eff_n_fast_lp):
            book_agents.append(FastRecyclerLP(
                exchange, cash=2e4, env=env,
                ttl=fast_lp_quote_life,
                base_qty=fast_lp_base_qty,
                levels=fast_lp_levels,
                base_spread_bps=fast_lp_base_spread_bps,
                base_withdraw_prob=fast_lp_base_withdraw_prob,
                stress_abstention=fast_lp_stress_abstention))
        for _ in range(eff_n_latent_lp):
            book_agents.append(LatentLP(exchange, cash=2e4, env=env))
        for _ in range(n_clob_fund):
            book_agents.append(Fundamentalist(
                exchange, cash=1e4, access=1, env=env,
                observation_noise_multiple=clob_fund_observation_noise,
                quote_offset_multiple=clob_fund_quote_offset))
        for _ in range(n_clob_chart):
            book_agents.append(Chartist(exchange, cash=1e4, env=env))
        for _ in range(n_clob_univ):
            book_agents.append(Universalist(exchange, cash=1e4, access=1, env=env))
        for agent in book_agents:
            _set_risk_limits(agent, cash_borrow=2.5e4, short_assets=75.0)
            _set_risk_policy(agent)

        # ── Market Makers ────────────────────────────────────────────
        mm = None
        for i in range(n_mm):
            dealer_withdraw_threshold = max(
                0.1, mm_withdraw_threshold + i * mm_withdraw_threshold_step
            )
            dealer_reentry_threshold = min(
                dealer_withdraw_threshold,
                mm_reentry_threshold + 0.55 * i * mm_withdraw_threshold_step,
            )
            _mm = MarketMaker(exchange, cash=7e4, env=env,
                              amm_pools=amm_pools,
                              alpha0=mm_alpha0_base + i * mm_alpha0_step,
                              alpha1=mm_alpha1, alpha2=mm_alpha2,
                              alpha3=mm_alpha3,
                              d0=max(5.0, mm_d0_base * clob_liq - i * mm_d0_step),
                              d1=mm_d1, d2=mm_d2, d3=mm_d3, d_min=3.0,
                              n_levels=mm_n_levels,
                              level_step_ticks=mm_level_step_ticks,
                              venue_interaction_mode=clob_amm_interaction,
                              amm_spread_impact_bps=clob_amm_spread_impact_bps,
                              amm_depth_impact=clob_amm_depth_impact,
                              withdrawal_threshold=dealer_withdraw_threshold,
                              reentry_threshold=dealer_reentry_threshold,
                              loss_threshold_bps=mm_loss_threshold_bps,
                              quote_life=mm_quote_life,
                              quote_refresh_tol_bps=mm_quote_refresh_tol_bps,
                              inv_skew_bps=mm_inv_skew_bps,
                              revenue_horizon=mm_revenue_horizon,
                              min_withdraw_ticks=mm_min_withdraw_ticks,
                              reentry_ticks=mm_reentry_ticks,
                              withdrawal_confirmation_ticks=(
                                  mm_withdraw_confirmation_ticks
                              ))
            _set_risk_limits(_mm, cash_borrow=8.0e4, short_assets=250.0)
            _set_risk_policy(_mm)
            if mm is None:
                # Compatibility handle only; all dealers share the shuffled
                # step-2 scheduler and hence the same opportunity for queue
                # priority.
                mm = _mm
            else:
                book_agents.append(_mm)

        # ── Liquidity takers with venue routing ─────────────────────
        fx_traders: List = []
        for i in range(n_fx_takers):
            trader = Random(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                venue_choice_rule=venue_choice_rule,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                routing_cost_scale_bps=routing_cost_scale_bps,
                routing_prior_mix_cap=routing_prior_mix_cap,
                routing_basis_scale_bps=routing_basis_scale_bps,
                routing_clob_depth_multiple=routing_clob_depth_multiple,
                routing_amm_depth_multiple=routing_amm_depth_multiple,
                flow_rng=_isolated_venue_rng(f'customer_flow:hedger:{i}'),
                routing_rng=_isolated_venue_rng(
                    f'customer_routing:hedger:{i}'
                ),
                trade_prob=min(1.0, 0.28 * max(0.0, fx_flow_intensity_scale)),
                q_min=1, q_max=4, label='Hedger',
                flow_role='Hedger', flow_persistence=hedger_flow_persistence, session_sensitivity=0.8,
                common_flow_response=common_flow_response,
            )
            _set_risk_limits(trader, cash_borrow=5.0e3, short_assets=12.0)
            _set_risk_policy(trader)
            fx_traders.append(trader)
        for i in range(n_fx_fund):
            trader = Fundamentalist(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                venue_choice_rule=venue_choice_rule,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                routing_cost_scale_bps=routing_cost_scale_bps,
                routing_prior_mix_cap=routing_prior_mix_cap,
                routing_basis_scale_bps=routing_basis_scale_bps,
                routing_clob_depth_multiple=routing_clob_depth_multiple,
                routing_amm_depth_multiple=routing_amm_depth_multiple,
                routing_rng=_isolated_venue_rng(
                    f'customer_routing:fundamental:{i}'
                ),
                fundamental_rate=price, fx_gamma=5e-3, fx_q_max=10,
                flow_role='LeveragedDirectional',
            )
            _set_risk_limits(trader, cash_borrow=7.5e3, short_assets=20.0)
            _set_risk_policy(trader)
            fx_traders.append(trader)
        for i in range(n_retail):
            trader = Random(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                venue_choice_rule=venue_choice_rule,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                routing_cost_scale_bps=routing_cost_scale_bps,
                routing_prior_mix_cap=routing_prior_mix_cap,
                routing_basis_scale_bps=routing_basis_scale_bps,
                routing_clob_depth_multiple=routing_clob_depth_multiple,
                routing_amm_depth_multiple=routing_amm_depth_multiple,
                flow_rng=_isolated_venue_rng(f'customer_flow:retail:{i}'),
                routing_rng=_isolated_venue_rng(
                    f'customer_routing:retail:{i}'
                ),
                trade_prob=min(1.0, 0.42 * max(0.0, fx_flow_intensity_scale)),
                q_min=1, q_max=2, label='RetailToxic',
                flow_role='RetailToxic', flow_persistence=retail_flow_persistence, session_sensitivity=1.1,
                common_flow_response=common_flow_response,
            )
            _set_risk_limits(trader, cash_borrow=2.5e3, short_assets=6.0)
            _set_risk_policy(trader)
            fx_traders.append(trader)
        for i in range(n_institutional):
            trader = Random(
                exchange, cash=5e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                venue_choice_rule=venue_choice_rule,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                routing_cost_scale_bps=routing_cost_scale_bps,
                routing_prior_mix_cap=routing_prior_mix_cap,
                routing_basis_scale_bps=routing_basis_scale_bps,
                routing_clob_depth_multiple=routing_clob_depth_multiple,
                routing_amm_depth_multiple=routing_amm_depth_multiple,
                flow_rng=_isolated_venue_rng(
                    f'customer_flow:institutional:{i}'
                ),
                routing_rng=_isolated_venue_rng(
                    f'customer_routing:institutional:{i}'
                ),
                trade_prob=min(1.0, 0.14 * max(0.0, fx_flow_intensity_scale)),
                q_min=15, q_max=50, label='RealMoney',
                flow_role='RealMoney', flow_persistence=institutional_flow_persistence, session_sensitivity=1.15,
                common_flow_response=common_flow_response,
            )
            _set_risk_limits(trader, cash_borrow=3.0e4, short_assets=120.0)
            _set_risk_policy(trader)
            fx_traders.append(trader)

        # Logger
        logger = MetricsLogger(
            Q_grid=[1, 2, 5, 10, 20, 50],
            slippage_thresholds=[5, 10, 25, 50],
        )

        # Full book-agent list for SimulatorInfo (Chartist / Universalist
        # need sentiment / strategy updates each iteration).
        all_book_agents = list(book_agents)
        if mm is not None:
            all_book_agents.append(mm)

        return cls(
            exchange=exchange,
            traders=all_book_agents,
            clob=clob,
            amm_pools=amm_pools,
            env=env,
            fx_traders=fx_traders,
            book_agents=book_agents,
            market_maker=mm,
            lp_providers=lp_providers,
            arbitrageur=arb,
            logger=logger,
            dealer_scheduler_rng=dealer_scheduler_rng,
            fx_scheduler_rng=fx_scheduler_rng,
            shock_iter=shock_iter,
            shock_pct=shock_pct,
            shock_mode=shock_mode,
            realism_shock_config={
                'fundamental_pct': fundamental_shock_pct,
                'order_flow_qty': order_flow_shock_qty,
                'order_flow_side': order_flow_shock_side,
                'liquidity_frac': liquidity_shock_frac,
                'force_mm_pause': force_mm_pause,
                'forced_pause_ticks': forced_pause_ticks,
                'funding_vol_intensity': funding_vol_shock_intensity,
                'reprice_prob_recovery': reprice_prob_recovery,
                'anchor_strength_recovery': anchor_strength_recovery,
                'bg_target_ratio_recovery': bg_target_ratio_recovery,
                'toxic_flow_decay': toxic_flow_decay,
                'liquidity_shock_decay': liquidity_shock_decay,
            },
        )

    @classmethod
    def default_fx_no_amm(cls, **kwargs) -> Simulator:
        """Shortcut: CLOB-only FX simulator (no AMM)."""
        kwargs.setdefault('enable_amm', False)
        return cls.default_fx(**kwargs)


class SimulatorInfo:
    """
    SimulatorInfo is responsible for capturing data during simulating
    """

    def __init__(self, exchange: ExchangeAgent = None, traders: list = None):
        self.exchange = exchange
        self.traders = {t.id: t for t in traders}

        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.dividends = list()  # dividend paid at each iteration
        self.orders = list()  # order book statistics

        # Agent statistics
        self.equities = list()  # agent: equity
        self.cash = list()  # agent: cash
        self.assets = list()  # agent: number of assets
        self.types = list()  # agent: current type
        self.sentiments = list()  # agent: current sentiment
        self.returns = [{tr_id: 0 for tr_id in self.traders.keys()}]  # agent: iteration return

        """
        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.spread_sizes = list()  # bid-ask spread sizes
        self.dividends = list()
        self.orders_quantities = list()  # list -> (bid, ask)
        self.orders_volumes = list()  # list -> (bid, ask) -> (sum, mean, q1, q3, std)
        self.orders_prices = list()  # list -> (bid, ask) -> (mean, q1, q3, std)

        # Agent Statistics
        self.equity = list()  # sum of equity of agents
        self.cash = list()  # sum of cash of agents
        self.assets_qty = list()  # sum of number of assets of agents
        self.assets_value = list()  # sum of value of assets of agents
        """

    def capture(self):
        """
        Method called at the end of each iteration to capture basic info on simulation.

        **Attributes:**

        *Market Statistics*

        - :class:`list[float]` **prices** --> stock prices on each iteration
        - :class:`list[dict]` **spreads** --> order book spreads on each iteration
        - :class:`list[float]` **dividends** --> dividend paid on each iteration
        - :class:`list[dict[dict]]` **orders** --> order book price, volume, quantity stats on each iteration

        *Traders Statistics*

        - :class:`list[dict]` **equities** --> each agent's equity on each iteration
        - :class:`list[dict]` **cash** --> each agent's cash on each iteration
        - :class:`list[dict]` **assets** --> each agent's number of stocks on each iteration
        - :class:`list[dict]` **types** --> each agent's type on each iteration
        """
        # Market Statistics
        self.prices.append(self.exchange.price())
        self.spreads.append((self.exchange.spread()))
        self.dividends.append(self.exchange.dividend())
        self.orders.append({
            'quantity': {'bid': len(self.exchange.order_book['bid']), 'ask': len(self.exchange.order_book['ask'])},
            # 'price mean': {
            #     'bid': mean([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.price for order in self.exchange.order_book['ask']])},
            # 'price std': {
            #     'bid': std([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.price for order in self.exchange.order_book['ask']])},
            # 'volume sum': {
            #     'bid': sum([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': sum([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume mean': {
            #     'bid': mean([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume std': {
            #     'bid': std([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.qty for order in self.exchange.order_book['ask']])}
        })

        # Trader Statistics
        self.equities.append({t_id: t.equity() for t_id, t in self.traders.items()})
        self.cash.append({t_id: t.cash for t_id, t in self.traders.items()})
        self.assets.append({t_id: t.assets for t_id, t in self.traders.items()})
        self.types.append({t_id: t.type for t_id, t in self.traders.items()})
        self.sentiments.append({t_id: t.sentiment for t_id, t in self.traders.items() if t.type == 'Chartist'})
        self.returns.append({tr_id: (self.equities[-1][tr_id] - self.equities[-2][tr_id]) / self.equities[-2][tr_id]
                             for tr_id in self.traders.keys()}) if len(self.equities) > 1 else None

    def fundamental_value(self, access: int = 1) -> list:
        divs = self.dividends.copy()
        n = len(divs)  # number of iterations
        divs.extend(self.exchange.dividend(access)[1:access])  # add not recorded future divs
        r = self.exchange.risk_free

        return [Fundamentalist.evaluate(divs[i:i+access], r) for i in range(n)]

    def stock_returns(self, roll: int = None) -> List[float] | float:
        p = self.prices
        div = self.dividends
        r = [(p[i+1] - p[i]) / p[i] + div[i] / p[i] for i in range(len(p) - 1)]
        return rolling(r, roll) if roll else mean(r)

    def abnormal_returns(self, roll: int = None) -> List[float]:
        rf = self.exchange.risk_free
        r = [r - rf for r in self.stock_returns()]
        return rolling(r, roll) if roll else r

    def return_volatility(self, window: int = None) -> List[float] | float:
        if window is None:
            return std(self.stock_returns())
        n = len(self.stock_returns(1))
        return [std(self.stock_returns(1)[i:i+window]) for i in range(n - window)]

    def price_volatility(self, window: int = None) -> List[float] | float:
        if window is None:
            return std(self.prices)
        return [std(self.prices[i:i+window]) for i in range(len(self.prices) - window)]

    def liquidity(self, roll: int = None) -> List[float] | float:
        n = len(self.prices)
        spreads = [el['ask'] - el['bid'] for el in self.spreads]
        prices = self.prices
        liq = [spreads[i] / prices[i] for i in range(n)]
        return rolling(liq, roll) if roll else mean(liq)
