"""
AMM Pool implementations for FX market simulation.

- CPMMPool: Constant Product Market Maker (Uniswap-like), invariant x*y = k
- HFMMPool: Hybrid Function Market Maker (Curve StableSwap-like),
  invariant 4A(x+y) + D = 4AD + D^3/(4xy)

Costs are reported in basis points (bps) relative to a benchmark price.
By default, the benchmark is the pool's own marginal mid-price (internal TCA).
An optional external benchmark S_t can still be passed for diagnostics.
"""

import math as _math
from typing import Optional, List, Dict


def _clip_tiny_negative_bps(value: float, eps: float = 1e-9) -> float:
    """Suppress floating-point artifacts close to zero in bps metrics."""
    return 0.0 if -eps < value < 0.0 else value


# ---------------------------------------------------------------------------
# Trade size classification  (θ = Q / R)
# ---------------------------------------------------------------------------

def classify_trade_size(Q: float, pool, mid_price: float) -> dict:
    """
    Classify trade as small / medium / large relative to pool depth.

    θ = Q / R,  where R = effective pool depth in base units.

    Thresholds (consistent with FX & AMM literature):
        small:  θ ≤ 1%    — near-zero slippage on CPMM
        medium: 1% < θ ≤ 5%
        large:  θ > 5%     — material slippage

    Returns dict(theta, size_bucket, R).
    """
    R = pool.effective_depth(mid_price)
    theta = Q / R if R > 0 else float('inf')
    if theta <= 0.01:
        bucket = 'small'
    elif theta <= 0.05:
        bucket = 'medium'
    else:
        bucket = 'large'
    return dict(theta=theta, size_bucket=bucket, R=R)


# ---------------------------------------------------------------------------
# CPMM — Constant Product Market Maker
# ---------------------------------------------------------------------------

class CPMMPool:
    """
    Constant Product Market Maker: x * y = k

    Reserves:
        x  – base currency
        y  – quote currency
    Internal mid-price: y / x
    Pool fee: fraction f of the input amount (e.g. 0.003 = 30 bps).
    """

    def __init__(self, x: float, y: float, fee: float = 0.003,
                 gas_cost_bps: float = 0.0,
                 dynamic_fee: bool = False,
                 fee_floor: Optional[float] = None,
                 fee_cap: Optional[float] = None):
        assert x > 0 and y > 0, "Reserves must be positive"
        self.x = float(x)
        self.y = float(y)
        self.k = self.x * self.y
        # Local second-order rebalancing-loss coefficient around balance.
        # For constant product, loss versus buy-and-hold is sigma^2 / 8.
        self.rebalancing_loss_curvature = 0.125
        self.fee = fee
        self._base_fee = fee
        self.gas_cost_bps = gas_cost_bps

        # Dynamic fee configuration
        self.dynamic_fee = dynamic_fee
        self._fee_floor = fee_floor if fee_floor is not None else fee * 0.2
        self._fee_cap = fee_cap if fee_cap is not None else fee * 5.0

        # Accumulated fee revenue (in quote terms) for LP income tracking
        self.fee_revenue = 0.0
        self.period_fee_revenue = 0.0  # fee revenue in current period
        # Fee income has two jobs that were being done by one number. The
        # counter above is cumulative telemetry and must only ever rise, since
        # a profit and loss statement reads its differences. The two balances
        # below are what is actually owed to the providers and fall to zero
        # when they claim. They are held in the currency the fee was charged
        # in, because a sell is charged in base and a buy in quote, and paying
        # a base fee out as quote hands over money nobody ever put in.
        self.fee_base = 0.0
        self.fee_quote = 0.0
        # The same two amounts for the current period alone. Every reader of a
        # fee figure, the participation signal and the profit and loss
        # statement included, takes them from here and applies one price of its
        # own choosing. The quote valued counter above is struck at the price
        # each trade happened to execute at, which is a different price, and
        # letting one reader use it while another used the balances made the
        # signal and the payment disagree by a third on a large trade.
        self.period_fee_base = 0.0
        self.period_fee_quote = 0.0

        # History
        self.x_history: List[float] = [self.x]
        self.y_history: List[float] = [self.y]
        self.fee_revenue_history: List[float] = [0.0]
        self.fee_history: List[float] = [self.fee]

    # ---- prices ----------------------------------------------------------

    def mid_price(self) -> float:
        """Internal pool mid-price: quote per base."""
        if self.x <= 0:
            return float('inf')
        return self.y / self.x

    # ---- quoting (read-only) ---------------------------------------------

    def quote_buy(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Estimate cost of buying *Q* base currency (trader pays quote).

        If S_t is omitted, slippage is measured versus the pool's local mid.

        Returns dict with:
            exec_price   – effective price paid (quote/base)
            delta_y      – total quote paid by trader
            slippage_bps – pure bonding-curve slippage vs S_t
            fee_bps      – 10 000 * f
            cost_bps     – all-in cost = slippage + fee + gas
        """

        if getattr(self, "closed", False):
            return self._inf_quote()
        if Q <= 0:
            return self._zero_quote(S_t)
        if Q >= self.x * 0.95:
            return self._inf_quote()

        S_ref = S_t if S_t is not None else self.mid_price()

        x_new = self.x - Q
        y_new = self.k / x_new
        dy_eff = y_new - self.y          # quote absorbed by pool (after fee)
        dy = dy_eff / (1.0 - self.fee)   # trader pays this

        # Fee-free execution price (for slippage decomposition)
        p_exec_no_fee = dy_eff / Q
        p_exec = dy / Q

        slippage_bps = 10_000.0 * (p_exec_no_fee - S_ref) / S_ref
        slippage_bps = _clip_tiny_negative_bps(slippage_bps)
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps
        cost_bps = _clip_tiny_negative_bps(cost_bps)

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    def quote_sell(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Estimate cost of selling *Q* base currency (trader receives quote).

        If S_t is omitted, slippage is measured versus the pool's local mid.
        """

        if getattr(self, "closed", False):
            return self._inf_quote()
        if Q <= 0:
            return self._zero_quote(S_t)

        S_ref = S_t if S_t is not None else self.mid_price()

        dx_eff = Q * (1.0 - self.fee)
        x_new = self.x + dx_eff
        y_new = self.k / x_new
        dy = self.y - y_new  # trader receives

        # Fee-free exec price
        x_new_nf = self.x + Q
        y_new_nf = self.k / x_new_nf
        dy_nf = self.y - y_new_nf
        p_exec_no_fee = dy_nf / Q

        p_exec = dy / Q
        slippage_bps = 10_000.0 * (S_ref - p_exec_no_fee) / S_ref
        slippage_bps = _clip_tiny_negative_bps(slippage_bps)
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps
        cost_bps = _clip_tiny_negative_bps(cost_bps)

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    # ---- execution (mutates state) ---------------------------------------

    def execute_buy(self, Q: float) -> dict:
        """Execute buying Q base. Updates reserves & fee revenue."""

        if getattr(self, "closed", False):
            return self._refusal(Q)
        quote = self.quote_buy(Q, self.mid_price())
        if quote['cost_bps'] == float('inf'):
            return self._refusal(Q)

        fee_amount = quote['delta_y'] * self.fee
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount
        self.fee_quote += fee_amount          # withheld from the quote reserve
        self.period_fee_quote += fee_amount

        self.y += quote['delta_y'] * (1.0 - self.fee)
        self.x -= Q
        # k is invariant by construction
        return self._filled(quote, Q)

    def execute_sell(self, Q: float) -> dict:
        """Execute selling Q base. Updates reserves & fee revenue."""

        if getattr(self, "closed", False):
            return self._refusal(Q)
        quote = self.quote_sell(Q, self.mid_price())
        if quote['cost_bps'] == float('inf'):
            return self._refusal(Q)

        # Sell-side fees are charged on the input base amount, so what the
        # pool actually withholds is Q * fee of base. In quote terms that is
        # Q * fee * exec_price, which equals delta_y * fee, and that is the
        # figure the cumulative counter carries so the two sides stay on one
        # basis. The claimable balance, though, has to be the base itself.
        fee_amount = quote['delta_y'] * self.fee
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount
        self.fee_base += Q * self.fee         # withheld from the base reserve
        self.period_fee_base += Q * self.fee

        self.x += Q * (1.0 - self.fee)
        self.y -= quote['delta_y']
        # Update k (may drift slightly due to fees collected)
        self.k = self.x * self.y
        return self._filled(quote, Q)

    # ---- liquidity management --------------------------------------------

    def liquidity_measure(self) -> float:
        """L = sqrt(k) — geometric mean of reserves."""
        return _math.sqrt(self.k)

    def effective_depth(self, mid_price: Optional[float] = None) -> float:
        """
        Effective pool depth in base units:  R = sqrt(x·y / S_t).

        Represents how many base units the pool can absorb before
        significant price impact occurs.

        The quantity has to carry base units. Writing sqrt(x·y)/S instead
        gives x/sqrt(S) on a balanced pool, so the same base liquidity would
        be reported as ten times smaller at a price of a hundred and a
        hundred times smaller at ten thousand. Depth would then depend on the
        currency the quote leg happens to be denominated in, which it must
        not, and routing, size buckets and depth matching would all inherit
        that dependence.
        """
        if getattr(self, "closed", False):
            return 0.0
        S = mid_price if mid_price else self.mid_price()
        return _math.sqrt(self.x * self.y / S) if S > 0 else self.x

    def add_liquidity(self, fraction: float):
        """Add liquidity by scaling reserves up by *fraction*."""
        self._record_flow(fraction)
        self.x *= (1.0 + fraction)
        self.y *= (1.0 + fraction)
        self.k = self.x * self.y

    def remove_liquidity(self, fraction: float):
        """Remove liquidity by scaling reserves down by *fraction*."""
        fraction = min(max(0.0, fraction), 1.0)
        self._record_flow(-fraction)
        self.x *= (1.0 - fraction)
        self.y *= (1.0 - fraction)
        self.x = max(self.x, 1e-6)
        self.y = max(self.y, 1e-6)
        self.k = self.x * self.y

    # ---- arbitrage -------------------------------------------------------

    def arbitrage_to_target(self, S_t: float,
                            max_trade_qty: Optional[float] = None) -> float:
        """
        Trade to align pool mid-price with external price S_t.
        Returns quantity of base traded (positive = bought base from pool).
        Only arbs if deviation exceeds fee band.
        """
        if getattr(self, "closed", False):
            return 0.0
        current = self.mid_price()
        deviation = abs(current - S_t) / S_t
        if deviation < self.fee * 2:
            return 0.0

        # Target reserves: y/x = S_t, x*y = k
        x_target = _math.sqrt(self.k / S_t)
        delta_x = x_target - self.x
        trade_cap = float('inf') if max_trade_qty is None else max(0.0, max_trade_qty)

        if delta_x < 0:
            # Pool has too much x → buy base from pool
            Q = min(abs(delta_x), self.x * 0.5, trade_cap)
            if Q <= 0:
                return 0.0
            self.execute_buy(Q)
            return Q
        else:
            # Pool needs more x → sell base to pool
            # Cap: never drain more than 50 % of quote reserves
            max_sell = self.y * 0.5 / max(S_t, 1e-9)
            Q = min(abs(delta_x), max_sell, trade_cap)
            if Q <= 0:
                return 0.0
            self.execute_sell(Q)
            return -Q

    # ---- volume–slippage profile -----------------------------------------

    def volume_slippage_max_Q(self, threshold_bps: float,
                              S_t: Optional[float] = None,
                              side: str = 'buy',
                              tol: float = 0.01) -> float:
        """
        Maximum Q such that all-in cost ≤ threshold_bps (binary search).
        """
        hi = self.x * 0.99 if side == 'buy' else self.y * 0.99 / self.mid_price()
        lo = 0.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if side == 'buy':
                c = self.quote_buy(mid, S_t)['cost_bps']
            else:
                c = self.quote_sell(mid, S_t)['cost_bps']
            if c <= threshold_bps:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo

    # ---- period management -----------------------------------------------

    def reset_period_fees(self):
        """Reset per-period fee accumulator."""
        self.period_fee_revenue = 0.0

    def update_fee(self, sigma_t: float, sigma_base: float, sigma_prev: Optional[float] = None):
        """Scale fee proportionally to σ_t / σ_base, clamped to [floor, cap].
        Uses lagged σ (sigma_prev) when available to avoid look-ahead bias."""
        if not self.dynamic_fee or sigma_base <= 0:
            return
        sigma_eff = sigma_prev if sigma_prev is not None else sigma_t
        ratio = sigma_eff / sigma_base
        self.fee = max(self._fee_floor,
                       min(self._fee_cap, self._base_fee * ratio))

    # ---- capital flow instrumentation -------------------------------
    # Adding and removing liquidity are capital flows, not investment result.
    # A profit and loss figure that compares terminal reserves against a static
    # benchmark charges the provider for its own withdrawals, so the value of
    # each flow is recorded here and netted out downstream.

    # A pool whose providers have all redeemed holds no capital and must not
    # attract flow. Marking the pool itself, rather than teaching every routing
    # site about the provider population, means quoting, depth and trade size
    # classification all inherit the state without further changes.
    closed = False

    def _refusal(self, Q: float) -> dict:
        """A refused trade, stated so that no caller can settle against it.

        The quote carried an infinite payment and no quantity at all, so a
        caller reading it loosely would move the trader's balances by the
        amount asked for and by an infinite amount of cash. Both figures are
        set to zero here and the refusal is stated explicitly.
        """
        r = self._inf_quote()
        r['delta_y'] = 0.0
        r['executed_qty'] = 0.0
        r['requested_qty'] = Q
        return r

    def _filled(self, quote: dict, Q: float) -> dict:
        """A completed trade, with the quantity that actually went through."""
        quote['executed_qty'] = Q
        quote['requested_qty'] = Q
        return quote

    def claim_fees(self):
        """Hand the accrued fee balances over and reset them.

        Returns the base and quote amounts owed. The cumulative counter is
        left alone, because it is telemetry rather than a balance and every
        reader of its history expects it to rise monotonically.
        """
        b, q = self.fee_base, self.fee_quote
        self.fee_base = 0.0
        self.fee_quote = 0.0
        return b, q

    def take_period_fees(self):
        """The fees of the current period in native amounts, and reset."""
        b, q = self.period_fee_base, self.period_fee_quote
        self.period_fee_base = 0.0
        self.period_fee_quote = 0.0
        return b, q

    def _record_flow(self, fraction: float):
        """Record a capital flow as reserve deltas, not as a value.

        Valuing the flow at the pool mid would bake one price into the record,
        while a profit and loss statement marks positions at the reference
        price. The two differ whenever the pool is off peg, and the gap then
        appears as a fictitious gain or loss. Storing the base and quote
        amounts leaves the choice of price to whoever reads them.
        """
        self.record_reserve_flow(fraction * self.x, fraction * self.y)

    def record_reserve_flow(self, dx: float, dy: float):
        """Record a capital flow given directly as reserve amounts.

        A redemption tilted toward one reserve is not proportional, so it
        cannot be described by a single fraction. Such a path writes the two
        amounts here. Without this the withdrawal leaves no trace in the
        record and a profit and loss statement reads the missing capital as a
        loss the provider never took.
        """
        if not hasattr(self, 'flow_history'):
            self.flow_history = []
            self.flow_dx_history = []
            self.flow_dy_history = []
            self._flow_dx = 0.0
            self._flow_dy = 0.0
        self._flow_dx += dx
        self._flow_dy += dy

    def _take_period_flow(self):
        dx = getattr(self, '_flow_dx', 0.0)
        dy = getattr(self, '_flow_dy', 0.0)
        self._flow_dx = 0.0
        self._flow_dy = 0.0
        return dx, dy

    def record_state(self):
        """Snapshot current reserves & cumulative fees."""
        if not hasattr(self, 'flow_history'):
            self.flow_history = []
            self.flow_dx_history = []
            self.flow_dy_history = []
            self._flow_dx = 0.0
            self._flow_dy = 0.0
        if not hasattr(self, 'fee_base_history'):
            self.fee_base_history = []
            self.fee_quote_history = []
        fb, fq = self.take_period_fees()
        self.fee_base_history.append(fb)
        self.fee_quote_history.append(fq)
        dx, dy = self._take_period_flow()
        self.flow_dx_history.append(dx)
        self.flow_dy_history.append(dy)
        self.flow_history.append(dx * self.mid_price() + dy)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.fee_revenue_history.append(self.fee_revenue)
        self.fee_history.append(self.fee)

    # ---- helpers ---------------------------------------------------------

    def _zero_quote(self, S_t=None):
        p = S_t if S_t else self.mid_price()
        return dict(exec_price=p, delta_y=0, slippage_bps=0,
                    fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), delta_y=float('inf'),
                    slippage_bps=float('inf'),
                    fee_bps=10_000 * self.fee,
                    cost_bps=float('inf'))


# ---------------------------------------------------------------------------
# HFMM — Hybrid Function Market Maker  (StableSwap / Curve-like)
# ---------------------------------------------------------------------------

def _hfmm_get_D(x1: float, x2: float, A: float) -> float:
    """
    Solve for D in the 2-token StableSwap invariant (Newton's method):

        4A(x₁ + x₂) + D = 4AD + D³/(4 x₁ x₂)
    """
    S = x1 + x2
    if S == 0:
        return 0.0
    if x1 <= 0 or x2 <= 0:
        return 0.0

    Ann = 4.0 * A
    D = S  # initial guess

    for _ in range(512):
        P = 4.0 * x1 * x2
        if P < 1e-30:
            return S  # fallback
        D_P = D ** 3 / P
        D_prev = D
        denom = (Ann - 1.0) * D + 3.0 * D_P
        if abs(denom) < 1e-30:
            return D
        D = (Ann * S + 2.0 * D_P) * D / denom
        if D < 0:
            D = D_prev * 0.5  # safeguard
        if abs(D - D_prev) < max(1e-12, abs(D) * 1e-14):
            return D

    return D  # return best estimate instead of raising


def _hfmm_get_y(x_new: float, D: float, A: float) -> float:
    """
    Solve for y given new x, invariant D, and amplification A.

    Quadratic in y:
        4A y² + b y - c = 0
    where b = 4A x + D(1 − 4A),  c = D³/(4x).
    """
    Ann = 4.0 * A
    b = Ann * x_new + D * (1.0 - Ann)
    c = D ** 3 / (4.0 * x_new)

    discriminant = b * b + 4.0 * Ann * c
    if discriminant < 0:
        raise ValueError("_hfmm_get_y: negative discriminant")

    y = (-b + _math.sqrt(discriminant)) / (2.0 * Ann)
    return max(y, 0.0)


def _hfmm_mid_price(x: float, y: float, A: float, D: float) -> float:
    """
    Marginal price (quote per base) on the StableSwap curve:

        price = (4A + D³/(4 x² y)) / (4A + D³/(4 x y²))
    """
    Ann = 4.0 * A
    D3 = D ** 3
    dFdx = Ann + D3 / (4.0 * x * x * y)
    dFdy = Ann + D3 / (4.0 * x * y * y)
    return dFdx / dFdy


class HFMMPool:
    """
    Hybrid Function Market Maker (StableSwap / Curve v1).

    Invariant (2 tokens, operating on *normalised* reserves):
        4A(x_n + y_n) + D = 4A D + D³ / (4 x_n y_n)

    where x_n = x_base × rate,  y_n = y_quote.
    *rate* is the expected equilibrium price (quote per base) and ensures
    the two reserve dimensions are comparable so the StableSwap curve
    behaves properly.

    Parameter A controls curvature:
        A → ∞  ⟹  behaviour ≈ CSMM (constant sum, near parity)
        reserves far from balance ⟹  behaviour ≈ CPMM (constant product)
    """

    def __init__(self, x: float, y: float, A: float = 100.0,
                 fee: float = 0.001, gas_cost_bps: float = 0.0,
                 rate: Optional[float] = None,
                 dynamic_fee: bool = False,
                 fee_floor: Optional[float] = None,
                 fee_cap: Optional[float] = None):
        """
        Parameters
        ----------
        x : float       base reserves
        y : float       quote reserves
        A : float       amplification coefficient
        fee : float     pool fee (fraction)
        rate : float    equilibrium price (quote/base).  If None → y/x.
        dynamic_fee : bool   scale fee with σ_t / σ_base
        fee_floor : float    minimum fee (default: base_fee × 0.2)
        fee_cap : float      maximum fee (default: base_fee × 5.0)
        """
        assert x > 0 and y > 0 and A > 0
        self.x = float(x)          # raw base reserves
        self.y = float(y)          # raw quote reserves
        self.A = A
        # The same local coefficient measured from this invariant.  Keeping it
        # on the curve prevents LP models from carrying an unrelated free
        # parameter for a mechanical property of the venue.
        self.rebalancing_loss_curvature = 0.125 + 0.25 * float(A)
        self.fee = fee
        self._base_fee = fee
        self.gas_cost_bps = gas_cost_bps
        self.rate = rate if rate is not None else self.y / self.x

        # Dynamic fee configuration
        self.dynamic_fee = dynamic_fee
        self._fee_floor = fee_floor if fee_floor is not None else fee * 0.2
        self._fee_cap = fee_cap if fee_cap is not None else fee * 5.0

        # Normalised reserves for the invariant
        self._xn = self.x * self.rate
        self._yn = self.y
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

        self.fee_revenue = 0.0
        self.period_fee_revenue = 0.0
        self.fee_base = 0.0
        self.fee_quote = 0.0
        self.period_fee_base = 0.0
        self.period_fee_quote = 0.0

        self.x_history: List[float] = [self.x]
        self.y_history: List[float] = [self.y]
        self.D_history: List[float] = [self.D]
        self.fee_revenue_history: List[float] = [0.0]
        self.fee_history: List[float] = [self.fee]

    def _sync_norm(self):
        """Re-compute normalised reserves from raw."""
        self._xn = self.x * self.rate
        self._yn = self.y

    # ---- prices ----------------------------------------------------------

    def mid_price(self) -> float:
        """Quote per base, derived from the marginal rate on the curve."""
        if self._xn <= 0 or self._yn <= 0:
            return float('inf')
        mp_norm = _hfmm_mid_price(self._xn, self._yn, self.A, self.D)
        return mp_norm * self.rate

    # ---- quoting ---------------------------------------------------------

    def quote_buy(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Cost of buying Q base (trader pays quote, receives base).

        If S_t is omitted, slippage is measured versus the pool's local mid.
        """

        if getattr(self, "closed", False):
            return self._inf_quote()
        if Q <= 0:
            return self._zero_quote(S_t)
        if Q >= self.x * 0.95:
            return self._inf_quote()

        S_ref = S_t if S_t is not None else self.mid_price()

        # Fee-free execution in normalised space
        xn_new = (self.x - Q) * self.rate
        yn_new = _hfmm_get_y(xn_new, self.D, self.A)
        dy_nf = yn_new - self._yn          # quote to pay (no fee)
        p_exec_no_fee = dy_nf / Q

        # With fee
        dy = dy_nf / (1.0 - self.fee)
        p_exec = dy / Q

        slippage_bps = 10_000.0 * (p_exec_no_fee - S_ref) / S_ref
        slippage_bps = _clip_tiny_negative_bps(slippage_bps)
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps
        cost_bps = _clip_tiny_negative_bps(cost_bps)

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    def quote_sell(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Cost of selling Q base (trader pays base, receives quote).

        If S_t is omitted, slippage is measured versus the pool's local mid.
        """

        if getattr(self, "closed", False):
            return self._inf_quote()
        if Q <= 0:
            return self._zero_quote(S_t)

        S_ref = S_t if S_t is not None else self.mid_price()

        # Fee-free
        xn_new_nf = (self.x + Q) * self.rate
        yn_new_nf = _hfmm_get_y(xn_new_nf, self.D, self.A)
        dy_nf = self._yn - yn_new_nf
        p_exec_no_fee = dy_nf / Q

        # With fee
        xn_new = (self.x + Q * (1.0 - self.fee)) * self.rate
        yn_new = _hfmm_get_y(xn_new, self.D, self.A)
        dy = self._yn - yn_new
        p_exec = dy / Q

        slippage_bps = 10_000.0 * (S_ref - p_exec_no_fee) / S_ref
        slippage_bps = _clip_tiny_negative_bps(slippage_bps)
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps
        cost_bps = _clip_tiny_negative_bps(cost_bps)

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    # ---- execution -------------------------------------------------------

    def execute_buy(self, Q: float) -> dict:

        if getattr(self, "closed", False):
            return self._refusal(Q)
        quote = self.quote_buy(Q, self.mid_price())
        if quote['cost_bps'] == float('inf'):
            return self._refusal(Q)

        fee_amount = quote['delta_y'] * self.fee
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount
        self.fee_quote += fee_amount          # withheld from the quote reserve
        self.period_fee_quote += fee_amount

        self.y += quote['delta_y'] * (1.0 - self.fee)
        self.x -= Q
        self._sync_norm()
        return self._filled(quote, Q)

    def execute_sell(self, Q: float) -> dict:

        if getattr(self, "closed", False):
            return self._refusal(Q)
        quote = self.quote_sell(Q, self.mid_price())
        if quote['cost_bps'] == float('inf'):
            return self._refusal(Q)

        # Sell-side fees are charged on the input base amount, so what the
        # pool actually withholds is Q * fee of base. In quote terms that is
        # Q * fee * exec_price, which equals delta_y * fee, and that is the
        # figure the cumulative counter carries so the two sides stay on one
        # basis. The claimable balance, though, has to be the base itself.
        fee_amount = quote['delta_y'] * self.fee
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount
        self.fee_base += Q * self.fee         # withheld from the base reserve
        self.period_fee_base += Q * self.fee

        self.x += Q * (1.0 - self.fee)
        self.y -= quote['delta_y']
        self._sync_norm()
        return self._filled(quote, Q)

    # ---- liquidity management --------------------------------------------

    def liquidity_measure(self) -> float:
        """D — the StableSwap invariant parameter."""
        return self.D

    def effective_depth(self, mid_price: Optional[float] = None) -> float:
        """
        Effective pool depth in base units, on the same basis as the constant
        product pool:  R = sqrt(x · y / S), from raw reserves and the pool mid.

        The history here is worth recording. The original form used normalised
        reserves, sqrt(x_n · y_n) / rate, and with x_n = x·rate that is exactly
        sqrt(x·y / rate), which is dimensionally right. It was then changed to
        match the constant product pool, whose own formula was the one at
        fault, and the change deflated the measure by sqrt(price) rather than
        correcting an inflation. Both are now written in the correct form.
        """
        if getattr(self, "closed", False):
            return 0.0
        S = mid_price if mid_price else self.mid_price()
        return _math.sqrt(self.x * self.y / S) if S > 0 else self.x

    def add_liquidity(self, fraction: float):
        """Scale reserves up proportionally; recompute D."""
        self._record_flow(fraction)
        self.x *= (1.0 + fraction)
        self.y *= (1.0 + fraction)
        self._sync_norm()
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

    def remove_liquidity(self, fraction: float):
        """Scale reserves down proportionally; recompute D."""
        fraction = min(max(0.0, fraction), 1.0)
        self._record_flow(-fraction)
        self.x *= (1.0 - fraction)
        self.y *= (1.0 - fraction)
        self.x = max(self.x, 1e-6)
        self.y = max(self.y, 1e-6)
        self._sync_norm()
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

    # ---- arbitrage -------------------------------------------------------

    def arbitrage_to_target(self, S_t: float,
                            max_trade_qty: Optional[float] = None) -> float:
        """
        Binary-search for x_target such that mid_price ≈ S_t,
        then execute the implied trade.  Returns signed quantity traded.
        """
        if getattr(self, "closed", False):
            return 0.0
        current = self.mid_price()
        deviation = abs(current - S_t) / S_t
        if deviation < self.fee * 2:
            return 0.0

        # Binary search for x_target (raw base reserves)
        lo = self.x * 0.01
        hi = self.x * 5.0
        D = self.D
        for _ in range(200):
            xm = (lo + hi) / 2.0
            xm_n = xm * self.rate
            ym_n = _hfmm_get_y(xm_n, D, self.A)
            if ym_n <= 0:
                hi = xm
                continue
            mp = _hfmm_mid_price(xm_n, ym_n, self.A, D) * self.rate
            if mp > S_t:
                lo = xm  # need more x to lower price
            else:
                hi = xm
            if abs(mp - S_t) / S_t < 1e-8:
                break

        x_target = (lo + hi) / 2.0
        delta = x_target - self.x
        trade_cap = float('inf') if max_trade_qty is None else max(0.0, max_trade_qty)

        if delta < 0:
            Q = min(abs(delta), self.x * 0.5, trade_cap)
            if Q <= 0:
                return 0.0
            self.execute_buy(Q)
            return Q
        else:
            # Cap: never drain more than 50 % of quote reserves
            max_sell = self.y * 0.5 / max(S_t, 1e-9)
            Q = min(abs(delta), max_sell, trade_cap)
            if Q <= 0:
                return 0.0
            self.execute_sell(Q)
            return -Q

    # ---- rate management -------------------------------------------------

    def update_rate(self, threshold: float = 0.01,
                    target_rate: Optional[float] = None):
        """
        Recenter the StableSwap curve around an external equilibrium rate.

        In real Curve v1 deployments the rate is governance-controlled and
        anchored on an *external* oracle (e.g. an on-chain price feed). It
        is *not* re-pegged to the pool's own drifted mid, because doing so
        would silently legitimise pool drift and eliminate the StableSwap
        amplification benefit relative to the equilibrium price.

        For the FX context, the equilibrium rate is the latent fair price
        ``S_t``. The arbitrageur supplies ``target_rate`` (typically the
        current CLOB mid or env.fair_price). When ``target_rate`` is None,
        the legacy behaviour of pegging to the pool's own mid is preserved
        for backward compatibility, but production code paths should
        always pass ``target_rate``.

        Parameters
        ----------
        threshold : float
            Minimum relative drift |target − rate| / rate to trigger.
        target_rate : float or None
            External equilibrium rate. If None, falls back to the pool mid.
        """
        if target_rate is None:
            target_rate = self.mid_price()
        if target_rate is None or target_rate <= 0:
            return
        if abs(target_rate - self.rate) / self.rate < threshold:
            return
        self.rate = float(target_rate)
        self._sync_norm()
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

    # ---- volume–slippage profile -----------------------------------------

    def volume_slippage_max_Q(self, threshold_bps: float,
                              S_t: Optional[float] = None,
                              side: str = 'buy',
                              tol: float = 0.01) -> float:
        hi = self.x * 0.99 if side == 'buy' else self.y * 0.99 / max(self.mid_price(), 1e-9)
        lo = 0.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if side == 'buy':
                c = self.quote_buy(mid, S_t)['cost_bps']
            else:
                c = self.quote_sell(mid, S_t)['cost_bps']
            if c <= threshold_bps:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo

    # ---- period management -----------------------------------------------

    def reset_period_fees(self):
        self.period_fee_revenue = 0.0

    def update_fee(self, sigma_t: float, sigma_base: float, sigma_prev: Optional[float] = None):
        """Scale fee proportionally to σ_t / σ_base, clamped to [floor, cap].
        Uses lagged σ (sigma_prev) when available to avoid look-ahead bias."""
        if not self.dynamic_fee or sigma_base <= 0:
            return
        sigma_eff = sigma_prev if sigma_prev is not None else sigma_t
        ratio = sigma_eff / sigma_base
        self.fee = max(self._fee_floor,
                       min(self._fee_cap, self._base_fee * ratio))

    # ---- capital flow instrumentation -------------------------------
    # Adding and removing liquidity are capital flows, not investment result.
    # A profit and loss figure that compares terminal reserves against a static
    # benchmark charges the provider for its own withdrawals, so the value of
    # each flow is recorded here and netted out downstream.

    # A pool whose providers have all redeemed holds no capital and must not
    # attract flow. Marking the pool itself, rather than teaching every routing
    # site about the provider population, means quoting, depth and trade size
    # classification all inherit the state without further changes.
    closed = False

    def _refusal(self, Q: float) -> dict:
        """A refused trade, stated so that no caller can settle against it.

        The quote carried an infinite payment and no quantity at all, so a
        caller reading it loosely would move the trader's balances by the
        amount asked for and by an infinite amount of cash. Both figures are
        set to zero here and the refusal is stated explicitly.
        """
        r = self._inf_quote()
        r['delta_y'] = 0.0
        r['executed_qty'] = 0.0
        r['requested_qty'] = Q
        return r

    def _filled(self, quote: dict, Q: float) -> dict:
        """A completed trade, with the quantity that actually went through."""
        quote['executed_qty'] = Q
        quote['requested_qty'] = Q
        return quote

    def claim_fees(self):
        """Hand the accrued fee balances over and reset them.

        Returns the base and quote amounts owed. The cumulative counter is
        left alone, because it is telemetry rather than a balance and every
        reader of its history expects it to rise monotonically.
        """
        b, q = self.fee_base, self.fee_quote
        self.fee_base = 0.0
        self.fee_quote = 0.0
        return b, q

    def take_period_fees(self):
        """The fees of the current period in native amounts, and reset."""
        b, q = self.period_fee_base, self.period_fee_quote
        self.period_fee_base = 0.0
        self.period_fee_quote = 0.0
        return b, q

    def _record_flow(self, fraction: float):
        """Record a capital flow as reserve deltas, not as a value.

        Valuing the flow at the pool mid would bake one price into the record,
        while a profit and loss statement marks positions at the reference
        price. The two differ whenever the pool is off peg, and the gap then
        appears as a fictitious gain or loss. Storing the base and quote
        amounts leaves the choice of price to whoever reads them.
        """
        self.record_reserve_flow(fraction * self.x, fraction * self.y)

    def record_reserve_flow(self, dx: float, dy: float):
        """Record a capital flow given directly as reserve amounts.

        A redemption tilted toward one reserve is not proportional, so it
        cannot be described by a single fraction. Such a path writes the two
        amounts here. Without this the withdrawal leaves no trace in the
        record and a profit and loss statement reads the missing capital as a
        loss the provider never took.
        """
        if not hasattr(self, 'flow_history'):
            self.flow_history = []
            self.flow_dx_history = []
            self.flow_dy_history = []
            self._flow_dx = 0.0
            self._flow_dy = 0.0
        self._flow_dx += dx
        self._flow_dy += dy

    def _take_period_flow(self):
        dx = getattr(self, '_flow_dx', 0.0)
        dy = getattr(self, '_flow_dy', 0.0)
        self._flow_dx = 0.0
        self._flow_dy = 0.0
        return dx, dy

    def record_state(self):
        if not hasattr(self, 'flow_history'):
            self.flow_history = []
            self.flow_dx_history = []
            self.flow_dy_history = []
            self._flow_dx = 0.0
            self._flow_dy = 0.0
        if not hasattr(self, 'fee_base_history'):
            self.fee_base_history = []
            self.fee_quote_history = []
        fb, fq = self.take_period_fees()
        self.fee_base_history.append(fb)
        self.fee_quote_history.append(fq)
        dx, dy = self._take_period_flow()
        self.flow_dx_history.append(dx)
        self.flow_dy_history.append(dy)
        self.flow_history.append(dx * self.mid_price() + dy)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.D_history.append(self.D)
        self.fee_revenue_history.append(self.fee_revenue)
        self.fee_history.append(self.fee)

    # ---- helpers ---------------------------------------------------------

    def _zero_quote(self, S_t=None):
        p = S_t if S_t else self.mid_price()
        return dict(exec_price=p, delta_y=0, slippage_bps=0,
                    fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), delta_y=float('inf'),
                    slippage_bps=float('inf'),
                    fee_bps=10_000 * self.fee,
                    cost_bps=float('inf'))
