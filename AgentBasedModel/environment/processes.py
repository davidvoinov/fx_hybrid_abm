"""
Exogenous market environment: volatility σ_t, funding cost c_t,
and fair price S_t.

Supports piecewise-constant regimes (normal → stress → normal)
or a simple mean-reverting stochastic process for σ, c.

S_t follows a mean-reverting latent-anchor process so that there is
always an authoritative fair price available — even when the CLOB order
book is thin or absent (AMM-only scenarios) — without producing the
hyper-drifting post-shock paths of a pure GBM.
"""

import random
import math
from typing import Optional, List


# Longest stress an episode may declare, as a per tick multiplier: a half
# life of about twenty three minutes, which outlasts any run in the design.
# The previous ceiling of 0.99 was a half life of sixty nine seconds, and it
# was applied silently, so the two episodes documented as lasting weeks
# declared 0.997 and ran at a third of the duration they stated.
_MAX_STRESS_DECAY: float = 0.9995


def _decay_per_second(half_life_seconds: float) -> float:
    """Per-tick multiplier implied by a half-life stated in seconds.

    One tick is one second. Stating the half-life and deriving the
    multiplier keeps the duration visible in the parameter and not
    buried in a decimal that has to be inverted to be understood.
    """
    half_life = max(1e-9, float(half_life_seconds))
    return float(0.5 ** (1.0 / half_life))


class MarketEnvironment:
    """
    Manages exogenous volatility (σ_t), funding liquidity cost (c_t),
    and fair price S_t.

    Two modes for σ / c:
    1. **Piecewise-constant** (default): σ and c jump between predefined
       normal / stress levels at specified times.
    2. **Stochastic**: Ornstein–Uhlenbeck around a base level, with
       optional regime-shift overlay.

    Fair price S_t
    --------------
    Always present.  Follows a mean-reverting process in log-price:

        log S_{t+1} = log S_t
            + μ
            + κ (log S* - log S_t)
            + λ σ_t ε

    where S* is a latent anchor price.  Price shocks move both S_t and
    S* downward (or upward), but only partially, so the process can
    recover in a controlled way instead of exploding under repeated
    high-σ draws.

    These parameters feed into:
      - CLOB market-maker spread:  qspr_t = α₀ + α₁ σ_t + α₂ c_t
      - AMM LP liquidity rule:     L_{t+1} = L_t + φ₁ Π^fee − φ₂ σ − φ₃ c
      - Arbitrageur target:        arb → S_t
    """

    def __init__(self,
                 sigma_low: float = 0.01,
                 sigma_high: float = 0.05,
                 c_low: float = 0.001,
                 c_high: float = 0.02,
                 stress_start: Optional[int] = None,
                 stress_end: Optional[int] = None,
                 mode: str = 'piecewise',
                 # ── exogenous fair price ──────────────────────
                 price: Optional[float] = None,
                 drift: float = 0.0,
                 price_reversion: float = 0.12,
                 price_vol_scale: float = 0.35,
                 funding_rate_scale: float = 1.3319e-9,
                 rng: Optional['random.Random'] = None,
                 shock_anchor_weight: float = 0.75,
                 shock_price_freeze_ticks: int = 1,
                 # session_cycle <= 0 disables the intraday cycle. The
                 # paper's primary scope (calibration manifest) by design
                 # excludes Asia/London/NY decomposition, so the default for
                 # FX runs is now disabled; a positive value re-enables a
                 # 4-session round-robin for studies that explicitly want it.
                 session_cycle: int = 0):
        """
        Parameters
        ----------
        sigma_low, sigma_high : float
            Volatility in normal / stress regimes.
        c_low, c_high : float
            Funding cost in normal / stress regimes.
        stress_start, stress_end : int or None
            Iteration range [start, end) during which the market is stressed.
            If None the market stays in normal regime.
        mode : str
            'piecewise' for regime-switch, 'stochastic' for OU-based.
        price : float or None
            Initial fair price S_0.  If None the fair price is not tracked
            (backward-compatible with legacy callers).
        drift : float
            Drift μ of the latent fair-price process.
        price_reversion : float
            Mean-reversion speed κ toward the latent anchor price.
        price_vol_scale : float
            Scale λ mapping σ_t into fair-price innovations.
        shock_anchor_weight : float
            Fraction of a shock that permanently shifts the latent anchor.
        shock_price_freeze_ticks : int
            Number of periods to freeze S_t immediately after a shock so
            the logged shock level is not contaminated by the same-tick
            stochastic price step.
        """
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.c_low = c_low
        self.c_high = c_high
        self.stress_start = stress_start
        self.stress_end = stress_end
        self.mode = mode

        # Current values
        self._sigma = sigma_low
        self._c = c_low
        self._t = 0

        # ── Shock-induced stress overlay ─────────────────────────────
        # When a price shock hits, σ and c spike and then decay
        # exponentially back to the regime baseline.
        self._shock_sigma_overlay: float = 0.0
        self._shock_c_overlay: float = 0.0
        # Stress durations are stated in seconds, because one tick is one
        # second and a per-tick multiplier hides the duration it implies.
        # These were set when a tick had no defined length and were never
        # restated when it acquired one: 0.93 per tick is a half-life of ten
        # seconds, and the dysfunction state below lasted under twenty. A
        # scenario named after a dealer liquidity crisis was therefore over
        # in the time it takes to read its name, which is why the spread
        # response is a spike with no decay to measure.
        self.stress_overlay_half_life_seconds: float = 10.0
        self._shock_decay: float = _decay_per_second(
            self.stress_overlay_half_life_seconds
        )
        # An episode declares how long its stress lasts, and both limbs have
        # to hear it. The liquidity limb was given a duration and the
        # volatility limb was not, so a scenario documented as lasting weeks
        # decayed its volatility on an eleven second clock: whichever shock
        # entry point fired last reset this to its own argument default.
        # Held separately so an episode setting survives that call.
        self._stress_overlay_decay: Optional[float] = None

        # ── Shock liquidity-crisis state ────────────────────────────────
        # Managed by apply_shock(), decremented by step().
        self.shock_ticks_remaining: int = 0       # >0 means shock aftermath
        self.mm_pause_ticks: int = 0              # MM doesn't quote
        self.cancel_wave_frac: float = 0.0        # fraction of book to cancel
        self.reprice_prob_override: Optional[float] = None  # reduced repricing
        self.anchor_strength_override: Optional[float] = None
        self.background_target_ratio_override: Optional[float] = None
        self.toxic_flow_bias: float = 0.0         # directional bias for takers
        self._order_flow_imbalance: float = 0.0   # signed taker flow pressure
        self._logged_order_flow_imbalance: float = 0.0
        self._clob_order_flow_imbalance: float = 0.0
        self._logged_clob_order_flow_imbalance: float = 0.0
        self._liquidity_shock: float = 0.0        # systemic liquidity damage

        # ── Dealer capacity cascade ─────────────────────────────────
        # Systemic liquidity was entirely exogenous: it responded to
        # volatility, funding and the exogenous liquidity shock, but never to
        # what the dealer sector had actually done. A shock could therefore
        # empty the book without the emptiness itself making conditions any
        # worse, so the model had no channel through which one dealer leaving
        # pushes the next one closer to leaving.
        #
        # The link runs through dealer capacity, following the finding in BIS
        # Working Paper 1138 that limited intermediation capacity worsens
        # illiquidity only at high levels of balance sheet utilisation. Below
        # the threshold the sector absorbs the loss of a dealer and nothing
        # propagates; above it the same marginal loss bites, and it bites
        # convexly. Capacity is read from the depth each dealer actually
        # supplies relative to its own unimpaired quote, which counts a dealer
        # quoting defensively at a fraction of its size as the fraction it
        # supplies, and counts one that has withdrawn as zero. Headcount and
        # depth are therefore the same measurement here and are not added
        # twice.
        self._dealer_capacity_impairment: float = 0.0
        self.dealer_capacity_history: List[float] = []
        self._dealer_capacity_threshold: float = 0.5
        self._dealer_cascade_gain: float = 0.0
        self.liquidity_half_life_seconds: float = 6.6
        self._liquidity_shock_decay: float = _decay_per_second(
            self.liquidity_half_life_seconds
        )
        # How long the dysfunction state itself lasts, in seconds, before the
        # book is treated as no longer in the aftermath of the shock.
        self.stress_duration_seconds: float = 24.0

        # How long the suppressed book states take to climb back, in
        # seconds. These were per-tick increments, and they are what
        # actually ended the crisis: repricing climbed from its suppressed
        # level to normal in about eight ticks whatever the dysfunction
        # state said, so lengthening that state changed nothing. The
        # duration is stated here and the per-tick step derived from it.
        self.state_recovery_seconds: float = 8.0
        self._reprice_prob_recovery: float = 0.6 / self.state_recovery_seconds
        self._anchor_strength_recovery: float = 0.25 / self.state_recovery_seconds
        self._bg_target_ratio_recovery: float = 1.0 / self.state_recovery_seconds
        self._toxic_flow_decay: float = 0.85            # multiplicative per tick
        self._systemic_liquidity: float = 1.0
        self._recovery_support: float = 1.0
        self._amm_order_flow_imbalance: float = 0.0
        self._logged_amm_order_flow_imbalance: float = 0.0
        self._amm_slippage_signal: float = 0.0
        self._amm_reserve_imbalance: float = 0.0
        self._venue_basis_bps: float = 0.0
        self._pool_dispersion_bps: float = 0.0
        self._arbitrage_capacity: float = 0.0

        # ── Fair price (latent-anchor mean reversion) ───────────────
        self._price: Optional[float] = float(price) if price is not None else None
        self._price_anchor: Optional[float] = float(price) if price is not None else None
        self._drift = drift
        # A negative value here is a calibration that the model cannot carry,
        # and clamping it to zero silently reported a cost of capital the run
        # never used. It matters for a pair whose policy rates are negative,
        # where the honest reading is a floor of zero and not a quiet one.
        for _name, _value in (('price_reversion', price_reversion),
                              ('price_vol_scale', price_vol_scale),
                              ('funding_rate_scale', funding_rate_scale)):
            if _value < 0.0:
                raise ValueError(
                    f'{_name} is {_value!r}. The model floors it at zero, so a '
                    'negative calibration would be silently discarded. Set it '
                    'to zero and record why in the manifest.')
        self._price_reversion = float(price_reversion)
        self._price_vol_scale = float(price_vol_scale)
        self._funding_rate_scale = float(funding_rate_scale)

        # The exogenous state is drawn from its own generator. Sharing the
        # global one made the price path depend on how many numbers the rest
        # of the model happened to consume, so a market with the facility and
        # a market without it walked different paths from the second tick even
        # on the same seed. Every paired comparison then carried the treatment
        # effect and a difference of realisations together. The caller passes
        # a generator seeded from the run seed, and the two arms therefore see
        # the same volatility, funding and price path by construction.
        self._rng = rng if rng is not None else random
        self._shock_anchor_weight = max(0.0, min(1.0, shock_anchor_weight))
        self._shock_price_freeze_default = max(0, int(shock_price_freeze_ticks))
        self._price_freeze_ticks = 0
        self._session_cycle = int(session_cycle)
        # When the cycle is disabled (<=0) all session multipliers are 1.0
        # and the session name is 'AllDay'. This avoids implicit intraday
        # heterogeneity in the FX baseline that the paper does not document.
        if self._session_cycle <= 0:
            self._session_name = 'AllDay'
            self._session_flow_multiplier = 1.0
            self._session_liquidity_multiplier = 1.0
            self._session_vol_multiplier = 1.0
        else:
            self._session_name = 'Asia'
            self._session_flow_multiplier = 0.85
            self._session_liquidity_multiplier = 0.90
            self._session_vol_multiplier = 0.92

        # History
        self.sigma_history: List[float] = []
        self.c_history: List[float] = []
        self.regime_history: List[str] = []
        self.session_history: List[str] = []
        self.price_history: List[float] = []  # S_t series
        self.flow_imbalance_history: List[float] = []
        self.systemic_liquidity_history: List[float] = []

    # ---- public API ------------------------------------------------------

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def price_sigma(self) -> float:
        """Standard deviation of the log price step, per tick.

        ``sigma`` is a dimensionless stress index. It drives how far quoting
        agents stand off, how dealers withdraw and how funding tightens, and it
        is calibrated in those roles. It is not the volatility of the price,
        which is this number. Quoting rules that charged for adverse selection
        in units of ``sigma`` were charging for a risk whose size they had no
        way of knowing, and the two moved together only because the price
        volatility happened to be a fixed multiple of the index.
        """
        return self._price_vol_scale * self._sigma

    @property
    def price_sigma_bps(self) -> float:
        """The same quantity in basis points, which is how quotes are set."""
        return 1e4 * self.price_sigma

    @property
    def funding_rate(self) -> float:
        """Cost of the capital committed, as a rate per tick.

        ``funding_cost`` is a stress index in the same sense as ``sigma``. It
        drives how far dealers widen and how hard balance sheets bind, and it
        is calibrated in those roles, but it is not a rate and carries no time
        scale. A provider deciding whether to keep capital in a pool compares
        what the pool earns over a period against what the capital costs over
        the same period, so it needs this instead. The level is anchored to an
        overnight money market rate of about three per cent a year, split
        across the two currencies a pair is quoted in, expressed per tick and
        scaled by the stress index so that a funding squeeze still raises it.
        """
        base = max(self.c_low, 1e-12)
        return self._funding_rate_scale * (self._c / base)

    @property
    def funding_cost(self) -> float:
        return self._c

    @property
    def fair_price(self) -> Optional[float]:
        """Exogenous fair price S_t (None if not configured)."""
        return self._price

    @property
    def order_flow_imbalance(self) -> float:
        """EWMA of signed taker flow, positive for buy pressure."""
        return self._order_flow_imbalance

    @property
    def logged_order_flow_imbalance(self) -> float:
        """Raw period signed flow imbalance used for diagnostics."""
        return self._logged_order_flow_imbalance

    @property
    def clob_order_flow_imbalance(self) -> float:
        """EWMA of signed CLOB taker flow, positive for buy pressure."""
        return self._clob_order_flow_imbalance

    @property
    def logged_clob_order_flow_imbalance(self) -> float:
        """Raw period signed CLOB flow imbalance used for diagnostics."""
        return self._logged_clob_order_flow_imbalance

    @property
    def amm_order_flow_imbalance(self) -> float:
        """EWMA of signed AMM taker flow, positive for buy pressure."""
        return self._amm_order_flow_imbalance

    @property
    def logged_amm_order_flow_imbalance(self) -> float:
        """Raw period AMM signed flow imbalance used for diagnostics."""
        return self._logged_amm_order_flow_imbalance

    @property
    def systemic_liquidity(self) -> float:
        """Common liquidity factor shared across venues in [0.2, 1.0]."""
        return self._systemic_liquidity

    @property
    def liquidity_shock(self) -> float:
        """Residual liquidity-crisis intensity carried across the shock aftermath."""
        return self._liquidity_shock

    @property
    def amm_slippage_signal(self) -> float:
        return self._amm_slippage_signal

    @property
    def amm_reserve_imbalance(self) -> float:
        return self._amm_reserve_imbalance

    @property
    def venue_basis_bps(self) -> float:
        return self._venue_basis_bps

    @property
    def pool_dispersion_bps(self) -> float:
        return self._pool_dispersion_bps

    @property
    def arbitrage_capacity(self) -> float:
        return self._arbitrage_capacity

    @property
    def session_name(self) -> str:
        return self._session_name

    @property
    def session_flow_multiplier(self) -> float:
        return self._session_flow_multiplier

    @property
    def session_liquidity_multiplier(self) -> float:
        return self._session_liquidity_multiplier

    @property
    def session_vol_multiplier(self) -> float:
        return self._session_vol_multiplier

    def _stress_seconds(self, intensity: float, base_share: float = 0.0,
                        slope_share: float = 1.0,
                        floor_share: float = 0.5) -> int:
        """Periods the dysfunction state lasts, scaled by shock intensity.

        Stated as a duration and not as an arithmetic expression in
        ticks, so that the length of a modelled crisis is a declared input
        and can be set against the episode it claims to represent.

        The three entry points do not agree on how a shock of a given
        intensity maps to a duration, and collapsing them onto one shape
        silently shortened two of them: the funding path fell from twenty
        six periods to nineteen at the calibrated intensity and the
        cancellation path from twenty two to fourteen. Each keeps its own
        floor and slope as fractions of the declared duration, so the
        translation into seconds is exact and not approximate.
        """
        scale = max(0.0, float(intensity))
        duration = self.stress_duration_seconds
        affine = (base_share + slope_share * scale) * duration
        return int(max(floor_share * duration, affine))

    def is_stress(self) -> bool:
        if self.stress_start is None:
            return False
        if self.stress_end is None:
            return self._t >= self.stress_start
        return self.stress_start <= self._t < self.stress_end

    def step(self):
        """Advance to next period and update σ_t, c_t, S_t."""
        self._t += 1

        recovery_support = max(0.75, min(1.75, self._recovery_support))
        reprice_recovery = self._reprice_prob_recovery * recovery_support
        anchor_recovery = self._anchor_strength_recovery * recovery_support
        bg_ratio_recovery = self._bg_target_ratio_recovery * recovery_support
        toxic_flow_decay = self._toxic_flow_decay ** recovery_support
        liquidity_shock_decay = self._liquidity_shock_decay ** recovery_support

        if self.mode == 'piecewise':
            self._step_piecewise()
        else:
            self._step_stochastic()

        self._update_session_state()
        self._sigma *= self._session_vol_multiplier
        self._c *= 1.0 + 0.20 * (self._session_vol_multiplier - 1.0)

        # Add shock-induced stress overlay (decays each step)
        if self._shock_sigma_overlay > 1e-8:
            self._sigma += self._shock_sigma_overlay
            self._shock_sigma_overlay *= self._shock_decay
        if self._shock_c_overlay > 1e-8:
            self._c += self._shock_c_overlay
            self._shock_c_overlay *= self._shock_decay

        # Decay shock liquidity-crisis state
        if self.shock_ticks_remaining > 0:
            self.shock_ticks_remaining -= 1
            if self.mm_pause_ticks > 0:
                self.mm_pause_ticks -= 1
            # Gradually restore reprice probability
            if self.reprice_prob_override is not None:
                self.reprice_prob_override = min(
                    0.6, self.reprice_prob_override + reprice_recovery)
                if self.reprice_prob_override >= 0.6:
                    self.reprice_prob_override = None
            if self.anchor_strength_override is not None:
                self.anchor_strength_override = min(
                    0.25, self.anchor_strength_override + anchor_recovery)
                if self.anchor_strength_override >= 0.25:
                    self.anchor_strength_override = None
            if self.background_target_ratio_override is not None:
                self.background_target_ratio_override = min(
                    1.0, self.background_target_ratio_override + bg_ratio_recovery)
                if self.background_target_ratio_override >= 0.95:
                    self.background_target_ratio_override = None
        else:
            self.reprice_prob_override = None
            self.anchor_strength_override = None
            self.background_target_ratio_override = None

        # Toxic flow decays on the clock its episode declares, outside the
        # dysfunction counter. Held inside it, the limb was cleared outright
        # when that counter reached zero at about thirty seconds, so an
        # episode declaring a half life of one hundred and seventy three
        # seconds ran this channel for thirty whatever it declared, while the
        # liquidity and volatility limbs beside it ran for hundreds.
        if abs(self.toxic_flow_bias) > 0.01:
            self.toxic_flow_bias *= toxic_flow_decay
        else:
            self.toxic_flow_bias = 0.0

        if abs(self._order_flow_imbalance) > 0.01:
            self._order_flow_imbalance *= 0.90
        else:
            self._order_flow_imbalance = 0.0

        if abs(self._clob_order_flow_imbalance) > 0.01:
            self._clob_order_flow_imbalance *= 0.90
        else:
            self._clob_order_flow_imbalance = 0.0

        if abs(self._amm_order_flow_imbalance) > 0.01:
            self._amm_order_flow_imbalance *= 0.90
        else:
            self._amm_order_flow_imbalance = 0.0

        if self._liquidity_shock > 1e-8:
            self._liquidity_shock *= liquidity_shock_decay

        # Fair-price step
        if self._price is not None:
            if self._price_freeze_ticks > 0:
                self._price_freeze_ticks -= 1
            else:
                self._step_price()

        self._update_systemic_liquidity()

        self.sigma_history.append(self._sigma)
        self.c_history.append(self._c)
        self.regime_history.append('stress' if self.is_stress() else 'normal')
        self.session_history.append(self._session_name)
        if self._price is not None:
            self.price_history.append(self._price)
        self.flow_imbalance_history.append(self._order_flow_imbalance)
        self.systemic_liquidity_history.append(self._systemic_liquidity)
        self.dealer_capacity_history.append(self._dealer_capacity_impairment)

    def apply_shock(self, pct: float):
        """Apply an instantaneous shock to the fair price.

        Also triggers an endogenous stress response: σ and c spike
        proportionally to the shock magnitude and then decay
        exponentially (half-life ≈ 10 steps).

        Parameters
        ----------
        pct : float
            Percentage change (e.g. -20 for a 20 % crash).
        """
        multiplier = 1.0 + pct / 100.0
        if self._price is not None:
            self._price *= multiplier
            self._price = max(self._price, 1e-6)
            self._price_freeze_ticks = self._shock_price_freeze_default
        if self._price_anchor is not None:
            anchor_multiplier = 1.0 + self._shock_anchor_weight * (multiplier - 1.0)
            self._price_anchor *= max(anchor_multiplier, 1e-6)
            self._price_anchor = max(self._price_anchor, 1e-6)

        # ── Endogenous stress response ───────────────────────────────
        intensity = min(abs(pct) / 20.0, 2.0)
        self._shock_sigma_overlay = (self.sigma_high - self.sigma_low) * intensity
        self._shock_c_overlay = (self.c_high - self.c_low) * intensity

        # ── Liquidity crisis state ──────────────────────────────────
        self.shock_ticks_remaining = self._stress_seconds(intensity)
        self.mm_pause_ticks = max(2, int(6 * intensity))
        self.cancel_wave_frac = min(0.8, 0.15 + 0.35 * intensity)
        self.reprice_prob_override = max(0.05, 0.25 - 0.08 * intensity)
        self.anchor_strength_override = max(0.10, 0.22 - 0.05 * intensity)
        self.background_target_ratio_override = max(0.45, 0.70 - 0.12 * intensity)
        # Toxic flow: takers skew toward shock direction
        self.toxic_flow_bias = -0.3 * (1.0 if pct < 0 else -1.0) * intensity
        self._liquidity_shock = max(self._liquidity_shock, 0.25 + 0.20 * intensity)
        self._update_systemic_liquidity()

    def apply_fundamental_shock(self, pct: float,
                                anchor_weight: float = 1.0,
                                freeze_ticks: Optional[int] = None):
        """Move the latent fair value without mechanically repricing venues."""
        if pct == 0:
            return

        multiplier = max(1e-6, 1.0 + pct / 100.0)
        if self._price is not None:
            self._price *= multiplier
            self._price = max(self._price, 1e-6)
            self._price_freeze_ticks = (
                self._shock_price_freeze_default
                if freeze_ticks is None else max(0, int(freeze_ticks))
            )
        if self._price_anchor is not None:
            eff_weight = max(0.0, min(1.0, anchor_weight))
            anchor_multiplier = 1.0 + eff_weight * (multiplier - 1.0)
            self._price_anchor *= max(anchor_multiplier, 1e-6)
            self._price_anchor = max(self._price_anchor, 1e-6)

    def configure_stress_overlay_decay(self, decay: Optional[float]) -> None:
        """Set how fast the volatility and funding cost overlay decays.

        Stated per tick, to match the liquidity limb, so that the two limbs
        of one episode can be given one duration.
        """
        if decay is None:
            return
        self._stress_overlay_decay = max(0.80, min(_MAX_STRESS_DECAY, float(decay)))
        self._shock_decay = self._stress_overlay_decay

    def apply_funding_volatility_shock(self, intensity: float,
                                       decay: float = 0.94):
        """Spike sigma and funding cost with gradual decay."""
        intensity = max(0.0, float(intensity))
        if intensity <= 0:
            return

        if self._stress_overlay_decay is not None:
            decay = self._stress_overlay_decay
        self._shock_decay = max(0.80, min(_MAX_STRESS_DECAY, float(decay)))
        sigma_jump = (self.sigma_high - self.sigma_low) * intensity
        c_jump = (self.c_high - self.c_low) * intensity
        self._shock_sigma_overlay = max(self._shock_sigma_overlay, sigma_jump)
        self._shock_c_overlay = max(self._shock_c_overlay, c_jump)
        # 12 + 18*intensity in tick form, as shares of the declared duration
        # of twenty four seconds.
        self.shock_ticks_remaining = max(
            self.shock_ticks_remaining,
            self._stress_seconds(intensity, base_share=0.5,
                                 slope_share=0.75, floor_share=0.0),
        )
        self._update_systemic_liquidity()

    def apply_liquidity_shock(self, cancel_frac: float,
                              direction: float = 0.0,
                              intensity: Optional[float] = None,
                              force_mm_pause: bool = False,
                              forced_pause_ticks: Optional[int] = None,
                              reprice_prob: float = 0.05,
                              anchor_strength: float = 0.03,
                              background_target_ratio: float = 0.35,
                              reprice_prob_recovery: Optional[float] = None,
                              anchor_strength_recovery: Optional[float] = None,
                              bg_target_ratio_recovery: Optional[float] = None,
                              toxic_flow_decay: Optional[float] = None,
                              liquidity_shock_decay: Optional[float] = None,
                              stress_overlay_decay: Optional[float] = None):
        """Trigger cancellations, MM withdrawal, stale quotes and toxic flow."""
        cancel_frac = max(0.0, min(1.0, float(cancel_frac)))
        intensity = cancel_frac if intensity is None else max(0.0, float(intensity))
        if cancel_frac <= 0 and intensity <= 0:
            return

        # 10 + 20*intensity in tick form.
        self.shock_ticks_remaining = max(
            self.shock_ticks_remaining,
            self._stress_seconds(intensity, base_share=10.0 / 24.0,
                                 slope_share=20.0 / 24.0, floor_share=0.0),
        )
        if force_mm_pause:
            pause = (2 + int(8 * intensity) if forced_pause_ticks is None
                     else max(0, int(forced_pause_ticks)))
            self.mm_pause_ticks = max(self.mm_pause_ticks, pause)
        self.cancel_wave_frac = max(self.cancel_wave_frac, cancel_frac)
        self.reprice_prob_override = (
            max(0.01, min(reprice_prob, self.reprice_prob_override))
            if self.reprice_prob_override is not None else max(0.01, reprice_prob)
        )
        self.anchor_strength_override = (
            max(0.0, min(anchor_strength, self.anchor_strength_override))
            if self.anchor_strength_override is not None else max(0.0, anchor_strength)
        )
        self.background_target_ratio_override = min(
            self.background_target_ratio_override if self.background_target_ratio_override is not None else 1.0,
            max(0.10, background_target_ratio),
        )
        signed_direction = 0.0
        if direction > 0:
            signed_direction = 1.0
        elif direction < 0:
            signed_direction = -1.0
        self.toxic_flow_bias = 0.30 * signed_direction * max(0.3, intensity)
        self._liquidity_shock = max(self._liquidity_shock, 0.20 + 0.35 * intensity)
        # Apply custom decay rates if provided
        if reprice_prob_recovery is not None:
            self._reprice_prob_recovery = max(0.001, float(reprice_prob_recovery))
        if anchor_strength_recovery is not None:
            self._anchor_strength_recovery = max(0.001, float(anchor_strength_recovery))
        if bg_target_ratio_recovery is not None:
            self._bg_target_ratio_recovery = max(0.001, float(bg_target_ratio_recovery))
        if toxic_flow_decay is not None:
            self._toxic_flow_decay = max(0.50, min(_MAX_STRESS_DECAY, float(toxic_flow_decay)))
        if liquidity_shock_decay is not None:
            self._liquidity_shock_decay = max(
                0.50, min(_MAX_STRESS_DECAY, float(liquidity_shock_decay))
            )
        if stress_overlay_decay is not None:
            self.configure_stress_overlay_decay(stress_overlay_decay)
        self._update_systemic_liquidity()

    def observe_order_flow(self, period_trades: List[dict]):
        """Update common liquidity state from realised taker order flow.

        The model previously reacted mostly to fair-price moves.  This
        method injects a common liquidity channel driven by signed flow,
        so dealer constraints and AMM liquidity can comove for the right
        reason.
        """
        buy_vol = sum(
            tr.get('quantity', 0.0)
            for tr in period_trades
            if tr.get('side') == 'buy'
        )
        sell_vol = sum(
            tr.get('quantity', 0.0)
            for tr in period_trades
            if tr.get('side') == 'sell'
        )
        gross = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / gross if gross > 0 else 0.0
        self._logged_order_flow_imbalance = imbalance
        self._order_flow_imbalance = 0.65 * self._order_flow_imbalance + 0.35 * imbalance

        clob_buy_vol = sum(
            tr.get('quantity', 0.0)
            for tr in period_trades
            if tr.get('side') == 'buy' and tr.get('venue') == 'clob'
        )
        clob_sell_vol = sum(
            tr.get('quantity', 0.0)
            for tr in period_trades
            if tr.get('side') == 'sell' and tr.get('venue') == 'clob'
        )
        clob_gross = clob_buy_vol + clob_sell_vol
        clob_imbalance = (clob_buy_vol - clob_sell_vol) / clob_gross if clob_gross > 0 else 0.0
        self._logged_clob_order_flow_imbalance = clob_imbalance
        self._clob_order_flow_imbalance = (
            0.65 * self._clob_order_flow_imbalance + 0.35 * clob_imbalance
        )

        amm_buy_vol = sum(
            tr.get('quantity', 0.0)
            for tr in period_trades
            if tr.get('side') == 'buy' and tr.get('venue') != 'clob'
        )
        amm_sell_vol = sum(
            tr.get('quantity', 0.0)
            for tr in period_trades
            if tr.get('side') == 'sell' and tr.get('venue') != 'clob'
        )
        amm_gross = amm_buy_vol + amm_sell_vol
        amm_imbalance = (amm_buy_vol - amm_sell_vol) / amm_gross if amm_gross > 0 else 0.0
        self._logged_amm_order_flow_imbalance = amm_imbalance
        self._amm_order_flow_imbalance = (
            0.70 * self._amm_order_flow_imbalance + 0.30 * amm_imbalance
        )

        sigma_base = max(self.sigma_low, 1e-9)
        stress_scale = max(1.0, self._sigma / sigma_base)
        stress_excess = max(0.0, stress_scale - 1.0)
        flow_damage = 0.08 * abs(self._order_flow_imbalance) * (stress_excess + self._liquidity_shock)
        self._liquidity_shock = min(0.85, self._liquidity_shock + flow_damage)
        self._update_systemic_liquidity()

    def observe_dealer_sector(self, dealers):
        """Read how much quoting capacity the dealer sector is still supplying.

        Called every period in every arm. The measure is one minus the mean
        share of its own unimpaired depth that each dealer offers, so a sector
        quoting in full reads zero and a sector wholly withdrawn reads one.

        By design this is the same quantity the dealer uses to size its own
        quote, so the cascade cannot be driven by a state variable that has no
        consequence for what the book actually shows.
        """
        supplied = []
        for dealer in dealers or ():
            adjust = getattr(dealer, '_state_quote_adjustments', None)
            if adjust is None:
                continue
            try:
                supplied.append(max(0.0, min(1.0, float(adjust()['depth_mult']))))
            except Exception:
                continue
        if not supplied:
            return
        self._dealer_capacity_impairment = 1.0 - sum(supplied) / len(supplied)

    def observe_venue_conditions(self, clob, amm_pools: dict, arbitrageur=None):
        """Update AMM-specific venue stress signals from live cross-venue state."""
        if not amm_pools:
            self._amm_slippage_signal = 0.0
            self._amm_reserve_imbalance = 0.0
            self._venue_basis_bps = 0.0
            self._pool_dispersion_bps = 0.0
            self._arbitrage_capacity = 0.0
            return

        clob_mid = None
        try:
            clob_mid = clob.mid_price()
        except Exception:
            clob_mid = None
        if clob_mid is not None and (not math.isfinite(clob_mid) or clob_mid <= 0):
            clob_mid = None

        pool_states = []
        reserve_imbalances = []
        depths = []
        for pool in amm_pools.values():
            try:
                mid = pool.mid_price()
            except Exception:
                continue
            if not math.isfinite(mid) or mid <= 0:
                continue
            pool_state = {
                'pool': pool,
                'mid': float(mid),
            }
            try:
                depth = pool.effective_depth(clob_mid if clob_mid is not None else mid)
            except Exception:
                depth = float('nan')
            pool_state['depth'] = depth
            if math.isfinite(depth) and depth > 0:
                depths.append(float(depth))
            try:
                reserve_ratio = pool.y / max(pool.x * mid, 1e-9)
                reserve_imbalances.append(min(1.0, abs(reserve_ratio - 1.0)))
            except Exception:
                pass
            pool_states.append(pool_state)

        if not pool_states:
            self._amm_slippage_signal = 0.0
            self._amm_reserve_imbalance = 0.0
            self._venue_basis_bps = 0.0
            self._pool_dispersion_bps = 0.0
            self._arbitrage_capacity = 0.0
            return

        pool_mids = [state['mid'] for state in pool_states]
        amm_mid = sum(pool_mids) / len(pool_mids)
        ref_mid = clob_mid if clob_mid is not None else amm_mid

        basis_vals = []
        dispersion_vals = []
        slippage_vals = []
        probe_qty = 5.0
        if depths:
            probe_qty = max(1.0, min(10.0, 0.01 * (sum(depths) / len(depths))))

        for state in pool_states:
            pool = state['pool']
            pool_mid = state['mid']
            if ref_mid > 0:
                basis_vals.append(abs(pool_mid - ref_mid) / ref_mid * 10_000.0)
            if amm_mid > 0:
                dispersion_vals.append(abs(pool_mid - amm_mid) / amm_mid * 10_000.0)
            try:
                buy_quote = pool.quote_buy(probe_qty, S_t=ref_mid)
                sell_quote = pool.quote_sell(probe_qty, S_t=ref_mid)
                buy_slip = max(0.0, float(buy_quote.get('slippage_bps', 0.0)))
                sell_slip = max(0.0, float(sell_quote.get('slippage_bps', 0.0)))
                slippage_vals.append(0.5 * (buy_slip + sell_slip))
            except Exception:
                pass

        self._amm_slippage_signal = (
            sum(slippage_vals) / len(slippage_vals) if slippage_vals else 0.0
        )
        self._amm_reserve_imbalance = (
            sum(reserve_imbalances) / len(reserve_imbalances) if reserve_imbalances else 0.0
        )
        self._venue_basis_bps = sum(basis_vals) / len(basis_vals) if basis_vals else 0.0
        self._pool_dispersion_bps = (
            sum(dispersion_vals) / len(dispersion_vals) if dispersion_vals else 0.0
        )

        trade_cap = getattr(arbitrageur, 'trade_fraction_cap', 0.0) if arbitrageur is not None else 0.0
        correction_cap = getattr(arbitrageur, 'max_correction_pct', 0.0) if arbitrageur is not None else 0.0
        depth_scale = min(1.0, (sum(depths) / len(depths)) / max(10.0 * probe_qty, 1.0)) if depths else 0.0
        basis_penalty = 1.0 / (1.0 + self._venue_basis_bps / 40.0)
        self._arbitrage_capacity = max(
            0.0,
            min(
                1.0,
                self._systemic_liquidity * depth_scale * basis_penalty
                * (trade_cap + correction_cap),
            ),
        )

    def set_recovery_support(self, support: float):
        """Set resiliency support from active book-side liquidity providers."""
        self._recovery_support = max(0.75, min(1.75, float(support)))

    # ---- private ---------------------------------------------------------

    def _step_piecewise(self):
        if self.is_stress():
            self._sigma = self.sigma_high
            self._c = self.c_high
        else:
            self._sigma = self.sigma_low
            self._c = self.c_low

    def _step_stochastic(self):
        """
        OU-like: σ_{t+1} = σ_t + κ_σ (μ_σ − σ_t) + η_σ ε
        with regime-dependent mean μ_σ.
        """
        kappa_s, kappa_c = 0.1, 0.1
        eta_s, eta_c = 0.002, 0.001

        mu_s = self.sigma_high if self.is_stress() else self.sigma_low
        mu_c = self.c_high if self.is_stress() else self.c_low

        self._sigma += kappa_s * (mu_s - self._sigma) + eta_s * self._rng.gauss(0, 1)
        self._c += kappa_c * (mu_c - self._c) + eta_c * self._rng.gauss(0, 1)

        self._sigma = max(self._sigma, 1e-6)
        self._c = max(self._c, 0.0)

    def _step_price(self):
        """
        Mean-reverting log-price step toward the latent anchor.
        """
        if self._price is None:
            return
        anchor = self._price_anchor if self._price_anchor is not None else self._price
        sigma = self._sigma
        mu = self._drift
        eps = self._rng.gauss(0, 1)
        log_price = math.log(max(self._price, 1e-6))
        log_anchor = math.log(max(anchor, 1e-6))
        log_price += mu
        log_price += self._price_reversion * (log_anchor - log_price)
        log_price += self._price_vol_scale * sigma * eps
        self._price = math.exp(log_price)
        self._price = max(self._price, 1e-6)

    def _update_systemic_liquidity(self):
        """Shared liquidity factor that links CLOB and AMM conditions."""
        sigma_base = max(self.sigma_low, 1e-9)
        c_base = max(self.c_low, 1e-9)
        sigma_stress = max(0.0, self._sigma / sigma_base - 1.0)
        c_stress = max(0.0, self._c / c_base - 1.0)
        flow_pressure = abs(self._order_flow_imbalance) * (0.5 * sigma_stress + self._liquidity_shock)

        friction = (
            0.12 * sigma_stress
            + 0.05 * c_stress
            + 0.35 * flow_pressure
            + self._liquidity_shock
        )

        # Dealer capacity enters above a threshold and convexly, so that the
        # sector absorbs ordinary attrition and only a genuinely impaired one
        # feeds back. The square is what makes a cascade a cascade: the second
        # dealer to leave an already strained sector does more damage than the
        # first did. The term is bounded by the gain, and systemic liquidity
        # keeps its own floor, so the loop cannot run away.
        # Impaired dealer capacity multiplies the friction the market is
        # already under instead of contributing a term of its own. Capacity
        # binds when there is a shock to absorb, which is the finding this
        # follows, and when the exogenous stress has decayed there is nothing
        # left to amplify. Two additive forms were tried and neither is usable.
        # A term in the level of impairment holds systemic liquidity at 1/(1+g)
        # for as long as the sector is impaired, which for any gain above
        # 1/0.55-1 sits under the liquidity a withdrawn dealer needs to return,
        # and since the impairment is itself the consequence of dealers being
        # out the state absorbs: measured on the dealer crisis the sector stayed
        # wholly withdrawn for 586 periods of 1000 against 244 with the channel
        # off. A term in the rise of impairment releases correctly but carries
        # no force, moving impaired capacity by 0.002 where the level form moved
        # it by 0.12. The multiplicative form amplifies and cannot latch.
        if self._dealer_cascade_gain > 0.0:
            span = max(1e-9, 1.0 - self._dealer_capacity_threshold)
            excess = min(1.0, max(0.0, self._dealer_capacity_impairment
                                  - self._dealer_capacity_threshold) / span)
            friction *= 1.0 + self._dealer_cascade_gain * excess ** 2

        liquidity = 1.0 / (1.0 + friction)
        liquidity *= self._session_liquidity_multiplier
        self._systemic_liquidity = max(0.2, min(1.0, liquidity))

    def _update_session_state(self):
        """Optional intraday FX session cycle for flow and liquidity timing.

        Disabled when session_cycle <= 0 (the FX baseline default). When
        enabled the cycle implements a four-session round-robin (Asia /
        London / Overlap / NewYork) over session_cycle ticks.
        """
        if self._session_cycle <= 0:
            return
        bucket = self._t % self._session_cycle
        cycle = self._session_cycle
        if bucket < cycle * 6 // 24:
            self._session_name = 'Asia'
            self._session_flow_multiplier = 0.85
            self._session_liquidity_multiplier = 0.90
            self._session_vol_multiplier = 0.92
        elif bucket < cycle * 12 // 24:
            self._session_name = 'London'
            self._session_flow_multiplier = 1.10
            self._session_liquidity_multiplier = 1.05
            self._session_vol_multiplier = 1.05
        elif bucket < cycle * 17 // 24:
            self._session_name = 'Overlap'
            self._session_flow_multiplier = 1.25
            self._session_liquidity_multiplier = 1.10
            self._session_vol_multiplier = 1.12
        else:
            self._session_name = 'NewYork'
            self._session_flow_multiplier = 1.00
            self._session_liquidity_multiplier = 0.98
            self._session_vol_multiplier = 1.00
