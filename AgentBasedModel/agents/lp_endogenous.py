"""Endogenous liquidity provision for the automated pools.

The shipped ``AMMProvider`` adjusts committed liquidity through a linear rule
in fee income, volatility and funding cost, with a floor that prevents the
pool from being drained. Liquidity therefore responds to market *conditions*
but never to the provider's own *outcome*, and the pool cannot be abandoned.
Both referees objected to that, the first asking whether the market maker
loses money and acts on it, the second asking for entry, exit, depletion and
the fee needed to prevent a run.

This module replaces the rule with a participation condition derived from a
payoff, and replaces the single representative provider with a heterogeneous
population, so that availability becomes an outcome of the model.

Ledger
------
Holdings are recorded as liquidity provider tokens against a supply, in the
way a deployed pool records them, rather than as a bare fraction. Writing

    NAV      V_t = x_t p_t + y_t
    supply   T_t
    price    V_t / T_t

a balanced commitment ``(dx, dy)`` proportional to the current reserves mints
``(dx p_t + dy) T_t / V_t`` tokens.  Redemption returns the actual base and
quote reserves released by the pool; there is no implicit conversion at the
reference price.  Both operations scale NAV and supply together and therefore
leave the price per token untouched.  Only trading, rebalancing losses and fees
move it.

Assets only reach a provider when reserves actually leave the pool, so a
request that the numerical floor cannot serve stays queued as unburned tokens
instead of becoming an unbacked claim.  Fees are also distributed in the
currency in which the venue charged them.

Payoff
------
Over one period, with reference price :math:`p_t` and reserves moving from
:math:`(x_{t-1}, y_{t-1})` to :math:`(x_t, y_t)`,

    hold value   H_t = x_{t-1} p_t + y_{t-1}
    pool value   V_t = x_t p_t + y_t
    rebalancing loss  L_t = H_t - V_t
    net payoff   pi_t = F_t - L_t

where :math:`F_t` is fee revenue earned in the period. The loss term is the
realised counterpart of loss versus rebalancing, measured against a hold and
rebalance benchmark rather than assumed. Mints and burns are removed from the
comparison, so the payoff measures the investment result alone.

Participation and the sponsor
-----------------------------
Provider *i* compares the smoothed net return on committed capital against its
own outside option, expands when the pool earns more and contracts when it
earns less. Sustained underperformance triggers exit and sustained
outperformance readmits a provider that has left.

A sponsor may pay a subsidy, expressed as a return on committed capital. It is
an actual transfer: the sponsor's account is debited and provider wallets are
credited pro rata to the tokens they hold, so the same number that enters the
participation decision also leaves somebody's pocket.
"""
from __future__ import annotations

import math as _math
import random as _random
from typing import List, Optional


class EndogenousLP:
    """A provider holding LP tokens plus native base and quote balances."""

    def __init__(self,
                 outside_option: float,
                 tokens: float,
                 wallet: float = 0.0,
                 wallet_base: float = 0.0,
                 kappa: float = 0.35,
                 response_scale: float = 1e-6,
                 max_adj: float = 0.05,
                 exit_patience: int = 25,
                 entry_patience: int = 40,
                 entry_margin: float = 0.25,
                 warmup: int = 0):
        self.outside_option = float(outside_option)
        # Periods before the exit counter is armed. Not a free parameter. The
        # population sets it from the horizon of its own smoother, so it is
        # whatever that smoother needs to mean anything.
        self.warmup = int(warmup)
        self.tokens = float(tokens)        # claim on the pool
        self.wallet_cash = float(wallet)   # quote currency held outside
        self.wallet_base = float(wallet_base)
        self.kappa = float(kappa)
        self.response_scale = max(1e-18, float(response_scale))
        self.max_adj = float(max_adj)
        self.exit_patience = int(exit_patience)
        self.entry_patience = int(entry_patience)
        self.entry_margin = float(entry_margin)

        self.active = True
        self.pending_burn = 0.0            # tokens queued for redemption
        self.subsidy_received = 0.0
        self.fees_received = 0.0           # quote-value telemetry at receipt
        self.fees_received_base = 0.0
        self.fees_received_quote = 0.0
        self._below = 0                    # consecutive periods under the option
        # Entry persistence is measured against the full economic hurdle, not
        # merely against the outside option.  The EWMA is already a smoothed
        # return, so this counter is kept visible in telemetry on purpose. A run
        # that ends before it reaches ``entry_patience`` is right-censored, not
        # evidence that the entry mechanism is absent.
        self._above = 0
        self.max_entry_streak = 0
        self.exited_at: Optional[int] = None
        self.last_exited_at: Optional[int] = None
        self.entered_at: Optional[int] = None
        self.reentered_at: Optional[int] = None
        self.last_reentered_at: Optional[int] = None
        self.exit_count = 0
        self.entry_count = 0
        self.reentry_count = 0
        self.ever_deployed = self.tokens > 1e-12

    @property
    def wallet(self) -> float:
        """Backward-compatible name for the quote-currency wallet."""
        return self.wallet_cash

    @wallet.setter
    def wallet(self, value: float):
        self.wallet_cash = float(value)

    def decide(self, rho: float, t: int) -> float:
        """Desired fractional change in this provider's own holding.

        Returns a fraction of its current holding. The population converts that
        into mints and burns and enforces the capital constraint, so a provider
        can never commit more cash than it actually holds. Exit is signalled by
        ``-1.0`` and re-entry by ``float('inf')``.

        The smoothed return opens at zero, because nothing has been observed
        yet, so at any strictly positive outside option every provider counts
        as short of its alternative from the first period. Left alone that
        turns survival into a race between the patience of the provider and
        the convergence of the smoother, which has no economic content and
        was killing pools before they had earned anything. During the warm up
        no sizing, entry or exit decision is taken, so a provider does not pass
        judgement on a record it has not yet seen.
        """
        excess = rho - self.outside_option
        # ``response_scale`` normalises the *size* adjustment below.  It is a
        # numerical/behavioural response coefficient, not an entry cost.  Using
        # it for both jobs made a 25% entry margin equal to 2.5e-7 per tick when
        # the outside option was only 1.33e-9: an economically unintended hurdle
        # hundreds of times larger than the alternative return.  Entry hysteresis
        # is therefore a dimensionless premium over the provider's own outside
        # option.  At a zero outside option the natural hurdle is zero.
        entry_excess_hurdle = self.entry_margin * abs(self.outside_option)

        # Warm-up means that the estimate is not yet decision-grade.  Holding
        # only the binary-exit counter while still allowing a five-per-cent
        # sizing response let an uninitialised EWMA drain most of the pool
        # before the guard expired.  No entry, exit or resizing is therefore
        # allowed until the same observation threshold has been reached.
        if t < self.warmup:
            self._below = 0
            self._above = 0
            return 0.0

        if excess < 0:
            self._below += 1
            self._above = 0
        elif excess > entry_excess_hurdle:
            self._above += 1
            self.max_entry_streak = max(self.max_entry_streak, self._above)
            self._below = 0
        else:
            # The interval between the exit and entry thresholds is a genuine
            # no-action band.  A single observation above the full hurdle must
            # not complete a streak accumulated below that hurdle.
            self._above = 0
            self._below = 0

        if self.active:
            if self._below >= self.exit_patience:
                self.active = False
                if self.exited_at is None:
                    self.exited_at = t
                self.last_exited_at = t
                self.exit_count += 1
                return -1.0
            # The two terms are added rather than maximised.  Taking the
            # larger of them pinned the denominator to ``response_scale``
            # whenever the outside option sat below it, which at the
            # calibrated values it does by a factor of about seven hundred.
            # The outside option then entered the size decision only as a
            # shift inside ``excess`` and not at all as a scale, so the
            # sensitivity of supply to the alternative return was flat over
            # the whole range a robustness panel varies it across: the panel
            # could not have reported anything but insensitivity, whatever
            # the model did.  The sum keeps both limits the maximum had, a
            # finite scale at a zero outside option and a scale set by the
            # outside option once it dominates, without the flat region
            # between them.  At the calibrated point the two forms differ by
            # about a tenth of a per cent.
            scale = abs(self.outside_option) + self.response_scale
            want = self.kappa * excess / scale
            return max(-self.max_adj, min(self.max_adj, want))

        if (self._above >= self.entry_patience
                and self.pending_burn <= 1e-12
                and self.wallet_cash > 1e-12
                and self.wallet_base > 1e-12):
            # This is an intent.  The population changes ``active`` only after
            # both native wallet legs have actually minted positive tokens, so
            # a failed or dust-sized request cannot create a ghost provider.
            return float('inf')
        return 0.0


class LPPopulation:
    """A heterogeneous population of providers holding tokens on one pool.

    Exposes ``update_liquidity`` so it is a drop in replacement for
    ``AMMProvider`` inside the simulator loop.
    """

    def __init__(self,
                 pool,
                 env,
                 n_providers: int = 5,
                 n_entrants: int = 5,
                 outside_option: float = 0.0,
                 option_dispersion: float = 0.6,
                 kappa: float = 0.35,
                 response_scale: float = 1e-6,
                 max_adj: float = 0.05,
                 exit_patience: int = 25,
                 entry_patience: int = 40,
                 entry_margin: float = 0.25,
                 wallet_ratio: float = 0.20,
                 withdraw_skew: float = 0.0,
                 subsidy_rate: float = 0.0,
                 loss_rebate_fraction: float = 0.0,
                 ewma_alpha: float = 0.10,
                 warmup: Optional[int] = None,
                 allow_exit: bool = True,
                 allow_entry: bool = True,
                 rng: Optional[_random.Random] = None):
        self.pool = pool
        self.env = env
        # Diagnostic/structural coefficient inherited from the venue.  The
        # endogenous decision rule uses realised loss directly, but exposing
        # the coefficient keeps the curve mechanics auditable and comparable
        # with the reduced-form provider.
        self.phi2 = float(getattr(pool, 'rebalancing_loss_curvature', 0.125))
        self.withdraw_skew = float(withdraw_skew)
        self.subsidy_rate = float(subsidy_rate)
        self.loss_rebate_fraction = max(0.0, min(1.0,
                                                 float(loss_rebate_fraction)))
        self.ewma_alpha = float(ewma_alpha)
        self.allow_exit = bool(allow_exit)
        self.allow_entry = bool(allow_entry)

        # Purely numerical floor. The invariant solver is not defined at
        # vanishing reserves, so redemptions stop there. Tokens queued for
        # redemption stay with their owner instead of being written off, which
        # is why the floor produces a delay rather than a loss.
        self.min_reserve_ratio = 0.01
        self.floor_binds = 0

        rng = rng or _random.Random(0)
        n = max(1, int(n_providers))
        # An outside option of exactly zero is meaningful, namely capital that
        # can sit idle at no return, so it must not be silently replaced.
        base = float(outside_option)

        # Three horizons of the smoother, which is where an exponentially
        # weighted mean has substantially converged. Tied to alpha so that
        # it is not a knob of its own. Passing a value explicitly is for
        # tests that drive the machinery directly and are not interested in
        # the timing of participation.
        warmup = (int(_math.ceil(3.0 / max(1e-9, float(ewma_alpha))))
                  if warmup is None else int(warmup))
        self.warmup = warmup

        p0 = self._reference_price()
        v0 = self._value(p0)
        # One token per unit of quote value at inception, so that the price per
        # token starts at one and any later deviation is investment result.
        self.total_supply = v0
        self._initial_value = v0

        self.providers: List[EndogenousLP] = []
        self.n_incumbents = n
        buffer_base = max(0.0, float(wallet_ratio)) * float(pool.x) / n
        buffer_quote = max(0.0, float(wallet_ratio)) * float(pool.y) / n
        for _ in range(n):
            # Dispersion is relative to the common outside option.  A zero
            # option therefore remains exactly zero; manufacturing positive
            # and negative per-tick rates around it would introduce an
            # uncalibrated return scale.
            jitter = option_dispersion * (2.0 * rng.random() - 1.0)
            opt_i = base * (1.0 + jitter)
            # Patience is dispersed as well, so that a common adverse shock
            # produces a staggered run instead of a single simultaneous exit.
            pat = max(3, int(round(exit_patience * (0.5 + rng.random()))))
            self.providers.append(EndogenousLP(
                outside_option=opt_i,
                tokens=v0 / n,
                wallet=buffer_quote,
                wallet_base=buffer_base,
                kappa=kappa, max_adj=max_adj,
                response_scale=response_scale,
                exit_patience=pat if allow_exit else 10 ** 9,
                entry_patience=entry_patience if allow_entry else 10 ** 9,
                entry_margin=entry_margin,
                warmup=warmup))

        # Potential entrants hold cash outside the pool and commit it only when
        # the return, inclusive of any subsidy, has exceeded their own outside
        # option for long enough.
        for _ in range(max(0, int(n_entrants))):
            jitter = option_dispersion * (2.0 * rng.random() - 1.0)
            opt_i = base * (1.0 + jitter)
            entrant_pat = max(3, int(round(
                exit_patience * (0.5 + rng.random())
            )))
            e = EndogenousLP(outside_option=opt_i, tokens=0.0,
                             wallet=buffer_quote,
                             wallet_base=buffer_base,
                             kappa=kappa, response_scale=response_scale,
                             max_adj=max_adj,
                             exit_patience=entrant_pat if allow_exit else 10 ** 9,
                             warmup=warmup,
                             entry_patience=entry_patience if allow_entry else 10 ** 9,
                             entry_margin=entry_margin)
            e.active = False
            self.providers.append(e)

        self._t = 0
        self._prev_x = float(pool.x)
        self._prev_y = float(pool.y)
        self._rho_ewma = 0.0
        self._rho_eff_ewma = 0.0
        self.sponsor_paid = 0.0            # cumulative cash paid by the sponsor
        self.sponsor_account_quote = 0.0   # explicit sponsor balance; may borrow
        self.fees_distributed = 0.0        # cumulative fee income paid to providers
        self.fees_distributed_base = 0.0
        self.fees_distributed_quote = 0.0
        # Reserves that the numerical floor stranded and that a later refounding
        # handed to whoever recapitalised the pool. Tracked so that wealth
        # accounting has no silent source.
        self.refounded_windfall = 0.0

        self.history = {'rho': [], 'net': [], 'lvr': [], 'fee': [], 'active': [],
                        'supply': [], 'nav_per_token': [], 'queued': [],
                        'closed': [], 'entered': [], 'subsidy': [],
                        'rate_subsidy': [], 'loss_rebate': [],
                        # Gross events, not changes in the active headcount.  A
                        # simultaneous exit and entry must remain observable.
                        'entries_gross': [], 'reentries_gross': [],
                        'exits_gross': [], 'deployed': [],
                        'deployed_share': [], 'entry_progress_max': []}

    # ---------------------------------------------------------------- helpers
    def _reference_price(self) -> float:
        p = getattr(self.env, 'fair_price', None)
        if p is None or not _math.isfinite(p) or p <= 0:
            p = self.pool.mid_price()
        return max(float(p), 1e-9)

    def _value(self, p: float) -> float:
        return max(1e-12, self.pool.x * p + self.pool.y)

    def nav_per_token(self, p: Optional[float] = None) -> float:
        if self.total_supply <= 1e-12:
            return 0.0
        return self._value(p if p is not None else self._reference_price()) \
            / self.total_supply

    @property
    def closed(self) -> bool:
        """No claim outstanding, so the venue has no capital behind it."""
        return self.total_supply <= 1e-9

    def _publish_closed(self):
        """Tell the pool whether it still has capital behind it.

        The flag is recomputed every period rather than latched, so a pool that
        is recapitalised starts quoting again.
        """
        try:
            self.pool.closed = bool(self.closed)
        except Exception:
            pass

    @property
    def deployed_share(self) -> float:
        """Supply relative to inception, the analogue of the old share sum."""
        return self.total_supply / max(self._initial_value, 1e-12)

    @property
    def n_active(self) -> int:
        return sum(1 for lp in self.providers if lp.active)

    @property
    def n_entered(self) -> int:
        return sum(1 for lp in self.providers[self.n_incumbents:]
                   if lp.active and lp.tokens > 0.0)

    @property
    def n_deployed(self) -> int:
        """Providers with a positive outstanding capital claim.

        Queued redemptions remain capital at work until the pool serves them,
        so they count as deployed even though their owner is no longer active.
        """
        return sum(1 for lp in self.providers
                   if lp.tokens + lp.pending_burn > 1e-12)

    @property
    def queued_tokens(self) -> float:
        return sum(lp.pending_burn for lp in self.providers)

    def _resync_invariant(self):
        pool = self.pool
        if hasattr(pool, '_sync_norm'):
            pool._sync_norm()
        try:
            from AgentBasedModel.venues.amm import _hfmm_get_D
            if hasattr(pool, 'A') and hasattr(pool, '_xn'):
                pool.D = _hfmm_get_D(pool._xn, pool._yn, pool.A)
        except Exception:
            pass
        if hasattr(pool, 'k'):
            pool.k = pool.x * pool.y

    # ------------------------------------------------------------------ main
    def update_liquidity(self):
        self._t += 1
        p = self._reference_price()

        # ---- investment result of the period, mints and burns excluded ----
        hold = self._prev_x * p + self._prev_y
        value = self._value(p)
        lvr = hold - value
        # The fee of the period is claimed before it is used, so the figure
        # that enters the participation signal is the very cash that reaches
        # the wallets, valued once and at the same reference price as every
        # other term. Reading the pool's quote valued counter instead put the
        # execution price of each trade into the signal while the payment was
        # struck at the reference price, and on a large trade the two differed
        # by a third.
        claimed_base, claimed_quote = 0.0, 0.0
        if self.total_supply > 1e-12:
            try:
                claimed_base, claimed_quote = self.pool.claim_fees()
            except AttributeError:
                claimed_base, claimed_quote = 0.0, 0.0
        fee = claimed_base * p + claimed_quote
        net = fee - lvr
        period_return = net / hold if hold > 1e-9 else 0.0
        a = self.ewma_alpha
        self._rho_ewma = (1.0 - a) * self._rho_ewma + a * period_return
        # A pool with nothing in it has no rate of return. Dividing the smoothed
        # result by a numerical floor turned the residue of the last period into
        # a signal of ten per cent and drove providers straight back into a venue
        # that had just wound down. A closed pool is reported as earning nothing,
        # which is what it does, and re-entry is then decided by the outside
        # option against the subsidy rather than by an artefact of the divisor.
        rho = self._rho_ewma if value > 1e-9 else 0.0

        # ---- fee income belongs to the providers ----
        # The pool moves fees out of the reserves into a separate accumulator,
        # so they are not in the net asset value and do not reach anyone by
        # themselves. Letting fee income enter the participation signal while
        # never reaching a wallet would mean providers acting on money that was
        # not theirs, and the balance would grow ownerless once they left.
        # The population therefore claims the period's fees and pays them out
        # pro rata to the tokens outstanding, queued tokens included, since the
        # capital behind them was still working.
        # What is claimed is the balance the pool actually withheld, in the
        # currency it was withheld in, and the base side is valued here at the
        # reference price. The cumulative counter on the pool is left untouched,
        # since it is telemetry whose differences a profit and loss statement
        # reads, and decrementing it made that history read as a flat zero.
        if fee > 0.0:
            for lp in self.providers:
                eligible = lp.tokens + lp.pending_burn
                if eligible > 0.0:
                    share = eligible / self.total_supply
                    cut_base = claimed_base * share
                    cut_quote = claimed_quote * share
                    lp.wallet_base += cut_base
                    lp.wallet_cash += cut_quote
                    lp.fees_received_base += cut_base
                    lp.fees_received_quote += cut_quote
                    lp.fees_received += cut_base * p + cut_quote
            self.fees_distributed += fee
            self.fees_distributed_base += claimed_base
            self.fees_distributed_quote += claimed_quote
        try:
            self.pool.period_fee_revenue = 0.0
        except Exception:
            pass

        # ---- the sponsor actually pays ----
        rate_subsidy_cash = 0.0
        loss_rebate_cash = 0.0
        subsidy_cash = 0.0
        if self.total_supply > 1e-12:
            rate_subsidy_cash = max(0.0, self.subsidy_rate * value)
            # A loss guarantee is state contingent rather than a continuously
            # paid return.  It reimburses only an actually realised negative
            # pool payoff, and the fraction is bounded by one so the policy
            # cannot turn a loss into a manufactured operating profit.
            loss_rebate_cash = (self.loss_rebate_fraction
                                * max(0.0, -net))
            subsidy_cash = rate_subsidy_cash + loss_rebate_cash
        if subsidy_cash > 0.0:
            self.sponsor_paid += subsidy_cash
            self.sponsor_account_quote -= subsidy_cash
            # Tokens queued for redemption have not been burned yet, so the
            # capital behind them is still in the pool and still earning. They
            # are eligible, and paying only on unqueued tokens would debit the
            # sponsor for more than the providers receive.
            for lp in self.providers:
                eligible = lp.tokens + lp.pending_burn
                if eligible > 0.0:
                    cut = subsidy_cash * eligible / self.total_supply
                    lp.wallet_cash += cut
                    lp.subsidy_received += cut
        # The supported payoff must pass through the same filter as the raw
        # payoff. Adding an unsmoothed current rebate to an already smoothed
        # raw return exaggerates the rebate by roughly 1/alpha and turns loss
        # insurance into a spurious expansion signal. A closed pool has no
        # realised loss to rebate, so only a posted rate can attract entry.
        if hold > 1e-9:
            effective_period_return = (net + subsidy_cash) / hold
            self._rho_eff_ewma = ((1.0 - a) * self._rho_eff_ewma
                                  + a * effective_period_return)
            rho_eff = self._rho_eff_ewma
        else:
            rho_eff = max(0.0, self.subsidy_rate)
            self._rho_eff_ewma = rho_eff

        # ---- decisions become balanced mint requests and burn requests in tokens
        mint_base = 0.0
        mint_quote = 0.0
        exits_gross = 0
        activation_candidates = []
        for lp in self.providers:
            want = lp.decide(rho_eff, self._t)
            if want == -1.0:
                exits_gross += 1
                lp.pending_burn += lp.tokens
                lp.tokens = 0.0
            elif want == float('inf'):
                tokens_before = lp.tokens
                req_base, req_quote = self._stage_mint(
                    lp, self._wallet_capacity_value(lp, p), p
                )
                mint_base += req_base
                mint_quote += req_quote
                activation_candidates.append((lp, tokens_before))
            elif lp.active and want > 0.0:
                own_value = lp.tokens * self.nav_per_token(p)
                desired = min(own_value * want, self._wallet_capacity_value(lp, p))
                req_base, req_quote = self._stage_mint(lp, desired, p)
                mint_base += req_base
                mint_quote += req_quote
            elif lp.active and want < 0.0:
                burn = lp.tokens * min(1.0, abs(want))
                lp.pending_burn += burn
                lp.tokens -= burn

        self._mint(mint_base, mint_quote, p)
        entries_gross = 0
        reentries_gross = 0
        for lp, tokens_before in activation_candidates:
            if lp.tokens > tokens_before + 1e-12:
                if lp.ever_deployed:
                    lp.reentry_count += 1
                    lp.reentered_at = self._t
                    lp.last_reentered_at = self._t
                    reentries_gross += 1
                else:
                    lp.entry_count += 1
                    lp.entered_at = self._t
                    lp.ever_deployed = True
                    entries_gross += 1
                lp.active = True
                lp._above = 0
            else:
                # ``decide`` only requested activation.  No capital was minted,
                # hence this provider remains outside and is not an event.
                lp.active = False
        self._serve_redemptions(p)

        self._publish_closed()
        self._prev_x, self._prev_y = float(self.pool.x), float(self.pool.y)
        h = self.history
        h['rho'].append(rho); h['net'].append(net); h['lvr'].append(lvr)
        h['fee'].append(fee); h['active'].append(self.n_active)
        h['supply'].append(self.total_supply)
        h['nav_per_token'].append(self.nav_per_token(p))
        h['queued'].append(self.queued_tokens)
        h['closed'].append(int(self.closed))
        h['entered'].append(self.n_entered)
        h['subsidy'].append(subsidy_cash)
        h['rate_subsidy'].append(rate_subsidy_cash)
        h['loss_rebate'].append(loss_rebate_cash)
        h.setdefault('rho_eff', []).append(rho_eff)
        h['entries_gross'].append(entries_gross)
        h['reentries_gross'].append(reentries_gross)
        h['exits_gross'].append(exits_gross)
        h['deployed'].append(self.n_deployed)
        h['deployed_share'].append(self.deployed_share)
        h['entry_progress_max'].append(max(
            (min(1.0, lp._above / max(1, lp.entry_patience))
             for lp in self.providers if not lp.active),
            default=0.0,
        ))

    # ------------------------------------------------------------- ledger ops
    def _wallet_capacity_value(self, lp: EndogenousLP, p: float) -> float:
        """Quote value that ``lp`` can add without an implicit FX conversion."""
        if self.total_supply <= 1e-12 or self.pool.x <= 1e-12 or self.pool.y <= 1e-12:
            # A refounding uses a fifty-fifty value basket at the reference
            # price.  Both legs must already exist in the provider wallet.
            return max(0.0, min(2.0 * lp.wallet_base * p,
                                2.0 * lp.wallet_cash))
        frac = min(lp.wallet_base / self.pool.x,
                   lp.wallet_cash / self.pool.y)
        return max(0.0, frac * self._value(p))

    def _stage_mint(self, lp: EndogenousLP, desired_value: float,
                    p: float) -> tuple[float, float]:
        """Record a feasible native-asset contribution request."""
        desired_value = max(0.0, min(float(desired_value),
                                     self._wallet_capacity_value(lp, p)))
        if desired_value <= 1e-12:
            lp._mint_base = 0.0
            lp._mint_quote = 0.0
            return 0.0, 0.0
        if self.total_supply <= 1e-12 or self.pool.x <= 1e-12 or self.pool.y <= 1e-12:
            req_base = 0.5 * desired_value / max(p, 1e-12)
            req_quote = 0.5 * desired_value
        else:
            frac = desired_value / self._value(p)
            req_base = frac * self.pool.x
            req_quote = frac * self.pool.y
        lp._mint_base = min(req_base, lp.wallet_base)
        lp._mint_quote = min(req_quote, lp.wallet_cash)
        return lp._mint_base, lp._mint_quote

    def _clear_mint_requests(self):
        for lp in self.providers:
            lp._mint_base = 0.0
            lp._mint_quote = 0.0

    def _mint(self, base: float, quote: float, p: float):
        """Commit native assets and mint tokens at the prevailing NAV."""
        requested_value = base * p + quote
        if requested_value <= 1e-12:
            self._clear_mint_requests()
            return

        refounding = self.total_supply <= 1e-12
        value_before = self._value(p)
        if refounding:
            # There is no incumbent token price.  Providers transfer the two
            # assets they actually hold and receive one token per unit of
            # quote value.  Any numerical residual stays visible as a windfall
            # rather than being silently converted into the missing leg.
            x0, y0 = float(self.pool.x), float(self.pool.y)
            self.pool.x += base
            self.pool.y += quote
            try:
                self.pool.record_reserve_flow(base, quote)
            except AttributeError:
                pass
            self._resync_invariant()
            residual_value = x0 * p + y0
            minted_total = self._value(p)
            self.refounded_windfall += residual_value
            contributed_tokens = max(0.0, minted_total - residual_value)
            for lp in self.providers:
                bx = getattr(lp, '_mint_base', 0.0)
                qy = getattr(lp, '_mint_quote', 0.0)
                contribution = bx * p + qy
                if contribution <= 0.0:
                    continue
                lp.wallet_base -= bx
                lp.wallet_cash -= qy
                lp.tokens += contributed_tokens * contribution / requested_value
            # The residual, if any, belongs pro rata to the refounders too.
            if residual_value > 0.0 and requested_value > 0.0:
                for lp in self.providers:
                    contribution = (getattr(lp, '_mint_base', 0.0) * p
                                    + getattr(lp, '_mint_quote', 0.0))
                    lp.tokens += residual_value * contribution / requested_value
            self.total_supply = minted_total
            self._initial_value = minted_total
            self._clear_mint_requests()
            return

        if value_before <= 1e-9 or self.pool.x <= 1e-12 or self.pool.y <= 1e-12:
            self._clear_mint_requests()
            return

        price_before = self.nav_per_token(p)
        requested_frac = min(base / self.pool.x, quote / self.pool.y)
        served_frac = min(max(0.0, requested_frac), 0.5)
        scale = served_frac / requested_frac if requested_frac > 1e-12 else 0.0
        served_base = base * scale
        served_quote = quote * scale
        served_value = served_base * p + served_quote
        if served_value <= 1e-12:
            self._clear_mint_requests()
            return

        self.pool.add_liquidity(served_frac)
        self._resync_invariant()
        minted = served_value / price_before
        for lp in self.providers:
            bx = getattr(lp, '_mint_base', 0.0) * scale
            qy = getattr(lp, '_mint_quote', 0.0) * scale
            contribution = bx * p + qy
            if contribution <= 0.0:
                continue
            lp.wallet_base -= bx
            lp.wallet_cash -= qy
            lp.tokens += minted * contribution / served_value
        self.total_supply += minted
        self._clear_mint_requests()

    def _serve_redemptions(self, p: float):
        """Burn queued tokens, shrink reserves and pay the cash out."""
        queued = self.queued_tokens
        if queued <= 1e-12 or self.total_supply <= 1e-12:
            return
        value = self._value(p)
        # A wind down is different from a partial redemption. When every token
        # outstanding has been queued nobody is left to be protected by a
        # reserve floor, and holding the last fraction back would leave a
        # venue that quotes with capital no one owns. The floor therefore
        # guards partial redemptions only, and a full wind down is allowed to
        # empty the pool.
        # Decide from the claims that are *not* queued, not from subtracting two
        # large nearly equal floats.  In two primary seeds all providers had
        # left, but 1.1e-9 of rounding dust made ``queued >= supply - 1e-9``
        # false and stranded the pool permanently behind the reserve floor.
        unqueued = sum(max(0.0, lp.tokens) for lp in self.providers)
        claim_dust = max(1e-9, 1e-12 * self.total_supply)
        winding_down = queued > 0.0 and unqueued <= claim_dust
        frac = queued / self.total_supply
        if winding_down:
            # Nothing is left to pace or to tilt, so the pool is simply emptied.
            # Removing a fraction first and sweeping the remainder afterwards
            # gave the same answer by a longer route, and left two branches
            # whose behaviour no test could tell apart from this one.
            price = self.nav_per_token(p)
            self._pay_out(queued, self._sweep_residual(), price, p)
            return
        else:
            # The floor has to bound the withdrawal itself, not merely refuse
            # one once the pool is already through it. Testing the value before
            # the removal let a permitted half leave in one step and land well
            # below the floor, which is how the reserve ratio reached 0.78 per
            # cent against a floor of one. The room left above the floor is
            # therefore what caps the fraction.
            floor = self.min_reserve_ratio * self._initial_value
            room = max(0.0, (value - floor) / value) if value > 1e-12 else 0.0
            paced = min(frac, 0.5)         # the per period cap, a separate thing
            capped = min(paced, room)
            if capped < paced - 1e-12:
                self.floor_binds += 1      # tokens stay queued, nothing is lost
            frac = capped
            if frac <= 1e-12:
                return
        # Burn against the value the pool actually released, not against the
        # request. A capped reserve releases less, and paying the request would
        # create a claim on value that never left. The price per token is read
        # before the reserves move, since that is the price the claim is on.
        price = self.nav_per_token(p)
        self._pay_out(queued, self._remove(frac), price, p)

    def _pay_out(self, queued: float, removed: tuple[float, float],
                 price: float, p: float):
        """Burn against released value and return both native reserve assets.

        A wind down releases the whole pool and extinguishes the queue. Any
        other redemption releases what the caps allowed and burns only that
        much, leaving the rest of the claim outstanding.
        """
        removed_base, removed_quote = removed
        removed_value = removed_base * p + removed_quote
        if removed_value <= 0.0 or queued <= 1e-12:
            return
        emptied = self._value(self._reference_price()) <= 1e-12
        if emptied:
            # The pool has been emptied, so nothing is left owed and the whole
            # queue goes. The price passed in was read before the reserves
            # moved and already equals the value removed per token queued,
            # since a wind down sweeps the pool and burns the whole queue.
            burned = queued
        else:
            burned = min(queued, removed_value / price) if price > 1e-12 else 0.0
        if burned <= 1e-12:
            return
        self.total_supply -= burned
        for lp in self.providers:
            if lp.pending_burn <= 0.0:
                continue
            share = lp.pending_burn / queued
            b = burned * share
            lp.pending_burn -= b
            lp.wallet_base += removed_base * share
            lp.wallet_cash += removed_quote * share
            if lp.pending_burn < 1e-12:
                lp.pending_burn = 0.0
        if emptied:
            # Any remaining supply is the same floating-point dust that was not
            # represented by a provider claim.  An empty venue cannot carry an
            # ownerless positive supply or remain formally open because of it.
            self.total_supply = 0.0

    def max_add(self) -> float:
        """Total capital available to be committed, held by anyone."""
        p = self._reference_price()
        return max(0.0, sum(self._wallet_capacity_value(lp, p)
                            for lp in self.providers))



    def _sweep_residual(self) -> tuple[float, float]:
        """Empty the reserves outright and report the native assets taken out.

        Only a wind down calls this. The venue refuses every request once it
        is closed, so reserves of exactly zero are safe, and leaving them at
        the numerical clamp instead would mean a claim was extinguished
        against value that was still there.
        """
        residual_base = float(self.pool.x)
        residual_quote = float(self.pool.y)
        if residual_base <= 0.0 and residual_quote <= 0.0:
            return 0.0, 0.0
        try:
            self.pool.record_reserve_flow(-self.pool.x, -self.pool.y)
        except AttributeError:
            pass
        self.pool.x = 0.0
        self.pool.y = 0.0
        self._resync_invariant()
        return residual_base, residual_quote

    def _remove(self, frac: float) -> tuple[float, float]:
        """Withdraw a pool fraction and report base and quote that left.

        Redemption in an automated pool is pro rata in both reserves, which is
        the default here. A positive ``withdraw_skew`` tilts redemption toward
        the scarcer reserve, and the two proportions are rescaled so that the
        value leaving equals the value requested.

        A reserve cannot go negative, so each side is capped. With a strongly
        tilted redemption on a lopsided pool the cap binds and less value
        leaves than was asked for. The amount actually removed is therefore
        returned, and the caller burns tokens against that figure rather than
        against the request. Paying out the request while the cap held back the
        reserves would hand the provider value the pool never released.
        """
        # The per period cap protects an ongoing pool from a disorderly
        # redemption. A wind down does not come through here at all, it empties
        # the pool outright.
        frac = max(0.0, min(float(frac), 0.5))
        if frac <= 0:
            return 0.0, 0.0
        p = self._reference_price()
        x0, y0 = float(self.pool.x), float(self.pool.y)
        if abs(self.withdraw_skew) < 1e-12:
            self.pool.remove_liquidity(frac)
            self._resync_invariant()
            return x0 - self.pool.x, y0 - self.pool.y

        vx, vy = self.pool.x * p, self.pool.y
        total = vx + vy
        if total <= 1e-12:
            return 0.0, 0.0
        s_ = max(0.0, min(1.0, self.withdraw_skew))
        scarce_is_x = vx < vy
        wx = (1.0 + s_) if scarce_is_x else (1.0 - s_)
        wy = (1.0 - s_) if scarce_is_x else (1.0 + s_)
        denom = wx * vx + wy * vy
        k = (frac * total) / denom if denom > 1e-12 else 0.0
        fx = min(wx * k, 0.9)
        fy = min(wy * k, 0.9)
        self.pool.x = max(self.pool.x * (1.0 - fx), 1e-9)
        self.pool.y = max(self.pool.y * (1.0 - fy), 1e-9)
        # A tilted redemption bypasses the pool's own proportional path, so the
        # capital that left has to be reported to the venue by hand.
        try:
            self.pool.record_reserve_flow(self.pool.x - x0, self.pool.y - y0)
        except AttributeError:
            pass
        self._resync_invariant()
        return x0 - self.pool.x, y0 - self.pool.y
