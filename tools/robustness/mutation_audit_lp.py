#!/usr/bin/env python3
"""Mutation audit for AgentBasedModel/agents/lp_endogenous.py.

A mutation that leaves the suite green is a hole in the tests, not a harmless
edit. Each entry below is a deliberate defect; the suite is expected to notice.

The first version of this script reported a perfect score with the test file
deleted, because it treated any crash or any unparseable output as evidence
that the mutation had been caught. The guards below exist so that the score
cannot be manufactured that way:

  * the test file must exist and the unmutated baseline must be green, with a
    plausible number of checks, otherwise the run aborts,
  * a mutation whose pattern is not found in the source is an error, not a
    catch, because nothing was actually changed,
  * output that cannot be parsed is an error unless the process also failed,
  * the module is restored in a ``finally`` block, so an interrupted run does
    not leave a mutated copy behind.

    python3 tools/robustness/mutation_audit_lp.py
    python3 tools/robustness/mutation_audit_lp.py --self-check
"""
from __future__ import annotations

import argparse
import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MOD = os.path.join(ROOT, 'AgentBasedModel', 'agents', 'lp_endogenous.py')
# The rule provider is the arm the command line and the paper use by default,
# and it was carrying a defect that no mutation could have caught, because the
# audit only ever touched the endogenous module. Both files are mutated now.
MOD_RULE = os.path.join(ROOT, 'AgentBasedModel', 'agents', 'agents.py')
MOD_SIM = os.path.join(ROOT, 'AgentBasedModel', 'simulator', 'simulator.py')
MOD_AMM = os.path.join(ROOT, 'AgentBasedModel', 'venues', 'amm.py')
# A mutation is routed to whichever file contains its pattern, so the list does
# not have to repeat the path and cannot name the wrong one.
EXTRA_FILES = (MOD_RULE, MOD_SIM, MOD_AMM)
SUITE = os.path.join(ROOT, 'tests', 'lp_endogenous_test.py')
PRISTINE = MOD + '.pristine'
MIN_CHECKS = 295          # the baseline is expected to run at least this many

CAUGHT, SURVIVED, ERROR = 'caught', 'survived', 'error'

RULE_MUTATIONS = [
    ('A1', 'ruleprovider', 'fee income never claimed from the pool',
     'claimed_base, claimed_quote = self.pool.claim_fees()',
     'claimed_base, claimed_quote = 0.0, 0.0'),
    ('A2', 'ruleprovider', 'the base fee is not credited',
     'self.wallet_base += claimed_base', 'pass'),
    ('A3', 'ruleprovider', 'the quote fee is not credited',
     'self.wallet_cash += claimed_quote', 'pass'),
    ('A4', 'ruleprovider', 'the base fee is treated as quote',
     'fee_income = claimed_base * ref_price + claimed_quote',
     'fee_income = claimed_base + claimed_quote'),
    ('A5', 'ruleprovider', 'the rule reads the numeraire again',
     'fee_return = fee_income / pool_value', 'fee_return = fee_income / L'),
    ('A11', 'ruleprovider', 'the loss curvature is assumed and not measured',
     'self.rebalancing_loss_curvature = 0.125 + 0.25 * float(A)',
     'self.rebalancing_loss_curvature = 2.25'),
    ('A6', 'ruleprovider', 'the loss term is linear in the volatility',
     'frac = self.phi1 * fee_return - self.phi2 * vol * vol - self.phi3 * c',
     'frac = self.phi1 * fee_return - self.phi2 * vol - self.phi3 * c'),
    ('A7', 'ruleprovider', 'the volatility term is dropped',
     'frac = self.phi1 * fee_return - self.phi2 * vol * vol - self.phi3 * c',
     'frac = self.phi1 * fee_return - self.phi3 * c'),
    ('A8', 'ruleprovider', 'the funding term is dropped',
     'frac = self.phi1 * fee_return - self.phi2 * vol * vol - self.phi3 * c',
     'frac = self.phi1 * fee_return - self.phi2 * vol * vol'),
    ('A9', 'ruleprovider', 'fees push capital out instead of in',
     'frac = self.phi1 * fee_return - self.phi2 * vol * vol - self.phi3 * c',
     'frac = -self.phi1 * fee_return - self.phi2 * vol * vol - self.phi3 * c'),
    ('A10', 'ruleprovider', 'the volatility index replaces the price volatility',
     "vol = float(getattr(self.env, 'price_sigma', self.env.sigma))",
     'vol = self.env.sigma'),
]

MUTATIONS = [
    ('D1', 'decide', 'sign of the excess return',
     'excess = rho - self.outside_option', 'excess = rho + self.outside_option'),
    ('D2', 'decide', 'counters swapped',
     'if excess < 0:\n            self._below += 1\n            self._above = 0',
     'if excess < 0:\n            self._above += 1\n            self._below = 0'),
    ('D3', 'decide', 'patience threshold made strict',
     'if self._below >= self.exit_patience:', 'if self._below > self.exit_patience:'),
    ('D4', 'decide', 'exit is not signalled',
     'self.exit_count += 1\n                return -1.0',
     'self.exit_count += 1\n                return 0.0'),
    ('D5', 'decide', 'adjustment speed ignored',
     'want = self.kappa * excess / scale', 'want = excess / scale'),
    ('D6', 'decide', 'step is not clamped',
     'return max(-self.max_adj, min(self.max_adj, want))', 'return want'),
    ('D7', 'decide', 'entry without capital',
     'and self.wallet > 1e-12):', 'and True):'),
    ('D8', 'decide', 'entry without a margin',
     'and excess > self.entry_margin * scale', 'and excess > -1e18'),
    ('D9', 'decide', 'favourable counter not reset',
     'self._above += 1\n            self._below = 0', 'self._above += 1'),

    ('L1', 'ledger', 'mint priced after the cash goes in',
     'minted = served_cash / price_before', 'minted = served_cash'),
    ('L2', 'ledger', 'supply not increased on mint',
     'lp._mint_cash = 0.0\n        self.total_supply += minted',
     'lp._mint_cash = 0.0'),
    ('L3', 'ledger', 'supply not reduced on burn',
     'self.total_supply -= burned', 'pass'),
    ('L4', 'ledger', 'unserved cash not returned',
     'lp.wallet += (c - take)', 'pass'),
    ('L5', 'ledger', 'tokens minted ignore the contribution',
     'take = c * served_cash / cash\n            lp.tokens += minted * (c / cash)',
     'take = c * served_cash / cash\n            lp.tokens += minted'),
    ('L6', 'ledger', 'redemption paid at the wrong price',
     'lp.wallet += b * price', 'lp.wallet += b'),
    ('L7', 'ledger', 'queue not reduced when served',
     'lp.pending_burn -= b', 'pass'),
    ('L8', 'ledger', 'exit does not queue the tokens',
     'lp.pending_burn += lp.tokens\n                lp.tokens = 0.0',
     'lp.tokens = 0.0'),
    ('L9', 'ledger', 'partial redemption does not reduce the holding',
     'lp.pending_burn += burn\n                lp.tokens -= burn',
     'lp.pending_burn += burn'),
    ('L10', 'ledger', 'commitment ignores the wallet',
     'cash = min(own_value * want, lp.wallet)', 'cash = own_value * want'),
    ('L11', 'ledger', 'wallet not debited on commitment',
     'mint_cash += cash\n                    lp._mint_cash = cash\n                    lp.wallet -= cash',
     'mint_cash += cash\n                    lp._mint_cash = cash'),
    ('L12', 'ledger', 'no cap on issuance',
     'frac = min(cash / value, 0.5)          # per period cap on growth',
     'frac = cash / value'),
    # The cap is applied again inside ``_remove``, which R2 covers, so removing
    # it at the call site changes nothing and the mutant is equivalent. What
    # the call site does carry is the wind down exemption, which W2 covers.

    ('L14', 'ledger', 'floor discards the queued claim',
     'self.floor_binds += 1      # tokens stay queued, nothing is lost',
     'self.floor_binds += 1\n                for _lp in self.providers:\n                    _lp.pending_burn = 0.0'),
    ('S2', 'sponsor', 'providers are not credited',
     'lp.wallet += cut\n                    lp.subsidy_received += cut',
     'lp.subsidy_received += cut'),
    ('S3', 'sponsor', 'queued tokens excluded from the payment',
     'cut = subsidy_cash * eligible / self.total_supply',
     'cut = subsidy_cash * lp.tokens / self.total_supply'),
    ('S4', 'sponsor', 'sponsor account not debited',
     'self.sponsor_paid += subsidy_cash', 'pass'),
    ('S5', 'sponsor', 'sign of the subsidy in the signal',
     'rho_eff = rho + self.subsidy_rate', 'rho_eff = rho - self.subsidy_rate'),

    ('Y1', 'payoff', 'sign of the rebalancing loss',
     'lvr = hold - value', 'lvr = value - hold'),
    ('Y2', 'payoff', 'sign of the net payoff', 'net = fee - lvr', 'net = fee + lvr'),
    ('Y3', 'payoff', 'fee income ignored', 'net = fee - lvr', 'net = -lvr'),
    ('Y4', 'payoff', 'smoothing weights inverted',
     'self._pi_ewma = (1.0 - a) * self._pi_ewma + a * net',
     'self._pi_ewma = a * self._pi_ewma + (1.0 - a) * net'),
    ('Y5', 'payoff', 'normalised by the initial value',
     'rho = self._pi_ewma / value if value > 1e-9 else 0.0',
     'rho = self._pi_ewma / self._initial_value'),
    ('Y6', 'payoff', 'an empty pool reports a return',
     'rho = self._pi_ewma / value if value > 1e-9 else 0.0',
     'rho = self._pi_ewma / max(value, 1e-12)'),
    ('Y7', 'payoff', 'hold benchmark does not advance',
     'self._prev_x, self._prev_y = float(self.pool.x), float(self.pool.y)', 'pass'),

    ('C1', 'construction', 'zero outside option replaced',
     'base = float(outside_option)',
     'base = outside_option if outside_option else 1e-5'),
    ('C2', 'construction', 'entrants start active', 'e.active = False', 'e.active = True'),
    ('C3', 'construction', 'entrants start with tokens',
     'e = EndogenousLP(outside_option=opt_i, tokens=0.0,',
     'e = EndogenousLP(outside_option=opt_i, tokens=5.0,'),
    ('C4', 'construction', 'price per token does not start at one',
     'self.total_supply = v0', 'self.total_supply = 2.0 * v0'),
    ('C5', 'construction', 'patience not dispersed',
     'pat = max(3, int(round(exit_patience * (0.5 + rng.random()))))',
     'pat = exit_patience'),
    ('C6', 'construction', 'initial holdings unequal', 'tokens=v0 / n,', 'tokens=v0,'),
    ('C7', 'construction', 'incumbent providers hold no cash',
     'tokens=v0 / n,\n                wallet=wallet_ratio * v0 / n,',
     'tokens=v0 / n,\n                wallet=0.0,'),
    ('C11', 'construction', 'entrants hold no cash',
     'tokens=0.0,\n                             wallet=wallet_ratio * v0 / n,',
     'tokens=0.0,\n                             wallet=0.0,'),
    ('R2', 'redemption', 'no cap on the requested fraction',
     'frac = max(0.0, min(float(frac), 0.5))', 'frac = max(0.0, float(frac))'),
    ('R3', 'redemption', 'scarce side identified backwards',
     'scarce_is_x = vx < vy', 'scarce_is_x = vx > vy'),
    ('R4', 'redemption', 'skew ignored',
     'if abs(self.withdraw_skew) < 1e-12:', 'if True:'),
    ('R5', 'redemption', 'skew not rescaled to the value requested',
     'k = (frac * total) / denom if denom > 1e-12 else 0.0', 'k = frac'),
    ('L15', 'ledger', 'the floor becomes a price for issuing claims',
     'if value <= 1e-9:\n            for lp in self.providers:',
     'if False:\n            for lp in self.providers:'),

    # ---- fee income belongs to the providers -------------------------------
    ('F1', 'fees', 'fee income never reaches a wallet',
     'if claimed > 0.0:', 'if False:'),
    ('F2', 'fees', 'the claimable balance is never emptied',
     'claimed_base, claimed_quote = self.pool.claim_fees()',
     'claimed_base, claimed_quote = self.pool.fee_base, self.pool.fee_quote'),
    ('F5', 'fees', 'the base fee is paid as if it were quote',
     'fee = claimed_base * p + claimed_quote', 'fee = claimed_base + claimed_quote'),
    ('F6', 'fees', 'the signal is struck at a different price from the payment',
     'claimed = fee', "claimed = float(getattr(self.pool, 'period_fee_revenue', 0.0))"),
    ('F3', 'fees', 'queued tokens excluded from the fee split',
     'cut = claimed * eligible / self.total_supply',
     'cut = claimed * lp.tokens / self.total_supply'),
    ('F4', 'fees', 'fee split not pro rata',
     'cut = claimed * eligible / self.total_supply',
     'cut = claimed / max(1, len(self.providers))'),
    ('W2', 'winddown', 'a wind down does not empty the pool',
     'self._pay_out(queued, self._sweep_residual(p), price)',
     'self._pay_out(queued, self._remove(0.5), price)'),
    ('W8', 'winddown', 'the floor no longer bounds the withdrawal',
     'capped = min(paced, room)', 'capped = paced'),
    ('W9', 'winddown', 'the swept value is not reported to the payout',
     'return residual', 'return 0.0'),
    ('W3', 'winddown', 'the sweep leaves the reserves in place',
     'self.pool.x = 0.0\n        self.pool.y = 0.0', 'pass'),
    ('W4', 'winddown', 'the last tokens are not extinguished',
     'burned = queued\n        else:', 'burned = 0.0\n        else:'),
    # ``W6`` used to mutate a reassignment of the burn price on an emptied
    # pool. A wind down sweeps the reserves and burns the whole queue, so the
    # price read before the reserves moved already equals the value removed per
    # token queued. The line was redundant and has been removed rather than
    # defended by a test that could not tell the two apart.
    ('W5', 'winddown', 'closure never published to the venue',
     'self.pool.closed = bool(self.closed)', 'pass'),

    # ---- refounding --------------------------------------------------------
    ('N1', 'refounding', 'an empty pool seeded with nothing',
     'total = cash + value\n            x_new, y_new',
     'total = 0.0\n            x_new, y_new'),
    # ``N2`` used to mutate ``total = cash + value`` at a refounding. A wind
    # down now empties the reserves outright, so nothing is ever stranded and
    # ``value`` is always zero there. The mutant is equivalent by construction
    # and has been withdrawn rather than propped up by an artificial state.
    ('N3', 'refounding', 'the floor not rebased on the new pool',
     'self._initial_value = minted', 'pass'),
    ('N4', 'refounding', 'refounded tokens not split by cash committed',
     'lp.tokens += minted * (c / cash)\n                lp._mint_cash = 0.0',
     'lp.tokens += minted\n                lp._mint_cash = 0.0'),

    # ---- the record a profit and loss statement reads -----------------------
    ('C8', 'flowrecord', 'a skewed withdrawal leaves no trace',
     'self.pool.record_reserve_flow(self.pool.x - x0, self.pool.y - y0)', 'pass'),
    ('C9', 'flowrecord', 'the skewed flow recorded with the wrong sign',
     'self.pool.record_reserve_flow(self.pool.x - x0, self.pool.y - y0)',
     'self.pool.record_reserve_flow(x0 - self.pool.x, y0 - self.pool.y)'),
    ('C10', 'flowrecord', 'seeding a refounded pool leaves no trace',
     'self.pool.record_reserve_flow(x_new - self.pool.x,\n                                              y_new - self.pool.y)',
     'pass'),
]

# The original catalogue above documents defects from the quote-only ledger.
# Once the population moved to native base and quote balances, many of those
# source patterns necessarily ceased to exist.  The executable catalogue below
# targets the current two-asset implementation; retaining the historical list
# above keeps the audit trail without pretending that a missing pattern was a
# mutation caught by a test.
AUDIT_MUTATIONS = [
    ('D1', 'decide', 'sign of the excess return',
     'excess = rho - self.outside_option', 'excess = rho + self.outside_option'),
    ('D3', 'decide', 'patience threshold made strict',
     'if self._below >= self.exit_patience:', 'if self._below > self.exit_patience:'),
    ('D4', 'decide', 'exit is not signalled',
     'self.exit_count += 1\n                return -1.0',
     'self.exit_count += 1\n                return 0.0'),
    ('D5', 'decide', 'adjustment speed ignored',
     'want = self.kappa * excess / scale', 'want = excess / scale'),
    ('D6', 'decide', 'step is not clamped',
     'return max(-self.max_adj, min(self.max_adj, want))', 'return want'),
    ('D8', 'decide', 'entry margin ignored',
     'entry_excess_hurdle = self.entry_margin * abs(self.outside_option)',
     'entry_excess_hurdle = -1e18'),
    ('D10', 'decide', 'entry permitted without base inventory',
     'and self.wallet_base > 1e-12):', 'and True):'),
    ('D11', 'decide', 'entry permitted while an old burn is pending',
     'and self.pending_burn <= 1e-12', 'and True'),
    ('D12', 'decide', 'warm-up still permits position resizing',
     'if t < self.warmup:\n            self._below = 0',
     'if False:\n            self._below = 0'),
    ('D13', 'decide', 'explicit response scale is ignored',
     'scale = max(abs(self.outside_option), self.response_scale)',
     'scale = max(abs(self.outside_option), 1e-6)'),
    ('D14', 'decide', 'sizing normalisation contaminates the entry hurdle',
     'entry_excess_hurdle = self.entry_margin * abs(self.outside_option)',
     'entry_excess_hurdle = self.entry_margin * max(abs(self.outside_option), self.response_scale)'),
    ('D15', 'decide', 'entry persistence ignores the full hurdle',
     'elif excess > entry_excess_hurdle:', 'elif excess >= 0:'),

    ('L3', 'ledger', 'supply not reduced on burn',
     'self.total_supply -= burned', 'pass'),
    ('L7', 'ledger', 'queue not reduced when served',
     'lp.pending_burn -= b', 'pass'),
    ('L8', 'ledger', 'exit does not queue the tokens',
     'lp.pending_burn += lp.tokens\n                lp.tokens = 0.0',
     'lp.tokens = 0.0'),
    ('L9', 'ledger', 'partial redemption does not reduce the holding',
     'lp.pending_burn += burn\n                lp.tokens -= burn',
     'lp.pending_burn += burn'),
    ('L16', 'ledger', 'base leg is not returned on redemption',
     'lp.wallet_base += removed_base * share', 'pass'),
    ('L17', 'ledger', 'quote leg is not returned on redemption',
     'lp.wallet_cash += removed_quote * share', 'pass'),
    ('L18', 'ledger', 'minted supply is not increased',
     'self.total_supply += minted\n        self._clear_mint_requests()',
     'self._clear_mint_requests()'),
    ('L19', 'ledger', 'mint price ignores current NAV',
     'minted = served_value / price_before', 'minted = served_value'),
    ('L20', 'ledger', 'mint ignores the base-wallet constraint',
     'frac = min(lp.wallet_base / self.pool.x,\n                   lp.wallet_cash / self.pool.y)',
     'frac = lp.wallet_cash / self.pool.y'),
    ('L21', 'ledger', 'issuance cap removed',
     'served_frac = min(max(0.0, requested_frac), 0.5)',
     'served_frac = max(0.0, requested_frac)'),
    ('L22', 'ledger', 'absolute dust strands a complete wind down',
     'claim_dust = max(1e-9, 1e-12 * self.total_supply)',
     'claim_dust = 1e-9'),

    ('S2', 'sponsor', 'providers are not credited',
     'lp.wallet_cash += cut\n                    lp.subsidy_received += cut',
     'lp.subsidy_received += cut'),
    ('S4', 'sponsor', 'sponsor account not debited',
     'self.sponsor_account_quote -= subsidy_cash', 'pass'),
    ('S5', 'sponsor', 'sign of the subsidy in the signal',
     'effective_period_return = (net + subsidy_cash) / hold',
     'effective_period_return = (net - subsidy_cash) / hold'),
    ('S6', 'sponsor', 'loss rebate is not paid',
     'loss_rebate_cash = (self.loss_rebate_fraction\n'
     '                                * max(0.0, -net))',
     'loss_rebate_cash = 0.0'),
    ('S7', 'sponsor', 'loss rebate rewards positive payoff instead',
     '* max(0.0, -net))', '* max(0.0, net))'),

    ('Y1', 'payoff', 'sign of the rebalancing loss',
     'lvr = hold - value', 'lvr = value - hold'),
    ('Y2', 'payoff', 'sign of the net payoff',
     'net = fee - lvr', 'net = fee + lvr'),
    ('Y3', 'payoff', 'fee income ignored',
     'net = fee - lvr', 'net = -lvr'),
    ('Y4', 'payoff', 'EWMA weights inverted',
     'self._rho_ewma = (1.0 - a) * self._rho_ewma + a * period_return',
     'self._rho_ewma = a * self._rho_ewma + (1.0 - a) * period_return'),
    ('Y5', 'payoff', 'cash P&L is smoothed instead of a return',
     'period_return = net / hold if hold > 1e-9 else 0.0',
     'period_return = net'),
    ('Y7', 'payoff', 'hold benchmark does not advance',
     'self._prev_x, self._prev_y = float(self.pool.x), float(self.pool.y)',
     'pass'),

    ('C1', 'construction', 'zero outside option replaced',
     'base = float(outside_option)',
     'base = outside_option if outside_option else 1e-5'),
    ('C2', 'construction', 'entrants start active',
     'e.active = False', 'e.active = True'),
    ('C4', 'construction', 'token price does not start at one',
     'self.total_supply = v0', 'self.total_supply = 2.0 * v0'),
    ('C5', 'construction', 'patience not dispersed',
     'pat = max(3, int(round(exit_patience * (0.5 + rng.random()))))',
     'pat = exit_patience'),
    ('C6', 'construction', 'entrant patience not dispersed',
     'entrant_pat = max(3, int(round(\n'
     '                exit_patience * (0.5 + rng.random())\n'
     '            )))',
     'entrant_pat = exit_patience'),
    ('C7', 'construction', 'incumbent base buffer removed',
     'wallet=buffer_quote,\n                wallet_base=buffer_base,',
     'wallet=buffer_quote,\n                wallet_base=0.0,'),

    ('F1', 'fees', 'fee income never reaches native wallets',
     'if fee > 0.0:', 'if False:'),
    ('F2', 'fees', 'claimable balances are never emptied',
     'claimed_base, claimed_quote = self.pool.claim_fees()',
     'claimed_base, claimed_quote = self.pool.fee_base, self.pool.fee_quote'),
    ('F5', 'fees', 'base fee is valued as one unit of quote',
     'fee = claimed_base * p + claimed_quote',
     'fee = claimed_base + claimed_quote'),
    ('F7', 'fees', 'base fee is credited to the quote wallet',
     'lp.wallet_base += cut_base\n                    lp.wallet_cash += cut_quote',
     'lp.wallet_cash += cut_base + cut_quote'),
    ('F8', 'fees', 'quote fee is credited to the base wallet',
     'lp.wallet_cash += cut_quote\n                    lp.fees_received_base += cut_base',
     'lp.wallet_base += cut_quote\n                    lp.fees_received_base += cut_base'),

    ('R2', 'redemption', 'redemption cap removed',
     'frac = max(0.0, min(float(frac), 0.5))',
     'frac = max(0.0, float(frac))'),
    ('R3', 'redemption', 'scarce reserve identified backwards',
     'scarce_is_x = vx < vy', 'scarce_is_x = vx > vy'),
    ('R4', 'redemption', 'withdrawal skew ignored',
     'if abs(self.withdraw_skew) < 1e-12:', 'if True:'),
    ('W2', 'winddown', 'wind down does not sweep the pool',
     'self._pay_out(queued, self._sweep_residual(), price, p)',
     'self._pay_out(queued, self._remove(0.5), price, p)'),
    ('W3', 'winddown', 'sweep leaves reserves in place',
     'self.pool.x = 0.0\n        self.pool.y = 0.0', 'pass'),
    ('W5', 'winddown', 'closure never reaches venue',
     'self.pool.closed = bool(self.closed)', 'pass'),
]


def run_suite():
    """Return (returncode, passed, failed). Counts are None if unparseable."""
    r = subprocess.run([sys.executable, SUITE], capture_output=True,
                       text=True, timeout=300, cwd=ROOT)
    m = re.search(r'passed (\d+), failed (\d+)', r.stdout)
    if not m:
        return r.returncode, None, None
    return r.returncode, int(m.group(1)), int(m.group(2))


def repair_if_interrupted():
    """Restore the module if a previous run was killed before it could.

    Signal handlers cover termination but not ``SIGKILL``, which cannot be
    caught, so a run that is killed outright leaves a mutated module behind.
    The pristine copy is written to disk before any mutation and removed on a
    clean finish, so its presence at startup means the previous run died and
    the module has to be put back before anything is measured.
    """
    if os.path.exists(PRISTINE):
        shutil.copy(PRISTINE, MOD)
        os.unlink(PRISTINE)
        print('a previous run was interrupted, the module has been restored\n')
        return True
    return False


def verify_baseline():
    """Refuse to score anything unless the unmutated suite really runs green."""
    problems = []
    if not os.path.exists(SUITE):
        problems.append(f'test suite missing at {SUITE}')
    if not os.path.exists(MOD):
        problems.append(f'module missing at {MOD}')
    if problems:
        return problems, None
    code, passed, failed = run_suite()
    if passed is None:
        problems.append('baseline produced no parseable summary')
    else:
        if failed:
            problems.append(f'baseline is not green, {failed} checks failed')
        if code != 0:
            problems.append(f'baseline exit code is {code}, expected 0')
        if passed < MIN_CHECKS:
            problems.append(f'baseline ran only {passed} checks, expected at '
                            f'least {MIN_CHECKS}, the suite may not be running')
    return problems, passed


def classify(pattern_found, code, passed, failed, base_passed):
    if not pattern_found:
        return ERROR, 'pattern absent, nothing was mutated'
    if passed is None:
        return ((CAUGHT, 'module crashed') if code != 0
                else (ERROR, 'no summary and a zero exit code'))
    if failed > 0:
        return CAUGHT, f'{failed} check(s) failed'
    if code != 0:
        return CAUGHT, f'exit code {code}'
    if passed < base_passed:
        return ERROR, f'only {passed} of {base_passed} checks ran'
    return SURVIVED, 'suite stayed green'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-check', action='store_true',
                    help='verify that the audit refuses to score without a suite')
    args = ap.parse_args()

    repair_if_interrupted()
    problems, base_passed = verify_baseline()
    if problems:
        print('AUDIT ABORTED, the baseline is not trustworthy:')
        for p in problems:
            print('  *', p)
        return 2
    print(f'baseline green, {base_passed} checks\n')

    if args.self_check:
        hidden = tempfile.mktemp(suffix='.py')
        shutil.move(SUITE, hidden)
        try:
            probs, _ = verify_baseline()
            ok = bool(probs)
            print('self check with the suite removed:',
                  'audit correctly refuses to score' if ok else 'AUDIT STILL SCORES')
            for p in probs:
                print('    reason:', p)
        finally:
            shutil.move(hidden, SUITE)
        return 0 if ok else 1

    backup = PRISTINE
    shutil.copy(MOD, backup)

    # A ``finally`` block does not run when the process is terminated by a
    # signal, and a run interrupted part way through would then leave a mutated
    # copy of the module in the tree. That happened once and the defect sat
    # there until the test suite caught it. The restore is therefore also
    # registered at exit and on the usual termination signals.
    def _restore(*_a):
        try:
            if os.path.exists(backup):
                shutil.copy(backup, MOD)
        except Exception:
            pass

    atexit.register(_restore)
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, lambda s, f: (_restore(), sys.exit(130)))
        except Exception:
            pass

    results = []
    try:
        pristine = open(backup).read()
        # ``str.replace(old, new, 1)`` rewrites the first occurrence, which is
        # not necessarily the branch a mutation names. An ambiguous pattern
        # therefore reports on one place while claiming another, and the run
        # is not a result at all. Every pattern is required to be unique.
        srcs = {f: open(f).read() for f in EXTRA_FILES}
        ambiguous = [(mid, pristine.count(old))
                     for mid, _, _, old, _ in AUDIT_MUTATIONS
                     if pristine.count(old) != 1]
        ambiguous += [(mid, sum(t.count(old) for t in srcs.values()))
                      for mid, _, _, old, _ in RULE_MUTATIONS
                      if sum(t.count(old) for t in srcs.values()) != 1]
        if ambiguous:
            print('These patterns do not identify one place in the module, so a '
                  'mutation would not land where it says it does.')
            for mid, n in ambiguous:
                print(f'  {mid}: {n} matches')
            return 2
        print(f"{'id':4s} {'area':13s} {'mutation':38s} {'verdict':9s} detail")
        print('-' * 100)
        extra = {}
        for path in EXTRA_FILES:
            extra[path] = open(path).read()
            shutil.copy(path, path + '.pristine')
        for mid, area, desc, old, new in AUDIT_MUTATIONS + RULE_MUTATIONS:
            if area == 'ruleprovider':
                target = next((f for f in EXTRA_FILES if old in extra[f]), MOD_RULE)
                source = extra[target]
            else:
                target, source = MOD, pristine
            found = old in source
            if found:
                with open(target, 'w') as f:
                    f.write(source.replace(old, new, 1))
            code, passed, failed = run_suite()
            shutil.copy(target + '.pristine' if area == 'ruleprovider' else backup,
                        target)
            verdict, detail = classify(found, code, passed, failed, base_passed)
            results.append((mid, area, desc, verdict, detail, failed))
            print(f'{mid:4s} {area:13s} {desc:38s} {verdict:9s} {detail}')
    finally:
        shutil.copy(backup, MOD)
        if os.path.exists(backup):
            os.unlink(backup)
        for path in EXTRA_FILES:
            rb = path + '.pristine'
            if os.path.exists(rb):
                shutil.copy(rb, path)
                os.unlink(rb)

    caught = [r for r in results if r[3] == CAUGHT]
    survived = [r for r in results if r[3] == SURVIVED]
    errors = [r for r in results if r[3] == ERROR]
    print('-' * 100)
    print(f'mutations {len(results)}, caught {len(caught)}, '
          f'survived {len(survived)}, errors {len(errors)}')
    thin = [r for r in caught if r[5] == 1]
    if thin:
        print(f'\ncaught by a single check ({len(thin)}), thin protection:')
        for r in thin:
            print(f'  {r[0]}  [{r[1]}]  {r[2]}')
    if survived:
        print('\ngaps in coverage:')
        for r in survived:
            print(f'  {r[0]}  [{r[1]}]  {r[2]}')
    if errors:
        print('\nerrors, these are not results:')
        for r in errors:
            print(f'  {r[0]}  [{r[1]}]  {r[2]}  ({r[4]})')
    return 1 if (survived or errors) else 0


if __name__ == '__main__':
    sys.exit(main())
