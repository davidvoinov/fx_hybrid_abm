# Response to reviewers

**Manuscript.** FX Liquidity Resilience. Can an Always On AMM Backstop Dealer Withdrawal?

**Context.** The manuscript was reviewed at Finance Research Letters and is being transferred to Economic Modelling. Under the Elsevier Article Transfer Service the reviewer reports travel with the submission, so this document states plainly what the reviewers raised and what has changed in response. Reviewer 1 recommended acceptance conditional on one question. Reviewer 2 recommended rejection on grounds that the identification of the effect was not established. We treat Reviewer 2's central objection as correct and have rebuilt the experimental design around it rather than defending the original comparison.

**Status of this document.** It distinguishes what has already been changed in the manuscript from
what requires simulation runs that have not yet been carried out. We do not describe planned runs as
though they had been completed.

**Already changed in the manuscript.**

1. Liquidity provider profit and loss has moved from the supplement into the main text as a primary
   result, with the reserve constraint and the behaviour of the pool near depletion described explicitly.
2. The primary outcome is now formally defined for a hybrid market, as the best executable round trip
   cost across venues at a stated order size, inclusive of fees.
3. Resilience is now defined as a post shock decay rate conditional on the initial displacement,
   with threshold crossing measures demoted to a secondary check.
4. The novelty claim has been narrowed against the literature the reviewer identifies, and the whole
   reference list has been audited. Three broken DOIs were found and corrected.
5. Claims that the facility cannot withdraw have been narrowed to the statement that it cannot
   decline to quote at given reserves.
6. Language characterising the subsidy as small or the trade off as favourable has been removed.

**Not yet done, and stated as such.**

The resource matched comparison that Reviewer 2 asks for in point 1 has been designed but not run.
The manuscript sets out the design in its conclusion and reports no estimate from it. We considered it
preferable to state the identification limit plainly than to present a planned experiment as a result.

---

## Reviewer 1---

## Reviewer 1

> This study should answer the simple question whether the automated market makers lose money in the volatile situation or not.

We thank the reviewer for identifying the single most consequential omission. The analysis existed but was placed in the supplement, where it was easy to miss, and the main text did not state the answer.

**The answer is yes, the pool loses money, and the loss is larger in stress.** Marking reserves against a hold and rebalance benchmark and crediting the fee revenue the pool actually earns, the provider loses about 0.14 percent of pool value over a calm window and about 0.38 percent over the crisis window at the 5 basis point benchmark fee. The loss shrinks as the fee rises, reaching about 0.21 percent in crisis at 20 basis points, with the calm window close to breakeven. This is far below the frictionless loss versus rebalancing bound, which assumes continuous arbitrage that the crisis itself caps.

**Changes made.** A dedicated section on provider economics now appears in the main text rather than the supplement, reporting the profit and loss decomposition, the fee frontier, and the resulting position that the venue is a subsidised facility rather than a self financing venue. We also connect this to the endogenous fee and inventory mechanism of Hasbrouck, Rivera and Saleh, which shows that provider compensation and liquidity supply are jointly determined, and we now discuss why our fixed reserve assumption makes our fee comparative static an incomplete description of that channel.

We are also grateful for the framing reference to Gilbert. We now cite it explicitly when setting the level of abstraction, which clarifies that the model is intended to expose a mechanism rather than to replicate the market.

---

## Reviewer 2

### Point 1. The comparison is not resource equivalent

**We accept this.** The objection is correct as stated, and our own fixed share placebo pointed in the
same direction. The original design added capital in the treatment arm without adding it in the control,
so the reported improvement pooled three effects that we described as one.

**What has changed in the manuscript.** We no longer claim to identify an effect specific to the pricing
rule. The headline comparison is now presented as what it is, namely the combined effect of committed
capital, an obligation to remain available, and a reserve based schedule. The introduction, the results
section and the conclusion all state this limitation explicitly rather than in passing.

**What the existing evidence does support.** The fixed share placebo, in which the facility stays open
but flow is allocated without regard to cost, still cushions the crisis. This bounds the contribution of
standing availability from one side and shows that a meaningful part of the effect is generic to any
facility that does not withdraw. It does not complete the separation, because the placebo arm still
carries capital the control arm lacks, and we say so.

**What remains to be done.** The conclusion sets out the resource matched design in full, namely a
committed dealer of last resort quoting a single spread, a passive order book facility with matched
capital and inventory limits, and a reallocation arm in which dealer capital is moved rather than
supplemented, all at equal subsidy. The quantity that identifies the pricing rule is the difference
between the reserve priced arm and the dealer of last resort arm at equal capital. Those runs have not
been carried out and no estimate from them appears in the paper.

### Point 2.### Point 2. The claim that the AMM cannot withdraw is too strong

**We accept this.** The contract persists but the capital need not, and a finite pool can be exhausted on one side.

**Changes made.** The claim has been narrowed throughout to the statement that the venue cannot withdraw quotes at given reserves. The main text now describes explicitly what happens as a reserve approaches its constraint and how the price schedule steepens. We have removed the phrasing that the pool quotes every size at every instant. We note candidly that endogenous provider entry and exit, reserve depletion and replenishment, and the subsidy required to prevent a run on the pool remain outside the model, and we have added this to the limitations rather than leaving the availability assumption implicit.

### Point 3. H2 and H3 are built into the routing rule and the chosen curve parameters

**We partly accept this.** The direction of the migration result does follow from the routing rule, and we have removed language suggesting that it constitutes independent evidence. The rule does not, however, pin the magnitude of the shift or the reversion after the shock, and we now present these as quantitative consistency rather than as a test.

On H3 we accept that the crossover location depends on amplification and fee. We now make a different and stronger use of the result. Barbon and Ranaldo document empirically that decentralised venues are relatively more competitive for larger trades. Our model, calibrated only to calm state FX moments and never to any crossover, reproduces that pattern. We therefore present the tail confined advantage as an out of sample validation of the mechanism rather than as a novel finding, which also addresses part of point 4.

### Point 4. The calibration is insufficient for the quantitative interpretation

**We largely accept this.** On preregistration we have checked the project history rather than relying
on recollection. The acceptance bands were fixed in a version stamped targets file committed to the
repository before the paired runs reported here were produced, so the ordering is verifiable from the
version history. No protocol was deposited in an external registry, so we no longer use the word
preregistered and now describe the bands as design targets fixed in advance, which is the claim the
evidence supports.

The manuscript now also states plainly which quantities are disciplined by data and which are design
choices, in a dedicated subsection. The facility share in calm, the amplification and fee, the
withdrawal threshold and the stress score dynamics fall in the second group.

Out of sample validation against an identifiable FX stress episode is flagged in the manuscript as
outstanding work and has not been completed. We note that the model does reproduce, without ever having
been calibrated to it, the ordering documented by Barbon and Ranaldo whereby reserve priced venues are
relatively more competitive for large trades.

### Point 5.### Point 5. The primary outcome is not well defined for a hybrid market

**We accept this without reservation.** This was an expositional failure caused in part by the length limit of the original outlet.

**Changes made.** The hybrid spread is now defined explicitly as the best executable round trip cost across venues for a stated order size, inclusive of pool fees, with the aggregation rule given in the main text. All headline comparisons state the order size. Because relative cost varies with size, we report the primary outcome across a range of sizes rather than at a single one.

### Point 6. The inference addresses Monte Carlo variation, not model uncertainty

**We partly accept this.** We agree that the intervals in the original table describe variation across random seeds conditional on the model, and we have removed the word confirmatory and the language implying that the tests establish what would happen in an actual crisis.

We would gently observe that the objection, applied literally, would deny statistical inference to computational models as a class, since every simulated result is conditional on its equations. Our response is therefore to separate the sources of uncertainty rather than to drop the statistics. The revised paper reports Monte Carlo error, parameter uncertainty and structural sensitivity in distinct rows, so that the reader can see how much of the spread comes from each. We follow Gilbert in treating the model as a device for isolating a mechanism.

### Point 7. The recovery analysis mixes amplitude with dynamics

**We accept this.** The threshold criteria are mechanically easier to satisfy when the peak is lower,
and describing the divergence as a normalisation artifact did not resolve the issue.

**What has changed.** The manuscript now defines resilience as the post shock decay rate estimated
conditional on the initial displacement, together with the normalised impulse response, and demotes
threshold crossing measures to a secondary check. The recomputation of the reported figures under this
definition, from the existing paired runs, is outstanding and is flagged in the text.

### Point 8.### Point 8. The welfare and policy interpretation goes beyond the analysis

**We accept this.** The words small, tunable and favourable have been removed as characterisations of the subsidy.

**What has changed.** The provider loss and the fee frontier now form a full section rather than a
closing paragraph, and the sponsor question is treated as part of the institutional design rather than
as implementation detail. The manuscript sets out the accounting that would place the resilience gain
and the provider cost in a common frame, as expected annual quantities under a crisis frequency, with
the opportunity cost of committed reserves. Populating that accounting requires the volume distribution
by trade size and an assumed crisis frequency, and it has not been completed. The paper therefore
reports the provider loss and refrains from any claim that the trade off is favourable.

### Point 9. The novelty statement should be narrowed and the references verified

**We accept this in full, and we are grateful for the correction.**

**Bibliographic errors.** Both errors are confirmed. The DOI attached to Lehar and Parlour resolved to an unrelated paper on mortgages and labour supply, and the correct identifier is 10.1111/jofi.13405. The Hasbrouck, Rivera and Saleh article was listed as a 2024 Journal of Financial Economics paper with a DOI that is not registered at all, and it is Management Science, published online in 2026, with identifier 10.1287/mnsc.2023.00726. Following the reviewer's advice we audited the entire reference list against Crossref and the doi.org registry and found a third defect, a Management Science DOI attached to Capponi and Jia that is likewise unregistered, since that paper is still a working paper. Publication years, volumes and pages have been corrected throughout.

**Novelty.** The claim has been narrowed. We no longer assert that the coexistence of order books and automated venues is unexamined. We now state that coexistence has been characterised in equilibrium by Aoyagi and Ito, that relative venue quality including the large trade advantage has been documented empirically by Barbon and Ranaldo, and that the joint determination of fees, inventory and volume has been modelled by Hasbrouck, Rivera and Saleh. We have added the two studies applying automated market making to traditional markets, by Foley, O'Neill and Putniņš and by Malinova and Park. Against that literature we now claim something narrower and, we believe, defensible. The contribution is the resource matched decomposition of a standing liquidity backstop in a dealer intermediated FX setting with endogenous dealer withdrawal, evaluated on resilience rather than on average cost. To our knowledge no existing study holds capital and subsidy constant while comparing backstop designs.

---

## Items we have not resolved

We prefer to state these plainly rather than imply completeness.

- Provider entry and exit and reserve replenishment remain exogenous. Endogenising them is the natural next step and would change the fee comparative static, as the Hasbrouck, Rivera and Saleh mechanism implies.
- The oracle remains pegged to the latent fair price in the baseline, with degradation examined only as a robustness check.
- The analysis covers a single currency pair, so the commonality measure remains weakly informative and is not presented as a finding.
