# FX ABM Model

Multi-venue FX agent-based model. A dealer CLOB coexists with an automated venue (HFMM with StableSwap invariant; CPMM kept as benchmark).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## `main.py` — calibrated baseline run

Runs the model once with the **calibrated baseline** stored in `calibration/primary_model.json`. The baseline is **calibrated on output moments** drawn from the FX-microstructure literature, not on agent counts. Targets (acceptance bands in `calibration/primary_model_targets.json`):

- Median quoted spread, 1.5–2.5 bps — Ranaldo & Santucci de Magistris (2022, JFE)
- Near-touch depth, 220 ± 30 % — Lo & Hall (2015, JEDC)
- Order-flow autocorrelation, 0.03–0.10 — Hasbrouck-style order-flow data
- Impact curve, 0.5 ± 50 % — Hasbrouck price-impact studies
- Cross-venue basis HFMM↔CLOB, 3–25 bps — DEX-CLOB coexistence literature
- Recovery half-life, 20 ± 35 % ticks — Lo & Hall (2015)
- Funding stress propagation, 0.4 ± 35 % — Brunnermeier & Pedersen (2009)

Many (count × activation × size) configurations could in principle produce the same moments; the baseline is one specific choice that hits the bands.

```bash
python main.py                                       # calibrated run, no shock
python main.py --preset dealer_liquidity_crisis      # stress preset
```

Output: `output/main_aware/` (`primary_acceptance_report.json` + diagnostic CSV/PNG).

## `tests/resilience_test.py` — event-time resilience study

Runs **300 paired Monte Carlo seeds** at the calibrated composition, across stress scenarios. For each seed, runs with-AMM and without-AMM with the **same random stream**, then characterises the post-shock dynamics of spread, depth, execution cost, and price.

Methodology in plain terms:

- **Recovery metric** = "absorb 80 % of the peak post-shock dislocation and stay there for 10 consecutive ticks". Anchored to each run's own peak — invariant to baseline-level differences between the two arms.
- **Paired design**: Δ = with − without on the same seed, removing stochastic noise.
- **Inference**: paired permutation test (10 000 sign-flips) for differences; bootstrap 95 % CI (2 000 resamples) for levels.
- **Scale-free impact metrics** alongside recovery time: `peak_abs_change_pct`, `normalized_avg_impact`, `normalized_auc_abs`.

```bash
python tests/resilience_test.py --seeds 300 --n-iter 900 --out-dir output/resilience
```

Output: `output/resilience/` (subfolders `scatter/`, `km_curves/`, `peak_impact/`, `normalized_impact/`, `recovery_boxplots/` + CSV).

## `tests/stat_tests.py` — scenario-level RQ tests

Same 300-seed paired design as the resilience study, but the comparison is on **aggregate scenario outcomes** rather than event-time metrics. Two research questions:

- **RQ1**: does AMM presence improve CLOB quality (spread, depth, vol, cost, recovery)?
- **RQ2**: does AMM presence change dealer behaviour and post-shock recovery (MM active / defensive / withdrawn shares, post-shock spread/depth)?

Also runs an **H2 phase analysis** — for every scenario splits the run into before / during / after shock and tracks how AMM share, dealer withdrawal score and liquidity spillover evolve across phases.

```bash
python tests/stat_tests.py --seeds 300 --n-iter 900 --out-dir output/stat_tests
```

Output: `output/stat_tests/` (`rq1/`, `rq2/`, `h2_phase/`, `mm_behavior/` + CSV).

## `tools/composition_sweep.py` — composition robustness sweep

Tests whether results hold across a wide range of alternative agent compositions, not just the calibrated baseline.

Mechanics:

1. **Sample**: `tools/composition_lhs.py` produces 200 agent configurations via Latin Hypercube Sampling over the 11-dim composition space (n_mm, n_fast_lp, n_latent_lp, n_clob_fund/chart/univ, n_fx_takers/fund, n_retail, n_institutional, n_noise). Each LHS point is one full agent composition.
2. **AMM parameters fixed.** Only agent counts vary; HFMM amplification, fees, reserves stay on calibrated defaults. Any observed differences come from ecology, not AMM design.
3. **Per LHS point**: 50 paired seeds × 3 scenarios (`default`, `dealer_liquidity_crisis`, `high_vol_stress`) × 2 branches (with/without AMM, same seed) = 300 sims per point. Total 60 000 sims.
4. **Stability filter**: each run checked against `calibration/stability_criteria.json` (spread bounded, depth non-trivial, finite vol, ...). Points whose runs violate the criteria are excluded from the AMM-effect analysis.
5. **Inference per LHS point**: paired permutation test (5 000 sign-flips) + bootstrap 95 % CI (2 000 resamples) on the per-seed Δ. So we get not just "mean Δ across compositions" but also "what fraction of compositions show statistically significant Δ".

Parallel execution: the 200 points can be split across multiple machines. Each runs its `--chunk-id` with `--total-chunks N`; supports `--resume` after interruption.

```bash
# Single machine, full run:
python tools/composition_sweep.py --out output/composition/results.csv

# Split across 4 machines (chunk-id 0..3 on each):
python tools/composition_sweep.py --chunk-id 0 --total-chunks 4 \
    --out output/composition/results_chunk_0_of_4.csv

# Merge + analyse after all chunks land:
python tools/composition_analysis.py \
    --results-dir output/composition --out-dir output/composition/analysis
```

Output: `output/composition/` (raw CSV per chunk + `analysis/` with stability diagnostic, per-composition Δ table, regression summary, threshold bars, fraction-significant plots, heatmaps, projections, marginals, importance).

## Repository layout

```
AgentBasedModel/         model code (agents, environment, venues, simulator, metrics, visualization)
calibration/             primary_model.json, primary_model_targets.json, stability_criteria.json, fitter.py
tests/                   unit tests, resilience_test.py, stat_tests.py
tools/                   composition sweep + plot regenerators
output/                  generated artifacts (main_aware, resilience, stat_tests, composition)
```
