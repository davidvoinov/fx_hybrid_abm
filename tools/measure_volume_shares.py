"""Measure realized volume share per agent class for the current baseline.

Runs the calibrated calm-state simulation for N seeds, walks every trader at
the end of each run, aggregates `quantity` from `self.trades`, and reports the
share of total turnover per agent class. The result is compared against the
BIS Triennial 2025 counterparty composition (spot).

The BIS taxonomy and our internal agent-class taxonomy do not map 1-to-1, so
the script reports both:
  - raw per-agent-class shares (no BIS mapping)
  - aggregated shares using the cleanest available BIS-correspondence mapping

Usage:
    python tools/measure_volume_shares.py --seeds 30 --n-iter 900
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AgentBasedModel.agents.agents import (
    Random, Fundamentalist, Chartist, Universalist, MarketMaker,
    FastRecyclerLP, LatentLP, Trader,
)
from main import build_parser, build_sim, _seed_all, _apply_preset_defaults  # noqa: F401


# ---------------------------------------------------------------------------
# Monkey-patch Trader.apply_fill to track provider-side volume (limit-fill).
# Without this, MM and LP volumes appear as 0 because self.trades only records
# orders the agent INITIATED (market orders / AMM hits), not limit orders that
# were filled when somebody else hit them.
# ---------------------------------------------------------------------------

_ORIGINAL_APPLY_FILL = Trader.apply_fill


def _apply_fill_with_volume_tracking(self, order, fill_qty, fill_price,
                                     t_cost, is_buy, qty_before):
    if not hasattr(self, '_provider_fill_volume'):
        self._provider_fill_volume = 0.0
    self._provider_fill_volume += float(fill_qty)
    return _ORIGINAL_APPLY_FILL(self, order, fill_qty, fill_price,
                                t_cost, is_buy, qty_before)


Trader.apply_fill = _apply_fill_with_volume_tracking


# BIS Triennial 2025 spot turnover counterparty shares (% of global)
# Source: https://www.bis.org/statistics/rpfx25_fx.pdf, Graph 3.C
BIS_2025_SHARES = {
    'Reporting dealers':      40.0,   # spot ~40% (overall 46%)
    'Non-reporting banks':    26.0,   # spot (~24% overall)
    'Institutional investors': 14.0,  # spot (~13% overall)
    'Hedge funds + PTFs':      9.0,   # spot (~8% overall)
    'Official sector':         3.0,
    'Non-financial':           5.0,
    'Retail / Other':          3.0,
}


def _classify_agent(agent) -> str:
    """Return our internal class label for an agent."""
    if isinstance(agent, MarketMaker):
        return 'MarketMaker'
    if isinstance(agent, FastRecyclerLP):
        return 'FastRecyclerLP'
    if isinstance(agent, LatentLP):
        return 'LatentLP'
    if isinstance(agent, Universalist):
        return 'Universalist'
    if isinstance(agent, Chartist):
        return 'Chartist'
    if isinstance(agent, Fundamentalist):
        # Distinguish CLOB-book Fundamentalist vs FX-fund Fundamentalist
        # FX-fund uses flow_role='LeveragedDirectional'
        role = getattr(agent, 'flow_role', '') or ''
        if 'Directional' in role or 'Leveraged' in role:
            return 'FX_Fundamentalist'
        return 'CLOB_Fundamentalist'
    if isinstance(agent, Random):
        # Use label / flow_role to distinguish Hedger / Retail / Institutional / Noise
        label = getattr(agent, 'label', '') or ''
        role = getattr(agent, 'flow_role', '') or ''
        tag = (label + '|' + role).lower()
        if 'retail' in tag:
            return 'Retail'
        if 'realmoney' in tag or 'institutional' in tag:
            return 'Institutional'
        if 'hedger' in tag or 'taker' in tag:
            return 'FX_Hedger_Taker'
        return 'Noise'
    return f'Unknown({type(agent).__name__})'


# Mapping from our agent classes to BIS 2025 counterparty groups.
# This is best-effort and documented in the article methodology.
AGENT_TO_BIS = {
    'MarketMaker':         'Reporting dealers',
    'FastRecyclerLP':      'Hedge funds + PTFs',     # PTF-like HFT intermediation
    'LatentLP':            'Non-reporting banks',    # reserve / regional bank LPs
    'CLOB_Fundamentalist': 'Institutional investors',
    'Chartist':            'Hedge funds + PTFs',     # directional algo / trend
    'Universalist':        'Hedge funds + PTFs',
    'FX_Fundamentalist':   'Institutional investors',
    'FX_Hedger_Taker':     'Institutional investors',  # macro asset managers / hedgers
    'Retail':              'Retail / Other',
    'Institutional':       'Non-financial',           # corporates / RealMoney
    'Noise':               'Retail / Other',
}


def measure_one_run(sim) -> Dict[str, float]:
    """Aggregate trade volume per agent class for a single simulation.

    Volume = initiator side (own market orders / AMM hits via self.trades) +
             provider side (limit orders that were filled, captured via the
             apply_fill monkey-patch into self._provider_fill_volume).

    This corresponds to BIS gross turnover (both sides counted), which is the
    natural unit before the inter-dealer net-net adjustment.
    """
    volume_by_class: Dict[str, float] = defaultdict(float)

    all_agents = []
    if sim.mm is not None:
        all_agents.append(sim.mm)
    all_agents.extend(sim.book_agents)
    all_agents.extend(sim.fx_traders)
    all_agents.extend(sim.lp_providers)

    for agent in all_agents:
        cls = _classify_agent(agent)
        # Initiator side
        trades = getattr(agent, 'trades', []) or []
        for tr in trades:
            qty = float(tr.get('quantity', 0.0) or 0.0)
            volume_by_class[cls] += qty
        # Provider side (limit fills hit by counterparties)
        volume_by_class[cls] += float(getattr(agent, '_provider_fill_volume', 0.0) or 0.0)

    return dict(volume_by_class)


def _shares(volumes: Dict[str, float]) -> Dict[str, float]:
    total = sum(volumes.values())
    if total <= 0:
        return {k: 0.0 for k in volumes}
    return {k: 100.0 * v / total for k, v in volumes.items()}


def _aggregate_by_bis(class_shares: Dict[str, float]) -> Dict[str, float]:
    bis = defaultdict(float)
    for cls, share in class_shares.items():
        bis_group = AGENT_TO_BIS.get(cls, 'Retail / Other')
        bis[bis_group] += share
    return dict(bis)


def main() -> None:
    parser = build_parser()
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--out-dir', default='output/volume_shares')
    args = parser.parse_args()

    # build_parser already loads defaults from calibration/primary_model.json,
    # so without --preset we get the calibrated baseline as-is.

    os.makedirs(args.out_dir, exist_ok=True)

    per_seed_class_volumes: List[Dict[str, float]] = []
    print(f'Running {args.seeds} seeds × {args.n_iter} ticks for baseline calm scenario...')
    for i in range(args.seeds):
        seed = args.base_seed + i
        _seed_all(seed)
        sim = build_sim(args)
        sim.simulate(args.n_iter, silent=True)
        v = measure_one_run(sim)
        per_seed_class_volumes.append(v)
        total = sum(v.values())
        if (i + 1) % 5 == 0 or i == args.seeds - 1:
            print(f'  seed {i+1}/{args.seeds} done (total volume = {total:.1f})')

    # Aggregate
    all_classes = sorted({c for v in per_seed_class_volumes for c in v})
    means_by_class: Dict[str, float] = {}
    for cls in all_classes:
        vols = [v.get(cls, 0.0) for v in per_seed_class_volumes]
        means_by_class[cls] = sum(vols) / len(vols) if vols else 0.0

    class_shares = _shares(means_by_class)
    bis_shares = _aggregate_by_bis(class_shares)

    # Sort by share descending
    sorted_classes = sorted(class_shares.items(), key=lambda kv: -kv[1])
    sorted_bis = sorted(bis_shares.items(), key=lambda kv: -kv[1])

    print()
    print('=' * 70)
    print('Realized volume share per agent class (% of total turnover)')
    print('=' * 70)
    print(f'{"Class":<24} {"Share %":>10} {"Mean vol":>14}')
    print('-' * 50)
    for cls, share in sorted_classes:
        print(f'{cls:<24} {share:>9.2f}% {means_by_class[cls]:>14.1f}')
    print('-' * 50)
    print(f'{"TOTAL":<24} {sum(class_shares.values()):>9.2f}% {sum(means_by_class.values()):>14.1f}')

    print()
    print('=' * 70)
    print('Aggregated by BIS Triennial 2025 counterparty group')
    print('=' * 70)
    print(f'{"BIS group":<28} {"Model share":>12} {"BIS 2025 spot":>16} {"Δ":>8}')
    print('-' * 70)
    for grp, share in sorted_bis:
        bis = BIS_2025_SHARES.get(grp, 0.0)
        delta = share - bis
        marker = ' ⚠' if abs(delta) > 10 else ''
        print(f'{grp:<28} {share:>11.2f}% {bis:>15.2f}% {delta:>+7.1f}{marker}')
    # Add missing BIS groups
    for grp in BIS_2025_SHARES:
        if grp not in bis_shares:
            print(f'{grp:<28} {0:>11.2f}% {BIS_2025_SHARES[grp]:>15.2f}% {-BIS_2025_SHARES[grp]:>+7.1f} ⚠ (missing)')

    # CSV outputs
    csv_class_path = os.path.join(args.out_dir, 'volume_shares_by_class.csv')
    with open(csv_class_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agent_class', 'mean_volume', 'share_pct'])
        for cls, share in sorted_classes:
            w.writerow([cls, means_by_class[cls], share])
    print(f'\nSaved per-class shares → {csv_class_path}')

    csv_bis_path = os.path.join(args.out_dir, 'volume_shares_by_bis.csv')
    with open(csv_bis_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bis_group', 'model_share_pct', 'bis_2025_share_pct', 'delta_pct'])
        for grp in BIS_2025_SHARES:
            model = bis_shares.get(grp, 0.0)
            w.writerow([grp, model, BIS_2025_SHARES[grp], model - BIS_2025_SHARES[grp]])
    print(f'Saved BIS comparison    → {csv_bis_path}')


if __name__ == '__main__':
    main()
