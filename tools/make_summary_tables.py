"""Generate paper-ready summary tables (CSV + ASCII).

Produces 5 tables consolidating the headline numbers from all three pipelines:

  Table 1: Calibration acceptance — pass/fail per literature target
  Table 2: H1 headline (paired Δ with CI + p) across scenarios × outcomes
  Table 3: H2 phase dynamics — before / during / after shock
  Table 4: Composition robustness — % significant help / hurt / NS per n_mm
  Table 5: Per-exact-n_mm AMM effect across scenarios (threshold view)

All tables go to output/tables/ as both CSV (for paper) and a single
human-readable .txt with ASCII rendering.

Usage:
    python tools/make_summary_tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _write_csv(rows: List[Dict], path: str, fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v, prec=3) -> str:
    if isinstance(v, str):
        return v
    try:
        x = float(v)
        if not math.isfinite(x):
            return 'NaN'
        if abs(x) < 1e-4 and x != 0:
            return f'{x:.2e}'
        return f'{x:.{prec}f}'
    except (TypeError, ValueError):
        return str(v)


def _sig(p) -> str:
    try:
        p = float(p)
        if not math.isfinite(p):
            return ''
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''
    except: return ''


# ---------------------------------------------------------------------------
# Table 1: Calibration acceptance
# ---------------------------------------------------------------------------

def table_1_calibration(in_path: str, out_dir: str) -> List[Dict]:
    if not os.path.exists(in_path):
        print(f'[skip] no acceptance report at {in_path}')
        return []
    with open(in_path) as f:
        report = json.load(f)
    rows = []
    for t in report.get('targets', []):
        target_str = t.get('target')
        band = t.get('accepted_error_band', {})
        band_str = ', '.join(f'{k}={v}' for k, v in band.items())
        rows.append({
            'target': t['observable'],
            'realized': _fmt(t.get('realized_value'), 4),
            'target_band': str(target_str),
            'tolerance': band_str,
            'status': t.get('status'),
            'gating': 'YES' if t.get('gating') else 'no',
            'source': t.get('source', ''),
        })
    _write_csv(rows, os.path.join(out_dir, 'table1_calibration.csv'))
    return rows


# ---------------------------------------------------------------------------
# Table 2: H1 headline paired tests
# ---------------------------------------------------------------------------

H1_METRICS = [
    ('avg_clob_spread_bps',      'Quoted spread (bps)',       True),   # lower is better
    ('avg_clob_depth',           'CLOB depth',                False),  # higher is better
    ('avg_realized_volatility',  'Realized volatility',       True),
    ('tail_realized_volatility', 'Tail realized vol',         True),
    ('avg_cost_clob_q5',         'Execution cost Q=5 (bps)',  True),
]

SCENARIO_ORDER = ['default', 'mm_withdrawal', 'flash_crash',
                  'dealer_liquidity_crisis', 'funding_liquidity_shock',
                  'high_vol_stress']


def table_2_h1(rq1_path: str, out_dir: str) -> List[Dict]:
    raw = _read_csv(rq1_path)
    rows = []
    for sc in SCENARIO_ORDER:
        for m, label, lower_better in H1_METRICS:
            match = next((r for r in raw if r['scenario_key'] == sc and r['metric_name'] == m), None)
            if not match:
                continue
            d = _f(match.get('mean_delta_with_minus_without'))
            ci_lo = _f(match.get('delta_ci_lo'))
            ci_hi = _f(match.get('delta_ci_hi'))
            p = _f(match.get('permutation_p_value'))
            verdict = ''
            if math.isfinite(d) and math.isfinite(p) and p < 0.05:
                if (d < 0 and lower_better) or (d > 0 and not lower_better):
                    verdict = 'AMM helps'
                else:
                    verdict = 'AMM hurts'
            elif math.isfinite(p):
                verdict = 'NS'
            rows.append({
                'scenario': sc,
                'metric': label,
                'delta_with_minus_without': _fmt(d, 4),
                'ci_95': f'[{_fmt(ci_lo, 4)}, {_fmt(ci_hi, 4)}]',
                'p_value': _fmt(p, 4) + _sig(p),
                'n_pairs': match.get('n_pairs', ''),
                'verdict': verdict,
            })
    _write_csv(rows, os.path.join(out_dir, 'table2_h1_paired_tests.csv'))
    return rows


# ---------------------------------------------------------------------------
# Table 3: H2 phase dynamics
# ---------------------------------------------------------------------------

H2_METRICS = [
    ('amm_customer_volume_share',
                              'AMM customer volume share (ratio of sums)'),
    ('amm_active_tick_flow_share',
                              'AMM customer share (active trade ticks)'),
    ('mm_share_active',       'MM active share'),
    ('mm_share_endogenous',   'MM endogenous withdrawal share'),
    ('mm_share_forced',       'MM forced pause share'),
    ('mm_withdrawal_score',   'MM withdrawal score'),
]


def table_3_h2_phases(phase_path: str, tests_path: str, out_dir: str) -> List[Dict]:
    summary = _read_csv(phase_path)
    tests = _read_csv(tests_path)

    rows = []
    for sc in SCENARIO_ORDER:
        if sc == 'default':
            continue  # H2 phase analysis only meaningful in stress scenarios
        for metric, label in H2_METRICS:
            before = next((r for r in summary if r['scenario_key'] == sc and r['phase'] == 'before'), None)
            during = next((r for r in summary if r['scenario_key'] == sc and r['phase'] == 'during'), None)
            after = next((r for r in summary if r['scenario_key'] == sc and r['phase'] == 'after'), None)
            if not (before and during and after):
                continue
            test_match = next((t for t in tests
                              if t['scenario_key'] == sc and t['metric_name'] == metric
                              and t['phase_a'] == 'during' and t['phase_b'] == 'before'), None)
            p_str = ''
            d_str = ''
            if test_match:
                p_str = _fmt(_f(test_match.get('permutation_p_value')), 4) + _sig(test_match.get('permutation_p_value'))
                d_str = _fmt(_f(test_match.get('mean_delta_a_minus_b')), 4)
            rows.append({
                'scenario': sc,
                'metric': label,
                'before': _fmt(_f(before.get(metric)), 4),
                'during': _fmt(_f(during.get(metric)), 4),
                'after': _fmt(_f(after.get(metric)), 4),
                'delta_during_minus_before': d_str,
                'p_value': p_str,
            })
    _write_csv(rows, os.path.join(out_dir, 'table3_h2_phases.csv'))
    return rows


# ---------------------------------------------------------------------------
# Table 4: Composition robustness — % significant by per-composition test
# ---------------------------------------------------------------------------

COMP_METRICS = [
    ('avg_clob_spread_bps',      'Quoted spread',       True),
    ('avg_clob_depth',           'CLOB depth',          False),
    ('avg_realized_volatility',  'Realized volatility', True),
    ('avg_cost_clob_q5',         'Execution cost Q=5',  True),
    ('tail_realized_volatility', 'Tail realized vol',   True),
]


def table_4_composition(amm_effect_path: str, out_dir: str, alpha: float = 0.05) -> List[Dict]:
    raw = _read_csv(amm_effect_path)
    rows = []
    scenarios = ['default', 'dealer_liquidity_crisis', 'high_vol_stress']

    for sc in scenarios:
        for metric, label, lower_better in COMP_METRICS:
            delta_k = f'{sc}__{metric}__delta'
            p_k = f'{sc}__{metric}__pvalue'

            # All compositions
            all_pts = [(int(r['n_mm']), _f(r.get(delta_k)), _f(r.get(p_k))) for r in raw
                       if r.get(delta_k) and r.get(p_k)]
            n0 = [(d, p) for (mm, d, p) in all_pts if mm == 0]
            n1plus = [(d, p) for (mm, d, p) in all_pts if mm >= 1]

            def _classify(items):
                total = len(items)
                if total == 0:
                    return 0, 0, 0, 0
                helps = sum(1 for d, p in items
                            if math.isfinite(p) and p < alpha
                            and ((d < 0 and lower_better) or (d > 0 and not lower_better)))
                hurts = sum(1 for d, p in items
                            if math.isfinite(p) and p < alpha
                            and ((d > 0 and lower_better) or (d < 0 and not lower_better)))
                ns = total - helps - hurts
                return total, helps, hurts, ns

            t_n0, h_n0, hu_n0, ns_n0 = _classify(n0)
            t_n1, h_n1, hu_n1, ns_n1 = _classify(n1plus)

            rows.append({
                'scenario': sc,
                'metric': label,
                'n_mm_0_helps': f'{h_n0}/{t_n0}',
                'n_mm_0_hurts': f'{hu_n0}/{t_n0}',
                'n_mm_0_NS': f'{ns_n0}/{t_n0}',
                'n_mm_ge1_helps': f'{h_n1}/{t_n1}',
                'n_mm_ge1_hurts': f'{hu_n1}/{t_n1}',
                'n_mm_ge1_NS': f'{ns_n1}/{t_n1}',
                'verdict': ('AMM uniformly helps (n_mm≥1)'
                            if h_n1 == t_n1 and t_n1 > 0
                            else 'mixed'),
            })

    _write_csv(rows, os.path.join(out_dir, 'table4_composition_robustness.csv'))
    return rows


# ---------------------------------------------------------------------------
# Table 5: Per-exact-n_mm AMM effect across scenarios (threshold view)
# ---------------------------------------------------------------------------

def table_5_threshold(amm_effect_path: str, out_dir: str) -> List[Dict]:
    raw = _read_csv(amm_effect_path)
    rows = []
    scenarios = ['default', 'dealer_liquidity_crisis', 'high_vol_stress']

    # Aggregate per (n_mm, scenario) for spread metric (key metric for threshold)
    metric = 'avg_clob_spread_bps'
    by_nmm = defaultdict(lambda: {sc: [] for sc in scenarios})
    for r in raw:
        mm = int(r.get('n_mm', -1))
        if mm < 0:
            continue
        for sc in scenarios:
            d = _f(r.get(f'{sc}__{metric}__delta'))
            p = _f(r.get(f'{sc}__{metric}__pvalue'))
            if math.isfinite(d):
                by_nmm[mm][sc].append((d, p))

    for mm in sorted(by_nmm):
        row = {'n_mm': mm}
        for sc in scenarios:
            items = by_nmm[mm][sc]
            if not items:
                row[f'{sc}_n'] = 0
                row[f'{sc}_mean_delta'] = ''
                row[f'{sc}_pct_significant_helps'] = ''
                continue
            mean_d = sum(d for d, _ in items) / len(items)
            sig_helps = sum(1 for d, p in items if math.isfinite(p) and p < 0.05 and d < 0)
            sig_hurts = sum(1 for d, p in items if math.isfinite(p) and p < 0.05 and d > 0)
            row[f'{sc}_n'] = len(items)
            row[f'{sc}_mean_delta'] = _fmt(mean_d, 3)
            row[f'{sc}_pct_significant_helps'] = f'{100*sig_helps/len(items):.0f}%'
            row[f'{sc}_pct_significant_hurts'] = f'{100*sig_hurts/len(items):.0f}%'
        rows.append(row)

    _write_csv(rows, os.path.join(out_dir, 'table5_threshold_by_nmm.csv'))
    return rows


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

def _render_ascii(title: str, rows: List[Dict], out_lines: List[str]) -> None:
    out_lines.append('')
    out_lines.append('=' * 100)
    out_lines.append(f'  {title}')
    out_lines.append('=' * 100)
    if not rows:
        out_lines.append('  (no data)')
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max((len(str(r.get(c, ''))) for r in rows), default=0)) for c in cols}
    header = '  ' + ' | '.join(c.ljust(widths[c]) for c in cols)
    out_lines.append(header)
    out_lines.append('  ' + '-+-'.join('-' * widths[c] for c in cols))
    for r in rows:
        out_lines.append('  ' + ' | '.join(str(r.get(c, '')).ljust(widths[c]) for c in cols))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='output/tables')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lines: List[str] = []
    lines.append('PAPER-READY SUMMARY TABLES')
    lines.append(f'Generated from current output/ artifacts.')

    print('Building Table 1 — Calibration acceptance...')
    t1 = table_1_calibration('output/main_aware/primary_acceptance_report.json', args.out_dir)
    _render_ascii('TABLE 1 — Calibration acceptance (baseline, default scenario)', t1, lines)

    print('Building Table 2 — H1 paired tests across scenarios...')
    t2 = table_2_h1('output/stat_tests/rq1_tests.csv', args.out_dir)
    _render_ascii('TABLE 2 — H1 headline (AMM-effect on CLOB quality, 300 paired seeds)',
                  t2, lines)

    print('Building Table 3 — H2 phase dynamics...')
    t3 = table_3_h2_phases('output/stat_tests/h2_phase_summary.csv',
                            'output/stat_tests/h2_phase_tests.csv', args.out_dir)
    _render_ascii('TABLE 3 — H2 phase dynamics (before / during / after shock)', t3, lines)

    print('Building Table 4 — Composition robustness...')
    t4 = table_4_composition('output/composition/analysis/amm_effect_by_point.csv', args.out_dir)
    _render_ascii('TABLE 4 — Composition robustness (% of 200 LHS compositions with significant Δ at α=0.05)',
                  t4, lines)

    print('Building Table 5 — Per-n_mm threshold view...')
    t5 = table_5_threshold('output/composition/analysis/amm_effect_by_point.csv', args.out_dir)
    _render_ascii('TABLE 5 — AMM effect on quoted spread by exact n_mm value',
                  t5, lines)

    ascii_path = os.path.join(args.out_dir, 'all_tables.txt')
    with open(ascii_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nASCII rendering → {ascii_path}')
    print(f'CSVs → {args.out_dir}/table*.csv')


if __name__ == '__main__':
    main()
