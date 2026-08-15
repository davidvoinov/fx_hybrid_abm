#!/usr/bin/env python3
"""Robustness panel for endogenous LP participation.

The LP response parameters are behavioural assumptions, not EBS calibration
targets. This panel varies them symmetrically around the primary specification
and reports survival/participation outcomes without fitting or assigning a
literature pass/fail label.  Most controls are one-at-a-time; patience and the
EWMA speed are also crossed on a complete 3-by-3 grid because both determine
how much evidence a provider has accumulated before acting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.robustness import lp_survival as S
from tools.robustness.signatures import measurement_signature


_REPORT_SIGNATURE = None
_REPORT_SCHEMA_VERSION = 2
_DESIGN = 'oat_plus_joint_patience_by_ewma_around_primary_lp_specification'


def _build_payload(signature, wanted, seed_start, results, invalid_total):
    raw_provenance_clean = int(invalid_total) == 0
    seed_complete = bool(results) and all(
        int(row.get('available_seeds', -1)) == len(wanted)
        for row in results
    )
    return {
        'model_signature': S.model_signature(S.ROOT),
        'measurement_signature': signature,
        'report_signature': sensitivity_report_signature(),
        'report_schema_version': _REPORT_SCHEMA_VERSION,
        'design': _DESIGN,
        'requested_seed_start': int(seed_start),
        'requested_seeds_per_specification': len(wanted),
        'requested_seeds': list(wanted),
        'invalid_or_stale_records_ignored': int(invalid_total),
        'raw_provenance_clean': raw_provenance_clean,
        'complete': bool(seed_complete and raw_provenance_clean),
        'parameters_are_behavioural_robustness_assumptions_not_ebs_targets': True,
        'results': results,
    }


def sensitivity_report_signature():
    """Digest of the declared design and its current aggregation."""
    global _REPORT_SIGNATURE
    if _REPORT_SIGNATURE is None:
        _REPORT_SIGNATURE = measurement_signature(
            'lp_sensitivity_report',
            S.survival_report_signature(),
            functions=(_specifications, _build_payload),
            constants=(
                _DESIGN, 15, _REPORT_SCHEMA_VERSION,
            ),
        )
    return _REPORT_SIGNATURE


def _specifications(n_iter):
    base = S._calibrated_lp_defaults()
    base['n_iter'] = int(n_iter)
    rows = [('baseline', dict(base))]

    def add(label, **changes):
        config = dict(base)
        config.update(changes)
        rows.append((label, config))

    for patience in (145, 580):
        add(
            f'patience_{patience}',
            exit_patience=patience,
            entry_patience=patience,
            max_adj=1.0 - 0.5 ** (1.0 / patience),
        )
    add('ewma_alpha_0.01', ewma_alpha=0.01)
    add('ewma_alpha_0.05', ewma_alpha=0.05)
    # Baseline plus the four OAT rows above already supply the centre row and
    # centre column.  These four corners complete the 3 x 3 design, exposing
    # interactions between smoothing and persistence that neither OAT arm can
    # identify on its own.
    for patience in (145, 580):
        for alpha in (0.01, 0.05):
            add(
                f'joint_patience_{patience}_ewma_{alpha:g}',
                exit_patience=patience,
                entry_patience=patience,
                max_adj=1.0 - 0.5 ** (1.0 / patience),
                ewma_alpha=alpha,
            )
    add('response_scale_0.5x', response_scale=0.5 * base['response_scale'])
    add('response_scale_2x', response_scale=2.0 * base['response_scale'])
    add('outside_option_0.5x', outside_option=0.5 * base['outside_option'])
    add('outside_option_2x', outside_option=2.0 * base['outside_option'])
    add('entry_margin_0', entry_margin=0.0)
    add('entry_margin_0.5', entry_margin=0.5)
    return rows


def _measure_job(job):
    return S.measure(job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=60)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--n-iter', type=int, default=S.N_ITER)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--output', default='output/resilience/lp_sensitivity.json')
    args = parser.parse_args(argv)

    signature = S.survival_signature()
    wanted = list(range(args.seed_start, args.seed_start + args.seeds))
    specs = _specifications(args.n_iter)
    pending = []
    for _, config in specs:
        have, _ = S.load_raw(S._raw_path(config, signature), signature, config)
        if not args.report_only:
            pending.extend((seed, config) for seed in wanted if seed not in have)

    if pending:
        if args.workers > 1:
            with Pool(args.workers) as pool:
                measured = pool.imap_unordered(_measure_job, pending)
                for rec in measured:
                    S.append_raw(rec, signature)
        else:
            for job in pending:
                rec = S.measure(job)
                S.append_raw(rec, signature)

    results = []
    invalid_total = 0
    for label, config in specs:
        have, stale = S.load_raw(
            S._raw_path(config, signature), signature, config
        )
        recs = [have[seed] for seed in wanted if seed in have]
        invalid_total += int(stale)
        results.append({
            'label': label,
            'available_seeds': len(recs),
            'stale_records_ignored': stale,
            'invalid_or_stale_records_ignored': stale,
            'raw_provenance_clean': int(stale) == 0,
            'summary': S.summarize(recs, config),
        })

    payload = _build_payload(
        signature, wanted, args.seed_start, results, invalid_total
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({
        'model_signature': payload['model_signature'],
        'measurement_signature': signature,
        'specifications': len(results),
        'complete': payload['complete'],
        'output': args.output,
    }, indent=2))
    return 0 if payload['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
