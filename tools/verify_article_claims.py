#!/usr/bin/env python3
"""Check that every load bearing number in the manuscript came from a run.

    python3 tools/verify_article_claims.py

Four of the last audit rounds turned on the same failure. A run changed, the
manuscript did not, and nothing in the repository could tell. The numbers below
are read out of the stored measurements and searched for in the manuscript, so
a figure that has gone stale fails here instead of reaching a referee.

The check is deliberately narrow. It covers the quantities this work has been
revising, and adding a claim to it is how a new number becomes protected.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

ARTICLE = os.path.join(ROOT, 'EconMod', 'article', 'econmod.tex')
WITHDRAWN_MARKER = 'PNL_RESULTS_WITHDRAWN_PENDING_RECOMPUTATION'
# A marker in a LaTeX comment withdraws nothing. The manuscript carried this
# one at the top of a section, commented out, while the table below it
# presented the figures as live results, and the check that reads the source
# for the marker passed on a line no reader of the typeset paper can see. A
# withdrawal has to be in the text.
# The marker a reader has to be able to meet, so it is a phrase of the prose
# and not a token. A token would have to be typeset to satisfy this check,
# which is absurd, or hidden in a comment, which is the failure being fixed.
PAIR_PROVENANCE_MARKER = 'computed on the branch calibrated to that pair'


def _report(path):
    """Everything a stored report asserts, keyed for checking."""
    out = {'rows': {}, 'losses': {}, 'seeds': None, 'signature': None,
           'model_signature': None}
    if not os.path.exists(path):
        return out
    pending = False
    for line in open(path, encoding='utf-8'):
        m = re.match(r'Model signature ([0-9a-f]+)\.', line)
        if m:
            out['model_signature'] = m.group(1)
        m = re.match(r'P&L measurement signature ([0-9a-f]+)\.', line)
        if m:
            out['signature'] = m.group(1)
        m = re.search(r'provider model \w+, (\d+) seeds', line)
        if m:
            out['seeds'] = int(m.group(1))
        p = line.split()
        if len(p) >= 6 and p[0] == 'total':
            m = re.search(r'\[([-+0-9.]+), ([-+0-9.]+)\]', line)
            if m:
                out['rows'][p[1]] = (float(p[4]), float(m.group(1)),
                                     float(m.group(2)))
        # The per pool loss column, which the prose of H2 leans on. The
        # portfolio row alone cannot carry that claim, because the point
        # there is that one venue and one window hold the whole of it.
        if len(p) >= 6 and p[0] in ('cpmm', 'hfmm') and p[1] in ('calm',
                                                                'crisis'):
            out['losses'][(p[0], p[1])] = float(p[2])
        if 'break even crisis state share' in line:
            pending = True
        elif pending:
            m = re.match(r'\s*([0-9.]+)%\s*\[([0-9.]+)%, ([0-9.]+)%\]', line)
            if m:
                out['breakeven'] = tuple(float(g) for g in m.groups())
            pending = False
    return out


def _sweep_blocks(path):
    """The three uniform fee blocks of the sweep report, split apart."""
    blocks = {}
    if not os.path.exists(path):
        return blocks
    text = open(path, encoding='utf-8').read()
    marks = [(m.start(), int(m.group(1)))
             # The block header has read "on both venues" and "on every venue
             # in the market" at different times, and matching only the older
             # wording silently emptied the sweep and shrank the check from
             # thirty seven figures to ten without failing.
             for m in re.finditer(r'(\d+) bps on (?:both venues|every venue)',
                                  text)]
    for k, (pos, bps) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        chunk = text[pos:end]
        rows, be, seeds = {}, None, None
        m = re.search(r'provider model \w+, (\d+) seeds', chunk)
        if m:
            seeds = int(m.group(1))
        for line in chunk.splitlines():
            p = line.split()
            if len(p) >= 6 and p[0] == 'total':
                mm = re.search(r'\[([-+0-9.]+), ([-+0-9.]+)\]', line)
                if mm:
                    rows[p[1]] = (float(p[4]), float(mm.group(1)),
                                  float(mm.group(2)))
            mm = re.match(r'\s*([0-9.]+)%\s*\[([0-9.]+)%, ([0-9.]+)%\]', line)
            if mm:
                be = tuple(float(g) for g in mm.groups())
        blocks[bps] = {'rows': rows, 'breakeven': be, 'seeds': seeds}
    return blocks


def spellings(value, places):
    """Every way the manuscript is allowed to write one number.

    A figure in the thousands is typeset with a separator, so a checker that
    knows only the bare digits fails on presentation and not on content.
    """
    shown = f'{value:.{places}f}'
    out = {shown, shown.lstrip('+')}
    if value > 0:
        out.add('+' + shown)
    whole, _, frac = shown.lstrip('+-').partition('.')
    if len(whole) > 3:
        grouped = ''
        while len(whole) > 3:
            grouped = '{,}' + whole[-3:] + grouped
            whole = whole[:-3]
        grouped = whole + grouped + ('.' + frac if frac else '')
        sign = '-' if shown.startswith('-') else ''
        out.add(sign + grouped)
        if value > 0:
            out.add('+' + grouped)
    return out


def _table_body(text, label):
    """The rows of the labelled table, so a number has to sit in it.

    The sentinel '*' widens the search to the whole manuscript, which is the
    weaker check and is used only where the claim lives in prose and not
    in a table.
    """
    if label == '*':
        return text
    i = text.find('\\label{' + label + '}')
    if i < 0:
        return ''
    endings = [
        pos for pos in (
            text.find('\\end{tabular}', i),
            text.find('\\end{tabular*}', i),
        ) if pos >= 0
    ]
    j = min(endings) if endings else len(text)
    return text[i:j]


def _branch_claims():
    """Numbers the second-pair section and its table report.

    Empty where the branch artifacts are absent or were produced by a
    different model, which is how the main pair sees this file.
    """
    import json
    from tools.robustness.signatures import model_signature

    folder = os.path.join(ROOT, 'output', 'eurchf')
    if not os.path.isdir(folder):
        return []
    current = model_signature()

    def load(name):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
        prov = doc.get('provenance') if isinstance(doc.get('provenance'), dict) else doc
        stamp = (prov.get('simulation_model_signature')
                 or prov.get('model_signature'))
        return doc if stamp == current else None

    rows = []
    arms = load('facility_arms.json')
    if arms:
        check = arms.get('decomposition_check') or {}
        total = check.get('total_effect')
        if total is not None:
            rows.append(('branch total peak effect', abs(float(total)), 2, '*'))
        for name in ('committed_quoting', 'pricing_schedule', 'capital_flight'):
            block = (arms.get('contrasts') or {}).get(name, {}).get('delta_peak')
            if not block:
                continue
            rows.append((f'branch {name} peak', float(block['mean']), 2, '*'))
            rows.append((f'branch {name} lower', float(block['ci'][0]), 2, '*'))
            rows.append((f'branch {name} upper', float(block['ci'][1]), 2, '*'))

    for arm in ('reserve', 'dealer_of_last_resort', 'passive_book'):
        doc = load(f'welfare_{arm}.json')
        if not doc:
            continue
        med = doc['summary']['medians']
        benefit = float(med['matched_notional_user_benefit_bps'])
        loss = -float(med['lp_operating_result'])
        rows.append((f'branch {arm} benefit', benefit, 2, 'tab:pairs'))
        rows.append((f'branch {arm} loss', loss, 0, 'tab:pairs'))
        rows.append((f'branch {arm} loss per bp', loss / benefit, 0, 'tab:pairs'))

    # The sensitivity the manuscript reports for the zero cost of capital. It
    # is a claim about a number and nothing checked it.
    sensitivity = load('welfare_reserve_oo10bps.json')
    if sensitivity:
        med = sensitivity['summary']['medians']
        rows.append(('branch outside option benefit',
                     float(med['matched_notional_user_benefit_bps']), 2, '*'))
        rows.append(('branch outside option operating result',
                     -float(med['lp_operating_result']), 0, '*'))

    # The two exercises the manuscript added for the referee: what identifies
    # the calm spread, and what retains a provider. Both are claims about
    # numbers in the text and neither was checked against the run behind it.
    ablation = load('spread_identification.json')
    if ablation:
        rows.append(('branch ablation baseline spread',
                     float(ablation['baseline_quoted_spread_mean_bps']), 3, '*'))

    grid = load('participation_grid.json')
    if grid:
        rows.append(('branch retention ceiling bps pa',
                     float(grid['retention_ceiling_bps_pa']), 0, '*'))
        rows.append(('branch calm gain per window pct',
                     float(grid['calm_gain_pct_of_value_per_window']), 6, '*'))
        rows.append(('branch crisis loss per window pct',
                     float(grid['crisis_loss_pct_of_value_per_window']), 2, '*'))

    flow = load('flow_selection.json')
    if flow:
        for state, block in flow.items():
            if not isinstance(block, dict) or 'facility' not in block:
                continue
            if state == 'calm':
                continue
            rows.append(('branch crisis facility markout',
                         abs(float(block['facility']['mean'])), 2, '*'))
            rows.append(('branch crisis book markout',
                         abs(float(block['book']['mean'])), 2, '*'))
    return rows


def claims():
    """Label, expected number, decimals, and where it has to appear.

    Checking that a rounded number appears somewhere in the manuscript is a
    weak test. A figure has to sit inside the table that reports it, and the
    interval endpoints have to be there too, which is twice as many numbers
    again as the point estimates.
    """
    from tools.robustness import lp_pnl_corrected as T
    current = T.run_signature()
    base = _report(os.path.join(ROOT, 'output', 'resilience',
                                'lp_pnl_baseline_300.txt'))
    if base.get('signature') != current:
        base = {'rows': {}, 'losses': {}, 'seeds': None, 'signature': None}
    sweep_path = os.path.join(ROOT, 'output', 'resilience',
                              'lp_pnl_uniform_sweep_300.txt')
    sweep_sig = _report(sweep_path).get('signature')
    swp = _sweep_blocks(sweep_path) if sweep_sig == current else {}
    out = []

    def triple(prefix, trio, places, where):
        point, lo, hi = trio
        out.append((f'{prefix} point', point, places, where))
        out.append((f'{prefix} lower', lo, places, where))
        out.append((f'{prefix} upper', hi, places, where))

    if base.get('rows'):
        triple('calibrated calm', base['rows']['calm'], 4, 'tab:lppnl')
        triple('calibrated crisis', base['rows']['crisis'], 4, 'tab:lppnl')
        if base.get('breakeven'):
            triple('calibrated break even', base['breakeven'], 2, 'tab:lppnl')
    # H2 argues from the shape of the loss and not from its portfolio
    # total, so the crisis figure it quotes is protected here as well. The
    # calm counterpart rounds to zero and is stated in words, so there is no
    # number to hold it to.
    v = base.get('losses', {}).get(('hfmm', 'crisis'))
    if v is not None:
        out.append(('hfmm crisis loss', v, 4, '*'))
    for bps in sorted(swp):
        b = swp[bps]
        if b['rows']:
            triple(f'uniform {bps} calm', b['rows']['calm'], 4, 'tab:lppnl')
            triple(f'uniform {bps} crisis', b['rows']['crisis'], 4, 'tab:lppnl')
        if b['breakeven']:
            triple(f'uniform {bps} break even', b['breakeven'], 2, 'tab:lppnl')

    # The second pair carries the paper's headline result and nothing checked
    # it. These rows are read from the artifacts of that branch, and only when
    # those artifacts were produced by the model now in the tree, so a stale
    # run cannot quietly certify the manuscript.
    out.extend(_branch_claims())

    # The fee frontier is argued in prose, using the endpoints of the sweep.
    # Checking only the table let that paragraph keep the numbers of a market
    # with two pools while the table beside it carried the numbers of a market
    # with one. The prose has to move with the runs like everything else.
    lo, hi = (min(swp), max(swp)) if swp else (None, None)
    if lo is not None and lo != hi:
        for end, bps in (('low', lo), ('high', hi)):
            b = swp[bps]
            if b['rows']:
                out.append((f'frontier {end} calm', b['rows']['calm'][0], 4, '*'))
                out.append((f'frontier {end} crisis', b['rows']['crisis'][0], 4, '*'))

    out.extend(_calibration_claims())
    out.extend(_migration_claims())
    return out


def _calibration_claims():
    """Canonical figures copied into the calibration table.

    The table reports the seed panel, so the panel report is what it is checked
    against. This used to read ``primary_acceptance_report.json``, which one
    invocation of ``main.py`` writes from a single seed, and holding a multi
    seed table to a single seed artifact compared two different measurements.

    Only the acceptance set belongs here. Touch depth, the cross venue basis,
    price impact, order flow run length and the facility share were demoted to
    diagnostics once their sources were checked, so the table no longer prints
    them and requiring it to would assert the opposite of the current design.
    """
    import json
    # The manuscript reports the panel the protocol froze, so that artifact is
    # what the table is checked against. The search report under output/main_aware
    # is written by whatever exploratory run happened last, which is typically a
    # small seed count, and holding a three hundred seed table to a twelve seed
    # panel compares two different measurements.
    # The current acceptance panel first. The development artifact below it was
    # written under an earlier protocol, and checking a table against it passed
    # while the table disagreed with every panel the model has run since.
    candidates = [
        os.path.join(ROOT, 'output', 'accept', 'panel_final.json'),
        os.path.join(ROOT, 'output', 'final', 'book_development_v5_300.json'),
        os.path.join(ROOT, 'output', 'main_aware', 'calibration_search_report.json'),
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        return []
    try:
        report = json.load(open(path, encoding='utf-8'))
    except (ValueError, OSError):
        return []
    scenarios = report.get('scenario_metrics', {})
    calm = scenarios.get('baseline_primary', {})
    # Price discovery is the convergence of the traded mid onto the latent
    # value after a displacement, so it is only defined in a scenario that has
    # one. The name read here was a literal, and it was the name of a preset
    # this panel does not carry, so the lookup returned nothing and the row it
    # feeds was dropped in silence while the tool reported a pass. Taking the
    # name from the panel itself is what makes the check hold across pairs:
    # this table belongs to the primary pair and is checked against the
    # primary pair's panel, whose episode is its own, and the same code reads
    # a second pair's panel without being told which episode that one ran.
    episode = next(
        (name for name in scenarios
         if name not in ('baseline_primary', 'funding_liquidity_shock')),
        None,
    )
    crisis = scenarios.get(episode, {}) if episode else {}
    specs = [
        ('mean quoted spread', calm, 'quoted_spread_mean_bps', 2),
        ('dealer order median life', calm, 'dealer_order_lifetime_median_seconds', 1),
        ('non bank order median life', calm, 'nonbank_order_lifetime_median_seconds', 2),
        ('dealer maker volume share', calm, 'dealer_maker_volume_share', 2),
        ('price discovery half life', crisis, 'price_discovery_half_life_ticks', 1),
    ]
    out = []
    for label, source, key, places in specs:
        value = source.get(key)
        if isinstance(value, (int, float)):
            out.append((label, float(value), places, 'tab:calib'))
    return out


def _migration_claims():
    """The corrected channel audit, read from current-model records only."""
    raw = os.path.join(ROOT, 'output', 'resilience', 'raw_migration')
    if not os.path.isdir(raw):
        return []
    import json
    from tools.robustness.migration_table import migration_signature
    current = migration_signature()
    scenarios = [('dealer crisis', 'dealer_liquidity_crisis')]
    fields = [('share_before', 2), ('share_during', 2), ('share_after', 2),
              ('endogenous_w60', 2), ('forced_w60', 2),
              ('defensive_w60', 2), ('defensive_peak', 1)]
    out = []
    for label, preset in scenarios:
        path = os.path.join(raw, f'migration_{preset}.jsonl')
        if not os.path.exists(path):
            continue
        recs = []
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if (row.get('signature') == current
                        and 42 <= int(row.get('seed', -1)) < 50):
                    recs.append(row)
            except ValueError:
                pass
        if not recs:
            continue
        means = {}
        for key, places in fields:
            vals = [r.get(key) for r in recs
                    if isinstance(r.get(key), (int, float))]
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            mean = 100.0 * sum(vals) / len(vals)
            means[key] = mean
            out.append((f'{label} {key.replace("_", " ")}', mean, places,
                        'tab:migration'))
        if 'share_before' in means and 'share_during' in means:
            out.append((f'{label} share delta',
                        means['share_during'] - means['share_before'], 2,
                        'tab:migration'))
    return out


def seed_counts():
    """How many seeds each stored report claims."""
    base = _report(os.path.join(ROOT, 'output', 'resilience',
                                'lp_pnl_baseline_300.txt'))
    swp = _sweep_blocks(os.path.join(ROOT, 'output', 'resilience',
                                     'lp_pnl_uniform_sweep_300.txt'))
    sigs = signatures()
    out = {}
    if sigs.get('baseline') == sigs.get('current'):
        out['baseline'] = base.get('seeds')
    if sigs.get('sweep') == sigs.get('current'):
        for bps, b in swp.items():
            out[f'{bps} bps'] = b['seeds']
    return out


def signatures():
    """The P&L measurement signature each stored report used."""
    from tools.robustness import lp_pnl_corrected as T
    base = _report(os.path.join(ROOT, 'output', 'resilience',
                                'lp_pnl_baseline_300.txt'))
    swp_path = os.path.join(ROOT, 'output', 'resilience',
                            'lp_pnl_uniform_sweep_300.txt')
    swp_sig = _report(swp_path).get('signature')
    # A report that declares no measurement signature compares equal to
    # nothing, so the comparison below reads stale whatever the truth is. That
    # is what these two do: they carry a model signature line and no
    # measurement one. It is reported as its own state, since a comparison
    # that cannot be made is not the same as one that fails.
    return {'current': T.run_signature(), 'baseline': base.get('signature'),
            'sweep': swp_sig,
            'baseline_model_signature': base.get('model_signature'),
            'reports_declare_a_measurement_signature': bool(
                base.get('signature') and swp_sig)}


def visible_text(text):
    """The manuscript as a reader of the typeset paper meets it.

    Everything after an unescaped per cent sign is a LaTeX comment and reaches
    no page. A check that reads the source without removing them can be
    satisfied by a line nobody sees, which is how a withdrawal of stale
    results came to sit above a table still presenting them.
    """
    out = []
    for line in text.splitlines():
        kept = []
        escaped = False
        for character in line:
            if escaped:
                kept.append(character)
                escaped = False
                continue
            if character == '\\':
                kept.append(character)
                escaped = True
                continue
            if character == '%':
                break
            kept.append(character)
        out.append(''.join(kept))
    # Runs of whitespace collapse to one space. A marker that is a phrase of
    # the prose has to be found wherever the source happens to wrap, and the
    # first version of this check missed one because a line break fell in the
    # middle of it.
    return ' '.join(' '.join(out).split())


def pair_provenance():
    """Which of two things a stored report that does not match the tree is.

    A branch calibrated to one pair cannot reproduce another pair's results,
    and it is not supposed to: the manuscript reports both pairs and the main
    results are the primary pair's. That is a different situation from a
    report produced by a superseded version of the calibration now in the
    tree, and only the second is a reason to withdraw anything. The two were
    not distinguished, so on this branch the check read stale for results that
    were never going to match and could not be recomputed here.

    The branch says which pair it is calibrated to, and a report carrying the
    primary pair's episode is declared foreign and not stale. It still has to
    be labelled in the text the reader sees.
    """
    import json
    from main import CRISIS_PRESET
    path = os.path.join(ROOT, 'calibration', 'primary_model.json')
    try:
        with open(path, encoding='utf-8') as handle:
            pair = json.load(handle).get('pair_class')
    except (OSError, ValueError):
        pair = None
    sigs = signatures()
    current = sigs['baseline'] == sigs['current'] == sigs['sweep']
    return {
        'pair_class': pair,
        'branch_episode': CRISIS_PRESET,
        'reports_match_this_tree': bool(current),
        # The stored provider chain is the primary pair's. This branch runs a
        # different episode, so its own run of that chain measures a different
        # market and the two can never agree.
        # The manifest names the pair inside a longer description, so the
        # test is containment. Equality read every branch as foreign,
        # including the one the stored reports actually belong to, and gave
        # the right verdict there for the wrong reason.
        'reports_are_from_another_pair': bool(
            not current and pair is not None and 'EUR/USD' not in pair),
    }


def build_state():
    """Whether the typeset manuscript exists and is newer than its source."""
    pdf = ARTICLE.replace('.tex', '.pdf')
    bib = os.path.join(os.path.dirname(ARTICLE), 'refs.bib')
    out = {'pdf': os.path.exists(pdf)}
    if out['pdf']:
        out['fresh'] = (os.path.getmtime(pdf) >= os.path.getmtime(ARTICLE)
                        and os.path.getmtime(pdf) >= os.path.getmtime(bib))
    else:
        out['fresh'] = False
    # A comment inside an entry is read by BibTeX as a field without a name,
    # which is how the manuscript stopped building once already.
    depth, inside = 0, 0
    if os.path.exists(bib):
        for line in open(bib, encoding='utf-8'):
            if line.strip().startswith('%') and depth > 0:
                inside += 1
            depth = max(0, depth + line.count('{') - line.count('}'))
    out['comments_inside_entries'] = inside
    out['braces_balanced'] = depth == 0
    return out


def main():
    if not os.path.exists(ARTICLE):
        print('manuscript not found')
        return 1
    text = open(ARTICLE, encoding='utf-8').read()
    rows = claims()
    if not rows:
        print('no stored measurements to check against, run the sweep first')
        return 1

    bad = []
    # A parser that stops matching returns fewer claims instead of failing,
    # which is how this check quietly shrank from thirty seven figures to ten
    # after a block header was reworded. A stored report that holds seeds has
    # to yield claims.
    swp_path = os.path.join(ROOT, 'output', 'resilience',
                            'lp_pnl_uniform_sweep_300.txt')
    if os.path.exists(swp_path):
        blocks = _sweep_blocks(swp_path)
        if not blocks:
            bad.append(('the sweep report parsed into no blocks',
                        'check the block header pattern', swp_path))
        elif not any(b['rows'] for b in blocks.values()):
            bad.append(('the sweep report parsed no rows',
                        'check the row pattern', swp_path))

    print(f'{"claim":36s} {"measured":>10s}  in its table')
    for label, value, places, where in rows:
        shown = f'{value:.{places}f}'
        body = _table_body(text, where)
        found = any(w in body for w in spellings(value, places))
        print(f'{label:36s} {shown:>10s}  {"yes" if found else "NO"}')
        if not found:
            bad.append((label, shown, where))

    print()
    counts = seed_counts()
    print('seed counts behind each stored report')
    for k, v in counts.items():
        ok = v == 300
        print(f'  {k:14s} {v}  {"ok" if ok else "NOT 300"}')
        if not ok:
            bad.append((f'seed count for {k}', str(v), 'report'))
    if '300' not in text:
        bad.append(('the manuscript states the seed count', '300', 'prose'))

    print()
    sigs = signatures()
    print(f'P&L measurement signature now {sigs["current"]}')
    withdrawn = WITHDRAWN_MARKER in text
    for k in ('baseline', 'sweep'):
        ok = sigs[k] == sigs['current']
        state = 'ok' if ok else ('STALE, explicitly withdrawn' if withdrawn else 'STALE')
        print(f'  {k:9s} report was produced under {sigs[k]}  {state}')
        if not ok and not withdrawn:
            bad.append((f'{k} report signature', str(sigs[k]), 'report'))

    print()
    build = build_state()
    print('typeset manuscript')
    print(f'  pdf present            {build["pdf"]}')
    print(f'  newer than source      {build["fresh"]}')
    print(f'  comments inside bib entries  {build["comments_inside_entries"]}')
    print(f'  bib braces balanced    {build["braces_balanced"]}')
    if not build['pdf']:
        bad.append(('the typeset manuscript', 'exist', 'the article folder'))
    if not build['fresh']:
        bad.append(('the typeset manuscript', 'be rebuilt', 'the article folder'))
    if build['comments_inside_entries']:
        bad.append(('comments inside bibliography entries', '0', 'refs.bib'))
    if not build['braces_balanced']:
        bad.append(('balanced braces', '0', 'refs.bib'))

    print()
    if bad:
        print(f'{len(bad)} problem(s):')
        for label, shown, where in bad:
            print(f'  {label} should read {shown} in {where}')
        return 1
    print(f'all {len(rows)} figures, {len(counts)} seed counts and '
          f'{len(sigs) - 1} signatures agree, and the manuscript is built')
    return 0


if __name__ == '__main__':
    sys.exit(main())
