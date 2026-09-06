#!/usr/bin/env python3
"""Tests for the entry point that names the workflows.

    python3 tests/cli_dispatch_test.py

Every command the usage text advertises has to reach a runner. A name listed
there but missing from the dispatch falls through to the unknown command
branch, so the entry point advertises work it cannot do; that is how the
figures of the article came to be drawn by hand outside the entry point for
as long as they were.

The figure command carries a second obligation. Its runner reports the figures
it managed to draw and swallows the ones it could not, so a run that half fails
still ends. Since the article keeps whatever the previous run wrote, a partial
set leaves stale figures beside fresh ones with nothing to say so, and the
command has to fail and not return quietly.
"""
from __future__ import annotations

import inspect
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as entry


def test_every_advertised_command_is_dispatched():
    """The registry and the dispatch must name the same set of commands.

    The dispatch is read and not executed, since the runners behind the
    heavy commands want artifacts and seeds that a unit test has no business
    producing. Reading it still catches both directions of the defect: a name
    advertised in the usage text with no branch to reach, and a branch no name
    announces.
    """
    dispatched = set(re.findall(r"command == '([a-z]+)'",
                                inspect.getsource(entry.main)))
    assert dispatched == set(entry.COMMANDS)
    for command in entry.COMMANDS:
        assert command in entry._usage()


def test_figures_command_is_registered():
    assert 'figures' in entry.COMMANDS
    assert 'figures' in entry._usage()


def test_figures_reports_a_partial_set_as_a_failure(monkeypatch, capsys):
    """Three figures drawn out of eleven is a failed run, not a warning."""
    from AgentBasedModel.visualization import paper_figures

    monkeypatch.setattr(paper_figures, 'draw_all',
                        lambda: ['EconMod/figures/one.pdf'])
    assert entry.main(['figures']) == 1
    assert 'were not drawn' in capsys.readouterr().err


def test_figures_returns_zero_on_a_complete_set(monkeypatch):
    from AgentBasedModel.visualization import paper_figures

    monkeypatch.setattr(paper_figures, 'draw_all',
                        lambda: ['f%d.pdf' % n for n in range(len(paper_figures.ALL))])
    assert entry.main(['figures']) == 0


def _repository_sources():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = []
    for folder in ('main.py', 'tests', 'tools', 'calibration', 'AgentBasedModel'):
        path = os.path.join(root, folder)
        if os.path.isfile(path):
            text.append(open(path, encoding='utf-8').read())
            continue
        for base, _, files in os.walk(path):
            for name in files:
                if name.endswith('.py'):
                    text.append(open(os.path.join(base, name),
                                     encoding='utf-8').read())
    return '\n'.join(text)


def test_every_declared_flag_is_read_somewhere():
    """A switch nothing reads advertises work the entry point will not do.

    ``--comparison`` survived this way. It defaulted to on and promised a
    second book only simulation for a with and without comparison, which is
    the counterfactual the resource matched arms replaced; nothing had read it
    for as long as the arms had existed.
    """
    sources = _repository_sources()
    unread = []
    for action in entry.build_parser()._actions:
        dest = action.dest
        if dest == 'help':
            continue
        read = (re.search(r'args\.%s\b' % re.escape(dest), sources)
                or re.search(r'[\'"]%s[\'"]\s*[:,)]' % re.escape(dest), sources)
                or re.search(r'\b%s\s*=' % re.escape(dest), sources))
        if not read:
            unread.append(dest)
    assert not unread, f'flags nothing reads: {unread}'


def test_no_preset_carries_the_retired_venue_split():
    """The bundles that fixed a share between the venues are gone.

    They served the binary counterfactual, and a preset that still set
    ``amm_share_pct`` would silently pull routing back to the fixed split.
    """
    for name, bundle in entry.PRESETS.items():
        assert 'amm_share_pct' not in bundle, name


def test_the_readme_lists_exactly_the_commands():
    """Documentation that drifts from the entry point sends people to nothing.

    The README named neither the markout of the flow nor the figures for as
    long as those commands existed.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    listed = set(re.findall(r'python -m main (\w+)', readme))
    assert listed == set(entry.COMMANDS)


def test_the_header_advertises_nothing_that_was_retired():
    """The header is the first thing read and outlived two rounds of removal.

    It offered ``--preset clob_only`` and ``--preset amm_only`` after both were
    gone, and invoked the model as ``python3 main.py --seed 42``, a form that
    now reads the first flag as a command name and refuses it.
    """
    header = entry.__doc__ or ''
    for retired in ('clob_only', 'amm_only', 'heavy_amm', 'heavy_clob',
                    'stress_test', 'low_liquidity', 'fx_calibrated',
                    '--comparison'):
        assert retired not in header, retired
    assert 'main.py --' not in header
    # Only the worked examples, so that prose naming the flag is not read as
    # an episode.
    for line in header.splitlines():
        if 'python -m main' not in line:
            continue
        for preset in re.findall(r'--preset (\w+)', line):
            assert preset in entry.PRESETS, preset


def test_unknown_command_is_refused(capsys):
    assert entry.main(['nonsense']) == 2
    assert 'unknown command' in capsys.readouterr().err


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
