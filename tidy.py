"""One-off local reorganisation of runs/ and artifacts/.

Moves are logged to MOVES.md in each root so an old path can always be traced to its new one.
The two trees are gitignored, so this script is the only record of the mapping.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPRITES = {
    'kawaii-character-01': 'kawaii-maid-early',
    'kawaii-character-02': 'kawaii-maid',
    'mint-legibility-character-01': 'mint-maid-legible',
    'mint-white-maid-character-01': 'mint-maid-white',
    'mint-white-maid-character-deeper': 'mint-maid-white-deeper-blink',
    'rado-chan-character': 'rado-chan-early',
    'rado-chan-gala-character': 'rado-chan-gala',
    'trained-best400-curly-01': 'best400-curly-prototype',
    'exported-long-wavy-01': 'long-wavy-export-test',
}
SPRITE_SOURCES = {
    'kawaii-maid': ['kawaii-magenta-white-keyed', 'kawaii-blink-03', 'kawaii-blink-01', 'kawaii-blink-02'],
    'rado-chan-gala': ['rado-chan-gala', 'rado-chan-gala-blink'],
    'rado-chan-early': ['rado-chan-01', 'rado-chan-blink-01', 'rado-chan-keyed'],
    'mint-maid-legible': ['mint-legibility-01'],
    'mint-maid-white': ['mint-white-maid-blink-01', 'mint-white-maid-blink-final'],
    'mint-maid-white-deeper-blink': ['mint-white-maid-blink-deeper'],
    'best400-curly-prototype': [],
}
REVIEWS = ['overnight-matrix', 'prototype-gallery', 'source-review', 'recovery-palette-probe.png',
           'source-sheet.png', 'alexwh-sheet.png', 'buch-sheet.png', 'druiddess-sheet.png',
           'elemental-sheet.png', 'everface-sheet.png', 'female64-sheet.png', 'nekomimi-sheet.png',
           'portrait13-inspect.png', 'portrait14-inspect.png', 'corpus-review-01.json',
           'corpus-source-review-02', 'baseline-review.json']
STUDIES = ['roadblock-audit-01', 'layerdiffuse-audit-01', 'bold-features-01', 'native64-feasibility-01',
           'grid-recovery-probe-01', 'everface-identity-audit-01', 'recovery-palette-probe-32',
           'recovery-palette-probe-48', 'recovery-palette-probe-62']
RUN_BUCKETS = {
    'sprites': ['kawaii-20261015', 'rado-chan-20261015', 'mint-cyan-20261013', 'mint-white-maid-20261013',
                'portrait-followup-20261013', 'layerdiffuse-pilot-20261013'],
    'training': ['autonomous-design-20261012', 'refinement-overnight-20261012', 'facial-design-20261012',
                 'overnight-20261013'],
    'studies': ['seam-corpus-20261013'],
}
RUN_LOGS = ('.log', '.stdout.log', '.stderr.log', '.jsonl', '.jsonl.corrupt-backup')
RUN_INTERNAL = {'loops', 'requests', 'results', 'reviews', 'baseline-challenge', 'baseline-primary',
                'baseline-challenge-longtokens', 'baseline-primary-longtokens', 'requests', 'extracted',
                'measure64', 'raw', 'logs-attempts'}
RUN_KEEP_SUFFIXES = ('.png', '.gif')
RUN_KEEP_NAMES = {'README.md', 'INDEX.md', 'final-report.json', 'goal.json', 'spec.json', 'mission.json',
                  'key.json', 'variant-notes.json'}


def move(path: Path, target: Path, moves: list):
    if not path.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f'target already exists: {target}')
    shutil.move(str(path), str(target))
    moves.append((str(path.relative_to(ROOT)), str(target.relative_to(ROOT))))


def tidy_artifacts(apply: bool):
    artifacts = ROOT / 'artifacts'
    moves: list = []
    plan = {'sprites': [], 'reviews': [], 'studies': [], 'archive': []}
    for name in sorted(p.name for p in artifacts.iterdir()):
        if name in {'sprites', 'reviews', 'studies', 'archive', 'INDEX.md', 'MOVES.md'}:
            continue
        base = name[:-len('.provenance.json')] if name.endswith('.provenance.json') else name
        if name in SPRITES or base in SPRITES:
            plan['sprites'].append(name)
        elif name in REVIEWS:
            plan['reviews'].append(name)
        elif name in STUDIES:
            plan['studies'].append(name)
        else:
            plan['archive'].append(name)
    if not apply:
        return plan, moves
    for name in plan['sprites']:
        base = name[:-len('.provenance.json')] if name.endswith('.provenance.json') else name
        slug = SPRITES[base]
        target = artifacts / 'sprites' / slug / ('provenance.json' if name.endswith('.provenance.json') else 'neutral.json')
        if name.endswith('.provenance.json'):
            move(artifacts / name, target, moves)
        else:
            move(artifacts / name / 'neutral.json', target, moves)
            leftover = [p for p in (artifacts / name).iterdir()] if (artifacts / name).exists() else []
            for item in leftover:
                move(item, artifacts / 'sprites' / slug / 'internal' / item.name, moves)
            if (artifacts / name).exists() and not any((artifacts / name).iterdir()):
                (artifacts / name).rmdir()
    for bucket, names in (('reviews', plan['reviews']), ('studies', plan['studies']), ('archive', plan['archive'])):
        for name in names:
            move(artifacts / name, artifacts / bucket / name, moves)
    for slug, sources in SPRITE_SOURCES.items():
        for source in sources:
            if (artifacts / source).exists():
                move(artifacts / source, artifacts / 'sprites' / slug / 'internal' / Path(source).name, moves)
    return plan, moves


def tidy_runs(apply: bool):
    runs = ROOT / 'runs'
    moves: list = []
    if not apply:
        plan = {bucket: names for bucket, names in RUN_BUCKETS.items()}
        plan['internal_per_run'] = {}
        for bucket, names in RUN_BUCKETS.items():
            for name in names:
                run = runs / name
                if run.exists():
                    plan['internal_per_run'][name] = sorted(
                        p.name for p in run.iterdir()
                        if p.name in RUN_INTERNAL or p.name.endswith(RUN_LOGS))
        return plan, moves
    for bucket, names in RUN_BUCKETS.items():
        for name in names:
            source = runs / name
            if source.exists():
                move(source, runs / bucket / name, moves)
    for bucket, names in RUN_BUCKETS.items():
        for name in names:
            run = runs / bucket / name
            if not run.exists():
                continue
            (run / 'logs').mkdir(exist_ok=True)
            (run / 'internal').mkdir(exist_ok=True)
            for item in sorted(run.iterdir()):
                if item.name in {'logs', 'internal'}:
                    continue
                if item.is_dir() and item.name in RUN_INTERNAL:
                    move(item, run / 'internal' / item.name, moves)
                elif item.is_file() and item.name.endswith(RUN_LOGS):
                    move(item, run / 'logs' / item.name, moves)
                elif item.is_file() and not item.name.endswith(RUN_KEEP_SUFFIXES) and item.name not in RUN_KEEP_NAMES:
                    move(item, run / 'internal' / item.name, moves)
    return {}, moves


def write_moves(name, moves):
    path = ROOT / name / 'MOVES.md'
    lines = ['# Path moves', '',
             'This tree was reorganised once by `tidy.py`. Historical manifests and provenance files still',
             'name the old locations, so use this table to trace an old path to its new one.', '',
             '| old | new |', '|---|---|']
    lines += [f'| `{old}` | `{new}` |' for old, new in sorted(moves)]
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='perform the moves; default is a dry run')
    parser.add_argument('--only', choices=('artifacts', 'runs'), help='limit to one tree')
    args = parser.parse_args()
    result = {}
    if args.only in (None, 'artifacts'):
        plan, moves = tidy_artifacts(args.apply)
        result['artifacts'] = {'plan': plan, 'moves': len(moves)}
        if args.apply:
            result['artifacts']['moves_file'] = str(write_moves('artifacts', moves))
    if args.only in (None, 'runs'):
        plan, moves = tidy_runs(args.apply)
        result['runs'] = {'plan': plan, 'moves': len(moves)}
        if args.apply:
            result['runs']['moves_file'] = str(write_moves('runs', moves))
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
