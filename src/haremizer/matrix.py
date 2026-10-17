"""Build comparison matrix pages: the original render plus one column per completed stage.

Column 1 of every page is the matched no-LoRA baseline ("original"). Each later column is one
training stage, so a page holds at most `columns` stage columns plus the original. Pages repeat the
original column, so every page is readable on its own.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

FONT_PATHS = ('/usr/share/fonts/TTF/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def pil():
    """Deferred PIL import so the module stays importable without Pillow."""
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


def font(size):
    ImageFont = pil()[2]
    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_log(path):
    """Index -> log entry, tolerant of concatenated lines."""
    entries = {}
    if not Path(path).exists():
        return entries
    decoder = json.JSONDecoder()
    text = Path(path).read_text()
    position = 0
    while position < len(text):
        while position < len(text) and text[position] in ' \n\r\t':
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        entries[value.get('index')] = value
    return entries


def discover_stages(run, cohort_size=8):
    """Completed stages ordered by global index, skipping superseded preflight loops."""
    stages = []
    for loop_dir in sorted((Path(run) / 'loops').glob('loop-*')):
        if 'preflight' in loop_dir.name or not loop_dir.name.split('-')[-1].isdigit():
            continue
        cohort = int(loop_dir.name.split('-')[-1]) - 1
        for stage_dir in sorted(loop_dir.glob('stage-*')):
            if not (stage_dir / 'sample-0').is_dir() or not (stage_dir / 'sample-1').is_dir():
                continue
            number = int(stage_dir.name.split('-')[-1])
            stages.append({'index': cohort * cohort_size + number, 'directory': stage_dir})
    stages.sort(key=lambda row: row['index'])
    for row in stages:
        row['name'] = 'unknown'
        result = Path(run) / 'results' / f"result-{row['index']:03}.json"
        if result.is_file():
            row['name'] = json.loads(result.read_bytes()).get('hypothesis', 'unknown')
    return stages


def native64(path, size):
    """Same 64px rendition used in review: BOX resize, alpha>=128, 32 opaque colours."""
    Image = pil()[0]
    with Image.open(path) as image:
        small = image.convert('RGBA').resize((64, 64), Image.Resampling.BOX)
    pixels = list(small.getdata())
    opaque = [index for index, value in enumerate(pixels) if value[3] >= 128]
    result = [(0, 0, 0, 0)] * len(pixels)
    if opaque:
        strip = Image.new('RGB', (len(opaque), 1))
        strip.putdata([pixels[index][:3] for index in opaque])
        quantized = strip.quantize(colors=32, method=Image.Quantize.MAXCOVERAGE,
                                   dither=Image.Dither.NONE).convert('RGB')
        for index, color in zip(opaque, quantized.getdata()):
            result[index] = (*color, 255)
    out = Image.new('RGBA', (64, 64))
    out.putdata(result)
    return out.resize((size, size), Image.Resampling.NEAREST)


def suite_cases(run, suite_name):
    path = Path(run) / f'suite-{suite_name}.json'
    if not path.is_file():
        return [f'case {i}' for i in range(1, 5)]
    return [case['id'] for case in json.loads(path.read_bytes())['cases']]


def build(run, output, columns=8, cell=176, max_pages=None):
    Image, ImageDraw = pil()[0], pil()[1]
    run = Path(run).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = read_log(run / 'log.jsonl')
    stages = discover_stages(run)
    suites = [('primary', run / 'baseline-primary'), ('challenge', run / 'baseline-challenge')]
    suites = [(name, path) for name, path in suites if path.is_dir()]
    if not suites:
        raise SystemExit('no baseline directories found')
    rows = []
    for name, path in suites:
        for case_number, case_id in enumerate(suite_cases(run, name), 1):
            rows.append({'suite': name, 'case': case_number, 'id': case_id,
                         'baseline': path / f'case-{case_number:02}.png'})
    per_page = max(1, columns - 1)
    label_width = 150
    header = 76
    cell_height = cell + cell // 2 + 30
    pages = [stages[i:i + per_page] for i in range(0, len(stages), per_page)] or [[]]
    if max_pages:
        pages = pages[:max_pages]
    written = []
    for page_number, page in enumerate(pages, 1):
        width = label_width + (1 + len(page)) * cell
        height = header + len(rows) * cell_height
        sheet = Image.new('RGB', (width, height), '#3c3c3c')
        draw = ImageDraw.Draw(sheet)
        small, bold = font(13), font(15)
        draw.text((6, 7), f'page {page_number}  |  column 1 = original no-LoRA render  |  '
                          f'columns 2+ are training stages  |  generated from {run}',
                  fill='white', font=bold)
        headers = [('original', entries.get(0))]
        for stage in page:
            entry = entries.get(stage['index'])
            score = f"{entry['score']:.3f}" if entry else 'n/a'
            mark = 'ok' if entry and entry.get('accepted') else 'rej'
            headers.append((f"{stage['index']} {stage['name']}\n{score} {mark}", entry))
        for column, (title, entry) in enumerate(headers):
            x = label_width + column * cell
            draw.rectangle((x, header, x + cell - 1, header + cell_height - 1), outline='#5a5a5a')
            draw.text((x + 4, 30), title.split('\n')[0][:24], fill='#ffe9a8', font=small)
            if '\n' in title:
                draw.text((x + 4, 48), title.split('\n')[1], fill='#9fd6ff', font=small)
        for row_number, row in enumerate(rows):
            y = header + row_number * cell_height
            draw.text((6, y + 6), f"{row['suite']} {row['case']}", fill='white', font=bold)
            draw.text((6, y + 24), row['id'][:18], fill='#bbbbbb', font=small)
            sources = [row['baseline']] + [stage['directory'] /
                                           ('sample-0' if row['suite'] == 'primary' else 'sample-1') /
                                           f"case-{row['case']:02}.png" for stage in page]
            for column, source in enumerate(sources):
                x = label_width + column * cell
                if not source.is_file():
                    draw.text((x + 6, y + cell // 2), 'missing', fill='#ff8080', font=small)
                    continue
                with Image.open(source) as image:
                    sheet.paste(image.convert('RGB').resize((cell - 4, cell - 4),
                                                            Image.Resampling.LANCZOS), (x + 2, y + 2))
                sheet.paste(native64(source, cell // 2), (x + 2, y + cell + 2))
        path = output / f'page-{page_number:02d}.png'
        sheet.save(path)
        written.append(str(path))
    legend = {'pages': written, 'stages': [{'index': s['index'], 'name': s['name'],
                                            'directory': str(s['directory'])} for s in stages],
              'columns_per_page': columns, 'rows': len(rows),
              'note': 'column 1 of every page is the matched no-LoRA baseline; '
                      'the native-64 rendition sits under each 768 render'}
    (output / 'legend.json').write_text(json.dumps(legend, indent=2) + '\n')
    (output / 'regenerate.sh').write_text(
        '#!/bin/sh\nexec ' + shlex.join([
            sys.executable, '-I', '-B', '-m', 'haremizer.matrix', '--run', str(run),
            '--output', str(output), '--columns', str(columns)]) + '\n')
    (output / 'regenerate.sh').chmod(0o755)
    return legend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--cell', type=int, default=176)
    parser.add_argument('--max-pages', type=int)
    args = parser.parse_args()
    legend = build(args.run, args.output, args.columns, args.cell, args.max_pages)
    print(json.dumps({'pages': legend['pages'], 'stages': len(legend['stages'])}, indent=2))


if __name__ == '__main__':
    main()
