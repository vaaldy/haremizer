"""Detect defect classes in a rendered image or a 64px sprite and emit masks. Never repairs anything.

Classes:
  stray_thin_lines  long thin opaque components that are not the character silhouette
  specks            opaque components smaller than four pixels
  backdrop_residue  opaque components whose colour matches the background but are not the silhouette
  border_contact    the subject reaches the top edge, so the head is likely cropped
  busy_background   the border is not dominated by one colour, so a frame, panel or scenery is present
  empty_alpha       almost everything is transparent, so the subject is missing
  format            wrong size, too many colours, or soft alpha on a sprite

The detector is read-only: it opens the input, writes a JSON report and mask PNGs, and never modifies
the input file. Face-region colour checks are deliberately not implemented yet: locating the face
reliably is unsolved, and a wrong box would produce confident nonsense.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

CLASSES = ('stray_thin_lines', 'specks', 'backdrop_residue', 'border_contact', 'busy_background',
           'empty_alpha', 'format')
SPECK_MAX = 4
THIN_MAX = 2
THIN_MIN_LENGTH = 8
MAX_SPRITE_COLOURS = 62


def load_png(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('single frame PNG required')
        image.load()
        return image.convert('RGBA')


def components(pixels, size):
    opaque = {index for index, value in enumerate(pixels) if value[3] == 255}
    seen, found = set(), []
    for start in opaque:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            index = stack.pop()
            comp.append(index)
            x, y = index % size, index // size
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < size and 0 <= ny < size:
                    other = ny * size + nx
                    if other in opaque and other not in seen:
                        seen.add(other)
                        stack.append(other)
        found.append(comp)
    found.sort(key=len, reverse=True)
    return found


def border_colour(pixels, size, band=3):
    from collections import Counter
    counts = Counter()
    for index, value in enumerate(pixels):
        if value[3] != 255:
            continue
        x, y = index % size, index // size
        if x < band or y < band or x >= size - band or y >= size - band:
            counts[value[:3]] += 1
    return counts.most_common(1)[0][0] if counts else (255, 255, 255)


def count_dark_lines(pixels, size, luminance=90, ratio=0.9, minimum=8):
    """Count rows and columns that are almost entirely dark ink, the signature of a drawn frame or panel.

    Measured on the corpus at 768: a drawn frame scores 15 dark rows, a grey side panel 23 dark columns,
    and the clean renders score zero on both. This replaces a thin-stroke test that could not separate a
    frame stroke from a large subject blob. The threshold is calibrated at render resolution: a 64px
    rendition thins a frame stroke to one or two pixels, so this rule may not fire there.
    """
    def dark(value):
        return value[3] == 255 and (value[0] + value[1] + value[2]) / 3 < luminance
    rows = 0
    columns = 0
    for y in range(size):
        if sum(1 for x in range(size) if dark(pixels[y * size + x])) >= ratio * size:
            rows += 1
    for x in range(size):
        if sum(1 for y in range(size) if dark(pixels[y * size + x])) >= ratio * size:
            columns += 1
    return rows, columns


def residue_colour(pixels, silhouette, tolerance=None):
    """Most common colour among opaque pixels outside the silhouette; the likely keyed backdrop."""
    from collections import Counter
    counts = Counter(value[:3] for index, value in enumerate(pixels)
                     if value[3] == 255 and index not in silhouette)
    return counts.most_common(1)[0][0] if counts else (0, 0, 0)


def near(colour, reference, tolerance):
    return sum((colour[i] - reference[i]) ** 2 for i in range(3)) <= tolerance * tolerance


def detect(path, size=None, tolerance=28):
    """Return a defect report plus a mask (set of pixel indexes) per class.

    Sprite mode is chosen when the image has any transparent pixel: the background is transparency and
    the additive classes apply. Render mode is chosen for fully opaque output: the background is the
    modal border colour and only background and framing classes apply.
    """
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError('bounded local PNG required')
    raw = path.read_bytes()
    image = load_png(path)
    if image.width != image.height:
        raise ValueError('square images only')
    if size is None:
        size = image.width
    if image.width != size:
        image = image.resize((size, size), image.Resampling.NEAREST)
    pixels = list(image.getdata())
    transparent = sum(1 for value in pixels if value[3] == 0)
    soft = sum(1 for value in pixels if 0 < value[3] < 255)
    visible = [value for value in pixels if value[3] == 255]
    colours = {value[:3] for value in visible}
    masks = {name: set() for name in CLASSES}
    problems = []
    sprite = size <= 128 and transparent > 0
    if sprite:
        comps = components(pixels, size)
        silhouette = set(comps[0]) if comps else set()
        background = border_colour(pixels, size)
        for position, comp in enumerate(comps):
            if position == 0:
                continue
            xs = [index % size for index in comp]
            ys = [index // size for index in comp]
            width = max(xs) - min(xs) + 1
            height = max(ys) - min(ys) + 1
            if len(comp) < SPECK_MAX:
                masks['specks'].update(comp)
            elif (width <= THIN_MAX and height >= THIN_MIN_LENGTH) or (height <= THIN_MAX and width >= THIN_MIN_LENGTH):
                masks['stray_thin_lines'].update(comp)
            else:
                reference = residue_colour(pixels, silhouette)
                close = sum(1 for index in comp if near(pixels[index][:3], reference, tolerance))
                if close / len(comp) >= 0.8:
                    masks['backdrop_residue'].update(comp)
        if silhouette:
            top = min(index // size for index in silhouette)
            if top <= 1:
                masks['border_contact'] = {index for index in silhouette if index // size == top}
        if size != 64:
            problems.append(f'size {size} is not 64')
        if len(colours) > MAX_SPRITE_COLOURS:
            problems.append(f'{len(colours)} opaque colours exceeds {MAX_SPRITE_COLOURS}')
        if soft:
            problems.append(f'{soft} soft-alpha pixels')
    else:
        background = border_colour(pixels, size)
        subject = {index for index, value in enumerate(pixels)
                   if not near(value[:3], background, tolerance)}
        size_subject = len(subject)
        comps = []
        seen = set()
        for start in sorted(subject):
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                index = stack.pop()
                comp.append(index)
                x, y = index % size, index // size
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < size and 0 <= ny < size:
                        other = ny * size + nx
                        if other in subject and other not in seen:
                            seen.add(other)
                            stack.append(other)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        silhouette = set(comps[0]) if comps else set()
        border = [value for index, value in enumerate(pixels)
                  if (index % size < 3 or index // size < 3 or
                      index % size >= size - 3 or index // size >= size - 3)]
        uniform = sum(1 for value in border if near(value[:3], background, tolerance)) / max(1, len(border))
        dark_rows, dark_columns = count_dark_lines(pixels, size)
        drawn_lines = max(dark_rows, dark_columns) >= 8
        if uniform < 0.6 or drawn_lines:
            masks['busy_background'] = {index for index in subject
                                        if index % size < 3 or index // size < 3 or
                                        index % size >= size - 3 or index // size >= size - 3}
            if drawn_lines and not masks['busy_background']:
                masks['busy_background'] = {index for index in subject
                                            if index % size < 40 or index // size < 40 or
                                            index % size >= size - 40 or index // size >= size - 40}
        large = [comp for comp in comps if len(comp) >= 0.02 * size * size]
        if large:
            top = min(index // size for comp in large for index in comp)
            if top <= 1:
                masks['border_contact'] = {index for comp in large for index in comp
                                           if index // size == top}
        if size_subject / (size * size) > 0.995:
            masks['busy_background'] = set(subject)
    if transparent / (size * size) > 0.9:
        masks['empty_alpha'] = {index for index, value in enumerate(pixels) if value[3] == 0}
    if problems:
        masks['format'] = set()
    report = {'input': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': size,
              'mode': 'sprite' if sprite else 'render', 'background_rgb': list(background),
              'tolerance': tolerance, 'opaque_pixels': len(visible), 'transparent_pixels': transparent,
              'soft_alpha_pixels': soft, 'opaque_colours': len(colours),
              'defects': {name: {'pixels': len(masks[name]), 'present': bool(masks[name]) or
                                 (name == 'format' and bool(problems))}
                          for name in CLASSES},
              'format_problems': problems,
              'scope': 'detection only; no repair, no quality verdict'}
    return report, masks


def write_masks(path, size, masks):
    from PIL import Image
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, pixels in sorted(masks.items()):
        image = Image.new('L', (size, size), 0)
        if pixels:
            data = bytearray(size * size)
            for index in pixels:
                data[index] = 255
            image.putdata(list(data))
            target = path / f'{name}.png'
            image.save(target)
            written[name] = str(target)
    return written


def analyse(path, output=None, size=None, tolerance=28):
    report, masks = detect(path, size=size, tolerance=tolerance)
    if output is not None:
        report['masks'] = write_masks(output, report['size'], masks)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--size', type=int)
    parser.add_argument('--tolerance', type=int, default=28)
    args = parser.parse_args()
    report = analyse(args.input, args.output, args.size, args.tolerance)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
