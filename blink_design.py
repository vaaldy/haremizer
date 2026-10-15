"""Design a blink for one sprite, then hand the edits to the guarded blink editor.

Two paths:

* Declared boxes (`--box x,y,w,h`, repeatable): the caller states where the eyes are, and the tool does
  the rest. It samples each eye's own ink colour and the skin colour around that eye, then draws a
  closed-lid arc sized to the box. This is the path to use.
* Automatic detection (`--auto`): experimental. It tries to find the eyes from the sprite's own pixels.
  In practice the face region merges into hair on several sprites and it refuses more often than it
  succeeds, so it is kept only for experimentation.

Either way the output is an annotation for blink.py: declared regions, edits inside them, alpha preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from pathlib import Path

MAX_REGION_WIDTH = 16
MAX_REGION_HEIGHT = 10
MAX_REGION_UNION = 128


def load(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.convert('RGBA').copy()


def near(a, b, tolerance):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) <= tolerance * tolerance


def luminance(colour):
    return (colour[0] + colour[1] + colour[2]) / 3


def cluster(points, size, diagonal=False):
    seen, clusters = set(), []
    neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        neighbours += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for start in sorted(points):
        if start in seen:
            continue
        queue, found = deque([start]), []
        seen.add(start)
        while queue:
            index = queue.popleft()
            found.append(index)
            x, y = index % size, index // size
            for dx, dy in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    other = ny * size + nx
                    if other in points and other not in seen:
                        seen.add(other)
                        queue.append(other)
        clusters.append(found)
    return sorted(clusters, key=len, reverse=True)


def find_face(pixels, size, tolerance=40):
    centre = size // 2
    box = [(x, y) for y in range(centre - size // 4, centre + size // 4)
           for x in range(centre - size // 4, centre + size // 4)]
    reference = Counter(pixels[y * size + x][:3] for x, y in box
                        if pixels[y * size + x][3] == 255).most_common(1)[0][0]
    inside = {y * size + x for x, y in box if pixels[y * size + x][3] == 255}
    same = {index for index in inside if near(pixels[index][:3], reference, tolerance)}
    for component in cluster(same, size, diagonal=True):
        if (size // 2) * size + size // 2 in component or len(component) > 40:
            return component, reference
    return max(cluster(same, size, diagonal=True), key=len), reference


def dilate(points, size, radius=2):
    grown = set(points)
    for index in points:
        x, y = index % size, index // size
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    grown.add(ny * size + nx)
    return grown


def find_eyes(pixels, size, face, skin_reference, tolerance=40):
    """Eyes are the dark, non-skin clusters that sit inside the skin area.

    The skin area has to be grown first: eyes are not skin-coloured, so they are never part of the skin
    component itself.
    """
    skin_luminance = luminance(skin_reference)
    area = dilate(face, size, radius=2)
    inside = {index for index in area
              if pixels[index][3] == 255
              and not near(pixels[index][:3], skin_reference, tolerance)
              and luminance(pixels[index][:3]) < skin_luminance * 0.55}
    candidates = []
    for component in cluster(inside, size, diagonal=True):
        if len(component) < 3:
            continue
        xs = [index % size for index in component]
        ys = [index // size for index in component]
        width, height = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if 3 <= width <= 14 and 2 <= height <= 10:
            candidates.append({'pixels': component, 'width': width, 'height': height,
                               'x': min(xs), 'y': min(ys), 'cx': sum(xs) / len(xs),
                               'cy': sum(ys) / len(ys)})
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            if abs(first['cx'] - second['cx']) >= 3 and abs(first['cy'] - second['cy']) <= 6:
                pairs.append((len(first['pixels']) + len(second['pixels']), first, second))
    if not pairs:
        raise ValueError('no pair of plausible eyes found; refusing to guess')
    _, first, second = max(pairs, key=lambda row: row[0])
    return sorted([first, second], key=lambda eye: eye['cx'])


def eye_colours(pixels, eye):
    values = [pixels[index][:3] for index in eye['pixels']]
    ordered = sorted(values, key=luminance)
    darkest = ordered[:max(1, len(ordered) // 3)]
    ink = Counter(darkest).most_common(1)[0][0]
    return ink, sorted(set(ordered))


def surrounding_skin(pixels, size, eye, fallback, depth=3):
    """Most common colour in the strip directly below the eye, which is cheek rather than hair."""
    strip = []
    for y in range(eye['y'] + eye['height'], eye['y'] + eye['height'] + depth):
        for x in range(eye['x'], eye['x'] + eye['width']):
            if 0 <= x < size and 0 <= y < size:
                index = y * size + x
                if index not in eye['pixels'] and pixels[index][3] == 255:
                    strip.append(pixels[index][:3])
    if not strip:
        return fallback
    brightest = max(luminance(colour) for colour in strip)
    light = [colour for colour in strip if luminance(colour) >= brightest - 25]
    return Counter(light).most_common(1)[0][0]


def arc_pixels(eye, amplitude=1):
    """Closed-eye lid: a shallow smile-shaped line across the eye, thicker in the middle."""
    x0, x1 = eye['x'], eye['x'] + eye['width'] - 1
    middle = round(eye['cy'])
    span = max(1, x1 - x0)
    pixels = {}
    for x in range(x0, x1 + 1):
        t = (x - x0) / span
        lift = round(amplitude * (1 - 2 * abs(t - 0.5)))
        pixels[(x, middle + lift)] = True
        if abs(t - 0.5) <= 0.25:
            pixels[(x, middle + lift + 1)] = True
    return pixels


def design_declared(sprite_path, boxes, amplitude=1):
    sprite_path = Path(sprite_path)
    image = load(sprite_path)
    size = image.width
    pixels = list(image.getdata())
    _, skin_reference = find_face(pixels, size)
    edits, regions, report = {}, [], []
    for box in boxes:
        x, y, width, height = box
        if width > MAX_REGION_WIDTH or height > MAX_REGION_HEIGHT:
            raise ValueError(f'declared box {box} is outside the editor limits')
        pixels_in_box = {yy * size + xx for yy in range(y, y + height) for xx in range(x, x + width)
                         if 0 <= xx < size and 0 <= yy < size and pixels[yy * size + xx][3] == 255}
        if not pixels_in_box:
            raise ValueError(f'declared box {box} contains no opaque pixels')
        values = sorted((pixels[index][:3] for index in pixels_in_box), key=luminance)
        darkest, brightest = luminance(values[0]), luminance(values[-1])
        cutoff = darkest + 0.25 * max(1.0, brightest - darkest)
        ink = Counter(colour for colour in values if luminance(colour) <= cutoff).most_common(1)[0][0]
        eye = {'pixels': pixels_in_box, 'x': x, 'y': y, 'width': width, 'height': height,
               'cx': x + (width - 1) / 2, 'cy': y + (height - 1) / 2}
        skin = surrounding_skin(pixels, size, eye, skin_reference)
        # the eye itself is the lash span, not the whole declared box: the box may clip hair or cheek
        lash = {index for index in pixels_in_box if luminance(pixels[index][:3]) <= cutoff}
        if not lash:
            raise ValueError(f'declared box {box} holds no lash pixels to anchor the lid')
        lash_x = [index % size for index in lash]
        lash_y = [index // size for index in lash]
        span = {'x': min(lash_x), 'y': min(lash_y), 'width': max(lash_x) - min(lash_x) + 1,
                'height': max(lash_y) - min(lash_y) + 1, 'cx': sum(lash_x) / len(lash_x),
                'cy': sum(lash_y) / len(lash_y), 'pixels': set()}
        skin_luminance = luminance(skin)
        for index in pixels_in_box:
            px_x, px_y = index % size, index // size
            if span['x'] <= px_x < span['x'] + span['width'] and luminance(pixels[index][:3]) < skin_luminance * 0.8:
                edits[(px_x, px_y)] = skin
        for (ax, ay), _ in arc_pixels(span, amplitude).items():
            if 0 <= ax < size and 0 <= ay < size and pixels[ay * size + ax][3] == 255:
                edits[(ax, ay)] = ink
        regions.append({'x': x, 'y': y, 'width': width, 'height': height})
        report.append({'region': regions[-1], 'pixels': len(pixels_in_box),
                       'ink': '#%02x%02x%02x' % ink, 'skin': '#%02x%02x%02x' % skin})
    edits = {coordinate: colour for coordinate, colour in edits.items()
             if pixels[coordinate[1] * size + coordinate[0]][:3] != colour}
    annotation = {'version': 1,
                  'reviewed_by': 'machine:blink_design; eye boxes declared by the caller and verified on a '
                                 'high-zoom grid, arc and colours derived from this sprite; not human review',
                  'source_sha256': hashlib.sha256(sprite_path.read_bytes()).hexdigest(),
                  'eye_regions': regions,
                  'edits': [{'x': x, 'y': y, 'color': '#%02x%02x%02x' % colour}
                            for (x, y), colour in sorted(edits.items())]}
    return annotation, report


def design(sprite_path, amplitude=1):
    sprite_path = Path(sprite_path)
    image = load(sprite_path)
    size = image.width
    if image.width != image.height:
        raise ValueError('square sprites only')
    pixels = list(image.getdata())
    face, skin_reference = find_face(pixels, size)
    eyes = find_eyes(pixels, size, face, skin_reference)
    edits, regions, report = {}, [], []
    for eye in eyes:
        ink, palette = eye_colours(pixels, eye)
        skin = surrounding_skin(pixels, size, eye, skin_reference)
        for index in eye['pixels']:
            edits[(index % size, index // size)] = skin
        for (x, y), _ in arc_pixels(eye, amplitude).items():
            if 0 <= x < size and 0 <= y < size:
                edits[(x, y)] = ink
        region = {'x': eye['x'], 'y': eye['y'], 'width': eye['width'], 'height': eye['height']}
        regions.append(region)
        report.append({'region': region, 'pixels': len(eye['pixels']), 'ink': '#%02x%02x%02x' % ink,
                       'skin': '#%02x%02x%02x' % skin, 'eye_palette': ['#%02x%02x%02x' % c for c in palette]})
    edits = {coordinate: colour for coordinate, colour in edits.items()
             if pixels[coordinate[1] * size + coordinate[0]][:3] != colour}
    union = set()
    for region in regions:
        if (region['width'] > MAX_REGION_WIDTH or region['height'] > MAX_REGION_HEIGHT or
                region['width'] < 1 or region['height'] < 1):
            raise ValueError(f"eye region {region} is outside the editor's limits")
        union.update((x, y) for y in range(region['y'], region['y'] + region['height'])
                     for x in range(region['x'], region['x'] + region['width']))
    if len(union) > MAX_REGION_UNION:
        raise ValueError('eye regions union is larger than the editor allows')
    annotation = {'version': 1,
                  'reviewed_by': 'machine:blink_design; eyes located from this sprite\'s own pixels, '
                                 'not declared by hand and not human review',
                  'source_sha256': hashlib.sha256(sprite_path.read_bytes()).hexdigest(),
                  'eye_regions': regions,
                  'edits': [{'x': x, 'y': y, 'color': '#%02x%02x%02x' % colour}
                            for (x, y), colour in sorted(edits.items())]}
    return annotation, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sprite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amplitude', type=int, default=1)
    parser.add_argument('--box', action='append', default=[], metavar='x,y,w,h',
                        help='declared eye box; repeat for each eye')
    parser.add_argument('--auto', action='store_true', help='experimental automatic eye detection')
    args = parser.parse_args()
    if args.box:
        boxes = [tuple(int(part) for part in value.split(',')) for value in args.box]
        for box in boxes:
            if len(box) != 4:
                raise SystemExit('each --box needs x,y,w,h')
        annotation, report = design_declared(args.sprite, boxes, args.amplitude)
    elif args.auto:
        annotation, report = design(args.sprite, args.amplitude)
    else:
        raise SystemExit('supply --box x,y,w,h (repeatable) or --auto')
    args.output.write_text(json.dumps(annotation, indent=2) + '\n')
    print(json.dumps({'annotation': str(args.output), 'edits': len(annotation['edits']),
                      'eyes': report}, indent=2))


if __name__ == '__main__':
    main()
