"""Render a term-chan sprite JSON frame to PNG so a folder can be read without opening JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

INDEX_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load(character_dir):
    character_dir = Path(character_dir)
    if not character_dir.is_dir():
        raise ValueError('character directory required')
    candidates = [p for p in sorted(character_dir.glob('*.json'))
                  if p.name not in ('provenance.json', 'manifest.json')]
    if len(candidates) != 1:
        raise ValueError(f'expected exactly one sprite JSON, found {[p.name for p in candidates]}')
    return candidates[0], json.loads(candidates[0].read_bytes())


def render(sprite, frame):
    from PIL import Image
    if frame not in sprite['frames']:
        raise ValueError(f'frame {frame} not in {sorted(sprite["frames"])}')
    width, height = sprite['size']
    palette = [tuple(bytes.fromhex(color[1:])) for color in sprite['palette']]
    rows = sprite['frames'][frame]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError('frame dimensions do not match the declared size')
    pixels = []
    for row in rows:
        for char in row:
            pixels.append((0, 0, 0, 0) if char == '.' else (*palette[INDEX_CHARS.index(char)], 255))
    image = Image.new('RGBA', (width, height))
    image.putdata(pixels)
    return image


def render_directory(character_dir, frames=('base', 'blink')):
    """Write <name>.<frame>.png next to the sprite JSON and return the written paths."""
    path, sprite = load(character_dir)
    written = []
    for frame in frames:
        if frame not in sprite['frames']:
            continue
        target = Path(character_dir) / f'{path.stem}-{frame}.png'
        render(sprite, frame).save(target)
        written.append(target)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--character', type=Path, required=True)
    parser.add_argument('--frames', default='base,blink')
    args = parser.parse_args()
    written = render_directory(args.character, [f.strip() for f in args.frames.split(',') if f.strip()])
    print(json.dumps([str(p) for p in written], indent=2))


if __name__ == '__main__':
    main()
