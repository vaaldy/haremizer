"""Measure dark-ink stroke continuity in sprites; no quality verdict or automatic repair."""
from pathlib import Path

from images import pillow


def ink_stats(source, ink_max_rgb=32, size=64):
    """Count dark-ink pixels and their 4-connected components after nearest resize."""
    if type(ink_max_rgb) is not int or not 0 <= ink_max_rgb <= 255:
        raise ValueError('ink_max_rgb must be an integer 0..255')
    path = Path(source)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError('bounded local PNG required')
    with pillow().open(path) as image:
        rgba = image.convert('RGBA').resize((size, size), pillow().Resampling.NEAREST)
    pixels = list(rgba.getdata())
    ink = {index for index, (r, g, b, a) in enumerate(pixels)
           if a == 255 and max(r, g, b) <= ink_max_rgb}
    seen, sizes = set(), []
    for start in ink:
        if start in seen:
            continue
        stack, component = [start], 0
        seen.add(start)
        while stack:
            index = stack.pop()
            component += 1
            x, y = index % size, index // size
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= neighbour[0] < size and 0 <= neighbour[1] < size:
                    candidate = neighbour[1] * size + neighbour[0]
                    if candidate in ink and candidate not in seen:
                        seen.add(candidate)
                        stack.append(candidate)
        sizes.append(component)
    sizes.sort(reverse=True)
    return {'path': str(path), 'size': size, 'ink_max_rgb': ink_max_rgb,
            'ink_pixels': len(ink), 'components': len(sizes), 'largest_component': sizes[0] if sizes else 0,
            'isolated_pixels': sum(1 for value in sizes if value == 1),
            'fragments_under_4': sum(1 for value in sizes if value < 4),
            'mean_component': round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
            'ink_fraction': round(len(ink) / (size * size), 4),
            'scope': 'dark ink pixels only; not an overall quality or adherence verdict'}
