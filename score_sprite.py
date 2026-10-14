"""Deterministic scorecard for the overnight goal; not an adherence or quality verdict."""
import hashlib
from io import BytesIO
from pathlib import Path

from images import pillow
from ink_stats import ink_stats

PALETTE_COLORS = 32
SUBJECT_THRESHOLD = 24


def _rendition(raw):
    """Same 64px rendition used in review: BOX resize, alpha>=128, 32 opaque colours."""
    Image = pillow()
    with Image.open(BytesIO(raw)) as probe:
        if probe.format != 'PNG' or getattr(probe, 'n_frames', 1) != 1:
            raise ValueError('single-frame PNG required')
        probe.verify()
    with Image.open(BytesIO(raw)) as image:
        small = image.convert("RGBA").resize((64, 64), Image.Resampling.BOX)
    pixels = list(small.getdata())
    opaque = [index for index, value in enumerate(pixels) if value[3] >= 128]
    result = [(0, 0, 0, 0)] * len(pixels)
    if opaque:
        strip = Image.new("RGB", (len(opaque), 1))
        strip.putdata([pixels[index][:3] for index in opaque])
        quantized = strip.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MAXCOVERAGE,
                                  dither=Image.Dither.NONE).convert("RGB")
        for index, color in zip(opaque, quantized.getdata()):
            result[index] = (*color, 255)
    return result


def measure(source):
    path = Path(source)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError('bounded local PNG required')
    raw = path.read_bytes()
    pixels = _rendition(raw)
    work = Path('/tmp')
    cache = work / f'haremizer-score-{hashlib.sha256(raw).hexdigest()[:16]}.png'
    Image = pillow()
    image = Image.new("RGBA", (64, 64))
    image.putdata(pixels)
    image.save(cache)
    try:
        ink = ink_stats(cache)
    finally:
        cache.unlink(missing_ok=True)
    edges = ([pixels[index] for index in range(64)] + [pixels[63 * 64 + index] for index in range(64)] +
             [pixels[row * 64] for row in range(64)] + [pixels[row * 64 + 63] for row in range(64)])
    visible = [value for value in edges if value[3] == 255]
    background = max(set(visible), key=visible.count) if visible else (255, 255, 255, 255)
    def differs(value):
        return value[3] == 255 and max(abs(a - b) for a, b in zip(value[:3], background[:3])) > SUBJECT_THRESHOLD
    subject = [differs(value) for value in pixels]
    clearance = 0
    for row in range(64):
        if any(subject[row * 64 + column] for column in range(64)):
            break
        clearance += 1
    return {'file': path.name, 'ink_pixels': ink['ink_pixels'], 'components': ink['components'],
            'isolated_share': round(ink['isolated_pixels'] / max(1, ink['components']), 4),
            'largest_component': ink['largest_component'],
            'border_rgb': list(background[:3]),
            'border_fraction': round(sum(1 for value in pixels if value[:3] == background[:3] and value[3] == 255) / 4096, 4),
            'top_clearance_rows': clearance,
            'subject_fraction': round(sum(subject) / 4096, 4),
            'opaque_colors': len({value[:3] for value in pixels if value[3] == 255}),
            'sha256': hashlib.sha256(raw).hexdigest()}


def aggregate(rows):
    if not rows:
        raise ValueError('at least one measured output required')
    keys = ('ink_pixels', 'components', 'isolated_share', 'largest_component', 'border_fraction',
            'top_clearance_rows', 'subject_fraction', 'opaque_colors')
    return {**{key: round(sum(row[key] for row in rows) / len(rows), 4) for key in keys},
            'cases': len(rows)}


def compare(candidate, incumbent, margin=0.02, thresholds=None):
    """Gain-weighted score; regressions are reported separately and block acceptance."""
    thresholds = thresholds or {}
    ink_min = thresholds.get('ink_min', 40)
    ink_max = thresholds.get('ink_max', 400)
    margin = thresholds.get('margin', margin)
    def gain(key, higher_better=True):
        base = incumbent[key]
        if base == 0:
            return 0.0
        value = (candidate[key] - base) / abs(base) if higher_better else (base - candidate[key]) / abs(base)
        return max(0.0, min(1.0, value))
    gains = {'isolated_share': gain('isolated_share', higher_better=False),
             'largest_component': gain('largest_component'),
             'top_clearance_rows': gain('top_clearance_rows')}
    score = round(0.45 * gains['isolated_share'] + 0.35 * gains['largest_component'] +
                  0.20 * gains['top_clearance_rows'], 4)
    regressions = []
    if candidate['isolated_share'] > incumbent['isolated_share'] + 0.05:
        regressions.append('isolated_share')
    if candidate['largest_component'] < incumbent['largest_component'] * 0.9:
        regressions.append('largest_component')
    if candidate['top_clearance_rows'] < incumbent['top_clearance_rows'] - 0.5:
        regressions.append('top_clearance_rows')
    if not ink_min <= candidate['ink_pixels'] <= ink_max:
        regressions.append('ink_pixels_range')
    if candidate.get('border_fraction', 0) < incumbent.get('border_fraction', 0) - 0.05:
        regressions.append('border_fraction')
    gates = {}
    if 'clearance_cases_required' in thresholds:
        gates['framing_clearance_cases'] = (candidate.get('clearance_cases_ok', 0) >=
                                            thresholds['clearance_cases_required'])
    if 'border_min' in thresholds:
        gates['background_border_fraction'] = candidate.get('border_fraction', 0) >= thresholds['border_min']
    if 'isolated_share' in thresholds.get('isolated_gate', {}):
        gates['isolated_share'] = candidate['isolated_share'] <= thresholds['isolated_gate']['isolated_share']
    return {'score': score, 'gains': gains, 'regressions': regressions, 'gates': gates,
            'accepted': score > margin and not regressions,
            'scope': 'deterministic gate only; adherence, attire, blink and alpha need review'}
