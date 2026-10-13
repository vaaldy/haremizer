"""Hash-bound, explicitly annotated sprite ink checks; not automatic landmarks or LoRA evaluation."""
import hashlib
from io import BytesIO
from pathlib import Path
import re

from images import pillow, validate_png


def check_legibility(source, annotation):
    source = Path(source)
    if not source.is_file() or not 0 < source.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError('bounded local PNG required')
    raw = source.read_bytes()
    if not isinstance(annotation, dict) or set(annotation) != {
            'version', 'reviewed_by', 'source_sha256', 'mouth_pixels',
            'mouth_colors', 'uniform_strokes', 'uniform_color'}:
        raise ValueError('explicit ink annotation fields required')
    if (type(annotation['version']) is not int or annotation['version'] != 1 or
            not isinstance(annotation['reviewed_by'], str) or not annotation['reviewed_by'].strip()):
        raise ValueError('version1 and truthful reviewer required')
    if annotation['source_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('source hash mismatch')
    metrics = validate_png(BytesIO(raw))
    with pillow().open(BytesIO(raw)) as image:
        image = image.convert('RGBA')

    def points(values, maximum):
        if not isinstance(values, list) or not 2 <= len(values) <= maximum:
            raise ValueError('bounded explicit ink pixel list required')
        result = []
        for value in values:
            if (not isinstance(value, list) or len(value) != 2 or
                    any(type(n) is not int for n in value) or
                    not 0 <= value[0] < image.width or not 0 <= value[1] < image.height):
                raise ValueError('ink coordinates must be in-bounds integers')
            point = tuple(value)
            if point in result or image.getpixel(point)[3] != 255:
                raise ValueError('ink points must be unique and opaque')
            result.append(point)
        return result

    def color(value):
        if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
            raise ValueError('ink color must be #RRGGBB')
        return tuple(bytes.fromhex(value[1:]))

    mouth = points(annotation['mouth_pixels'], 24)
    allowed = annotation['mouth_colors']
    if not isinstance(allowed, list) or not 1 <= len(allowed) <= 2:
        raise ValueError('mouth requires one or two declared ink colors')
    allowed = {color(c) for c in allowed}
    observed = {image.getpixel(p)[:3] for p in mouth}
    mouth_pass = len(observed) <= 2 and observed <= allowed
    strokes = annotation['uniform_strokes']
    if not isinstance(strokes, list) or not 1 <= len(strokes) <= 4:
        raise ValueError('one to four explicit uniform strokes required')
    expected = color(annotation['uniform_color'])
    reports = []
    for stroke in strokes:
        path = points(stroke, 64)
        if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) != 1 for a, b in zip(path, path[1:])):
            raise ValueError('uniform paths must be ordered eight-connected pixels')
        colors = {image.getpixel(p)[:3] for p in path}
        reports.append({'pixels': len(path), 'swatches': len(colors),
                        'passed': colors == {expected}})
    passed = mouth_pass and all(r['passed'] for r in reports)
    return {'source_sha256': hashlib.sha256(raw).hexdigest(), 'metrics': metrics,
            'mouth': {'pixels': len(mouth), 'swatches': len(observed), 'passed': mouth_pass},
            'uniform_strokes': reports, 'passed': passed, 'redo_required': not passed,
            'scope': 'explicit intended ink pixels only; no automatic localization, background or overall quality verdict',
            'reviewed_by': annotation['reviewed_by']}
