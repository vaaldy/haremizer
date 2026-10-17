"""Explicit, hash-bound local eye-region pixel edits; no eye detection or quality verdict."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from .images import pillow, validate_png


def blink(source, annotation, output):
    source, annotation, output = (Path(path).resolve() for path in (source, annotation, output))
    if (output.exists() or output.is_relative_to(source.parent) or
            source.is_relative_to(output) or annotation.is_relative_to(output)):
        raise ValueError("new output directory outside the input image directory required")
    if not source.is_file() or not 0 < source.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("source must be a nonempty local PNG <=16 MiB")
    with source.open("rb") as file:
        raw = file.read(16 * 1024 * 1024 + 1)
    if not 0 < len(raw) <= 16 * 1024 * 1024:
        raise ValueError("source must be a nonempty local PNG <=16 MiB")
    annotation_bytes = annotation.read_bytes()
    try:
        declaration = json.loads(annotation_bytes)
    except (ValueError, UnicodeError) as error:
        raise ValueError("annotation must be UTF-8 JSON") from error
    if not isinstance(declaration, dict) or set(declaration) != {
            "version", "reviewed_by", "source_sha256", "eye_regions", "edits"}:
        raise ValueError("annotation requires version, reviewed_by, source_sha256, eye_regions, edits")
    if type(declaration["version"]) is not int or declaration["version"] != 1:
        raise ValueError("annotation version must be integer 1")
    reviewer = declaration["reviewed_by"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewed_by must be a nonblank truthful reviewer declaration")
    source_hash = hashlib.sha256(raw).hexdigest()
    if declaration["source_sha256"] != source_hash:
        raise ValueError("source_sha256 must match the full source byte SHA256")
    source_metrics = validate_png(BytesIO(raw), allow_banners=True)
    Image = pillow()
    with Image.open(BytesIO(raw)) as image:
        rgba = image.convert("RGBA")
    width, height = rgba.size
    regions = declaration["eye_regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 2:
        raise ValueError("eye_regions must contain one or two rectangles")
    allowed = set()
    for region in regions:
        if (not isinstance(region, dict) or set(region) != {"x", "y", "width", "height"} or
                any(type(value) is not int for value in region.values())):
            raise ValueError("eye region coordinates and dimensions must be exact integers")
        x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
        if not (0 <= x < width and 0 <= y < height and 1 <= w <= 16 and 1 <= h <= 10 and
                x + w <= width and y + h <= height):
            raise ValueError("eye region must be in bounds, width <=16 and height <=10")
        allowed.update((px, py) for py in range(y, y + h) for px in range(x, x + w))
    if len(allowed) > 128:
        raise ValueError("eye region union must contain <=128 pixels")
    edits = declaration["edits"]
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a nonempty list")
    original = list(rgba.getdata())
    palette = {pixel[:3] for pixel in original if pixel[3] == 255}
    pixels = original.copy()
    seen = set()
    for edit in edits:
        if (not isinstance(edit, dict) or set(edit) != {"x", "y", "color"} or
                type(edit["x"]) is not int or type(edit["y"]) is not int):
            raise ValueError("each edit requires exact integer x,y and color")
        coordinate = (edit["x"], edit["y"])
        if coordinate in seen:
            raise ValueError("duplicate edit coordinates")
        seen.add(coordinate)
        if coordinate not in allowed:
            raise ValueError("edit must be inside a declared eye region")
        color = edit["color"]
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("color must be #RRGGBB")
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        if rgb not in palette:
            raise ValueError("edit color must exist in the opaque source palette")
        index = edit["y"] * width + edit["x"]
        if original[index][3] != 255:
            raise ValueError("edit source pixel must be opaque")
        pixels[index] = (*rgb, 255)
    changed = sum(before != after for before, after in zip(original, pixels))
    if not 0 < changed <= 128:
        raise ValueError("actual changed pixel count must be 1..128")
    rgba.putdata(pixels)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.png").write_bytes(raw)
    (output / "annotation.json").write_bytes(annotation_bytes)
    destination = output / "blink.png"
    rgba.save(destination, format="PNG")
    metrics = validate_png(destination, (width,), allow_banners=True)
    with Image.open(destination) as image:
        result = list(image.convert("RGBA").getdata())
    outside_unchanged = all(before == after for index, (before, after) in
                            enumerate(zip(original, result))
                            if (index % width, index // width) not in allowed)
    alpha_unchanged = all(before[3] == after[3] for before, after in zip(original, result))
    if result != pixels or not outside_unchanged or not alpha_unchanged:
        raise ValueError("saved output does not preserve specified edits, outside pixels and alpha")
    report = {
        "version": 1,
        "source": {"path": str(source), "copy": "source.png", "sha256": source_hash,
                   "bytes": len(raw), "metrics": source_metrics},
        "annotation": {"path": str(annotation), "copy": "annotation.json",
                       "sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                       "bytes": len(annotation_bytes), "reviewed_by": reviewer},
        "eye_regions": regions, "allowed_pixels": len(allowed),
        "specified_edits": len(edits), "changed_pixels": changed,
        "output": {"file": "blink.png", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                   "metrics": metrics},
        "outside_eye_unchanged": outside_unchanged, "alpha_unchanged": alpha_unchanged,
        "semantics": "declared eye regions not automatic detection",
        "visual_quality_assessed": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        report = blink(**vars(parser.parse_args()))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
