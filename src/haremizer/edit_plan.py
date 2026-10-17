"""Source-bound edit permissions, not detection or guaranteed semantic generation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

from .images import pillow


class Abstention(ValueError):
    """The annotation cannot safely support the requested operation."""


def _image(raw):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 16 * 1024 * 1024:
        raise ValueError("source must be PNG bytes <=16 MiB")
    Image = pillow()
    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG" or not (0 < image.width <= 1024 and 0 < image.height <= 1024):
            raise ValueError("PNG dimensions must be 1..1024")
        return image.convert("RGBA")


def _rect(value, size):
    if (not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"} or
            any(type(v) is not int for v in value.values())):
        raise ValueError("rectangle requires exact integer x, y, width, height")
    x, y, w, h = (value[k] for k in ("x", "y", "width", "height"))
    if not (0 <= x < size[0] and 0 <= y < size[1] and w > 0 and h > 0 and
            x + w <= size[0] and y + h <= size[1]):
        raise ValueError("rectangle out of bounds")
    return x, y, w, h


def _pixels(rect):
    x, y, w, h = rect
    return frozenset((px, py) for py in range(y, y + h) for px in range(x, x + w))


@dataclass(frozen=True)
class BlinkPlan:
    source_sha256: str
    size: tuple
    fraction: float
    eyes: tuple
    targets: tuple
    safe: frozenset
    mask: frozenset


def plan_blink(source: bytes, annotation: dict, fraction: float) -> BlinkPlan:
    """Round f*eye height (Python ties-to-even); refuse uncertain or clipped targets.

    Coordinates use native pixels and half-open rectangles. Confidence is recorded by
    the provider; only explicitly confirmed, fully confident geometry is actionable.
    """
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1 or not math.isfinite(fraction):
        raise ValueError("fraction must be finite in [0,1], not boolean")
    image = _image(source)
    digest = hashlib.sha256(source).hexdigest()
    if (not isinstance(annotation, dict) or set(annotation) != {
            "version", "source_sha256", "provider", "eyes"} or
            type(annotation["version"]) is not int or annotation["version"] != 1 or
            not isinstance(annotation["provider"], str) or not annotation["provider"].strip()):
        raise ValueError("invalid source-bound annotation")
    if annotation["source_sha256"] != digest:
        raise ValueError("source hash mismatch")
    entries = annotation["eyes"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise Abstention("one or two explicitly confirmed eyes required")
    eyes, targets, safe, mask = [], [], set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"status", "confidence", "rect", "safe_regions"}:
            raise ValueError("invalid eye annotation")
        confidence = entry["confidence"]
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1 or not math.isfinite(confidence):
            raise ValueError("confidence must be finite in [0,1]")
        if entry["status"] != "confirmed" or confidence != 1:
            raise Abstention("uncertain, missing or occluded eye")
        eye = _rect(entry["rect"], image.size)
        permissions = entry["safe_regions"]
        if not isinstance(permissions, list) or not permissions:
            raise Abstention("explicit safe eye/skin regions required")
        local_safe = set().union(*(_pixels(_rect(r, image.size)) for r in permissions))
        x, y, w, h = eye
        target = (x, y + round(fraction * h), w, h)
        target_pixels = _pixels(target)
        envelope = _pixels(eye) | target_pixels
        if any(px >= image.width or py >= image.height for px, py in target_pixels):
            raise Abstention("target out of bounds")
        if not envelope <= local_safe:
            raise Abstention("target or old eye escapes safe domain; clipping would lose motion")
        if any(image.getpixel(p)[3] != 255 for p in envelope):
            raise Abstention("blink envelope must be opaque")
        if any(_pixels(eye) & _pixels(other) for other in eyes) or mask & envelope:
            raise ValueError("overlapping eye envelopes")
        eyes.append(eye)
        targets.append(target)
        safe.update(local_safe)
        mask.update(envelope & local_safe)
    return BlinkPlan(digest, image.size, float(fraction), tuple(eyes), tuple(targets),
                     frozenset(safe), frozenset(mask))


@dataclass(frozen=True)
class ExpressionPlan:
    source_sha256: str
    size: tuple
    operation: str
    regions: tuple
    mask: frozenset


def _expression_mask(operation, regions, size):
    if operation not in ("blink", "happy", "surprised"):
        raise ValueError("unsupported expression operation")
    allowed = {"eyes"} if operation == "blink" else {"eyes", "brows", "mouth"}
    required = {"eyes"} if operation == "blink" else {"eyes", "mouth"}
    labels, roles, mask = set(), set(), set()
    for label, role, rect in regions:
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError("region labels must be nonempty and unique")
        if not isinstance(role, str) or role not in allowed:
            raise ValueError("region role not permitted for operation")
        checked = _rect(dict(zip(("x", "y", "width", "height"), rect)), size)
        if len(rect) != 4:
            raise ValueError("invalid region rectangle")
        pixels = _pixels(checked)
        if mask & pixels:
            raise ValueError("overlapping expression regions")
        labels.add(label)
        roles.add(role)
        mask.update(pixels)
    if not required <= roles:
        raise Abstention("declared eyes and, for expressions, mouth required")
    return frozenset(mask)


def plan_expression(source: bytes, annotation: dict, operation: str) -> ExpressionPlan:
    """Union explicit semantic regions; brows are optional, never inferred.

    Role labels and confidence are trusted provider declarations, not segmentation
    evidence. Every supplied region must be confirmed; none is silently dropped.
    """
    image = _image(source)
    digest = hashlib.sha256(source).hexdigest()
    if (not isinstance(annotation, dict) or set(annotation) != {
            "version", "source_sha256", "size", "provider", "regions"} or
            type(annotation["version"]) is not int or annotation["version"] != 1 or
            not isinstance(annotation["provider"], str) or not annotation["provider"].strip()):
        raise ValueError("invalid source-bound expression annotation")
    if annotation["source_sha256"] != digest:
        raise ValueError("source hash mismatch")
    if (not isinstance(annotation["size"], list) or len(annotation["size"]) != 2 or
            any(type(n) is not int for n in annotation["size"]) or
            tuple(annotation["size"]) != image.size):
        raise ValueError("source dimensions mismatch")
    if not isinstance(annotation["regions"], list):
        raise ValueError("regions must be a list")
    regions = []
    for entry in annotation["regions"]:
        if not isinstance(entry, dict) or set(entry) != {"label", "role", "status", "confidence", "rect"}:
            raise ValueError("invalid semantic region")
        confidence = entry["confidence"]
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1 or not math.isfinite(confidence):
            raise ValueError("confidence must be finite in [0,1]")
        if entry["status"] != "confirmed" or confidence != 1:
            raise Abstention("uncertain, missing or occluded feature")
        regions.append((entry["label"], entry["role"], _rect(entry["rect"], image.size)))
    mask = _expression_mask(operation, regions, image.size)
    if any(image.getpixel(p)[3] != 255 for p in mask):
        raise Abstention("expression regions must be opaque")
    return ExpressionPlan(digest, image.size, operation, tuple(regions), mask)


def _checked_source(source, plan):
    image = _image(source)
    if hashlib.sha256(source).hexdigest() != plan.source_sha256 or image.size != plan.size:
        raise ValueError("source hash or dimensions mismatch")
    if isinstance(plan, ExpressionPlan):
        expected = _expression_mask(plan.operation, plan.regions, image.size)
        safe = expected
    else:
        expected = frozenset().union(*(_pixels(r) for r in (*plan.eyes, *plan.targets)))
        safe = plan.safe
    if (plan.mask != expected or not plan.mask <= safe or
            any(not (0 <= x < image.width and 0 <= y < image.height) or
                image.getpixel((x, y))[3] != 255 for x, y in plan.mask)):
        raise ValueError("plan mask escapes geometry, safe domain or opaque image")
    return image


def validate_candidate(source: bytes, plan: BlinkPlan | ExpressionPlan, candidate) -> dict:
    """Check preservation only; changed pixels do not prove the requested operation."""
    image = _checked_source(source, plan)
    if candidate.mode != "RGBA" or candidate.size != image.size:
        raise ValueError("candidate must be same-size RGBA")
    original, result = list(image.getdata()), list(candidate.getdata())
    palette = {p[:3] for p in original if p[3] == 255}
    changed = 0
    colors = set()
    for i, (before, after) in enumerate(zip(original, result)):
        if before[3] != after[3]:
            raise ValueError("alpha drift")
        if before != after:
            if (i % image.width, i // image.width) not in plan.mask:
                raise ValueError("outside-mask RGBA drift")
            changed += 1
        if after[3] == 255:
            colors.add(after[:3])
    if not colors <= palette or len(colors) > 62:
        raise ValueError("source palette violation or >62 opaque colors")
    return {**({"expression_verified": False} if isinstance(plan, ExpressionPlan) else {}),
            "changed_pixels": changed, "no_op": changed == 0,
            "outside_rgba_changes": 0, "alpha_changes": 0,
            "opaque_colors": len(colors), "placement_verified": False}


def lock_candidate(source: bytes, plan: BlinkPlan | ExpressionPlan, proposal):
    """Composite palette-mapped model RGB only within the planned envelope."""
    image = _checked_source(source, plan)
    if proposal.mode not in ("RGB", "RGBA") or proposal.size != image.size:
        raise ValueError("proposal must be same-size RGB or RGBA")
    palette = sorted({p[:3] for p in image.getdata() if p[3] == 255})
    result = image.copy()
    for coordinate in plan.mask:
        color = proposal.getpixel(coordinate)[:3]
        nearest = min(palette, key=lambda p: sum((p[i] - color[i]) ** 2 for i in range(3)))
        result.putpixel(coordinate, (*nearest, 255))
    validate_candidate(source, plan, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    args = parser.parse_args()
    try:
        plan = plan_blink(args.source.read_bytes(), json.loads(args.annotation.read_text()), args.fraction)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps({"source_sha256": plan.source_sha256, "size": plan.size,
                      "fraction": plan.fraction, "eyes": plan.eyes, "targets": plan.targets,
                      "mask": sorted(plan.mask), "placement_verified": False}, indent=2))


if __name__ == "__main__":
    main()
