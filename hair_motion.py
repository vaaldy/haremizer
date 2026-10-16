"""Optional source-bound isolated-tip rig, not segmentation or wind physics."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math

from edit_plan import _image


def _number(value, name, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be finite in [{low}, {high}], not boolean")
    return value


def _mask(value, size, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError("nonempty explicit pixel mask required")
    result = set()
    for point in value:
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) is not int for v in point) or
                not (0 <= point[0] < size[0] and 0 <= point[1] < size[1])):
            raise ValueError("mask requires in-bounds integer [x,y] pixels")
        if tuple(point) in result:
            raise ValueError("duplicate mask pixel")
        result.add(tuple(point))
    return frozenset(result)


@dataclass(frozen=True)
class HairPlan:
    source_sha256: str
    size: tuple
    hair: frozenset
    protected: frozenset
    envelope: frozenset
    root_y: int
    tip_y: int
    amplitude: float
    period: float
    backplate: tuple


def _check(image, plan):
    _number(plan.amplitude, "amplitude (native pixels)", 0, 2)
    _number(plan.period, "period", 1e-9, 1e9)
    if (type(plan.root_y) is not int or type(plan.tip_y) is not int or
            not 0 <= plan.root_y < plan.tip_y < image.height):
        raise ValueError("root/tip require ordered native rows")
    if plan.size != image.size or plan.backplate != (0, 0, 0, 0):
        raise ValueError("explicit transparent-black backplate and source dimensions required")
    for mask in (plan.hair, plan.protected, plan.envelope):
        _mask([list(p) for p in mask], image.size, allow_empty=mask == plan.protected)
    if not plan.hair <= plan.envelope or plan.hair & plan.protected:
        raise ValueError("hair must be inside envelope and disjoint from protection")
    if not any(y == plan.root_y for x, y in plan.hair) or not any(y == plan.tip_y for x, y in plan.hair):
        raise ValueError("hair must include root and tip")
    for y in range(image.height):
        for x in range(image.width):
            p = (x, y)
            alpha = image.getpixel(p)[3]
            if alpha not in (0, 255):
                raise ValueError("binary source alpha required")
            if alpha == 255 and p not in plan.hair and p not in plan.protected:
                raise ValueError("all non-hair opaque pixels must be protected")
    for x, y in plan.hair:
        if not plan.root_y <= y <= plan.tip_y or image.getpixel((x, y))[3] != 255:
            raise ValueError("hair must be opaque within root-to-tip rows")
        reach = math.floor(plan.amplitude * (y - plan.root_y) / (plan.tip_y - plan.root_y) + .5)
        for dx in range(-reach, reach + 1):
            dest = (x + dx, y)
            if not (0 <= dest[0] < image.width) or dest not in plan.envelope:
                raise ValueError("motion clips canvas or declared envelope")
            if dest in plan.protected:
                raise ValueError("motion collides with protected content")
    for p in plan.envelope - plan.hair:
        if image.getpixel(p) != plan.backplate:
            raise ValueError("envelope requires known transparent backplate outside hair")


def plan_hair(source: bytes, annotation: dict, amplitude: float, period: float) -> HairPlan:
    """Validate manual/machine source-bound masks; never infer an occluded layer.

    Backplate is an explicit annotation assertion about the isolated tip, not an
    estimate of unseen pixels. Row weight is (y-root_y)/(tip_y-root_y).
    """
    image = _image(source)
    if (not isinstance(annotation, dict) or set(annotation) != {
            "version", "provider", "source_sha256", "hair", "protected", "envelope",
            "root_y", "tip_y", "backplate"} or
            type(annotation["version"]) is not int or annotation["version"] != 1 or
            not isinstance(annotation["provider"], str) or not annotation["provider"].strip()):
        raise ValueError("invalid source-bound hair annotation")
    digest = hashlib.sha256(source).hexdigest()
    if annotation["source_sha256"] != digest:
        raise ValueError("source hash mismatch")
    backplate = annotation["backplate"]
    if (backplate != {"kind": "known-transparent", "rgba": [0, 0, 0, 0]} or
            any(type(v) is not int for v in backplate["rgba"])):
        raise ValueError("explicit known transparent backplate required; no occlusion guessing")
    plan = HairPlan(digest, image.size,
                    *(_mask(annotation[k], image.size, allow_empty=k == "protected")
                      for k in ("hair", "protected", "envelope")),
                    annotation["root_y"], annotation["tip_y"], amplitude, period, (0, 0, 0, 0))
    _check(image, plan)
    return plan


def render_frame(source: bytes, plan: HairPlan, time: float):
    """Grid-preserving horizontal translation; half pixels round away from zero.

    Alpha intentionally moves inside the envelope; all other RGBA is exact.
    Returns a new Pillow RGBA image, leaving source bytes untouched.
    """
    if not isinstance(plan, HairPlan) or hashlib.sha256(source).hexdigest() != plan.source_sha256:
        raise ValueError("source hash mismatch or invalid plan")
    image = _image(source)
    _check(image, plan)
    _number(time, "time", -1e9, 1e9)
    phase = (time % plan.period) / plan.period
    wave = 0.0 if phase == 0 else math.sin(2 * math.pi * phase)
    output = image.copy()
    for p in plan.hair:
        output.putpixel(p, plan.backplate)
    for x, y in plan.hair:
        displacement = plan.amplitude * (y - plan.root_y) / (plan.tip_y - plan.root_y) * wave
        dx = int(math.copysign(math.floor(abs(displacement) + .5), displacement))
        output.putpixel((x + dx, y), image.getpixel((x, y)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()


if __name__ == "__main__":
    main()
