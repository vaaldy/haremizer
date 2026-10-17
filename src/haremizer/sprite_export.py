"""Lossless visual PNG-to-term-chan export; no generation or quality acceptance."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from .images import pillow, validate_png

INDEX_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_INPUT_BYTES = 16 * 1024 * 1024


def _snapshot(path: Path) -> tuple[bytes, dict, list[tuple]]:
    if not path.is_file():
        raise ValueError("input must be a local PNG file")
    with path.open("rb") as source:
        raw = source.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("input PNG exceeds 16 MiB")
    metrics = validate_png(BytesIO(raw), allow_banners=True)
    with pillow().open(BytesIO(raw)) as image:
        pixels = list(image.convert("RGBA").getdata())
    return raw, metrics, pixels


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def export_sprite(base, newcharacter_directory, blink=None, name="neutral", provenance=None) -> dict:
    """Export supplied frames with a shared sorted palette; return factual provenance."""
    if not isinstance(name, str) or len(name) > 64 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("name must be a lowercase filename slug of at most 64 characters")
    supplied = {"base": Path(base)}
    if blink is not None:
        supplied["blink"] = Path(blink)
    paths = {frame: path.resolve() for frame, path in supplied.items()}
    requested_output = Path(newcharacter_directory)
    output = requested_output.resolve()
    requested_provenance = (Path(provenance) if provenance is not None else
                            output.parent / (output.name + ".provenance.json"))
    sidecar = requested_provenance.resolve()
    if requested_output.exists() or requested_output.is_symlink() or output.exists():
        raise ValueError("character directory must be new")
    for frame, path in paths.items():
        # Check both the supplied and resolved image directory for symlink inputs.
        for directory in (path.parent, supplied[frame].absolute().parent.resolve()):
            if output == directory or output.is_relative_to(directory):
                raise ValueError("character directory must be outside input image directories")
        if path.is_relative_to(output):
            raise ValueError("character directory must not contain inputs")
    if sidecar == output or sidecar.is_relative_to(output) or output.is_relative_to(sidecar):
        raise ValueError("provenance must be outside the character directory")
    if (requested_provenance.exists() or requested_provenance.is_symlink() or
            sidecar.exists() or sidecar in paths.values()):
        raise ValueError("provenance must be a new path and must not overwrite inputs")

    snapshots = {frame: _snapshot(path) for frame, path in paths.items()}
    dimensions = [(metrics["width"], metrics["height"]) for _, metrics, _ in snapshots.values()]
    if len(set(dimensions)) != 1:
        raise ValueError("base and blink must have the same geometry")
    width, height = dimensions[0]
    colors = sorted({pixel[:3] for _, _, pixels in snapshots.values()
                     for pixel in pixels if pixel[3] == 255})
    if len(colors) > len(INDEX_CHARS):
        raise ValueError("combined palette exceeds 62 opaque colors")
    indices = {rgb: INDEX_CHARS[index] for index, rgb in enumerate(colors)}
    palette = ["#{:02X}{:02X}{:02X}".format(*rgb) for rgb in colors]
    frames = {}
    for frame, (_, _, pixels) in snapshots.items():
        encoded = "".join(indices[pixel[:3]] if pixel[3] else "." for pixel in pixels)
        frames[frame] = [encoded[start:start + width] for start in range(0, len(encoded), width)]
    sprite = {"name": name, "size": [width, height], "palette": palette, "frames": frames}
    encoded_json = _json_bytes(sprite)
    expression = output / (name + ".json")
    report = {
        "version": 1, "tool": "sprite_export", "character_directory": str(output),
        "inputs": {frame: {"path": str(paths[frame]), "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest(), "metrics": metrics}
                   for frame, (raw, metrics, _) in snapshots.items()},
        "output": {"path": str(expression), "bytes": len(encoded_json),
                   "sha256": hashlib.sha256(encoded_json).hexdigest()},
        "size": [width, height], "palette": palette,
        "preservation_scope": "opaque RGB and binary alpha exact; hidden transparent RGB not serialized",
        "blink_semantics": "exports given frames only; no closed-eye verification or localization assertion",
        "visual_quality_assessed": False, "automatic_acceptance": False,
    }
    # Exclusive creation prevents overwrites; an I/O failure may leave partial output.
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    with expression.open("xb") as target:
        target.write(encoded_json)
    with sidecar.open("xb") as target:
        target.write(_json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new character directory")
    parser.add_argument("--blink", type=Path)
    parser.add_argument("--name", default="neutral")
    parser.add_argument("--provenance", type=Path, help="new JSON path outside character directory")
    args = parser.parse_args()
    try:
        report = export_sprite(args.base, args.output, args.blink, args.name, args.provenance)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
