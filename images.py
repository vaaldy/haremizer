"""Factual PNG checks and nearest-neighbor contact sheets, not visual evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for PNG processing (missing dependency: Pillow)") from error
    return Image


BANNER_SIZES = ((128, 48), (96, 32), (64, 32))


def validate_png(path: Path, sizes: tuple[int, ...] = (48, 64), allow_banners: bool = False) -> dict:
    """Validate one PNG against the sprite contract.

    Square portrait sprites keep the original 48/64 rule. Banners are non-square and are only accepted
    when `allow_banners` is passed, so nothing silently loosens the portrait contract.
    """
    Image = pillow()
    if not sizes or (any(size not in (48, 64) for size in sizes) and not allow_banners):
        raise ValueError("allowed PNG sizes must be 48 and/or 64")
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("expected a single-frame PNG")
            if image.width == image.height:
                if image.width not in sizes:
                    raise ValueError(f"expected square PNG of size {sizes}")
            elif not allow_banners:
                raise ValueError(f"expected square PNG of size {sizes}; banners need allow_banners")
            elif (image.width, image.height) not in BANNER_SIZES:
                raise ValueError(f"banner size must be one of {BANNER_SIZES}")
            image.verify()
        with Image.open(path) as image:
            image.load()
            rgba = image.convert("RGBA")
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError(f"unreadable PNG: {error}") from error
    width, height = rgba.width, rgba.height
    data = rgba.tobytes()
    pixels = [tuple(data[index:index + 4]) for index in range(0, len(data), 4)]
    if any(pixel[3] not in (0, 255) for pixel in pixels):
        raise ValueError("alpha must be binary (0 or 255)")
    opaque = [pixel[:3] for pixel in pixels if pixel[3] == 255]
    transparent = len(pixels) - len(opaque)
    colors = len(set(opaque))
    if not opaque or not transparent:
        # A full-bleed banner is opaque by design and has no transparent pixels. Square sprites still
        # need both, so an all-opaque portrait cannot masquerade as a sprite.
        if not (allow_banners and width != height):
            raise ValueError("both transparent and opaque pixels are required")
    if colors > 62:
        raise ValueError("palette exceeds 62 opaque colors")
    # Ignore hidden RGB under transparency when detecting re-encoded duplicates.
    canonical = bytes(channel for pixel in pixels
                      for channel in (pixel if pixel[3] else (0, 0, 0, 0)))
    pixel_digest = hashlib.sha256(f"{rgba.width}x{rgba.height}:".encode() + canonical).hexdigest()
    return {"width": rgba.width, "height": rgba.height, "opaque_colors": colors,
            "opaque_pixels": len(opaque), "transparent_pixels": transparent,
            "opaque_fraction": len(opaque) / len(pixels), "pixel_sha256": pixel_digest,
            "visual_quality_assessed": False}


def contact_sheet(paths: list[Path], output: Path, scale: int = 4, columns: int = 4) -> dict:
    if not paths:
        raise ValueError("contact sheet requires at least one PNG")
    if type(scale) is not int or not 1 <= scale <= 16 or type(columns) is not int or not 1 <= columns <= 16:
        raise ValueError("scale and columns must be integers from 1 to 16")
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("contact sheet must not overwrite an input")
    metrics = [validate_png(path) for path in paths]
    Image = pillow()
    cell = max(item["width"] for item in metrics) * scale
    columns = min(columns, len(paths))
    sheet = Image.new("RGBA", (columns * cell, ((len(paths) + columns - 1) // columns) * cell))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            resized = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
            sheet.paste(resized, ((index % columns) * cell, (index // columns) * cell))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return {"output": str(output), "width": sheet.width, "height": sheet.height,
            "scale": scale, "images": metrics, "visual_quality_assessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument("--size", type=int, choices=(48, 64), action="append")
    sheet = commands.add_parser("sheet")
    sheet.add_argument("paths", nargs="+", type=Path)
    sheet.add_argument("--output", type=Path, required=True)
    sheet.add_argument("--scale", type=int, default=4)
    sheet.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    try:
        result = ([{"path": str(path), **validate_png(path, tuple(args.size or (48, 64)))}
                   for path in args.paths] if args.command == "validate" else
                  contact_sheet(args.paths, args.output, args.scale, args.columns))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
