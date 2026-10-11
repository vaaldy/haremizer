"""Explicit local CPU PNG postprocessing; not native-layout diffusion or quality acceptance."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
from io import BytesIO
import json
from pathlib import Path

from images import pillow, validate_png


MAX_SOURCE_BYTES = 16 * 1024 * 1024


def recover(source, output, background, size=64, tolerance=16, colors=32):
    if (not isinstance(background, (tuple, list)) or len(background) != 3 or
            any(type(channel) is not int or not 0 <= channel <= 255 for channel in background)):
        raise ValueError("explicit background RGB must contain three integers from 0..255")
    if type(size) is not int or size not in (48, 64):
        raise ValueError("size must be 48 or 64")
    if type(tolerance) is not int or not 0 <= tolerance <= 64:
        raise ValueError("tolerance must be an integer from 0..64")
    if type(colors) is not int or not 2 <= colors <= 62:
        raise ValueError("colors must be an integer from 2..62")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source.parent) or source.is_relative_to(output):
        raise ValueError("new output directory outside the input directory required")
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError("source must be a nonempty local PNG <=16 MiB")
    with source.open("rb") as file:
        raw = file.read(MAX_SOURCE_BYTES + 1)
    if not 0 < len(raw) <= MAX_SOURCE_BYTES:
        raise ValueError("source must be a nonempty local PNG <=16 MiB")
    Image = pillow()
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("source must be a single-frame PNG")
            if image.width != image.height or not 64 <= image.width <= 1024:
                raise ValueError("source must be square, 64..1024 pixels")
            original_size = image.width
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image.load()
            rgba = image.convert("RGBA")
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError(f"unreadable PNG: {error}") from error

    pixels = list(rgba.getdata())
    eligible = bytearray(pixel[3] == 0 or sum((pixel[c] - background[c]) ** 2
                                             for c in range(3)) <= tolerance ** 2
                         for pixel in pixels)
    removed = bytearray(len(pixels))
    queue = deque()

    def visit(index):
        if eligible[index] and not removed[index]:
            removed[index] = 1
            queue.append(index)

    for coordinate in range(original_size):
        visit(coordinate)
        visit((original_size - 1) * original_size + coordinate)
        visit(coordinate * original_size)
        visit(coordinate * original_size + original_size - 1)
    while queue:
        index = queue.popleft()
        x, y = index % original_size, index // original_size
        if x:
            visit(index - 1)
        if x + 1 < original_size:
            visit(index + 1)
        if y:
            visit(index - original_size)
        if y + 1 < original_size:
            visit(index + original_size)
    rgba.putdata([(0, 0, 0, 0) if removed[index] or pixel[3] == 0 else pixel
                  for index, pixel in enumerate(pixels)])
    resized = rgba.resize((size, size), Image.Resampling.BOX)
    pixels = list(resized.getdata())
    indices = [index for index, pixel in enumerate(pixels) if pixel[3] >= 128]
    final = Image.new("RGBA", (size, size))
    if indices:
        # Quantize opaque pixels only so hidden RGB cannot consume palette entries.
        strip = Image.new("RGB", (len(indices), 1))
        strip.putdata([pixels[index][:3] for index in indices])
        quantized = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                  dither=Image.Dither.NONE).convert("RGB")
        canonical = [(0, 0, 0, 0)] * len(pixels)
        for index, rgb in zip(indices, quantized.getdata()):
            canonical[index] = (*rgb, 255)
        final.putdata(canonical)

    output.mkdir(parents=True, exist_ok=False)
    (output / "source.png").write_bytes(raw)
    destination = output / "recovered.png"
    final.save(destination, format="PNG")
    metrics = validate_png(destination, (size,))
    report = {
        "source": {"path": str(source), "copy": "source.png", "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest(), "width": original_size,
                   "height": original_size},
        "settings": {"background_rgb": list(background), "background_selection": "explicit",
                     "tolerance": tolerance, "distance": "RGB Euclidean",
                     "removal": "four-connected edge flood fill; existing alpha zero preserved",
                     "resize": "BOX direct from original composition", "size": size,
                     "alpha_threshold": 128, "colors": colors,
                     "palette_algorithm": "Pillow MEDIANCUT on opaque RGB only",
                     "dither": "NONE", "transparent_rgb": [0, 0, 0]},
        "output": {"file": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                   "metrics": metrics},
        "output_contract": "CPU postprocessed PNG; not native-layout diffusion or term-chan export",
        "visual_quality_assessed": False, "training_started": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", type=int, nargs=3, required=True, metavar=("R", "G", "B"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--tolerance", type=int, default=16)
    parser.add_argument("--colors", type=int, default=32)
    try:
        report = recover(**vars(parser.parse_args()))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
