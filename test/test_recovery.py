import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.source = self.inputs / "raw.png"
        image = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        for y in range(12, 52):
            for x in range(12, 52):
                image.putpixel((x, y), (x * 3, y * 3, (x + y) * 2, 255))
        for y in range(25, 35):
            for x in range(25, 35):
                image.putpixel((x, y), (255, 255, 255, 255))
        image.putpixel((20, 20), (255, 255, 255, 0))
        image.save(self.source)

    def recover(self, name="output", **kwargs):
        return recovery.recover(self.source, self.root / name, (255, 255, 255), **kwargs)

    def test_edge_only_binary_palette_provenance_and_determinism(self):
        original = self.source.read_bytes()
        first = self.recover(colors=8)
        second = self.recover("second", colors=8)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual((self.root / "output/source.png").read_bytes(), original)
        self.assertEqual(first["source"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertFalse(first["visual_quality_assessed"])
        self.assertFalse(first["output"]["metrics"]["visual_quality_assessed"])
        self.assertEqual(first["settings"]["background_rgb"], [255, 255, 255])
        self.assertEqual(first["settings"]["alpha_threshold"], 128)
        self.assertEqual(first["settings"]["dither"], "NONE")
        self.assertEqual(first["output"]["metrics"]["pixel_sha256"],
                         second["output"]["metrics"]["pixel_sha256"])
        path = self.root / "output/recovered.png"
        self.assertEqual(first["output"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.root / "output/manifest.json").read_text()), first)
        with Image.open(path) as image, Image.open(self.root / "second/recovered.png") as again:
            self.assertEqual(image.convert("RGBA").tobytes(), again.convert("RGBA").tobytes())
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0, 0))
            self.assertEqual(image.getpixel((30, 30))[3], 255)
            self.assertEqual(image.getpixel((20, 20)), (0, 0, 0, 0))
            pixels = list(image.getdata())
        self.assertEqual({pixel[3] for pixel in pixels}, {0, 255})
        self.assertLessEqual(len({pixel[:3] for pixel in pixels if pixel[3]}), 8)
        self.assertTrue(all(pixel == (0, 0, 0, 0) for pixel in pixels if not pixel[3]))

    def test_direct_box_resize_to_both_sizes(self):
        with Image.open(self.source) as source:
            source.resize((128, 128), Image.Resampling.NEAREST).save(self.source)
        for size in (48, 64):
            report = self.recover(str(size), size=size)
            self.assertEqual(report["source"]["width"], 128)
            self.assertEqual(report["output"]["metrics"]["width"], size)
            self.assertEqual(report["settings"]["resize"], "BOX direct from original composition")
            with Image.open(self.root / str(size) / "recovered.png") as image:
                self.assertEqual(image.getpixel((size // 2, size // 2))[3], 255)

    def test_euclidean_tolerance_and_alpha_threshold(self):
        image = Image.new("RGBA", (64, 64), (10, 10, 10, 255))
        image.putpixel((0, 1), (13, 14, 10, 255))  # Distance exactly five.
        image.putpixel((0, 2), (14, 14, 10, 255))  # Distance greater than five.
        image.putpixel((25, 25), (90, 0, 0, 127))
        image.putpixel((26, 25), (90, 0, 0, 128))
        image.save(self.source)
        recovery.recover(self.source, self.root / "output", (10, 10, 10), tolerance=5)
        with Image.open(self.root / "output/recovered.png") as final:
            self.assertEqual(final.getpixel((0, 1))[3], 0)
            self.assertEqual(final.getpixel((0, 2))[3], 255)
            self.assertEqual(final.getpixel((25, 25)), (0, 0, 0, 0))
            self.assertEqual(final.getpixel((26, 25))[3], 255)

    def test_invalid_options_and_output_rejected_before_processing(self):
        for kwargs in ({"size": 32}, {"size": True}, {"tolerance": -1}, {"tolerance": 65},
                       {"tolerance": 1.5}, {"colors": 1}, {"colors": 63}, {"colors": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.recover(**kwargs)
        for background in (None, (1, 2), (1, 2, 256), (True, 0, 0), (1.5, 2, 3)):
            with self.subTest(background=background), self.assertRaises(ValueError):
                recovery.recover(self.source, self.root / "output", background)
        with self.assertRaises(TypeError):
            recovery.recover(self.source, self.root / "output")
        empty = self.root / "empty"
        empty.mkdir()
        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "keep").write_text("unchanged")
        alias = self.root / "alias"
        alias.symlink_to(self.inputs, target_is_directory=True)
        for output in (empty, occupied, self.source, self.inputs / "new", alias / "new", self.root):
            with self.subTest(output=output), patch("recovery.pillow") as pillow:
                with self.assertRaises(ValueError):
                    recovery.recover(self.source, output, (255, 255, 255))
                pillow.assert_not_called()
        self.assertEqual((occupied / "keep").read_text(), "unchanged")
        self.assertFalse((self.inputs / "new").exists())

    def test_invalid_sources_dimensions_frames_and_file_bound(self):
        for dimensions in ((48, 48), (64, 65), (1025, 1025)):
            Image.new("RGB", dimensions).save(self.source)
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, "square"):
                self.recover()
        Image.new("RGB", (64, 64)).save(self.source, format="JPEG")
        with self.assertRaisesRegex(ValueError, "single-frame"):
            self.recover()
        frames = [Image.new("RGBA", (64, 64), color) for color in ("red", "blue")]
        frames[0].save(self.source, save_all=True, append_images=frames[1:])
        with self.assertRaisesRegex(ValueError, "single-frame"):
            self.recover()
        self.source.write_bytes(b"not PNG")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.recover()
        Image.new("RGB", (64, 64)).save(self.source)
        data = bytearray(self.source.read_bytes())
        data[-13] ^= 1
        self.source.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.recover()
        with self.source.open("wb") as file:
            file.truncate(recovery.MAX_SOURCE_BYTES + 1)
        with patch("recovery.pillow") as pillow, self.assertRaisesRegex(ValueError, "16 MiB"):
            self.recover()
        pillow.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_all_transparent_and_all_opaque_rejected_by_factual_validator(self):
        for color in ((255, 255, 255, 255), (20, 30, 40, 255), (20, 30, 40, 0)):
            Image.new("RGBA", (64, 64), color).save(self.source)
            with self.subTest(color=color), self.assertRaisesRegex(ValueError, "both transparent and opaque"):
                self.recover(str(color))

    def test_cli_explicit_background_and_errors(self):
        command = [sys.executable, "-B", str(Path(recovery.__file__)), "--source", str(self.source),
                   "--output", str(self.root / "cli")]
        missing = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--background", missing.stderr)
        command += ["--background", "255", "255", "255", "--size", "48",
                    "--tolerance", "0", "--colors", "2"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["output"]["metrics"]["width"], 48)
        self.assertLessEqual(report["output"]["metrics"]["opaque_colors"], 2)
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("new output directory", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
