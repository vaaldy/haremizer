"""Non-semantic temporary PNG fixtures: export conformance, not visual quality."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from images import pillow, validate_png
from sprite_export import export_sprite, INDEX_CHARS, MAX_INPUT_BYTES


class SpriteExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.base = self.inputs / "base.png"
        self.blink = self.inputs / "blink.png"
        self.output = self.root / "character"
        self.make_png(self.base)

    def make_png(self, path, size=64, colors=None):
        colors = colors if colors is not None else [(index, 20, 30) for index in range(62)]
        image = pillow().new("RGBA", (size, size), (222, 111, 99, 0))
        for index, rgb in enumerate(colors):
            image.putpixel((index % size, index // size), (*rgb, 255))
        image.save(path)

    def assert_raster(self, sprite, frame, path):
        with pillow().open(path) as image:
            original = list(image.convert("RGBA").getdata())
        palette = [tuple(bytes.fromhex(color[1:])) for color in sprite["palette"]]
        decoded = [(0, 0, 0, 0) if char == "." else (*palette[INDEX_CHARS.index(char)], 255)
                   for row in sprite["frames"][frame] for char in row]
        canonical = [pixel if pixel[3] else (0, 0, 0, 0) for pixel in original]
        self.assertEqual(decoded, canonical)
        width, height = sprite["size"]
        digest = hashlib.sha256(f"{width}x{height}:".encode() +
                                bytes(channel for pixel in decoded for channel in pixel)).hexdigest()
        self.assertEqual(digest, validate_png(path)["pixel_sha256"])

    def test_lossless_deterministic_hashes_48_and_64(self):
        for size in (48, 64):
            with self.subTest(size=size):
                self.make_png(self.base, size)
                raw = self.base.read_bytes()
                output = self.root / f"character-{size}"
                report = export_sprite(self.base, output)
                encoded = (output / "neutral.json").read_bytes()
                sprite = json.loads(encoded)
                self.assertEqual(set(sprite), {"name", "size", "palette", "frames"})
                self.assertEqual(sprite["size"], [size, size])
                self.assertEqual(set(sprite["frames"]), {"base"})
                self.assertEqual(len(sprite["palette"]), 62)
                self.assertEqual(sprite["palette"], sorted(sprite["palette"]))
                self.assertIn("Z", "".join(sprite["frames"]["base"]))
                self.assertIn(".", "".join(sprite["frames"]["base"]))
                self.assert_raster(sprite, "base", self.base)
                self.assertEqual(report["inputs"]["base"]["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(report["inputs"]["base"]["metrics"], validate_png(self.base))
                self.assertEqual(report["output"]["sha256"], hashlib.sha256(encoded).hexdigest())
                self.assertEqual(report["palette"], sprite["palette"])
                self.assertFalse(report["visual_quality_assessed"])
                self.assertFalse(report["automatic_acceptance"])
                self.assertEqual(self.base.read_bytes(), raw)
                self.assertEqual([p.name for p in output.iterdir()], ["neutral.json"])
                sidecar = output.parent / (output.name + ".provenance.json")
                self.assertEqual(json.loads(sidecar.read_bytes()), report)
                second = self.root / f"repeat-{size}"
                export_sprite(self.base, second)
                self.assertEqual((second / "neutral.json").read_bytes(), encoded)

    def test_shared_palette_union_62_and_reject_63(self):
        self.make_png(self.base, colors=[(i, 0, 0) for i in range(31)])
        self.make_png(self.blink, colors=[(i, 0, 0) for i in range(31, 62)])
        export_sprite(self.base, self.output, self.blink)
        sprite = json.loads((self.output / "neutral.json").read_bytes())
        self.assertEqual(len(sprite["palette"]), 62)
        self.assertEqual(set(sprite["frames"]), {"base", "blink"})
        for frame, path in (("base", self.base), ("blink", self.blink)):
            self.assert_raster(sprite, frame, path)
        self.make_png(self.blink, colors=[(i, 0, 0) for i in range(31, 63)])
        rejected = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "combined palette"):
            export_sprite(self.base, rejected, self.blink)
        self.assertFalse(rejected.exists())

    def test_geometry_alpha_and_png_contract(self):
        self.make_png(self.blink, size=48)
        with self.assertRaisesRegex(ValueError, "geometry"):
            export_sprite(self.base, self.output, self.blink)
        for kind in ("alpha", "opaque", "transparent", "size", "63", "animated", "not-png"):
            with self.subTest(kind=kind):
                image = pillow().new("RGBA", (64, 64), (1, 2, 3, 0))
                image.putpixel((1, 1), (1, 2, 3, 255))
                if kind == "alpha":
                    image.putpixel((2, 2), (1, 2, 3, 128))
                elif kind == "opaque":
                    image = pillow().new("RGBA", (64, 64), (1, 2, 3, 255))
                elif kind == "transparent":
                    image = pillow().new("RGBA", (64, 64))
                elif kind == "size":
                    image = image.resize((32, 32))
                elif kind == "63":
                    for i in range(63):
                        image.putpixel((i, 0), (i, 0, 0, 255))
                if kind == "animated":
                    other = image.copy()
                    other.putpixel((3, 3), (1, 2, 3, 255))
                    image.save(self.base, save_all=True, append_images=[other])
                elif kind == "not-png":
                    image.save(self.base, format="BMP")
                else:
                    image.save(self.base)
                with self.assertRaises(ValueError):
                    export_sprite(self.base, self.output)
                self.assertFalse(self.output.exists())

    def test_names_and_new_paths_no_overwrite(self):
        raw = self.base.read_bytes()
        for name in ("../escape", "Neutral", "a/b", "a\\b", "\x1b[31m", "", ".", "a" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                export_sprite(self.base, self.output, name=name)
        for output in (self.inputs, self.inputs / "nested", self.base, self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                export_sprite(self.base, output)
        for sidecar in (self.output / "manifest.json", self.output, self.base, self.inputs):
            with self.subTest(sidecar=sidecar), self.assertRaises(ValueError):
                export_sprite(self.base, self.output, provenance=sidecar)
        self.output.symlink_to(self.root / "missing", target_is_directory=True)
        with self.assertRaises(ValueError):
            export_sprite(self.base, self.output)
        self.output.unlink()
        default_sidecar = self.root / "character.provenance.json"
        default_sidecar.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            export_sprite(self.base, self.output)
        self.assertEqual(default_sidecar.read_bytes(), b"existing")
        default_sidecar.unlink()
        sidecar = self.root / "reports" / "export.json"
        export_sprite(self.base, self.output, name="neutral-1", provenance=sidecar)
        before = (self.output / "neutral-1.json").read_bytes()
        with self.assertRaises(ValueError):
            export_sprite(self.base, self.output, provenance=self.root / "other.json")
        self.assertEqual((self.output / "neutral-1.json").read_bytes(), before)
        self.assertEqual(self.base.read_bytes(), raw)
        self.assertTrue(sidecar.is_file())

    def test_bounded_snapshot_and_validation_uses_bytesio(self):
        with self.base.open("wb") as target:
            target.truncate(MAX_INPUT_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "16 MiB"):
            export_sprite(self.base, self.output)
        self.make_png(self.base)
        raw = self.base.read_bytes()
        original_validate = validate_png

        def validate_snapshot(stream):
            self.assertIsInstance(stream, BytesIO)
            self.assertEqual(stream.getvalue(), raw)
            metrics = original_validate(stream)
            # Source mutation after snapshot must not change encoding or byte hash.
            self.base.write_bytes(b"changed after snapshot")
            return metrics

        with patch("sprite_export.validate_png", side_effect=validate_snapshot):
            report = export_sprite(self.base, self.output)
        self.assertEqual(report["inputs"]["base"]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(json.loads((self.output / "neutral.json").read_bytes())["palette"]), 62)

    def test_cli_base_and_blink(self):
        self.make_png(self.blink, colors=[(61, 20, 30), (0, 20, 30)])
        for with_blink in (False, True):
            output = self.root / f"cli-{with_blink}"
            command = [sys.executable, str(Path(__file__).with_name("sprite_export.py")),
                       "--base", str(self.base), "--output", str(output)]
            if with_blink:
                command += ["--blink", str(self.blink), "--provenance", str(self.root / "cli.json")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["visual_quality_assessed"])
            sprite = json.loads((output / "neutral.json").read_bytes())
            self.assertEqual(set(sprite["frames"]), {"base", "blink"} if with_blink else {"base"})

    @unittest.skipUnless(os.environ.get("TERM_CHAN_LOADER_TEST") == "1" and shutil.which("node"),
                         "optional read-only parent Node integration")
    def test_parent_loader_and_renderer(self):
        parent = Path(__file__).resolve().parent.parent
        loader = parent / "src" / "sprites" / "loader.js"
        renderer = parent / "src" / "render" / "sprite.js"
        if not loader.is_file() or not renderer.is_file():
            self.skipTest("parent term-chan sources unavailable")
        self.make_png(self.blink, colors=[(61, 20, 30), (0, 20, 30)])
        export_sprite(self.base, self.output, self.blink)
        script = """
            const {loadCharacter} = await import(process.argv[1]);
            const {spriteToLines} = await import(process.argv[2]);
            const pack = loadCharacter(process.argv[3]);
            const sprite = pack.expressions.neutral;
            console.log(JSON.stringify({expressions: Object.keys(pack.expressions),
                sprite, lines: spriteToLines(sprite).length,
                talkFallback: sprite.frames.talk === sprite.frames.base}));
        """
        result = subprocess.run(["node", "--input-type=module", "-e", script,
                                 loader.as_uri(), renderer.as_uri(), str(self.output)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["expressions"], ["neutral"])
        self.assertEqual(parsed["lines"], 32)
        self.assertTrue(parsed["talkFallback"])
        sprite = parsed["sprite"]
        self.assertEqual((sprite["width"], sprite["height"]), (64, 64))
        expected = json.loads((self.output / "neutral.json").read_bytes())
        self.assertEqual(sprite["palette"], [list(bytes.fromhex(c[1:])) for c in expected["palette"]])
        for frame, path in (("base", self.base), ("blink", self.blink)):
            with pillow().open(path) as image:
                canonical = [p if p[3] else (0, 0, 0, 0) for p in image.convert("RGBA").getdata()]
            decoded = [(0, 0, 0, 0) if index == -1 else (*sprite["palette"][index], 255)
                       for row in sprite["frames"][frame] for index in row]
            self.assertEqual(decoded, canonical)


if __name__ == "__main__":
    unittest.main()
