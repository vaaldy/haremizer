"""Non-semantic synthetic fixtures test local pixel preservation, not closed eyes."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from blink import blink
from images import pillow, validate_png


class BlinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.source = self.inputs / "fixture.png"
        self.annotation = self.root / "annotation.json"
        self.output = self.root / "output"
        self.make_source(64)

    def make_source(self, size):
        image = pillow().new("RGBA", (size, size), (7, 8, 9, 0))
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                image.putpixel((x, y), (20, 40, 60, 255))
        image.putpixel((4, 4), (80, 100, 120, 255))
        image.save(self.source)
        self.declaration = {
            "version": 1, "reviewed_by": "machine:sol",
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "eye_regions": [{"x": 2, "y": 2, "width": 3, "height": 2}],
            "edits": [{"x": 2, "y": 2, "color": "#506478"},
                      {"x": 3, "y": 2, "color": "#506478"}],
        }

    def write_annotation(self, declaration=None):
        self.annotation.write_text(json.dumps(declaration or self.declaration) + "\n")

    def assert_rejected(self, declaration):
        self.write_annotation(declaration)
        with self.assertRaises(ValueError):
            blink(self.source, self.annotation, self.output)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_preservation_hashes_and_canonical_metrics_48_and_64(self):
        for size in (48, 64):
            with self.subTest(size=size):
                self.make_source(size)
                self.write_annotation()
                output = self.root / f"output-{size}"
                raw = self.source.read_bytes()
                annotation_bytes = self.annotation.read_bytes()
                report = blink(self.source, self.annotation, output)
                self.assertEqual((output / "source.png").read_bytes(), raw)
                self.assertEqual((output / "annotation.json").read_bytes(), annotation_bytes)
                self.assertEqual(report["source"]["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(report["annotation"]["sha256"],
                                 hashlib.sha256(annotation_bytes).hexdigest())
                self.assertEqual(report["output"]["sha256"],
                                 hashlib.sha256((output / "blink.png").read_bytes()).hexdigest())
                self.assertEqual(report["output"]["metrics"], validate_png(output / "blink.png"))
                self.assertEqual(report["source"]["metrics"], validate_png(self.source))
                self.assertEqual(report["output"]["metrics"]["width"], size)
                self.assertEqual(report["output"]["metrics"]["opaque_colors"], 2)
                self.assertEqual(report["changed_pixels"], 2)
                self.assertEqual(report["specified_edits"], 2)
                self.assertEqual(report["allowed_pixels"], 6)
                self.assertEqual(report["eye_regions"], self.declaration["eye_regions"])
                self.assertTrue(report["outside_eye_unchanged"])
                self.assertTrue(report["alpha_unchanged"])
                self.assertFalse(report["visual_quality_assessed"])
                self.assertEqual(json.loads((output / "manifest.json").read_bytes()), report)
                with pillow().open(self.source) as image:
                    before = list(image.convert("RGBA").getdata())
                with pillow().open(output / "blink.png") as image:
                    after = list(image.convert("RGBA").getdata())
                self.assertEqual([p[3] for p in before], [p[3] for p in after])
                self.assertEqual({i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]},
                                 {2 * size + 2, 2 * size + 3})
                for index, pixel in enumerate(before):
                    if not (2 <= index % size < 5 and 2 <= index // size < 4):
                        self.assertEqual(after[index], pixel)

    def test_union_not_sum_and_128_pixel_boundary(self):
        region = {"x": 1, "y": 1, "width": 16, "height": 8}
        self.declaration["eye_regions"] = [region, region.copy()]
        self.write_annotation()
        self.assertEqual(blink(self.source, self.annotation, self.output)["allowed_pixels"], 128)

    def test_invalid_regions(self):
        cases = [[], [{}] * 3, [{"x": 2, "y": 2, "width": 16, "height": 9}]]
        for field, value in (("x", -1), ("x", 64), ("x", 63), ("y", 63),
                             ("x", True), ("width", False), ("height", 2.0),
                             ("width", 17), ("height", 11), ("width", 0)):
            region = self.declaration["eye_regions"][0].copy()
            region[field] = value
            cases.append([region])
        cases.append([{"x": 1, "y": 1, "width": 16, "height": 8},
                      {"x": 1, "y": 20, "width": 1, "height": 1}])
        for regions in cases:
            with self.subTest(regions=regions):
                declaration = copy.deepcopy(self.declaration)
                declaration["eye_regions"] = regions
                self.assert_rejected(declaration)

    def test_invalid_edits(self):
        cases = [[], self.declaration["edits"] * 2,
                 [{"x": 1, "y": 2, "color": "#506478"}],
                 [{"x": True, "y": 2, "color": "#506478"}],
                 [{"x": 2, "y": 2.0, "color": "#506478"}],
                 [{"x": 2, "y": 2, "color": "#FFFFFF"}],
                 [{"x": 2, "y": 2, "color": [80, 100, 120]}],
                 [{"x": 2, "y": 2, "color": "#14283C"}]]
        for edits in cases:
            with self.subTest(edits=edits):
                declaration = copy.deepcopy(self.declaration)
                declaration["edits"] = edits
                self.assert_rejected(declaration)
        declaration = copy.deepcopy(self.declaration)
        declaration["eye_regions"] = [{"x": 0, "y": 0, "width": 1, "height": 1}]
        declaration["edits"] = [{"x": 0, "y": 0, "color": "#506478"}]
        self.assert_rejected(declaration)

    def test_invalid_hash_version_and_reviewer(self):
        for key, value in (("source_sha256", "0" * 64), ("source_sha256", "abc"),
                           ("source_sha256", self.declaration["source_sha256"].upper()),
                           ("version", True), ("version", 2), ("reviewed_by", " ")):
            with self.subTest(key=key, value=value):
                declaration = copy.deepcopy(self.declaration)
                declaration[key] = value
                self.assert_rejected(declaration)

    def test_invalid_source_contract(self):
        for kind in ("size", "opaque", "partial_alpha", "palette"):
            with self.subTest(kind=kind):
                self.make_source(64)
                with pillow().open(self.source) as source:
                    image = source.convert("RGBA")
                if kind == "size":
                    image = pillow().new("RGBA", (32, 32))
                elif kind == "opaque":
                    image = pillow().new("RGBA", (64, 64), (20, 40, 60, 255))
                elif kind == "partial_alpha":
                    image.putpixel((10, 10), (20, 40, 60, 128))
                else:
                    for i in range(63):
                        image.putpixel((i, 10), (i, 0, 0, 255))
                image.save(self.source)
                self.declaration["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
                self.assert_rejected(self.declaration)

    def test_existing_or_input_outputs_do_not_modify_inputs(self):
        self.write_annotation()
        source_bytes, annotation_bytes = self.source.read_bytes(), self.annotation.read_bytes()
        self.output.mkdir()
        for output in (self.output, self.inputs / "new", self.source, self.annotation, self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                blink(self.source, self.annotation, output)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.annotation.read_bytes(), annotation_bytes)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli(self):
        self.write_annotation()
        command = [sys.executable, str(Path(__file__).with_name("blink.py"))]
        help_result = subprocess.run(command + ["--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        result = subprocess.run(command + ["--source", str(self.source), "--annotation",
                                          str(self.annotation), "--output", str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["changed_pixels"], 2)
        failed = subprocess.run(command + ["--source", str(self.source), "--annotation",
                                          str(self.annotation), "--output", str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertEqual(failed.stdout, "")


if __name__ == "__main__":
    unittest.main()
