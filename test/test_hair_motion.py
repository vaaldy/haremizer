"""Translated synthetic isolated tips; no character coordinates or colors."""
import copy
from dataclasses import replace
import hashlib
from io import BytesIO
import subprocess
import sys
import unittest

from haremizer.hair_motion import plan_hair, render_frame
from haremizer.images import pillow


class HairMotionTests(unittest.TestCase):
    def fixture(self, x=7, root=4, width=2, height=5):
        image = pillow().new("RGBA", (24, 24), (0, 0, 0, 0))
        hair = [[px, y] for y in range(root, root + height) for px in range(x, x + width)]
        for px, y in hair:
            image.putpixel((px, y), (30 + y, 90, 130, 255))
        image.putpixel((20, 20), (240, 210, 200, 255))
        stream = BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
        ann = {"version": 1, "provider": "synthetic translated tip", "source_sha256": hashlib.sha256(raw).hexdigest(),
               "hair": hair, "protected": [[20, 20]],
               "envelope": [[px, y] for y in range(root, root + height) for px in range(x - 2, x + width + 2)],
               "root_y": root, "tip_y": root + height - 1,
               "backplate": {"kind": "known-transparent", "rgba": [0, 0, 0, 0]}}
        return image, raw, ann

    def test_translation_dimensions_signed_root_and_invariants(self):
        for x, root, width, height in ((7, 4, 2, 5), (10, 9, 3, 7), (3, 1, 1, 2)):
            image, raw, ann = self.fixture(x, root, width, height)
            before = bytes(raw)
            plan = plan_hair(raw, ann, 2, 8)
            for t, dx in ((2, 2), (6, -2), (-2, -2)):
                output = render_frame(raw, plan, t)
                self.assertEqual(output.size, image.size)
                self.assertEqual(output.getpixel((x + dx, root + height - 1)), image.getpixel((x, root + height - 1)))
                for px in range(x, x + width):
                    self.assertEqual(output.getpixel((px, root)), image.getpixel((px, root)))
                for y in range(image.height):
                    for px in range(image.width):
                        if (px, y) not in plan.envelope or (px, y) in plan.protected:
                            self.assertEqual(output.getpixel((px, y)), image.getpixel((px, y)))
                self.assertTrue(set(output.getdata()) <= set(image.getdata()))
                self.assertTrue(set(output.getchannel("A").getdata()) <= {0, 255})
            self.assertEqual(raw, before)

    def test_zero_endpoint_determinism(self):
        image, raw, ann = self.fixture()
        for amplitude in (0, .5, 1, 2):
            plan = plan_hair(raw, ann, amplitude, 8)
            for t in (0, 8, -8):
                self.assertEqual(render_frame(raw, plan, t).tobytes(), image.tobytes())
            self.assertEqual(render_frame(raw, plan, 2).tobytes(), render_frame(raw, plan, 10).tobytes())
            if amplitude == 0:
                self.assertEqual(render_frame(raw, plan, 3).tobytes(), image.tobytes())

    def test_malformed_controls(self):
        _, raw, ann = self.fixture()
        for value in (True, False, None, "1", float("nan"), float("inf"), -1, 3, 10**1000):
            with self.subTest(amplitude=value), self.assertRaises(ValueError):
                plan_hair(raw, ann, value, 8)
        for value in (True, 0, -1, None, float("nan"), float("inf")):
            with self.subTest(period=value), self.assertRaises(ValueError):
                plan_hair(raw, ann, 1, value)
        plan = plan_hair(raw, ann, 1, 8)
        for value in (True, None, "1", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                render_frame(raw, plan, value)
        for key, value in (("root_y", True), ("tip_y", 99), ("hair", []), ("hair", [[1.5, 2]]),
                           ("envelope", [[-1, 0]]), ("hair", [[7, 4], [7, 4]]), ("provider", "")):
            bad = copy.deepcopy(ann)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                plan_hair(raw, bad, 1, 8)

    def test_hash_backplate_and_nonbinary_alpha(self):
        image, raw, ann = self.fixture()
        bad = copy.deepcopy(ann)
        bad["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            plan_hair(raw, bad, 1, 8)
        plan = plan_hair(raw, ann, 1, 8)
        with self.assertRaises(ValueError):
            render_frame(raw + b"changed", plan, 2)
        for backplate in (None, {"kind": "unknown"}, {"kind": "known-transparent", "rgba": [1, 2, 3, 0]},
                          {"kind": "known-transparent", "rgba": [False, 0, 0, 0]}):
            bad = copy.deepcopy(ann)
            bad["backplate"] = backplate
            with self.assertRaises(ValueError):
                plan_hair(raw, bad, 1, 8)
        for rgba in ((0, 0, 0, 128), (1, 2, 3, 0)):
            image.putpixel((5, 4), rgba)
            stream = BytesIO()
            image.save(stream, format="PNG")
            bad = copy.deepcopy(ann)
            bad["source_sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
            with self.assertRaises(ValueError):
                plan_hair(stream.getvalue(), bad, 1, 8)

    def test_clipping_mask_escape_and_protection(self):
        _, raw, ann = self.fixture()
        for key, value in (("protected", [[20, 20], [7, 4]]), ("protected", [[5, 8]]),
                           ("envelope", ann["hair"]), ("root_y", 5)):
            bad = copy.deepcopy(ann)
            bad[key] = value
            with self.assertRaises(ValueError):
                plan_hair(raw, bad, 2, 8)
        image, _, ann = self.fixture(1)
        ann["envelope"] = [[x, y] for x, y in ann["envelope"] if x >= 0]
        stream = BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
        with self.assertRaisesRegex(ValueError, "clips"):
            plan_hair(raw, ann, 2, 8)
        _, raw, ann = self.fixture()
        plan = plan_hair(raw, ann, 1, 8)
        with self.assertRaises(ValueError):
            render_frame(raw, replace(plan, envelope=plan.hair), 2)

    def test_actual_opaque_collision_and_unprotected_content(self):
        image, _, ann = self.fixture()
        image.putpixel((5, 8), (200, 50, 70, 255))
        stream = BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
        ann["source_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "must be protected"):
            plan_hair(raw, ann, 2, 8)
        ann["protected"].append([5, 8])
        with self.assertRaisesRegex(ValueError, "collides"):
            plan_hair(raw, ann, 2, 8)
        with self.assertRaises((ValueError, OSError)):
            plan_hair(b"not a png", ann, 1, 8)

    def test_import_and_help_without_gpu_libraries(self):
        code = "import sys; from haremizer import hair_motion; assert not any(k in sys.modules for k in ('torch','diffusers'))"
        subprocess.run([sys.executable, "-B", "-c", code], check=True)
        subprocess.run([sys.executable, "-B", "-m", "haremizer.hair_motion", "--help"], check=True, stdout=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
