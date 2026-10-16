"""Synthetic geometry/preservation checks, independent of any character."""
import copy
from dataclasses import replace
import hashlib
from io import BytesIO
import subprocess
import sys
import unittest

from edit_plan import Abstention, plan_blink, lock_candidate, validate_candidate
from images import pillow


class EditPlanTests(unittest.TestCase):
    def setUp(self):
        self.image = pillow().new("RGBA", (32, 32), (120, 130, 140, 255))
        self.image.putpixel((0, 0), (1, 2, 3, 0))
        self.image.putpixel((1, 1), (10, 20, 30, 255))
        stream = BytesIO()
        self.image.save(stream, format="PNG")
        self.raw = stream.getvalue()
        self.annotation = {"version": 1, "provider": "synthetic test",
                           "source_sha256": hashlib.sha256(self.raw).hexdigest(),
                           "eyes": [self.eye(3, 4, 4, 8), self.eye(20, 12, 5, 4)]}

    def eye(self, x, y, w, h):
        return {"status": "confirmed", "confidence": 1,
                "rect": {"x": x, "y": y, "width": w, "height": h},
                "safe_regions": [{"x": x, "y": y, "width": w, "height": h + 3}]}

    def plan(self, fraction=.25):
        return plan_blink(self.raw, self.annotation, fraction)

    def test_zero_positive_asymmetric_and_translation(self):
        zero = self.plan(0)
        self.assertEqual(zero.eyes, zero.targets)
        self.assertEqual(len(zero.mask), 52)
        plan = self.plan()
        self.assertEqual(plan.targets, ((3, 6, 4, 8), (20, 13, 5, 4)))
        self.assertEqual(len(plan.mask), 65)
        self.assertEqual(self.plan(.125).targets[0][1], 5)
        self.annotation["eyes"] = [self.eye(8, 2, 4, 8)]
        self.assertEqual(self.plan().targets, ((8, 4, 4, 8),))

    def test_missing_occluded_uncertain(self):
        for status, confidence in (("occluded", 1), ("uncertain", 1), ("confirmed", .99)):
            with self.subTest(status=status, confidence=confidence):
                self.annotation["eyes"][0].update(status=status, confidence=confidence)
                with self.assertRaises(Abstention):
                    self.plan()
        self.annotation["eyes"] = []
        with self.assertRaises(Abstention):
            self.plan()

    def test_malformed_fractions_and_geometry(self):
        for value in (True, False, "0.25", None, float("nan"), float("inf"), -1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(value)
        for value in (True, 2.5, "3", None, float("inf"), -1, 33):
            annotation = copy.deepcopy(self.annotation)
            annotation["eyes"][0]["rect"]["x"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan_blink(self.raw, annotation, .25)
        for value in (True, float("nan"), float("inf"), -1, "1"):
            annotation = copy.deepcopy(self.annotation)
            annotation["eyes"][0]["confidence"] = value
            with self.assertRaises(ValueError):
                plan_blink(self.raw, annotation, .25)
        self.annotation["version"] = True
        with self.assertRaises(ValueError):
            self.plan()

    def test_safe_domain_edges_and_overlap(self):
        self.annotation["eyes"] = [self.eye(28, 21, 4, 8)]
        self.assertEqual(self.plan().targets, ((28, 23, 4, 8),))
        self.annotation["eyes"][0]["safe_regions"][0]["height"] = 8
        with self.assertRaises(Abstention):
            self.plan()
        self.annotation["eyes"] = [self.eye(2, 24, 4, 8)]
        self.annotation["eyes"][0]["safe_regions"][0]["height"] = 8
        with self.assertRaises(Abstention):
            self.plan()
        self.assertEqual(self.plan(0).targets, ((2, 24, 4, 8),))
        self.annotation["eyes"] = [self.eye(3, 4, 4, 8), self.eye(3, 13, 4, 8)]
        with self.assertRaises(ValueError):
            self.plan()

    def test_hash_mask_dimensions_and_immutability(self):
        plan = self.plan()
        before = copy.deepcopy(self.annotation)
        proposal = self.image.convert("RGB")
        pixels = list(proposal.getdata())
        lock_candidate(self.raw, plan, proposal)
        self.assertEqual(list(proposal.getdata()), pixels)
        self.assertEqual(self.annotation, before)
        with self.assertRaises(ValueError):
            lock_candidate(self.raw + b"x", plan, proposal)
        self.annotation["source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.plan()
        with self.assertRaises(ValueError):
            lock_candidate(self.raw, replace(plan, mask=plan.mask | {(0, 0)}), proposal)
        with self.assertRaises(ValueError):
            lock_candidate(self.raw, plan, proposal.resize((16, 16)))
        with self.assertRaises(ValueError):
            validate_candidate(self.raw, plan, proposal)

    def test_noop_not_placement_palette_alpha_outside(self):
        plan = self.plan()
        metrics = validate_candidate(self.raw, plan, self.image)
        self.assertTrue(metrics["no_op"])
        self.assertFalse(metrics["placement_verified"])
        changed = self.image.copy()
        changed.putpixel((3, 4), (10, 20, 30, 255))
        metrics = validate_candidate(self.raw, plan, changed)
        self.assertEqual(metrics["changed_pixels"], 1)
        self.assertFalse(metrics["placement_verified"])
        for coordinate, color in (((3, 4), (1, 2, 3, 255)),
                                  ((3, 4), (10, 20, 30, 0)),
                                  ((30, 30), (10, 20, 30, 255)),
                                  ((0, 0), (2, 2, 3, 0))):
            candidate = self.image.copy()
            candidate.putpixel(coordinate, color)
            with self.assertRaises(ValueError):
                validate_candidate(self.raw, plan, candidate)
        proposal = pillow().new("RGB", self.image.size, (11, 21, 31))
        locked = lock_candidate(self.raw, plan, proposal)
        self.assertEqual(locked.getpixel((3, 4)), (10, 20, 30, 255))
        self.assertEqual(locked.getpixel((0, 0)), self.image.getpixel((0, 0)))

    def test_palette_overflow(self):
        for x in range(1, 31):
            for y in (1, 2, 3):
                self.image.putpixel((x, y), (x, y, 0, 255))
        stream = BytesIO()
        self.image.save(stream, format="PNG")
        self.raw = stream.getvalue()
        self.annotation["source_sha256"] = hashlib.sha256(self.raw).hexdigest()
        with self.assertRaises(ValueError):
            validate_candidate(self.raw, self.plan(), self.image)

    def test_lazy_import_and_help(self):
        code = "import sys, edit_plan; assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", code], check=True)
        subprocess.run([sys.executable, "-B", "edit_plan.py", "--help"], check=True,
                       stdout=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
