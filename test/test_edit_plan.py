"""Synthetic geometry/preservation checks, independent of any character."""
import copy
from dataclasses import replace
import hashlib
from io import BytesIO
import subprocess
import sys
import unittest

from edit_plan import Abstention, plan_blink, plan_expression, lock_candidate, validate_candidate
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

    def expression_annotation(self):
        return {"version": 1, "provider": "synthetic independent of Mint",
                "source_sha256": hashlib.sha256(self.raw).hexdigest(), "size": [32, 32],
                "regions": [self.region("left", "eyes", 3, 4),
                            self.region("right", "eyes", 20, 12),
                            self.region("smile", "mouth", 12, 24)]}

    def region(self, label, role, x, y):
        return {"label": label, "role": role, "status": "confirmed", "confidence": 1,
                "rect": {"x": x, "y": y, "width": 3, "height": 2}}

    def test_expression_operation_permissions_and_union(self):
        annotation = self.expression_annotation()
        for operation in ("happy", "surprised"):
            plan = plan_expression(self.raw, annotation, operation)
            self.assertEqual(len(plan.mask), 18)
            self.assertEqual(plan.mask, frozenset((x, y) for r in annotation["regions"]
                for x in range(r["rect"]["x"], r["rect"]["x"] + 3)
                for y in range(r["rect"]["y"], r["rect"]["y"] + 2)))
            annotation["regions"].append(self.region("brow", "brows", 3, 2))
            self.assertEqual(len(plan_expression(self.raw, annotation, operation).mask), 24)
            annotation["regions"].pop()
        for role in ("mouth", "brows"):
            annotation["regions"] = [self.region("eye", "eyes", 3, 4),
                                      self.region("other", role, 10, 10)]
            with self.assertRaises(ValueError):
                plan_expression(self.raw, annotation, "blink")
        annotation["regions"] = annotation["regions"][:1]
        self.assertEqual(len(plan_expression(self.raw, annotation, "blink").mask), 6)
        for operation in ("happy", "surprised"):
            with self.assertRaises(Abstention):
                plan_expression(self.raw, annotation, operation)
        for operation in ("smug", None, True, []):
            with self.assertRaises(ValueError):
                plan_expression(self.raw, annotation, operation)

    def test_expression_annotation_refusals(self):
        base = self.expression_annotation()
        changes = [("source_sha256", "0" * 64), ("size", [32, 31]),
                   ("size", [True, 32]), ("provider", ""), ("version", True)]
        for key, value in changes:
            annotation = copy.deepcopy(base)
            annotation[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                plan_expression(self.raw, annotation, "happy")
        for key, value in (("label", ""), ("label", "right"), ("role", "face"),
                           ("role", {}), ("confidence", True), ("confidence", float("nan")),
                           ("status", "occluded"), ("status", "missing"),
                           ("status", "uncertain"), ("confidence", .5)):
            annotation = copy.deepcopy(base)
            annotation["regions"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                plan_expression(self.raw, annotation, "happy")
        for rect in ({"x": 31, "y": 4, "width": 3, "height": 2},
                     {"x": True, "y": 4, "width": 3, "height": 2},
                     {"x": 0, "y": 0, "width": 1, "height": 1},
                     base["regions"][1]["rect"]):
            annotation = copy.deepcopy(base)
            annotation["regions"][0]["rect"] = rect
            with self.assertRaises(ValueError):
                plan_expression(self.raw, annotation, "happy")
        for regions in ([], base["regions"][-1:]):
            annotation = {**base, "regions": regions}
            with self.assertRaises(Abstention):
                plan_expression(self.raw, annotation, "happy")
        annotation = copy.deepcopy(base)
        del annotation["regions"][0]["role"]
        with self.assertRaises(ValueError):
            plan_expression(self.raw, annotation, "happy")

    def test_expression_preservation_noop_and_immutable_input(self):
        annotation = self.expression_annotation()
        before = copy.deepcopy(annotation)
        plan = plan_expression(self.raw, annotation, "surprised")
        self.assertEqual(annotation, before)
        metrics = validate_candidate(self.raw, plan, self.image)
        self.assertTrue(metrics["no_op"])
        self.assertFalse(metrics["expression_verified"])
        proposal = pillow().new("RGB", self.image.size, (11, 21, 31))
        pixels = list(proposal.getdata())
        locked = lock_candidate(self.raw, plan, proposal)
        self.assertEqual(list(proposal.getdata()), pixels)
        self.assertEqual(list(locked.getdata()), list(lock_candidate(self.raw, plan, proposal).getdata()))
        self.assertFalse(validate_candidate(self.raw, plan, locked)["expression_verified"])
        for coordinate, color in (((0, 0), (2, 2, 3, 0)),
                                  ((3, 4), (10, 20, 30, 0)),
                                  ((3, 4), (11, 21, 31, 255))):
            corrupt = locked.copy()
            corrupt.putpixel(coordinate, color)
            with self.assertRaises(ValueError):
                validate_candidate(self.raw, plan, corrupt)
        for forged in (replace(plan, mask=plan.mask | {(31, 31)}),
                       replace(plan, mask=frozenset()), replace(plan, operation="blink"),
                       replace(plan, size=(31, 32))):
            with self.assertRaises(ValueError):
                validate_candidate(self.raw, forged, locked)
        with self.assertRaises(ValueError):
            validate_candidate(self.raw + b"x", plan, locked)
        with self.assertRaises(ValueError):
            validate_candidate(self.raw, plan, locked.resize((16, 16)))
        with self.assertRaises(ValueError):
            validate_candidate(self.raw, plan, locked.convert("RGB"))

    def test_lazy_import_and_help(self):
        code = "import sys, edit_plan; assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", code], check=True)
        subprocess.run([sys.executable, "-B", "edit_plan.py", "--help"], check=True,
                       stdout=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
