import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import images
import prepare
import runtime

try:
    from PIL import Image
except ImportError:
    Image = None


SUITE = Path(__file__).with_name("evaluation-suite.json")


class RuntimeTests(unittest.TestCase):
    def test_visibility_is_not_tensor_verification_and_environment_is_unchanged(self):
        before = dict(os.environ)
        with patch("runtime.shutil.which", return_value="/visible-tool"), patch(
                "runtime.bounded_command", return_value={"stdout": "GPU visible", "returncode": 0}) as command:
            evidence = runtime.probe_runtime()
        self.assertFalse(evidence["gpu_operations_verified"])
        self.assertEqual(evidence["torch_probe"]["status"], "not_requested")
        self.assertEqual(command.call_count, 3)
        self.assertEqual(dict(os.environ), before)

    def test_optional_torch_probe_reports_only_explicit_operation_evidence(self):
        for verified in (False, True):
            with self.subTest(verified=verified), patch("runtime.shutil.which", return_value=None), patch(
                    "runtime.bounded_command", return_value={"returncode": 0, "stdout_truncated": False,
                    "stdout": json.dumps({"gpu_visible": True, "gpu_operations_verified": verified})}) as command:
                evidence = runtime.probe_runtime(torch_probe=True)
                self.assertEqual(evidence["gpu_operations_verified"], verified)
                self.assertEqual(command.call_args.args[0][:3], [sys.executable, "-B", "-c"])
                self.assertEqual(command.call_args.kwargs["timeout"], 30)

    def test_failed_or_missing_torch_cannot_verify_gpu(self):
        with patch("runtime.shutil.which", return_value=None), patch(
                "runtime.bounded_command", return_value={"returncode": 1, "stdout_truncated": False,
                                                         "stderr": "ModuleNotFoundError: No module named 'torch'"}):
            evidence = runtime.probe_runtime(True)
        self.assertFalse(evidence["gpu_operations_verified"])
        self.assertIn("No module named 'torch'", evidence["torch_probe"]["stderr"])

    def test_signaled_and_timed_out_gpu_child_are_blocked(self):
        for child in ({"returncode": -11, "stdout": "", "stdout_truncated": False},
                      {"error": "TimeoutExpired: GPU child", "stdout": "", "stdout_truncated": False}):
            with self.subTest(child=child), patch("runtime.shutil.which", return_value=None), patch(
                    "runtime.bounded_command", return_value=dict(child)):
                evidence = runtime.probe_runtime(True)
            self.assertFalse(evidence["gpu_operations_verified"])
            self.assertEqual(evidence["torch_probe"]["status"], "blocked")
            if child.get("returncode") == -11:
                self.assertEqual(evidence["torch_probe"]["signal"], 11)

    def test_explicit_override_is_recorded_not_applied_and_not_a_native_pass(self):
        with patch.dict(os.environ, {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}), patch(
                "runtime.shutil.which", return_value=None), patch("runtime.bounded_command", return_value={
                    "returncode": 0, "stdout_truncated": False,
                    "stdout": json.dumps({"gpu_visible": True, "gpu_operations_verified": True})}):
            before = dict(os.environ)
            evidence = runtime.probe_runtime(True)
            self.assertEqual(dict(os.environ), before)
        self.assertTrue(evidence["gpu_operations_verified"])
        self.assertEqual(evidence["environment"]["HSA_OVERRIDE_GFX_VERSION"], "10.3.0")
        self.assertEqual(evidence["probe_mode"], "explicit_environment_override")
        self.assertTrue(evidence["compatibility_workaround_requires_human_acknowledgment"])

    def test_subprocess_timeout_and_diagnostic_size_are_bounded(self):
        result = runtime.bounded_command([sys.executable, "-B", "-c",
            "import sys; print('x' * 20000); print('e' * 20000, file=sys.stderr)"])
        self.assertTrue(result["stdout_truncated"])
        self.assertEqual(len(result["stdout"]), 16384)
        self.assertTrue(result["stderr_truncated"])
        self.assertEqual(len(result["stderr"]), 16384)
        result = runtime.bounded_command([sys.executable, "-B", "-c", "import time; time.sleep(5)"], timeout=0.05)
        self.assertIn("TimeoutExpired", result["error"])

    def test_valid_json_stdout_survives_warning_stderr(self):
        result = runtime.bounded_command([sys.executable, "-B", "-c",
            "import json, sys; print(json.dumps({'gpu_operations_verified': True})); "
            "print('(null): No such file or directory', file=sys.stderr)"])
        self.assertEqual(json.loads(result["stdout"]), {"gpu_operations_verified": True})
        self.assertIn("(null): No such file or directory", result["stderr"])
        with patch("runtime.shutil.which", return_value=None), patch(
                "runtime.bounded_command", return_value=result):
            evidence = runtime.probe_runtime(True)
        self.assertTrue(evidence["gpu_operations_verified"])
        self.assertEqual(evidence["torch_probe"]["status"], "verified")
        self.assertIn("No such file", evidence["torch_probe"]["stderr"])

    def test_missing_pillow_has_exact_dependency_message(self):
        with patch.dict(sys.modules, {"PIL": None}):
            with self.assertRaisesRegex(RuntimeError, "missing dependency: Pillow"):
                images.pillow()


@unittest.skipIf(Image is None, "Pillow is required for PNG fixture tests (missing dependency: Pillow)")
class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = self.root / "train"
        self.evaluation = self.root / "eval"
        self.rights = self.root / "rights.json"
        for split, root, color in (("train", self.train, (200, 10, 20, 255)),
                                   ("eval", self.evaluation, (10, 200, 20, 255))):
            root.mkdir()
            self.png(root / "sample.png", color=color)
            (root / "metadata.jsonl").write_text(json.dumps({"file_name": "sample.png", "text": f"{split} adult portrait"}) + "\n")
        self.manifest = {"version": 1, "reviewed_by": "fixture author",
                         "entries": [self.permission("train", "sample.png"), self.permission("eval", "sample.png")]}
        self.save_rights()

    def png(self, path, size=48, color=(200, 10, 20, 255)):
        image = Image.new("RGBA", (size, size))
        image.putpixel((1, 1), color)
        image.save(path)

    def permission(self, split, name):
        return {"split": split, "file_name": name, "source": "synthetic test fixture",
                "license": "fixture only", "training_allowed" if split == "train" else "evaluation_allowed": True}

    def save_rights(self):
        self.rights.write_text(json.dumps(self.manifest))

    def dataset(self):
        return prepare.validate_dataset(self.train, self.evaluation, self.rights)

    def plan(self, verified=True, runtime_evidence=None, **changes):
        arguments = dict(train=self.train, evaluation=self.evaluation, rights=self.rights,
                         suite=SUITE, model="explicit/example", revision="a" * 40)
        arguments.update(changes)
        evidence = runtime_evidence if runtime_evidence is not None else {
            "python": "3.12.0", "python_executable": "/fixture/python",
            "environment": {"HSA_OVERRIDE_GFX_VERSION": None}, "probe_mode": "native",
            "gpu_operations_verified": verified,
            "torch_probe": {"evidence": {"torch_version": "fixture-torch", "hip_version": "fixture-hip",
                "cuda_version": None, "gpu_visible": True, "device": "fixture-device"}},
        }
        with patch("prepare.probe_runtime", return_value=evidence), patch(
                "subprocess.run", side_effect=AssertionError("must not launch training")):
            return prepare.build_plan(**arguments)

    def test_valid_png_metrics_are_not_visual_passes(self):
        metrics = images.validate_png(self.train / "sample.png", (48,))
        self.assertEqual(metrics["opaque_colors"], 1)
        self.assertEqual(metrics["opaque_pixels"], 1)
        self.assertEqual(metrics["transparent_pixels"], 48 * 48 - 1)
        self.assertFalse(metrics["visual_quality_assessed"])
        with self.assertRaisesRegex(ValueError, "square PNG"):
            images.validate_png(self.train / "sample.png", (64,))

    def test_invalid_images_rejected(self):
        path = self.root / "invalid.png"
        for size, color, message in ((32, (1, 2, 3, 255), "square PNG"),
                                     (48, (1, 2, 3, 128), "binary"),
                                     (48, (1, 2, 3, 0), "both transparent")):
            with self.subTest(message=message):
                self.png(path, size, color)
                with self.assertRaisesRegex(ValueError, message):
                    images.validate_png(path)
        Image.new("RGBA", (48, 48), (1, 2, 3, 255)).save(path)
        with self.assertRaisesRegex(ValueError, "both transparent"):
            images.validate_png(path)
        image = Image.new("RGBA", (48, 48))
        for number in range(63):
            image.putpixel((number % 48, number // 48), (number, 0, 0, 255))
        image.save(path)
        with self.assertRaisesRegex(ValueError, "62 opaque colors"):
            images.validate_png(path)
        path.write_bytes(b"not a PNG")
        with self.assertRaisesRegex(ValueError, "unreadable PNG"):
            images.validate_png(path)
        Image.new("RGB", (48, 48)).save(path, format="JPEG")
        with self.assertRaisesRegex(ValueError, "single-frame PNG"):
            images.validate_png(path)
        self.png(path, 64)
        self.assertEqual(images.validate_png(path, (64,))["width"], 64)
        data = bytearray(path.read_bytes())
        data[-13] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "unreadable PNG"):
            images.validate_png(path)

    def test_traversal_absolute_and_symlink_escapes_rejected(self):
        outside = self.root / "outside.png"
        self.png(outside)
        for name in ("../outside.png", str(outside), "sub/../../outside.png", "sub\\image.png", "./sample.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare.local_file(self.train, name)
        (self.train / "escape.png").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            prepare.local_file(self.train, "escape.png")
        (self.train / "metadata.jsonl").unlink()
        (self.train / "metadata.jsonl").symlink_to(self.evaluation / "metadata.jsonl")
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            self.dataset()

    def test_path_escape_is_a_blocked_plan(self):
        (self.train / "metadata.jsonl").write_text(json.dumps({"file_name": "../evil.png", "text": "caption"}))
        plan = self.plan()
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("path escape", " ".join(plan["blockers"]))

    def test_missing_caption_empty_split_missing_rights_and_denied_rights_block(self):
        metadata = self.train / "metadata.jsonl"
        original = metadata.read_text()
        for content in ("", json.dumps({"file_name": "sample.png", "text": " "}), "[]", "{malformed"):
            metadata.write_text(content)
            self.assertEqual(self.plan()["status"], "blocked")
        metadata.write_text(original)
        self.rights.unlink()
        self.assertEqual(self.plan()["status"], "blocked")
        self.manifest["entries"][0]["training_allowed"] = False
        self.save_rights()
        self.assertEqual(self.plan()["status"], "blocked")
        self.manifest["entries"] = self.manifest["entries"][1:]
        self.save_rights()
        self.assertEqual(self.plan()["status"], "blocked")

    def test_duplicate_hashes_and_reencoded_pixels_rejected(self):
        source = self.train / "sample.png"
        target = self.evaluation / "sample.png"
        target.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate image: train/sample.png and eval/sample.png"):
            self.dataset()
        with Image.open(source) as image:
            image.save(target, compress_level=0)
        self.assertNotEqual(source.read_bytes(), target.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            self.dataset()

    def test_deterministic_dataset_fingerprint_ignores_row_and_rights_order(self):
        self.png(self.train / "second.png", color=(2, 3, 4, 255))
        metadata = self.train / "metadata.jsonl"
        rows = [json.loads(metadata.read_text()), {"file_name": "second.png", "text": "another adult portrait"}]
        metadata.write_text("\n".join(map(json.dumps, rows)))
        self.manifest["entries"].append(self.permission("train", "second.png"))
        self.save_rights()
        first = self.dataset()["sha256"]
        metadata.write_text("\n".join(map(json.dumps, reversed(rows))) + "\n\n")
        self.manifest["entries"].reverse()
        self.save_rights()
        self.assertEqual(first, self.dataset()["sha256"])
        rows[0]["text"] = "changed caption"
        metadata.write_text("\n".join(map(json.dumps, rows)))
        self.assertNotEqual(first, self.dataset()["sha256"])
        caption_digest = self.dataset()["sha256"]
        self.manifest["entries"][0]["license"] = "changed declaration"
        self.save_rights()
        self.assertNotEqual(caption_digest, self.dataset()["sha256"])

    def test_contact_sheet_nearest_neighbor_scaling_preserves_pixels(self):
        output = self.root / "sheet.png"
        source = self.train / "sample.png"
        result = images.contact_sheet([source], output, scale=3, columns=2)
        self.assertEqual((result["width"], result["height"]), (144, 144))
        with Image.open(source) as original, Image.open(output) as sheet:
            for y in range(sheet.height):
                for x in range(sheet.width):
                    self.assertEqual(sheet.getpixel((x, y)), original.getpixel((x // 3, y // 3)))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            images.contact_sheet([source], source)
        with self.assertRaises(ValueError):
            images.contact_sheet([source], output, scale=0)

    def test_mixed_size_sheet_uses_native_integer_scaling(self):
        self.png(self.evaluation / "sample.png", size=64, color=(1, 2, 3, 255))
        output = self.root / "mixed.png"
        result = images.contact_sheet([self.train / "sample.png", self.evaluation / "sample.png"], output, 2, 2)
        self.assertEqual((result["width"], result["height"]), (256, 128))
        with Image.open(output) as sheet:
            self.assertEqual(sheet.getpixel((2, 2)), (200, 10, 20, 255))
            self.assertEqual(sheet.getpixel((130, 2)), (1, 2, 3, 255))
            self.assertEqual(sheet.getpixel((120, 2)), (0, 0, 0, 0))

    def test_all_preparation_gates_and_no_training_side_effects(self):
        for changes in ({"train": None}, {"rights": None}, {"model": None}, {"revision": None},
                        {"steps": 0}, {"steps": 1001}, {"rank": 17}, {"training_resolution": 513}):
            with self.subTest(changes=changes):
                self.assertEqual(self.plan(**changes)["status"], "blocked")
        self.assertEqual(self.plan(verified=False)["status"], "blocked")
        before = sorted(self.root.rglob("*"))
        plan = self.plan()
        self.assertEqual(plan["status"], "needs_human")
        self.assertFalse(plan["training_started"])
        self.assertFalse(plan["training_authorized"])
        self.assertIsNone(plan["training_command"])
        self.assertTrue(all(value == "unassessed" for value in plan["quality_assessments"].values()))
        self.assertEqual(prepare.canonical_digest(plan["preparation_binding"]), plan["preparation_sha256"])
        self.assertNotEqual(plan["preparation_sha256"], self.plan(steps=101)["preparation_sha256"])
        self.assertNotEqual(plan["preparation_sha256"], self.plan(revision="b" * 40)["preparation_sha256"])
        self.assertIsNone(self.plan(model=None)["preparation_sha256"])
        self.assertTrue(any("acknowledge the unsupported compatibility workaround" in gate
                            for gate in plan["human_gate"]))
        self.assertEqual(plan["config"]["batch_size"], 1)
        self.assertEqual(plan["config"]["output_target_sizes"], [48, 64])
        self.assertEqual(plan["config"]["training_resolution"], 512)
        self.assertEqual(before, sorted(self.root.rglob("*")))
        frozen = plan["evaluation_suite"]
        self.assertEqual(hashlib.sha256(frozen["frozen_json"].encode()).hexdigest(), frozen["sha256"])
        self.assertEqual(frozen["sha256"], hashlib.sha256(SUITE.read_bytes()).hexdigest())

    def test_mutable_or_partial_model_revisions_are_blocked(self):
        for revision in ("main", "v1.0", "abcdef", "g" * 40, "a" * 41):
            with self.subTest(revision=revision):
                plan = self.plan(revision=revision)
                self.assertEqual(plan["status"], "blocked")
                self.assertIsNone(plan["preparation_sha256"])
                self.assertIn("immutable commit/content digest", " ".join(plan["blockers"]))
        self.assertEqual(self.plan(revision="b" * 64)["status"], "needs_human")

    def test_preparation_digest_binds_runtime_backend_and_override(self):
        baseline = self.plan()
        self.assertEqual(baseline["preparation_binding"]["runtime_contract"], baseline["runtime_contract"])
        for path, value in ((["python"], "3.12.1"), (["python_executable"], "/another/python"),
                            (["torch_probe", "evidence", "torch_version"], "other-torch"),
                            (["torch_probe", "evidence", "hip_version"], "other-hip"),
                            (["environment", "HSA_OVERRIDE_GFX_VERSION"], "10.3.0"),
                            (["gpu_operations_verified"], False)):
            with self.subTest(path=path):
                evidence = json.loads(json.dumps(baseline["runtime"]))
                target = evidence
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                changed = self.plan(runtime_evidence=evidence)
                self.assertNotEqual(baseline["preparation_sha256"], changed["preparation_sha256"])
                self.assertFalse(changed["training_authorized"])

    def test_bad_suite_blocks_and_fixed_suite_has_both_layouts(self):
        bad = self.root / "bad-suite.json"
        bad.write_text('{}')
        self.assertEqual(self.plan(suite=bad)["status"], "blocked")
        suite = json.loads(prepare.freeze_suite(SUITE)["frozen_json"])
        self.assertEqual({layout["size"] for layout in suite["target_layouts"]}, {48, 64})
        self.assertTrue(all("adult anime woman" in case["prompt"] for case in suite["cases"]))
        suite["cases"][0]["seed"] = None
        bad.write_text(json.dumps(suite))
        self.assertEqual(self.plan(suite=bad)["status"], "blocked")

    def test_cli_writes_only_blocked_manifest_and_never_overwrites_inputs(self):
        output = self.root / "plan.json"
        with patch("sys.argv", ["prepare.py", "--output", str(output)]), patch(
                "prepare.probe_runtime", return_value={"gpu_operations_verified": False}), patch("builtins.print"):
            with self.assertRaises(SystemExit) as error:
                prepare.main()
        self.assertEqual(error.exception.code, 2)
        plan = json.loads(output.read_text())
        self.assertEqual(plan["status"], "blocked")
        self.assertFalse(plan["training_started"])
        self.assertFalse(plan["training_authorized"])
        for destination in (output, self.train / "new-plan.json", self.rights):
            with patch("sys.argv", ["prepare.py", "--train", str(self.train), "--rights", str(self.rights),
                                     "--output", str(destination)]), patch("sys.stderr"), patch("prepare.build_plan") as build:
                with self.assertRaises(SystemExit):
                    prepare.main()
                build.assert_not_called()
        self.assertEqual(json.loads(self.rights.read_text()), self.manifest)
        self.assertFalse(list(self.root.rglob("*.safetensors")))


if __name__ == "__main__":
    unittest.main()
