"""CPU fixtures and real tiny child processes; no GPU/model/network operations."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from haremizer.core import REQUIRED_CHECKS
from haremizer.prepare import canonical_digest
from haremizer import refinement_loop as rl


class RefinementLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.safetensors").write_bytes(b"CPU fixture, not weights")
        revision = canonical_digest({"weights.safetensors": rl.fingerprint(self.model / "weights.safetensors")})
        self.spec = {"path": str(self.model), "id": "fixture/model", "revision": revision}
        self.settings = {**rl.SETTINGS, "negative_prompt": "cropped, duplicate face"}
        self.plan = {"model": self.spec, "python": sys.executable,
                     "deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                     "max_stages": 4, "train_seconds": 900, "sample_seconds": 600,
                     "storage_paths": [str(self.root), str(Path(sys.executable).resolve())],
                     "settings": self.settings, "suites": [], "validation_suites": []}
        for group, offset in (("suites", 0), ("validation_suites", 100)):
            for index in range(2):
                suite = self.root / f"{group}-{index}.json"
                self.dump(suite, {"version": 1, "cases": [
                    {"id": f"case-{n}", "prompt": f"adult portrait {index} {n}", "seed": offset + index * 4 + n}
                    for n in range(4)], "target_layouts": [
                    {"size": 48, "composition": "independent"}, {"size": 64, "composition": "independent"}]})
                baseline = self.root / f"base-{group}-{index}"
                self.samples(baseline, suite, None)
                self.plan[group].append({"path": str(suite), "baseline": str(baseline)})
        self.plan_path = self.root / "plan-input.json"
        self.dump(self.plan_path, self.plan)
        self.manifest_patch = patch.object(rl, "model_manifest", return_value={
            "class": "StableDiffusionXLPipeline", "files": {"weights.safetensors": {}}})
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)
        self.preflight_patch = patch.object(rl.smoke_sdxl, "preflight", return_value={"quality_assessed": False})
        self.preflight_mock = self.preflight_patch.start()
        self.addCleanup(self.preflight_patch.stop)
        self.adapter_patch = patch.object(rl.smoke_sdxl, "validate_adapter_file")
        self.adapter_patch.start()
        self.addCleanup(self.adapter_patch.stop)
        self.loop = rl.Loop.init(self.root / "run", self.plan_path)
        self.request = self.root / "request.json"
        self.dump(self.request, {"model": self.spec, "examples": [], "heldout_examples": [], "continuation": None})
        self.real_popen = subprocess.Popen
        self.mock_children = patch.object(rl.subprocess, "Popen", side_effect=self.child)
        self.mock_children.start()
        self.addCleanup(self.mock_children.stop)

    @staticmethod
    def dump(path, value):
        path.write_text(json.dumps(value))

    def samples(self, directory, suite, checkpoint):
        directory.mkdir()
        records = []
        for index, case in enumerate(json.loads(suite.read_bytes())["cases"], 1):
            path = directory / f"case-{index:02}.png"
            Image.new("RGB", (768, 768), (case["seed"] % 256, 40, 60)).save(path)
            records.append({**case, "file": path.name, "sha256": rl.fingerprint(path), "gpu_memory": {"fixture": True}})
        self.dump(directory / "manifest.json", {"model": self.spec,
                  "settings": {**self.settings, "device": "cuda",
                               **({"scheduler_class": "EulerAncestralDiscreteScheduler"}
                                  if self.settings.get("scheduler_override") == "euler_ancestral" else {})},
                  "lora": checkpoint,
                  "suite_sha256": rl.fingerprint(suite), "records": records})

    def child(self, argv, **kwargs):
        # du stays real. Trusted trainer/sampler argv is recorded by the loop, but
        # tests replace only the actual ML process with a tiny CPU subprocess.
        if "haremizer.smoke_sdxl" not in argv and "haremizer.sample" not in argv:
            return self.real_popen(argv, **kwargs)
        def arg(name):
            return argv[argv.index(name) + 1]
        output = Path(arg("--output"))
        if "--refine" in argv:
            output.mkdir()
            checkpoint = output / "pytorch_lora_weights.safetensors"
            checkpoint.write_bytes(b"CPU fixture adapter")
            self.dump(output / "outcome.json", {"model": self.spec, "status": "refinement_completed_unreviewed",
                      "phase": "completed", "training_started": True,
                      "runtime": {"memory": {"fixture": True}},
                      "checkpoint": {"path": checkpoint.name, "sha256": rl.fingerprint(checkpoint), "reload_exact": True}})
        else:
            self.samples(output, Path(arg("--suite")), {"path": arg("--lora"), "sha256": arg("--lora-sha256")})
        return self.real_popen([sys.executable, "-B", "-c",
                               "import os,resource; print(os.environ['HSA_OVERRIDE_GFX_VERSION']); "
                               "assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0)"], **kwargs)

    def stage(self):
        self.loop.enqueue(self.request, "explicit CPU test stage")
        self.loop.pipeline()
        self.assertEqual(self.loop.state["status"], "needs_visual_review")
        return self.loop.state["stages"][-1]

    def review_file(self, stage, *, uncertain=False, negative=False, robust=False):
        native = Path(stage["directory"]) / "native64.png"
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        image.putpixel((20, 20), (200, 20, 30, 255))
        image.save(native)
        value = {"reviewed_by": "machine:sol", "notes": "CPU declaration only, not actual semantic review",
                 "uncertain": uncertain, "robust_requested": robust,
                 "checks": dict.fromkeys(REQUIRED_CHECKS, True),
                 "binding": self.loop.status()["review_binding"],
                 "cases": {r["case"]: {k: "fixture observation, not quality proof" for k in
                            ("glitches", "crop_duplicate_face", "adherence", "readability")}
                           for r in stage["cases"]}, "format_evidence": [rl.identity(native)]}
        if negative:
            value["checks"]["pixel_quality"] = False
        path = self.root / "review-input.json"
        self.dump(path, value)
        return path

    def test_single_stage_pauses_and_records_bound_evidence(self):
        stage = self.stage()
        self.assertEqual(len(stage["cases"]), 8)
        self.assertEqual(len(stage["processes"]), 3)
        self.assertIsNone(self.loop.state["incumbent"])
        self.assertTrue(all(p["code"] == 0 for p in stage["processes"]))
        self.assertIn("--refine", stage["processes"][0]["argv"])
        self.assertTrue(all("--scheduler" not in p["argv"] for p in stage["processes"]))
        self.assertEqual(stage["processes"][0]["memory_outcome"], {"fixture": True})
        self.assertIn("10.3.0", Path(stage["processes"][0]["stdout"]).read_text())
        self.assertEqual(self.preflight_mock.call_args.kwargs, {"refine": True})
        with self.assertRaises(ValueError):
            self.loop.pipeline()
        with self.assertRaises(ValueError):
            self.loop.enqueue(self.request, "cannot bypass review")
        self.assertEqual(rl.Loop(self.loop.directory).state["status"], "needs_visual_review")

    def test_optional_ancestral_scheduler_is_frozen_forwarded_and_verified(self):
        self.settings["scheduler_override"] = "euler_ancestral"
        for row in self.plan["suites"] + self.plan["validation_suites"]:
            path = Path(row["baseline"]) / "manifest.json"
            report = json.loads(path.read_bytes())
            report["settings"].update(scheduler_override="euler_ancestral",
                                      scheduler_class="EulerAncestralDiscreteScheduler")
            self.dump(path, report)
        self.dump(self.plan_path, self.plan)
        self.loop = rl.Loop.init(self.root / "ancestral", self.plan_path)
        stage = self.stage()
        for process in stage["processes"][1:]:
            argv = process["argv"]
            self.assertEqual(argv[argv.index("--scheduler") + 1], "euler_ancestral")
        row = self.plan["suites"][0]
        path = Path(row["baseline"]) / "manifest.json"
        report = json.loads(path.read_bytes())
        report["settings"]["scheduler_class"] = "EulerDiscreteScheduler"
        self.dump(path, report)
        with self.assertRaisesRegex(ValueError, "settings mismatch"):
            rl.Loop.manifest(Path(row["baseline"]), Path(row["path"]), self.spec, self.settings, None)
        for value in (None, True, "euler", "EulerAncestralDiscreteScheduler", []):
            plan = copy.deepcopy(self.plan)
            plan["settings"]["scheduler_override"] = value
            self.dump(self.plan_path, plan)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fixed 768"):
                rl.Loop.init(self.root / "invalid-scheduler", self.plan_path)

    def test_negative_and_uncertain_reviews_preserve_incumbent(self):
        for negative, uncertain in ((True, False), (False, True)):
            stage = self.stage()
            self.loop.review(self.review_file(stage, negative=negative, uncertain=uncertain, robust=True))
            self.assertEqual(self.loop.state["status"], "rejected")
            self.assertIsNone(self.loop.state["incumbent"])
            self.assertIn("failure", stage)
            request = json.loads(self.request.read_bytes())
            request["test_explicit_change"] = negative
            self.dump(self.request, request)

    def test_promising_not_robust_and_same_weights_second_pass(self):
        first = self.stage()
        with self.assertRaisesRegex(ValueError, "two accepted"):
            self.loop.review(self.review_file(first, robust=True))
        self.loop.review(self.review_file(first))
        self.assertEqual(self.loop.state["status"], "promising")
        self.assertIsNone(self.loop.state["incumbent"])
        self.loop.pipeline(Path(first["checkpoint"]["path"]), Path(first["outcome"]["path"]))
        second = self.loop.state["stages"][-1]
        self.assertTrue(second["validation_only"])
        self.assertEqual(len(second["processes"]), 2)
        self.assertEqual(first["checkpoint"], second["checkpoint"])
        self.assertNotEqual(first["suite_hashes"], second["suite_hashes"])
        self.loop.review(self.review_file(second, robust=True))
        self.assertEqual(self.loop.state["status"], "robust")
        self.assertEqual(self.loop.state["incumbent"], first["checkpoint"])
        # A later rejected stage cannot replace the accepted checkpoint.
        request = json.loads(self.request.read_bytes())
        request["explicit_change"] = "third"
        self.dump(self.request, request)
        third = self.stage()
        self.loop.review(self.review_file(third, negative=True))
        self.assertEqual(self.loop.state["incumbent"], first["checkpoint"])

    def test_mutated_evidence_and_missing_case_notes_block_review(self):
        stage = self.stage()
        path = self.review_file(stage)
        review = json.loads(path.read_bytes())
        del review["cases"][next(iter(review["cases"]))]
        self.dump(path, review)
        with self.assertRaisesRegex(ValueError, "all eight"):
            self.loop.review(path)
        path = self.review_file(stage)
        Path(stage["cases"][0]["candidate"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.review(path)
        self.assertEqual(self.loop.state["status"], "needs_visual_review")

    def test_suite_model_and_checkpoint_changes_are_blocked(self):
        suite = Path(self.plan["suites"][0]["path"])
        original = suite.read_bytes()
        suite.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.enqueue(self.request, "changed suite")
        suite.write_bytes(original)
        model = self.model / "weights.safetensors"
        original_model = model.read_bytes()
        model.write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.enqueue(self.request, "changed model")
        model.write_bytes(original_model)
        stage = self.stage()
        review = self.review_file(stage)
        Path(stage["checkpoint"]["path"]).write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.review(review)

    def test_format_preview_is_not_native_evidence(self):
        stage = self.stage()
        path = self.review_file(stage)
        review = json.loads(path.read_bytes())
        review["format_evidence"] = [stage["evidence"][-1]]
        self.dump(path, review)
        with self.assertRaisesRegex(ValueError, "separate postprocessed"):
            self.loop.review(path)

    def test_hash_changes_and_no_overwrite(self):
        self.loop.enqueue(self.request, "hash test")
        frozen = Path(self.loop.state["queue"]["request"]["path"])
        frozen.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.pipeline()
        with self.assertRaises(FileExistsError):
            rl.Loop.init(self.loop.directory, self.plan_path)
        baseline = Path(self.plan["suites"][0]["baseline"]) / "case-01.png"
        baseline.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.verify()

    def test_deadline_storage_and_stage_bounds(self):
        expired = copy.deepcopy(self.plan)
        expired["deadline_utc"] = (datetime.now(timezone.utc) + timedelta(hours=13)).isoformat()
        self.dump(self.plan_path, expired)
        with self.assertRaisesRegex(ValueError, "12 hours"):
            rl.Loop.init(self.root / "other", self.plan_path)
        self.loop.plan["deadline_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "deadline_exhausted"):
            self.loop.enqueue(self.request, "expired")
        self.loop.plan["deadline_utc"] = self.plan["deadline_utc"]
        with patch.object(rl, "LIMIT", 1):
            with self.assertRaisesRegex(ValueError, "storage ceiling"):
                self.loop.enqueue(self.request, "over budget")
        first = self.stage()
        self.loop.review(self.review_file(first))
        self.loop.plan["max_stages"] = 1
        with self.assertRaisesRegex(ValueError, "max_stages_exhausted"):
            self.loop.enqueue(self.request, "over stages")

    def test_timeout_kills_owned_group_and_records_failure(self):
        stage = {"directory": str(self.loop.directory), "processes": []}
        with self.assertRaises(TimeoutError):
            self.loop.process(stage, [sys.executable, "-B", "-c", "import time; time.sleep(10)"], 0.2, "timeout")
        record = stage["processes"][0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["code"], -9)
        with self.assertRaises(ProcessLookupError):
            os.killpg(record["owned_pgid"], 0)
        self.loop.plan["deadline_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=0.3)).isoformat()
        with self.assertRaises(TimeoutError):
            self.loop.process(stage, [sys.executable, "-B", "-c", "import time; time.sleep(10)"], 900, "deadline")
        self.assertLessEqual(stage["processes"][-1]["timeout_seconds"], 0.3)

    def test_unreaped_owned_child_blocks_further_stages(self):
        from unittest.mock import Mock
        import signal
        stage = {"directory": str(self.loop.directory), "processes": []}
        child = Mock(pid=123456, returncode=None)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("fixture", 5)
        with patch.object(rl.subprocess, "Popen", return_value=child), \
                patch.object(self.loop, "bounds"), \
                patch.object(rl.os, "killpg") as kill:
            with self.assertRaises(TimeoutError):
                self.loop.process(stage, ["fixture"], 0.01, "stuck")
        child.wait.assert_called_once_with(timeout=5)
        kill.assert_called_once_with(123456, signal.SIGKILL)
        self.assertTrue(stage["processes"][0]["cleanup_pending"])
        self.loop.state["stages"].append(stage)
        with self.assertRaisesRegex(ValueError, "cleanup pending"):
            self.loop.available()

    def test_failed_child_does_not_sample_or_promote(self):
        self.loop.enqueue(self.request, "failed CPU subprocess")
        self.mock_children.stop()
        def fail(argv, **kwargs):
            if "--refine" in argv:
                return self.real_popen([sys.executable, "-B", "-c", "raise SystemExit(3)"], **kwargs)
            return self.real_popen(argv, **kwargs)
        with patch.object(rl.subprocess, "Popen", side_effect=fail):
            self.loop.pipeline()
        self.assertEqual(self.loop.state["status"], "blocked")
        self.assertEqual(len(self.loop.state["stages"][-1]["processes"]), 1)
        self.assertIsNone(self.loop.state["best_promising"])
        self.assertEqual(self.loop.state["stages"][-1]["processes"][0]["code"], 3)

    def test_second_pass_rechecks_prior_evidence(self):
        first = self.stage()
        self.loop.review(self.review_file(first))
        self.loop.pipeline(Path(first["checkpoint"]["path"]), Path(first["outcome"]["path"]))
        second = self.loop.state["stages"][-1]
        path = self.review_file(second, robust=True)
        Path(first["cases"][0]["candidate"]["path"]).write_bytes(b"changed prior evidence")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.loop.review(path)
        self.assertIsNone(self.loop.state["incumbent"])


if __name__ == "__main__":
    unittest.main()
