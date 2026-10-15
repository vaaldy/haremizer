import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import backend
from core import Attempt, Run, fingerprint
from images import pillow
import prepare


class BackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
            if k not in (*backend.ENV_KEYS, "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.deps = patch("backend.dependencies", return_value=[("torch", "fixture-torch", "fixture-installation")])
        self.deps.start()
        self.addCleanup(self.deps.stop)
        self.facts = {
            "python": ".".join(map(str, sys.version_info[:3])), "python_executable": sys.executable,
            "environment": dict.fromkeys(backend.ENV_KEYS), "probe_mode": "native",
            "compatibility_workaround_requires_human_acknowledgment": False,
            "gpu_operations_verified": True,
            "torch_probe": {"evidence": {"torch_version": "fixture-torch", "hip_version": "fixture-hip",
                "cuda_version": None, "gpu_visible": True, "device": "fixture-device"}},
        }
        rights = {"version": 1, "reviewed_by": "fixture author", "entries": []}
        for split, color in (("train", (20, 30, 40, 255)), ("eval", (50, 60, 70, 255))):
            folder = self.root / split
            folder.mkdir()
            image = pillow().new("RGBA", (64, 64))
            image.putpixel((1, 1), color)
            image.save(folder / "sample.png")
            (folder / "metadata.jsonl").write_text(json.dumps({"file_name": "sample.png", "text": split + " adult fixture"}))
            rights["entries"].append({"split": split, "file_name": "sample.png", "source": "synthetic fixture",
                "license": "test only", "training_allowed" if split == "train" else "evaluation_allowed": True})
        (self.root / "rights.json").write_text(json.dumps(rights))
        self.suite = self.root / "suite.json"
        self.suite.write_bytes((ROOT / "evaluation-suite.json").read_bytes())
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}))
        (self.model / "weights.fake").write_bytes(b"synthetic model identity only")
        revision = prepare.canonical_digest({str(p.relative_to(self.model)): fingerprint(p) for p in sorted(self.model.iterdir())})
        self.script = self.root / "script ; $(touch INJECTION)" / "train_text_to_image_lora.py"
        self.script.parent.mkdir()
        self.script.write_text("raise AssertionError('TEST SCRIPT MUST NEVER EXECUTE')\n")
        with patch("prepare.probe_runtime", return_value=self.facts):
            self.plan = prepare.build_plan(train=self.root / "train", evaluation=self.root / "eval",
                rights=self.root / "rights.json", suite=self.suite, model="fixture/sd1", revision=revision,
                training_resolution=128, steps=1, rank=1)
        self.plan_path = self.root / "preparation.json"
        self.save_plan()
        self.run = Run.create(self.root / "run", self.suite, self.plan["dataset"]["sha256"], max_attempts=1)
        self.spec = {"preparation": str(self.plan_path), "train": str(self.root / "train"),
            "evaluation": str(self.root / "eval"), "rights": str(self.root / "rights.json"),
            "suite": str(self.suite), "run_directory": str(self.run.directory), "python": sys.executable,
            "trainer": str(self.script), "trainer_sha256": fingerprint(self.script), "model_path": str(self.model),
            "timeout_seconds": 10, "max_attempts": 1, "mixed_precision": "no",
            "preprocessing": {"resize": "NEAREST", "alpha": "flatten", "background": [1, 2, 3]},
            "script_contract": {"name": backend.CONTRACT, "trainer_sha256": fingerprint(self.script),
                "reviewed_by": "synthetic fixture", "note": "mock contract only, never execute"}}
        self.request = self.root / "request.json"
        self.save_spec()
        self.approval = self.root / "human.json"
        self.trainer = backend.LoraTrainer(self.request, self.approval)

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def save_spec(self):
        self.request.write_text(json.dumps(self.spec))

    def authorize_fixture(self, **changes):
        # Synthetic test authorization only; production adapter exposes no approval writer.
        approval = self.trainer.preview()["approval_template"]
        approval.update(training_authorized=True, approved_by="synthetic test human", note="test only")
        approval.update(changes)
        self.approval.write_text(json.dumps(approval))

    def attempt(self, number=1):
        directory = self.run.directory / f"attempt-{number:03}"
        directory.mkdir(exist_ok=True)
        self.run.state.update(status="running", attempts=number)
        self.run._save()
        return Attempt(directory, number, self.run.directory / "suite.json", self.plan["dataset"]["sha256"], None)

    def output_process(self, argv, **kwargs):
        output = Path(argv[argv.index("--output_dir") + 1])
        output.mkdir()
        (output / "pytorch_lora_weights.safetensors").write_bytes(b"mock LoRA, not real weights")
        return Mock(pid=12345, wait=Mock(return_value=0))

    def test_preview_and_cli_never_launch_or_populate_human_approval(self):
        with patch("backend.subprocess.Popen", side_effect=AssertionError("no process")), patch(
                "backend.probe_runtime", side_effect=AssertionError("no GPU")), patch("builtins.print") as printed, patch(
                "sys.argv", ["backend.py", str(self.request)]):
            preview = self.trainer.preview()
            backend.main()
        self.assertFalse(preview["training_authorized"])
        self.assertFalse(preview["approval_template"]["training_authorized"])
        self.assertEqual(preview["approval_template"]["approved_by"], "")
        self.assertEqual(json.loads(printed.call_args.args[0]), json.loads(json.dumps(preview)))
        self.assertFalse(self.approval.exists())
        argv = preview["execution"]["argv_template"]
        self.assertEqual(argv[argv.index("--rank") + 1], "1")
        self.assertNotIn("--lora_rank", argv)
        self.assertNotIn("--allow_tf32", argv)
        self.assertNotIn("--use_8bit_adam", argv)
        self.assertNotIn("--enable_xformers_memory_efficient_attention", argv)

    def test_absent_false_and_mismatched_approval_fail_before_gpu_or_process(self):
        attempt = self.attempt()
        cases = ({}, {"training_authorized": False}, {"execution_sha256": "wrong"},
                 {"preparation_sha256": "wrong"}, {"approved_by": ""}, {"note": ""})
        with patch("backend.probe_runtime") as probe, patch("backend.subprocess.Popen") as launch:
            with self.assertRaises(FileNotFoundError):
                self.trainer.train(attempt)
            for changes in cases:
                self.authorize_fixture(**changes)
                if changes == {}:
                    self.approval.write_text("{}")
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.trainer.train(attempt)
        probe.assert_not_called()
        launch.assert_not_called()

    def test_changed_input_identities_fail_closed(self):
        attempt = self.attempt()
        targets = (self.root / "train" / "metadata.jsonl", self.root / "eval" / "metadata.jsonl",
                   self.root / "rights.json", self.suite, self.script, self.model / "weights.fake",
                   self.run.directory / "suite.json")
        for target in targets:
            self.authorize_fixture()
            original = target.read_bytes()
            target.write_bytes(original + b"changed")
            with self.subTest(target=target), patch("backend.probe_runtime") as probe, patch(
                    "backend.subprocess.Popen") as launch, self.assertRaises((ValueError, json.JSONDecodeError)):
                self.trainer.train(attempt)
            probe.assert_not_called()
            launch.assert_not_called()
            target.write_bytes(original)

    def test_fake_binding_mutable_revision_runtime_and_dependency_changes_block(self):
        for key, value in (("preparation_sha256", "fake"), ("status", "ready"), ("training_authorized", True)):
            original = self.plan[key]
            self.plan[key] = value
            self.save_plan()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.trainer.preview()
            self.plan[key] = original
        self.save_plan()
        with patch.dict(os.environ, {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}), self.assertRaisesRegex(ValueError, "environment"):
            self.trainer.preview()
        self.authorize_fixture()
        with patch("backend.dependencies", return_value=[("torch", "changed", "changed")]), self.assertRaisesRegex(ValueError, "torch"):
            self.trainer.preview()
        self.plan["model"]["revision"] = "main"
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.trainer.preview()

    def test_execution_digest_covers_policy_limits_contract_runner_and_dependencies(self):
        attempt = self.attempt()
        for key, value in (("timeout_seconds", 11), ("mixed_precision", "fp16"),
                           ("preprocessing", {"resize": "NEAREST", "alpha": "flatten", "background": [3, 2, 1]}),
                           ("script_contract", {**self.spec["script_contract"], "note": "changed review"})):
            self.authorize_fixture()
            original = self.spec[key]
            self.spec[key] = value
            self.save_spec()
            with self.subTest(key=key), patch("backend.probe_runtime") as probe, self.assertRaisesRegex(ValueError, "authorization"):
                self.trainer.train(attempt)
            probe.assert_not_called()
            self.spec[key] = original
            self.save_spec()
        self.authorize_fixture()
        with patch("backend.dependencies", return_value=[("torch", "fixture-torch", "different-install")]), self.assertRaisesRegex(ValueError, "authorization"):
            self.trainer.train(attempt)
        real_fingerprint = backend.fingerprint
        with patch("backend.fingerprint", side_effect=lambda p: "different-python" if Path(p) == Path(sys.executable) else real_fingerprint(p)), self.assertRaisesRegex(ValueError, "authorization"):
            self.trainer.train(attempt)

    def test_unrecorded_interpreter_loader_overrides_are_rejected_not_changed(self):
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            with self.subTest(key=key), patch.dict(os.environ, {key: "/unrecorded"}), patch(
                    "backend.probe_runtime") as probe, patch("backend.subprocess.Popen") as launch:
                before = dict(os.environ)
                with self.assertRaisesRegex(ValueError, key + ".*clean dedicated runner"):
                    self.trainer.preview()
                self.assertEqual(dict(os.environ), before)
                probe.assert_not_called()
                launch.assert_not_called()

    def test_runtime_reprobe_and_runner_identity_changes_block_before_launch(self):
        self.authorize_fixture()
        attempt = self.attempt()
        for key in ("torch_version", "hip_version", "cuda_version", "device", "gpu_visible"):
            facts = json.loads(json.dumps(self.facts))
            facts["torch_probe"]["evidence"][key] = "changed"
            with self.subTest(key=key), patch("backend.probe_runtime", return_value=facts), patch(
                    "backend.subprocess.Popen") as launch, self.assertRaisesRegex(ValueError, "runtime changed"):
                self.trainer.train(attempt)
            launch.assert_not_called()
            provenance = attempt.directory / "backend.json"
            self.assertEqual(json.loads(provenance.read_text())["runtime_revalidation"], facts)
            provenance.unlink()
        self.spec["python"] = str(self.root / "different-python")
        self.save_spec()
        with self.assertRaisesRegex(ValueError, "Python runner"):
            self.trainer.preview()

    def test_cached_hf_revision_requires_model_id_and_rejects_changed_bytes(self):
        import shutil
        cache = self.root / "models--fixture--sd1" / "snapshots" / ("a" * 40)
        cache.parent.mkdir(parents=True)
        shutil.copytree(self.model, cache)
        self.spec["model_path"] = str(cache)
        self.save_spec()
        self.plan["model"]["revision"] = "a" * 40
        self.plan["preparation_binding"]["model"] = self.plan["model"]
        self.plan["preparation_sha256"] = prepare.canonical_digest(self.plan["preparation_binding"])
        self.save_plan()
        self.authorize_fixture()
        attempt = self.attempt()
        (cache / "weights.fake").write_bytes(b"changed cache")
        with patch("backend.probe_runtime") as probe, self.assertRaisesRegex(ValueError, "authorization"):
            self.trainer.train(attempt)
        probe.assert_not_called()
        self.plan["model"]["id"] = "different/model"
        self.save_plan()
        with self.assertRaises(ValueError):
            self.trainer.preview()

    def test_invalid_limits_policy_and_undeclared_script_contract_are_blocked(self):
        for key, value in (("timeout_seconds", float("inf")), ("timeout_seconds", 0),
                           ("max_attempts", 4), ("mixed_precision", "bf16"), ("script_contract", {}),
                           ("preprocessing", {"resize": "NEAREST", "alpha": "preserve"})):
            original = self.spec[key]
            self.spec[key] = value
            self.save_spec()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.trainer.preview()
            self.spec[key] = original
            self.save_spec()

    def test_override_requires_explicit_human_note_and_reprobe_must_match(self):
        override = "10.3.0"
        self.facts.update(probe_mode="explicit_environment_override", compatibility_workaround_requires_human_acknowledgment=True)
        self.facts["environment"]["HSA_OVERRIDE_GFX_VERSION"] = override
        self.plan["runtime"] = self.facts
        self.plan["runtime_contract"] = backend.runtime_contract(self.facts)
        self.plan["preparation_binding"]["runtime_contract"] = self.plan["runtime_contract"]
        self.plan["preparation_sha256"] = prepare.canonical_digest(self.plan["preparation_binding"])
        self.save_plan()
        attempt = self.attempt()
        with patch.dict(os.environ, {"HSA_OVERRIDE_GFX_VERSION": override}):
            self.authorize_fixture()
            with patch("backend.probe_runtime") as probe, self.assertRaisesRegex(ValueError, "acknowledge"):
                self.trainer.train(attempt)
            probe.assert_not_called()
            self.authorize_fixture(note="I acknowledge HSA_OVERRIDE_GFX_VERSION=10.3.0 unsupported compatibility workaround")
            with patch("backend.probe_runtime", return_value={}), patch("backend.subprocess.Popen") as launch, self.assertRaisesRegex(ValueError, "runtime changed"):
                self.trainer.train(attempt)
            launch.assert_not_called()

    def test_valid_mock_output_nearest_flatten_offline_argv_and_provenance(self):
        self.authorize_fixture()
        attempt = self.attempt()
        with patch("backend.probe_runtime", return_value=self.facts) as probe, patch(
                "backend.subprocess.Popen", side_effect=self.output_process) as launch, patch("backend.os.killpg"):
            checkpoint = self.trainer.train(attempt)
        probe.assert_called_once_with(True)
        self.assertGreater(checkpoint.stat().st_size, 0)
        argv = launch.call_args.args[0]
        self.assertIn(str(self.script), argv)  # Shell metacharacters remain one literal argument.
        self.assertFalse(launch.call_args.kwargs["shell"])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        env = launch.call_args.kwargs["env"]
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", env)
        with pillow().open(attempt.directory / "imagefolder" / "000000.png") as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (1, 2, 3))
            self.assertEqual(image.size, (128, 128))
            for x in (2, 3):
                for y in (2, 3):
                    self.assertEqual(image.getpixel((x, y)), (20, 30, 40))
        record = json.loads((attempt.directory / "backend.json").read_text())
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["checkpoint_sha256"], fingerprint(checkpoint))
        self.assertTrue((attempt.directory / "trainer.log").exists())

    def test_timeout_kills_entire_process_group_and_keeps_failure_evidence(self):
        self.authorize_fixture()
        attempt = self.attempt()
        process = Mock(pid=12345, wait=Mock(side_effect=[subprocess.TimeoutExpired("mock", 10), -9]))
        with patch("backend.probe_runtime", return_value=self.facts), patch(
                "backend.subprocess.Popen", return_value=process), patch("backend.os.killpg") as kill, self.assertRaises(subprocess.TimeoutExpired):
            self.trainer.train(attempt)
        kill.assert_called_once_with(12345, signal.SIGKILL)
        process.wait.assert_any_call(timeout=10)
        self.assertEqual(json.loads((attempt.directory / "backend.json").read_text())["status"], "failed")

    def test_cleanup_wait_is_bounded_when_killed_process_does_not_exit(self):
        self.authorize_fixture()
        attempt = self.attempt()
        original = subprocess.TimeoutExpired("mock training", 10)
        process = Mock(pid=12345, wait=Mock(side_effect=[original, subprocess.TimeoutExpired("mock cleanup", 5)]))
        with patch("backend.probe_runtime", return_value=self.facts), patch(
                "backend.subprocess.Popen", return_value=process), patch("backend.os.killpg"), self.assertRaises(
                subprocess.TimeoutExpired) as raised:
            self.trainer.train(attempt)
        self.assertIs(raised.exception, original)
        process.wait.assert_any_call(timeout=5)
        record = json.loads((attempt.directory / "backend.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["cleanup_pending"])

    def test_failed_exit_and_missing_empty_escaped_output_are_not_candidates(self):
        self.authorize_fixture()
        attempt = self.attempt()
        for case in ("exit", "missing", "empty", "escape"):
            def launch(argv, **kwargs):
                if case in ("empty", "escape"):
                    output = attempt.directory / "output"
                    output.mkdir()
                    weights = output / "pytorch_lora_weights.safetensors"
                    if case == "empty":
                        weights.write_bytes(b"")
                    else:
                        weights.symlink_to(self.model / "weights.fake")
                return Mock(pid=12345, wait=Mock(return_value=1 if case == "exit" else 0))
            with self.subTest(case=case), patch("backend.probe_runtime", return_value=self.facts), patch(
                    "backend.subprocess.Popen", side_effect=launch), patch("backend.os.killpg"), self.assertRaises(ValueError):
                self.trainer.train(attempt)
            import shutil
            for child in attempt.directory.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    def test_attempt_bounds_and_core_integration_never_promote_without_reviewer(self):
        self.authorize_fixture()
        bad = self.attempt(2)
        with patch("backend.probe_runtime") as probe, self.assertRaises(ValueError):
            self.trainer.train(bad)
        probe.assert_not_called()
        bad.directory.rmdir()
        self.run.state.update(status="needs_human", reason="setup_approval", attempts=0)
        self.run._save()
        self.run.approve_setup("synthetic harness setup only")
        reviewer = Mock()
        reviewer.evaluate.side_effect = RuntimeError("independent real visual evaluator remains missing")
        with patch("backend.probe_runtime", return_value=self.facts), patch(
                "backend.subprocess.Popen", side_effect=self.output_process) as launch, patch("backend.os.killpg"):
            state = self.run.execute(self.trainer, reviewer)
        launch.assert_called_once()
        self.assertEqual(state["status"], "failed")
        self.assertIsNone(state["incumbent"])
        self.assertTrue((self.run.directory / "attempt-001" / "output" / "pytorch_lora_weights.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
