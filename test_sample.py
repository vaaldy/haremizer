import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import sample

SUITE = Path(__file__).with_name("evaluation-suite.json")


class EulerDiscreteScheduler:
    def __init__(self):
        self.config = {"mock_config": True}


class EulerAncestralDiscreteScheduler:
    pass


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.index = {"_class_name": "StableDiffusionPipeline", "_diffusers_version": "test",
                      "unet": ["diffusers", "UNet2DConditionModel"],
                      "text_encoder": ["transformers", "CLIPTextModel"],
                      "safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"],
                      "requires_safety_checker": True, "feature_extractor": [None, None]}
        self.write_index()
        # Fake bytes only: never loaded by ML libraries.
        (self.model / "weights.safetensors").write_bytes(b"mock weights")
        self.lora = self.root / "adapter.safetensors"
        self.lora.write_bytes(b"mock adapter, not a real checkpoint")
        self.lora_sha256 = hashlib.sha256(self.lora.read_bytes()).hexdigest()
        self.output = self.root / "output"
        self.pipeline = MagicMock()
        self.pipeline.scheduler = EulerDiscreteScheduler()
        self.scheduler_factory = MagicMock()
        self.scheduler_factory.from_config.return_value = EulerAncestralDiscreteScheduler()
        self.pipeline.side_effect = lambda **kw: SimpleNamespace(
            images=[Image.new("RGB", (kw["width"], kw["height"]), "red")])
        self.loader = MagicMock()
        self.loader.from_pretrained.return_value = self.pipeline
        self.torch = MagicMock()
        self.torch.__version__ = "mock-torch"
        self.torch.version = SimpleNamespace(hip="mock-hip", cuda=None)
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.get_device_name.return_value = "mock GPU"
        self.torch.cuda.max_memory_allocated.return_value = 100
        self.torch.cuda.max_memory_reserved.return_value = 200
        self.torch.inference_mode.side_effect = contextlib.nullcontext
        self.torch.Generator.side_effect = lambda **kw: SimpleNamespace(
            manual_seed=lambda seed: (kw["device"], seed))
        self.diffusers = SimpleNamespace(__version__="mock-diffusers",
            StableDiffusionPipeline=self.loader, StableDiffusionXLPipeline=self.loader,
            EulerAncestralDiscreteScheduler=self.scheduler_factory)

    def write_index(self):
        (self.model / "model_index.json").write_text(json.dumps(self.index))

    def run_sample(self, **kwargs):
        with patch.dict(sys.modules, {"torch": self.torch, "diffusers": self.diffusers}):
            return sample.sample(self.model, self.output, **kwargs)

    def test_offline_loading_fixed_seed_and_actual_png_manifest(self):
        with patch.dict(os.environ, {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}):
            before = dict(os.environ)
            report = self.run_sample(prompt="../../do not use as filename", seed=9,
                                     width=768, height=1024, variant="fp16",
                                     model_id="owner/model", revision="a" * 40)
            self.assertEqual(dict(os.environ), before)
        self.loader.from_pretrained.assert_called_once_with(
            str(self.model), local_files_only=True, use_safetensors=True,
            torch_dtype=self.torch.float16, variant="fp16")
        self.pipeline.to.assert_called_once_with("cuda")
        self.pipeline.enable_model_cpu_offload.assert_not_called()
        self.pipeline.load_lora_weights.assert_not_called()
        self.assertIsNone(report["lora"])
        self.scheduler_factory.from_config.assert_not_called()
        self.assertEqual(report["settings"]["scheduler_class"], "EulerDiscreteScheduler")
        self.assertIsNone(report["settings"]["scheduler_override"])
        self.assertNotIn("scheduler", self.loader.from_pretrained.call_args.kwargs)
        self.assertNotIn("scheduler", self.pipeline.call_args.kwargs)
        self.assertEqual(self.pipeline.call_args.kwargs["generator"], ("cuda", 9))
        self.assertEqual(self.pipeline.call_args.kwargs["num_inference_steps"], 20)
        self.assertIsNone(self.pipeline.call_args.kwargs["negative_prompt"])
        self.assertEqual(self.pipeline.call_args.kwargs["guidance_scale"], 7.5)
        self.assertIsNone(report["settings"]["negative_prompt"])
        self.assertEqual(report["settings"]["guidance_scale"], 7.5)
        self.assertEqual(set(p.name for p in self.output.iterdir()), {"case-01.png", "manifest.json"})
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), report)
        record = report["records"][0]
        with Image.open(self.output / record["file"]) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (768, 1024))
            self.assertEqual(image.mode, "RGB")
        self.assertEqual(record["sha256"], hashlib.sha256((self.output / record["file"]).read_bytes()).hexdigest())
        self.assertGreaterEqual(record["duration_seconds"], 0)
        self.assertEqual(record["gpu_memory"]["peak_allocated_bytes"], 100)
        self.assertEqual(record["gpu_memory"]["peak_reserved_bytes"], 200)
        self.assertIn("not total GPU usage", record["gpu_memory"]["scope"])
        self.assertFalse(report["visual_quality_assessed"])
        self.assertFalse(report["sprite_format_validated"])
        self.assertFalse(report["training_started"])
        self.assertIn("not final 48/64", report["output_contract"])
        self.assertEqual(report["runtime"]["HSA_OVERRIDE_GFX_VERSION"], "10.3.0")
        self.assertEqual(report["model"]["id"], "owner/model")
        self.assertEqual(report["model"]["revision"], "a" * 40)
        self.assertEqual(report["model"]["files"]["weights.safetensors"], {"bytes": 12})
        self.assertIn("not weight-content attestation", report["model"]["identity_scope"])

    def test_sdxl_explicit_offload_preserves_frozen_suite_seeds(self):
        self.index.update(_class_name="StableDiffusionXLPipeline", force_zeros_for_empty_prompt=True)
        self.write_index()
        report = self.run_sample(suite=SUITE, max_cases=2, steps=1, cpu_offload=True,
                                 negative_prompt="  blurry, text  ", guidance_scale=15)
        self.pipeline.to.assert_not_called()
        self.pipeline.enable_model_cpu_offload.assert_called_once_with(gpu_id=0, device="cuda")
        self.assertNotIn("variant", self.loader.from_pretrained.call_args.kwargs)
        self.assertEqual([c.kwargs["generator"] for c in self.pipeline.call_args_list],
                         [("cuda", 1701), ("cuda", 2939)])
        self.assertEqual([r["width"] for r in report["records"]], [768, 768])
        self.assertEqual([r["height"] for r in report["records"]], [768, 768])
        self.assertEqual(report["settings"]["width"], 768)
        self.assertEqual(report["settings"]["height"], 768)
        self.assertEqual(report["suite_sha256"], hashlib.sha256(SUITE.read_bytes()).hexdigest())
        self.assertEqual(report["suite"]["frozen_json"], SUITE.read_text())
        cases = json.loads(SUITE.read_text())["cases"][:2]
        self.assertEqual([c.kwargs["prompt"] for c in self.pipeline.call_args_list],
                         [c["prompt"] for c in cases])
        for call in self.pipeline.call_args_list:
            self.assertEqual(call.kwargs["negative_prompt"], "  blurry, text  ")
            self.assertEqual(call.kwargs["guidance_scale"], 15)
        self.assertEqual(report["settings"]["negative_prompt"], "  blurry, text  ")
        self.assertEqual(report["settings"]["guidance_scale"], 15)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), report)
        self.assertEqual(report["model"]["class"], "StableDiffusionXLPipeline")

    def test_lora_local_loading_precedes_placement_and_records_identity(self):
        for offload in (False, True):
            with self.subTest(cpu_offload=offload):
                self.pipeline.reset_mock()
                self.index["_class_name"] = "StableDiffusionXLPipeline" if offload else "StableDiffusionPipeline"
                self.write_index()
                self.output = self.root / f"output-{offload}"
                report = self.run_sample(prompt="test", lora=self.lora,
                                         lora_sha256=self.lora_sha256, cpu_offload=offload)
                self.pipeline.load_lora_weights.assert_called_once_with(
                    str(self.lora.parent), weight_name=self.lora.name,
                    local_files_only=True, use_safetensors=True)
                methods = [call[0] for call in self.pipeline.mock_calls]
                placement = "enable_model_cpu_offload" if offload else "to"
                self.assertLess(methods.index("load_lora_weights"), methods.index(placement))
                self.assertEqual(report["lora"], {"path": str(self.lora),
                    "bytes": self.lora.stat().st_size, "sha256": self.lora_sha256})
                self.assertEqual(json.loads((self.output / "manifest.json").read_text()), report)
                self.assertFalse(report["training_started"])
                self.assertFalse(report["visual_quality_assessed"])
                self.assertFalse(report["sprite_format_validated"])
                self.assertEqual(report["settings"]["dtype"], "float16")

    def test_scheduler_override_uses_loaded_config_before_placement(self):
        for offload in (False, True):
            with self.subTest(cpu_offload=offload):
                self.output = self.root / f"scheduler-output-{offload}"
                original = EulerDiscreteScheduler()
                self.pipeline.scheduler = original
                order = MagicMock()
                order.attach_mock(self.loader.from_pretrained, "load")
                order.attach_mock(self.scheduler_factory.from_config, "scheduler")
                placement = (self.pipeline.enable_model_cpu_offload if offload else self.pipeline.to)
                order.attach_mock(placement, "place")
                placement.side_effect = lambda *args, **kwargs: self.assertIs(
                    self.pipeline.scheduler, self.scheduler_factory.from_config.return_value)
                report = self.run_sample(prompt="test", scheduler="euler_ancestral",
                                         cpu_offload=offload)
                self.assertIs(self.scheduler_factory.from_config.call_args.args[0], original.config)
                self.scheduler_factory.from_config.assert_called_with(original.config)
                self.assertEqual([call[0] for call in order.mock_calls], ["load", "scheduler", "place", "load()"])
                self.loader.from_pretrained.assert_called_with(
                    str(self.model), local_files_only=True, use_safetensors=True,
                    torch_dtype=self.torch.float16)
                self.assertEqual(report["settings"]["scheduler_class"], "EulerAncestralDiscreteScheduler")
                self.assertEqual(report["settings"]["scheduler_override"], "euler_ancestral")
                self.assertEqual(json.loads((self.output / "manifest.json").read_text()), report)
                self.assertNotIn("scheduler", self.pipeline.call_args.kwargs)
                self.assertFalse(report["visual_quality_assessed"])
                self.assertFalse(report["training_started"])

    def test_invalid_scheduler_fails_before_ml_import(self):
        with patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            for value in ("", "EulerAncestralDiscreteScheduler", "euler", True, 1, [], {}):
                with self.subTest(scheduler=value), self.assertRaisesRegex(ValueError, "scheduler"):
                    sample.sample(self.model, self.output, prompt="test", scheduler=value)
        self.assertFalse(self.output.exists())
        self.loader.from_pretrained.assert_not_called()

    def test_scheduler_cli_choice_and_default(self):
        args = ["sample.py", "--model", str(self.model), "--output", str(self.output),
                "--prompt", "test"]
        for extra, expected in (([], None), (["--scheduler", "euler_ancestral"], "euler_ancestral")):
            with patch.object(sys, "argv", args + extra), patch.object(sample, "sample", return_value={}) as run:
                with patch("builtins.print"):
                    sample.main()
            self.assertEqual(run.call_args.kwargs["scheduler"], expected)
        with patch.object(sys, "argv", args + ["--scheduler", "euler"]), patch.object(sample, "sample") as run:
            with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                sample.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_invalid_lora_fails_before_ml_import(self):
        empty = self.root / "empty.safetensors"
        empty.touch()
        pickle = self.root / "adapter.bin"
        pickle.write_bytes(self.lora.read_bytes())
        oversized = self.root / "oversized.safetensors"
        with oversized.open("wb") as file:
            file.truncate(sample.MAX_LORA_BYTES + 1)
        invalid = [{"lora": self.lora}, {"lora_sha256": self.lora_sha256}] + [
            {"lora": self.lora, "lora_sha256": value}
            for value in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 123, "0" * 64)] + [
            {"lora": path, "lora_sha256": self.lora_sha256}
            for path in (empty, pickle, oversized, self.root, self.root / "missing.safetensors",
                         "https://example.invalid/adapter.safetensors")]
        with patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    sample.sample(self.model, self.output, prompt="test", **options)
        self.assertFalse(self.output.exists())
        self.loader.from_pretrained.assert_not_called()

    def test_lora_source_cannot_be_inside_output_even_via_symlink(self):
        source_directory = self.root / "source"
        source_directory.mkdir()
        source = source_directory / "adapter.safetensors"
        source.write_bytes(self.lora.read_bytes())
        link = self.root / "linked.safetensors"
        link.symlink_to(source)
        with patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            for output in (source_directory, source, self.model / "new-output"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    sample.sample(self.model, output, prompt="test", lora=link,
                                  lora_sha256=self.lora_sha256)
        self.assertEqual(source.read_bytes(), self.lora.read_bytes())
        self.loader.from_pretrained.assert_not_called()

    def test_lora_hash_rechecked_after_base_load(self):
        def mutate(*args, **kwargs):
            self.lora.write_bytes(b"changed mock adapter")
            return self.pipeline
        self.loader.from_pretrained.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "match lora_sha256"):
            self.run_sample(prompt="test", lora=self.lora, lora_sha256=self.lora_sha256)
        self.pipeline.load_lora_weights.assert_not_called()
        self.pipeline.to.assert_not_called()
        self.pipeline.enable_model_cpu_offload.assert_not_called()

    def test_four_case_and_step_upper_bounds(self):
        report = self.run_sample(suite=SUITE, max_cases=4, steps=40, width=64, height=64,
                                 guidance_scale=1)
        self.assertEqual(report["settings"]["guidance_scale"], 1)
        self.assertTrue(all(c.kwargs["guidance_scale"] == 1 for c in self.pipeline.call_args_list))
        self.assertEqual(len(report["records"]), 4)
        self.assertTrue(all(c.kwargs["num_inference_steps"] == 40 for c in self.pipeline.call_args_list))

    def test_invalid_bounds_and_inputs_fail_before_import(self):
        invalid = [{"max_cases": v} for v in (0, 5, True)] + [
            {"steps": v} for v in (0, 41, True)] + [
            {"negative_prompt": v} for v in ("", " \t\n", 1, True, [], {})] + [
            {"guidance_scale": v} for v in (0.99, 15.01, True, False, float("nan"),
                                           float("inf"), -float("inf"), "7.5", None,
                                           10**400)] + [
            {"width": v} for v in (0, 48, 65, 1088, True)] + [
            {"height": 63}, {"device": "cpu"}, {"variant": "bf16"},
            {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"prompt": " "},
            {"revision": "main"}, {"revision": "A" * 40}, {"cpu_offload": 1},
            {"suite": SUITE}]
        with patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    sample.sample(self.model, self.output, **{"prompt": "test", **options})
            with self.assertRaises(ValueError):
                sample.sample(self.model, self.output)
        self.assertFalse(self.output.exists())

    def test_class_component_and_python_safety(self):
        invalid = [{"_class_name": "DiffusionPipeline"}, {"_class_name": ["evil", "Pipeline"]},
                   {"unet": ["remote_module", "UNet2DConditionModel"]},
                   {"text_encoder": ["transformers", "CustomModel"]},
                   {"_module": "evil"}, {"unet": [None, "CustomModel"]},
                   {"unet": [[], []]}, {"requires_safety_checker": "false"}]
        original = dict(self.index)
        for changes in invalid:
            self.index = {**original, **changes}
            self.write_index()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_sample(prompt="test")
        self.loader.from_pretrained.assert_not_called()
        self.index = original
        self.write_index()
        (self.model / "custom.py").write_text("raise RuntimeError('never execute')")
        with self.assertRaisesRegex(ValueError, "Python code"):
            self.run_sample(prompt="test")

    def test_pickle_only_model_cannot_load_or_convert(self):
        (self.model / "weights.safetensors").unlink()
        (self.model / "pytorch_model.bin").write_bytes(b"not actually pickle")
        with self.assertRaisesRegex(ValueError, "pickle checkpoints"):
            self.run_sample(prompt="test")
        (self.model / "pytorch_model.bin").unlink()
        with self.assertRaisesRegex(ValueError, "safetensors"):
            self.run_sample(prompt="test")
        (self.model / "weights.safetensors").write_bytes(b"mock weights")
        (self.model / "pytorch_model.bin").write_bytes(b"fallback forbidden even alongside safe weights")
        with self.assertRaisesRegex(ValueError, "component fallback"):
            self.run_sample(prompt="test")
        self.loader.from_pretrained.assert_not_called()

    def test_no_gpu_means_no_load_or_cpu_fallback(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            self.run_sample(prompt="test", cpu_offload=True)
        self.loader.from_pretrained.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_output_must_be_new_and_outside_model(self):
        for path in (self.model / "new-output", self.model, SUITE, self.root):
            self.output = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_sample(prompt="test")
        self.loader.from_pretrained.assert_not_called()

    def test_memory_unavailable_is_not_claimed_as_zero(self):
        self.torch.cuda.reset_peak_memory_stats.side_effect = RuntimeError("stats unavailable")
        report = self.run_sample(prompt="test", width=64, height=64)
        self.assertEqual(report["records"][0]["gpu_memory"]["error"], "stats unavailable")
        self.assertNotIn("peak_allocated_bytes", report["records"][0]["gpu_memory"])
        self.torch.cuda.max_memory_allocated.assert_not_called()

    def test_unexpected_dimensions_are_not_recorded_as_requested(self):
        self.pipeline.side_effect = None
        self.pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (1, 1))])
        with self.assertRaisesRegex(RuntimeError, "unexpected image dimensions"):
            self.run_sample(prompt="test")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_forwards_sampling_controls(self):
        args = ["sample.py", "--model", str(self.model), "--output", str(self.output),
                "--prompt", "test", "--negative-prompt", "blurry", "--guidance-scale", "6.5",
                "--lora", str(self.lora), "--lora-sha256", self.lora_sha256]
        with patch.object(sys, "argv", args), patch.object(sample, "sample", return_value={}) as run:
            with patch("builtins.print"):
                sample.main()
        self.assertEqual(run.call_args.kwargs["negative_prompt"], "blurry")
        self.assertEqual(run.call_args.kwargs["guidance_scale"], 6.5)
        self.assertEqual(run.call_args.kwargs["lora"], self.lora)
        self.assertEqual(run.call_args.kwargs["lora_sha256"], self.lora_sha256)

    def test_import_and_cli_help_do_not_import_ml_or_generate(self):
        code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'diffusers'):
        raise AssertionError('ML import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_path('sample.py', run_name='import_check')
sys.argv = ['sample.py', '--help']
runpy.run_path('sample.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scheduler", result.stdout)
        self.assertIn("--cpu-offload", result.stdout)
        self.assertIn("--suite", result.stdout)
        self.assertIn("--negative-prompt", result.stdout)
        self.assertIn("--guidance-scale", result.stdout)
        self.assertIn("--lora", result.stdout)
        self.assertIn("--lora-sha256", result.stdout)


if __name__ == "__main__":
    unittest.main()
