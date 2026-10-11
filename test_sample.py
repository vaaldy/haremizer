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
        self.output = self.root / "output"
        self.pipeline = MagicMock()
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
            StableDiffusionPipeline=self.loader, StableDiffusionXLPipeline=self.loader)

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
        self.assertEqual(self.pipeline.call_args.kwargs["generator"], ("cuda", 9))
        self.assertEqual(self.pipeline.call_args.kwargs["num_inference_steps"], 20)
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
        report = self.run_sample(suite=SUITE, max_cases=2, steps=1, cpu_offload=True)
        self.pipeline.to.assert_not_called()
        self.pipeline.enable_model_cpu_offload.assert_called_once_with(gpu_id=0, device="cuda")
        self.assertNotIn("variant", self.loader.from_pretrained.call_args.kwargs)
        self.assertEqual([c.kwargs["generator"] for c in self.pipeline.call_args_list],
                         [("cuda", 1701), ("cuda", 2939)])
        self.assertEqual([r["width"] for r in report["records"]], [1024, 1024])
        self.assertEqual(report["suite_sha256"], hashlib.sha256(SUITE.read_bytes()).hexdigest())
        self.assertEqual(report["suite"]["frozen_json"], SUITE.read_text())
        self.assertEqual(report["model"]["class"], "StableDiffusionXLPipeline")

    def test_four_case_and_step_upper_bounds(self):
        report = self.run_sample(suite=SUITE, max_cases=4, steps=40, width=64, height=64)
        self.assertEqual(len(report["records"]), 4)
        self.assertTrue(all(c.kwargs["num_inference_steps"] == 40 for c in self.pipeline.call_args_list))

    def test_invalid_bounds_and_inputs_fail_before_import(self):
        invalid = [{"max_cases": v} for v in (0, 5, True)] + [
            {"steps": v} for v in (0, 41, True)] + [
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
        self.assertIn("--cpu-offload", result.stdout)
        self.assertIn("--suite", result.stdout)


if __name__ == "__main__":
    unittest.main()
