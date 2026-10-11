"""SDXL smoke preflight rejects invalid declarations before the trainer boundary."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import smoke_sdxl as smoke


class SmokePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = "a" * 40
        self.model = self.root / "models--owner--model" / "snapshots" / self.revision
        self.model.mkdir(parents=True)
        index = {"_class_name": "StableDiffusionXLPipeline",
                 "unet": ["diffusers", "UNet2DConditionModel"],
                 "vae": ["diffusers", "AutoencoderKL"],
                 "text_encoder": ["transformers", "CLIPTextModel"],
                 "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
                 "tokenizer": ["transformers", "CLIPTokenizer"],
                 "tokenizer_2": ["transformers", "CLIPTokenizer"],
                 "scheduler": ["diffusers", "EulerDiscreteScheduler"]}
        (self.model / "model_index.json").write_text(json.dumps(index))
        # Deliberately not a real checkpoint; mocked training only, no GPU access.
        (self.model / "fixture.safetensors").write_bytes(b"test-only fake weight bytes")
        self.image = self.root / "image.png"
        Image.new("RGBA", (64, 64), (10, 20, 30, 127)).save(self.image)
        self.page = self.root / "source.html"
        self.page.write_text("test-only source/license fixture")
        self.request_path = self.root / "request.json"
        self.output = self.root / "output"
        self.request = {
            "model": {"path": str(self.model.relative_to(self.root)), "id": "owner/model",
                      "revision": self.revision},
            "config": {**smoke.CONFIG, "steps": 2},
            "reviewed_by": "machine:sol", "declared_smoke_only": True,
            "authorization": {"user_authorized": True,
                              "record": "Test fixture recording broad local user authorization; not human review"},
            "examples": [{"path": "image.png", "sha256": smoke.file_hash(self.image),
                          "caption": "Test-only caption", "source_url": "https://example.org/art",
                          "license": "CC-BY-4.0", "credit": "Test artist",
                          "source_page": {"path": str(self.page), "sha256": smoke.file_hash(self.page)}}]}
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def write(self, request=None):
        self.request_path.write_text(json.dumps(self.request if request is None else request))

    def test_valid_local_manifest_no_ml_import(self):
        self.write()
        import builtins
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft", "accelerate"}:
                raise AssertionError("preflight imported ML")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard):
            report = smoke.preflight(self.request_path, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(report["request"]["reviewed_by"], "machine:sol")
        self.assertEqual(report["inputs"][0]["path"], str(self.image))
        self.assertIn("fixture.safetensors", report["model"]["file_sha256"])
        self.assertFalse(report["quality_assessed"])

    def test_invalid_requests_never_reach_trainer(self):
        cases = []

        def case(label, mutate):
            request = copy.deepcopy(self.request)
            mutate(request)
            cases.append((label, request))

        for key, value in (("steps", 0), ("steps", 3), ("steps", True),
                           ("batch_size", 2), ("resolution", 1024), ("rank", 8),
                           ("seed", 1), ("learning_rate", float("nan")), ("gradient_clip", 2.0)):
            case(f"config {key}={value}", lambda r, k=key, v=value: r["config"].update({k: v}))
        for key, value in (("caption", " "), ("license", "unknown"), ("credit", ""),
                           ("source_url", "file:///source"), ("sha256", "0" * 64)):
            case(key, lambda r, k=key, v=value: r["examples"][0].update({k: v}))
        case("source hash", lambda r: r["examples"][0]["source_page"].update(sha256="0" * 64))
        case("source missing", lambda r: r["examples"][0]["source_page"].update(path="missing.html"))
        case("empty examples", lambda r: r.update(examples=[]))
        case("too many examples", lambda r: r.update(examples=r["examples"] * 3))
        case("duplicate examples", lambda r: r.update(examples=r["examples"] * 2))
        case("unknown key", lambda r: r.update(retries=2))
        case("unknown config", lambda r: r["config"].update(cpu_fallback=True))
        case("no authorization", lambda r: r["authorization"].update(user_authorized=False))
        case("blank authorization", lambda r: r["authorization"].update(record=""))
        case("invented human review", lambda r: r.update(reviewed_by="human:sol"))
        case("not smoke", lambda r: r.update(declared_smoke_only=False))
        case("unpinned model", lambda r: r["model"].update(revision="main"))
        case("cache id mismatch", lambda r: r["model"].update(id="other/model"))
        case("cache revision mismatch", lambda r: r["model"].update(revision="b" * 40))
        with patch.object(smoke, "train") as trainer:
            for label, request in cases:
                with self.subTest(label=label):
                    self.write(request)
                    with self.assertRaises((ValueError, OSError)):
                        smoke.run(self.request_path, self.output)
                    self.assertFalse(self.output.exists())
            trainer.assert_not_called()

    def test_existing_output_and_pickle_model_rejected(self):
        self.write()
        self.output.mkdir()
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.output)
        self.output.rmdir()
        (self.model / "unsafe.bin").write_bytes(b"unsafe")
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.output)

    def test_content_revision(self):
        hashes = {p.name: smoke.file_hash(p) for p in self.model.iterdir()}
        self.request["model"]["revision"] = smoke.canonical_digest(hashes)
        self.write()
        smoke.preflight(self.request_path, self.output)
        (self.model / "fixture.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "content revision mismatch"):
            smoke.preflight(self.request_path, self.output)

    def test_native_size_required(self):
        Image.new("RGBA", (48, 48)).save(self.image)
        self.request["examples"][0]["sha256"] = smoke.file_hash(self.image)
        self.write()
        with self.assertRaisesRegex(ValueError, "64x64"):
            smoke.preflight(self.request_path, self.output)

    def test_mocked_failure_outcome_no_retry(self):
        self.write()
        with patch.object(smoke.importlib.metadata, "version", side_effect=smoke.VERSIONS.__getitem__), \
                patch.object(smoke.sys, "version_info", (3, 12)), \
                patch.object(smoke, "train", side_effect=RuntimeError("test OOM")) as trainer:
            with self.assertRaisesRegex(RuntimeError, "test OOM"):
                smoke.run(self.request_path, self.output)
        trainer.assert_called_once()
        outcome = json.loads((self.output / "outcome.json").read_text())
        self.assertEqual(outcome["status"], "failed")
        self.assertNotIn("checkpoint", outcome)
        self.assertFalse(outcome["quality_assessed"])

    def test_optional_management_metric_missing_does_not_abort(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        query = Mock(side_effect=ModuleNotFoundError("amdsmi missing"))
        torch = SimpleNamespace(cuda=SimpleNamespace(device_memory_used=query))
        result = smoke.device_usage(torch)
        self.assertIsNone(result["device_used_bytes"])
        self.assertIn("amdsmi missing", result["device_usage_error"])
        query.assert_called_once_with(0)
        query.side_effect = None
        query.return_value = 123
        self.assertEqual(smoke.device_usage(torch), {"device_used_bytes": 123})

    def test_training_progress_is_durable_before_failure(self):
        self.write()

        def trainer(report, output, persist):
            report.update(training_started=True, phase="training", steps=[])
            persist()
            snapshot = json.loads((output / "outcome.json").read_text())
            self.assertTrue(snapshot["training_started"])
            self.assertEqual(snapshot["phase"], "training")
            report["steps"].append({"optimizer_step": 1})
            persist()
            snapshot = json.loads((output / "outcome.json").read_text())
            self.assertEqual(snapshot["steps"], [{"optimizer_step": 1}])
            self.assertFalse((output / "outcome.json.tmp").exists())
            raise RuntimeError("interrupted after first step")

        with patch.object(smoke.importlib.metadata, "version", side_effect=smoke.VERSIONS.__getitem__), \
                patch.object(smoke.sys, "version_info", (3, 12)), \
                patch.object(smoke, "train", side_effect=trainer):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                smoke.run(self.request_path, self.output)
        outcome = json.loads((self.output / "outcome.json").read_text())
        self.assertEqual(outcome["status"], "failed")
        self.assertTrue(outcome["training_started"])
        self.assertEqual(len(outcome["steps"]), 1)

    def test_pinned_versions_block_before_training(self):
        self.write()
        with patch.object(smoke.importlib.metadata, "version", return_value="wrong"), \
                patch.object(smoke, "train") as trainer:
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                smoke.run(self.request_path, self.output)
            trainer.assert_not_called()
        self.assertFalse(json.loads((self.output / "outcome.json").read_text())["training_started"])

    @unittest.skipUnless(os.environ.get("SDXL_SMOKE_CPU_API_TEST") == "1", "optional pinned-runtime CPU API test")
    def test_real_diffusers_save_delete_reload_cpu(self):
        import torch
        from diffusers import UNet2DConditionModel
        from peft import LoraConfig
        unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                                   layers_per_block=1, block_out_channels=(32, 32),
                                   down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                   up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                   cross_attention_dim=32, attention_head_dim=4).half()
        unet.requires_grad_(False)
        unet.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=smoke.TARGETS))
        with torch.no_grad():
            for name, param in unet.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()
                    if ".lora_B." in name:
                        param.fill_(0.001234567)
        frozen = {k: p.detach().clone() for k, p in unet.named_parameters() if not p.requires_grad}
        self.output.mkdir()
        expected = smoke.save_adapter(unet, self.output)
        smoke.reload_adapter(unet, self.output, expected)
        actual = {k: p for k, p in unet.named_parameters() if not p.requires_grad}
        self.assertEqual(set(frozen), set(actual))
        self.assertTrue(all(torch.equal(v, actual[k]) for k, v in frozen.items()))
        self.assertTrue((self.output / "pytorch_lora_weights.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
