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

    def pilot_fixture(self, train_count=4, heldout_count=2):
        request = copy.deepcopy(self.request)
        request.update(declared_smoke_only=False, heldout_examples=[])
        request["config"]["steps"] = 100
        request["examples"] = []
        for index in range(train_count + heldout_count):
            image = self.root / f"pilot-{index}.png"
            png = Image.new("RGBA", (64, 64), (index + 1, 20, 30, 255))
            png.putpixel((0, 0), (10, 20, 30, 0))
            # Distinct non-semantic geometry, not a collection of recolored blocks.
            png.putpixel((index + 1, 1), (10, 20, 30, 0))
            png.save(image)
            row = {**copy.deepcopy(self.request["examples"][0]), "path": str(image),
                   "sha256": smoke.file_hash(image), "identity_group": f"identity-{index}"}
            request["examples" if index < train_count else "heldout_examples"].append(row)
        return request

    def test_pilot_selector_requires_explicit_boolean(self):
        self.write(self.pilot_fixture())
        for value in (None, 1, 0, "false", "true"):
            with self.subTest(pilot=value), self.assertRaisesRegex(ValueError, "explicit boolean"):
                smoke.preflight(self.request_path, self.output, pilot=value)
        self.assertFalse(self.output.exists())

    def test_pilot_preflight_no_ml_and_train_only_inputs(self):
        import builtins
        request = self.pilot_fixture()
        self.write(request)
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft", "accelerate"}:
                raise AssertionError("preflight imported ML")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard):
            report = smoke.preflight(self.request_path, self.output, pilot=True)
        self.assertEqual(len(report["inputs"]), 4)
        self.assertEqual(report["heldout_count"], 2)
        self.assertEqual(len({r["pixel_sha256"] for r in report["inputs"] + report["heldout_inputs"]}), 6)

        def trainer(report, output, persist):
            self.assertEqual([r["sha256"] for r in report["inputs"]],
                             [r["sha256"] for r in request["examples"]])
            self.assertTrue({r["path"] for r in report["inputs"]}.isdisjoint(
                {r["path"] for r in report["heldout_inputs"]}))
            self.assertTrue(all("cached_input" not in r for r in report["heldout_inputs"]))

        with patch.object(smoke.importlib.metadata, "version", side_effect=smoke.VERSIONS.__getitem__), \
                patch.object(smoke.sys, "version_info", (3, 12)), \
                patch.object(smoke, "train", side_effect=trainer) as mock:
            result = smoke.run(self.request_path, self.output, pilot=True)
        mock.assert_called_once()
        self.assertEqual(result["status"], "pilot_completed_unreviewed")
        for key in ("quality_assessed", "promotion", "synthetic_feedback"):
            self.assertIs(result[key], False)

    def test_pilot_caps_groups_and_contracts_before_trainer(self):
        cases = []
        base = self.pilot_fixture()

        def case(label, mutate):
            request = copy.deepcopy(base)
            mutate(request)
            cases.append((label, request))

        for steps in (2, 99, 101, True):
            case(f"steps {steps}", lambda r, n=steps: r["config"].update(steps=n))
        for split, minimum in (("examples", 4), ("heldout_examples", 2)):
            case(f"{split} too few", lambda r, k=split, n=minimum: r.update({k: r[k][:n-1]}))
            case(f"{split} too many", lambda r, k=split: r.update({k: r[k] * 17}))
            case(f"{split} blank group", lambda r, k=split: r[k][0].update(identity_group=" "))
            case(f"{split} too few groups", lambda r, k=split: r[k][0].update(
                identity_group=r[k][1]["identity_group"]))
            for key, value in (("sha256", "0" * 64), ("license", "unknown"), ("caption", " ")):
                case(f"{split} {key}", lambda r, k=split, a=key, v=value: r[k][0].update({a: v}))
            case(f"{split} source hash", lambda r, k=split: r[k][0]["source_page"].update(sha256="0" * 64))
        case("intersection", lambda r: r["heldout_examples"][0].update(
            identity_group=r["examples"][0]["identity_group"]))
        case("byte duplicate", lambda r: r["heldout_examples"][0].update(
            path=r["examples"][0]["path"], sha256=r["examples"][0]["sha256"]))
        case("smoke declaration", lambda r: r.update(declared_smoke_only=True))
        case("unknown key", lambda r: r.update(resume=True))
        case("unknown example key", lambda r: r["examples"][0].update(extra=True))
        case("missing heldout", lambda r: r.pop("heldout_examples"))
        with patch.object(smoke, "train") as trainer:
            for label, request in cases:
                with self.subTest(label=label):
                    self.write(request)
                    with self.assertRaises(ValueError):
                        smoke.run(self.request_path, self.output, pilot=True)
                    self.assertFalse(self.output.exists())
            trainer.assert_not_called()
        self.write(base)
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.output)  # no implicit pilot
        self.write()
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.output, pilot=True)

    def test_pilot_canonical_duplicates_reencoding_and_hidden_rgb(self):
        for hidden_rgb in (False, True):
            with self.subTest(hidden_rgb=hidden_rgb):
                request = self.pilot_fixture()
                source = Path(request["examples"][0]["path"])
                target = Path(request["heldout_examples"][0]["path"])
                with Image.open(source) as png:
                    if hidden_rgb:
                        png.putpixel((0, 0), (99, 88, 77, 0))
                    png.save(target, compress_level=0)
                request["heldout_examples"][0]["sha256"] = smoke.file_hash(target)
                self.assertNotEqual(smoke.file_hash(source), smoke.file_hash(target))
                self.write(request)
                with self.assertRaisesRegex(ValueError, "canonical pixels"):
                    smoke.preflight(self.request_path, self.output, pilot=True)

    def test_pilot_rejects_palette_swaps_with_distinct_declared_groups(self):
        for split in ("examples", "heldout_examples"):
            with self.subTest(split=split):
                request = self.pilot_fixture()
                source = Path(request["examples"][0]["path"])
                target = Path(request[split][-1]["path"])
                with Image.open(source) as original:
                    rgba = original.convert("RGBA")
                    recolored = Image.new("RGBA", rgba.size)
                    recolored.putdata([((r + 17) % 256, (g + 29) % 256, (b + 43) % 256, a)
                                       for r, g, b, a in rgba.getdata()])
                recolored.save(target)
                request[split][-1]["sha256"] = smoke.file_hash(target)
                self.write(request)
                with self.assertRaisesRegex(ValueError, "palette recolor"):
                    smoke.preflight(self.request_path, self.output, pilot=True)
                self.assertFalse(self.output.exists())

    def test_palette_guard_declines_true_color_not_claiming_identity(self):
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
        for index in range(63):
            image.putpixel((index, 0), (index + 1, 0, 0, 255))
        self.assertIsNone(smoke.palette_pattern_digest(image.tobytes()))

    def test_pilot_maximum_split_counts_and_heldout_native_size(self):
        request = self.pilot_fixture(train_count=16, heldout_count=16)
        self.write(request)
        report = smoke.preflight(self.request_path, self.output, pilot=True)
        self.assertEqual(len(report["inputs"]), 16)
        self.assertEqual(report["heldout_count"], 16)
        heldout = request["heldout_examples"][0]
        Image.new("RGBA", (48, 48)).save(heldout["path"])
        heldout["sha256"] = smoke.file_hash(heldout["path"])
        self.write(request)
        with self.assertRaisesRegex(ValueError, "64x64"):
            smoke.preflight(self.request_path, self.output, pilot=True)

    def test_pilot_cli_forwards_flag_and_prints_actual_status(self):
        import io
        with patch.object(smoke.sys, "argv", ["smoke_sdxl.py", "--pilot", "--request",
                                              str(self.request_path), "--output", str(self.output)]), \
                patch.object(smoke, "run", return_value={"status": "pilot_completed_unreviewed"}) as run, \
                patch.object(smoke.sys, "stdout", new_callable=io.StringIO) as stdout:
            smoke.main()
            self.assertEqual(json.loads(stdout.getvalue()),
                             {"status": "pilot_completed_unreviewed", "quality_assessed": False})
        run.assert_called_once_with(self.request_path, self.output, pilot=True)

    def test_pilot_distinct_expression_pixels_same_training_group_allowed(self):
        request = self.pilot_fixture(train_count=5)
        request["examples"][-1]["identity_group"] = request["examples"][0]["identity_group"]
        self.write(request)
        report = smoke.preflight(self.request_path, self.output, pilot=True)
        self.assertEqual(len(report["inputs"]), 5)
        self.assertEqual(len({r["identity_group"] for r in report["inputs"]}), 4)

    def refine_fixture(self, continuation=False):
        request = self.pilot_fixture()
        request["config"].update(steps=50, content_resolution=384, seed=42, learning_rate=2e-5)
        request["continuation"] = None
        if continuation:
            import struct
            adapter = self.root / "parent.safetensors"
            header = {}
            offset = 0
            for part, shape in (("down", [4, 8]), ("up", [8, 4])):
                header[f"unet.down_blocks.0.attentions.0.to_q.lora.{part}.weight"] = {
                    "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + 128]}
                offset += 128
            raw = json.dumps(header).encode()
            adapter.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
            self.write(self.pilot_fixture())
            parent = smoke.preflight(self.request_path, self.output, pilot=True)
            parent.update(status="pilot_completed_unreviewed", phase="completed", training_started=True,
                          checkpoint={"path": adapter.name, "sha256": smoke.file_hash(adapter), "reload_exact": True})
            outcome = self.root / "parent-outcome.json"
            outcome.write_text(json.dumps(parent))
            request["continuation"] = {"path": str(adapter), "sha256": smoke.file_hash(adapter),
                                       "outcome": str(outcome), "outcome_sha256": smoke.file_hash(outcome)}
        return request

    def test_refine_bounds_modes_and_preimport(self):
        import builtins
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft", "safetensors"}:
                raise AssertionError("preflight imported ML")
            return original(name, *args, **kwargs)

        for parent in (False, True):
            request = self.refine_fixture(parent)
            for content in (384, 448, 512):
                request["config"]["content_resolution"] = content
                self.write(request)
                with patch("builtins.__import__", side_effect=guard):
                    report = smoke.preflight(self.request_path, self.output, refine=True)
                self.assertEqual(len(report["inputs"]), 4)
                self.assertEqual(report["heldout_count"], 2)
        for kwargs in ({"refine": 1}, {"refine": None}, {"refine": "true"},
                       {"pilot": True, "refine": True}):
            with self.assertRaises(ValueError):
                smoke.preflight(self.request_path, self.output, **kwargs)
        for key, values in (("steps", [49, 601, True]), ("seed", [-1, 2**32, True, 1.5]),
                            ("learning_rate", [0, 0.001, float("nan"), float("inf"), True]),
                            ("content_resolution", [383, 449, True, 512.0]),
                            ("rank", [8]), ("resolution", [768])):
            for value in values:
                request = self.refine_fixture()
                request["config"][key] = value
                self.write(request)
                with self.subTest(key=key, value=value), patch.object(smoke, "train") as trainer:
                    with self.assertRaises(ValueError):
                        smoke.run(self.request_path, self.output, refine=True)
                    trainer.assert_not_called()
                    self.assertFalse(self.output.exists())
        request = self.refine_fixture()
        request["config"].update(steps=600, seed=2**32 - 1, learning_rate=1e-5)
        self.write(request)
        smoke.preflight(self.request_path, self.output, refine=True)
        request.pop("continuation")
        self.write(request)
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.output, refine=True)

    def test_refine_source_prior_weight_bounds_before_ml(self):
        for value in (0, 0.1, 1):
            request = self.refine_fixture()
            request["config"]["prior_preservation_weight"] = value
            self.write(request)
            report = smoke.preflight(self.request_path, self.output, refine=True)
            self.assertEqual(report["request"]["config"]["prior_preservation_weight"], value)
        for value in (-1, 1.01, True, float("nan"), float("inf"), 10**400):
            request = self.refine_fixture()
            request["config"]["prior_preservation_weight"] = value
            self.write(request)
            with patch.object(smoke, "train") as trainer, self.assertRaises(ValueError):
                smoke.run(self.request_path, self.output, refine=True)
            trainer.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_facial_refine_annotations_preflight_without_ml(self):
        import builtins
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft", "accelerate"}:
                raise AssertionError("preflight imported ML")
            return original(name, *args, **kwargs)

        request = self.refine_fixture()
        regions = [{"x": 12, "y": 20, "width": 30, "height": 16}]
        for row in request["examples"]:
            row["facial_regions"] = regions
        for weight in (1, 3.0, 4):
            request["config"]["facial_loss_weight"] = weight
            self.write(request)
            with patch("builtins.__import__", side_effect=guard):
                report = smoke.preflight(self.request_path, self.output, refine=True)
            self.assertEqual(report["inputs"][0]["facial_regions"], regions)
            self.assertTrue(all("facial_regions" not in r for r in report["heldout_inputs"]))
        self.assertFalse(self.output.exists())

    def test_invalid_facial_requests_block_before_ml(self):
        import builtins
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft", "accelerate"}:
                raise AssertionError("invalid request imported ML")
            return original(name, *args, **kwargs)

        base = self.refine_fixture()
        base["config"]["facial_loss_weight"] = 3
        region = {"x": 12, "y": 20, "width": 30, "height": 16}
        for row in base["examples"]:
            row["facial_regions"] = [region.copy()]
        cases = []
        for weight in (0, 4.01, True, "3", float("nan"), float("inf"), 10**400):
            request = copy.deepcopy(base)
            request["config"]["facial_loss_weight"] = weight
            cases.append(request)
        for regions in (None, [], [region] * 9, [{**region, "extra": 1}],
                        *([{**region, key: value}] for key, value in
                          (("x", -1), ("x", 64), ("y", 64), ("width", 0),
                           ("width", 53), ("height", 45), ("x", True), ("y", 1.5),
                           ("width", float("nan"))))):
            request = copy.deepcopy(base)
            request["examples"][0]["facial_regions"] = regions
            cases.append(request)
        request = copy.deepcopy(base)
        request["config"]["facial_loss_weight"] = 1
        request["examples"][0]["facial_regions"] = []
        cases.append(request)
        request = copy.deepcopy(base)
        request["examples"][0].pop("facial_regions")
        cases.append(request)
        request = copy.deepcopy(base)
        request["heldout_examples"][0]["facial_regions"] = [region]
        cases.append(request)
        with patch("builtins.__import__", side_effect=guard), patch.object(smoke, "train") as trainer:
            for request in cases:
                self.write(request)
                with self.subTest(request=request), self.assertRaises(ValueError):
                    smoke.run(self.request_path, self.output, refine=True)
                self.assertFalse(self.output.exists())
            trainer.assert_not_called()
        for pilot in (False, True):
            for feature in ("config", "regions"):
                request = self.pilot_fixture() if pilot else copy.deepcopy(self.request)
                if feature == "config":
                    request["config"]["facial_loss_weight"] = 1
                else:
                    request["examples"][0]["facial_regions"] = [region]
                self.write(request)
                with self.assertRaises(ValueError):
                    smoke.preflight(self.request_path, self.output, pilot=pilot)

    def test_facial_mask_exact_layout_union_and_native_size_first(self):
        regions = [{"x": 16, "y": 16, "width": 32, "height": 32}]
        source = Image.new("RGBA", (64, 64), "black")
        from PIL import ImageDraw
        ImageDraw.Draw(source).rectangle((16, 16, 47, 47), fill="white")
        for resolution, contents in ((512, (384, 448, 512)), (768, (576, 672, 768))):
            for content in contents:
                latent_size = (resolution // 8,) * 2
                mask = smoke.facial_mask(regions, content, resolution, latent_size)
                rgb = smoke.prepare_rgb(source, content, resolution)
                # prepare_rgb pads white; independently clear only that padding for comparison.
                pad = (resolution - content) // 2
                expected = Image.new("L", (resolution, resolution), 0)
                expected.paste(rgb.convert("L").crop((pad, pad, pad + content, pad + content)), (pad, pad))
                expected = expected.resize(latent_size, Image.Resampling.NEAREST)
                self.assertEqual(bytes(v * 255 for v in mask.getdata()), expected.tobytes())
                union = smoke.facial_mask(regions + [{"x": 24, "y": 24, "width": 16, "height": 16}],
                                          content, resolution, latent_size)
                self.assertEqual(mask.tobytes(), union.tobytes())
                self.assertEqual(mask.getpixel((0, 0)), 0)
        request = self.refine_fixture()
        request["config"]["facial_loss_weight"] = 3
        row = request["examples"][0]
        row["facial_regions"] = []
        Image.new("RGBA", (48, 48)).save(row["path"])
        row["sha256"] = smoke.file_hash(row["path"])
        self.write(request)
        with self.assertRaisesRegex(ValueError, "64x64"):
            smoke.preflight(self.request_path, self.output, refine=True)

    @unittest.skipUnless(os.environ.get("SDXL_SMOKE_CPU_API_TEST") == "1", "optional pinned-runtime CPU helper test")
    def test_real_torch_facial_weighted_loss_cpu(self):
        import torch
        mask = smoke.facial_mask([{"x": 0, "y": 0, "width": 32, "height": 64}], 512, 512, (64, 64))
        weights = smoke.facial_weights(mask, 3)
        self.assertEqual(set(weights.flatten().tolist()), {1, 3})
        self.assertEqual(weights[0, 0, 0, 63].item(), 1)
        target = torch.zeros((2, 4, 64, 64))
        selected = target.clone().requires_grad_()
        with torch.no_grad():
            selected[:, :, :, :32] = 2
        outside = target.clone()
        outside[:, :, :, 32:] = 2
        selected_loss = smoke.denoising_loss(selected, target, weights)
        self.assertEqual(selected_loss.item(), 3)
        self.assertEqual(smoke.denoising_loss(outside, target, weights).item(), 1)
        selected_loss.backward()
        self.assertGreater(selected.grad.abs().sum().item(), 0)
        prediction = torch.randn_like(target)
        self.assertTrue(torch.equal(smoke.denoising_loss(prediction, target),
                                    torch.nn.functional.mse_loss(prediction.float(), target.float())))

    def test_detached_base_prediction_restores_adapter_and_training_state(self):
        from contextlib import contextmanager, nullcontext
        from types import SimpleNamespace
        active = {"no_grad": False}

        @contextmanager
        def no_grad():
            active["no_grad"] = True
            try:
                yield
            finally:
                active["no_grad"] = False

        fake_torch = SimpleNamespace(no_grad=no_grad, autocast=lambda *a, **k: nullcontext(), float16="half")
        test = self

        class Unet:
            training = True
            disabled = False
            fail = False
            def disable_adapters(self): self.disabled = True
            def enable_adapters(self): self.disabled = False
            def eval(self): self.training = False
            def train(self, flag): self.training = flag
            def __call__(self, noisy, timestep, **conditioning):
                test.assertTrue(self.disabled)
                test.assertFalse(self.training)
                test.assertTrue(active["no_grad"])
                test.assertEqual((noisy, timestep, conditioning), ("source", 7, {"tokens": "train-only"}))
                if self.fail: raise RuntimeError("teacher fixture failure")
                return SimpleNamespace(sample=SimpleNamespace(detach=lambda: "detached"))

        unet = Unet()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(smoke.base_prediction(unet, "source", 7, {"tokens": "train-only"}), "detached")
            self.assertFalse(unet.disabled)
            self.assertTrue(unet.training)
            self.assertFalse(active["no_grad"])
            unet.fail = True
            with self.assertRaisesRegex(RuntimeError, "teacher fixture failure"):
                smoke.base_prediction(unet, "source", 7, {"tokens": "train-only"})
            self.assertFalse(unet.disabled)
            self.assertTrue(unet.training)
            self.assertFalse(active["no_grad"])

    def test_refine_matched_768_resolution_preflight_and_canvas(self):
        for content in (576, 672, 768):
            request = self.refine_fixture(True)
            request["config"].update(resolution=768, content_resolution=content)
            self.write(request)
            with patch.object(smoke, "train") as trainer:
                report = smoke.preflight(self.request_path, self.output, refine=True)
                trainer.assert_not_called()
            self.assertEqual(report["request"]["config"]["resolution"], 768)
            image = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
            image.putpixel((0, 0), (0, 0, 0, 0))
            rgb = smoke.prepare_rgb(image, content, 768)
            pad = (768 - content) // 2
            self.assertEqual(rgb.size, (768, 768))
            self.assertEqual(rgb.getpixel((pad, pad)), (255, 255, 255))
            self.assertEqual(rgb.getpixel((384, 384)), (10, 20, 30))
            if pad:
                self.assertEqual(rgb.getpixel((pad - 1, 384)), (255, 255, 255))
        for resolution, content in ((1024, 768), (768, 512), (512, 672), (True, 768)):
            request = self.refine_fixture()
            request["config"].update(resolution=resolution, content_resolution=content)
            self.write(request)
            with self.assertRaises(ValueError):
                smoke.preflight(self.request_path, self.output, refine=True)

    def test_refine_continuation_rejects_hash_base_rank_and_incomplete_outcomes(self):
        for mutate in (lambda p: p.update(status="running"), lambda p: p.update(status="failed"),
                       lambda p: p.update(training_started=False),
                       lambda p: p["model"].update(id="other/base"),
                       lambda p: p["model"].update(revision="b" * 40),
                       lambda p: p["model"].update(file_sha256={}),
                       lambda p: p["request"]["config"].update(rank=8),
                       lambda p: p["checkpoint"].update(sha256="0" * 64),
                       lambda p: p["checkpoint"].update(path="elsewhere.safetensors")):
            request = self.refine_fixture(True)
            outcome = Path(request["continuation"]["outcome"])
            parent = json.loads(outcome.read_text())
            mutate(parent)
            outcome.write_text(json.dumps(parent))
            request["continuation"]["outcome_sha256"] = smoke.file_hash(outcome)
            self.write(request)
            with patch.object(smoke, "train") as trainer, self.assertRaises(ValueError):
                smoke.run(self.request_path, self.output, refine=True)
            trainer.assert_not_called()
        for field in ("sha256", "outcome_sha256"):
            request = self.refine_fixture(True)
            request["continuation"][field] = "0" * 64
            self.write(request)
            with self.assertRaises(ValueError):
                smoke.preflight(self.request_path, self.output, refine=True)
        request = self.refine_fixture(True)
        adapter = Path(request["continuation"]["path"])
        adapter.write_bytes(b"invalid")
        request["continuation"]["sha256"] = smoke.file_hash(adapter)
        self.write(request)
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.output, refine=True)
        with self.assertRaises(ValueError):
            smoke.preflight(self.request_path, self.root, refine=True)

    def test_refine_adapter_header_rank_dtype_and_size_gates(self):
        import struct
        for mutate in (lambda h: next(iter(h.values())).update(dtype="F16"),
                       lambda h: next(iter(h.values())).update(shape=[8, 4]),
                       lambda h: h.pop(next(iter(h))),
                       lambda h: next(iter(h.values())).update(data_offsets=[1, 129])):
            request = self.refine_fixture(True)
            adapter = Path(request["continuation"]["path"])
            raw = adapter.read_bytes()
            length = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + length])
            mutate(header)
            modified = json.dumps(header).encode()
            adapter.write_bytes(struct.pack("<Q", len(modified)) + modified + raw[8 + length:])
            request["continuation"]["sha256"] = smoke.file_hash(adapter)
            self.write(request)
            with self.assertRaises(ValueError):
                smoke.preflight(self.request_path, self.output, refine=True)
        with adapter.open("wb") as stream:
            stream.truncate(64 * 1024**2 + 1)
        with self.assertRaisesRegex(ValueError, "64 MiB"):
            smoke.validate_adapter_file(adapter)

    def test_refine_preprocessing_actual_pixels_and_legacy_equivalence(self):
        image = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
        image.putpixel((0, 0), (99, 88, 77, 0))
        image.putpixel((1, 0), (10, 20, 30, 127))
        for content in (384, 448, 512):
            rgb = smoke.prepare_rgb(image, content)
            pad, scale = (512 - content) // 2, content // 64
            self.assertEqual(rgb.size, (512, 512))
            self.assertEqual(rgb.getpixel((pad, pad)), (255, 255, 255))
            self.assertEqual(rgb.getpixel((pad + 2 * scale, pad)), (10, 20, 30))
            for x in range(scale):
                for y in range(scale):
                    self.assertEqual(rgb.getpixel((pad + 2 * scale + x, pad + y)), (10, 20, 30))
            if pad:
                self.assertEqual(rgb.getpixel((pad - 1, 256)), (255, 255, 255))
                self.assertEqual(rgb.getpixel((512 - pad, 256)), (255, 255, 255))
        rgba = image.resize((512, 512), Image.Resampling.NEAREST)
        legacy = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        self.assertEqual(smoke.prepare_rgb(image).tobytes(), legacy.tobytes())

    def test_refine_mocked_status_and_cli(self):
        self.write(self.refine_fixture())
        with patch.object(smoke.importlib.metadata, "version", side_effect=smoke.VERSIONS.__getitem__), \
                patch.object(smoke.sys, "version_info", (3, 12)), patch.object(smoke, "train"):
            result = smoke.run(self.request_path, self.output, refine=True)
        self.assertEqual(result["status"], "refinement_completed_unreviewed")
        self.assertIs(result["quality_assessed"], False)
        self.assertIs(result["promotion"], False)
        import io
        with patch.object(smoke.sys, "argv", ["smoke_sdxl.py", "--refine", "--request",
                                               str(self.request_path), "--output", str(self.output)]), \
                patch.object(smoke, "run", return_value=result) as run, \
                patch.object(smoke.sys, "stdout", new_callable=io.StringIO):
            smoke.main()
        run.assert_called_once_with(self.request_path, self.output, pilot=False, refine=True)
        with patch.object(smoke.sys, "argv", ["smoke_sdxl.py", "--refine", "--pilot", "--request",
                                               str(self.request_path), "--output", str(self.output)]):
            with self.assertRaises(SystemExit):
                smoke.main()

    def test_decoded_feature_config_preimport_and_training_only(self):
        import builtins
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft", "accelerate"}:
                raise AssertionError("preflight imported ML")
            return original(name, *args, **kwargs)

        base = self.refine_fixture()
        for row in base["examples"]:
            row["facial_regions"] = [{"x": 20, "y": 24, "width": 10, "height": 12}]
        for value in (0, 0.1, 0.25):
            request = copy.deepcopy(base)
            request["config"]["decoded_feature_loss_weight"] = value
            self.write(request)
            with patch("builtins.__import__", side_effect=guard):
                report = smoke.preflight(self.request_path, self.output, refine=True)
            self.assertTrue(all("facial_regions" not in row for row in report["heldout_inputs"]))
        bad = []
        for value in (-0.01, 0.250001, float("nan"), float("inf"), True, "0.1", 10**400):
            request = copy.deepcopy(base)
            request["config"]["decoded_feature_loss_weight"] = value
            bad.append(request)
        request = copy.deepcopy(base)
        request["config"]["decoded_feature_loss_weight"] = 0.1
        request["examples"][0].pop("facial_regions")
        bad.append(request)
        request = copy.deepcopy(base)
        request["config"]["decoded_feature_loss_weight"] = 0.1
        request["examples"][0]["facial_regions"] = [{"x": 0, "y": 0, "width": 64, "height": 64}]
        request["config"]["content_resolution"] = request["config"]["resolution"]
        bad.append(request)
        request = copy.deepcopy(base)
        request["heldout_examples"][0]["facial_regions"] = base["examples"][0]["facial_regions"]
        bad.append(request)
        request = copy.deepcopy(self.request)
        request["config"]["decoded_feature_loss_weight"] = 0
        bad.append(request)  # refine-only, even at zero
        with patch("builtins.__import__", side_effect=guard), patch.object(smoke, "train") as trainer:
            for request in bad:
                self.write(request)
                with self.assertRaises(ValueError):
                    smoke.run(self.request_path, self.output, refine="continuation" in request)
            trainer.assert_not_called()
        request = self.refine_fixture()
        request["config"]["decoded_feature_loss_weight"] = 0
        self.write(request)
        with patch("builtins.__import__", side_effect=guard):
            smoke.preflight(self.request_path, self.output, refine=True)  # zero needs no mask

    def test_decoded_default_does_not_retain_or_decode_vae(self):
        # Pin the opt-in boundary: auxiliary-only calls live beneath strict >0 branches.
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(smoke.train))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
                    "decoded_feature_cache", "decoded_feature_loss", "predicted_x0"}:
                ancestors = []
                current = node
                while current in parents:
                    current = parents[current]
                    ancestors.append(current)
                self.assertTrue(any(isinstance(a, ast.If) and "decoded_weight > 0" in ast.unparse(a.test)
                                    for a in ancestors))
        source = inspect.getsource(smoke.train)
        self.assertIn('decoded_weight = config.get("decoded_feature_loss_weight", 0)', source)
        self.assertIn('else:\n            del vae', source)

    @unittest.skipUnless(os.environ.get("SDXL_SMOKE_CPU_API_TEST") == "1", "optional pinned-runtime CPU API test")
    def test_decoded_crop_layout_gt_mask_padding_and_clamp(self):
        import torch
        regions = [{"x": 20, "y": 24, "width": 10, "height": 12},
                   {"x": 23, "y": 30, "width": 10, "height": 12}]
        source = Image.new("RGB", (64, 64), (25, 80, 130))
        for content, resolution in ((448, 512), (672, 768), (768, 768)):
            rgb = smoke.prepare_rgb(source, content, resolution)
            cache = smoke.decoded_feature_cache(rgb, regions, content, resolution,
                                                (resolution // 8, resolution // 8))
            bounds = smoke.facial_mask(regions, content, resolution, (resolution // 8,) * 2).getbbox()
            alignment = 3 if resolution == 768 else 1
            expected_box = ((bounds[0] - 2) // alignment * alignment,
                            (bounds[1] - 2) // alignment * alignment,
                            (bounds[2] + 2 + alignment - 1) // alignment * alignment,
                            (bounds[3] + 2 + alignment - 1) // alignment * alignment)
            self.assertEqual(cache["box"], expected_box)
            rgb_box = cache["provenance"]["rgb_box_xyxy"]
            stride = resolution // 64
            self.assertTrue(all(value % stride == 0 and value % 8 == 0 for value in rgb_box))
            self.assertEqual(rgb_box, [n * 8 for n in cache["box"]])
            self.assertEqual(cache["provenance"]["display_size_hw"],
                             [(rgb_box[3] - rgb_box[1]) // stride, (rgb_box[2] - rgb_box[0]) // stride])
            global_box = tuple(value // stride for value in rgb_box)
            full64 = rgb.resize((64, 64), Image.Resampling.BOX).crop(global_box)
            expected_pixels = torch.tensor(list(full64.getdata()), dtype=torch.float32)
            expected_pixels = expected_pixels.reshape(full64.height, full64.width, 3).permute(2, 0, 1).unsqueeze(0) / 255
            self.assertTrue(torch.allclose(cache["gt"], expected_pixels, atol=1e-6))
            selected = cache["mask"].expand_as(cache["gt"]) > 0
            expected = torch.tensor([25, 80, 130]).reshape(1, 3, 1, 1) / 255
            self.assertTrue(torch.allclose(cache["gt"][selected], expected.expand_as(cache["gt"])[selected]))
            self.assertTrue(torch.all((cache["mask"] >= 0) & (cache["mask"] <= 1)))
        # Native64 one-pixel ink feature retains global sampling phase, not a fractional crop grid.
        sparse = Image.new("RGB", (64, 64), "white")
        sparse.putpixel((27, 31), (0, 0, 0))
        rgb = smoke.prepare_rgb(sparse, 768, 768)
        cache = smoke.decoded_feature_cache(rgb, [{"x": 27, "y": 31, "width": 1, "height": 1}],
                                            768, 768, (96, 96))
        global_box = cache["provenance"]["global_display_box_xyxy"]
        self.assertTrue(torch.equal(cache["gt"][0, :, 31 - global_box[1], 27 - global_box[0]], torch.zeros(3)))
        self.assertEqual((cache["gt"] == 0).sum().item(), 3)
        full64 = rgb.resize((64, 64), Image.Resampling.BOX).crop(tuple(global_box))
        expected_pixels = torch.tensor(list(full64.getdata()), dtype=torch.float32)
        expected_pixels = expected_pixels.reshape(full64.height, full64.width, 3).permute(2, 0, 1).unsqueeze(0) / 255
        self.assertTrue(torch.equal(cache["gt"], expected_pixels))
        edge = [{"x": 0, "y": 0, "width": 1, "height": 1},
                {"x": 63, "y": 63, "width": 1, "height": 1}]
        with self.assertRaisesRegex(ValueError, "full image"):
            smoke.decoded_feature_cache(source.resize((768, 768)), edge, 768, 768, (96, 96))
        cache = smoke.decoded_feature_cache(source.resize((768, 768)), edge[:1], 768, 768, (96, 96))
        self.assertEqual(cache["box"][:2], (0, 0))
        with self.assertRaisesRegex(ValueError, "8x"):
            smoke.decoded_feature_cache(source.resize((768, 768)), regions, 768, 768, (95, 95))

    @unittest.skipUnless(os.environ.get("SDXL_SMOKE_CPU_API_TEST") == "1", "optional pinned-runtime CPU API test")
    def test_decoded_x0_epsilon_velocity_exact_and_gradient(self):
        import torch
        clean, noise = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        alpha = torch.tensor(0.7)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
        for kind, target in (("epsilon", noise), ("v_prediction", velocity)):
            prediction = target.clone().requires_grad_()
            recovered = smoke.predicted_x0(noisy, prediction, alpha, kind)
            self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))
            recovered.sum().backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(prediction.grad.abs().sum().item(), 0)

    @unittest.skipUnless(os.environ.get("SDXL_SMOKE_CPU_API_TEST") == "1", "optional pinned-runtime CPU API test")
    def test_decoded_cpu_toy_frozen_input_grad_and_crop_locality(self):
        import torch
        from types import SimpleNamespace
        class ToyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv2d(4, 3, 1, bias=False)
                torch.nn.init.constant_(self.layer.weight, 0.1)
                self.shape_seen = None
            def decode(self, latent):
                self.shape_seen = latent.shape
                return SimpleNamespace(sample=torch.nn.functional.interpolate(self.layer(latent), scale_factor=8))
        vae = ToyDecoder().float().eval().requires_grad_(False)
        before = [p.detach().clone() for p in vae.parameters()]
        source = Image.new("RGB", (768, 768), (25, 80, 130))
        cache = smoke.decoded_feature_cache(source, [{"x": 25, "y": 25, "width": 8, "height": 8}],
                                            768, 768, (96, 96))
        latent = torch.zeros(1, 4, 96, 96, requires_grad=True)
        loss = smoke.decoded_feature_loss(vae, latent, cache, 0.13025)
        loss.backward()
        left, top, right, bottom = cache["box"]
        self.assertEqual(vae.shape_seen[-2:], (bottom - top, right - left))
        self.assertLess(right - left, 96)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        outside = latent.grad.clone()
        outside[..., top:bottom, left:right] = 0
        self.assertEqual(outside.abs().sum().item(), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None and torch.equal(p, old)
                            for p, old in zip(vae.parameters(), before)))
        # A change completely outside the source union does not affect the masked RGB objective.
        masked = cache["mask"] == 0
        altered = {**cache, "gt": cache["gt"] + masked.expand_as(cache["gt"]) * 0.4}
        self.assertTrue(torch.equal(loss, smoke.decoded_feature_loss(vae, latent, altered, 0.13025)))

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
        smoke.validate_adapter_file(self.output / "pytorch_lora_weights.safetensors")
        import copy
        child = copy.deepcopy(unet)
        child.delete_adapters("default")
        parent_hash = smoke.file_hash(self.output / "pytorch_lora_weights.safetensors")
        reloaded = smoke.load_continuation(child, self.output / "pytorch_lora_weights.safetensors")
        self.assertEqual(parent_hash, smoke.file_hash(self.output / "pytorch_lora_weights.safetensors"))
        child_frozen = {k: p for k, p in child.named_parameters() if not p.requires_grad}
        self.assertEqual(set(frozen), set(child_frozen))
        self.assertTrue(all(torch.equal(v, child_frozen[k]) for k, v in frozen.items()))
        self.assertTrue(any(not torch.equal(v, v.half().float()) for v in expected.values()))
        self.assertEqual(set(expected), set(reloaded))
        self.assertTrue(all(torch.equal(v, reloaded[k]) for k, v in expected.items()))
        self.assertTrue(all(p.dtype == torch.float32 if p.requires_grad else p.dtype == torch.float16
                            for p in child.parameters()))
        smoke.reload_adapter(unet, self.output, expected)
        actual = {k: p for k, p in unet.named_parameters() if not p.requires_grad}
        self.assertEqual(set(frozen), set(actual))
        self.assertTrue(all(torch.equal(v, actual[k]) for k, v in frozen.items()))
        self.assertTrue((self.output / "pytorch_lora_weights.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
