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
