"""CPU-only synthetic sampler replacement; these tests are NOT inference."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from haremizer import workflow as wf
from haremizer.process_guard import run_process


class ProcessGuardTests(unittest.TestCase):
    def test_nonzero_signal_timeout_and_interrupt_cleanup(self):
        for code, timeout, interrupt in [("raise SystemExit(7)", 3, False),
                ("import os,signal; os.kill(os.getpid(),signal.SIGTERM)", 3, False),
                ("import time; time.sleep(10)", .15, False),
                ("import time; time.sleep(10)", 3, True)]:
            with self.subTest(code=code, interrupt=interrupt), tempfile.TemporaryDirectory() as tmp:
                stage = {"directory": tmp, "processes": []}
                def bounds():
                    if interrupt and stage["processes"] and "owned_pgid" in stage["processes"][0]:
                        raise KeyboardInterrupt()
                real_sleep = time.sleep
                interrupted = False
                def sleep(seconds):
                    nonlocal interrupted
                    if interrupt and not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt()
                    real_sleep(seconds)
                with patch("haremizer.process_guard.time.sleep", side_effect=sleep):
                    with self.assertRaises(KeyboardInterrupt if interrupt else (TimeoutError if timeout < 1 else RuntimeError)):
                        run_process(stage, [sys.executable, "-c", code], timeout, "test",
                                    bounds=bounds, save=lambda: None, deadline=time.time()+10)
                record = stage["processes"][0]
                self.assertEqual(record["status"], "failed")
                self.assertIsNotNone(record["code"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(record["owned_pgid"], 0)
                self.assertTrue(Path(record["stdout"]).exists())
                self.assertTrue(Path(record["stderr"]).exists())


    def test_missing_executable_and_midrun_bounds_failure(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                stage = {"directory":tmp,"processes":[]}
                calls = 0
                def bounds():
                    nonlocal calls
                    calls += 1
                    if calls > 1:
                        raise ValueError("storage ceiling fixture")
                argv = [str(Path(tmp) / "absent")] if missing else [sys.executable,"-c","import time; time.sleep(10)"]
                # Make the five-second periodic check deterministic, without waiting.
                ticks = iter([0, 0, 0, 6, 6, 6, 6, 6])
                with patch("haremizer.process_guard.time.monotonic", side_effect=lambda: next(ticks, 6)):
                    with self.assertRaises(OSError if missing else ValueError):
                        run_process(stage,argv,9,"test",bounds=bounds,save=lambda:None,deadline=time.time()+20)
                record = stage["processes"][0]
                self.assertEqual(record["status"],"failed")
                if not missing:
                    self.assertIsNotNone(record["code"])
                    with self.assertRaises(ProcessLookupError):
                        os.kill(record["owned_pgid"],0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "model_index.json").write_text('{"_class_name":"StableDiffusionXLPipeline"}')
        blob = self.root / "weight-blob"
        blob.write_bytes(b"mock only, not real weights")
        weight = model / "weights.safetensors"
        weight.symlink_to(blob)
        self.recipe = {"model": {"path":"model", "id":"mock/model", "revision":"a"*40,
            "weight_sha256":{"weights.safetensors":wf.fingerprint(weight)}},
            "prompt":"mock portrait", "negative_prompt":"mock negative", "seed":33013,
            "steps":28,"width":64,"height":64,"guidance_scale":5,"scheduler":"euler_ancestral",
            "cpu_offload":True,"max_cases":1,"layerdiffuse":None,
            "recovery":{"background":[255,0,255],"tolerance":0,"colors":8,"size":64},
            "export":{"name":"neutral"}}
        self.config = {"recipe":"recipe.json", "recipe_sha256":"", "timeout_seconds":3,
            "storage_roots":{key: (str(Path(sys.executable).resolve().parent) if key == "python" else str(self.root)) for key in wf.STORAGE_ROOTS}}
        self.config_path = self.root / "config.json"
        self.freeze()
        self.output = self.root / "output"

    def freeze(self):
        path = self.root / "recipe.json"
        path.write_text(json.dumps(self.recipe))
        self.config["recipe_sha256"] = wf.fingerprint(path)
        self.config_path.write_text(json.dumps(self.config))

    def child(self, argv, **kwargs):
        # Replace only the ML child; run the real owned-child guard around CPU code.
        if "--model" not in argv:
            return self.real_popen(argv, **kwargs)
        raw = self.output / "raw"
        raw.mkdir()
        image = Image.new("RGB", (64,64), (255,0,255))
        for y in range(16,48):
            for x in range(16,48):
                image.putpixel((x,y),(0,200,100))
        image.save(raw / "case-01.png")
        (raw / "manifest.json").write_text(json.dumps({"model": {**self.recipe["model"], "path":str(self.root / "model")},
            "settings": {**{k:self.recipe[k] for k in ("width","height","steps","guidance_scale","max_cases","negative_prompt")},
                         "model_cpu_offload":True,"scheduler_override":"euler_ancestral","device":"cuda"},
            "records":[{"file":"case-01.png",
            "sha256":wf.fingerprint(raw / "case-01.png"), "seed":33013,"prompt":"mock portrait"}]}))
        self.environment = kwargs["env"]
        return self.real_popen([sys.executable,"-c","print('mock sampler only')"], **kwargs)

    def test_mock_end_to_end_and_collision(self):
        self.real_popen = subprocess.Popen
        with patch("haremizer.process_guard.subprocess.Popen", side_effect=self.child):
            report = wf.run(self.config_path, self.output)
        self.assertEqual(report["status"], "completed_unreviewed")
        self.assertEqual(self.environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(self.environment["HSA_OVERRIDE_GFX_VERSION"], "10.3.0")
        sprite = json.loads((self.output / "character/neutral.json").read_text())
        self.assertEqual(sprite["size"], [64,64])
        self.assertEqual(sum(c != "." for row in sprite["frames"]["base"] for c in row), 1024)
        with self.assertRaisesRegex(ValueError,"new directory"):
            wf.run(self.config_path, self.output)

    def test_partial_output_fails_closed(self):
        def partial(stage, *args, **kwargs):
            (self.output / "raw").mkdir()
            (self.output / "raw/case-01.png").write_bytes(b"partial")
        with patch.object(wf,"run_process", side_effect=partial):
            with self.assertRaises(OSError):
                wf.run(self.config_path,self.output)
        self.assertEqual(json.loads((self.output / "workflow.json").read_text())["status"], "failed")
        self.assertFalse((self.output / "character").exists())

    def test_missing_model_malformed_recipe_and_hash(self):
        for field, value in [("steps", 0), ("width", 1024), ("recovery", {}), ("cpu_offload", False)]:
            with self.subTest(field=field):
                old = self.recipe[field]
                self.recipe[field] = value
                self.freeze()
                with self.assertRaises(ValueError):
                    wf.run(self.config_path,self.output)
                self.assertFalse(self.output.exists())
                self.recipe[field] = old
        self.recipe["model"]["path"] = "absent"
        self.freeze()
        with self.assertRaisesRegex(ValueError,"model directory"):
            wf.run(self.config_path,self.output)
        self.config["recipe_sha256"] = "0"*64
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError,"recipe hash"):
            wf.run(self.config_path,self.output)

    def test_changed_binding_and_input_output_collision(self):
        self.output = self.root / "model" / "new-output"
        with self.assertRaisesRegex(ValueError,"collides"):
            wf.run(self.config_path,self.output)
        self.output = self.root / "output"
        def changed(stage, *args, **kwargs):
            (self.root / "model/weights.safetensors").write_bytes(b"changed")
            kwargs["bounds"]()
        with patch.object(wf, "run_process", side_effect=changed):
            with self.assertRaisesRegex(ValueError,"hash changed"):
                wf.run(self.config_path,self.output)
        self.assertEqual(json.loads((self.output / "workflow.json").read_text())["status"], "failed")
