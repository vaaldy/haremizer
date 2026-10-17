"""Durable local stage coordinator. External Sol image review is never automated."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from .core import REQUIRED_CHECKS, fingerprint
from .images import validate_png
from .prepare import canonical_digest, freeze_suite
from . import smoke_sdxl
from .sample import model_manifest

LIMIT = 100_000_000_000
ALLOWANCE = 200_000_000 + 2_000_000_000
SETTINGS = {"width": 768, "height": 768, "steps": 28, "guidance_scale": 5,
            "model_cpu_offload": True}
SOURCE = Path(__file__).resolve().parent


def module_argv(python, module):
    """Run a module from this same installation, not another interpreter's copy."""
    bootstrap = (
        "import pathlib,runpy,sys,haremizer; "
        "origin=pathlib.Path(haremizer.__file__).resolve().parent; "
        f"assert origin == pathlib.Path({str(SOURCE)!r}), 'package origin mismatch'; "
        "module=sys.argv.pop(1); runpy.run_module(module,run_name='__main__')"
    )
    return [str(python), "-I", "-B", "-c", bootstrap, f"haremizer.{module}"]


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("deadline must be UTC")
    return parsed.timestamp()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": fingerprint(path)}


def check_identity(record):
    if fingerprint(Path(record["path"])) != record["sha256"]:
        raise ValueError(f"hash changed: {record['path']}")


def tree(path):
    return [identity(file) for file in sorted(Path(path).rglob("*")) if file.is_file()]


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("all local input paths must be absolute")
    return Path(value).resolve()


def child_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class Loop:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "loop.json"
        self.state = json.loads(self.path.read_bytes())
        self.plan = json.loads((self.directory / "plan.json").read_bytes())

    def save(self):
        write_json(self.path, self.state)

    @classmethod
    def init(cls, directory, plan_path):
        directory = Path(directory).resolve()
        raw = Path(plan_path).read_bytes()
        plan = json.loads(raw)
        expected = {"model", "python", "deadline_utc", "max_stages", "train_seconds",
                    "sample_seconds", "storage_paths", "settings", "suites"}
        if set(plan) not in (expected, expected | {"validation_suites"}):
            raise ValueError("invalid plan fields")
        now = time.time()
        if not now < utc(plan["deadline_utc"]) <= now + 12 * 3600:
            raise ValueError("deadline must be in the next 12 hours")
        if type(plan["max_stages"]) is not int or not 1 <= plan["max_stages"] <= 8:
            raise ValueError("max_stages must be 1..8")
        if (type(plan["train_seconds"]) is not int or plan["train_seconds"] != 900 or
                type(plan["sample_seconds"]) is not int or plan["sample_seconds"] != 600):
            raise ValueError("train_seconds=900 and sample_seconds=600 required")
        settings = plan["settings"]
        required_settings = {*SETTINGS, "negative_prompt"}
        if (set(settings) not in (required_settings, required_settings | {"scheduler_override"}) or
                any(type(settings[k]) is not type(v) or settings[k] != v for k, v in SETTINGS.items()) or
                ("scheduler_override" in settings and settings["scheduler_override"] != "euler_ancestral") or
                not isinstance(settings["negative_prompt"], str) or not settings["negative_prompt"].strip()):
            raise ValueError("fixed 768 square/28 steps/CFG5/CPU offload and negative prompt required")
        model = plan["model"]
        smoke_sdxl.keys(model, {"path", "id", "revision"}, "model")
        model_path = absolute(model["path"])
        if (model["path"] != str(model_path) or
                not isinstance(model["id"], str) or re.fullmatch(r"[^/\s]+/[^/\s]+", model["id"]) is None or
                not isinstance(model["revision"], str) or
                re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model["revision"]) is None):
            raise ValueError("canonical model path, OWNER/MODEL and immutable revision required")
        manifest = model_manifest(model_path)
        if manifest["class"] != "StableDiffusionXLPipeline":
            raise ValueError("local SDXL required")
        revision = model["revision"]
        if len(revision) == 40:
            if (model_path.name != revision or model_path.parent.name != "snapshots" or
                    model_path.parent.parent.name != "models--" + model["id"].replace("/", "--")):
                raise ValueError("model path must match pinned HF revision")
        elif canonical_digest({name: fingerprint(model_path / name) for name in manifest["files"]}) != revision:
            raise ValueError("model content revision mismatch")
        python = absolute(plan["python"])
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("executable Python required")
        paths = plan["storage_paths"]
        if not isinstance(paths, list) or not paths:
            raise ValueError("explicit separately counted storage_paths required")
        for name in paths:
            absolute(name)
        if not any(directory.is_relative_to(absolute(name)) for name in paths):
            raise ValueError("storage_paths must cover run directory")
        modules = sorted(SOURCE.glob("*.py"))
        bindings = [identity(Path(plan["python"])), *map(identity, modules), *tree(model_path)]
        suites = {}
        for group in ("suites", "validation_suites"):
            if group not in plan:
                continue
            if not isinstance(plan[group], list) or len(plan[group]) != 2:
                raise ValueError("each suite pair must have primary and challenge")
            suites[group] = []
            for row in plan[group]:
                smoke_sdxl.keys(row, {"path", "baseline"}, "suite")
                source, baseline = absolute(row["path"]), absolute(row["baseline"])
                frozen = freeze_suite(source)
                cases = json.loads(frozen["frozen_json"])["cases"]
                if len(cases) != 4:
                    raise ValueError("each suite requires exactly four cases")
                cls.manifest(baseline, source, model, settings, None)
                baseline_files = tree(baseline)
                suites[group].append({"source": identity(source), "baseline": str(baseline),
                                      "baseline_files": baseline_files, "cases": cases})
                bindings.extend(baseline_files)
            if len({r["source"]["sha256"] for r in suites[group]}) != 2:
                raise ValueError("primary and challenge suites must differ")
        if "validation_suites" in suites:
            if ({r["source"]["sha256"] for r in suites["suites"]} &
                    {r["source"]["sha256"] for r in suites["validation_suites"]}):
                raise ValueError("validation suite bytes must be new")
            for first, second in zip(suites["suites"], suites["validation_suites"]):
                if (first["source"]["sha256"] == second["source"]["sha256"] or
                        any(a["seed"] == b["seed"] for a, b in zip(first["cases"], second["cases"]))):
                    raise ValueError("validation suites require additional fixed seeds")
        for path in [model_path, python, *(Path(r["baseline"]) for rows in suites.values() for r in rows)]:
            if not any(path.is_relative_to(absolute(name)) for name in paths):
                raise ValueError(f"storage_paths do not cover {path}")
        cls.storage(plan)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "plan.json").write_bytes(raw)
        for group, rows in suites.items():
            for index, row in enumerate(rows):
                file = directory / f"{group}-{index}.json"
                file.write_bytes(Path(row["source"]["path"]).read_bytes())
                row["frozen"] = identity(file)
        obj = cls.__new__(cls)
        obj.directory, obj.path, obj.plan = directory, directory / "loop.json", plan
        obj.state = {"version": 1, "initialized_utc": datetime.now(timezone.utc).isoformat(),
                     "plan": identity(directory / "plan.json"), "bindings": bindings,
                     "suites": suites, "status": "ready", "reason": None, "queue": None,
                     "stages": [], "incumbent": None, "best_promising": None, "previous": []}
        obj.save()
        obj.verify()
        return obj

    @staticmethod
    def storage(plan):
        entries = []
        # One du per path: overlapping roots and hardlinks intentionally count again.
        for name in plan["storage_paths"]:
            path = absolute(name)
            if not path.exists():
                # Only a new run descendant may be absent; count its existing ancestor.
                while not path.exists():
                    path = path.parent
            result = subprocess.run(["du", "-s", "-B1", "--", str(path)],
                                    capture_output=True, text=True, timeout=10, check=True)
            entries.append({"path": name, "counted_path": str(path),
                            "bytes": int(result.stdout.split()[0])})
        total = sum(item["bytes"] for item in entries) + ALLOWANCE
        if total >= LIMIT:
            raise ValueError(f"storage ceiling: {total} >= {LIMIT}")
        return {"entries": entries, "allowance_and_reserve": ALLOWANCE, "total": total}

    def verify(self):
        check_identity(self.state["plan"])
        for item in self.state["bindings"]:
            check_identity(item)
        for rows in self.state["suites"].values():
            for row in rows:
                check_identity(row["source"])
                check_identity(row["frozen"])
        for stage in self.state["stages"]:
            if stage.get("request"):
                check_identity(stage["request"])
        for item in (self.state["incumbent"], self.state["best_promising"]):
            if item:
                check_identity(item)

    def bounds(self):
        if time.time() >= utc(self.plan["deadline_utc"]):
            raise ValueError("deadline_exhausted")
        self.state["storage"] = self.storage(self.plan)

    def available(self):
        if any(p.get("cleanup_pending") for stage in self.state["stages"]
               for p in stage.get("processes", [])):
            raise ValueError("owned child cleanup pending; no further stages")
        if self.state["status"] in ("running", "needs_visual_review", "queued"):
            raise ValueError("stage must finish and receive external review before another stage")
        self.verify()
        self.bounds()
        if len(self.state["stages"]) >= self.plan["max_stages"]:
            raise ValueError("max_stages_exhausted")

    def preflight(self, request, output):
        value = json.loads(Path(request).read_bytes())
        if value.get("model") != self.plan["model"]:
            raise ValueError("request base model must equal frozen plan model")
        inputs = [value["model"]["path"]]
        for group in ("examples", "heldout_examples"):
            for row in value[group]:
                inputs.extend((row["path"], row["source_page"]["path"]))
        if value.get("continuation"):
            inputs.extend(value["continuation"][k] for k in ("path", "outcome"))
        for name in inputs:
            path = absolute(name)
            if path.is_relative_to(self.directory) and not value.get("continuation"):
                raise ValueError("generated run artifacts cannot become source data")
            if not any(path.is_relative_to(absolute(root)) for root in self.plan["storage_paths"]):
                raise ValueError("storage_paths must cover request inputs")
        # Never allow generated run PNGs into either supervised source partition.
        for group in ("examples", "heldout_examples"):
            if any(absolute(row["path"]).is_relative_to(self.directory) for row in value[group]):
                raise ValueError("generated outputs cannot be source feedback")
        return smoke_sdxl.preflight(request, output, refine=True)

    def enqueue(self, request, reason):
        self.available()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("explicit new request reason required")
        source = Path(request).resolve()
        report = self.preflight(source, self.directory / f"stage-{len(self.state['stages']) + 1:03}" / "train")
        digest = fingerprint(source)
        if any((stage.get("request") or {}).get("sha256") == digest for stage in self.state["stages"]):
            raise ValueError("each training stage requires an explicit new request")
        frozen = self.directory / f"request-{len(self.state['stages']) + 1:03}.json"
        with frozen.open("xb") as stream:
            stream.write(source.read_bytes())
        check_identity({"path": str(frozen), "sha256": digest})
        self.state.update(status="queued", reason=None,
                          queue={"request": identity(frozen), "source": identity(source),
                                 "reason": reason, "preflight": report})
        self.save()
        return self.state

    @staticmethod
    def manifest(directory, suite, model, settings, checkpoint):
        report = json.loads((directory / "manifest.json").read_bytes())
        cases = json.loads(Path(suite).read_bytes())["cases"]
        if (report.get("suite_sha256") != fingerprint(Path(suite)) or
                any(report.get("settings", {}).get(k) != v for k, v in settings.items()) or
                any(report.get("model", {}).get(k) != v for k, v in model.items()) or
                report.get("settings", {}).get("device") != "cuda" or
                (settings.get("scheduler_override") == "euler_ancestral" and
                 report.get("settings", {}).get("scheduler_class") != "EulerAncestralDiscreteScheduler")):
            raise ValueError("sample/baseline suite, model or settings mismatch")
        lora = report.get("lora")
        if (checkpoint is None and lora is not None) or (checkpoint is not None and
                (not isinstance(lora, dict) or lora.get("sha256") != checkpoint["sha256"])):
            raise ValueError("sample weights mismatch")
        records = report.get("records", [])
        if len(records) != 4:
            raise ValueError("all four frozen cases required")
        evidence = [identity(directory / "manifest.json")]
        for index, (case, record) in enumerate(zip(cases, records), 1):
            if record.get("file") != f"case-{index:02}.png":
                raise ValueError("canonical sample case filenames required")
            if any(record.get(k) != case[k] for k in ("id", "prompt", "seed")):
                raise ValueError("sample case mismatch")
            path = (directory / record["file"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("sample escaped directory")
            check_identity({"path": str(path), "sha256": record["sha256"]})
            from PIL import Image
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (768, 768) or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("raw 768 square single PNG required")
                image.verify()
            evidence.append(identity(path))
        return report, evidence

    def process(self, stage, argv, timeout, name):
        self.bounds()
        limit = min(timeout, utc(self.plan["deadline_utc"]) - time.time())
        if limit <= 0:
            raise ValueError("deadline_exhausted")
        environment = {**os.environ, "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        directory = Path(stage["directory"])
        record = {"argv": argv, "stdout": str(directory / f"{name}.stdout.log"),
                  "stderr": str(directory / f"{name}.stderr.log"), "timeout_seconds": limit,
                  "status": "running", "code": None, "memory_outcome": "not_reported"}
        stage["processes"].append(record)
        self.save()
        process = None
        started = time.monotonic()
        try:
            with open(record["stdout"], "xb") as stdout, open(record["stderr"], "xb") as stderr:
                process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=environment,
                                           cwd=directory, start_new_session=True, preexec_fn=child_limits)
                record["owned_pgid"] = process.pid
                self.save()
                last_storage = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started >= limit:
                        raise TimeoutError(f"{name} timeout")
                    if time.monotonic() - last_storage >= 5:
                        self.bounds()
                        last_storage = time.monotonic()
                    time.sleep(0.1)
                record["code"] = process.returncode
                if process.returncode:
                    raise RuntimeError(f"{name} exit {process.returncode}")
                record["status"] = "completed"
        except BaseException as error:
            if process is not None:
                # Only this child's new session/group, never unrelated GPU processes.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record["cleanup_pending"] = True
                record["code"] = process.returncode
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["duration_seconds"] = time.monotonic() - started
            self.save()
        self.bounds()

    def sheets(self, stage, rows):
        from PIL import Image, ImageDraw
        evidence = []
        for preview in (False, True):
            size = 64 if preview else 768
            cell_width = max(size, 256)
            sheet = Image.new("RGB", (cell_width * 2, (size + 32) * 8), "#303030")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(rows):
                for column, key in enumerate(("baseline", "candidate")):
                    with Image.open(row[key]["path"]) as image:
                        view = image.convert("RGB")
                        if preview:
                            view = view.resize((64, 64), Image.Resampling.NEAREST)
                        sheet.paste(view, (column * cell_width + (cell_width - size) // 2, index * (size + 32) + 32))
                    draw.text((column * cell_width, index * (size + 32)),
                              f"{row['case']} {key}", fill="white")
            path = Path(stage["directory"]) / ("preview-64.png" if preview else "raw-comparison.png")
            if path.exists():
                raise ValueError("no overwrite")
            sheet.save(path)
            evidence.append(identity(path))
        return evidence

    def pipeline(self, checkpoint=None, outcome=None):
        validation = checkpoint is not None
        if validation:
            self.available()
            if "validation_suites" not in self.state["suites"]:
                raise ValueError("no frozen validation_suites")
            checkpoint, outcome = identity(absolute(str(checkpoint))), identity(absolute(str(outcome)))
            if not any(s.get("checkpoint") == checkpoint and s.get("outcome") == outcome and
                       s.get("review", {}).get("accepted") for s in self.state["stages"]):
                raise ValueError("validation requires an already accepted promising checkpoint")
            queue = None
        else:
            if self.state["status"] != "queued":
                raise ValueError("next requires exactly one explicitly queued request")
            self.verify()
            self.bounds()
            if len(self.state["stages"]) >= self.plan["max_stages"]:
                raise ValueError("max_stages_exhausted")
            queue = self.state["queue"]
            check_identity(queue["request"])
            check_identity(queue["source"])
        number = len(self.state["stages"]) + 1
        directory = self.directory / f"stage-{number:03}"
        directory.mkdir(exist_ok=False)
        stage = {"number": number, "directory": str(directory), "validation_only": validation,
                 "status": "running", "processes": [], "request": queue["request"] if queue else None,
                 "preflight": queue["preflight"] if queue else None,
                 "reason": queue["reason"] if queue else "explicit same-checkpoint validation"}
        self.state["stages"].append(stage)
        self.state.update(status="running", queue=None, reason=None)
        self.save()
        try:
            if not validation:
                self.preflight(Path(queue["request"]["path"]), directory / "train")
                self.verify()
                self.process(stage, [*module_argv(self.plan["python"], "smoke_sdxl"),
                                     "--refine", "--request", queue["request"]["path"],
                                     "--output", str(directory / "train")], self.plan["train_seconds"], "train")
                outcome = identity(directory / "train" / "outcome.json")
                result = json.loads(Path(outcome["path"]).read_bytes())
                checkpoint = identity(directory / "train" / result["checkpoint"]["path"])
                if not Path(checkpoint["path"]).is_relative_to(directory / "train"):
                    raise ValueError("checkpoint escaped stage")
            result = json.loads(Path(outcome["path"]).read_bytes())
            if (result.get("status") != "refinement_completed_unreviewed" or result.get("phase") != "completed" or
                    result.get("checkpoint", {}).get("sha256") != checkpoint["sha256"] or
                    result.get("checkpoint", {}).get("reload_exact") is not True or result.get("training_started") is not True or
                    (Path(outcome["path"]).parent / result["checkpoint"]["path"]).resolve() != Path(checkpoint["path"]) or
                    any(result.get("model", {}).get(k) != v for k, v in self.plan["model"].items())):
                raise ValueError("completed exact-reload checkpoint provenance required")
            smoke_sdxl.validate_adapter_file(Path(checkpoint["path"]))
            stage.update(checkpoint=checkpoint, outcome=outcome,
                         memory_outcome=result.get("runtime", {}).get("memory", "see outcome"))
            if not validation:
                stage["processes"][0]["memory_outcome"] = stage["memory_outcome"]
            group = "validation_suites" if validation else "suites"
            stage["suite_hashes"] = [row["frozen"]["sha256"] for row in self.state["suites"][group]]
            evidence, rows = [outcome], []
            for suite in self.state["suites"][group]:
                evidence.extend([suite["frozen"], *suite["baseline_files"]])
            for index, suite in enumerate(self.state["suites"][group]):
                self.verify()
                check_identity(checkpoint)
                output = directory / f"sample-{index}"
                argv = [*module_argv(self.plan["python"], "sample"), "--model", self.plan["model"]["path"],
                        "--model-id", self.plan["model"]["id"], "--revision", self.plan["model"]["revision"],
                        "--output", str(output), "--suite", suite["frozen"]["path"],
                        "--width", "768", "--height", "768", "--steps", "28", "--max-cases", "4",
                        "--guidance-scale", "5", "--negative-prompt", self.plan["settings"]["negative_prompt"],
                        "--cpu-offload", "--lora", checkpoint["path"], "--lora-sha256", checkpoint["sha256"]]
                if "scheduler_override" in self.plan["settings"]:
                    argv.extend(("--scheduler", self.plan["settings"]["scheduler_override"]))
                self.process(stage, argv, self.plan["sample_seconds"], f"sample-{index}")
                report, files = self.manifest(output, Path(suite["frozen"]["path"]), self.plan["model"],
                                              self.plan["settings"], checkpoint)
                stage["processes"][-1]["memory_outcome"] = [r.get("gpu_memory") for r in report["records"]]
                evidence.extend(files)
                for case_index, case in enumerate(suite["cases"]):
                    rows.append({"case": f"{index}:{case['id']}", "candidate": files[case_index + 1],
                                 "baseline": identity(Path(suite["baseline"]) / f"case-{case_index + 1:02}.png")})
            evidence.extend(self.sheets(stage, rows))
            stage.update(evidence=evidence, cases=rows,
                         preview_contract="64 NEAREST unquantized viewing only; not transparent validated sprite or quality proof")
            self.verify()
            check_identity(checkpoint)
            for item in evidence:
                check_identity(item)
            self.bounds()
            stage["status"] = "needs_visual_review"
            self.state.update(status="needs_visual_review", reason="external_actual_image_review_required")
        except BaseException as error:
            failure_outcome = directory / "train" / "outcome.json"
            if failure_outcome.is_file():
                try:
                    failed = json.loads(failure_outcome.read_bytes())
                    stage["failure_outcome"] = identity(failure_outcome)
                    stage["memory_outcome"] = failed.get("runtime", {}).get("failure_memory", "see outcome")
                    if stage["processes"]:
                        stage["processes"][0]["memory_outcome"] = stage["memory_outcome"]
                except (ValueError, OSError):
                    pass
            stage.update(status="failed", failure=f"{type(error).__name__}: {error}")
            self.state.update(status="blocked", reason=stage["failure"])
            self.save()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        self.save()
        return self.state

    def review(self, review_path):
        if self.state["status"] != "needs_visual_review":
            raise ValueError("no stage awaiting visual review")
        self.verify()
        stage = self.state["stages"][-1]
        check_identity(stage["checkpoint"])
        for row in stage["evidence"]:
            check_identity(row)
        raw = Path(review_path).read_bytes()
        review = json.loads(raw)
        reviewer = review.get("reviewed_by")
        if (not isinstance(reviewer, str) or not reviewer.startswith("machine:") or
                not reviewer.removeprefix("machine:").strip() or
                not isinstance(review.get("notes"), str) or
                not review["notes"].strip() or type(review.get("uncertain")) is not bool or
                type(review.get("robust_requested", False)) is not bool):
            raise ValueError("explicit Sol review, notes, uncertainty and boolean robust_requested required")
        checks = review.get("checks", {})
        if set(checks) != set(REQUIRED_CHECKS) or any(type(v) is not bool for v in checks.values()):
            raise ValueError("review requires exactly core.REQUIRED_CHECKS booleans")
        binding = {"stage": stage["number"], "checkpoint": stage["checkpoint"],
                   "suite_hashes": stage["suite_hashes"], "evidence": stage["evidence"],
                   "plan_sha256": self.state["plan"]["sha256"]}
        if review.get("binding") != binding:
            raise ValueError("review stage/checkpoint/suites/evidence/plan binding mismatch")
        observations = review.get("cases", {})
        if set(observations) != {r["case"] for r in stage["cases"]}:
            raise ValueError("case observations required for all eight cases")
        for notes in observations.values():
            if (not isinstance(notes, dict) or set(notes) != {"glitches", "crop_duplicate_face", "adherence", "readability"} or
                    any(not isinstance(v, str) or not v.strip() for v in notes.values())):
                raise ValueError("each case needs glitches/crop_duplicate_face/adherence/readability notes")
        format_evidence = review.get("format_evidence", [])
        if checks["format"]:
            if not format_evidence:
                raise ValueError("format=True requires actual native PNG evidence, not preview")
            widths = set()
            for row in format_evidence:
                path = Path(row["path"]).resolve()
                if not path.is_relative_to(Path(stage["directory"])) or any(row == e for e in stage["evidence"]):
                    raise ValueError("format evidence must be separate postprocessed PNGs inside current stage")
                check_identity(row)
                widths.add(validate_png(path)["width"])
            if review.get("robust_requested") and all(checks.values()) and not review["uncertain"] and 64 not in widths:
                raise ValueError("robust request requires actual validated native 64 PNG")
        accepted = all(checks.values()) and not review["uncertain"]
        prior = [s for s in self.state["stages"][:-1] if s.get("review", {}).get("accepted") and
                 s.get("checkpoint", {}).get("sha256") == stage["checkpoint"]["sha256"] and
                 set(s.get("suite_hashes", [])) != set(stage["suite_hashes"]) and len(s.get("cases", [])) == 8]
        if review.get("robust_requested") and accepted:
            if not prior or len(stage["cases"]) != 8:
                raise ValueError("robust requires two accepted different-suite eight-case passes on same weights")
            for earlier in prior:
                check_identity(earlier["checkpoint"])
                check_identity(earlier["review"]["file"])
                for item in earlier["evidence"] + earlier["review"]["declaration"].get("format_evidence", []):
                    check_identity(item)
        path = Path(stage["directory"]) / "review.json"
        with path.open("xb") as stream:
            stream.write(raw)
        stage["review"] = {"file": identity(path), "accepted": accepted, "declaration": review,
                           "scope": "external declarations; not legal, human or semantic proof"}
        if accepted:
            candidate = stage["checkpoint"]
            previous = self.state["best_promising"]
            if previous and previous != candidate:
                self.state["previous"].append(previous)
            self.state["best_promising"] = candidate
            if review.get("robust_requested"):
                if self.state["incumbent"] and self.state["incumbent"] != candidate:
                    self.state["previous"].append(self.state["incumbent"])
                self.state["incumbent"] = candidate
                stage["status"] = "robust"
            else:
                stage["status"] = "promising"
        else:
            stage.update(status="rejected", failure={"checks": checks, "uncertain": review["uncertain"],
                                                     "notes": review["notes"]})
        self.state.update(status=stage["status"], reason=None)
        self.save()
        return self.state

    def status(self):
        # Reviews remain possible after compute exhaustion; no quality verdict is inferred.
        try:
            self.verify()
            self.bounds()
            budget = "max_stages_exhausted" if len(self.state["stages"]) >= self.plan["max_stages"] else "available"
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            budget = str(error)
        result = {**self.state, "budget": budget}
        if self.state["status"] == "running":
            result["recovery"] = "interrupted or active stage: no automatic resume; inspect recorded owned process and evidence"
        if self.state["status"] == "needs_visual_review":
            stage = self.state["stages"][-1]
            result["review_binding"] = {"stage": stage["number"], "checkpoint": stage["checkpoint"],
                                        "suite_hashes": stage["suite_hashes"], "evidence": stage["evidence"],
                                        "plan_sha256": self.state["plan"]["sha256"]}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "enqueue", "next", "review", "status", "validate-existing"):
        command = commands.add_parser(name)
        command.add_argument("run", type=Path)
        if name == "init":
            command.add_argument("plan", type=Path)
        elif name == "enqueue":
            command.add_argument("request", type=Path)
            command.add_argument("reason")
        elif name == "review":
            command.add_argument("review_json", type=Path)
        elif name == "validate-existing":
            command.add_argument("checkpoint", type=Path)
            command.add_argument("outcome", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = Loop.init(args.run, args.plan).status()
        else:
            # Serializes CLI writers and prevents two next calls from owning a stage.
            # Lock a stable directory inode: loop.json is atomically replaced.
            fd = os.open(args.run, os.O_DIRECTORY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                loop = Loop(args.run)
                if args.command == "enqueue":
                    loop.enqueue(args.request, args.reason)
                elif args.command == "next":
                    loop.pipeline()
                elif args.command == "review":
                    loop.review(args.review_json)
                elif args.command == "validate-existing":
                    loop.pipeline(args.checkpoint, args.outcome)
                result = loop.status()
            finally:
                os.close(fd)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"blocked: {error}\n")


if __name__ == "__main__":
    main()
