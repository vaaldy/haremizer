"""Human-gated local upstream LoRA adapter. CLI previews only; never authorizes/runs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .core import Attempt, fingerprint
from .images import pillow
from .prepare import canonical_digest, freeze_suite, nonempty, validate_dataset
from .runtime import probe_runtime


CONTRACT = "reviewed-diffusers-sd1-text-to-image-lora-v1"
ENV_KEYS = ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_contract(facts):
    tensor = facts.get("torch_probe", {}).get("evidence", {})
    keys = ("python", "python_executable", "environment", "probe_mode",
            "compatibility_workaround_requires_human_acknowledgment")
    return {**{key: facts.get(key, False if key.startswith("compatibility_") else None) for key in keys},
            **{key: tensor.get(key) for key in ("torch_version", "hip_version", "cuda_version", "device", "gpu_visible")},
            "gpu_operations_verified": facts.get("gpu_operations_verified") is True}


def dependencies():
    # Version/installation metadata identity; not a sandbox or a package-code attestation.
    return sorted((dist.metadata["Name"], dist.version,
                   canonical_digest([dist.read_text("METADATA"), dist.read_text("RECORD")]))
                  for dist in importlib.metadata.distributions())


def model_identity(path, model):
    require(path.is_dir(), "local model directory required")
    files = {str(file.relative_to(path)): fingerprint(file) for file in sorted(path.rglob("*")) if file.is_file()}
    require(files and "model_index.json" in files, "complete cached diffusers model required")
    index = json.loads((path / "model_index.json").read_text())
    require(index.get("_class_name") == "StableDiffusionPipeline", "only SD1 StableDiffusionPipeline supported")
    # Reject custom pipeline/component modules that could request remote Python code.
    for key, value in index.items():
        if not key.startswith("_"):
            require(isinstance(value, list) and len(value) == 2 and
                    value[0] in (None, "diffusers", "transformers", "stable_diffusion"), "custom model component rejected")
    revision = model["revision"]
    digest = canonical_digest(files)
    if len(revision) == 64:
        require(digest == revision.lower(), "model content revision mismatch")
    else:
        require(path.name.lower() == revision.lower() and path.parent.name == "snapshots" and
                path.parent.parent.name == "models--" + model["id"].replace("/", "--"), "HF cache model/revision mismatch")
    return {"path": str(path), "files": files, "sha256": digest}


class LoraTrainer:
    """One finite core.Run envelope; approval is separate, externally written JSON."""

    def __init__(self, request: Path, authorization: Path | None = None):
        self.request = Path(request)
        self.authorization = Path(authorization) if authorization else None

    def preview(self):
        """Read-only exact argv template and incomplete approval payload; no subprocess."""
        unrecorded = [key for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD") if os.environ.get(key)]
        require(not unrecorded, f"unrecorded runtime overrides: {', '.join(unrecorded)}; prepare a clean dedicated runner environment")
        spec = json.loads(self.request.read_text())
        plan = json.loads(Path(spec["preparation"]).read_text())
        require(plan["status"] == "needs_human" and plan["blockers"] == [] and
                plan["training_authorized"] is False and plan["training_started"] is False, "preparation is not human-gated/clear")
        config, model, runtime = plan["config"], plan["model"], plan["runtime_contract"]
        require(re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", model["revision"]) and nonempty(model["id"]), "immutable model identity required")
        require(type(config["max_steps"]) is int and 1 <= config["max_steps"] <= 1000 and
                type(config["lora_rank"]) is int and 1 <= config["lora_rank"] <= 16 and
                type(config["training_resolution"]) is int and 64 <= config["training_resolution"] <= 1024 and
                config["training_resolution"] % 8 == 0 and config["batch_size"] == 1 and
                type(config["seed"]) is int, "invalid training bounds")
        roots = {key: str(Path(spec[key]).resolve()) for key in ("train", "evaluation", "rights", "suite", "run_directory")}
        dataset = validate_dataset(Path(roots["train"]), Path(roots["evaluation"]), Path(roots["rights"]),
                                   tuple(plan["dataset"]["allowed_sizes"]))
        suite = freeze_suite(Path(roots["suite"]))
        require(dataset == plan["dataset"] and suite == plan["evaluation_suite"], "dataset/rights/suite changed")
        binding = {"dataset_sha256": dataset["sha256"], "suite_sha256": suite["sha256"],
                   "model": model, "config": config, "runtime_contract": runtime}
        require(binding == plan["preparation_binding"] and canonical_digest(binding) == plan["preparation_sha256"], "fake preparation binding")
        require(runtime == runtime_contract(plan["runtime"]) and runtime["gpu_operations_verified"] is True, "runtime evidence mismatch")
        python = Path(spec["python"]).absolute()
        require(str(python) == runtime["python_executable"] and python.resolve() == Path(sys.executable).resolve() and
                python.is_file() and os.access(python, os.X_OK), "run backend under prepared Python runner")
        environment = {key: os.environ.get(key) for key in ENV_KEYS}
        require(environment == {key: runtime["environment"].get(key) for key in ENV_KEYS}, "ambient runtime environment changed")
        override = environment["HSA_OVERRIDE_GFX_VERSION"]
        require(runtime["probe_mode"] == ("explicit_environment_override" if override else "native") and
                runtime["compatibility_workaround_requires_human_acknowledgment"] == bool(override), "runtime override evidence mismatch")
        dep = dependencies()
        torch_versions = [version for name, version, _ in dep if name.lower() == "torch"]
        require(torch_versions == [runtime["torch_version"]], "installed torch identity changed")
        require(runtime["python"] == ".".join(map(str, sys.version_info[:3])), "Python version changed")
        script = Path(spec["trainer"]).resolve()
        require(script.name == "train_text_to_image_lora.py" and fingerprint(script) == spec["trainer_sha256"], "trainer hash/name mismatch")
        contract = spec["script_contract"]
        require(contract.get("name") == CONTRACT and contract.get("trainer_sha256") == spec["trainer_sha256"] and
                nonempty(contract.get("reviewed_by")) and nonempty(contract.get("note")), "reviewed pinned upstream script contract required")
        timeout, maximum = spec["timeout_seconds"], spec["max_attempts"]
        require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 86400 and
                type(maximum) is int and 1 <= maximum <= 3, "finite timeout and max_attempts 1..3 required")
        run = json.loads((Path(roots["run_directory"]) / "run.json").read_text())
        require(run["max_attempts"] == maximum and run["dataset_id"] == dataset["sha256"] and
                run["suite_sha256"] == suite["sha256"] and
                fingerprint(Path(roots["run_directory"]) / "suite.json") == suite["sha256"], "core run envelope mismatch")
        policy = spec["preprocessing"]
        require(policy.get("resize") == "NEAREST" and policy.get("alpha") == "flatten" and
                isinstance(policy.get("background"), list) and len(policy["background"]) == 3 and
                all(type(v) is int and 0 <= v <= 255 for v in policy["background"]), "explicit NEAREST/alpha flatten RGB background policy required")
        precision = spec["mixed_precision"]
        require(precision in ("no", "fp16"), "mixed_precision must be explicit no or fp16")
        env = {"PATH": str(python.parent) + ":/usr/bin:/bin", "HOME": str(Path.home()),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
               **{key: value for key, value in environment.items() if value is not None}}
        cached = model_identity(Path(spec["model_path"]).resolve(), model)
        argv = [str(python), "-I", "-B", str(script), "--pretrained_model_name_or_path", cached["path"],
                "--train_data_dir", "{attempt}/imagefolder", "--resolution", str(config["training_resolution"]),
                "--center_crop", "--train_batch_size", "1", "--gradient_accumulation_steps", "4",
                "--gradient_checkpointing", "--max_train_steps", str(config["max_steps"]),
                "--rank", str(config["lora_rank"]), "--seed", str(config["seed"]),
                "--mixed_precision", precision, "--learning_rate", "0.0001", "--lr_scheduler", "constant",
                "--lr_warmup_steps", "0", "--report_to", "none", "--output_dir", "{attempt}/output"]
        execution = {"preparation_sha256": plan["preparation_sha256"], "binding": binding, "roots": roots,
                     "model_cache": cached, "trainer_sha256": spec["trainer_sha256"], "script_contract": contract,
                     "runner_sha256": fingerprint(python), "dependencies": dep, "environment": env,
                     "argv_template": argv, "preprocessing": policy, "timeout_seconds": timeout,
                     "max_attempts": maximum, "maximum_training_wall_seconds": timeout * maximum}
        digest = canonical_digest(execution)
        return {"training_authorized": False, "training_started": False, "execution": execution,
                "execution_sha256": digest, "approval_template": {
                    "training_authorized": False, "preparation_sha256": plan["preparation_sha256"],
                    "execution_sha256": digest, "approved_by": "", "note": ""}}

    def train(self, attempt: Attempt) -> Path:
        preview = self.preview()
        require(self.authorization is not None, "distinct human training authorization required")
        approval = json.loads(self.authorization.read_text())
        execution = preview["execution"]
        require(approval.get("training_authorized") is True and nonempty(approval.get("approved_by")) and
                nonempty(approval.get("note")) and approval.get("preparation_sha256") == execution["preparation_sha256"] and
                approval.get("execution_sha256") == preview["execution_sha256"], "missing/mismatched human authorization")
        override = execution["environment"].get("HSA_OVERRIDE_GFX_VERSION")
        if override:
            require(f"HSA_OVERRIDE_GFX_VERSION={override}" in approval["note"] and
                    "unsupported compatibility workaround" in approval["note"].lower(), "approval must explicitly acknowledge runtime override")
        require(sys.platform == "linux", "process-group containment requires Linux")
        root = Path(execution["roots"]["run_directory"])
        run = json.loads((root / "run.json").read_text())
        require(run["status"] == "running" and run["attempts"] == attempt.number, "attempt must be active in core.Run")
        directory = attempt.directory.absolute()
        require(type(attempt.number) is int and 1 <= attempt.number <= execution["max_attempts"] and
                directory == root / f"attempt-{attempt.number:03}" and directory.resolve() == directory and directory.is_dir() and
                not list(directory.iterdir()), "new run-owned bounded attempt directory required")
        require(attempt.dataset_id == execution["binding"]["dataset_sha256"] and
                fingerprint(attempt.suite) == execution["binding"]["suite_sha256"], "attempt dataset/suite changed")
        argv = [str(directory) + item[len("{attempt}"):] if item.startswith("{attempt}/") else item
                for item in execution["argv_template"]]
        provenance = directory / "backend.json"
        record = {**preview, "authorization": approval, "argv": argv, "status": "runtime_verification"}
        def save():
            provenance.write_text(json.dumps(record, indent=2) + "\n")
        save()
        process = None
        try:
            # Only after authorization. This probe performs tiny primitives, not fitting.
            facts = probe_runtime(True)
            record["runtime_revalidation"] = facts
            save()
            require(runtime_contract(facts) == execution["binding"]["runtime_contract"], "runtime changed since preparation")
            require(self.preview()["execution_sha256"] == preview["execution_sha256"], "inputs changed during runtime verification")
            record["status"] = "preprocessing"
            imagefolder = directory / "imagefolder"
            imagefolder.mkdir()
            Image = pillow()
            dataset = json.loads(Path(json.loads(self.request.read_text())["preparation"]).read_text())["dataset"]
            rows = []
            for number, row in enumerate(row for row in dataset["records"] if row["split"] == "train"):
                source = Path(execution["roots"]["train"]) / row["file_name"]
                require(fingerprint(source) == row["sha256"], "image changed before preprocessing")
                with Image.open(source) as image:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, tuple(execution["preprocessing"]["background"]) + (255,))
                    rgb = Image.alpha_composite(background, rgba).convert("RGB")
                    size = execution["binding"]["config"]["training_resolution"]
                    rgb.resize((size, size), Image.Resampling.NEAREST).save(imagefolder / f"{number:06}.png")
                rows.append({"file_name": f"{number:06}.png", "text": row["text"]})
            (imagefolder / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            require(self.preview()["execution_sha256"] == preview["execution_sha256"], "inputs changed before launch")
            record.update(status="running", imagefolder_sha256=canonical_digest(
                {file.name: fingerprint(file) for file in sorted(imagefolder.iterdir())}))
            save()
            with (directory / "trainer.log").open("wb") as log:
                process = subprocess.Popen(argv, cwd=directory, env=execution["environment"],
                                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           shell=False, start_new_session=True)
                code = process.wait(timeout=execution["timeout_seconds"])
                require(code == 0, f"trainer failed with exit {code}")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            checkpoint = directory / "output" / "pytorch_lora_weights.safetensors"
            require(checkpoint.is_file() and checkpoint.resolve().is_relative_to(directory) and
                    not checkpoint.is_symlink() and checkpoint.stat().st_size > 0, "missing/empty/nonlocal final LoRA weights")
            record.update(status="completed", checkpoint_sha256=fingerprint(checkpoint))
            save()
            return checkpoint
        except BaseException as error:
            if process is not None:
                # Kill the whole Linux session on timeout/interruption/error, including descendants.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record["cleanup_pending"] = True
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="local adapter request JSON; preview only")
    args = parser.parse_args()
    print(json.dumps(LoraTrainer(args.request).preview(), indent=2))


if __name__ == "__main__":
    main()
