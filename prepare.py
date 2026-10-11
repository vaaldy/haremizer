"""Validate local inputs and write a human-gated preparation plan. Never train."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from images import validate_png
from runtime import probe_runtime


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("file_name must be a relative POSIX path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"path escape or ambiguous path: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"file missing or escapes dataset root: {name}")
    return path


def nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_dataset(train: Path, evaluation: Path, rights: Path,
                     sizes: tuple[int, ...] = (48, 64)) -> dict:
    records = []
    seen_bytes, seen_pixels = {}, {}
    for split, root in (("train", train), ("eval", evaluation)):
        metadata = local_file(root, "metadata.jsonl")
        names = set()
        for number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not nonempty(row.get("text")):
                raise ValueError(f"{split} metadata line {number}: nonempty text caption required")
            name = row.get("file_name")
            image = local_file(root, name)
            if image.suffix.lower() != ".png":
                raise ValueError(f"{split}/{name}: expected .png filename")
            if name in names:
                raise ValueError(f"duplicate metadata file_name: {split}/{name}")
            names.add(name)
            metrics = validate_png(image, sizes)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            identity = f"{split}/{name}"
            for seen, key in ((seen_bytes, digest), (seen_pixels, metrics["pixel_sha256"])):
                if key in seen:
                    raise ValueError(f"duplicate image: {seen[key]} and {identity}")
                seen[key] = identity
            records.append({"split": split, "file_name": name, "text": row["text"],
                            "sha256": digest, "metrics": metrics})
        if not names:
            raise ValueError(f"{split} split has no captioned PNGs")
    manifest = json.loads(rights.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int or
            manifest["version"] != 1 or not nonempty(manifest.get("reviewed_by")) or
            not isinstance(manifest.get("entries"), list)):
        raise ValueError("rights manifest requires version 1, reviewed_by, and entries")
    permissions = {}
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("rights entries must be objects")
        split, name = entry.get("split"), entry.get("file_name")
        if split not in ("train", "eval") or not isinstance(name, str):
            raise ValueError("rights entry requires split train/eval and file_name")
        key = (split, name)
        use = "training_allowed" if split == "train" else "evaluation_allowed"
        if (key in permissions or entry.get(use) is not True or
                not nonempty(entry.get("source")) or not nonempty(entry.get("license"))):
            raise ValueError(f"rights missing, denied, or duplicated for {split}/{name}")
        permissions[key] = entry
    if set(permissions) != {(row["split"], row["file_name"]) for row in records}:
        raise ValueError("rights entries must cover exactly the captioned train and eval files")
    records.sort(key=lambda row: (row["split"], row["file_name"]))
    normalized_rights = {"version": 1, "reviewed_by": manifest["reviewed_by"],
                         "entries": [permissions[key] for key in sorted(permissions)]}
    content = {"records": records, "rights": normalized_rights, "allowed_sizes": sorted(set(sizes))}
    return {**content, "sha256": canonical_digest(content),
            "rights_sha256": canonical_digest(normalized_rights),
            "rights_are_human_declarations": True}


def freeze_suite(path: Path) -> dict:
    raw = path.read_bytes()
    suite = json.loads(raw)
    if (not isinstance(suite, dict) or type(suite.get("version")) is not int or
            suite["version"] != 1 or not isinstance(suite.get("cases"), list) or
            not suite["cases"] or not isinstance(suite.get("target_layouts"), list)):
        raise ValueError("suite requires version 1, nonempty cases, and target_layouts")
    ids = set()
    for case in suite["cases"]:
        if (not isinstance(case, dict) or not nonempty(case.get("id")) or
                case["id"] in ids or not nonempty(case.get("prompt")) or
                type(case.get("seed")) is not int or not 0 <= case["seed"] < 2**32):
            raise ValueError("suite cases require unique id, prompt, and fixed uint32 seed")
        ids.add(case["id"])
    layouts = suite["target_layouts"]
    if (len(layouts) != 2 or any(not isinstance(layout, dict) or
            type(layout.get("size")) is not int or not nonempty(layout.get("composition"))
            for layout in layouts) or {layout["size"] for layout in layouts} != {48, 64}):
        raise ValueError("suite requires independent 48 and 64 target layouts")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "frozen_json": raw.decode("utf-8")}


def build_plan(*, train: Path | None, evaluation: Path | None, rights: Path | None,
               suite: Path, model: str | None, revision: str | None,
               torch_probe: bool = False, steps: int = 100, rank: int = 4,
               training_resolution: int = 512, sizes: tuple[int, ...] = (48, 64)) -> dict:
    blockers = []
    if not nonempty(model) or not nonempty(revision):
        blockers.append("explicit base model and revision are required; there is no default model")
    revision_valid = isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision) is not None
    if nonempty(revision) and not revision_valid:
        blockers.append("model revision must be a full 40/64-hex immutable commit/content digest, not a branch or tag")
    config_valid = (type(steps) is int and 1 <= steps <= 1000 and type(rank) is int and
                    1 <= rank <= 16 and type(training_resolution) is int and
                    64 <= training_resolution <= 1024 and training_resolution % 8 == 0)
    if not config_valid:
        blockers.append("config requires steps 1..1000, rank 1..16, resolution 64..1024 divisible by 8")
    dataset = None
    if train is None or evaluation is None or rights is None:
        blockers.append("local train/eval datasets and an explicit rights manifest are required")
    else:
        try:
            dataset = validate_dataset(train, evaluation, rights, sizes)
        except (ValueError, RuntimeError, OSError, TypeError) as error:
            blockers.append(f"dataset/rights: {error}")
    frozen = None
    try:
        frozen = freeze_suite(suite)
    except (ValueError, OSError, TypeError) as error:
        blockers.append(f"evaluation suite: {error}")
    runtime = probe_runtime(torch_probe)
    if runtime.get("gpu_operations_verified") is not True:
        blockers.append("GPU tensor math/inference/backward operations are not verified")
    tensor_evidence = runtime.get("torch_probe", {}).get("evidence", {})
    runtime_contract = {
        "python": runtime.get("python"), "python_executable": runtime.get("python_executable"),
        "torch_version": tensor_evidence.get("torch_version"),
        "hip_version": tensor_evidence.get("hip_version"),
        "cuda_version": tensor_evidence.get("cuda_version"), "device": tensor_evidence.get("device"),
        "gpu_visible": tensor_evidence.get("gpu_visible"),
        "gpu_operations_verified": runtime.get("gpu_operations_verified") is True,
        "environment": runtime.get("environment", {}), "probe_mode": runtime.get("probe_mode"),
        "compatibility_workaround_requires_human_acknowledgment": runtime.get(
            "compatibility_workaround_requires_human_acknowledgment", False),
    }
    model_spec = {"id": model, "revision": revision}
    config = {"batch_size": 1, "lora_rank": rank, "max_steps": steps,
              "seed": 1234, "training_resolution": training_resolution,
              "output_target_sizes": [48, 64],
              "output_policy": "independent native target layouts; not automatic downscaling"}
    binding = ({"dataset_sha256": dataset["sha256"], "suite_sha256": frozen["sha256"],
                "model": model_spec, "config": config, "runtime_contract": runtime_contract}
               if dataset and frozen and nonempty(model) and revision_valid and config_valid else None)
    return {
        "version": 1, "status": "blocked" if blockers else "needs_human",
        "blockers": blockers, "training_authorized": False,
        "training_started": False, "training_command": None,
        "preparation_sha256": canonical_digest(binding) if binding else None,
        "preparation_binding": binding,
        "model": model_spec, "dataset": dataset,
        "evaluation_suite": frozen, "runtime": runtime, "runtime_contract": runtime_contract,
        "config": config,
        "quality_assessments": {name: "unassessed" for name in (
            "pixel_quality", "prompt_adherence", "character_diversity", "non_regression")},
        "generated_data_feedback": "disabled; never automatically add generated outputs to training data",
        "human_gate": [
            "Confirm source rights declarations, caption accuracy, adult subjects, and held-out separation.",
            "Approve base-model license and immutable revision; no model has been downloaded or tested.",
            "If runtime evidence uses HSA_OVERRIDE_GFX_VERSION, explicitly acknowledge the unsupported compatibility workaround before any separate training authorization; a workaround pass is not a native-runtime pass.",
            "Select and pin a real trainer and its dependency versions; approve compute/time/memory limits.",
            "Implement and calibrate independent visual evaluation; PNG metrics are not aesthetic or semantic passes.",
            "Preparation/setup approval is not training authorization. Explicitly authorize a separate training implementation/run bound to preparation_sha256; this CLI cannot authorize or launch it.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--eval", dest="evaluation", type=Path)
    parser.add_argument("--rights", type=Path)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("evaluation-suite.json"))
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--torch-probe", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--training-resolution", type=int, default=512)
    parser.add_argument("--size", type=int, choices=(48, 64), action="append")
    parser.add_argument("--output", type=Path, default=Path("artifacts/preparation.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() or output in {path.resolve() for path in (args.rights, args.suite) if path} or
            any(output.is_relative_to(root.resolve()) for root in (args.train, args.evaluation) if root)):
        parser.exit(2, "output must be new and outside dataset roots and input files\n")
    plan = build_plan(train=args.train, evaluation=args.evaluation, rights=args.rights,
                      suite=args.suite, model=args.model, revision=args.revision,
                      torch_probe=args.torch_probe, steps=args.steps, rank=args.rank,
                      training_resolution=args.training_resolution, sizes=tuple(args.size or (48, 64)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": plan["status"], "output": str(output),
                      "blockers": plan["blockers"], "training_authorized": False,
                      "training_started": False}, indent=2))
    if plan["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
