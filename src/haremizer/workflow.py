"""Bounded local neutral portrait -> explicit CPU color-key recovery -> sprite export."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .core import fingerprint
from .images import validate_png
from .process_guard import run_process
from .recovery import recover
from .refinement_loop import Loop, check_identity, identity, tree, write_json
from .sample import model_manifest, validate_lora
from .sprite_export import export_sprite

STORAGE_ROOTS = {"project", "runtime", "uv", "python", "coredump", "miopen"}


def resolve(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty path required")
    return (base / value).resolve()


def load_recipe(path):
    recipe = json.loads(path.read_bytes())
    for key in ("model", "prompt", "negative_prompt", "seed", "steps", "width", "height",
                "guidance_scale", "scheduler", "cpu_offload", "max_cases", "recovery", "export"):
        if key not in recipe:
            raise ValueError(f"recipe missing {key}")
    if (recipe["scheduler"] != "euler_ancestral" or recipe["cpu_offload"] is not True or
            recipe["max_cases"] != 1 or recipe.get("layerdiffuse") is not None or
            recipe["export"] != {"name": "neutral"}):
        raise ValueError("one neutral, CPU-offloaded Euler ancestral case without LayerDiffuse required")
    for key in ("prompt", "negative_prompt"):
        if not isinstance(recipe[key], str) or not recipe[key].strip():
            raise ValueError(f"nonblank {key} required")
    for key, low, high in (("seed", 0, 2**32 - 1), ("steps", 1, 40), ("width", 64, 768), ("height", 64, 768)):
        if type(recipe[key]) is not int or not low <= recipe[key] <= high:
            raise ValueError(f"invalid {key}")
    if recipe["width"] != recipe["height"] or recipe["width"] % 64:
        raise ValueError("square multiple-of-64 dimensions required")
    cfg = recipe["guidance_scale"]
    if type(cfg) not in (int, float) or not math.isfinite(cfg) or not 1 <= cfg <= 15:
        raise ValueError("invalid guidance_scale")
    recovery = recipe["recovery"]
    if (not isinstance(recovery, dict) or recovery.get("size") != 64 or
            type(recovery.get("tolerance")) is not int or not 0 <= recovery["tolerance"] <= 64 or
            type(recovery.get("colors")) is not int or not 2 <= recovery["colors"] <= 62 or
            not isinstance(recovery.get("background"), list) or len(recovery["background"]) != 3 or
            any(type(c) is not int or not 0 <= c <= 255 for c in recovery["background"])):
        raise ValueError("explicit RGB recovery, size64, tolerance0..64, colors2..62 required")
    spec = recipe["model"]
    if (not isinstance(spec, dict) or not isinstance(spec.get("id"), str) or
            not isinstance(spec.get("revision"), str) or len(spec["revision"]) not in (40, 64) or
            any(c not in "0123456789abcdef" for c in spec["revision"]) or
            not isinstance(spec.get("weight_sha256"), dict) or not spec["weight_sha256"]):
        raise ValueError("local model identity/revision and weight hashes required")
    return recipe


def run(config_path, output):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    if set(config) != {"recipe", "recipe_sha256", "timeout_seconds", "storage_roots"}:
        raise ValueError("config requires recipe, recipe_sha256, timeout_seconds, storage_roots only")
    timeout = config["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError("timeout_seconds must be 1..600")
    roots = config["storage_roots"]
    if not isinstance(roots, dict) or set(roots) != STORAGE_ROOTS:
        raise ValueError("six separate storage roots required: " + ", ".join(sorted(STORAGE_ROOTS)))
    roots = [resolve(config_path.parent, roots[key]) for key in sorted(roots)]
    if any(not p.exists() for p in roots):
        raise ValueError("storage roots must exist")
    recipe_path = resolve(config_path.parent, config["recipe"])
    if fingerprint(recipe_path) != config["recipe_sha256"]:
        raise ValueError("recipe hash mismatch")
    recipe = load_recipe(recipe_path)
    model = resolve(recipe_path.parent, recipe["model"]["path"])
    if not model.is_dir():
        raise ValueError("local model directory required")
    model_manifest(model)
    bindings = [identity(config_path), identity(recipe_path), identity(Path(sys.executable).resolve())]
    bindings += [identity(p) for p in sorted(Path(__file__).resolve().parent.iterdir())
                 if p.is_file() and p.suffix in (".py", ".json")]
    bindings += tree(model)
    for name, expected in recipe["model"]["weight_sha256"].items():
        relative = Path(name)
        weight = model / relative
        if relative.is_absolute() or ".." in relative.parts or fingerprint(weight) != expected:
            raise ValueError(f"model weight hash mismatch: {name}")
    lora = recipe.get("lora")
    lora_path = resolve(recipe_path.parent, lora["path"]) if lora else None
    validate_lora(lora_path, lora["sha256"] if lora else None)
    if lora_path:
        bindings.append(identity(lora_path))
    requested = Path(output).absolute()
    output = requested.resolve()
    if requested.exists() or requested.is_symlink() or output.exists():
        raise ValueError("output must be a new directory")
    if any(Path(b["path"]).is_relative_to(output) or output.is_relative_to(Path(b["path"]))
           for b in bindings) or output.is_relative_to(model) or output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("output collides with input")
    if any(not any(p.is_relative_to(root) for root in roots) for p in (model, output, config_path, Path(sys.executable).resolve())):
        raise ValueError("storage roots must cover model, output, config and interpreter")
    if lora_path and not any(lora_path.is_relative_to(root) for root in roots):
        raise ValueError("storage roots must cover LoRA")
    plan = {"storage_paths": [str(p) for p in roots]}
    Loop.storage(plan)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "config": str(config_path), "bindings": bindings,
              "module_origin": __file__, "processes": [], "directory": str(output),
              "training_started": False, "visual_quality_assessed": False,
              "alpha_policy": "explicit CPU color-key recovery; not native learned transparency"}
    def save():
        write_json(output / "workflow.json", report)
    def bounds():
        for binding in bindings:
            check_identity(binding)
        report["storage"] = Loop.storage(plan)
    save()
    try:
        # The isolated child verifies every bound byte before importing the ML runtime.
        bootstrap = (
            "import json,pathlib,runpy,sys,haremizer; "
            f"assert pathlib.Path(haremizer.__file__).resolve().parent == pathlib.Path({str(Path(__file__).resolve().parent)!r}), 'package origin mismatch'; "
            "from haremizer.refinement_loop import check_identity; "
            f"[check_identity(b) for b in json.loads(pathlib.Path({str(output / 'workflow.json')!r}).read_text())['bindings']]; "
            "import torch; torch.cuda.init(); "
            "assert torch.cuda.is_available(), 'GPU required; no CPU fallback'; "
            "torch.cuda.set_per_process_memory_fraction(min(1.0,11*1024**3/torch.cuda.get_device_properties(0).total_memory),0); "
            "sys.argv=['haremizer.sample',*sys.argv[1:]]; runpy.run_module('haremizer.sample',run_name='__main__')"
        )
        report["child_bootstrap_sha256"] = hashlib.sha256(bootstrap.encode()).hexdigest()
        argv = [sys.executable, "-I", "-B", "-c", bootstrap, "--model", str(model),
                "--output", str(output / "raw"), "--model-id", recipe["model"]["id"],
                "--revision", recipe["model"]["revision"], "--cpu-offload"]
        for key in ("prompt", "negative_prompt", "seed", "steps", "width", "height", "guidance_scale", "scheduler", "max_cases"):
            argv += ["--" + key.replace("_", "-"), str(recipe[key])]
        if lora_path:
            argv += ["--lora", str(lora_path), "--lora-sha256", lora["sha256"]]
        run_process(report, argv, timeout, "sample", bounds=bounds, save=save,
                    deadline=time.time() + timeout)
        manifest = json.loads((output / "raw/manifest.json").read_bytes())
        if not isinstance(manifest, dict):
            raise ValueError("sampler manifest must be an object")
        settings = manifest.get("settings", {})
        expected_settings = {key: recipe[key] for key in ("width", "height", "steps", "guidance_scale", "max_cases", "negative_prompt")}
        expected_settings.update(model_cpu_offload=True, scheduler_override="euler_ancestral", device="cuda")
        expected_model = {"path": str(model), "id": recipe["model"]["id"], "revision": recipe["model"]["revision"]}
        if (any(settings.get(key) != value for key, value in expected_settings.items()) or
                any(manifest.get("model", {}).get(key) != value for key, value in expected_model.items()) or
                (manifest.get("lora", {}).get("sha256") if manifest.get("lora") else None) != (lora["sha256"] if lora else None)):
            raise ValueError("sampler manifest identity/settings mismatch")
        records = manifest.get("records")
        raw = output / "raw/case-01.png"
        if (not isinstance(records, list) or len(records) != 1 or records[0].get("file") != raw.name or
                records[0].get("sha256") != fingerprint(raw) or records[0].get("seed") != recipe["seed"] or
                records[0].get("prompt") != recipe["prompt"]):
            raise ValueError("incomplete or mismatched sampler manifest")
        from .images import pillow
        with pillow().open(raw) as image:
            image.verify()
        with pillow().open(raw) as image:
            if image.format != "PNG" or image.size != (recipe["width"], recipe["height"]) or getattr(image, "n_frames", 1) != 1:
                raise ValueError("invalid raw PNG geometry")
        recovery = recipe["recovery"]
        recover(raw, output / "recovery", recovery["background"], size=64,
                tolerance=recovery["tolerance"], colors=recovery["colors"])
        validate_png(output / "recovery/recovered.png", (64,))
        export_sprite(output / "recovery/recovered.png", output / "character", name="neutral")
        bounds()
        report["artifacts"] = tree(output / "raw") + tree(output / "recovery") + tree(output / "character") + [identity(output / "character.provenance.json")]
        report["status"] = "completed_unreviewed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new immutable run directory")
    args = parser.parse_args()
    try:
        report = run(args.config, args.output)
    except (ValueError, RuntimeError, OSError, ImportError, TimeoutError, KeyError, TypeError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
