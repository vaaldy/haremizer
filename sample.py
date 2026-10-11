"""Bounded, local safetensors SD1/SDXL inference; raw PNGs, not validated sprites."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from prepare import freeze_suite

PIPELINES = ("StableDiffusionPipeline", "StableDiffusionXLPipeline")
COMPONENTS = {
    "diffusers": {"AutoencoderKL", "UNet2DConditionModel", "DDIMScheduler", "DDPMScheduler",
                  "PNDMScheduler", "LMSDiscreteScheduler", "EulerDiscreteScheduler",
                  "EulerAncestralDiscreteScheduler", "DPMSolverMultistepScheduler",
                  "HeunDiscreteScheduler", "UniPCMultistepScheduler"},
    "transformers": {"CLIPTextModel", "CLIPTextModelWithProjection", "CLIPTokenizer",
                     "CLIPImageProcessor", "CLIPFeatureExtractor", "CLIPVisionModelWithProjection"},
    "stable_diffusion": {"StableDiffusionSafetyChecker"},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def model_manifest(path):
    index = json.loads((path / "model_index.json").read_bytes())
    if not isinstance(index, dict) or index.get("_class_name") not in PIPELINES:
        raise ValueError("only standard SD1/SDXL pipeline classes are allowed")
    for name, component in index.items():
        if name in ("_class_name", "_diffusers_version", "_name_or_path"):
            continue
        if name in ("requires_safety_checker", "force_zeros_for_empty_prompt", "add_watermarker"):
            if component is not None and type(component) is not bool:
                raise ValueError(f"invalid standard pipeline setting: {name}")
            continue
        if (not isinstance(component, list) or len(component) != 2 or
                (component != [None, None] and
                 (not isinstance(component[0], str) or not isinstance(component[1], str) or
                  component[1] not in COMPONENTS.get(component[0], set())))):
            raise ValueError(f"custom or unsupported model component: {name}")
    files = {}
    for file in sorted(path.rglob("*")):
        if file.is_file():
            if file.suffix == ".py":
                raise ValueError("model cache must not contain Python code")
            if file.suffix.lower() in (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"):
                raise ValueError("pickle checkpoints forbidden, including component fallback weights")
            files[file.relative_to(path).as_posix()] = {"bytes": file.stat().st_size}
            if file.suffix == ".json":
                files[file.relative_to(path).as_posix()]["sha256"] = digest(file.read_bytes())
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("local safetensors weights required; no checkpoint conversion")
    return {"class": index["_class_name"], "files": files,
            "local_manifest_sha256": digest(json.dumps(files, sort_keys=True).encode()),
            "identity_scope": "file names/sizes and JSON hashes; not weight-content attestation"}


def validate_lora(lora, lora_sha256):
    if (lora is None) != (lora_sha256 is None):
        raise ValueError("lora and lora_sha256 must be supplied together")
    if lora is None:
        return None, None
    if (not isinstance(lora_sha256, str) or len(lora_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in lora_sha256)):
        raise ValueError("lora_sha256 must be a full lowercase 64-hex SHA256")
    path = Path(lora).resolve()
    limit = 64 * 1024 * 1024
    if (path.suffix != ".safetensors" or not path.is_file() or
            not 0 < path.stat().st_size <= limit):
        raise ValueError("lora must be a nonempty local safetensors file <=64 MiB")
    with path.open("rb") as file:
        data = file.read(limit + 1)
    if not 0 < len(data) <= limit or digest(data) != lora_sha256:
        raise ValueError("lora bytes must be <=64 MiB and match lora_sha256")
    return path, {"path": str(lora), "bytes": len(data), "sha256": lora_sha256}


def sample(model, output, *, suite=None, prompt=None, seed=1234, max_cases=4, steps=20,
           width=None, height=None, variant=None, device="cuda", cpu_offload=False,
           model_id=None, revision=None, negative_prompt=None, guidance_scale=7.5,
           lora=None, lora_sha256=None):
    if negative_prompt is not None and (not isinstance(negative_prompt, str) or
                                        not negative_prompt.strip()):
        raise ValueError("negative_prompt must be None or a nonblank string")
    if (type(guidance_scale) not in (int, float) or not 1 <= guidance_scale <= 15 or
            not math.isfinite(guidance_scale)):
        raise ValueError("guidance_scale must be a finite number from 1..15")
    if type(max_cases) is not int or not 1 <= max_cases <= 4:
        raise ValueError("max_cases must be 1..4")
    if type(steps) is not int or not 1 <= steps <= 40:
        raise ValueError("steps must be 1..40")
    if device != "cuda" or variant not in (None, "fp16") or type(cpu_offload) is not bool:
        raise ValueError("CUDA/ROCm GPU only; variant must be fp16 or omitted")
    if revision is not None and (len(revision) not in (40, 64) or
                                any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("revision must be a full lowercase 40/64-hex immutable revision")
    if (suite is None) == (prompt is None):
        raise ValueError("supply exactly one frozen suite or prompt")
    frozen = freeze_suite(Path(suite)) if suite is not None else None
    cases = json.loads(frozen["frozen_json"])["cases"] if frozen else [
        {"id": "prompt", "prompt": prompt, "seed": seed}]
    if not frozen and (not isinstance(prompt, str) or not prompt.strip() or
                       type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("nonempty prompt and fixed uint32 seed required")
    model, output = Path(model).resolve(), Path(output).resolve()
    if not model.is_dir() or output.exists() or output.is_relative_to(model):
        raise ValueError("local model directory and new output directory outside model required")
    lora_path, lora_identity = validate_lora(lora, lora_sha256)
    if lora_path is not None and lora_path.is_relative_to(output):
        raise ValueError("output directory must not contain the lora source")
    identity = model_manifest(model)
    default_size = 768 if identity["class"] == "StableDiffusionXLPipeline" else 512
    width, height = default_size if width is None else width, default_size if height is None else height
    if any(type(n) is not int or not 64 <= n <= 1024 or n % 64 for n in (width, height)):
        raise ValueError("dimensions must be multiples of 64 from 64..1024")
    import torch
    import diffusers
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/ROCm GPU available; no CPU fallback")
    output.mkdir(parents=True, exist_ok=False)
    pipeline = getattr(diffusers, identity["class"]).from_pretrained(
        str(model), local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
        **({"variant": variant} if variant else {}))
    if lora_path is not None:
        validate_lora(lora_path, lora_sha256)
        pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name,
                                   local_files_only=True, use_safetensors=True)
    if cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=0, device="cuda")
    else:
        pipeline.to("cuda")
    report = {"model": {**identity, "path": str(model), "id": model_id or model.name,
                        "revision": revision}, "suite": frozen,
              "suite_sha256": frozen["sha256"] if frozen else None,
              "lora": lora_identity,
              "settings": {"steps": steps, "width": width, "height": height,
                           "max_cases": max_cases, "dtype": "float16", "device": device,
                           "variant": variant, "model_cpu_offload": cpu_offload,
                           "negative_prompt": negative_prompt,
                           "guidance_scale": guidance_scale, "num_images_per_prompt": 1},
              "runtime": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                          "hip": torch.version.hip, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0),
                          "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION")},
              "output_contract": ("raw inference PNG" if lora_path else "raw pretrained PNG") +
                                 "; alpha may be opaque; not final 48/64 indexed sprites",
              "visual_quality_assessed": False, "sprite_format_validated": False,
              "training_started": False, "records": []}
    for number, case in enumerate(cases[:max_cases], 1):
        torch.cuda.synchronize()
        memory = {"scope": "PyTorch allocator per case; not total GPU usage or a memory cap"}
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except (RuntimeError, AttributeError) as error:
            memory["error"] = str(error)
        started = time.monotonic()
        with torch.inference_mode():
            image = pipeline(prompt=case["prompt"], generator=torch.Generator(device="cuda").manual_seed(case["seed"]),
                             num_inference_steps=steps, width=width, height=height,
                             negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                             num_images_per_prompt=1).images[0]
        torch.cuda.synchronize()
        duration = time.monotonic() - started
        if "error" not in memory:
            try:
                memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            except (RuntimeError, AttributeError) as error:
                memory["error"] = str(error)
        if image.size != (width, height):
            raise RuntimeError("pipeline returned unexpected image dimensions")
        file = output / f"case-{number:02d}.png"
        image.save(file, format="PNG")
        report["records"].append({**case, "file": file.name, "sha256": digest(file.read_bytes()),
                                  "width": image.width, "height": image.height, "mode": image.mode,
                                  "duration_seconds": duration, "gpu_memory": memory})
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--suite", type=Path)
    inputs.add_argument("--prompt")
    for name, default in (("seed", 1234), ("max-cases", 4), ("steps", 20), ("width", None), ("height", None)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--negative-prompt", help="nonblank negative prompt; omitted by default")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="finite guidance scale from 1..15 (default: 7.5)")
    parser.add_argument("--variant", choices=("fp16",))
    parser.add_argument("--device", choices=("cuda",), default="cuda", help="CUDA or ROCm; GPU 0 only")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--lora", type=Path, help="local safetensors adapter <=64 MiB; requires --lora-sha256")
    parser.add_argument("--lora-sha256", help="full lowercase SHA256 of the supplied adapter")
    try:
        report = sample(**vars(parser.parse_args()))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
