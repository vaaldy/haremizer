"""Local GPU-only SDXL attention-LoRA plumbing smoke; never quality acceptance."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

from prepare import canonical_digest, nonempty
from sample import model_manifest

CONFIG = {"batch_size": 1, "resolution": 512, "rank": 4, "seed": 1234,
          "learning_rate": 1e-4, "gradient_clip": 1.0}
VERSIONS = {"torch": "2.11.0+rocm7.2", "diffusers": "0.41.0", "transformers": "5.19.0",
            "peft": "0.21.2", "accelerate": "1.15.0"}
TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
LIMIT = 11 * 1024**3


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def device_usage(torch):
    """Optional management-library metric; allocator limits remain mandatory."""
    try:
        return {"device_used_bytes": torch.cuda.device_memory_used(0)}
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        return {"device_used_bytes": None,
                "device_usage_error": f"{type(error).__name__}: {error}"}


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")


def preflight(request_path, output):
    """Validate local declarations and byte identities without importing ML packages."""
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    raw = request_path.read_bytes()
    request = json.loads(raw)
    keys(request, {"model", "config", "examples", "reviewed_by", "declared_smoke_only",
                   "authorization"}, "request")
    if request["reviewed_by"] != "machine:sol" or request["declared_smoke_only"] is not True:
        raise ValueError("truthful machine:sol review and declared_smoke_only=true required")
    authorization = request["authorization"]
    keys(authorization, {"user_authorized", "record"}, "authorization")
    if authorization["user_authorized"] is not True or not nonempty(authorization["record"]):
        raise ValueError("record the existing broad user authorization; do not invent human approval")
    config = request["config"]
    keys(config, {*CONFIG, "steps"}, "config")
    if any((type(config[k]) not in ((int, float) if type(v) is float else (int,))) or
           config[k] != v for k, v in CONFIG.items()):
        raise ValueError("smoke config must use the fixed batch/resolution/rank/seed/lr/clip")
    if type(config["steps"]) is not int or not 1 <= config["steps"] <= 2:
        raise ValueError("steps must be 1..2 total optimizer steps")
    spec = request["model"]
    keys(spec, {"path", "id", "revision"}, "model")
    if (not nonempty(spec["path"]) or not isinstance(spec["id"], str) or
            re.fullmatch(r"[^/\s]+/[^/\s]+", spec["id"]) is None or
            not isinstance(spec["revision"], str) or
            re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", spec["revision"]) is None):
        raise ValueError("explicit local model, OWNER/MODEL and immutable lowercase revision required")
    root = request_path.parent

    def local(name):
        if not nonempty(name):
            raise ValueError("nonempty local path required")
        return (root / name).resolve()

    model = local(spec["path"])
    if not model.is_dir() or output.exists() or output.is_relative_to(model):
        raise ValueError("local model and new output outside model required")
    identity = model_manifest(model)
    if identity["class"] != "StableDiffusionXLPipeline":
        raise ValueError("SDXL only")
    index = json.loads((model / "model_index.json").read_bytes())
    for name, component in {"unet": ["diffusers", "UNet2DConditionModel"],
                            "vae": ["diffusers", "AutoencoderKL"],
                            "text_encoder": ["transformers", "CLIPTextModel"],
                            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
                            "tokenizer": ["transformers", "CLIPTokenizer"],
                            "tokenizer_2": ["transformers", "CLIPTokenizer"]}.items():
        if index.get(name) != component:
            raise ValueError(f"standard dual-encoder SDXL component required: {name}")
    hashes = {name: file_hash(model / name) for name in identity["files"]}
    revision = spec["revision"]
    if len(revision) == 40:
        if (model.name != revision or model.parent.name != "snapshots" or
                model.parent.parent.name != "models--" + spec["id"].replace("/", "--")):
            raise ValueError("model path must match pinned HF cache id/revision convention")
    elif canonical_digest(hashes) != revision:
        raise ValueError("64-hex model content revision mismatch")
    examples = request["examples"]
    if not isinstance(examples, list) or not 1 <= len(examples) <= 2:
        raise ValueError("supply 1..2 local PNG examples")
    records, seen = [], set()
    from PIL import Image
    for row in examples:
        keys(row, {"path", "sha256", "caption", "source_url", "license", "credit",
                   "source_page"}, "example")
        source = row["source_page"]
        keys(source, {"path", "sha256"}, "source_page")
        url = urlparse(row["source_url"] if isinstance(row["source_url"], str) else "")
        if (not nonempty(row["caption"]) or not nonempty(row["credit"]) or
                row["license"] not in ("CC-BY-4.0", "CC0-1.0") or
                url.scheme not in ("http", "https") or not url.netloc):
            raise ValueError("nonblank caption/credit, source URL and CC-BY-4.0 or CC0-1.0 required")
        image, page = local(row["path"]), local(source["path"])
        for path, expected in ((image, row["sha256"]), (page, source["sha256"])):
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"missing input or SHA256 mismatch: {path}")
            if path.is_relative_to(output):
                raise ValueError("output must not contain inputs")
        if row["sha256"] in seen:
            raise ValueError("duplicate example")
        seen.add(row["sha256"])
        with Image.open(image) as png:
            if png.format != "PNG" or png.size != (64, 64) or getattr(png, "n_frames", 1) != 1:
                raise ValueError("single-frame native 64x64 PNG required")
            png.load()
        records.append({**row, "path": str(image), "source_page": {**source, "path": str(page)}})
    if request_path.is_relative_to(output):
        raise ValueError("output must not contain request")
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override not in (None, "10.3.0"):
        raise ValueError("only the authorized process-local gfx 10.3.0 override is supported")
    return {"request": request, "request_sha256": hashlib.sha256(raw).hexdigest(),
            "config_sha256": canonical_digest(config), "model": {**spec, "path": str(model),
            "manifest": identity, "file_sha256": hashes,
            "provenance_scope": "HF path asserts revision; content hashes bind local bytes, not upstream authenticity"},
            "inputs": records, "script_sha256": file_hash(__file__),
            "runtime_override": override,
            "override_scope": "existing process-local environment only; unsupported workaround, not native support",
            "quality_assessed": False,
            "promotion": False, "synthetic_feedback": False}


def save_adapter(unet, output):
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict
    state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(unet).items()}
    StableDiffusionXLPipeline.save_lora_weights(
        str(output), unet_lora_layers=convert_state_dict_to_diffusers(state),
        safe_serialization=True, weight_name="pytorch_lora_weights.safetensors")
    return state


def reload_adapter(unet, output, expected):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    unet.delete_adapters("default")
    # The loader follows the base dtype. Temporarily upcast on CPU so fp32
    # checkpoint values are not rounded to fp16 during adapter injection.
    unet.float()
    state = load_file(str(output / "pytorch_lora_weights.safetensors"), device="cpu")
    StableDiffusionXLPipeline.load_lora_into_unet(state, network_alphas=None, unet=unet,
                                               adapter_name="default")
    for param in unet.parameters():
        if not param.requires_grad:
            param.data = param.data.half()
    actual = get_peft_model_state_dict(unet)
    if set(actual) != set(expected) or any(
            not torch.equal(actual[k].detach().cpu(), expected[k]) for k in expected):
        raise RuntimeError("saved/reloaded adapter mismatch")


def train(report, output, persist):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU training fallback")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.cuda.reset_peak_memory_stats()
    model = Path(report["model"]["path"])
    runtime = report["runtime"]
    runtime.update(gpu=torch.cuda.get_device_name(0), hip=torch.version.hip, cuda=torch.version.cuda)

    def finite(*tensors):
        if any(not torch.isfinite(t).all().item() for t in tensors):
            raise RuntimeError("nonfinite tensor/loss/gradient/weight")

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  **device_usage(torch),
                  "device_free_bytes": free, "device_total_bytes": total,
                  "scope": "allocator peaks; device usage is an instantaneous all-process reading"}
        runtime["memory"] = result
        if result["peak_allocated_bytes"] > LIMIT:
            raise RuntimeError("peak allocated exceeds 11 GiB; stop, no retry")

    def frozen_hash(unet):
        value = hashlib.sha256()
        count = 0
        for name, param in unet.named_parameters():
            if not param.requires_grad:
                finite(param)
                value.update(name.encode())
                value.update(str((tuple(param.shape), param.dtype)).encode())
                value.update(param.detach().cpu().contiguous().numpy().tobytes())
                count += param.numel()
        return {"sha256": value.hexdigest(), "parameters": count,
                "scope": "all frozen named UNet parameters, streamed CPU bytes; not buffers"}

    cache = [{"tokens": []} for _ in report["inputs"]]
    load = {"local_files_only": True, "use_safetensors": True}
    report["phase"] = "conditioning_cache"
    persist()
    # Encode the two token sequences separately, then concatenate penultimate hidden states.
    with torch.no_grad():
        for sub, cls in (("text_encoder", CLIPTextModel),
                         ("text_encoder_2", CLIPTextModelWithProjection)):
            tokenizer = CLIPTokenizer.from_pretrained(str(model / sub.replace("text_encoder", "tokenizer")),
                                                       local_files_only=True)
            encoder = cls.from_pretrained(str(model / sub), torch_dtype=torch.float16, **load).to("cuda")
            encoder.eval().requires_grad_(False)
            for item, row in zip(cache, report["inputs"]):
                ids = tokenizer(row["caption"], padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                encoded = encoder(ids.to("cuda"), output_hidden_states=True)
                finite(encoded.hidden_states[-2])
                item["tokens"].append(encoded.hidden_states[-2].cpu())
                if sub == "text_encoder_2":
                    finite(encoded.text_embeds)
                    item["pooled"] = encoded.text_embeds.cpu()
            del encoded, ids, encoder, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            memory()
        vae = AutoencoderKL.from_pretrained(str(model / "vae"), torch_dtype=torch.float32, **load).to("cuda")
        vae.eval().requires_grad_(False)
        report["conditioning"] = {"tokens": "dual penultimate hidden states concatenated",
                                  "pooled": "CLIPTextModelWithProjection.text_embeds",
                                  "time_ids": [512, 512, 0, 0, 512, 512],
                                  "latent_encoding": "fp32 VAE posterior mode * scaling_factor",
                                  "scaling_factor": vae.config.scaling_factor}
        for index, (item, row) in enumerate(zip(cache, report["inputs"])):
            source = Path(row["path"])
            if file_hash(source) != row["sha256"]:
                raise ValueError("input changed after preflight")
            original = output / f"source-{index + 1}.png"
            original.write_bytes(source.read_bytes())
            if file_hash(original) != row["sha256"]:
                raise ValueError("source changed while caching")
            with Image.open(original) as image:
                rgba = image.convert("RGBA").resize((512, 512), Image.Resampling.NEAREST)
                rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            cached = output / f"input-{index + 1}.png"
            rgb.save(cached)
            row["cached_input"] = {"path": cached.name, "sha256": file_hash(cached),
                                   "transform": "64->512 nearest; alpha flattened onto white RGB",
                                   "original_preserved": original.name, "original_sha256": file_hash(original)}
            pixels = torch.tensor(list(rgb.getdata()), dtype=torch.float32).reshape(512, 512, 3)
            pixels = (pixels.permute(2, 0, 1).unsqueeze(0).to("cuda") / 127.5) - 1
            latent = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
            finite(latent, item["pooled"], *item["tokens"])
            item["latents"] = latent.cpu().to(torch.float16)
            item["tokens"] = torch.cat(item["tokens"], dim=-1)
            item["time_ids"] = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=torch.float16)
        del pixels, latent, vae
    gc.collect()
    torch.cuda.empty_cache()
    memory()
    report["phase"] = "unet_setup"
    persist()
    scheduler = DDPMScheduler.from_pretrained(str(model / "scheduler"), local_files_only=True)
    prediction = scheduler.config.prediction_type
    if prediction not in ("epsilon", "v_prediction"):
        raise ValueError("only epsilon/v_prediction scheduler targets supported")
    report["prediction_type"] = prediction
    unet = UNet2DConditionModel.from_pretrained(str(model / "unet"), torch_dtype=torch.float16, **load)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights=True, target_modules=TARGETS))
    for name, param in unet.named_parameters():
        if param.requires_grad:
            if ".lora_A." not in name and ".lora_B." not in name:
                raise RuntimeError("unexpected non-adapter trainable parameter")
            param.data = param.data.float()
    before = frozen_hash(unet)
    report["frozen_before"] = before
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(unet).items()}
    unet.to("cuda").train()
    unet.enable_gradient_checkpointing()
    adapters = [p for p in unet.parameters() if p.requires_grad]
    if not adapters:
        raise RuntimeError("no trainable adapters")
    report["parameters"] = {"trainable": sum(p.numel() for p in adapters),
                            "frozen": before["parameters"], "targets": TARGETS,
                            "adapter_dtype": "float32", "base_dtype": "float16"}
    optimizer = torch.optim.AdamW(adapters, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device="cuda").manual_seed(1234)
    report["steps"] = []
    memory()
    report.update(training_started=True, phase="training")
    persist()
    for step in range(report["request"]["config"]["steps"]):
        started = time.monotonic()
        item = {k: v.to("cuda") for k, v in cache[step % len(cache)].items()}
        noise = torch.randn(item["latents"].shape, generator=generator, device="cuda", dtype=torch.float16)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=generator, device="cuda")
        noisy = scheduler.add_noise(item["latents"], noise, timestep)
        target = noise if prediction == "epsilon" else scheduler.get_velocity(item["latents"], noise, timestep)
        finite(*item.values(), noise, noisy, target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            predicted = unet(noisy, timestep, encoder_hidden_states=item["tokens"],
                             added_cond_kwargs={"text_embeds": item["pooled"], "time_ids": item["time_ids"]}).sample
            loss = torch.nn.functional.mse_loss(predicted.float(), target.float())
        finite(predicted, loss)
        memory()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is None for p in adapters):
            raise RuntimeError("adapter missing gradient")
        finite(*(p.grad for p in adapters))
        norm = torch.nn.utils.clip_grad_norm_(adapters, 1.0)
        finite(norm)
        memory()
        scaler.step(optimizer)
        scaler.update()
        finite(*adapters)
        memory()
        report["steps"].append({"optimizer_step": step + 1, "loss": loss.item(),
                                "gradient_norm_before_clip": norm.item(),
                                "seconds": time.monotonic() - started})
        persist()
    report["phase"] = "frozen_verification"
    persist()
    after = frozen_hash(unet)
    report["frozen_after"] = after
    if before != after:
        raise RuntimeError("frozen UNet parameters changed")
    final = get_peft_model_state_dict(unet)
    delta = sum((final[k].detach().cpu() - initial[k]).abs().sum().item() for k in initial)
    if not delta > 0:
        raise RuntimeError("no adapter update")
    report["adapter_update_l1"] = delta
    report["phase"] = "save_reload"
    persist()
    expected = save_adapter(unet, output)
    # Reload on CPU preserves fp32 serialization rather than rounding via the fp16 base.
    del optimizer, scaler, adapters, final, predicted, loss, item, noisy, target, noise
    unet.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    reload_adapter(unet, output, expected)
    if frozen_hash(unet) != before:
        raise RuntimeError("frozen parameters changed during reload")
    report["checkpoint"] = {"path": "pytorch_lora_weights.safetensors",
                            "sha256": file_hash(output / "pytorch_lora_weights.safetensors"),
                            "reload_exact": True, "scope": "same UNet; old adapter deleted; actual diffusers loader with temporary CPU fp32 base; frozen base restored fp16; CPU tensor equality, not inference"}
    memory()


def run(request, output):
    output = Path(output).resolve()
    report = preflight(request, output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report.update(status="running", training_started=False)
    report["runtime"] = {"python": sys.version, "executable": sys.executable,
                         "versions": {}, "environment": {name: os.environ.get(name) for name in
                         ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}}

    def persist():
        report["duration_seconds"] = time.monotonic() - started
        temporary = output / "outcome.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "outcome.json")

    persist()  # A killed/crashed process remains running, never a false success.
    try:
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        report["runtime"]["versions"] = versions
        if versions != VERSIONS or sys.version_info[:2] != (3, 12):
            raise RuntimeError("requires the pinned Python 3.12 smoke runtime")
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        report["runtime"]["offline"] = True
        persist()
        train(report, output, persist)
        report.update(status="smoke_passed", phase="completed")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        # Best effort only: never hide the original OOM/crash/nonfinite error.
        if "torch" in sys.modules:
            try:
                torch = sys.modules["torch"]
                if torch.cuda.is_initialized():
                    report["runtime"]["failure_memory"] = {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        **device_usage(torch)}
            except Exception:
                pass
        persist()
        raise
    persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.request, args.output)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "quality_assessed": False}))


if __name__ == "__main__":
    main()
