"""Local GPU-only bounded SDXL attention-LoRA smoke/pilot/refinement; never quality acceptance."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import struct
from pathlib import Path
import re
import random
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


def palette_pattern_digest(rgba):
    """Exact 64px alpha/color partitions, not identity; decline >62 RGB colors."""
    labels, pattern = {}, bytearray()
    for index in range(0, len(rgba), 4):
        alpha = rgba[index + 3]
        if not alpha:
            pattern.extend((0, 0))
            continue
        rgb = rgba[index:index + 3]
        label = labels.setdefault(rgb, len(labels) + 1)
        if len(labels) > 62:
            return None
        pattern.extend((label, alpha))
    return hashlib.sha256(b"64x64:" + pattern).hexdigest()


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")


def preflight(request_path, output, *, pilot=False, refine=False):
    """Validate local declarations and byte identities without importing ML packages."""
    if type(pilot) is not bool or type(refine) is not bool:
        raise ValueError("pilot and refine must be an explicit boolean")
    if pilot and refine:
        raise ValueError("pilot and refine are mutually exclusive")
    style = pilot or refine
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    raw = request_path.read_bytes()
    request = json.loads(raw)
    request_keys = {"model", "config", "examples", "reviewed_by", "declared_smoke_only",
                    "authorization"}
    if style:
        request_keys.add("heldout_examples")
    if refine:
        request_keys.add("continuation")
    keys(request, request_keys, "request")
    if request["reviewed_by"] != "machine:sol" or request["declared_smoke_only"] is not (not style):
        raise ValueError(f"truthful machine:sol review and declared_smoke_only={not style} required")
    authorization = request["authorization"]
    keys(authorization, {"user_authorized", "record"}, "authorization")
    if authorization["user_authorized"] is not True or not nonempty(authorization["record"]):
        raise ValueError("record the existing broad user authorization; do not invent human approval")
    config = request["config"]
    keys(config, {*CONFIG, "steps", *({"content_resolution"} if refine else set()),
                  *({"prior_preservation_weight"} if refine and "prior_preservation_weight" in config else set()),
                  *({"facial_loss_weight"} if refine and "facial_loss_weight" in config else set()),
                  *({"decoded_feature_loss_weight"} if refine and "decoded_feature_loss_weight" in config else set()),
                  *({"caption_dropout_probability"} if refine and "caption_dropout_probability" in config else set()),
                  *({"snr_gamma"} if refine and "snr_gamma" in config else set()),
                  *({"decoded_ink_weight"} if refine and "decoded_ink_weight" in config else set())}, "config")
    fixed = {k: v for k, v in CONFIG.items() if not refine or k not in ("seed", "learning_rate", "resolution")}
    if any((type(config[k]) not in ((int, float) if type(v) is float else (int,))) or
           config[k] != v for k, v in fixed.items()):
        raise ValueError("smoke config must use the fixed batch/resolution/rank/seed/lr/clip")
    if refine:
        if (type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1 or
                type(config["learning_rate"]) not in (int, float) or
                not math.isfinite(config["learning_rate"]) or
                not 1e-5 <= config["learning_rate"] <= 1e-4 or
                type(config["resolution"]) is not int or config["resolution"] not in (512, 768) or
                type(config["content_resolution"]) is not int or
                config["content_resolution"] not in {512: (384, 448, 512), 768: (576, 672, 768)}.get(config["resolution"], ())):
            raise ValueError("refine requires uint32 seed, finite learning_rate 1e-5..1e-4; resolution512/content384,448,512 or resolution768/content576,672,768")
    if refine and "prior_preservation_weight" in config:
        weight = config["prior_preservation_weight"]
        if type(weight) not in (int, float) or not 0 <= weight <= 1 or not math.isfinite(weight):
            raise ValueError("prior_preservation_weight must be finite 0..1")
    snr_gamma = config.get("snr_gamma")
    if snr_gamma is not None and (type(snr_gamma) not in (int, float) or
            not 0 < snr_gamma <= 20 or not math.isfinite(snr_gamma)):
        raise ValueError("snr_gamma must be None or finite numeric >0..20")
    dropout_probability = config.get("caption_dropout_probability", 0)
    if (type(dropout_probability) not in (int, float) or not 0 <= dropout_probability <= 0.5 or
            not math.isfinite(dropout_probability)):
        raise ValueError("caption_dropout_probability must be finite 0..0.5")
    decoded_weight = config.get("decoded_feature_loss_weight", 0)
    if (type(decoded_weight) not in (int, float) or not 0 <= decoded_weight <= 0.25 or
            not math.isfinite(decoded_weight)):
        raise ValueError("decoded_feature_loss_weight must be finite 0..0.25")
    ink_weight = config.get("decoded_ink_weight", 1)
    if (type(ink_weight) not in (int, float) or not 1 <= ink_weight <= 8 or
            not math.isfinite(ink_weight) or (ink_weight > 1 and decoded_weight == 0)):
        raise ValueError("decoded_ink_weight must be finite 1..8; >1 requires decoded feature loss")
    facial_weight = config.get("facial_loss_weight", 1)
    if (type(facial_weight) not in (int, float) or not 1 <= facial_weight <= 4 or
            not math.isfinite(facial_weight)):
        raise ValueError("facial_loss_weight must be finite 1..4")
    if type(config["steps"]) is not int or not (
            50 <= config["steps"] <= 600 if refine else
            config["steps"] == 100 if pilot else 1 <= config["steps"] <= 2):
        raise ValueError("steps must be 50..600" if refine else "steps must be exactly 100" if pilot else
                         "steps must be 1..2 total optimizer steps")
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
    continuation = None
    if refine and request["continuation"] is not None:
        parent = request["continuation"]
        keys(parent, {"path", "sha256", "outcome", "outcome_sha256"}, "continuation")
        adapter, outcome = local(parent["path"]), local(parent["outcome"])
        validate_adapter_file(adapter)
        for path, expected in ((adapter, parent["sha256"]), (outcome, parent["outcome_sha256"])):
            if (not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or
                    not path.is_file() or file_hash(path) != expected or path.is_relative_to(output)):
                raise ValueError("continuation input/hash/output mismatch")
        evidence = json.loads(outcome.read_bytes())
        if (not isinstance(evidence, dict) or evidence.get("status") not in
                ("smoke_passed", "pilot_completed_unreviewed", "refinement_completed_unreviewed") or
                evidence.get("phase") != "completed" or evidence.get("training_started") is not True):
            raise ValueError("continuation requires completed haremizer outcome")
        checkpoint = evidence.get("checkpoint", {})
        parent_model = evidence.get("model", {})
        parent_request = evidence.get("request", {})
        if not isinstance(parent_request, dict):
            raise ValueError("invalid continuation request provenance")
        parent_config = parent_request.get("config", {})
        if any(not isinstance(value, dict) for value in (checkpoint, parent_model, parent_config)):
            raise ValueError("invalid continuation checkpoint/base/config provenance")
        if (checkpoint.get("sha256") != parent["sha256"] or
                not nonempty(checkpoint.get("path")) or
                (outcome.parent / checkpoint["path"]).resolve() != adapter or
                checkpoint.get("reload_exact") is not True or
                type(parent_config.get("rank")) is not int or parent_config["rank"] != 4 or
                any(parent_model.get(k) != v for k, v in
                    (("id", spec["id"]), ("revision", revision), ("file_sha256", hashes)))):
            raise ValueError("continuation checkpoint/base/rank provenance mismatch")
        continuation = {**parent, "path": str(adapter), "outcome": str(outcome),
                        "base_model": parent_model, "rank": 4,
                        "scope": "weights continuation only; new optimizer, not exact optimizer resume"}
    splits = [("examples", 4 if style else 1, 16 if style else 2)]
    if style:
        splits.append(("heldout_examples", 2, 16))
    validated, seen, seen_pixels, seen_patterns = {}, set(), set(), set()
    from PIL import Image
    for name, minimum, maximum in splits:
        examples = request[name]
        if not isinstance(examples, list) or not minimum <= len(examples) <= maximum:
            raise ValueError(f"{name}: supply {minimum}..{maximum} local PNG examples")
        records = []
        for row in examples:
            example_keys = {"path", "sha256", "caption", "source_url", "license", "credit",
                            "source_page"}
            if style:
                example_keys.add("identity_group")
            if refine and name == "examples" and "facial_regions" in row:
                example_keys.add("facial_regions")
            keys(row, example_keys, "example")
            if style and not nonempty(row["identity_group"]):
                raise ValueError("nonblank identity_group required")
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
                if style:
                    data = png.convert("RGBA").tobytes()
                    canonical = bytes(channel for index in range(0, len(data), 4)
                                      for channel in (data[index:index + 4] if data[index + 3]
                                                      else b"\0\0\0\0"))
                    pixel_hash = hashlib.sha256(b"64x64:" + canonical).hexdigest()
                    if pixel_hash in seen_pixels:
                        raise ValueError("duplicate canonical pixels across pilot inputs")
                    seen_pixels.add(pixel_hash)
                    pattern_hash = palette_pattern_digest(data)
                    if pattern_hash is not None:
                        if pattern_hash in seen_patterns:
                            raise ValueError("duplicate pixel pattern under a whole-image palette recolor")
                        seen_patterns.add(pattern_hash)
            if refine and name == "examples":
                if "facial_regions" in row:
                    validate_facial_regions(row["facial_regions"])
                    if decoded_weight > 0:
                        decoded_crop_box(row["facial_regions"], config["content_resolution"],
                                         config["resolution"], (config["resolution"] // 8,) * 2)
                elif facial_weight > 1 or decoded_weight > 0:
                    raise ValueError("facial_regions required on every weighted training example")
            records.append({**row, "path": str(image), "source_page": {**source, "path": str(page)},
                            **({"pixel_sha256": pixel_hash,
                                "palette_pattern_sha256": pattern_hash} if style else {})})
        validated[name] = records
        if style and len({row["identity_group"] for row in records}) < minimum:
            raise ValueError(f"{name}: require at least {minimum} distinct identity groups")
    if style and ({row["identity_group"] for row in validated["examples"]} &
                  {row["identity_group"] for row in validated["heldout_examples"]}):
        raise ValueError("identity_group intersection across splits")
    if request_path.is_relative_to(output):
        raise ValueError("output must not contain request")
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override not in (None, "10.3.0"):
        raise ValueError("only the authorized process-local gfx 10.3.0 override is supported")
    return {**({"continuation": continuation} if refine else {}),
            "request": request, "request_sha256": hashlib.sha256(raw).hexdigest(),
            "config_sha256": canonical_digest(config), "model": {**spec, "path": str(model),
            "manifest": identity, "file_sha256": hashes,
            "provenance_scope": "HF path asserts revision; content hashes bind local bytes, not upstream authenticity"},
            "inputs": validated["examples"],
            **({"heldout_inputs": validated["heldout_examples"],
                "heldout_count": len(validated["heldout_examples"]),
                "holdout_scope": "declared identity groups, not independently proven identities"}
               if style else {}),
            "script_sha256": file_hash(__file__),
            "runtime_override": override,
            "override_scope": "existing process-local environment only; unsupported workaround, not native support",
            "quality_assessed": False,
            "promotion": False, "synthetic_feedback": False}


def validate_adapter_file(path):
    """Inspect the local safetensors header without importing an ML runtime."""
    size = path.stat().st_size
    if path.suffix != ".safetensors" or not 8 < size <= 64 * 1024**2:
        raise ValueError("continuation requires nonempty safetensors <=64 MiB")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= min(size - 8, 1024**2):
            raise ValueError("invalid adapter header")
        try:
            header = json.loads(stream.read(length))
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid adapter header") from error
    if not isinstance(header, dict):
        raise ValueError("invalid adapter header")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    pairs, intervals = {}, []
    for name, tensor in tensors.items():
        match = re.fullmatch(r"unet\.(.+\.(?:to_q|to_k|to_v|to_out\.0))\.lora\.(down|up)\.weight", name)
        if not match or not isinstance(tensor, dict):
            raise ValueError("only own diffusers UNet attention adapters supported")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        dtype = tensor.get("dtype")
        if (dtype != "F32" or not isinstance(shape, list) or len(shape) != 2 or
                any(type(n) is not int or n <= 0 for n in shape) or
                shape[0 if match[2] == "down" else 1] != 4 or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(n) is not int for n in offsets) or
                not 0 <= offsets[0] < offsets[1] <= size - 8 - length or
                offsets[1] - offsets[0] != math.prod(shape) * 4):
            raise ValueError("continuation requires own fp32 rank4 adapter tensors")
        pairs.setdefault(match[1], {})[match[2]] = shape
        intervals.append(offsets)
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("invalid adapter tensor offsets")
        cursor = end
    if (not pairs or cursor != size - 8 - length or
            any(set(pair) != {"down", "up"} for pair in pairs.values())):
        raise ValueError("incomplete rank4 adapter")


def prepare_rgb(image, content_resolution=512, resolution=512):
    from PIL import Image
    rgba = image.convert("RGBA").resize((content_resolution, content_resolution), Image.Resampling.NEAREST)
    padding = (resolution - content_resolution) // 2
    canvas = Image.new("RGBA", (resolution, resolution), "white")
    canvas.alpha_composite(rgba, (padding, padding))
    return canvas.convert("RGB")


def validate_facial_regions(regions):
    if not isinstance(regions, list) or not 1 <= len(regions) <= 8:
        raise ValueError("facial_regions requires 1..8 source rectangles")
    for region in regions:
        keys(region, {"x", "y", "width", "height"}, "facial region")
        if (any(type(value) is not int for value in region.values()) or
                not 0 <= region["x"] < 64 or not 0 <= region["y"] < 64 or
                not 1 <= region["width"] <= 64 - region["x"] or
                not 1 <= region["height"] <= 64 - region["y"]):
            raise ValueError("facial region requires integer bounds inside native 64x64")


def facial_mask(regions, content_resolution, resolution, latent_size):
    """Union of source rectangles, with the exact prepare_rgb resize/padding layout."""
    from PIL import Image, ImageDraw
    validate_facial_regions(regions)
    mask = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        x, y, width, height = (region[k] for k in ("x", "y", "width", "height"))
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill=1)
    content = mask.resize((content_resolution, content_resolution), Image.Resampling.NEAREST)
    canvas = Image.new("L", (resolution, resolution), 0)
    padding = (resolution - content_resolution) // 2
    canvas.paste(content, (padding, padding))
    return canvas.resize(latent_size, Image.Resampling.NEAREST)


def facial_weights(mask, weight):
    import torch
    binary = torch.tensor(list(mask.getdata()), dtype=torch.float32).reshape(1, 1, mask.height, mask.width)
    return 1 + (weight - 1) * binary


def denoising_loss(predicted, target, weights=None):
    import torch
    if weights is None:
        return torch.nn.functional.mse_loss(predicted.float(), target.float())
    squared = (predicted.float() - target.float()).square()
    return (squared * weights).sum() / weights.expand_as(squared).sum()


def min_snr_weight(alpha_cumprod, gamma, prediction_type):
    """Scalar Min-SNR weight for the existing batch-one epsilon/velocity objective."""
    import torch
    alpha = alpha_cumprod.float()
    if alpha.numel() != 1 or not torch.isfinite(alpha).all() or not 0 < alpha.item() < 1:
        raise ValueError("Min-SNR requires one finite alpha strictly between zero and one")
    snr = alpha / (1 - alpha)
    capped = torch.minimum(snr, snr.new_tensor(gamma))
    if prediction_type == "epsilon":
        return capped / snr, snr
    if prediction_type == "v_prediction":
        return capped / (snr + 1), snr
    raise ValueError("only epsilon/v_prediction scheduler targets supported")


def select_caption_conditioning(full, empty, rng, probability):
    """Independent stdlib RNG only; zero never draws or needs an empty cache."""
    dropped = probability > 0 and rng.random() < probability
    return (empty if dropped else full), dropped


def cache_empty_caption(item, encoder, tokenizer, *, projection):
    """One shared empty prompt using the same frozen dual-CLIP encoding contract."""
    ids = tokenizer("", padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    encoded = encoder(ids.to("cuda"), output_hidden_states=True)
    item["tokens"].append(encoded.hidden_states[-2].cpu())
    if projection:
        item["pooled"] = encoded.text_embeds.cpu()


def caption_conditioning_digest(item):
    value = hashlib.sha256()
    for name in ("tokens", "pooled"):
        tensor = item[name]
        value.update(name.encode())
        value.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        value.update(tensor.contiguous().numpy().tobytes())
    return value.hexdigest()


def predicted_x0(noisy, predicted, alpha_cumprod, prediction_type):
    """Unclipped clean latent estimate; preserve the prediction gradient."""
    import torch
    alpha = torch.as_tensor(alpha_cumprod, device=noisy.device, dtype=torch.float32)
    if prediction_type == "epsilon":
        return (noisy.float() - (1 - alpha).sqrt() * predicted.float()) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * noisy.float() - (1 - alpha).sqrt() * predicted.float()
    raise ValueError("only epsilon/v_prediction scheduler targets supported")


def decoded_crop_box(regions, content_resolution, resolution, latent_size):
    """Pre-ML safety/layout check shared by preflight and actual CPU cache."""
    mask = facial_mask(regions, content_resolution, resolution, latent_size)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("facial union vanished at latent resolution")
    left, top, right, bottom = bounds
    box = (max(0, left - 2), max(0, top - 2),
           min(latent_size[0], right + 2), min(latent_size[1], bottom + 2))
    if resolution != latent_size[0] * 8 or resolution != latent_size[1] * 8:
        raise ValueError("decoded feature loss requires 8x VAE latent layout")
    # Align outward to both VAE8 and the global native64 display sampling grid.
    display_stride = resolution // 64
    alignment = math.lcm(8, display_stride) // 8
    left, top, right, bottom = box
    box = (left // alignment * alignment, top // alignment * alignment,
           min(latent_size[0], ((right + alignment - 1) // alignment) * alignment),
           min(latent_size[1], ((bottom + alignment - 1) // alignment) * alignment))
    if box == (0, 0, latent_size[0], latent_size[1]):
        raise ValueError("decoded feature crop must not span the full image")
    return box, mask


def decoded_feature_cache(rgb, regions, content_resolution, resolution, latent_size, ink_weight=1):
    """CPU-only real prepared source RGB and union mask, never VAE-generated GT."""
    import torch
    import torch.nn.functional as F
    box, mask = decoded_crop_box(regions, content_resolution, resolution, latent_size)
    rgb_box = tuple(value * 8 for value in box)
    crop = rgb.crop(rgb_box)
    union = facial_mask(regions, content_resolution, resolution, (resolution, resolution)).crop(rgb_box)
    gt = torch.tensor(list(crop.getdata()), dtype=torch.float32).reshape(crop.height, crop.width, 3)
    gt = gt.permute(2, 0, 1).unsqueeze(0) / 255
    binary = torch.tensor(list(union.getdata()), dtype=torch.float32).reshape(1, 1, crop.height, crop.width)
    display_stride = resolution // 64
    display_size = (crop.height // display_stride, crop.width // display_stride)
    gt64 = F.interpolate(gt, size=display_size, mode="area")
    mask64 = F.interpolate(binary, size=display_size, mode="area")
    if ink_weight > 1:
        ink = (gt64.amax(dim=1, keepdim=True) <= 32 / 255).float()
        mask64 = mask64 * (1 + (ink_weight - 1) * ink)
    if mask64.sum().item() <= 0:
        raise ValueError("decoded feature mask is empty")
    provenance = {"latent_box_xyxy": list(box), "rgb_box_xyxy": list(rgb_box),
                  "display_size_hw": list(display_size), "margin_latent_cells": 2,
                  "latent_alignment": math.lcm(8, display_stride) // 8,
                  "display_stride_rgb": display_stride,
                  "global_display_box_xyxy": [value // display_stride for value in rgb_box],
                  "gt_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                  "mask_sha256": hashlib.sha256(union.tobytes()).hexdigest(),
                  "latent_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                  "mapping": "prepare_rgb source crop; source union NEAREST with centered padding; bbox+2 latent cells aligned outward to LCM(VAE8,global64 stride); both AREA to exact crop_px/display_stride",
                  "cache_bytes": (gt64.numel() + mask64.numel()) * 4,
                  "ink_weight": ink_weight, "ink_threshold_max_rgb": 32,
                  "weighted_mask_sha256": hashlib.sha256(mask64.contiguous().numpy().tobytes()).hexdigest()}
    return {"box": box, "gt": gt64, "mask": mask64, "provenance": provenance}


def decoded_feature_loss(vae, clean_latent, cached, scaling_factor):
    """Differentiable GPU->CPU crop decode, frozen VAE; no full-image decode."""
    import torch
    import torch.nn.functional as F
    left, top, right, bottom = cached["box"]
    crop = clean_latent[..., top:bottom, left:right].to(device="cpu", dtype=torch.float32)
    decoded = (vae.decode(crop / scaling_factor).sample + 1) / 2
    if decoded.shape[-2:] != ((bottom - top) * 8, (right - left) * 8):
        raise RuntimeError("VAE crop output layout mismatch")
    display = F.interpolate(decoded, size=cached["gt"].shape[-2:], mode="area")
    weights = cached["mask"].expand_as(display)
    return ((display - cached["gt"]).abs() * weights).sum() / weights.sum()


def load_continuation(unet, path):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    expected = convert_unet_state_dict_to_peft({k.removeprefix("unet."): v for k, v in state.items()})
    # Injection follows base dtype: upcast on CPU first, never round adapters via fp16.
    unet.float()
    StableDiffusionXLPipeline.load_lora_into_unet(state, network_alphas=None, unet=unet,
                                               adapter_name="default")
    actual = get_peft_model_state_dict(unet)
    if set(actual) != set(expected) or any(
            actual[k].dtype != torch.float32 or not torch.equal(actual[k].detach().cpu(), expected[k])
            for k in expected):
        raise RuntimeError("continuation adapter precision/state mismatch")
    for param in unet.parameters():
        if not param.requires_grad:
            param.data = param.data.half()
    return {k: v.detach().cpu().clone() for k, v in actual.items()}


def base_prediction(unet, noisy, timestep, conditioning):
    """Detached base denoising on existing source inputs, with adapter state restored."""
    import torch
    training = unet.training
    unet.disable_adapters()
    try:
        unet.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return unet(noisy, timestep, **conditioning).sample.detach()
    finally:
        unet.enable_adapters()
        unet.train(training)


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
    unet.delete_adapters("default")
    actual = load_continuation(unet, output / "pytorch_lora_weights.safetensors")
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
    config = report["request"]["config"]
    resolution = config["resolution"]
    snr_gamma = config.get("snr_gamma")
    if snr_gamma is not None:
        report["snr_weighting"] = {"gamma": snr_gamma,
            "formula": "min(SNR,gamma)/SNR for epsilon; min(SNR,gamma)/(SNR+1) for velocity",
            "scope": "scalar multiplier on spatially weighted noise loss only; prior/decoded terms unchanged"}
    dropout_probability = config.get("caption_dropout_probability", 0)
    empty_caption = {"tokens": []} if dropout_probability > 0 else None
    caption_rng = random.Random(config["seed"]) if dropout_probability > 0 else None
    if dropout_probability > 0:
        report["caption_dropout"] = {
            "probability": dropout_probability, "rng_seed": config["seed"],
            "rng": "independent stdlib random.Random; no Torch/global RNG draws",
            "empty_caption_sha256": hashlib.sha256(b"").hexdigest(),
            "text_encoder_weights": {name: report["model"]["file_sha256"][name + "/model.safetensors"]
                                     for name in ("text_encoder", "text_encoder_2")},
            "scope": "one shared frozen dual-CLIP empty conditioning; same chosen conditioning for student and prior; source RGB/masks unchanged"}
    decoded_weight = config.get("decoded_feature_loss_weight", 0)
    auxiliary_cache = []
    if decoded_weight > 0:
        torch.set_num_threads(2)
        report["decoded_feature_loss"] = {
            "weight": decoded_weight, "active_timestep_max": 250, "cpu_threads": torch.get_num_threads(),
            "ink_weight": config.get("decoded_ink_weight", 1),
            "mask_weighting": "source union * (1+(ink_weight-1)*[actual GT64 max RGB<=32/255]); normalized over expanded RGB weights",
            "formula": "loss=noise_loss+prior_weight*prior_loss+decoded_weight*sum(abs(decoded64-real_source64)*decoded_weights)/sum(decoded_weights.expand_as(RGB)); decoded_weights=AREA(source_union_mask)*(1+(ink_weight-1)*GT_dark_mask)",
            "x0": "epsilon:(xt-sqrt(1-alpha)*eps)/sqrt(alpha); v:sqrt(alpha)*xt-sqrt(1-alpha)*v; no detach/clamp",
            "scope": "training source-only cropped CPU fp32 frozen local VAE; no heldout/generated GT; no full-image gradient decode"}
    facial_weight = config.get("facial_loss_weight", 1)
    if facial_weight > 1:
        report["facial_loss"] = {"weight": facial_weight,
            "scope": "source-annotated training-only weighted noise/velocity MSE; rectangle union; no detection or heldout geometry",
            "normalization": "sum(weight * squared error) / sum(weights expanded over batch and channels)"}
    time_ids = [resolution, resolution, 0, 0, resolution, resolution]
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
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
            if dropout_probability > 0:
                cache_empty_caption(empty_caption, encoder, tokenizer, projection=sub == "text_encoder_2")
                finite(*empty_caption["tokens"], *([empty_caption["pooled"]] if sub == "text_encoder_2" else []))
            del encoded, ids, encoder, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            memory()
        if dropout_probability > 0:
            empty_caption["tokens"] = torch.cat(empty_caption["tokens"], dim=-1)
            report["caption_dropout"].update(empty_conditioning_sha256=caption_conditioning_digest(empty_caption),
                empty_cache_bytes=sum(t.numel() * t.element_size() for t in empty_caption.values()))
        vae = AutoencoderKL.from_pretrained(str(model / "vae"), torch_dtype=torch.float32, **load).to("cuda")
        vae.eval().requires_grad_(False)
        report["conditioning"] = {"tokens": "dual penultimate hidden states concatenated",
                                  "pooled": "CLIPTextModelWithProjection.text_embeds",
                                  "time_ids": time_ids,
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
                rgb = prepare_rgb(image, config.get("content_resolution", resolution), resolution)
            cached = output / f"input-{index + 1}.png"
            rgb.save(cached)
            row["cached_input"] = {"path": cached.name, "sha256": file_hash(cached),
                                   "transform": "64->512 nearest; alpha flattened onto white RGB",
                                   "original_preserved": original.name, "original_sha256": file_hash(original)}
            if "content_resolution" in config:
                content = config["content_resolution"]
                pad = (resolution - content) // 2
                row["cached_input"].update(transform=f"64->{content} nearest; centered {resolution} white alpha flatten",
                    resize={"from": [64, 64], "to": [content, content], "filter": "NEAREST"},
                    padding={"left": pad, "top": pad, "right": pad, "bottom": pad},
                    background=[255, 255, 255], rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            pixels = torch.tensor(list(rgb.getdata()), dtype=torch.float32).reshape(resolution, resolution, 3)
            pixels = (pixels.permute(2, 0, 1).unsqueeze(0).to("cuda") / 127.5) - 1
            latent = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
            finite(latent, item["pooled"], *item["tokens"])
            item["latents"] = latent.cpu().to(torch.float16)
            if facial_weight > 1:
                mask = facial_mask(row["facial_regions"], config["content_resolution"], resolution,
                                   (latent.shape[-1], latent.shape[-2]))
                item["facial_weights"] = facial_weights(mask, facial_weight)
                row["cached_facial_mask"] = {
                    "source_sha256": row["sha256"], "regions": row["facial_regions"],
                    "mapping": "64px binary rectangle union -> content NEAREST -> centered zero padding -> latent NEAREST; independent of alpha",
                    "content_resolution": config["content_resolution"], "resolution": resolution,
                    "padding": (resolution - config["content_resolution"]) // 2,
                    "latent_size": list(mask.size), "binary_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "selected_latent_pixels": sum(mask.getdata()),
                    "weight": facial_weight, "cpu_cache_bytes": item["facial_weights"].numel() * 4}
            if decoded_weight > 0:
                auxiliary = decoded_feature_cache(rgb, row["facial_regions"], config["content_resolution"],
                                                  resolution, (latent.shape[-1], latent.shape[-2]),
                                                  config.get("decoded_ink_weight", 1))
                row["cached_decoded_features"] = {**auxiliary["provenance"], "source_sha256": row["sha256"],
                                                  "prepared_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
                auxiliary_cache.append(auxiliary)
            item["tokens"] = torch.cat(item["tokens"], dim=-1)
            if dropout_probability > 0:
                row["cached_caption_conditioning"] = {
                    "caption_sha256": hashlib.sha256(row["caption"].encode()).hexdigest(),
                    "conditioning_sha256": caption_conditioning_digest(item)}
            item["time_ids"] = torch.tensor([time_ids], dtype=torch.float16)
        del pixels, latent
        if decoded_weight > 0:
            vae.to("cpu").float().eval().requires_grad_(False)
            report["decoded_feature_loss"].update(vae_frozen=True, vae_device="cpu", vae_dtype="float32",
                vae_weights_sha256=report["model"]["file_sha256"]["vae/diffusion_pytorch_model.safetensors"])
        else:
            del vae
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
    parent = report.get("continuation")
    if parent is not None:
        if (file_hash(parent["path"]) != parent["sha256"] or
                file_hash(parent["outcome"]) != parent["outcome_sha256"]):
            raise ValueError("continuation changed after preflight")
        load_continuation(unet, Path(parent["path"]))
        report["continuation"]["initial_reload_exact"] = True
    else:
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
    if "continuation" in report:
        report["optimizer_initialization"] = "new AdamW state; weights continuation is not optimizer resume"
        report["initial_adapter_sha256"] = hashlib.sha256(b"".join(
            k.encode() + initial[k].contiguous().numpy().tobytes() for k in sorted(initial))).hexdigest()
        report["adapter_update_reference"] = "starting parent weights" if parent else "fresh initializer"
    optimizer = torch.optim.AdamW(adapters, lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    prior_weight = config.get("prior_preservation_weight", 0)
    report["prior_preservation"] = {"weight": prior_weight,
        "scope": "detached base epsilon/velocity prediction on licensed training latents only; no heldout or new generated image feedback"}
    report["steps"] = []
    memory()
    report.update(training_started=True, phase="training")
    persist()
    for step in range(report["request"]["config"]["steps"]):
        started = time.monotonic()
        if dropout_probability > 0:
            chosen, caption_dropped = select_caption_conditioning(cache[step % len(cache)], empty_caption,
                                                                 caption_rng, dropout_probability)
            item = {k: v.to("cuda") for k, v in cache[step % len(cache)].items()
                    if k not in ("tokens", "pooled")}
            item.update({k: chosen[k].to("cuda") for k in ("tokens", "pooled")})
        else:
            item = {k: v.to("cuda") for k, v in cache[step % len(cache)].items()}
        noise = torch.randn(item["latents"].shape, generator=generator, device="cuda", dtype=torch.float16)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=generator, device="cuda")
        noisy = scheduler.add_noise(item["latents"], noise, timestep)
        target = noise if prediction == "epsilon" else scheduler.get_velocity(item["latents"], noise, timestep)
        finite(*item.values(), noise, noisy, target)
        optimizer.zero_grad(set_to_none=True)
        conditioning = {"encoder_hidden_states": item["tokens"],
                        "added_cond_kwargs": {"text_embeds": item["pooled"], "time_ids": item["time_ids"]}}
        prior = base_prediction(unet, noisy, timestep, conditioning) if prior_weight else None
        if prior is not None:
            finite(prior)
            memory()
        with torch.autocast("cuda", dtype=torch.float16):
            predicted = unet(noisy, timestep, **conditioning).sample
            noise_loss = denoising_loss(predicted, target, item.get("facial_weights"))
            if snr_gamma is not None:
                unweighted_noise_loss = noise_loss
                snr_weight, snr = min_snr_weight(
                    scheduler.alphas_cumprod[timestep.item()].to(predicted.device), snr_gamma, prediction)
                noise_loss = noise_loss * snr_weight
            prior_loss = (torch.nn.functional.mse_loss(predicted.float(), prior.float())
                          if prior is not None else noise_loss.new_zeros(()))
            loss = noise_loss + prior_weight * prior_loss
        decoded_loss = None
        if decoded_weight > 0 and timestep.item() <= 250:
            clean = predicted_x0(noisy, predicted, scheduler.alphas_cumprod[timestep.item()], prediction)
            decoded_loss = decoded_feature_loss(vae, clean, auxiliary_cache[step % len(cache)],
                                                vae.config.scaling_factor)
            finite(decoded_loss)
            loss = loss + decoded_weight * decoded_loss.to(predicted.device)
        finite(predicted, loss)
        memory()
        scaler.scale(loss).backward()
        if decoded_weight > 0 and any(p.requires_grad or p.grad is not None for p in vae.parameters()):
            raise RuntimeError("auxiliary VAE must remain frozen and gradient-free")
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
                                "noise_loss": noise_loss.item(), "prior_loss": prior_loss.item(),
                                **({"unweighted_noise_loss": unweighted_noise_loss.item(),
                                    "snr_weight": snr_weight.item(), "snr": snr.item(),
                                    "timestep": timestep.item()} if snr_gamma is not None else {}),
                                **({"weighted_noise_loss": noise_loss.item(),
                                    "source_sha256": report["inputs"][step % len(cache)]["sha256"]}
                                   if facial_weight > 1 else {}),
                                **({"caption_dropped": caption_dropped,
                                    "chosen_caption_sha256": hashlib.sha256(
                                        ("" if caption_dropped else report["inputs"][step % len(cache)]["caption"]).encode()).hexdigest(),
                                    "caption_source_sha256": report["inputs"][step % len(cache)]["sha256"]}
                                   if dropout_probability > 0 else {}),
                                **({"decoded_feature_active": decoded_loss is not None,
                                    "timestep": timestep.item(),
                                    "decoded_loss": decoded_loss.item() if decoded_loss is not None else 0.0,
                                    "decoded_source_sha256": report["inputs"][step % len(cache)]["sha256"]}
                                   if decoded_weight > 0 else {}),
                                "gradient_norm_before_clip": norm.item(),
                                "seconds": time.monotonic() - started})
        persist()
        if decoded_loss is not None:
            del clean, decoded_loss
    if decoded_weight > 0:
        report["decoded_feature_loss"]["vae_frozen_after"] = all(
            not p.requires_grad and p.grad is None for p in vae.parameters())
        del vae, auxiliary_cache
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
    if "continuation" in report:
        report["final_adapter_sha256"] = hashlib.sha256(b"".join(
            k.encode() + final[k].detach().cpu().contiguous().numpy().tobytes()
            for k in sorted(final))).hexdigest()
        if parent is not None:
            if (file_hash(parent["path"]) != parent["sha256"] or
                    file_hash(parent["outcome"]) != parent["outcome_sha256"]):
                raise RuntimeError("continuation inputs changed during training")
            report["continuation"]["inputs_unchanged"] = True
    report["phase"] = "save_reload"
    persist()
    expected = save_adapter(unet, output)
    # Reload on CPU preserves fp32 serialization rather than rounding via the fp16 base.
    del optimizer, scaler, adapters, final, predicted, loss, noise_loss, prior_loss, prior, conditioning, item, noisy, target, noise
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


def run(request, output, *, pilot=False, refine=False):
    output = Path(output).resolve()
    report = preflight(request, output, pilot=pilot, refine=refine)
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
        report.update(status="refinement_completed_unreviewed" if refine else "pilot_completed_unreviewed" if pilot else "smoke_passed", phase="completed")
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refine", action="store_true", help="bounded weights refinement stage")
    modes.add_argument("--pilot", action="store_true", help="explicit experimental 100-step style pilot")
    args = parser.parse_args()
    try:
        report = run(args.request, args.output, pilot=args.pilot, **({"refine": True} if args.refine else {}))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "quality_assessed": False}))


if __name__ == "__main__":
    main()
