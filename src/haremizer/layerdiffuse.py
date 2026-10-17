"""Optional, hash-pinned LayerDiffuse SDXL attention offset and CPU RGBA decoder.

Decoder architecture/augmentation adapted from lllyasviel/LayerDiffuse_DiffusersCLI
commit3061d9aed52a6c52a13fcf2b196c0fef4d727824, lib_layerdiffuse/vae.py.
Apache-2.0 code: see LICENSE.layerdiffuse.txt. Changes: lazy imports, fixed decoder
configuration, CPU fp32 inference, streaming offset merge, no OpenCV/encoder.
Weights are separately labeled OpenRAIL by their publisher, not Apache-2.0.
"""
from pathlib import Path
import hashlib

REVISION = "e76ba7e9031888ee9c2595391f218a69a5cbfc20"
WEIGHTS = {
    "attention": ("ld_diffusers_sdxl_attn.safetensors", 1910453328,
                  "22edbe45583644952c1613c6a62eda33503e4c032a8798d0acf198b3ebef2f13"),
    "decoder": ("ld_diffusers_sdxl_vae_transparent_decoder.safetensors", 208266320,
                "59dc8feefe40d26b6cb7186aed7af70c60d94ac4f9db8ebe7f121a01ba27a2fc"),
}


def validate(directory):
    directory = Path(directory).resolve()
    records = {}
    for role, (name, size, expected) in WEIGHTS.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"LayerDiffuse {role}: exact official file/size required")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != expected:
            raise ValueError(f"LayerDiffuse {role}: SHA256 mismatch")
        records[role] = {"path": str(path), "bytes": size, "sha256": expected}
    return {"repository": "lllyasviel/LayerDiffuse_Diffusers", "revision": REVISION,
            "license_card": "openrail (unspecified version; no redistribution approval claimed)",
            "files": records}


def merge_attention(unet, path):
    import torch
    from safetensors import safe_open
    state = unet.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as offsets:
        keys = list(offsets.keys())
        if not keys or any(key not in state for key in keys):
            raise ValueError("LayerDiffuse offset keys do not match this UNet")
        # Check the entire asset before mutating the in-memory model.
        for key in keys:
            value = offsets.get_tensor(key)
            if (value.shape != state[key].shape or not value.is_floating_point() or
                    state[key].device.type != "cpu" or not torch.isfinite(value).all()):
                raise ValueError(f"LayerDiffuse offset shape/dtype/device/finite mismatch: {key}")
        with torch.no_grad():
            for key in keys:
                state[key].add_(offsets.get_tensor(key).to(dtype=state[key].dtype))
                if not torch.isfinite(state[key]).all():
                    raise ValueError(f"LayerDiffuse merged tensor is nonfinite: {key}")
    return len(keys)


def build_decoder():
    import torch
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
    from diffusers.models.attention_processor import AttnProcessor2_0
    nn = torch.nn

    class TransparencyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (32, 32, 64, 128, 256, 512, 512)
            down = ("DownBlock2D",) * 4 + ("AttnDownBlock2D",) * 3
            up = ("AttnUpBlock2D",) * 3 + ("UpBlock2D",) * 4
            self.conv_in = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
            self.latent_conv_in = nn.Conv2d(4, channels[2], kernel_size=1)
            self.down_blocks = nn.ModuleList()
            common = dict(temb_channels=None, resnet_eps=1e-5, resnet_act_fn="silu",
                          resnet_groups=4, attention_head_dim=8,
                          resnet_time_scale_shift="default", dropout=0.0)
            output_channel = channels[0]
            for i, kind in enumerate(down):
                input_channel, output_channel = output_channel, channels[i]
                self.down_blocks.append(get_down_block(
                    kind, num_layers=2, in_channels=input_channel, out_channels=output_channel,
                    add_downsample=i != 6, downsample_padding=1, downsample_type="conv", **common))
            self.mid_block = UNetMidBlock2D(
                in_channels=channels[-1], output_scale_factor=1, attn_groups=None,
                add_attention=True, **common)
            self.up_blocks = nn.ModuleList()
            reverse = list(reversed(channels))
            output_channel = reverse[0]
            for i, kind in enumerate(up):
                previous, output_channel = output_channel, reverse[i]
                self.up_blocks.append(get_up_block(
                    kind, num_layers=3, in_channels=reverse[min(i + 1, 6)],
                    out_channels=output_channel, prev_output_channel=previous,
                    add_upsample=i != 6, upsample_type="conv", **common))
            self.conv_norm_out = nn.GroupNorm(4, channels[0], eps=1e-5)
            self.conv_act = nn.SiLU()
            self.conv_out = nn.Conv2d(channels[0], 4, kernel_size=3, padding=1)
            for module in self.modules():
                if hasattr(module, "set_processor"):
                    module.set_processor(AttnProcessor2_0())

        def forward(self, pixel, latent):
            sample_latent = self.latent_conv_in(latent)
            sample = self.conv_in(pixel)
            residuals = (sample,)
            for i, block in enumerate(self.down_blocks):
                if i == 3:
                    sample = sample + sample_latent
                sample, extra = block(hidden_states=sample, temb=None)
                residuals += extra
            sample = self.mid_block(sample, None)
            for block in self.up_blocks:
                extra = residuals[-len(block.resnets):]
                residuals = residuals[:-len(block.resnets)]
                sample = block(sample, extra, None)
            return self.conv_out(self.conv_act(self.conv_norm_out(sample)))

    return TransparencyDecoder()


def augmented_decode(model, pixel, latent):
    import torch
    results = []
    for flip in (False, True):
        for turns in range(4):
            p, z = pixel, latent
            if flip:
                p, z = p.flip(3), z.flip(3)
            result = model(p.rot90(turns, (2, 3)), z.rot90(turns, (2, 3)))
            if not torch.isfinite(result).all():
                raise ValueError("LayerDiffuse decoder returned nonfinite values")
            result = result.clamp(0, 1).rot90(-turns, (2, 3))
            results.append(result.flip(3) if flip else result)
    return torch.stack(results).median(dim=0).values


def rgba_image(result):
    import torch
    from PIL import Image
    if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 4:
        raise ValueError("LayerDiffuse decoder must return one A,R,G,B tensor")
    if not torch.isfinite(result).all():
        raise ValueError("LayerDiffuse RGBA values must be finite")
    rgba = torch.cat((result[:, 1:], result[:, :1]), dim=1)
    array = (rgba.clamp(0, 1)[0].movedim(0, -1) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGBA")


class Decoder:
    def __init__(self, model_directory, assets):
        import torch
        from safetensors.torch import load_file
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            str(model_directory), subfolder="vae", local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32).eval().requires_grad_(False)
        self.model = build_decoder()
        self.model.load_state_dict(load_file(assets["files"]["decoder"]["path"], device="cpu"), strict=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, latents):
        import torch
        if latents.ndim != 4 or latents.shape[0] != 1 or latents.shape[1] != 4:
            raise ValueError("LayerDiffuse requires one four-channel scaled latent")
        with torch.inference_mode():
            latent = latents.detach().to(device="cpu", dtype=torch.float32) / self.vae.config.scaling_factor
            pixel = (self.vae.decode(latent).sample * 0.5 + 0.5).clamp(0, 1)
            return rgba_image(augmented_decode(self.model, pixel, latent))
