# Local-model setup (experimental GPU path)

The CPU quickstart in [README](../README.md) needs only Python 3.11+ and Pillow.
Inference additionally requires an independently provisioned, GPU-compatible
PyTorch/diffusers environment. There is no automatic model or dependency download
in these source tools. Do not install a generic torch build into a working GPU
runtime merely to run the CPU utilities.

## Inputs

- A complete local diffusers snapshot with `model_index.json`, scheduler, tokenizer,
  text encoders, UNet and VAE components. The historical SDXL model is
  `cagliostrolab/animagine-xl-4.0`, revision
  `2b7c1b397761bf5bd3cc42e5b39ec99314a75a96`; its model-card/license conditions
  are separate from the code license. Supply your own authorized local snapshot.
- Optional local LoRA safetensors (<=256 MiB), with its full SHA256 passed explicitly.
- Optional local LayerDiffuse assets in one directory. [layerdiffuse.py](../src/haremizer/layerdiffuse.py)
  pins exact names, lengths and hashes. They are not shipped; their weight terms
  are not the Apache license of the code adaptation.

The local validation environment at documentation cleanup had Python 3.12.14,
Pillow 12.3.0, torch 2.11.0+rocm7.2, diffusers 0.41.0, transformers 5.19.0,
accelerate 1.15.0, peft 0.21.2 and safetensors 0.8.0. This is an observed inventory,
not a portable lockfile or a guarantee on another GPU. No GPU inference was run
for this documentation phase. ROCm compatibility overrides are hardware-specific,
process-local opt-ins, not global configuration instructions.

## Inspect the installed CLI

After installing the package into your provisioned environment (see [packaging](packaging.md)),
these module commands work independently of the source checkout:

```sh
python -B -m haremizer.sample --help
python -B -m haremizer.recovery --help
python -B -m haremizer.sprite_export --help
```

For a separately authorized inference run, supply explicit paths:

```sh
python -B -m haremizer.sample --model /path/to/local/snapshot \
  --output runs/my-new-sample --prompt '1girl, solo, bust portrait, pixel art' \
  --seed 1234 --steps 28 --width 768 --height 768 \
  --guidance-scale 5 --scheduler euler_ancestral --cpu-offload
```

This is a low-level CLI example, not a quality-accepted recipe or a bounded
end-to-end workflow. Direct `python -m haremizer.sample` does not supply the coordinator's timeout
or storage/allocator guard. Arrange owned-process time/resource bounds before real
GPU execution; see [coordinator architecture](architecture/pipelines.md).
The sampler requires CUDA/ROCm and refuses CPU fallback. A prompt alone does not
provide transparency. Inspect raw images before choosing any explicit recovery
key; retain raw, recovered and exported artifacts separately.

`--lora /path/to/adapter.safetensors --lora-sha256 FULL_SHA256` optionally loads
an adapter. `--layerdiffuse /path/to/local/assets` is an experimental alpha path,
not reliable background separation. Neither option establishes image quality.
Training is outside this onboarding guide: data and historical LoRAs are absent,
and the old research archive is not permission to reconstruct or run experiments.

## Interoperability

Exported JSON targets term-chan's palette-indexed frames. A sibling term-chan
checkout is not bundled. To verify interoperability, use an explicit separate
term-chan checkout and its `src/sprites/loader.js` and pure renderer in `src/render/`;
point that checkout's read-only checks at your exported character directory.
Do not assume a private absolute path or mutate that checkout. CPU unit tests may
skip sibling-dependent checks when it is unavailable; such skips are not an
interoperability pass.

See [NOTICE](../NOTICE.md) before redistributing any weights, data or samples.
