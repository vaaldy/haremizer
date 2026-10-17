# haremizer

An experimental Python toolkit for anime-style pixel-art portraits and
term-chan-format sprite export.
It combines local SDXL inference with explicit CPU post-processing and source-bound edits.
It is not a reliable automatic character, expression, or animation generator.

![Historical kawaii-maid base frame](artifacts/sprites/kawaii-maid/neutral-base.png)

Historical generated example, downsampled and color-keyed on CPU, with separately
annotated blink edits. This is not native learned transparency or a model-quality
promotion. See [sample provenance and limitations](docs/samples.md).

## Current capabilities

| Status | Capability |
|---|---|
| Implemented, CPU-tested | PNG validation, palette reduction, explicit background recovery, sprite JSON export/render |
| Implemented, annotation required | Source-hash-bound localized blink/edit plans; isolated hair-tip rig for declared layers |
| Experimental, optional GPU | Local SDXL inference, LoRA training/evaluation, LayerDiffuse adaptation |
| Not implemented | Reliable automatic eye/part detection, robust learned expressions, general image world model |

## CPU quickstart

Use Python 3.11+. Install the package in a fresh environment; the only base
dependency is Pillow (no PyTorch/model download):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
```

For offline wheel build/install instructions, see [packaging](docs/packaging.md).
From the checkout, these commands read the tracked example and create a **new**,
ignored output; they do not run a model:

```sh
python -B -m haremizer.sprite_export \
  --base artifacts/sprites/kawaii-maid/neutral-base.png \
  --blink artifacts/sprites/kawaii-maid/neutral-blink.png \
  --output runs/quickstart/kawaii --name neutral
python -B -m haremizer.sprite_png --character runs/quickstart/kawaii
```

Here `python` means your Pillow-enabled interpreter (e.g. `.venv/bin/python`).
Inspect `runs/quickstart/kawaii/neutral.json`, `neutral-base.png`, and
`neutral-blink.png`; `runs/quickstart/kawaii.provenance.json` records export facts.
Choose a different new output directory on repeat runs; overwrites are refused.
Export validates supplied frames, not expression meaning or aesthetic quality.
Transparent hidden RGB is not serialized; opaque RGB and binary alpha are preserved.

## GPU and model setup

Read [local-model setup](docs/models.md) before invoking inference. Models, LoRAs,
training images and the heavy ML runtime are not supplied or downloaded by these
tools. CPU post-processing does not require PyTorch. No training or inference is
needed for the quickstart above.

## Tests

After non-editable installation, from the repository root using that interpreter:

```sh
python -B -m unittest discover -s test -q
```

The suite preserves all 193 original tests plus installed-package checks. Optional integration checks can skip when
their dependencies or sibling term-chan checkout are unavailable; skips are not
GPU/model-quality validation. Tests do not establish reliable generation.

## Documentation

- [Pipeline architecture and source entrypoints](docs/architecture/pipelines.md)
- [Planned image-world-model illustration](docs/architecture/README.md)
- [Historical research and original contracts](docs/history/README.md) — archival,
  not current permissions or installation instructions
- [Sample provenance](docs/samples.md) and [publication notice](NOTICE.md)

## Limitations and licenses

A prompt saying “transparent” does not give an RGB model alpha. Color-key recovery
is explicit, can remove wanted colors, and must be inspected. Localized edits need
source-bound annotations; unknown occluded content is not reconstructed. Existing
examples do not prove descriptor adherence, reliable transparency or general quality.

Original code is [MIT](LICENSE), copyright 2026 Haremizer contributors.
The LayerDiffuse adaptation remains [Apache-2.0](LICENSE.layerdiffuse.txt).
Model weights, training references and generated examples have separate terms;
MIT does not clear those assets for redistribution. See [NOTICE.md](NOTICE.md) for
attribution and outstanding rights/provenance limits. Data, weights, private working
files and run directories are deliberately excluded.

For an explicit bounded local generation → CPU recovery → neutral JSON example,
see [workflow](docs/workflow.md). Format success is not visual acceptance.
