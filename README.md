# haremizer

An experimental Animagine XL 4 + LoRA image generation model for bitmap / pixel-art anime girls.

The main use case is term-chan: fully animated, with 9 expressions + blink from a text prompt... though it's an ambitious goal.

## From prompt to Mint maid

<table>
  <tr>
    <td valign="middle" width="65%">
      <sub>YOUR PROMPT</sub>
      <blockquote>A pixel-art anime girl with pale cyan bobbed hair, green eyes and fair skin, wearing a white maid uniform and a frilled headdress.</blockquote>
      <sub>Example wording · existing generated render</sub>
    </td>
    <td align="center" width="35%">
      <img src="docs/images/world-model/01-baseline.png" width="256" height="256" alt="Mint maid">
    </td>
  </tr>
</table>

## Image world model approximation

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <p><strong>01 · Accept baseline</strong></p>
      <img src="docs/images/world-model/01-baseline.png" width="256" height="256" alt="Original Mint-maid sprite">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>02 · Propose boxes</strong></p>
      <img src="docs/images/world-model/02-candidate-boxes.png" width="256" height="256" alt="Candidate regions with red outline-only boxes">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>03 · Classify regions</strong></p>
      <img src="docs/images/world-model/03-part-labels.png" width="256" height="256" alt="Illustrative hair, eye and mouth region outlines">
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <p><strong>04 · Plan safe edit</strong></p>
      <img src="docs/images/world-model/04-edit-mask.png" width="256" height="256" alt="Actual eye-and-mouth edit mask; white pixels are editable">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>05 · Regenerate part</strong></p>
      <img src="docs/images/world-model/05-regenerated.png" width="256" height="256" alt="Saved raw happy-expression proposal">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>06 · Overlay and review</strong></p>
      <img src="docs/images/world-model/06-overlay-result.png" width="256" height="256" alt="Happy sprite with original pixels restored outside the edit mask">
    </td>
  </tr>
</table>

<sub>Illustrative region labels · Generated pixels inside the mask, original pixels outside · <a href="docs/architecture/image-world-model.md">Architecture</a> · <a href="docs/samples.md">Provenance</a></sub>

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
