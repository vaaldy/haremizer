# Bounded neutral portrait workflow (experimental)

This CPU coordinator invokes the installed local GPU sampler in one owned child
(maximum 600 seconds), then uses the unchanged CPU recovery and sprite exporter.
It does not train, fetch models, judge aesthetics, or promote a model. The base
package and `--help` need no torch/diffusers import. Actual inference requires the
independently provisioned environment and authorized local inputs in [models](models.md).

## Prepare explicit inputs before inference

Copy [neutral-recipe.json](../examples/neutral-recipe.json) and
[neutral-workflow.json](../examples/neutral-workflow.json) into your workspace.
These are templates, not bundled weights or a visual acceptance guarantee.
Replace the model and optional LoRA paths with your authorized local paths.
The example's hashes describe the historical Animagine XL4 revision and exp27
adapter; verify your bytes, never substitute a different model under those hashes.
To omit the adapter set `lora` to `null`. Model `weight_sha256` maps relative
weight filenames to full SHA256s. All model files (including configs/tokenizers),
installed Python/JSON source, interpreter, recipe and config are additionally
bound in the run receipt. Relative recipe/model/LoRA paths resolve from their
containing config/recipe file, not from package source or current directory.

Set all six `storage_roots` to existing separately counted project, ML runtime,
uv cache, Python installations, coredump and miopen cache directories. They must
cover inputs/interpreter/output. Overlapping roots intentionally count twice;
this is not a disk-free-space estimate. The original conservative 100,000,000,000
byte ceiling includes a 2GB reserve plus 200MB image allowance. Storage is checked
before launch and during the child. Do not delete unrelated caches to pass it.

Freeze the recipe before inference, including explicit recovery RGB, tolerance,
colors and size. Then put its hash in your workflow config:

```sh
sha256sum /absolute/workspace/neutral-recipe.json
/path/to/installed-env/bin/python -I -B -m haremizer.workflow --help
/path/to/installed-env/bin/python -I -B -m haremizer.workflow \
  --config /absolute/workspace/neutral-workflow.json \
  --output /absolute/workspace/new-neutral-run
```

Run from any directory with the same installed environment in the coordinator
and child. No editable install, `PYTHONPATH`, private coordinator construction,
or shared-runtime modification is needed. Output must not exist; failed outputs
are evidence and cannot be reused. There is no retry or CPU fallback.

The historical prompt requests magenta, but its actual saved background was
pastel pink. The example freezes RGB252,215,220 / tolerance24 / colors32 / size64
based on that prior evidence, **not** on a newly generated image. This is explicit
color-key postprocessing, not native learned transparency or automatic semantic
segmentation. Do not change the recipe after looking at results to claim a pass.

## Artifacts and failure behavior

- `raw/case-01.png` and sampler `raw/manifest.json`: original model output.
- `recovery/recovered.png`, source copy and recovery manifest: CPU postprocessing.
- `character/neutral.json` and `character.provenance.json`: lossless binary-alpha
  palette export, not an eye/blink detector.
- `workflow.json`: byte identities, argv, storage, status and output hashes.
- `sample.stdout.log`, `sample.stderr.log`: separate unparsed child diagnostics.

The child checks the bound installation/inputs, initializes CUDA/ROCm explicitly,
sets the PyTorch allocator fraction to at most 11GiB, and runs the existing sampler
with offline flags and process-local `HSA_OVERRIDE_GFX_VERSION=10.3.0` (hardware
specific, not global setup advice). This is an allocator guard, not a claim about
total GPU/device memory. Core dumps are disabled. Timeout, interrupt, signal,
nonzero exit, changed inputs or missing/partial output fail closed; owned-child
failure cleanup kills only its new process group. Logs/partial artifacts remain.

`completed_unreviewed` means the pipeline and PNG/export format checks completed,
**not** that the image is good or even has an adequately separated subject. Inspect
raw/native/enlarged output, require nonempty subject and exterior transparency,
round-trip the JSON and use explicit read-only term-chan loader/renderer checks
([interoperability](models.md#interoperability)). Independent image review is a
separate gate. CPU synthetic/mock tests are not real inference. Weights, training
data and generated artwork have separate terms; see [NOTICE](../NOTICE.md).
