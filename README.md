# haremizer

Text-described anime pixel-art portraits: an independent repository inside
term-chan. **Current milestone: real SDXL training/inference testbed, annotated
localized blink, and independently verified sprite JSON export. The 100-step
style pilots were rejected for quality; no model or sprite is quality-promoted.**
Fully automatic eye landmarks remain missing. Python 3.11+; Pillow for PNG
processing. GPU inference additionally needs the dedicated PyTorch/diffusers
runtime. Source tools do not install packages or download models/data; separately
user-authorized setup has populated the local runtime, model cache, and candidate
artwork. No system-driver or global environment changes are authorized.

## Autonomous work authorization and current evidence

The current end goal, visual gates, checkpoint policy and bounded autonomous
loop are defined in [GOAL.md](GOAL.md). It supersedes informal “looks good”
judgments, not the immutable history of earlier experiments.

The user authorized bounded autonomous local training/evaluation, overnight
RX 6700 XT use with the process-local unsupported compatibility workaround
`HSA_OVERRIDE_GFX_VERSION=10.3.0`, and verified CC0/CC-BY artwork downloads with
attribution. No routine pre-training approval is required for newly designed
experiments inside that envelope. Hard storage ceiling: **100 GB**, including
runtime/cache/model/data/artifact reservations. No paid services, pushes, system
driver changes, unrelated shared-cache deletion, or term-chan source mutations.

Downloaded base: `cagliostrolab/animagine-xl-4.0`, immutable revision
`2b7c1b397761bf5bd3cc42e5b39ec99314a75a96`, local safe diffusers component
snapshot (~6.94 GB), subject to its model-card OpenRAIL++ conditions. Real
pretrained inference succeeded at 768×768, 12 steps, seed 1234 with CPU offload.
Evidence: ignored `artifacts/base-model.json`,
`artifacts/animagine-smoke-01/manifest.json` and `case-01.png`. The inspected
portrait is recognizable but opaque, with inconsistent pseudo-pixel blocks and
noisy highlights: **not a valid 64px sprite or trained LoRA**.

`data/candidates/` contains small source packs and source/license evidence, not
an approved training corpus. Group holdouts by identity/source, never expression.
Source verification and machine visual review must not be labeled human review.
Training completion and PNG format checks are not quality acceptance.

### Latest reviewed outcomes

- Global **16** completed the matched 25% caption-dropout branch. All eight
  actual/native64 images remained substantially similar; no reliable feature
  gain, so promotion is rejected. The serial-executor workflow is retired at
  the user's request; parent-direct execution continues without handoff waits.
  Global **17** is queued from the stronger retained 400-update style checkpoint
  with 100 updates of the tested facial objective. This changes initialization,
  not the total-history counter or acceptance gates.

- Autonomous global **15** reached 150 cumulative decoded-feature updates.
  Cross-device loss, active coverage on all sources, frozen weights and reload
  passed, but all-eight/native64 review found no reliable design gain. This
  branch is closed rather than extended again. Batch one used **3/6** slots.
  The next controlled hypothesis is training-caption dropout for generalization;
  no new GPU stage is permitted before its CPU tests and frozen plan.

- Autonomous global **14** completed the 50-step decoded-feature safety stage:
  CPU→GPU backward worked, frozen VAE/base and exact reload passed, and all four
  sources received active auxiliary steps (11 total). All eight actual/native64
  outputs remained near baseline without new major geometry regression; this
  is not model acceptance. One explicit 100-step continuation is queued to test
  the same objective at 150 cumulative updates, with new Adam state disclosed.

- Autonomous global experiment **13** retained every facial150 setting except
  integer-grid content768. Numerical/frozen/reload checks passed, but actual64
  inspection found no reliable automatic bold-feature gain; crops, pale nose
  marks and decoration persisted. It is rejected for promotion, not discarded.
  The input-grid correction remains valid. The next hypothesis is a bounded
  decoded64 source-feature objective, CPU-proven before GPU. Mission/visual
  ledger: `runs/autonomous-design-20261012/mission.json` and `review-013.json`.

- A renewed facial-design pilot trained a **fresh 150-update LoRA** with 3×
  source-annotated facial loss and a 0.1 source-only prior anchor. Four training
  identities/three publishers and three heldout identity groups were used.
  Licensed-reference crop/contrast changes and one machine-authored two-pixel
  nose mark are disclosed; no model-generated portrait entered training.
  All numerical/frozen-weight/reload checks passed, and all **133 CPU tests**
  passed. **The requested learned design goal did not pass visual review.**
- Fixed original-prompt outputs showed modest/mixed facial changes. A separate
  single design-phrase control applied identical conditioning to baseline and
  candidate: darker mouths largely occurred in the unloaded control too, while
  the LoRA regressed silver-hair eye proportions. Actual 64px BOX/32-color views
  were inspected without reinking; visibility/proportions remain inconsistent.
  Checkpoint retained, not promoted; no further identical fitting is queued.
  Evidence: `runs/facial-design-20261012/visual-review.json` and
  `runs/facial-design-20261012/design-profile/raw64-paired.png`.
  This renewed run used its own explicit
  deadline, preserved history, and is not an extension of the archived clock.

- The 2026-10-12 overnight work completed nine weight-training stages and one
  paired sampler diagnostic. **No model is quality-accepted or robust.** The
  original work deadline remains `2026-10-12T11:57:11Z`; changing trainer/data
  regimes did not reset it. Storage remained about 72.8 GB including reserve.
- The original-corpus 400-update checkpoint remains the strongest **unpromoted**
  rollback candidate. Extra padding, matched 768px training, source-only prior
  anchoring, and a corrected uncropped-reference corpus did not pass visual
  gates: head clipping, descriptor/background failures, or eye regressions
  persisted. Those fitting regimes are stopped, not automatically extended.
- A bounded byte-range audit found three cached UNet tensor samples matching
  Animagine OPT rather than ORIGINAL. This is not complete pipeline identity
  proof, but does not justify downloading OPT as an unexplored alternative.
- The recommended Euler Ancestral diagnostic compared eight fixed descriptors
  with unloaded baseline and unchanged 400-update weights. All four batches
  completed, after a preserved dispatcher-schema failure and authorized recovery.
  Actual image/64px-preview inspection found some cleaner backgrounds and useful
  learned styling/skin/hair gains, but persistent crops and adherence failures.
  This is evaluation evidence, not another weight-training success or promotion.
  Hash-bound review: `runs/refinement-overnight-20261012/scheduler-diagnostic-01/visual-review.json`.
- One **learned 400-update checkpoint** prototype now has transparent 64px
  neutral/blink PNGs and `artifacts/trained-best400-curly-01/character/neutral.json`.
  Independent AI image review passed this example only. Parent verification
  confirmed exact 4,096-pixel PNG↔JSON frames, 32 colors, and 23 blink edits inside
  26 declared eye pixels with unchanged alpha/surroundings. Soft eye detail and
  enclosed pale hair gaps remain; this does not promote the generator or prove
  automatic blinking. Preview: `artifacts/trained-best400-curly-01/gallery.png`.
- The dedicated-runtime CPU suite passed all **129 tests**, including actual
  adapter save/reload APIs and read-only Node/term-chan loader integration.
  These validate implementation contracts, not portrait quality.

- Two-step 512px SDXL LoRA runtime smoke passed; a 100-step style pilot on six
  training identities/two whole-identity holdouts also completed in about 117
  seconds, with finite updates, unchanged frozen weights and exact CPU reload.
  Peak allocated GPU memory was 5.66 GB. **Quality promotion was rejected:**
  coarse styling improved in some cases, but one matched descriptor produced an
  extreme cropped face; red-hair adherence also remained wrong.
- A fresh second 100-step generic-style pilot added CC0 EverFace references.
  Conservative ambiguity grouping yielded 12 training images/10 declared groups
  and five heldout images/four groups across two publishers. It also completed
  in about 118 seconds with unchanged frozen weights and exact reload. Its
  matched 512px baseline and candidate both suffered framing/repeated-face
  failures; candidate texture/color changes did not establish quality. This
  experiment is **not promoted**, and canonical ages, genders and creation
  methods of the additional references are not claimed verified.
- One constant whole-head/shoulder/margin scaffold was tested on all four
  descriptors with the unchanged pilot. It did not resolve the crop/geometry
  and adherence failures. No training extension or prompt sweep followed.
- A pinned `nerijs/pixel-art-xl` pretrained alternative (revision
  `8bf4a4d9ea283e00a51fafda8e0539f8248ea037`, 170.5 MB, CreativeML Open RAIL-M)
  was tested at 512px with the local Animagine base. This combination produced
  tiled faces/decorative scenes and was rejected, not used for training feedback.
  Retain nerijs attribution, the model card and applicable license restrictions;
  its training-data rights are **not** claimed verified.
- A separate **unloaded pretrained-baseline** prototype now has transparent
  64px base/blink frames. Its blink changes 54 pixels inside 70 declared eye
  pixels, with identical alpha/outside pixels. Independent read-only term-chan
  checks passed exact PNG↔JSON pixels, palette, loader, renderer and fallback
  behavior. This is format/locality evidence, **not quality acceptance**.
- Conservative, separately counted storage is approximately **76.7 decimal GB**
  including the entire shared uv cache, runtime, all system core dumps, and a
  200 MB diagnostic-image allowance. Earlier combined `du` estimates omitted
  shared hardlinks from the conservative double-counting policy. A further
  2 GB reservation still fits the strict 100 GB ceiling; no shared cache was
  deleted. Recheck before new downloads.

Ignored evidence: `artifacts/style-pilot-01/review.json`,
`artifacts/style-pilot-02/review.json`,
`artifacts/style-pilot-02-curation-review.json`,
`data/curation-draft-02/everface/curation.json`,
`artifacts/composition-guided-pilot-01/review.json`,
`artifacts/alternative-model-review/`, `artifacts/storage-accounting-01.json`,
`artifacts/exported-long-wavy-01/neutral.json` and sibling
`artifacts/exported-long-wavy-01.compatibility.json`. GPU iteration is paused
pending broader source/identity curation, not awaiting routine training permission.

The legacy SD1 adapter below retains its separate hash-bound human declaration
gates; general user authorization does not fabricate those declarations or make
that adapter SDXL-compatible. Its historical preparation gates are not the
current user authorization policy for future reviewed training implementations.

## Implemented versus missing

- `images.py`: real local PNG validation and nearest-neighbor contact sheets.
  Checks single-frame PNG, native square 48/64 size, binary alpha, at most 62
  opaque RGB colors, and both transparent and opaque pixels. Reports dimensions,
  color count, occupancy, and canonical pixel hash. **These are factual format
  checks, not aesthetic, semantic, prompt-adherence, or diversity passes.** No
  portrait generator or fixed face template is included.
- `runtime.py`: read-only platform/environment and GPU-management diagnostics.
  Each fixed diagnostic has a 10-second timeout and separate 16-KiB stdout/stderr
  return caps; probe JSON is parsed from stdout only, retaining stderr warnings.
  Visibility is not verified GPU tensor operation support. `--torch-probe` opts
  into a 30-second child process testing tiny GPU matrix/inference and backward
  primitives. No optimizer, fitted model, CPU fallback, or automatic
  `HSA_OVERRIDE_GFX_VERSION`. Existing override/visibility settings are recorded,
  never changed. Timeouts, nonzero exits, and signaled child crashes report blocked
  rather than crashing the coordinator. Native and explicit-environment-override
  evidence is labeled separately; an override pass is not a native-runtime pass.
  Unsupported compatibility workarounds require explicit human acknowledgment
  before any separate training authorization. A successful primitive probe still
  does not prove model/LoRA compatibility, memory capacity, or training stability.
- `prepare.py`: validates local captioned train/eval PNGs and rights declarations,
  prevents path traversal and escaping symlinks, rejects byte-identical and
  canonically identical pixel images (including re-encodings), fingerprints the
  dataset, and freezes exact evaluation-suite bytes in a JSON plan. It never
  invokes a trainer or the core execution loop.
- `evaluation-suite.json`: four fixed diverse adult anime-woman descriptors and
  fixed seeds, plus instructions for **independently composed** 48 and 64 layouts.
  These are descriptors, not generated samples or proof of visual diversity.
- `core.py`: the existing GPU-independent experiment orchestration harness with
  injected `Trainer`/`Reviewer` interfaces. `test_core.py` uses fake checkpoint
  files and fake reviews only. Its approval/ready state is a harness setup gate,
  **not authorization for real training**.

Missing: a broader quality training corpus, validated quality-learning recipes,
independent calibrated visual evaluation, accepted transparent native-grid
portraits, and fully automatic eye landmarks/semantic blinking. Local annotated
blink editing and lossless term-chan export are available, not quality acceptance.
A pinned base model, pretrained/adapter inference and bounded GPU training are
exercised; a successful short pilot does not prove long-run stability. Nothing here declares
learned quality as passing. Generated outputs must never be silently fed back
into training.

## Supervised durable refinement loop

A frozen coordinator plan may explicitly include
`settings.scheduler_override: "euler_ancestral"`; this forwards the existing local
sampler option and requires matching baseline/candidate override and actual
scheduler-class metadata. Omission preserves the cached scheduler and old plans.

`refinement_loop.py` coordinates **one explicitly chosen stage at a time**, not a
hyperparameter sweep or an independent LLM service. Sol must actually inspect
images externally after every successful stage. No GPU operation happens at
initialization/enqueue; enqueue calls the existing CPU `preflight(..., refine=True)`.
The loop does not select learning rates, source padding, steps, continuation, or
training data. Existing `smoke_sdxl.py --refine` source/identity partition checks
remain authoritative. Generated run PNGs cannot enter either source partition.

CLI (already-installed Python/Pillow; no install/download operation):

```sh
python -B refinement_loop.py init /absolute/run /absolute/plan.json
python -B refinement_loop.py enqueue /absolute/run /absolute/request.json 'explicit change and why'
python -B refinement_loop.py next /absolute/run
python -B refinement_loop.py status /absolute/run
# Inspect actual images, add native-format PNG evidence inside stage-NNN, then:
python -B refinement_loop.py review /absolute/run /absolute/review.json
# Explicit validation-only pass; no optimizer/training:
python -B refinement_loop.py validate-existing /absolute/run /absolute/weights.safetensors /absolute/outcome.json
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v test_refinement_loop
```

Plan JSON fields (no implicit defaults):

```json
{
  "model": {"path": "/canonical/local/SDXL", "id": "OWNER/MODEL", "revision": "immutable-40-or-64-hex"},
  "python": "/absolute/already-installed/runtime/bin/python",
  "deadline_utc": "2026-01-01T12:00:00Z",
  "max_stages": 4,
  "train_seconds": 900,
  "sample_seconds": 600,
  "storage_paths": ["/absolute/runtime", "/absolute/shared-cache", "/absolute/model-cache", "/absolute/data", "/absolute/artifacts", "/absolute/core-dumps"],
  "settings": {"width": 768, "height": 768, "steps": 28, "guidance_scale": 5, "model_cpu_offload": true, "negative_prompt": "fixed nonblank negative prompt"},
  "suites": [{"path": "/absolute/primary.json", "baseline": "/absolute/primary-baseline"}, {"path": "/absolute/challenge.json", "baseline": "/absolute/challenge-baseline"}],
  "validation_suites": [{"path": "/absolute/primary-new-seeds.json", "baseline": "/absolute/primary-new-seeds-baseline"}, {"path": "/absolute/challenge-new-seeds.json", "baseline": "/absolute/challenge-new-seeds-baseline"}]
}
```

Replace the illustrative deadline/revision/paths with real values. Deadline must
be future UTC, at most 12 hours from init; stages 1..8, including failed and
validation-only stages. `validation_suites` is optional, but without it robust
acceptance/validation is unavailable. Every suite uses `prepare.freeze_suite`'s
schema, exactly four cases, and fixed seeds; each second-pass suite has new seeds.
All supplied baseline directories must already contain four unloaded-base
`sample.py` PNGs and manifests matching the exact model, suite, and settings.
This API requires absolute request input paths, so freezing exact request bytes
into the run cannot reinterpret relative paths. The model path must be canonical.
A 40-hex model revision uses the pinned HF snapshot convention; a 64-hex revision
is the existing canonical file-content digest. Each training enqueue needs new
request bytes and an explicit reason; continuation is never inferred.

Plan/suite/request bytes, Python executable, current local trainer/sampler and
local helper modules, model bytes and supplied baseline files are hash-bound.
Directories/outputs are fresh and never overwritten. CLI writers serialize with
a directory lock; direct `Loop` API callers must remain single-writer. Atomic
fsynced state persists running stages before spawning. An interrupted `running`
stage **does not auto-resume**: inspect its logs/owned process and preserve the run;
there is intentionally no automatic recovery/retry or rollback deletion command.
Old incumbent, best-promising, and `previous` checkpoint paths are retained.

Storage roots must conservatively include the whole runtime, shared caches,
models, source data, artifacts, and existing core dumps, not merely known input
files. The caller supplies that inventory; the coordinator cannot discover all
unlisted external storage. Each path is counted by a **separate** `du -s -B1`
invocation: shared hardlinks across listed roots conservatively count again,
but a symlink is not fictional extra allocated weight storage within a root.
Include the resolved Python interpreter root as well as its virtual environment.
Add 200 MB image allowance plus 2 GB reserve;
strictly less than 100,000,000,000 bytes is required. Bounds are checked before
children, approximately every five seconds during children, and afterward.
Reserve/periodic checks are not a filesystem quota and cannot prevent a child
from transiently allocating beyond the ceiling between checks.

Children use Python `-B` argv (no shell), a new owned process group, zero core
limit, process-local gfx10.3.0 and HF/transformers/datasets offline environment.
Training is capped at 900 seconds; each four-case sampler at 600 seconds, further
limited by the remaining deadline. Only the owned group is killed on failure or
timeout. Cleanup waits at most five seconds; an unreaped owned child is recorded
as `cleanup_pending` and blocks every subsequent stage. There is no CPU training
fallback. Logs, argv, return codes, durations,
reported memory outcomes and checkpoint hashes are retained. Deadline/storage/
stage exhaustion is a budget result, **not** model quality. Reviews may finish
after the compute deadline, but no further child starts.

`status` exposes `review_binding`; copy that object unchanged into the review
JSON's `binding`. Review fields:

- `reviewed_by: "machine:sol"`, nonblank `notes`, boolean `uncertain`, optional
  boolean `robust_requested` (default false).
- `checks`: **exactly** `format`, `pixel_quality`, `prompt_adherence`,
  `character_diversity`, `non_regression`, each a boolean.
- `cases`: exactly eight keys from stage `cases[].case` (`0:id` and `1:id`). Each
  has nonblank string `glitches`, `crop_duplicate_face`, `adherence`, and
  `readability` observations, including explicit absence/uncertainty as appropriate.
- `format_evidence`: `{ "path": "/absolute/run/stage-NNN/native.png", "sha256": "..." }`
  records for actual postprocessed 48/64 PNGs. `format: true` requires these
  separate current-stage files to pass `images.validate_png`; robust requests
  additionally need native 64 evidence. Raw outputs/contact sheets are not eligible.

The binding covers stage, actual checkpoint hash, suite byte hashes, plan hash,
all generated PNGs/manifests/sheets, and supplied baselines. Raw labeled comparison
sheets and **64px NEAREST, unquantized viewing previews** are provided: previews
are not transparent validated sprites or quality proof. A failed check or any
uncertainty records rejection and a finding; it cannot replace an incumbent.
All checks true with no uncertainty yields **promising only**. Robust requires an
explicit `robust_requested: true`, separately recorded accepted first and
validation-only second reviews on the **same weights**, different frozen suite
sets and eight raw cases each (16 total), rechecked earlier evidence, and actual
native 64 format evidence. No loss/PNG/process-completion heuristic promotes.
External declarations are trusted as declarations, never legal, human, or
machine-verifiable semantic proof. CPU tests exercise orchestration only, not
training or visual quality. Main Sol owns actual training orchestration and image
review; this tool cannot fulfill a 12-hour autonomous training goal by itself.

## Observed local runtime

On this CachyOS / RX 6700 XT (`gfx1031`) machine, an isolated Python 3.12
installation of PyTorch `2.11.0+rocm7.2` detects the GPU and allocates memory,
but native kernel execution exits with SIGSEGV. The delivered handler records
that as blocked without crashing its coordinator.

An explicitly process-scoped `HSA_OVERRIDE_GFX_VERSION=10.3.0` passes the handler's
small matrix/backward probe. Separate primitive probes also passed convolution
backward and fp16 attention backward. Those primitive probes did not fit a model;
the separate two-step SDXL smoke below did. This is
an unsupported compatibility workaround, not proof of complete training stability.
No global environment or system-package changes were made.

The isolated environment is `/home/valdy/.cache/haremizer-runtime`; observed
native/override evidence lives in ignored `artifacts/runtime-native.json` and
`artifacts/runtime-override.json`. A bounded Ubuntu 24.04 container diagnostic using the same mounted host
PyTorch 2.11 + ROCm 7.2 libraries mirrored the host: native primitive probe
SIGSEGV, explicit gfx1030 override primitive probe success. This was not a
self-contained production image and did not train. Diagnostic image:
`sha256:ce310911b08dd152269a16831d74a71cabfd5c4816f6c104e708119fe750522a`;
ignored evidence: `artifacts/runtime-container-native.json` and
`artifacts/runtime-container-override.json`. No further version shuffle is planned.

## Local PNG testbed

Pillow must already be available; missing Pillow is reported by name, not installed.
Tests synthesize tiny non-semantic PNG fixtures inside temporary directories, not
inside the ignored `/test/` directory. No ML dependencies, network, or TTY needed:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v
python images.py validate /path/to/portrait.png --size 48
python images.py sheet /path/to/a.png /path/to/b.png \
  --output artifacts/contact-sheet.png --scale 4 --columns 2
python runtime.py                       # facts only, no torch import
python runtime.py --torch-probe         # optional tiny GPU primitives, not training
```

Contact sheets use integer nearest-neighbor scaling with transparent padding for
mixed native sizes, never stretch a 48 portrait to match a 64 portrait. Sheets
are viewing aids, not format candidates or quality verdicts.

## Explicit CPU grid recovery

`recovery.py` postprocesses a local raw single-frame square PNG (64..1024px,
<=16 MiB, verified before decoding). Pillow must already be installed. Python API:
`recover(source, output, background, size=64, tolerance=16, colors=32)`; explicit
RGB background is required, never inferred from borders. Example:

```sh
python recovery.py --source /local/raw/case-01.png --output /local/recovery/case-01 \
  --background 255 255 255 --size 64 --tolerance 16 --colors 32
```

Output must be a new directory outside the input directory, separate from any
term-chan export directory; even existing empty directories are rejected. The
four-connected edge flood fill removes only pixels within RGB Euclidean tolerance
0..64 of the explicit key (and traverses existing transparent pixels). Enclosed
same-key regions stay opaque; boundary-connected same-color foreground can be
removed incorrectly. Original composition is directly BOX-resized to 48 or 64,
alpha binarized at >=128, and opaque RGB MEDIANCUT-quantized without dithering to
2..62 colors. Hidden transparent RGB is zeroed. `source.png` retains exact input
bytes; `manifest.json` records source/output hashes, all controls, and factual
`images.validate_png` metrics. All-opaque/all-transparent results are rejected;
failed final validation can leave the source copy and rejected output, without a
success manifest. Use another new directory for a retry.

This is CPU postprocessing, **not native-layout diffusion, visual quality
acceptance, or sprite export**. No eye/face templates or automatic training feed
are used. Format validity does not make a recovered portrait an accepted sprite.

## Annotated local blink frame testbed

`blink.py` requires a **per-generated-portrait eye annotation**, not an autonomous
landmark detector. It is a limited frame-editing testbed, **not a trained blink**,
closed-eye semantic pass, quality acceptance, promotion, or sprite export. The
user/main reviewer owns actual eye regions, closed-lid edits and visual review;
changing arbitrary skin pixels does not establish a blink. No face skeleton,
generic eye template, filtering, interpolation or cropping is used.

API: `blink(source, annotation, output)`; CLI:

```sh
python blink.py --source /local/portraits/portrait.png \
  --annotation /local/eye-annotation.json --output /local/blink-frame
```

Annotation JSON has exactly these keys (replace hash/regions/edits with actual
portrait-specific evidence; `machine:sol` records machine, not human review):

```json
{"version":1,"reviewed_by":"machine:sol","source_sha256":"REPLACE_WITH_FULL_SOURCE_BYTE_SHA256","eye_regions":[{"x":2,"y":2,"width":3,"height":2}],"edits":[{"x":2,"y":2,"color":"#506478"}]}
```

The hash must match exact source PNG bytes before editing. Source must satisfy
`images.py`'s 48/64 single-frame, mixed binary-alpha, <=62-color contract.
One or two in-bounds rectangles use exact integers (no booleans), width <=16,
height <=10, union <=128 pixels. Edits are nonempty, unique integer coordinates
inside those regions on opaque source pixels; `#RRGGBB` colors must already exist
in the opaque source palette. At least one and at most 128 pixels must actually
change. Only specified pixels change; the full alpha channel and every RGBA pixel
outside the declared regions are explicitly checked after saving.

Output must be a new directory outside the input image folder and must not
contain/overwrite either input. It retains exact `source.png` and
`annotation.json` bytes, writes `blink.png`, and records hashes, regions, counts,
factual PNG metrics and preservation checks in `manifest.json` with
`visual_quality_assessed: false`. Declared eye regions are not automatic detection.
A failed save/validation may leave artifacts but no success manifest; retry in a
new directory. Pillow is loaded lazily through `images.py`; no ML is invoked.

## Lossless local term-chan sprite export

`export_sprite(base, newcharacter_directory, blink=None, name='neutral',
provenance=None)` in `sprite_export.py` exports supplied single-frame 48/64 PNGs
using `images.py` checks and bounded snapshots (16 MiB per input). No resizing,
quantization, recoloring, generation or quality acceptance occurs. Optional blink
must have the same geometry. The combined opaque RGB palette must fit 62 colors;
RGB-sorted indices use `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`,
with `.` for transparency. Opaque RGB and binary alpha round-trip exactly; hidden
RGB beneath transparent pixels is intentionally not serialized. The shared palette
is **local to these frames**; the parent loader does not enforce shared palettes
across expressions. Exporting a blink does not assert localization or closed eyes:
this tool exports given frames only, without blink-tool manifest verification.
Missing blink/talk frames use the parent's base fallback; no fake animation is added.

```sh
python sprite_export.py --base /local/portraits/base.png \
  --blink /local/blink-frame/blink.png --output /local/characters/new-character
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_sprite_export
# Optional pure Node integration; no TTY or parent writes:
PYTHONDONTWRITEBYTECODE=1 TERM_CHAN_LOADER_TEST=1 python -m unittest -v test_sprite_export
```

The character directory must be new and outside input image directories; it
contains only `<name>.json` (default `neutral`, safe lowercase filename slug).
Provenance defaults to the sibling `<character-directory>.provenance.json`, or
use `--provenance` for a new path outside the character directory. No existing
paths/inputs are overwritten. Do not put a manifest in the character directory:
the parent loader reads **all** JSON files as expressions. Provenance records
input byte hashes, output JSON hash, source paths, dimensions, shared palette and
factual PNG metrics, not a quality pass or automatic acceptance. I/O failure may
leave partial output; retry with new paths. Sources remain unchanged.

**Pane fit is separate from format validity:** 64px portraits need roughly 35
terminal rows, 48px roughly 27, before a speech bubble (two pixel rows per terminal
row plus UI overhead). A valid export need not fit a short terminal split.

## Local pretrained sampling controls

`sample.py` samples only a locally supplied SD1/SDXL safetensors pipeline; it
never downloads models. Supply `--model`, a new `--output` directory, and exactly
one of `--prompt` or `--suite`. Omitted dimensions now default to **768×768 for
SDXL**, the exercised local inference size, and 512×512 for SD1. Explicit sizes
remain supported for other environments; the observed 1024px route stays
operationally disabled on this machine after its GPU hang. A safer default is
not a guarantee against instability in this unsupported runtime.
Optional `--negative-prompt` must be a nonblank
string (default: omitted / Python `None`). `--guidance-scale` must be finite and
within 1..15 inclusive (default: 7.5). Both controls are forwarded unchanged to
the pipeline and recorded in manifest `settings`; frozen suite bytes, positive
prompts, and seeds are unchanged. Outputs remain raw inference PNGs, not
validated sprites or quality acceptance.

Optional `--scheduler euler_ancestral` (Python `scheduler=None` by default) is
an explicit diagnostic override, not a quality-improvement claim. Omission
preserves the locally cached scheduler. The only override uses standard
`EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)` after
local loading and before GPU placement/offload; it does not modify the model
cache or weights. Manifest `settings.scheduler_class` records the actual selected
class and `settings.scheduler_override` records the explicit choice or `null`.
Compare new diagnostic baselines and candidates with matching scheduler settings;
older manifests lack this metadata. No prompts, seeds, steps, CFG or dimensions
are changed by the override. Outputs remain raw PNGs, not quality acceptance.

Optional `--lora /local/path/pytorch_lora_weights.safetensors` and
`--lora-sha256 FULL_LOWERCASE_64_HEX_SHA256` must be supplied together (Python
keywords: `lora=None`, `lora_sha256=None`). The adapter must be a real nonempty
local `.safetensors` file at most **256 MiB**; this bounded allowance covers the
170.5 MB pretrained Pixel Art XL alternative as well as the 23.4 MB local
runtime/pilot adapters. It is not permission for unrestricted checkpoint loads.
Its full byte hash is verified before ML
imports and again just before loading. No downloads, pickle fallback, fusion,
or strength control are added. The standard diffusers local-only safetensors
loader runs before GPU placement/CPU offload. Output must be new and outside the
base model and adapter source; manifest `lora` records supplied path, byte count,
and SHA256 (or `null` without an adapter). Inference uses the fp16 pipeline;
normal inference dtype casting is distinct from the smoke's exact fp32 CPU
training save/reload verification. Sampling never starts training or promotes
an adapter. GPU inference with the runtime-smoke adapter passed four matched
cases in `artifacts/smoke-lora-suite-01` (~6.4 GB peak allocated); this is runtime
evidence, not quality improvement or an accepted corpus.

## Bounded SDXL attention-LoRA runtime smoke

`smoke_sdxl.py` is a separate **1–2 total optimizer-step plumbing smoke**, not the
legacy SD1 adapter, a quality trainer, promotion, or synthetic-feedback loop.
It records the existing broad user authorization and truthful `machine:sol`
review, never invents human approval or calls the legacy preparation gates.
Actual GPU evidence: `artifacts/sdxl-smoke-02/outcome.json` records
`smoke_passed`, two 512px optimizer steps, finite losses/gradients, a nonzero
adapter update, identical exhaustive frozen-UNet hashes, and exact fp32 CPU
adapter reload through diffusers. It took about 91 seconds wall time (86.6 seconds
recorded smoke duration), with peak allocated memory 5.66 GB. This is runtime
plumbing evidence only, not a quality corpus, accepted sprite, or promotion.
GPU inference with the trained smoke adapter passed four matched cases in
`artifacts/smoke-lora-suite-01` (~6.4 GB peak allocated), with no demonstrated
quality improvement or accepted corpus. Attempt 01
failed before training on the optional `amdsmi` device-memory metric; unavailable
metrics are now recorded nonfatally, not as zero.
Separate 1024px pretrained inference hung after 28 denoising steps (exit 134);
the exact failing post-denoising operation was not instrumented. That experiment
remains disabled, not retried. Earlier 768px inference passed. Neither proves
training stability. This smoke stays at 512px, with no automatic scaling/retry.
Native 64px pretrained samples were not viable portraits and are not training
data or a quality pass.

Supply `--request` JSON and a new `--output` directory. Paths are absolute or
relative to the request file. Example shape (replace all evidence placeholders
with real local paths/hashes, captions, sources, and attribution):

```json
{
  "model": {"path": "../models/hub/models--cagliostrolab--animagine-xl-4.0/snapshots/2b7c1b397761bf5bd3cc42e5b39ec99314a75a96", "id": "cagliostrolab/animagine-xl-4.0", "revision": "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96"},
  "config": {"steps": 2, "batch_size": 1, "resolution": 512, "rank": 4, "seed": 1234, "learning_rate": 0.0001, "gradient_clip": 1.0},
  "reviewed_by": "machine:sol", "declared_smoke_only": true,
  "authorization": {"user_authorized": true, "record": "User authorized bounded autonomous local training and process-local HSA_OVERRIDE_GFX_VERSION=10.3.0; this records broad authorization, not human artwork review."},
  "examples": [{"path": "../data/example.png", "sha256": "REPLACE_WITH_IMAGE_SHA256", "caption": "REPLACE_WITH_REVIEWED_CAPTION", "source_url": "https://REPLACE_WITH_REAL_SOURCE", "license": "CC-BY-4.0", "credit": "REPLACE_WITH_ACTUAL_ATTRIBUTION", "source_page": {"path": "../data/source.html", "sha256": "REPLACE_WITH_SOURCE_PAGE_SHA256"}}]
}
```

Only 1–2 supplied native 64×64 PNGs and CC-BY-4.0/CC0-1.0 declarations are
accepted; hashes bind source PNG and local source-page bytes. This validates
machine declarations, not legal truth or aesthetics. Model pinning uses the
same HF cache convention below, or a 64-hex content digest. ML imports are lazy,
loads are local-only safetensors without remote code, and offline flags are set;
this is not an OS network sandbox. The isolated Python 3.12 runtime must match
torch 2.11.0+rocm7.2, diffusers 0.41.0, transformers 5.19.0, peft 0.21.2 and
accelerate 1.15.0. No installs/downloads, CPU training fallback, or retries.

Original PNG/alpha bytes are retained; cached RGB input explicitly records
64→512 nearest resizing and white alpha flattening. Dual SDXL conditioning and
fp32 VAE posterior-mode scaled latents are cached, then encoders/VAE are released
before UNet backward. Frozen fp16 UNet, fp32 rank-4 attention adapters, AdamW,
checkpointing, fp16 autocast/GradScaler, finite checks, exhaustive frozen-parameter
hashes, and a nonzero update check test plumbing only. The stop threshold is
**11 GiB peak allocated**, checked between phases/operations, not a hard allocator
cap; peak reserved/device usage are also recorded. CPU save/reload verification
uses the actual diffusers API, deleting the old adapter and temporarily upcasting
the CPU base to preserve fp32 adapter bytes (extra host memory required).
`outcome.json` atomically checkpoints phase boundaries, training start and each
completed optimizer step, plus hashes, versions, memory, timing and failures.
Native crash or external timeout can leave `status: running`; that is not
success and the coordinator must record process termination separately.

Parent/coordinator must wrap the entire process in a **900-second timeout**:

```sh
timeout --signal=KILL 900s env HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  /home/valdy/.cache/haremizer-runtime/bin/python smoke_sdxl.py \
  --request artifacts/sdxl-smoke-request.json --output artifacts/sdxl-smoke-01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_sdxl_smoke
# Optional CPU-only real save/delete/reload API check, not a model/GPU smoke:
PYTHONDONTWRITEBYTECODE=1 SDXL_SMOKE_CPU_API_TEST=1 \
  /home/valdy/.cache/haremizer-runtime/bin/python -m unittest -v \
  test_sdxl_smoke.SmokePreflightTests.test_real_diffusers_save_delete_reload_cpu
```

### Explicit experimental style-learning pilot

The same script additionally supports one bounded **experimental 100-step style
pilot**, only with CLI `--pilot` (Python `preflight(..., pilot=True)` or
`run(..., pilot=True)`; default `pilot=False`). Default smoke request/CLI contracts
remain unchanged: 1–2 steps, 1–2 examples, `declared_smoke_only: true`, and no
heldout keys. Pilot requests instead require `declared_smoke_only: false`, exactly
100 steps, 4–16 training `examples`, and 2–16 `heldout_examples`. All other config
values remain fixed: rank 4, 512px, batch 1, seed 1234, learning rate 1e-4 and
clip 1. Every example in either split has the existing exact keys plus a nonblank
`identity_group`. At least four distinct training groups and two distinct heldout
groups are required; multiple distinct expression images within a training group
are allowed. Groups must not intersect across splits. Group labels are declarations,
not automatic proof of identity independence or source diversity.

Both splits undergo local native 64×64 single-frame PNG, byte-hash, source-page
hash, caption and CC-BY-4.0/CC0-1.0 declaration validation before ML imports.
Duplicate image bytes or canonical RGBA pixels are rejected across all pilot
inputs, including different PNG encodings and hidden RGB under zero alpha.
For inputs with at most 62 visible RGB colors, an additional exact alpha/color-
partition fingerprint rejects whole-image palette swaps, even under different
claimed identity groups. Higher-color inputs decline this narrow guard rather
than treating every unique-color photo as a recolor. This is **not** a semantic
identity, shared-face-skeleton, near-duplicate, age, or rights detector; grouping
and source review remain necessary.
`outcome.json` keeps training records in `inputs` only; `heldout_inputs` and
`heldout_count` record separate heldout provenance/hashes. Heldout captions and
pixels never enter training conditioning or caches. Each run starts from the fresh
pinned base, with the same finite/frozen/update/checkpoint/memory checks and atomic
progress as the smoke; no incremental resume, automatic evaluation, early-best
selection, sweep, retry, scaling, promotion or synthetic feedback is added.

Successful execution reports `pilot_completed_unreviewed`, **not** `smoke_passed`
or quality acceptance; `quality_assessed`, `promotion` and `synthetic_feedback`
remain false. The executed `artifacts/style-pilot-request-01.json` contains
six training and two heldout identities from the same pebonius source, with
machine-reviewed rights and adult-appearing descriptions (canonical ages not
supplied). It is a small learning experiment, **not an accepted quality corpus,
production training or a quality claim**. GPU execution completed all 100 steps;
`artifacts/style-pilot-01/review.json` records the subsequent quality rejection.
Parent/coordinator must bound any separately justified future pilot to 900 seconds;
1024px remains disabled, with no automatic extension. Historical invocation
(the existing output is deliberately not overwritable):

```sh
timeout --signal=KILL 900s env HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  /home/valdy/.cache/haremizer-runtime/bin/python smoke_sdxl.py --pilot \
  --request artifacts/style-pilot-request-01.json --output artifacts/style-pilot-01
```

### Bounded weights refinement

`--refine` (Python `refine=True`, default `False`) is mutually exclusive with
`--pilot`; both Python selectors require actual booleans. The original smoke
and pilot contracts are unchanged. Refinement uses the pilot's train/heldout,
source/hash, identity-group and anti-recolor gates, with
`declared_smoke_only: false`. It permits 50–600 total optimizer steps, a uint32
seed and finite learning rate 1e-5–1e-4; batch 1, rank 4 and gradient clip 1
remain fixed. Effective training `resolution` may be **512 or 768**, never 1024.

Refinement config additionally requires `content_resolution`: 384/448/512 for a
512px canvas, or 576/672/768 for a 768px canvas. Original 64px RGBA is nearest-
scaled to that square, centered on the effective-resolution white canvas with
explicit alpha compositing, then converted to RGB. Exact resize, symmetric
padding, background, cached PNG hash and raw RGB hash are recorded. SDXL time
IDs, input tensors and latents use the actual training resolution consistently.
Only training captions/latents are cached; heldout inputs never condition
training. Legacy smoke/pilot and 512-content preprocessing are unchanged.

768px backward memory/stability must be exercised in one bounded 50-step stage
before scaling; 512px success does not establish its fit. The same 11 GiB
allocator stop threshold and coordinator deadlines apply. A trainer upgrade
changes the frozen module binding: archive the old run and initialize a new
bounded coordinator with the **original work-window deadline**, never silently
reset the 12-hour budget.

An optional refine-only `prior_preservation_weight` (finite 0..1, default 0)
adds `weight * MSE(student_prediction, detached_base_prediction)` to the source
noise/velocity loss. The base prediction uses the **same licensed training
latents, noise, timestep and conditioning**, with adapters disabled, gradient-free
evaluation, and adapter/training state restored even on error. No heldout
conditioning or new synthetic image data is used. This anchors the base denoiser,
not a guarantee of preserved geometry or style; both losses are separately
recorded and actual visual comparison must detect style erasure/regression.
Predeclare the weight per request; initial experiment is 0.1 for 50 steps.

An optional **refine-only** `facial_loss_weight` (finite numeric 1..4, default 1)
weights source-annotated facial denoising errors. Weight 1 uses exactly the legacy
Torch MSE path, without a mask cache. At weight >1, **every training example**
must include `facial_regions`, a list of 1..8 manually supplied rectangles such as
`[{"x":18,"y":22,"width":28,"height":20}]`. Coordinates are native 64px,
strict integers (not booleans), with positive extents wholly inside the source;
x/y are 0..63 and right/bottom endpoints are exclusive. Overlaps form a binary
union, never multiply weights. Annotations are optional but validated at weight
1; heldout rows, smoke and pilot requests cannot carry them. These are supplied
source annotations, not detected landmarks or claimed human review.

Masks follow the exact content nearest-neighbor resize and centered padding of
`prepare_rgb`, then nearest-neighbor downsampling to the actual 64×64 or 96×96
latent grid. Padding is unselected; source annotations are independent of alpha.
Each source caches only one fp32 `1 + (weight - 1) * mask` CPU tensor (at most
36 KiB; at most 16 sources), transferring only the current source's tensor to GPU.
The noise/velocity loss is `sum(weights * squared_error) / sum(expanded_weights)`
across batch/channels/spatial dimensions: outside weight remains 1 and the global
loss scale is normalized rather than implicitly multiplying the learning rate.
The detached source-only prior term is unchanged. Facial noise weighting alone
introduces no VAE decode gradients, new model/detection, heldout geometry/loss,
generated-image feedback, or pixel postprocessing.

Outcome evidence records source hashes/regions, content/padding/latent layout,
binary-mask hash and selected-pixel count, cache size, weight, and per-step
`weighted_noise_loss` plus source hash. Reconstructing the documented mapping
allows auditing mask placement against cached input layout. This is a learned
weighted-noise objective, **not a guarantee of bold eye/nose/mouth contrast,
strict proportions, or unclipped heads**. Any new explicit retraining needs a
separately bounded source-annotated pilot and independent visual evaluation;
it is not a robust model or promotion. The completed original 12-hour work
window above is archival, not a reusable clock for a renewed request.

An experimental refine-only `decoded_feature_loss_weight` (finite numeric
0..0.25, default 0) adds a **source-only decoded facial L1** term. Missing/zero
keeps the legacy loss and VAE deletion path exactly; no auxiliary cache, decode,
or CPU-thread change occurs. Positive weight requires valid `facial_regions`
on every training row; heldout annotations remain forbidden. Full-image facial
crop requests are rejected before ML imports. This is not a contrast/geometry
success claim: a new frozen coordinator and bounded 50-step memory/visual gate
are required before any longer fit.

At timesteps `t <= 250`, derive unclipped clean latents without detach:
`x0=(xt-sqrt(1-alpha_bar_t)*epsilon)/sqrt(alpha_bar_t)` or, for velocity,
`x0=sqrt(alpha_bar_t)*xt-sqrt(1-alpha_bar_t)*v`. Crop the latent facial-union
bounding box plus two latent cells of context, then align bounds outward to
`LCM(VAE stride 8, display stride resolution/64)` and clamp to the image.
This means latent alignment 3 at768 and1 at512. Only this
crop is transferred differentiably to CPU and decoded after dividing by the
local VAE scaling factor. After the ordinary latent cache, the frozen local VAE
is retained on CPU fp32 in eval mode; its parameters never receive gradients.
Process-local Torch CPU threads are bounded to two only when enabled. No full
768px gradient decode, new model, GPU VAE during training, generated GT or
heldout conditioning is introduced.

Ground truth is the corresponding **actual `prepare_rgb` source RGB crop**,
not a VAE reconstruction. Both decoded RGB `(decode+1)/2` (unclamped) and source
RGB `[0,1]` are AREA resized to exact `crop_height/display_stride` by
`crop_width/display_stride`. Crop boundaries are aligned to the global64 display
grid: no fractional resize or phase shift is introduced. The same source-rectangle union is resized with
AREA to form fractional coverage weights. The auxiliary loss is
`sum(abs(decoded64-source64)*mask64) / sum(mask64 expanded over RGB channels)`.
No additional dark-ink weighting is used. The differentiable CPU scalar moves
back to the prediction device and adds `decoded_feature_loss_weight*decoded_loss`
to the unchanged noise/prior objective. Inactive steps add no auxiliary term.

Evidence records crop bounds, two-cell margin, source/prepared-RGB/crop-GT/mask
hashes, mapping, cache bytes, VAE weight hash/frozen/device/dtype state and
per-step active/inactive status, timestep, source hash and decoded loss. Tests
cover pre-ML bounds/annotations, padding/layout/global display phase (including
native one-pixel ink), both x0 formulas and CPU toy input gradients with frozen
parameters and crop locality. ML-dependent tests use the existing optional
`SDXL_SMOKE_CPU_API_TEST=1` guard. Cropped VAE context, CPU time/RAM and
cross-device backward remain risks;
these tests are not a real-GPU memory or visual acceptance gate.

An experimental refine-only `caption_dropout_probability` (finite numeric
0..0.5, default0; booleans rejected) explores source-conditioning generalization.
Positive probability caches **one shared empty-caption dual-CLIP conditioning**
using the same frozen local encoders as full captions, including the projected
pooled embedding. Full/empty conditioning tensor digests, caption hashes, encoder
weight hashes, cache bytes, probability and independent RNG seed are recorded.
Per optimizer step, an independent `random.Random(config_seed)` chooses full or
empty caption; it never advances Torch noise/timestep or global random generators.
Only the chosen token/pooled tensors transfer to GPU. Student and detached
source-only prior receive exactly the same chosen conditioning. Licensed source
latents, prepared RGB GT and masks are unchanged; no text-encoder training,
source-caption edits, generated-image feedback or inference-prompt changes occur.
Per-step evidence records `caption_dropped`, chosen-caption SHA256 and source
SHA256. Default/missing0 performs no extra encode/cache/RNG construction or draw
and retains the legacy training path. This hypothesis is not a quality claim;
a fresh bounded coordinator and parent actual-image gate are required.

Refinement requests also require `continuation`, either `null` for the unchanged
fresh-base LoRA initializer or exactly:

```json
{"path":"/local/parent/pytorch_lora_weights.safetensors","sha256":"FULL_LOWERCASE_SHA256","outcome":"/local/parent/outcome.json","outcome_sha256":"FULL_LOWERCASE_SHA256"}
```

Continuation accepts only local, nonempty own fp32 rank-4 diffusers attention
safetensors up to 64 MiB, bound by full hashes to a completed haremizer outcome's
checkpoint, rank, identical base ID/revision and full model-file hashes. These
checks do not import ML packages. Output must be new and must not contain or
replace continuation inputs. Parent checkpoint bytes are never rewritten.
The actual diffusers loader injects parent weights on CPU with an fp32 base,
checks exact adapter tensor equality, then restores frozen base parameters to
fp16 before GPU placement. No silent fp16 adapter round-trip is permitted.
Training still uses fp32 adapters, checkpointing/autocast and the existing
finite/frozen/update/save-reload checks. Each stage creates **new AdamW state**:
this is weights continuation, not exact optimizer resume. Parent provenance,
starting adapter identity and final update relative to starting weights are
recorded atomically with progress.

Completion is `refinement_completed_unreviewed`, with quality and promotion
false. No autonomous loop, evaluation, quality acceptance or synthetic feedback
runs inside this trainer. The parent owns iterative visual review and must wrap
each stage in the same 900-second timeout, within the user's 12-hour autonomous
window and <100 GB total storage ceiling. No driver changes, installs, network
access or paid services are added by this seam. CPU API tests establish exact
reload only; GPU refinement/continuation still requires separate execution
verification and independent quality review.

## Dataset and rights input contract

Use separate local imagefolder roots, e.g. `data/approved/train/` and
`data/approved/eval/`. Each needs `metadata.jsonl`, one object per captioned PNG:

```json
{"file_name":"portraits/example.png","text":"Adult anime woman with short curly hair, warm brown skin, green jacket"}
```

Names are relative POSIX paths; absolute paths, `..`, ambiguous components,
backslashes, escaping symlinks, and duplicate names are rejected. Blank lines are
ignored, but each split must have at least one image. Only captioned entries are
included; unlisted files are not training inputs. Captions must be nonempty strings.
PNG validation defaults to either 48 or 64; repeat `--size` to explicitly select
allowed dataset sizes. Train and held-out eval must contain no shared image hashes;
this catches exact duplicates, **not** near-duplicates or semantic leakage. Human
review must check those, adult subjects, and real character diversity.

Supply a separate explicit rights manifest covering exactly the listed files:

```json
{
  "version": 1,
  "reviewed_by": "human reviewer identity",
  "entries": [
    {"split":"train","file_name":"portraits/example.png","source":"original source reference","license":"actual license or permission reference","training_allowed":true},
    {"split":"eval","file_name":"heldout.png","source":"original source reference","license":"actual license or permission reference","evaluation_allowed":true}
  ]
}
```

These are auditable **human declarations**, not automatic legal verification. A
missing, denied, duplicated, or incomplete permission blocks preparation. Record
real sources/permissions, not these example placeholders.

Dataset SHA-256 binds sorted split/path/caption records, image byte and canonical
pixel hashes, factual metrics, accepted sizes, and normalized rights declarations.
It ignores absolute root location, metadata row order, JSON formatting, and blank
lines; changing captions, pixels, PNG bytes, or rights changes its identity.

## Preparation plan and hard stop

No default model is selected. Supply model identity and a full 40/64-hex immutable
commit/content revision explicitly; branches, tags, arbitrary names, and partial
hashes are blocked. Humans must verify license and revision before separate training:

```sh
python prepare.py \
  --train data/approved/train --eval data/approved/eval \
  --rights data/approved/rights.json \
  --model OWNER/APPROVED-MODEL --revision FULL_40_OR_64_HEX_DIGEST \
  --suite evaluation-suite.json --torch-probe \
  --training-resolution 512 --steps 100 --rank 4 \
  --output artifacts/preparation.json
```

This only writes a plan. Output must be new and outside dataset roots/input files.
Preparation never downloads a model/dataset or launches training. The separately
gated process adapter below is implemented but has never run real training. Default prospective
config: batch 1, rank 4, finite 100 steps, seed 1234. Limits: rank 1..16, steps
1..1000, training resolution 64..1024 divisible by 8. **Training resolution is
separate from final independently composed 48/64 output targets**; a working
native-layout generation pipeline remains to be designed and evaluated.

- Missing data, rights, model/revision, valid suite/config, or verified GPU
  primitives: `status: blocked`, exit code 2, with reasons and runtime evidence.
  Without `--torch-probe`, operations deliberately remain unverified.
- All factual prerequisites present: `status: needs_human`, exit code 0, **not
  ready to train**. Quality fields remain `unassessed`.
- Every plan: `training_authorized: false`, `training_started: false`, no training
  command. Setup/preparation approval is not training authorization. There is no
  automatic approval/resume path and no generated-data feedback loop.

The plan embeds exact suite JSON and its byte hash. `preparation_sha256` binds
valid dataset digest, suite hash, explicit model/revision, finite config, and the
runtime contract (Python interpreter/version, torch/HIP/CUDA versions, GPU device
and verification, recorded override/visibility environment). It is null when
model/data/suite/config inputs are incomplete. Runtime failure may leave a binding
but still blocks preparation; changing backend or override changes the digest.
This digest is an audit identity, not enforcement
by the existing core harness or a training capability token.

Before any future training: humans must approve rights and held-out separation,
model/license/immutable revision, a real version-pinned trainer, compute and
wall-time/memory/storage budget, and independent calibrated visual evaluation.
A separate explicit training authorization must name the preparation digest.
No real visual evaluator or approved training stack exists yet. Passing unit tests or PNG format
checks cannot authorize promotion or claim quality/diversity/non-regression.

## LoRA process adapter: preview only at this checkpoint

`backend.py` implements `core.Trainer` for a **locally supplied, SHA-256-pinned**
upstream diffusers SD1.x `train_text_to_image_lora.py`. No script, model, or
package is supplied/downloaded. Its only CLI previews; it has no approve/run CLI.
**Stop here for human review. Do not call `train` or `Run.execute` now.** Tests
mock both runtime revalidation and training processes; test checkpoints are fake.

First create a core run with the prepared dataset SHA-256 and frozen suite, with
an explicit finite `--max-attempts` (1..3). The adapter must run under the same
Python executable recorded by preparation. Nonempty `PYTHONPATH`, `PYTHONHOME`,
`LD_LIBRARY_PATH`, or `LD_PRELOAD` are rejected by name: prepare a clean dedicated
runner environment, not an implicitly different one. They are never unset
globally. This intentionally excludes the mounted-sitepackages diagnostic
container above from production adapter use. Supply local `request.json`:

```json
{
  "preparation": "/absolute/path/preparation.json",
  "train": "/absolute/path/approved/train",
  "evaluation": "/absolute/path/approved/eval",
  "rights": "/absolute/path/rights.json",
  "suite": "/absolute/path/frozen-suite.json",
  "run_directory": "/absolute/path/core-run",
  "python": "/absolute/path/prepared/python",
  "trainer": "/absolute/path/train_text_to_image_lora.py",
  "trainer_sha256": "EXPLICIT_SCRIPT_CONTENT_SHA256",
  "model_path": "/absolute/path/cached-model",
  "timeout_seconds": 600,
  "max_attempts": 1,
  "mixed_precision": "no",
  "preprocessing": {"resize": "NEAREST", "alpha": "flatten", "background": [255, 255, 255]},
  "script_contract": {
    "name": "reviewed-diffusers-sd1-text-to-image-lora-v1",
    "trainer_sha256": "EXPLICIT_SCRIPT_CONTENT_SHA256",
    "reviewed_by": "ACTUAL_HUMAN_REVIEWER",
    "note": "ACTUAL_PINNED_SOURCE_AND_DEPENDENCY_REVIEW_REFERENCE"
  }
}
```

```sh
python backend.py /absolute/path/request.json > /absolute/path/preview.json
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_backend
```

The script contract is a **human source review declaration**, not an automatic
upstream verification. Before declaring it, verify the pinned script and dependency
versions support every emitted argument (notably `--rank`, `--report_to none`,
`--center_crop`, checkpointing and precision), default to ordinary AdamW without
8-bit/xformers/TF32, honor offline local-only loading without remote/custom Python
code, and save final `pytorch_lora_weights.safetensors`. Review SD1 architecture
and model/license compatibility too. No help command executes in preview. Current
upstream main may require development diffusers; do not assume installed stable
versions suffice. Preprocessed images already equal the training resolution, so
upstream resize/crop must be identity-sized; review this behavior to avoid its
usual non-nearest resize default. No unverified interpolation flag is guessed.

Only explicit alpha flattening onto the recorded RGB background is supported:
upstream RGB training cannot preserve transparency. Approval must cover that
loss and nearest-neighbor scaling; final native-layout generation is not
implemented. Explicit CPU transparent output postprocessing is separate (above).
`mixed_precision` must be `no` or explicitly approved `fp16`; batch 1,
accumulation 4, rank/steps/resolution from
preparation, fixed AdamW learning-rate/scheduler arguments, and no TF32 flag.

Local models require either a 64-hex revision equal to
`prepare.canonical_digest({relative_posix_filename: file_sha256, ...})` over all
files, or standard HF `models--OWNER--MODEL/snapshots/FULL_40_HEX_COMMIT` matching
the prepared model ID/revision. All cached file hashes additionally enter the
execution digest. HF cache directory names assert upstream revision provenance;
humans must verify their authenticity. Model components cannot declare custom
pipeline modules. Partial/incompatible caches fail during upstream loading.

Preview emits an exact argv template (`{attempt}` is substituted only in the two
adapter-owned directory arguments), execution SHA-256, and incomplete approval
payload with `training_authorized: false`, empty human identity and note. It never
fills human approval. A separate human-written JSON must explicitly set
`training_authorized: true`, `approved_by`, and `note`, matching both preparation
and execution digests. For an override, the note must include the exact
`HSA_OVERRIDE_GFX_VERSION=VALUE` and `unsupported compatibility workaround`.
No digest-specific human approval has been supplied to this legacy SD1 adapter.
The broader autonomous-work authorization is recorded above; it is not an
invented digest-specific approval file.

Future authorized integration uses `LoraTrainer(request_path, authorization_path)`
as the injected trainer for the matching `core.Run`, with a separately implemented
independent reviewer. Setup approval alone cannot authorize it. Authorization
covers the bounded run envelope, not an arbitrary new run: roots/run directory,
model files, dataset/rights, frozen suite, preparation config/runtime, script hash
and reviewed contract, interpreter binary hash, installed distribution versions
and METADATA/RECORD hashes, sanitized child environment, preprocessing, argv
and per-process timeout/max attempts are bound. Attempts must be empty active
core-owned directories. Authorization is rechecked on every attempt. After
approval only, a tiny GPU primitive reprobe must match the frozen runtime;
all inputs are revalidated again before launching. Native and override evidence
remain distinct.

The child uses argv without a shell, isolated Python, only a local cached model,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (effective local-only hub loading),
`HF_DATASETS_OFFLINE=1`, no inherited Python-path/telemetry environment, and no
remote-code flag. This is not a network sandbox or protection from malicious
trusted local scripts/packages. Runner distribution metadata is not full package
code attestation; mutable files still have normal filesystem check/use races.
Timeout/error/interruption sends SIGKILL to the entire Linux child process group.
Cleanup waits at most five seconds; an unresponsive process is recorded as
`cleanup_pending`, and the failed run stops rather than starting another attempt.
The run binds at most `max_attempts * timeout_seconds` of training-process waits, not a
persisted global deadline: preprocessing, runtime probes and cleanup are outside
that budget; GPU memory/disk quotas are not enforced. Keep independent compute,
storage and script trust review before any execution.

Each started attempt retains `backend.json` with exact argv, approval, runtime
revalidation, input identities, preprocessed imagefolder digest and outcome, plus
`trainer.log`. Only a nonempty local final weights file is returned; the adapter
never evaluates/promotes, and no calibrated evaluator exists. Each candidate starts
from the approved base model; continuing an incumbent LoRA is not implemented.
Versioned experiments therefore do not yet imply incremental weight updates.
No real training, model access or GPU calls were performed to implement/test this seam.

## Existing core scaffold

```sh
python core.py --help
python core.py init /tmp/haremizer-run \
  --suite evaluation-suite.json --dataset-id approved-dataset-digest --max-attempts 3
python core.py status /tmp/haremizer-run
```

Future trusted adapters can use `Run.execute(trainer, reviewer)` after separately
approved integration. The harness freezes suite bytes, bounds attempts, requires
all five boolean review checks plus evidence, rejects uncertain candidates, and
preserves prior checkpoints for rollback. It does not yet enforce real compute,
wall-time, or visual-evaluation policy; do not attach real training during this
preparation milestone. Synthetic core tests prove control flow only.

Keep `.venv/`, caches, datasets, models, artifacts, and run outputs out of source
Git. `prompts/` is an independent locally ignored repository; do not stage it
through this repository. No hooks or global Git configuration are created by
these tools.

## Local commit policy

The user authorizes autonomous, reviewed per-hunk or cohesive milestone commits
**in haremizer only**. Keep commits local: never push. Inspect each staged diff,
run relevant tests, and use all-lowercase subjects prefixed with `[chore]`,
`[fix]`, `[test]`, or `[feat]`. Never stage the parent term-chan repository,
`prompts/`, model weights, datasets, caches, or generated artifacts.

This repository preference does not override higher-priority session
instructions. If an inherited session instruction prohibits commit commands,
prepare reviewable changes without committing until that instruction is updated.
