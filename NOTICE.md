# Publication notice

What this repository contains, what it deliberately leaves out, and the third-party work it depends on.

## Published

- Source, tests and documentation for the haremizer pipeline: prompt-conditioned portrait generation,
  bounded LoRA training and evaluation, deterministic sprite post-processing, and term-chan export.
- The generated term-chan sprites under `artifacts/sprites/<character>/`: the sprite JSON, PNG renders of
  those frames, and a preview. Each folder keeps its working files in an unpublished `internal/` folder.

## Not published

- **Training data.** The reference sets under `data/` are not redistributed. Local lineage records
  identify CC0 and CC-BY-4.0 references; this notice is not a blanket clearance of all local data.
- **Model files and adapters.** No `*.safetensors`, checkpoint or base-model file is included. The
  trained LoRA adapters are local artefacts and are not distributed here.
- **Third-party weights.** Models and LayerDiffuse transparency assets must be provisioned separately
  as local files. The tracked sampler/trainer use local-only loading, and the LayerDiffuse validator
  checks existing hash-pinned assets; these tools do not download weights. No weights are redistributed.
- **Runs.** `runs/` holds training runs, raw renders, logs, matrices and evaluation records and is
  excluded.

## Third-party components

| component | role | licence | redistributed here |
|---|---|---|---|
| Animagine XL 4.0 (`cagliostrolab`) | SDXL base model used for generation and as the frozen reference in training | CreativeML OpenRAIL++-M | no |
| LayerDiffuse code (`lllyasviel/LayerDiffuse_DiffusersCLI`, pinned commit `3061d9aed52a6c52a13fcf2b196c0fef4d727824`) | transparency attention offset and decoder | Apache-2.0 (see `LICENSE.layerdiffuse.txt`) | code adaptation only, in `src/haremizer/layerdiffuse.py` |
| LayerDiffuse weights (`lllyasviel/LayerDiffuse_Diffusers`) | SDXL transparency offset and VAE decoder, hash-pinned | publisher labels the weight repository `openrail`, without a version or licence text | no |
| CC0 reference art (`everface`, `nekomimi`/cawfeecrow, druiddess) | training references | CC0 | no |
| CC-BY-4.0 reference art (`pebonius`, `alexwh`) | training references | CC-BY-4.0, attribution recorded in the run lineage | no |

Attribution for the reference art is recorded in local dataset manifests and run lineage files, which
are not distributed. The minimal [public sample record](docs/samples.md) retains verified available
credits and hashes, and explicitly identifies gaps. These records do not establish legal clearance of
generated samples or complete upstream model authenticity.

## Reproducing the pipeline

A checkout alone cannot retrain an adapter: reference images, weights and the ML runtime are absent.
CPU tools implement deterministic keying, palette reduction and sprite export for valid supplied
inputs; localized blink edits additionally require source-bound annotations. Tests cover these steps
without establishing semantic quality, automatic detection or robust native transparency.

## Licence of this repository

Original code is MIT licensed, copyright 2026 Haremizer contributors; see [LICENSE](LICENSE).
The adapted LayerDiffuse code in `src/haremizer/layerdiffuse.py` remains Apache-2.0 under
[LICENSE.layerdiffuse.txt](LICENSE.layerdiffuse.txt), with its existing upstream attribution preserved.
MIT does not relicense third-party code, model weights, datasets or generated-example artwork.
Model/data/generated-output terms must be checked separately before redistribution. Available local
provenance is incomplete for some samples; no artist permission, human approval or comprehensive
rights clearance is asserted. Historical documents predate this license decision and are not current
licensing or operational authority.
