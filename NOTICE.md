# Publication notice

What this repository contains, what it deliberately leaves out, and the third-party work it depends on.

## Published

- Source, tests and documentation for the haremizer pipeline: prompt-conditioned portrait generation,
  bounded LoRA training and evaluation, deterministic sprite post-processing, and term-chan export.
- The generated term-chan sprites under `artifacts/sprites/<character>/`: the sprite JSON, PNG renders of
  those frames, and a preview. Each folder keeps its working files in an unpublished `internal/` folder.

## Not published

- **Training data.** The curated reference sets under `data/` are not redistributed. They are CC0 or
  CC-BY-4.0 pixel art used only to train adapters locally.
- **Model files and adapters.** No `*.safetensors`, checkpoint or base-model file is included. The
  trained LoRA adapters are local artefacts and are not distributed here.
- **Third-party weights.** The LayerDiffuse transparency assets are downloaded at runtime from their
  publisher and are not redistributed.
- **Runs.** `runs/` holds training runs, raw renders, logs, matrices and evaluation records and is
  excluded.

## Third-party components

| component | role | licence | redistributed here |
|---|---|---|---|
| Animagine XL 4.0 (`cagliostrolab`) | SDXL base model used for generation and as the frozen reference in training | CreativeML OpenRAIL++-M | no |
| LayerDiffuse code (`lllyasviel/LayerDiffuse_DiffusersCLI`, pinned commit `3061d9aed52a6c52a13fcf2b196c0fef4d727824`) | transparency attention offset and decoder | Apache-2.0 (see `LICENSE.layerdiffuse.txt`) | code adaptation only, in `layerdiffuse.py` |
| LayerDiffuse weights (`lllyasviel/LayerDiffuse_Diffusers`) | SDXL transparency offset and VAE decoder, hash-pinned | publisher labels the weight repository `openrail`, without a version or licence text | no |
| CC0 reference art (`everface`, `nekomimi`/cawfeecrow, druiddess) | training references | CC0 | no |
| CC-BY-4.0 reference art (`pebonius`, `alexwh`) | training references | CC-BY-4.0, attribution recorded in the run lineage | no |

Attribution for the reference art is recorded in the dataset manifests and run lineage files, which name
each artist, the source page, the licence and the transform applied.

## Reproducing the pipeline

Nothing here can retrain an adapter by itself: the reference images are deliberately absent. The code will
reproduce every deterministic step — keying, palette reduction, blink edits, sprite export — from any
render, and the tests cover those steps without a GPU.

## Licence of this repository

Not yet chosen. Until a `LICENSE` file is added, all rights are reserved by the author.
