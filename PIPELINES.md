# Haremizer pipelines

How a text description becomes a 64×64 term-chan sprite, and where every adapter and every
deterministic step fits. Five segments: training, inference, post-processing, the evaluation loop, and
the adapter inventory.

Code entry points referenced below:

| file | role |
|---|---|
| `smoke_sdxl.py` | LoRA training (`--refine`) |
| `sample.py` | inference, optional LoRA and LayerDiffuse |
| `layerdiffuse.py` | LayerDiffuse offset merge and CPU transparency decoder |
| `recovery.py` | explicit background colour key |
| `blink.py` | hash-bound eye-region blink edit |
| `sprite_export.py` | term-chan sprite JSON export |
| `refinement_loop.py` | bounded stage coordinator |
| `auto_loop.py` | unattended driver over a hypothesis list |
| `score_sprite.py`, `ink_stats.py` | deterministic scoring |
| `matrix.py` | side-by-side review pages |

## 1. Training

```mermaid
flowchart TD
    subgraph DATA["Data in"]
        A1["64x64 PNG references<br/>+ captions + licence"] --> A2["prepare_rgb<br/>NEAREST resize, alpha flattened, padded"]
        A1 --> A3["dual CLIP tokenizer<br/>token budget checked at 77"]
    end
    A2 --> B1["frozen SDXL VAE encode<br/>fp16 latents cached"]
    A3 --> B2["text_encoder + text_encoder_2<br/>conditioning cached"]
    subgraph ADAPTER["Adapter attached here"]
        C1["unet.add_adapter<br/>LoraConfig r=4, alpha=4<br/>targets: to_q, to_k, to_v, to_out.0"]
        C2["base UNet frozen fp16<br/>adapter fp32<br/>optional: freeze Q and K"]
        C3["optional continuation<br/>load previous adapter weights<br/>new AdamW state"]
    end
    B1 --> D1["per step: noise latent,<br/>predict noise"]
    B2 --> D1
    C1 --> D1
    C2 --> D1
    C3 --> D1
    D1 --> E1["loss = noise MSE"]
    E1 --> E2["+ prior_weight x prior loss<br/>frozen base predicts the same<br/>source latent, detached"]
    E2 --> E3["+ decoded_weight x decoded feature loss<br/>frozen CPU fp32 VAE decodes the<br/>facial latent crop at low t <= 250<br/>compared with the real 64px source"]
    E3 --> F1["backward, gradient clip, AdamW step"]
    F1 --> G1["save fp32 rank-4 safetensors<br/>+ outcome.json with hashes,<br/>durations, memory, reload check"]
```

Bounds applied at every stage: 900 seconds, 11 GiB allocator check, owned process group, storage
ceiling, and the plan's deadline and stage cap. Inputs are hash-pinned: model manifest, image bytes,
source pages, captions, groups and configuration.

## 2. Inference — where each adapter slots in

```mermaid
flowchart TD
    S1["load SDXL from local snapshot<br/>fp16, safetensors only"] --> S2["optional: EulerAncestral scheduler"]
    S2 --> S3["optional LayerDiffuse step 1:<br/>merge dense attention offset (1.9 GB)<br/>into UNet weights on CPU"]
    S3 --> S4["optional LayerDiffuse step 2:<br/>load portrait LoRA<br/>via diffusers load_lora_weights"]
    S4 --> S5["cpu offload or move to GPU"]
    S5 --> S6["prompt to dual CLIP"]
    S6 --> S7["28 denoise steps, CFG 5, 768x768"]
    S7 --> S8{"LayerDiffuse active?"}
    S8 -->|no| T1["VAE decode to RGB PNG"]
    S8 -->|yes| T2["output_type=latent<br/>latents / scaling_factor"]
    T2 --> T3["frozen VAE decode to pixels"]
    T3 --> T4["transparency decoder<br/>UNet1024, 12 ch in, 4 out: A R G B<br/>8 flip/rotate passes, take median"]
    T4 --> T5["RGBA PNG"]
    T1 --> U1["manifest.json<br/>hashes, settings, scheduler class,<br/>allocator peaks"]
    T5 --> U1
```

Order matters: the LayerDiffuse offset is merged **before** the style LoRA is loaded, so the two do not
fight over the same tensors. Example command:

```sh
sample.py --model <snapshot> --output <new dir> --prompt "..." --width 768 --height 768 \
  --steps 28 --guidance-scale 5 --cpu-offload --scheduler euler_ancestral \
  --lora <adapter.safetensors> --lora-sha256 <sha> \
  --layerdiffuse models/layerdiffuse
```

## 3. Post-processing — from render to term-chan sprite

```mermaid
flowchart LR
    P1["768 render"] --> P2["BOX resize to 64"]
    P2 --> P3["alpha threshold 128 if present"]
    P3 --> P4["palette quantise<br/>MAXCOVERAGE, 32 or up to 62 colours"]
    P4 --> P5["optional explicit colour key<br/>4-connected edge flood fill<br/>from a declared RGB + tolerance"]
    P5 --> P6["optional blink<br/>hash-bound annotation,<br/>only declared eye pixels change"]
    P6 --> P7["sprite_export<br/>shared sorted palette,<br/>rows of index chars,<br/>frames base / blink / talk"]
    P7 --> P8["term-chan loader + renderer<br/>read-only check, exact RGBA round trip"]
```

This segment contains no learning. It is where the current kawaii sprite gets its transparency, since
native alpha did not separate the subject for that prompt. A colour key is labelled background
recovery, not native transparency.

## 4. The evaluation loop

```mermaid
flowchart TD
    L1["plan: model, frozen prompt suites,<br/>baseline directories, deadline, max_stages"] --> L2["enqueue one hypothesis"]
    L2 --> L3["train one stage"]
    L3 --> L4["sample 2 suites x 4 cases<br/>+ matched no-LoRA baseline"]
    L4 --> L5["score_sprite: ink pixels,<br/>isolated share, largest stroke,<br/>top clearance, border fraction"]
    L5 --> L6{"score up and<br/>no regression?"}
    L6 -->|yes| L7["become incumbent"]
    L6 -->|no| L8["reject, keep incumbent"]
    L7 --> L9["matrix.py builds review pages"]
    L8 --> L9
    L9 --> L2
```

One GPU job at a time. A stage that finishes but was not yet recorded is picked up again after a driver
restart. `matrix.py` writes `page-NN.png` with eight columns: column 1 is always the matched no-LoRA
render, later columns are training stages, and the native-64 rendition sits under each 768 render.

## 5. Adapter inventory

| artifact | what it is | size | where it attaches | who made it |
|---|---|---|---|---|
| **Style LoRA** | rank-4 bypass on UNet attention (`to_q`, `to_k`, `to_v`, `to_out.0`) | ~23 MB | loaded by diffusers `load_lora_weights` at inference; `add_adapter` at training | us, e.g. experiment 27, stage 6, stage 22 |
| **LayerDiffuse attention offset** | dense delta for SDXL attention weights, not a LoRA | 1.9 GB | added into UNet weights on CPU before the style LoRA | third party, hash-pinned |
| **LayerDiffuse transparency decoder** | separate UNet1024 network, 12 channels in, 4 out | 208 MB | after the VAE decode, produces alpha | third party, hash-pinned |
| **SDXL VAE** | the base model's VAE | part of the snapshot | frozen in training for the decoded loss, and at inference for decoding | Animagine XL 4.0 |
| **Frozen base UNet + dual CLIP** | 2.5B parameters | snapshot | never trained; supplies the prior loss and all conditioning | Animagine XL 4.0 |

## Two things worth knowing

**Training and inference touch the UNet differently.** Training inserts trainable rank-4 matrices into
attention and keeps everything else frozen. Inference can additionally fuse the LayerDiffuse offset
straight into the base weights. That is a different mechanism from a LoRA even though both change the
same layers.

**The sprite pipeline is separate from the model.** Everything in segment 3 is deterministic CPU code:
resizing, palette, colour key, blink, export. Model quality and sprite legibility are therefore
separately testable, and a rejected adapter can still produce a usable sprite.
