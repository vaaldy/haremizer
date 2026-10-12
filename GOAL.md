# Haremizer: autonomous improvement contract

## End goal

Given a text description, generate a genuinely distinct, readable **64×64 anime-style portrait** suitable for term-chan, with bold facial features, controlled proportions, a complete head, an actually empty transparent background, and a localized blink. The model must learn the visual design; manually repairing each generated face is not the finished solution.

This contract records the conversation's requirements. Numeric ranges below are the initial design specification, not claims that the current model meets them. Change them only with explicit recorded aesthetic evidence/user feedback; never quietly loosen a gate to promote a checkpoint.

## Core portrait gate — first priority

Assess the actual final 64px rendition, not just an attractive 768px preview.

- One coherent head/bust; no tiled or duplicate faces, distorted anatomy, cropped scalp/hair, or accidental extra eyes/mouths. Hair/head has at least two native pixels of top/side clearance; shoulders may reach the bottom.
- Initial head-height target: **36–54 pixels**, including ordinary hair but excluding deliberately requested accessories. Face-height target: **20–34 pixels**. Foreshortening/accessories require an explicit review note, not automatic failure or silent exceptions.
- Initial near-eye width target: **3–7 pixels**, with plausible iris/pupil proportions; the farther eye may be smaller. No oversized/chibi-eye regression. Eyelids/pupils are clearly dark while requested iris colors and highlights remain identifiable.
- A readable **1–3-pixel nose mark** and **2–8-pixel mouth**, allowing expression-appropriate shapes. Facial ink is confidently near-black (roughly RGB channels ≤32), with visible contrast against local skin; do not globally crush shadows or erase colors.
- Requested hair length/style/color, skin tone, eye color, expression, accessories and clothing remain distinguishable. Do not get bolder lines by sacrificing descriptor adherence or turning all faces into the same character.
- Clean pixel clusters and coherent stroke thickness; no dependence on manual output reinking, a fixed face skeleton, recolor/template banks, nadeshiko/yuki geometry, or per-case prompt rescue.
- Blank composition: no unrequested scenery, frames, checkerboards, text, reaction symbols or loose decoration.

These are visual/annotated measurements, not permission to fabricate automatic landmark detection. Evaluation may use explicitly labeled machine per-image annotations; source/output annotation is not human review.

## Full deliverable gate — transparency and animation

Transparency remains mandatory, but native-transparency/LayerDiffuse integration is deferred until core portraits pass.

- Final PNGs are **64×64 RGBA**, with genuine background alpha zero—not a white canvas or painted checkerboard. Hair gaps are clear without deleting eye whites, skin highlights or clothing. For term-chan, alpha is binary and there are ≤62 opaque colors.
- Our ordinary SDXL style LoRA/RGB decoder cannot add alpha by learning a prompt. Preserve source alpha/provenance; distinguish a validated alpha-generation decoder from explicitly labeled background recovery. Do not claim native transparency until its architecture, licensing, loading, ROCm compatibility and actual outputs are verified.
- Base/blink use the same silhouette/alpha/palette. Blink changes only declared eye regions; no hair/mouth/background movement. Automatic eye localization or an explicitly user-selected annotation workflow is needed for an automatic end product; one manually annotated example is only a prototype.
- term-chan JSON export passes its actual loader and renderer, and both frames round-trip to exactly the PNG RGBA pixels. No parent term-chan mutations.

## Acceptance evidence

1. Freeze baseline/checkpoint hashes, source/caption/group splits, runtime/module hashes, sampler, prompts, seeds, output processing and this contract for each experiment.
2. Inspect **all eight existing primary/challenge cases** at large and native-64 sizes. Use identical conditioning/processing for baseline and candidate; no cherry-picking. Distinguish phrase-induced improvements in the unloaded baseline from learned LoRA gains.
3. A promising checkpoint must pass two different frozen seed sets, plus a separately frozen novel-descriptor set before final robustness acceptance. Validation descriptors/images never enter training. Source identity families/expressions stay in one partition.
4. Record per-case proportions, visibility/contrast, adherence, artifacts and uncertainty. Compare against the unloaded baseline and the strongest retained unpromoted checkpoint. A material regression blocks promotion.
5. Training completion, finite gradients, exact reload, a lower loss, CPU tests, one nice portrait or a valid alpha channel do **not** establish model quality. Preserve failed evidence and rollback weights.

## Improvement loop

One writer and one GPU job at a time. Parent owns decisions, source fixes, visual gates and final acceptance.

**hypothesis → smallest change → CPU/preflight checks → one bounded training/evaluation stage → actual image inspection → accept/reject → cohesive local commit → choose next evidence-based hypothesis**

- First hypothesis: remove the confirmed fractional pixel grid. The last 64px references were scaled to 672 inside 768; the CPU probe retained only 47/138 projected dark-stroke cells, versus 153/153 for aligned 64→768→64, including VAE reconstruction. Start with content resolution 768; keep seed/LR/steps/losses/data/conditioning fixed. The references already have native-grid margins. This does not establish that geometry or all generation failures are solved.
- If needed next: audit/expand proportion-consistent original references; investigate source-supervised decoded-64px edge/contrast/geometry losses with a measured memory gate. Min-SNR is a separate optimization ablation, not a guaranteed contrast fix. Do not combine speculative changes just to get a result.
- Failed gates stop that branch, **not necessarily the whole mission**. Roll back and change the hypothesis/data/objective. Do not repeat identical failed fits, increase steps/rank blindly, tune individual evaluation prompts, or silently recycle/reset budgets.
- Commit the contract, tested implementation changes and truthful checkpoint summaries with lowercase `[feat]/[fix]/[test]/[chore]`. Keep model/data/run binaries out of Git; commit their identities/evidence summaries, not huge weights. No pushes. A rejected experiment may merit a documentation checkpoint, not an unrelated code change.

## Authority and bounds

User explicitly renewed autonomous iterative improvement and checkpoint commits. The serial-executor subagent was retired at the user's request; the active loop now runs directly in the parent, with no subagent execution handoff barrier. Do not launch another executor unless the user requests it. Any later explicitly authorized delegate remains Sol-only (`openai-codex/gpt-6.1-sol:medium`); no paid services or provider substitution.

Work only in `/home/valdy/Projects/term-chan/haremizer`. Never mutate parent term-chan or the independently tracked `prompts/` repository. Use verified CC0/CC-BY original artwork with attribution and disclosed derivative edits. No generated-image training feedback, modular recolor datasets, invented human/legal/canonical-age/gender/source-method approval, or unsupported identity counts.

- Hard storage ceiling **100,000,000,000 bytes**, conservative separate-root accounting with **2 GB reserve + 200 MB image allowance**. No unrelated shared-cache deletion.
- Process-local `HSA_OVERRIDE_GFX_VERSION=10.3.0` only; preserve tmux, no drivers/global compatibility changes. **1024px GPU route remains disabled.** No CPU training fallback or unrelated-process kills.
- Training ≤900 seconds and each sampler ≤600 seconds; owned-process-group cleanup and original immutable evidence required. Allocator peaks are not total GPU memory; respect the existing 11 GiB allocator check and retain uncertainty.
- This newly authorized autonomous window is **12 hours maximum**. Its initial six-experiment batch has completed. Under the user's renewed parent-direct continuous-loop instruction, the parent explicitly predeclared one further three-experiment batch: the first revised cap was nine experiments. After that batch's recorded gates, a further three-check architectural batch was explicitly predeclared: that architectural batch has completed. A further three-check conditioning-alignment batch was explicitly recorded: the active cap is now **fifteen new GPU experiments including evaluation-only diagnostics/validation**, with the original UTC deadline unchanged. This parent-selected safety bound is not claimed as a user-specified number. Every revision must be recorded and communicated before compute; no quiet resets. Historical total is 12 at initialization; never erase it. Work in serial batches of at most three experiments, rechecking safety/resources/quality before each. Do not silently roll the window or cap over.
- Stop/pause for user interruption, accepted full goal, exhausted declared budget/deadline, unsafe GPU/resource behavior, infrastructure failure, or a genuine data/permission/strategy blocker. Capture partial results/diff, report the exact blocker and preserve evidence. Do not switch execution protocols silently.

“Until oblivion” means persistent, surgical improvement while these bounds permit it—not infinite unmonitored execution or pretending success. If only an authorized async job is active, return control and resume on its native completion notification.

## Current state

No model is promoted/robust. Original-corpus best400 remains a useful unpromoted rollback/prototype; the facial-weighted150 checkpoint failed consistent bold-feature/proportion gates. The pixel-grid/VAE research is a verified preprocessing lead, not a completed model improvement. Existing trained-prototype/blink format checks are not full-generator acceptance.
