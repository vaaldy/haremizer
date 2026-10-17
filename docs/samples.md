# Historical sample provenance and release limits

The tracked folders under `artifacts/sprites/` remain at their original paths.
They are historical examples, not a promise of model quality, human acceptance or
reliable automatic expression generation. [sample-provenance.json](sample-provenance.json)
records every tracked sample file's SHA256/length and minimal evidence read from
existing local sidecars. Private paths and full run dumps are deliberately omitted.
Sidecar hashes identify the evidence used; those private sidecars are not shipped.

| Sample | What the available evidence establishes |
|---|---|
| [kawaii-maid](../artifacts/sprites/kawaii-maid/neutral.json) | Export of supplied recovered base and separately edited blink; README example |
| [kawaii-maid-early](../artifacts/sprites/kawaii-maid-early/neutral.json) | Earlier blink variant of the same supplied recovered base |
| [mint-maid-legible](../artifacts/sprites/mint-maid-legible/neutral.json) | Export of supplied legibility-edited base/blink; only 56 transparent pixels, not an empty exterior backdrop |
| [mint-maid-white](../artifacts/sprites/mint-maid-white/neutral.json) | Export of supplied white-maid base and blink; likewise only 56 transparent pixels |
| [mint-maid-white-deeper-blink](../artifacts/sprites/mint-maid-white-deeper-blink/neutral.json) | Same supplied base, deeper blink variant |
| [long-wavy-export-test](../artifacts/sprites/long-wavy-export-test/neutral.json) | Export test of supplied recovered and blink PNGs |
| [rado-chan](../artifacts/sprites/rado-chan/neutral.json) | Export of supplied rebuilt base/blink; export alone does not establish the reconstruction lineage |
| [best400-curly-prototype](../artifacts/sprites/best400-curly-prototype/neutral.json) | One unpromoted learned-weight example, with recorded model/adapter identity, white-key CPU recovery and per-image annotated eye edits |

The export-only sidecars bind input/output bytes and frame properties, not the
full upstream model, source artwork, artist license or transformation chain.
Unknown attribution is left unknown rather than inferred from a filename.
The best400 sidecar additionally records Animagine XL 4.0 and adapter identity,
and training-reference credits for peb / pebonius (CC-BY-4.0,
<https://opengameart.org/content/64px-portraits>) and Efilheim (CC0-1.0,
<https://opengameart.org/content/everface>). Those credits are retained as local
lineage evidence, not a claim that either artist authored or approved the output.

The kawaii README image is a downsampled, explicitly color-keyed historical render,
not native learned alpha. Blink tools export/edit declared regions; they do not
certify closed-eye semantics. Opaque colors and binary alpha can round-trip through
JSON; hidden transparent RGB is not serialized. A valid file is not a visual verdict.

## Outstanding publication issues

Original code's MIT grant does not clear generated artwork, training sources,
base models or adapters. Full source/processing lineage and generated-output
redistribution rights are not established for every example, including Mint artwork
embedded in the concept illustration. Reference manifests and model cards are
local evidence, not comprehensive upstream authenticity or legal review. No new
artist/license/approval claims are made. Resolve these gaps before asserting a
fully rights-cleared public release; see [NOTICE](../NOTICE.md).
