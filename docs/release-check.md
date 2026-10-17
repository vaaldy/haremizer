# Installed workflow check — 2026-10-17

**Operational PASS / image REJECT / publication not approved**.
The producing source was commit
`e559c1677265b8ab49d2879f31f86dd6a8bb20e8`, package version `0.1.0`.
This note changes no producing Python code.

- Built wheel and sdist offline; checked archive exclusions and installed Python/
  JSON bytes against that source. Installed into a fresh Pillow-only environment
  and a separate wheel-installed environment inheriting existing ML dependencies
  read-only. No editable install or checkout source-path bootstrap.
- From outside the checkout, each environment passed all 206 discovered CPU tests
  (11 original optional skips in Pillow-only, 9 in inherited-ML). Read-only external
  term-chan integration was enabled. Lazy CLI/import/resource checks, including
  missing-resource refusal, and exact frozen synthetic recovery/export passed.
- One real installed inference used the frozen Animagine XL4 revision
  `2b7c1b397761bf5bd3cc42e5b39ec99314a75a96`, hash-bound exp27 adapter,
  seed33013, 768 square, 28 Euler ancestral steps, CFG5 and CPU offload.
  The owned child exited0 under the 600s/11GiB allocator guards; its group was absent
  afterward. Bound inputs, weights and external renderer source stayed unchanged.
- Raw768 PNG → explicit RGB252,215,220/tolerance24 color-key recovery → 64RGBA
  produced 2611 opaque pixels, 1485 total transparent pixels (1484 four-connected
  exterior plus 1 enclosed), and 32 opaque colors. Alpha was binary.
  `neutral.json` and the real term-chan loader
  decoded exactly to those RGBA pixels; pure renderers worked without a TTY.
- New and historical raw PNG bytes matched in this single comparison. This is
  **not** a general GPU reproducibility guarantee or a second sampling experiment.

Replay the installed command from [the workflow guide](workflow.md), using an
explicit hash-verified recipe/config and a new output directory:

```sh
/path/to/installed-env/bin/python -I -B -m haremizer.workflow \
  --config /absolute/workspace/neutral-workflow.json \
  --output /absolute/workspace/new-neutral-run
```

The local validation receipt retains exact argv, dependency inventory, wheel
hashes, process diagnostics, raw/recovered/exported artifacts and previews.
Independent image review **REJECTED** the example for clipped upper head/headdress
framing. Operational format success does not override that visual failure. The
historical raw is byte-identical: this is a model/recipe limitation, not a
packaging regression. No model promotion or visual acceptance is claimed.
No model improvement, training, publishing or additional artistic attempts were
performed. Original code is MIT; third-party adaptation, model, data and generated
art terms remain separate ([NOTICE](../NOTICE.md)). Publication remains blocked
on independent review and applicable rights/provenance/notice/secret checks;
this operational check does not grant those rights.
