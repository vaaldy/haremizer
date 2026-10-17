# Package installation and module commands

Python >=3.11 and Pillow >=9.1 are the base requirements. A normal non-editable
`python -m pip install .` builds and installs `haremizer`; it does not install ML
libraries or supply models. `src/haremizer/__init__.py` has no eager ML imports.
There are no flat-script compatibility wrappers or console-script aliases.

## Offline build and installation

Use an already provisioned build environment with setuptools >=77. Its bundled
wheel backend suffices; do not download a build backend merely to run these steps.
From the checkout:

```sh
python -c "import setuptools.build_meta as b; b.build_wheel('/absolute/path/to/dist'); b.build_sdist('/absolute/path/to/dist')"
python3 -m venv /absolute/path/to/cpu-env
/absolute/path/to/cpu-env/bin/python -m pip install --no-index \
  --find-links /path/to/local/wheels /absolute/path/to/dist/haremizer-0.1.0-py3-none-any.whl
```

Pillow must already be available as a compatible local wheel/cache entry. If using
uv, `uv venv --offline --python /path/to/python /path/to/env` and
`uv pip install --offline --python /path/to/env/bin/python /path/to/package.whl`
use its existing local cache. No network fallback is needed or implied.
The wheel includes `evaluation-suite.json`, original-code MIT and separate
LayerDiffuse Apache-2.0 license/NOTICE files, not runtime/model/data/run assets.
The distribution's combined license expression describes those two code parts;
it does not apply MIT to the Apache adaptation or grant rights to model/data/art.

From an unrelated working directory, use the installed interpreter:

```sh
/path/to/env/bin/python -I -c 'import haremizer; print(haremizer.__file__)'
/path/to/env/bin/python -I -m haremizer.prepare --help
/path/to/env/bin/python -I -m haremizer.sample --help
```

`prepare` defaults to the packaged evaluation suite; `--suite` selects an explicit
caller file. A missing suite is an error/blocker, never a source-checkout fallback.

## Command migration and workspace paths

Old `python sample.py ...` becomes `python -m haremizer.sample ...`. The same rule
applies to all former root modules with a CLI: `auto_loop`, `backend`, `blink`,
`blink_design`, `core`, `defects`, `edit_plan`, `hair_motion`, `images`, `matrix`,
`prepare`, `recovery`, `refinement_loop`, `runtime`, `smoke_sdxl`, `sprite_export`,
`sprite_png`, and `tidy`. All expose `--help` without ML libraries or models.
Library imports likewise change from `from recovery import recover` to
`from haremizer.recovery import recover`. The small scoring/LayerDiffuse modules
remain library APIs, not advertised CLI dispatchers.

Relative CLI file paths resolve against the caller's current directory, not the
installed package. Supply explicit model/data/run paths. Coordinators require
absolute input paths in plans, hash-bind installed source and inputs, run children
in owned stage directories, and refuse a child interpreter resolving a different
package installation. Install the same wheel in the selected interpreter's
environment; do not rely on source `PYTHONPATH`. `matrix` regeneration scripts
invoke the installed module with explicit run/output paths. The historical
reorganization tool additionally requires `python -m haremizer.tidy --workspace
/path/to/workspace`; dry run is still the default. Installing a package is not
permission to reorganize old evidence or run training.

## Optional ML and tests

Provision the heavy GPU-compatible libraries manually in a separate environment;
see the observed working inventory in [models](models.md). Install only this
wheel with `pip install --no-deps /path/to/package.whl` into that isolated
environment, never upgrade a working shared GPU runtime for the base install.
For local validation an isolated venv may explicitly expose an existing runtime's
site-packages using a venv-local `.pth` file; this is a read-only dependency
bootstrap, not a modification of that runtime or a source-checkout import path.

After installing a wheel, from the checkout:

```sh
/path/to/env/bin/python -I -B -m unittest discover -s test -v
```

The original 193 tests are retained. `test/test_package.py` adds installed-origin,
lazy-help, resource/missing-resource and child-launch boundary tests. Base-only
environments skip optional ML helpers under the original conditions. To opt into
read-only interoperability against an external checkout:

```sh
TERM_CHAN_ROOT=/path/to/term-chan TERM_CHAN_LOADER_TEST=1 \
  /path/to/env/bin/python -I -B -m unittest discover -s test -v
```

No optional skip is evidence of successful GPU inference or visual quality.
Historical documents retain old flat commands as archival evidence, not current
installation instructions.
