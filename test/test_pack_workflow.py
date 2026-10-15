"""Regression for the default single-character pack workflow.

The seam this guards: an export must publish the default expression as `neutral.json`, the term-chan
loader must expose it as `expressions.neutral`, and a companion built from that pack must actually draw
the sprite — including when the requested mood is absent and the mood falls back to neutral.

Optional and read-only: needs Node and the term-chan sources, never writes to the parent repo, no GPU and
no network. Enable with TERM_CHAN_LOADER_TEST=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from sprite_export import export_sprite  # noqa: E402

PARENT = Path(__file__).resolve().parent.parent.parent
NODE_SCRIPT = """
const { loadCharacter } = await import(process.argv[1]);
const { createCompanion } = await import(process.argv[2]);
const character = loadCharacter(process.argv[3]);
const companion = createCompanion({ character, random: () => 0.5, startAt: 0 });
companion.react({ mood: 'happy', quip: 'Tests passed. Obviously.', status: 'test' }, 0);
const frameText = companion.render(0, { cols: 120, rows: 30 });
const blocks = (frameText.match(/[\\u2580\\u2584]/g) ?? []).length;
const lines = frameText.split('\\n').length;
console.log(JSON.stringify({
  expressions: Object.keys(character.expressions),
  hasNeutral: Boolean(character.expressions.neutral),
  name: character.name,
  width: character.expressions.neutral ? character.expressions.neutral.width : null,
  height: character.expressions.neutral ? character.expressions.neutral.height : null,
  lines: lines,
  blocks,
  mood: companion.state.mood,
}));
"""


def write_png(path, size, colors):
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    for y, colour in enumerate(colors):
        for x in range(size):
            image.putpixel((x, y), (*colour, 255))
    image.save(path)


class DefaultPackWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # the pack must live outside the input image directory, as the exporter requires
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.base = self.inputs / 'base.png'
        self.blink = self.inputs / 'blink.png'
        write_png(self.base, 64, [(20, 30, 40), (200, 180, 160)])
        write_png(self.blink, 64, [(20, 30, 40), (10, 10, 10)])
        self.output = self.root / 'pack'

    def test_export_publishes_neutral_json(self):
        """The exported pack must carry the default expression filename, not a custom name."""
        export_sprite(self.base, self.output, self.blink)
        self.assertTrue((self.output / 'neutral.json').is_file(),
                        'the default export must publish neutral.json')

    @unittest.skipUnless(os.environ.get('TERM_CHAN_LOADER_TEST') == '1' and shutil.which('node'),
                         'optional read-only term-chan integration')
    def test_loader_and_companion_render_the_default_pack(self):
        loader = PARENT / 'src' / 'sprites' / 'loader.js'
        app = PARENT / 'src' / 'app.js'
        if not loader.is_file() or not app.is_file():
            self.skipTest('parent term-chan sources unavailable')
        export_sprite(self.base, self.output, self.blink)
        before = {path: path.stat().st_mtime for path in (loader, app)}
        result = subprocess.run(['node', '--input-type=module', '-e', NODE_SCRIPT,
                                 loader.as_uri(), app.as_uri(), str(self.output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertIn('neutral', parsed['expressions'])
        self.assertTrue(parsed['hasNeutral'])
        self.assertEqual((parsed['width'], parsed['height']), (64, 64))
        self.assertEqual(parsed['lines'], 30)
        # The requested mood 'happy' is absent, so the companion must fall back to neutral and still
        # draw the sprite rather than render an empty pane.
        self.assertEqual(parsed['mood'], 'happy')
        self.assertGreater(parsed['blocks'], 0, 'the companion rendered no sprite cells')
        self.assertEqual({path.stat().st_mtime for path in (loader, app)}, set(before.values()),
                         'the parent sources must not be modified')


if __name__ == '__main__':
    unittest.main()
