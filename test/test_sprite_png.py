import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from haremizer.sprite_png import load, render, render_directory


class SpritePngTests(unittest.TestCase):
    def test_render_matches_source_pixels_and_rejects_bad_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            sprite = {'name': 't', 'size': [2, 2], 'palette': ['#FF0000', '#00FF00'],
                      'frames': {'base': ['0.', '.1'], 'blink': ['..', '..']}}
            (directory / 'neutral.json').write_text(json.dumps(sprite))
            path, loaded = load(directory)
            self.assertEqual(path.name, 'neutral.json')
            image = render(loaded, 'base')
            self.assertEqual(list(image.getdata())[0], (255, 0, 0, 255))
            self.assertEqual(list(image.getdata())[1], (0, 0, 0, 0))
            self.assertEqual(list(image.getdata())[3], (0, 255, 0, 255))
            self.assertEqual(render(loaded, 'blink').getpixel((0, 0)), (0, 0, 0, 0))
            with self.assertRaisesRegex(ValueError, 'frame'):
                render(loaded, 'talk')
            written = render_directory(directory)
            self.assertEqual([p.name for p in written], ['neutral-base.png', 'neutral-blink.png'])
            self.assertEqual(Image.open(written[0]).size, (2, 2))
            (directory / 'extra.json').write_text(json.dumps(sprite))
            with self.assertRaisesRegex(ValueError, 'exactly one sprite JSON'):
                load(directory)

    def test_bad_dimensions_rejected(self):
        sprite = {'size': [4, 4], 'palette': ['#000000'], 'frames': {'base': ['0', '0']}}
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            render(sprite, 'base')


if __name__ == '__main__':
    unittest.main()
