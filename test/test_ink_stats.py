from pathlib import Path
import tempfile
import unittest

from PIL import Image

from haremizer.ink_stats import ink_stats


class InkStatsTests(unittest.TestCase):
    def test_connected_stroke_and_isolated_dots(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sprite.png'
            image = Image.new('RGBA', (8, 8), (200, 120, 90, 255))
            for x in range(5):
                image.putpixel((x, 1), (0, 0, 0, 255))
            image.putpixel((7, 7), (10, 10, 10, 255))
            image.save(path)
            result = ink_stats(path, size=8)
            self.assertEqual(result['ink_pixels'], 6)
            self.assertEqual(result['components'], 2)
            self.assertEqual(result['largest_component'], 5)
            self.assertEqual(result['isolated_pixels'], 1)
            self.assertEqual(result['fragments_under_4'], 1)
            image.putpixel((7, 7), (200, 120, 90, 255))
            image.save(path)
            self.assertEqual(ink_stats(path, size=8)['components'], 1)
            with self.assertRaises(ValueError):
                ink_stats(path, ink_max_rgb=999)


if __name__ == '__main__':
    unittest.main()
