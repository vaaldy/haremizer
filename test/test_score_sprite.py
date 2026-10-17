from pathlib import Path
import tempfile
import unittest

from PIL import Image


from haremizer.score_sprite import aggregate, compare, measure


def write(path, rows, size=64):
    image = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    for y, xs in rows:
        for x in xs:
            image.putpixel((x, y), (0, 0, 0, 255))
    image.save(path)
    return path


class ScoreSpriteTests(unittest.TestCase):
    def test_measure_reports_ink_clearance_and_border(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write(Path(folder) / 'a.png', [(20, list(range(10, 20))), (21, [30])])
            result = measure(path)
            self.assertEqual(result['ink_pixels'], 11)
            self.assertEqual(result['components'], 2)
            self.assertEqual(result['isolated_share'], 0.5)
            self.assertEqual(result['ink_pixels'], 11)
            self.assertEqual(result['top_clearance_rows'], 20)
            self.assertEqual(result['border_rgb'], [255, 255, 255])
            self.assertGreater(result['border_fraction'], 0.9)


    def test_compare_blocks_regression_and_accepts_gain(self):
        base = {'ink_pixels': 120, 'components': 24, 'isolated_share': 0.51,
                'largest_component': 41, 'top_clearance_rows': 0, 'border_fraction': 0.5,
                'subject_fraction': 0.5, 'opaque_colors': 20}
        improved = {**base, 'isolated_share': 0.30, 'largest_component': 70, 'top_clearance_rows': 3}
        worse = {**base, 'isolated_share': 0.62, 'largest_component': 30}
        self.assertTrue(compare(improved, base)['accepted'])
        blocked = compare(worse, base)
        self.assertFalse(blocked['accepted'])
        self.assertIn('isolated_share', blocked['regressions'])
        self.assertIn('largest_component', blocked['regressions'])
        self.assertEqual(aggregate([base, improved])['cases'], 2)


if __name__ == '__main__':
    unittest.main()
