import hashlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from haremizer.sprite_check import check_legibility


class LegibilityTests(unittest.TestCase):
    def test_fragmented_marks_require_redo_and_consistent_ink_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sprite.png'
            image = Image.new('RGBA', (64, 64), (255, 240, 220, 255))
            image.putpixel((0, 0), (0, 0, 0, 0))
            mouth = [[20, 20], [21, 21], [22, 21]]
            stroke = [[30, 30], [30, 31], [31, 32]]
            for p, c in zip(mouth, [(120, 20, 30), (200, 50, 70), (230, 80, 100)]):
                image.putpixel(tuple(p), (*c, 255))
            image.putpixel((30, 30), (100, 50, 40, 255))
            image.save(path)
            ann = {'version': 1, 'reviewed_by': 'machine:test explicit fixture',
                   'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'mouth_pixels': mouth, 'mouth_colors': ['#78141e'],
                   'uniform_strokes': [stroke], 'uniform_color': '#643228'}
            before = path.read_bytes()
            result = check_legibility(path, ann)
            self.assertEqual(result['mouth']['swatches'], 3)
            self.assertTrue(result['redo_required'])
            self.assertEqual(path.read_bytes(), before)
            for p in mouth:
                image.putpixel(tuple(p), (120, 20, 30, 255))
            for p in stroke:
                image.putpixel(tuple(p), (100, 50, 40, 255))
            image.save(path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                check_legibility(path, ann)
            ann['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(check_legibility(path, ann)['passed'])
            ann['uniform_strokes'] = [[[30, 30], [32, 32]]]
            with self.assertRaisesRegex(ValueError, 'eight-connected'):
                check_legibility(path, ann)


if __name__ == '__main__':
    unittest.main()
