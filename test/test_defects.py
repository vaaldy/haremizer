import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from haremizer.defects import CLASSES, analyse, detect, residue_colour


def sprite(rows, size=16, colours=None):
    """rows: list of (y, [(x, colour)]) painted over a transparent canvas."""
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    for y, points in rows:
        for x, colour in points:
            image.putpixel((x, y), (*colour, 255))
    return image


class SpriteTests(unittest.TestCase):
    def test_thin_line_specks_and_contact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sprite.png'
            image = sprite([(8, [(x, (10, 20, 30)) for x in range(2, 10)]),   # silhouette blob
                            (8, [(x, (10, 20, 30)) for x in range(2, 10)]),
                            (11, [(14, (10, 20, 30))]),                        # speck
                            (3, [(15, (200, 10, 10)) for _ in range(1)])])     # detached mark
            for y in (7, 8, 9):
                for x in range(2, 10):
                    image.putpixel((x, y), (10, 20, 30, 255))
            for y in range(1, 11):                                            # 1px wide, 10px tall line
                image.putpixel((15, y), (200, 10, 10, 255))
            image.putpixel((0, 0), (200, 10, 10, 255))                        # speck at the corner
            image.save(path)
            before = path.read_bytes()
            report, masks = detect(path)
            self.assertEqual(str(path.resolve()), report['input'])
            self.assertEqual(report['mode'], 'sprite')
            self.assertTrue(report['defects']['stray_thin_lines']['present'])
            self.assertTrue(report['defects']['specks']['present'])
            self.assertEqual(path.read_bytes(), before, 'detector must not modify the input')
            self.assertTrue(all(name in masks for name in CLASSES))

    def test_residue_and_border_contact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sprite.png'
            image = sprite([(0, [(x, (30, 30, 30)) for x in range(4, 12)])])
            for y in range(0, 6):
                for x in range(4, 12):
                    image.putpixel((x, y), (30, 30, 30, 255))
            for y in range(6, 14):                                            # 3px wide pale band, detached
                for x in range(13, 16):
                    image.putpixel((x, y), (245, 201, 197, 255))
            image.save(path)
            report, masks = detect(path)
            self.assertTrue(report['defects']['border_contact']['present'])
            self.assertTrue(report['defects']['backdrop_residue']['present'])
            self.assertIn((14, 6), {(index % 16, index // 16) for index in masks['backdrop_residue']})

    def test_format_problems_on_a_non_64_sprite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sprite.png'
            image = sprite([], size=32)
            image.putpixel((0, 0), (1, 2, 3, 255))
            image.save(path)
            report = analyse(path)
            self.assertEqual(report['mode'], 'sprite')
            self.assertTrue(report['defects']['format']['present'])
            self.assertTrue(any('not 64' in problem for problem in report['format_problems']))


class RenderTests(unittest.TestCase):
    def test_drawn_frame_and_panel_are_busy_background(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, painter, expected in (('frame', 'frame', True), ('panel', 'panel', True),
                                            ('clean', 'clean', False)):
                path = Path(folder) / f'{name}.png'
                size = 256
                image = Image.new('RGBA', (size, size), (240, 236, 232, 255))
                if painter == 'frame':
                    for x in range(size):
                        for offset in range(8):
                            image.putpixel((x, 6 + offset), (5, 5, 5, 255))
                            image.putpixel((x, size - 14 + offset), (5, 5, 5, 255))
                elif painter == 'panel':
                    for y in range(size):
                        for x in range(40, 48):
                            image.putpixel((x, y), (5, 5, 5, 255))
                image.save(path)
                report = analyse(path)
                self.assertEqual(report['mode'], 'render')
                self.assertEqual(report['defects']['busy_background']['present'], expected, name)

    def test_clean_render_reports_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'render.png'
            image = Image.new('RGBA', (256, 256), (250, 250, 250, 255))
            for y in range(40, 200):
                for x in range(60, 180):
                    image.putpixel((x, y), (40, 60, 90, 255))
            image.save(path)
            report = analyse(path)
            self.assertEqual(report['mode'], 'render')
            self.assertEqual(report['defects']['busy_background']['present'], False)
            self.assertEqual(report['defects']['border_contact']['present'], False)

    def test_residue_colour_falls_back_to_black(self):
        self.assertEqual(residue_colour([(0, 0, 0, 0)], set()), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
