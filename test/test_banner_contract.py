import unittest
from io import BytesIO
from pathlib import Path
import tempfile

from PIL import Image

from haremizer.images import BANNER_SIZES, validate_png


class BannerContractTests(unittest.TestCase):
    def make(self, path, size, colors=((20, 30, 40), (200, 180, 160))):
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        for x in range(size[0]):
            image.putpixel((x, 0), (*colors[0], 255))
        for x in range(size[0]):
            image.putpixel((x, size[1] - 1), (*colors[1], 255))
        image.save(path)

    def test_banner_sizes_need_the_explicit_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            for size in BANNER_SIZES:
                path = Path(folder) / f'{size[0]}x{size[1]}.png'
                self.make(path, size)
                with self.assertRaisesRegex(ValueError, 'banners need allow_banners'):
                    validate_png(path)
                metrics = validate_png(path, allow_banners=True)
                self.assertEqual((metrics['width'], metrics['height']), size)
                self.assertEqual(metrics['opaque_colors'], 2)

    def test_full_bleed_banner_may_be_opaque(self):
        with tempfile.TemporaryDirectory() as folder:
            banner = Path(folder) / 'opaque.png'
            Image.new('RGBA', (128, 48), (30, 40, 60, 255)).save(banner)
            metrics = validate_png(banner, allow_banners=True)
            self.assertEqual(metrics['transparent_pixels'], 0)
            square = Path(folder) / 'opaque-square.png'
            Image.new('RGBA', (64, 64), (30, 40, 60, 255)).save(square)
            with self.assertRaisesRegex(ValueError, 'both transparent and opaque'):
                validate_png(square, allow_banners=True)

    def test_undeclared_aspect_ratio_still_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'odd.png'
            self.make(path, (100, 48))
            with self.assertRaisesRegex(ValueError, 'banner size must be one of'):
                validate_png(path, allow_banners=True)

    def test_square_rules_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'square.png'
            self.make(path, (64, 64))
            self.assertEqual(validate_png(path)['width'], 64)
            other = Path(folder) / 'wrong.png'
            self.make(other, (32, 32))
            with self.assertRaisesRegex(ValueError, 'expected square PNG'):
                validate_png(other, allow_banners=True)


if __name__ == '__main__':
    unittest.main()
