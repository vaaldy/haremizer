import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import layerdiffuse


class AssetTests(unittest.TestCase):
    def test_exact_asset_validation_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.safetensors'
            path.write_bytes(b'fixture')
            fixture = {'decoder': (path.name, 7, hashlib.sha256(b'fixture').hexdigest())}
            with patch.dict(layerdiffuse.WEIGHTS, fixture, clear=True):
                self.assertEqual(layerdiffuse.validate(folder)['files']['decoder']['bytes'], 7)
                path.write_bytes(b'tamper!')
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    layerdiffuse.validate(folder)
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'file/size'):
                    layerdiffuse.validate(folder)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),
                     'optional ML runtime unavailable')
class TensorTests(unittest.TestCase):
    def test_offset_merge_and_pre_mutation_rejection(self):
        import torch
        from safetensors.torch import save_file
        model = torch.nn.Linear(2, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'offset.safetensors'
            save_file({'weight': torch.ones(2, 2)}, str(path))
            self.assertEqual(layerdiffuse.merge_attention(model, path), 1)
            self.assertTrue(torch.equal(model.weight, before['weight'] + 1))
            self.assertTrue(torch.equal(model.bias, before['bias']))
            current = model.weight.clone()
            for invalid in ({'weight': torch.ones(1)}, {'unknown': torch.ones(2, 2)},
                            {'weight': torch.full((2, 2), float('nan'))}):
                save_file(invalid, str(path))
                with self.assertRaises(ValueError):
                    layerdiffuse.merge_attention(model, path)
                self.assertTrue(torch.equal(model.weight, current))

    def test_augmentation_inverse_and_rgba_order(self):
        import torch
        pixel = torch.rand(1, 3, 64, 64)
        latent = torch.rand(1, 4, 8, 8)
        original_pixel, original_latent = pixel.clone(), latent.clone()
        def fake(p, z):
            alpha = z[:, :1].repeat_interleave(8, 2).repeat_interleave(8, 3)
            return torch.cat((alpha, p), dim=1)
        result = layerdiffuse.augmented_decode(fake, pixel, latent)
        self.assertTrue(torch.equal(result, fake(pixel, latent)))
        self.assertTrue(torch.equal(pixel, original_pixel))
        self.assertTrue(torch.equal(latent, original_latent))
        image = layerdiffuse.rgba_image(torch.tensor([[[[0.5]], [[1.0]], [[0.0]], [[0.0]]]]))
        self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0, 127))
        with self.assertRaises(ValueError):
            layerdiffuse.rgba_image(torch.full((1, 4, 1, 1), float('nan')))


if __name__ == '__main__':
    unittest.main()
