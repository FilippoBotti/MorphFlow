"""CPU contract tests for source formats and interruption-safe caches."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np
import morph_assets as pipeline


class InputAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pack = {'feats': torch.arange(16).reshape(2, 8).float(),
                     'coords': torch.tensor([[1, 2, 3], [4, 5, 6]])}

    def tearDown(self):
        self.temp.cleanup()

    def test_same_asset_across_supported_slat_formats(self):
        torch.save(self.pack, self.root / 'latent.pt')
        np.savez(self.root / 'latent.npz', **{k: v.numpy() for k, v in self.pack.items()})
        directory = self.root / 'asset'
        directory.mkdir()
        torch.save(self.pack['feats'], directory / 'slat_feats.pt')
        torch.save(self.pack['coords'], directory / 'slat_coords.pt')
        for path in (self.root / 'latent.pt', self.root / 'latent.npz', directory):
            with self.subTest(path=path):
                result = pipeline.read_slat(path)
                self.assertTrue(torch.equal(result['feats'], self.pack['feats']))
                self.assertTrue(torch.equal(result['coords'][:, 1:], self.pack['coords']))
                self.assertTrue((result['coords'][:, 0] == 0).all())

    def test_invalid_or_normalized_sources_are_rejected(self):
        bad = [
            {**self.pack, 'metadata': {'latent_scale': 'normalized'}},
            {**self.pack, 'coords': torch.tensor([[1., 2., 3.5], [4., 5., 6.]])},
            {**self.pack, 'coords': torch.tensor([[1, 2, 3], [1, 2, 3]])},
            {**self.pack, 'coords': torch.tensor([[1, 2, 64], [4, 5, 6]])},
            {**self.pack, 'feats': torch.full((2, 8), float('nan'))},
            {**self.pack, 'coords': torch.tensor([[1, 1, 2, 3], [0, 4, 5, 6]])},
        ]
        for pack in bad:
            with self.subTest(pack=pack), self.assertRaises(ValueError):
                pipeline.validate_pack(pack)

    def test_nearby_alphas_do_not_reuse_the_endpoint(self):
        values = pipeline.parse_alphas(['0,0.5', '0.999999999', '1'])
        self.assertEqual(len({pipeline.alpha_tag(a) for a in values}), 4)
        for values in (['nan'], ['1.1'], ['0', '0.0']):
            with self.assertRaises(ValueError):
                pipeline.parse_alphas(values)

    def test_cache_requires_complete_marker_and_intact_content(self):
        path = self.root / 'slat.pt'
        pipeline.save_tensor(path, self.pack)
        self.assertFalse(pipeline.ready(self.root, 'slat'))
        pipeline.commit(self.root, 'slat', ['slat.pt'])
        self.assertTrue(pipeline.ready(self.root, 'slat'))
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        self.assertFalse(pipeline.ready(self.root, 'slat'))

    def test_upstream_invalidation_keeps_reusable_tensors(self):
        pipeline.save_tensor(self.root / 'ss.pt', {'latent': torch.ones(1)})
        pipeline.commit(self.root, 'ss', ['ss.pt'])
        (self.root / 'slat.json').write_text(json.dumps({'status': 'complete'}))
        pipeline.invalidate(self.root, ('slat', 'mesh', 'gaussian', 'export'))
        self.assertTrue(pipeline.ready(self.root, 'ss'))
        self.assertFalse((self.root / 'slat.json').exists())

    def test_explicit_blender_overrides_environment_and_fails_if_invalid(self):
        from unittest.mock import patch
        executable = self.root / 'blender'
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
        with patch.dict('os.environ', {'BLENDER_BIN': '/invalid/environment/blender'}):
            self.assertEqual(pipeline.resolve_blender(executable), executable)
            with self.assertRaises(FileNotFoundError):
                pipeline.resolve_blender(self.root / 'absent')


if __name__ == '__main__':
    unittest.main()
