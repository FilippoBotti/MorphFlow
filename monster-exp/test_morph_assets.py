"""CPU contract tests for source formats and interruption-safe caches."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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

    def test_up_axis_is_explicit_and_can_differ_between_sources(self):
        trellis = self.root / 'TRELLIS'
        for relative in ['trellis/models/__init__.py', 'dataset_toolkits/blender_script/render.py']:
            path = trellis / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in ['a.glb', 'b.glb']:
            (self.root / name).write_bytes(b'glTF')
        for name in ['ss.pt', 'slat.pt']:
            (self.root / name).touch()
        blender = self.root / 'blender'
        blender.write_text('#!/bin/sh\nexit 0\n')
        blender.chmod(0o755)
        args = pipeline.parse_args([str(self.root / 'a.glb'), str(self.root / 'b.glb'),
            '--alphas', '0.5', '--ss-checkpoint', str(self.root / 'ss.pt'),
            '--slat-checkpoint', str(self.root / 'slat.pt'), '--output-dir', str(self.root / 'run'),
            '--trellis-root', str(trellis), '--blender', str(blender),
            '--glb-up-axis', 'z', '--src2-up-axis', 'y'])
        self.assertEqual(args.src1_up_axis, 'z')
        self.assertEqual(args.src2_up_axis, 'y')

    def precision_args(self, *extra):
        trellis = self.root / 'TRELLIS'
        model_file = trellis / 'trellis/models/__init__.py'
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()
        for name in ('input.pt', 'ss.pt', 'slat.pt'):
            (self.root / name).touch()
        return pipeline.parse_args([str(self.root / 'input.pt'), str(self.root / 'input.pt'),
            '--alphas', '0.5', '--ss-checkpoint', str(self.root / 'ss.pt'),
            '--slat-checkpoint', str(self.root / 'slat.pt'), '--output-dir', str(self.root / 'run'),
            '--trellis-root', str(trellis), '--mode', 'check', *extra])

    def test_sampling_defaults_to_fp32_and_reduced_precision_is_explicit(self):
        cases = [((), 'fp32', 'no'),
                 (('--mixed-precision', 'fp32'), 'fp32', 'no'),
                 (('--mixed-precision', 'no'), 'no', 'no'),
                 (('--mixed-precision', 'bf16'), 'bf16', 'bf16'),
                 (('--precision', 'fp16'), 'fp16', 'fp16'),
                 (('--mixed-precision', 'fp32', '--mixed-precision', 'bf16'), 'bf16', 'bf16'),
                 (('--mixed-precision', 'fp32', '--precision', 'fp16'), 'fp16', 'fp16')]
        for flags, requested, resolved in cases:
            with self.subTest(flags=flags):
                args = self.precision_args(*flags)
                trellis = SimpleNamespace(__file__=str(args.trellis_root / 'trellis/__init__.py'))
                backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace())
                with patch.dict('sys.modules', {'trellis': trellis}), \
                     patch.object(torch.cuda, 'is_available', return_value=True), \
                     patch.object(torch.cuda, 'is_bf16_supported', return_value=True) as support, \
                     patch.object(torch, 'set_num_threads'), \
                     patch.object(torch, 'backends', backends):
                    pipeline.runtime(args)
                self.assertEqual(args.mixed_precision, requested)
                self.assertEqual(args.precision, resolved)
                self.assertFalse(backends.cuda.matmul.allow_tf32)
                self.assertFalse(backends.cudnn.allow_tf32)
                if requested != 'bf16':
                    support.assert_not_called()

    def test_explicit_unsupported_bf16_fails_without_fallback(self):
        args = self.precision_args('--mixed-precision', 'bf16')
        trellis = SimpleNamespace(__file__=str(args.trellis_root / 'trellis/__init__.py'))
        with patch.dict('sys.modules', {'trellis': trellis}), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'is_bf16_supported', return_value=False), \
             self.assertRaisesRegex(RuntimeError, 'does not support bf16'):
            pipeline.runtime(args)

    def test_launcher_sets_fp32_and_forwards_explicit_overrides(self):
        executable = self.root / 'capture-python'
        executable.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
        executable.chmod(0o755)
        launcher = Path(pipeline.__file__).with_name('run_local.sh')
        environment = {**os.environ, 'MORPH_PYTHON': str(executable)}
        for flags in ([], ['--mixed-precision', 'bf16'], ['--precision', 'fp16']):
            with self.subTest(flags=flags):
                result = subprocess.check_output(['bash', str(launcher), 'a.glb', 'b.glb', *flags],
                                                 env=environment, text=True).splitlines()
                self.assertEqual(result, [str(launcher.with_name('morph_assets.py')),
                                         '--mixed-precision', 'fp32', 'a.glb', 'b.glb', *flags])
        doctor = subprocess.check_output(['bash', str(launcher), '--doctor', '--gpu'],
                                         env=environment, text=True).splitlines()
        self.assertEqual(doctor, [str(launcher.with_name('check_environment.py')), '--gpu'])


if __name__ == '__main__':
    unittest.main()
