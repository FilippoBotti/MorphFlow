import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image


spec = importlib.util.spec_from_file_location('render_repair', Path(__file__).with_name('resume_threeway_render_cache.py'))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class RenderCacheRepairTests(unittest.TestCase):
    def test_relocation_reproduces_original_identity_with_preserved_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root/'old', root/'new'
            old.mkdir(); new.mkdir()
            source = old/'mesh_raw.pt'
            source.write_bytes(b'raw Gaussian data')
            target = new/source.name
            shutil.copy2(source, target)
            args = SimpleNamespace(resolution=512, radius=2, fov_degrees=40, camera_distribution='hammersley')
            before = repair.cache_identity(source, args, 64, old, new)
            after = repair.cache_identity(target, args, 64, old, new)
            self.assertEqual(before, after)
            # Includes mtime/size: changed assets must not reuse old features.
            target.write_bytes(b'different asset')
            self.assertNotEqual(before, repair.cache_identity(target, args, 64, old, new))

    def test_atomic_write_preserves_exact_png_pixels_and_restores_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.png'
            image = Image.new('RGB', (8,8), color=(23,45,67))
            save = Image.Image.save
            with repair.atomic_png_saves():
                image.save(path)
            self.assertIs(Image.Image.save, save)
            with Image.open(path) as result:
                self.assertEqual(result.tobytes(), image.tobytes())
            self.assertEqual([p.name for p in path.parent.iterdir()], ['image.png'])

    def test_failed_atomic_commit_keeps_previous_valid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.png'
            Image.new('RGB', (8,8), color='red').save(path)
            previous = path.read_bytes()
            original = Image.Image.save
            with patch.object(repair.os, 'replace', side_effect=OSError('simulated disk full')):
                with self.assertRaises(OSError), repair.atomic_png_saves():
                    Image.new('RGB', (8,8), color='blue').save(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertIs(Image.Image.save, original)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_only_invalid_cached_images_are_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (8,8)).save(root/'000.png')
            (root/'001.png').write_bytes(b'')
            (root/'002.png').write_bytes((root/'000.png').read_bytes()[:30])
            (root/'identity.json').write_text('')
            repair.validate_png_directory(root, 3, 8)
            self.assertTrue((root/'000.png').is_file())
            self.assertFalse((root/'001.png').exists())
            self.assertFalse((root/'002.png').exists())
            self.assertFalse((root/'identity.json').exists())
            self.assertEqual(len(list(root.glob('*.invalid_*'))), 3)


if __name__ == '__main__':
    unittest.main()
