"""CPU tests for pairing, immutable inputs and resumable canonical endpoints."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from benchmarks import prepare_interp3ddata as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="interp3ddata_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw" / "all_cases_morphing"
        self.output = self.root / "prepared"
        # Preserve the real archive's subtle plane vs plane0 naming distinction.
        cases = {"easy": ("plane", "plane0"), "mid": ("plane",), "hard": ("plane",)}
        for difficulty, names in cases.items():
            directory = self.raw / difficulty
            directory.mkdir(parents=True)
            for index, case in enumerate(names):
                for endpoint in (0, 1):
                    Image.new("RGBA", (8, 6), (40 * index, 120 * endpoint, 10, 180)).save(
                        directory / (case + str(endpoint) + ".png"))

    def index(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return prep.index_dataset(self.raw.parent, self.output, **kwargs)

    def test_pairs_preserve_difficulty_trailing_digits_direction_and_original_pixels(self):
        provenance = self.index()
        self.assertEqual(provenance["num_pairs"], 4)
        self.assertEqual(provenance["num_assets"], 8)
        self.assertEqual(provenance["difficulty_pairs"], {"easy": 2, "mid": 1, "hard": 1})
        metadata = json.loads((self.output / "metadata_test.json").read_text())
        self.assertEqual([(row["src_1"], row["src_2"]) for row in metadata[:2]], [
            ("easy__plane0", "easy__plane1"), ("easy__plane00", "easy__plane01")])
        self.assertEqual([row["difficulty"] for row in metadata], ["easy", "easy", "mid", "hard"])
        for asset in provenance["assets"]:
            self.assertEqual((self.raw / asset["source_path"]).read_bytes(),
                             (self.output / asset["image"]).read_bytes())
            self.assertEqual(asset["seed"], prep.endpoint_seed(42, asset["name"]))
        self.assertEqual(len({asset["name"] for asset in provenance["assets"]}), 8)
        self.assertFalse((self.output / "targets").exists())

    def test_resume_keeps_metadata_bytes_mtime_and_rejects_changed_images_or_settings(self):
        self.index()
        paths = [self.output / name for name in ("metadata_test.json", "split_assets.json", "dataset_provenance.json")]
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
        self.index()
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths])
        with self.assertRaisesRegex(prep.PreparationError, "new --output-root"):
            self.index(steps=30)
        Image.new("RGBA", (8, 6), (255, 0, 0, 255)).save(self.raw / "easy/plane0.png")
        with self.assertRaisesRegex(prep.PreparationError, "new --output-root"):
            self.index()
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths])

    def test_incomplete_or_unrecognized_pair_is_rejected_without_silent_filtering(self):
        (self.raw / "hard/plane1.png").unlink()
        with self.assertRaisesRegex(prep.PreparationError, "Incomplete source pair"):
            self.index()
        self.assertFalse((self.output / "metadata_test.json").exists())
        Image.new("RGBA", (8, 6)).save(self.raw / "hard/plane1.png")
        Image.new("RGBA", (8, 6)).save(self.raw / "easy/extra2.png")
        with self.assertRaisesRegex(prep.PreparationError, "0.png or 1.png"):
            self.index()

    def test_canonical_endpoint_sampling_and_corrupt_cache_repair_feed_real_benchmark(self):
        import torch
        from data.morph_dataset import MorphingDistillDataset
        from benchmarks.threeway_textured_benchmark import select_test_pairs

        provenance = self.index()
        metadata_before = (self.output / "dataset_provenance.json").read_bytes()
        calls = []

        class FakePipeline:
            sparse_structure_sampler_params = {"steps": 25, "cfg_strength": 5., "cfg_interval": [.5, 1.], "rescale_t": 3.}
            slat_sampler_params = dict(sparse_structure_sampler_params)

            def preprocess_image(self, image):
                return image

            def get_cond(self, images):
                return {"cond": images[0].size}

            def sample_sparse_structure(self, cond, noise, num_samples, params):
                calls.append((torch.initial_seed(), params))
                return torch.tensor([[0, 1, 2, 3]], dtype=torch.int32), torch.randn(1, 8, 2, 2, 2)

            def sample_slat(self, cond, coords, params):
                return SimpleNamespace(feats=torch.randn(len(coords), 8), coords=coords)

        with redirect_stdout(io.StringIO()):
            prep.prepare_endpoints(self.output, lambda model_id: FakePipeline())
        self.assertEqual(len(calls), 8)
        self.assertEqual([seed for seed, params in calls], [asset["seed"] for asset in provenance["assets"]])
        self.assertTrue(all(params == {"steps": 25, "cfg_strength": 5.} for seed, params in calls))
        with redirect_stdout(io.StringIO()):
            prep.prepare_endpoints(self.output, lambda _: self.fail("Complete endpoint cache must not load a pipeline"))
        self.assertEqual(len(calls), 8)
        first = provenance["assets"][0]
        path = self.output / "assets" / first["name"] / "ss_latent.pt"
        expected = path.read_bytes()
        path.write_bytes(b"corrupted")
        self.assertFalse(prep.asset_ready(self.output, first, provenance["endpoint_generation"]))
        with redirect_stdout(io.StringIO()):
            prep.prepare_endpoints(self.output, lambda model_id: FakePipeline())
        self.assertEqual(len(calls), 9)
        self.assertTrue(prep.asset_ready(self.output, first, provenance["endpoint_generation"]))
        # Tensor contents (unlike torch archive bytes with random temporary names) are deterministic.
        recovered = torch.load(path, weights_only=True)
        with tempfile.NamedTemporaryFile() as previous:
            previous.write(expected)
            previous.flush()
            self.assertTrue(torch.equal(torch.load(previous.name, weights_only=True), recovered))
        self.assertEqual(metadata_before, (self.output / "dataset_provenance.json").read_bytes())
        dataset = MorphingDistillDataset(str(self.output), metadata_file="metadata_test.json", split="test",
                                        strict_split=True, load_occupancy=False, skip_missing=False,
                                        verbose=False, source_images_root=str(self.output / "images"))
        pairs, available = select_test_pairs(dataset, 0, 42)
        self.assertEqual((len(pairs), available), (4, 4))
        self.assertFalse((self.output / "targets").exists())


if __name__ == "__main__":
    unittest.main()
