"""CPU checks for leakage prevention, temporal alignment, and portable outputs."""

import contextlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import generate_teacher_student_comparison as comparison


def fake_bundle(path, image=None):
    path.mkdir(parents=True, exist_ok=True)
    files = list(comparison.BUNDLE_FILES)
    if image:
        files.append(image)
    for filename in files:
        (path / filename).write_text("{}" if filename.endswith(".json") else "fixture")
    comparison.write_json(path / "complete.json", {
        "files": {name: (path / name).stat().st_size for name in files}
    })


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "flux inputs"
        self.dataset = self.root / "dataset"
        self.output = self.root / "comparison output"
        self.assets.mkdir()
        self.dataset.mkdir()
        for name in "abcd":
            (self.assets / f"{name}.png").write_bytes(b"image fixture")
        comparison.write_json(self.dataset / "metadata_train.json", [{"src_1": "b", "src_2": "a"}])
        comparison.write_json(self.dataset / "metadata_val.json", {"samples": [{"src_1": "c", "src_2": "a"}]})
        comparison.write_json(self.dataset / "metadata_test.json", [{"src_1": "d", "src_2": "a"}])
        self.argv = [
            "--assets-dir", str(self.assets), "--dataset-dir", str(self.dataset),
            "--output-dir", str(self.output), "--checkpoint-path", str(self.root / "ss.pt"),
            "--slat-checkpoint-path", str(self.root / "slat.pt"),
            "--num-pairs", "3", "--num-intermediates", "7",
        ]

    def plan(self):
        return comparison.build_plan(comparison.parse_args(self.argv + ["--dry-run"]))

    def test_exact_complement_dense_sparse_and_reverse_exclusions(self):
        rng = random.Random(37)
        for count in range(2, 35):
            names = [f"asset_{i:03}" for i in range(count)]
            all_pairs = set(itertools.combinations(names, 2))
            for fraction in (0, 0.2, 0.95, 1):
                excluded = set(rng.sample(sorted(all_pairs), int(len(all_pairs) * fraction)))
                expected = all_pairs - excluded
                reversed_pairs = {(b, a) for a, b in excluded}
                if not expected:
                    with self.assertRaisesRegex(ValueError, "only 0"):
                        comparison.select_pairs(names, reversed_pairs, 1, 42)
                    continue
                selected, available = comparison.select_pairs(names, reversed_pairs, len(expected), 42)
                self.assertEqual(set(selected), expected)
                self.assertEqual(len(selected), available)
                self.assertEqual(selected, comparison.select_pairs(names, excluded, available, 42)[0])

    def test_all_splits_and_planned_and_partial_pairs_are_excluded(self):
        comparison.write_json(self.dataset / "pair_sequence_alphas.json", {"train": {"c+b": [0.5]}})
        (self.dataset / "targets" / "d+c" / "alpha_0p5").mkdir(parents=True)
        excluded, provenance = comparison.dataset_pairs(self.dataset)
        self.assertEqual(len(excluded), 5)
        self.assertEqual(len(provenance), 4)
        self.assertEqual(comparison.select_pairs("abcd", excluded, 1, 1)[0], [("b", "d")])
        with self.assertRaisesRegex(ValueError, "only 1"):
            comparison.select_pairs("abcd", excluded, 2, 1)

    def test_invalid_metadata_and_name_collisions_fail_closed(self):
        comparison.write_json(self.dataset / "metadata_extra.json", [{"src_1": "x"}])
        with self.assertRaisesRegex(ValueError, "Missing src_1/src_2"):
            comparison.dataset_pairs(self.dataset)
        (self.assets / "a.jpg").write_bytes(b"duplicate")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            comparison.discover_images(self.assets)

    def test_dry_run_works_without_torch_or_checkpoints(self):
        completed = subprocess.run([sys.executable, comparison.__file__, *self.argv, "--dry-run"],
                                   text=True, capture_output=True, check=True)
        self.assertIn("3 unseen pairs", completed.stdout)
        plan = comparison.read_json(self.output / "plan.dry_run.json")
        self.assertEqual(plan["alphas"], [i / 8 for i in range(7, 0, -1)])
        self.assertFalse((self.output / "plan.json").exists())
        self.assertFalse((self.output / "assets").exists())
        self.assertEqual({(p["src_1"], p["src_2"]) for p in plan["pairs"]},
                         {("b", "c"), ("b", "d"), ("c", "d")})

    def test_resume_rejects_changed_checkpoint_or_exclusions(self):
        for filename in ("ss.pt", "slat.pt"):
            (self.root / filename).write_bytes(b"checkpoint")
        # A teacher-only stage does not publish matched results.
        with patch.object(comparison.subprocess, "run") as run:
            comparison.main(self.argv + ["--stage", "teacher"])
            self.assertEqual(run.call_args.args[0][2], "_worker")
            self.assertEqual(run.call_args.args[0][3], "teacher")
            comparison.main(self.argv + ["--stage", "teacher", "--resume"])
            self.assertEqual(run.call_count, 2)
            (self.root / "ss.pt").write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "Resume plan differs"):
                comparison.main(self.argv + ["--resume"])
            self.assertEqual(run.call_count, 2)

    def test_incomplete_mesh_cannot_be_published_as_success(self):
        plan = self.plan()
        with self.assertRaisesRegex(RuntimeError, "Missing matched"):
            comparison.publish_comparison(self.output, plan)
        self.assertFalse((self.output / "comparison_manifest.json").exists())
        bundle = self.output / "partial"
        fake_bundle(bundle)
        self.assertTrue(comparison.bundle_ready(bundle))
        (bundle / "mesh.glb").write_bytes(b"")
        self.assertFalse(comparison.bundle_ready(bundle))

    def test_output_paths_and_metric_layout_survive_moving_run(self):
        plan = self.plan()
        for name in plan["sources"]:
            fake_bundle(self.output / "assets" / name, "input.png")
        for version in ("teacher", "student"):
            comparison.prepare_layout(self.output, plan, version)
            for pair in plan["pairs"]:
                for i in range(1, 8):
                    fake_bundle(self.output / version / pair["id"] / comparison.step_name(i))
            comparison.publish_version(self.output, plan, version)
        comparison.publish_comparison(self.output, plan)
        moved = self.root / "moved run"
        self.output.rename(moved)
        manifest = comparison.read_json(moved / "comparison_manifest.json")
        self.assertEqual(manifest["num_matched_samples"], 21)
        for row in manifest["samples"]:
            folder = moved / row["comparison_dir"]
            for name in ("src1.glb", "src2.glb", "target.glb", "pred_final.glb"):
                self.assertTrue((folder / name).is_file())
            for version in ("teacher", "student"):
                source = moved / version / row["id"] / "src1" / "ss_latent.pt"
                self.assertTrue(source.is_file())
        for version in ("teacher", "student"):
            metadata = comparison.read_json(moved / version / "metadata.json")
            self.assertEqual(len(metadata), 21)
            for row in metadata:
                for key, value in row.items():
                    if key.endswith(("_feats", "_coords", "_latent", "_occupancy", "_image")):
                        self.assertTrue((moved / version / value).is_file(), (key, value))

    def test_teacher_replays_tfsa_prefix_and_passes_exact_grid(self):
        plan = self.plan()
        plan["pairs"] = plan["pairs"][:1]
        for name in plan["sources"]:
            fake_bundle(self.output / "assets" / name, "input.png")
        pair = plan["pairs"][0]
        first = self.output / "teacher" / pair["id"] / comparison.step_name(1)
        fake_bundle(first)  # Interrupted run: only frame 1 is already complete.
        pipeline = SimpleNamespace(models={"slat_decoder_mesh": Mock(), "sparse_structure_decoder": Mock()},
                                   device="cuda", cuda=Mock())
        pipeline_class = Mock()
        pipeline_class.from_pretrained.return_value = pipeline
        teacher = ModuleType("data.dataset_scripts.generate_morphany3d_split_dataset")
        teacher.encode_image_condition = Mock(return_value={"cond": "condition"})
        teacher.run_one_morphany3d_step = Mock(side_effect=lambda *a, **kw: (Mock(), Mock(), Mock()))
        teacher.cleanup_old_index = Mock()
        teacher.cleanup_old_tfsa_cache = Mock()
        teacher.directory_size_bytes = Mock(return_value=0)
        modules = {
            "torch": SimpleNamespace(no_grad=contextlib.nullcontext, float32="float32",
                                     cuda=SimpleNamespace(empty_cache=Mock())),
            "trellis": ModuleType("trellis"),
            "trellis.pipelines": SimpleNamespace(TrellisImageTo3DPipeline=pipeline_class),
            "trellis.modules": ModuleType("trellis.modules"),
            "trellis.modules.sparse": ModuleType("trellis.modules.sparse"),
            "trellis.modules.sparse.basic": SimpleNamespace(SparseTensor=Mock()),
            "eval_validation_latents": SimpleNamespace(patch_trellis_mesh_dtype=Mock()),
            teacher.__name__: teacher,
        }
        with patch.dict(sys.modules, modules), patch.object(comparison, "save_bundle", side_effect=lambda path, *a, **kw: fake_bundle(path)) as save:
            comparison.teacher_phase(self.output, plan, str(self.root / "cache"))
        calls = teacher.run_one_morphany3d_step.call_args_list
        self.assertEqual(len(calls), 7)  # Frame 1 replayed despite its saved bundle.
        self.assertEqual([c.args[5] for c in calls], list(range(1, 8)))
        self.assertEqual([c.args[6] for c in calls], [9] * 7)
        self.assertEqual([c.args[7] for c in calls], plan["alphas"])
        self.assertEqual(save.call_count, 6)
        self.assertEqual(list((self.root / "cache").iterdir()), [])

    def test_student_uses_own_ss_coordinates_and_identical_alphas(self):
        plan = self.plan()
        plan["pairs"] = plan["pairs"][:1]
        for name in plan["sources"]:
            fake_bundle(self.output / "assets" / name, "input.png")
        pair = plan["pairs"][0]
        for i in range(1, 8):
            fake_bundle(self.output / "teacher" / pair["id"] / comparison.step_name(i))
        models = [Mock(), Mock()]
        for model in models:
            model.to.return_value.eval.return_value = model
        logits = MagicMock()
        occupancy = object()
        logits.__gt__.return_value = occupancy
        raw_coords = MagicMock()
        coords = object()
        raw_coords.__getitem__.return_value.int.return_value = coords
        torch = SimpleNamespace(device=lambda name: name, no_grad=contextlib.nullcontext,
                                cuda=SimpleNamespace(empty_cache=Mock()), zeros_like=Mock(),
                                argwhere=Mock(return_value=raw_coords))
        inference = SimpleNamespace(
            resolve_mixed_precision=Mock(return_value="no"),
            load_checkpoint=Mock(side_effect=["ss", "slat"]),
            detect_flow_target=lambda ckpt: ckpt,
            detect_model_type=Mock(return_value="image_large"),
            build_model=Mock(side_effect=models), preload_dino_if_needed=Mock(),
            checkpoint_requires_source_images=Mock(return_value=True),
            load_decoders=Mock(return_value=(Mock(), Mock(), Mock())),
            sample_ss=Mock(return_value=object()), ss_logits_raw=Mock(return_value=logits),
            sample_slat_on_coords=Mock(return_value=object()),
        )
        loader = Mock()
        seeds = Mock()
        generate = SimpleNamespace(
            load_asset=lambda root, name: {"name": name, "ss_latent": object()},
            batch_for_alpha=lambda a, b, alpha: {"alpha": alpha, "src1_image": a["image"], "src2_image": b["image"]},
        )
        modules = {
            "torch": torch,
            "data.dataset_scripts.generate_morphany3d_split_dataset": SimpleNamespace(seed_everything=seeds),
            "data.morph_dataset": SimpleNamespace(MorphingDistillDataset=Mock(return_value=loader)),
            "eval_validation_latents": inference, "generate_alpha_steps": generate,
        }
        with patch.dict(sys.modules, modules), patch.object(comparison, "save_bundle", side_effect=lambda path, *a, **kw: fake_bundle(path)):
            comparison.student_phase(self.output, plan)
        calls = inference.sample_slat_on_coords.call_args_list
        self.assertEqual([c.args[1]["alpha"] for c in calls], plan["alphas"])
        self.assertTrue(all(c.args[2] is coords for c in calls))
        self.assertTrue(all(c.args[0] is models[1] for c in calls))
        self.assertEqual(loader._load_source_image.call_count, 2)
        self.assertEqual([c.args[0] for c in seeds.call_args_list], [42, 43] * 7)

    def test_slurm_forwards_paths_with_spaces_and_gpu_visibility(self):
        binary_dir = self.root / "bin"
        binary_dir.mkdir()
        (binary_dir / "module").write_text("#!/bin/bash\nexit 0\n")
        (binary_dir / "module").chmod(0o755)
        capture = self.root / "singularity.json"
        (binary_dir / "singularity").write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            "with open(os.environ['CAPTURE'], 'w') as f:\n"
            "    json.dump({'args': sys.argv[1:], 'cuda': os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES'), 'script': sys.stdin.read()}, f)\n"
        )
        (binary_dir / "singularity").chmod(0o755)
        base = self.root / "trellis base"
        (base / "env").mkdir(parents=True)
        morph = self.root / "MorphAny3D checkout"
        (morph / "trellis").mkdir(parents=True)
        for name in ("ss.pt", "slat.pt", "container.sif"):
            (self.root / name).write_bytes(b"fixture")
        cache = self.root / "cache space"
        project = Path(comparison.__file__).parent
        env = {**os.environ, "PATH": f"{binary_dir}:{os.environ['PATH']}",
               "PROJECT_DIR": str(project), "MORPHANY3D_DIR": str(morph),
               "TRELLIS_BASE": str(base), "SIF": str(self.root / "container.sif"),
               "ASSETS_DIR": str(self.assets), "DATASET_DIR": str(self.dataset),
               "CHECKPOINT_PATH": str(self.root / "ss.pt"), "SLAT_CHECKPOINT_PATH": str(self.root / "slat.pt"),
               "OUTPUT_DIR": str(self.output), "WORK_CACHE_ROOT": str(cache),
               "NUM_PAIRS": "17", "NUM_INTERMEDIATES": "9", "CUDA_VISIBLE_DEVICES": "3",
               "SLURM_JOB_ID": "12345", "CAPTURE": str(capture)}
        subprocess.run(["bash", str(project / "slurm/generate_teacher_student_comparison.slurm"), "--dry-run"],
                       env=env, check=True, capture_output=True, text=True)
        result = comparison.read_json(capture)
        self.assertEqual(result["cuda"], "3")
        for flag, value in (("--assets-dir", str(self.assets)), ("--output-dir", str(self.output)),
                            ("--num-pairs", "17"), ("--num-intermediates", "9")):
            self.assertEqual(result["args"][result["args"].index(flag) + 1], value)
        self.assertEqual(result["args"][-1], "--dry-run")
        self.assertIn(f"{self.assets}:{self.assets}:ro", result["args"])
        self.assertEqual(list(cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
