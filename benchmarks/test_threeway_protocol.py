"""CPU regression checks for protocol and aggregation, without model downloads."""

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import eval_perceptual_sequence as base
import eval_textured_benchmark as evaluation
import threeway_textured_benchmark as benchmark


def minimal_glb() -> bytes:
    document = {"asset": {"version": "2.0"}, "meshes": [{"primitives": []}], "buffers": [{"byteLength": 4}]}
    encoded = json.dumps(document).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    body = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded + struct.pack("<II", 4, 0x004E4942) + b"\x00" * 4
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


class ProtocolTests(unittest.TestCase):
    def test_all_test_pairs_are_unique_complete_repeatable_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / "source.pt"
            available.write_bytes(b"source input")
            entries = [{"src_1": "a", "src_2": "b", "target": "ab1"},
                       {"src_1": "c", "src_2": "d", "target": "cd1"},
                       {"src_1": "a", "src_2": "b", "target": "ab2"}]
            dataset = SimpleNamespace(
                metadata=entries, _entry_allowed_by_split=lambda entry: True,
                _paths_for_entry=lambda entry: {key: available for key in (
                    "src1_feats", "src1_coords", "src1_ss_latent", "src2_feats", "src2_coords", "src2_ss_latent")},
                _find_source_image_path=lambda asset, entry, prefix: available)
            selected, count = benchmark.select_test_pairs(dataset, 0, 42)
            self.assertEqual(count, 2)
            self.assertEqual({(entry["src_1"], entry["src_2"]) for _, entry in selected}, {("a", "b"), ("c", "d")})
            self.assertEqual({index for index, _ in selected}, {0, 1})
            self.assertEqual(selected, benchmark.select_test_pairs(dataset, 0, 42)[0])
            self.assertEqual(len(benchmark.select_test_pairs(dataset, 1, 42)[0]), 1)
            with self.assertRaises(ValueError):
                benchmark.select_test_pairs(dataset, -1, 42)
            with self.assertRaises(RuntimeError):
                benchmark.select_test_pairs(dataset, 3, 42)
            dataset._find_source_image_path = lambda asset, entry, prefix: None if asset == "c" else available
            with self.assertRaisesRegex(FileNotFoundError, "no pairs were dropped"):
                benchmark.select_test_pairs(dataset, 0, 42)
            dataset._find_source_image_path = lambda asset, entry, prefix: available
            old_paths = dataset._paths_for_entry
            dataset._paths_for_entry = lambda entry: dict(old_paths(entry), src1_feats=root / "absent.pt")
            with self.assertRaisesRegex(FileNotFoundError, "missing src1_feats"):
                benchmark.select_test_pairs(dataset, 0, 42)
            dataset._entry_allowed_by_split = lambda entry: entry["src_1"] != "c"
            with self.assertRaisesRegex(ValueError, "strict test split"):
                benchmark.select_test_pairs(dataset, 0, 42)

    def test_all_pair_plan_persists_actual_count_and_unfiltered_index_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / "source.pt"
            available.write_bytes(b"source input")
            dataset = SimpleNamespace(
                metadata=[{"src_1": "a", "src_2": "b", "target": "ab"},
                          {"src_1": "c", "src_2": "d", "target": "cd"}],
                _entry_allowed_by_split=lambda entry: True,
                _paths_for_entry=lambda entry: {key: available for key in (
                    "src1_feats", "src1_coords", "src1_ss_latent", "src2_feats", "src2_coords", "src2_ss_latent")},
                _find_source_image_path=lambda asset, entry, prefix: available)
            args = benchmark.build_parser().parse_args([
                "plan", "--run-root", str(root / "run"), "--data-root", str(root),
                "--source-images-root", str(root), "--num-pairs", "0", "--k", "5"])
            with mock.patch("data.morph_dataset.MorphingDistillDataset", return_value=dataset) as constructor:
                benchmark.command_plan(args)
                self.assertFalse(constructor.call_args.kwargs["skip_missing"])
                benchmark.command_plan(args)
                self.assertEqual(constructor.call_count, 1)
            plan = json.loads((root / "run" / "plan.json").read_text())
            self.assertEqual(plan["num_pairs"], 2)
            self.assertEqual(plan["requested_num_pairs"], 0)
            self.assertEqual(plan["available_test_pairs"], 2)
            self.assertEqual(plan["metadata_index_basis"], "unfiltered")
            shard_ids = [pair["pair_id"] for shard in range(8) for pair in benchmark.shard_pairs(plan, shard, 8)]
            self.assertEqual(sorted(shard_ids), [0, 1])
            args.num_pairs = 2
            with self.assertRaisesRegex(ValueError, "Existing plan requested"):
                benchmark.command_plan(args)
            # Positive-N plans made before the new field remain resumable.
            del plan["requested_num_pairs"]
            benchmark.write_json(root / "run" / "plan.json", plan)
            benchmark.command_plan(args)

    def test_atomic_glb_export_repairs_truncation_and_preserves_completed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "pred_final.glb"
            complete = minimal_glb()
            destination.write_bytes(complete[:-3])
            self.assertFalse(benchmark.glb_ready(destination))
            exporter = mock.Mock(side_effect=lambda path: path.write_bytes(complete))
            self.assertTrue(benchmark.atomic_export_glb(exporter, destination))
            self.assertTrue(benchmark.glb_ready(destination))
            self.assertFalse(benchmark.atomic_export_glb(exporter, destination))
            self.assertEqual(exporter.call_count, 1)
            with self.assertRaisesRegex(ValueError, "Invalid GLB export"):
                benchmark.atomic_export_glb(lambda path: path.write_bytes(complete[:20]), destination, overwrite=True)
            self.assertEqual(destination.read_bytes(), complete)
            self.assertEqual(list(root.glob(".*.tmp.glb")), [])
            linked = root / "src1.glb"
            linked.write_bytes(b"glTF")
            benchmark.hardlink_or_copy(destination, linked)
            self.assertTrue(benchmark.glb_ready(linked))
            destination.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "Invalid source GLB"):
                benchmark.hardlink_or_copy(destination, root / "src2.glb")

    def test_texture_selected_method_and_audit_include_all_endpoint_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = {"pair_id": 0, "pair_dir": "pair_0000_a_to_b", "src1": "a", "src2": "b"}
            plan = {"num_pairs": 1, "k_intermediate": 2, "alphas": [0.7, 0.3], "pairs": [pair]}
            benchmark.write_json(root / "plan.json", plan)
            pair_root = root / "morphflow" / pair["pair_dir"]
            for alpha in plan["alphas"]:
                raw = pair_root / benchmark.alpha_dir(alpha) / "mesh_raw.pt"
                raw.parent.mkdir(parents=True)
                raw.write_bytes(b"unused by fake exporter")
            args = benchmark.build_parser().parse_args([
                "texture", "--run-root", str(root), "--methods", "morphflow"])
            def fake_texture(raw, glb, *unused):
                benchmark.atomic_export_glb(lambda path: path.write_bytes(minimal_glb()), glb)
            fake_nvdiffrast = SimpleNamespace(torch=SimpleNamespace())
            with mock.patch.dict(sys.modules, {"nvdiffrast": fake_nvdiffrast, "nvdiffrast.torch": fake_nvdiffrast.torch}), \
                    mock.patch.object(benchmark, "texture_one", side_effect=fake_texture), mock.patch.object(torch.cuda, "empty_cache"):
                benchmark.command_texture(args)
            self.assertFalse((root / "morphany3d").exists())
            sequence = json.loads((pair_root / "sequence.json").read_text())
            self.assertEqual(len(sequence), 4)
            benchmark.command_texture_status(args)
            status = json.loads((root / "texture_status_morphflow.json").read_text())
            self.assertEqual(status["checked_glbs"], 6)
            self.assertEqual(status["methods"], ["morphflow"])
            (pair_root / "src1.glb").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "GLBs missing or invalid"):
                benchmark.command_texture_status(args)
            args.methods = list(benchmark.METHODS)
            with self.assertRaises(FileNotFoundError):
                benchmark.command_texture_status(args)

    def test_paper_beta_schedule_direction_and_endpoints(self):
        from scipy.stats import beta
        alphas = benchmark.canonical_alphas(5)
        internal_t = [0.0] + [1 - a for a in alphas] + [1.0]
        np.testing.assert_allclose(internal_t, beta.ppf(np.linspace(0, 1, 7), 5, 5), atol=1e-10)
        self.assertTrue(all(a > b for a, b in zip(alphas, alphas[1:])))
        self.assertEqual(alphas[2], 0.5)
        np.testing.assert_allclose(benchmark.canonical_alphas(5, "uniform"), np.linspace(1, 0, 7)[1:-1])

    def test_spherical_cameras_match_upstream_hammersley_origins(self):
        args = SimpleNamespace(num_views=64, resolution=512, radius=2, fov_degrees=40, camera_distribution="hammersley")
        for index, view in enumerate(base.uniform_views(args)):
            digits, denominator, inverse = index, 2, 0.0
            while digits:
                inverse += (digits % 2) / denominator
                digits //= 2
                denominator *= 2
            official_yaw = 2 * math.pi * inverse
            pitch = math.acos(1 - 2 * index / 64) - math.pi / 2
            upstream = [math.sin(official_yaw) * math.cos(pitch), math.cos(official_yaw) * math.cos(pitch), math.sin(pitch)]
            blender = [math.cos(view["yaw"]) * math.cos(view["pitch"]), math.sin(view["yaw"]) * math.cos(view["pitch"]), math.sin(view["pitch"])]
            np.testing.assert_allclose(blender, upstream, atol=1e-14)
        self.assertGreater(len({round(view["pitch"], 8) for view in base.uniform_views(args)}), 50)

    def test_random_fid_states_are_repeatable_and_independent_of_method(self):
        selections = [evaluation.random_frame_indices(5, 2, 42, pair) for pair in range(20)]
        self.assertEqual(selections, [evaluation.random_frame_indices(5, 2, 42, pair) for pair in range(20)])
        self.assertTrue(all(len(set(selected)) == 2 for selected in selections))
        self.assertGreater(len({tuple(selected) for selected in selections}), 5)
        self.assertNotEqual(selections, [evaluation.random_frame_indices(5, 2, 43, pair) for pair in range(20)])

    def test_full_path_includes_both_reference_transitions_and_normalizes(self):
        args = SimpleNamespace(num_views=1, lpips_batch_size=8)
        frames = [SimpleNamespace(key="a0p7"), SimpleNamespace(key="a0p3")]
        with mock.patch.object(base, "image_path", side_effect=lambda root, key, view: root / key / str(view)), mock.patch.object(base, "batched_lpips", side_effect=[[0.1, 0.2, 0.3], [0.4]]):
            metrics, transitions, _ = evaluation.full_path_metrics(args, Path("/tmp/metrics"), frames, None, None)
        self.assertEqual([(row["from"], row["to"]) for row in transitions], [("src1", "a0p7"), ("a0p7", "a0p3"), ("a0p3", "src2")])
        self.assertAlmostEqual(metrics["ppl_full_sum"]["mean"], 0.6)
        self.assertAlmostEqual(metrics["ppl_full_normalized_reference_endpoints"]["mean"], 1.5)

    def test_geometry_only_raw_is_incomplete_and_appearance_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh_raw.pt"
            torch.save({"vertices": torch.zeros(3, 3)}, path)
            self.assertFalse(benchmark.raw_asset_ready(path))
            mesh = SimpleNamespace(vertices=torch.zeros(3, 3), faces=torch.tensor([[0, 1, 2]]), vertex_attrs=None, res=64)
            gaussian = SimpleNamespace(init_params={"aabb": [0, 0, 0, 1, 1, 1]}, active_sh_degree=0,
                                       **{name: torch.ones(3, 1) if name != "_features_rest" else None for name in benchmark.GAUSSIAN_TENSOR_FIELDS})
            benchmark.save_raw_mesh(mesh, path, gaussian=gaussian)
            self.assertTrue(benchmark.raw_asset_ready(path))
            state = benchmark.torch_load_cpu(path)["gaussian"]
            self.assertEqual(state["init_params"], gaussian.init_params)
            for name in benchmark.GAUSSIAN_TENSOR_FIELDS:
                if getattr(gaussian, name) is not None:
                    torch.testing.assert_close(state["tensors"][name], getattr(gaussian, name))
            with self.assertRaises(ValueError):
                benchmark.save_raw_mesh(mesh, Path(directory) / "missing.pt")

    def test_trajectory_mean_fid_is_preserved_separately_from_pooled_fid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pair, fid in enumerate([2.0, 8.0]):
                output = root / f"pair_{pair:04d}_a_to_b"
                output.mkdir()
                value = {"num_generated_frames": 5, "num_views": 64, "lpips_backbone": "vgg",
                         "fid_protocol": {"views_per_state": 1000}, "render_backend": "gaussian",
                         "aggregate_across_views": {"fid_endpoint_manifold": {"mean": fid, "std": 0, "median": fid}}}
                (output / "metrics.json").write_text(json.dumps(value))
                rng = np.random.default_rng(pair)
                np.savez(output / "fid_features.npz", reference=rng.normal(size=(8, 3)), generated=rng.normal(size=(8, 3)))
            args = argparse.Namespace(output_dir=str(root), pair_glob="pair_*", expected_num_pairs=2, allow_incomplete=False, fid_min_samples=2)
            with mock.patch.object(evaluation, "frechet_distance", return_value=99.0):
                evaluation.merge_outputs(args)
            batch = json.loads((root / "metrics_batch.json").read_text())
            self.assertEqual(batch["aggregate_across_pairs"]["fid_endpoint_manifold"]["mean"], 5.0)
            self.assertEqual(batch["aggregate_across_pairs"]["fid_endpoint_manifold"]["std"], 3.0)
            self.assertEqual(batch["aggregate_across_pairs"]["fid_endpoint_manifold_pooled"]["mean"], 99.0)
            args.expected_num_pairs = 3
            with self.assertRaises(RuntimeError):
                evaluation.merge_outputs(args)

    def test_fid_gaussian_translation_has_known_distance(self):
        features = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        self.assertAlmostEqual(evaluation.frechet_distance(features, features), 0.0, places=10)
        self.assertAlmostEqual(evaluation.frechet_distance(features, features + [1., 2.]), 5.0, places=10)

    def test_completed_metric_signature_changes_with_inputs_or_fid_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "asset.pt"
            raw.write_bytes(b"first asset")
            args = SimpleNamespace(num_views=64, resolution=512, radius=2, fov_degrees=40,
                                   camera_distribution="hammersley", lpips_net="vgg", fid_num_views=1000,
                                   fid_num_intermediate_states=2, fid_selection_seed=42, render_backend="gaussian")
            frames = [SimpleNamespace(mesh=raw, alpha=0.5)]
            original = evaluation.evaluation_signature(args, raw, raw, frames)
            self.assertEqual(original, evaluation.evaluation_signature(args, raw, raw, frames))
            args.fid_selection_seed = 43
            self.assertNotEqual(original, evaluation.evaluation_signature(args, raw, raw, frames))
            args.fid_selection_seed = 42
            raw.write_bytes(b"different source asset")
            self.assertNotEqual(original, evaluation.evaluation_signature(args, raw, raw, frames))

    def test_upstream_tfsa_file_calls_use_bounded_ram_and_spill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "node_tmp"
            work.mkdir()
            cache = benchmark.HybridTFSACache(32, root / "persistent_spill")
            adapter = benchmark.TFSAFileCacheAdapter(cache, work)
            module = SimpleNamespace(torch=torch, os=os)
            original_save, original_exists = torch.save, os.path.exists
            first = work / "ss_sa_morphing1_step0_block0.pt"
            second = work / "slat_sa_morphing2_step0_block0.pt"
            coordinates = work / "feat_coords_morphing2.pt"
            endpoint = work / "slat_init.pt"
            value = torch.arange(8, dtype=torch.float32)
            with adapter.install([module]):
                module.torch.save({"k": value}, first)
                value.add_(100)
                module.torch.save({"k": value}, second)
                module.torch.save(torch.arange(4), coordinates)
                module.torch.save(value, endpoint)
                self.assertIs(torch.save, original_save)
                self.assertIs(os.path.exists, original_exists)
                self.assertTrue(module.os.path.exists(first))
                self.assertFalse(os.path.exists(first))
                self.assertFalse(second.exists())
                self.assertTrue(endpoint.exists())
                torch.testing.assert_close(module.torch.load(first)["k"], torch.arange(8, dtype=torch.float32))
                torch.testing.assert_close(module.torch.load(second)["k"], value)
                self.assertLessEqual(cache.peak_ram_bytes, 32)
                self.assertGreater(cache.spill_bytes_written, 0)
                self.assertIn(first.name, cache)
                self.assertIn(second.name, cache)
                cache.prune_to_morphing_idx(2)
                self.assertFalse(module.os.path.exists(first))
                self.assertTrue(module.os.path.exists(second))
                self.assertTrue(module.os.path.exists(coordinates))
            self.assertIs(module.torch, torch)
            self.assertIs(module.os, os)
            cache.clear()
            self.assertEqual(cache.ram_bytes, 0)
            self.assertEqual(cache.spill_bytes, 0)
            self.assertFalse(cache.spill_dir.exists())

    def test_explicit_tfsa_spill_storage_is_per_pair_and_preserves_endpoint_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            persistent = root / "method_cache" / "morphany3d"
            args = benchmark.build_parser().parse_args([
                "morphany3d", "--run-root", str(root), "--local-cache-root", str(root / "staging")])
            pair_dir = "pair_0000_a_to_b"
            self.assertEqual(benchmark.morphany_tfsa_spill_directory(args, persistent, pair_dir), persistent / pair_dir / "tfsa_spill")
            args.tfsa_spill_root = str(root / "nvme")
            selected = benchmark.morphany_tfsa_spill_directory(args, persistent, pair_dir)
            self.assertEqual(selected, root / "nvme" / pair_dir / "tfsa_spill")
            self.assertNotEqual(selected, benchmark.morphany_tfsa_spill_directory(args, persistent, "pair_0001_c_to_d"))
            # An explicit persistent parent is safe: clear only removes spill
            # state, leaving endpoint and unrelated pair files in place.
            args.tfsa_spill_root = str(persistent)
            endpoint = persistent / pair_dir / "src" / "slat_init.pt"
            endpoint.parent.mkdir(parents=True)
            endpoint.write_bytes(b"persistent endpoint cache")
            selected = benchmark.morphany_tfsa_spill_directory(args, persistent, pair_dir)
            cache = benchmark.HybridTFSACache(0, selected)
            cache["ss_sa_morphing1_step0_block0.pt"] = torch.ones(4)
            self.assertGreater(cache.spill_bytes_written, 0)
            cache.clear()
            self.assertFalse(selected.exists())
            self.assertEqual(endpoint.read_bytes(), b"persistent endpoint cache")
            with self.assertRaisesRegex(ValueError, "Unsafe pair directory"):
                benchmark.morphany_tfsa_spill_directory(args, persistent, "../references_raw")

    def test_tfsa_pressure_guard_and_exception_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = benchmark.HybridTFSACache(128, root / "spill")
            cache["ss_sa_morphing1_step0_block0.pt"] = torch.ones(8)
            adapter = benchmark.TFSAFileCacheAdapter(cache, root, min_free_bytes=100)
            module = SimpleNamespace(torch=torch, os=os)
            with self.assertRaisesRegex(RuntimeError, "sampler failed"):
                with adapter.install([module]), mock.patch.object(benchmark, "node_mem_available_bytes", return_value=50):
                    module.torch.save(torch.ones(8), root / "slat_sa_morphing2_step0_block0.pt")
                    self.assertTrue(adapter.pressure_relief_triggered)
                    self.assertEqual(cache.max_ram_bytes, 0)
                    self.assertEqual(cache.ram_bytes, 0)
                    raise RuntimeError("sampler failed")
            self.assertIs(module.torch, torch)
            self.assertIs(module.os, os)
            cache.clear()


if __name__ == "__main__":
    unittest.main()
