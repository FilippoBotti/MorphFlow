"""CPU tests for immutable inputs and portable multi-CFG run layouts."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("threeway_ateneo", Path(__file__).with_name("threeway_ateneo.py"))
ateneo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ateneo)


class AteneoLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "run"
        self.project = self.base / "project"
        (self.project / "models").mkdir(parents=True)
        (self.project / "models/model.py").write_text("VERSION = 1\n")
        (self.project / "outputs").mkdir()
        (self.project / "outputs/ignored.py").write_text("ignored\n")
        self.data = self.base / "data"
        self.data.mkdir()
        (self.data / "metadata_test.json").write_text(json.dumps([
            {"src_1": "a", "src_2": "b"}, {"src_1": "a", "src_2": "b"},
            {"src_1": "c", "src_2": "d"},
        ]))
        (self.data / "split_assets.json").write_text(json.dumps({"test": ["a", "b", "c", "d"]}))
        self.env = dict(ateneo.PROTOCOL, PROJECT_DIR=str(self.project), DATA_ROOT_HOST=str(self.data),
                        SOURCE_IMAGES_ROOT_HOST=str(self.base / "images"), METADATA="metadata_test.json",
                        N="0", SEED="42", CFG_VALUES="2 3", SLAT_CFG_VALUES="1.5 2")
        for name in ("CHECKPOINT_PATH", "SLAT_CHECKPOINT_PATH", "SIF"):
            p = self.base / name
            p.write_bytes((name + " contents").encode())
            self.env[name] = str(p)
        for name in ("ORIGINAL_ENV", "MORPHANY_ENV", "INTERP_ENV", "PROTOCOL_PYTHON"):
            p = self.base / name
            p.mkdir()
            self.env[name] = str(p)
        for name in ateneo.REPOS.values():
            p = self.base / name / "trellis"
            p.mkdir(parents=True)
            (p / "__init__.py").write_text("FORK = {!r}\n".format(name))
            (p / "resource.bin").write_bytes(b"needed non-Python resource")
            self.env[name] = str(p.parent)
        self.args = argparse.Namespace(run_root=str(self.root))

    def prepare(self):
        with patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(io.StringIO()):
            ateneo.prepare(self.args)

    def manifest(self):
        return json.loads((self.root / "run_manifest.json").read_text())

    def test_all_unique_pairs_and_cartesian_grid(self):
        self.prepare()
        m = self.manifest()
        self.assertEqual(m["metadata_entries"], 3)
        self.assertEqual(m["evaluated_pairs"], 2)
        self.assertEqual(m["configuration"]["CFG_VALUES"], ["2", "3"])
        self.assertEqual(m["configuration"]["SLAT_CFG_VALUES"], ["1.5", "2"])
        self.assertNotIn("MorphFlow/outputs/ignored.py", m["sources"])
        self.assertEqual((self.root / "source_snapshot/TRELLIS/trellis/resource.bin").read_bytes(), b"needed non-Python resource")

    def test_cfg_lists_reject_equivalent_duplicates_and_nonfinite(self):
        self.assertEqual(ateneo.cfg_values("2, 3 5.0"), ["2", "3", "5"])
        for value in ("2 2.0", "nan", "inf", "-1", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ateneo.cfg_values(value)

    def test_native_configuration_uses_same_grid_without_container_paths(self):
        environment = dict(self.env, EXECUTION_BACKEND="native", DATASET_INPUT_ROOT=str(self.base / "original-images"))
        del environment["SIF"]
        del environment["PROTOCOL_PYTHON"]
        configuration = ateneo.requested_configuration(environment)
        self.assertEqual(configuration["EXECUTION_BACKEND"], "native")
        self.assertEqual(configuration["DATASET_INPUT_ROOT"], str(self.base / "original-images"))
        self.assertEqual(configuration["N"], 0)
        self.assertEqual(configuration["FID_AGGREGATION"], "both")
        self.assertNotIn("SIF", configuration)
        self.assertNotIn("PROTOCOL_PYTHON", configuration)
        # The default must preserve the existing container manifest shape.
        self.assertNotIn("EXECUTION_BACKEND", ateneo.requested_configuration(self.env))

    def test_native_dataset_provenance_is_frozen_and_checked_on_resume(self):
        self.env["EXECUTION_BACKEND"] = "native"
        del self.env["SIF"]
        del self.env["PROTOCOL_PYTHON"]
        provenance = self.data / "dataset_provenance.json"
        provenance.write_text('{"endpoint_seed":42}\n')
        self.prepare()
        self.assertEqual((self.root / "input_snapshot/dataset_provenance.json").read_bytes(), provenance.read_bytes())
        provenance.write_text('{"endpoint_seed":43}\n')
        with self.assertRaisesRegex(ValueError, "Code or test metadata changed"):
            self.prepare()

    def test_fixed_protocol_counts_cannot_silently_drift(self):
        for key, value in (("K", "7"), ("NUM_VIEWS", "32"), ("FID_NUM_VIEWS", "64"), ("FID_AGGREGATION", "concatenated")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "paper protocol"):
                ateneo.requested_configuration(dict(self.env, **{key: value}))

    def test_requested_count_must_fit_test_set(self):
        self.env["N"] = "3"
        with self.assertRaisesRegex(ValueError, "metadata has 2"):
            self.prepare()

    def test_snapshots_pin_weights_despite_training_updates(self):
        self.prepare()
        before = (self.root / "checkpoints/ss/checkpoint.pt").read_bytes()
        Path(self.env["CHECKPOINT_PATH"]).write_bytes(b"new training best")
        self.prepare()
        self.assertEqual((self.root / "checkpoints/ss/checkpoint.pt").read_bytes(), before)

    def test_corrupt_checkpoint_snapshot_is_rejected(self):
        self.prepare()
        (self.root / "checkpoints/ss/checkpoint.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checkpoint snapshot changed"):
            self.prepare()

    def test_changed_configuration_is_rejected(self):
        self.prepare()
        self.env["CFG_VALUES"] = "2 5"
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            self.prepare()

    def test_glb_export_defaults_to_disabled_and_accepts_only_boolean_flags(self):
        self.assertEqual(ateneo.requested_configuration(self.env)['EXPORT_GLB'], 0)
        self.assertEqual(ateneo.requested_configuration(dict(self.env, EXPORT_GLB='1'))['EXPORT_GLB'], 1)
        self.assertEqual(ateneo.requested_configuration(dict(self.env, EXPORT_GLB='0'))['EXPORT_GLB'], 0)
        for value in ('2', '-1', 'true', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'EXPORT_GLB'):
                ateneo.requested_configuration(dict(self.env, EXPORT_GLB=value))

    def test_glb_export_policy_is_frozen_on_resume(self):
        self.prepare()
        manifest = self.manifest()
        self.assertEqual(manifest['configuration']['EXPORT_GLB'], 0)
        self.assertEqual(manifest['asset_export'], {
            'glb_enabled': False,
            'raw_mesh_and_gaussian': True,
            'glb_can_be_baked_later': True,
        })
        self.env['EXPORT_GLB'] = '1'
        with self.assertRaisesRegex(ValueError, 'Configuration changed'):
            self.prepare()

    def test_live_or_frozen_source_change_is_rejected(self):
        self.prepare()
        frozen = self.root / "source_snapshot/MorphFlow/models/model.py"
        frozen.write_text("corrupt\n")
        with self.assertRaisesRegex(ValueError, "Source snapshot changed"):
            self.prepare()
        frozen.write_text("VERSION = 1\n")
        (self.project / "models/model.py").write_text("VERSION = 2\n")
        with self.assertRaisesRegex(ValueError, "Code or test metadata changed"):
            self.prepare()

    def test_frozen_metadata_and_split_are_checked_on_resume(self):
        self.prepare()
        metadata = self.root / "input_snapshot/metadata_test.json"
        original = metadata.read_bytes()
        metadata.write_bytes(b"[]")
        with self.assertRaisesRegex(ValueError, "Frozen test metadata changed"):
            self.prepare()
        metadata.write_bytes(original)
        (self.root / "input_snapshot/split_assets.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Frozen split changed"):
            self.prepare()

    def test_layout_links_remain_valid_after_moving_run_tree(self):
        self.prepare()
        shared = self.root / "shared"
        (shared / "plan.json").write_text("{}")
        args = argparse.Namespace(run_root=str(self.root), cfg="2.0", slat_cfg="1.5")
        with contextlib.redirect_stdout(io.StringIO()):
            ateneo.cell(args)
        cell = self.root / "cfg_ss_2_slat_1.5"
        for path in (cell / "plan.json", cell / "morphany3d", cell / "metrics/interp3d"):
            self.assertFalse(Path(os.readlink(str(path))).is_absolute())
            self.assertTrue(path.exists())
        moved = self.base / "moved"
        self.root.rename(moved)
        self.assertEqual((moved / "cfg_ss_2_slat_1.5/plan.json").read_text(), "{}")
        self.assertTrue((moved / "cfg_ss_2_slat_1.5/morphany3d").is_dir())

    def test_summary_requires_all_pairs_both_fids_and_complete_grid(self):
        self.prepare()
        args = argparse.Namespace(run_root=str(self.root), require_complete=True)
        with self.assertRaisesRegex(ValueError, "Expected 4 CFG cells"):
            ateneo.summarize(args)
        for ss in ("2", "3"):
            for slat in ("1.5", "2"):
                cell = self.root / ("cfg_ss_{}_slat_{}".format(ss, slat))
                cell.mkdir()
                aggregate = {k: {"mean": 1.0} for k in ("fid_endpoint_manifold", "fid_endpoint_manifold_mean_per_state", "ppl_full_normalized_reference_endpoints", "pdv_full")}
                methods = {m: {"num_pairs": 2, "missing_pairs": [], "aggregate_across_pairs": aggregate} for m in ("morphflow", "morphany3d", "interp3d")}
                ateneo.write_json(cell / "final_metrics.json", {"num_pairs": 2, "methods": methods})
                (cell / "final_metrics.csv").write_text("method,metric,mean\nmorphflow,fid_endpoint_manifold_mean_per_state,1\n")
        with contextlib.redirect_stdout(io.StringIO()):
            ateneo.summarize(args)
        self.assertEqual(json.loads((self.root / "cfg_grid_metrics.json").read_text())["completed_configurations"], 4)
        bad = self.root / "cfg_ss_2_slat_1.5/final_metrics.json"
        data = json.loads(bad.read_text())
        data["methods"]["interp3d"]["num_pairs"] = 1
        ateneo.write_json(bad, data)
        with self.assertRaisesRegex(ValueError, "Incomplete method"):
            ateneo.summarize(args)

    def test_dense_layout_and_summary_keep_the_paper_protocol_separate(self):
        self.prepare()
        manifest = self.manifest()
        self.assertEqual(manifest["configuration"]["K"], "5")
        self.assertEqual(manifest["weighted_dense_protocol"]["num_sequence_frames"], 50)
        dense = self.root / "weighted_50"
        references = dense / "shared/references_raw"
        self.assertTrue(references.is_symlink())
        self.assertEqual(references.resolve(), self.root / "shared/references_raw")
        (dense / "shared/plan.json").write_text("{}")
        alphas = [i/49 for i in range(48, 0, -1)]
        for ss in ("2", "3"):
            for slat in ("1.5", "2"):
                args = argparse.Namespace(run_root=str(self.root), cfg=ss, slat_cfg=slat, evaluation="weighted50")
                with contextlib.redirect_stdout(io.StringIO()):
                    ateneo.cell(args)
                directory = dense / ("cfg_ss_{}_slat_{}".format(ss, slat))
                self.assertEqual((directory / "plan.json").resolve(), dense / "shared/plan.json")
                aggregate = {k: {"mean": .2} for k in ("weighted_endpoint_lpips_mean", "weighted_endpoint_lpips_auc", "weighted_endpoint_lpips_full_mean", "weighted_endpoint_lpips_full_auc")}
                methods = {m: {"num_pairs": 2, "missing_pairs": [], "aggregate_across_pairs": aggregate} for m in ("morphflow", "morphany3d", "interp3d")}
                ateneo.write_json(directory / "final_metrics.json", {"num_pairs": 2, "methods": methods,
                    "metric_mode": "weighted-endpoints-only", "k_intermediate": 48, "alphas": alphas})
                (directory / "final_metrics.csv").write_text("method,metric,mean\nmorphflow,weighted_endpoint_lpips_full_mean,.2\n")
        args = argparse.Namespace(run_root=str(self.root), require_complete=True, evaluation="weighted50")
        with contextlib.redirect_stdout(io.StringIO()):
            ateneo.summarize(args)
        self.assertEqual(json.loads((dense / "cfg_grid_metrics.json").read_text())["completed_configurations"], 4)
        self.assertFalse((self.root / "cfg_grid_metrics.json").exists())
        path = dense / "cfg_ss_2_slat_1.5/final_metrics.json"
        broken = json.loads(path.read_text())
        broken["k_intermediate"] = 5
        ateneo.write_json(path, broken)
        with self.assertRaisesRegex(ValueError, "48-intermediate"):
            ateneo.summarize(args)

    def test_dense_disabled_cannot_create_dense_cfg_cells(self):
        self.env["WEIGHTED_DENSE"] = "0"
        self.prepare()
        self.assertFalse((self.root / "weighted_50").exists())
        args = argparse.Namespace(run_root=str(self.root), cfg="2", slat_cfg="1.5", evaluation="weighted50")
        with self.assertRaisesRegex(ValueError, "not enabled"):
            ateneo.cell(args)
        self.env["WEIGHTED_DENSE"] = "2"
        with self.assertRaisesRegex(ValueError, "WEIGHTED_DENSE"):
            ateneo.requested_configuration(self.env)


if __name__ == "__main__":
    unittest.main()
