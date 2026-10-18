"""CPU regression checks for shared outputs and safe benchmark resumption.

Run with the configured Python: python -m unittest discover -s slurm -p test_threeway_leonardo.py
"""

from contextlib import redirect_stdout
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "threeway_leonardo", Path(__file__).with_name("threeway_leonardo.py")
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class SharedOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "shared").mkdir()
        (self.root / "shared" / "plan.json").write_text('{"pairs": []}\n')

    def make_cell(self, cfg, slat_cfg):
        with redirect_stdout(io.StringIO()):
            helper.cell(argparse.Namespace(run_root=str(self.root), cfg=cfg, slat_cfg=slat_cfg))
        return self.root / f"cfg_ss_{cfg}_slat_{slat_cfg}"

    def test_cfg_cells_share_baselines_but_preserve_distinct_morphflow_outputs(self):
        first = self.make_cell("2", "1.5")
        second = self.make_cell("5", "2.5")
        for cell, prediction in ((first, "prediction at CFG 2"), (second, "prediction at CFG 5")):
            (cell / "morphflow").mkdir()
            (cell / "morphflow" / "result.txt").write_text(prediction)
        baseline = first / "morphany3d" / "pair_0000"
        baseline.mkdir()
        (baseline / "result.txt").write_text("one shared baseline")
        (first / "metrics" / "morphany3d" / "metrics_batch.json").write_text('{"num_pairs": 1}')

        self.assertEqual((second / "morphany3d" / "pair_0000" / "result.txt").read_text(), "one shared baseline")
        self.assertEqual((second / "metrics" / "morphany3d" / "metrics_batch.json").read_text(), '{"num_pairs": 1}')
        self.assertNotEqual((first / "morphflow" / "result.txt").read_text(), (second / "morphflow" / "result.txt").read_text())
        for name in ("references_raw", "references_textured", "morphany3d", "interp3d", "plan.json"):
            self.assertEqual((first / name).resolve(), (second / name).resolve())

    def test_resuming_layout_keeps_existing_predictions_and_metrics(self):
        cell = self.make_cell("3", "2")
        (cell / "morphflow").mkdir()
        prediction = cell / "morphflow" / "result.txt"
        prediction.write_text("completed prediction")
        metrics = cell / "metrics" / "interp3d" / "metrics_batch.json"
        metrics.write_text("completed shared metrics")
        original_inode = metrics.stat().st_ino

        self.make_cell("3", "2")

        self.assertEqual(prediction.read_text(), "completed prediction")
        self.assertEqual(metrics.read_text(), "completed shared metrics")
        self.assertEqual(metrics.stat().st_ino, original_inode)

    def test_reused_cell_refuses_unrelated_baseline_without_replacing_it(self):
        cell = self.make_cell("2", "2")
        link = cell / "interp3d"
        link.unlink()
        unrelated = self.root / "other_run_baseline"
        unrelated.mkdir()
        marker = unrelated / "important_result.txt"
        marker.write_text("preserve this result")
        link.symlink_to(unrelated, target_is_directory=True)

        with self.assertRaises(ValueError):
            self.make_cell("2", "2")

        self.assertEqual(link.resolve(), unrelated)
        self.assertEqual(marker.read_text(), "preserve this result")


class ResumeProvenanceTests(unittest.TestCase):
    """Exercise input mutation with real fixture files and no model loading."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        for name in helper.SOURCE_FILES:
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# original {name}\n")
        data = self.root / "dataset"
        data.mkdir()
        self.metadata = data / "metadata_test.json"
        self.metadata.write_text('[{"pair": 1}]\n')
        (data / "split_assets.json").write_text('{"test": []}\n')
        self.environment = {"PROJECT_DIR": str(self.project), "ROOT_DIR": str(data),
                            "METADATA": "metadata_test.json", "RUN_ROOT": str(self.root / "run"),
                            "CHECKPOINT_PATH": str(self.root / "ss.pt"),
                            "SLAT_CHECKPOINT_PATH": str(self.root / "slat.pt")}
        for key in helper.REPOSITORIES[:-1]:
            path = self.root / key
            path.mkdir()
            self.environment[key] = str(path)
        self.diffs = {}
        self.revisions = {}
        self.untracked = {}
        environment_patch = mock.patch.dict(os.environ, self.environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        git_patch = mock.patch.object(helper, "git_output", side_effect=self.git_output)
        git_patch.start()
        self.addCleanup(git_patch.stop)
        checkpoint_patch = mock.patch.object(helper, "checkpoint_metadata", side_effect=lambda path: {"path": path, "epoch": 13})
        checkpoint_patch.start()
        self.addCleanup(checkpoint_patch.stop)
        self.args = argparse.Namespace(run_root=self.environment["RUN_ROOT"])

    def git_output(self, repository, *arguments):
        repository = str(repository)
        if arguments[0] == "rev-parse":
            return self.revisions.get(repository, "0123456789abcdef").encode() + b"\n"
        if arguments[0] == "diff":
            return self.diffs.get(repository, b"")
        if arguments[0] == "ls-files" and "--others" in arguments:
            return b"\0".join(name.encode() for name in self.untracked.get(repository, []))
        if arguments[0] == "ls-files":
            return b""
        raise AssertionError(arguments)

    def initialize(self):
        with redirect_stdout(io.StringIO()):
            helper.initialize(self.args)

    @property
    def manifest(self):
        return Path(self.args.run_root) / "run_manifest.json"

    def test_snapshot_contains_launcher_dataset_and_all_repository_diffs(self):
        baseline = self.environment["INTERP_REPO"]
        self.diffs[baseline] = b"diff --git a/pipeline.py b/pipeline.py\n"
        self.initialize()
        manifest = json.loads(self.manifest.read_text())
        snapshot = self.manifest.parent / "source_snapshot"
        for name in helper.SOURCE_FILES:
            self.assertEqual((snapshot / name).read_bytes(), (self.project / name).read_bytes())
            self.assertEqual(manifest["source_sha256"][name], helper.sha256_file(snapshot / name))
        self.assertEqual((snapshot / "dataset_inputs" / "metadata.json").read_bytes(), self.metadata.read_bytes())
        for key in helper.REPOSITORIES:
            self.assertEqual((snapshot / "repositories" / key / "working_tree.patch").read_bytes(), self.diffs.get(self.environment[key], b""))

    def test_unchanged_resume_preserves_original_manifest_and_outputs(self):
        self.initialize()
        original = self.manifest.read_bytes()
        completed = self.manifest.parent / "completed_prediction.txt"
        completed.write_text("keep completed output")
        self.initialize()
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(completed.read_text(), "keep completed output")

    def test_changed_launcher_refuses_resume_before_reusing_results(self):
        self.initialize()
        original = self.manifest.read_bytes()
        (self.project / helper.SOURCE_FILES[-1]).write_text("# changed launcher\n")
        with self.assertRaisesRegex(ValueError, "new RUN_ROOT"):
            self.initialize()
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_changed_metadata_content_refuses_resume(self):
        self.initialize()
        self.metadata.write_text('[{"pair": 2}]\n')
        with self.assertRaisesRegex(ValueError, "dataset_inputs.*new RUN_ROOT"):
            self.initialize()

    def test_changed_baseline_dirty_diff_refuses_resume_at_same_revision(self):
        self.initialize()
        self.diffs[self.environment["MORPHANY_REPO"]] = b"changed tracked attention implementation\n"
        with self.assertRaisesRegex(ValueError, "repositories.*new RUN_ROOT"):
            self.initialize()

    def test_changed_untracked_baseline_source_refuses_resume(self):
        baseline = self.environment["INTERP_REPO"]
        source = Path(baseline) / "new_attention.py"
        source.write_text("alpha = 5\n")
        self.untracked[baseline] = [source.name]
        self.initialize()
        source.write_text("alpha = 6\n")
        with self.assertRaisesRegex(ValueError, "repositories.*new RUN_ROOT"):
            self.initialize()

    def test_changed_baseline_revision_refuses_resume(self):
        self.initialize()
        self.revisions[self.environment["TRELLIS_REPO"]] = "different-commit"
        with self.assertRaisesRegex(ValueError, "repositories.*new RUN_ROOT"):
            self.initialize()

    def test_source_change_during_initialization_leaves_no_partial_manifest(self):
        snapshot_repository = helper.snapshot_repository

        def change_input(repository, destination):
            snapshot_repository(repository, destination)
            self.metadata.write_text('[{"pair": 9}]\n')

        with mock.patch.object(helper, "snapshot_repository", side_effect=change_input):
            with self.assertRaisesRegex(ValueError, "changed during initialization"):
                self.initialize()
        self.assertFalse(self.manifest.exists())
        self.assertFalse((self.manifest.parent / "source_snapshot").exists())
        self.assertFalse(list(self.manifest.parent.glob(".source_snapshot.*")))


if __name__ == "__main__":
    unittest.main()
