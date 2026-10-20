"""CPU-only regression tests for the opt-in repair of a frozen benchmark run."""

import contextlib
import enum
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "resume_threeway_mesh", Path(__file__).with_name("resume_threeway_mesh.py")
)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


INT32_MESSAGE = (
    "/io/build/spconv/build/core_cc/src/cumm/gemm/main/"
    "GemmMainUnitTest_matmul_split_Simt_f32f32f32_1.cu(142)\n"
    "int64_t(a.dim(0)) * int64_t(a.dim(1)) * tv::bit_size(algo_desp.dtype_a) / 8 "
    "< int_max assert faild. your data exceed int32 range. "
    "this will be fixed in cumm + nvrtc (spconv 2.2/2.3)."
)


class ConvAlgo(enum.Enum):
    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2


class FakeSparseConvolution:
    def __init__(self, kernel_size, algo=ConvAlgo.Native):
        self.kernel_size = kernel_size
        self.algo = algo
        self.subm = True
        self.stride = (1, 1, 1)
        self.dilation = (1, 1, 1)
        self.groups = 1
        self.inverse = False
        self.transposed = False

    def forward(self, sparse_input):
        return sparse_input


class FakeTensor:
    """Enough tensor surface for cloning without loading Torch."""

    def __init__(self, value):
        self.value = value
        self.shape = (37, 16)

    def clone(self):
        return FakeTensor(self.value)

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self


def fake_runtime(all_weight_is_krsc=True):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache=Mock(), is_available=lambda: False)
    @contextlib.contextmanager
    def no_grad():
        yield
    torch.no_grad = no_grad
    torch.float32 = "float32"
    torch.int32 = "int32"
    spconv = types.ModuleType("spconv")
    spconv.__path__ = []
    constants = types.ModuleType("spconv.constants")
    constants.ALL_WEIGHT_IS_KRSC = all_weight_is_krsc
    core = types.ModuleType("spconv.core")
    core.ConvAlgo = ConvAlgo
    pytorch = types.ModuleType("spconv.pytorch")
    pytorch.__path__ = []
    pytorch.ConvAlgo = ConvAlgo
    conv = types.ModuleType("spconv.pytorch.conv")
    conv.SparseConvolution = FakeSparseConvolution
    pytorch.conv = conv
    spconv.core = core
    spconv.constants = constants
    spconv.pytorch = pytorch
    return {
        "torch": torch,
        "spconv": spconv,
        "spconv.constants": constants,
        "spconv.core": core,
        "spconv.pytorch": pytorch,
        "spconv.pytorch.conv": conv,
    }


class MeshFallbackTests(unittest.TestCase):
    def setUp(self):
        self.runtime = patch.dict(sys.modules, fake_runtime())
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.feats = FakeTensor("features")
        self.coords = FakeTensor("coordinates")
        self.sparse_cls = object
        self.device = "cuda:0"
        recorder = patch.object(repair, "record_fallback")
        self.record_fallback = recorder.start()
        self.addCleanup(recorder.stop)
        self.block_calls = []

        def block(module, sparse_input, **kwargs):
            self.block_calls.append((module, sparse_input, kwargs))
            return kwargs["original_forward"](sparse_input)

        loader = patch.object(repair, "load_channel_blocker", return_value=block, create=True)
        self.load_channel_blocker = loader.start()
        self.addCleanup(loader.stop)

    def decoder(self, layers):
        return types.SimpleNamespace(modules=lambda: iter(layers))

    def decode(self, original, decoder):
        return repair.decode_with_mesh_fallback(
            original, decoder, self.sparse_cls, self.feats, self.coords, self.device
        )

    def test_error_match_is_specific_to_native_int32_assertion(self):
        self.assertTrue(repair.native_int32_error(RuntimeError(INT32_MESSAGE)))
        for error in (
            RuntimeError("CUDA out of memory"),
            RuntimeError("Expected coords dtype int32"),
            ValueError("Unexpected coords shape: (0,)"),
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            RuntimeError("shape mismatch in cumm gemm"),
        ):
            with self.subTest(error=str(error)):
                self.assertFalse(repair.native_int32_error(error))

    def test_success_uses_original_once_without_changing_algorithms(self):
        layer = FakeSparseConvolution((3, 3, 3))
        decoder = self.decoder([layer])
        result = object()
        original = Mock(return_value=result)
        self.assertIs(self.decode(original, decoder), result)
        original.assert_called_once()
        self.assertEqual(layer.algo, ConvAlgo.Native)

    def test_unrelated_error_is_not_retried_and_preserves_identity(self):
        layer = FakeSparseConvolution((3, 3, 3))
        decoder = self.decoder([layer])
        for error in (RuntimeError("CUDA out of memory"), ValueError("wrong coords shape")):
            with self.subTest(error=str(error)):
                original = Mock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    self.decode(original, decoder)
                self.assertIs(caught.exception, error)
                original.assert_called_once()
                self.assertEqual(layer.algo, ConvAlgo.Native)

    def test_fallback_wraps_native_nonpointwise_forward_without_changing_algorithms(self):
        native = FakeSparseConvolution((3, 3, 3))
        anisotropic = FakeSparseConvolution((1, 1, 3))
        pointwise = FakeSparseConvolution((1, 1, 1))
        existing_implicit = FakeSparseConvolution((3, 3, 3), ConvAlgo.MaskSplitImplicitGemm)
        unrelated = types.SimpleNamespace(kernel_size=(3, 3, 3), algo=ConvAlgo.Native)
        layers = [native, anisotropic, pointwise, existing_implicit, unrelated]
        decoder = self.decoder(layers)
        attempts = []
        wrapped = []
        sparse_inputs = []
        result = object()

        def original(*args, **kwargs):
            attempts.append(tuple(layer.algo for layer in layers))
            wrapped.append(tuple("forward" in vars(layer) for layer in layers))
            # Each invocation reconstructs sparse state instead of reusing failed caches.
            sparse_inputs.append(object())
            if len(attempts) == 1:
                raise RuntimeError(INT32_MESSAGE)
            self.assertIs(native.forward(sparse_inputs[-1]), sparse_inputs[-1])
            self.assertIs(anisotropic.forward(sparse_inputs[-1]), sparse_inputs[-1])
            return result

        self.assertIs(self.decode(original, decoder), result)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], (
            ConvAlgo.Native, ConvAlgo.Native, ConvAlgo.Native,
            ConvAlgo.MaskSplitImplicitGemm, ConvAlgo.Native,
        ))
        self.assertEqual(attempts[1], attempts[0])
        self.assertEqual(wrapped, [(False,) * 5, (True, True, False, False, False)])
        self.assertEqual([item[0] for item in self.block_calls], [native, anisotropic])
        self.assertTrue(all(call[2]["original_forward"].__self__ is call[0]
                            for call in self.block_calls))
        self.assertEqual(tuple(layer.algo for layer in layers), attempts[0])
        self.assertFalse(any("forward" in vars(layer) for layer in layers))
        self.assertIsNot(sparse_inputs[0], sparse_inputs[1])

    def test_forward_restores_when_fallback_also_fails(self):
        layer = FakeSparseConvolution((3, 3, 3))
        second_error = RuntimeError("second decoding failure")
        original = Mock(side_effect=[RuntimeError(INT32_MESSAGE), second_error])
        with self.assertRaises(RuntimeError) as caught:
            self.decode(original, self.decoder([layer]))
        self.assertIs(caught.exception, second_error)
        self.assertEqual(original.call_count, 2)
        self.assertEqual(layer.algo, ConvAlgo.Native)
        self.assertNotIn("forward", vars(layer))

    def test_existing_instance_forward_restores_by_identity(self):
        layer = FakeSparseConvolution((3, 3, 3))
        saved = Mock(return_value="custom sparse output")
        layer.forward = saved
        calls = []

        def original(*args):
            calls.append(layer.forward)
            if len(calls) == 1:
                raise RuntimeError(INT32_MESSAGE)
            return layer.forward("sparse input")

        self.assertEqual(self.decode(original, self.decoder([layer])), "custom sparse output")
        self.assertIs(layer.forward, saved)
        self.assertIsNot(calls[1], saved)
        saved.assert_called_once_with("sparse input")

    def test_incompatible_weight_layout_refuses_retry(self):
        layer = FakeSparseConvolution((3, 3, 3))
        original = Mock(side_effect=RuntimeError(INT32_MESSAGE))
        with patch.dict(sys.modules, fake_runtime(all_weight_is_krsc=False)):
            with self.assertRaises(RuntimeError):
                self.decode(original, self.decoder([layer]))
        original.assert_called_once()
        self.assertEqual(layer.algo, ConvAlgo.Native)

    def test_no_compatible_layers_does_not_retry(self):
        for layer in (FakeSparseConvolution((1, 1, 1)),
                      FakeSparseConvolution((5, 5, 5)),
                      FakeSparseConvolution((3, 3, 3), ConvAlgo.MaskImplicitGemm)):
            with self.subTest(kernel=layer.kernel_size, algorithm=layer.algo):
                before = layer.algo
                original = Mock(side_effect=RuntimeError(INT32_MESSAGE))
                with self.assertRaises(RuntimeError):
                    self.decode(original, self.decoder([layer]))
                original.assert_called_once()
                self.assertEqual(layer.algo, before)

    def test_worker_patches_the_real_function_globals_from_runpy(self):
        original = Mock(return_value="decoded mesh")
        actual_globals = {"decode_slat_mesh": original, "calls": [],
            "decoder": object(), "sparse_cls": self.sparse_cls,
            "feats": self.feats, "coords": self.coords, "device": self.device}
        exec("def main():\n    calls.append(decode_slat_mesh(decoder, sparse_cls, feats, coords, device))\n",
             actual_globals)
        copied_namespace = dict(actual_globals)
        with patch.object(repair.runpy, "run_path", return_value=copied_namespace) as run_path, \
                patch.object(sys, "argv", ["worker.py", "morphflow"]), \
                patch.object(sys, "path", list(sys.path)):
            repair.worker_main()
        self.assertEqual(actual_globals["calls"], ["decoded mesh"])
        original.assert_called_once_with(actual_globals["decoder"], self.sparse_cls,
                                         self.feats, self.coords, self.device)
        self.assertIs(copied_namespace["decode_slat_mesh"], original)
        self.assertIsNot(actual_globals["decode_slat_mesh"], original)
        run_path.assert_called_once_with(repair.SCRIPT, run_name="threeway_mesh_repair_worker")

    def test_native_worker_uses_frozen_benchmark_override(self):
        original = Mock(return_value="decoded native mesh")
        actual_globals = {"decode_slat_mesh": original, "calls": [],
            "decoder": object(), "sparse_cls": self.sparse_cls,
            "feats": self.feats, "coords": self.coords, "device": self.device}
        exec("def main():\n    calls.append(decode_slat_mesh(decoder, sparse_cls, feats, coords, device))\n",
             actual_globals)
        native_script = "/scratch/run/source_snapshot/MorphFlow/benchmarks/threeway_textured_benchmark.py"
        with patch.object(repair.runpy, "run_path", return_value=dict(actual_globals)) as run_path, \
                patch.dict(repair.os.environ, {"MORPHFLOW_MESH_REPAIR_SCRIPT": native_script}), \
                patch.object(sys, "argv", ["worker.py", "morphflow"]), \
                patch.object(sys, "path", list(sys.path)):
            repair.worker_main()
            self.assertEqual(sys.argv[0], native_script)
            self.assertEqual(sys.path[0], str(Path(native_script).parent))
        self.assertEqual(actual_globals["calls"], ["decoded native mesh"])
        run_path.assert_called_once_with(native_script, run_name="threeway_mesh_repair_worker")


class FrozenResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.project = self.root / "source_snapshot/MorphFlow"
        self.project.mkdir(parents=True)
        self.launcher = self.project / "slurm/benchmark_threeway_interp3d_l40s.slurm"
        self.launcher.parent.mkdir()
        self.launcher.write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            'SCRIPT_IN=/workspace/MorphFlow/benchmarks/threeway_textured_benchmark.py\n'
            'container_exec() {\n'
            '    singularity exec \\\n'
            '        --bind "$SNAPSHOT/MorphFlow:/workspace/MorphFlow:ro" \\\n'
            '        "$SIF" python "$SCRIPT_IN" "$@"\n'
            '}\n'
            'for CELL_NAME in cfg_ss_5_slat_1.5; do\n'
            '    PHASE_NAME="${phase_prefix}${CELL_NAME}"\n'
            '    echo "$PHASE_NAME"\n'
            'done\n'
        )
        helper = self.project / "slurm/threeway_ateneo.py"
        helper.write_text("# Frozen host helper\n")
        benchmark = self.project / "benchmarks/threeway_textured_benchmark.py"
        benchmark.parent.mkdir()
        benchmark.write_text("# Frozen benchmark entry point\n")
        paths = {
            "PROJECT_DIR": str(Path(self.temp.name) / "live-project"),
            "DATA_ROOT_HOST": "/hpc/scratch/user/3d_dataset/morphing_dataset_v4",
            "SOURCE_IMAGES_ROOT_HOST": "/hpc/scratch/user/3d_dataset/flux_outputs",
            "METADATA_HOST": "/hpc/scratch/user/3d_dataset/morphing_dataset_v4/metadata_test.json",
            "CHECKPOINT_PATH": "/hpc/archive/user/checkpoints/ss_best_e17.pt",
            "SLAT_CHECKPOINT_PATH": "/hpc/archive/user/checkpoints/slat_best_e9.pt",
            "ORIGINAL_ENV": "/hpc/home/user/trellis-singularity/env",
            "MORPHANY_ENV": "/hpc/scratch/user/morph-benchmark/morphany3d/env",
            "INTERP_ENV": "/hpc/scratch/user/morph-benchmark/interp3d/env",
            "TRELLIS_REPO_HOST": "/hpc/home/user/trellis-singularity/work/TRELLIS",
            "MORPHANY_REPO": "/hpc/scratch/user/morph-benchmark/morphany3d/work/MorphAny3D",
            "INTERP_REPO": "/hpc/scratch/user/morph-benchmark/interp3d/work/Interp3D",
            "SIF": "/hpc/home/user/containers/trellis-py240-cu118.sif",
            "PROTOCOL_PYTHON": "/hpc/scratch/user/morph-benchmark/protocol-python",
        }
        self.configuration = dict(paths, N=50, SEED=42,
            CFG_VALUES=["1", "2", "3", "5", "7.5"],
            SLAT_CFG_VALUES=["1", "1.5", "2"], WEIGHTED_DENSE=1,
            EXPORT_GLB=0, K="5", STEPS="25", SLAT_STEPS="25", BETA_SHAPE="5",
            NUM_VIEWS="64", FID_NUM_VIEWS="1000", LPIPS_NET="vgg",
            FID_AGGREGATION="both", RENDER_BACKEND="gaussian",
            MIXED_PRECISION="bf16", TEXTURE_SIZE=1024, SIMPLIFY=.95,
            FILL_HOLES=1, SOURCE_IMAGE_FILENAME="")
        sources = {}
        for path in (self.launcher, helper, benchmark):
            sources[path.relative_to(self.root / "source_snapshot").as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoints = {}
        for name in ("ss", "slat"):
            path = self.root / "checkpoints" / name / "checkpoint.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes((name + " pinned checkpoint").encode())
            checkpoints[name] = {"snapshot": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        (self.root / "input_snapshot").mkdir()
        (self.root / "input_snapshot/metadata_test.json").write_text("[]\n")
        manifest = {"schema_version": 1, "configuration": self.configuration,
            "sources": sources, "checkpoints": checkpoints, "evaluated_pairs": 50}
        self.manifest_path = self.root / "run_manifest.json"
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def test_preparation_preserves_frozen_inputs_and_adds_a_separate_repair(self):
        before = {p: p.read_bytes() for p in (self.manifest_path, self.launcher)}
        result = repair.prepare_resume(self.root)
        directory = Path(result["repair_dir"])
        self.assertIn(self.root / "runtime_repairs", directory.parents)
        self.assertTrue((directory / "resume_threeway_mesh.py").is_file())
        self.assertTrue((directory / "mesh_channel_blocks.py").is_file())
        self.assertTrue((directory / "repair.json").is_file())
        launcher = Path(result["launcher"])
        self.assertTrue(launcher.is_file())
        self.assertIn(directory, launcher.parents)
        generated = launcher.read_text()
        self.assertIn("SCRIPT_IN=/opt/morphflow-mesh-repair/resume_threeway_mesh.py", generated)
        marker = '--bind "$SNAPSHOT/MorphFlow:/workspace/MorphFlow:ro"'
        helper_bind_position = generated.index(":/opt/morphflow-mesh-repair:ro")
        self.assertLess(helper_bind_position, generated.index(marker))
        self.assertIn('"$CELL_HOST/final_metrics.json"', generated)
        self.assertIn('"$CELL_HOST/final_metrics.csv"', generated)
        self.assertIn("BASELINES_DONE=1", generated)
        self.assertIn("continue", generated)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_preparation_restores_exact_grid_and_metric_settings(self):
        result = repair.prepare_resume(self.root)
        env = result["environment"]
        expected = {"CFG_VALUES": "1 2 3 5 7.5", "SLAT_CFG_VALUES": "1 1.5 2",
            "N": "50", "SEED": "42", "WEIGHTED_DENSE": "1", "EXPORT_GLB": "0",
            "K": "5", "NUM_VIEWS": "64", "FID_NUM_VIEWS": "1000",
            "FID_AGGREGATION": "both", "MIXED_PRECISION": "bf16"}
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(str(env[name]), value)
        self.assertEqual(Path(env["RUN_ROOT"]), self.root)
        for name in ("ORIGINAL_ENV", "MORPHANY_ENV", "INTERP_ENV", "SIF", "PROTOCOL_PYTHON"):
            self.assertEqual(env[name], self.configuration[name])


class NativeFrozenResumeTests(FrozenResumeTests):
    """Native repair uses pinned files without weakening original verification."""

    def setUp(self):
        super().setUp()
        self.launcher.unlink()
        self.launcher = self.project / "slurm/benchmark_interp3ddata_leonardo.slurm"
        self.launcher.write_bytes(Path(__file__).with_name(self.launcher.name).read_bytes())
        helper = self.project / "slurm/threeway_ateneo.py"
        helper.write_bytes(Path(__file__).with_name("threeway_ateneo.py").read_bytes())
        self.configuration.pop("SIF")
        self.configuration.pop("PROTOCOL_PYTHON")
        data = self.root / "prepared_dataset"
        data.mkdir()
        metadata = data / "metadata_test.json"
        metadata.write_text('[{"src_1": "a", "src_2": "b"}]\n')
        self.configuration.update(EXECUTION_BACKEND="native", N=0,
            DATA_ROOT_HOST=str(data), SOURCE_IMAGES_ROOT_HOST=str(data / "images"),
            DATASET_INPUT_ROOT=str(self.root / "raw_dataset"), METADATA_HOST=str(metadata))
        (self.root / "input_snapshot/metadata_test.json").write_bytes(metadata.read_bytes())
        manifest = json.loads(self.manifest_path.read_text())
        manifest["configuration"] = self.configuration
        manifest["evaluated_pairs"] = 1
        manifest["metadata_entries"] = 1
        manifest["unique_test_pairs"] = 1
        manifest["dataset_inputs"] = {"metadata": hashlib.sha256(metadata.read_bytes()).hexdigest()}
        paths = (self.launcher, helper, self.project / "benchmarks/threeway_textured_benchmark.py")
        manifest["sources"] = {
            p.relative_to(self.root / "source_snapshot").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def test_preparation_preserves_frozen_inputs_and_adds_a_separate_repair(self):
        manifest = json.loads(self.manifest_path.read_text())
        frozen = [self.manifest_path] + [self.root / "source_snapshot" / p for p in manifest["sources"]]
        before = {p: p.read_bytes() for p in frozen}
        result = repair.prepare_resume(self.root)
        directory = Path(result["repair_dir"])
        generated = Path(result["launcher"]).read_text()
        self.assertIn(self.root / "runtime_repairs", directory.parents)
        self.assertTrue((directory / "resume_threeway_mesh.py").is_file())
        self.assertTrue((directory / "mesh_channel_blocks.py").is_file())
        self.assertTrue((directory / "repair.json").is_file())
        self.assertIn('SCRIPT="$MORPHFLOW_MESH_REPAIR_DIR/resume_threeway_mesh.py"', generated)
        self.assertIn("--verify-frozen --run-root", generated)
        self.assertNotIn('"$HELPER" prepare --run-root', generated)
        self.assertNotIn("singularity", generated.lower())
        self.assertNotIn("--bind", generated)
        self.assertNotIn("$SIF", generated)
        self.assertIn('"$cell/final_metrics.json"', generated)
        self.assertIn('"$cell/final_metrics.csv"', generated)
        self.assertIn("baselines_done=1", generated)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_preparation_restores_exact_grid_and_metric_settings(self):
        result = repair.prepare_resume(self.root)
        env = result["environment"]
        for name in ("N", "SEED", "K", "STEPS", "SLAT_STEPS", "NUM_VIEWS", "FID_NUM_VIEWS",
                     "WEIGHTED_DENSE", "EXPORT_GLB", "FID_AGGREGATION", "MIXED_PRECISION",
                     "ORIGINAL_ENV", "MORPHANY_ENV", "INTERP_ENV", "DATA_ROOT_HOST",
                     "SOURCE_IMAGES_ROOT_HOST", "DATASET_INPUT_ROOT"):
            self.assertEqual(str(env[name]), str(self.configuration[name]), name)
        self.assertEqual(env["CFG_VALUES"], "1 2 3 5 7.5")
        self.assertEqual(env["SLAT_CFG_VALUES"], "1 1.5 2")
        frozen = self.root / "source_snapshot"
        expected = {"PROJECT_DIR": frozen / "MorphFlow", "TRELLIS_REPO_HOST": frozen / "TRELLIS",
            "MORPHANY_REPO": frozen / "MorphAny3D", "INTERP_REPO": frozen / "Interp3D",
            "CHECKPOINT_PATH": self.root / "checkpoints/ss/checkpoint.pt",
            "SLAT_CHECKPOINT_PATH": self.root / "checkpoints/slat/checkpoint.pt"}
        for key, value in expected.items():
            self.assertEqual(Path(env[key]), value, key)
        self.assertNotIn("SIF", env)
        self.assertNotIn("PROTOCOL_PYTHON", env)
        self.assertEqual(env["EXECUTION_BACKEND"], "native")
        self.assertEqual(Path(env["MORPHFLOW_MESH_REPAIR_SCRIPT"]),
                         frozen / "MorphFlow/benchmarks/threeway_textured_benchmark.py")
        self.assertEqual(env["MORPHFLOW_MESH_REPAIR_LOG_DIR"], result["repair_dir"])

    def test_frozen_verification_uses_original_helper_and_retains_manifest(self):
        before = self.manifest_path.read_bytes()
        result = repair.prepare_resume(self.root)
        # Additions after the frozen source list was established must not replace
        # that list, and the live checkout need not still match the old run.
        (self.project / "unrelated_new_source.py").write_text("# Untracked later addition\n")
        with patch.dict(repair.os.environ, result["environment"], clear=True):
            repair.verify_frozen_run(self.root)
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_frozen_verification_rejects_configuration_changes(self):
        result = repair.prepare_resume(self.root)
        changed = dict(result["environment"], CFG_VALUES="1 2 3")
        with patch.dict(repair.os.environ, changed, clear=True):
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                repair.verify_frozen_run(self.root)

    def test_frozen_verification_rejects_metadata_changes(self):
        result = repair.prepare_resume(self.root)
        Path(self.configuration["METADATA_HOST"]).write_text('[{"src_1":"a","src_2":"c"}]\n')
        with patch.dict(repair.os.environ, result["environment"], clear=True):
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                repair.verify_frozen_run(self.root)

    def test_frozen_verification_rejects_pinned_checkpoint_changes(self):
        result = repair.prepare_resume(self.root)
        (self.root / "checkpoints/ss/checkpoint.pt").write_bytes(b"Changed pinned checkpoint")
        with patch.dict(repair.os.environ, result["environment"], clear=True):
            with self.assertRaisesRegex(ValueError, "Pinned checkpoint snapshot changed"):
                repair.verify_frozen_run(self.root)

    def test_frozen_verification_rejects_frozen_source_changes(self):
        result = repair.prepare_resume(self.root)
        (self.project / "benchmarks/threeway_textured_benchmark.py").write_text("# Tampered frozen source\n")
        with patch.dict(repair.os.environ, result["environment"], clear=True):
            with self.assertRaisesRegex(ValueError, "Code or test metadata changed|Source snapshot changed"):
                repair.verify_frozen_run(self.root)


if __name__ == "__main__":
    unittest.main()
