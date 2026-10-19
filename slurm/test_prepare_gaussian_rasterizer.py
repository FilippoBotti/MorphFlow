"""Focused CPU tests for Gaussian overlay ABI/API checks and atomic preparation."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("prepare_gaussian_rasterizer", Path(__file__).with_name("prepare_gaussian_rasterizer.py"))
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gaussian_prepare_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.overlay = self.root / "protocol-python/gaussian"
        self.source_root = self.root / "source"
        self.package = self.root / "cached-package"
        self.package.mkdir()
        (self.package / "setup.py").write_text("# pinned fixture\n")
        self.runtime = dict(python=[3, 10], torch="2.4.0+cu118", cuda="11.8",
                            cxx11_abi=False, machine="x86_64", architectures="8.9")
        self.result = dict(runtime=self.runtime, module_file="/fake/module.py", extension_file="/fake/_C.so")

    def metadata(self, target, runtime=None):
        target.mkdir(parents=True, exist_ok=True)
        (target / helper.METADATA_NAME).write_text(json.dumps(dict(
            runtime=self.runtime if runtime is None else runtime,
            source_commit=helper.SOURCE_COMMIT)))

    def test_missing_extension_check_is_offline_and_actionable(self):
        with mock.patch.object(helper, "probe", side_effect=helper.PreparationError("No module named diff_gaussian_rasterization")), \
                mock.patch.object(helper, "build_overlay") as build:
            with self.assertRaisesRegex(helper.PreparationError, "--prepare before submitting"):
                helper.ensure_rasterizer(self.overlay, self.source_root)
            build.assert_not_called()
        self.assertFalse(self.overlay.parent.exists())
        self.assertFalse(self.source_root.exists())

    def test_compatible_existing_environment_skips_install(self):
        with mock.patch.object(helper, "probe", return_value=self.result) as probe, \
                mock.patch.object(helper, "build_overlay") as build:
            helper.ensure_rasterizer(self.overlay, self.source_root, prepare=True)
            probe.assert_called_once_with()
            build.assert_not_called()
        self.assertFalse(self.overlay.parent.exists())

    def test_overlay_runtime_mismatch_is_not_silently_accepted(self):
        self.metadata(self.overlay, dict(self.runtime, torch="2.1.0"))
        with mock.patch.object(helper, "probe", return_value=self.result), \
                mock.patch.object(helper, "build_overlay") as build:
            with self.assertRaisesRegex(helper.PreparationError, "ABI/runtime mismatch"):
                helper.ensure_rasterizer(self.overlay, self.source_root)
            build.assert_not_called()

    def test_compatible_overlay_skips_source_download(self):
        self.metadata(self.overlay)
        with mock.patch.object(helper, "probe", return_value=self.result) as probe, \
                mock.patch.object(helper, "source_package") as source:
            helper.ensure_rasterizer(self.overlay, self.source_root, prepare=True)
            probe.assert_called_once_with(self.overlay)
            source.assert_not_called()

    def fake_build_command(self, arguments, **kwargs):
        if "wheel" in arguments:
            self.assertIn("--no-deps", arguments)
            self.assertIn("--no-build-isolation", arguments)
            self.assertEqual(kwargs["env"]["TORCH_CUDA_ARCH_LIST"], getattr(self, "expected_architectures", "8.9"))
            self.assertEqual(kwargs["env"]["MAX_JOBS"], getattr(self, "expected_jobs", "2"))
            package = Path(arguments[-1])
            self.assertNotEqual(package, self.package)
            self.assertEqual((package / "setup.py").read_text(), "# pinned fixture\n")
            for pattern in ("build", "dist", "*.egg-info", "__pycache__"):
                self.assertEqual(list(package.rglob(pattern)), [])
            wheels = Path(arguments[arguments.index("--wheel-dir") + 1])
            (wheels / "diff_gaussian_rasterization-test.whl").write_bytes(b"test wheel")
        elif "install" in arguments:
            self.assertIn("--no-deps", arguments)
            self.assertIn("--no-compile", arguments)
            self.assertIn("--no-index", arguments)
            destination = Path(arguments[arguments.index("--target") + 1])
            destination.mkdir()
            (destination / "test-extension.so").write_bytes(b"compiled test payload")
        else:
            self.fail("Unexpected command: " + repr(arguments))
        return ""

    def test_staged_build_failure_preserves_published_overlay_and_cleans_stage(self):
        old_version = self.overlay.parent / ".gaussian-old"
        self.metadata(old_version)
        self.overlay.symlink_to(old_version.name, target_is_directory=True)
        self.source_root.mkdir()
        with mock.patch.object(helper, "source_package", return_value=self.package), \
                mock.patch.object(helper, "configure_build_toolchain", side_effect=lambda env, workspace, **kwargs: dict(env)), \
                mock.patch.object(helper, "run_command", side_effect=self.fake_build_command), \
                mock.patch.object(helper, "probe", side_effect=helper.PreparationError("undefined symbol: torch ABI")):
            with self.assertRaisesRegex(helper.PreparationError, "torch ABI"):
                helper.build_overlay(self.overlay, self.source_root)
        self.assertEqual(self.overlay.resolve(), old_version)
        self.assertTrue((old_version / helper.METADATA_NAME).exists())
        self.assertEqual(sorted(path.name for path in self.overlay.parent.iterdir()), [".gaussian-old", "gaussian"])
        self.assertEqual(list(self.source_root.iterdir()), [])

    def test_build_publishes_relative_symlink_and_pinned_metadata(self):
        self.source_root.mkdir()
        for name in ("build", "dist", "old.egg-info", "__pycache__"):
            directory = self.package / name
            directory.mkdir()
            (directory / "stale.o").write_bytes(b"wrong old environment")
        with mock.patch.object(helper, "source_package", return_value=self.package), \
                mock.patch.object(helper, "configure_build_toolchain", side_effect=lambda env, workspace, **kwargs: dict(env)), \
                mock.patch.object(helper, "run_command", side_effect=self.fake_build_command), \
                mock.patch.object(helper, "probe", return_value=self.result):
            helper.build_overlay(self.overlay, self.source_root)
        self.assertTrue(self.overlay.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(self.overlay)))
        metadata = json.loads((self.overlay / helper.METADATA_NAME).read_text())
        self.assertEqual(metadata["runtime"], self.runtime)
        self.assertEqual(metadata["source_commit"], helper.SOURCE_COMMIT)
        self.assertEqual(metadata["source_url"], helper.SOURCE_URL)
        self.assertEqual(list(self.source_root.iterdir()), [])
        self.assertTrue((self.package / "build/stale.o").is_file())

    def test_atomic_replacement_keeps_previous_version_available(self):
        old_version = self.overlay.parent / ".gaussian-old"
        self.metadata(old_version)
        self.overlay.symlink_to(old_version.name, target_is_directory=True)
        self.source_root.mkdir()
        with mock.patch.object(helper, "source_package", return_value=self.package), \
                mock.patch.object(helper, "configure_build_toolchain", side_effect=lambda env, workspace, **kwargs: dict(env)), \
                mock.patch.object(helper, "run_command", side_effect=self.fake_build_command), \
                mock.patch.object(helper, "probe", return_value=self.result):
            helper.build_overlay(self.overlay, self.source_root)
        self.assertNotEqual(self.overlay.resolve(), old_version)
        self.assertTrue(old_version.is_dir())

    def test_build_respects_explicit_architecture_and_parallelism(self):
        self.source_root.mkdir()
        self.expected_architectures, self.expected_jobs = "8.9+PTX", "3"
        overridden = dict(self.result, runtime=dict(self.runtime, architectures="8.9+PTX"))
        with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.9+PTX", "MAX_JOBS": "3"}), \
                mock.patch.object(helper, "source_package", return_value=self.package), \
                mock.patch.object(helper, "configure_build_toolchain", side_effect=lambda env, workspace, **kwargs: dict(env)), \
                mock.patch.object(helper, "run_command", side_effect=self.fake_build_command), \
                mock.patch.object(helper, "probe", return_value=overridden):
            helper.build_overlay(self.overlay, self.source_root)
        metadata = json.loads((self.overlay / helper.METADATA_NAME).read_text())
        self.assertEqual(metadata["runtime"]["architectures"], "8.9+PTX")

    def fake_python_packages(self, correct_api):
        packages = self.root / "python-fixture"
        torch = packages / "torch"
        torch.mkdir(parents=True)
        (torch / "__init__.py").write_text(
            "from types import SimpleNamespace\n__version__='2.4.0+cu118'\n"
            "version=SimpleNamespace(cuda='11.8')\n_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)\n")
        rasterizer = self.overlay / "diff_gaussian_rasterization"
        rasterizer.mkdir(parents=True)
        fields = "kernel_size subpixel_offset" if correct_api else "image_height image_width"
        (rasterizer / "__init__.py").write_text(
            "from collections import namedtuple\n"
            "GaussianRasterizationSettings=namedtuple('GaussianRasterizationSettings', {!r})\n"
            "class GaussianRasterizer: pass\n".format(fields))
        (rasterizer / "_C.py").write_text("import torch\nassert torch.__version__ == '2.4.0+cu118'\n")
        return packages

    def test_real_subprocess_rejects_non_mip_api(self):
        packages = self.fake_python_packages(correct_api=False)
        with mock.patch.dict(os.environ, {"PYTHONPATH": str(packages)}):
            with self.assertRaisesRegex(helper.PreparationError, "requires mip-splatting"):
                helper.probe(self.overlay)

    def test_real_subprocess_imports_torch_then_extension_and_reports_abi(self):
        packages = self.fake_python_packages(correct_api=True)
        with mock.patch.dict(os.environ, {"PYTHONPATH": str(packages), "TORCH_CUDA_ARCH_LIST": "8.9"}):
            result = helper.probe(self.overlay)
        self.assertEqual(result["runtime"]["torch"], "2.4.0+cu118")
        self.assertEqual(result["runtime"]["cuda"], "11.8")
        self.assertIn("kernel_size", result["fields"])
        self.assertIn("subpixel_offset", result["fields"])

    def smoke_fixture(self, invalid_image=False):
        # Exercise smoke control flow with real CPU tensors, redirecting every
        # requested CUDA allocation. No CUDA runtime or physical GPU is used.
        import torch as real_torch

        def cpu_factory(factory):
            def allocate(*args, **kwargs):
                self.assertEqual(kwargs.pop("device"), "cuda")
                return factory(*args, device="cpu", **kwargs)
            return allocate

        image = real_torch.ones((3, 32, 32))
        if invalid_image:
            image[0, 0, 0] = float("nan")
        rasterize = mock.Mock(return_value=(image, real_torch.ones(1)))
        settings = mock.Mock(side_effect=lambda **values: values)
        rasterizer = mock.Mock(return_value=rasterize)
        cuda = SimpleNamespace(is_available=lambda: True, synchronize=mock.Mock())
        class CpuTorch(SimpleNamespace):
            def __getattr__(self, name):
                return getattr(real_torch, name)

        torch = CpuTorch(
            zeros=cpu_factory(real_torch.zeros), eye=cpu_factory(real_torch.eye),
            tensor=cpu_factory(real_torch.tensor), isfinite=real_torch.isfinite,
            no_grad=real_torch.no_grad, float32=real_torch.float32, cuda=cuda,
        )
        module = SimpleNamespace(GaussianRasterizationSettings=settings,
                                 GaussianRasterizer=rasterizer)
        return torch, module, settings, rasterize

    def test_cuda_smoke_uses_mip_fields_and_checks_visible_output(self):
        torch, module, settings, rasterize = self.smoke_fixture()
        with mock.patch.dict(sys.modules, {"torch": torch, "diff_gaussian_rasterization": module}):
            helper.smoke_gpu()
        torch.cuda.synchronize.assert_called_once_with()
        values = settings.call_args.kwargs
        self.assertEqual(values["kernel_size"], .1)
        self.assertEqual(tuple(values["subpixel_offset"].shape), (32, 32, 2))
        self.assertEqual(tuple(rasterize.call_args.kwargs["means3D"].shape), (1, 3))

    def test_cuda_smoke_rejects_nonfinite_image(self):
        torch, module, _, _ = self.smoke_fixture(invalid_image=True)
        with mock.patch.dict(sys.modules, {"torch": torch, "diff_gaussian_rasterization": module}):
            with self.assertRaisesRegex(helper.PreparationError, "invalid image"):
                helper.smoke_gpu()


class ToolchainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gaussian_toolchain_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cuda = self.root / "cuda"
        self.compiler = str(self.cuda / "bin/x86_64-conda-linux-gnu-g++")
        self.nvcc = str(self.cuda / "bin/nvcc")
        self.environment = dict(CUDA_HOME=str(self.cuda), CUDACXX=self.nvcc,
                                CC=str(self.cuda / "bin/x86_64-conda-linux-gnu-gcc"),
                                CXX=self.compiler, TORCH_CUDA_ARCH_LIST="8.9",
                                CONDA_PREFIX="/model-specific/env", PATH="/model-specific/env/bin")
        self.calls = []
        self.header_failures = set()
        self.cuda_failures = set()
        self.available = {self.compiler, self.nvcc, "/usr/bin/g++"}

    def fake_which(self, name, **kwargs):
        return name if name in self.available else None

    def fake_command(self, arguments, **kwargs):
        self.calls.append((arguments, dict(kwargs.get("env", {}))))
        compiler = arguments[0]
        if "-dumpfullversion" in arguments:
            return "11.4.0\n"
        if "-dumpmachine" in arguments:
            return "x86_64-conda-linux-gnu\n"
        if "--version" in arguments:
            return "g++ (fixture GCC) 11.4.0\n"
        if "-E" in arguments:
            self.assertIn("#include <cfloat>", kwargs["input"])
            if compiler in self.header_failures and not kwargs["env"].get("CPLUS_INCLUDE_PATH"):
                raise helper.PreparationError("fatal error: cfloat: No such file or directory")
            return "preprocessed C++\n"
        if compiler == self.nvcc:
            selected = arguments[arguments.index("-ccbin") + 1]
            self.assertNotIn("--allow-unsupported-compiler", arguments)
            self.assertEqual(kwargs["env"]["CC"], selected)
            self.assertEqual(kwargs["env"]["CXX"], selected)
            self.assertEqual(kwargs["env"]["CUDAHOSTCXX"], selected)
            self.assertIn("arch=compute_89,code=sm_89", arguments)
            if selected in self.cuda_failures:
                raise helper.PreparationError("nvcc rejected unsupported host compiler")
            Path(arguments[arguments.index("-o") + 1]).write_bytes(b"real compile fixture object")
            return ""
        self.fail("Unexpected compiler command: " + repr(arguments))

    def configure(self):
        with mock.patch.object(helper.shutil, "which", side_effect=self.fake_which), \
                mock.patch.object(helper, "run_command", side_effect=self.fake_command):
            return helper.configure_build_toolchain(self.environment, self.root / "probe")

    def test_existing_compiler_keeps_model_env_and_torch_uses_verified_gxx(self):
        original = dict(self.environment)
        configured = self.configure()
        self.assertEqual(self.environment, original)
        self.assertEqual(configured["CONDA_PREFIX"], "/model-specific/env")
        self.assertEqual(configured["PATH"], original["PATH"])
        self.assertEqual(configured["CC"], self.compiler)
        self.assertNotIn("CPLUS_INCLUDE_PATH", configured)
        self.assertEqual(sum(args[0] == self.nvcc for args, _ in self.calls), 1)

    def test_missing_cfloat_recovers_only_matching_version_and_target_headers(self):
        self.header_failures.add(self.compiler)
        base = self.cuda / "x86_64-conda-linux-gnu/include/c++/11.4.0"
        base.mkdir(parents=True)
        (base / "cfloat").write_text("// standard C++ header")
        (base / "x86_64-conda-linux-gnu").mkdir()
        (base / "backward").mkdir()
        configured = self.configure()
        self.assertEqual(configured["CC"], self.compiler)
        self.assertEqual(configured["CPLUS_INCLUDE_PATH"].split(os.pathsep),
                         [str(base), str(base / "x86_64-conda-linux-gnu"), str(base / "backward")])
        self.assertTrue(any(args[0] == self.nvcc for args, _ in self.calls))

    def test_missing_headers_selects_system_compiler_only_after_real_nvcc_probe(self):
        self.header_failures.add(self.compiler)
        configured = self.configure()
        self.assertEqual(configured["CC"], "/usr/bin/g++")
        probes = [args for args, _ in self.calls if args[0] == self.nvcc]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0][probes[0].index("-ccbin") + 1], "/usr/bin/g++")

    def test_header_success_does_not_accept_nvcc_failure(self):
        self.cuda_failures.add(self.compiler)
        configured = self.configure()
        self.assertEqual(configured["CC"], "/usr/bin/g++")
        self.assertEqual(sum(args[0] == self.nvcc for args, _ in self.calls), 2)

    def test_different_gcc_version_headers_are_not_used_and_all_fail_actionably(self):
        self.header_failures.add(self.compiler)
        self.cuda_failures.add("/usr/bin/g++")
        wrong = self.cuda / "include/c++/10.3.0"
        wrong.mkdir(parents=True)
        (wrong / "cfloat").write_text("// incompatible GCC header")
        with self.assertRaisesRegex(helper.PreparationError, "GCC/G\\+\\+ and matching libstdc\\+\\+ development headers"):
            self.configure()
        self.assertFalse(any(env.get("CPLUS_INCLUDE_PATH") for _, env in self.calls))

    def test_no_nvcc_object_is_not_success(self):
        real_fake = self.fake_command

        def no_object(arguments, **kwargs):
            result = real_fake(arguments, **kwargs)
            if arguments[0] == self.nvcc:
                Path(arguments[arguments.index("-o") + 1]).unlink()
            return result

        with mock.patch.object(helper.shutil, "which", side_effect=self.fake_which), \
                mock.patch.object(helper, "run_command", side_effect=no_object):
            with self.assertRaisesRegex(helper.PreparationError, "NVCC did not create"):
                helper.configure_build_toolchain(self.environment, self.root / "probe")

    def test_absent_container_gxx_and_missing_headers_prepare_private_compiler(self):
        self.header_failures.add(self.compiler)
        self.available.remove("/usr/bin/g++")
        prefix = self.root / "private-gcc11"
        private_compiler = str(prefix / "bin/x86_64-conda-linux-gnu-g++")
        original = dict(self.environment)

        def prepare(target, environment):
            self.assertEqual(target, prefix)
            self.assertEqual(environment, original)
            self.available.add(private_compiler)
            return private_compiler

        with mock.patch.object(helper.shutil, "which", side_effect=self.fake_which), \
                mock.patch.object(helper, "run_command", side_effect=self.fake_command), \
                mock.patch.object(helper, "prepare_private_compiler", side_effect=prepare) as installer:
            result = helper.configure_build_toolchain(self.environment, self.root / "probe", compiler_root=prefix)
        installer.assert_called_once()
        self.assertEqual(self.environment, original)
        self.assertEqual(result["CONDA_PREFIX"], "/model-specific/env")
        self.assertEqual(result["CC"], private_compiler)
        self.assertTrue(result["PATH"].startswith(str(prefix / "bin") + os.pathsep))
        self.assertNotIn("LD_LIBRARY_PATH", result)
        self.assertTrue(any(args[0] == self.nvcc and args[args.index("-ccbin") + 1] == private_compiler
                            for args, _ in self.calls))


class PrivateCompilerTests(unittest.TestCase):
    def test_conda_installs_only_into_dedicated_prefix_and_retains_model_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "build-tools/gcc11"
            model = root / "model-env"
            model.mkdir()
            marker = model / "keep.txt"
            marker.write_text("unchanged")
            environment = {"CONDA_PREFIX": str(model), "CONDA_EXE": "/opt/conda/bin/conda", "PATH": str(model / "bin")}

            def install(arguments, **kwargs):
                self.assertEqual(arguments[:3], ["/opt/conda/bin/conda", "create", "--yes"])
                self.assertEqual(arguments[arguments.index("--prefix") + 1], str(prefix))
                self.assertIn("gxx_linux-64=11", arguments)
                self.assertIn("sysroot_linux-64=2.17", arguments)
                self.assertEqual(kwargs["env"]["CONDA_PREFIX"], str(model))
                self.assertTrue(kwargs["env"]["CONDA_PKGS_DIRS"].startswith(str(root / "build-tools")))
                (prefix / "bin").mkdir(parents=True)
                (prefix / "bin/x86_64-conda-linux-gnu-g++").write_text("compiler")
                return ""

            with mock.patch.object(helper, "run_command", side_effect=install) as command:
                result = helper.prepare_private_compiler(prefix, environment)
                self.assertEqual(result, str(prefix / "bin/x86_64-conda-linux-gnu-g++"))
                helper.prepare_private_compiler(prefix, environment)
                self.assertEqual(command.call_count, 1)
            self.assertEqual(marker.read_text(), "unchanged")
            self.assertEqual(sorted(path.name for path in model.iterdir()), ["keep.txt"])
            self.assertNotIn("CONDA_PKGS_DIRS", environment)

    def test_incomplete_compiler_prefix_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "gcc11"
            prefix.mkdir()
            marker = prefix / "partial.txt"
            marker.write_text("preserve")
            with mock.patch.object(helper, "run_command") as command:
                with self.assertRaisesRegex(helper.PreparationError, "Incomplete private compiler"):
                    helper.prepare_private_compiler(prefix, {})
                command.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
