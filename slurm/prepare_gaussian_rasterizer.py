#!/usr/bin/env python3
"""Prepare the TRELLIS mip Gaussian rasterizer without modifying Conda envs.

The CUDA compiler and host compiler are supplied by the surrounding container
helper. A GPU is not needed: TORCH_CUDA_ARCH_LIST explicitly selects L40S.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid


SOURCE_URL = "https://github.com/autonomousvision/mip-splatting.git"
SOURCE_COMMIT = "dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
PACKAGE_SUBDIRECTORY = "submodules/diff-gaussian-rasterization"
METADATA_NAME = "gaussian-rasterizer-build.json"
ARCHITECTURES = "8.9"
PROBE_PREFIX = "GAUSSIAN_PROBE_JSON="
PROBE_CODE = r'''
import importlib
import json
import os
import platform
import sys
import torch

module = importlib.import_module("diff_gaussian_rasterization")
extension = importlib.import_module("diff_gaussian_rasterization._C")
fields = set(getattr(module.GaussianRasterizationSettings, "_fields", ()))
required = {"kernel_size", "subpixel_offset"}
if not required.issubset(fields):
    raise RuntimeError("Wrong Gaussian rasterizer fork: TRELLIS requires mip-splatting "
                       "GaussianRasterizationSettings fields kernel_size and subpixel_offset")
if not callable(getattr(module, "GaussianRasterizer", None)):
    raise RuntimeError("GaussianRasterizer is missing or not callable")
runtime = {
    "python": list(sys.version_info[:2]),
    "torch": str(torch.__version__),
    "cuda": torch.version.cuda,
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "machine": platform.machine(),
    "architectures": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9"),
}
print("GAUSSIAN_PROBE_JSON=" + json.dumps({
    "runtime": runtime, "module_file": module.__file__,
    "extension_file": extension.__file__, "fields": sorted(fields),
}), flush=True)
'''


class PreparationError(RuntimeError):
    pass


def smoke_gpu():
    """Render one Gaussian before starting expensive multi-GPU phases."""
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    if not torch.cuda.is_available():
        raise PreparationError("Gaussian CUDA smoke check requires the allocated GPU")
    with torch.no_grad():
        settings = GaussianRasterizationSettings(
            image_height=32, image_width=32, tanfovx=1.0, tanfovy=1.0,
            kernel_size=0.1,
            subpixel_offset=torch.zeros((32, 32, 2), dtype=torch.float32, device="cuda"),
            bg=torch.zeros(3, dtype=torch.float32, device="cuda"), scale_modifier=1.0,
            viewmatrix=torch.eye(4, dtype=torch.float32, device="cuda"),
            projmatrix=torch.eye(4, dtype=torch.float32, device="cuda"), sh_degree=0,
            campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
            prefiltered=False, debug=False,
        )
        image, radii = GaussianRasterizer(raster_settings=settings)(
            means3D=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"),
            means2D=torch.zeros((1, 3), dtype=torch.float32, device="cuda"),
            shs=None,
            colors_precomp=torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float32, device="cuda"),
            opacities=torch.tensor([[0.9]], dtype=torch.float32, device="cuda"),
            scales=torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, device="cuda"),
            rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"),
            cov3D_precomp=None,
        )
        torch.cuda.synchronize()
        if tuple(image.shape) != (3, 32, 32) or not bool(torch.isfinite(image).all()):
            raise PreparationError("Gaussian CUDA smoke render returned invalid image data")
        if not bool(image.abs().sum() > 0) or not bool((radii > 0).any()):
            raise PreparationError("Gaussian CUDA smoke render did not rasterize the visible test Gaussian")
    print("Gaussian CUDA smoke render OK: 32x32, one visible Gaussian", flush=True)


def run_command(arguments, **kwargs):
    """Keep compiler output useful on failure without emitting package internals."""
    result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, **kwargs)
    if result.returncode:
        raise PreparationError("Command failed: {}\n{}".format(
            " ".join(map(str, arguments)), result.stdout[-12000:]))
    return result.stdout


def prepare_private_compiler(prefix, environment):
    """Install a build-only GCC/G++ prefix, without touching any model env."""
    prefix = Path(prefix)
    compiler = prefix / "bin/x86_64-conda-linux-gnu-g++"
    if compiler.is_file():
        return str(compiler)
    if prefix.exists() and any(prefix.iterdir()):
        raise PreparationError("Incomplete private compiler prefix {}; move it aside before preparing again".format(prefix))
    conda = environment.get("CONDA_EXE") or shutil.which("conda", path=environment.get("PATH"))
    if not conda:
        raise PreparationError("No C++ development toolchain and no Conda executable available to prepare one")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache = prefix.parent / ".compiler-conda-pkgs"
    cache.mkdir(exist_ok=True)
    installer_environment = dict(environment, CONDA_PKGS_DIRS=str(cache))
    print("Preparing build-only GCC/G++ 11 with C++ headers in {}; model environments stay unchanged".format(prefix), flush=True)
    # Install at the final path: Conda compiler specs contain absolute prefixes.
    run_command([conda, "create", "--yes", "--prefix", str(prefix),
                 "--override-channels", "--channel", "conda-forge", "--strict-channel-priority",
                 "gcc_linux-64=11", "gxx_linux-64=11", "sysroot_linux-64=2.17"],
                env=installer_environment)
    if not compiler.is_file():
        raise PreparationError("Conda did not provide the expected C++ compiler {}".format(compiler))
    return str(compiler)


def matching_cxx_headers(cuda_home, compiler, environment):
    """Recover only headers matching this compiler's version and target.

    Conda compiler prefixes can be mounted at a different path in Singularity.
    Explicit include directories repair that relocation without changing the
    installed compiler or mixing its headers with another GCC version.
    """
    version = run_command([compiler, "-dumpfullversion"], env=environment, timeout=30).strip()
    target = run_command([compiler, "-dumpmachine"], env=environment, timeout=30).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)*", version) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", target):
        raise PreparationError("Compiler returned invalid GCC version or target")
    compiler_prefix = Path(compiler).parent.parent
    for include_root in dict.fromkeys((cuda_home / "include/c++", cuda_home / target / "include/c++",
                                       compiler_prefix / "include/c++", compiler_prefix / target / "include/c++")):
        if not include_root.is_dir():
            continue
        # GCC often names its include directory with the major version only.
        versions = [version]
        if "." in version:
            versions.append(version.split(".")[0])
        for header_version in versions:
            base = include_root / header_version
            if not (base / "cfloat").is_file():
                continue
            directories = [base]
            for suffix in (target, "backward"):
                if (base / suffix).is_dir():
                    directories.append(base / suffix)
            return directories
    return []


def configure_build_toolchain(environment, workspace, compiler_root=None):
    """Return a tested build environment; never alter the model environment.

    Every candidate must compile both standard C++ headers and a real CUDA
    translation unit. In particular, a system compiler is never accepted merely
    because its executable exists. Neither probe requires a GPU.
    """
    environment = dict(environment)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    cuda_home = Path(environment.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc_name = environment.get("CUDACXX", str(cuda_home / "bin/nvcc"))
    nvcc = shutil.which(nvcc_name, path=environment.get("PATH"))
    if not nvcc:
        raise PreparationError("Missing CUDA compiler {}; check TOOLCHAIN_ROOT_HOST".format(nvcc_name))
    candidates = []
    for name in (environment.get("CXX"),
                 str(cuda_home / "bin/x86_64-conda-linux-gnu-g++"),
                 "/usr/bin/g++"):
        if not name:
            continue
        compiler = shutil.which(name, path=environment.get("PATH"))
        if compiler and compiler not in candidates:
            candidates.append(compiler)
    if not candidates and compiler_root is None:
        raise PreparationError("No usable C++ compiler found; install GCC/G++ development headers in the toolchain")
    match = re.search(r"(?:^|[;\s])(\d+)\.(\d+)(?:\+PTX)?(?:$|[;\s])",
                      environment.get("TORCH_CUDA_ARCH_LIST", ARCHITECTURES))
    if not match:
        raise PreparationError("TORCH_CUDA_ARCH_LIST must specify a numeric CUDA architecture, e.g. 8.9")
    architecture = match[1] + match[2]
    source = workspace / "gaussian-toolchain-probe.cu"
    source.write_text("#include <cfloat>\n#include <vector>\n#include <cuda_runtime.h>\n"
                      "__global__ void probe(float* x) { *x = FLT_EPSILON; }\n")
    errors = []
    for index, compiler in enumerate(candidates):
        candidate = dict(environment)
        # Torch's cpp_extension selects NVCC -ccbin from CC, not CUDAHOSTCXX.
        # A checked g++ is appropriate for all C++ sources in this extension.
        candidate.update(CC=compiler, CXX=compiler, CUDAHOSTCXX=compiler)
        try:
            try:
                run_command([compiler, "-std=c++17", "-E", "-x", "c++", "-"],
                            input="#include <cfloat>\n#include <vector>\n",
                            env=candidate, timeout=30)
            except PreparationError:
                headers = matching_cxx_headers(cuda_home, compiler, candidate)
                if not headers:
                    raise PreparationError("standard C++ headers missing, including <cfloat>; no matching headers found under {}".format(cuda_home))
                previous = candidate.get("CPLUS_INCLUDE_PATH", "")
                candidate["CPLUS_INCLUDE_PATH"] = os.pathsep.join(map(str, headers)) + (os.pathsep + previous if previous else "")
                run_command([compiler, "-std=c++17", "-E", "-x", "c++", "-"],
                            input="#include <cfloat>\n#include <vector>\n",
                            env=candidate, timeout=30)
            target = workspace / ("gaussian-toolchain-probe-{}.o".format(index))
            target.unlink(missing_ok=True)
            run_command([nvcc, "-ccbin", compiler, "-std=c++17", "-gencode",
                         "arch=compute_{0},code=sm_{0}".format(architecture),
                         "-c", str(source), "-o", str(target)], env=candidate, timeout=180)
            if not target.is_file():
                raise PreparationError("NVCC did not create the CUDA probe object")
            version = run_command([compiler, "--version"], env=candidate, timeout=30).splitlines()[0]
            print("Verified Gaussian build compiler: {} ({})".format(compiler, version), flush=True)
            if candidate.get("CPLUS_INCLUDE_PATH") != environment.get("CPLUS_INCLUDE_PATH"):
                print("Recovered matching C++ headers: {}".format(candidate["CPLUS_INCLUDE_PATH"]), flush=True)
            print("CUDA compile probe passed with {} (no GPU required)".format(nvcc), flush=True)
            return candidate
        except (PreparationError, OSError, subprocess.TimeoutExpired) as error:
            diagnostic = "\n".join(str(error).splitlines()[-6:])
            errors.append("{}: {}".format(compiler, diagnostic))
            print("Gaussian compiler candidate rejected: {}".format(errors[-1]), flush=True)
    if compiler_root is not None:
        compiler = prepare_private_compiler(compiler_root, environment)
        private = dict(environment)
        # The model env stays active. Only this build's compiler search paths
        # change; probe(version) later uses the untouched runtime environment.
        for variable in ("GCC_EXEC_PREFIX", "COMPILER_PATH", "CPLUS_INCLUDE_PATH", "CPATH", "C_INCLUDE_PATH", "CONDA_BUILD_SYSROOT"):
            private.pop(variable, None)
        private.update(CC=compiler, CXX=compiler, CUDAHOSTCXX=compiler)
        private["PATH"] = str(Path(compiler).parent) + os.pathsep + private.get("PATH", "")
        sysroot = Path(compiler_root) / "x86_64-conda-linux-gnu/sysroot"
        if sysroot.is_dir():
            private["CONDA_BUILD_SYSROOT"] = str(sysroot)
        return configure_build_toolchain(private, workspace / "private-compiler-probe")
    raise PreparationError("No C++/CUDA compiler pair passed the Gaussian build checks. "
                           "The model environments were not changed. Repair the CUDA toolchain's "
                           "GCC/G++ and matching libstdc++ development headers, or set CXX to a "
                           "CUDA-compatible compiler.\n" + "\n".join(errors))


def probe(overlay=None):
    """Import Torch before _C in a fresh interpreter, detecting ABI/API errors."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("TORCH_CUDA_ARCH_LIST", ARCHITECTURES)
    if overlay is not None:
        previous = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(overlay) + (os.pathsep + previous if previous else "")
    output = run_command([sys.executable, "-c", PROBE_CODE], env=environment,
                         timeout=120)
    for line in reversed(output.splitlines()):
        if line.startswith(PROBE_PREFIX):
            result = json.loads(line[len(PROBE_PREFIX):])
            if overlay is not None:
                root = Path(overlay).resolve()
                for name in ("module_file", "extension_file"):
                    location = Path(result[name]).resolve()
                    if root not in location.parents:
                        raise PreparationError("Gaussian import escaped overlay {}: {}".format(root, location))
            return result
    raise PreparationError("Gaussian import check did not return runtime metadata")


def validate_overlay(overlay):
    result = probe(overlay)
    try:
        metadata = json.loads((overlay / METADATA_NAME).read_text())
    except (OSError, ValueError) as error:
        raise PreparationError("Missing or invalid Gaussian overlay build metadata: {}".format(error))
    if metadata.get("runtime") != result["runtime"]:
        raise PreparationError("Gaussian overlay ABI/runtime mismatch: built for {}; current {}".format(
            metadata.get("runtime"), result["runtime"]))
    if metadata.get("source_commit") != SOURCE_COMMIT:
        raise PreparationError("Gaussian overlay source commit differs from the pinned mip-splatting revision")
    return result


def source_package(source_root):
    """Fetch a pinned, tracked source tree only when preparation is requested."""
    source_root.mkdir(parents=True, exist_ok=True)
    repository = source_root / ("mip-splatting-" + SOURCE_COMMIT)
    if repository.exists():
        revision = run_command(["git", "-C", str(repository), "rev-parse", "HEAD"]).strip()
        if revision != SOURCE_COMMIT:
            raise PreparationError("Unexpected revision in {}; choose a new source root".format(repository))
        run_command(["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--", PACKAGE_SUBDIRECTORY])
    else:
        stage = Path(tempfile.mkdtemp(prefix=".mip-source-", dir=str(source_root)))
        try:
            run_command(["git", "init", str(stage)])
            run_command(["git", "-C", str(stage), "remote", "add", "origin", SOURCE_URL])
            run_command(["git", "-C", str(stage), "fetch", "--depth", "1", "origin", SOURCE_COMMIT])
            run_command(["git", "-C", str(stage), "checkout", "--detach", "FETCH_HEAD"])
            revision = run_command(["git", "-C", str(stage), "rev-parse", "HEAD"]).strip()
            if revision != SOURCE_COMMIT:
                raise PreparationError("Downloaded mip-splatting revision does not match pinned commit")
            stage.rename(repository)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    package = repository / PACKAGE_SUBDIRECTORY
    for required in ("setup.py", "diff_gaussian_rasterization/__init__.py", "third_party/glm/glm/glm.hpp"):
        if not (package / required).is_file():
            raise PreparationError("Incomplete pinned Gaussian source tree: {}".format(package / required))
    return package


def build_overlay(overlay, source_root):
    package = source_package(source_root)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    # The final name is a relative symlink so it remains valid in container binds.
    version = overlay.parent / ("." + overlay.name + "-" + uuid.uuid4().hex)
    workspace = Path(tempfile.mkdtemp(prefix=".gaussian-build-", dir=str(source_root)))
    link = overlay.parent / ("." + overlay.name + ".publish-" + uuid.uuid4().hex)
    published = False
    try:
        wheel_directory = workspace / "wheels"
        wheel_directory.mkdir()
        environment = dict(os.environ)
        environment.setdefault("TORCH_CUDA_ARCH_LIST", ARCHITECTURES)
        environment.setdefault("MAX_JOBS", "2")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment = configure_build_toolchain(environment, workspace, compiler_root=source_root / "gcc11")
        # Cached sources can contain objects from an interrupted or failed build
        # in another model environment. Compile a clean, private copy instead.
        package_copy = workspace / "package"
        shutil.copytree(package, package_copy,
                        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))
        print("Building pinned mip Gaussian rasterizer for CUDA architectures {}; no GPU is required".format(
            environment["TORCH_CUDA_ARCH_LIST"]), flush=True)
        run_command([sys.executable, "-m", "pip", "wheel", "--disable-pip-version-check",
                     "--no-deps", "--no-build-isolation", "--no-cache-dir",
                     "--wheel-dir", str(wheel_directory), str(package_copy)], env=environment)
        wheels = list(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise PreparationError("Expected exactly one Gaussian rasterizer wheel, found {}".format(len(wheels)))
        run_command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-deps", "--no-compile", "--no-index", "--target", str(version),
                     str(wheels[0])], env=environment)
        result = probe(version)
        metadata = dict(runtime=result["runtime"], source_url=SOURCE_URL,
                        source_commit=SOURCE_COMMIT, source_subdirectory=PACKAGE_SUBDIRECTORY)
        (version / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
        validate_overlay(version)
        if overlay.exists() and not overlay.is_symlink():
            raise PreparationError("Cannot replace existing non-symlink Gaussian overlay {}; move it aside or choose a new overlay".format(overlay))
        link.symlink_to(version.name, target_is_directory=True)
        os.replace(str(link), str(overlay))
        published = True
        return result
    finally:
        link.unlink(missing_ok=True)
        if not published and version.exists():
            shutil.rmtree(version)
        shutil.rmtree(workspace)


def ensure_rasterizer(overlay, source_root, prepare=False):
    """Use a compatible existing install, otherwise stage a separate overlay."""
    if os.path.lexists(str(overlay)):
        try:
            result = validate_overlay(overlay)
            print("Gaussian rasterizer ready from overlay: {}".format(result["extension_file"]), flush=True)
            return result
        except (PreparationError, OSError, ValueError) as error:
            if not prepare:
                raise PreparationError("{}\nRun prepare_threeway_interp3d_l40s.sh --prepare to rebuild the Gaussian rasterizer overlay.".format(error))
            print("Existing Gaussian overlay requires rebuilding: {}".format(error), flush=True)
    else:
        try:
            result = probe()
            print("Existing environment already provides the compatible mip Gaussian rasterizer: {}".format(result["extension_file"]), flush=True)
            return result
        except (PreparationError, OSError, ValueError) as error:
            if not prepare:
                raise PreparationError("Gaussian rasterizer is missing or incompatible: {}\nRun prepare_threeway_interp3d_l40s.sh --prepare before submitting the job.".format(error))
            print("Preparing missing mip Gaussian rasterizer in a separate overlay", flush=True)
    return build_overlay(overlay, source_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    for name in ("overlay", "source_root"):
        path = getattr(arguments, name)
        if not path.is_absolute() or path == Path("/"):
            parser.error("--{} must be an absolute path other than /".format(name.replace("_", "-")))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ARCHITECTURES)
    try:
        ensure_rasterizer(arguments.overlay, arguments.source_root, arguments.prepare)
    except (PreparationError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, "ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
