#!/usr/bin/env python3
"""Runtime I/O repair for a relocated frozen native benchmark.

Keeps the frozen evaluator, camera protocol, metric definitions and pixels.
Relocation preserves original cache identities; PNG writes become atomic and
invalid cached PNGs are quarantined before the original renderer regenerates
them. Used by a recorded external runtime launcher, not inside source_snapshot.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import uuid


def cache_identity(path, args, num_views, old_root, new_root):
    path = Path(path)
    resolved = path.resolve()
    try:
        stable_path = Path(old_root) / resolved.relative_to(Path(new_root).resolve())
    except ValueError:
        stable_path = resolved
    stat = path.stat()
    identity = {"path": str(stable_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "views": num_views, "resolution": args.resolution, "radius": args.radius,
                "fov": args.fov_degrees, "cameras": args.camera_distribution,
                "background": "white", "renderer": "TRELLIS GaussianRenderer kernel_size=0.1 use_mip_gaussian=True",
                "feature_backbone": "pytorch-fid TensorFlow-compatible pool3 2048D"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def quarantine(path, reason):
    path = Path(path)
    target = path.with_name(path.name + '.invalid_' + uuid.uuid4().hex[:10])
    path.rename(target)
    print('RENDER CACHE REPAIR: ' + json.dumps(dict(path=str(path), quarantined=str(target), reason=reason)),
          flush=True)


def validate_png_directory(directory, count, resolution):
    from PIL import Image
    directory = Path(directory)
    marker = directory/'identity.json'
    if marker.is_file():
        try:
            json.loads(marker.read_text())
        except (ValueError, UnicodeError) as error:
            quarantine(marker, str(error))
    for i in range(count):
        path = directory/('%03d.png' % i)
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                if image.format != 'PNG' or image.size != (resolution, resolution):
                    raise ValueError('Unexpected PNG format or dimensions')
                image.verify()
        except (OSError, ValueError, SyntaxError) as error:
            quarantine(path, str(error))


@contextlib.contextmanager
def atomic_png_saves():
    from PIL import Image
    original = Image.Image.save

    def save(image, fp, format=None, **params):
        if not isinstance(fp, (str, bytes, os.PathLike)) or Path(fp).suffix.lower() != '.png':
            return original(image, fp, format=format, **params)
        path = Path(fp)
        temporary = path.with_name('.' + path.name + '.partial_' + uuid.uuid4().hex)
        try:
            original(image, temporary, format=format or 'PNG', **params)
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()

    Image.Image.save = save
    try:
        yield
    finally:
        Image.Image.save = original


def repair_prepare(original, namespace):
    def prepare(args, pair_output, src1, src2, frames):
        for key, raw_path in [('src1', src1)] + [(f.key, f.mesh) for f in frames] + [('src2', src2)]:
            directory = pair_output/'images'/key
            if key in ('src1', 'src2') and args.reference_cache_root:
                identity = namespace['cache_identity'](raw_path, args, args.num_views)
                directory = Path(args.reference_cache_root).resolve()/'images'/identity
            if directory.is_dir():
                validate_png_directory(directory, args.num_views, args.resolution)
        with atomic_png_saves():
            return original(args, pair_output, src1, src2, frames)
    return prepare


def main():
    config = json.loads(Path(os.environ['MORPHFLOW_RENDER_REPAIR_CONFIG']).read_text())
    retention_path = os.environ.get('MORPHFLOW_DENSE_RETENTION_SCRIPT')
    if retention_path and len(sys.argv) > 1 and sys.argv[1] == 'shard' and '--run-dir' in sys.argv:
        run_dir = Path(sys.argv[sys.argv.index('--run-dir')+1])
        retention = runpy.run_path(retention_path)
        if retention['retired'](run_dir.parent, run_dir.name):
            retention['release'](run_dir.parent, run_dir.name)
            print('RESUME: dense metrics already verified; retained PNGs and metric caches', flush=True)
            return
    script = Path(config['frozen_evaluator'])
    sys.path[:0] = [str(script.parent), str(script.parent.parent)]
    namespace = runpy.run_path(str(script), run_name='threeway_render_cache_worker')
    globals_ = namespace['main'].__globals__
    globals_['cache_identity'] = lambda path, args, views: cache_identity(
        path, args, views, config['old_root'], config['new_root'])
    globals_['prepare_gaussian_images'] = repair_prepare(globals_['prepare_gaussian_images'], globals_)
    sys.argv[0] = str(script)
    globals_['main']()


if __name__ == '__main__':
    main()
