"""Run the existing TRELLIS Blender renderer on a JSON list of meshes."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_single_renderer():
    path = Path(__file__).with_name("render.py")
    spec = importlib.util.spec_from_file_location("morphflow_single_render", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    with open(args.manifest, "r", encoding="utf-8") as handle:
        jobs = json.load(handle)
    if not isinstance(jobs, list):
        raise ValueError("Render manifest must contain a JSON list.")

    renderer = load_single_renderer()
    for index, job in enumerate(jobs):
        print(
            f"[BATCH {index + 1}/{len(jobs)}] {job['object']} -> {job['output_folder']}",
            flush=True,
        )
        render_args = argparse.Namespace(
            views=args.views,
            object=job["object"],
            output_folder=job["output_folder"],
            resolution=args.resolution,
            engine=args.engine,
            geo_mode=args.geo_mode,
            save_depth=False,
            save_normal=False,
            save_albedo=False,
            save_mist=False,
            split_normal=False,
            save_mesh=False,
        )
        renderer.main(render_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch wrapper around render.py")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--geo_mode", action="store_true")
    argv = __import__("sys").argv
    args = parser.parse_args(argv[argv.index("--") + 1 :])
    main(args)
