"""Morph two GLBs at one alpha, also exporting decoder and model reconstructions.

Uses generate_glb_pair's multiview TRELLIS encoding and SS -> SLat pipeline.
alpha=1 selects src1; alpha=0 selects src2. Endpoints are sampled by the model.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil

import generate_glb_pair as pipeline


OUTPUT_FILES = {
    "decoder_src1": "reconstruction_decoder_src1.glb",
    "decoder_src2": "reconstruction_decoder_src2.glb",
    "model_src1": "reconstruction_model_src1_alpha1.glb",
    "model_src2": "reconstruction_model_src2_alpha0.glb",
    "morph": "morph.glb",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src1", type=Path, help="First GLB (alpha=1)")
    parser.add_argument("src2", type=Path, help="Second GLB (alpha=0)")
    parser.add_argument("--ss-checkpoint", "--checkpoint_path", type=Path, required=True)
    parser.add_argument("--slat-checkpoint", "--slat_checkpoint_path", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Weight of src1, in [0,1]")
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=os.environ.get("BLENDER_BIN") or shutil.which("blender"),
                        help="Blender 3.x executable; also accepts BLENDER_BIN")
    parser.add_argument("--cfg-ss", "--cfg_scale", type=float, default=3.0)
    parser.add_argument("--cfg-slat", "--slat_cfg_scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50, help="SS flow integration steps")
    parser.add_argument("--slat-steps", "--slat_steps", type=int, default=50)
    parser.add_argument("--encoding-views", type=int, default=50)
    parser.add_argument("--dino-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "no"), default="bf16")
    parser.add_argument("--mode", choices=("run", "check", "prepare"), default="run")
    args = parser.parse_args(argv)
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be finite and in [0,1]")
    if args.blender is None:
        parser.error("Supply --blender /path/to/blender or set BLENDER_BIN")
    # Even alpha=0/1 must generate both endpoints; deduplicate the requested alpha.
    alphas = list(dict.fromkeys((1.0, args.alpha, 0.0)))
    args.output_dir = args.output_dir.expanduser().resolve()
    backend = pipeline.parse_args([
        str(args.src1), str(args.src2),
        "--ss-checkpoint", str(args.ss_checkpoint),
        "--slat-checkpoint", str(args.slat_checkpoint),
        "--output-dir", str(args.output_dir / "pipeline"), "--blender", str(args.blender),
        "--alphas", ",".join(map(str, alphas)),
        "--cfg-ss", str(args.cfg_ss), "--cfg-slat", str(args.cfg_slat),
        "--steps", str(args.steps), "--slat-steps", str(args.slat_steps),
        "--encoding-views", str(args.encoding_views), "--dino-batch-size", str(args.dino_batch_size),
        "--seed", str(args.seed), "--mixed-precision", args.mixed_precision, "--mode", args.mode,
    ])
    return args, backend


def export_outputs(args, backend):
    """Publish explicit filenames only after the entire pipeline has succeeded."""
    root = backend.output_dir
    cfg_dir = root / f"cfg_ss_{pipeline.tag(args.cfg_ss)}_slat_{pipeline.tag(args.cfg_slat)}"

    def prediction(alpha):
        return cfg_dir / f"alpha_{pipeline.tag(alpha)}" / "prediction.glb"

    sources = {
        "decoder_src1": root / "encoding/src1/reconstruction.glb",
        "decoder_src2": root / "encoding/src2/reconstruction.glb",
        "model_src1": prediction(1.0),
        "model_src2": prediction(0.0),
        "morph": prediction(args.alpha),
    }
    for path in sources.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing pipeline output: {path}")
    outputs = {}
    for role, source in sources.items():
        target = args.output_dir / OUTPUT_FILES[role]
        temporary = target.with_suffix(".glb.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        outputs[role] = str(target)

    run = json.loads((root / "run.json").read_text())
    summary = {
        "status": "complete", "alpha": args.alpha,
        "alpha_convention": "1=src1, 0=src2",
        "inputs": {"src1": str(backend.src1), "src2": str(backend.src2)},
        "checkpoints": run["checkpoints"],
        "cfg_ss": args.cfg_ss, "cfg_slat": args.cfg_slat,
        "ss_steps": args.steps, "slat_steps": args.slat_steps, "seed": args.seed,
        "outputs": outputs,
        "decoder_reconstruction": "multiview DINO -> SLat encoder posterior mean -> mesh decoder",
        "model_reconstruction": "full SS -> SS decoder -> SLat -> mesh decoder at alpha=1 and alpha=0",
        "encoding": {"views": args.encoding_views, "voxel_resolution": 64,
                     "feature_model": "dinov2_vitl14_reg", "sample_posterior": False},
        "geometry": "individually normalized to the unit cube; exported GLBs are Y-up",
        "appearance": "decoded vertex colors",
        "pipeline_manifest": str(root / "run.json"),
    }
    pipeline.write_json(args.output_dir / "summary.json", summary)
    for role, path in outputs.items():
        print(f"{role}: {path}", flush=True)
    return summary


def main(argv=None):
    args, backend = parse_args(argv)
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ["HF_HUB_OFFLINE"] = "0" if args.mode == "prepare" else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_HUB_OFFLINE"]
    metadata = pipeline.preflight(backend)
    if args.mode != "run":
        print("CHECK PASSED: inputs, checkpoints, Blender, TRELLIS and DINO cache", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".morph.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Output directory is already in use: {args.output_dir}") from exc
        # Avoid replacing unrelated files on the first use of an output directory.
        if not (backend.output_dir / "run.json").exists():
            for name in (*OUTPUT_FILES.values(), "summary.json"):
                if (args.output_dir / name).exists():
                    raise FileExistsError(f"Output already exists; choose another directory: {args.output_dir / name}")
        try:
            pipeline.run(backend, metadata)
            export_outputs(args, backend)
        except Exception as exc:
            pipeline.write_json(args.output_dir / "last_error.json",
                                {"error": type(exc).__name__, "message": str(exc)})
            raise
        (args.output_dir / "last_error.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
