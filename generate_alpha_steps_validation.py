import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch

if os.environ.get("TRELLIS_REPO"):
    sys.path.append(os.environ["TRELLIS_REPO"])

from data.morph_dataset import MorphingDistillDataset
from eval_validation_latents import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    build_model,
    checkpoint_args,
    checkpoint_requires_source_images,
    detect_flow_target,
    detect_model_type,
    detect_slat_condition_source,
    load_checkpoint,
    load_decoders,
    preload_dino_if_needed,
    safe_slug,
    sample_slat_on_coords,
    sample_ss,
    save_slat_glb,
    save_voxel_glb,
    ss_coords_from_latent,
)
from generate_alpha_steps import (
    alpha_color,
    batch_for_alpha,
    load_asset,
    parse_alpha_list,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/alpha_steps_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full MorphFlow alpha-step pipeline on k validation asset pairs. "
            "Full pipeline means SS flow -> SS decoder sparse coords -> SLat flow -> mesh decoder."
        )
    )
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--metadata", type=str, default="metadata_val.json")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT), help="SS-flow checkpoint.")
    parser.add_argument("--slat_checkpoint_path", type=str, required=True, help="Second-flow SLat checkpoint for the final mesh stage.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))

    parser.add_argument("--num_assets", "--num_pairs", dest="num_assets", type=int, default=8, help="Number of validation asset pairs to process.")
    parser.add_argument("--start_index", type=int, default=0, help="Offset in the selected validation candidate list before taking k pairs.")
    parser.add_argument(
        "--selection",
        type=str,
        choices=["first", "random", "evenly_spaced"],
        default="first",
        help="How to choose k validation pairs after optional pair deduplication.",
    )
    parser.add_argument("--deduplicate_pairs", type=int, choices=[0, 1], default=1, help="Use each src1/src2 pair only once.")

    parser.add_argument("--num_intermediate_steps", type=int, default=8, help="Number of intermediate alpha values between src1 and src2.")
    parser.add_argument("--include_endpoints", type=int, choices=[0, 1], default=0, help="Also generate alpha=1 and alpha=0 predictions. Source endpoint GLBs are always saved.")
    parser.add_argument("--alphas", type=str, default=None, help="Optional explicit comma/space separated alpha list. Overrides --num_intermediate_steps.")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["src1_to_src2", "src2_to_src1", "ascending"],
        default="src1_to_src2",
        help="Default alpha order. Convention: alpha=1 is src1 and alpha=0 is src2.",
    )

    parser.add_argument("--steps", type=int, default=25, help="SS flow integration steps.")
    parser.add_argument("--slat_steps", type=int, default=None, help="SLat flow integration steps. Defaults to --steps.")
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--slat_cfg_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trellis_model", type=str, choices=["auto", "text_base", "image_large"], default="auto")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_latents", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_voxels", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_sources", type=int, choices=[0, 1], default=1)
    parser.add_argument("--empty_cache_each_pair", type=int, choices=[0, 1], default=1)
    parser.add_argument("--source_images_root", type=str, default=None, help="Root containing source images for DINO-conditioned SLat checkpoints.")
    parser.add_argument("--source_image_filename", type=str, default="", help="Optional fixed image filename inside each asset directory.")
    return parser.parse_args()


def default_alpha_steps(num_intermediate_steps: int, include_endpoints: bool, direction: str) -> List[float]:
    if num_intermediate_steps < 0:
        raise ValueError("--num_intermediate_steps must be >= 0")

    if include_endpoints:
        values = np.linspace(1.0, 0.0, num_intermediate_steps + 2, dtype=np.float64)
    else:
        if num_intermediate_steps == 0:
            values = np.asarray([], dtype=np.float64)
        else:
            values = np.linspace(1.0, 0.0, num_intermediate_steps + 2, dtype=np.float64)[1:-1]

    if direction in ("src2_to_src1", "ascending"):
        values = values[::-1]

    return [float(round(v, 10)) for v in values.tolist()]


def resolve_metadata_path(root: Path, metadata: Optional[str]) -> Optional[Path]:
    if not metadata:
        return None
    path = Path(metadata).expanduser()
    return path if path.is_absolute() else root / path


def validation_candidates(dataset: MorphingDistillDataset, deduplicate_pairs: bool) -> List[Tuple[int, Dict[str, Any]]]:
    out: List[Tuple[int, Dict[str, Any]]] = []
    seen = set()
    for idx, entry in enumerate(dataset.metadata):
        src1 = str(entry["src_1"])
        src2 = str(entry["src_2"])
        key = (src1, src2)
        if deduplicate_pairs and key in seen:
            continue
        seen.add(key)
        out.append((idx, dict(entry)))
    return out


def select_candidates(
    candidates: Sequence[Tuple[int, Dict[str, Any]]],
    num_assets: int,
    start_index: int,
    selection: str,
    seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    if num_assets < 1:
        raise ValueError("--num_assets must be >= 1")
    if not candidates:
        raise RuntimeError("No validation candidates found.")
    if start_index < 0 or start_index >= len(candidates):
        raise IndexError(f"--start_index out of range [0, {len(candidates) - 1}]: {start_index}")

    pool = list(candidates[start_index:])
    k = min(num_assets, len(pool))

    if selection == "first":
        return pool[:k]
    if selection == "random":
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(pos)] for pos in positions.tolist()]
    if selection == "evenly_spaced":
        positions = np.linspace(0, len(pool) - 1, k, dtype=int)
        return [pool[int(pos)] for pos in positions.tolist()]
    raise ValueError(f"Unknown selection: {selection}")


def attach_source_images_for_entry(dataset: MorphingDistillDataset, src1: Dict[str, Any], src2: Dict[str, Any], entry: Dict[str, Any]) -> None:
    src1["image"] = dataset._load_source_image(src1["name"], entry, "src1")
    src2["image"] = dataset._load_source_image(src2["name"], entry, "src2")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def pair_dir_name(pair_id: int, src1_name: str, src2_name: str) -> str:
    return f"pair_{pair_id:04d}_{safe_slug(src1_name, 36)}_to_{safe_slug(src2_name, 36)}"


def run_pair(
    pair_id: int,
    metadata_index: int,
    entry: Dict[str, Any],
    root: Path,
    output_dir: Path,
    alphas: Sequence[float],
    args: argparse.Namespace,
    dataset: MorphingDistillDataset,
    model: torch.nn.Module,
    slat_model: torch.nn.Module,
    ss_decoder: torch.nn.Module,
    mesh_decoder: torch.nn.Module,
    sparse_tensor_cls: Any,
    device: torch.device,
    needs_source_images: bool,
    ckpt: Dict[str, Any],
    slat_ckpt: Dict[str, Any],
) -> Dict[str, Any]:
    src1_name = str(entry["src_1"])
    src2_name = str(entry["src_2"])
    src1 = load_asset(root, src1_name)
    src2 = load_asset(root, src2_name)

    if needs_source_images:
        attach_source_images_for_entry(dataset, src1, src2, entry)

    pair_out = output_dir / pair_dir_name(pair_id, src1_name, src2_name)
    pair_out.mkdir(parents=True, exist_ok=True)

    if args.save_sources == 1:
        save_slat_glb(
            mesh_decoder,
            sparse_tensor_cls,
            src1["feats"],
            src1["coords"],
            pair_out / "src1.glb",
            device,
            args.mixed_precision,
            fallback_color=(80, 150, 255),
        )
        save_slat_glb(
            mesh_decoder,
            sparse_tensor_cls,
            src2["feats"],
            src2["coords"],
            pair_out / "src2.glb",
            device,
            args.mixed_precision,
            fallback_color=(255, 150, 80),
        )

    slat_steps = args.slat_steps if args.slat_steps is not None else args.steps
    slat_cfg_scale = args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale
    template_ss = torch.zeros_like(src1["ss_latent"])
    step_rows: List[Dict[str, Any]] = []

    for step_idx, alpha in enumerate(alphas):
        ss_seed = int(args.seed + pair_id * 100_000 + step_idx)
        slat_seed = int(args.seed + pair_id * 100_000 + 10_000 + step_idx)
        torch.manual_seed(ss_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ss_seed)

        batch = batch_for_alpha(src1, src2, alpha)
        pred_ss = sample_ss(
            model,
            batch,
            template_ss.to(device=device, dtype=torch.float32),
            args.steps,
            device,
            args.cfg_scale,
            args.mixed_precision,
        )

        step_name = f"alpha_{float(alpha):.6f}".replace(".", "p")
        step_out = pair_out / step_name
        step_out.mkdir(parents=True, exist_ok=True)

        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "metadata_index": metadata_index,
            "step": step_idx,
            "alpha": float(alpha),
            "src1": src1_name,
            "src2": src2_name,
            "target_from_metadata": str(entry.get("target", "")),
            "ss_seed": ss_seed,
            "slat_seed": slat_seed,
            "step_dir": str(step_out),
        }

        if args.save_voxels == 1:
            voxel_path = step_out / "pred_voxels.glb"
            row["pred_voxels"] = str(voxel_path)
            row["voxel_info"] = save_voxel_glb(
                ss_decoder,
                pred_ss,
                voxel_path,
                device,
                args.mixed_precision,
                color=alpha_color(alpha),
            )

        pred_coords = ss_coords_from_latent(ss_decoder, pred_ss, device, args.mixed_precision)
        row["pred_ss_points"] = int(pred_coords.shape[0])

        torch.manual_seed(slat_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(slat_seed)
        pred_final_slat = sample_slat_on_coords(
            slat_model,
            batch,
            pred_coords,
            slat_steps,
            device,
            slat_cfg_scale,
            args.mixed_precision,
        )

        if pred_final_slat is None:
            row["pred_final_saved"] = False
            row["pred_final_reason"] = "empty_ss_coords"
        else:
            final_path = step_out / "pred_final.glb"
            row["pred_final"] = str(final_path)
            row["pred_final_saved"] = save_slat_glb(
                mesh_decoder,
                sparse_tensor_cls,
                pred_final_slat.feats,
                pred_final_slat.coords,
                final_path,
                device,
                args.mixed_precision,
                fallback_color=alpha_color(alpha),
            )
            row["pred_final_points"] = int(pred_final_slat.feats.shape[0])

        if args.save_latents == 1:
            latent_path = step_out / "latents.pt"
            payload = {
                "flow_target": "ss_to_slat_pipeline",
                "ss_flow_arch": checkpoint_args(ckpt).get("ss_flow_arch", "standard"),
                "slat_condition_source": detect_slat_condition_source(slat_ckpt),
                "src1": src1_name,
                "src2": src2_name,
                "metadata_index": metadata_index,
                "metadata_entry": entry,
                "alpha": float(alpha),
                "pred_ss_latent": pred_ss.detach().cpu(),
                "pred_ss_coords": pred_coords.detach().cpu(),
            }
            if pred_final_slat is not None:
                payload.update(
                    {
                        "pred_final_slat_feats": pred_final_slat.feats.detach().cpu(),
                        "pred_final_slat_coords": pred_final_slat.coords.detach().cpu(),
                    }
                )
            torch.save(payload, latent_path)
            row["latents"] = str(latent_path)

        write_json(step_out / "metadata.json", row)
        step_rows.append(row)
        append_jsonl(output_dir / "steps.jsonl", row)

        print(
            f"[{pair_id + 1}] step {step_idx + 1}/{len(alphas)} "
            f"alpha={float(alpha):.6f} ss_points={row['pred_ss_points']} "
            f"final={row.get('pred_final_saved')} -> {step_out}",
            flush=True,
        )

        del pred_ss, pred_coords, pred_final_slat

    sequence = []
    if args.save_sources == 1:
        sequence.append({"kind": "source", "alpha": 1.0, "path": str(pair_out / "src1.glb"), "name": src1_name})
    for row in step_rows:
        sequence.append({"kind": "prediction", "alpha": row["alpha"], "path": row.get("pred_final"), "saved": row.get("pred_final_saved", False)})
    if args.save_sources == 1:
        sequence.append({"kind": "source", "alpha": 0.0, "path": str(pair_out / "src2.glb"), "name": src2_name})

    pair_summary = {
        "pair_id": pair_id,
        "metadata_index": metadata_index,
        "src1": src1_name,
        "src2": src2_name,
        "metadata_target": str(entry.get("target", "")),
        "metadata_alpha": float(entry.get("alpha", -1.0)) if "alpha" in entry else None,
        "pair_dir": str(pair_out),
        "num_generated_steps": len(step_rows),
        "alphas": [float(a) for a in alphas],
        "steps": step_rows,
        "sequence": sequence,
    }
    write_json(pair_out / "sequence.json", sequence)
    write_json(pair_out / "summary.json", pair_summary)
    return pair_summary


def main() -> None:
    args = parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    metadata_path = resolve_metadata_path(root, args.metadata)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    slat_checkpoint_path = Path(args.slat_checkpoint_path).expanduser().resolve()

    if args.allow_tf32 == 1 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SS checkpoint not found: {checkpoint_path}")
    if not slat_checkpoint_path.is_file():
        raise FileNotFoundError(f"SLat checkpoint not found: {slat_checkpoint_path}")

    if args.alphas:
        alphas = parse_alpha_list(args.alphas)
    else:
        alphas = default_alpha_steps(args.num_intermediate_steps, bool(args.include_endpoints), args.direction)
    if not alphas:
        raise ValueError("No alpha values to generate. Increase --num_intermediate_steps or pass --alphas.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = load_checkpoint(str(checkpoint_path))
    flow_target = detect_flow_target(ckpt)
    if flow_target != "ss":
        raise ValueError(f"--checkpoint_path must be an SS checkpoint for the full pipeline, got {flow_target!r}.")

    slat_ckpt = load_checkpoint(str(slat_checkpoint_path))
    slat_flow_target = detect_flow_target(slat_ckpt)
    if slat_flow_target != "slat":
        raise ValueError(f"--slat_checkpoint_path must be a SLat checkpoint, got {slat_flow_target!r}.")

    needs_source_images = checkpoint_requires_source_images(slat_ckpt, "slat")
    source_images_root = args.source_images_root or checkpoint_args(slat_ckpt).get("source_images_root")
    if needs_source_images and not source_images_root:
        raise ValueError("This DINO-conditioned SLat checkpoint requires --source_images_root.")

    model_type = detect_model_type(ckpt, args.trellis_model)
    slat_model_type = detect_model_type(slat_ckpt, args.trellis_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MorphingDistillDataset(
        root=str(root),
        metadata_file=str(metadata_path) if metadata_path is not None else None,
        split=args.split,
        verbose=True,
        load_occupancy=False,
        load_source_images=False,
        source_images_root=source_images_root,
        source_image_filename=args.source_image_filename,
    )
    candidates = validation_candidates(dataset, bool(args.deduplicate_pairs))
    selected = select_candidates(candidates, args.num_assets, args.start_index, args.selection, args.seed)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{args.split}_k{len(selected):03d}_n{args.num_intermediate_steps:03d}_pipeline"
    )
    output_dir = Path(args.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== VALIDATION ALPHA-STEPS PIPELINE =====")
    print(f"dataset: {root}")
    print(f"metadata: {metadata_path}")
    print(f"split: {args.split}")
    print(f"validation candidates: {len(candidates)}")
    print(f"selected pairs: {len(selected)}")
    print(f"selection: {args.selection}, start_index={args.start_index}, deduplicate_pairs={args.deduplicate_pairs}")
    print(f"ss checkpoint: {checkpoint_path}")
    print(f"slat checkpoint: {slat_checkpoint_path}")
    print("pipeline: SS flow -> SS decoder coords -> SLat flow -> mesh decoder")
    print(f"ss_flow_arch: {checkpoint_args(ckpt).get('ss_flow_arch', 'standard')}")
    print(f"slat_condition_source: {detect_slat_condition_source(slat_ckpt)}")
    if needs_source_images:
        print(f"source_images_root: {source_images_root}")
        print(f"source_image_filename: {args.source_image_filename or '<auto>'}")
    print(f"model_type: {model_type}")
    print(f"slat_model_type: {slat_model_type}")
    print(f"alphas: {alphas}")
    print("alpha convention: alpha=1 is src1, alpha=0 is src2")
    print(f"steps: {args.steps}, slat_steps: {args.slat_steps if args.slat_steps is not None else args.steps}")
    print(f"cfg_scale: {args.cfg_scale}, slat_cfg_scale: {args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale}")
    print(f"mixed_precision: {args.mixed_precision}")
    print(f"output: {output_dir}")
    print("==========================================")

    write_json(
        output_dir / "selected_pairs.json",
        [
            {
                "pair_id": i,
                "metadata_index": int(metadata_index),
                "src1": str(entry["src_1"]),
                "src2": str(entry["src_2"]),
                "target": str(entry.get("target", "")),
                "metadata_alpha": float(entry.get("alpha", -1.0)) if "alpha" in entry else None,
            }
            for i, (metadata_index, entry) in enumerate(selected)
        ],
    )

    model = build_model(ckpt, model_type, "ss").to(device).eval()
    slat_model = build_model(slat_ckpt, slat_model_type, "slat").to(device).eval()
    preload_dino_if_needed(model, device)
    preload_dino_if_needed(slat_model, device)
    ss_decoder, mesh_decoder, sparse_tensor_cls = load_decoders("ss", device)

    pair_summaries: List[Dict[str, Any]] = []
    summary_path = output_dir / "summary.json"

    for pair_id, (metadata_index, entry) in enumerate(selected):
        print(
            f"===== PAIR {pair_id + 1}/{len(selected)} | metadata_index={metadata_index} | "
            f"{entry['src_1']} -> {entry['src_2']} =====",
            flush=True,
        )
        try:
            pair_summary = run_pair(
                pair_id=pair_id,
                metadata_index=int(metadata_index),
                entry=entry,
                root=root,
                output_dir=output_dir,
                alphas=alphas,
                args=args,
                dataset=dataset,
                model=model,
                slat_model=slat_model,
                ss_decoder=ss_decoder,
                mesh_decoder=mesh_decoder,
                sparse_tensor_cls=sparse_tensor_cls,
                device=device,
                needs_source_images=needs_source_images,
                ckpt=ckpt,
                slat_ckpt=slat_ckpt,
            )
        except Exception as exc:
            pair_summary = {
                "pair_id": pair_id,
                "metadata_index": int(metadata_index),
                "src1": str(entry.get("src_1", "")),
                "src2": str(entry.get("src_2", "")),
                "failed": True,
                "error": repr(exc),
            }
            write_json(output_dir / f"pair_{pair_id:04d}_FAILED.json", pair_summary)
            print(f"[FAILED] pair_id={pair_id} error={exc!r}", flush=True)
            raise
        finally:
            if args.empty_cache_each_pair == 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        pair_summaries.append(pair_summary)
        write_json(
            summary_path,
            {
                "dataset": str(root),
                "metadata": str(metadata_path) if metadata_path is not None else None,
                "split": args.split,
                "checkpoint": str(checkpoint_path),
                "slat_checkpoint": str(slat_checkpoint_path),
                "pipeline_mode": True,
                "flow_target": "ss_to_slat_pipeline",
                "ss_flow_arch": checkpoint_args(ckpt).get("ss_flow_arch", "standard"),
                "slat_condition_source": detect_slat_condition_source(slat_ckpt),
                "source_images_root": str(source_images_root) if source_images_root else None,
                "model_type": model_type,
                "slat_model_type": slat_model_type,
                "num_assets_requested": args.num_assets,
                "num_assets_done": len(pair_summaries),
                "num_intermediate_steps": args.num_intermediate_steps,
                "include_endpoints": bool(args.include_endpoints),
                "alphas": [float(a) for a in alphas],
                "selection": args.selection,
                "start_index": args.start_index,
                "deduplicate_pairs": bool(args.deduplicate_pairs),
                "cfg_scale": args.cfg_scale,
                "slat_cfg_scale": args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale,
                "steps": args.steps,
                "slat_steps": args.slat_steps if args.slat_steps is not None else args.steps,
                "seed": args.seed,
                "output_dir": str(output_dir),
                "pairs": pair_summaries,
            },
        )

    print("===== DONE =====")
    print(f"output: {output_dir}")
    print(f"summary: {summary_path}")
    print("================")


if __name__ == "__main__":
    main()
