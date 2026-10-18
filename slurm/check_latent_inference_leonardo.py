"""CPU preflight for the Leonardo latent eval and alpha-step launchers."""

import importlib
import math
import sys
from pathlib import Path


def main():
    mode, entrypoint, *arguments = sys.argv[1:]
    if mode not in ("--eval", "--generate", "--check", "--prepare"):
        raise ValueError(f"Modalita' non supportata: {mode}")
    modules = {
        "eval_validation_latents.py": "eval_validation_latents",
        "generate_alpha_steps_validation.py": "generate_alpha_steps_validation",
    }
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module(modules[entrypoint])
    sys.argv = [entrypoint, *arguments]
    args = module.parse_args()
    alpha_sweep = entrypoint == "generate_alpha_steps_validation.py"

    import torch
    from huggingface_hub import hf_hub_download
    from data.morph_dataset import MorphingDistillDataset
    from eval_validation_latents import checkpoint_args, detect_flow_target

    torch.set_num_threads(2)
    for name in ("steps", "slat_steps", "num_assets" if alpha_sweep else "num_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} deve essere >= 1")
    for name in ("cfg_scale", "slat_cfg_scale"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} deve essere finito e >= 0")

    root = Path(args.root_dir).expanduser().resolve()
    metadata = module.resolve_metadata_path(root, args.metadata) if alpha_sweep else Path(args.metadata)
    if not metadata.is_absolute():
        metadata = root / metadata
    dataset = MorphingDistillDataset(
        root=str(root), metadata_file=str(metadata),
        split=args.split if alpha_sweep else "test",
        load_occupancy=not alpha_sweep, load_source_images=False, verbose=True,
    )
    if alpha_sweep:
        alphas = (
            module.parse_alpha_list(args.alphas) if args.alphas else
            module.default_alpha_steps(args.num_intermediate_steps, bool(args.include_endpoints), args.direction)
        )
        if not alphas or any(not math.isfinite(a) for a in alphas):
            raise ValueError("Occorre almeno un alpha finito; aumentare NUM_INTERMEDIATE_STEPS.")
        candidates = module.validation_candidates(dataset, bool(args.deduplicate_pairs))
        selected = module.select_candidates(candidates, args.num_assets, args.start_index, args.selection, args.seed)
        print(f"Dataset: {root} | coppie selezionate={len(selected)}/{len(candidates)} | alpha={alphas}")
        if len(selected) < args.num_assets:
            print(f"ATTENZIONE: richieste {args.num_assets} coppie, disponibili {len(selected)} dopo START_INDEX.")
        # The generator loads sources directly from assets/<name>.
        for _, entry in selected:
            for name in (entry["src_1"], entry["src_2"]):
                for filename in ("slat_feats.pt", "slat_coords.pt", "ss_latent.pt"):
                    path = root / "assets" / name / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
        for name in (selected[0][1]["src_1"], selected[0][1]["src_2"]):
            module.load_asset(root, name)
    else:
        selected = module.select_fixed_pair_indices(dataset, args.num_samples, args.seed)
        print(f"Dataset: {root} | coppie selezionate={len(selected)} | un alpha del metadata per coppia")
        if len(selected) < args.num_samples:
            print(f"ATTENZIONE: richieste {args.num_samples} coppie, disponibili {len(selected)}.")
        dataset[selected[0]]

    targets = []
    for path in (args.checkpoint_path, args.slat_checkpoint_path):
        if not path:
            raise ValueError("Impostare sia CHECKPOINT_PATH=SS sia SLAT_CHECKPOINT_PATH=SLat.")
        # mmap inspects metadata without materializing student or optimizer tensors.
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        config = checkpoint_args(ckpt)
        target = detect_flow_target(ckpt)
        targets.append(target)
        print(f"Checkpoint: {path} | flow={target} epoch={ckpt.get('epoch')} "
              f"step={ckpt.get('step')} val_loss={ckpt.get('val_loss')}")
        if config.get("slat_condition_source", "slat") != "slat":
            raise ValueError("Checkpoint con conditioning diretto DINO non piu' supportato.")
        if config.get("ss_flow_arch", "standard") != "standard":
            raise ValueError("Checkpoint SS residuale non piu' supportato.")
        if int(config.get("separate_cond", 0)) and config.get("separate_cond_gate") != "token":
            raise ValueError("Sono supportati solo checkpoint con gating token.")
        del ckpt
    if targets != ["ss", "slat"]:
        raise ValueError("La pipeline richiede CHECKPOINT_PATH=SS e SLAT_CHECKPOINT_PATH=SLat.")

    for decoder in ("slat_dec_mesh_swin8_B_64l8m256c_fp16", "ss_dec_conv3d_16l8_fp16"):
        for ext in ("json", "safetensors"):
            filename = f"ckpts/{decoder}.{ext}"
            try:
                path = hf_hub_download("microsoft/TRELLIS-image-large", filename,
                                       local_files_only=mode != "--prepare")
            except Exception as exc:
                raise RuntimeError(f"Decoder mancante: {filename}. Eseguire --prepare dal login "
                                   "con gli stessi checkpoint.") from exc
            print(f"Decoder: {path}")
    # These are imported lazily by inference; check them on the login node too.
    from trellis.models import SLatMeshDecoder, SparseStructureDecoder  # noqa: F401


if __name__ == "__main__":
    main()
