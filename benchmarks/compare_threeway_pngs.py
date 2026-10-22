#!/usr/bin/env python3
"""Compare cached view 025 across three methods and seven or fifty states.

Uses only Pillow and the standard library. No models, metrics, GLBs or CUDA are
loaded. Inputs are the full benchmark results on scratch, including the shared
render caches and relative symlinks, rather than the compact metrics archive.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


METHODS = ("morphflow", "morphany3d", "interp3d")
LABELS = {"morphflow": "MorphFlow", "morphany3d": "MorphAny3D", "interp3d": "Interp3D"}
VIEW = 25
CELL_PATTERN = re.compile(r"cfg_ss_([0-9.eE+-]+)_slat_([0-9.eE+-]+)")


class ComparisonError(ValueError):
    pass


class MissingRenders(ComparisonError):
    pass


def cfg_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ComparisonError("Invalid CFG value: {}".format(value))
    if not math.isfinite(number) or number < 0:
        raise ComparisonError("CFG values must be finite and nonnegative")
    return number


def find_cell(run_root, cfg_ss, cfg_slat):
    ss, slat = cfg_number(cfg_ss), cfg_number(cfg_slat)
    if not run_root.is_dir():
        raise ComparisonError("Benchmark run directory not found: {}".format(run_root))
    cells = []
    for path in run_root.iterdir():
        match = CELL_PATTERN.fullmatch(path.name)
        if match and path.is_dir():
            try:
                values = float(match[1]), float(match[2])
            except ValueError:
                continue
            if values == (ss, slat):
                cells.append(path)
    if len(cells) != 1:
        raise ComparisonError("Expected one CFG cell for SS={} SLat={}, found {} under {}".format(
            format(ss, ".12g"), format(slat, ".12g"), len(cells), run_root))
    return cells[0], ss, slat


def load_plan(run_root, cell, protocol="interp3d"):
    path = cell / "plan.json"
    if not path.is_file():
        path = run_root / "shared/plan.json"
    if not path.is_file():
        path = run_root / "plan.json"
    if not path.is_file():
        raise ComparisonError("Missing plan.json in {} or {}/shared".format(cell, run_root))
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
        alphas = [float(value) for value in plan["alphas"]]
        pairs = plan["pairs"]
    except (ValueError, TypeError, KeyError) as error:
        raise ComparisonError("Invalid benchmark plan {}: {}".format(path, error))
    expected = 48 if protocol == "weighted50" else 5
    if (len(alphas) != expected or not all(math.isfinite(a) and 0 < a < 1 for a in alphas)
            or not all(a > b for a, b in zip(alphas, alphas[1:]))
            or len(set(alpha_key(a) for a in alphas)) != expected):
        raise ComparisonError("Expected {} distinct intermediate alphas in descending order "
                              "for the {}-state protocol; check {} and --protocol".format(expected, expected+2, path))
    if protocol == "weighted50" and any(not math.isclose(a, i/49, rel_tol=0, abs_tol=1e-8)
                                        for a, i in zip(alphas, range(48, 0, -1))):
        raise ComparisonError("The 50-state protocol requires 48 uniform alphas i/49")
    if int(plan.get("k_intermediate", expected)) != expected:
        raise ComparisonError("Plan intermediate count differs from alpha schedule")
    if not isinstance(pairs, list) or not pairs:
        raise ComparisonError("No pairs in {}".format(path))
    ids, directories = set(), set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ComparisonError("Invalid pair record in {}".format(path))
        pair_id, name = pair.get("pair_id"), pair.get("pair_dir")
        if (not isinstance(pair_id, int) or isinstance(pair_id, bool) or pair_id < 0
                or not isinstance(name, str) or not re.fullmatch(r"pair_[A-Za-z0-9_.-]+", name)
                or name in directories or pair_id in ids):
            raise ComparisonError("Invalid or duplicate pair ID/directory in {}: {}".format(path, pair))
        ids.add(pair_id)
        directories.add(name)
    return plan, alphas


def alpha_key(alpha):
    # Same key as eval_perceptual_sequence.alpha_key, without its model imports.
    return "a{:.6f}".format(alpha).replace(".", "p")


def select_pairs(pairs, selectors):
    if not selectors:
        return pairs
    selected_ids = set()
    for selector in selectors:
        matches = [pair for pair in pairs if str(pair["pair_id"]) == str(selector)
                   or pair["pair_dir"] == str(selector)]
        if len(matches) != 1:
            raise ComparisonError("Pair {!r} is not in this run's plan.json".format(selector))
        selected_ids.add(matches[0]["pair_id"])
    return [pair for pair in pairs if pair["pair_id"] in selected_ids]


def image_paths(cell, pair, alphas):
    keys = ["src1"] + [alpha_key(a) for a in alphas] + ["src2"]
    paths, missing = [], []
    for method in METHODS:
        row = []
        for key in keys:
            path = cell / "metrics" / method / pair["pair_dir"] / "images" / key / "025.png"
            row.append(path)
            if not path.is_file():
                missing.append("{} / {}: {}".format(LABELS[method], key, path))
        paths.append(row)
    if missing:
        raise MissingRenders("Missing view 025 renders for {} ({} files):\n{}\n"
                             "Use the full scratch run with its shared caches; wait for its metrics "
                             "rendering, select a complete --pair, or use --skip-incomplete.".format(
                                 pair["pair_dir"], len(missing), "\n".join(missing)))
    return paths


def pillow_modules():
    try:
        from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
    except ImportError as error:
        raise ComparisonError("Pillow is required. On Ateneo use "
                              "slurm/compare_threeway_pngs_l40s.sh with the existing model env.") from error
    return Image, ImageDraw, ImageFont, PngImagePlugin


def font_at(ImageFont, size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def fit_text(draw, text, font, width):
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    while text and draw.textbbox((0, 0), text + "...", font=font)[2] > width:
        text = text[:-1]
    return text + "..."


def compose(pair, paths, alphas, cfg_ss, cfg_slat, tile_size, destination, protocol="interp3d"):
    Image, ImageDraw, ImageFont, PngImagePlugin = pillow_modules()
    margin, label_width, gap, row_gap = 20, 150, 8, 14
    left, top = margin + label_width, 116
    columns = len(alphas) + 2
    width = left + columns * tile_size + (columns-1) * gap + margin
    height = top + 3 * tile_size + 2 * row_gap + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font_at(ImageFont, 20)
    label_font = font_at(ImageFont, 16)
    small_font = font_at(ImageFont, 13)
    draw.text((margin, 14), fit_text(draw, "Three-way morph comparison | {} states | CFG SS={} | CFG SLat={} | view 025".format(
        columns, format(cfg_ss, ".12g"), format(cfg_slat, ".12g")), title_font, width - 2 * margin),
        font=title_font, fill="#17263c")
    pair_text = "Pair {} | {} -> {}".format(pair["pair_id"], pair.get("src1", "src1"), pair.get("src2", "src2"))
    # Use ASCII labels so old Pillow's fallback bitmap font works in containers.
    pair_text = pair_text.encode("ascii", "replace").decode("ascii")
    draw.text((margin, 44), fit_text(draw, pair_text, small_font, width - 2 * margin),
              font=small_font, fill="#4d5b70")
    full_alphas = [1.0] + alphas + [0.0]
    for column, alpha in enumerate(full_alphas):
        x = left + column * (tile_size + gap)
        label = "alpha={}".format(format(alpha, ".6f").rstrip("0").rstrip("."))
        draw.text((x + 4, 88), label, font=small_font, fill="#17263c")
        if column in (0, columns-1):
            draw.text((x + 4, 70), "Source 1" if column == 0 else "Source 2", font=small_font, fill="#4d5b70")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for row, method in enumerate(METHODS):
        y = top + row * (tile_size + row_gap)
        draw.text((margin, y + tile_size // 2 - 10), LABELS[method], font=label_font, fill="#17263c")
        for column, path in enumerate(paths[row]):
            x = left + column * (tile_size + gap)
            try:
                with Image.open(path) as original:
                    image = original.convert("RGBA")
                    image.thumbnail((tile_size, tile_size), resampling)
                    white = Image.new("RGBA", image.size, "white")
                    white.alpha_composite(image)
                    image = white.convert("RGB")
            except OSError as error:
                raise ComparisonError("Cannot read {} / {}: {} ({})".format(method, pair["pair_dir"], path, error))
            canvas.paste(image, (x + (tile_size - image.width) // 2, y + (tile_size - image.height) // 2))
            draw.rectangle((x - 1, y - 1, x + tile_size, y + tile_size), outline="#dce3ec")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("comparison", json.dumps({
        "methods": list(METHODS), "alphas": full_alphas, "view": VIEW,
        "protocol": protocol, "num_states": columns,
        "pair_id": pair["pair_id"], "pair_dir": pair["pair_dir"],
        "cfg_ss": cfg_ss, "cfg_slat": cfg_slat, "tile_size": tile_size,
        "layout": {"left": left, "top": top, "gap": gap, "row_gap": row_gap},
        "alpha_convention": "alpha=1 -> src1; alpha=0 -> src2",
        "source_paths": [[str(path) for path in row] for row in paths],
    }))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp." + str(os.getpid()))
    try:
        canvas.save(temporary, format="PNG", pnginfo=metadata)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_comparisons(run_root, cfg_ss, cfg_slat, output_dir=None, pair_selectors=None,
                       tile_size=256, overwrite=False, skip_incomplete=False, protocol="interp3d"):
    run_root = Path(run_root).expanduser().resolve()
    if protocol not in ("interp3d", "weighted50"):
        raise ComparisonError("Unknown comparison protocol: {}".format(protocol))
    if protocol == "weighted50" and (run_root / "weighted_50").is_dir():
        run_root = run_root / "weighted_50"
    cell, ss, slat = find_cell(run_root, cfg_ss, cfg_slat)
    plan, alphas = load_plan(run_root, cell, protocol)
    pairs = select_pairs(plan["pairs"], pair_selectors)
    if not isinstance(tile_size, int) or tile_size < 32:
        raise ComparisonError("--tile-size must be an integer >= 32")
    output = (Path(output_dir).expanduser().resolve() if output_dir is not None
              else run_root / "comparisons" / cell.name / "view_025")
    results = []
    for pair in pairs:
        destination = output / (pair["pair_dir"] + ".png")
        if destination.is_file() and not overwrite:
            raise ComparisonError("Comparison already exists: {}. Use --overwrite to regenerate it "
                                  "or choose another --output-dir.".format(destination))
        try:
            paths = image_paths(cell, pair, alphas)
        except MissingRenders as error:
            if not skip_incomplete:
                raise
            print("SKIP {}".format(error), file=sys.stderr, flush=True)
            continue
        compose(pair, paths, alphas, ss, slat, tile_size, destination, protocol)
        results.append(destination)
        print("SAVED {}".format(destination), flush=True)
    if not results:
        raise ComparisonError("No complete pairs to compare; no comparison PNGs were produced")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="Full scratch benchmark run, containing the CFG cells and shared caches")
    parser.add_argument("--cfg-ss", required=True, help="MorphFlow sparse-structure CFG")
    parser.add_argument("--cfg-slat", required=True, help="MorphFlow SLat CFG")
    parser.add_argument("--protocol", choices=("interp3d", "weighted50"), default="interp3d",
                        help="Seven states (default) or 50 states; weighted50 accepts the main run root or weighted_50 itself")
    parser.add_argument("--pair", action="append", dest="pair_selectors", help="Pair ID or full pair directory name; repeat to select several (default: all)")
    parser.add_argument("--output-dir", help="Default: RUN_ROOT/comparisons/CFG_CELL/view_025")
    parser.add_argument("--tile-size", type=int, default=256, help="Square tile size, without cropping; default: 256")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing comparison PNGs")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip pairs whose view-025 renders are not all available yet")
    args = parser.parse_args(argv)
    try:
        results = create_comparisons(**vars(args))
    except (ComparisonError, OSError) as error:
        parser.exit(1, "ERROR: {}\n".format(error))
    print("DONE: {} comparisons, view 025, 3 methods x {} states".format(
        len(results), 50 if args.protocol == "weighted50" else 7), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
