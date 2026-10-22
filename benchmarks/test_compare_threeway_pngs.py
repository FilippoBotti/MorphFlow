"""Compare cached benchmark PNGs using Pillow, without models or a GPU."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_threeway_pngs as compare


METHODS = ("morphflow", "morphany3d", "interp3d")
ALPHAS = [.8, .65, .5, .35, .2]
PAIRS = ("pair_0000_a_to_b", "pair_0001_c_to_d", "pair_0002_e_to_f")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_png(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color).save(path)


def frame_color(row, column):
    # Canonical endpoints are identical for all methods, just as in the cache.
    if column in (0, 6):
        return (25, 35, 45) if column == 0 else (215, 225, 235)
    return (45 + row * 65, 20 + column * 30, 35 + row * 20 + column * 5)


def relative_link(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def fixture(root, pairs=PAIRS, alphas=ALPHAS):
    cell = root / "cfg_ss_5_slat_1.5"
    cell.mkdir(parents=True)
    plan = {"alphas": list(alphas), "k_intermediate": len(alphas),
            "pairs": [{"pair_id": PAIRS.index(pair), "pair_dir": pair,
                       "src1": f"source_{pair}", "src2": f"target_{pair}"}
                      for pair in pairs]}
    # Older benchmark plans have no sequence_alphas. Relative links survive
    # archiving or moving a run, including the source-image cache links.
    write_json(root / "shared" / "plan.json", plan)
    relative_link(cell / "plan.json", root / "shared" / "plan.json")
    for row, method in enumerate(METHODS):
        method_root = cell / "metrics" / method
        if row:
            target = root / "shared" / "metrics" / method
            target.mkdir(parents=True)
            relative_link(method_root, target)
        for pair in pairs:
            images = method_root / pair / "images"
            for column, endpoint in ((0, "src1"), (6, "src2")):
                cached = root / "shared" / "reference_cache" / pair / endpoint
                write_png(cached / "025.png", frame_color(row, column))
                relative_link(images / endpoint, cached)
            for column, alpha in enumerate(alphas, start=1):
                key = f"a{alpha:.6f}".replace(".", "p")
                write_png(images / key / "025.png", frame_color(row, column))
    return cell


def png_metadata(path):
    with Image.open(path) as image:
        return json.loads(image.info["comparison"])


def dense_fixture(root):
    cell = fixture(root, pairs=PAIRS[:1])
    plan_path = root / 'shared/plan.json'
    plan = json.loads(plan_path.read_text())
    plan.update(alphas=[round(i/49, 10) for i in range(48, 0, -1)],
                k_intermediate=48, num_sequence_frames=50,
                protocol='weighted-endpoints-50', metric_mode='weighted-endpoints-only')
    write_json(plan_path, plan)
    for row, method in enumerate(METHODS):
        for column, alpha in enumerate(plan['alphas'], 1):
            color = (10+row*60, 10+column*4, 40+row*10)
            write_png(cell/'metrics'/method/PAIRS[0]/'images'/compare.alpha_key(alpha)/'025.png', color)
    return cell, plan['alphas']


class CompareThreewayPngTests(unittest.TestCase):
    def test_dense_grid_all_150_tiles_and_main_root_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell, alphas = dense_fixture(root/'weighted_50')
            outputs = compare.create_comparisons(root, 5, 1.5, protocol='weighted50', tile_size=64)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].parent, root/'weighted_50/comparisons/cfg_ss_5_slat_1.5/view_025')
            meta = png_metadata(outputs[0])
            self.assertEqual(meta['num_states'], 50)
            self.assertEqual(meta['protocol'], 'weighted50')
            self.assertEqual(meta['alphas'], [1.0, *alphas, 0.0])
            layout=meta['layout']
            with Image.open(outputs[0]) as image:
                self.assertEqual(image.width, layout['left']+50*64+49*layout['gap']+20)
                for row in range(3):
                    for column in range(50):
                        expected = (frame_color(row, 0 if column==0 else 6)
                                    if column in (0,49) else (10+row*60,10+column*4,40+row*10))
                        center=(layout['left']+column*(64+layout['gap'])+32,
                                layout['top']+row*(64+layout['row_gap'])+32)
                        self.assertEqual(image.convert('RGB').getpixel(center), expected, (row,column))

    def test_dense_requires_all_intermediates_and_uniform_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cell, alphas=dense_fixture(root)
            missing=cell/'metrics/interp3d'/PAIRS[0]/'images'/compare.alpha_key(alphas[30])/'025.png'
            missing.unlink()
            with self.assertRaisesRegex(compare.MissingRenders, '025.png'):
                compare.create_comparisons(root,5,1.5,protocol='weighted50',tile_size=64)
            plan_path=root/'shared/plan.json'
            plan=json.loads(plan_path.read_text())
            plan['alphas'][10]-=.001
            write_json(plan_path,plan)
            with self.assertRaisesRegex(compare.ComparisonError,'uniform'):
                compare.create_comparisons(root,5,1.5,protocol='weighted50',tile_size=64)

    def test_dense_shell_wrapper_uses_existing_python_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dense_fixture(root/'weighted_50')
            wrapper=Path(compare.__file__).resolve().parents[1]/'slurm/compare_threeway_dense_pngs_leonardo.sh'
            env=dict(os.environ,COMPARISON_PYTHON=sys.executable)
            env.pop('COMPARISON_OUT',None)
            env.pop('PROJECT_DIR',None)
            result=subprocess.run(['bash',str(wrapper),str(root),'5','1.5','--pair','0','--tile-size','64'],
                                  env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                  universal_newlines=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('3 methods x 50 states',result.stdout)

    def test_grid_uses_shared_sources_relative_baselines_and_descending_alphas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, pairs=PAIRS[:1])
            outputs = compare.create_comparisons(root, 5.0, 1.5, tile_size=64)
            self.assertEqual(len(outputs), 1)
            metadata = png_metadata(outputs[0])
            self.assertEqual(metadata["methods"], list(METHODS))
            self.assertEqual(metadata["alphas"], [1.0, *ALPHAS, 0.0])
            self.assertEqual(metadata["view"], 25)
            self.assertEqual(metadata["pair_id"], 0)
            self.assertEqual(metadata["pair_dir"], PAIRS[0])
            self.assertEqual(float(metadata["cfg_ss"]), 5)
            self.assertEqual(float(metadata["cfg_slat"]), 1.5)
            self.assertEqual(metadata["tile_size"], 64)
            layout = metadata["layout"]
            with Image.open(outputs[0]) as image:
                image = image.convert("RGB")
                for row in range(3):
                    for column in range(7):
                        x = layout["left"] + column * (64 + layout["gap"]) + 32
                        y = layout["top"] + row * (64 + layout["row_gap"]) + 32
                        self.assertEqual(image.getpixel((x, y)), frame_color(row, column),
                                         (row, column))

    def test_pair_selection_accepts_ids_and_full_names_and_custom_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            fixture(root)
            destination = Path(directory) / "selected"
            outputs = compare.create_comparisons(
                root, "5.0", "1.50", output_dir=destination,
                pair_selectors=["2", PAIRS[0]], tile_size=64)
            self.assertEqual({png_metadata(path)["pair_dir"] for path in outputs},
                             {PAIRS[0], PAIRS[2]})
            self.assertTrue(all(Path(path).parent == destination for path in outputs))
            with self.assertRaisesRegex((ValueError, FileNotFoundError), "999|pair|Pair"):
                compare.create_comparisons(root, 5, 1.5, pair_selectors=["999"])

    def test_missing_view_is_actionable_and_skip_incomplete_keeps_complete_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = fixture(root, pairs=PAIRS[:2])
            missing = cell / "metrics" / "interp3d" / PAIRS[0] / "images" / "a0p500000" / "025.png"
            missing.unlink()
            with self.assertRaises(compare.MissingRenders) as raised:
                compare.create_comparisons(root, 5, 1.5, tile_size=64)
            message = str(raised.exception)
            self.assertIn("025.png", message)
            self.assertIn("interp3d", message.lower())
            self.assertIn(PAIRS[0], message)
            outputs = compare.create_comparisons(root, 5, 1.5, tile_size=64,
                                                  skip_incomplete=True)
            self.assertEqual([png_metadata(path)["pair_id"] for path in outputs], [1])

    def test_refuses_dense_or_ambiguous_alpha_schedule(self):
        invalid_schedules = (
            [index / 49 for index in range(48, 0, -1)],
            [.8, .5, .65, .35, .2],
            [.8, .65, .5, .5, .2],
        )
        for alphas in invalid_schedules:
            with self.subTest(alphas=alphas), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, pairs=PAIRS[:1])
                plan_path = root / "shared" / "plan.json"
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                plan["alphas"] = alphas
                write_json(plan_path, plan)
                with self.assertRaisesRegex(ValueError, "alpha|Alpha|intermediate|states|stati|five|5"):
                    compare.create_comparisons(root, 5, 1.5, tile_size=64)

    def test_cli_builds_only_selected_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            fixture(root)
            destination = Path(directory) / "cli-output"
            result = subprocess.run(
                [sys.executable, str(Path(compare.__file__)), "--run-root", str(root),
                 "--cfg-ss", "5.0", "--cfg-slat", "1.5", "--pair", "1",
                 "--output-dir", str(destination), "--tile-size", "64"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs = list(destination.glob("*.png"))
            self.assertEqual(len(outputs), 1)
            self.assertEqual(png_metadata(outputs[0])["pair_dir"], PAIRS[1])

    def test_existing_output_requires_overwrite_and_regenerates_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = fixture(root, pairs=PAIRS[:1])
            output = compare.create_comparisons(root, 5, 1.5, tile_size=64)[0]
            original_bytes = output.read_bytes()
            changed_color = (10, 230, 80)
            source = cell / "metrics" / "morphflow" / PAIRS[0] / "images" / "a0p500000" / "025.png"
            write_png(source, changed_color)
            with self.assertRaisesRegex(compare.ComparisonError, "--overwrite"):
                compare.create_comparisons(root, 5, 1.5, tile_size=64)
            self.assertEqual(output.read_bytes(), original_bytes)
            rebuilt = compare.create_comparisons(root, 5, 1.5, tile_size=64,
                                                overwrite=True)
            self.assertEqual(rebuilt, [output])
            metadata = png_metadata(output)
            layout = metadata["layout"]
            center = (layout["left"] + 3 * (64 + layout["gap"]) + 32,
                      layout["top"] + 32)
            with Image.open(output) as image:
                self.assertEqual(image.convert("RGB").getpixel(center), changed_color)


if __name__ == "__main__":
    unittest.main()
