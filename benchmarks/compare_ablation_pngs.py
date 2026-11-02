#!/usr/bin/env python3
"""Compose cached MorphFlow ablation renders: one row per run, seven states.

Only Pillow and the standard library are required. The default view is 025.png.
Use the local presets or repeat --row 'Label=/path/to/run' for arbitrary runs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import textwrap

from compare_threeway_pngs import (
    ComparisonError, MissingRenders, alpha_key, find_cell, font_at,
    load_plan, pillow_modules, select_pairs,
)

WORKSPACE = Path(__file__).resolve().parents[3]
BENCH = WORKSPACE / 'outputs/threeway_benchmark'
SUITE = BENCH / 'ablation_suite_20261026'
BASELINE = BENCH / 'baseline_ss10_slat34_n50_seed42_cfg11_cfg515_20261023'
FULL = BENCH / 'full_ss17_slat12_n50_seed42_ss3_4_4.5_5_slat1.5_20261026'


def identity(pair):
    source, target = pair.get('src1'), pair.get('src2')
    if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
        raise ComparisonError('Every pair must specify src1 and src2')
    return source, target


def read_row(label, root, cfg_ss, cfg_slat):
    root = Path(root).expanduser().resolve()
    cell, ss, slat = find_cell(root, cfg_ss, cfg_slat)
    plan, alphas = load_plan(root, cell, 'interp3d')
    by_pair = {}
    for pair in plan['pairs']:
        key = identity(pair)
        if key in by_pair:
            raise ComparisonError('Repeated oriented source pair in {}'.format(root))
        by_pair[key] = pair
    return dict(label=label, root=root, cell=cell, plan=plan, alphas=alphas,
                by_pair=by_pair, cfg_ss=ss, cfg_slat=slat)


def validate_rows(rows):
    if len(rows) < 2:
        raise ComparisonError('Choose at least two experiment rows')
    if len({r['label'] for r in rows}) != len(rows):
        raise ComparisonError('Experiment row labels must be unique')
    first = rows[0]
    for row in rows[1:]:
        if set(row['by_pair']) != set(first['by_pair']):
            raise ComparisonError('Different source pairs or orientation: {} versus {}'.format(
                first['label'], row['label']))
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
               for a, b in zip(row['alphas'], first['alphas'])):
            raise ComparisonError('Different alpha schedules: {}'.format(row['label']))
        if row['plan'].get('seed') != first['plan'].get('seed'):
            raise ComparisonError('Different generation seeds: {}'.format(row['label']))
        for key, pair in row['by_pair'].items():
            if pair.get('pair_seed') != first['by_pair'][key].get('pair_seed'):
                raise ComparisonError('Different noise seed for {} in {}'.format(key, row['label']))


def input_paths(rows, pair, view):
    paths = []
    missing = []
    for row in rows:
        # Match by ordered source identities, independently of local pair IDs.
        local_pair = row['by_pair'][identity(pair)]
        images = row['cell'] / 'metrics/morphflow' / local_pair['pair_dir'] / 'images'
        keys = ['src1'] + [alpha_key(a) for a in row['alphas']] + ['src2']
        current = [images / key / '{:03d}.png'.format(view) for key in keys]
        paths.append(current)
        missing.extend('{} / {}: {}'.format(row['label'], key, path)
                       for key, path in zip(keys, current) if not path.is_file())
    if missing:
        raise MissingRenders('Missing cached renders for {}:\n{}'.format(
            pair['pair_dir'], '\n'.join(missing)))
    return paths


def source_signature(paths):
    return [[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in row] for row in paths]


def compose(rows, pair, paths, tile_size, view, destination):
    Image, ImageDraw, ImageFont, PngImagePlugin = pillow_modules()
    margin, label_width, gap, row_gap = 20, 220, 8, 14
    left, top = margin + label_width, 132
    width = left + 7 * tile_size + 6 * gap + margin
    height = top + len(rows) * tile_size + (len(rows) - 1) * row_gap + margin
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = font_at(ImageFont, 21)
    label_font = font_at(ImageFont, 18)
    small_font = font_at(ImageFont, 14)
    first = rows[0]
    title = 'MorphFlow ablations | 7 states | CFG SS={} / SLat={} | view {:03d}'.format(
        format(first['cfg_ss'], '.12g'), format(first['cfg_slat'], '.12g'), view)
    draw.text((margin, 12), title, font=title_font, fill='#17263c')
    sources = 'Pair {} | {} -> {}'.format(pair['pair_id'], pair['src1'], pair['src2'])
    draw.text((margin, 44), sources, font=small_font, fill='#4d5b70')
    draw.text((margin, 69), 'alpha=1: source 1; alpha=0: source 2. Endpoints are canonical source renders.',
              font=small_font, fill='#4d5b70')
    alphas = [1.0] + first['alphas'] + [0.0]
    for column, alpha in enumerate(alphas):
        x = left + column * (tile_size + gap)
        label = ('Source 1 | ' if column == 0 else 'Source 2 | ' if column == 6 else '')
        label += 'a=' + format(alpha, '.4f')
        draw.text((x + 3, 106), label, font=small_font, fill='#17263c')
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    for row_index, row in enumerate(rows):
        y = top + row_index * (tile_size + row_gap)
        label = '\n'.join(textwrap.wrap(row['label'], width=22, break_long_words=False))
        box = draw.multiline_textbbox((0, 0), label, font=label_font, spacing=6)
        draw.multiline_text((margin, y + (tile_size - (box[3] - box[1])) // 2),
                            label, font=label_font, spacing=6, fill='#17263c')
        for column, path in enumerate(paths[row_index]):
            x = left + column * (tile_size + gap)
            try:
                with Image.open(path) as original:
                    tile = original.convert('RGBA')
                    tile.thumbnail((tile_size, tile_size), resampling)
                    white = Image.new('RGBA', tile.size, 'white')
                    white.alpha_composite(tile)
                    tile = white.convert('RGB')
            except (OSError, ValueError) as error:
                raise ComparisonError('Cannot read {}: {}'.format(path, error)) from error
            canvas.paste(tile, (x + (tile_size - tile.width) // 2,
                                y + (tile_size - tile.height) // 2))
            draw.rectangle((x-1, y-1, x+tile_size, y+tile_size), outline='#dce3ec')
    signature = source_signature(paths)
    info = dict(kind='morphflow_ablation_comparison', num_states=7, view=view,
                pair_id=pair['pair_id'], pair_dir=pair['pair_dir'],
                src1=pair['src1'], src2=pair['src2'], alphas=alphas,
                cfg_ss=first['cfg_ss'], cfg_slat=first['cfg_slat'], tile_size=tile_size,
                rows=[dict(label=r['label'], run_root=str(r['root']),
                           pair_dir=r['by_pair'][identity(pair)]['pair_dir']) for r in rows],
                source_images=signature,
                layout=dict(left=left, top=top, gap=gap, row_gap=row_gap))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('comparison', json.dumps(info))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.tmp.' + str(os.getpid()))
    try:
        canvas.save(temporary, format='PNG', pnginfo=metadata)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_comparisons(row_specs, cfg_ss, cfg_slat, output_dir, pair_selectors=None,
                       tile_size=256, view=25, overwrite=False, skip_incomplete=False):
    if not isinstance(tile_size, int) or tile_size < 64:
        raise ComparisonError('--tile-size must be an integer >=64')
    if not isinstance(view, int) or not 0 <= view < 64:
        raise ComparisonError('--view must be between 0 and 63')
    rows = [read_row(label, root, cfg_ss, cfg_slat) for label, root in row_specs]
    validate_rows(rows)
    output = Path(output_dir).expanduser().resolve()
    selected = select_pairs(rows[0]['plan']['pairs'], pair_selectors)
    ready = []
    skipped = []
    # Preflight every selected pair before producing a partial comparison set.
    for pair in selected:
        destination = output / (pair['pair_dir'] + '.png')
        if destination.exists() and not overwrite:
            raise ComparisonError('Already exists: {}. Use --overwrite.'.format(destination))
        try:
            paths = input_paths(rows, pair, view)
        except MissingRenders as error:
            if not skip_incomplete:
                raise
            skipped.append(pair['pair_dir'])
            print('SKIP ' + str(error), file=sys.stderr)
            continue
        ready.append((pair, paths, destination))
    if not ready:
        raise ComparisonError('No complete pairs to compare')
    for pair, paths, destination in ready:
        compose(rows, pair, paths, tile_size, view, destination)
        print('SAVED ' + str(destination), flush=True)
    manifest = dict(cfg_ss=rows[0]['cfg_ss'], cfg_slat=rows[0]['cfg_slat'], view=view,
                    num_states=7, tile_size=tile_size,
                    rows=[dict(label=r['label'], run_root=str(r['root'])) for r in rows],
                    outputs=[str(p[2]) for p in ready], skipped_pairs=skipped)
    target = output / 'comparison_manifest.json'
    temporary = target.with_name('.' + target.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(target)
    return [p[2] for p in ready]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg-ss', type=float, default=5)
    parser.add_argument('--cfg-slat', type=float, default=1.5)
    parser.add_argument('--suite-root', type=Path, default=SUITE)
    parser.add_argument('--baseline-root', type=Path, default=BASELINE)
    parser.add_argument('--include-full', action='store_true', help='Append the full SS17/SLat12 model row')
    parser.add_argument('--full-root', type=Path, default=FULL)
    parser.add_argument('--row', action='append', help='Custom row: LABEL=RUN_ROOT; repeat at least twice. Replaces all default rows.')
    parser.add_argument('--pair', action='append', dest='pair_selectors', help='Pair ID or directory name; repeat to select several')
    parser.add_argument('--view', type=int, default=25)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-incomplete', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.row:
            if args.include_full:
                raise ComparisonError('Use --row also for the full model when specifying custom rows')
            specs = []
            for item in args.row:
                label, separator, path = item.partition('=')
                if not separator or not label.strip() or not path.strip():
                    raise ComparisonError('--row requires LABEL=RUN_ROOT')
                specs.append((label.strip(), path.strip()))
        else:
            specs = [('Baseline', args.baseline_root),
                     ('Gate SS + SLat', args.suite_root/'gate_only_both_n50_seed42'),
                     ('Prior SS + SLat', args.suite_root/'prior_only_both_n50_seed42'),
                     ('Swap SS only', args.suite_root/'swap_ss_only_n50_seed42')]
            if args.include_full:
                specs.append(('Full SS17 / SLat12', args.full_root))
        output = args.output_dir or args.suite_root/'comparisons'/(
            'cfg_ss_{}_slat_{}'.format(format(args.cfg_ss, '.12g'), format(args.cfg_slat, '.12g')))/(
                'view_{:03d}{}'.format(args.view, '_with_full' if args.include_full else ''))
        results = create_comparisons(specs, args.cfg_ss, args.cfg_slat, output,
            pair_selectors=args.pair_selectors, tile_size=args.tile_size, view=args.view,
            overwrite=args.overwrite, skip_incomplete=args.skip_incomplete)
    except (ComparisonError, OSError) as error:
        parser.exit(1, 'ERROR: {}\n'.format(error))
    print('DONE: {} pairs, {} experiment rows x 7 states, view {:03d}'.format(
        len(results), len(specs), args.view), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
