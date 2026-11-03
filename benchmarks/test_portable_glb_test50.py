"""Materialize 50 v4 test GLBs and exercise the portable runner on random pairs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from html import escape
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parents[1]
BASELINE = BASE / 'outputs/threeway_benchmark/full_ss17_slat12_n50_seed42_ss3_4_4.5_5_slat1.5_20261026'
OLD = BASE / 'outputs/morphflow_glb_reencoded_vs_dataset_full_test10_20261101'
DEFAULT_RUN = BASE / 'outputs/morphflow_glb_test50_random4_20261103'
ASSETS = BASE / 'assets/test_v4_50'
sys.path[:0] = [str(PROJECT / 'monster-exp'), str(PROJECT), str(PROJECT / 'benchmarks')]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root):
    from morph_assets import glb_error
    plan = read_json(BASELINE / 'shared/plan.json')
    test = set(read_json(Path(plan['data_root']) / 'split_assets.json')['test'])
    assets = []
    for pair in plan['pairs'][:25]:
        for source in ('src1', 'src2'):
            name = pair[source]
            if name not in test:
                raise ValueError('Asset outside test split: ' + name)
            raw = BASELINE / 'shared/references_raw' / pair['pair_dir'] / (source + '_raw.pt')
            if not raw.is_file():
                raise FileNotFoundError(raw)
            assets.append({'name': name, 'glb': str(ASSETS / (name + '.glb')),
                           'raw': str(raw.resolve()),
                           'cached_glb': str(OLD / 'assets' / name / 'source.glb')})
    if len({a['name'] for a in assets}) != 50:
        raise ValueError('Expected 50 unique sources')
    ids = random.Random(42).sample(range(25), 4)
    pairs = [dict(plan['pairs'][i], index=j) for j, i in enumerate(ids)]
    selected = {p[k] for p in pairs for k in ('src1', 'src2')}
    pending = [a['name'] for a in assets if not Path(a['cached_glb']).is_file() and a['name'] not in selected]
    old_manifest = read_json(OLD / 'manifest.json')
    manifest = {'version': 1, 'split': 'test', 'data_root': plan['data_root'],
                'selection': '50 unique assets from first 25 frozen test pairs; four pairs randomly sampled without replacement, seed 42',
                'pair_selection_seed': 42, 'assets': assets, 'asset_dir': str(ASSETS), 'pairs': pairs,
                'bulk_chunks': [pending[i::2] for i in range(2)],
                'alphas': [0.0, 0.05, 0.5, 0.95, 1.0],
                'checkpoints': {k: old_manifest['checkpoints'][k]['path'] for k in ('ss', 'slat')},
                'sampling': {'ss_steps': 25, 'slat_steps': 25, 'cfg_ss': 5, 'cfg_slat': 1.5,
                             'precision': 'bf16', 'matmul_tf32': False, 'cudnn_tf32': True,
                             'seed': 'historical pair_seed; SLat seed=pair_seed+10000'},
                'encoding': {'views': 150, 'lighting': 'uniform_texture', 'input_up_axis': 'y'},
                'export': {'texture_size': 2048, 'simplify': 0, 'fill_holes': False,
                           'material_mode': 'unlit', 'original_pbr_also_saved': True},
                'source_glbs': 'Decoded dataset source mesh plus Gaussian-baked texture, as in the reference experiment; source exports use seed 42.',
                'inference': 'Portable morph_assets.py receives GLBs and freshly encodes both inputs. Every alpha including 0/1 is sampled; no source replacement.',
                'reference_overview': str(OLD / 'overview.png'),
                'code_sha256': {str(p): digest(p) for p in [PROJECT / 'monster-exp/morph_assets.py',
                               PROJECT / 'monster-exp/make_glb_material_variants.py', Path(__file__)]}}
    root.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    target = root / 'manifest.json'
    if target.exists() and read_json(target) != manifest:
        raise FileExistsError('Different experiment manifest: ' + str(root))
    write_json(target, manifest)
    write_json(ASSETS / 'manifest.json', manifest)
    for asset in assets:
        src, dst = Path(asset['cached_glb']), Path(asset['glb'])
        if src.is_file() and not dst.exists():
            if glb_error(src, textured=True):
                raise ValueError('Invalid cached GLB: ' + str(src))
            temporary = dst.with_suffix('.tmp.glb')
            shutil.copy2(src, temporary)
            temporary.replace(dst)
            write_json(dst.with_suffix('.json'), {'status': 'complete', 'source': str(src),
                       'source_raw': asset['raw'], 'sha256': digest(dst), 'method': 'exact copy of previous official Gaussian-textured source'})
    report(root)
    print(json.dumps({'assets': len(assets), 'copied': sum(Path(a['glb']).is_file() for a in assets),
                      'selected_pair_ids': ids, 'bulk_sources': len(pending), 'root': str(root)}, indent=2))


def ensure_source(asset):
    from morph_assets import glb_error
    from trellis_asset_codec import OfficialAssetCodec
    from benchmark_morph_glb_conditioning import export_raw_glb
    from threeway_textured_benchmark import seed_everything
    destination = Path(asset['glb'])
    with destination.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file() and read_json(destination.with_suffix('.json'), {}).get('status') == 'complete':
            if glb_error(destination, textured=True) is None:
                return
        seed_everything(42)
        codec = OfficialAssetCodec(trellis_root=BASE / 'src/TRELLIS', texture_size=2048,
                                    lighting='uniform_texture')
        export_raw_glb(asset['raw'], destination, codec, textured=True)
        codec.release()


def materialize(root, index):
    manifest = read_json(root / 'manifest.json')
    mapping = {a['name']: a for a in manifest['assets']}
    for name in manifest['bulk_chunks'][index]:
        print('SOURCE_START ' + name, flush=True)
        ensure_source(mapping[name])
        report(root)
        print('SOURCE_COMPLETE ' + name, flush=True)


def worker(root, index):
    manifest = read_json(root / 'manifest.json')
    pair = manifest['pairs'][index]
    folder = root / 'pairs' / pair['pair_dir']
    folder.mkdir(parents=True, exist_ok=True)
    assets = {a['name']: a for a in manifest['assets']}
    try:
        for key in ('src1', 'src2'):
            ensure_source(assets[pair[key]])
        command = [sys.executable, str(PROJECT / 'monster-exp/morph_assets.py'),
                   assets[pair['src1']]['glb'], assets[pair['src2']]['glb'],
                   '--alphas', *map(str, manifest['alphas']),
                   '--ss-checkpoint', manifest['checkpoints']['ss'],
                   '--slat-checkpoint', manifest['checkpoints']['slat'],
                   '--trellis-root', str(BASE / 'src/TRELLIS'),
                   '--blender', str(BASE / 'tools/blender-3.0.1-linux-x64/blender'),
                   '--glb-up-axis', 'y', '--encoding-views', '150', '--lighting', 'uniform_texture',
                   '--steps-ss', '25', '--steps-slat', '25', '--cfg-ss', '5', '--cfg-slat', '1.5',
                   '--seed', str(pair['pair_seed']), '--slat-seed', str(pair['pair_seed'] + 10000),
                   '--mixed-precision', 'bf16', '--cudnn-tf32', '1',
                   '--texture-size', '2048', '--simplify', '0', '--material-mode', 'unlit',
                   '--output-dir', str(folder / 'run')]
        write_json(folder / 'invocation.json', {'command': command, 'job_id': __import__('os').environ.get('SLURM_JOB_ID')})
        subprocess.run(command, check=True, cwd=PROJECT)
        render_pair(root, index)
        (folder / 'error.json').unlink(missing_ok=True)
    except Exception as exc:
        write_json(folder / 'error.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        report(root)
        raise


def render_pair(root, index):
    import torch
    from morph_assets import alpha_tag, glb_error
    from morph_glb_conditioning_metrics import _RawRenderer, _save_view
    from trellis_codec_metrics import render_glb, _save_sheet, appearance_metrics
    torch.set_num_threads(4)
    def save_view(view, path):
        path = Path(path)
        temporary = path.with_name('.' + path.stem + '.' + uuid.uuid4().hex + '.png')
        _save_view(view, temporary)
        temporary.replace(path)
    manifest = read_json(root / 'manifest.json')
    pair = manifest['pairs'][index]
    folder = root / 'pairs' / pair['pair_dir']
    run = folder / 'run'
    renderer = _RawRenderer(resolution=512)
    source_views = {}
    assets = {a['name']: a for a in manifest['assets']}
    for key in ('src1', 'src2'):
        norm = read_json(run / 'inputs' / key / 'encoding/encode.json')['normalization']
        views = render_glb(assets[pair[key]]['glb'], normalization=norm, cameras=renderer.cameras, resolution=512)
        source_views[key] = views
        save_view(views[7], folder / (key + '_view07.png'))
    gaussian_rows, glb_rows, metrics = [], [], []
    for alpha in manifest['alphas']:
        state = run / 'morphs' / alpha_tag(alpha)
        if glb_error(state / 'prediction_textured.glb', textured=True):
            raise ValueError('Invalid final GLB: ' + str(state))
        raw = torch.load(state / 'mesh.pt', map_location='cpu', weights_only=True)
        raw['gaussian'] = torch.load(state / 'gaussian.pt', map_location='cpu', weights_only=True)
        gs = renderer.render(raw)
        mesh = render_glb(state / 'prediction_textured.glb', cameras=renderer.cameras, resolution=512)
        for name, views in [('gaussian', gs), ('final_glb', mesh)]:
            for j, view in enumerate(views):
                save_view(view, folder / 'renders' / alpha_tag(alpha) / (name + '_view%02d.png' % j))
        gaussian_rows.append(('Gaussian alpha ' + str(alpha), gs))
        glb_rows.append(('Final GLB alpha ' + str(alpha), mesh))
        metric = {'alpha': alpha, 'baking_vs_gaussian': appearance_metrics(gs, mesh)}
        if alpha in (0, 1):
            source = source_views['src1' if alpha == 1 else 'src2']
            metric['endpoint_gaussian_vs_input'] = appearance_metrics(source, gs)
            metric['endpoint_glb_vs_input'] = appearance_metrics(source, mesh)
        metrics.append(metric)
    _save_sheet(gaussian_rows, folder / 'gaussian_views.png', thumb_size=192)
    _save_sheet(glb_rows, folder / 'final_glb_views.png', thumb_size=192)
    historical = OLD / 'pairs' / pair['pair_dir'] / 'glb_encoded/alpha_0p500000/mesh_raw.pt'
    if historical.is_file():
        reference = renderer.render(torch.load(historical, map_location='cpu', weights_only=True))
        current = next(views for label, views in gaussian_rows if label == 'Gaussian alpha 0.5')
        _save_sheet([('Reference old GLB encoding', reference), ('Portable runner / fresh encoding', current)],
                    folder / 'historical_midpoint_comparison.png', thumb_size=192)
    result = {'status': 'complete', 'pair_id': pair['pair_id'], 'alphas': metrics,
              'validation': 'All five final GLBs have embedded textures; all endpoints are sampled model outputs.',
              'quality': 'Inspect native Gaussian and final GLB images; format validation does not imply visual fidelity.'}
    (folder / 'error.json').unlink(missing_ok=True)
    write_json(folder / 'result.json', result)
    report(root)


def report(root):
    from morph_assets import alpha_tag
    from PIL import Image, ImageDraw
    from trellis_codec_overview import _font
    manifest = read_json(root / 'manifest.json')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        available = sum(Path(a['glb']).is_file() and read_json(Path(a['glb']).with_suffix('.json'), {}).get('status') == 'complete' for a in manifest['assets'])
        completed = 0; glbs = 0; errors = []; sections = []
        tile, gap, left, rowheight = 224, 8, 165, 555
        canvas = Image.new('RGB', (left + 7 * (tile + gap) + gap, 120 + len(manifest['pairs']) * rowheight), '#edf0f4')
        draw = ImageDraw.Draw(canvas); title = _font(22, True); body = _font(14); small = _font(12)
        draw.text((18, 16), 'MorphFlow: input GLB, 50 asset test, 4 coppie casuali', font=title, fill='#172234')
        draw.text((18, 50), 'Sopra: Gaussiane native. Sotto: GLB finale unlit. Stessa camera della overview precedente, senza ritagli indipendenti.', font=body, fill='#334155')
        labels = ['Sorgente B'] + ['alpha ' + str(a) for a in manifest['alphas']] + ['Sorgente A']
        for j, label in enumerate(labels):
            draw.text((left + j * (tile + gap), 87), label, font=body, fill='#172234')
        for i, pair in enumerate(manifest['pairs']):
            folder = root / 'pairs' / pair['pair_dir']
            result = read_json(folder / 'result.json')
            if result and result.get('status') == 'complete': completed += 1
            error = read_json(folder / 'error.json')
            if error: errors.append({'pair_id': pair['pair_id'], **error})
            y = 120 + i * rowheight
            draw.text((12, y + 18), 'Coppia %02d' % pair['pair_id'], font=title, fill='#172234')
            draw.text((12, y + 55), 'Gaussian', font=body, fill='#334155')
            draw.text((12, y + 300), 'GLB unlit', font=body, fill='#334155')
            for row, kind in enumerate(('gaussian', 'final_glb')):
                paths = [folder / 'src2_view07.png']
                paths += [folder / 'renders' / alpha_tag(a) / (kind + '_view07.png') for a in manifest['alphas']]
                paths += [folder / 'src1_view07.png']
                for j, imagepath in enumerate(paths):
                    x, yy = left + j * (tile + gap), y + row * 252
                    draw.rectangle((x, yy, x + tile, yy + tile), fill='white')
                    if imagepath.is_file():
                        with Image.open(imagepath) as image:
                            image = image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS)
                            canvas.paste(image, (x, yy))
                    else:
                        draw.text((x + 55, yy + 100), 'In attesa', font=body, fill='#64748b')
            links = []
            for alpha in manifest['alphas']:
                file = folder / 'run/morphs' / alpha_tag(alpha) / 'prediction_textured.glb'
                if file.is_file():
                    glbs += 1
                    links.append('<a href="' + file.relative_to(root).as_posix() + '">GLB alpha ' + str(alpha) + '</a>')
            extra = ''
            for filename in ['gaussian_views.png', 'final_glb_views.png', 'historical_midpoint_comparison.png', 'result.json', 'run/index.html']:
                if (folder / filename).is_file():
                    extra += ' | <a href="' + (folder / filename).relative_to(root).as_posix() + '">' + filename + '</a>'
            sections.append('<h2>Coppia %02d</h2><p>%s → %s</p><p>%s%s</p>' % (
                pair['pair_id'], escape(pair['src2']), escape(pair['src1']), ' | '.join(links) or 'In attesa dei GLB', extra))
        temporary = root / 'overview.tmp.png'; canvas.save(temporary); temporary.replace(root / 'overview.png')
        status = 'complete' if completed == len(manifest['pairs']) and available == 50 and not errors else 'in_progress'
        summary = {'status': status, 'assets_ready': available, 'assets_expected': 50,
                   'pairs_complete': completed, 'pairs_expected': 4, 'final_glbs': glbs, 'final_glbs_expected': 20,
                   'errors': errors, 'reference_overview': manifest['reference_overview']}
        write_json(root / 'summary.json', summary)
        (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Test 50 asset GLB</title>'
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}a{margin-right:8px}</style>'
            + f'<h1>Test del runner portatile da GLB</h1><p>Asset pronti: {available}/50. Coppie complete: {completed}/4. GLB finali: {glbs}/20.</p>'
            + '<p>Alpha 0 = B; alpha 1 = A. Tutti gli alpha sono generati dal modello, inclusi gli estremi. Input codificati dai GLB con 150 viste; SS 17 / SLat 12, 25+25 passi. GLB finali unlit per preservare l\'aspetto baked; originale PBR salvato a parte.</p>'
            + '<p><a href="manifest.json">Configurazione</a> | <a href="summary.json">Stato</a> | <a href="../../assets/test_v4_50/index.html">50 GLB sorgente</a> | <a href="../morphflow_glb_reencoded_vs_dataset_full_test10_20261101/overview.png">Overview precedente</a></p><img src="overview.png">'
            + ''.join(sections))
        asset_links = ['<li><a href="' + escape(Path(a['glb']).name) + '">' + escape(a['name']) + '</a>'
                       + (' — pronto' if Path(a['glb']).is_file() else ' — in esportazione') + '</li>' for a in manifest['assets']]
        (ASSETS / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>50 asset del test v4</h1><p>%d/50 GLB pronti. Mesh e texture ottenute dal decoding dei latenti del dataset.</p><ul>%s</ul>' % (available, ''.join(asset_links)))
        print(json.dumps({k: v for k, v in summary.items() if k != 'errors'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'materialize', 'worker', 'render', 'report'])
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    if args.mode in ('prepare', 'report'):
        {'prepare': prepare, 'report': report}[args.mode](root)
    else:
        {'materialize': materialize, 'worker': worker, 'render': render_pair}[args.mode](root, args.index)
