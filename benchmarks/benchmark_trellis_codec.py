"""Direct v4 decoding versus GLB autoencoding, without a generative flow.

prepare selects unique test endpoints; worker runs one asset on a GPU; report
collects only completed measurements. The v4 dataset contains generated latents,
not ground-truth input GLBs, so the experiment measures an export roundtrip.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_DATA = Path('/leonardo_scratch/fast/IscrC_MORPHFL/mbarezzi/datasets/morphing_dataset_v4')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(args):
    data = args.dataset.resolve()
    rows = json.loads((data / 'metadata_test.json').read_text())
    names = sorted({r[k] for r in rows for k in ('src_1', 'src_2')})[:args.count]
    split = set(json.loads((data / 'split_assets.json').read_text())['test'])
    if len(names) != args.count or any(name not in split for name in names):
        raise ValueError('Requested count must fit the unique v4 test endpoints')
    assets = []
    for index, name in enumerate(names):
        folder = data / 'assets' / name
        for filename in ('slat_feats.pt', 'slat_coords.pt'):
            if not (folder / filename).is_file():
                raise FileNotFoundError(folder / filename)
        assets.append({'index': index, 'asset': name, 'directory': str(folder),
                       'test_membership': True,
                       'test_pairs': sum(r['src_1'] == name or r['src_2'] == name for r in rows)})
    manifest = {'dataset': str(data), 'split': 'test', 'selection': 'first unique test endpoints in lexicographic order',
                'count': len(assets), 'assets': assets, 'seed': 42,
                'reference': 'direct decoding of decoder-scale v4 source SLat; no original GLB exists in this dataset',
                'branches': ['direct', 'legacy_50', 'official_150'],
                'input_to_both_encoders': 'direct/textured.glb (identical source GLB)',
                'export': {'texture_size': args.texture_size, 'simplify': 0, 'fill_holes': False},
                'metadata_sha256': hashlib.sha256((data / 'metadata_test.json').read_bytes()).hexdigest()}
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise FileExistsError(f'Incompatible manifest already exists: {path}')
    write_json(path, manifest)
    print(json.dumps({'manifest': str(path), 'assets': names}, indent=2))


def latent_comparison(source, encoded):
    import numpy as np
    a = source['coords'].numpy().astype(np.int64)
    b = encoded['coords'].numpy().astype(np.int64)
    ka = a[:, -3:] @ np.array([4096, 64, 1])
    kb = b[:, -3:] @ np.array([4096, 64, 1])
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    out = {'source_voxels': len(a), 'encoded_voxels': len(b), 'common_voxels': len(common),
           'coordinate_iou': float(len(common) / len(np.union1d(ka, kb))),
           'caveat': 'Coordinate overlap is diagnostic only: source mesh is normalized again by official GLB preprocessing.'}
    if len(common):
        af = source['feats'].numpy()[ia].astype(np.float64)
        bf = encoded['feats'].numpy()[ib].astype(np.float64)
        out['raw_feature_rmse_common_coords'] = float(np.sqrt(np.mean((af - bf) ** 2)))
        out['raw_feature_cosine_common_coords'] = float(np.mean(np.sum(af * bf, axis=1) /
            np.maximum(np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1), 1e-12)))
    return out


def worker(args):
    import torch
    from trellis_asset_codec import OfficialAssetCodec
    from trellis_codec_metrics import compare_glbs, render_montage
    manifest = json.loads((args.output / 'manifest.json').read_text())
    item = manifest['assets'][args.index]
    out = args.output / f"{args.index:02d}_{item['asset']}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').is_file():
        print(f'Already complete: {out}', flush=True)
        return
    started = time.perf_counter()
    write_json(out / 'status.json', {'stage': 'loading', 'asset': item['asset']})
    codec = OfficialAssetCodec(blender_bin=args.blender, texture_size=manifest['export']['texture_size'],
                               dino_batch_size=args.dino_batch_size, seed=manifest['seed'])
    srcdir = Path(item['directory'])
    source = {'feats': torch.load(srcdir / 'slat_feats.pt', map_location='cpu', weights_only=True).float(),
              'coords': torch.load(srcdir / 'slat_coords.pt', map_location='cpu', weights_only=True).int()}
    if source['coords'].shape[1] == 3:
        source['coords'] = torch.cat([torch.zeros(len(source['coords']), 1, dtype=torch.int32), source['coords']], 1)
    result = {'asset': item['asset'], 'index': args.index, 'source': item,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'branches': {}}
    try:
        write_json(out / 'status.json', {'stage': 'direct decoding'})
        result['direct'] = codec.decode(source, out / 'direct', textured=True)
        source_glb = out / 'direct/textured.glb'
        for branch, views in [('legacy_50', 50), ('official_150', 150)]:
            stage_start = time.perf_counter()
            folder = out / branch
            write_json(out / 'status.json', {'stage': branch, 'phase': 'encoding'})
            folder.mkdir(parents=True, exist_ok=True)
            if branch == 'legacy_50':
                from encode import TrellisSLatEncoderDebug
                cache = folder / 'latent.pt'
                if cache.is_file():
                    encoded = torch.load(cache, map_location='cpu', weights_only=True)
                else:
                    legacy = TrellisSLatEncoderDebug(blender_bin=str(codec.blender_bin),
                                                    dino_batch_size=args.dino_batch_size)
                    latent = legacy.encode_to_slat(str(source_glb), num_views=views,
                                                  tmp_dir=str(folder / 'encoding'), sample_posterior=False)
                    encoded = {'feats': latent.feats.detach().float().cpu(),
                               'coords': latent.coords.detach().int().cpu()}
                    torch.save(encoded, cache)
                    del legacy, latent
                    torch.cuda.empty_cache()
                transform_path = folder / 'encoding/renders/transforms.json'
            else:
                encoded = codec.encode(source_glb, folder / 'encoding', num_views=views)
                transform_path = folder / 'encoding/renders/transforms.json'
            normalization = json.loads(transform_path.read_text())
            normalization = {k: normalization[k] for k in ('scale', 'offset')}
            write_json(out / 'status.json', {'stage': branch, 'phase': 'decoding'})
            decoded = codec.decode(encoded, folder / 'decoded', textured=True)
            write_json(out / 'status.json', {'stage': branch, 'phase': 'metrics'})
            metrics = compare_glbs(source_glb, folder / 'decoded/textured.glb', folder / 'comparison',
                                    normalization=normalization)
            vertex_metrics = compare_glbs(source_glb, folder / 'decoded/vertex.glb', folder / 'comparison_vertex',
                                           normalization=normalization)
            result['branches'][branch] = {'views': views, 'decode': decoded,
                'normalization': normalization, 'textured_metrics': metrics,
                'vertex_metrics': vertex_metrics, 'latent_diagnostics': latent_comparison(source, encoded),
                'seconds': time.perf_counter() - stage_start}
            write_json(out / 'partial_result.json', result)
        norm = result['branches']['official_150']['normalization']
        entries = [
            {'label': 'Direct / vertex', 'path': out / 'direct/vertex.glb', 'normalization': norm},
            {'label': 'Direct / official texture', 'path': source_glb, 'normalization': norm},
            {'label': 'Legacy 50 / vertex', 'path': out / 'legacy_50/decoded/vertex.glb'},
            {'label': 'Legacy 50 / texture', 'path': out / 'legacy_50/decoded/textured.glb'},
            {'label': 'Official 150 / vertex', 'path': out / 'official_150/decoded/vertex.glb'},
            {'label': 'Official 150 / texture', 'path': out / 'official_150/decoded/textured.glb'},
        ]
        render_montage(entries, out / 'comparison.png')
        result['seconds_total'] = time.perf_counter() - started
        write_json(out / 'result.json', result)
        write_json(out / 'status.json', {'stage': 'complete', 'seconds': result['seconds_total']})
        (out / 'error.json').unlink(missing_ok=True)
        print(f"COMPLETE {args.index}: {item['asset']} ({result['seconds_total']:.1f}s)", flush=True)
    except Exception as exc:
        write_json(out / 'error.json', {'type': type(exc).__name__, 'message': str(exc),
                                       'traceback': traceback.format_exc()})
        raise


def enhance(args):
    """Keep VAE geometry; optionally fit original GLB appearance with official tools."""
    from trellis_source_texture import transfer_source_texture
    from trellis_codec_metrics import compare_glbs, render_montage
    manifest = json.loads((args.output / 'manifest.json').read_text())
    items = manifest['assets'] if args.index < 0 else [manifest['assets'][args.index]]
    for item in items:
        out = args.output / f"{item['index']:02d}_{item['asset']}"
        target = out / 'source_texture'
        if all((target / name).is_file() for name in ('result.json', 'source_baked.glb', 'comparison.png')):
            continue
        result = json.loads((out / 'result.json').read_text())
        norm = result['branches']['official_150']['normalization']
        source = out / 'direct/textured.glb'
        started = time.perf_counter()
        meta = transfer_source_texture(source, out / 'official_150/decoded/textured.glb',
                                       target / 'source_baked.glb', normalization=norm,
                                       texture_size=manifest['export']['texture_size'], resolution=1024)
        metrics = compare_glbs(source, target / 'source_baked.glb', target / 'comparison', normalization=norm)
        write_json(target / 'result.json', {
            'views': 100, 'decode': meta, 'normalization': norm,
            'textured_metrics': metrics,
            'vertex_metrics': result['branches']['official_150']['vertex_metrics'],
            'latent_diagnostics': result['branches']['official_150']['latent_diagnostics'],
            'seconds': time.perf_counter() - started,
            'appearance_uses_original_glb': True,
            'interpretation': 'External original appearance transfer onto unchanged VAE geometry/UVs, using official bake_texture; not latent-only decoding.'})
        entries = [
            {'label': 'Direct / vertex', 'path': out / 'direct/vertex.glb', 'normalization': norm},
            {'label': 'Direct / official texture', 'path': source, 'normalization': norm},
            {'label': 'Legacy 50 / vertex', 'path': out / 'legacy_50/decoded/vertex.glb'},
            {'label': 'Legacy 50 / texture', 'path': out / 'legacy_50/decoded/textured.glb'},
            {'label': 'Official 150 / vertex', 'path': out / 'official_150/decoded/vertex.glb'},
            {'label': 'Official 150 / texture', 'path': out / 'official_150/decoded/textured.glb'},
            {'label': '150 / original appearance', 'path': target / 'source_baked.glb'},
        ]
        render_montage(entries, target / 'comparison.png')
        print(f"SOURCE_TEXTURE_COMPLETE {item['index']}: {item['asset']}", flush=True)


def status(args):
    manifest = json.loads((args.output / 'manifest.json').read_text())
    for item in manifest['assets']:
        out = args.output / f"{item['index']:02d}_{item['asset']}"
        state = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {'stage': 'pending'}
        state['index'] = item['index']
        state['asset'] = item['asset']
        for name in ('legacy_50', 'official_150'):
            folder = out / name / 'encoding/renders'
            state[name + '_rendered'] = len(list(folder.glob('[0-9][0-9][0-9].png')))
        state['source_texture_complete'] = (out / 'source_texture/result.json').is_file()
        print(json.dumps(state))


def report(args):
    manifest = json.loads((args.output / 'manifest.json').read_text())
    results, missing, failures = [], [], []
    for item in manifest['assets']:
        folder = args.output / f"{item['index']:02d}_{item['asset']}"
        if (folder / 'result.json').is_file():
            result = json.loads((folder / 'result.json').read_text())
            if (folder / 'source_texture/result.json').is_file():
                result['branches']['source_texture_150'] = json.loads((folder / 'source_texture/result.json').read_text())
            results.append(result)
        else:
            missing.append(item['asset'])
            if (folder / 'error.json').exists():
                failures.append({'asset': item['asset'], **json.loads((folder / 'error.json').read_text())})
    rows = []
    for r in results:
        for branch, b in r['branches'].items():
            row = {'asset': r['asset'], 'branch': branch, 'seconds': b['seconds']}
            for export in ('textured_metrics', 'vertex_metrics'):
                def flatten(value, prefix):
                    if isinstance(value, dict):
                        for key, nested in value.items():
                            flatten(nested, prefix + '_' + key)
                    elif isinstance(value, (float, int)):
                        row[prefix] = value
                flatten(b[export], export)
            row.update({f'latent_{k}': v for k, v in b['latent_diagnostics'].items()
                        if isinstance(v, (int, float))})
            rows.append(row)
    columns = sorted({k for row in rows for k in row})
    with (args.output / 'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    means = {}
    branches = manifest['branches'][1:] + (['source_texture_150'] if any(r['branch'] == 'source_texture_150' for r in rows) else [])
    for branch in branches:
        selected = [r for r in rows if r['branch'] == branch]
        means[branch] = {k: sum(r[k] for r in selected if k in r) / sum(k in r for r in selected)
                         for k in columns if k not in ('asset', 'branch') and any(k in r for r in selected)}
    summary = {'status': 'complete' if len(results) == manifest['count'] else 'incomplete',
               'completed': len(results), 'expected': manifest['count'], 'missing': missing,
               'failures': failures, 'mean_metrics': means,
               'original_appearance_transfer_completed': sum('source_texture_150' in r['branches'] for r in results),
               'branch_asset_counts': {branch: sum(row['branch'] == branch for row in rows) for branch in branches},
               'interpretation': 'Measures reconstruction from exported generated GLBs; no ground-truth original mesh is available.',
               'protocol_caveat': 'legacy_50 uses the existing zero-offset camera sequence; official_150 uses the official seeded random offset. The comparison changes both coverage and count, so improvements cannot be attributed to view count alone.',
               'diagnosis': [
                   'Legacy default Blender path did not point to the installed executable; now resolves BLENDER_BIN/PATH/workspace tools.',
                   'Some old eval GLBs omit official Z-up to Y-up rotation; canonical direct GLBs here use official axes.',
                   'Raw v4 and VAE encoder features are already decoder-scale; do not denormalize again.',
                   'Existing scripts export vertex colors; official textured export uses Gaussian decoder plus UV texture optimization.',
                   'SLat encoder is a lossy autoencoder: original latent equality after exporting and reencoding a mesh is not guaranteed.',
                   'An SS/SLat flow sampled at alpha=0 or 1 is a generative endpoint, separate from autoencoder reconstruction.',
               ]}
    write_json(args.output / 'summary.json', summary)
    from trellis_codec_overview import create_overview
    create_overview(args.output)
    # A portable gallery opens without a server and links every GLB and measurement.
    import html
    metric_keys = [('Chamfer ↓', 'textured_metrics_geometry_chamfer_l1'),
                   ('F-score 0.01 ↑', 'textured_metrics_geometry_fscore_0.01'),
                   ('Silhouette IoU ↑', 'textured_metrics_appearance_silhouette_iou'),
                   ('PSNR foreground ↑', 'textured_metrics_appearance_psnr_foreground_union'),
                   ('PSNR vertex foreground ↑', 'vertex_metrics_appearance_psnr_foreground_union')]
    table_rows = []
    for branch, values in means.items():
        cells = ''.join(f'<td>{values[key]:.5f}</td>' if key in values else '<td>—</td>' for _, key in metric_keys)
        count = summary['branch_asset_counts'][branch]
        table_rows.append(f'<tr><th>{branch}</th><td>{count}</td>{cells}</tr>')
    mean_table = '<table><tr><th>Protocollo</th><th>Asset misurati</th>' + ''.join(f'<th>{label}</th>' for label, _ in metric_keys) + '</tr>' + ''.join(table_rows) + '</table>'
    blocks = []
    for item in manifest['assets']:
        name = f"{item['index']:02d}_{item['asset']}"
        escaped = html.escape(item['asset'])
        links = ' | '.join(f'<a href="{name}/{p}">{label}</a>' for p, label in [
            ('direct/textured.glb', 'Direct GLB'), ('legacy_50/decoded/textured.glb', '50 views GLB'),
            ('official_150/decoded/textured.glb', '150 views GLB'), ('result.json', 'Metrics')])
        if (args.output / name / 'source_texture/result.json').is_file():
            links += f' | <a href="{name}/source_texture/source_baked.glb">150 + aspetto originale</a>'
            picture = 'source_texture/comparison.png'
        else:
            picture = 'comparison.png'
        blocks.append(f'<article><h2>{escaped}</h2><p>{links}</p><img loading="lazy" src="{name}/{picture}"></article>')
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TRELLIS v4 codec comparison</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%}article{background:white;padding:20px;margin:20px 0}a{color:#1259a7}table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #ccc}</style>'
        f'<h1>TRELLIS codec: {len(results)}/{manifest["count"]} assets</h1>'
        '<p>Confronto del ciclo latente v4 → GLB → encoder → decoder. Il riferimento è il GLB prodotto dal decoding diretto: il dataset contiene latenti generati da immagini, senza mesh originali.</p>'
        '<p><a href="preview.zip">Scarica anteprima con immagini e metriche</a> | '
        '<a href="overview.png">Tavola riassuntiva dei 10 asset</a> | <a href="metrics.csv">Metriche CSV</a> | '
        '<a href="summary.json">Riepilogo JSON</a> | <a href="manifest.json">Selezione degli asset</a></p>'
        '<p>Per controllare i risultati: confronta le immagini qui sotto, poi apri i GLB in Blender. '
        'In ogni cartella asset, direct/textured.glb è il riferimento; legacy_50/decoded/textured.glb è la ricostruzione a 50 viste; '
        'official_150/decoded/textured.glb è la ricostruzione ufficiale a 150 viste; '
        'source_texture/source_baked.glb conserva meglio i colori del GLB di partenza. '
        'result.json e i file comparison/metrics.json contengono i dettagli misurati.</p>'
        + mean_table +
        '<p>Diagnosi: il vecchio percorso Blender predefinito non era valido; alcuni GLB storici conservano Z-up senza la conversione ufficiale a Y-up. '
        'I latenti v4 e quelli dell’encoder sono già nella scala del decoder: una seconda denormalizzazione è errata. '
        'Il GLB con colori ai vertici differisce dall’export ufficiale con texture ottenute dal decoder Gaussian.</p>'
        '<p>La versione a 50 viste usa il codice esistente; quella a 150 usa le impostazioni ufficiali con offset casuale fissato dal seed. '
        'Il confronto misura l’effetto combinato del protocollo. Nessun miglioramento viene assunto prima delle misure. '
        'Mesh non decimate, texture 2048 px, posterior mean deterministico, nessun flow generativo. '
        'L’opzione source_texture_150 trasferisce i colori dal GLB originale sulla stessa geometria VAE: richiede l’aspetto originale e non è decoding dal solo latente. '
        'Chamfer su 20.000 punti per mesh; F-score con soglia 0,01 nel cubo unitario; PSNR/IoU su 8 viste comuni estranee all’encoding. '
        'Le ricostruzioni non vengono riallineate o riscalate separatamente.</p>'
        '<p>Righe: decoding diretto con colori ai vertici / diretto con texture / 50 viste con colori ai vertici / 50 con texture / '
        '150 viste con colori ai vertici / 150 con texture / 150 con colori recuperati dal GLB originale (dove completato). '
        'Le colonne mostrano le stesse otto camere di valutazione.</p>' + ''.join(blocks))
    from trellis_codec_preview import build_preview_zip
    build_preview_zip(args.output)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'worker', 'enhance', 'status', 'report'])
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--blender', type=Path, default=Path(os.environ.get('BLENDER_BIN', str(REPO.parents[1] / 'tools/blender-3.0.1-linux-x64/blender'))))
    args = parser.parse_args()
    args.output = args.output.resolve()
    {'prepare': prepare, 'worker': worker, 'enhance': enhance, 'status': status, 'report': report}[args.mode](args)


if __name__ == '__main__':
    main()
