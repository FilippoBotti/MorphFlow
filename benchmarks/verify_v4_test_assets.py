"""Verify v4 test GLBs on CPU and write their download gallery.

Geometry, UVs, indices, embedded texture dimensions and split membership are
checked from the actual GLB bytes. Pillow is used only when making an overview.
"""
from __future__ import annotations

import argparse
import array
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import struct
import sys
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_glb(path, texture_size=2048):
    """Validate one self-contained, indexed GLB; return measured properties."""
    payload = Path(path).read_bytes()
    if len(payload) < 20:
        raise ValueError('Truncated GLB header')
    magic, version, declared_length = struct.unpack_from('<4sII', payload)
    if magic != b'glTF' or version != 2 or declared_length != len(payload):
        raise ValueError('Invalid GLB magic/version/length')
    chunks = []
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError('Truncated chunk header')
        length, kind = struct.unpack_from('<I4s', payload, offset)
        offset += 8
        if length % 4 or offset + length > len(payload):
            raise ValueError('Invalid chunk alignment/length')
        chunks.append((kind, memoryview(payload)[offset:offset + length]))
        offset += length
    if [kind for kind, _ in chunks] != [b'JSON', b'BIN\x00']:
        raise ValueError('Expected one JSON chunk and one embedded BIN chunk')
    document = json.loads(bytes(chunks[0][1]))
    binary = chunks[1][1]
    buffers = document.get('buffers', [])
    if len(buffers) != 1 or buffers[0].get('uri') or not 0 <= len(binary) - buffers[0]['byteLength'] <= 3:
        raise ValueError('Expected one embedded buffer with valid length')
    buffer_length = buffers[0]['byteLength']
    views = document.get('bufferViews', [])
    for view in views:
        start, length = view.get('byteOffset', 0), view['byteLength']
        if view.get('buffer', 0) != 0 or start < 0 or length < 0 or start + length > buffer_length:
            raise ValueError('Buffer view outside the embedded buffer')

    component = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2),
                 5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4)}
    widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
              'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
    accessors = document.get('accessors', [])
    cache = {}

    def accessor(index):
        if index in cache:
            return cache[index]
        descriptor = accessors[index]
        if 'sparse' in descriptor or 'bufferView' not in descriptor:
            raise ValueError('Sparse/missing accessor buffers are unsupported in this export')
        code, size = component[descriptor['componentType']]
        width = widths[descriptor['type']]
        count = descriptor['count']
        if count <= 0:
            raise ValueError('Empty accessor')
        view = views[descriptor['bufferView']]
        local = descriptor.get('byteOffset', 0)
        step = view.get('byteStride', size * width)
        if local < 0 or step < size * width or step % size:
            raise ValueError('Invalid accessor offset/stride')
        end = local + (count - 1) * step + size * width
        if end > view['byteLength']:
            raise ValueError('Accessor extends beyond its buffer view')
        start = view.get('byteOffset', 0) + local
        values = array.array(code)
        if step == size * width:
            values.frombytes(binary[start:start + count * step])
        else:
            unpacker = struct.Struct('<' + code * width)
            for i in range(count):
                values.extend(unpacker.unpack_from(binary, start + i * step))
        if sys.byteorder != 'little' and step == size * width:
            values.byteswap()
        if descriptor['componentType'] == 5126 and not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite floating-point accessor')
        cache[index] = (descriptor, values)
        return descriptor, values

    # Include all stored attributes, not only POSITION and TEXCOORD_0.
    for index in range(len(accessors)):
        accessor(index)

    image_rows = []
    for image in document.get('images', []):
        if image.get('uri') or 'bufferView' not in image:
            raise ValueError('Texture is not embedded in the GLB')
        view = views[image['bufferView']]
        start = view.get('byteOffset', 0)
        data = binary[start:start + view['byteLength']]
        if len(data) < 33 or bytes(data[:8]) != b'\x89PNG\r\n\x1a\n' or bytes(data[12:16]) != b'IHDR':
            raise ValueError('Expected an embedded PNG texture')
        dimensions = list(struct.unpack_from('>II', data, 16))
        if dimensions != [texture_size, texture_size]:
            raise ValueError('Unexpected texture dimensions: ' + str(dimensions))
        image_rows.append({'mime_type': image.get('mimeType'), 'width': dimensions[0],
                           'height': dimensions[1], 'embedded': True})
    if not image_rows:
        raise ValueError('No embedded texture')

    primitives = []
    for mesh in document.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            if primitive.get('mode', 4) != 4:
                raise ValueError('Expected triangle primitives')
            attrs = primitive['attributes']
            position, coordinates = accessor(attrs['POSITION'])
            uv, texcoords = accessor(attrs['TEXCOORD_0'])
            indices, values = accessor(primitive['indices'])
            if position['type'] != 'VEC3' or position['componentType'] != 5126:
                raise ValueError('Expected float VEC3 vertex positions')
            if uv['type'] != 'VEC2' or uv['componentType'] != 5126 or uv['count'] != position['count']:
                raise ValueError('UV accessor does not match the vertices')
            if min(texcoords) < -1e-5 or max(texcoords) > 1.00001:
                raise ValueError('UV coordinates outside the texture atlas')
            if indices['type'] != 'SCALAR' or indices['componentType'] not in (5121, 5123, 5125):
                raise ValueError('Invalid triangle index type')
            if indices['count'] % 3 or max(values) >= position['count']:
                raise ValueError('Invalid triangle indices')
            material = document['materials'][primitive['material']]
            texture = material['pbrMetallicRoughness']['baseColorTexture']
            if texture.get('texCoord', 0) != 0:
                raise ValueError('Material does not use TEXCOORD_0')
            texture_source = document['textures'][texture['index']]['source']
            if not 0 <= texture_source < len(image_rows):
                raise ValueError('Invalid material texture source')
            primitives.append({'vertices': position['count'], 'triangles': indices['count'] // 3,
                               'position_bounds': [[min(coordinates[i::3]) for i in range(3)],
                                                   [max(coordinates[i::3]) for i in range(3)]],
                               'uv_range': [min(texcoords), max(texcoords)],
                               'material_extensions': sorted(material.get('extensions', {}))})
    if not primitives:
        raise ValueError('GLB has no mesh primitives')
    return {'file_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            'mesh_count': len(document['meshes']), 'primitive_count': len(primitives),
            'vertices': sum(p['vertices'] for p in primitives),
            'triangles': sum(p['triangles'] for p in primitives),
            'images': image_rows, 'primitives': primitives}


def gallery(output, assets, report):
    cards = []
    for item in report['assets']:
        name = item['asset']
        file_name = name + '.glb'
        preview = output / 'previews' / (name + '.png')
        picture = ''
        if preview.is_file():
            source = escape(os.path.relpath(preview, assets), quote=True)
            picture = '<img loading="lazy" src="' + source + '" alt="' + escape(name, quote=True) + '">'
        if item['status'] == 'valid':
            details = '%s vertici · %s triangoli · texture 2048 px' % (item['vertices'], item['triangles'])
            link = '<a href="' + escape(file_name, quote=True) + '">Scarica GLB</a>'
        else:
            details = '; '.join(item.get('issues', [])) or 'In attesa'
            link = ''
        cards.append('<article>' + picture + '<h2>' + escape(name) + '</h2><p>' + escape(details) + '</p>' + link + '</article>')
    html = ('<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>50 asset test v4 — Gaussian texture 2048</title><style>'
            'body{font:16px system-ui;background:#edf0f4;color:#172234;margin:24px}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}'
            'article{background:white;padding:16px;border-radius:8px;overflow-wrap:anywhere}'
            'img{width:100%;height:280px;object-fit:contain}h2{font-size:16px}a{color:#1259a7}</style>' +
            '<h1>Asset del test v4: %d/%d verificati</h1>' % (report['valid_count'], report['expected_count']) +
            '<p>Decoding diretto degli SLat v4 e texture 2048 × 2048 ottenute dalle Gaussiane originali con l’export ufficiale TRELLIS.</p>'
            '<p>Mesh completa, senza decimazione o chiusura dei fori; GLB Y-up.</p><main>' + ''.join(cards) + '</main></html>')
    target = assets / 'index.html'
    temporary = target.with_name('.index.' + uuid.uuid4().hex + '.html')
    temporary.write_text(html)
    temporary.replace(target)


def overview(output, report):
    if not any((output / 'previews' / (a['asset'] + '.png')).is_file() for a in report['assets']):
        return None
    from PIL import Image, ImageDraw
    from trellis_codec_overview import _font, _wrapped
    columns, tile, padding, label = 5, 240, 14, 90
    row_height = tile + label + padding
    canvas = Image.new('RGB', (columns * (tile + padding) + padding,
                               100 + math.ceil(len(report['assets']) / columns) * row_height), '#edf0f4')
    draw = ImageDraw.Draw(canvas)
    title, body = _font(24, True), _font(13)
    draw.text((padding, 14), 'Test v4: %d/%d GLB verificati' % (report['valid_count'], report['expected_count']),
              font=title, fill='#172234')
    draw.text((padding, 52), 'Decoding diretto · Gaussiane originali · texture 2048 px · geometria completa', font=body, fill='#334155')
    for index, item in enumerate(report['assets']):
        x = padding + index % columns * (tile + padding)
        y = 100 + index // columns * row_height
        draw.rectangle((x, y, x + tile, y + tile + label), fill='white')
        path = output / 'previews' / (item['asset'] + '.png')
        if path.is_file():
            with Image.open(path) as picture:
                picture = picture.convert('RGB')
                picture.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(picture, (x + (tile - picture.width) // 2, y + (tile - picture.height) // 2))
        else:
            draw.text((x + 16, y + tile // 2), 'In attesa', font=body, fill='#64748b')
        for j, line in enumerate(_wrapped(draw, item['asset'].replace('-', ' '), body, tile - 16)[:4]):
            draw.text((x + 8, y + tile + 8 + j * 17), line, font=body, fill='#334155')
    target = output / 'overview.png'
    temporary = target.with_name('.overview.' + uuid.uuid4().hex + '.png')
    canvas.save(temporary)
    temporary.replace(target)
    return str(target)


def verify(output, assets, allow_incomplete=False):
    output, assets = Path(output).resolve(), Path(assets).resolve()
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    dataset = Path(manifest['dataset'])
    metadata_path = dataset / 'metadata_test.json'
    metadata = json.loads(metadata_path.read_text())
    test = set(json.loads((dataset / 'split_assets.json').read_text())['test'])
    endpoints = {row[key] for row in metadata for key in ('src_1', 'src_2')}
    names = [a['asset'] for a in manifest['assets']]
    issues = []
    if len(names) != manifest['count'] or len(names) != len(set(names)):
        issues.append('Manifest count/unique asset names mismatch')
    if manifest.get('split') != 'test':
        issues.append('Manifest split is not test')
    metadata_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if manifest.get('metadata_sha256') and manifest['metadata_sha256'] != metadata_digest:
        issues.append('Dataset metadata hash differs from the selection manifest')
    rows = []
    for item in manifest['assets']:
        name = item['asset']
        path = assets / (name + '.glb')
        row = {'index': item['index'], 'asset': name, 'glb': str(path),
               'test_split': name in test, 'test_endpoint': name in endpoints, 'issues': []}
        if not row['test_split'] or not row['test_endpoint']:
            row['issues'].append('Asset is not a v4 test endpoint')
            issues.append('Selection contains an asset outside the v4 test endpoints: ' + name)
        if not path.is_file():
            row['status'] = 'pending'
            row['issues'].append('GLB missing')
        else:
            try:
                row.update(read_glb(path, texture_size=manifest['export']['texture_size']))
            except (KeyError, IndexError, ValueError, struct.error, OSError) as exc:
                row['issues'].append(type(exc).__name__ + ': ' + str(exc))
            try:
                provenance = json.loads(path.with_suffix('.json').read_text())
                identity = provenance['decode']['identity']
                expected_identity = {'texture_size': manifest['export']['texture_size'],
                                     'simplify': 0, 'fill_holes': False, 'textured': True,
                                     'latent_scale': 'raw_decoder', 'seed': manifest['seed'],
                                     'mesh_decoder': 'slat_dec_mesh_swin8_B_64l8m256c_fp16',
                                     'gaussian_decoder': 'slat_dec_gs_swin8_B_64l8gs32_fp16'}
                if any(identity.get(k) != v for k, v in expected_identity.items()):
                    row['issues'].append('Export provenance does not match the requested pipeline')
                if (provenance.get('status') != 'complete' or provenance.get('asset') != name
                        or provenance.get('split') != 'test' or provenance['decode'].get('status') != 'complete'):
                    row['issues'].append('Invalid asset provenance status/name/split')
                if row.get('sha256') and provenance.get('sha256') != row['sha256']:
                    row['issues'].append('GLB hash differs from its provenance')
                if not identity.get('latent_sha256') or not identity.get('gaussian_decoder'):
                    row['issues'].append('Missing original SLat/Gaussian decoder provenance')
                if Path(provenance['source_slat']).resolve() != Path(item['directory']).resolve():
                    row['issues'].append('Provenance source differs from the selected dataset asset')
                row['provenance'] = {'method': provenance.get('method'), 'source_slat': provenance.get('source_slat'),
                                     'reference_overview': provenance.get('reference_overview'), 'decode_identity': identity}
            except (KeyError, ValueError, OSError) as exc:
                row['issues'].append('Invalid provenance: ' + str(exc))
            row['status'] = 'invalid' if row['issues'] else 'valid'
        rows.append(row)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'manifest': str(manifest_path), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'dataset': str(dataset), 'metadata_sha256': metadata_digest,
              'asset_directory': str(assets), 'expected_count': manifest['count'],
              'present_count': sum(row['status'] != 'pending' for row in rows),
              'valid_count': sum(row['status'] == 'valid' for row in rows),
              'invalid_count': sum(row['status'] == 'invalid' for row in rows),
              'pending_count': sum(row['status'] == 'pending' for row in rows),
              'issues': issues, 'assets': rows}
    report['status'] = 'complete' if not issues and report['valid_count'] == report['expected_count'] else 'incomplete'
    assets.mkdir(parents=True, exist_ok=True)
    write_json(assets / 'manifest.json', manifest)
    gallery(output, assets, report)
    report['overview'] = overview(output, report)
    write_json(output / 'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'assets'}, indent=2))
    if issues or report['invalid_count'] or (report['pending_count'] and not allow_incomplete):
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Run directory containing manifest.json')
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing <asset>.glb')
    parser.add_argument('--allow-incomplete', action='store_true', help='Allow missing GLBs, while still rejecting invalid ones')
    args = parser.parse_args()
    raise SystemExit(verify(args.output, args.assets, args.allow_incomplete))


if __name__ == '__main__':
    main()
