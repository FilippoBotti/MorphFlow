"""Create matte/unlit GLB copies; preserve geometry, UVs and texture bytes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct


def read_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError('Truncated GLB')
    magic, version, length = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise ValueError('Expected a complete GLB 2.0 file')
    size, kind = struct.unpack_from('<II', data, 12)
    end = 20 + size
    if kind != 0x4E4F534A or size % 4 or end > len(data):
        raise ValueError('Invalid JSON chunk')
    document = json.loads(data[20:end])
    if not document.get('materials'):
        raise ValueError('Expected an explicitly assigned GLB material')
    return data, document, data[end:]


def write_glb(path, document, tail):
    chunk = json.dumps(document, separators=(',', ':'), allow_nan=False).encode('utf-8')
    chunk += b' ' * (-len(chunk) % 4)
    data = struct.pack('<4sII', b'glTF', 2, 20 + len(chunk) + len(tail))
    data += struct.pack('<II', len(chunk), 0x4E4F534A) + chunk + tail
    temporary = path.with_suffix('.tmp.glb')
    temporary.write_bytes(data)
    temporary.replace(path)


def material_document(document, mode):
    if mode not in ('matte', 'unlit'):
        raise ValueError('Expected matte or unlit')
    variant = copy.deepcopy(document)
    for material in variant['materials']:
        pbr = material.setdefault('pbrMetallicRoughness', {})
        pbr.update(metallicFactor=0.0, roughnessFactor=1.0)
        if mode == 'unlit':
            material.setdefault('extensions', {})['KHR_materials_unlit'] = {}
        else:
            material.get('extensions', {}).pop('KHR_materials_unlit', None)
    for key in ('extensionsUsed', 'extensionsRequired'):
        names = [x for x in variant.get(key, []) if x != 'KHR_materials_unlit']
        if mode == 'unlit' and key == 'extensionsUsed':
            names.append('KHR_materials_unlit')
        if names:
            variant[key] = names
        else:
            variant.pop(key, None)
    return variant


def make_variants(source, output_dir=None):
    source = Path(source).resolve()
    original, document, tail = read_glb(source)
    output_dir = Path(output_dir).resolve() if output_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {'source': str(source), 'source_sha256': hashlib.sha256(original).hexdigest(),
              'binary_chunks_sha256': hashlib.sha256(tail).hexdigest(), 'outputs': {}}
    for mode in ('matte', 'unlit'):
        variant = material_document(document, mode)
        path = output_dir / (source.stem + '_' + mode + '.glb')
        write_glb(path, variant, tail)
        _, checked, checked_tail = read_glb(path)
        if checked_tail != tail:
            raise RuntimeError('Binary payload changed')
        for key in ('meshes', 'accessors', 'bufferViews', 'buffers', 'images', 'textures', 'nodes', 'scenes'):
            if checked.get(key) != document.get(key):
                raise RuntimeError('Asset data changed: ' + key)
        result['outputs'][mode] = str(path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(make_variants(args.source, args.output_dir), indent=2))
