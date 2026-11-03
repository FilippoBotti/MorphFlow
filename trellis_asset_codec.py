"""Encode existing GLBs with TRELLIS's training-data pipeline and reconstruct them.

This is a VAE round trip, not image-to-3D generation: no flow sampler, CFG, or
sampling steps are involved. SLat features are stored in raw decoder scale.
Only standard-library dependencies are imported until computation starts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


VERSION = 1
MODEL_ROOT = "microsoft/TRELLIS-image-large/ckpts"
ENCODER = "slat_enc_swin8_B_64l8_fp16"
MESH_DECODER = "slat_dec_mesh_swin8_B_64l8m256c_fp16"
GAUSSIAN_DECODER = "slat_dec_gs_swin8_B_64l8gs32_fp16"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_manifest(path, identity):
    path = Path(path)
    if path.is_file():
        old = json.loads(path.read_text())
        if old["identity"] != identity:
            raise ValueError(f"Output cache has different inputs/settings: {path.parent}")
        return old
    data = {"identity": identity, "status": "running"}
    write_json(path, data)
    return data


# Keep upstream cameras/normalization; optionally render existing appearance
# without adding a second lighting pass to textures baked from Gaussians.
_BLENDER_WRAPPER = r'''
import json
import os
import sys
from types import SimpleNamespace
import bpy

config = json.load(open(sys.argv[sys.argv.index('--') + 1]))
official = config.pop('official_script')
lighting = config.pop('lighting', 'official')
namespace = {'__name__': 'trellis_official_renderer', '__file__': official}
with open(official) as stream:
    source = stream.read()
    # Cycles renamed this RNA property; the value/renderer are unchanged.
    if hasattr(bpy.context.scene.cycles, 'pixel_filter_type'):
        source = source.replace('scene.cycles.filter_type', 'scene.cycles.pixel_filter_type')
    exec(compile(source, official, 'exec'), namespace)
original_load_object = namespace['load_object']

def load_object(path):
    original_load_object(path)
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not obj.data.vertex_colors:
            continue
        mesh = obj.data
        if mesh.materials and all(m is not None for m in mesh.materials):
            continue
        layer = mesh.vertex_colors.active or mesh.vertex_colors[0]
        material = bpy.data.materials.new(name=obj.name + '_VertexColor')
        material.use_nodes = True
        color = material.node_tree.nodes.new('ShaderNodeVertexColor')
        color.layer_name = layer.name
        bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        material.node_tree.links.new(color.outputs['Color'], bsdf.inputs['Base Color'])
        if not mesh.materials:
            mesh.materials.append(material)
        else:
            for index in range(len(mesh.materials)):
                if mesh.materials[index] is None:
                    mesh.materials[index] = material
        print('[codec] Enabled missing vertex-color material:', obj.name)
    if lighting == 'uniform_texture':
        for material in bpy.data.materials:
            if not material.use_nodes:
                material.use_nodes = True
            nodes, links = material.node_tree.nodes, material.node_tree.links
            bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
            if bsdf is None or output is None:
                raise RuntimeError('Uniform texture rendering requires a Principled material: ' + material.name)
            emission = nodes.new('ShaderNodeEmission')
            base = bsdf.inputs['Base Color']
            if base.is_linked:
                links.new(base.links[0].from_socket, emission.inputs['Color'])
            else:
                emission.inputs['Color'].default_value = base.default_value
            emission.inputs['Strength'].default_value = 1.0
            transparent = nodes.new('ShaderNodeBsdfTransparent')
            mix = nodes.new('ShaderNodeMixShader')
            alpha = bsdf.inputs['Alpha']
            if alpha.is_linked:
                links.new(alpha.links[0].from_socket, mix.inputs[0])
            else:
                mix.inputs[0].default_value = alpha.default_value
            links.new(transparent.outputs[0], mix.inputs[1])
            links.new(emission.outputs[0], mix.inputs[2])
            for link in list(output.inputs['Surface'].links):
                links.remove(link)
            links.new(mix.outputs[0], output.inputs['Surface'])

if lighting == 'uniform_texture':
    def init_lighting():
        for obj in list(bpy.data.objects):
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)
        scene = bpy.context.scene
        if scene.world:
            scene.world.use_nodes = True
            for node in scene.world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    node.inputs['Strength'].default_value = 0.0
        scene.display_settings.display_device = 'sRGB'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.view_settings.use_curve_mapping = False
        print('[codec] Uniform texture appearance: emission=1, no additional lights, Standard sRGB')
        return {}
    namespace['init_lighting'] = init_lighting

namespace['load_object'] = load_object
namespace['main'](SimpleNamespace(**config))
'''


class OfficialAssetCodec:
    """Lazy, reusable encoder/decoder using only the selected official checkout.

    ``encode(glb, outdir, num_views=150)`` returns CPU ``feats`` [N,8] and
    ``coords`` [N,4], with batch coordinate zero. ``decode(pack, outdir)``
    returns geometry, file paths and timings. GLBs are Y-up; mesh_zup.ply
    preserves the internal decoder coordinate frame for aligned comparisons.
    """

    def __init__(self, trellis_root=None, blender_bin=None, device="cuda",
                 dino_batch_size=4, texture_size=2048, seed=42, lighting="official"):
        default_root = Path(__file__).resolve().parent.parent / "TRELLIS"
        self.trellis_root = Path(trellis_root or os.environ.get("TRELLIS_REPO", default_root)).resolve()
        default_blender = Path(__file__).resolve().parents[2] / "tools/blender-3.0.1-linux-x64/blender"
        self.blender_bin = str(blender_bin or os.environ.get("BLENDER_BIN") or shutil.which("blender") or default_blender)
        self.device = device
        self.dino_batch_size = int(dino_batch_size)
        self.texture_size = int(texture_size)
        self.seed = int(seed)
        if lighting not in ("official", "uniform_texture"):
            raise ValueError("lighting must be official or uniform_texture")
        self.lighting = lighting
        if self.dino_batch_size < 1 or self.texture_size < 1:
            raise ValueError("dino_batch_size and texture_size must be positive")
        self._torch = None
        self._models = {}
        self._dino = None

    def _init(self):
        if self._torch is not None:
            return
        if not (self.trellis_root / "trellis/models/__init__.py").is_file():
            raise FileNotFoundError(f"Missing official TRELLIS checkout: {self.trellis_root}")
        existing = sys.modules.get("trellis")
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(self.trellis_root):
            raise RuntimeError(f"Wrong trellis package already imported: {existing.__file__}")
        sys.path.insert(0, str(self.trellis_root))
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        os.environ.setdefault("SPCONV_ALGO", "native")
        import torch
        import numpy as np
        import trellis
        from trellis import models
        from trellis.modules import sparse
        if not Path(trellis.__file__).resolve().is_relative_to(self.trellis_root):
            raise RuntimeError(f"Unexpected trellis package: {trellis.__file__}")
        self._torch, self._np, self._official_models, self._sparse = torch, np, models, sparse
        self._trellis_file = str(Path(trellis.__file__).resolve())
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Keep DINO/projected features in FP32 and model-configured VAE FP16.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run encoding/decoding inside a GPU allocation")

    def _model(self, name):
        self._init()
        if name not in self._models:
            started = time.monotonic()
            print(f"[codec] Loading official {name}", flush=True)
            self._models[name] = self._official_models.from_pretrained(f"{MODEL_ROOT}/{name}").to(self.device).eval()
            self._models[name].requires_grad_(False)
            print(f"[codec] Loaded {name} ({time.monotonic() - started:.1f}s)", flush=True)
        return self._models[name]

    def _ensure_dino(self):
        self._init()
        if self._dino is not None:
            return
        torch = self._torch
        local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if (local / "hubconf.py").is_file():
            model = torch.hub.load(str(local), "dinov2_vitl14_reg", source="local", pretrained=True)
        elif os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            raise FileNotFoundError(f"Missing offline DINOv2 cache: {local}")
        else:
            model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitl14_reg", pretrained=True, trust_repo=True)
        self._dino = model.to(self.device).eval().requires_grad_(False)

    def release(self):
        """Release model memory between independent stages, preserving disk caches."""
        self._models.clear()
        self._dino = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _render(self, glb, outdir, num_views):
        outdir.mkdir(parents=True, exist_ok=True)
        transforms = outdir / "transforms.json"
        if transforms.is_file() and (outdir / "mesh.ply").is_file():
            data = json.loads(transforms.read_text())
            if len(data["frames"]) == num_views and all((outdir / f["file_path"]).is_file() for f in data["frames"]):
                return data
        utils_path = self.trellis_root / "dataset_toolkits/utils.py"
        spec = importlib.util.spec_from_file_location("_official_dataset_utils", utils_path)
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        # Official rendering uses a random Hammersley offset; seed it explicitly.
        offset = tuple(self._np.random.RandomState(self.seed).rand(2))
        views = []
        for index in range(num_views):
            yaw, pitch = utils.sphere_hammersley_sequence(index, num_views, offset)
            views.append({"yaw": float(yaw), "pitch": float(pitch), "radius": 2.0,
                          "fov": math.radians(40)})
        config = {"official_script": str(self.trellis_root / "dataset_toolkits/blender_script/render.py"),
                  "views": json.dumps(views), "object": str(glb), "resolution": 512,
                  "output_folder": str(outdir), "engine": "CYCLES", "geo_mode": False,
                  "save_depth": False, "save_normal": False, "save_albedo": False,
                  "save_mist": False, "split_normal": False, "save_mesh": True,
                  "lighting": self.lighting}
        config_path, wrapper = outdir / "render_config.json", outdir / "official_render_wrapper.py"
        write_json(config_path, config)
        wrapper.write_text(_BLENDER_WRAPPER)
        command = [self.blender_bin, "-b", "--threads", os.environ.get("OMP_NUM_THREADS", "4"),
                   "--python-exit-code", "1", "-P", str(wrapper), "--", str(config_path)]
        print(f"[codec] Rendering {num_views} Cycles views ({self.lighting}): {glb}", flush=True)
        with (outdir / "blender.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not transforms.is_file() or not (outdir / "mesh.ply").is_file():
            raise RuntimeError(f"Official Blender rendering failed ({result.returncode}); see {outdir / 'blender.log'}")
        data = json.loads(transforms.read_text())
        if len(data["frames"]) != num_views:
            raise RuntimeError(f"Only {len(data['frames'])}/{num_views} rendered views")
        return data

    def _voxelize(self, mesh_path, outpath):
        import open3d as o3d
        import utils3d
        np = self._np
        if not outpath.is_file():
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                raise ValueError(f"Empty normalized mesh: {mesh_path}")
            vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh, voxel_size=1 / 64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
            indices = np.asarray([v.grid_index for v in grid.get_voxels()])
            if not len(indices) or not ((indices >= 0) & (indices < 64)).all():
                raise ValueError("Invalid or empty 64^3 surface voxelization")
            utils3d.io.write_ply(str(outpath), (indices + 0.5) / 64 - 0.5)
        # Read back in the exact order subsequently used for features, rather
        # than assuming the PLY reader preserves a separate Open3D voxel list.
        positions = utils3d.io.read_ply(str(outpath))[0].astype(np.float32)
        indices = ((positions + 0.5) * 64).astype(np.int32)
        if not len(indices) or not ((indices >= 0) & (indices < 64)).all():
            raise ValueError("Invalid voxel centers after PLY reload")
        return positions, indices

    def _features(self, render_dir, frames, positions, indices, outpath):
        np, torch = self._np, self._torch
        if outpath.is_file():
            with np.load(outpath) as saved:
                if not np.array_equal(saved["indices"].astype(np.int32), indices):
                    raise ValueError("Cached DINO indices do not match voxel PLY")
                return saved["patchtokens"].astype(np.float32)
        from PIL import Image
        import torch.nn.functional as F
        import utils3d
        self._ensure_dino()
        points = torch.from_numpy(positions).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        accumulated = torch.zeros((len(indices), 1024), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(frames), self.dino_batch_size):
                batch = frames[start:start + self.dino_batch_size]
                images, extrinsics, intrinsics = [], [], []
                for frame in batch:
                    with Image.open(render_dir / frame["file_path"]) as image:
                        rgba = np.asarray(image.convert("RGBA").resize((518, 518), Image.Resampling.LANCZOS), dtype=np.float32) / 255
                    images.append(torch.from_numpy(rgba[:, :, :3] * rgba[:, :, 3:]).permute(2, 0, 1))
                    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                    c2w[:3, 1:3] *= -1
                    extrinsics.append(torch.linalg.inv(c2w))
                    fov = torch.tensor(frame["camera_angle_x"])
                    intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
                inputs = (torch.stack(images).to(self.device) - mean) / std
                output = self._dino(inputs, is_training=True)
                tokens = output["x_prenorm"][:, 1 + self._dino.num_register_tokens:]
                tokens = tokens.permute(0, 2, 1).reshape(len(batch), 1024, 37, 37)
                uv = utils3d.torch.project_cv(points, torch.stack(extrinsics).to(self.device),
                                             torch.stack(intrinsics).to(self.device))[0] * 2 - 1
                sampled = F.grid_sample(tokens, uv.unsqueeze(1), mode="bilinear", align_corners=False)
                accumulated += sampled.squeeze(2).permute(0, 2, 1).sum(dim=0).float().cpu()
                print(f"[codec] DINO views {min(start + len(batch), len(frames))}/{len(frames)}", flush=True)
        features = (accumulated / len(frames)).numpy().astype(np.float16)
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite DINO features")
        temp = outpath.with_name(outpath.stem + ".tmp.npz")
        np.savez_compressed(temp, indices=indices.astype(np.uint8), patchtokens=features)
        temp.replace(outpath)
        return features.astype(np.float32)

    def encode(self, glb, outdir, num_views=150):
        self._init()
        torch, np = self._torch, self._np
        glb, outdir = Path(glb).expanduser().resolve(), Path(outdir).expanduser().resolve()
        if not glb.is_file():
            raise FileNotFoundError(glb)
        if int(num_views) < 1:
            raise ValueError("num_views must be positive")
        num_views = int(num_views)
        script = self.trellis_root / "dataset_toolkits/blender_script/render.py"
        identity = {"version": VERSION, "source": str(glb), "sha256": file_sha256(glb),
                    "renderer_sha256": file_sha256(script), "num_views": num_views,
                    "resolution": 512, "seed": self.seed, "encoder": ENCODER,
                    "feature_model": "dinov2_vitl14_reg", "feature_storage": "fp16",
                    "feature_kind": "x_prenorm", "pooling": "mean_all_views",
                    "sample_posterior": False, "voxel_resolution": 64,
                    "vertex_color_material_fallback": True}
        if self.lighting != "official":
            identity.update(lighting=self.lighting, emission_strength=1.0,
                            color_transform="Standard sRGB, exposure=0, gamma=1",
                            wrapper_sha256=hashlib.sha256(_BLENDER_WRAPPER.encode()).hexdigest())
        manifest = outdir / "encode.json"
        old = _guard_manifest(manifest, identity)
        latent_path = outdir / "latent.pt"
        if old.get("status") == "complete" and latent_path.is_file():
            return self.validate_pack(torch.load(latent_path, map_location="cpu", weights_only=True))
        start = time.monotonic()
        transform = self._render(glb, outdir / "renders", num_views)
        render_end = time.monotonic()
        positions, indices = self._voxelize(outdir / "renders/mesh.ply", outdir / "voxels.ply")
        voxel_end = time.monotonic()
        features = self._features(outdir / "renders", transform["frames"], positions, indices, outdir / "features.npz")
        feature_end = time.monotonic()
        coords = np.concatenate([np.zeros((len(indices), 1), dtype=np.int32), indices], axis=1)
        inputs = self._sparse.SparseTensor(feats=torch.from_numpy(features).to(self.device),
                                           coords=torch.from_numpy(coords).to(self.device))
        with torch.no_grad():
            latent = self._model(ENCODER)(inputs, sample_posterior=False)
        pack = self.validate_pack({"feats": latent.feats.float().cpu(), "coords": latent.coords.int().cpu()})
        pack["metadata"] = {"latent_scale": "raw_decoder", "identity": identity}
        temp = latent_path.with_suffix(".pt.tmp")
        torch.save(pack, temp)
        temp.replace(latent_path)
        write_json(manifest, {"identity": identity, "status": "complete", "latent": str(latent_path),
                             "num_voxels": len(indices), "trellis_import": self._trellis_file,
                             "normalization": {k: transform[k] for k in ("scale", "offset", "aabb")},
                             "seconds": {"render": render_end - start, "voxelize": voxel_end - render_end,
                                         "dino": feature_end - voxel_end, "vae": time.monotonic() - feature_end,
                                         "total": time.monotonic() - start}})
        return pack

    def validate_pack(self, pack):
        self._init()
        torch = self._torch
        feats = torch.as_tensor(pack["feats"]).detach().float().cpu().contiguous()
        coords = torch.as_tensor(pack["coords"]).detach().cpu()
        if feats.ndim != 2 or feats.shape[1] != 8 or not feats.shape[0] or not torch.isfinite(feats).all():
            raise ValueError("Expected finite raw SLat feats with shape [N,8]")
        if coords.ndim != 2 or coords.shape[0] != feats.shape[0] or coords.shape[1] not in (3, 4):
            raise ValueError("Expected SLat coords with shape [N,3] or [N,4]")
        if torch.is_floating_point(coords) and not torch.equal(coords, coords.round()):
            raise ValueError("Voxel coordinates must be integers")
        coords = coords.int().contiguous()
        if coords.shape[1] == 3:
            coords = torch.cat([torch.zeros((len(coords), 1), dtype=torch.int32), coords], dim=1)
        if (coords[:, 0] != 0).any() or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= 64).any():
            raise ValueError("Expected one batch and spatial coordinates in [0,63]")
        if len(torch.unique(coords, dim=0)) != len(coords):
            raise ValueError("Duplicate sparse voxel coordinates")
        return {"feats": feats, "coords": coords}

    def decode(self, pack, outdir, textured=True):
        self._init()
        import trimesh
        torch, np = self._torch, self._np
        pack = self.validate_pack(pack)
        outdir = Path(outdir).expanduser().resolve()
        digest = hashlib.sha256(pack["feats"].numpy().tobytes() + pack["coords"].numpy().tobytes()).hexdigest()
        identity = {"version": VERSION, "latent_sha256": digest, "latent_scale": "raw_decoder",
                    "mesh_decoder": MESH_DECODER, "textured": bool(textured), "seed": self.seed,
                    "gaussian_decoder": GAUSSIAN_DECODER if textured else None,
                    "texture_size": self.texture_size, "simplify": 0.0, "fill_holes": False,
                    "trellis_root": str(self.trellis_root)}
        manifest = outdir / "decode.json"
        old = _guard_manifest(manifest, identity)
        expected = [outdir / "vertex.glb", outdir / "mesh_zup.ply"]
        if textured:
            expected.append(outdir / "textured.glb")
        if old.get("status") == "complete" and all(p.is_file() and p.stat().st_size for p in expected):
            return old
        start = time.monotonic()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        inputs = self._sparse.SparseTensor(feats=pack["feats"].to(self.device), coords=pack["coords"].to(self.device))
        with torch.no_grad():
            mesh = self._model(MESH_DECODER)(inputs)[0]
        if not mesh.success or not torch.isfinite(mesh.vertices).all():
            raise RuntimeError("Official mesh decoder produced an empty or nonfinite mesh")
        vertices = mesh.vertices.detach().float().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        attrs = mesh.vertex_attrs
        attrs = attrs.detach().float().cpu().numpy() if attrs is not None else None
        colors = None
        if attrs is not None:
            # Upstream MeshRenderer: :3 is color and 3: is normal_map.
            rgb = np.clip(attrs[:, :3], 0, 1)
            colors = np.concatenate([(rgb * 255).astype(np.uint8), np.full((len(rgb), 1), 255, np.uint8)], axis=1)
        trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False).export(outdir / "mesh_zup.ply")
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        vertex_mesh = trimesh.Trimesh(vertices @ rotation, faces, vertex_colors=colors, process=False)
        vertex_mesh.export(outdir / "vertex.glb")
        mesh_end = time.monotonic()
        print(f"[codec] Mesh decoded/exported: {len(vertices)} vertices, {len(faces)} faces "
              f"({mesh_end - start:.1f}s)", flush=True)
        paths = {"vertex_glb": str(outdir / "vertex.glb"), "mesh_zup": str(outdir / "mesh_zup.ply")}
        if textured:
            with torch.no_grad():
                gaussian = self._model(GAUSSIAN_DECODER)(inputs)[0]
            print(f"[codec] Gaussian appearance decoded ({time.monotonic() - mesh_end:.1f}s)", flush=True)
            from trellis.utils import postprocessing_utils
            # Baking optimizes a texture tensor. Enabling grad explicitly is
            # required even when the caller disabled global grad for inference.
            texture_started = time.monotonic()
            print(f"[codec] Starting official to_glb: full geometry, {self.texture_size}px texture "
                  "(UV parametrization, 100 appearance views, texture optimization)", flush=True)
            with torch.inference_mode(False), torch.enable_grad():
                textured_mesh = postprocessing_utils.to_glb(gaussian, mesh, simplify=0.0,
                                                            fill_holes=False,
                                                            texture_size=self.texture_size, verbose=False)
            print(f"[codec] Official texture baked ({time.monotonic() - texture_started:.1f}s)", flush=True)
            textured_mesh.export(outdir / "textured.glb")
            paths["textured_glb"] = str(outdir / "textured.glb")
        data = {"identity": identity, "status": "complete", "paths": paths,
                "vertices": int(len(vertices)), "faces": int(len(faces)),
                "voxels": int(len(pack["coords"])), "trellis_import": self._trellis_file,
                "axis_convention": "GLB Y-up, mesh_zup.ply internal Z-up",
                "vertex_rgb_channels": [0, 1, 2],
                "vertex_attrs_min": attrs.min(axis=0).tolist() if attrs is not None else None,
                "vertex_attrs_max": attrs.max(axis=0).tolist() if attrs is not None else None,
                "seconds": {"mesh": mesh_end - start, "texture": time.monotonic() - mesh_end,
                            "total": time.monotonic() - start}}
        write_json(manifest, data)
        print(f"[codec] Decoded {len(vertices)} vertices / {len(faces)} faces: {outdir}", flush=True)
        return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("encode", "decode", "roundtrip"))
    parser.add_argument("input", type=Path, help="GLB for encode/roundtrip, raw latent .pt/.npz for decode")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trellis-root", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--views", type=int, default=150)
    parser.add_argument("--lighting", choices=("official", "uniform_texture"), default="official",
                        help="uniform_texture preserves baked RGB without additional shading")
    parser.add_argument("--dino-batch-size", type=int, default=4)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-texture", action="store_true", help="Export vertex colors only")
    parser.add_argument("--preserve-input-texture", action="store_true",
                        help="For roundtrip, also fit original GLB appearance onto reconstructed geometry using the official texture baker")
    args = parser.parse_args(argv)
    if args.preserve_input_texture and (args.operation != "roundtrip" or args.no_texture):
        parser.error("--preserve-input-texture requires roundtrip with textured export")
    codec = OfficialAssetCodec(trellis_root=args.trellis_root, blender_bin=args.blender,
                               dino_batch_size=args.dino_batch_size,
                               texture_size=args.texture_size, seed=args.seed, lighting=args.lighting)
    if args.operation in ("encode", "roundtrip"):
        pack = codec.encode(args.input, args.output_dir, args.views)
        if args.operation == "roundtrip":
            codec.release()
            codec.decode(pack, args.output_dir / "reconstruction", textured=not args.no_texture)
            if args.preserve_input_texture:
                codec.release()
                from trellis_source_texture import transfer_source_texture
                transform = json.loads((args.output_dir / "renders/transforms.json").read_text())
                normalization = {key: transform[key] for key in ("scale", "offset")}
                transfer_source_texture(args.input, args.output_dir / "reconstruction/textured.glb",
                                        args.output_dir / "reconstruction/source_baked.glb", normalization,
                                        trellis_root=codec.trellis_root,
                                        texture_size=args.texture_size, seed=args.seed)
    else:
        codec._init()
        if args.input.suffix == ".npz":
            with codec._np.load(args.input) as data:
                pack = {"feats": data["feats"], "coords": data["coords"]}
        else:
            pack = codec._torch.load(args.input, map_location="cpu", weights_only=True)
        codec.decode(pack, args.output_dir, textured=not args.no_texture)


if __name__ == "__main__":
    main()
