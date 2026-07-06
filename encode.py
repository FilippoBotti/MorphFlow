from typing import Optional
import os
import json
import subprocess
import tempfile

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import open3d as o3d
import utils3d

from TRELLIS.trellis import models
from TRELLIS.trellis.modules import sparse as sp
from dataset_toolkits.utils import sphere_hammersley_sequence
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


class TrellisSLatEncoderDebug:
    def __init__(
        self,
        device: str = "cuda",
        slat_encoder=None,
        blender_bin: Optional[str] = None,
    ):
        self.device = torch.device(device)

        self._slat_encoder = slat_encoder or models.from_pretrained(
            "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
        )
        self._slat_encoder = self._slat_encoder.to(self.device).eval()

        self._dino_model = None
        self._dino_transform = None
        self._blender_bin = blender_bin

    def _ensure_dino(self):
        if self._dino_model is not None:
            return

        self._dino_model = (
            torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitl14_reg",
                pretrained=True,
            )
            .eval()
            .to(self.device)
        )

        self._dino_transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _ensure_blender(self) -> str:
        if self._blender_bin is not None:
            return self._blender_bin

        blender_dir = os.path.join(_PROJECT_ROOT, "workspace_encoding", "blender")
        self._blender_bin = os.path.join(
            blender_dir,
            "blender-3.0.1-linux-x64",
            "blender",
        )
        return self._blender_bin

    def _render_views(self, mesh_path: str, output_dir: str, num_views: int = 50):
        blender_bin = self._ensure_blender()

        os.makedirs(output_dir, exist_ok=True)

        views = []
        for i in range(num_views):
            yaw, pitch = sphere_hammersley_sequence(i, num_views)
            views.append({
                "yaw": float(yaw),
                "pitch": float(pitch),
                "radius": 2.0,
                "fov": float(40 / 180 * np.pi),
            })

        script_path = os.path.join(
            _PROJECT_ROOT,
            "dataset_toolkits",
            "blender_script",
            "render.py",
        )

        cmd = [
            blender_bin,
            "-b",
            "-P",
            script_path,
            "--",
            "--views",
            json.dumps(views),
            "--object",
            os.path.abspath(mesh_path),
            "--resolution",
            "512",
            "--output_folder",
            os.path.abspath(output_dir),
            "--engine",
            "CYCLES",
            "--save_mesh",
        ]

        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        return os.path.abspath(output_dir)

    def _voxelize_ply(
        self,
        mesh_ply_path: str,
        output_ply: str,
        resolution: int = 64,
    ):
        mesh = o3d.io.read_triangle_mesh(mesh_ply_path)

        vertices = np.clip(
            np.asarray(mesh.vertices),
            -0.5 + 1e-6,
            0.5 - 1e-6,
        )
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1.0 / resolution,
            min_bound=np.array([-0.5, -0.5, -0.5]),
            max_bound=np.array([0.5, 0.5, 0.5]),
        )

        indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
        indices = indices[((indices >= 0) & (indices < resolution)).all(axis=1)]

        centres = (indices + 0.5) / resolution - 0.5
        utils3d.io.write_ply(output_ply, centres)

        return indices

    @torch.no_grad()
    def _extract_dino_features(
        self,
        render_dir: str,
        voxel_ply_path: str,
        batch_size: int = 8,
    ):
        self._ensure_dino()

        positions = torch.from_numpy(
            utils3d.io.read_ply(voxel_ply_path)[0]
        ).float().to(self.device)

        with open(os.path.join(render_dir, "transforms.json"), "r") as f:
            frames = json.load(f)["frames"]

        patchtokens_acc = []

        for i in tqdm(range(0, len(frames), batch_size), desc="DINOv2 features"):
            batch = frames[i:i + batch_size]

            imgs = []
            extr = []
            intr = []

            for frame in batch:
                img_path = os.path.join(render_dir, frame["file_path"])

                pil_img = Image.open(img_path).resize(
                    (518, 518),
                    Image.Resampling.LANCZOS,
                )

                img_np = np.array(pil_img).astype(np.float32) / 255.0

                if img_np.shape[-1] == 4:
                    img_np = img_np[:, :, :3] * img_np[:, :, 3:]
                else:
                    img_np = img_np[:, :, :3]

                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
                imgs.append(self._dino_transform(img_t))

                c2w = torch.tensor(frame["transform_matrix"])
                c2w[:3, 1:3] *= -1
                extr.append(torch.inverse(c2w))

                fov = frame["camera_angle_x"]
                intr.append(
                    utils3d.torch.intrinsics_from_fov_xy(
                        torch.tensor(fov),
                        torch.tensor(fov),
                    )
                )

            imgs_t = torch.stack(imgs).to(self.device)
            extr_t = torch.stack(extr).to(self.device)
            intr_t = torch.stack(intr).to(self.device)

            features = self._dino_model(imgs_t, is_training=True)

            n_reg = getattr(self._dino_model, "num_register_tokens", 0)
            raw = features["x_prenorm"][:, 1 + n_reg:, :]

            b, n, c = raw.shape
            h = int(np.sqrt(n))

            patch_tokens = raw.permute(0, 2, 1).reshape(b, c, h, h)

            uv = utils3d.torch.project_cv(positions, extr_t, intr_t)
            uv = (uv[0] if isinstance(uv, tuple) else uv) * 2 - 1

            sampled = F.grid_sample(
                patch_tokens,
                uv.unsqueeze(1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(2).permute(0, 2, 1)

            patchtokens_acc.append(sampled.cpu())

        return torch.cat(patchtokens_acc, dim=0).mean(dim=0).numpy()

    @torch.no_grad()
    def encode_to_slat(
        self,
        mesh_path: str,
        num_views: int = 50,
        tmp_dir: Optional[str] = None,
        sample_posterior: bool = False,
    ):
        cleanup = tmp_dir is None

        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="trellis_slat_")

        os.makedirs(tmp_dir, exist_ok=True)

        try:
            render_dir = self._render_views(
                mesh_path,
                os.path.join(tmp_dir, "renders"),
                num_views=num_views,
            )

            normalised_mesh = os.path.join(render_dir, "mesh.ply")
            voxel_ply = os.path.join(tmp_dir, "voxels.ply")

            indices = self._voxelize_ply(normalised_mesh, voxel_ply)
            features = self._extract_dino_features(render_dir, voxel_ply)

            coords = torch.cat([
                torch.zeros(len(indices), 1),
                torch.from_numpy(indices).float(),
            ], dim=1).int()

            inputs = sp.SparseTensor(
                feats=torch.from_numpy(features).float(),
                coords=coords,
            ).to(self.device)

            return self._slat_encoder(
                inputs,
                sample_posterior=sample_posterior,
            )

        finally:
            if cleanup:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)