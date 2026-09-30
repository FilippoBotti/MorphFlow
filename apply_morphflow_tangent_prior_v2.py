from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path.cwd()
FILES = [
    "models/trellis_ss_prior.py",
    "modules/trellis_prior_training.py",
    "modules/prior_gradient_balance.py",
    "train.py",
    "slurm/train_morphflow_v3_prior_strong.slurm",
    "tools/launch_prior_strong.py",
    "tests/test_prior_gradient_balance.py",
]
for rel in FILES:
    if not (ROOT / rel).is_file():
        raise SystemExit(f"Missing {rel}; run from MorphFlow repository root.")

head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
print("HEAD:", head)

stamp = time.strftime("%Y%m%d-%H%M%S")
backup = ROOT / ".git" / "morphflow_patches" / f"tangent-prior-v2-{stamp}"
backup.mkdir(parents=True, exist_ok=False)
for rel in FILES:
    dst = backup / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)
print("Backup:", backup)


def replace_once(rel, old, new):
    p = ROOT / rel
    s = p.read_text()
    if new in s:
        print(f"[already] {rel}")
        return
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{rel}: expected exactly one marker, found {n}\nMARKER:\n{old}")
    p.write_text(s.replace(old, new, 1))
    print(f"[patched] {rel}")


replace_once(
    "models/trellis_ss_prior.py",
    '    "trellis_prior_projection_delta_rms",\n    "trellis_prior_projection_delta_clipped_rms",\n',
    '    "trellis_prior_projection_delta_rms",\n    "trellis_prior_projection_tangent_delta_rms",\n    "trellis_prior_projection_radial_fraction",\n    "trellis_prior_projection_raw_cosine_z",\n    "trellis_prior_projection_delta_clipped_rms",\n',
)

replace_once(
    "models/trellis_ss_prior.py",
    '        projection_clip_ratio=0.10,\n        rms_guard_weight=1.0,\n',
    '        projection_clip_ratio=0.10,\n        tangent_projection=False,\n        rms_guard_weight=1.0,\n',
)

replace_once(
    "models/trellis_ss_prior.py",
    '        self.projection_clip_ratio = float(projection_clip_ratio)\n\n        self.rms_guard_weight = float(rms_guard_weight)\n',
    '        self.projection_clip_ratio = float(projection_clip_ratio)\n        self.tangent_projection = bool(tangent_projection)\n\n        self.rms_guard_weight = float(rms_guard_weight)\n',
)

replace_once(
    "models/trellis_ss_prior.py",
    '        return rms\n\n    def forward(\n',
    '''        return rms

    @staticmethod
    def _split_radial_tangent(delta, z, batch_size):
        # Split delta into the component parallel to z and its tangent component.
        delta_f = delta.float().reshape(batch_size, -1)
        z_f = z.float().reshape(batch_size, -1)

        dot = (delta_f * z_f).sum(dim=1)
        z_sq = z_f.square().sum(dim=1).clamp_min(1e-12)
        delta_norm = delta_f.square().sum(dim=1).clamp_min(1e-12).sqrt()
        z_norm = z_sq.sqrt()

        coeff = dot / z_sq
        shape = (batch_size,) + (1,) * (delta.ndim - 1)
        radial = z.float() * coeff.reshape(shape)
        tangent = delta.float() - radial

        radial_rms = (
            radial.reshape(batch_size, -1).square().mean(dim=1)
            .clamp_min(1e-12).sqrt()
        )
        raw_rms = (
            delta_f.square().mean(dim=1).clamp_min(1e-12).sqrt()
        )
        radial_fraction = radial_rms / raw_rms.clamp_min(1e-12)
        raw_cosine_z = dot / (delta_norm * z_norm).clamp_min(1e-12)

        return tangent.contiguous(), radial_fraction, raw_cosine_z

    def forward(
''',
)

replace_once(
    "models/trellis_ss_prior.py",
    '''            raw_delta_rms_per_item = self._rms_per_item(
                raw_delta,
                batch_size,
            )

            endpoint_rms = self._rms_per_item(
''',
    '''            raw_delta_rms_per_item = self._rms_per_item(
                raw_delta,
                batch_size,
            )

            tangent_delta, radial_fraction_per_item, raw_cosine_z_per_item = (
                self._split_radial_tangent(
                    raw_delta,
                    detached_z,
                    batch_size,
                )
            )
            tangent_delta_rms_per_item = self._rms_per_item(
                tangent_delta,
                batch_size,
            )
            projection_delta = (
                tangent_delta
                if self.tangent_projection
                else raw_delta
            )
            projection_delta_rms_per_item = (
                tangent_delta_rms_per_item
                if self.tangent_projection
                else raw_delta_rms_per_item
            )

            endpoint_rms = self._rms_per_item(
''',
)

replace_once(
    "models/trellis_ss_prior.py",
    '                    / raw_delta_rms_per_item.clamp_min(1e-12)\n',
    '                    / projection_delta_rms_per_item.clamp_min(1e-12)\n',
)

replace_once(
    "models/trellis_ss_prior.py",
    '                    raw_delta\n                    * delta_scale.reshape(\n',
    '                    projection_delta\n                    * delta_scale.reshape(\n',
)

replace_once(
    "models/trellis_ss_prior.py",
    '                delta = raw_delta\n                clip_fraction = detached_z.new_zeros(())\n',
    '                delta = projection_delta\n                clip_fraction = detached_z.new_zeros(())\n',
)

replace_once(
    "models/trellis_ss_prior.py",
    '''            "trellis_prior_projection_delta_rms":
                raw_delta_rms_per_item.mean().detach(),

            "trellis_prior_projection_delta_clipped_rms":
''',
    '''            "trellis_prior_projection_delta_rms":
                raw_delta_rms_per_item.mean().detach(),

            "trellis_prior_projection_tangent_delta_rms":
                tangent_delta_rms_per_item.mean().detach(),

            "trellis_prior_projection_radial_fraction":
                radial_fraction_per_item.mean().detach(),

            "trellis_prior_projection_raw_cosine_z":
                raw_cosine_z_per_item.mean().detach(),

            "trellis_prior_projection_delta_clipped_rms":
''',
)

replace_once(
    "modules/trellis_prior_training.py",
    '    group.add_argument(\n        "--trellis_prior_rms_guard_weight",\n',
    '''    group.add_argument(
        "--trellis_prior_tangent_projection",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Remove the TRELLIS correction component parallel to the current "
            "student latent before trust-region clipping."
        ),
    )
    group.add_argument(
        "--trellis_prior_rms_guard_weight",
''',
)

replace_once(
    "train.py",
    '''                projection_clip_ratio=(
                    args.trellis_prior_projection_clip_ratio
                ),
                rms_guard_weight=(
''',
    '''                projection_clip_ratio=(
                    args.trellis_prior_projection_clip_ratio
                ),
                tangent_projection=bool(
                    args.trellis_prior_tangent_projection
                ),
                rms_guard_weight=(
''',
)

replace_once(
    "train.py",
    '            f"projection_clip_ratio={args.trellis_prior_projection_clip_ratio}, "\n            f"rms_guard_weight={args.trellis_prior_rms_guard_weight}, "\n',
    '            f"projection_clip_ratio={args.trellis_prior_projection_clip_ratio}, "\n            f"tangent_projection={bool(args.trellis_prior_tangent_projection)}, "\n            f"rms_guard_weight={args.trellis_prior_rms_guard_weight}, "\n',
)

replace_once(
    "modules/prior_gradient_balance.py",
    'PATCH_VERSION = "mf-prior-balance-v1"\n',
    'PATCH_VERSION = "mf-prior-balance-v2"\n',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '    g.add_argument("--trellis_prior_ratio_max", type=float, default=1.0)\n    g.add_argument("--trellis_prior_balance_ema", type=float, default=0.95)\n',
    '    g.add_argument("--trellis_prior_ratio_max", type=float, default=1.0)\n    g.add_argument("--trellis_prior_group_ratio_max", type=float, default=1.0)\n    g.add_argument("--trellis_prior_balance_ema", type=float, default=0.95)\n',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '    finite_positive = ("trellis_prior_ratio_target", "trellis_prior_ratio_max",\n                       "trellis_prior_lambda_min", "trellis_prior_lambda_max")\n',
    '''    finite_positive = (
        "trellis_prior_ratio_target",
        "trellis_prior_ratio_max",
        "trellis_prior_group_ratio_max",
        "trellis_prior_lambda_min",
        "trellis_prior_lambda_max",
    )
''',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '    max_ratio: float = 1.0\n    ema: float = 0.95\n',
    '    max_ratio: float = 1.0\n    group_max_ratio: float = 1.0\n    ema: float = 0.95\n',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '        return cls(target=args.trellis_prior_ratio_target,\n                   max_ratio=args.trellis_prior_ratio_max,\n                   ema=args.trellis_prior_balance_ema,\n',
    '        return cls(target=args.trellis_prior_ratio_target,\n                   max_ratio=args.trellis_prior_ratio_max,\n                   group_max_ratio=args.trellis_prior_group_ratio_max,\n                   ema=args.trellis_prior_balance_ema,\n',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '''        # Cap instantaneous projection norm, not just an EMA estimate.
        cap = self.config.max_ratio * nf / np_
        weight = min(nominal, cap)
        report = self._metrics_for_vectors(gf, gp, gg, weight)
''',
    '''        # Cap instantaneous projection norm globally and within every
        # trainable optimizer group. A single scalar weight is retained.
        cap = self.config.max_ratio * nf / np_
        group_cap = float("inf")
        for _name, start, end in self.groups:
            nf_group = self._norm(gf[start:end])
            np_group = self._norm(gp[start:end])
            if nf_group > self.config.min_norm and np_group > self.config.min_norm:
                group_cap = min(
                    group_cap,
                    self.config.group_max_ratio * nf_group / np_group,
                )
        weight = min(nominal, cap, group_cap)
        report = self._metrics_for_vectors(gf, gp, gg, weight)
''',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '            "balance/ratio_cap_hit": float(nominal > cap),\n            "balance/weak_streak": float(self.weak_streak),\n',
    '            "balance/ratio_cap_hit": float(nominal > cap),\n            "balance/group_ratio_cap_hit": float(nominal > group_cap),\n            "balance/weak_streak": float(self.weak_streak),\n',
)

replace_once(
    "modules/prior_gradient_balance.py",
    '            ("balance/ratio_cap_hit", "ratio_capped"),\n            ("perf/peak_allocated_gib_rank0", "peak_GiB_r0"),\n',
    '            ("balance/ratio_cap_hit", "ratio_capped"),\n            ("balance/group_ratio_cap_hit", "group_capped"),\n            ("trellis_prior_projection_tangent_delta_rms", "delta_tan"),\n            ("trellis_prior_projection_radial_fraction", "radial_frac"),\n            ("trellis_prior_projection_raw_cosine_z", "cos_dz"),\n            ("perf/peak_allocated_gib_rank0", "peak_GiB_r0"),\n',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    ''': "${TRELLIS_PRIOR_PROJECTION_CLIP_RATIO:=0.15}"
: "${TRELLIS_PRIOR_GRAD_BALANCE:=1}"
: "${TRELLIS_PRIOR_RATIO_TARGET:=0.5}"
: "${TRELLIS_PRIOR_RATIO_MAX:=1.0}"
: "${TRELLIS_PRIOR_BALANCE_EMA:=0.95}"
''',
    ''': "${TRELLIS_PRIOR_PROJECTION_CLIP_RATIO:=0.15}"
: "${TRELLIS_PRIOR_TANGENT_PROJECTION:=1}"
: "${TRELLIS_PRIOR_GRAD_BALANCE:=1}"
: "${TRELLIS_PRIOR_RATIO_TARGET:=0.35}"
: "${TRELLIS_PRIOR_RATIO_MAX:=0.60}"
: "${TRELLIS_PRIOR_GROUP_RATIO_MAX:=0.75}"
: "${TRELLIS_PRIOR_BALANCE_EMA:=0.99}"
''',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    ': "${TRELLIS_PRIOR_GUARD_SCALE:=0.1}"\n',
    ': "${TRELLIS_PRIOR_GUARD_SCALE:=0.5}"\n',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    'TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="${TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO:-0.25}"\n',
    'TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="${TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO:-0.50}"\n',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    'export SINGULARITYENV_TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO"\n',
    'export SINGULARITYENV_TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO"\nexport SINGULARITYENV_TRELLIS_PRIOR_TANGENT_PROJECTION="$TRELLIS_PRIOR_TANGENT_PROJECTION"\n',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    'export SINGULARITYENV_TRELLIS_PRIOR_RATIO_MAX="$TRELLIS_PRIOR_RATIO_MAX"\nexport SINGULARITYENV_TRELLIS_PRIOR_BALANCE_EMA="$TRELLIS_PRIOR_BALANCE_EMA"\n',
    'export SINGULARITYENV_TRELLIS_PRIOR_RATIO_MAX="$TRELLIS_PRIOR_RATIO_MAX"\nexport SINGULARITYENV_TRELLIS_PRIOR_GROUP_RATIO_MAX="$TRELLIS_PRIOR_GROUP_RATIO_MAX"\nexport SINGULARITYENV_TRELLIS_PRIOR_BALANCE_EMA="$TRELLIS_PRIOR_BALANCE_EMA"\n',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    '  --trellis_prior_projection_clip_ratio "$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO" \\\n  --trellis_prior_rms_guard_weight "$TRELLIS_PRIOR_RMS_GUARD_WEIGHT" \\\n',
    '  --trellis_prior_projection_clip_ratio "$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO" \\\n  --trellis_prior_tangent_projection "$TRELLIS_PRIOR_TANGENT_PROJECTION" \\\n  --trellis_prior_rms_guard_weight "$TRELLIS_PRIOR_RMS_GUARD_WEIGHT" \\\n',
)

replace_once(
    "slurm/train_morphflow_v3_prior_strong.slurm",
    '  --trellis_prior_ratio_max "$TRELLIS_PRIOR_RATIO_MAX" \\\n  --trellis_prior_balance_ema "$TRELLIS_PRIOR_BALANCE_EMA" \\\n',
    '  --trellis_prior_ratio_max "$TRELLIS_PRIOR_RATIO_MAX" \\\n  --trellis_prior_group_ratio_max "$TRELLIS_PRIOR_GROUP_RATIO_MAX" \\\n  --trellis_prior_balance_ema "$TRELLIS_PRIOR_BALANCE_EMA" \\\n',
)

replace_once(
    "tools/launch_prior_strong.py",
    '''        TRELLIS_PRIOR_PROJECTION_CLIP_RATIO='0.15',
        TRELLIS_PRIOR_RMS_GUARD_WEIGHT='1.0',
        TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO='0.25',
        TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO='2.0',
        TRELLIS_PRIOR_GRAD_BALANCE='1', TRELLIS_PRIOR_RATIO_TARGET='0.5',
        TRELLIS_PRIOR_RATIO_MAX='1.0', TRELLIS_PRIOR_BALANCE_EMA='0.95',
''',
    '''        TRELLIS_PRIOR_PROJECTION_CLIP_RATIO='0.15',
        TRELLIS_PRIOR_TANGENT_PROJECTION='1',
        TRELLIS_PRIOR_RMS_GUARD_WEIGHT='1.0',
        TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO='0.50',
        TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO='2.0',
        TRELLIS_PRIOR_GRAD_BALANCE='1', TRELLIS_PRIOR_RATIO_TARGET='0.35',
        TRELLIS_PRIOR_RATIO_MAX='0.60', TRELLIS_PRIOR_GROUP_RATIO_MAX='0.75',
        TRELLIS_PRIOR_BALANCE_EMA='0.99',
''',
)

replace_once(
    "tools/launch_prior_strong.py",
    "        TRELLIS_PRIOR_PHASE_WARMUP_STEPS='500', TRELLIS_PRIOR_GUARD_SCALE='0.1',\n",
    "        TRELLIS_PRIOR_PHASE_WARMUP_STEPS='500', TRELLIS_PRIOR_GUARD_SCALE='0.5',\n",
)

replace_once(
    "tests/test_prior_gradient_balance.py",
    '    def test_persistent_weak_signal_aborts(self):\n',
    '''    def test_group_ratio_cap_is_respected(self):
        net, opt, bal = setup(BalanceConfig(
            target=0.5,
            max_ratio=1.0,
            group_max_ratio=0.30,
            warmup_steps=0,
            ema=0,
        ))
        _, metrics = bal.backward_parts(
            net(torch.randn(2, 3), torch.randn(2, 2)),
            FakeAccelerator(),
            net,
            70000,
        )
        for group in ("condition", "lora"):
            self.assertLessEqual(
                metrics["balance/ratio_projection_fm/" + group],
                0.300001,
            )
        self.assertEqual(metrics["balance/group_ratio_cap_hit"], 1.0)

    def test_persistent_weak_signal_aborts(self):
''',
)

test_path = ROOT / "tests" / "test_trellis_prior_tangent.py"
if not test_path.exists():
    test_path.write_text(
'''import unittest

import torch

from models.trellis_ss_prior import TrellisSSPrior


class TangentProjectionTests(unittest.TestCase):
    def test_tangent_delta_is_orthogonal_to_current_latent(self):
        torch.manual_seed(7)
        z = torch.randn(3, 8, 4, 4, 4)
        delta = -0.7 * z + 0.2 * torch.randn_like(z)
        tangent, radial_fraction, raw_cosine = (
            TrellisSSPrior._split_radial_tangent(delta, z, z.shape[0])
        )
        dot = (tangent.flatten(1) * z.flatten(1)).sum(dim=1)
        scale = (
            tangent.flatten(1).norm(dim=1) * z.flatten(1).norm(dim=1)
        ).clamp_min(1e-12)
        self.assertTrue(torch.all((dot.abs() / scale) < 2e-6))
        self.assertTrue(torch.all(radial_fraction > 0))
        self.assertTrue(torch.all(torch.isfinite(raw_cosine)))

    def test_pure_radial_delta_is_removed(self):
        z = torch.randn(2, 8, 4, 4, 4)
        tangent, radial_fraction, _ = (
            TrellisSSPrior._split_radial_tangent(-0.4 * z, z, z.shape[0])
        )
        self.assertLess(float(tangent.abs().max()), 2e-6)
        torch.testing.assert_close(
            radial_fraction,
            torch.ones_like(radial_fraction),
            atol=2e-6,
            rtol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
''')
    print("[new] tests/test_trellis_prior_tangent.py")

targets = [
    "models/trellis_ss_prior.py",
    "modules/trellis_prior_training.py",
    "modules/prior_gradient_balance.py",
    "train.py",
    "tools/launch_prior_strong.py",
    "tests/test_prior_gradient_balance.py",
    "tests/test_trellis_prior_tangent.py",
]
subprocess.run(["python3", "-m", "py_compile", *targets], check=True)
subprocess.run(["bash", "-n", "slurm/train_morphflow_v3_prior_strong.slurm"], check=True)
subprocess.run(["git", "diff", "--check"], check=True)

print("\nPATCH OK")
print("tangent_projection=1")
print("target/global_cap/group_cap = 0.35 / 0.60 / 0.75")
print("balance_ema=0.99")
print("guard_scale=0.5, guard_low_ratio=0.50")
print("rollout=8, grad_steps=4, every=2, clip=0.15")
