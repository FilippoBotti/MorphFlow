# Dataset v4 Interp3D: 12,500 unique train pairs x 4 alpha

Ready-to-submit HPC package. Based on the existing MorphFlow 8xL40S Singularity layout.

## Budget

| Split | Unique pairs | Samples per pair | Physical targets | Alpha=0 | Alpha=1 | Interior |
|---|---:|---:|---:|---:|---:|---:|
| Train | 12,500 | 4 | 50,000 | 500 | 500 | 49,000 |
| Validation | 500 | 4 | 2,000 | 20 | 20 | 1,960 |
| Test | 500 | 4 | 2,000 | 20 | 20 | 1,960 |
| Total | 13,500 | 4 | 54,000 | 540 | 540 | 52,920 |

Training with `--augment_source_swap 1` makes **100,000 virtual train examples**, without duplicate teacher generation or files. Validation and test are not doubled by default.

Every pair is assigned four **different** alpha values, stratified to cover (0,1); 2% of physical targets are exact alpha=0/1 endpoint aliases, one endpoint in each of 1,000 train pairs. The interpolation for alpha=1..0 follows A->B order; inside Interp3D `t=1-alpha`, increasing from 0 to 1. Teacher SS and SLat latents are captured and saved as compatible `.pt` files.

## Install

Download the archive and copy it to the HPC login server, then:

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
tar -xzf "$HOME/morphflow_v4_interp3d_12500x4_monitor_patch.tar.gz"
sbatch slurm/generate_interp3d_v4.slurm
```

Only one production job is submitted; there is **no pilot or intermediate ablation**. A fast import preflight is part of the job and will abort for missing GPU/environment support. Teacher inference on the university HPC is not verified by these static checks.

Output: `/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_interp3d_12500x4_e02`.

## Monitor (login node, second terminal)

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
OUT=/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_interp3d_12500x4_e02
python3 slurm/monitor_interp3d_v4.py status --out "$OUT" --watch 30
```

The view includes stage, worker heartbeats, per-GPU progress, completed physical targets, per-split targets, ETA, and errors. General job status: `squeue -u "$USER"`, after completion use `sacct -j JOBID --format=JobID,State,Elapsed,ExitCode`.

Worker logs: `/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/dataset_v4_interp3d/job_JOBID/{assets,pairs}_gpu_N.log`. The generator is resumable: relaunch the same `sbatch` in case of interruption, with the same OUT_NAME.

Requirements: original MorphFlow v3 `split_assets.json`, FLUX inputs, Interp3D checkout and Singularity environments as used by the working three-way benchmark. The train directory remains unchanged until the dataset is completed.

**Caution:** the Python/Slurm syntax and CPU-only planner are verified locally but Interp3D inference, GPU memory, checkpoint compatibility and accuracy remain to be checked by running on wn49. Grouping four alpha for the same pair saves repeated conditioning but increases instantaneous GPU memory.
