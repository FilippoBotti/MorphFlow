MorphFlow v4 Interp3D dataset audit and first trainings

Installation:
  cd $HOME/trellis-singularity/work/MorphFlow
  tar -xzf $HOME/morphflow_v4_training_ready.tar.gz

Audit (no training/pilot run):
  python3 slurm/audit_interp3d_v4.py --out /hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_interp3d_12500x4_e02 --repair-image-manifests

  The --repair-image-manifests option creates image manifest.json files only for
  source assets that the training DataLoader cannot otherwise find by name.
  It does NOT edit any latent .pt, source splits, alpha or target metadata.

  You need RESULT: PASS, and run_state.json phase/status complete.

When audit reports PASS:
  bash slurm/queue_train_v4.sh

  This generates slurm/train_morphflow_v4_{ss,slat}.slurm using existing
  slurm/train_morphflow_v3.slurm, and queues SS then SLat with Slurm afterok.
  Both stages use 8 GPUs, 20 epochs, prior every 4 optimizer updates,
  rollout_steps=25, grad_steps=4; SS lr=1e-4, SLat lr=5e-5.
  Checkpoints are stored under /hpc/archive/G_VBD/marco.barezzi/morphflow_runs.

  If the audit fails, do not queue training. Inspect the printed errors.
