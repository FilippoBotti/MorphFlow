# MorphFlow: measured strong projection prior

Base reviewed: `52c05c56d9d9a074610d51ab803ab1788ee70484`.

## Experimental definition

Resume the immutable epoch-14 / step-70000 checkpoint with its optimizer and
scheduler. The original student architecture and ordinary launcher are unchanged.
The dedicated launcher finishes epochs 15 through 25, not 25 additional epochs.

Projection: tau [0.05, 0.20], no CFG, clip 0.15, every 2 updates, one sample/GPU,
8 Euler steps with 4 differentiated suffix steps, activation checkpointing on.
The old teacher/FM weighting, LoRA configuration and learning rates are retained.

On every active update, measure GLOBAL mean gradients separately for:
- weighted teacher FM + existing semantic usage regularization;
- local projection loss (WITHOUT the RMS guard);
- RMS guard, only when active on any rank.

No `autograd.grad` calls on DDP. The active forward and backwards are enclosed
in `no_sync`, then FP32 gradients are explicitly all-reduced and divided by the
number of ranks. Off-prior updates retain the ordinary DDP backward.

Use an EMA in log space of lambda_ideal = 0.5 ||g_FM|| / ||g_projection||.
Ramp from lambda=0.1 to the calibrated estimate over 500 NEW optimizer steps.
Nominal lambda is bounded to [0.01,100]; instantaneous projection/FM norm ratio
is capped at 1.0. Low ratios below 0.3 after warmup are explicitly warned. After 25 consecutive
weak active updates training aborts before the next optimizer step, rather than
silently running a weak-prior experiment.
This is a heuristic balance controller, not the full GradNorm algorithm.

The guard has a separate fixed external coefficient 0.1. The existing internal
guard weight remains 1.0. It is never multiplied by the adaptive projection
coefficient. This preserves the previous guard scale per application, although
applications are twice as frequent.

Large scalar lambda alone is NOT evidence of strong influence. R_P_F, computed
from mean gradients, is the diagnostic. It is not the ratio of AdamW parameter
updates. Increasing the ratio can strengthen endpoint bias or worsen geometry.

## Logs

Console [PRIOR]: phase counter, lambda/EMA, projection/FM ratio, cosine,
guard/FM ratio, combined/FM ratio, per-group ratios and cosines, guard fraction,
raw correction, relative clipped correction, clipping fraction, latent scale,
gradient norm before/after clipping, and rank-0 compute time / CUDA memory peak.

`logs/prior_diagnostics.jsonl` contains every active update. TensorBoard stores
prior values only on active updates; the fake zero prior series in validation
and duplicate slat/prior tags are removed. Raw latent statistics are retained
in the diagnostic data but no longer clutter every console line.

Additional snapshots at +500, +1500 and +3000 updates are EVALUATION ONLY.
The trainer rejects resuming them because dataloader position is not stored.
Resume later phases from best/last/epoch-end checkpoints instead.

The balancer's state (phase start, EMA, counters and settings) is saved in each
checkpoint. Set reset_phase=0 for resuming this same phase. A new phase resets
the best-FM comparator in the new output directory, not model/optimizer/scheduler.
The old checkpoint stays untouched. Its RNG and dataloader positions were not
saved by the original trainer, so this is not a bitwise-exact continuation.

## Supported environment

Ordinary single-device-per-process DDP and AdamW; FP32 trainable parameters;
BF16 or no autocast; gradient accumulation=1; non-reentrant checkpointing.
FP16/GradScaler, DeepSpeed and FSDP are explicitly rejected in balanced mode.

Extra FP32 gradient storage is proportional to trainable parameters, not frozen
TRELLIS weights: normally two packed component vectors, three if guard is active,
in addition to parameter gradients and DDP buckets. Runtime and real GPU peak
must be measured. More backward/communication work is expected on active steps.

## Tests

The package includes 9 CPU gradient/controller tests and a two-process Gloo
smoke test. The smoke test alternates split and ordinary DDP updates and checks
against a serial global-batch reference, including a guard active on one rank,
checkpointed forwards, AdamW updates and balancer-state resume.
Run both tests in the user's PyTorch 2.4 TRELLIS container before training.
The development run used PyTorch 2.10 CPU; no full TRELLIS GPU run was performed.

The installer validates every transformation and syntax before writing; backups
and a manifest are saved under `.git/morphflow_patches/`, not as tracked bak files.
The transformer was exercised on integration fixtures matching the inspected
source structure. `--check` additionally validates the actual local files.

Do not interpret clipped projection loss or clip_fraction as a distance-to-
manifold metric. Stochastic denoising and endpoint conditioning can produce
nonzero corrections even for valid samples. Decode fixed held-out SS examples
and examine central alpha values, endpoint bias and full-pipeline geometry.
