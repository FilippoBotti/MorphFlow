# Paired ablation evaluation on Leonardo

Four fixed MorphFlow configurations only: baseline, gate only in both flows,
prior only in both flows, and full model. CFG is SS 5 / SLat 1.5 for every
configuration. The full model uses the lowest saved validation loss for each
flow (currently SS epoch 17, SLat epoch 12). The prior ablation uses the
previously selected global-batch-8 SLat checkpoint. No new training is performed.

The evaluation covers all 500 unique test pairs. The original 50-pair sample
is marked as an exploratory pilot. Before observing new outputs, a fixed
greedy selection from the other 450 pairs chooses 349 pairs whose source
assets are all distinct and absent from the pilot. These form the primary
confirmation cohort. The full 500-pair summary is also retained; its
descriptive intervals resample connected components of shared source assets.

The protocol uses seven states (five Beta(5,5) intermediates and two canonical
sources), 64 matched LPIPS views, 1,000 FID views per state, and FID from
concatenating two selected intermediate states per pair. The reported scores
are FID, full-path PDV, source-normalized full-path PPL, and full-interval EPRE.
Checkpoint identities, input metadata, source snapshot and the statistical
protocol are frozen before submission.

## Launch

From `/leonardo_work/IscrC_MORPHFL/mbarezzi`:

```bash
envs/morphflow/bin/python src/MorphFlow/slurm/prepare_ablation_statistics_leonardo.py \
  --run-root /leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/threeway_benchmark/ablation_statistics_n500_seed42_20261028 \
  --batch-size 5 --jobs 3 --submit
```

The launcher reserves one four-GPU node and runs one configuration per GPU.
Three 24-hour allocations are chained with `afterany` for safe continuation;
later allocations exit immediately when the complete report already exists.
Mail is enabled for BEGIN, END and FAIL. The generated `submissions.json`
records all job IDs. The same frozen launcher can be resubmitted with the
suite root as its sole argument.

For independent pieces of the test set, the same launcher also accepts
`RUN_ROOT SHARD_INDEX SHARD_COUNT` (zero-based index). The count must match
`batch_shards` in the run manifest. Original batch positions are distributed
round-robin, retaining pair identities, seeds, and existing batch paths.
Each piece runs all four model variants on its own four-GPU node, with
separate locks and temporary directories. Continuations depend only on
the previous allocation of the same piece. Global progress still appears
in `variants/MODEL/status.json`, and piece-specific progress is saved in
`status_shard_INDEX_of_COUNT.json`. The final report is written once all
four variants have completed all test pairs.

Only five pairs per GPU occupy temporary work storage at a time. After a
batch is measured, its metrics, per-view CSVs and seven `025.png` renders per
pair are committed and hashed before its new temporary assets are removed.
Previous runs, dataset files and checkpoints are never deleted. A partially
committed batch resumes without repeating valid pair commits.

## Progress and outputs

Each allocation prints the worker log paths. Progress is also available in:

```text
SUITE/variants/baseline/status.json
SUITE/variants/gate_only/status.json
SUITE/variants/prior_only/status.json
SUITE/variants/full/status.json
SUITE/logs/job_JOBID/MODEL/worker.out
SUITE/logs/job_JOBID/MODEL/batch_XXXX_generation.out
SUITE/logs/job_JOBID/MODEL/batch_XXXX_metrics.out
```

Cached metrics and comparison images use:

```text
SUITE/variants/MODEL/cfg_ss_5_slat_1.5/metrics/morphflow/PAIR/metrics.json
SUITE/variants/MODEL/cfg_ss_5_slat_1.5/metrics/morphflow/PAIR/images/STATE/025.png
```

After all four configurations finish, `reports/index.html` contains the
comparison, `summary.csv` contains means and confidence intervals, and
`paired_comparisons.csv` contains paired differences and adjusted p-values.
`per_pair_metrics.csv` preserves the observations. A report invoked before
completion writes progress and descriptive means only, without hypothesis
tests or significance decisions.

Primary inference uses 10,000 paired bootstrap resamples and 99,999 two-sided
paired sign-flip permutations. Holm correction covers the 12 primary tests
(all six model contrasts for FID and PDV). PPL/EPRE use a separately labelled
exploratory family. Confidence intervals are marginal, not simultaneous.
Sign-flip tests assume exchangeability of method labels within a pair.
All uncertainty is conditional on these trained checkpoints and test assets;
it does not estimate variation across training seeds or remove differences
in the original training budgets.
