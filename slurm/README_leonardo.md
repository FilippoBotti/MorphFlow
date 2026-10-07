# MorphFlow v3 su Leonardo

I launcher usano l'ambiente nativo `/leonardo_work/IscrC_MORPHFL/mbarezzi/env.sh`,
2 nodi Booster, 4 GPU per nodo, 32 CPU e 128 GiB di RAM per nodo.
Slurm avvia un launcher Accelerate per nodo e ogni launcher crea 4 processi GPU.
Le notifiche mail `ALL` includono inizio, fine ed errore.

## Esperimenti encDual (7 ottobre 2026)

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
# SLat: replica train_morphflow_v3.slurm, pesi TRELLIS originali, 25 epoche.
sbatch slurm/train_morphflow_v3_leonardo.sbatch
# SS: 5 nuove epoche dal best importato, prior sui due endpoint, rollout 24/6.
sbatch slurm/train_morphflow_v3_ss_prior_dual_leonardo.sbatch
```

| Parametro | SLat ablation | SS prior dual |
|---|---|---|
| Run | `slat_ablation_encDual_lora` | `ss_ablation_encDual_lora_priorDual_s24_g6` |
| Epoche | 25 | 5 nuove |
| Batch/GPU (globale) | 2 (16) | 1 (8) |
| Inizializzazione | TRELLIS originale | `checkpoints/ss_ablation_encDual_lora/morphflow_best.pt` |
| LR cond/flow/LoRA | 5e-5 | 5e-5 |
| Weight decay / warmup LR | 5e-5 / 10000 | 1e-4 / 500 |
| Prior | disabilitato | peso 1.01, ogni 4 update, max 1 sample/GPU |
| Rollout prior | non eseguito | 24 passi, ultimi 6 con gradiente e checkpointing |
| Checkpoint periodici | ogni 2 epoche | ogni epoca e a meta' epoca |

Entrambi usano encoder sparse_conv3d, conditioning separato con gate token,
LoRA cross-attention rank 32 / alpha 64, BF16, scheduler cosine, source swap,
semantic matching/cycle/usage disabilitati e CFG dropout 0.
Il fine-tuning SS carica solo i pesi (`--init_from --init_strict 1`):
optimizer, scheduler e contatore delle cinque epoche iniziano da zero.

Per SS **e** SLat il prior ora valuta entrambe le immagini endpoint sullo
stesso campione rumoroso, con identici tau e rumore. Ogni correzione viene
clippata rispetto all'RMS del proprio endpoint prima di calcolare
`Lproj = alpha * Lsrc1 + (1-alpha) * Lsrc2`.
Il teacher rimane congelato; scale/stat anchor e guard vengono applicati una volta.
Per SLat la riduzione resta una media sui token, pesati con l'alpha del loro sample.

Il preset SS mantiene tau `[0.05, 0.20]`, clip ratio `0.08`, rapporto gradiente
projection/FM target `0.15` (cap `0.30`, cap per gruppo `0.40`), EMA `0.99`
e warmup della fase prior 500 update. I 24 passi riguardano lo student;
il teacher fa una proiezione per ciascuno dei due endpoint, senza CFG.
Nei diagnostici sono presenti `trellis_prior_projection_src1_loss` e
`trellis_prior_projection_src2_loss`. La metrica storica `src1_fraction`
indica ora la media di alpha; le diagnostiche dei due endpoint sono pesate
per alpha. Il valore di Lproj non e' la loss verso il target medio: conserva
anche il disaccordo tra i target, mentre il gradiente combina le due correzioni.

## Percorsi e controlli

Radice del progetto: `/leonardo_work/IscrC_MORPHFL/mbarezzi`.

- Dataset: `/leonardo_scratch/fast/IscrC_MORPHFL/mbarezzi/datasets/morphing_dataset_v3`.
- Checkpoint SLat: `checkpoints/slat_ablation_encDual_lora/`.
- Checkpoint SS: `checkpoints/ss_ablation_encDual_lora_priorDual_s24_g6/`.
- Run/TensorBoard/diagnostici: `outputs/<run>/{tb,logs}/`.
- Log Slurm: `logs/<jobname>_<jobid>.out` e `.err`.

`MF_CHECKPOINT_DIR` imposta una directory checkpoint esplicita e viene passato
al nuovo `--checkpoint_dir`. Senza questo argomento il training continua a
usare `<out_dir>/<run_name>/checkpoints`. Il launcher protegge sia la directory
della run sia quella dei checkpoint da scritture concorrenti.

```bash
bash slurm/train_morphflow_v3_leonardo.sbatch --check
bash slurm/train_morphflow_v3_ss_prior_dual_leonardo.sbatch --check
# Solo se mancano pesi in cache, dal login:
bash slurm/train_morphflow_v3_ss_prior_dual_leonardo.sbatch --prepare
```

`--check` verifica import, parsing, cache e caricamento DINO su CPU per il prior.
I job usano le cache offline. DINO e immagini endpoint servono al prior anche
quando lo student usa i latenti come conditioning.

## Ripresa

```bash
AUTO_RESUME=1 sbatch slurm/train_morphflow_v3_leonardo.sbatch
AUTO_RESUME=1 sbatch slurm/train_morphflow_v3_ss_prior_dual_leonardo.sbatch
```

Il limite Slurm e' 24 ore. La ripresa resta manuale: `morphflow_last.pt` viene
salvato a fine epoca; gli snapshot di meta' epoca/fase sono solo per eval.
La ripresa ripristina optimizer/scheduler e conserva per default lo stato del
bilanciamento del prior. `TRAIN_EPOCHS` e' il totale della fase, non il numero
di epoche da aggiungere a un resume. Conservare batch e numero di GPU.
Non rilanciare una nuova fase sulla stessa cartella gia' popolata.

Altri override: `RUN_NAME`, `MF_OUT_ROOT`, `MF_DATA_ROOT`, `INIT_FROM`,
`RESUME_FROM`, `TRAIN_BS`, `TRAIN_EPOCHS`, `NUM_WORKERS`, `WARMUP_STEPS`,
`WEIGHT_DECAY` e le variabili `TRELLIS_PRIOR_*` del launcher comune.
`INIT_FROM=` disabilita l'inizializzazione predefinita. Gli script di ateneo
restano la sorgente della configurazione baseline SLat.

## Valutazione v3 su Leonardo

`eval_morphflow_v3_leonardo.slurm` usa lo stesso `eval_validation_latents.py`
del launcher di ateneo, con `env.sh`, cache e percorsi nativi di Leonardo.
Richiede un nodo, una GPU, 8 CPU, 32 GiB di RAM e quattro ore.
I default sono 50 coppie di test, seed 42, 50 passi, CFG 1, BF16 e
`SAVE_LATENTS=0`. La configurazione del modello viene letta dal checkpoint.
Il teacher TRELLIS del prior non viene caricato durante l'eval.

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow

# Prepara i decoder Hugging Face dal login e verifica gli import.
bash slurm/eval_morphflow_v3_leonardo.slurm --prepare

# Verifica offline senza richiedere GPU o lanciare la generazione.
bash slurm/eval_morphflow_v3_leonardo.slurm --check

# Valuta il best della run SLat con prior, letto all'avvio del job.
sbatch slurm/eval_morphflow_v3_leonardo.slurm
```

Il checkpoint predefinito e'
`/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/v3_slat_trellisPrior_r015_stat015_bs1_fromBest_leonardo/checkpoints/morphflow_best.pt`.
`CHECKPOINT_PATH` permette di scegliere un altro checkpoint; per confronti
riproducibili mentre il training continua, usare uno snapshot con epoca e step.

```bash
# Esempio: checkpoint fisso, coppie di validation e due CFG.
CHECKPOINT_PATH=/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/v3_slat_trellisPrior_r015_stat015_bs1_fromBest_leonardo/checkpoints/morphflow_epoch_0002_step_0020000.pt \
    METADATA=metadata_val.json RUN_NAME=prior_step20000_val \
    CFG_VALUES="1.0 3.0" SAVE_LATENTS=1 \
    sbatch slurm/eval_morphflow_v3_leonardo.slurm

# Baseline importata dall'ateneo, con le stesse coppie e lo stesso seed.
CHECKPOINT_PATH=/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/slat_from_hpc/morphflow_best.pt \
    METADATA=metadata_val.json RUN_NAME=baseline_hpc_val \
    CFG_VALUES="1.0 3.0" SAVE_LATENTS=1 \
    sbatch slurm/eval_morphflow_v3_leonardo.slurm
```

Per la pipeline completa, impostare `CHECKPOINT_PATH` a un checkpoint **SS** e
`SLAT_CHECKPOINT_PATH` a un checkpoint **SLat**. Eseguire prima `--prepare`
con entrambi i percorsi: serve anche il decoder SS. `STEPS` e `CFG_VALUES`
controllano il primo flow; `SLAT_STEPS` e `SLAT_CFG_SCALE` il secondo.
Un checkpoint SLat da solo genera sulle coordinate sparse del target:
questa modalita' isola il flow SLat e non valuta la generazione delle coordinate SS.

Risultati:
`/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/<EVAL_TYPE>/<RUN_NAME>/job_<jobid>/cfg_<cfg>/<timestamp>/`.
L'evaluator salva i mesh `.glb`, le metriche per campione, `summary.json`,
`selected_samples.json` e, con `SAVE_LATENTS=1`, i latenti. `OUTPUT_DIR`
modifica la directory base. I log Slurm sono in `$WORK/$USER/logs/`.

Restano disponibili le variabili del launcher di ateneo: `ROOT_DIR`,
`SOURCE_IMAGES_ROOT`, `SOURCE_IMAGE_FILENAME`, `PROJECT_DIR`, `METADATA`,
`EVAL_TYPE`, `RUN_NAME`, `NUM_SAMPLES`, `SEED`, `STEPS`, `CFG_VALUES`,
`SLAT_STEPS`, `SLAT_CFG_SCALE`, `TRELLIS_MODEL`, `MIXED_PRECISION`, `SAVE_LATENTS`.
I checkpoint SLat condizionati con DINO richiedono immagini e cache Torch Hub;
`--prepare` prepara anche quest'ultima quando necessaria.
