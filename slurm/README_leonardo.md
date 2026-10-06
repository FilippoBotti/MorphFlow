# MorphFlow v3 SLat su Leonardo

`train_morphflow_v3_leonardo.sbatch` usa l'env nativo attivato da
`/leonardo_work/IscrC_MORPHFL/mbarezzi/env.sh`.
La configurazione iniziale richiede 4 nodi Booster, 4 A100 per nodo, 32 CPU e
128 GiB di RAM per nodo. Slurm avvia un launcher Accelerate per nodo; ogni launcher
crea 4 processi GPU. Il rank del nodo viene assegnato da `SLURM_PROCID`.

Con 16 GPU e `train_bs=2` il batch globale e' 32 (nell'allegato di ateneo era 16).
Il training usa `flow_target=slat` e `slat_condition_source=slat`.
Gli altri iperparametri seguono l'allegato r32/semMatch: LoRA rank 32, alpha 64,
`flow_lr=lora_lr=1e-4`, semantic token matching abilitato con max align 0.25,
semantic cycle loss con peso 0.01 e probabilita' 0.25, 60 epoche, `val_bs=1`,
BF16, scheduler cosine e 8 worker del DataLoader per processo.
I 32 worker per nodo si aggiungono ai processi del training: se la CPU diventa
un limite, provare `NUM_WORKERS=4` e confrontare la velocita'.

Percorsi predefiniti:

- Dataset: `/leonardo_scratch/fast/IscrC_MORPHFL/mbarezzi/datasets/morphing_dataset_v3`.
- Run: `/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/v3_slat_lora_cross_r32_tokenGate_semMatch_loralr1e-4_leonardo`.
- Checkpoint: `<run>/checkpoints/`; TensorBoard: `<run>/tb/`.
- Log training: `<run>/logs/train_<jobid>.log`.
- Log Slurm: `/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/<jobname>_<jobid>.out` e `.err`.
- Pesi Hugging Face: `/leonardo_work/IscrC_MORPHFL/mbarezzi/cache/huggingface`.

Lo student con `flow_target=slat` e `slat_condition_source=slat` usa i latenti.
Quando si abilita il prior TRELLIS, il supervisore usa invece le immagini
endpoint e DINOv2: servono `source_images_root` e la cache Torch Hub.
NVLink e InfiniBand restano abilitati per NCCL.
Le notifiche Slurm `ALL` sono inviate a `marco.barezzi@unipr.it`.

## Preparazione dal login node

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
TRELLIS_PRIOR_WEIGHT=1.0 bash slurm/train_morphflow_v3_leonardo.sbatch --prepare
```

Scarica nella cache condivisa pesi e configurazione TRELLIS SLat image_large,
il codice DINOv2 e i pesi `dinov2_vitl14_reg4_pretrain.pth`. Carica DINOv2
su CPU e controlla import e argomenti, incluso il bilanciamento dei gradienti.
Non avvia il training e non richiede GPU. Con lo stesso peso e `--check`
verifica anche il caricamento DINOv2 con i download Torch Hub disabilitati.
I job controllano la presenza della cache prima di avviare i worker GPU.
Il prior richiede i suoi pesi TRELLIS e DINOv2 anche con `INIT_FROM`.
Con il bilanciamento attivo, `TRELLIS_PRIOR_WEIGHT` deve essere positivo
(default SLat: `1.0`); passare lo stesso valore a `--prepare`, `--check`
e `sbatch` quando si usa un override.
La cartella dei log Slurm deve esistere prima di `sbatch`;
`--prepare` la crea (esiste gia' nell'installazione corrente).

## Prova della comunicazione su quattro nodi

```bash
sbatch --qos=boost_qos_dbg --time=00:10:00 \
    --job-name=morphflow_nccl_check \
    slurm/train_morphflow_v3_leonardo.sbatch --comm-check
```

Esegue all-reduce NCCL sulle 16 GPU, verifica i rank distribuiti su quattro nodi
e importa il codice del training. Non carica dataset o checkpoint e non salva
una run. Cercare `MULTINODE CHECK PASSED` nel log `.out`.
Questo controllo non esegue forward/backward del modello: l'uso di memoria
e i kernel del training si verificano avviando il training vero.
Per questo test viene impostato `NCCL_DEBUG=INFO`; controllare nei log il
trasporto selezionato se si vuole verificare l'uso di InfiniBand.

## Training

```bash
sbatch slurm/train_morphflow_v3_leonardo.sbatch
```

Il limite iniziale e' 24 ore, QoS `normal`. Per una run fino a quattro giorni,
con l'account corrente abilitato alla QoS lunga:

```bash
sbatch --qos=boost_qos_lprod --time=4-00:00:00 \
    slurm/train_morphflow_v3_leonardo.sbatch
```

La QoS lunga ammette fino a 8 nodi per progetto. La disponibilita' dipende
anche dagli altri job del progetto.

## Ripresa e varianti

```bash
# Riprende morphflow_last.pt; in sua assenza il checkpoint compatibile piu' recente.
AUTO_RESUME=1 sbatch slurm/train_morphflow_v3_leonardo.sbatch

# Percorso assoluto di un checkpoint di questa stessa architettura.
RESUME_FROM=/percorso/checkpoint.pt RUN_NAME=run_ripresa \
    sbatch slurm/train_morphflow_v3_leonardo.sbatch

# 4 nodi = 16 GPU; train_bs=1 conserva il batch globale 16.
TRAIN_BS=1 RUN_NAME=v3_slat_r32_semMatch_4n_bs1 sbatch --nodes=4 \
    slurm/train_morphflow_v3_leonardo.sbatch

# Override dei percorsi, senza modificare env.sh.
MF_DATA_ROOT=/percorso/dataset MF_OUT_ROOT=/percorso/risultati RUN_NAME=nuova_run \
    sbatch slurm/train_morphflow_v3_leonardo.sbatch
```

`AUTO_RESUME=1` parte da zero se non trova checkpoint. La ripresa resta manuale
tramite `sbatch`: non vengono accodati job automaticamente.
Il nome della run SLat e' distinto dalle precedenti run SS. Usare checkpoint
SLat della stessa architettura r32/semMatch per `RESUME_FROM`; i checkpoint SS
non sono compatibili con questo flow.
Il codice salva `morphflow_last.pt` alla fine di ogni epoca; al limite di tempo
si perde il lavoro dell'epoca incompleta. Non e' un salvataggio su segnale Slurm.
Il checkpoint ripristina modello, optimizer e scheduler secondo la logica
esistente in `train.py`. Conservare numero di GPU e batch per una ripresa
coerente; cambiare parallelismo modifica la scansione dei dati e i passi
del scheduler. Usare un nuovo `RUN_NAME` per confronti con parallelismo o batch
diversi. Gli altri iperparametri non vengono riscalati automaticamente.

`--nodes` si puo' cambiare alla submission. Conservare `--ntasks-per-node=1`
e `--gres=gpu:4`: i processi per GPU sono gestiti da Accelerate.
`TRAIN_EPOCHS`, `NUM_WORKERS`, `OMP_NUM_THREADS`, `MASTER_PORT`, `AUTO_RESUME`,
`RESUME_FROM` e `RUN_NAME` sono anch'essi modificabili tramite ambiente.

```bash
squeue -u "$USER"
tail -f /leonardo_work/IscrC_MORPHFL/mbarezzi/logs/<jobname>_<jobid>.out
```

Riferimenti: [Leonardo, risorse e QoS](https://docs.hpc.cineca.it/hpc/leonardo.html),
[esempio ufficiale Accelerate/Slurm](https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh).

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
