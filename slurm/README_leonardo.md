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

Le immagini FLUX e DINO non vengono caricate in questa configurazione
(`flow_target=slat`, `slat_condition_source=slat`), quindi `source_images_root`
e' omesso. NVLink e InfiniBand restano abilitati per NCCL.
Le notifiche Slurm `ALL` sono inviate a `marco.barezzi@unipr.it`.

## Preparazione dal login node

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
bash slurm/train_morphflow_v3_leonardo.sbatch --prepare
```

Scarica nella cache condivisa i pesi TRELLIS SLat image_large
(`ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors`) e
controlla import e argomenti. Non avvia il training e non richiede GPU.
Con `--check` esegue lo stesso controllo senza scaricare nulla.
I job usano la cache offline; se si riparte da un checkpoint, non servono i
pesi iniziali TRELLIS. La cartella dei log Slurm deve esistere prima di `sbatch`;
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
