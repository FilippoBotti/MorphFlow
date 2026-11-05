# Dataset v4 Interp3D con 8 e 2 step per coppia

Un solo job genera entrambi i dataset sull'HPC di Ateneo, con lo stesso
container, ambiente Interp3D e allocazione di otto L40S del launcher a 4 step.
Ogni dataset contiene 50.000 campioni train, che diventano 100.000 con
`augment_source_swap`, e 2.000 campioni ciascuno per validation e test.

| Dataset | Coppie train | Coppie val | Coppie test | Campioni per coppia | Campioni train / val / test |
|---|---:|---:|---:|---:|---:|
| `morphing_dataset_v4_steps8` | 6.250 | 250 | 250 | 8 | 50.000 / 2.000 / 2.000 |
| `morphing_dataset_v4_steps2` | 25.000 | 1.000 | 1.000 | 2 | 50.000 / 2.000 / 2.000 |

Gli split degli asset sono riutilizzati dal v3 e rimangono disgiunti.
Le coppie della versione a 8 sono incluse nella versione a 2. Su queste coppie
il teacher viene eseguito una sola volta, con otto alpha: la versione a 2
riutilizza due dei latenti ottenuti, uno per ciascuna metà dell'intervallo.
Sulle coppie aggiuntive vengono generati soltanto due alpha.
Nel train questo riutilizza 12.500 campioni e ne genera altri 37.500 per la
versione a 2, evitando una seconda inferenza sulle coppie comuni.

Gli alpha sono casuali e deterministici rispetto al seed: uno per ciascun bin
di uguale ampiezza in `[0.001, 0.999]`. Il 2% dei campioni di **ciascun** dataset
è costituito da endpoint esatti, bilanciati tra alpha 0 e 1: 500 per estremo
nel train e 20 per estremo in ciascuno degli altri split. Gli endpoint
sostituiscono campioni dei bin estremi e riutilizzano i latenti sorgenti.
La selezione della versione a 2 mantiene questa quota anche quando prende i
campioni dalle sequenze a 8. I metadati conservano la sequenza teacher da cui
proviene ciascun target condiviso.

## Lancio sull'Ateneo

Aggiornare il checkout MorphFlow sull'Ateneo con `git pull`, quindi inviare
il job. Non servono file da copiare separatamente:

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
git pull --ff-only
sbatch slurm/generate_interp3d_v4_multistep.slurm
```

Il launcher usa account `g_vbd`, partizione/QoS `gpu_vbd`, nodo `wn49`,
otto `gpu:l40s_vbd`, 64 CPU, 700 GiB RAM e limite di 10 giorni. Rimangono i
percorsi degli ambienti modello già usati sull'Ateneo:

- Container: `$HOME/containers/trellis-py240-cu118.sif`.
- Ambiente Interp3D: `/hpc/scratch/marco.barezzi/morph-benchmark/interp3d/env`.
- Repository Interp3D: `/hpc/scratch/marco.barezzi/morph-benchmark/interp3d/work/Interp3D`.
- Input v3 e immagini: `/hpc/scratch/marco.barezzi/3d_dataset/{morphing_dataset_v3,flux_outputs}`.

Il job prepara il piano sulla CPU, verifica l'ambiente e la cache DINOv2 con
un processo, poi distribuisce asset e coppie sugli otto worker. Dopo la
generazione finalizza entrambi i dataset e verifica il DataLoader su ciascuno.
Non è necessario inviare un secondo job per la versione a 2.

## Output e training

Percorsi predefiniti:

```text
/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_steps8_steps2/
├── assets/                         # latenti sorgenti condivisi
├── targets/                        # target teacher condivisi
├── plan.json                       # piano con sequenze da 8 o da 2
├── morphing_dataset_v4_steps8/      # root da passare al training a 8 step
│   ├── assets -> ../assets
│   ├── targets -> ../targets
│   └── metadata*.json
└── morphing_dataset_v4_steps2/      # root da passare al training a 2 step
    ├── assets -> ../assets
    ├── targets -> ../targets
    └── metadata*.json
```

I due dataset hanno metadati e piani separati, mentre i tensori condividono lo
stesso spazio su disco tramite link relativi. Per spostarli o copiarli,
conservare l'intera directory padre `morphing_dataset_v4_steps8_steps2` e i link
simbolici, per esempio con `rsync -a`. Ogni versione usa esclusivamente i
campioni elencati nei propri metadati.

Nel container di training va montata **l'intera directory padre**, così i link
`../assets` e `../targets` rimangono raggiungibili. Il bind della sola directory
`morphing_dataset_v4_steps8` o `morphing_dataset_v4_steps2` non basta. Per esempio:

```bash
OUT=/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_steps8_steps2
# Aggiungere al comando singularity exec del training:
# --bind "$OUT:/data/morphing_dataset_v4_steps8_steps2:ro"
# Root dataset da passare al programma nel container:
# /data/morphing_dataset_v4_steps8_steps2/morphing_dataset_v4_steps8
# oppure:
# /data/morphing_dataset_v4_steps8_steps2/morphing_dataset_v4_steps2
```

I launcher di training esistenti che montano soltanto `DATA_ROOT_HOST` in
`DATA_ROOT_IN` richiedono questo bind del padre aggiuntivo e il percorso della
versione scelta nel container. Il launcher di generazione esegue già gli
smoke test con il padre montato.

La precedente versione a 4 step conserva il proprio launcher e la propria
directory. Il nuovo job usa un nome di output distinto.

## Training per numero di step con LoRA sulle cross-attention

Il trainer supporta `max_train_steps`: uno step e' un aggiornamento
dell'optimizer, indipendentemente dal numero di GPU. Il contatore viene
salvato nei checkpoint e ripristinato alla ripresa; il limite e' il totale
della run, non un numero di aggiornamenti aggiuntivi a ogni rilancio.

Il nuovo preset usa LoRA solo sulle due cross-attention (`cross_attn` e
`cross_attn2`, proiezioni `to_q,to_kv`) per **entrambe** le fasi:

| Training | Nodo | GPU | Batch per GPU | LR | Step optimizer |
|---|---|---|---:|---:|---:|
| SS | `wn49` | 8 `l40s_vbd` | 1 | `1e-4` | 300.000 |
| SLat | `wn51` | 2 `h100_vbd` | 1 | `5e-5` | 300.000 |

Dal login dell'Ateneo:

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
git pull --ff-only
bash slurm/queue_train_v4_steps.sh
```

Il dataset predefinito e' `morphing_dataset_v4_steps8`. Per usare la versione
a 2 stadi, con lo stesso numero totale di campioni e gli stessi budget:

```bash
bash slurm/queue_train_v4_steps.sh --dataset-stages 2
```

Si possono configurare i budget senza calcolare manualmente le epoche:

```bash
bash slurm/queue_train_v4_steps.sh --ss-steps 300000 --slat-steps 300000 --check
```

`--check` verifica piano, dataset e risorse senza inviare job. Togliendolo,
il comando invia SS e SLat come job indipendenti, con eventuale dipendenza
dalla generazione. I wrapper diretti sono
`train_morphflow_v4_steps_ss.slurm` e `train_morphflow_v4_steps_slat.slurm`;
accettano anche `MAX_TRAIN_STEPS` e `DATA_STAGES` come variabili di ambiente.

Il launcher calcola un tetto di epoche sufficiente per raggiungere il budget,
poi passa al trainer `--max_train_steps` e `--expected_train_steps`. Con
100.000 campioni dopo lo swap e batch 1 per GPU, i default equivalgono a
24 epoche SS e 6 epoche SLat. Per budget non multipli della lunghezza
dell'epoca, il trainer si ferma allo step esatto e completa validation e
salvataggio anche alla fine di un'epoca parziale. Il cosine scheduler usa
il budget di step richiesto.

Le nuove run includono dataset, scope LoRA e budget nel nome, per esempio
`v4_steps8_ss_lora_cross_bs1_lr1e-4_s300000` e
`v4_steps8_slat_lora_cross_bs1_lr5e-5_s300000`. La ripresa richiede la stessa
configurazione. Restano attivi token gate, source swap e prior 0,1 della
configurazione precedente.

## Preset precedente a epoche con LoRA su tutte le attention

I launcher dedicati montano automaticamente la directory padre e usano
esclusivamente `morphing_dataset_v4_steps8`:

| Training | Nodo | GPU | Batch per GPU | Batch globale | LR | Epoche |
|---|---|---|---:|---:|---:|---:|
| SS | `wn49` | 8 `l40s_vbd` | 1 | 8 | `1e-4` | 40 |
| SLat | `wn51` | 2 `h100_vbd` | 1 | 2 | `5e-5` | 20 |

Dal login dell'Ateneo, dopo aver aggiornato la repo:

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
git pull --ff-only
bash slurm/queue_train_v4_steps8.sh
```

Il comando verifica con `sinfo` i nodi e i tipi di GPU prima di inviare i due
job. Se la generazione è ancora attiva, legge il suo job ID dal `run_state.json`
della directory padre e assegna **a entrambi** la dipendenza `afterok` da quel
job. Se il dataset è già completo, li invia senza dipendenze. SS e SLat possono
allenarsi in parallelo: SLat usa i latenti teacher e non aspetta il training SS.
Una generazione fallita o non più in coda deve essere ripresa prima del training.

Per controllare il piano senza inviare job:

```bash
bash slurm/queue_train_v4_steps8.sh --check
```

LoRA usa `attention_scope=all`, rank 32, alpha 64 e dropout 0,1. I target
`to_qkv,to_q,to_kv` includono la self-attention e le due cross-attention di ogni
blocco. I pesi base rimangono congelati; vengono allenati gli adapter LoRA, il
conditioning e il token gate. Restano attivi source swap e prior con peso 0,1,
rollout 25 e 4 passi con gradiente ogni 4 batch, come nella configurazione v4.

Con 100.000 campioni train dopo lo swap, SS esegue 500.000 aggiornamenti e SLat
1.000.000. Questi launcher completano le epoche richieste e non applicano il
budget di 250.000 aggiornamenti usato dalle ablation.

I nomi predefiniti delle run distinguono dataset e configurazione:
`v4_steps8_ss_lora_all_bs1_lr1e4_e40` e
`v4_steps8_slat_lora_all_bs1_lr5e5_e20`. Per nuove run si possono impostare
`SS_RUN_NAME` e `SLAT_RUN_NAME`; rilanciando gli stessi nomi si riprendono i
checkpoint compatibili già presenti.

## Monitoraggio e ripresa

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
OUT=/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_steps8_steps2
python3 slurm/monitor_interp3d_v4.py status --out "$OUT" --watch 30
squeue -u "$USER"
```

Il monitor va puntato alla directory padre, che contiene i progressi degli
otto worker e il totale effettivo dei target da generare. I log sono in
`/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/dataset_v4_interp3d_steps8_steps2/job_JOBID/`.

Per riprendere un job interrotto, rilanciare lo stesso comando con la medesima
configurazione e lo stesso `OUT_NAME`. I target completi vengono riutilizzati;
se una coppia è incompleta il teacher ricostruisce la sua sequenza originale.
Configurazioni diverse richiedono un nuovo output. Gli override mantengono
la convenzione del launcher precedente; per esempio:

```bash
OUT_NAME=morphing_dataset_v4_steps8_steps2_seed123 SEED=123 \
    sbatch --export=ALL slurm/generate_interp3d_v4_multistep.slurm
```

La sintassi Bash e la pianificazione/finalizzazione CPU sono verificabili
localmente. L'inferenza Interp3D, l'uso di memoria delle sequenze a 8 e lo smoke
test CUDA richiedono l'esecuzione sul nodo dell'Ateneo.
