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
