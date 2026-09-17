# Confronto su nuove coppie FLUX

`generate_teacher_student_comparison.py` campiona **n coppie non ordinate** dalle
immagini nella directory FLUX. Esclude l'unione di tutti i `metadata*.json` del
dataset (train/val/test, anche coppie invertite), dei registri `pair_alphas.json`,
`pair_sequence_alphas.json`, `pair_selected_indices.json` e delle directory coppia
in `targets/`. Esclude anche coppie pianificate ma non completate. Le singole
immagini possono essere già presenti nel training: questo confronto misura la
generalizzazione a **combinazioni nuove**, non necessariamente a oggetti nuovi.
Se non ci sono abbastanza coppie, oppure i metadata non sono leggibili, termina
con errore senza generare un numero inferiore a quello richiesto.

Gli intermedi sono esattamente **k**, agli alpha `i/(k+1)`, esclusi 0 e 1. La
sequenza procede da `src1` a `src2`, quindi gli alpha vengono elaborati in ordine
decrescente. La convenzione è `alpha * src1 + (1-alpha) * src2` per entrambi i
modelli. Il teacher usa MCA e TFSA, `morphing_num=k+2` e tutti i passi consecutivi.
Lo student usa SS → coordinate predette → SLat, senza coordinate del target teacher.

## Lancio sul cluster di ateneo

Dal checkout MorphFlow sul cluster:

```bash
export CHECKPOINT_PATH=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/run_ss/checkpoints/morphflow_best.pt
export SLAT_CHECKPOINT_PATH=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/run_slat/checkpoints/morphflow_best.pt
export OUTPUT_DIR=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/comparison/experiment_01
export NUM_PAIRS=20
export NUM_INTERMEDIATES=9
bash slurm/submit_teacher_student_comparison.sh
```

Default: account/partition/QOS `g_vbd`/`gpu_vbd`/`gpu_vbd`, una L40S, 8 CPU,
128 GB RAM, 24 ore, modulo `singularity/3.8.7`, immagine
`$HOME/containers/trellis-py240-cu118.sif`, ambiente `/opt/trellis/env`.
Il submitter crea `OUTPUT_DIR` prima di inviare il job e imposta i log Slurm:

```text
$OUTPUT_DIR/slurm-comparison-<jobid>.out
$OUTPUT_DIR/slurm-comparison-<jobid>.err
```

Questi file includono anche gli errori di avvio del job. Le opzioni prima di `--`
vanno a `sbatch`, quelle dopo a Python. Ad esempio:

```bash
bash slurm/submit_teacher_student_comparison.sh --time=2-00:00:00 -- --resume
tail -f "$OUTPUT_DIR"/slurm-comparison-*.err
```

Resta possibile `sbatch slurm/generate_teacher_student_comparison.slurm`: lo
script redirige subito stdout/stderr negli stessi file dentro `OUTPUT_DIR`,
prima di caricare moduli e container. Per includere anche gli errori Slurm che
precedono l'avvio dello script, usare il submitter.

Percorsi configurabili tramite variabili esportate:

| Variabile | Default |
| --- | --- |
| `PROJECT_DIR` | `$HOME/trellis-singularity/work/MorphFlow` |
| `MORPHANY3D_DIR` | `$HOME/trellis-singularity/work/MorphAny3D` |
| `TRELLIS_BASE` | `$HOME/trellis-singularity` |
| `SIF` | `$HOME/containers/trellis-py240-cu118.sif` |
| `ASSETS_DIR` | `/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs` |
| `DATASET_DIR` | `/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v3` |
| `OUTPUT_DIR` | archivio `morphflow_runs/comparison/run_<timestamp>_<pid>` con il submitter; `job_<jobid>` con sbatch diretto |
| `WORK_CACHE_ROOT` | `$SLURM_TMPDIR`, altrimenti `/tmp` |

Il checkout MorphAny3D deve contenere `trellis/` con i metodi di morphing già usati
dal generatore del dataset. Il TRELLIS originale da solo non è sufficiente.
Il launcher monta esplicitamente i percorsi di input, checkpoint, output e cache,
anche se i checkpoint si trovano fuori dall'archivio predefinito. Le GPU visibili
restano quelle assegnate da Slurm. Teacher e student girano in processi separati
sulla stessa GPU, liberando i pesi del teacher prima di caricare i due checkpoint.

Parametri principali esportabili: `NUM_PAIRS=10`, `NUM_INTERMEDIATES=5`, `SEED=42`,
`STEPS=50`, `SLAT_STEPS=50`, `CFG_SCALE=3.0`, `SLAT_CFG_SCALE=3.0`,
`TEACHER_SS_STEPS=25`, `TEACHER_SLAT_STEPS=25`, `TEACHER_SS_CFG=7.5`,
`TEACHER_SLAT_CFG=3.0`, `TFSA_ALPHA=0.8`, `TFSA_CACHE_MODE=memory`,
`MAX_WORK_CACHE_GB=60`, `MIXED_PRECISION=auto`, `MODEL_ID=microsoft/TRELLIS-image-large`.
Il default `memory` conserva in RAM i tensori di attenzione TFSA, evitando le
grandi scritture su `/tmp` che possono causare `PytorchStreamWriter ... file write
failed` / `unexpected pos`. Restano solo eventuali piccoli file ausiliari nella
directory temporanea. MCA e TFSA restano attive.
Con `file` la cache richiede spazio su disco: prima di caricare i modelli si
verifica la disponibilità di `MAX_WORK_CACHE_GB + 2` GiB. Su nodi con NVMe si può
impostare `WORK_CACHE_ROOT=/nvme/$USER`. Il controllo dello spazio libero non
verifica le quote personali; gli errori di scrittura riportano directory e spazio
residuo. Il limite è per coppia; `MAX_WORK_CACHE_GB=0` disabilita il limite dei
tensori (resta il controllo di almeno 2 GiB liberi in modalità file).
La cache temporanea viene eliminata
all'uscita, compresi gli errori intercettati. Un arresto forzato del job può
richiedere la pulizia della sua directory temporanea.

## Dry run e uso diretto

Il dry run usa solo la libreria standard Python e non carica checkpoint o modelli:

```bash
python generate_teacher_student_comparison.py \
  --assets-dir /path/flux_outputs \
  --dataset-dir /path/morphing_dataset_v3 \
  --checkpoint-path /path/ss.pt \
  --slat-checkpoint-path /path/slat.pt \
  --output-dir ./outputs/comparison/experiment_01 \
  --num-pairs 20 --num-intermediates 9 --seed 42 --dry-run
```

Scrive `plan.dry_run.json`. Togliendo `--dry-run`, gli stessi parametri avviano
la generazione e scrivono `plan.json`. Per uso diretto serve l'ambiente GPU del
progetto, con `TRELLIS_REPO=/path/MorphAny3D` e il checkout nel `PYTHONPATH`.
La modalità dry run è disponibile anche con
`bash slurm/submit_teacher_student_comparison.sh -- --dry-run`; il launcher
verifica comunque l'esistenza dei checkpoint e richiede le risorse del job.

## Output e confronto metriche

```text
output/
  slurm-comparison-<jobid>.out      # avanzamento e stdout
  slurm-comparison-<jobid>.err      # traceback, warning e stderr
  plan.json                       # input, checkpoint, esclusioni, seed, alpha
  status.json                     # running / failed / teacher_complete / complete
  assets/<asset>/                  # sorgenti TRELLIS condivise
    input.png                     # o input.jpg/jpeg/webp, copia originale
    ss_latent.pt
    slat_feats.pt
    slat_coords.pt
    structured_latent.pt
    occupancy.pt
    mesh.glb
    manifest.json
    complete.json
  teacher/
    assets -> ../assets
    metadata.json
    pair_000000/
      src1, src2                  # link agli asset completi
      src1.glb, src2.glb
      sequence.json
      alpha_0001/                 # indice temporale; valore alpha nei JSON
        ...                       # stessi latenti, occupancy e GLB degli asset
        pred_final.glb -> mesh.glb
        metadata.json
  student/                        # stesso layout e stessi alpha del teacher
  comparison/
    pair_000000_alpha_0001/
      src1.glb, src2.glb, target.glb, pred_final.glb
      metrics.json                # identificatori; nessuna metrica calcolata
  comparison_manifest.json        # esattamente n*k corrispondenze completate
```

Le sorgenti sono ricostruite una volta con TRELLIS, poi usate identiche per
teacher e student; gli endpoint non sono predizioni dello student. I link sono
relativi e interni all'output: spostare/copiare l'intera directory preservando i
symlink. Le mesh usano lo stesso esportatore GLB con colori ai vertici delle
valutazioni esistenti (colore neutro se il decoder non restituisce colori).
I latenti SLat salvati sono denormalizzati, come quelli del dataset.
Non vengono prodotti Gaussian PLY né rendering.

`eval_fid_kid.py` può leggere `--eval_dir <output>/comparison`.
`eval_perceptual_sequence.py` può leggere separatamente
`--run_dir <output>/teacher` e `--run_dir <output>/student`, con
`--strict_uniform_alphas` (servono almeno due intermedi per le metriche di
sequenza). Questi evaluator eseguono rendering se lanciati: lo script di
generazione prepara solo gli asset, senza eseguirli.

## Ripresa e riproducibilità

Per riprendere sullo stesso output, mantenere gli stessi parametri e aggiungere
`--resume`, oppure esportare `RESUME=1` prima di `sbatch`. È necessario mantenere
anche lo stesso `OUTPUT_DIR`, perché il default cambia con il job ID.
`--stage teacher` / `STAGE=teacher` esegue solo il teacher;
`--stage student --resume` / `STAGE=student RESUME=1` completa lo student.

Il piano registra hash SHA-256 delle immagini selezionate e dei metadata di
esclusione, dimensione/mtime dei checkpoint e tutti i parametri. La ripresa
rifiuta input o parametri di generazione differenti. Si possono cambiare la
modalità `TFSA_CACHE_MODE` e il limite `MAX_WORK_CACHE_GB` (oltre alla directory
temporanea): il piano registra le nuove impostazioni e riutilizza gli asset
completi. Per riprendere un job fallito scrivendo la cache su `/tmp`, mantenere
gli stessi checkpoint, n, k, seed e output, poi usare:

```bash
export TFSA_CACHE_MODE=memory
export RESUME=1
bash slurm/submit_teacher_student_comparison.sh
```

Gli asset completi vengono saltati;
se una sequenza teacher è parziale, viene ricalcolata dall'inizio per ricostruire
la TFSA, conservando gli asset già completati. Errori CUDA, mesh vuote o limiti di
cache interrompono il job, evitando di pubblicare silenziosamente meno di n*k
campioni. Non viene sostituita una coppia fallita con una coppia più facile.

Teacher e sorgenti usano `SEED`. Lo student mantiene il seed fisso fra gli alpha
di ciascuna coppia, con seed distinti per SS e SLat registrati nei manifest.
Questo riduce la variabilità della sequenza dovuta al rumore; non implica lo
stesso rumore nei due modelli o risultati bit-identici fra hardware diversi.

Verifiche CPU: `python -m unittest discover -s tests -p 'test_teacher_student_comparison.py'`.
L'inferenza completa richiede CUDA, i checkpoint e il checkout MorphAny3D.
