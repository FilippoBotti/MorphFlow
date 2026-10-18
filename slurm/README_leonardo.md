# MorphFlow su Leonardo

I launcher usano l'ambiente nativo `/leonardo_work/IscrC_MORPHFL/mbarezzi/env.sh`,
2 nodi Booster, 4 GPU per nodo, 32 CPU e 128 GiB di RAM per nodo.
Slurm avvia un launcher Accelerate per nodo e ogni launcher crea 4 processi GPU.
I launcher v3 includono notifiche mail `ALL` per inizio, fine ed errore.

## SLat v4 Interp3D (13 ottobre 2026)

`train_morphflow_v4_slat_leonardo.sbatch` replica la configurazione di
`train_morphflow_v4_ss.slurm` dell'ateneo per il target **SLat**, con LR
`5e-5` per condition, flow adapter e LoRA. Usa 2 nodi Booster x 4 GPU,
batch 1 per GPU (globale 8), BF16 e 20 epoche dal pretrained TRELLIS.
Il dataset e' `datasets/morphing_dataset_v4` su `$FAST/$USER`:
50000 esempi train, 2000 validation e 2000 test; source swap porta il
training a 100000 esempi virtuali per epoca.

Restano i parametri v4: encoder sparse_conv3d, conditioning separato con
gate token, LoRA cross-attention rank 32 / alpha 64, scheduler cosine,
warmup LR 10000, weight decay 5e-5 e 2 DataLoader worker per GPU.
Il prior ha peso 0.1, warmup 10000, cadenza 4, rollout 25 passi con
gradiente sugli ultimi 4 e checkpointing; tau [0.05, 0.20], clip 0.10,
RMS guard peso 1 e limiti [0.25, 2.0]. Semantic matching/cycle/usage,
gradient balancing, EMA e CFG dropout sono disabilitati.

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
bash slurm/train_morphflow_v4_slat_leonardo.sbatch --check
sbatch slurm/train_morphflow_v4_slat_leonardo.sbatch
# Ripresa dopo il limite di 24 ore; TRAIN_EPOCHS resta il totale (20).
AUTO_RESUME=1 sbatch slurm/train_morphflow_v4_slat_leonardo.sbatch
```

Run predefinita: `v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo`.
Log/TensorBoard: `$WORK/$USER/outputs/<run>/`; checkpoint:
`$WORK/$USER/checkpoints/<run>/`; log Slurm:
`$WORK/$USER/logs/v4_slat_pr25g4_lr5e-5_bs1_<jobid>.{out,err}`.
`morphflow_last.pt` viene salvato a fine epoca; snapshot ogni 2 epoche.
La ripresa e' manuale. Il launcher verifica stato e dimensioni del dataset
v4 e riusa ambiente, cache offline e avvio multinodo del launcher Leonardo
comune. Sono disponibili anche `--prepare` e `--comm-check`.

### Ripresa a blocchi di tre epoche (16 ottobre 2026)

Per la ripresa dal checkpoint di epoca 3 sono usati `TRAIN_EPOCHS=20`
e `STOP_AFTER_EPOCH=6`: il primo mantiene lo scheduler cosine originale,
il secondo ferma il ciclo dopo validazione e salvataggio dell'epoca 6.
Il job dipendente usa `STOP_AFTER_EPOCH=9` e `AUTO_RESUME=1` per eseguire
le epoche 7–9 solo dopo il successo del primo (`afterok`). Entrambi hanno
limite di 24 ore, checkpoint periodico ogni epoca e ripristino completo di
modello, optimizer e scheduler da `morphflow_last.pt`.
Gli ID e la configurazione della sottomissione sono salvati in
`$WORK/$USER/logs/v4_slat_resume_20261016.json`.

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

## Eval e sequenze alpha sul test v4

I due nuovi launcher usano `env.sh`, cache offline e, per default,
`$FAST/$USER/datasets/morphing_dataset_v4/metadata_test.json`.
Il test v4 contiene **500 coppie uniche e 2000 target** (quattro alpha per coppia).
Entrambi richiedono `CHECKPOINT_PATH` per il modello SS e
`SLAT_CHECKPOINT_PATH` per il modello SLat; l'architettura viene letta dai
checkpoint. Gli override mantengono i nomi dei launcher precedenti.

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
export CHECKPOINT_PATH=/percorso/checkpoint_ss.pt
export SLAT_CHECKPOINT_PATH=/percorso/checkpoint_slat.pt

# Controlli CPU: import, dati/selezione, checkpoint SS/SLat e cache decoder.
bash slurm/eval_morphflow_v4_leonardo.slurm --check
NUM_PAIRS=10 NUM_INTERMEDIATE_STEPS=9 \
    bash slurm/generate_alpha_steps_test_v4_leonardo.slurm --check
# Se mancano decoder in cache, usare --prepare dal login con gli stessi parametri.
```

`eval_morphflow_v4_leonardo.slurm` richiama lo stesso evaluator del v3:
seleziona `NUM_SAMPLES` coppie uniche con seed fisso e un alpha del metadata
per coppia. Confronta le predizioni con i target Interp3D del v4, usando le
stesse coppie e alpha in tutta la griglia `CFG_VALUES x SLAT_CFG_VALUES`.
I default restano 10 coppie, 50 passi SS e SLat, seed 42, BF16,
`SAVE_LATENTS=0` e CFG `1 1.5 2 2.5 3` per entrambi i flow (25 combinazioni).
Risorse: 1 GPU, 8 CPU, 32 GiB, 4 ore.

```bash
NUM_SAMPLES=50 STEPS=50 SLAT_STEPS=50 \
CFG_VALUES="1 2 3" SLAT_CFG_VALUES="1 2 3" \
RUN_NAME=v4_test_cfggrid \
    sbatch slurm/eval_morphflow_v4_leonardo.slurm
```

Rimangono disponibili `ROOT_DIR`, `METADATA`, `SOURCE_IMAGES_ROOT`,
`SOURCE_IMAGE_FILENAME`, `PROJECT_DIR`, `EVAL_TYPE`, `RUN_NAME`, `OUTPUT_DIR`,
`NUM_SAMPLES`, `SEED`, `STEPS`, `SLAT_STEPS`, `CFG_VALUES`, `SLAT_CFG_VALUES`,
`TRELLIS_MODEL`, `MIXED_PRECISION`, `SAVE_LATENTS` e i due checkpoint.
`SLAT_CFG_SCALE` e' accettato come CFG SLat singolo se `SLAT_CFG_VALUES`
non e' impostato. Output predefinito:
`$WORK/$USER/outputs/eval_test_latents_v4/<RUN_NAME>/job_<id>/cfg_ss_<cfg>_slat_<cfg>/<timestamp>/`.
Contiene GLB, metriche per campione, `summary.json`, `selected_samples.json`
e, se richiesti, latenti.

`generate_alpha_steps_test_v4_leonardo.slurm` esegue la pipeline
SS -> coordinate sparse -> SLat -> mesh su `NUM_PAIRS` coppie di test e
`NUM_INTERMEDIATE_STEPS` alpha intermedi equispaziati. Sono due conteggi
distinti da `STEPS` e `SLAT_STEPS`, che controllano l'integrazione dei flow.
Con `NUM_INTERMEDIATE_STEPS=9` genera alpha `0.9, 0.8, ..., 0.1`;
alpha 1 indica src1 e alpha 0 indica src2.

```bash
NUM_PAIRS=10 NUM_INTERMEDIATE_STEPS=9 \
SELECTION=random SEED=42 STEPS=50 SLAT_STEPS=50 \
CFG_SCALE=3 SLAT_CFG_SCALE=3 RUN_NAME=v4_test_10pairs_9alpha \
    sbatch slurm/generate_alpha_steps_test_v4_leonardo.slurm
```

I default restano 10 coppie, 5 intermedi, `SELECTION=first`,
`START_INDEX=0`, `DEDUPLICATE_PAIRS=1`, `INCLUDE_ENDPOINTS=0`,
`DIRECTION=src1_to_src2`, seed 42, 50 passi per flow, CFG 3 per flow,
`TRELLIS_MODEL=image_large`, BF16, `SAVE_LATENTS=0`, `SAVE_VOXELS=1` e
`SAVE_SOURCES=1`. `SELECTION` accetta `first`, `random`, `evenly_spaced`;
`START_INDEX` si applica alla lista dopo la deduplicazione. Se si richiedono
piu' coppie di quelle disponibili, vengono usate le rimanenti con un avviso.
`INCLUDE_ENDPOINTS=1` aggiunge due predizioni, alpha 1 e 0; i GLB delle
sorgenti sono salvati separatamente quando `SAVE_SOURCES=1`.
`ALPHAS="0.75,0.5,0.25"` permette anche una lista esplicita, che sostituisce
la griglia automatica e ne determina l'ordine.

Risorse: 1 GPU, 8 CPU, 64 GiB, 8 ore. Output predefinito:
`$WORK/$USER/outputs/alpha_steps_test_v4/<RUN_NAME>/job_<id>/<timestamp>_test_k..._n..._pipeline/`,
con `selected_pairs.json`, `summary.json` e una directory per coppia
contenente sorgenti, GLB generati e metadati della sequenza. Anche qui
`OUTPUT_DIR`, `ROOT_DIR`, `METADATA`, `PROJECT_DIR` e i parametri delle immagini
sono modificabili. I log di entrambi i launcher sono in `$WORK/$USER/logs/`.

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

## Morphing di due GLB a un alpha, con tutte le ricostruzioni

`morph_glb_pair.py` accetta due GLB, checkpoint SS e SLat espliciti, un alpha
e la cartella di output. Richiama l'encoding e la pipeline di
`generate_glb_pair.py` con un solo CFG per flow e genera sempre anche gli
endpoint alpha 1 e 0. Gli endpoint attraversano entrambi i modelli SS/SLat;
le ricostruzioni dirette encoder/decoder sono salvate separatamente.

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
sbatch slurm/morph_glb_pair_leonardo.sbatch \
    /percorso/primo.glb /percorso/secondo.glb \
    --ss-checkpoint /percorso/checkpoint_ss.pt \
    --slat-checkpoint /percorso/checkpoint_slat.pt \
    --alpha 0.35 \
    --output-dir /leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/morph_personale \
    --cfg-ss 3 --cfg-slat 3 --steps 50 --slat-steps 50
```

La convenzione e' **alpha 1 = primo GLB, alpha 0 = secondo GLB**.
Con alpha 0 o 1 vengono comunque prodotti tutti e cinque i file: il morph
coincide con la relativa generazione endpoint, senza rigenerarla.
I GLB finali sono direttamente nella cartella passata a `--output-dir`,
senza aggiungere timestamp o ID del job:

| File | Contenuto |
|---|---|
| `reconstruction_decoder_src1.glb` | Encoding del primo GLB -> mesh decoder TRELLIS |
| `reconstruction_decoder_src2.glb` | Encoding del secondo GLB -> mesh decoder TRELLIS |
| `reconstruction_model_src1_alpha1.glb` | Generazione completa del modello ad alpha 1 |
| `reconstruction_model_src2_alpha0.glb` | Generazione completa del modello ad alpha 0 |
| `morph.glb` | Generazione completa all'alpha richiesto |
| `summary.json` | Input, checkpoint, alpha, CFG, seed e percorsi dei risultati |

L'encoding segue la pipeline dataset TRELLIS presente nel checkout: Blender
normalizza ogni scena nel cubo unitario e renderizza 50 viste a 512px;
voxelizzazione 64^3; DINOv2 ViT-L/14 con register su immagini 518px;
token `x_prenorm` proiettati sui voxel, interpolati bilinearmente e mediati
sulle viste; feature arrotondate a FP16 come nella cache TRELLIS; encoder
SLat con `sample_posterior=False`. Si passa al modello il latente grezzo:
la normalizzazione del conditioning viene gestita dal modello secondo il
checkpoint, e il latente generato viene denormalizzato prima del decoder.
Per GLB con soli colori ai vertici (`COLOR_0`) e senza materiale, il renderer
collega esplicitamente il layer colore al materiale Blender: altrimenti
Blender 3.0/Cycles renderizza grigio. I materiali gia' presenti sono conservati.
L'encoding versione 2 include questa correzione: per run precedenti usare una
nuova cartella di output, cosi' render e latenti vengono ricalcolati.
Rendering, trasformazioni, voxel, latenti e risultati della pipeline restano
in `pipeline/`. I GLB sono normalizzati, Y-up, con colori ai vertici prodotti
dal decoder; non ripristinano dimensioni/traslazioni originali o texture UV.

Default: 50 passi per flow, CFG 3 per flow, seed 42, BF16, 50 viste,
batch DINO 4. Modificabili con `--steps`, `--slat-steps`, `--cfg-ss`,
`--cfg-slat`, `--seed`, `--mixed-precision`, `--encoding-views`,
`--dino-batch-size`, `--blender`. Sono accettati anche gli alias
`--checkpoint_path`, `--slat_checkpoint_path`, `--output_dir`,
`--cfg_scale`, `--slat_cfg_scale`, `--slat_steps`.

Per verificare dal login, usare **gli stessi argomenti** sostituendo
`sbatch slurm/morph_glb_pair_leonardo.sbatch` con
`bash slurm/morph_glb_pair_leonardo.sbatch --check`. `--prepare` prepara le
cache quando mancano. Queste modalita' non generano mesh. Per esecuzione
diretta in un ambiente GPU gia' attivo, usare `python morph_glb_pair.py`
con gli stessi argomenti e `--blender /percorso/blender` (o `BLENDER_BIN`).
Il launcher Leonardo richiede 1 GPU, 8 CPU, 64 GiB e 4 ore.

La cartella di output ha un lock per evitare scritture concorrenti. A parita'
di configurazione una riesecuzione riusa encoding e predizioni completati;
per input/checkpoint o parametri diversi, usare una nuova cartella.
Gli errori della generazione sono riportati in `last_error.json` e nel log
Slurm; `summary.json` viene pubblicato solo quando tutti i risultati esistono.

## Inferenza da due GLB con encoding multiview e griglia CFG

`generate_glb_pair.py` accetta due GLB arbitrari, checkpoint SS/SLat espliciti
(SS standard e conditioning SLat) e una griglia cartesiana di CFG.
Il launcher `generate_glb_pair_leonardo.sbatch` richiede 1 GPU, 8 CPU, 64 GiB
per 4 ore. Le notifiche includono inizio, fine e fallimento.

```bash
# Default: assets/Werewolf_Warrior.glb, assets/pool3913.glb;
# ultimi checkpoint con prior disponibili al 7 ottobre 2026:
# SS tangentScaleAnchor ep25/step125000; SLat prior ep5/step50000.
bash slurm/generate_glb_pair_leonardo.sbatch --prepare
sbatch slurm/generate_glb_pair_leonardo.sbatch

# Asset, checkpoint e CFG personalizzati:
SS_CHECKPOINT=/percorso/ss.pt SLAT_CHECKPOINT=/percorso/slat.pt \
CFG_SS='2,2.5,3' CFG_SLAT='1.5,2,2.5' \
sbatch slurm/generate_glb_pair_leonardo.sbatch /percorso/primo.glb /percorso/secondo.glb
```

Per ogni asset: rendering Blender di 50 viste a 512px, normalizzazione nello
stesso cubo del dataset, voxelizzazione 64^3, DINOv2 ViT-L/14 con register,
proiezione e media delle feature sulle viste, encoder TRELLIS SLat con media
posterior deterministica. `ENCODING_VIEWS` e `DINO_BATCH_SIZE` sono modificabili.
I latenti di condizionamento vengono salvati e riutilizzati, senza ricodifica
per ogni CFG. Le ricostruzioni dell'encoder sono salvate separatamente.

Con i default vengono prodotti 27 GLB: 3 CFG SS x 3 CFG SLat x 3 alpha.
La convenzione e' `alpha=1 -> primo GLB`, `alpha=0 -> secondo GLB`.
Anche alpha 0 e 1 sono generati dal modello, non copie dei file originali.
SS e SLat usano 50 passi Euler e BF16; il seed SS e quello SLat sono fissi
per confrontare i CFG. Le 9 strutture SS vengono generate una sola volta.

Output: `outputs/glb_pair/werewolf_pool_prior_cfggrid/job_<id>/`:

- `encoding/src{1,2}/`: viste, mesh normalizzata, voxel, `slat.pt`, `reconstruction.glb`.
- `ss/`: i 9 latenti SS con le coordinate generate.
- `cfg_ss_<cfg>_slat_<cfg>/alpha_<alpha>/`: `prediction.glb`, `slat.pt`, `result.json`.
- `run.json` e `results.json`: configurazione, checkpoint, stato e indice risultati.

I GLB esportano i colori ai vertici del mesh decoder, senza texture baking UV.
Gli assi tornano a Y-up per la visualizzazione GLB. Encoder, flow SS e flow
SLat vengono caricati in fasi successive per contenere la memoria GPU.

`OUTPUT_DIR` consente di riprendere la stessa cartella con configurazione e
file sorgente invariati: gli output completati vengono riutilizzati. In caso
di errore vengono prodotti traceback Slurm e `last_error.json`.
`--check` verifica import, checkpoint e cache senza fare inferenza GPU.
