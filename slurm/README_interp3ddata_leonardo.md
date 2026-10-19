# Interp3DData: tre modelli su Leonardo

`benchmark_interp3ddata_leonardo.slurm` confronta MorphFlow, MorphAny3D e
Interp3D sulle immagini di Interp3DData. Usa gli ambienti nativi Leonardo gia'
installati, separati per modello: `envs/morphflow`, `envs/morphany3d`,
`envs/interp3d`. Il renderer comune e le metriche usano MorphAny3D; la
preparazione degli endpoint usa Interp3D. Non servono container o installazioni
negli ambienti. Le cache HF/Torch/U2Net restano quelle di `env.sh`.

Risorse: un nodo Booster, una GPU A100, otto CPU, 64 GiB, massimo 24 ore.
Account `IscrC_MORPHFL`, partizione `boost_usr_prod`, QoS `normal`.
Mail Slurm BEGIN, END e FAIL a `marco.barezzi@unipr.it`.

## Checkpoint e dataset

Da **Ateneo**, copiare il checkpoint SS in una directory dedicata su Leonardo:

```bash
ssh mbarezzi@login.leonardo.cineca.it 'mkdir -p /leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/ateneo_v4_ss17'
rsync -avP --partial-dir=.rsync-partial \
  /hpc/archive/G_VBD/marco.barezzi/morph-benchmark/checkpoints/ss_best_e17.pt \
  mbarezzi@login.leonardo.cineca.it:/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/ateneo_v4_ss17/
```

Il checkpoint SLat predefinito e' il `morphflow_best.pt` del training v4 su
Leonardo. Entrambi sono selezionabili tramite `CHECKPOINT_PATH` e
`SLAT_CHECKPOINT_PATH`; il job ne crea copie immutabili prima di generare.
La ripresa continua a usare queste copie anche se il training aggiorna il best.

Le immagini estratte del dataset devono essere sotto
`/leonardo_work/IscrC_MORPHFL/mbarezzi/datasets/Interp3DData`.
Il dataset e' gia' stato scaricato e indicizzato in questa workspace: 57 coppie
(19 easy, 19 mid, 19 hard), con 114 PNG originali. L'archivio ufficiale e'
conservato in `datasets/downloads/Interp3DData.zip` (26.374.718 byte,
SHA256 `ed2baaa33f235614cd8487e9979ad227f900724b140fda0f09e15db10cbb59e5`).
Il preparatore indicizza le coppie, copia le immagini originali e costruisce
`datasets/interp3ddata_prepared/metadata_test.json`, `split_assets.json` e
`dataset_provenance.json`. Il controllo CPU non richiede latenti gia' presenti.
Il controllo sul login verifica indice, percorsi degli env/checkpoint e parametri
usando Pillow e libreria standard. Gli import dei modelli e il controllo del tipo
SS/SLat dei checkpoint avvengono nel job allocato, prima della generazione.
Nel job GPU, la pipeline Interp3D genera una sola volta gli endpoint canonici
SS/SLat per ciascuna immagine, con seed stabile derivato dal seed della run.
La ricostruzione canonica usa i default TRELLIS: 25 passi e CFG 5 per entrambi
gli stadi; questi valori sono indipendenti dalla griglia CFG di MorphFlow.
Tutti i modelli, CFG e protocolli usano gli stessi endpoint. Questi sono
endpoint ricostruiti dalle immagini: il dataset non fornisce ground truth
tridimensionale, e la preparazione e' registrata nella provenienza.

Override dei percorsi: `DATASET_INPUT_ROOT`, `DATA_ROOT_HOST` e
`SOURCE_IMAGES_ROOT_HOST` (default `<prepared>/images`). Per cambiare seed o
impostazioni della ricostruzione usare una nuova directory preparata e una
nuova run, conservando la provenienza degli endpoint precedenti.

## Configurazione e lancio

Da **Leonardo**, dopo il trasferimento del checkpoint:

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
export CHECKPOINT_PATH=/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/ateneo_v4_ss17/ss_best_e17.pt
export SLAT_CHECKPOINT_PATH=/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo/morphflow_best.pt
export RUN_NAME=interp3ddata_ss17_slatbest_cfg15_seed42_raw_v1
unset RUN_ROOT FINAL_ARCHIVE
export N=0 SEED=42
export CFG_VALUES="1 2 3 5 7.5" SLAT_CFG_VALUES="1 1.5 2"
export WEIGHTED_DENSE=1 EXPORT_GLB=0

# Solo indice e controlli CPU, nessuna inferenza o sottomissione.
bash slurm/submit_interp3ddata_leonardo.sh --check

# Controlla e sottomette un job, quando si decide di lanciarlo.
bash slurm/submit_interp3ddata_leonardo.sh
```

`N=0` usa tutte le coppie; `N=50 SEED=42` seleziona 50 coppie deterministiche
uguali per ogni modello e configurazione. La griglia di default contiene
15 combinazioni SS x SLat. I baseline mantengono i CFG degli autori e vengono
generati e valutati una sola volta per protocollo.

## Due protocolli e output

Il protocollo Interp3D usa cinque intermedi Beta(5,5) e due endpoint, 25 passi
SS e 25 SLat, LPIPS VGG/PPL su 64 viste, FID su due intermedi e 1000 viste
ciascuno. Calcola entrambi i FID: distribuzione dei due intermedi concatenata
e media dei due FID separati, oltre alla diagnostica FID pooled.

La valutazione aggiuntiva `weighted_50/` usa 48 intermedi uniformi
`alpha=i/49` e due endpoint, sempre 64 viste. Calcola LPIPS pesata rispetto
alle due sorgenti e la sua evoluzione; non ricalcola FID/PPL sulla sequenza
densa. Curve, medie, AUC e plot PNG/PDF vengono conservati per ogni CFG e
nel confronto complessivo. La metrica descrive la vicinanza alle sorgenti,
non misura da sola la fluidita' o il realismo della transizione.

I risultati completi sono su FAST:

```text
/leonardo_scratch/fast/IscrC_MORPHFL/mbarezzi/outputs/threeway_benchmark/<RUN_NAME>/
  cfg_grid_metrics.{csv,json}
  cfg_ss_.../final_metrics.{csv,json}
  weighted_endpoint_lpips/evolution.{png,pdf}
  weighted_50/cfg_grid_metrics.{csv,json}
  weighted_50/weighted_endpoint_lpips/evolution.{png,pdf}
  shared/{morphany3d,interp3d}/...
  logs/job_<JOBID>_<timestamp>/<fase>.{out,err}
  progress.log
```

Mesh e Gaussian complete sono nei `mesh_raw.pt`, per tutti gli intermedi,
modelli e CFG. `EXPORT_GLB=0` salta export e baking per tutti i modelli;
i raw consentono di esportare GLB in seguito senza rigenerare le transizioni.
`EXPORT_GLB=1` abilita anche GLB e relativo audit. Restano PNG percettive e
feature FID: evitare di stimare lo spazio contando soltanto gli eventuali GLB.

A job completato, i riepiloghi compatti vengono copiati anche in
`/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/threeway_benchmark/<RUN_NAME>/`.
`RUN_ROOT` e `FINAL_ARCHIVE` permettono altre destinazioni. I temporanei TFSA
usano `<RUN_ROOT>/runtime-tmp`; `THREEWAY_TMP_ROOT` li sposta altrove. Non usare
il piccolo `/tmp` del nodo per questa cache.

Per seguire il job:

```bash
tail -n 80 -F /leonardo_work/IscrC_MORPHFL/mbarezzi/logs/interp3ddata_threeway_<JOBID>.out
tail -n 80 -F /leonardo_scratch/fast/IscrC_MORPHFL/mbarezzi/outputs/threeway_benchmark/<RUN_NAME>/progress.log
```

Rilanciare con gli stessi percorsi, sorgenti, parametri e `RUN_NAME` riprende
endpoint, intermedi e metriche completati. Un lock impedisce scritture
concorrenti nella run; un errore ferma le fasi successive. Cambiando codice,
checkpoint selezionati, dataset o configurazione usare un nuovo `RUN_NAME`.
Le 24 ore sono il limite del singolo job, non una stima della durata totale.
