# Confronto completo su otto L40S dell'Ateneo

`benchmark_threeway_interp3d_l40s.slurm` usa la configurazione dei launcher
funzionanti `train_morphflow_v3.slurm`, `generate_interp3d_v4.slurm` e del
precedente benchmark: account `g_vbd`, partizione/QoS `gpu_vbd`, nodo `wn49`,
otto `gpu:l40s_vbd`, 64 CPU, 700 GiB RAM, massimo 10 giorni; modulo
`singularity/3.8.7` e SIF `$HOME/containers/trellis-py240-cu118.sif`.
Mail Slurm BEGIN, END e FAIL a `marco.barezzi@unipr.it`.
Il default `EXPORT_GLB=0` conserva mesh e Gaussian nei raw e rimanda l'export GLB.
Si applica a MorphFlow, MorphAny3D e Interp3D, in entrambi i protocolli.

## Preparazione e lancio sull'HPC Ateneo

Aggiornare il checkout MorphFlow sull'Ateneo con **tutti** i file modificati
di questo lavoro: launcher, helper, benchmark driver, evaluator e
`eval_perceptual_sequence.py`. I soli file `.slurm` non bastano.
I checkpoint devono essere gia' disponibili su quel filesystem. Pubblicare tutti
i file modificati e nuovi nel repository, quindi fare `git pull` sull'Ateneo.

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
git pull
cp slurm/threeway_interp3d_l40s.env.example threeway_l40s.env
# Modificare CHECKPOINT_PATH, SLAT_CHECKPOINT_PATH e RUN_NAME nel file.
set -a
source ./threeway_l40s.env
set +a

# Una volta dal login con accesso internet: dipendenze metriche e pesi.
bash slurm/prepare_threeway_interp3d_l40s.sh --prepare

# Verifica offline, checkpoint SS/SLat e piano completo senza inferenza GPU.
bash slurm/benchmark_threeway_interp3d_l40s.slurm --check

sbatch --export=ALL slurm/benchmark_threeway_interp3d_l40s.slurm
```

Per la coppia migliore disponibile al 19 ottobre 2026 (SS epoca 13, SLat epoca 9),
eseguire **dal login Ateneo**; i file sorgente sono snapshot immutabili:

```bash
DEST=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/checkpoints/v4_ss13_slat9
mkdir -p "$DEST"
rsync -avP \
  mbarezzi@login.leonardo.cineca.it:/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/ateneo_v4_ss13_slat9_20261019/ \
  "$DEST/"

cd "$HOME/trellis-singularity/work/MorphFlow"
unset CONFIG_FILE RUN_ROOT FINAL_ARCHIVE
export CHECKPOINT_PATH="$DEST/ss_best.pt"
export SLAT_CHECKPOINT_PATH="$DEST/slat_best.pt"
export RUN_NAME=v4_ss13_slat9_interp3d_weighted50
export N=0 CFG_VALUES="2 3 5" SLAT_CFG_VALUES="1.5 2 2.5" WEIGHTED_DENSE=1
export EXPORT_GLB=0
# La preparazione e' necessaria solo se dipendenze/pesi non sono ancora presenti.
bash slurm/prepare_threeway_interp3d_l40s.sh --prepare
sbatch --export=ALL slurm/benchmark_threeway_interp3d_l40s.slurm
```

E' possibile usare direttamente variabili esportate senza file di configurazione.
Il launcher supporta anche `CONFIG_FILE=/percorso/assoluto/config.env`; i valori
in quel file prevalgono sulle variabili ereditate. Per usare gli stessi override
nel preparatore, esportarli con `set -a; source ...; set +a` come sopra.
I due checkpoint sono liberamente selezionabili, uno SS e uno SLat; ogni
esecuzione usa la stessa coppia su tutte le combinazioni dei due elenchi CFG.

Per una prova completa piccola, mantenere gli stessi conteggi di viste/passi:

```bash
N=2 CFG_VALUES=2 SLAT_CFG_VALUES=1.5 RUN_NAME=v4_l40s_smoke \
sbatch --export=ALL --time=04:00:00 slurm/benchmark_threeway_interp3d_l40s.slurm
```

L'allocazione resta di otto GPU; con meno di otto coppie partono soltanto i worker
necessari. `--check` verifica gli input e gli import CPU; il job verifica CUDA,
fork corretto e renderer prima della generazione. Con `EXPORT_GLB=1` verifica
anche il contesto nvdiffrast per il baking. La verifica CUDA
su L40S richiede l'esecuzione sull'Ateneo: non e' stata simulata come prova GPU.

## Ambienti e input

Gli ambienti esistenti sono montati in sola lettura, sempre come `/opt/trellis/env`:

| Metodo | Env host predefinito | Repository host predefinito |
| --- | --- | --- |
| MorphFlow | `$HOME/trellis-singularity/env` | TRELLIS rilevato dai percorsi del launcher v3 |
| MorphAny3D | `$BENCH_ROOT/morphany3d/env` | `$BENCH_ROOT/morphany3d/work/MorphAny3D` |
| Interp3D | `$BENCH_ROOT/interp3d/env` | `$BENCH_ROOT/interp3d/work/Interp3D` |

Ogni processo importa un solo fork TRELLIS dal proprio `/workspace/MethodRepo`.
Toolchain CUDA 11.8/GCC11: `$BENCH_ROOT/morphany3d/toolchain/cuda118-gcc11`.
Cache: `$BENCH_ROOT/shared-cache/{hf-home,torch-home,u2net}`. `pytorch-fid==0.3.0`
viene preparato, se necessario, in un overlay Python distinto
`$BENCH_ROOT/protocol-python`; gli env di training non vengono modificati.
Il preparatore verifica anche gli hash dei pesi canonici Inception FID e VGG16.
Il batch esegue nuovamente il controllo offline, senza installazioni/download.

Dataset predefinito: `/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4`;
immagini: directory sorella `flux_outputs`. **N=0 significa tutte le coppie uniche
ordinate di metadata_test.json**, non un campione di righe alpha: il v4 corrente
contiene 2000 righe e 500 coppie. Il piano controlla split, latenti sorgenti e
immagini senza scartare silenziosamente input mancanti.

Il launcher salva copie immutabili dei checkpoint, dei metadati, degli split,
del codice MorphFlow e dei tre fork prima di iniziare. I processi eseguono le
copie del codice. I checkpoint possono aggiornarsi durante il training: una
ripresa usa le copie gia' fissate. Per scegliere altri pesi usare un nuovo
RUN_NAME, anche se il filename originale resta morphflow_best.pt.

## Protocollo e due FID

Riferimento: [Interp3D, sezioni 5.1, 5.2 e A.2.2](https://arxiv.org/html/2601.14103v1#A2.SS2).
Cinque intermedi e due endpoint, Beta(5,5), 25 passi SS e 25 SLat, 64 viste per
stato per VGG LPIPS/PPL, due intermedi casuali e 1000 viste per stato per FID.
PPL principale: somma delle distanze LPIPS sui sei passaggi dell'intera sequenza,
normalizzata per LPIPS fra gli endpoint, poi media delle 64 viste.
Le liste CFG, coppie, rumori MorphFlow per coppia, alpha, camere e selezione FID
restano uguali nel confronto. I baseline mantengono i CFG degli autori.

Sia R la concatenazione delle feature dei due endpoint (2000 immagini), e
G1/G2 le feature dei due intermedi selezionati (1000 immagini ciascuno):

| Campo CSV | Calcolo |
| --- | --- |
| `fid_endpoint_manifold` | media sulle coppie di `FID(R, concat(G1,G2))` |
| `fid_endpoint_manifold_mean_per_state` | media sulle coppie di `(FID(R,G1)+FID(R,G2))/2` |
| `fid_endpoint_manifold_pooled` | FID aggregato su tutte le coppie; diagnostica distinta |

Entrambi i FID principali usano **gli stessi due stati e gli stessi render**.
I JSON conservano anche i FID per stato e le formule. Le feature rimangono
disponibili per altre verifiche; i render delle 1000 viste vengono elaborati in
streaming senza salvarne tutte le PNG. Le 64 viste percettive sono salvate.

L'eval usa le Gaussian autentiche con renderer TRELLIS comune, camere sferiche
Hammersley, raggio 2, FOV 40, sfondo bianco, 512x512. Queste scelte sono esplicite:
il paper non descrive ogni dettaglio del renderer/FID. Il confronto riguarda
il test v4 scelto, non i valori della tabella su Interp3DData. PDV resta una
metrica aggiuntiva. I conteggi di frame/passi/viste del paper sono fissi nel
launcher: override incompatibili vengono rifiutati prima di generare.

## LPIPS pesata sulle due sorgenti

Per ogni intermedio viene calcolata anche
`alpha * LPIPS(frame, src1) + (1-alpha) * LPIPS(frame, src2)`, con alpha=1
corrispondente a src1. Vengono conservate sia le due distanze originali sia
i contributi pesati. Il valore e' mediato prima sulle 64 viste, poi sui cinque
intermedi e infine sulle coppie con uguale peso. Il campo
`weighted_endpoint_lpips_mean` compare in `final_metrics.csv` e
`cfg_grid_metrics.csv`; `weighted_endpoint_lpips_auc` e' l'integrale trapezoidale
nel solo intervallo alpha campionato, senza estrapolazione agli endpoint.
Si tratta di una misura di vicinanza alle sorgenti, aggiuntiva al protocollo
Interp3D, che non misura da sola fluidita' o realismo.

Il launcher produce automaticamente, dopo ogni CFG completato:

- `cfg_ss_.../weighted_endpoint_lpips/evolution.{png,pdf}`: confronto dei tre
  modelli lungo il passaggio, distanze dalle due sorgenti e contributi pesati;
- `weighted_endpoint_lpips/evolution.{png,pdf}` nella radice: confronto di tutti
  i CFG MorphFlow con i due baseline, riportati una volta;
- `curves.csv`, `summary.csv`, `summary.json` nelle stesse directory: valori
  per alpha, medie e variabilita' tra le coppie;
- per ciascuna coppia/metodo, i file gia' prodotti dall'evaluator
  `metrics/<metodo>/pair_.../weighted_endpoint_lpips.{csv,png,pdf}` e
  `weighted_endpoint_lpips_per_view.csv` conservano l'evoluzione individuale.

L'asse del confronto va da src1 a src2 usando progresso=1-alpha. Mostra soltanto
gli intermedi realmente campionati; nessun endpoint a distanza zero e' aggiunto
artificialmente. Le bande dei confronti descrivono la variabilita' tra coppie,
quelle dei grafici individuali la variabilita' tra camere.
I grafici e CSV aggregati vengono copiati anche nell'archivio compatto.
Il report riusa esclusivamente i risultati LPIPS gia' calcolati, senza inferenza,
nuovi render o ricalcolo FID. Si puo' rigenerare separatamente nell'env metriche:

```bash
python benchmarks/weighted_endpoint_report.py --run-root /benchmark
# Opzionale: anche un confronto dei tre modelli per ciascuna coppia.
python benchmarks/weighted_endpoint_report.py --run-root /benchmark --per-pair-plots
```

`/benchmark` e' il mount del container: su host usare il percorso RUN_ROOT
reale se si esegue Python direttamente in un ambiente con NumPy e Matplotlib.

## Sequenza aggiuntiva da 50 frame

`WEIGHTED_DENSE=1` (default) aggiunge, dopo il confronto Interp3D, una valutazione
distinta in `weighted_50/`. La prima valutazione conserva i cinque intermedi
Beta(5,5), LPIPS/PPL e i due FID. La seconda genera **48 intermedi uniformi**, con
alpha `48/49, 47/49, ..., 1/49`, e usa le due sorgenti canoniche come primo e ultimo
frame: **50 frame totali**. Sono gli stessi due endpoint per tutti gli approcci.
Si usano sempre 25 passi SS, 25 SLat, 64 viste comuni e VGG LPIPS. Questa seconda
valutazione calcola esclusivamente le distanze LPIPS dalle sorgenti: nessun FID,
PPL o distanza fra intermedi viene ricalcolato sulla sequenza densa.

| Campo | Frame usati |
| --- | --- |
| `weighted_endpoint_lpips_mean` | media sui 48 intermedi, dopo viste e coppie |
| `weighted_endpoint_lpips_auc` | integrale nel solo intervallo degli intermedi |
| `weighted_endpoint_lpips_full_mean` | media su tutti i 50 frame |
| `weighted_endpoint_lpips_full_auc` | integrale sull'intero intervallo alpha [0,1] |

La distanza pesata agli endpoint e' zero per definizione: ciascun frame coincide
con la sorgente avente peso uno. Le distanze dall'altra sorgente vengono comunque
misurate. La media sui 50 frame vale quindi `48/50` della media sui soli intermedi;
la riduzione dovuta ai due zeri non e' un miglioramento del modello. Le AUC delle
due valutazioni coprono intervalli diversi e vanno confrontate fra approcci a
parita' di protocollo, non direttamente fra 5 e 48 intermedi.

I report PNG/PDF mostrano tutti i 50 punti, distanze originali e contributi pesati:

```text
weighted_50/cfg_grid_metrics.{csv,json}
weighted_50/weighted_endpoint_lpips/evolution.{png,pdf}
weighted_50/cfg_ss_.../weighted_endpoint_lpips/{curves.csv,pairs.csv,summary.csv,evolution.png,evolution.pdf}
weighted_50/cfg_ss_.../morphflow/pair_.../alpha_.../mesh_raw.pt
weighted_50/shared/{morphany3d,interp3d}/pair_.../alpha_.../mesh_raw.pt
```

Vengono salvati i raw completi di tutti i 48 intermedi per ogni coppia/modello/CFG.
Gli eventuali GLB sono controllati da `EXPORT_GLB`, anche per questa sequenza. I baseline
sono generati e valutati una sola volta per protocollo; sorgenti e cache dei render
di riferimento sono condivise anche fra i due protocolli. I riepiloghi densi
vengono copiati nell'archivio sotto `weighted_50/`. Per eseguire soltanto il
protocollo precedente impostare `WEIGHTED_DENSE=0` e scegliere un nuovo RUN_NAME.

## GLB, parallelismo e output

Con **`EXPORT_GLB=0` (default)** il job salta baking, export e audit GLB per tutti
i modelli, inclusi gli endpoint. Restano i file `mesh_raw.pt` con geometria e
Gaussian complete, i marker `mesh_raw.appearance.json`, i riferimenti
`references_raw/`, i piani, i metadati, i checkpoint e il codice congelati.
Questi raw consentono di esportare in seguito i GLB con il sottocomando `texture`,
nel container con GPU e dipendenze di baking, senza nuova inferenza dei modelli.
Le metriche leggono direttamente i raw Gaussian e non richiedono i GLB.
CSV, PNG/PDF, render delle viste e feature FID vengono conservati come prima.

Per la configurazione Ateneo corrente, su 50 coppie deterministiche e 15 CFG:

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
# Dopo commit/push sul repository, aggiornare il checkout dell'Ateneo.
git pull
unset CONFIG_FILE RUN_ROOT FINAL_ARCHIVE
export CHECKPOINT_PATH=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/checkpoints/ss_best_e17.pt
export SLAT_CHECKPOINT_PATH=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/checkpoints/slat_best_e9.pt
export RUN_NAME=v4_ss17_slat9_cfg15_n50_seed42_weighted50_raw
export N=50 SEED=42
export CFG_VALUES="1 2 3 5 7.5" SLAT_CFG_VALUES="1 1.5 2"
export WEIGHTED_DENSE=1 EXPORT_GLB=0
sbatch --export=ALL slurm/benchmark_threeway_interp3d_l40s.slurm
```

`EXPORT_GLB` e' registrato nel manifest immutabile: cambiare modalita' di export
richiede un nuovo RUN_NAME. Il sottocomando `texture` puo' invece esportare i raw
di una run esistente senza rilanciare il benchmark o cambiare quel manifest.
Per abilitare il baking durante una nuova valutazione scegliere `EXPORT_GLB=1`.

I baseline vengono generati e valutati una sola volta per protocollo. Ogni CFG
MorphFlow ha la propria generazione e le proprie metriche. Le fasi vengono ripartite per
`pair_id % worker_count`, con una GPU per processo; una barriera impedisce di
sovrapporre fasi pesanti. Un errore termina i worker rimanenti e blocca le fasi
successive. Ogni processo ha home e temporanei privati. TFSA usa RAM limitata
e overflow sul NVMe privato, selezionato esplicitamente con --tfsa-spill-root.
Il default richiede 64 GiB di NVMe libero per worker piu' 20 GiB di margine;
NODE_LOCAL_PARENT e NVME_MIN_GB sono configurabili.

Con `EXPORT_GLB=1`, mesh + aspetto Gaussian passano attraverso lo
stesso baking TRELLIS `to_glb(gaussian, mesh)`, texture 1024, simplify 0.95,
fill_holes abilitato per default. Gli export sono atomici; file troncati o non
validi vengono rigenerati. Un audit controlla ogni GLB intermedio e ogni endpoint
di ogni modello prima di dichiarare completata una cella.
Il baking produce i file da ispezionare; le metriche usano i render Gaussian
per evitare che il baking alteri il confronto percettivo.

Run host: `$BENCH_ROOT/threeway-runs/$RUN_NAME/`:

```text
cfg_grid_metrics.csv                         # tutte le configurazioni, entrambi i FID
cfg_grid_metrics.json
run_manifest.json                           # input, hash, formule e checkpoint fissati
checkpoint_metadata.json                    # epoca/loss/tipo dei pesi scelti
cfg_ss_2_slat_1.5/
  final_metrics.{csv,json}
  texture_status.json                       # solo con EXPORT_GLB=1
  morphflow/pair_.../
    src1.glb                                # solo con EXPORT_GLB=1
    alpha_.../pred_final.glb                 # solo con EXPORT_GLB=1
    alpha_.../mesh_raw.pt                    # geometria + aspetto autentico
    alpha_.../mesh_raw.appearance.json       # marker necessario per validare il raw
    src2.glb                                # solo con EXPORT_GLB=1
    sequence.json                           # scritto dal baking
  metrics/morphflow/pair_.../images/...
  morphany3d -> ../shared/morphany3d
  interp3d -> ../shared/interp3d
shared/
  morphany3d/pair_.../alpha_.../mesh_raw.pt
  interp3d/pair_.../alpha_.../mesh_raw.pt
  references_raw/pair_.../{src1_raw,src2_raw}.pt
  references_textured/pair_.../{src1,src2}.glb # solo con EXPORT_GLB=1
  metrics/{morphany3d,interp3d}/...
logs/job_<jobid>_<timestamp>/<phase>_g<shard>.{out,err}
progress.log
```

Con `EXPORT_GLB=1`, 500 coppie e nove CFG vengono salvati **27.500 GLB intermedi** per il
protocollo Interp3D (`500*5*(9+2)`), piu' **264.000 GLB intermedi** per quello
denso (`500*48*(9+2)`), oltre ai 1000 endpoint condivisi e ai relativi link.
La parte densa richiede 9,6 volte le generazioni/baking degli intermedi del solo
protocollo precedente. La ripresa conserva quanto completato anche in caso di
raggiungimento del limite temporale del job.
Con `EXPORT_GLB=0` non vengono creati GLB; restano gli stessi conteggi di raw e
render intermedi. Per 50 coppie e 15 CFG sono 45.050 raw intermedi e circa
2,88 milioni di PNG, oltre alle sorgenti condivise: il risparmio riguarda GLB
e tempo di baking, senza eliminare gli altri output.
RUN_ROOT conserva anche raw, PNG e feature: scegliere lo scratch con spazio
adeguato per l'intero output, non una directory temporanea del container.
Solo i riepiloghi compatti vengono copiati in
`$BENCH_ARCHIVE/final-results/$RUN_NAME/`; i GLB rimangono nella run scratch.

Rilanciare con lo stesso RUN_NAME/configurazione riprende coppie, GLB e metriche
completati. Un lock impedisce scritture concorrenti. Cambiare codice, metadati,
CFG o parametri richiede un nuovo nome; gli output precedenti sono preservati.
