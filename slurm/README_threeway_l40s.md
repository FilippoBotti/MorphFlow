# Confronto completo su otto L40S dell'Ateneo

`benchmark_threeway_interp3d_l40s.slurm` usa la configurazione dei launcher
funzionanti `train_morphflow_v3.slurm`, `generate_interp3d_v4.slurm` e del
precedente benchmark: account `g_vbd`, partizione/QoS `gpu_vbd`, nodo `wn49`,
otto `gpu:l40s_vbd`, 64 CPU, 700 GiB RAM, massimo 10 giorni; modulo
`singularity/3.8.7` e SIF `$HOME/containers/trellis-py240-cu118.sif`.
Mail Slurm BEGIN, END e FAIL a `marco.barezzi@unipr.it`.

## Preparazione e lancio sull'HPC Ateneo

Aggiornare il checkout MorphFlow sull'Ateneo con **tutti** i file modificati
di questo lavoro: launcher, helper, benchmark driver, evaluator e
`eval_perceptual_sequence.py`. I soli file `.slurm` non bastano.
I checkpoint devono essere gia' disponibili su quel filesystem.
Il pacchetto `threeway_l40s_20261018.tar.gz` contiene questi file e il codice
di inferenza MorphFlow; si estrae nella radice del checkout. Include un manifest
SHA256 dei file per verificare il trasferimento:

```bash
tar -xzf threeway_l40s_20261018.tar.gz -C "$HOME/trellis-singularity/work/MorphFlow"
```

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
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
fork corretto e contesto nvdiffrast prima della generazione. La verifica CUDA
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

## GLB, parallelismo e output

I baseline vengono generati, esportati e valutati una sola volta. Ogni CFG
MorphFlow ha la propria generazione/GLB/metriche. Le fasi vengono ripartite per
`pair_id % worker_count`, con una GPU per processo; una barriera impedisce di
sovrapporre fasi pesanti. Un errore termina i worker rimanenti e blocca le fasi
successive. Ogni processo ha home e temporanei privati. TFSA usa RAM limitata
e overflow sul NVMe privato, selezionato esplicitamente con --tfsa-spill-root.
Il default richiede 64 GiB di NVMe libero per worker piu' 20 GiB di margine;
NODE_LOCAL_PARENT e NVME_MIN_GB sono configurabili.

Gli export GLB sono obbligatori: mesh + aspetto Gaussian passano attraverso lo
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
  texture_status.json                       # audit di tutti e tre i modelli
  morphflow/pair_.../
    src1.glb
    alpha_.../pred_final.glb                 # ogni intermedio, con texture
    alpha_.../mesh_raw.pt                    # geometria + aspetto autentico
    src2.glb
    sequence.json
  metrics/morphflow/pair_.../images/...
  morphany3d -> ../shared/morphany3d
  interp3d -> ../shared/interp3d
shared/
  morphany3d/pair_.../alpha_.../pred_final.glb
  interp3d/pair_.../alpha_.../pred_final.glb
  references_textured/pair_.../{src1,src2}.glb
  metrics/{morphany3d,interp3d}/...
logs/job_<jobid>_<timestamp>/<phase>_g<shard>.{out,err}
progress.log
```

Con 500 coppie e nove CFG vengono salvati **27.500 GLB intermedi**
(`500*5*(9+2)`), oltre ai 1000 endpoint condivisi e ai relativi link.
RUN_ROOT conserva anche raw, PNG e feature: scegliere lo scratch con spazio
adeguato per l'intero output, non una directory temporanea del container.
Solo i riepiloghi compatti vengono copiati in
`$BENCH_ARCHIVE/final-results/$RUN_NAME/`; i GLB rimangono nella run scratch.

Rilanciare con lo stesso RUN_NAME/configurazione riprende coppie, GLB e metriche
completati. Un lock impedisce scritture concorrenti. Cambiare codice, metadati,
CFG o parametri richiede un nuovo nome; gli output precedenti sono preservati.
