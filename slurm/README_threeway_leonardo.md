# Confronto v4 su Leonardo

`benchmark_threeway_textured_leonardo.slurm` confronta MorphFlow v4,
MorphAny3D e Interp3D su **un nodo Booster e una GPU A100**, con ambienti
Python separati. Le notifiche Slurm BEGIN, END e FAIL arrivano a
`marco.barezzi@unipr.it`. Il limite di ogni job e' 24 ore.

La configurazione pronta in `submit_threeway_v4_leonardo.sh` usa 30 coppie
uniche del test v4, seed 42, checkpoint SS best epoca 13 e SLat best epoca 6,
copiati in `checkpoints/threeway_v4_20261017/`. `snapshots.json` contiene loss,
step e verifica della copia. Il training puo' continuare indipendentemente.

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
bash slurm/submit_threeway_v4_leonardo.sh --check
bash slurm/submit_threeway_v4_leonardo.sh
```

La griglia MorphFlow e' SS `2 3 5` per SLat `1.5 2 2.5` (nove celle).
I due baseline mantengono la configurazione CFG degli autori e sono
generati e valutati una volta; le loro metriche e gli endpoint canonici
sono condivisi fra tutte le celle. Le immagini sorgenti, le coppie,
gli alpha, le viste e gli intermedi scelti per il FID coincidono.

## Protocollo

Il riferimento e' [Interp3D, sezioni 5 e A.2.2](https://arxiv.org/html/2601.14103v1#A2.SS2):
sette stati totali (cinque intermedi e due endpoint), coefficienti
Beta(5,5), 25 passi SS e 25 passi SLat, VGG LPIPS e PPL su 64 viste,
FID su due intermedi scelti casualmente con seed fisso e 1000 viste per
stato. Il PPL principale normalizza la somma delle distanze adiacenti
per la distanza fra gli endpoint. PDV e altre diagnostiche sono aggiuntive.

L'eval usa direttamente le Gaussian decodificate dai rispettivi SLat,
con renderer TRELLIS comune, camere sferiche Hammersley, raggio 2,
FOV 40 gradi, risoluzione 512 e sfondo bianco. La geometria grezza salva
anche l'aspetto Gaussian autentico; i vecchi output contenenti solo mesh
vengono rigenerati. `BAKE_TEXTURES=1` abilita anche i GLB con texture usando
la stessa routine TRELLIS e le Gaussian come sorgente dell'aspetto.

Questo e' un **adattamento sul test v4**, non la riproduzione numerica
della tabella del paper (che usa Interp3DData). Il codice pubblico corrente
Interp3D usa Beta(6,6); qui la schedule condivisa segue Beta(5,5) del paper.
Gli endpoint di riferimento sono gli asset canonici del dataset v4.
Il FID usa Inception pool3 2048D TensorFlow-compatible di `pytorch-fid`;
il paper non pubblica l'intera implementazione delle metriche. Il FID
principale e' la media dei FID per traiettoria; il FID ottenuto aggregando
le feature di tutte le coppie e' una diagnostica separata.
Per ogni traiettoria, il riferimento concatena le 1000 feature di ciascun
endpoint e la distribuzione generata concatena le 1000 feature di ciascuno
dei due intermedi selezionati. Il testo del paper e' ambiguo sulla media
fra intermedi e non fornisce il codice FID: qui la definizione e' esplicita
e le feature vengono conservate per ulteriori verifiche senza altri render.

## Ambienti e verifica

La preparazione riproducibile dei baseline e' in
`setup_baseline_envs_leonardo.sh` e `.sbatch`; gli ambienti sono
`envs/morphany3d` e `envs/interp3d`, i repository `src/MorphAny3D` e
`src/Interp3D`. `envs/morphflow` e' l'ambiente principale TRELLIS/MorphFlow.
I job usano le cache offline gia' preparate sul login node.

Prima del confronto completo, eseguire una prova GPU su una coppia:

```bash
export CHECKPOINT_PATH=/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/threeway_v4_20261017/ss_best_epoch_13.pt
export SLAT_CHECKPOINT_PATH=/leonardo_work/IscrC_MORPHFL/mbarezzi/checkpoints/threeway_v4_20261017/slat_best_epoch_6.pt
N=1 K=5 CFG_VALUES=2 SLAT_CFG_VALUES=1.5 NUM_VIEWS=8 FID_NUM_VIEWS=32 \
FID_MIN_SAMPLES=2 RENDER_RESOLUTION=128 \
RUN_ROOT=/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/threeway_benchmark/threeway_smoke_20261017 \
sbatch --job-name=threeway_v4_smoke --time=01:00:00 \
    slurm/benchmark_threeway_textured_leonardo.slurm
```

Questa prova ridotta verifica i tre modelli e il percorso completo delle
metriche; i suoi numeri non appartengono al benchmark completo.

## Output e ripresa

La run pronta scrive in
`/leonardo_work/IscrC_MORPHFL/mbarezzi/outputs/threeway_benchmark/threeway_v4_ss13_slat6_n30_beta5_cfg235_slat15225_20261017/`.
Il riepilogo e' `cfg_grid_metrics.{json,csv}`; ogni directory
`cfg_ss_<x>_slat_<y>` contiene `final_metrics.{json,csv}` e risultati
per coppia. `shared/` contiene baseline e cache dei riferimenti.
I log Slurm sono in `$WORK/$USER/logs/threeway_v4_cfggrid_<jobid>.{out,err}`.

Rilanciare la stessa configurazione e `RUN_ROOT` riprende i risultati
completati. Il manifest controlla pesi, dati, codice e configurazione;
per cambiare uno di questi input scegliere una nuova `RUN_ROOT`.
Un lock impedisce due scritture contemporanee nella stessa run.
I file temporanei usano WORK (`<RUN_ROOT>/runtime-tmp`): `/tmp` sul nodo A100
verificato e' un tmpfs da 10 GiB, insufficiente per l'intera cache TFSA
(il login node ha un `/tmp` distinto da 2 GiB).
`THREEWAY_TMP_ROOT` permette una directory alternativa con spazio sufficiente.
Il launcher originale `benchmark_threeway_textured.slurm` resta
l'adattamento precedente per Parma, con alpha uniformi e render GLB.
