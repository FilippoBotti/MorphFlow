# Supervisione SS con TRELLIS congelato

La prima opzione dell'allegato aggiunge un prior RFDS ai campioni dello student.
Il training mantiene la regressione FM sul teacher MorphAny3D e le loss semantiche
esistenti. A cadenza configurabile genera una SS con un rollout Euler dello
student, aggiunge rumore a un tempo indipendente e interroga il primo flow
TRELLIS originale congelato, con conditioning nullo nativo. Il residuo RFDS
aggiorna lo student con stop-gradient sul prior. Non vengono caricati DINO,
immagini o decoder per questa supervisione.

Il prior richiede `flow_target=ss`, `ss_flow_arch=standard` e
`trellis_model=image_large`. Il gradiente attraversa gli ultimi passi del
rollout; i passi precedenti forniscono il campione iniziale senza mantenere il
grafo. Con `TRELLIS_PRIOR_GRAD_STEPS=0` attraversa l'intero rollout.
Il checkpointing riduce la memoria delle attivazioni ricalcolandole nel backward.
La copia congelata di TRELLIS occupa comunque memoria su ciascuna GPU.

## Segnale RFDS e rollout

Il rollout parte da rumore gaussiano indipendente dal batch FM, integra da
`t=1` a `t=0` su una griglia lineare (8 passi per default), usa `CFG=1` e
disabilita dropout e CFG dropout durante il campionamento. Lo student resta
differenziabile nei passi selezionati. Il latente `z` è quindi un campione
generato, non una stima one-step ricavata dal target teacher.

Con rumore `eps` e tempo `tau` uniformemente estratto nell'intervallo configurato,
entrambi indipendenti dal rollout, viene applicato:

```text
sigma_tau = sigma_min + (1 - sigma_min) * tau
x_tau = (1 - tau) * z + sigma_tau * eps
g = v_prior(x_tau, tau) - ((1 - sigma_min) * eps - z)
w(tau) = 1
L_prior = 0.5 * mean((z - stopgrad(z - g)) ** 2)
```

`sigma_min` è quello dello student. Il flow nativo riceve il tempo `1000 * tau`
e token nulli a zero, distinti dalla `null_cond` appresa dallo student.
Il prior, l'aggiunta di rumore e il residuo non sono differenziati. Il gradiente
della loss rispetto a `z` è `g / z.numel()`, prima del peso RFDS;
non si aggiunge un fattore `(1 - tau)`. Se il clipping RMS è attivo, limita
`g` per campione prima di costruire il target con stop-gradient.

Il comportamento nativo segue il
[flow SS originale Microsoft](https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_flow.py)
e il conditioning nullo della
[pipeline image-to-3D TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py).

## Cluster di ateneo

`train_morphflow_v3.slurm` attiva il prior con peso a regime configurato a `0.01`.
Mantiene `teacher_mid_weight=0.60`, matching semantico, hubness, learning rate e
gli altri parametri della baseline SS. Il nome predefinito della run è distinto
dalla baseline precedente.

```bash
RUN_NAME=v3_ss_prior_001 sbatch slurm/train_morphflow_v3.slurm

# Ablazione: nessun caricamento della copia prior, nessun rollout aggiuntivo.
TRELLIS_PRIOR_WEIGHT=0 RUN_NAME=v3_ss_prior_off \
    sbatch slurm/train_morphflow_v3.slurm

# Ripresa della stessa run; oppure specificare RESUME_FROM=/percorso/checkpoint.pt.
AUTO_RESUME=1 RUN_NAME=v3_ss_prior_001 sbatch slurm/train_morphflow_v3.slurm
```

Il launcher scarica o verifica una volta i due asset del prior prima di avviare
Accelerate. La cache condivisa resta `$HOME/trellis-singularity/hf-home`
(`HF_HOME=/opt/trellis/hf-home` nel container). Con `HF_HUB_OFFLINE=1` i file
devono già essere presenti nella cache; non vengono tentati download.
Anche una ripresa dello student richiede i pesi originali del prior.

## Leonardo

Il launcher Leonardo mantiene per default il training **SLat senza prior**.
Per il primo flow SS selezionare `FLOW_TARGET=ss`: attiva anche il prior a `0.01`
e usa una directory di run SS distinta. Gli altri iperparametri rimangono quelli
del launcher Leonardo; questa configurazione non replica automaticamente la
baseline del cluster di ateneo.

```bash
# Dal login node: download dei pesi SS e della configurazione nativa del prior.
FLOW_TARGET=ss bash slurm/train_morphflow_v3_leonardo.sbatch --prepare

# Stessa verifica usando solo la cache locale.
FLOW_TARGET=ss bash slurm/train_morphflow_v3_leonardo.sbatch --check

FLOW_TARGET=ss RUN_NAME=v3_ss_prior_001_leonardo \
    sbatch slurm/train_morphflow_v3_leonardo.sbatch

FLOW_TARGET=ss TRELLIS_PRIOR_WEIGHT=0 RUN_NAME=v3_ss_prior_off_leonardo \
    sbatch slurm/train_morphflow_v3_leonardo.sbatch

FLOW_TARGET=ss AUTO_RESUME=1 RUN_NAME=v3_ss_prior_001_leonardo \
    sbatch slurm/train_morphflow_v3_leonardo.sbatch
```

La preparazione scarica solo gli asset necessari da
`microsoft/TRELLIS-image-large`: `ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors`
e, quando il prior è abilitato, `ckpts/ss_flow_img_dit_L_16l8_fp16.json`.
I job restano offline. La cache del prior viene verificata anche con
`RESUME_FROM`; con peso zero e ripresa non servono i pesi iniziali TRELLIS.
Usare sempre checkpoint dello stesso flow e della stessa architettura.
Per allocazione, NCCL e percorsi vedere [README_leonardo.md](README_leonardo.md).

## Parametri comuni

Le variabili valgono per entrambi i launcher e corrispondono alle opzioni CLI
omonime in minuscolo, per esempio `TRELLIS_PRIOR_WEIGHT` →
`--trellis_prior_weight`.

| Variabile | Default SS | Significato |
| --- | ---: | --- |
| `TRELLIS_PRIOR_WEIGHT` | `0.01` | Peso RFDS a regime; `0` disabilita il prior |
| `TRELLIS_PRIOR_EVERY` | `4` | Un rollout ogni N microbatch di training, incluso il primo |
| `TRELLIS_PRIOR_WARMUP_STEPS` | `1000` | Rampa lineare del peso in passi optimizer; `0` usa subito il peso pieno |
| `TRELLIS_PRIOR_ROLLOUT_STEPS` | `8` | Passi Euler del rollout student |
| `TRELLIS_PRIOR_GRAD_STEPS` | `2` | Ultimi passi con gradiente; `0` mantiene tutto il rollout |
| `TRELLIS_PRIOR_MAX_ITEMS` | `1` | Campioni per GPU nel rollout; `0` usa tutto il microbatch |
| `TRELLIS_PRIOR_CHECKPOINT` | `1` | Checkpointing delle attivazioni nei passi con gradiente |
| `TRELLIS_PRIOR_T_MIN` | `0.05` | Estremo inferiore del tempo RFDS indipendente |
| `TRELLIS_PRIOR_T_MAX` | `0.95` | Estremo superiore del tempo RFDS indipendente |
| `TRELLIS_PRIOR_GRAD_CLIP` | `0.0` | Limite RMS per campione del residuo RFDS; `0` disabilita il clipping |

La rampa inizia dal primo aggiornamento: il peso è
`weight * min((global_step + 1) / warmup_steps, 1)`.
La frequenza non viene compensata moltiplicando la loss per N: cambiare
`TRELLIS_PRIOR_EVERY` modifica anche il contributo medio del prior.
La validation FM resta confrontabile con quella precedente e non include
rollout RFDS. Il prior non aggiunge pesi al checkpoint dello student e non serve
in inferenza; i checkpoint SS precedenti restano utilizzabili con la stessa
architettura. La ripresa mantiene il contatore dei passi del training.

## Metriche

TensorBoard registra le seguenti serie, con prefisso `train/trellis_prior_`:

| Suffisso | Significato |
| --- | --- |
| `active` | `1` negli aggiornamenti con rollout RFDS; `0` negli altri |
| `weight` | Peso effettivo dopo il warmup, oppure `0` quando inattivo |
| `loss` | Valore della loss surrogata RFDS senza peso |
| `loss_weighted` | Contributo della loss RFDS al totale |
| `residual_rms` | RMS del residuo prima del clipping |
| `gradient_rms` | RMS del residuo iniettato dopo il clipping, prima di peso e divisione per `z.numel()` |
| `t_mean` | Tempo medio del noising indipendente |
| `velocity_rms` | RMS della velocità predetta dal prior |
| `sample_rms` | RMS del latente generato dallo student |

Le metriche sono zero negli aggiornamenti senza prior; usare `active` per
leggere le serie o calcolare medie sui soli rollout. La loss surrogata serve
a iniettare il gradiente RFDS: il suo valore non misura direttamente la qualità
della forma. La selezione del checkpoint `best` rimane basata sulla validation
rispetto al teacher, senza RFDS.

Il valore `0.01` è un punto di partenza da confrontare con l'ablazione a zero.
Valutare qualità delle SS decodificate e continuità delle sequenze, oltre alla
validation sul teacher. RFDS favorisce plausibilità, ma non introduce una nuova
supervisione semantica esterna.
