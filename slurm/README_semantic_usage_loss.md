# Prova hubness loss con cap

La prova usa la formulazione della conversazione fornita: penalizzare solo i
token che ricevono troppa attenzione. Per ogni matrice di attenzione
`A[B, N_query, N_dest]`, l'usage è:

```text
u = N_dest * A.mean(dim=1)
L_hub = 0.5 * (mean(relu(u12 - cap)^2) + mean(relu(u21 - cap)^2))
L_total = L_base + usage_weight * L_hub       # con cycle disattivata
```

L'usage medio è 1 anche con sorgenti di lunghezza diversa. Con `cap=4` un usage
non uniforme sotto soglia non viene penalizzato. Il cap è una soglia morbida:
la loss contrasta gli hub, senza imporre un limite rigido all'usage.
Il calcolo usa le attention già disponibili, senza nuove matrici di matching.
La penalità usa riduzioni FP32, resta differenziabile verso Q/K ed esclude gli
style token se esclusi dal matcher. Non viene moltiplicata per alpha o campionata
con la probabilità della cycle.

## Parametri

| CLI Python | Variabile Slurm | Default |
| --- | --- | --- |
| `--semantic_usage_loss_weight` | `SEMANTIC_USAGE_LOSS_WEIGHT` | `0.0` (disattivata) |
| `--semantic_usage_cap` | `SEMANTIC_USAGE_CAP` | `4.0` |
| `--semantic_cycle_loss_weight` | `SEMANTIC_CYCLE_LOSS_WEIGHT` | Python `0.0`; script v3 `1.0`, come prima |
| `--semantic_cycle_loss_prob` | `SEMANTIC_CYCLE_LOSS_PROB` | Python `1.0`; script v3 `0.25`, come prima |

Il peso deve essere finito e non negativo; il cap finito e almeno 1. Un peso
positivo richiede `--use_semantic_token_matching 1`. La loss è supportata da SS,
SS residual e SLat condizionato su SLat. La variante SLat-DINO non usa questo
matcher e rifiuta un peso usage positivo.

## Lancio sul cluster di ateneo

Dal checkout aggiornato, per la prova richiesta (cycle OFF, cap 4, peso 0.01):

```bash
export RUN_NAME=v3_ss_semMatch_hub_w0p01_cap4
export SEMANTIC_CYCLE_LOSS_WEIGHT=0
export SEMANTIC_USAGE_LOSS_WEIGHT=0.01
export SEMANTIC_USAGE_CAP=4.0
export AUTO_RESUME=0
unset RESUME_FROM

sbatch --job-name="$RUN_NAME" slurm/train_morphflow_v3.slurm
```

Il launcher passa queste variabili all'interno di Singularity anche con
`--cleanenv`. Conserva il resto della configurazione v3: matching ON,
temperature 0.1, max_align 0.25, source swap ON. Usare un nome nuovo per ogni
esperimento; per un confronto controllato mantenere gli altri iperparametri e
l'inizializzazione uguali alla baseline. Questo comando avvia una nuova run,
senza riprendere automaticamente un checkpoint precedente.

Per la CLI diretta, aggiungere alla configurazione abituale:

```text
--use_semantic_token_matching 1
--semantic_cycle_loss_weight 0
--semantic_usage_loss_weight 0.01
--semantic_usage_cap 4.0
```

## Cosa osservare

I log periodici mostrano `base_mse`, `hub_raw`, `hub`, `usage12_max`,
`usage21_max`, `sem_H12` e `sem_H21`. Su TensorBoard le nuove serie sono:

```text
train/slat_semantic_usage_loss
train/slat_semantic_usage_loss_weighted
train/slat_semantic_usage_active
```

Le stesse metriche vengono salvate sotto `val/slat_...` in validazione.
`semantic_usage_active=1` indica che il termine è stato calcolato: la sua loss
può comunque essere zero se nessun token supera il cap. `usage*_max` mantiene
la convenzione esistente: media del massimo per elemento del batch, poi media
fra i processi; non è il massimo assoluto dell'intero dataset.

Confrontare il rapporto `semantic_usage_loss_weighted / base_mse`, l'andamento
di entrambe le usage massime, la base loss di validazione e le metriche
generative. La sola diminuzione degli hub non garantisce una generazione migliore.

La usage loss viene aggiunta a ogni forward di training e validazione quando il
peso è positivo. La validation loss totale (e quindi la selezione del best
checkpoint) la include: per confrontare pesi diversi osservare anche `base_mse`,
non soltanto la loss totale. La cycle resta disponibile come termine separato.

Lo schema delle metriche è fisso su tutti i rank, inclusi i valori zero, e la
riduzione distribuita usa un unico vettore ordinato. La modifica non aggiunge
parametri al modello: i vecchi state dict restano caricabili. Il loader di
valutazione ripristina peso/cap dai nuovi checkpoint e usa i default per quelli
precedenti. Il sampling `forward_flow` non calcola la nuova loss.

## Verifica locale

In un ambiente con PyTorch (CPU sufficiente):

```bash
python -m unittest discover -s tests -p 'test_semantic_usage_loss.py' -v
python -m unittest discover -s tests -p 'test_training_metrics.py' -v
```

I test coprono soglia, normalizzazione con numeri diversi di token, gradienti a
Q/K, esclusione degli style token, loss in validazione, compatibilità dei pesi e
training DDP a due processi Gloo con cycle attiva su rank differenti e hubness
abilitata/disabilitata. Le prestazioni e la qualità sul dataset richiedono la run
HPC; questi test non le misurano.
