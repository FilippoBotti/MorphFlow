# Ablation SS v4 su Ateneo: 250000 aggiornamenti

I quattro esperimenti partono dai pesi originali TRELLIS, non dal checkpoint
MorphFlow già addestrato. Allenano soltanto SS; LoRA cross-attention e condition
encoder sono configurati come `train_morphflow_v4_ss.slurm`. Le due cross-attention
sono presenti in tutti gli esperimenti.

| Script | Gate | Source swap | Prior | Esempi/epoca | Epoche |
| --- | --- | --- | --- | --- | --- |
| `train_ss_ablation_baseline_ateneo.slurm` | alpha fisso | no | no | 50000 | 10 |
| `train_ss_ablation_gate_only_ateneo.slurm` | token appreso | no | no | 50000 | 10 |
| `train_ss_ablation_swap_only_ateneo.slurm` | alpha fisso | entrambe le direzioni | no | 100000 | 5 |
| `train_ss_ablation_prior_only_ateneo.slurm` | alpha fisso | no | v4, peso 0.1 | 50000 | 10 |

Senza gate appreso ogni blocco calcola `alpha*h1 + (1-alpha)*h2`, dove `h1` e `h2`
sono le uscite delle due cross-attention. Non interpola i token prima
softmax e non usa `separate_cond=0`: quest'ultima modalità usa la fusione MLP
storica e sarebbe un'altra architettura. `separate_cond_gate=alpha` non crea
parametri `alpha_gate`; `token` conserva struttura e nomi dei checkpoint esistenti.
L'evaluator ricostruisce la modalità dagli argomenti salvati nel checkpoint.

Il riferimento e' il training SS originale: **250000 aggiornamenti optimizer**,
registrati alla fine di 20 epoche su 8 GPU con batch 1 per GPU e 100000 esempi
per epoca (50000 originali + 50000 invertiti).

Con i default di 2 GPU e batch 1 per GPU, baseline, gate_only e prior_only usano
**10 epoche sui 50000 esempi originali**; swap_only usa **5 epoche sui 100000 esempi**.
Il numero di epoche si calcola automaticamente dal batch globale per raggiungere
esattamente `TARGET_TRAIN_STEPS` (default 250000). Quando non corrisponde a
epoche intere, il launcher imposta `--max_train_steps` e interrompe l'ultima
epoca al budget esatto, eseguendo poi validazione e salvataggio dei checkpoint.
Con **3 GPU x batch 1**, sono al massimo 15 epoche senza swap oppure 8 con swap;
l'ultima e' parziale. Il totale resta 250000, non viene arrotondato.
Tutti usano seed 42, learning rate 1e-4, warmup 10000,
cosine, BF16, LoRA rank 32/alpha 64/dropout 0.1 su `to_q,to_kv`, matching semantico,
endpoint loss e symmetry loss disabilitati. Il prior mantiene rollout 25,
4 passi differenziabili, cadenza 4 e warmup 10000. Non viene caricato nei tre
esperimenti senza prior; DINO e immagini sorgenti servono soltanto a prior_only.

## Risorse e lancio

I quattro file richiedono per default `wn46`, **2 A100**, **28 CPU**, **400G di RAM**,
1 nodo, 1 task Slurm, fino a 10 giorni e mail BEGIN/END/FAIL a
`marco.barezzi@unipr.it`. `gpu:a100_vbd:2` evita di mescolare A100 e RTX6000 sul nodo
eterogeneo. Accelerate avvia un processo per GPU effettivamente assegnata.
`CUDA_VISIBLE_DEVICES` viene trasferita esplicitamente nel container cleanenv.

Sono riutilizzati container `~/containers/trellis-py240-cu118.sif`, env
`~/trellis-singularity/env` e cache condivise del benchmark già configurati su
Ateneo. Il launcher comune è `train_ss_ablation_ateneo.sh`.

Dopo aver sincronizzato queste modifiche sul repository di Ateneo:

```bash
cd "$HOME/trellis-singularity/work/MorphFlow"
unset RUN_NAME RESUME_FROM
export TRAIN_BS=1 SEED=42 AUTO_RESUME=1 TARGET_TRAIN_STEPS=250000

sbatch slurm/train_ss_ablation_baseline_ateneo.slurm
sbatch slurm/train_ss_ablation_gate_only_ateneo.slurm
sbatch slurm/train_ss_ablation_swap_only_ateneo.slurm
sbatch slurm/train_ss_ablation_prior_only_ateneo.slurm
```

Il batch globale è `TRAIN_BS * numero_GPU_assegnate`, senza accumulo.
Con i default è **1 * 2 = 2**: 25000 step/epoca senza swap, 50000 con swap.
Per un batch globale 4 sullo stesso nodo:

```bash
TRAIN_BS=2 sbatch slurm/train_ss_ablation_baseline_ateneo.slurm
```

Per un altro nodo e numero di GPU, sovrascrivere le direttive con sbatch
(o modificare gli header dei quattro file):

```bash
TRAIN_BS=1 sbatch --nodelist=wn49 --gres=gpu:l40s_vbd:8 \
  --cpus-per-task=28 --mem=400G \
  slurm/train_ss_ablation_baseline_ateneo.slurm
```

Con batch globale 4, le epoche diventano 20 senza swap e 10 con swap.
L'esempio su wn49 ha batch globale 8: 40 epoche senza swap, 20 con swap.
In tutti questi casi il totale resta 250000 step. Il numero dei processi viene rilevato
dall'allocazione, non da `NGPUS`; `NGPUS` serve solo alla simulazione `--dry-run`.
Un batch più grande va scelto in base alla memoria GPU disponibile, soprattutto
con il prior. Mantenere lo stesso batch globale nei quattro esperimenti.

Per verificare le opzioni senza prenotare GPU:

```bash
bash slurm/train_ss_ablation_baseline_ateneo.slurm --dry-run
NGPUS=8 TRAIN_BS=2 bash slurm/train_ss_ablation_prior_only_ateneo.slurm --dry-run
```

`PROJECT_DIR`, `DATA_ROOT_HOST`, `ARCHIVE_HOST`, `TRELLIS_BASE`, `SIF`,
`BENCH_HF_HOME`, `BENCH_TORCH_HOME`, `FLUX_ROOT_HOST`, `TRAIN_BS`, `SEED`,
`NUM_WORKERS`, `TARGET_TRAIN_STEPS` e `RUN_NAME` sono configurabili via ambiente. Se occorre mantenere
intatto il checkout di un benchmark ancora da riprendere, sincronizzare una copia
separata e impostare per esempio `PROJECT_DIR=$HOME/trellis-singularity/work/MorphFlow-ablations`.

## Budget e source swap

Baseline, gate_only e prior_only leggono i 50000 esempi originali una volta
per epoca, nella direzione del metadata. Solo swap_only usa la funzionalità
esistente `--augment_source_swap 1`: aggiunge virtualmente ogni coppia con sorgenti
scambiate, alpha sostituito da `1-alpha` e target invariato. Tutti i campi sorgente
vengono scambiati, comprese immagini e latenti SS. Non si duplicano file su disco.

Lo swap è deterministico: una copia originale e una invertita per ogni esempio.
Lo shuffle cambia l'ordine, non la composizione del dataset. Gli altri tre
esperimenti usano `--augment_source_swap 0`.

Il budget di tutti e quattro e' **250000 step optimizer**. Con batch globale 2:
`10 * (50000 / 2) = 5 * (100000 / 2) = 250000`. Gli esempi visti sono 500000,
mentre il vecchio training con batch globale 8 ne vedeva 2000000: si pareggiano
gli aggiornamenti, mantenendo il batch globale scelto per le nuove risorse.

Il calcolo usa `steps_per_epoch = ceil(train_samples / global_batch)` e rifiuta
combinazioni che non consentono di raggiungere esattamente il target con epoche
intere. Per esempio 3 GPU con batch 1 non sono compatibili con 250000 step.
Il training verifica anche la lunghezza effettiva del dataset e del DataLoader
distribuito: filtraggio di dati mancanti o una diversa distribuzione dei batch
producono un errore, invece di cambiare il budget senza segnalarlo.

Per un nuovo controllo completo con questo budget e due GPU, usare 5 epoche
se include il dataset raddoppiato. Il cosine scheduler usa il totale di epoche
calcolato; seed, generatori DataLoader e gli altri argomenti v4 sono espliciti.

## Output e ripresa

Default:

```text
/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/ablations_ss/
  v4_ss_ablation_<esperimento>_steps250000_e<10_o_5>_s42_g2_bs1/
    ablation_config.json
    checkpoints/morphflow_best.pt
    checkpoints/morphflow_last.pt
    checkpoints/morphflow_epoch_*.pt
    logs/train.log
```

I log Slurm `ss-ablation-<nomejob>-<jobid>.out/.err` sono nella directory da cui
si esegue sbatch. Per seguire una run:

```bash
tail -n 30 -F /hpc/archive/G_VBD/marco.barezzi/morphflow_runs/ablations_ss/v4_ss_ablation_baseline_steps250000_e10_s42_g2_bs1/logs/train.log
```

Risottomettere lo stesso script riprende automaticamente `morphflow_last.pt`,
con modello, optimizer, scheduler ed epoca. Il totale rimane quello calcolato
(10 epoche, oppure 5 per swap_only, con i default), senza aggiungerne alla ripresa.
Il nome delle nuove run include il target di step: i checkpoint del precedente
budget 20/10 epoche non vengono ripresi automaticamente. Un'epoca
interrotta ricomincia dall'ultimo checkpoint di fine epoca; lo stato RNG non è
salvato nei checkpoint storici. La configurazione viene confrontata con
`ablation_config.json` e un lock impedisce scritture simultanee sulla stessa run.
Per cambiare esperimento/batch/seed usare un nuovo RUN_NAME; il default li include.
`AUTO_RESUME=0` consente una nuova run ma rifiuta di sovrascrivere un last esistente.

## Altri controlli prioritari

1. **Modello completo e rimozione di un componente alla volta**: i quattro
   esperimenti richiesti misurano effetti singoli sulla baseline, ma non le
   interazioni. Aggiungere gate+swap+prior con lo stesso batch e budget di 250000 aggiornamenti; poi
   completo senza gate, senza swap, senza prior.
2. **Almeno tre seed**, almeno per baseline e modello completo, e differenze
   appaiate sulle stesse coppie test. Una sola seed non misura la variabilità
   del training.
3. **Stesso SLat congelato in tutte le valutazioni SS**: stessi checkpoint SLat,
   CFG, coppie, seed, viste e alpha. Valutare sia occupazione/geometry SS sia
   output finale, perché SLat può attenuare o amplificare le differenze.
4. **Consistenza allo scambio**: confrontare la generazione `(src1,src2,alpha)`
   con `(src2,src1,1-alpha)` a rumore fissato; è il controllo specifico dello swap.
5. Se una rivendicazione riguarda l'encoder, confrontare sparse-conv con block
   pooling lasciando invariati gli altri componenti. Se riguarda CFG, valutare
   separatamente il dropout condizionale: negli script v4 attuali è zero.
