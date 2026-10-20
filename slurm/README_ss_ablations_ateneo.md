# Ablation SS v4 su Ateneo: 20 epoche sui 50k originali

I quattro esperimenti partono dai pesi originali TRELLIS, non dal checkpoint
MorphFlow già addestrato. Allenano soltanto SS; LoRA cross-attention e condition
encoder sono configurati come `train_morphflow_v4_ss.slurm`. Le due cross-attention
sono presenti in tutti gli esperimenti.

| Script | Gate | Source swap | Prior | Esempi/epoca | Epoche |
| --- | --- | --- | --- | --- | --- |
| `train_ss_ablation_baseline_ateneo.slurm` | alpha fisso | no | no | 50000 | 20 |
| `train_ss_ablation_gate_only_ateneo.slurm` | token appreso | no | no | 50000 | 20 |
| `train_ss_ablation_swap_only_ateneo.slurm` | alpha fisso | entrambe le direzioni | no | 100000 | 10 |
| `train_ss_ablation_prior_only_ateneo.slurm` | alpha fisso | no | v4, peso 0.1 | 50000 | 20 |

Senza gate appreso ogni blocco calcola `alpha*h1 + (1-alpha)*h2`, dove `h1` e `h2`
sono le uscite delle due cross-attention. Non interpola i token prima
softmax e non usa `separate_cond=0`: quest'ultima modalità usa la fusione MLP
storica e sarebbe un'altra architettura. `separate_cond_gate=alpha` non crea
parametri `alpha_gate`; `token` conserva struttura e nomi dei checkpoint esistenti.
L'evaluator ricostruisce la modalità dagli argomenti salvati nel checkpoint.

Baseline, gate_only e prior_only usano **20 epoche sui 50000 esempi originali**.
Solo swap_only usa **10 epoche sui 100000 esempi**, includendo entrambe le direzioni.
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

sbatch slurm/train_ss_ablation_baseline_ateneo.slurm
sbatch slurm/train_ss_ablation_gate_only_ateneo.slurm
sbatch slurm/train_ss_ablation_swap_only_ateneo.slurm
sbatch slurm/train_ss_ablation_prior_only_ateneo.slurm
```

Il batch globale è `TRAIN_BS * numero_GPU_assegnate`, senza accumulo.
Con i default è **1 * 2 = 2**. Per un batch globale 4 sullo stesso nodo:

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

Questo esempio ha batch globale 8. Il numero dei processi viene rilevato
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
`NUM_WORKERS` e `RUN_NAME` sono configurabili via ambiente. Se occorre mantenere
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

Il budget di tutti e quattro è **1000000 esempi visti**:
`50000 * 20 = 100000 * 10`. Con il batch globale predefinito 2 corrisponde a
500000 aggiornamenti per ogni training. La loss schedule copre il totale
configurato (20 oppure 10 epoche); seed e generatori DataLoader sono espliciti.

Per confrontare anche il modello completo con questo budget, usare 10 epoche
se include il dataset raddoppiato e lo stesso batch globale. Il vecchio completo
su 8 GPU usava batch globale 8 e 20 epoche con swap: il suo budget è diverso.

## Output e ripresa

Default:

```text
/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/ablations_ss/
  v4_ss_ablation_<esperimento>_e<20_o_10>_s42_g2_bs1/
    ablation_config.json
    checkpoints/morphflow_best.pt
    checkpoints/morphflow_last.pt
    checkpoints/morphflow_epoch_*.pt
    logs/train.log
```

I log Slurm `ss-ablation-<nomejob>-<jobid>.out/.err` sono nella directory da cui
si esegue sbatch. Per seguire una run:

```bash
tail -n 30 -F /hpc/archive/G_VBD/marco.barezzi/morphflow_runs/ablations_ss/v4_ss_ablation_baseline_e20_s42_g2_bs1/logs/train.log
```

Risottomettere lo stesso script riprende automaticamente `morphflow_last.pt`,
con modello, optimizer, scheduler ed epoca. Il totale rimane 20 epoche
(10 per swap_only), senza aggiungerne altre alla ripresa. Un'epoca
interrotta ricomincia dall'ultimo checkpoint di fine epoca; lo stato RNG non è
salvato nei checkpoint storici. La configurazione viene confrontata con
`ablation_config.json` e un lock impedisce scritture simultanee sulla stessa run.
Per cambiare esperimento/batch/seed usare un nuovo RUN_NAME; il default li include.
`AUTO_RESUME=0` consente una nuova run ma rifiuta di sovrascrivere un last esistente.

## Altri controlli prioritari

1. **Modello completo e rimozione di un componente alla volta**: i quattro
   esperimenti richiesti misurano effetti singoli sulla baseline, ma non le
   interazioni. Aggiungere gate+swap+prior con lo stesso batch e budget di un milione di esempi; poi
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
