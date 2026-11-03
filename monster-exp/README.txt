MORPHING DA DUE GLB O DUE SLAT
============================

Dal progetto: monster-exp/morph_assets.py e monster-exp/run_local.sh.
Launcher Leonardo: slurm/morph_assets_leonardo.slurm.
Non modificare i vecchi launcher morph_glb_pair: questo e un percorso separato.

Convenzione: alpha=1 seleziona src1; alpha=0 seleziona src2.
Anche 0 e 1 sono generati dal modello; non si sostituiscono con le sorgenti.

MONSTER / UBUNTU
---------------
Eseguire dalla cartella del progetto clonato:

conda activate trellis
bash monster-exp/run_local.sh --doctor --gpu

Sincronizzare anche trellis_asset_codec.py e il codice del modello con la
versione di questo progetto. Il pacchetto portable in outputs contiene i
file Python necessari e puo essere estratto nella radice del clone; non
contiene checkpoint, dataset, TRELLIS o librerie dell'ambiente.

Il launcher usa, se non gia impostati:
TRELLIS_REPO=/home/filippo/projects/TRELLIS
BLENDER_BIN=/home/filippo/projects/morph_old/.cache/blender/blender-3.0.1-linux-x64/blender
CUDA_HOME=/usr/local/cuda-12.4

Questi percorsi corrispondono all'ambiente verificato: Python 3.10,
PyTorch 2.4.0+cu121, RTX 4090, BF16, Blender 3.0.1. Il renderer CUDA e
l'import dell'esportatore texture sono stati provati sul computer locale.
Anche il render Cycles GPU su RTX 4090 e stato verificato con Blender 3.0.1.
--doctor --gpu verifica anche le dipendenze del modello e la rasterizzazione.
Il launcher usa l'ambiente attivo; MORPH_PYTHON puo indicare un altro Python.

GLB:
bash monster-exp/run_local.sh /path/A.glb /path/B.glb \
  --alphas 0 0.25 0.5 0.75 1 \
  --ss-checkpoint /path/ss.pt --slat-checkpoint /path/slat.pt \
  --blender /home/filippo/projects/morph_old/.cache/blender/blender-3.0.1-linux-x64/blender \
  --output-dir /path/output/glb_run

SLAT:
bash monster-exp/run_local.sh /path/asset_A /path/asset_B \
  --alphas 0,0.25,0.5,0.75,1 \
  --ss-checkpoint /path/ss.pt --slat-checkpoint /path/slat.pt \
  --output-dir /path/output/slat_run

Per scegliere una GPU:
CUDA_VISIBLE_DEVICES=1 bash monster-exp/run_local.sh ...
Ogni esecuzione usa una sola GPU; i modelli sono caricati in fasi separate.
Due run indipendenti possono usare rispettivamente le GPU 0 e 1.

LEONARDO
--------
sbatch slurm/morph_assets_leonardo.slurm /path/A.glb /path/B.glb \
  --alphas 0 0.25 0.5 0.75 1 \
  --ss-checkpoint /path/ss.pt --slat-checkpoint /path/slat.pt \
  --blender /leonardo_work/IscrC_MORPHFL/mbarezzi/tools/blender-3.0.1-linux-x64/blender \
  --output-dir /path/output/run

I due input possono invece essere directory/file SLat, con la stessa CLI.
Per controllare input, checkpoint e import prima del run, aggiungere --mode check.
Per il solo ambiente: bash slurm/morph_assets_leonardo.slurm --doctor.
Il test --doctor --gpu richiede una allocazione GPU.

Python Leonardo predefinito: envs/interp3d/bin/python, con i rasterizer.
Librerie TRELLIS: src/TRELLIS ufficiale; cache offline da env.sh.
Default Slurm: 1 GPU, 8 CPU, 64 GB RAM, 4 ore. Modificare --time al lancio
se il numero di alpha o la complessita delle mesh richiedono piu tempo.
Il launcher non invia email. Log in /leonardo_work/IscrC_MORPHFL/mbarezzi/logs/.
Variabili opzionali: MORPH_BASE, MF_ENV_FILE, MF_PROJECT_DIR, MF_TRELLIS_REPO,
MORPH_PYTHON. Gli argomenti --trellis-root e --blender prevalgono sui default.

BLENDER
-------
Precedenza: --blender, poi BLENDER_BIN, poi ricerca dei default noti.
Un percorso esplicito errato causa un errore: non viene sostituito in silenzio.
Per i GLB si verifica che sia eseguibile e che --version restituisca Blender 3.x.
Il codec riceve esattamente questo eseguibile; il percorso e la versione sono
salvati in run.json. Il comando effettivo di rendering e documentato nelle
cartelle inputs/src*/encoding/renders/ (render_config.json, wrapper, blender.log).
Per input SLat Blender non viene usato.

FORMATO SLAT
------------
Sono accettati:
1. Directory con slat_feats.pt e slat_coords.pt (nessun ss_latent.pt richiesto).
2. File .pt contenente {'feats': Tensor[N,8], 'coords': Tensor[N,3 o 4]}.
3. File .npz con gli array feats e coords.
4. Directory con latent.pt o slat.pt nel formato precedente.

Le feature devono essere in scala raw_decoder TRELLIS, non normalizzate.
Coordinate intere fra 0 e 63; in Nx4 il primo campo e il batch, sempre zero.
Il modello genera la struttura SS: non si riutilizza la struttura sorgente.
Gli input non vengono modificati. Per GLB, geometria e render vengono
normalizzati dalla pipeline ufficiale nel cubo unitario; i due oggetti hanno
quindi una scala relativa normalizzata, non la loro scala metrica originale.

QUALITA E PARAMETRI
------------------
Default: SS 25 step / CFG 5; SLat 25 step / CFG 1.5; seed SS 42,
seed SLat 10042; precisione auto (BF16 sulle GPU qui verificate).
--steps-ss, --steps-slat, --cfg-ss, --cfg-slat, --seed, --slat-seed.
Encoding: 150 viste, DINOv2, VAE posterior mean, luce uniform_texture.
--lighting official ripristina il rig di luci upstream; --dino-batch-size 1
riduce la memoria necessaria per l'encoding.
Texture: 2048 px, 100 viste a 1024 px e ottimizzazione ufficiale TRELLIS.
--texture-size 4096 aumenta la risoluzione e il consumo di memoria/tempo.
Default --simplify 0 e nessun riempimento di buchi: si preserva la superficie
decodificata. Opzioni --simplify (frazione da rimuovere) e --fill-holes.
Le texture provengono dal Gaussian decoder dello SLat generato dal modello;
non sono copiate dagli oggetti sorgente. La parametrizzazione UV di mesh
complesse puo richiedere diversi minuti per ogni alpha.

ASSI DEI GLB
------------
Il default --glb-up-axis y segue glTF: la verticale nel file e Y.
Alcuni esportatori salvano invece coordinate Z-up in un file .glb.
Se gli oggetti risultano sdraiati anche nelle viste quasi orizzontali,
provare --glb-up-axis z in una NUOVA cartella di output. La correzione
ruota la scena importata prima di normalizzare, renderizzare e creare
la mesh per la voxelizzazione; non ruota le sole immagini.
Sono disponibili --src1-up-axis e --src2-up-axis per sorgenti diverse.
Queste opzioni valgono solo per GLB, non per SLat gia codificati.
L'orientamento errato rispetto al training puo influire su geometria e
aspetto; correggerlo non garantisce da solo la fedelta dei colori.

Per separare la ricostruzione VAE dal morphing, si puo decodificare un
input gia salvato, senza rigenerare il modello:
python trellis_asset_codec.py decode outputs/morph_test/inputs/src1/slat.pt \
  --output-dir outputs/morph_test/reconstruction_src1 --no-texture
Il file risultante e reconstruction_src1/vertex.glb. Se la ricostruzione
ha colori corretti ma il morph no, il problema si manifesta nel modello;
se e gia sbagliata, occorre approfondire encoding e rappresentazione.

OUTPUT E RIPRESA
---------------
output/run.json: configurazione, checkpoint, Blender, ambiente e memoria GPU.
output/index.html: collegamenti ai risultati, apribile dal browser.
output/inputs/src1/slat.pt, src2/slat.pt: condizioni realmente usate.
output/morphs/alpha_*/ss.pt e slat.pt: risultati intermedi riutilizzabili.
output/morphs/alpha_*/prediction.glb: colori ai vertici, superficie Y-up.
output/morphs/alpha_*/prediction_textured.glb: risultato finale con texture.
output/morphs/alpha_*/mesh.pt e gaussian.pt: decodifica riutilizzabile.
output/summary.json e last_error.json (solo in caso di errore).

Per riprendere, rilanciare lo stesso comando con lo stesso output-dir.
Si possono aggiungere alpha: i risultati gia completi vengono riutilizzati.
Hash e marker di completamento evitano di usare file troncati; la cartella
e protetta da run simultanei. Cambi a input, checkpoint, parametri o codice
di modello richiedono una nuova cartella. Le fasi successive a un artefatto
rigenerato vengono invalidate e ricalcolate.
Lo stesso seed viene riutilizzato lungo la sequenza. Se SS cambia i voxel,
il rumore SLat e associato a coordinate diverse. Dispositivi e librerie CUDA
diversi possono produrre lievi differenze numeriche.

Test CPU dei formati e della ripresa:
python monster-exp/test_morph_assets.py

VERIFICHE ED ESEMPI SALVATI
--------------------------
Sette test CPU passati per formati, coordinate, cache e precedenza Blender.
Prove GPU completate: GLB ad alpha 0.5 (8 viste per test rapido; default 150),
SLat ad alpha 0/0.5/1, ripresa ed estensione della lista senza modificare
i risultati gia prodotti. Tutti i GLB textured hanno UV e PNG incorporata
a 2048x2048. Dettagli in VALIDATION.json.
Esempi Leonardo: outputs/monster_exp_smoke_20261103/{glb,slat}/index.html.

AGGIORNAMENTO ASSI
------------------
La versione con --glb-up-axis richiede di aggiornare anche il file
trellis_asset_codec.py nella radice del progetto. Il pacchetto con suffisso
_axes contiene entrambi. Usare una cartella nuova, ad esempio
outputs/morph_test_zup, per evitare gli SLat gia codificati con gli assi vecchi.
Correzione testata su due GLB equivalenti Y-up e Z-up: mesh normalizzata
e render coincidono entro gli errori numerici. Questo verifica la coerenza
della trasformazione, non garantisce la resa del modello su asset nuovi.
