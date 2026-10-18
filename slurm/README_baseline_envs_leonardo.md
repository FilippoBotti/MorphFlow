# MorphAny3D e Interp3D su Leonardo

Radice: `/leonardo_work/IscrC_MORPHFL/mbarezzi`.

Gli ambienti `envs/morphany3d` e `envs/interp3d` sono copie indipendenti
di `envs/morphflow`, con Python 3.10.8, PyTorch 2.4.0+cu121,
torchvision 0.19.0+cu121, xformers 0.0.27.post2, spconv-cu120 2.3.6 e
Kaolin 0.17.0. Non ci sono hard link ai file dell'ambiente originale.
Gli script di attivazione e gli shebang sono adattati al rispettivo percorso.
La copia omette cache bytecode e suite di test opzionali delle dipendenze.

Le sorgenti ufficiali sono in `src/MorphAny3D` (commit
`37012252528175a2eec00cd709c2e9bbdbb9313b`) e `src/Interp3D` (commit
`bb4ae3cc0591457dbd08f3eec1b1ff49b918c0ce`). Ogni processo importa il
proprio fork `trellis` tramite `PYTHONPATH`. Per entrambi la sola modifica
locale iniziale riguarda DINOv2: con `HF_HUB_OFFLINE=1` il codice carica
la sorgente Torch Hub locale, senza interrogare GitHub.

L'installazione aggiunge PyTorch3D 0.7.8 (pacchetto binario ufficiale
`pytorch3d-0.7.8-py310_cu121_pyt240.tar.bz2` dal canale Anaconda `pytorch3d`,
con verifica ABI CPU e CUDA), nvdiffrast 0.3.3,
diffoctreerast, il rasterizzatore gaussiano di mip-splatting e pytorch-fid 0.3.0.
I sorgenti sono in `src/baseline_extensions`; le wheel compilate per
CUDA 12.1 / A100 sono salvate in `cache/baseline_wheels` e riutilizzate
nei due ambienti. L'attenzione usa xformers, supportato da entrambi i fork.

Revisioni delle estensioni: nvdiffrast
`729261dc64c4241ea36efda84fbf532cc8b425b8`, diffoctreerast
`b09c20b84ec3aace4729e6e18a613112320eca3a`, mip-splatting
`dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`.
SHA256 del pacchetto binario PyTorch3D:
`50f8677ca7cc5b59d89cbca66f443a52eadc5f86bf32516c6b5d92693830814f`.

```bash
cd /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow
# Attivazione del fork scelto
source slurm/activate_baseline_leonardo.sh morphany3d
# oppure
source slurm/activate_baseline_leonardo.sh interp3d

# Verifica CPU di entrambi gli ambienti
bash slurm/setup_baseline_envs_leonardo.sh --check
# Compilazione delle estensioni e verifica su una A100
sbatch slurm/setup_baseline_envs_leonardo.sbatch
```

Il job di installazione usa un nodo, una GPU, 8 CPU, 48 GiB e massimo
90 minuti; invia mail BEGIN, END e FAIL a `marco.barezzi@unipr.it`.
La verifica GPU esegue Chamfer distance, rasterizzazione CUDA, caricamento
completo della pipeline e conditioning DINOv2. Non esegue un benchmark.

Verifica completata il 17 ottobre 2026 con job **60809990**, stato
`COMPLETED`, exit code `0:0`, durata 11m29s su NVIDIA A100-SXM-64GB.
Entrambi gli ambienti hanno superato tutti i controlli CPU e GPU, incluso
il caricamento offline di sette modelli e conditioning DINOv2 finito.
Log: `logs/baseline_env_install_60809990.{out,err}` nella radice del progetto.
L'elenco completo delle versioni installate e' in
`envs/<metodo>/requirements-installed.txt`.

Cache condivise: `cache/huggingface` (pipeline TRELLIS completa e encoder),
`cache/torch` (DINOv2 e pesi percettivi), `cache/u2net` (u2net.onnx),
`cache/torch_extensions` (compilazione JIT nvdiffrast). Per preparare da zero
le copie si usa `bash slurm/setup_baseline_envs_leonardo.sh --clone`
da un login node; i repository e i pesi devono essere scaricati prima del
job, che usa solo cache locali.

Sono disponibili anche gli alias esterni al repository:
`source /leonardo_work/IscrC_MORPHFL/mbarezzi/env_morphany3d.sh` e
`source /leonardo_work/IscrC_MORPHFL/mbarezzi/env_interp3d.sh`.
