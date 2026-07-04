# Cross-Subject Contrastive GNN for EEG

A graph neural network for EEG classification that learns connectivity
patterns invariant across subjects within the same class.

Two pathways run in parallel:

- **Topology encoder** — takes the adjacency matrix only, learns
  subject-invariant graph structure.
- **Feature encoder** — takes signals + adjacency, temporal GNN (A3T-GCN).

Three losses train jointly:

- Cross-entropy for classification.
- Supervised contrastive on topology embeddings (same-label subjects are
  positive pairs).
- Multi-view contrastive on PLV vs. Coherence graphs of the same subject.

## Datasets

Switchable via `data.dataset` in `config.yaml`. Status of each:

| Dataset | Status | Classes | Channels |
|---|---|---|---|
| **MMIDB** | ✅ active default | 2 (real vs. imagined movement) | 64 |
| **ADFTD** | ✅ working, opt in | 3 (AD / FTD / HC) | 19 |

### MMIDB (`data.dataset: "MMIDB"`) — active default

[PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/)
— 109 subjects, 64 channels, 160 Hz. No registration.

Task: **real fist movement (runs 3/7/11) vs. imagined fist movement
(runs 4/8/12)**, one label per run recording. (True left-hand-vs-right-hand
motor imagery is a *trial-level* label — both classes alternate within a
single run — so it doesn't fit the current "one label per item" framework;
that would need per-trial epoching instead of per-run items.)

`data/loader.py` calls `mne.datasets.eegbci.load_data(...)` on first access,
fetching each subject/run `.edf` from PhysioNet and caching it under
`~/mne_data` (override with `data.mmidb.download_dir`). Subjects 88, 92,
100, and 104 are natively 128 Hz instead of 160 Hz, but every recording is
resampled to `sfreq` during preprocessing regardless of source rate, so all
109 subjects are used.

### ADFTD (`data.dataset: "ADFTD"`)

[OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) — EEG from
Alzheimer's disease (AD), frontotemporal dementia (FTD), and healthy
controls (HC). CC0, no registration. 88 subjects (36 AD / 23 FTD / 29 HC),
19 channels, resting-state eyes-closed, ~10 min each. This is the task the
contrastive design was actually built for (subject-invariant structure
within a pathology class), unlike MMIDB's motor-imagery task.

One label per subject recording (AD=0, FTD=1, HC=2). `data/loader.py` uses
`openneuro.download(...)` to fetch each subject's already-preprocessed
derivative (filtered, re-referenced, ICA-cleaned) `.set` file, cached under
`data.adftd.download_dir` (default `./data_cache/ds004504`).

**Class-imbalance caveat:** FTD has only 23 subjects total, so the
stratified batch sampler is capped accordingly — `data.adftd.batch_size`
is 24 (8 per class), smaller than MMIDB's 32 (16 per class). If results
are noisy, dropping to binary AD-vs-HC (65 subjects, better balanced) is
the fallback.

Both MMIDB and ADFTD split subjects deterministically (`split_ratios`,
seeded by `training.seed`) so no subject appears in more than one split —
ADFTD's split is additionally stratified by diagnostic group.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py       # trains, saves best_model.pt on validation accuracy
python evaluate.py --checkpoint best_model.pt --split test --plots_dir plots
```

Connectivity (PLV, Coherence) is computed once per item (run/recording) and
cached to `./cache/` as `.pt` files, since recomputing it from raw signals
on every epoch is expensive.

**Known limitation:** batches are stratified by label only, not by
subject, so a batch can contain multiple items from the same subject (only
possible for MMIDB, which has 6 runs/subject — ADFTD has one item per
subject so this doesn't apply there). The supervised contrastive loss then
treats some same-subject pairs as "positive," which dilutes the
cross-subject-invariance signal it's meant to isolate. Worth revisiting
with a subject-diverse sampler if MMIDB results are weaker than expected.

## Ablations

Toggle loss terms in `config.yaml` under `ablation:`

| Config | topo contrastive | feat contrastive | multi-view |
|--------|:---:|:---:|:---:|
| A — baseline           | ✗ | ✗ | ✗ |
| B — feature contrastive | ✗ | ✓ | ✗ |
| C — topology contrastive (novel) | ✓ | ✗ | ✗ |
| D — full model          | ✓ | ✓ | ✓ |

The key comparison is **C vs. B**: same framework, differing only in
whether the contrastive loss acts on topology embeddings or feature
embeddings.

## Layout

```
config.yaml
train.py / evaluate.py
data/
  loader.py           MMIDB + ADFTD loaders, shared windowing/connectivity caching, build_datasets()
  connectivity.py      PLV + Coherence computation
  pair_builder.py       positive/negative pair index, stratified batch sampler
models/
  cnn_encoder.py        2D-conv window encoder (node features)
  topology_encoder.py    adjacency -> topology embedding
  feature_encoder.py     A3T-GCN temporal graph model
  model.py               full two-pathway model
losses/
  contrastive.py         SupCon + multi-view losses
utils/
  metrics.py             accuracy, F1, per-class
  visualise.py            UMAP, similarity matrix
```
