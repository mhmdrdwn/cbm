# EEG Abnormality Classification: CNN, Graph-Coupling, and Concept Bottleneck Models

Binary normal/abnormal EEG classification on two datasets -- the
[NMT Scalp EEG Dataset](https://dll.seecs.nust.edu.pk/downloads/) and the
TUH Abnormal EEG Corpus (v3.0.0) -- comparing a plain CNN backbone against
a graph-coupling extension and a concept bottleneck model that routes
classification through interpretable clinical EEG features.

This project earlier explored transformer/attention-based architectures
(channel-graph attention with Jansen-Rit physics constraints and Granger-
causality edges) and other datasets (ds004504, an fMRI connectivity
dataset); all were dropped after every fully-learned cross-channel
attention mechanism tried underperformed a much simpler CNN backbone on
real held-out accuracy (see `models/shallow_cnn.py`'s docstring). That
earlier code has been removed from this repo -- see git history if it's
ever needed again.

## Models

- **`ShallowConvNet`** (`models/shallow_cnn.py`) -- temporal-conv ->
  spatial-conv -> square -> pool -> log EEG decoding backbone
  (Schirrmeister et al. 2017, "ShallowFBCSPNet"). The best-performing
  architecture found in this project; every attention/graph-based
  alternative tried has underperformed it.
- **`ShallowCNNPhysicsLoss`** (`models/shallow_cnn.py`) -- optional
  physics/anatomy-informed auxiliary losses on top of `ShallowConvNet`
  (NMT only): an electrode-coupling prior pulling the spatial filter's
  implied channel similarity toward a learnable, scalp-position-
  initialized embedding, plus a class-conditional Jansen-Rit excitatory-
  gain term. Used by `train_nmt_shallow_cnn.py`; `train_tuh_shallow_cnn.py`
  intentionally skips it (see that script's docstring for why).
- **`ShallowGNN`** (`models/graph_coupling_cnn.py`) -- multi-hop
  extension of `ShallowConvNet`: after the temporal conv, each hop
  recomputes a per-sample cosine-similarity adjacency over the model's
  own channel features and mixes across it with a learnable per-hop
  scalar, before the usual spatial collapse. At `n_hops=1` this reduces
  exactly to a single-hop coupling design. TUH only.
- **`ConceptBottleneckShallowCNN`** / **`ConceptBottleneckGNN`**
  (`models/concept_bottleneck.py`) -- CNN or graph-coupling backbone ->
  28 predicted clinical EEG concepts (regional band powers, hemispheric
  asymmetries, theta/alpha ratios, alpha peak frequency/amplitude) -> a
  classifier that consumes the concepts (not raw backbone features)
  instead. TUH only. Makes intervention meaningful: overriding a
  predicted concept and re-running the classifier actually changes the
  output, since the classifier has no shortcut back to the raw features.
  `ConceptBottleneckGNN` adds a small dedicated FFT-isolated beta-band
  branch (13-30Hz) alongside the main graph-coupling backbone, and both
  variants hard-zero 4 concepts confirmed to carry no signal (Spearman
  rank correlation not statistically significant) before the classifier
  sees them.


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Each script reads its own YAML config (same base name, `config_*.yaml`)
for data paths, model hyperparameters, and training settings.

```bash
# NMT: ShallowConvNet + physics-informed losses
python train_nmt_shallow_cnn.py

# TUH: plain ShallowConvNet (no physics loss)
python train_tuh_shallow_cnn.py

# TUH: multi-hop graph-coupling extension
python train_tuh_graph_cnn.py

# TUH: concept bottleneck, CNN backbone
python train_tuh_concept_bottleneck.py

# TUH: concept bottleneck, graph-coupling backbone + beta branch
python train_tuh_concept_bottleneck_gnn.py

# Concept bottleneck analysis (run against an already-trained GNN checkpoint):
python check_concept_rank_correlation.py       # R^2 vs Spearman per concept
python tune_tuh_concept_bottleneck_threshold.py # decision-threshold tuning (fit on val, apply to eval)
python recalibrate_tuh_concept_bottleneck.py    # isotonic recalibration (fit on val, apply to eval)
```

All TUH scripts share one raw-EEG cache (`data_cache/tuh_e2e_cache`); the
concept-bottleneck scripts additionally use a separate concept cache
(`data_cache/tuh_concepts_cache`) so raw concept values aren't
recomputed on every run. Both caches, and all `*.pt` checkpoints, are
gitignored (see `.gitignore`) -- they're local build artifacts, not
source.

## Layout

```
config_nmt_shallow_cnn.yaml              NMT ShallowConvNet + physics losses
config_tuh_shallow_cnn.yaml              TUH plain ShallowConvNet
config_tuh_graph_cnn.yaml                TUH ShallowGNN (graph coupling)
config_tuh_concept_bottleneck.yaml       TUH ConceptBottleneckShallowCNN
config_tuh_concept_bottleneck_gnn.yaml   TUH ConceptBottleneckGNN
train_nmt_shallow_cnn.py
train_tuh_shallow_cnn.py
train_tuh_graph_cnn.py
train_tuh_concept_bottleneck.py
train_tuh_concept_bottleneck_gnn.py
train_utils.py                split_validation (class-stratified train/val split)
check_concept_rank_correlation.py       R^2 vs Spearman per concept
tune_tuh_concept_bottleneck_threshold.py  decision-threshold tuning
recalibrate_tuh_concept_bottleneck.py     isotonic recalibration
data/
  nmt_e2e_loader.py            NMTEndToEndDataset -- raw time series, CAR, fixed-scale norm
  tuh_e2e_loader.py            TUHEndToEndDataset -- TUH counterpart
  electrode_positions.py       real 10-20 scalp coordinates (MNE standard_1020 montage)
  concept_cache.py             compute-once-then-cache for raw clinical EEG concepts
  tuh_concepts_loader.py       TUHWithConceptsDataset -- wraps TUHEndToEndDataset + cached concepts
models/
  shallow_cnn.py                ShallowConvNet, ShallowCNNPhysicsLoss
  graph_coupling_cnn.py          ShallowGNN (multi-hop graph coupling)
  concept_bottleneck.py          ConceptBottleneckShallowCNN, ConceptBottleneckGNN, concept computation
utils/
  metrics.py                   compute_loso_metrics (accuracy/F1/sensitivity/specificity)
```
