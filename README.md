# EEG Classification with Concept Bottleneck Models

EEG classification on two datasets: the TUH Abnormal EEG Corpus
(v3.0.0, binary normal/abnormal) and CAUEEG (Chung-Ang University
Hospital EEG dataset for 3-class normal/MCI/dementia). Here we compare a plain
CNN backbone against a concept bottleneck model that routes classification through interpretable clinical EEG features.

## Models

- **`ShallowConvNet`** (`models/shallow_cnn.py`) -- temporal-conv ->
  spatial-conv -> square -> pool -> log EEG decoding backbone
  (Schirrmeister et al. 2017, "ShallowFBCSPNet"). The best-performing
  architecture found in this project; every attention/graph-based
  alternative tried has underperformed it.
- **`ConceptBottleneckShallowCNN`** (`models/concept_bottleneck.py`) --
  same `ShallowConvNet` backbone -> `N_CONCEPTS` (30) predicted clinical EEG
  concepts (regional band powers, hemispheric asymmetries, theta/alpha
  ratios, alpha peak frequency/amplitude, interhemispheric alpha-band PLV
  connectivity) -> a classifier that consumes the concepts (not raw
  backbone features) instead. Makes intervention
  meaningful: overriding a predicted concept and re-running the
  classifier actually changes the output, since the classifier has no
  shortcut back to the raw features. Region/asymmetry-pair-to-channel-index
  mappings are dataset-specific (`REGIONS`/`ASYM_PAIRS` for TUH,
  `REGIONS_CAUEEG`/`ASYM_PAIRS_CAUEEG` for CAUEEG -- verified programmatically
  from each dataset's own channel order, not assumed by analogy). TUH's
  variant additionally hard-zeroes 4 concepts confirmed to carry no signal
  there (Spearman rank correlation not statistically significant) --
  CAUEEG's doesn't assume that finding transfers, so nothing is masked there
  without separately checking.

All models take `n_channels`/`n_classes` as constructor parameters, so
the same classes serve both datasets and both CAUEEG tasks.

## CAUEEG-specific details

CAUEEG's clinical protocol alternates brief eyes-open/closed instructions
with photic driving-response blocks (3-30Hz flash) rather than one long
continuous resting recording like TUH. `CAUEEGEndToEndDataset`
(`data/caueeg_e2e_loader.py`) selects only eyes-closed, non-photic
segments (via each subject's event log) to avoid the photic response
contaminating the alpha/beta band-power concepts, concatenates them, and
chops the result into non-overlapping `window_sec` (60s default) windows.
**Train** subjects contribute several windows each (capped at
`max_windows_per_subject`, so a few very long recordings don't dominate);
**val** always stays one window per subject (fast per-epoch checkpoint
selection); **eval** uses one window per subject unless `eval_tta: true`
(default), in which case it gets the same multi-window treatment as
train and predictions are averaged back to one-per-subject
(`aggregate_predictions_by_subject`, `utils/metrics.py`) before scoring
-- test-time augmentation, matching the validated approach in the
official CAUEEG reference implementation
([ipis-mjkim/caueeg-ceednet](https://github.com/ipis-mjkim/caueeg-ceednet)).
Concept R^2 (concept-bottleneck scripts) is reported per-window, not
TTA-aggregated -- each window has its own true concepts.

CAUEEG also gets an additional per-channel z-score normalization
(`compute_eeg_channel_norm`/`normalize_eeg`), fit on the training set and
applied at batch time on top of the existing fixed-divisor scale, right
before the model sees the signal -- also matching the reference
implementation's normalization. It does NOT affect concept computation,
which still reads the divisor-scaled (not z-scored) raw signal.

CAUEEG's loader (`data.task` in the config) also supports an `"abnormal"`
binary task sharing the same signal files/splits as `"dementia"`, but
this project only trains/evaluates the `"dementia"` task -- no
`config_caueeg_abnormal_*.yaml` ships here. The config path is a
command-line argument rather than hardcoded (one script, one config per
model):
`python run/train_caueeg_shallow_cnn.py config/config_caueeg_dementia_shallow_cnn.yaml`
(defaults to the `dementia` config if omitted). The class remapping
(`Dementia=0, MCI=1, Normal=2`, healthy class last) matches this
project's "pathology first" convention used everywhere else.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

All scripts live in `run/` and must be invoked from the repo root (e.g.
`python run/train_tuh_shallow_cnn.py`, not from inside `run/`) -- each one
inserts the repo root onto `sys.path` at import time so `models.*`,
`data.*`, `utils.*`, and `train_utils` (all still at the repo root)
resolve regardless of where `python` itself is invoked from, but config
paths, `logs/`, `saved_models/`, and the data caches are all plain
relative paths resolved against the current working directory, so it
must be the repo root.

Each script reads a YAML config for data paths, model hyperparameters,
and training settings. Every `train_*.py` script also writes its full
run output (epoch-by-epoch progress, final eval results) to a `.log`
file in `logs/` (matching its checkpoint's name), in addition to
printing to the console (`train_utils.tee_stdout_to_file`) -- both are
overwritten each run, same as the checkpoint. `logs/` is created
automatically if it doesn't exist yet, and is gitignored (local run
output, not source).

```bash
# --- TUH ---
python run/train_tuh_shallow_cnn.py               # plain ShallowConvNet (config hardcoded)
python run/train_tuh_concept_bottleneck.py          # CB-CNN (hybrid, residual=true) -- default config if no argv
python run/train_tuh_concept_bottleneck.py config/config_tuh_concept_bottleneck_plain.yaml   # "Plain" CBM, residual=false

# --- CAUEEG (config path as argv; defaults to dementia if omitted) ---
python run/train_caueeg_shallow_cnn.py               config/config_caueeg_dementia_shallow_cnn.yaml
python run/train_caueeg_concept_bottleneck.py          config/config_caueeg_dementia_concept_bottleneck.yaml
python run/train_caueeg_concept_bottleneck.py          config/config_caueeg_dementia_concept_bottleneck_plain.yaml
```

`residual: true` (the default) is a **hybrid** CBM: the classifier sees the predicted
concepts concatenated with a residual projection of the raw backbone features
(`eq:classifier` in article.tex), which is why the leakage analysis (below) exists at
all. `residual: false` -- the `*_plain.yaml` configs -- is a **pure** bottleneck: the
classifier sees only the concept vector, no escape hatch. Both share the same
`ConceptBottleneckShallowCNN` class (`models/concept_bottleneck.py`); only the
`residual` flag and `checkpoint_name` (so the two variants don't overwrite each
other's checkpoint/log) differ between the configs.

### Stability, significance, and interpretability analysis (article.tex results)

These run against the already-trained seed-42 checkpoints above (inference-only
except the multi-seed scripts, which retrain with 2 extra seeds) and produce the
numbers reported in article.tex:

```bash
# seeds: retrain each of the 3 headline models (ShallowCNN, CB-ShallowCNN
# hybrid, CB-ShallowCNN-Plain) for all 3 seeds (42, 43, 44) -- mean +/- std
# accuracy per model (article.tex section 5.1). torch.save is monkeypatched
# so seeds 43/44 get a suffixed filename and seed 42 saves under its normal
# name (the canonical checkpoint every other analysis script below loads).
python run/run_multiseed_tuh.py
python run/run_multiseed_caueeg.py

# significance: McNemar's exact test + paired-bootstrap 95% CI, pairwise
# across ShallowCNN, CB-ShallowCNN (hybrid), and CB-ShallowCNN-Plain (pure
# bottleneck, residual=false) -- needs saved_models/*_concept_bottleneck_
# plain_best_model.pt too (see "Plain concept bottleneck" ablation below).
# Saves pairwise_significance_{tuh,caueeg}_results.json.
# (article.tex section 5.2, tables tab:mcnemar / tab:mcnemar_caueeg).
python run/run_pairwise_significance_tuh.py
python run/run_pairwise_significance_caueeg.py

# intervention: replace ALL predicted concepts with their true computed values
# SIMULTANEOUSLY at inference (single full-intervention pass, no per-concept or
# progressive staging) and measure the TUAB accuracy change -- runs for BOTH
# CB-ShallowCNN (hybrid) and CB-ShallowCNN-Plain in one invocation, to compare
# intervention leverage with vs without the residual escape hatch. Saves
# intervention_experiment_results.json.
# (article.tex section 5.4, table tab:intervention).
python run/intervention_experiment.py

# leakage: freeze the seed-42 CB-ShallowCNN (hybrid) backbone and train fresh
# linear probes on concept-only / residual-only / combined extracted features,
# to decompose how much classification signal bypasses the concept bottleneck.
# CB-ShallowCNN-Plain has no residual pathway to analyze this way by
# construction, so it's intentionally excluded here (see the script's docstring).
# Saves leakage_analysis_tuh_results.json.
# (article.tex section 5.5, table tab:leakage).
python run/run_leakage_analysis_tuh.py

# concept data-quality audit: validity/low-variance/redundancy/train-test-shift/
# clinical-sanity checks on the cached concept values directly -- no trained
# checkpoint needed, just a populated concept cache. Console output only.
python run/check_concept_quality.py --dataset tuh --raw
python run/check_concept_quality.py --dataset caueeg --raw
```

TUH scripts share one raw-EEG cache (`data_cache/tuh_e2e_cache`); CAUEEG
scripts share `data_cache/caueeg_e2e_cache` (both tasks -- preprocessing
doesn't depend on which task's labels/split are used). The concept-
bottleneck scripts additionally use a separate concept cache
(`data_cache/tuh_concepts_cache` / `data_cache/caueeg_concepts_cache`)
so raw concept values aren't recomputed on every run. All caches, and
every `*.pt` checkpoint, are gitignored (see `.gitignore`) -- they're
local build artifacts, not source.

Every `train_*.py` script saves its checkpoint to `saved_models/<name>.pt`
via `train_utils.checkpoint_path` (creates the directory on first run), and
every analysis script that loads a checkpoint (`intervention_experiment.py`,
`run_pairwise_significance_*.py`, `run_leakage_analysis_tuh.py`) reads from
the same helper, so the checkpoint directory only needs to change in one
place (`train_utils.CHECKPOINT_DIR`). Every multiseed/significance/
intervention/leakage script also saves its own results to a `*_results.json`
file in the repo root (in addition to printing to console) -- nothing here
is console-only.

## Layout

```
config/config_tuh_*.yaml                  TUH configs (one per model)
config/config_caueeg_dementia_*.yaml      CAUEEG configs (one per model, dementia task only)
train_utils.py                split_validation (TUH's class-stratified train/val split),
                               tee_stdout_to_file (run output -> console + logs/*.log),
                               checkpoint_path (saved_models/ helper, save + load)
logs/                          run output per script (gitignored, auto-created)
saved_models/                  checkpoints per script (gitignored, auto-created)
run/                           every runnable script (see Usage) -- each inserts the repo
                                 root onto sys.path at import time (see Usage) so it can still
                                 import models/data/utils/train_utils from one level up
  train_tuh_shallow_cnn.py
  train_tuh_concept_bottleneck.py
  train_caueeg_shallow_cnn.py
  train_caueeg_concept_bottleneck.py
  run_multiseed_tuh.py                    seeds 42/43/44 for the 3 headline models (TUH)
  run_multiseed_caueeg.py                 same, CAUEEG
  run_pairwise_significance_tuh.py        McNemar's test + bootstrap CI, 3 headline models (TUH)
  run_pairwise_significance_caueeg.py     same, CAUEEG
  intervention_experiment.py              concept intervention, CB-ShallowCNN hybrid + Plain (TUH)
  run_leakage_analysis_tuh.py             frozen-backbone linear-probe leakage decomposition (TUH)
  check_concept_quality.py                concept data-quality audit (validity, variance, redundancy,
                                            train/test shift, clinical sanity) -- reads the concept
                                            cache directly, no raw EEG or trained checkpoint needed
data/
  tuh_e2e_loader.py            TUHEndToEndDataset -- raw time series, CAR, fixed-scale norm
  tuh_concepts_loader.py       TUHWithConceptsDataset -- wraps TUHEndToEndDataset + cached concepts
  caueeg_e2e_loader.py         CAUEEGEndToEndDataset -- eyes-closed segment selection + windowing,
                                per-channel normalization helpers, TTA-eval support
  caueeg_concepts_loader.py    CAUEEGWithConceptsDataset -- wraps CAUEEGEndToEndDataset + cached concepts
  concept_cache.py             compute-once-then-cache for raw clinical EEG concepts (shared)
models/
  shallow_cnn.py                ShallowConvNet
  concept_bottleneck.py          ConceptBottleneckShallowCNN, concept computation
utils/
  metrics.py                   compute_loso_metrics (accuracy/F1/sensitivity/specificity),
                                aggregate_predictions_by_subject (TTA aggregation)
```
