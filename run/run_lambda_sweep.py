"""
Lambda (concept-loss weight) sweep for the PURE concept bottleneck
(CB-CNN-Plain) ONLY, on both datasets, across all 3 seeds: lambda_concept
in {5, 20}, compared against the EXISTING lambda=0.5 results already in
multiseed_tuh_results.json / multiseed_caueeg_results.json (NOT retrained
here -- reused as the baseline point of the sweep).

Scoped to the pure bottleneck only: lambda controls how hard concept
predictions are anchored to their true analytic values, which is the ONLY
lever available to the pure bottleneck (it has no residual pathway to fall
back on). For the hybrid, raising lambda would likely improve concept R^2
without necessarily reducing how much of its decision routes through the
residual (that split is set by the classification cross-entropy alone,
independent of lambda) -- so a lambda sweep doesn't test the same thing
there and isn't run by this script.

For each of the 2 new lambda values x 2 datasets x 3 seeds (12 runs):
  1. Trains CB-CNN-Plain from scratch (fresh random init, NOT warm-started
     from the lambda=0.5 checkpoint -- a clean sweep where every point is
     trained the same way) under its own NEW checkpoint name (e.g.
     tuh_concept_bottleneck_plain_lambda5_best_model_seed43.pt via
     train_utils.seed_checkpoint_name's usual seed-42-unsuffixed /
     seed-43/44-suffixed convention) -- the existing lambda=0.5
     checkpoints and multiseed results are never touched or overwritten.
  2. Immediately runs the full-concept intervention experiment (replace
     all 30 predicted concepts with their true values at once, compare
     accuracy) on that seed's freshly-trained checkpoint, reusing
     run_intervention_one_seed from intervention_experiment.py /
     intervention_experiment_caueeg.py UNMODIFIED -- this is the
     "leakage measure for the pure model" requested: a big post-
     intervention drop means the concepts the classifier actually learned
     to use differ from their true values (hidden information / miscalib-
     ration); a drop near zero means the pure model's concepts are already
     faithful enough that correcting them barely matters.

Saves everything to lambda_sweep_results.json (a NEW file -- does not
touch multiseed_tuh_results.json, multiseed_caueeg_results.json,
intervention_experiment_results.json, or
intervention_experiment_caueeg_results.json).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import json
import random
import time

import numpy as np
import torch
import yaml

import train_tuh_concept_bottleneck
import train_caueeg_concept_bottleneck
from run.intervention_experiment import run_intervention_one_seed as run_intervention_tuh
from run.intervention_experiment_caueeg import run_intervention_one_seed as run_intervention_caueeg
from train_utils import tee_stdout_to_file, SEEDS

_orig_save = torch.save
_suffix = [None]


def _patched_save(obj, path, *args, **kwargs):
    if _suffix[0] is not None and isinstance(path, str) and path.endswith(".pt"):
        path = path[:-3] + f"_{_suffix[0]}.pt"
    return _orig_save(obj, path, *args, **kwargs)


torch.save = _patched_save

LAMBDAS = [5, 20]

DATASETS = [
    ("tuh", train_tuh_concept_bottleneck, run_intervention_tuh, {
        5: "config/config_tuh_concept_bottleneck_plain_lambda5.yaml",
        20: "config/config_tuh_concept_bottleneck_plain_lambda20.yaml",
    }),
    ("caueeg", train_caueeg_concept_bottleneck, run_intervention_caueeg, {
        5: "config/config_caueeg_dementia_concept_bottleneck_plain_lambda5.yaml",
        20: "config/config_caueeg_dementia_concept_bottleneck_plain_lambda20.yaml",
    }),
]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one(dataset_name, module, config_path, lam, seed):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["seed"] = seed
    set_all_seeds(seed)
    _suffix[0] = f"seed{seed}" if seed != 42 else None
    t0 = time.time()
    log_path = f"logs/lambda_sweep_{dataset_name}_lambda{lam}_seed{seed}.log"
    with tee_stdout_to_file(log_path):
        print(f"\n>>> starting {dataset_name} lambda={lam} seed={seed}", flush=True)
        out = module.train_and_evaluate(cfg, torch.device("cpu"))
        elapsed = time.time() - t0
        print(f">>> done {dataset_name} lambda={lam} seed={seed} "
              f"acc={out['metrics']['accuracy']:.4f} ({elapsed/60:.1f} min)", flush=True)
    _suffix[0] = None
    record = dict(out["metrics"])
    record["concept_r2"] = out["concept_r2"]
    return record


def main():
    results = {}

    for dataset_name, module, run_intervention_fn, lambda_configs in DATASETS:
        results[dataset_name] = {}
        for lam in LAMBDAS:
            config_path = lambda_configs[lam]
            print(f"\n{'='*70}\n{dataset_name} lambda={lam}\n{'='*70}", flush=True)

            training_per_seed = {}
            for seed in SEEDS:
                training_per_seed[seed] = train_one(dataset_name, module, config_path, lam, seed)

            intervention_per_seed = {}
            for seed in SEEDS:
                variant_label = f"CB-CNN-Plain (lambda={lam})"
                intervention_per_seed[seed] = run_intervention_fn(variant_label, config_path, seed)

            accs = np.array([r["accuracy"] for r in training_per_seed.values()])
            r2_means = np.array([np.mean(list(r["concept_r2"].values())) for r in training_per_seed.values()])
            base_accs = np.array([r["baseline"]["accuracy"] for r in intervention_per_seed.values()])
            full_accs = np.array([r["full_intervention_accuracy"] for r in intervention_per_seed.values()])
            gains = np.array([r["gain"] for r in intervention_per_seed.values()])

            summary = {
                "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
                "concept_r2_mean": float(r2_means.mean()), "concept_r2_std": float(r2_means.std()),
                "intervention_baseline_accuracy_mean": float(base_accs.mean()),
                "intervention_baseline_accuracy_std": float(base_accs.std()),
                "intervention_full_accuracy_mean": float(full_accs.mean()),
                "intervention_full_accuracy_std": float(full_accs.std()),
                "intervention_gain_mean": float(gains.mean()), "intervention_gain_std": float(gains.std()),
            }
            print(f"\n{dataset_name} lambda={lam} summary: "
                  f"acc={summary['accuracy_mean']:.4f}+/-{summary['accuracy_std']:.4f}  "
                  f"R2={summary['concept_r2_mean']:.4f}+/-{summary['concept_r2_std']:.4f}  "
                  f"intervention_gain={summary['intervention_gain_mean']:+.4f}+/-{summary['intervention_gain_std']:.4f}",
                  flush=True)

            results[dataset_name][f"lambda{lam}"] = {
                "training_per_seed": {str(s): r for s, r in training_per_seed.items()},
                "intervention_per_seed": {str(s): r for s, r in intervention_per_seed.items()},
                "summary": summary,
            }

            with open("lambda_sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)

    print("\nSaved to lambda_sweep_results.json")


if __name__ == "__main__":
    main()
