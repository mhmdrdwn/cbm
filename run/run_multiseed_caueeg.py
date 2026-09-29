"""
Multi-seed significance analysis for the 3 headline CAUEEG-Dementia models
(ShallowCNN, CB-ShallowCNN hybrid, CB-ShallowCNN-Plain pure bottleneck).
Mirrors run_multiseed_tuh.py -- see that file's docstring for the full
rationale (retrains all 3 seeds from scratch, seed 42 saved under its
normal unsuffixed checkpoint name so this script alone produces the
canonical seed-42 checkpoints that run_pairwise_significance_caueeg.py
loads).
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

import train_caueeg_shallow_cnn
import train_caueeg_concept_bottleneck
from train_utils import tee_stdout_to_file

_orig_save = torch.save
_suffix = [None]


def _patched_save(obj, path, *args, **kwargs):
    if _suffix[0] is not None and isinstance(path, str) and path.endswith(".pt"):
        path = path[:-3] + f"_{_suffix[0]}.pt"
    return _orig_save(obj, path, *args, **kwargs)


torch.save = _patched_save

SEEDS = [42, 43, 44]

MODELS = [
    ("shallow_cnn", train_caueeg_shallow_cnn, "config/config_caueeg_dementia_shallow_cnn.yaml"),
    ("cb_shallow_cnn", train_caueeg_concept_bottleneck, "config/config_caueeg_dementia_concept_bottleneck.yaml"),
    ("cb_shallow_cnn_plain", train_caueeg_concept_bottleneck, "config/config_caueeg_dementia_concept_bottleneck_plain.yaml"),
]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    device = torch.device("cpu")
    results = {name: [] for name, _, _ in MODELS}

    for name, module, config_path in MODELS:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for seed in SEEDS:
            cfg["training"]["seed"] = seed
            set_all_seeds(seed)
            _suffix[0] = f"seed{seed}" if seed != 42 else None
            t0 = time.time()
            log_path = f"logs/multiseed_caueeg_{name}_seed{seed}.log"
            with tee_stdout_to_file(log_path):
                print(f"\n>>> starting {name} seed={seed}", flush=True)
                out = module.train_and_evaluate(cfg, device)
                metrics = out["metrics"] if isinstance(out, dict) and "metrics" in out else out
                concept_r2 = out.get("concept_r2") if isinstance(out, dict) else None
                elapsed = time.time() - t0
                print(f">>> done {name} seed={seed} acc={metrics['accuracy']:.4f} "
                      f"f1={metrics['f1']:.4f} sens={metrics['sensitivity']:.4f} "
                      f"spec={metrics['specificity']:.4f} ({elapsed/60:.1f} min)", flush=True)
            print(f"    (full run output, including any concept R^2 table, saved to {log_path})", flush=True)
            record = dict(metrics)
            if concept_r2 is not None:
                record["concept_r2"] = concept_r2
            results[name].append(record)
            _suffix[0] = None

            with open("multiseed_caueeg_results.json", "w") as f:
                json.dump(results, f, indent=2)

    print("\n\n=== SUMMARY (mean +/- std across 3 seeds: 42, 43, 44) ===")
    for name, runs in results.items():
        accs = [r["accuracy"] for r in runs]
        print(f"{name}: acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}  raw={accs}")
    print("\nSaved to multiseed_caueeg_results.json")


if __name__ == "__main__":
    main()
