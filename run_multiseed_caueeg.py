"""
Multi-seed significance analysis for the 4 headline CAUEEG-Dementia models
(ShallowCNN, ShallowGNN, CB-ShallowCNN, CB-GNN). Mirrors run_multiseed_tuh.py
-- see that file's docstring for the full rationale (torch.save monkeypatch
to avoid overwriting seed-42 checkpoints, seed-42 results reused rather than
re-run since they're already on disk / already reported in article.tex).
"""
import json
import random
import time

import numpy as np
import torch
import yaml

import train_caueeg_shallow_cnn
import train_caueeg_graph_cnn
import train_caueeg_concept_bottleneck
import train_caueeg_concept_bottleneck_gnn

_orig_save = torch.save
_suffix = [None]


def _patched_save(obj, path, *args, **kwargs):
    if _suffix[0] is not None and isinstance(path, str) and path.endswith(".pt"):
        path = path[:-3] + f"_{_suffix[0]}.pt"
    return _orig_save(obj, path, *args, **kwargs)


torch.save = _patched_save

# seed=42 results already on disk / already reported in article.tex tab:classification.
SEED42_RESULTS = {
    "shallow_cnn":    {"accuracy": 0.602, "f1": 0.601, "sensitivity": 0.601, "specificity": 0.802},
    "cb_shallow_cnn": {"accuracy": 0.610, "f1": 0.615, "sensitivity": 0.611, "specificity": 0.804},
    "gnn":            {"accuracy": 0.525, "f1": 0.532, "sensitivity": 0.534, "specificity": 0.762},
    "cb_gnn":         {"accuracy": 0.534, "f1": 0.531, "sensitivity": 0.535, "specificity": 0.762},
}

NEW_SEEDS = [43, 44]

MODELS = [
    ("shallow_cnn", train_caueeg_shallow_cnn, "config_caueeg_dementia_shallow_cnn.yaml"),
    ("gnn", train_caueeg_graph_cnn, "config_caueeg_dementia_graph_cnn.yaml"),
    ("cb_shallow_cnn", train_caueeg_concept_bottleneck, "config_caueeg_dementia_concept_bottleneck.yaml"),
    ("cb_gnn", train_caueeg_concept_bottleneck_gnn, "config_caueeg_dementia_concept_bottleneck_gnn.yaml"),
]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    device = torch.device("cpu")  # match original seed-42 runs (no CUDA on this machine)
    results = {name: [dict(SEED42_RESULTS[name])] for name, _, _ in MODELS}

    for name, module, config_path in MODELS:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for seed in NEW_SEEDS:
            cfg["training"]["seed"] = seed
            set_all_seeds(seed)
            _suffix[0] = f"seed{seed}"
            t0 = time.time()
            print(f"\n>>> starting {name} seed={seed}", flush=True)
            out = module.train_and_evaluate(cfg, device)
            metrics = out["metrics"] if isinstance(out, dict) and "metrics" in out else out
            elapsed = time.time() - t0
            print(f">>> done {name} seed={seed} acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1']:.4f} sens={metrics['sensitivity']:.4f} "
                  f"spec={metrics['specificity']:.4f} ({elapsed/60:.1f} min)", flush=True)
            results[name].append(metrics)
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
