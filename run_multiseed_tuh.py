"""
Multi-seed significance analysis for the 4 headline TUAB models
(ShallowCNN, ShallowGNN, CB-ShallowCNN, CB-GNN). Retrains each with 2
additional seeds (43, 44) and combines with the existing seed-42 result
(already on disk / already reported in article.tex, verified against
logs/tuh_*_run.log) to get mean +/- std accuracy across 3 seeds.

torch.save is monkeypatched for the duration of each run to redirect to a
seed-suffixed checkpoint filename (e.g. tuh_shallow_cnn_best_model_seed43.pt)
so the original seed-42 checkpoints -- used by check_concept_rank_correlation.py,
tune_tuh_concept_bottleneck_threshold.py, recalibrate_tuh_concept_bottleneck.py,
intervention_experiment.py/_gnn.py -- are never overwritten.

Run in background; writes incremental progress to stdout and final results
to multiseed_tuh_results.json.
"""
import json
import random
import time

import numpy as np
import torch
import yaml

import train_tuh_shallow_cnn
import train_tuh_graph_cnn
import train_tuh_concept_bottleneck
import train_tuh_concept_bottleneck_gnn

_orig_save = torch.save
_suffix = [None]


def _patched_save(obj, path, *args, **kwargs):
    if _suffix[0] is not None and isinstance(path, str) and path.endswith(".pt"):
        path = path[:-3] + f"_{_suffix[0]}.pt"
    return _orig_save(obj, path, *args, **kwargs)


torch.save = _patched_save

# seed=42 results already on disk / already reported in article.tex tab:classification,
# verified against logs/tuh_*_run.log earlier in this project -- not re-run here.
SEED42_RESULTS = {
    "shallow_cnn":    {"accuracy": 0.851, "f1": 0.850, "sensitivity": 0.778, "specificity": 0.913},
    "cb_shallow_cnn": {"accuracy": 0.841, "f1": 0.838, "sensitivity": 0.730, "specificity": 0.933},
    "gnn":            {"accuracy": 0.837, "f1": 0.836, "sensitivity": 0.786, "specificity": 0.880},
    "cb_gnn":         {"accuracy": 0.859, "f1": 0.858, "sensitivity": 0.818, "specificity": 0.893},
}

NEW_SEEDS = [43, 44]

MODELS = [
    ("shallow_cnn", train_tuh_shallow_cnn, "config_tuh_shallow_cnn.yaml"),
    ("gnn", train_tuh_graph_cnn, "config_tuh_graph_cnn.yaml"),
    ("cb_shallow_cnn", train_tuh_concept_bottleneck, "config_tuh_concept_bottleneck.yaml"),
    ("cb_gnn", train_tuh_concept_bottleneck_gnn, "config_tuh_concept_bottleneck_gnn.yaml"),
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

            with open("multiseed_tuh_results.json", "w") as f:
                json.dump(results, f, indent=2)

    print("\n\n=== SUMMARY (mean +/- std across 3 seeds: 42, 43, 44) ===")
    for name, runs in results.items():
        accs = [r["accuracy"] for r in runs]
        print(f"{name}: acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}  raw={accs}")
    print("\nSaved to multiseed_tuh_results.json")


if __name__ == "__main__":
    main()
