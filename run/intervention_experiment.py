"""
Concept intervention experiment, run for BOTH CB-ShallowCNN variants: the
hybrid (residual=true, tuh_concept_bottleneck_best_model.pt) and the Plain
pure bottleneck (residual=false, tuh_concept_bottleneck_plain_best_model.pt).

Single full-intervention pass: ALL N_CONCEPTS (30) concepts' predicted
values are replaced with their true computed values SIMULTANEOUSLY at
inference time (models/concept_bottleneck.py's `intervention=` forward
argument), and the resulting TUAB test accuracy is compared against the
no-intervention baseline. No per-concept or progressive/staged breakdown,
and no R^2-based "working concept" prefilter (earlier versions of this
script had both; see models/concept_bottleneck.py -- no concept is excluded
anywhere in the pipeline any more, training or evaluation).

Running both variants side by side tests a specific prediction: the Plain
model has NO residual escape hatch, so correcting every concept at once is
the classifier's ONLY new information -- intervention should move its
accuracy at least as much as the hybrid's, and if the hybrid's residual is
absorbing concept errors, the hybrid's intervention gain should be smaller.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import json
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.tuh_e2e_loader import TUHEndToEndDataset
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneckShallowCNN, normalize_concepts
from train_utils import checkpoint_path
from utils.metrics import compute_loso_metrics


def run_pass(model, loader, device, concept_median, concept_iqr, intervene_indices=None):
    """One eval pass. If intervene_indices is given (a list of concept indices),
    every batch has ALL of those concepts' predicted values simultaneously
    overridden with their own true (per-sample) values before the classifier
    runs. Returns (preds, labels, true_concepts, pred_concepts)."""
    model.eval()
    preds, labels, true_c, pred_c = [], [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), concept_median, concept_iqr,
            )
            intervention = None
            if intervene_indices:
                intervention = {idx: true_concepts[:, idx] for idx in intervene_indices}
            logits, pred_concepts = model(x, intervention=intervention)

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(raw_batch["label"].tolist())
            true_c.append(true_concepts.cpu().numpy())
            pred_c.append(pred_concepts.cpu().numpy())
    return preds, labels, np.concatenate(true_c), np.concatenate(pred_c)


def run_intervention(variant_label, config_path):
    print(f"\n\n{'#'*70}\n# {variant_label}  ({config_path})\n{'#'*70}", flush=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, t, d = cfg["model"], cfg["training"], cfg["data"]

    seed = t["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = TUHWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    print(f"eval(test)={len(eval_ds)}", flush=True)

    model = ConceptBottleneckShallowCNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_filters=m.get("n_filters", 40),
        dropout=m.get("dropout", 0.5), residual=m.get("residual", True),
    ).to(device)
    checkpoint_name = m.get("checkpoint_name", "tuh_concept_bottleneck")
    ckpt = torch.load(checkpoint_path(f"{checkpoint_name}_best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    concept_median = ckpt["concept_median"].to(device)
    concept_iqr = ckpt["concept_iqr"].to(device)
    print(f"loaded checkpoint (best_epoch={ckpt['best_epoch']}, "
          f"best_val_bal_acc={ckpt['best_val_bal_acc']:.4f})", flush=True)

    # baseline pass (no intervention)
    base_preds, base_labels, _, _ = run_pass(
        model, eval_loader, device, concept_median, concept_iqr, intervene_indices=None,
    )
    base_metrics = compute_loso_metrics(base_preds, base_labels)
    base_acc = base_metrics["accuracy"]
    print(f"\n=== Baseline (no intervention) ===")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}")

    # full intervention: ALL N_CONCEPTS concepts corrected SIMULTANEOUSLY in one pass.
    all_indices = list(range(len(CONCEPT_NAMES)))
    preds, labels, _, _ = run_pass(
        model, eval_loader, device, concept_median, concept_iqr, intervene_indices=all_indices,
    )
    full_acc = compute_loso_metrics(preds, labels)["accuracy"]
    full_gain = full_acc - base_acc

    print(f"\n=== Full intervention (all {len(all_indices)} concepts corrected at once) ===")
    print(f"  baseline acc={base_acc:.4f}  post-intervention acc={full_acc:.4f}  gain={full_gain:+.4f}")

    print("\n=== LaTeX table row ===")
    sign = "+" if full_gain >= 0 else "-"
    print(f"All {len(all_indices)} concepts & {full_acc:.4f} & {sign}{abs(full_gain):.4f} \\\\")

    return {
        "baseline": base_metrics,
        "full_intervention_accuracy": full_acc,
        "gain": full_gain,
        "n_concepts": len(all_indices),
    }


def main():
    results = {
        "hybrid": run_intervention("CB-ShallowCNN (hybrid, residual=true)", "config/config_tuh_concept_bottleneck.yaml"),
        "plain": run_intervention("CB-ShallowCNN-Plain (pure bottleneck, residual=false)", "config/config_tuh_concept_bottleneck_plain.yaml"),
    }
    with open("intervention_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to intervention_experiment_results.json")


if __name__ == "__main__":
    main()
