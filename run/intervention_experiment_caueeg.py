"""
Concept intervention experiment for CAUEEG-Dementia, mirroring
run/intervention_experiment.py (TUH) -- see that file's docstring for the
full rationale (hybrid vs Plain pure-bottleneck comparison, all N_CONCEPTS
(30) concepts corrected SIMULTANEOUSLY in one pass, no per-concept or
progressive/staged breakdown), run across ALL 3 seeds (42, 43, 44) and
aggregated as mean +/- std.

The one CAUEEG-specific difference: eval_tta means an eval "subject" can
have several windows, each with its OWN true concepts (different EEG
segment) -- intervention overrides each window's predicted concepts with
that window's own true concepts, then predictions are aggregated by subject
(softmax-averaged) exactly like the main classifier's own reported accuracy
(aggregate_predictions_by_subject), so the intervened accuracy stays
comparable to the baseline's.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.caueeg_e2e_loader import CAUEEGEndToEndDataset, compute_eeg_channel_norm, normalize_eeg
from data.caueeg_concepts_loader import CAUEEGWithConceptsDataset, collate_caueeg_concepts
from models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneckShallowCNN, normalize_concepts
from train_utils import checkpoint_path, seed_checkpoint_name, SEEDS
from utils.metrics import aggregate_predictions_by_subject, compute_loso_metrics

TASK = "dementia"


def run_pass(model, loader, device, concept_median, concept_iqr, intervene_indices=None):
    """One eval pass over (possibly multi-window-per-subject) CAUEEG data.
    Returns per-subject-aggregated (preds, labels) via TTA, same aggregation
    the main classifier's own reported accuracy uses."""
    model.eval()
    all_probs, all_labels_raw, all_sids = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg_norm"]
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), concept_median, concept_iqr,
            )
            intervention = None
            if intervene_indices:
                intervention = {idx: true_concepts[:, idx] for idx in intervene_indices}
            logits, _ = model(x, intervention=intervention)

            all_probs.extend(F.softmax(logits, dim=-1).cpu().tolist())
            all_labels_raw.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])
    preds, labels = aggregate_predictions_by_subject(all_probs, all_sids, all_labels_raw)
    return preds, labels


def run_intervention_one_seed(variant_label, config_path, seed):
    print(f"\n\n{'#'*70}\n# {variant_label}  ({config_path})  seed={seed}\n{'#'*70}", flush=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, d = cfg["model"], cfg["data"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=TASK, cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=TASK, cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"], eval_tta=d.get("eval_tta", True),
    )
    train_ds = CAUEEGWithConceptsDataset(train_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    eval_ds = CAUEEGWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    # eeg_mean/std are population stats over train subjects, deterministic given the
    # (seed-independent) train split -- recomputing here reproduces training-time exactly.
    eeg_mean, eeg_std = compute_eeg_channel_norm(train_ds, list(range(len(train_ds))))

    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_caueeg_concepts)
    print(f"eval(test windows)={len(eval_ds)}", flush=True)

    model = ConceptBottleneckShallowCNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_filters=m.get("n_filters", 40),
        dropout=m.get("dropout", 0.5), residual=m.get("residual", True),
    ).to(device)
    checkpoint_name = m.get("checkpoint_name", f"caueeg_{TASK}_concept_bottleneck")
    ckpt = torch.load(checkpoint_path(seed_checkpoint_name(checkpoint_name, seed)), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    concept_median = ckpt["concept_median"].to(device)
    concept_iqr = ckpt["concept_iqr"].to(device)
    print(f"loaded checkpoint (best_epoch={ckpt['best_epoch']}, "
          f"best_val_bal_acc={ckpt['best_val_bal_acc']:.4f})", flush=True)

    # normalize_eeg is applied once per batch up front (same for baseline and
    # intervention passes), so wrap the loader to attach it to each batch.
    def normed_loader():
        for raw_batch in eval_loader:
            raw_batch["raw_eeg_norm"] = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            yield raw_batch

    # baseline pass (no intervention)
    base_preds, base_labels = run_pass(
        model, list(normed_loader()), device, concept_median, concept_iqr, intervene_indices=None,
    )
    base_metrics = compute_loso_metrics(base_preds, base_labels)
    base_acc = base_metrics["accuracy"]
    print(f"\n=== Baseline (no intervention) ===")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}")

    # full intervention: ALL N_CONCEPTS concepts corrected SIMULTANEOUSLY in one pass.
    all_indices = list(range(len(CONCEPT_NAMES)))
    preds, labels = run_pass(
        model, list(normed_loader()), device, concept_median, concept_iqr, intervene_indices=all_indices,
    )
    full_acc = compute_loso_metrics(preds, labels)["accuracy"]
    full_gain = full_acc - base_acc

    print(f"\n=== Full intervention (all {len(all_indices)} concepts corrected at once) ===")
    print(f"  baseline acc={base_acc:.4f}  post-intervention acc={full_acc:.4f}  gain={full_gain:+.4f}")

    return {
        "baseline": base_metrics,
        "full_intervention_accuracy": full_acc,
        "gain": full_gain,
        "n_concepts": len(all_indices),
    }


def run_intervention(variant_label, config_path):
    """Runs run_intervention_one_seed for every seed in SEEDS, then aggregates
    baseline/full-intervention accuracy and gain as mean +/- std across seeds
    (same convention as run_multiseed_caueeg.py's accuracy/R^2 reporting)."""
    per_seed = {seed: run_intervention_one_seed(variant_label, config_path, seed) for seed in SEEDS}

    base_accs = np.array([r["baseline"]["accuracy"] for r in per_seed.values()])
    full_accs = np.array([r["full_intervention_accuracy"] for r in per_seed.values()])
    gains = np.array([r["gain"] for r in per_seed.values()])
    n_concepts = next(iter(per_seed.values()))["n_concepts"]

    summary = {
        "baseline_accuracy_mean": float(base_accs.mean()),
        "baseline_accuracy_std": float(base_accs.std()),
        "full_intervention_accuracy_mean": float(full_accs.mean()),
        "full_intervention_accuracy_std": float(full_accs.std()),
        "gain_mean": float(gains.mean()),
        "gain_std": float(gains.std()),
        "n_concepts": n_concepts,
        "seeds": SEEDS,
    }

    print(f"\n\n{'='*70}\n# {variant_label}: aggregate over seeds {SEEDS}\n{'='*70}")
    print(f"  baseline acc  = {summary['baseline_accuracy_mean']:.4f} +/- {summary['baseline_accuracy_std']:.4f}")
    print(f"  full-interv.  = {summary['full_intervention_accuracy_mean']:.4f} +/- {summary['full_intervention_accuracy_std']:.4f}")
    print(f"  gain          = {summary['gain_mean']:+.4f} +/- {summary['gain_std']:.4f}")

    sign = "+" if summary["gain_mean"] >= 0 else "-"
    print("\n=== LaTeX table row ===")
    print(f"All {n_concepts} concepts & "
          f"{summary['full_intervention_accuracy_mean']:.4f} $\\pm$ {summary['full_intervention_accuracy_std']:.4f} & "
          f"{sign}{abs(summary['gain_mean']):.4f} $\\pm$ {summary['gain_std']:.4f} \\\\")

    return {
        "per_seed": {str(seed): r for seed, r in per_seed.items()},
        "summary": summary,
    }


def main():
    results = {
        "hybrid": run_intervention(
            "CB-ShallowCNN (hybrid, residual=true)", "config/config_caueeg_dementia_concept_bottleneck.yaml",
        ),
        "plain": run_intervention(
            "CB-ShallowCNN-Plain (pure bottleneck, residual=false)",
            "config/config_caueeg_dementia_concept_bottleneck_plain.yaml",
        ),
    }
    with open("intervention_experiment_caueeg_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to intervention_experiment_caueeg_results.json")


if __name__ == "__main__":
    main()
