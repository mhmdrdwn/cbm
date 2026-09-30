"""
Frozen-backbone linear-probe leakage analysis for CAUEEG-Dementia, mirroring
run_leakage_analysis_tuh.py (see that file's docstring and
run/leakage_probe_common.py for the shared probe-training/aggregation
logic) -- run across ALL 3 seeds (42, 43, 44) and aggregated as mean +/- std
per probe type (concept-only / residual-only / combined).

CAUEEG-specific differences from the TUH version:
  - 3 classes (dementia/MCI/normal), not 2.
  - Train/val/eval splits come directly from CAUEEGEndToEndDataset (no
    split_validation() call needed -- unlike TUH, CAUEEG's train/val split
    is already fixed by the dataset itself).
  - The classifier uses an additional per-channel z-score (normalize_eeg,
    stats from compute_eeg_channel_norm over train subjects) on top of the
    fixed-divisor scaling TUH uses alone; these stats aren't saved in the
    checkpoint but are deterministic given the (seed-independent) train
    split, so recomputing them here reproduces training-time exactly.
  - eval_tta=true means eval can have multiple windows per subject. Probe
    TRAINING stays window-level (train/val each have exactly one window per
    subject per this project's convention -- see the data config's comment),
    but probe EVALUATION on the eval split uses subject-level TTA
    aggregation (evaluate_probe_tta), matching how the main classifier's own
    reported accuracy is computed -- a window-level probe accuracy would
    not be comparable to that number.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import json

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.caueeg_e2e_loader import CAUEEGEndToEndDataset, compute_eeg_channel_norm, normalize_eeg
from data.caueeg_concepts_loader import CAUEEGWithConceptsDataset, collate_caueeg_concepts
from models.concept_bottleneck import ConceptBottleneckShallowCNN
from run.leakage_probe_common import (
    set_all_seeds, train_probe, evaluate_probe, evaluate_probe_tta, aggregate_over_seeds,
)
from train_utils import checkpoint_path, seed_checkpoint_name, SEEDS

TASK = "dementia"


def extract_features_caueeg(model, loader, device, eeg_mean, eeg_std):
    """Returns (concepts, residual, labels, subject_ids)."""
    model.eval()
    all_concepts, all_resid, all_labels, all_sids = [], [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            feat = model.get_backbone_features(x)  # dropout is off in eval mode
            concepts = model.concept_predictor(feat)
            resid = F.relu(model.residual_proj(feat))
            all_concepts.append(concepts.cpu())
            all_resid.append(resid.cpu())
            all_labels.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])
    return torch.cat(all_concepts), torch.cat(all_resid), torch.tensor(all_labels), all_sids


def run_for_backbone_one_seed(name, ckpt_path, config_path, model_kwargs, seed):
    print(f"\n{'='*70}\n{name}  seed={seed}\n{'='*70}", flush=True)
    set_all_seeds(seed)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, d = cfg["model"], cfg["data"]
    device = torch.device("cpu")

    train_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=TASK, cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    val_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=TASK, cache_dir=d["cache_dir"], split="val",
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
    val_ds = CAUEEGWithConceptsDataset(val_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    eval_ds = CAUEEGWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])

    eeg_mean, eeg_std = compute_eeg_channel_norm(train_ds, list(range(len(train_ds))))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_caueeg_concepts)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_caueeg_concepts)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_caueeg_concepts)

    model = ConceptBottleneckShallowCNN(n_channels=d["n_channels"], **model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print("extracting frozen features...", flush=True)
    train_concepts, train_resid, train_y, _ = extract_features_caueeg(model, train_loader, device, eeg_mean, eeg_std)
    val_concepts, val_resid, val_y, _ = extract_features_caueeg(model, val_loader, device, eeg_mean, eeg_std)
    eval_concepts, eval_resid, eval_y, eval_sids = extract_features_caueeg(model, eval_loader, device, eeg_mean, eeg_std)

    class_counts = np.bincount(train_y.numpy(), minlength=m["n_classes"]).astype(np.float32)
    inv_freq = len(train_y) / (m["n_classes"] * np.maximum(class_counts, 1))
    class_weights = torch.tensor(inv_freq ** 1.5, dtype=torch.float32).to(device)

    results = {}
    for probe_name, train_feat, val_feat, eval_feat in [
        ("concept-only", train_concepts, val_concepts, eval_concepts),
        ("residual-only", train_resid, val_resid, eval_resid),
        ("combined", torch.cat([train_concepts, train_resid], dim=1),
                     torch.cat([val_concepts, val_resid], dim=1),
                     torch.cat([eval_concepts, eval_resid], dim=1)),
    ]:
        print(f"training {probe_name} probe (dim={train_feat.shape[1]})...", flush=True)
        probe = train_probe(
            train_feat.shape[1], m["n_classes"], train_feat, train_y, val_feat, val_y, class_weights, device,
        )
        metrics = evaluate_probe_tta(probe, eval_feat, eval_y, eval_sids, device)
        results[probe_name] = metrics
        print(f"  {probe_name}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
              f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f}", flush=True)

    return results


def run_for_backbone(name, checkpoint_name, config_path, model_kwargs):
    """Runs run_for_backbone_one_seed for every seed in SEEDS (each seed loads
    its own checkpoint via seed_checkpoint_name), then aggregates each probe's
    metrics as mean +/- std across seeds."""
    per_seed = {}
    for seed in SEEDS:
        ckpt_path = checkpoint_path(seed_checkpoint_name(checkpoint_name, seed))
        per_seed[seed] = run_for_backbone_one_seed(name, ckpt_path, config_path, model_kwargs, seed)

    summary = aggregate_over_seeds(per_seed, SEEDS)

    print(f"\n{'='*70}\n{name}: aggregate over seeds {SEEDS}\n{'='*70}")
    for probe_name in summary:
        s = summary[probe_name]
        print(f"  {probe_name:<15} acc={s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}")

    return {
        "per_seed": {str(seed): r for seed, r in per_seed.items()},
        "summary": summary,
    }


def main():
    results = {}

    results["CB-ShallowCNN"] = run_for_backbone(
        "CB-ShallowCNN", "caueeg_dementia_concept_bottleneck",
        "config/config_caueeg_dementia_concept_bottleneck.yaml",
        dict(n_classes=3, n_filters=40, dropout=0.5, residual=True),
    )

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    for backbone, r in results.items():
        print(f"\n{backbone}:")
        for probe_name, s in r["summary"].items():
            print(f"  {probe_name:<15} acc={s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}")

    with open("leakage_analysis_caueeg_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to leakage_analysis_caueeg_results.json")


if __name__ == "__main__":
    main()
