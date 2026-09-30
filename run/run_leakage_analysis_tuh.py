
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from data.tuh_e2e_loader import TUHEndToEndDataset
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.concept_bottleneck import ConceptBottleneckShallowCNN, normalize_concepts
from train_utils import checkpoint_path, split_validation
from utils.metrics import compute_loso_metrics

PROBE_EPOCHS = 100
PROBE_LR = 1e-3
PROBE_WD = 1e-4


def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_features_shallow(model, loader, device, concept_median, concept_iqr):
    """Returns (concepts, residual, labels) as (N, C), (N, R), (N,) tensors."""
    model.eval()
    all_concepts, all_resid, all_labels = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            feat = model.get_backbone_features(x)  # dropout is off in eval mode
            concepts = model.concept_predictor(feat)
            resid = F.relu(model.residual_proj(feat))
            all_concepts.append(concepts.cpu())
            all_resid.append(resid.cpu())
            all_labels.extend(raw_batch["label"].tolist())
    return torch.cat(all_concepts), torch.cat(all_resid), torch.tensor(all_labels)


def train_probe(input_dim, n_classes, train_x, train_y, val_x, val_y, class_weights, device):
    probe = nn.Linear(input_dim, n_classes).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    best_val_bal_acc, best_state = -1.0, None
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        optim.zero_grad()
        logits = probe(train_x)
        loss = F.cross_entropy(logits, train_y, weight=class_weights)
        loss.backward()
        optim.step()

        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_x).argmax(dim=1).cpu().tolist()
        val_metrics = compute_loso_metrics(val_preds, val_y.cpu().tolist())
        val_bal_acc = (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = copy.deepcopy(probe.state_dict())

    probe.load_state_dict(best_state)
    return probe


def evaluate_probe(probe, x, y, device):
    probe.eval()
    with torch.no_grad():
        preds = probe(x.to(device)).argmax(dim=1).cpu().tolist()
    return compute_loso_metrics(preds, y.tolist())


def run_for_backbone(name, model_cls, ckpt_path, config_path, extract_fn, model_kwargs):
    print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
    set_all_seeds(42)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    device = torch.device("cpu")

    train_base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    train_ds = TUHWithConceptsDataset(train_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    eval_ds = TUHWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])

    all_indices = list(range(len(train_ds)))
    train_indices, val_indices = split_validation(train_ds, all_indices, t.get("val_frac", 0.2), t["seed"])

    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    val_loader = DataLoader(Subset(train_ds, val_indices), batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)

    model = model_cls(n_channels=d["n_channels"], **model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    concept_median = ckpt["concept_median"].to(device)
    concept_iqr = ckpt["concept_iqr"].to(device)

    print("extracting frozen features...", flush=True)
    train_concepts, train_resid, train_y = extract_fn(model, train_loader, device, concept_median, concept_iqr)
    val_concepts, val_resid, val_y = extract_fn(model, val_loader, device, concept_median, concept_iqr)
    eval_concepts, eval_resid, eval_y = extract_fn(model, eval_loader, device, concept_median, concept_iqr)

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
        metrics = evaluate_probe(probe, eval_feat, eval_y, device)
        results[probe_name] = metrics
        print(f"  {probe_name}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
              f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f}", flush=True)

    return results


def main():
    results = {}

    results["CB-ShallowCNN"] = run_for_backbone(
        "CB-ShallowCNN", ConceptBottleneckShallowCNN,
        checkpoint_path("tuh_concept_bottleneck_best_model.pt"), "config/config_tuh_concept_bottleneck.yaml",
        extract_features_shallow,
        dict(n_classes=2, n_filters=40, dropout=0.5, residual=True),
    )

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    for backbone, probes in results.items():
        print(f"\n{backbone}:")
        for probe_name, metrics in probes.items():
            print(f"  {probe_name:<15} acc={metrics['accuracy']:.4f}")

    import json
    with open("leakage_analysis_tuh_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to leakage_analysis_tuh_results.json")


if __name__ == "__main__":
    main()
