"""
Leakage analysis for CB-ShallowCNN and CB-GNN on TUAB (seed-42 checkpoints,
frozen -- no retraining of the backbone). Replaces the vague Limitations
claim ("logistic regression on true concepts achieves ~4% lower accuracy
than CB-GNN, indicating moderate leakage") with a proper decomposition.

For each backbone, extracts frozen (gated_concepts, residual) features for
every train/eval sample, then trains THREE linear probes (matching the
model's own nn.Linear(classifier_input, n_classes) classifier exactly, same
class-weighted cross-entropy loss) on:
  1. concept-only:  gated_concepts alone
  2. residual-only: residual alone
  3. combined:      [gated_concepts; residual] -- reproduces the model's own
                     reported accuracy exactly, serving as a sanity check on
                     the feature-extraction code.

If concept-only accuracy is close to combined, the classifier's decisions
are mostly explained by concepts (low leakage). If residual-only alone
already recovers most of combined's accuracy, most of the signal bypasses
the concept bottleneck (high leakage).
"""
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
from models.concept_bottleneck import (
    ConceptBottleneckShallowCNN, ConceptBottleneckGNN, normalize_concepts,
)
from train_utils import split_validation
from utils.metrics import compute_loso_metrics

PROBE_EPOCHS = 100
PROBE_LR = 1e-3
PROBE_WD = 1e-4


def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_features_shallow(model, loader, device, band_power_median, band_power_iqr):
    """Returns (gated_concepts, residual, labels) as (N, C), (N, R), (N,) tensors."""
    model.eval()
    all_gated, all_resid, all_labels = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            feat = model.get_backbone_features(x)  # dropout is off in eval mode
            concepts = model.concept_predictor(feat)
            gated = concepts * model.dead_mask
            resid = F.relu(model.residual_proj(feat))
            all_gated.append(gated.cpu())
            all_resid.append(resid.cpu())
            all_labels.extend(raw_batch["label"].tolist())
    return torch.cat(all_gated), torch.cat(all_resid), torch.tensor(all_labels)


def extract_features_gnn(model, loader, device, band_power_median, band_power_iqr):
    model.eval()
    all_gated, all_resid, all_labels = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            h = model.get_channel_features(x)
            for k in range(model.n_hops):
                A = model.per_sample_adjacency(h)
                h = h + model.hop_alphas[k] * torch.einsum("bij,bjf->bif", A, h)
                if k < model.n_hops - 1:
                    h = F.elu(h)
            h_flat = h.reshape(h.shape[0], -1)
            main_feat = model.channel_collapse(h_flat)
            main_feat = model.bn(main_feat)
            main_feat = F.relu(main_feat)
            # dropout is off in eval mode

            x_beta = None
            from models.concept_bottleneck import fft_bandpass, INVERSE_PERM
            x_beta = fft_bandpass(x, model.sfreq, *model.beta_band)
            xb = x_beta.unsqueeze(1)
            xb = model.beta_temporal_conv(xb)
            xb = model.beta_spatial_conv(xb)
            xb = model.beta_bn(xb)
            xb = xb ** 2
            xb = model.beta_pool(xb)
            xb = torch.log(torch.clamp(xb, min=1e-6))
            beta_feat = model.beta_global_pool(xb).flatten(1)

            main_concepts_raw = model.main_concept_head(main_feat)
            beta_concepts_raw = model.beta_concept_head(beta_feat)
            concepts_unordered = torch.cat([main_concepts_raw, beta_concepts_raw], dim=-1)
            concepts = torch.sigmoid(concepts_unordered[:, INVERSE_PERM.to(x.device)])
            gated = concepts * model.dead_mask
            resid = F.relu(model.residual_proj(main_feat))

            all_gated.append(gated.cpu())
            all_resid.append(resid.cpu())
            all_labels.extend(raw_batch["label"].tolist())
    return torch.cat(all_gated), torch.cat(all_resid), torch.tensor(all_labels)


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
    band_power_median = ckpt["band_power_median"].to(device)
    band_power_iqr = ckpt["band_power_iqr"].to(device)

    print("extracting frozen features...", flush=True)
    train_gated, train_resid, train_y = extract_fn(model, train_loader, device, band_power_median, band_power_iqr)
    val_gated, val_resid, val_y = extract_fn(model, val_loader, device, band_power_median, band_power_iqr)
    eval_gated, eval_resid, eval_y = extract_fn(model, eval_loader, device, band_power_median, band_power_iqr)

    class_counts = np.bincount(train_y.numpy(), minlength=m["n_classes"]).astype(np.float32)
    inv_freq = len(train_y) / (m["n_classes"] * np.maximum(class_counts, 1))
    class_weights = torch.tensor(inv_freq ** 1.5, dtype=torch.float32).to(device)

    results = {}
    for probe_name, train_feat, val_feat, eval_feat in [
        ("concept-only", train_gated, val_gated, eval_gated),
        ("residual-only", train_resid, val_resid, eval_resid),
        ("combined", torch.cat([train_gated, train_resid], dim=1),
                     torch.cat([val_gated, val_resid], dim=1),
                     torch.cat([eval_gated, eval_resid], dim=1)),
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
        "tuh_concept_bottleneck_best_model.pt", "config_tuh_concept_bottleneck.yaml",
        extract_features_shallow,
        dict(n_classes=2, n_filters=40, dropout=0.5, residual=True),
    )

    results["CB-GNN"] = run_for_backbone(
        "CB-GNN", ConceptBottleneckGNN,
        "tuh_concept_bottleneck_gnn_best_model.pt", "config_tuh_concept_bottleneck_gnn.yaml",
        extract_features_gnn,
        dict(n_classes=2, n_filters_time=40, n_filters_spat=40, n_hops=2, sfreq=100,
             beta_band=(13, 30), beta_filters=16, dropout=0.5, residual=True),
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
