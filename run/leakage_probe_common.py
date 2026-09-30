"""
Shared frozen-backbone linear-probe helpers for run_leakage_analysis_tuh.py
and run_leakage_analysis_caueeg.py: probe training/eval and the per-seed
mean +/- std aggregation, so both datasets' leakage scripts stay in sync
instead of maintaining two copies of the same probe logic.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import compute_loso_metrics

PROBE_EPOCHS = 100
PROBE_LR = 1e-3
PROBE_WD = 1e-4


def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_probe(input_dim, n_classes, train_x, train_y, val_x, val_y, class_weights, device,
                 batch_size=32):
    """
    Mini-batch training -- PROBE_EPOCHS real epochs, each a full pass over
    train_x in shuffled batch_size-sized batches (~len(train_x)/batch_size
    gradient steps per epoch), not one full-batch step per epoch. An earlier
    version did full-batch gradient descent here (one Adam step per "epoch",
    so PROBE_EPOCHS=100 meant only 100 total updates regardless of dataset
    size), which under-trained higher-dimensional probes (e.g. "combined",
    ~4x the parameters of "residual-only") more than lower-dimensional ones
    in the same fixed step budget, and was confirmed as the cause of a real,
    reproducible anomaly: full-batch training gave combined accuracy LOWER
    than residual-only alone (impossible in principle, since combined has
    strictly more information available -- it can always learn to ignore the
    extra input dimensions). Mini-batch training on the exact same extracted
    features fixed the ordering (combined > residual-only, as it must be).
    """
    probe = nn.Linear(input_dim, n_classes).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    n = train_x.shape[0]

    best_val_bal_acc, best_state = -1.0, None
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optim.zero_grad()
            loss = F.cross_entropy(probe(train_x[idx]), train_y[idx], weight=class_weights)
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
    """Window-level evaluation (no subject/TTA aggregation) -- used directly
    by TUH (one window per eval subject already) and by CAUEEG's train/val
    splits (also one window per subject there)."""
    probe.eval()
    with torch.no_grad():
        preds = probe(x.to(device)).argmax(dim=1).cpu().tolist()
    return compute_loso_metrics(preds, y.tolist())


def evaluate_probe_tta(probe, x, y, subject_ids, device):
    """Subject-level TTA evaluation for eval splits with multiple windows per
    subject (CAUEEG's eval_tta=true): averages per-window softmax probs by
    subject before arg-maxing, matching aggregate_predictions_by_subject's
    use elsewhere in this codebase for the main classifier's own reported
    accuracy -- a probe accuracy computed window-level would not be
    comparable to that number."""
    from utils.metrics import aggregate_predictions_by_subject

    probe.eval()
    with torch.no_grad():
        probs = F.softmax(probe(x.to(device)), dim=-1).cpu().tolist()
    preds, agg_labels = aggregate_predictions_by_subject(probs, subject_ids, y.tolist())
    return compute_loso_metrics(preds, agg_labels)


def aggregate_over_seeds(per_seed, seeds):
    """per_seed: {seed: {probe_name: metrics_dict}}. Returns {probe_name:
    {metric_mean, metric_std, ...}} aggregated across seeds."""
    probe_names = next(iter(per_seed.values())).keys()
    metric_names = next(iter(next(iter(per_seed.values())).values())).keys()
    summary = {}
    for probe_name in probe_names:
        summary[probe_name] = {}
        for metric_name in metric_names:
            vals = np.array([per_seed[seed][probe_name][metric_name] for seed in seeds])
            summary[probe_name][f"{metric_name}_mean"] = float(vals.mean())
            summary[probe_name][f"{metric_name}_std"] = float(vals.std())
    return summary
