"""
Concept intervention experiment, run for BOTH CB-ShallowCNN variants:
the hybrid (residual=true, tuh_concept_bottleneck_best_model.pt) and the
Plain pure bottleneck (residual=false, tuh_concept_bottleneck_plain_best_model.pt).

For each "working" concept (R^2 >= 0.10 against its analytically computed
true value -- recomputed fresh here, not trusted from a stale log), the
model's own predicted value for that ONE concept is replaced with the true
value at inference time (models/concept_bottleneck.py's `intervention=`
forward argument) and the resulting TUAB test accuracy is reported. This
is the standard CBM intervention test (Koh et al. 2020): correcting a
concept prediction can only change the output because the classifier
consumes concepts, not raw backbone features, directly (see
ConceptBottleneckShallowCNN's docstring).

Single-concept correction alone is a weak test -- with every OTHER concept
(and, for the hybrid, the residual pathway) still carrying the model's own
(imperfect) belief, one corrected input rarely has enough leverage over a
linear classifier to flip a prediction, especially with correlated concepts
(e.g. band power across regions) making any one concept partly redundant.
So this script also runs a PROGRESSIVE multi-concept intervention: concepts
are ordered by their own R^2 (most reliable first) and corrected k at a
time simultaneously (k=1,5,10,15,all), which is the standard way CBM
papers actually test whether concepts matter collectively, not just
individually.

Running both variants side by side tests a specific prediction: the Plain
model has NO residual escape hatch, so a correct concept is the classifier's
ONLY new information -- intervention should move its accuracy at least as
much as the hybrid's, and if the hybrid's residual is absorbing concept
errors, the hybrid's intervention gains should be smaller.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import random

import numpy as np
import torch
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from data.tuh_e2e_loader import TUHEndToEndDataset
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneckShallowCNN, normalize_concepts
from train_utils import checkpoint_path
from utils.metrics import compute_loso_metrics

R2_WORKING_THRESHOLD = 0.10


def run_pass(model, loader, device, band_power_median, band_power_iqr, intervene_indices=None):
    """One eval pass. If intervene_indices is given (a list of concept indices),
    every batch has ALL of those concepts' predicted values simultaneously
    overridden with their own true (per-sample) values before the classifier
    runs. Returns (preds, labels, true_concepts, pred_concepts) -- the latter
    two only meaningful/used on the baseline (intervene_indices=None) pass."""
    model.eval()
    preds, labels, true_c, pred_c = [], [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), band_power_median, band_power_iqr,
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
    band_power_median = ckpt["band_power_median"].to(device)
    band_power_iqr = ckpt["band_power_iqr"].to(device)
    print(f"loaded checkpoint (best_epoch={ckpt['best_epoch']}, "
          f"best_val_bal_acc={ckpt['best_val_bal_acc']:.4f})", flush=True)

    # baseline pass (no intervention) -- also gives us fresh per-concept R^2 to
    # decide which concepts count as "working" (R^2 >= 0.10), same criterion as
    # section 4.3 of the paper, recomputed here rather than trusted from a log.
    base_preds, base_labels, true_c, pred_c = run_pass(
        model, eval_loader, device, band_power_median, band_power_iqr, intervene_indices=None,
    )
    base_metrics = compute_loso_metrics(base_preds, base_labels)
    base_acc = base_metrics["accuracy"]
    print(f"\n=== Baseline (no intervention) ===")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}")

    r2_by_concept = {name: r2_score(true_c[:, i], pred_c[:, i]) for i, name in enumerate(CONCEPT_NAMES)}
    working = [name for name, r2 in r2_by_concept.items() if r2 >= R2_WORKING_THRESHOLD]
    print(f"\n{len(working)} working concepts (R^2 >= {R2_WORKING_THRESHOLD}): {working}")

    print(f"\n=== Single-concept intervention (baseline acc={base_acc:.4f}) ===")
    print(f"{'concept':<28}{'post-intervention acc':<24}{'gain':<10}")
    single_results = []
    for name in working:
        idx = CONCEPT_NAMES.index(name)
        preds, labels, _, _ = run_pass(
            model, eval_loader, device, band_power_median, band_power_iqr, intervene_indices=[idx],
        )
        acc = compute_loso_metrics(preds, labels)["accuracy"]
        gain = acc - base_acc
        single_results.append((name, acc, gain))
        print(f"{name:<28}{acc:<24.4f}{gain:+.4f}")

    single_results.sort(key=lambda r: r[2], reverse=True)
    print("\n=== Sorted by gain (descending) ===")
    for name, acc, gain in single_results:
        print(f"{name:<28}{acc:<24.4f}{gain:+.4f}")

    print("\n=== LaTeX table rows (tab:intervention, single-concept) ===")
    for name, acc, gain in single_results:
        latex_name = name.replace("_", "\\_")
        sign = "+" if gain >= 0 else "-"
        print(f"{latex_name} & {acc:.4f} & {sign}{abs(gain):.4f} \\\\")

    # progressive multi-concept intervention: concepts ordered by their own R^2
    # (most reliable first), corrected k-at-a-time SIMULTANEOUSLY -- tests whether
    # concepts matter collectively, not just individually. k=len(working) is the
    # "full intervention" (every working concept corrected at once).
    order = sorted(working, key=lambda n: r2_by_concept[n], reverse=True)
    k_values = sorted(set([1, 5, 10, 15, len(order)]))
    k_values = [k for k in k_values if k <= len(order)]

    print(f"\n=== Progressive multi-concept intervention (baseline acc={base_acc:.4f}) ===")
    print(f"{'k':<6}{'concepts corrected':<24}{'post-intervention acc':<24}{'gain':<10}")
    progressive_results = []
    for k in k_values:
        names_k = order[:k]
        indices_k = [CONCEPT_NAMES.index(n) for n in names_k]
        preds, labels, _, _ = run_pass(
            model, eval_loader, device, band_power_median, band_power_iqr, intervene_indices=indices_k,
        )
        acc = compute_loso_metrics(preds, labels)["accuracy"]
        gain = acc - base_acc
        progressive_results.append((k, names_k, acc, gain))
        label = "all" if k == len(order) else str(k)
        print(f"{label:<6}{k:<24}{acc:<24.4f}{gain:+.4f}")

    full_k, full_names, full_acc, full_gain = progressive_results[-1]
    print(f"\nFull intervention (all {full_k} working concepts corrected at once): "
          f"acc={full_acc:.4f} gain={full_gain:+.4f}")

    print("\n=== LaTeX table rows (progressive intervention) ===")
    for k, names_k, acc, gain in progressive_results:
        label = "All working concepts" if k == len(order) else f"Top {k} (by $R^2$)"
        sign = "+" if gain >= 0 else "-"
        print(f"{label} & {acc:.4f} & {sign}{abs(gain):.4f} \\\\")


def main():
    run_intervention("CB-ShallowCNN (hybrid, residual=true)", "config/config_tuh_concept_bottleneck.yaml")
    run_intervention("CB-ShallowCNN-Plain (pure bottleneck, residual=false)", "config/config_tuh_concept_bottleneck_plain.yaml")


if __name__ == "__main__":
    main()
