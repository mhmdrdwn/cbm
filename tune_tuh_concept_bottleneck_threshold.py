"""
Decision-threshold tuning for the best ConceptBottleneckGNN checkpoint
(tuh_concept_bottleneck_gnn_best_model.pt, 83.33% accuracy at the default
argmax/0.5 threshold, sensitivity=0.778 vs specificity=0.880). Every
result in this project so far has used the implicit 0.5 threshold; this
is the first time thresholds get tuned at all.

Threshold is selected using the VALIDATION set (the same split
split_validation produced during training, reconstructed here with the
same seed) and only then applied ONCE to the held-out eval set -- picking
a threshold based on eval-set performance and then reporting eval-set
performance at that threshold would be a real form of leakage (the same
issue flagged for concept pruning). This keeps the eval set an honest,
untouched final read, same discipline as every other result reported in
this project.

Label convention (0=abnormal/pathology, 1=normal, this project's
convention throughout): P(abnormal) = P(label=0) is the score sensitivity
is computed against, NOT P(label=1) -- getting this backwards (as a
pasted proposal earlier did) would silently tune the threshold for the
wrong class.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Subset

from data.tuh_e2e_loader import TUHEndToEndDataset, collate_tuh_e2e
from models.concept_bottleneck import ConceptBottleneckGNN
from train_utils import split_validation

TARGET_SENSITIVITIES = [0.80, 0.85, 0.90, 0.95]


def get_probs_and_labels(model, ds, device):
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_e2e)
    model.eval()
    probs_abnormal, labels = [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            logits, _ = model(x)
            p_abnormal = F.softmax(logits, dim=-1)[:, 0]  # P(label=0="abnormal")
            probs_abnormal.extend(p_abnormal.cpu().tolist())
            labels.extend(raw_batch["label"].tolist())
    return np.array(probs_abnormal), np.array(labels)


def metrics_at_threshold(probs_abnormal, labels, threshold):
    """Predict abnormal (0) if P(abnormal) >= threshold, else normal (1)."""
    preds = np.where(probs_abnormal >= threshold, 0, 1)
    tp = int(((preds == 0) & (labels == 0)).sum())  # correctly predicted abnormal
    fn = int(((preds == 1) & (labels == 0)).sum())  # abnormal predicted as normal
    tn = int(((preds == 1) & (labels == 1)).sum())  # correctly predicted normal
    fp = int(((preds == 0) & (labels == 1)).sum())  # normal predicted as abnormal
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    accuracy = (preds == labels).mean()
    bal_acc = (sensitivity + specificity) / 2
    return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "balanced_accuracy": bal_acc}


def main():
    with open("config_tuh_concept_bottleneck_gnn.yaml") as f:
        cfg = yaml.safe_load(f)
    m, t, d = cfg["model"], cfg["training"], cfg["data"]

    seed = t["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    train_ds = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    all_indices = list(range(len(train_ds)))
    train_indices, val_indices = split_validation(train_ds, all_indices, t.get("val_frac", 0.2), seed)
    val_ds = Subset(train_ds, val_indices)
    print(f"val={len(val_ds)} eval(test)={len(eval_ds)}", flush=True)

    model = ConceptBottleneckGNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"],
        n_filters_time=m.get("n_filters_time", 40), n_filters_spat=m.get("n_filters_spat", 40),
        n_hops=m.get("n_hops", 2), sfreq=d["sfreq"], beta_band=tuple(m.get("beta_band", (13, 30))),
        beta_filters=m.get("beta_filters", 16), dropout=m.get("dropout", 0.5),
        residual=m.get("residual", True),
    ).to(device)
    ckpt = torch.load("tuh_concept_bottleneck_gnn_best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded checkpoint (best_epoch={ckpt['best_epoch']})", flush=True)

    val_probs, val_labels = get_probs_and_labels(model, val_ds, device)
    eval_probs, eval_labels = get_probs_and_labels(model, eval_ds, device)

    # default (argmax / 0.5) threshold, for direct comparison
    default = metrics_at_threshold(eval_probs, eval_labels, 0.5)
    print("\n=== Default threshold (0.5) on eval set ===")
    for k, v in default.items():
        print(f"  {k}: {v:.4f}")

    # ROC on VAL set, pos_label=0 ("abnormal") -- this project's label convention
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs, pos_label=0)

    print(f"\n{'target_sens':<14}{'val_thresh':<12}{'eval_acc':<10}{'eval_sens':<11}{'eval_spec':<11}{'eval_bal_acc':<13}")
    for target in TARGET_SENSITIVITIES:
        valid = tpr >= target
        if not valid.any():
            print(f"{target:<14} -- no threshold on val reaches this sensitivity --")
            continue
        # among thresholds reaching >= target sensitivity on val, take the one with
        # the HIGHEST threshold (best specificity) that still clears the bar.
        idx = np.where(valid)[0][np.argmax(thresholds[valid])]
        chosen_thresh = thresholds[idx]
        val_sens_at_thresh, val_spec_at_thresh = tpr[idx], 1 - fpr[idx]

        eval_metrics = metrics_at_threshold(eval_probs, eval_labels, chosen_thresh)
        print(f"{target:<14.2f}{chosen_thresh:<12.4f}{eval_metrics['accuracy']:<10.4f}"
              f"{eval_metrics['sensitivity']:<11.4f}{eval_metrics['specificity']:<11.4f}"
              f"{eval_metrics['balanced_accuracy']:<13.4f}")
        print(f"    (val sensitivity at this threshold: {val_sens_at_thresh:.4f}, "
              f"val specificity: {val_spec_at_thresh:.4f})")

    print("\nInterpretation: each row picks the threshold on the VAL set that first reaches the "
          "target sensitivity, then reports what that SAME threshold achieves on the untouched "
          "eval set. If eval sensitivity tracks the target reasonably well, the threshold "
          "generalizes; a big gap would mean the val-selected threshold overfits val specifically.")


if __name__ == "__main__":
    main()
