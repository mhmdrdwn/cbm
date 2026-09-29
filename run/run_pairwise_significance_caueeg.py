"""
Pairwise statistical significance for the 3 headline CAUEEG-Dementia models
(ShallowCNN, CB-ShallowCNN (hybrid), CB-ShallowCNN-Plain (pure bottleneck,
residual=false)), using the EXISTING seed-42 checkpoints (no retraining --
inference-only, fast). Mirrors run_pairwise_significance_tuh.py -- see that
file's docstring for the full rationale, including why CB-ShallowCNN vs
CB-ShallowCNN-Plain is the key added comparison. McNemar's test and paired
bootstrap CI work identically here despite this being a 3-class task: both
operate on the binary correct/incorrect outcome per subject, not the
specific predicted class.

CAUEEG's official split (not seed-dependent) plus eval_tta multi-window
aggregation means per-subject predictions are directly comparable across
models as long as the same eval_ds construction is used for all of them
(aggregate_predictions_by_subject returns subjects in first-occurrence
order, which is identical across runs since the eval loader is
deterministic/shuffle=False).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils/train_utils

import numpy as np
import torch
import yaml
from scipy.stats import binomtest
from torch.utils.data import DataLoader

from data.caueeg_e2e_loader import (
    CAUEEGEndToEndDataset, collate_caueeg_e2e, compute_eeg_channel_norm, normalize_eeg,
)
from data.caueeg_concepts_loader import CAUEEGWithConceptsDataset, collate_caueeg_concepts
from models.shallow_cnn import ShallowConvNet
from models.concept_bottleneck import ConceptBottleneckShallowCNN, normalize_concepts
from train_utils import checkpoint_path
from utils.metrics import aggregate_predictions_by_subject

N_BOOTSTRAP = 10000
RNG_SEED = 0
TASK = "dementia"


def get_plain_predictions(model_cls, ckpt_path, config_path, **model_kwargs):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    train_ds = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=TASK, cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=TASK, cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"], eval_tta=d.get("eval_tta", True),
    )
    eeg_mean, eeg_std = compute_eeg_channel_norm(train_ds, list(range(len(train_ds))))
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_caueeg_e2e)
    device = torch.device("cpu")
    model = model_cls(n_channels=d["n_channels"], **model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    all_probs, all_labels_raw, all_sids = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            logits, _ = model(x)
            all_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            all_labels_raw.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])
    preds, labels = aggregate_predictions_by_subject(all_probs, all_sids, all_labels_raw)
    return np.array(preds), np.array(labels)


def get_cb_predictions(model_cls, ckpt_path, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, d = cfg["model"], cfg["data"]
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
    eeg_mean, eeg_std = compute_eeg_channel_norm(train_ds, list(range(len(train_ds))))
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_caueeg_concepts)
    device = torch.device("cpu")

    model = model_cls(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_filters=m.get("n_filters", 40),
        dropout=m.get("dropout", 0.5), residual=m.get("residual", True), dead_concept_indices=[],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    all_probs, all_labels_raw, all_sids = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            logits, _ = model(x)
            all_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            all_labels_raw.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])
    preds, labels = aggregate_predictions_by_subject(all_probs, all_sids, all_labels_raw)
    return np.array(preds), np.array(labels)


def mcnemar_test(correct_a, correct_b):
    n01 = int(np.sum((~correct_a) & correct_b))
    n10 = int(np.sum(correct_a & (~correct_b)))
    n_discordant = n01 + n10
    if n_discordant == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    p = binomtest(min(n01, n10), n_discordant, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_value": p}


def bootstrap_diff_ci(correct_a, correct_b, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_b[idx].mean() - correct_a[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(diffs.mean())


def bootstrap_acc_ci(correct, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(correct)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[i] = correct[idx].mean()
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    print("Loading predictions from existing seed-42 CAUEEG-Dementia checkpoints...", flush=True)

    preds_cnn, labels = get_plain_predictions(
        ShallowConvNet, checkpoint_path("caueeg_dementia_shallow_cnn_best_model.pt"),
        "config/config_caueeg_dementia_shallow_cnn.yaml", n_classes=3, dropout=0.5,
    )
    preds_cb_cnn, labels_c = get_cb_predictions(
        ConceptBottleneckShallowCNN, checkpoint_path("caueeg_dementia_concept_bottleneck_best_model.pt"),
        "config/config_caueeg_dementia_concept_bottleneck.yaml",
    )
    preds_cb_plain, labels_p = get_cb_predictions(
        ConceptBottleneckShallowCNN, checkpoint_path("caueeg_dementia_concept_bottleneck_plain_best_model.pt"),
        "config/config_caueeg_dementia_concept_bottleneck_plain.yaml",
    )

    assert np.array_equal(labels, labels_c) and np.array_equal(labels, labels_p), \
        "eval label order mismatch across models -- cannot compare per-sample"

    models = {
        "ShallowCNN": preds_cnn,
        "CB-ShallowCNN": preds_cb_cnn,
        "CB-ShallowCNN-Plain": preds_cb_plain,
    }
    correct = {name: (preds == labels) for name, preds in models.items()}

    print(f"\nn={len(labels)} subjects")
    print("\n=== Per-model accuracy + bootstrap 95% CI (n=10000 resamples) ===")
    for name, c in correct.items():
        acc = c.mean()
        lo, hi = bootstrap_acc_ci(c)
        print(f"  {name:<16} acc={acc:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    print("\n=== Pairwise McNemar's test + paired bootstrap CI on accuracy difference ===")
    names = list(models.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mc = mcnemar_test(correct[a], correct[b])
            lo, hi, mean_diff = bootstrap_diff_ci(correct[a], correct[b])
            sig = "*" if mc["p_value"] < 0.05 else " "
            print(f"  {b} - {a}: diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
                  f"McNemar p={mc['p_value']:.4f}{sig}  (n01={mc['n01']}, n10={mc['n10']})")

    print("\n(* = McNemar p < 0.05)")


if __name__ == "__main__":
    main()
