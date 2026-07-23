import copy
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.caueeg_e2e_loader import (
    CAUEEGEndToEndDataset, collate_caueeg_e2e, compute_eeg_channel_norm, normalize_eeg,
)
from models.shallow_cnn import ShallowConvNet
from train_utils import tee_stdout_to_file
from utils.metrics import aggregate_predictions_by_subject, compute_loso_metrics


def train_and_evaluate(cfg, device):
    """
    Plain ShallowConvNet on CAUEEG -- no physics loss, same "plain, best-
    performing configuration" choice as train_tuh_shallow_cnn.py. cfg["data"]
    ["task"] selects which of CAUEEG's two ready-made benchmarks ("abnormal"
    binary or "dementia" 3-class) this run targets; ShallowConvNet's
    classifier head, F.cross_entropy, and compute_loso_metrics all generalize
    to n_classes directly, same as the ds004504 3-class scripts did.

    Unlike TUH/NMT/ds004504, CAUEEG ships its own official train/val/test
    split per task (no split_validation train-pool carve-out needed here) --
    see CAUEEGEndToEndDataset's docstring for why (event-marker-based eyes-
    closed segment selection, not simple truncation, is the other real
    difference from those loaders).
    """
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    task = d["task"]

    train_ds = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=task, cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    val_ds = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=task, cache_dir=d["cache_dir"], split="val",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=task, cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"], eval_tta=d.get("eval_tta", True),
    )
    print(f"task={task} train={len(train_ds)} val={len(val_ds)} eval(test)={len(eval_ds)}", flush=True)

    print("computing per-channel EEG normalization from train windows...", flush=True)
    t0 = time.time()
    eeg_mean, eeg_std = compute_eeg_channel_norm(train_ds, list(range(len(train_ds))))
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    model = ShallowConvNet(
        n_channels=d["n_channels"], n_classes=m["n_classes"], dropout=m.get("dropout", 0.5),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    train_labels = [s["label"] for s in train_ds.subjects]
    class_counts = np.bincount(train_labels, minlength=m["n_classes"]).astype(np.float32)
    class_weight_power = t.get("class_weight_power", 1.0)
    inv_freq = len(train_labels) / (m["n_classes"] * np.maximum(class_counts, 1))
    class_weights = torch.tensor(inv_freq ** class_weight_power, dtype=torch.float32).to(device)
    print(f"class weights (power={class_weight_power}): "
          f"{dict(zip(range(m['n_classes']), class_weights.tolist()))}", flush=True)

    batch_size = t.get("batch_size", 32)
    patience = t.get("early_stop_patience", 15)
    ema_decay = t.get("val_ema_decay", 0.9)
    best_val_ema, best_val_acc, best_epoch, epochs_since_best = -1.0, -1.0, -1, 0
    best_state = None
    val_ema = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_caueeg_e2e,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_caueeg_e2e)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_caueeg_e2e)

    model.train()
    for epoch in range(t["epochs"]):
        epoch_loss = 0.0
        train_correct = n_seen = 0
        epoch_t0 = time.time()
        for raw_batch in train_loader:
            raw_eeg = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            labels = raw_batch["label"].to(device)
            logits, _ = model(raw_eeg)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()

            bsz = labels.shape[0]
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += bsz
            epoch_loss += loss.item() * bsz

        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for raw_batch in val_loader:
                raw_eeg = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
                logits, _ = model(raw_eeg)
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels.extend(raw_batch["label"].tolist())
        val_metrics = compute_loso_metrics(val_preds, val_labels)
        val_acc = val_metrics["accuracy"]
        val_bal_acc = (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2
        model.train()

        val_ema = val_bal_acc if val_ema is None else ema_decay * val_ema + (1 - ema_decay) * val_bal_acc
        if val_ema > best_val_ema:
            best_val_ema, best_val_acc, best_epoch, epochs_since_best = val_ema, val_bal_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_best += 1

        train_acc = train_correct / n_seen
        print(f"epoch {epoch+1:>3}/{t['epochs']} -- loss {epoch_loss/n_seen:.4f} "
              f"train_acc {train_acc:.3f} val_acc {val_acc:.3f} val_bal_acc {val_bal_acc:.3f} "
              f"val_ema {val_ema:.3f} (best_bal {best_val_acc:.3f} @ep{best_epoch+1}) "
              f"[{time.time()-epoch_t0:.1f}s]", flush=True)

        if patience is not None and epochs_since_best >= patience:
            print(f"early stopping (no val improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "best_epoch": int(best_epoch),
        "best_val_bal_acc": float(best_val_acc),
    }, f"caueeg_{task}_shallow_cnn_best_model.pt")

    model.eval()
    all_probs, all_labels_raw, all_sids = [], [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            raw_eeg = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            logits, _ = model(raw_eeg)
            all_probs.extend(F.softmax(logits, dim=-1).cpu().tolist())
            all_labels_raw.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])

    # TTA aggregation: if eval_tta is on, eval_loader yields multiple windows per
    # subject -- average their softmax scores back to one prediction per subject
    # before scoring (see aggregate_predictions_by_subject's docstring). A no-op
    # when eval_tta=False (each subject already has exactly one window).
    all_preds, all_labels = aggregate_predictions_by_subject(all_probs, all_sids, all_labels_raw)

    metrics = compute_loso_metrics(all_preds, all_labels)
    print(f"\n=== CAUEEG Shallow-CNN ({task}, no physics, eval_tta={d.get('eval_tta', True)}) Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def main():
    # one script, two configs (abnormal/dementia differ in task/n_classes/checkpoint
    # name) -- config path taken from argv rather than hardcoded, unlike this
    # project's other train_*.py scripts, since there's no separate script per task.
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_caueeg_abnormal_shallow_cnn.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_stdout_to_file(f"caueeg_{cfg['data']['task']}_shallow_cnn_run.log"):
        print(f"Using device: {device}, config: {config_path}")
        train_and_evaluate(cfg, device)


if __name__ == "__main__":
    main()
