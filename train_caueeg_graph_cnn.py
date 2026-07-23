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
from models.graph_coupling_cnn import ShallowGNN
from train_utils import tee_stdout_to_file
from utils.metrics import aggregate_predictions_by_subject, compute_loso_metrics


def train_and_evaluate(cfg, device):
    """
    ShallowGNN (models/graph_coupling_cnn.py) on CAUEEG -- same multi-hop
    graph-coupling backbone as train_tuh_graph_cnn.py. cfg["data"]["task"]
    selects "abnormal" (binary) or "dementia" (3-class); see
    train_caueeg_shallow_cnn.py's docstring for the shared CAUEEG-specific
    details (official splits, event-marker-based segment selection).
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

    model = ShallowGNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_hops=m.get("n_hops", 2),
        dropout=m.get("dropout", 0.5),
    ).to(device)

    hop_alpha_params = [p for n, p in model.named_parameters() if n.startswith("hop_alphas.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("hop_alphas.")]
    assert len(hop_alpha_params) == m.get("n_hops", 2), \
        f"expected {m.get('n_hops', 2)} hop_alpha params, found {len(hop_alpha_params)}"
    optim = torch.optim.Adam([
        {"params": other_params, "weight_decay": t["weight_decay"]},
        {"params": hop_alpha_params, "weight_decay": 0.0},
    ], lr=t["lr"])

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_caueeg_e2e)
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
        with torch.no_grad():
            alphas_str = ",".join(f"{a.item():.4f}" for a in model.hop_alphas)
        print(f"epoch {epoch+1:>3}/{t['epochs']} -- loss {epoch_loss/n_seen:.4f} "
              f"hop_alphas [{alphas_str}] "
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
    }, f"caueeg_{task}_graph_cnn_best_model.pt")

    model.eval()
    all_probs, all_labels_raw, all_sids = [], [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            raw_eeg = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            logits, _ = model(raw_eeg)
            all_probs.extend(F.softmax(logits, dim=-1).cpu().tolist())
            all_labels_raw.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])

    all_preds, all_labels = aggregate_predictions_by_subject(all_probs, all_sids, all_labels_raw)

    metrics = compute_loso_metrics(all_preds, all_labels)
    print(f"\n=== CAUEEG ShallowGNN ({task}, multi-hop coupling, eval_tta={d.get('eval_tta', True)}) "
          f"Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    with torch.no_grad():
        print(f"  final hop_alphas: {[round(a.item(), 4) for a in model.hop_alphas]}")
    return metrics


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_caueeg_abnormal_graph_cnn.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_stdout_to_file(f"caueeg_{cfg['data']['task']}_graph_cnn_run.log"):
        print(f"Using device: {device}, config: {config_path}")
        train_and_evaluate(cfg, device)


if __name__ == "__main__":
    main()
