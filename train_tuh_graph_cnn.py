import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from data.tuh_e2e_loader import TUHEndToEndDataset, collate_tuh_e2e
from models.graph_coupling_cnn import ShallowGNN
from train_utils import split_validation
from utils.metrics import compute_loso_metrics


def train_and_evaluate(cfg, device):
    """
    ShallowGNN (models/graph_coupling_cnn.py) on TUH: multi-hop extension
    of a single-hop coupling design that previously reached 84.42%
    accuracy, roughly matching the 85.14% plain-CNN baseline. Same data
    pipeline/cache, plain classification loss -- every hop_alpha gets its
    only gradient from the classification objective itself.

    Compare against: plain CNN 85.14%, single-hop coupling CNN 84.42%.
    If more hops helps, this should beat 84.42%; if it doesn't, or gets
    worse, that's evidence a single hop of learned-feature coupling
    already captures what's useful here and added depth just adds
    overfitting risk (same failure mode this project's seen with every
    "more parameters" attempt).
    """
    m, t, d = cfg["model"], cfg["training"], cfg["data"]

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
    train_indices, val_indices = split_validation(
        train_ds, all_indices, t.get("val_frac", 0.2), t["seed"]
    )
    print(f"train={len(train_indices)} val={len(val_indices)} eval(test)={len(eval_ds)}", flush=True)

    model = ShallowGNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_hops=m.get("n_hops", 2),
        dropout=m.get("dropout", 0.5),
    ).to(device)

    # hop_alphas excluded from weight_decay -- uniform L2 shrinkage
    # would pull every hop_alpha back toward 0 each step regardless of
    # whether the classification loss wants a given hop to matter, making
    # it impossible to tell "this hop doesn't help" from "weight_decay
    # never let it find out."
    hop_alpha_params = [p for n, p in model.named_parameters() if n.startswith("hop_alphas.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("hop_alphas.")]
    assert len(hop_alpha_params) == m.get("n_hops", 2), \
        f"expected {m.get('n_hops', 2)} hop_alpha params, found {len(hop_alpha_params)}"
    optim = torch.optim.Adam([
        {"params": other_params, "weight_decay": t["weight_decay"]},
        {"params": hop_alpha_params, "weight_decay": 0.0},
    ], lr=t["lr"])

    train_labels = [train_ds.subjects[i]["label"] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=m["n_classes"]).astype(np.float32)
    class_weight_power = t.get("class_weight_power", 1.0)
    inv_freq = len(train_labels) / (m["n_classes"] * np.maximum(class_counts, 1))
    class_weights = torch.tensor(inv_freq ** class_weight_power, dtype=torch.float32).to(device)
    print(f"class weights (power={class_weight_power}): "
          f"{dict(zip(range(m['n_classes']), class_weights.tolist()))}", flush=True)

    batch_size = t.get("batch_size", 32)
    patience = t.get("early_stop_patience", 15)
    best_val_acc, best_epoch, epochs_since_best = -1.0, -1, 0
    best_state = None

    train_loader = DataLoader(
        Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_tuh_e2e,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_tuh_e2e,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_tuh_e2e)

    model.train()
    for epoch in range(t["epochs"]):
        epoch_loss = 0.0
        train_correct = n_seen = 0
        epoch_t0 = time.time()
        for raw_batch in train_loader:
            raw_eeg = raw_batch["raw_eeg"].to(device)
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
                raw_eeg = raw_batch["raw_eeg"].to(device)
                logits, _ = model(raw_eeg)
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels.extend(raw_batch["label"].tolist())
        val_metrics = compute_loso_metrics(val_preds, val_labels)
        val_acc = val_metrics["accuracy"]
        val_bal_acc = (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2
        model.train()

        if val_bal_acc > best_val_acc:
            best_val_acc, best_epoch, epochs_since_best = val_bal_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_best += 1

        train_acc = train_correct / n_seen
        with torch.no_grad():
            alphas_str = ",".join(f"{a.item():.4f}" for a in model.hop_alphas)
        print(f"epoch {epoch+1:>3}/{t['epochs']} -- loss {epoch_loss/n_seen:.4f} "
              f"hop_alphas [{alphas_str}] "
              f"train_acc {train_acc:.3f} val_acc {val_acc:.3f} val_bal_acc {val_bal_acc:.3f} "
              f"(best_bal {best_val_acc:.3f} @ep{best_epoch+1}) [{time.time()-epoch_t0:.1f}s]", flush=True)

        if patience is not None and epochs_since_best >= patience:
            print(f"early stopping (no val improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "best_epoch": int(best_epoch),
        "best_val_bal_acc": float(best_val_acc),
    }, "tuh_graph_cnn_best_model.pt")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            raw_eeg = raw_batch["raw_eeg"].to(device)
            logits, _ = model(raw_eeg)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(raw_batch["label"].tolist())

    metrics = compute_loso_metrics(all_preds, all_labels)
    print("\n=== TUH ShallowGNN (multi-hop coupling) Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    with torch.no_grad():
        print(f"  final hop_alphas: {[round(a.item(), 4) for a in model.hop_alphas]}")
    print("  (single-hop coupling CNN baseline: accuracy=0.8442; plain CNN baseline: accuracy=0.8514; "
          "this raw-cosine multi-hop config previously reached accuracy=0.8370)")
    return metrics


def main():
    with open("config_tuh_graph_cnn.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_and_evaluate(cfg, device)


if __name__ == "__main__":
    main()
