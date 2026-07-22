import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from data.electrode_positions import get_channel_positions
from data.nmt_e2e_loader import NMTEndToEndDataset, collate_e2e
from models.shallow_cnn import ShallowConvNet, ShallowCNNPhysicsLoss
from train_utils import split_validation
from utils.metrics import compute_loso_metrics


def train_and_evaluate(cfg, device):
    """
    ShallowConvNet on NMT (NMTEndToEndDataset: raw time series, CAR,
    skip_sec/max_sec windowing, fixed-scale normalization) -- this CNN
    beat every attention-based variant tried in this project on real NMT
    eval accuracy. Physics/anatomy-informed constraints
    (ShallowCNNPhysicsLoss) are added directly on top of this proven
    backbone: an electrode-coupling prior on the spatial filter's implied
    channel similarity (real 10-20 scalp distances), plus the same
    class-conditional Jansen-Rit excitatory-gain convention used
    elsewhere in this project -- no new attention/graph module, since
    every fully-learned cross-channel attention attempt so far has
    underperformed this CNN.
    """
    m, t, d = cfg["model"], cfg["training"], cfg["data"]

    train_ds = NMTEndToEndDataset(
        raw_dir=d["raw_dir"], cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = NMTEndToEndDataset(
        raw_dir=d["raw_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )

    all_indices = list(range(len(train_ds)))
    train_indices, val_indices = split_validation(
        train_ds, all_indices, t.get("val_frac", 0.2), t["seed"]
    )
    print(f"train={len(train_indices)} val={len(val_indices)} eval(test)={len(eval_ds)}", flush=True)

    model = ShallowConvNet(
        n_channels=d["n_channels"], n_classes=m["n_classes"], dropout=m.get("dropout", 0.5),
    ).to(device)

    channel_positions = get_channel_positions(d["channel_names"]).to(device)
    phys_loss_fn = ShallowCNNPhysicsLoss(
        channel_positions, lambda_coupling=t["lambda_coupling"], lambda_class=t["lambda_class"],
    ).to(device)
    # phys_loss_fn.channel_embed is a learnable nn.Parameter now, not a
    # fixed buffer -- must be included here or it silently never updates.
    optim = torch.optim.Adam(
        list(model.parameters()) + list(phys_loss_fn.parameters()),
        lr=t["lr"], weight_decay=t["weight_decay"],
    )

    train_labels = [train_ds.subjects[i]["label"] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=m["n_classes"]).astype(np.float32)
    class_weight_power = t.get("class_weight_power", 1.0)
    inv_freq = len(train_labels) / (m["n_classes"] * np.maximum(class_counts, 1))
    class_weights = torch.tensor(inv_freq ** class_weight_power, dtype=torch.float32).to(device)
    print(f"class weights (power={class_weight_power}): "
          f"{dict(zip(range(m['n_classes']), class_weights.tolist()))}", flush=True)

    batch_size = t.get("batch_size", 32)
    warmup_epochs = max(1, t.get("phys_warmup_epochs", 5))
    patience = t.get("early_stop_patience", 15)
    best_val_acc, best_epoch, epochs_since_best = -1.0, -1, 0
    best_state = None

    train_loader = DataLoader(
        Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_e2e,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_e2e,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_e2e)

    model.train()
    for epoch in range(t["epochs"]):
        phys_scale = min(1.0, (epoch + 1) / warmup_epochs)
        epoch_loss = epoch_cls_loss = epoch_phys_loss = 0.0
        train_correct = n_seen = 0
        epoch_t0 = time.time()
        for raw_batch in train_loader:
            raw_eeg = raw_batch["raw_eeg"].to(device)
            labels = raw_batch["label"].to(device)
            logits, A_pred = model(raw_eeg)
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
            phys_l = phys_loss_fn(model.spatial_coupling(), A_pred, labels)
            loss = cls_loss + phys_scale * phys_l

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()

            bsz = labels.shape[0]
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += bsz
            epoch_loss += loss.item() * bsz
            epoch_cls_loss += cls_loss.item() * bsz
            epoch_phys_loss += phys_l.item() * bsz

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
        print(f"epoch {epoch+1:>3}/{t['epochs']} -- avg loss {epoch_loss/n_seen:.4f} "
              f"(cls {epoch_cls_loss/n_seen:.4f}, phys {epoch_phys_loss/n_seen:.4f} x{phys_scale:.2f}) "
              f"train_acc {train_acc:.3f} val_acc {val_acc:.3f} val_bal_acc {val_bal_acc:.3f} "
              f"(best_bal {best_val_acc:.3f} @ep{best_epoch+1}) [{time.time()-epoch_t0:.1f}s]", flush=True)

        if patience is not None and epochs_since_best >= patience:
            print(f"early stopping (no val improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "phys_loss_state": phys_loss_fn.state_dict(),  # includes the learned channel_embed
        "best_epoch": int(best_epoch),
        "best_val_bal_acc": float(best_val_acc),
    }, "nmt_shallow_cnn_best_model.pt")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            raw_eeg = raw_batch["raw_eeg"].to(device)
            logits, _ = model(raw_eeg)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(raw_batch["label"].tolist())

    metrics = compute_loso_metrics(all_preds, all_labels)
    print("\n=== NMT Shallow-CNN + Physics Constraints Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def main():
    with open("config_nmt_shallow_cnn.yaml") as f:
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
