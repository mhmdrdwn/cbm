import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

from data.tuh_e2e_loader import TUHEndToEndDataset
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.concept_bottleneck import (
    CONCEPT_NAMES, DEAD_CONCEPT_INDICES, ConceptBottleneckGNN, compute_concept_norm, normalize_concepts,
)
from train_utils import split_validation
from utils.metrics import compute_loso_metrics


def compute_concept_norm_from_dataset(ds, indices):
    """Same as train_tuh_concept_bottleneck.py's version -- see that docstring."""
    raw = torch.stack([ds[i]["concepts_raw"] for i in indices])
    median, iqr = compute_concept_norm(raw, list(range(len(indices))))
    return median, iqr


def train_and_evaluate(cfg, device):
    """
    ConceptBottleneckGNN on TUH: same joint-training setup as
    train_tuh_concept_bottleneck.py, swapping in the graph-coupling main
    backbone + dedicated beta branch (models/concept_bottleneck.py's
    ConceptBottleneckGNN docstring). Compare against the plain version's
    real result -- accuracy=0.8188, mean concept R^2=-0.142, with every
    beta concept negative (worst -2.344) and all 3 asymmetry concepts
    negative -- to see whether either targeted fix actually moves those
    specific numbers, not just overall accuracy.
    """
    m, t, d = cfg["model"], cfg["training"], cfg["data"]

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
    print(f"train={len(train_indices)} val={len(val_indices)} eval(test)={len(eval_ds)}", flush=True)

    print("computing population concept normalization from train subjects...", flush=True)
    t0 = time.time()
    band_power_median, band_power_iqr = compute_concept_norm_from_dataset(train_ds, train_indices)
    band_power_median, band_power_iqr = band_power_median.to(device), band_power_iqr.to(device)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    model = ConceptBottleneckGNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"],
        n_filters_time=m.get("n_filters_time", 40), n_filters_spat=m.get("n_filters_spat", 40),
        n_hops=m.get("n_hops", 2), sfreq=d["sfreq"], beta_band=tuple(m.get("beta_band", (13, 30))),
        beta_filters=m.get("beta_filters", 16), dropout=m.get("dropout", 0.5),
        residual=m.get("residual", True),
    ).to(device)

    # hop_alphas excluded from weight_decay -- same reasoning as every other
    # gate/gain scalar in this project (config_tuh_graph_cnn.yaml's precedent):
    # uniform L2 shrinkage would pull these back toward their neutral starting
    # point regardless of whether the classification loss wants them to move,
    # making it impossible to tell "this doesn't help" from "weight_decay
    # never let it find out."
    no_decay_prefixes = ("hop_alphas.",)
    no_decay_params = [p for n, p in model.named_parameters() if n.startswith(no_decay_prefixes)]
    other_params = [p for n, p in model.named_parameters() if not n.startswith(no_decay_prefixes)]
    assert len(no_decay_params) == m.get("n_hops", 2)  # hop_alphas (n_hops)
    optim = torch.optim.Adam([
        {"params": other_params, "weight_decay": t["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
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
    lambda_concept = t["lambda_concept"]
    ema_decay = t.get("val_ema_decay", 0.9)
    best_val_ema, best_val_acc, best_epoch, epochs_since_best = -1.0, -1.0, -1, 0
    best_state = None
    val_ema = None

    # concept loss excludes the 4 confirmed-dead concepts (see DEAD_CONCEPT_INDICES's
    # docstring) -- no point spending gradient fitting pure noise; the model's classifier
    # never sees them either (models/concept_bottleneck.py's dead_mask handles that side).
    live_concept_idx = torch.tensor(
        [i for i in range(len(CONCEPT_NAMES)) if i not in DEAD_CONCEPT_INDICES], device=device,
    )
    print(f"excluding {len(DEAD_CONCEPT_INDICES)} confirmed-dead concepts from the concept loss: "
          f"{[CONCEPT_NAMES[i] for i in DEAD_CONCEPT_INDICES]}", flush=True)

    train_loader = DataLoader(
        Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_tuh_concepts,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_tuh_concepts,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_tuh_concepts)

    model.train()
    for epoch in range(t["epochs"]):
        epoch_t0 = time.time()
        epoch_cls = epoch_conc = 0.0
        train_correct = n_seen = 0
        for raw_batch in train_loader:
            x = raw_batch["raw_eeg"].to(device)
            labels = raw_batch["label"].to(device)
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), band_power_median, band_power_iqr,
            )

            logits, pred_concepts = model(x)
            l_cls = F.cross_entropy(logits, labels, weight=class_weights)
            l_conc = F.mse_loss(pred_concepts[:, live_concept_idx], true_concepts[:, live_concept_idx])
            loss = l_cls + lambda_concept * l_conc

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()

            bsz = labels.shape[0]
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += bsz
            epoch_cls += l_cls.item() * bsz
            epoch_conc += l_conc.item() * bsz

        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for raw_batch in val_loader:
                x = raw_batch["raw_eeg"].to(device)
                logits, _ = model(x)
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels.extend(raw_batch["label"].tolist())
        val_metrics = compute_loso_metrics(val_preds, val_labels)
        val_acc = val_metrics["accuracy"]
        val_bal_acc = (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2
        model.train()

        # best-checkpoint tracking selects by an EMA-SMOOTHED val_bal_acc, not the raw
        # per-epoch value and not train accuracy. Raw val_bal_acc is noisy (543-subject
        # val set) -- a single lucky spike could otherwise get locked in as "best". A
        # fixed warmup period (tried first) fixes this too, but has an arbitrary
        # all-or-nothing cutoff: a genuinely good epoch just under the cutoff gets
        # excluded, while a noisy one just past it gets accepted. EMA is smoother --
        # it takes several consecutive good epochs to move, regardless of when they
        # happen, without needing to guess the right cutoff length. Not train accuracy:
        # train_acc climbs steadily all training long in real runs here, so rewarding
        # it directly would bias selection toward later, more-overfit epochs.
        val_ema = val_bal_acc if val_ema is None else ema_decay * val_ema + (1 - ema_decay) * val_bal_acc
        if val_ema > best_val_ema:
            best_val_ema, best_val_acc, best_epoch, epochs_since_best = val_ema, val_bal_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_best += 1

        train_acc = train_correct / n_seen
        with torch.no_grad():
            hop_str = ",".join(f"{a.item():.4f}" for a in model.hop_alphas)
        print(f"epoch {epoch+1:>3}/{t['epochs']} -- cls_loss {epoch_cls/n_seen:.4f} "
              f"conc_loss {epoch_conc/n_seen:.4f} hop_alphas [{hop_str}] train_acc {train_acc:.3f} "
              f"val_acc {val_acc:.3f} val_bal_acc {val_bal_acc:.3f} val_ema {val_ema:.3f} "
              f"(best_bal {best_val_acc:.3f} @ep{best_epoch+1}) [{time.time()-epoch_t0:.1f}s]", flush=True)

        if patience is not None and epochs_since_best >= patience:
            print(f"early stopping (no val improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(), "best_epoch": int(best_epoch),
        "best_val_bal_acc": float(best_val_acc),
        "band_power_median": band_power_median.cpu(), "band_power_iqr": band_power_iqr.cpu(),
    }, "tuh_concept_bottleneck_gnn_best_model.pt")

    model.eval()
    all_preds, all_labels = [], []
    all_true_concepts, all_pred_concepts = [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            x = raw_batch["raw_eeg"].to(device)
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), band_power_median, band_power_iqr,
            )
            logits, pred_concepts = model(x)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(raw_batch["label"].tolist())
            all_true_concepts.append(true_concepts.cpu().numpy())
            all_pred_concepts.append(pred_concepts.cpu().numpy())

    metrics = compute_loso_metrics(all_preds, all_labels)
    true_c = np.concatenate(all_true_concepts)
    pred_c = np.concatenate(all_pred_concepts)

    print("\n=== TUH Concept Bottleneck GNN Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("  (plain CNN baseline: accuracy=0.8514; plain concept bottleneck: accuracy=0.8188)")

    print(f"\n=== Concept Prediction Quality (R^2 per concept, n={len(true_c)}) ===")
    r2s = []
    for i, name in enumerate(CONCEPT_NAMES):
        r2 = r2_score(true_c[:, i], pred_c[:, i])
        r2s.append(r2)
        marker = " <- beta" if name.endswith("_beta") else (" <- asym" if "asym" in name else "")
        print(f"  {name:<28} R2={r2:.3f}{marker}")
    print(f"  mean R^2 across all 28 concepts: {np.mean(r2s):.3f} "
          f"(plain concept bottleneck: -0.142)")

    return {"metrics": metrics, "concept_r2": dict(zip(CONCEPT_NAMES, r2s))}


def main():
    with open("config_tuh_concept_bottleneck_gnn.yaml") as f:
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
