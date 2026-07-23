import copy
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from data.caueeg_e2e_loader import CAUEEGEndToEndDataset, compute_eeg_channel_norm, normalize_eeg
from data.caueeg_concepts_loader import CAUEEGWithConceptsDataset, collate_caueeg_concepts
from models.concept_bottleneck import (
    CONCEPT_NAMES, ConceptBottleneckShallowCNN, compute_concept_norm, normalize_concepts,
)
from train_utils import tee_stdout_to_file
from utils.metrics import aggregate_predictions_by_subject, compute_loso_metrics


def compute_concept_norm_from_dataset(ds, indices):
    """Same as train_tuh_concept_bottleneck.py's version -- see that docstring."""
    raw = torch.stack([ds[i]["concepts_raw"] for i in indices])
    median, iqr = compute_concept_norm(raw, list(range(len(indices))))
    return median, iqr


def train_and_evaluate(cfg, device):
    """
    ConceptBottleneckShallowCNN on CAUEEG. Concepts are computed with
    REGIONS_CAUEEG/ASYM_PAIRS_CAUEEG (CAUEEGWithConceptsDataset), since
    CAUEEG's 19-channel ordering differs from both TUH/NMT's and ds004504's.

    Like the ds004504 scripts, NO concepts are hard-masked here
    (dead_concept_indices=[]): DEAD_CONCEPT_INDICES is an empirical finding
    from TUH's own population and isn't assumed to transfer without
    separately checking via check_concept_rank_correlation.py-style analysis
    against this dataset's own checkpoint.
    """
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    task = d["task"]

    train_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=task, cache_dir=d["cache_dir"], split="train",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    val_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=task, cache_dir=d["cache_dir"], split="val",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=task, cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"], eval_tta=d.get("eval_tta", True),
    )
    train_ds = CAUEEGWithConceptsDataset(train_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    val_ds = CAUEEGWithConceptsDataset(val_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    eval_ds = CAUEEGWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    print(f"task={task} train={len(train_ds)} val={len(val_ds)} eval(test)={len(eval_ds)}", flush=True)

    print("computing population concept normalization from train subjects...", flush=True)
    t0 = time.time()
    band_power_median, band_power_iqr = compute_concept_norm_from_dataset(train_ds, list(range(len(train_ds))))
    band_power_median, band_power_iqr = band_power_median.to(device), band_power_iqr.to(device)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("computing per-channel EEG normalization from train windows...", flush=True)
    t0 = time.time()
    eeg_mean, eeg_std = compute_eeg_channel_norm(train_ds, list(range(len(train_ds))))
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    model = ConceptBottleneckShallowCNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_filters=m.get("n_filters", 40),
        dropout=m.get("dropout", 0.5), residual=m.get("residual", True),
        dead_concept_indices=[],  # no dead concepts confirmed for this dataset yet.
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
    lambda_concept = t["lambda_concept"]
    ema_decay = t.get("val_ema_decay", 0.9)
    best_val_ema, best_val_acc, best_epoch, epochs_since_best = -1.0, -1.0, -1, 0
    best_state = None
    val_ema = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_caueeg_concepts)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_caueeg_concepts)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_caueeg_concepts)

    model.train()
    for epoch in range(t["epochs"]):
        epoch_t0 = time.time()
        epoch_cls = epoch_conc = 0.0
        train_correct = n_seen = 0
        for raw_batch in train_loader:
            x = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            labels = raw_batch["label"].to(device)
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), band_power_median, band_power_iqr,
            )

            logits, pred_concepts = model(x)
            l_cls = F.cross_entropy(logits, labels, weight=class_weights)
            l_conc = F.mse_loss(pred_concepts, true_concepts)
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
                x = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
                logits, _ = model(x)
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
        print(f"epoch {epoch+1:>3}/{t['epochs']} -- cls_loss {epoch_cls/n_seen:.4f} "
              f"conc_loss {epoch_conc/n_seen:.4f} train_acc {train_acc:.3f} val_acc {val_acc:.3f} "
              f"val_bal_acc {val_bal_acc:.3f} val_ema {val_ema:.3f} "
              f"(best_bal {best_val_acc:.3f} @ep{best_epoch+1}) "
              f"[{time.time()-epoch_t0:.1f}s]", flush=True)

        if patience is not None and epochs_since_best >= patience:
            print(f"early stopping (no val improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(), "best_epoch": int(best_epoch),
        "best_val_bal_acc": float(best_val_acc),
        "band_power_median": band_power_median.cpu(), "band_power_iqr": band_power_iqr.cpu(),
    }, f"caueeg_{task}_concept_bottleneck_best_model.pt")

    model.eval()
    all_probs, all_labels_raw, all_sids = [], [], []
    all_true_concepts, all_pred_concepts = [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            x = normalize_eeg(raw_batch["raw_eeg"].to(device), eeg_mean, eeg_std)
            true_concepts = normalize_concepts(
                raw_batch["concepts_raw"].to(device), band_power_median, band_power_iqr,
            )
            logits, pred_concepts = model(x)
            all_probs.extend(F.softmax(logits, dim=-1).cpu().tolist())
            all_labels_raw.extend(raw_batch["label"].tolist())
            all_sids.extend(raw_batch["subject_id"])
            all_true_concepts.append(true_concepts.cpu().numpy())
            all_pred_concepts.append(pred_concepts.cpu().numpy())

    # classification metrics use TTA-aggregated (one-per-subject) predictions; concept
    # R^2 stays PER-WINDOW below (n=len(eval_ds), not n_subjects) -- each window has its
    # own true concepts (different EEG segment), so averaging concept predictions across
    # a subject's windows the way TTA does for class probabilities isn't the same kind of
    # aggregation and isn't done here.
    all_preds, all_labels = aggregate_predictions_by_subject(all_probs, all_sids, all_labels_raw)
    metrics = compute_loso_metrics(all_preds, all_labels)
    true_c = np.concatenate(all_true_concepts)
    pred_c = np.concatenate(all_pred_concepts)

    print(f"\n=== CAUEEG Concept Bottleneck ShallowCNN Eval Results ({task}, "
          f"eval_tta={d.get('eval_tta', True)}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n=== Concept Prediction Quality (R^2 per concept, n={len(true_c)}) ===")
    r2s = []
    for i, name in enumerate(CONCEPT_NAMES):
        r2 = r2_score(true_c[:, i], pred_c[:, i])
        r2s.append(r2)
        print(f"  {name:<28} R2={r2:.3f}")
    print(f"  mean R^2 across all 28 concepts: {np.mean(r2s):.3f}")

    return {"metrics": metrics, "concept_r2": dict(zip(CONCEPT_NAMES, r2s))}


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_caueeg_abnormal_concept_bottleneck.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_stdout_to_file(f"caueeg_{cfg['data']['task']}_concept_bottleneck_run.log"):
        print(f"Using device: {device}, config: {config_path}")
        train_and_evaluate(cfg, device)


if __name__ == "__main__":
    main()
