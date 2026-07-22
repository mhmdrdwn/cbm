"""
Recalibrates the concept predictor's outputs using isotonic regression --
motivated by check_concept_rank_correlation.py's finding that several
concepts with poor/negative R^2 (delta_alpha_ratio, dtabr,
alpha_peak_amplitude) actually have strong, significant Spearman rank
correlation (up to 0.55), meaning the model reliably orders subjects
correctly but isn't predicting the right absolute scale. Isotonic
regression fits a monotonic (rank-preserving) mapping from predicted to
true values, which can only improve R^2 by fixing calibration -- it
can't fix genuine ranking failures, since it's constrained to be
monotonic.

Same discipline as tune_tuh_concept_bottleneck_threshold.py: the
recalibration mapping is FIT on the VALIDATION set (the same split used
throughout training), then APPLIED to the untouched eval set and R^2
recomputed there. Fitting and evaluating the recalibration on the same
eval set would let a flexible mapping overfit that specific set and
trivially inflate R^2 -- this keeps it an honest, out-of-sample check.
"""
import numpy as np
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

from data.tuh_e2e_loader import TUHEndToEndDataset
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneckGNN, normalize_concepts
from train_utils import split_validation


def get_true_pred(model, loader, band_power_median, band_power_iqr):
    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            true_c = normalize_concepts(raw_batch["concepts_raw"], band_power_median, band_power_iqr)
            _, pred_c = model(raw_batch["raw_eeg"])
            all_true.append(true_c.numpy())
            all_pred.append(pred_c.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    with open("config_tuh_concept_bottleneck_gnn.yaml") as f:
        cfg = yaml.safe_load(f)
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    seed = t["seed"]

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
    _, val_indices = split_validation(train_ds, all_indices, t.get("val_frac", 0.2), seed)
    val_loader = DataLoader(Subset(train_ds, val_indices), batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    print(f"val={len(val_indices)} eval(test)={len(eval_ds)}", flush=True)

    model = ConceptBottleneckGNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"],
        n_filters_time=m.get("n_filters_time", 40), n_filters_spat=m.get("n_filters_spat", 40),
        n_hops=m.get("n_hops", 2), sfreq=d["sfreq"], beta_band=tuple(m.get("beta_band", (13, 30))),
        beta_filters=m.get("beta_filters", 16), dropout=m.get("dropout", 0.5),
        residual=m.get("residual", True),
    )
    ckpt = torch.load("tuh_concept_bottleneck_gnn_best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    band_power_median, band_power_iqr = ckpt["band_power_median"], ckpt["band_power_iqr"]

    val_true, val_pred = get_true_pred(model, val_loader, band_power_median, band_power_iqr)
    eval_true, eval_pred = get_true_pred(model, eval_loader, band_power_median, band_power_iqr)

    print(f"\n{'concept':<28}{'R2_raw':>9}{'R2_recal':>10}{'gain':>8}")
    raw_r2s, recal_r2s = [], []
    for i, name in enumerate(CONCEPT_NAMES):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_pred[:, i], val_true[:, i])  # fit mapping on VAL only
        eval_pred_recal = iso.predict(eval_pred[:, i])  # apply to EVAL

        r2_raw = r2_score(eval_true[:, i], eval_pred[:, i])
        r2_recal = r2_score(eval_true[:, i], eval_pred_recal)
        raw_r2s.append(r2_raw)
        recal_r2s.append(r2_recal)
        gain = r2_recal - r2_raw
        flag = "  <- big fix" if gain > 0.15 else ""
        print(f"{name:<28}{r2_raw:>9.3f}{r2_recal:>10.3f}{gain:>8.3f}{flag}")

    print(f"\nmean R^2 raw:        {np.mean(raw_r2s):.3f}")
    print(f"mean R^2 recalibrated: {np.mean(recal_r2s):.3f}")
    n_positive_now = sum(1 for r in recal_r2s if r > 0.2)
    print(f"concepts with R^2 > 0.2 after recalibration: {n_positive_now}/{len(CONCEPT_NAMES)} "
          f"(vs 15/28 before)")

    print("\nInterpretation: isotonic regression is monotonic, so it can only fix miscalibration "
          "(wrong scale/offset), not genuine ranking failures -- a concept that gains a lot here "
          "had real signal masked by bad calibration; a concept that doesn't improve had a real "
          "ranking problem, not just a calibration one.")


if __name__ == "__main__":
    main()
