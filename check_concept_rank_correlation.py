"""
Tests the specific claim in "Problem 3" (from a pasted external proposal)
that this concept bottleneck "captures direction and relative ranking,
not precise values" -- rather than accepting that framing on faith,
this checks it directly: R^2 penalizes any deviation from exact value
matching (including scale/calibration mismatches), but "is delta
elevated or suppressed" and "which subjects have higher TAR" are ranking
claims, which Spearman rank correlation measures directly and R^2 does
not. If Spearman is meaningfully higher than R^2 for a concept, the
"directional/ranking, not precise value" framing is genuinely earned for
that concept, not just a rationalization for a mediocre R^2.

Uses the already-trained tuh_concept_bottleneck_gnn_best_model.pt --
read-only evaluation, no retraining.
"""
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from data.tuh_e2e_loader import TUHEndToEndDataset
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneckGNN, normalize_concepts


def main():
    with open("config_tuh_concept_bottleneck_gnn.yaml") as f:
        cfg = yaml.safe_load(f)
    m, d = cfg["model"], cfg["data"]

    eval_base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = TUHWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)

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
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for raw_batch in eval_loader:
            true_c = normalize_concepts(raw_batch["concepts_raw"], band_power_median, band_power_iqr)
            _, pred_c = model(raw_batch["raw_eeg"])
            all_true.append(true_c.numpy())
            all_pred.append(pred_c.numpy())
    true_c = np.concatenate(all_true)
    pred_c = np.concatenate(all_pred)

    print(f"{'concept':<28}{'R2':>8}{'spearman':>10}{'spearman_p':>12}{'gap':>8}")
    r2s, rhos, ps = [], [], []
    for i, name in enumerate(CONCEPT_NAMES):
        r2 = r2_score(true_c[:, i], pred_c[:, i])
        rho, p = spearmanr(true_c[:, i], pred_c[:, i])
        r2s.append(r2)
        rhos.append(rho)
        ps.append(p)
        gap = rho - r2
        flag = "  <- ranking >> R2" if gap > 0.15 else ""
        print(f"{name:<28}{r2:>8.3f}{rho:>10.3f}{p:>12.4f}{gap:>8.3f}{flag}")

    print(f"\nmean R^2:       {np.mean(r2s):.3f}")
    print(f"mean Spearman:  {np.mean(rhos):.3f}")
    n_meaningful_rho = sum(1 for rho, p in zip(rhos, ps) if rho > 0.2 and p < 0.05)
    print(f"concepts with Spearman > 0.2 AND p < 0.05: {n_meaningful_rho}/{len(CONCEPT_NAMES)}")

    print("\nInterpretation: if Spearman correlations are substantially higher than R^2 across "
          "most concepts, the 'directional/ranking, not precise value' framing is genuinely "
          "earned -- the model reliably orders subjects correctly even where it doesn't hit exact "
          "values. If Spearman tracks R^2 closely (small gap), the framing isn't adding anything "
          "real -- the concepts that are weak by R^2 are just as weak by ranking.")


if __name__ == "__main__":
    main()
