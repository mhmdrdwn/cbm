"""
Pairwise statistical significance for the 4 headline TUAB models, using
the EXISTING seed-42 checkpoints (no retraining -- this is inference-only,
fast). Two complementary tests, both using per-sample correct/incorrect
outcomes on the same eval set (same order across all 4 models, since
TUHEndToEndDataset's eval split is deterministic):

1. McNemar's test on each of the 6 pairwise comparisons -- tests whether
   the two models' error patterns differ significantly on this specific
   trained pair, using the discordant-pairs contingency table.
2. Paired bootstrap 95% CI on each pairwise accuracy DIFFERENCE -- resample
   the same eval indices for both models in a pair each iteration (10,000
   iterations), giving a CI on e.g. acc(CB-GNN) - acc(ShallowCNN).

Complements run_multiseed_tuh.py (which tests robustness to training seed);
this tests significance for the SPECIFIC seed-42 models already reported
in article.tex's tab:classification.
"""
import numpy as np
import torch
import yaml
from scipy.stats import binomtest
from torch.utils.data import DataLoader

from data.tuh_e2e_loader import TUHEndToEndDataset, collate_tuh_e2e
from data.tuh_concepts_loader import TUHWithConceptsDataset, collate_tuh_concepts
from models.shallow_cnn import ShallowConvNet
from models.graph_coupling_cnn import ShallowGNN
from models.concept_bottleneck import ConceptBottleneckShallowCNN, ConceptBottleneckGNN, normalize_concepts

N_BOOTSTRAP = 10000
RNG_SEED = 0


def get_plain_predictions(model_cls, ckpt_path, config_path, **model_kwargs):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    eval_ds = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_e2e)
    device = torch.device("cpu")
    model = model_cls(n_channels=d["n_channels"], **model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            logits, _ = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(raw_batch["label"].tolist())
    return np.array(preds), np.array(labels)


def get_cb_shallow_predictions(ckpt_path, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, d = cfg["model"], cfg["data"]
    eval_base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = TUHWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    device = torch.device("cpu")
    model = ConceptBottleneckShallowCNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"], n_filters=m.get("n_filters", 40),
        dropout=m.get("dropout", 0.5), residual=m.get("residual", True),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            logits, _ = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(raw_batch["label"].tolist())
    return np.array(preds), np.array(labels)


def get_cb_gnn_predictions(ckpt_path, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m, d = cfg["model"], cfg["data"]
    eval_base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split="eval",
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    eval_ds = TUHWithConceptsDataset(eval_base, d["concept_cache_dir"], sfreq=d["sfreq"])
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_tuh_concepts)
    device = torch.device("cpu")
    model = ConceptBottleneckGNN(
        n_channels=d["n_channels"], n_classes=m["n_classes"],
        n_filters_time=m.get("n_filters_time", 40), n_filters_spat=m.get("n_filters_spat", 40),
        n_hops=m.get("n_hops", 2), sfreq=d["sfreq"], beta_band=tuple(m.get("beta_band", (13, 30))),
        beta_filters=m.get("beta_filters", 16), dropout=m.get("dropout", 0.5),
        residual=m.get("residual", True),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for raw_batch in loader:
            x = raw_batch["raw_eeg"].to(device)
            logits, _ = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(raw_batch["label"].tolist())
    return np.array(preds), np.array(labels)


def mcnemar_test(correct_a, correct_b):
    """correct_a, correct_b: (n,) bool arrays, same sample order.
    Exact McNemar via binomial test on discordant pairs (n01 vs n10)."""
    n01 = int(np.sum((~correct_a) & correct_b))   # a wrong, b right
    n10 = int(np.sum(correct_a & (~correct_b)))    # a right, b wrong
    n_discordant = n01 + n10
    if n_discordant == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    p = binomtest(min(n01, n10), n_discordant, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_value": p}


def bootstrap_diff_ci(correct_a, correct_b, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Paired bootstrap 95% CI on acc(b) - acc(a): resample the SAME indices
    for both models each iteration (preserves pairing)."""
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
    print("Loading predictions from existing seed-42 checkpoints...", flush=True)

    preds_cnn, labels = get_plain_predictions(
        ShallowConvNet, "tuh_shallow_cnn_best_model.pt", "config_tuh_shallow_cnn.yaml",
        n_classes=2, dropout=0.5,
    )
    preds_gnn, labels_g = get_plain_predictions(
        ShallowGNN, "tuh_graph_cnn_best_model.pt", "config_tuh_graph_cnn.yaml",
        n_classes=2, n_hops=2, dropout=0.5,
    )
    preds_cb_cnn, labels_c = get_cb_shallow_predictions(
        "tuh_concept_bottleneck_best_model.pt", "config_tuh_concept_bottleneck.yaml",
    )
    preds_cb_gnn, labels_cg = get_cb_gnn_predictions(
        "tuh_concept_bottleneck_gnn_best_model.pt", "config_tuh_concept_bottleneck_gnn.yaml",
    )

    assert np.array_equal(labels, labels_g) and np.array_equal(labels, labels_c) and np.array_equal(labels, labels_cg), \
        "eval label order mismatch across models -- cannot compare per-sample"

    models = {
        "ShallowCNN": preds_cnn,
        "GNN": preds_gnn,
        "CB-ShallowCNN": preds_cb_cnn,
        "CB-GNN": preds_cb_gnn,
    }
    correct = {name: (preds == labels) for name, preds in models.items()}

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
