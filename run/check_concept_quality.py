"""
Concept data-quality audit -- checks the CACHED analytically-computed
concept values (compute_concepts_raw's output, models/concept_bottleneck.py)
directly against the concept cache, without loading any raw EEG: builds the
lightweight dataset manifest (subject id / label / age -- no signal I/O)
and reads each sample's already-cached concept vector straight off disk.
Fast even across thousands of subjects.

Checks:
  1. Basic validity: NaN/Inf, constant concepts, values outside [0,1]
     (checked on NORMALIZED concepts, i.e. what actually feeds the MSE
     loss and the classifier -- that's the only representation where
     "[0,1]" is a meaningful bound for the band-power family; band power
     in RAW form is log1p(power), deliberately unbounded, and is NOT
     checked against [0,1]).
  2. Low variance: std of each concept, normalized -- the leading suspect
     for catastrophic negative R^2 (near-zero target variance makes any
     residual look enormous relative to it; see run_intervention's
     R^2 = 1 - SS_res/SS_tot, which blows up as SS_tot -> 0).
  3. Raw physiological plausibility (--raw only): asymmetry/PLV/ratio
     range checks on the RAW (pre-normalize_concepts) values, and whether
     alpha peak frequency is stuck at the 6/14Hz search-range boundary
     (meaning no real interior peak was found, not that one exists at
     the edge).
  4. Redundancy: exact-duplicate concept vectors across samples (cache
     bug or genuine duplicate recordings), and near-duplicate CONCEPTS
     (pairwise |correlation| above a threshold). This check is what
     originally found delta_alpha_ratio (DAR) redundant with dtabr
     (r=0.97-0.98 on both datasets) -- DAR has since been removed from
     CONCEPT_NAMES entirely (models/concept_bottleneck.py), so it no
     longer appears here.
  5. Train/test shift + R^2 floor: mean/std shift between train and eval
     per concept, and the R^2 a model would get by always predicting the
     TRAIN mean on the EVAL set -- a floor to compare the trained model's
     own reported R^2 against (if the model doesn't clear this floor by
     much, it isn't doing better than a constant predictor).
  6. Clinical sanity: per-class means for every concept, plus an explicit
     expected-direction check for the handful of concepts with a
     documented direction in the literature already cited in article.tex
     (e.g. delta band power higher in pathology, occipital alpha lower).
     Label 0 is always the most pathological class in this project's
     convention (LABEL_MAP/TASK_LABEL_MAPS, "pathology first"). Also
     reports correlation with age where available -- CAUEEG only; TUH's
     loader has no age field at all, and this is reported explicitly
     rather than silently skipped.

Usage:
  python run/check_concept_quality.py --dataset tuh
  python run/check_concept_quality.py --dataset caueeg --task dementia --raw
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for models/data/utils

import numpy as np
import torch
import yaml

from data.caueeg_e2e_loader import CAUEEGEndToEndDataset
from data.concept_cache import concept_cache_path
from data.tuh_e2e_loader import TUHEndToEndDataset
from models.concept_bottleneck import (
    CONCEPT_NAMES, N_BAND_POWER_CONCEPTS, N_CONCEPTS,
    compute_concept_norm, normalize_concepts,
)

RATIO_NAMES = {"theta_alpha_ratio", "dtabr"}
PEAK_NAMES = {"alpha_peak_freq", "alpha_peak_amplitude"}

# expected mean direction: pathology (label 0) vs the most normal class (max label),
# for the concepts with a documented direction (see article.tex section 2.2).
EXPECTED_DIRECTION = {
    "frontal_delta": "higher_in_pathology", "temporal_delta": "higher_in_pathology",
    "central_delta": "higher_in_pathology", "parietal_delta": "higher_in_pathology",
    "occipital_delta": "higher_in_pathology",
    "occipital_alpha": "lower_in_pathology",
    "theta_alpha_ratio": "higher_in_pathology",
}


def concept_family(name):
    if CONCEPT_NAMES.index(name) < N_BAND_POWER_CONCEPTS:
        return "band_power"
    if name.endswith("_asym"):
        return "asymmetry"
    if name.endswith("_plv"):
        return "plv"
    if name in RATIO_NAMES:
        return "ratio"
    if name in PEAK_NAMES:
        return "peak"
    raise ValueError(name)


def load_split_tuh(cfg, split):
    d = cfg["data"]
    base = TUHEndToEndDataset(
        root_dir=d["root_dir"], cache_dir=d["cache_dir"], split=split,
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        max_sec=d["max_sec"], clip_uv=d["clip_uv"], divisor=d["divisor"],
    )
    raws, labels, keys, missing = [], [], [], 0
    for s in base.subjects:
        path = concept_cache_path(d["concept_cache_dir"], s["id"])
        if not os.path.exists(path):
            missing += 1
            continue
        raws.append(torch.load(path)["concepts_raw"])
        labels.append(s["label"])
        keys.append(s["id"])
    if missing:
        print(f"  WARNING: {missing}/{len(base.subjects)} subjects have no cached concepts (skipped)", flush=True)
    return torch.stack(raws), torch.tensor(labels), None, keys


def load_split_caueeg(cfg, split):
    d = cfg["data"]
    base = CAUEEGEndToEndDataset(
        root_dir=d["root_dir"], task=d["task"], cache_dir=d["cache_dir"], split=split,
        sfreq=d["sfreq"], bandpass=tuple(d["bandpass"]), skip_sec=d["skip_sec"],
        window_sec=d["window_sec"], max_windows_per_subject=d.get("max_windows_per_subject", 5),
        clip_uv=d["clip_uv"], divisor=d["divisor"],
        eval_tta=d.get("eval_tta", True) if split == "eval" else False,
    )
    raws, labels, ages, keys, missing = [], [], [], [], 0
    for s in base.subjects:
        cache_key = f"{s['id']}_w{s['window_idx']}"
        path = concept_cache_path(d["concept_cache_dir"], cache_key)
        if not os.path.exists(path):
            missing += 1
            continue
        raws.append(torch.load(path)["concepts_raw"])
        labels.append(s["label"])
        ages.append(s["age"])
        keys.append(cache_key)
    if missing:
        print(f"  WARNING: {missing}/{len(base.subjects)} windows have no cached concepts (skipped)", flush=True)
    return torch.stack(raws), torch.tensor(labels), torch.tensor(ages, dtype=torch.float32), keys


def check_basic_validity(tag, norm):
    print(f"\n=== [1] Basic validity ({tag}, normalized) ===")
    nan_inf = (~torch.isfinite(norm)).sum(dim=0)
    any_nan = False
    for i, name in enumerate(CONCEPT_NAMES):
        if nan_inf[i] > 0:
            any_nan = True
            print(f"  {name:<28} {nan_inf[i].item()} NaN/Inf")
    if not any_nan:
        print("  no NaN/Inf")

    const = norm.std(dim=0) < 1e-6
    if const.any():
        print("  CONSTANT concepts (std < 1e-6):", [CONCEPT_NAMES[i] for i in torch.where(const)[0].tolist()])
    else:
        print("  no constant concepts")

    any_oob = False
    for i, name in enumerate(CONCEPT_NAMES):
        col = norm[:, i]
        n_oob = ((col < -1e-4) | (col > 1 + 1e-4)).sum().item()
        if n_oob > 0:
            any_oob = True
            print(f"  {name:<28} {n_oob} values outside [0,1]  min={col.min():.4f} max={col.max():.4f}")
    if not any_oob:
        print("  no out-of-[0,1] values")

    print("  saturation (fraction stuck within 1e-3 of 0 or 1):")
    for i, name in enumerate(CONCEPT_NAMES):
        col = norm[:, i]
        frac0 = (col < 1e-3).float().mean().item()
        frac1 = (col > 1 - 1e-3).float().mean().item()
        if frac0 + frac1 > 0.3:
            print(f"    {name:<28} stuck~0={frac0:.1%}  stuck~1={frac1:.1%}")


def check_low_variance(tag, norm, threshold=0.03):
    print(f"\n=== [2] Low variance (normalized, {tag}) -- sorted ascending, flagged below {threshold} ===")
    stds = norm.std(dim=0)
    order = torch.argsort(stds)
    for i in order.tolist():
        flag = "  <-- LOW" if stds[i] < threshold else ""
        print(f"  {CONCEPT_NAMES[i]:<28} std={stds[i].item():.4f}{flag}")


def check_raw_physiological(raw):
    print("\n=== [3] Raw physiological plausibility ===")
    for i, name in enumerate(CONCEPT_NAMES):
        fam = concept_family(name)
        vals = raw[:, i]
        lo, hi = vals.min().item(), vals.max().item()
        if fam in ("asymmetry", "plv"):
            bad = ((vals < -1e-4) | (vals > 1 + 1e-4)).sum().item()
            print(f"  {name:<28} [{lo:.3f},{hi:.3f}]  out-of-[0,1]={bad}")
        elif fam == "ratio":
            neg = (vals < -1e-4).sum().item()
            print(f"  {name:<28} [{lo:.3f},{hi:.3f}]  negative={neg}")
        elif name == "alpha_peak_freq":
            hz = 6 + vals * 8
            stuck_lo = (vals < 1e-3).float().mean().item()
            stuck_hi = (vals > 1 - 1e-3).float().mean().item()
            print(f"  {name:<28} {hz.min():.2f}-{hz.max():.2f}Hz  "
                  f"stuck@6Hz={stuck_lo:.1%}  stuck@14Hz={stuck_hi:.1%} (no clear interior peak)")
        elif fam == "peak":  # alpha_peak_amplitude
            print(f"  {name:<28} [{lo:.3f},{hi:.3f}]")
        else:  # band_power -- raw log1p(power), deliberately unbounded
            neg = (vals < -1e-4).sum().item()
            print(f"  {name:<28} [{lo:.3f},{hi:.3f}] (raw log1p power)  negative={neg}")


def check_redundancy(raw, keys, corr_threshold=0.9):
    print(f"\n=== [4] Redundancy ===")
    rounded = torch.round(raw * 1e4) / 1e4
    groups = {}
    for i in range(rounded.shape[0]):
        h = tuple(rounded[i].tolist())
        groups.setdefault(h, []).append(keys[i])
    dup_groups = [ks for ks in groups.values() if len(ks) > 1]
    print(f"  exact-duplicate concept vectors: {len(dup_groups)} group(s) "
          f"covering {sum(len(k) for k in dup_groups)}/{rounded.shape[0]} samples")
    for ks in dup_groups[:5]:
        print(f"    {ks[:5]}{' ...' if len(ks) > 5 else ''}")

    corr = torch.corrcoef(raw.T)
    print(f"  concept-concept |correlation| > {corr_threshold}:")
    found = False
    for i, j in itertools.combinations(range(N_CONCEPTS), 2):
        r = corr[i, j].item()
        if abs(r) > corr_threshold:
            found = True
            print(f"    {CONCEPT_NAMES[i]:<24} vs {CONCEPT_NAMES[j]:<24} r={r:+.3f}")
    if not found:
        print("    none above threshold")


def check_shift_and_floor(train_norm, eval_norm):
    print("\n=== [5] Train/test shift + R^2 floor (predict train mean on eval) ===")
    train_mean = train_norm.mean(dim=0)
    eval_mean = eval_norm.mean(dim=0)
    print(f"  {'concept':<28}{'train mean':<12}{'eval mean':<12}{'shift':<10}{'R2 floor':<10}")
    for i, name in enumerate(CONCEPT_NAMES):
        ss_res = ((eval_norm[:, i] - train_mean[i]) ** 2).sum()
        ss_tot = ((eval_norm[:, i] - eval_mean[i]) ** 2).sum()
        r2_floor = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
        shift = (eval_mean[i] - train_mean[i]).item()
        print(f"  {name:<28}{train_mean[i].item():<12.4f}{eval_mean[i].item():<12.4f}"
              f"{shift:<+10.4f}{r2_floor:<+10.4f}")


def check_clinical_sanity(raw, labels, ages):
    print("\n=== [6] Clinical sanity ===")
    classes = sorted(set(labels.tolist()))
    print(f"  classes present (0=most pathological, this project's convention): {classes}")
    pathology_label, normal_label = 0, max(classes)
    for i, name in enumerate(CONCEPT_NAMES):
        means = {c: raw[labels == c, i].mean().item() for c in classes}
        means_str = "  ".join(f"class{c}={v:.4f}" for c, v in means.items())
        expected = EXPECTED_DIRECTION.get(name)
        note = ""
        if expected:
            diff = means[pathology_label] - means[normal_label]
            matches = (diff > 0) == (expected == "higher_in_pathology")
            note = f"  [expected {expected}: {'OK' if matches else 'MISMATCH'}]"
        print(f"  {name:<28} {means_str}{note}")

    if ages is None:
        print("  age correlation: NOT AVAILABLE (this dataset's loader has no age field)")
    else:
        print("  correlation with age:")
        for i, name in enumerate(CONCEPT_NAMES):
            r = torch.corrcoef(torch.stack([raw[:, i], ages]))[0, 1].item()
            flag = "  <-- |r|>0.2" if abs(r) > 0.2 else ""
            print(f"    {name:<28} r={r:+.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tuh", "caueeg"], default="tuh")
    ap.add_argument("--task", choices=["abnormal", "dementia"], default="dementia",
                     help="CAUEEG only")
    ap.add_argument("--raw", action="store_true", help="also run the raw physiological-range checks")
    args = ap.parse_args()

    if args.dataset == "tuh":
        config_path = "config/config_tuh_concept_bottleneck.yaml"
        loader = load_split_tuh
    else:
        config_path = "config/config_caueeg_dementia_concept_bottleneck.yaml"
        loader = load_split_caueeg

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading {args.dataset} train split...", flush=True)
    train_raw, train_labels, train_ages, train_keys = loader(cfg, "train")
    print(f"Loading {args.dataset} eval split...", flush=True)
    eval_raw, eval_labels, eval_ages, eval_keys = loader(cfg, "eval")
    print(f"train n={len(train_keys)}  eval n={len(eval_keys)}", flush=True)

    concept_median, concept_iqr = compute_concept_norm(train_raw, list(range(len(train_raw))))
    train_norm = normalize_concepts(train_raw, concept_median, concept_iqr)
    eval_norm = normalize_concepts(eval_raw, concept_median, concept_iqr)

    check_basic_validity("train", train_norm)
    check_basic_validity("eval", eval_norm)
    check_low_variance("train", train_norm)
    check_low_variance("eval", eval_norm)
    if args.raw:
        check_raw_physiological(train_raw)
    check_redundancy(train_raw, train_keys)
    check_shift_and_floor(train_norm, eval_norm)
    check_clinical_sanity(eval_raw, eval_labels, eval_ages)


if __name__ == "__main__":
    main()
