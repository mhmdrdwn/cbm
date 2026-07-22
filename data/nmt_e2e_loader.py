import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"abnormal": 0, "normal": 1}  # 0 = pathology first, matching the project convention


class NMTEndToEndDataset(Dataset):
    """
    NMT dataset returning raw (band-passed) time series only -- no
    connectivity/feature precomputation baked into the cache, keeping this
    cheap to build regardless of which downstream model consumes it (any
    per-sample connectivity gets computed by the model itself, not
    precomputed here).

    Preprocessing follows the NMT dataset's own reference pipeline
    (dll-ncai/eeg_pre-diagnostic_screening, code/CNN/config.py) rather
    than this project's earlier global z-score convention:
      - resample to 100 Hz (that pipeline's sampling_freq, half our
        earlier 200 Hz)
      - skip the first 60s of each recording (their sec_to_cut) as
        settling/artifact burn-in, instead of starting the window at t=0
      - fixed-scale normalization: convert volts -> microvolts, clip to
        +-clip_uv (their max_abs_val=800), divide by `divisor` (their
        divisor=10) -- a simple, data-independent transform, unlike the
        earlier per-channel global z-score (compute_channel_norm), so it
        can be baked directly into the cache instead of computed from
        training-set statistics at train time. Padding introduced by
        collate_e2e for batching stays exactly 0 under this transform
        (0 -> 0 through clip/divide), unlike z-scoring, which shifted
        padding to -mean/std and needed an explicit re-zeroing step.
    """

    def __init__(self, raw_dir, cache_dir, split="train", sfreq=100, bandpass=(0.5, 45),
                 skip_sec=60, max_sec=120, clip_uv=800.0, divisor=10.0):
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.sfreq = sfreq
        self.bandpass = bandpass
        self.skip_sec = skip_sec
        self.max_sec = max_sec
        self.clip_uv = clip_uv
        self.divisor = divisor

        os.makedirs(cache_dir, exist_ok=True)
        self.subjects = self._build_manifest(split)

    def _build_manifest(self, split):
        df = pd.read_csv(os.path.join(self.raw_dir, "Labels.csv"))
        subjects = []
        for _, row in df.iterrows():
            if row["loc"] != split:
                continue
            label_name = row["label"]
            path = os.path.join(self.raw_dir, label_name, split, row["recordname"])
            if not os.path.exists(path):
                continue
            sid = row["recordname"].replace(".edf", "")
            subjects.append({
                "id": sid, "path": path, "group": label_name,
                "label": LABEL_MAP[label_name], "age": int(row["age"]),
            })
        print(f"NMTEndToEndDataset: {len(subjects)} subjects for split={split}")
        return subjects

    def _load_raw(self, path):
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.filter(self.bandpass[0], self.bandpass[1], method="iir", verbose=False)
        raw.resample(self.sfreq, npad="auto", verbose=False)
        data = raw.get_data().astype(np.float32)  # volts

        # Common average reference: subtract the per-timepoint mean across
        # all 21 channels from each channel. NMT's original acquisition
        # reference isn't documented, and this project's models learn
        # connectivity directly from cross-channel relationships -- CAR
        # puts those relationships on a consistent, montage-independent
        # footing instead of leaving them dependent on an unknown/uncontrolled
        # original reference choice. Applied before windowing/clipping/scaling
        # since it's a per-timepoint operation, unaffected by either.
        data = data - data.mean(axis=0, keepdims=True)

        # Skip the first skip_sec as burn-in, then take up to max_sec from
        # there -- but only if the recording actually has more than
        # skip_sec of signal. The previous `min(skip_sec*sfreq, T-1)` clamp
        # was meant to handle recordings shorter than skip_sec, but for
        # those it evaluates to T-1 (skip nearly everything), leaving
        # exactly 1 sample instead of "whatever's left" -- verified against
        # the cache: 14 subjects had been silently reduced to a single
        # timepoint this way. Skip nothing when the recording doesn't clear
        # skip_sec, so short recordings keep their real (short) signal
        # instead of being destroyed.
        want_skip = int(self.skip_sec * self.sfreq)
        skip_samples = want_skip if data.shape[1] > want_skip else 0
        data = data[:, skip_samples:skip_samples + int(self.max_sec * self.sfreq)]

        data = data * 1e6  # volts -> microvolts, matching the reference pipeline's raw scale
        data = np.clip(data, -self.clip_uv, self.clip_uv)
        data = data / self.divisor
        return data.astype(np.float32)

    def _cache_path(self, sid):
        return os.path.join(self.cache_dir, f"{sid}_e2e.pt")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        sid = subject["id"]
        cache_path = self._cache_path(sid)

        if os.path.exists(cache_path):
            cached = torch.load(cache_path)
        else:
            data = self._load_raw(subject["path"])
            cached = {"raw_eeg": torch.tensor(data[None])}  # (1, n_ch, n_samples), fixed-scale normalized
            torch.save(cached, cache_path)

        return {
            **cached,
            "label": torch.tensor(subject["label"], dtype=torch.long),
            "age": torch.tensor(subject["age"], dtype=torch.float32),
            "subject_id": sid,
            "group": subject["group"],
        }


def collate_e2e(items):
    """
    Batches variable-length recordings (subjects shorter than max_sec are
    NOT padded at load time -- see _load_raw's truncate-only slicing) by
    zero-padding every sample up to this batch's own max length, and
    tracking each sample's real length so a model can mask out the
    padded time region if it needs to. Pads to the batch's max, not the
    dataset-wide max_sec*sfreq, to avoid wasting compute on padding that
    no subject in a given batch needs.
    """
    lengths = torch.tensor([item["raw_eeg"].shape[-1] for item in items], dtype=torch.long)
    T_max = int(lengths.max())
    n_ch = items[0]["raw_eeg"].shape[1]
    batch_raw = torch.zeros(len(items), n_ch, T_max, dtype=items[0]["raw_eeg"].dtype)
    for i, item in enumerate(items):
        T_i = item["raw_eeg"].shape[-1]
        batch_raw[i, :, :T_i] = item["raw_eeg"].squeeze(0)
    labels = torch.stack([item["label"] for item in items])
    ages = torch.stack([item["age"] for item in items])
    return {"raw_eeg": batch_raw, "label": labels, "age": ages, "lengths": lengths}
