import os

import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"abnormal": 0, "normal": 1}  # same convention as NMT: 0 = pathology first

# Same 21-channel order used throughout this project for NMT (data/electrode_positions.py).
TARGET_CHANNELS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "CZ", "A1", "A2"]


class TUHEndToEndDataset(Dataset):
    """
    TUH Abnormal EEG Corpus counterpart to NMTEndToEndDataset (data/
    nmt_e2e_loader.py) -- same preprocessing convention (CAR, skip_sec/
    max_sec windowing, fixed-scale normalization baked into the cache),
    but a different manifest source and an added channel-selection step:

    - No Labels.csv: label and train/eval split are both encoded directly
      in the directory structure (edf/{train,eval}/{normal,abnormal}/
      01_tcp_ar/*.edf), which is also TUH's own official patient-
      disjoint train/eval partition, not something built here.
    - TUH's raw channel names are "EEG FP1-REF" style (not NMT's plain
      "FP1"), and each file has 30 channels -- the 21 target EEG channels
      used elsewhere in this project, plus EKG/eye-movement/photic-
      stimulus/annotation channels -- in a file-dependent order. Verified
      against 80 real sampled files (train/eval x normal/abnormal) that
      all 21 target channels are present in every file; explicit
      rename+pick+reorder (unlike NMT, where the stored order already
      matched) is required so channel index i means the same electrode
      for every subject.
    - Native sampling rate varies by file (250/256 Hz seen) -- already
      handled by resampling to a fixed target sfreq regardless, same as NMT.

    Known limitation, not addressed here: TUH filenames encode
    patient+session+token (e.g. aaaaaeph_s004_t000.edf and
    aaaaaeph_s005_t000.edf are the same patient, different sessions).
    TUH's own train/eval split is patient-disjoint, but the train-pool
    validation split carved out by split_validation (train_utils.py) is
    NOT patient-aware -- the same patient's sessions could land split
    across train-proper and val within the training pool. Not fixed here
    to keep this a direct port of the NMT pipeline's existing convention.
    """

    def __init__(self, root_dir, cache_dir, split="train", sfreq=100, bandpass=(0.5, 45),
                 skip_sec=60, max_sec=120, clip_uv=800.0, divisor=0.3):
        self.root_dir = root_dir
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
        subjects = []
        for label_name in ("normal", "abnormal"):
            split_dir = os.path.join(self.root_dir, "edf", split, label_name)
            if not os.path.isdir(split_dir):
                continue
            for dirpath, _, filenames in os.walk(split_dir):
                for fn in sorted(filenames):
                    if not fn.endswith(".edf"):
                        continue
                    path = os.path.join(dirpath, fn)
                    sid = fn.replace(".edf", "")
                    subjects.append({
                        "id": sid, "path": path, "group": label_name,
                        "label": LABEL_MAP[label_name],
                    })
        print(f"TUHEndToEndDataset: {len(subjects)} subjects for split={split}")
        return subjects

    def _load_raw(self, path):
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        rename = {ch: ch.replace("EEG ", "").replace("-REF", "").strip() for ch in raw.ch_names}
        raw.rename_channels(rename)
        raw.pick_channels(TARGET_CHANNELS, ordered=True)

        raw.filter(self.bandpass[0], self.bandpass[1], method="iir", verbose=False)
        raw.resample(self.sfreq, npad="auto", verbose=False)
        data = raw.get_data().astype(np.float32)  # volts

        # Common average reference -- see data/nmt_e2e_loader.py's docstring.
        data = data - data.mean(axis=0, keepdims=True)

        # Skip skip_sec as burn-in, then take up to max_sec -- same convention as NMT.
        want_skip = int(self.skip_sec * self.sfreq)
        skip_samples = want_skip if data.shape[1] > want_skip else 0
        data = data[:, skip_samples:skip_samples + int(self.max_sec * self.sfreq)]

        data = data * 1e6  # volts -> microvolts
        data = np.clip(data, -self.clip_uv, self.clip_uv)
        data = data / self.divisor
        return data.astype(np.float32)

    def _cache_path(self, sid):
        return os.path.join(self.cache_dir, f"{sid}_tuh.pt")

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
            cached = {"raw_eeg": torch.tensor(data[None])}  # (1, n_ch, n_samples)
            torch.save(cached, cache_path)

        return {
            **cached,
            "label": torch.tensor(subject["label"], dtype=torch.long),
            "subject_id": sid,
            "group": subject["group"],
        }


def collate_tuh_e2e(items):
    """Same padding/length-tracking convention as data/nmt_e2e_loader.py's collate_e2e."""
    lengths = torch.tensor([item["raw_eeg"].shape[-1] for item in items], dtype=torch.long)
    T_max = int(lengths.max())
    n_ch = items[0]["raw_eeg"].shape[1]
    batch_raw = torch.zeros(len(items), n_ch, T_max, dtype=items[0]["raw_eeg"].dtype)
    for i, item in enumerate(items):
        T_i = item["raw_eeg"].shape[-1]
        batch_raw[i, :, :T_i] = item["raw_eeg"].squeeze(0)
    labels = torch.stack([item["label"] for item in items])
    return {"raw_eeg": batch_raw, "label": labels, "lengths": lengths}
