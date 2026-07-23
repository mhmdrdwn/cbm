import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# CAUEEG's own channel order (annotation.json's signal_header, EKG/Photic dropped,
# "-AVG" suffix stripped and uppercased) -- verified identical across a random
# sample of real EDF files' own headers, and matches signal_header exactly. A
# THIRD distinct ordering from both TUH/NMT's TARGET_CHANNELS and ds004504's.
TARGET_CHANNELS = ["FP1", "F3", "C3", "P3", "O1", "FP2", "F4", "C4", "P4", "O2",
                    "F7", "T3", "T5", "F8", "T4", "T6", "FZ", "CZ", "PZ"]

# Verified constant across all 1379 EDF files (checked directly, not sampled) --
# used to convert event timestamps (native-sfreq sample units) to seconds when
# building the manifest, without needing to open each EDF just to count duration.
NATIVE_SFREQ = 200.0

# Original CAUEEG label conventions (abnormal.json/dementia.json) put the healthy
# class at 0 -- opposite of this project's "pathology first" convention used
# everywhere else. Remapped once here rather than left inconsistent per script.
# Dementia severity ordering (worst-to-best) rather than an arbitrary pick between
# the two pathology classes, matching this project's "0=most severe" spirit.
_ABNORMAL_LABEL_MAP = {"Abnormal": 0, "Normal": 1}
_DEMENTIA_LABEL_MAP = {"Dementia": 0, "MCI": 1, "Normal": 2}
TASK_LABEL_MAPS = {"abnormal": _ABNORMAL_LABEL_MAP, "dementia": _DEMENTIA_LABEL_MAP}
TASK_JSON_FILES = {"abnormal": "abnormal.json", "dementia": "dementia.json"}
TASK_N_CLASSES = {"abnormal": 2, "dementia": 3}


def _eyes_closed_intervals(events):
    """
    events: list of [sample_idx, label] from event/{serial}.json (native-sfreq
    sample units). Returns [(start, end), ...] intervals covering only "Eyes
    Closed" periods that are NOT concurrent with photic stimulation.

    CAUEEG's clinical protocol alternates brief eyes-open/closed instructions
    (often only 5-90s each) with photic driving-response blocks (3-30Hz flash),
    unlike TUH/NMT/ds004504's one long continuous resting recording. Including
    photic-stimulation periods would inject an artificial band-power response
    directly overlapping this project's alpha/beta concept definitions -- not
    real endogenous rhythm. A trial can technically still be marked "Eyes
    Closed" during a photic block (patients are sometimes asked to keep eyes
    closed through part of it too), so photic_on is tracked explicitly from
    "Photic On"/"Photic Off" markers rather than trusting "Eyes Closed" alone.
    Any other marker (Eyes Open, artifact, Move, swallowing, Paused, ...) ends
    the current interval -- conservatively excluding borderline periods rather
    than assuming they're still clean.
    """
    photic_on = False
    intervals = []
    open_start = None
    for samp, label in events:
        if "Photic On" in label:
            photic_on = True
        elif "Photic Off" in label:
            photic_on = False
        if label == "Eyes Closed" and not photic_on:
            open_start = samp
        elif open_start is not None:
            intervals.append((open_start, samp))
            open_start = None
    return intervals


def _total_eyes_closed_sec(events, max_native_samples=None):
    """
    max_native_samples: clip intervals to the actual recording length before
    summing. REQUIRED for an accurate answer -- verified directly that 343/1379
    subjects (25%) have event timestamps that run PAST the end of their own EDF
    recording (up to 700s overshoot, likely a clock/logging mismatch upstream in
    the dataset, not something introduced by this loader). Without clipping,
    this silently overestimates available content, which then overestimates how
    many train windows a subject can support -- caught via a real crash
    (IndexError on an empty window) during smoke testing, not assumed away.
    """
    ivs = _eyes_closed_intervals(events)
    if max_native_samples is not None:
        ivs = [(a, min(b, max_native_samples)) for a, b in ivs if a < max_native_samples]
    return sum(b - a for a, b in ivs) / NATIVE_SFREQ


class CAUEEGEndToEndDataset(Dataset):
    """
    CAUEEG (Chung-Ang University Hospital EEG dataset) -- 1379 subjects, 200Hz
    native, 19 EEG channels (+EKG/Photic, dropped) already in average
    reference at acquisition ("-AVG" channel names), with two ready-made
    benchmark tasks sharing the same signal files:
      - "abnormal": binary Normal/Abnormal, official 1107/136/136 train/val/test.
      - "dementia": 3-class Normal/MCI/Dementia, official 950/119/118 split
        (a different, smaller subject pool than "abnormal" -- not every
        subject has a dementia-relevant diagnosis).
    Official splits are used directly (task JSON's train_split/validation_split/
    test_split), not rebuilt here, unlike ds004504 which shipped none.

    Preprocessing: CAR, band-pass, resample -- same convention as TUH/NMT/
    ds004504 -- PLUS an event-marker-based segment selection step those
    datasets didn't need (_eyes_closed_intervals): eyes-closed, non-photic
    segments are concatenated, then chopped into non-overlapping window_sec
    windows. skip_sec skips from the START OF THE CONCATENATED STREAM before
    windowing, not from the start of the raw recording -- these are already
    short, instruction-triggered clean segments, not one continuous recording
    needing a burn-in discard.

    MULTIPLE WINDOWS PER SUBJECT, TRAIN SPLIT ONLY: most subjects have well
    over window_sec (60s default) of total usable eyes-closed content (median
    ~354s across the whole dataset) -- a single-window-per-subject scheme
    throws most of that away. For split="train", each subject contributes
    floor(available_sec / window_sec) non-overlapping windows (capped at
    max_windows_per_subject, default 5, so a handful of very long recordings
    don't dominate the train set with many correlated windows from the same
    underlying brain/session), each becoming its own (raw_eeg, label) sample
    -- self.subjects has one entry per WINDOW, not per subject, for train.
    val/eval deliberately stay ONE window per subject (index 0) so reported
    val/eval metrics remain directly comparable to a plain per-subject
    accuracy, not inflated/distorted by subjects with more usable content
    getting more evaluation weight. Window count is computed from the (cheap
    to parse) event JSON alone, before any EDF is opened.

    9/1379 subjects have no usable eyes-closed content at all -- these get
    exactly 1 window (index 0) regardless of split, via the same plain
    skip_sec/window_sec truncation-of-the-whole-recording fallback used
    before; not multi-windowed, since there's no marked-clean content to
    slice further without risking photic contamination across windows.

    eval_tta: if True, the "eval" split ALSO gets multiple windows per
    subject (same scheme as train), instead of val/eval's normal single-
    window-per-subject. Matches the official CAUEEG reference implementation
    (github.com/ipis-mjkim/caueeg-ceednet)'s test-time augmentation --
    predictions from each subject's windows get averaged before scoring (see
    utils/metrics.py's aggregate_predictions_by_subject), not scored per-
    window independently. val stays single-window regardless (checkpoint
    selection runs every epoch, multi-window there would slow training for
    a metric that's only used for relative epoch comparison, not reported).
    """

    def __init__(self, root_dir, task, cache_dir, split="train", sfreq=100, bandpass=(0.5, 45),
                 skip_sec=0, window_sec=60, max_windows_per_subject=5, clip_uv=800.0, divisor=10.8,
                 eval_tta=False):
        assert task in TASK_LABEL_MAPS, f"unknown task {task!r}, expected 'abnormal' or 'dementia'"
        self.root_dir = root_dir
        self.task = task
        self.cache_dir = cache_dir
        self.sfreq = sfreq
        self.bandpass = bandpass
        self.skip_sec = skip_sec
        self.window_sec = window_sec
        self.max_windows_per_subject = max_windows_per_subject
        self.clip_uv = clip_uv
        self.divisor = divisor
        self.eval_tta = eval_tta

        os.makedirs(cache_dir, exist_ok=True)
        self.subjects = self._build_manifest(split)

    def _build_manifest(self, split):
        with open(os.path.join(self.root_dir, TASK_JSON_FILES[self.task])) as f:
            task_json = json.load(f)
        split_key = {"train": "train_split", "val": "validation_split", "eval": "test_split"}[split]
        label_map = TASK_LABEL_MAPS[self.task]
        multi_window = split == "train" or (split == "eval" and self.eval_tta)

        subjects = []
        for row in task_json[split_key]:
            sid = row["serial"]
            path = os.path.join(self.root_dir, "signal", "edf", f"{sid}.edf")
            if not os.path.exists(path):
                continue
            base = {"id": sid, "path": path, "group": row["class_name"],
                    "label": label_map[row["class_name"]], "age": row["age"]}

            if multi_window:
                event_path = os.path.join(self.root_dir, "event", f"{sid}.json")
                with open(event_path) as f:
                    events = json.load(f)
                # header-only read (preload=False) just to get the real recording
                # length -- fast (~20ms/file), needed because event timestamps can
                # run past the actual EDF (see _total_eyes_closed_sec's docstring).
                import mne
                mne.set_log_level("ERROR")
                native_n_samples = mne.io.read_raw_edf(path, preload=False, verbose=False).n_times
                available_sec = max(0.0, _total_eyes_closed_sec(events, native_n_samples) - self.skip_sec)
                n_windows = int(available_sec // self.window_sec)
                n_windows = max(1, min(n_windows, self.max_windows_per_subject))
                # n_windows=1 for the 0-eyes-closed-content subjects too (available_sec=0
                # -> floor gives 0, clamped up to 1) -- that single window falls back to
                # plain truncation of the whole recording in _load_raw_full.
                for w in range(n_windows):
                    subjects.append({**base, "window_idx": w})
            else:
                subjects.append({**base, "window_idx": 0})

        n_unique = len(set(s["id"] for s in subjects))
        print(f"CAUEEGEndToEndDataset(task={self.task}): {len(subjects)} samples "
              f"({n_unique} subjects) for split={split}")
        return subjects

    def _load_raw_full(self, sid, path):
        """
        Full prepared (concatenated eyes-closed, CAR'd, scaled) stream for this
        subject, capped at max_windows_per_subject*window_sec seconds -- cached
        ONCE per subject regardless of how many windows get sliced from it
        (see __getitem__), since this depends only on the subject's own
        signal+events, not on which window_idx a given manifest entry wants.
        """
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        rename = {ch: ch.replace("-AVG", "").upper() for ch in raw.ch_names}
        raw.rename_channels(rename)
        raw.pick_channels(TARGET_CHANNELS, ordered=True)

        raw.filter(self.bandpass[0], self.bandpass[1], method="iir", verbose=False)
        native_sfreq = raw.info["sfreq"]
        raw.resample(self.sfreq, npad="auto", verbose=False)
        data = raw.get_data().astype(np.float32)  # volts, now at target sfreq

        # Common average reference -- see TUHEndToEndDataset's docstring.
        data = data - data.mean(axis=0, keepdims=True)

        with open(os.path.join(self.root_dir, "event", f"{sid}.json")) as f:
            events = json.load(f)
        # event timestamps are in NATIVE-sfreq sample units -- rescale to match the
        # just-resampled data's indexing (resampling is a linear time-scaling, so
        # sample index i at native_sfreq corresponds to i*(sfreq/native_sfreq) here).
        scale = self.sfreq / native_sfreq
        # clip to data.shape[1] -- defensive second check (manifest-building already
        # clips against the real recording length, see _total_eyes_closed_sec's
        # docstring for why this is necessary, not paranoia) so a window can never
        # come up empty/short here even if some other edge case slips past that.
        n_data_samples = data.shape[1]
        intervals = [
            (int(a * scale), min(int(b * scale), n_data_samples))
            for a, b in _eyes_closed_intervals(events) if int(a * scale) < n_data_samples
        ]

        max_total_samples = int(self.max_windows_per_subject * self.window_sec * self.sfreq)
        if intervals:
            chunks, total = [], 0
            for a, b in intervals:
                if total >= max_total_samples:
                    break
                take_end = min(b, a + (max_total_samples - total))
                if take_end > a:
                    chunks.append(data[:, a:take_end])
                    total += take_end - a
            if chunks:
                data = np.concatenate(chunks, axis=1)
        # else (or no chunks survived rounding): no usable eyes-closed segment for
        # this subject -- data stays the full resampled+CAR'd recording, falling
        # through to plain skip/truncate below.

        want_skip = int(self.skip_sec * self.sfreq)
        skip_samples = want_skip if data.shape[1] > want_skip else 0
        data = data[:, skip_samples:skip_samples + max_total_samples]

        data = data * 1e6  # volts -> microvolts
        data = np.clip(data, -self.clip_uv, self.clip_uv)
        data = data / self.divisor
        return data.astype(np.float32)

    def _cache_path(self, sid):
        # shared across both tasks AND every window of a subject -- preprocessing
        # doesn't depend on task/window_idx, only on the subject's own signal+events.
        return os.path.join(self.cache_dir, f"{sid}_caueeg.pt")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        sid = subject["id"]
        window_idx = subject["window_idx"]
        cache_path = self._cache_path(sid)

        if os.path.exists(cache_path):
            cached = torch.load(cache_path)
        else:
            data = self._load_raw_full(sid, subject["path"])
            cached = {"raw_eeg_full": torch.tensor(data[None])}  # (1, n_ch, n_samples)
            torch.save(cached, cache_path)

        full = cached["raw_eeg_full"].squeeze(0)  # (n_ch, T_full)
        window_samples = int(self.window_sec * self.sfreq)
        start = window_idx * window_samples
        raw_eeg = full[:, start:start + window_samples].unsqueeze(0)  # (1, n_ch, <=window_samples)
        assert raw_eeg.shape[-1] > 0, (
            f"empty window for sid={sid} window_idx={window_idx} (full stream length="
            f"{full.shape[-1]} samples) -- manifest window-count and available signal "
            f"disagree; see _total_eyes_closed_sec's docstring for the known cause "
            f"(event timestamps past the EDF's actual length) this should already guard against."
        )

        return {
            "raw_eeg": raw_eeg,
            "label": torch.tensor(subject["label"], dtype=torch.long),
            "age": torch.tensor(subject["age"], dtype=torch.float32),
            "subject_id": sid,
            "group": subject["group"],
            "window_idx": window_idx,
        }


def collate_caueeg_e2e(items):
    """Same padding/length-tracking convention as data/tuh_e2e_loader.py's collate_tuh_e2e.
    subject_id stays a plain list (not stackable into a tensor) -- needed downstream to
    average TTA predictions back to one-per-subject (see aggregate_predictions_by_subject
    in utils/metrics.py)."""
    lengths = torch.tensor([item["raw_eeg"].shape[-1] for item in items], dtype=torch.long)
    T_max = int(lengths.max())
    n_ch = items[0]["raw_eeg"].shape[1]
    batch_raw = torch.zeros(len(items), n_ch, T_max, dtype=items[0]["raw_eeg"].dtype)
    for i, item in enumerate(items):
        T_i = item["raw_eeg"].shape[-1]
        batch_raw[i, :, :T_i] = item["raw_eeg"].squeeze(0)
    labels = torch.stack([item["label"] for item in items])
    ages = torch.stack([item["age"] for item in items])
    subject_ids = [item["subject_id"] for item in items]
    return {"raw_eeg": batch_raw, "label": labels, "age": ages, "lengths": lengths, "subject_id": subject_ids}


def compute_eeg_channel_norm(dataset, indices):
    """
    Per-channel mean/std, computed from train subjects' (already CAR'd, clipped,
    divisor-scaled) raw EEG -- an additional normalization layer on top of the
    existing fixed-divisor scale, applied at batch time right before the model
    sees the signal (not baked into the cache, and NOT applied to what concept
    computation sees -- CAUEEGWithConceptsDataset reads the divisor-scaled but
    NOT z-scored raw_eeg directly from the base dataset, same as before this was
    added). Matches the official CAUEEG reference implementation's
    EegNormalizeMeanStd (github.com/ipis-mjkim/caueeg-ceednet), computed once per
    channel across the whole training set rather than per-recording, so absolute
    amplitude differences between subjects (a real clinical signal, e.g. slowing/
    attenuation in dementia) aren't normalized away per-sample.

    Returns (mean, std), each (n_channels,) float32 tensors.
    """
    sums, sqsums, n = None, None, 0
    for i in indices:
        x = dataset[i]["raw_eeg"].squeeze(0)  # (n_ch, T)
        if sums is None:
            sums = torch.zeros(x.shape[0])
            sqsums = torch.zeros(x.shape[0])
        sums += x.sum(dim=-1)
        sqsums += (x ** 2).sum(dim=-1)
        n += x.shape[-1]
    mean = sums / n
    var = (sqsums / n) - mean ** 2
    std = var.clamp(min=1e-12).sqrt()
    return mean, std


def normalize_eeg(raw_eeg, mean, std, eps=1e-8):
    """raw_eeg: (batch, n_ch, T). mean/std: (n_ch,), from compute_eeg_channel_norm."""
    mean = mean.to(raw_eeg.device).view(1, -1, 1)
    std = std.to(raw_eeg.device).view(1, -1, 1)
    return (raw_eeg - mean) / (std + eps)
