import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert, welch


CHANNEL_NAMES = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                  "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "CZ", "A1", "A2"]

REGIONS = {
    "frontal":   [0, 1, 2, 3, 10, 11, 16],   # FP1,FP2,F3,F4,F7,F8,FZ
    "temporal":  [12, 13, 14, 15],            # T3,T4,T5,T6
    "central":   [4, 5, 18],                  # C3,C4,CZ
    "parietal":  [6, 7, 17],                  # P3,P4,PZ
    "occipital": [8, 9],                      # O1,O2
}
# Indices for asymmetry pairs, in TARGET_CHANNELS' ordering.
F3_IDX, F4_IDX = 2, 3
T3_IDX, T4_IDX = 12, 13
O1_IDX, O2_IDX = 8, 9
ASYM_PAIRS = [(F3_IDX, F4_IDX), (T3_IDX, T4_IDX), (O1_IDX, O2_IDX)]


REGIONS_CAUEEG = {
    "frontal":   [0, 5, 1, 6, 10, 13, 16],   # FP1,FP2,F3,F4,F7,F8,FZ
    "temporal":  [11, 14, 12, 15],            # T3,T4,T5,T6
    "central":   [2, 7, 17],                  # C3,C4,CZ
    "parietal":  [3, 8, 18],                  # P3,P4,PZ
    "occipital": [4, 9],                      # O1,O2
}
ASYM_PAIRS_CAUEEG = [(1, 6), (11, 14), (4, 9)]  # F3/F4, T3/T4, O1/O2

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

CONCEPT_NAMES = (
    [f"{r}_{b}" for r in REGIONS for b in BANDS] +
    ["frontal_alpha_asym", "temporal_alpha_asym", "posterior_alpha_asym",
     "theta_alpha_ratio", "dtabr",
     "alpha_peak_freq", "alpha_peak_amplitude",
     "frontal_alpha_plv", "temporal_alpha_plv", "posterior_alpha_plv"]
)
# delta_alpha_ratio (DAR) was dropped: r=0.97-0.98 with dtabr on both datasets
# (run/check_concept_quality.py's redundancy check) -- dtabr = (delta+theta)/
# (alpha+beta) is the more complete "slow-vs-fast" index and subsumes DAR's
# delta/alpha comparison as a special case; keeping both bought no independent
# signal, just two copies of the same (also tanh-saturated, see that script's
# low-variance check) concept.
N_CONCEPTS = len(CONCEPT_NAMES)  # 30
N_BAND_POWER_CONCEPTS = len(REGIONS) * len(BANDS)  # 20 -- used only to identify the band-power family (e.g. run/check_concept_quality.py); all families are population-normalized equally now (see compute_concept_norm)

def _fft_bandpass(x, sfreq, lo, hi):

    T = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    mask = (freqs >= lo) & (freqs < hi)
    return np.fft.irfft(X * mask, n=T, axis=-1)


def _plv(sig_a, sig_b):

    phase_a = np.angle(hilbert(sig_a))
    phase_b = np.angle(hilbert(sig_b))
    return float(np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))


def compute_concepts_raw(eeg, sfreq=100, regions=None, asym_pairs=None):
    """
    Compute all N_CONCEPTS (30) clinical EEG concepts analytically.
    """
    regions = REGIONS if regions is None else regions
    asym_pairs = ASYM_PAIRS if asym_pairs is None else asym_pairs

    n_ch, n_samples = eeg.shape
    freqs, psd = welch(eeg, fs=sfreq, nperseg=min(512, n_samples))  # psd: (n_ch, n_freqs)

    def band_power(psd_row, lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return np.log1p(psd_row[mask].mean()) if mask.any() else 0.0

    concepts = []
    for region, ch_idx in regions.items():
        region_psd = psd[ch_idx].mean(axis=0)
        for lo, hi in BANDS.values():
            concepts.append(band_power(region_psd, lo, hi))
    # concepts[0:20] = raw (unnormalized) family 1

    alpha_lo, alpha_hi = BANDS["alpha"]
    alpha_mask = (freqs >= alpha_lo) & (freqs < alpha_hi)

    def alpha_power_ch(ch_idx):
        return psd[ch_idx, alpha_mask].mean() + 1e-8

    for lo_idx, hi_idx in asym_pairs:
        lo_p, hi_p = alpha_power_ch(lo_idx), alpha_power_ch(hi_idx)
        asym = (hi_p - lo_p) / (hi_p + lo_p)  # in [-1, 1]
        concepts.append((asym + 1) / 2)  # rescale to [0, 1]

    theta_mask = (freqs >= BANDS["theta"][0]) & (freqs < BANDS["theta"][1])
    delta_mask = (freqs >= BANDS["delta"][0]) & (freqs < BANDS["delta"][1])
    beta_mask = (freqs >= BANDS["beta"][0]) & (freqs < BANDS["beta"][1])
    alpha_glob = psd[:, alpha_mask].mean()
    theta_glob = psd[:, theta_mask].mean()
    delta_glob = psd[:, delta_mask].mean()
    beta_glob = psd[:, beta_mask].mean()

    concepts.append(np.tanh(np.log1p(theta_glob / (alpha_glob + 1e-8))))  # theta/alpha ratio
    concepts.append(np.tanh(np.log1p((delta_glob + theta_glob) / (alpha_glob + beta_glob + 1e-8))))  # dtabr

    occ_psd = psd[regions["occipital"]].mean(axis=0)
    peak_range = (freqs >= 6) & (freqs <= 14)
    if peak_range.any() and occ_psd[peak_range].max() > 0:
        peak_freq = freqs[peak_range][occ_psd[peak_range].argmax()]
    else:
        peak_freq = 8.0
    concepts.append(np.clip((peak_freq - 6) / (14 - 6), 0.0, 1.0))

    peak_amp = occ_psd[alpha_mask].max() if alpha_mask.any() else 0.0
    concepts.append(np.tanh(np.log1p(peak_amp)))

    alpha_sig = _fft_bandpass(eeg, sfreq, *BANDS["alpha"])
    for lo_idx, hi_idx in asym_pairs:
        concepts.append(_plv(alpha_sig[lo_idx], alpha_sig[hi_idx]))

    return np.array(concepts, dtype=np.float32)


def compute_concept_norm(raw_concepts, indices):
    """
    raw_concepts: (n_subjects, N_CONCEPTS) RAW concepts (compute_concepts_raw's
    output, cached per-subject). indices: which subjects (e.g. train_indices)
    to compute population statistics from. Returns (median, iqr_scaled), each
    (N_CONCEPTS,) -- ALL concepts are standardized against their own
    population statistics, not just band power.

    Previously only the 20 band-power concepts were standardized this way;
    families 2-4 (asymmetry/ratio/peak/PLV) were passed through normalize_concepts
    unchanged, since compute_concepts_raw already bounds them to [0,1)-ish via
    their own fixed tanh/clip transforms. run/check_concept_quality.py's
    low-variance check found this was exactly backwards: every concept that
    showed pathologically low normalized variance (alpha_peak_amplitude, the 3
    asymmetry concepts, dtabr, the 3 PLV concepts) was one of the unstandardized
    ones -- a FIXED nonlinearity chosen without knowing where a given
    population's values actually cluster can squash most subjects into a
    narrow sub-range of its own output even though its theoretical range is
    [0,1). Standardizing against the population's own median/IQR first
    recenters that cluster onto sigmoid's steep, variance-preserving middle
    zone regardless of the upstream nonlinearity -- exactly why band power
    (the only family already treated this way) never showed this problem.

    ROBUST (median + IQR) statistics, not mean/std -- see this project's
    earlier band-power-only bug (population std blown up ~11x by a handful of
    outlier recordings) for why; the same risk applies to every other family
    too, so the same robust estimator is used uniformly now.
    """
    family = raw_concepts[indices, :]
    median = family.median(dim=0).values
    q75 = family.quantile(0.75, dim=0)
    q25 = family.quantile(0.25, dim=0)
    iqr_scaled = ((q75 - q25) / 1.349).clamp(min=1e-6)
    return median, iqr_scaled


def normalize_concepts(concepts_raw, concept_median, concept_iqr):
    """
    concepts_raw: (..., N_CONCEPTS) raw concepts. Applies population-relative
    robust standardization (median + IQR/1.349, from compute_concept_norm)
    followed by a sigmoid squash to ALL N_CONCEPTS concepts uniformly --
    see compute_concept_norm's docstring for why this replaced the earlier
    band-power-only treatment. Returns (..., N_CONCEPTS).
    """
    return torch.sigmoid((concepts_raw - concept_median) / concept_iqr)


class ConceptBottleneckShallowCNN(nn.Module):
    """
    ShallowCNN backbone -> concept bottleneck -> classifier. Same proven
    backbone (temporal_conv -> spatial_conv -> bn -> square -> pool ->
    log -> global_pool) as models/shallow_cnn.py's ShallowConvNet, feeding
    a small head that predicts the N_CONCEPTS clinical concepts.
    """

    def __init__(self, n_channels, n_classes=2, n_filters=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15, dropout=0.5, residual=True):
        super().__init__()
        self.residual = residual
        self.temporal_conv = nn.Conv2d(1, n_filters, kernel_size=(1, filter_time_length))
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.concept_predictor = nn.Sequential(
            nn.Linear(n_filters, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, N_CONCEPTS), nn.Sigmoid(),
        )

        if residual:
            residual_dim = n_filters // 4
            self.residual_proj = nn.Linear(n_filters, residual_dim)
            classifier_input = N_CONCEPTS + residual_dim
        else:
            classifier_input = N_CONCEPTS
        self.classifier = nn.Linear(classifier_input, n_classes)

    def get_backbone_features(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return self.global_pool(x).flatten(1)  # (batch, n_filters)

    def forward(self, x, lengths=None, intervention=None):

        feat = self.dropout(self.get_backbone_features(x))
        concepts = self.concept_predictor(feat)

        if intervention is not None:
            concepts = concepts.clone()
            for idx, val in intervention.items():
                concepts[:, idx] = val

        if self.residual:
            resid = F.relu(self.residual_proj(feat))
            classifier_input = torch.cat([concepts, resid], dim=-1)
        else:
            classifier_input = concepts

        logits = self.classifier(classifier_input)
        return logits, concepts
