import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert, welch

# This project's ACTUAL channel order (data/tuh_e2e_loader.py's TARGET_CHANNELS)
# -- NOT the ordering in the original proposal, which used a different
# arrangement entirely. Using the wrong ordering wouldn't
# crash; it would silently pull the wrong channels into each "region" (e.g. the
# original proposal's "frontal" indices [0,1,2,3,4,5,6] map to FP1,FP2,F3,F4,C3,
# C4,P3 under THIS project's real ordering -- a mix of frontal/central/parietal
# channels, not frontal at all). Recomputed directly against TARGET_CHANNELS.
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

# CAUEEG (data/caueeg_e2e_loader.py's TARGET_CHANNELS) counterpart -- same 19
# electrodes but a THIRD distinct ordering from both TUH/NMT and ds004504.
# REGIONS/ASYM_PAIRS above were verified to reproduce byte-identical index
# lists when re-derived programmatically from region/pair NAMES (not
# hand-copied) -- REGIONS_CAUEEG uses that same name-based derivation against
# CAUEEG's own channel order, not assumed by analogy to either other dataset.
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
     "theta_alpha_ratio", "delta_alpha_ratio", "dtabr",
     "alpha_peak_freq", "alpha_peak_amplitude",
     "frontal_alpha_plv", "temporal_alpha_plv", "posterior_alpha_plv"]
)
N_CONCEPTS = len(CONCEPT_NAMES)  # 31
N_BAND_POWER_CONCEPTS = len(REGIONS) * len(BANDS)  # 20 -- the only family needing population normalization

# Confirmed dead by the STRICTEST evidence available (a real run checking
# Spearman rank correlation, not just R^2): not statistically significant (p>=0.05) against the
# true concept value -- i.e., no detectable rank-order signal at all, not just a weak
# one. NOTE this is a smaller, more precisely justified list than an earlier R^2<0.2
# cutoff would have given -- several concepts that looked dead by R^2 alone
# (central_alpha, posterior_alpha_asym, delta_alpha_ratio, dtabr, alpha_peak_amplitude)
# turned out to have real, significant Spearman correlation once checked, so they are
# NOT included here.
DEAD_CONCEPT_NAMES = ["frontal_alpha", "temporal_alpha", "parietal_alpha", "frontal_alpha_asym"]
DEAD_CONCEPT_INDICES = [CONCEPT_NAMES.index(n) for n in DEAD_CONCEPT_NAMES]


def _fft_bandpass(x, sfreq, lo, hi):
    """
    x: (..., T) real time-domain signal (numpy). Zero out all frequency
    content outside [lo, hi) Hz via a hard mask in the frequency domain --
    fixed/deterministic, same convention as this project's other analytical
    concept computations (Welch PSD below): no learned filter to fail.
    """
    T = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    mask = (freqs >= lo) & (freqs < hi)
    return np.fft.irfft(X * mask, n=T, axis=-1)


def _plv(sig_a, sig_b):
    """
    Phase-locking value between two REAL, 1-D, already narrowband-filtered
    signals: PLV = |mean_t(exp(i*(phase_a(t) - phase_b(t))))|, using
    scipy.signal.hilbert's analytic signal for instantaneous phase (a
    standard EEG connectivity measure -- see compute_concepts_raw's
    connectivity block for why narrowband filtering matters here).
    Returns a scalar in [0, 1] (1 = perfectly phase-locked).
    """
    phase_a = np.angle(hilbert(sig_a))
    phase_b = np.angle(hilbert(sig_b))
    return float(np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))


def compute_concepts_raw(eeg, sfreq=100, regions=None, asym_pairs=None):
    """
    Compute all N_CONCEPTS (31) clinical EEG concepts analytically.

    eeg: (n_channels, n_samples) -- raw EEG in THIS project's channel order
         and preprocessing (already band-passed 0.5-45Hz, CAR-referenced,
         fixed-scale normalized). sfreq: this project's actual rate (100Hz
         everywhere, NOT the original proposal's 256Hz default -- using the
         wrong sfreq would silently misplace every frequency-band boundary
         in the Welch PSD). regions/asym_pairs: which channel indices count
         as which region/asymmetry pair -- default to REGIONS/ASYM_PAIRS
         (TUH/NMT's 21-channel ordering); pass REGIONS_CAUEEG/ASYM_PAIRS_CAUEEG
         for CAUEEG's different (19-channel) ordering. CONCEPT_NAMES stays
         identical either way (region NAMES, not their channel indices,
         drive it), only which channels a given concept is computed from.

    Returns: (N_CONCEPTS,) float32 array. Family 1 (band power, indices 0-19) is
    RAW log1p(power) here, NOT yet normalized -- population mean/std for
    those 20 values should come from the training set (see
    data/concept_cache.py), same convention as this project's age
    normalization (compute_age_norm), not a per-subject self-relative
    z-score (the original proposal's version, which changes what the
    concept represents: "elevated relative to THIS subject's own other
    bands" is not the same claim as "elevated relative to a normal
    population," which is what these concepts are clinically meant to
    capture). Asymmetry/ratio/spectral-structure concepts (indices 20-27)
    are already self-contained/bounded and returned as-is.
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
    concepts.append(np.tanh(np.log1p(delta_glob / (alpha_glob + 1e-8))))  # delta/alpha ratio
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

    # Family 5: functional connectivity (PLV), same channel pairs as the
    # asymmetry family above (asym_pairs) but a DIFFERENT clinical question:
    # asymmetry asks "which side has more power," this asks "do the two
    # sides oscillate in step" -- reduced interhemispheric alpha-band phase
    # synchrony is a separately well-documented EEG marker of dementia/AD,
    # distinct from (and not derivable from) the power-asymmetry value.
    # Alpha band specifically: PLV needs an approximately narrowband signal
    # for instantaneous phase to be physiologically meaningful -- a
    # broadband Hilbert phase mixes unrelated oscillations together, unlike
    # the band-power family above (which pools raw broadband power, where
    # that mixing isn't a problem). Already bounded in [0, 1] by
    # construction (PLV magnitude), so -- like the asymmetry/ratio families
    # -- no population normalization is needed (see normalize_concepts).
    alpha_sig = _fft_bandpass(eeg, sfreq, *BANDS["alpha"])
    for lo_idx, hi_idx in asym_pairs:
        concepts.append(_plv(alpha_sig[lo_idx], alpha_sig[hi_idx]))

    return np.array(concepts, dtype=np.float32)


def compute_concept_norm(raw_concepts, indices):
    """
    raw_concepts: (n_subjects, N_CONCEPTS) tensor of ALL subjects' RAW concepts
    (compute_concepts_raw's output, cached per-subject). indices: which
    subjects (e.g. train_indices) to compute population statistics from.
    Returns (median, iqr_scaled), each (N_BAND_POWER_CONCEPTS,) -- family
    1 (band power) only; families 2-4 are already self-contained/bounded
    and don't need this.

    ROBUST (median + IQR) statistics, not mean/std -- an earlier version
    of this used mean/std (same convention as this project's age
    regression normalization) and it was a real, confirmed bug: a real
    run showed EVERY beta-band concept catastrophically failing (R^2 down
    to -79.9), traced to population std being ~11x LARGER than the
    entire P5-P95 raw value range for parietal_beta -- a small number of
    extreme outliers (very plausibly EMG/muscle-artifact contamination, a
    well-known source of excess beta-range power) inflated the non-robust
    std estimate enough to crush z-scores for the vast majority of
    subjects into a sliver near 0, which sigmoid then mapped to a
    near-constant ~0.5 regardless of real underlying differences. Raw
    beta's coefficient of variation was actually comparable to or higher
    than alpha's (which normalized fine), confirming this was a
    normalization artifact, not a genuine absence of signal. IQR/1.349
    (the constant that makes IQR-based scale comparable to std under a
    normal distribution, same convention as MAD-based robust z-scores)
    is far less sensitive to a handful of contaminated recordings.
    """
    family1 = raw_concepts[indices, :N_BAND_POWER_CONCEPTS]
    median = family1.median(dim=0).values
    q75 = family1.quantile(0.75, dim=0)
    q25 = family1.quantile(0.25, dim=0)
    iqr_scaled = ((q75 - q25) / 1.349).clamp(min=1e-6)
    return median, iqr_scaled


def normalize_concepts(concepts_raw, band_power_median, band_power_iqr):
    """
    concepts_raw: (..., N_CONCEPTS) raw concepts. Applies population sigmoid
    normalization to family 1 (indices 0:N_BAND_POWER_CONCEPTS) using
    the ROBUST stats from compute_concept_norm (median + IQR/1.349, not
    mean/std -- see that function's docstring for why); leaves families
    2-4 (already bounded in [0,1] or [0,1) by construction) unchanged.
    Returns (..., N_CONCEPTS).
    """
    family1 = concepts_raw[..., :N_BAND_POWER_CONCEPTS]
    family1_norm = torch.sigmoid((family1 - band_power_median) / band_power_iqr)
    return torch.cat([family1_norm, concepts_raw[..., N_BAND_POWER_CONCEPTS:]], dim=-1)


class ConceptBottleneckShallowCNN(nn.Module):
    """
    ShallowCNN backbone -> concept bottleneck -> classifier. Same proven
    backbone (temporal_conv -> spatial_conv -> bn -> square -> pool ->
    log -> global_pool) as models/shallow_cnn.py's ShallowConvNet, feeding
    a small head that predicts the N_CONCEPTS clinical concepts, which the
    classifier then consumes INSTEAD OF the raw CNN features directly --
    this routing-through-concepts is what makes intervention (a clinician
    overriding a wrong concept prediction and re-running the classifier)
    meaningful; if the classifier saw raw features too, an intervention
    on a concept could be ignored by the classifier via a shortcut through
    the untouched raw path.

    Small residual path (n_filters//4 dims, bypassing the bottleneck) is
    included to absorb concept incompleteness without destroying accuracy
    entirely -- but note this directly trades off against intervention
    meaningfulness (concepts no longer fully determine the output), which
    is worth checking empirically (the leakage test) rather than assuming.
    """

    def __init__(self, n_channels, n_classes=2, n_filters=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15, dropout=0.5, residual=True,
                 dead_concept_indices=None):
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

        # fixed (non-learnable) hard mask for confirmed-dead concepts. Defaults to
        # DEAD_CONCEPT_INDICES (TUH's own confirmed-dead list, see its docstring) for
        # backward compatibility, but this is an empirical finding specific to TUH's
        # population -- pass dead_concept_indices=[] (or a separately-verified list)
        # for a different dataset rather than assuming TUH's findings transfer.
        dead_concept_indices = DEAD_CONCEPT_INDICES if dead_concept_indices is None else dead_concept_indices
        dead_mask = torch.ones(N_CONCEPTS)
        dead_mask[dead_concept_indices] = 0.0
        self.register_buffer("dead_mask", dead_mask)

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
        """
        x: (batch, n_ch, n_samples)
        intervention: optional dict {concept_idx: value} -- value can be a
            python scalar (applies to every sample in the batch) or a
            (batch,) tensor (per-sample override, e.g. substituting each
            sample's own true concept value).
        Returns: logits (batch, n_classes), concepts (batch, N_CONCEPTS)
        """
        feat = self.dropout(self.get_backbone_features(x))
        concepts = self.concept_predictor(feat)

        if intervention is not None:
            concepts = concepts.clone()
            for idx, val in intervention.items():
                concepts[:, idx] = val

        # dead-concept mask applies to what the CLASSIFIER sees, not to the returned
        # `concepts` (which stays the model's actual belief, for R^2/Spearman/reporting
        # purposes).
        gated_concepts = concepts * self.dead_mask

        if self.residual:
            resid = F.relu(self.residual_proj(feat))
            classifier_input = torch.cat([gated_concepts, resid], dim=-1)
        else:
            classifier_input = gated_concepts

        logits = self.classifier(classifier_input)
        return logits, concepts
