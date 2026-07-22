import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch

# This project's ACTUAL channel order (data/tuh_e2e_loader.py's TARGET_CHANNELS,
# identical to NMT's channel_names) -- NOT the ordering in the original proposal,
# which used a different arrangement entirely. Using the wrong ordering wouldn't
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
     "alpha_peak_freq", "alpha_peak_amplitude"]
)
N_CONCEPTS = len(CONCEPT_NAMES)  # 28
N_BAND_POWER_CONCEPTS = len(REGIONS) * len(BANDS)  # 20 -- the only family needing population normalization

# Beta concepts predicted well: real run (train_tuh_concept_bottleneck.py) showed
# EVERY beta-band concept failing (R^2 all negative, worst -2.344) while alpha/ratio
# concepts worked (up to R^2=0.511) -- see ConceptBottleneckGNN's docstring for the
# fix. Indices computed programmatically from CONCEPT_NAMES, not hardcoded.
BETA_CONCEPT_INDICES = [i for i, name in enumerate(CONCEPT_NAMES) if name.endswith("_beta")]
NON_BETA_CONCEPT_INDICES = [i for i in range(N_CONCEPTS) if i not in BETA_CONCEPT_INDICES]
# For reassembling a (batch, 28) tensor from [non_beta_preds, beta_preds] (that
# concatenation order) back into CONCEPT_NAMES' true order: concepts_unordered[:,
# INVERSE_PERM] gives the correctly-ordered (batch, 28) tensor. Fully differentiable
# (advanced indexing), no in-place assignment ambiguity.
_CONCAT_ORDER = NON_BETA_CONCEPT_INDICES + BETA_CONCEPT_INDICES
_inverse_perm = [0] * N_CONCEPTS
for _pos, _orig_idx in enumerate(_CONCAT_ORDER):
    _inverse_perm[_orig_idx] = _pos
INVERSE_PERM = torch.tensor(_inverse_perm, dtype=torch.long)

# Confirmed dead by the STRICTEST evidence available (check_concept_rank_correlation.py,
# a real run): Spearman correlation not statistically significant (p>=0.05) against the
# true concept value -- i.e., no detectable rank-order signal at all, not just a weak
# one. NOTE this is a smaller, more precisely justified list than an earlier R^2<0.2
# cutoff would have given -- several concepts that looked dead by R^2 alone
# (central_alpha, posterior_alpha_asym, delta_alpha_ratio, dtabr, alpha_peak_amplitude)
# turned out to have real, significant Spearman correlation once checked, so they are
# NOT included here.
DEAD_CONCEPT_NAMES = ["frontal_alpha", "temporal_alpha", "parietal_alpha", "frontal_alpha_asym"]
DEAD_CONCEPT_INDICES = [CONCEPT_NAMES.index(n) for n in DEAD_CONCEPT_NAMES]


def fft_bandpass(x, sfreq, lo, hi):
    """
    x: (..., T) time-domain signal. Zero out all frequency content outside
    [lo, hi) Hz via a hard mask in the frequency domain, then inverse-
    transform back to the time domain. Fixed (not learned), fully
    differentiable via torch.fft -- deliberately NOT a learned filter bank
    (this project already tried and removed one, in a different model,
    for a different purpose, because it added complexity without earning
    its keep; a fixed filter has no equivalent failure mode since there's
    nothing for it to fail to learn).
    """
    T = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sfreq).to(x.device)
    mask = ((freqs >= lo) & (freqs < hi)).to(X.real.dtype)
    return torch.fft.irfft(X * mask, n=T, dim=-1)


def compute_concepts_raw(eeg, sfreq=100):
    """
    Compute all 28 clinical EEG concepts analytically.

    eeg: (n_channels, n_samples) -- raw EEG in THIS project's channel order
         and preprocessing (already band-passed 0.5-45Hz, CAR-referenced,
         fixed-scale normalized). sfreq: this project's actual rate (100Hz
         everywhere, NOT the original proposal's 256Hz default -- using the
         wrong sfreq would silently misplace every frequency-band boundary
         in the Welch PSD).

    Returns: (28,) float32 array. Family 1 (band power, indices 0-19) is
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
    n_ch, n_samples = eeg.shape
    freqs, psd = welch(eeg, fs=sfreq, nperseg=min(512, n_samples))  # psd: (n_ch, n_freqs)

    def band_power(psd_row, lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return np.log1p(psd_row[mask].mean()) if mask.any() else 0.0

    concepts = []
    for region, ch_idx in REGIONS.items():
        region_psd = psd[ch_idx].mean(axis=0)
        for lo, hi in BANDS.values():
            concepts.append(band_power(region_psd, lo, hi))
    # concepts[0:20] = raw (unnormalized) family 1

    alpha_lo, alpha_hi = BANDS["alpha"]
    alpha_mask = (freqs >= alpha_lo) & (freqs < alpha_hi)

    def alpha_power_ch(ch_idx):
        return psd[ch_idx, alpha_mask].mean() + 1e-8

    for lo_idx, hi_idx in [(F3_IDX, F4_IDX), (T3_IDX, T4_IDX), (O1_IDX, O2_IDX)]:
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

    occ_psd = psd[REGIONS["occipital"]].mean(axis=0)
    peak_range = (freqs >= 6) & (freqs <= 14)
    if peak_range.any() and occ_psd[peak_range].max() > 0:
        peak_freq = freqs[peak_range][occ_psd[peak_range].argmax()]
    else:
        peak_freq = 8.0
    concepts.append(np.clip((peak_freq - 6) / (14 - 6), 0.0, 1.0))

    peak_amp = occ_psd[alpha_mask].max() if alpha_mask.any() else 0.0
    concepts.append(np.tanh(np.log1p(peak_amp)))

    return np.array(concepts, dtype=np.float32)


def compute_concept_norm(raw_concepts, indices):
    """
    raw_concepts: (n_subjects, 28) tensor of ALL subjects' RAW concepts
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
    concepts_raw: (..., 28) raw concepts. Applies population sigmoid
    normalization to family 1 (indices 0:N_BAND_POWER_CONCEPTS) using
    the ROBUST stats from compute_concept_norm (median + IQR/1.349, not
    mean/std -- see that function's docstring for why); leaves families
    2-4 (already bounded in [0,1] or [0,1) by construction) unchanged.
    Returns (..., 28).
    """
    family1 = concepts_raw[..., :N_BAND_POWER_CONCEPTS]
    family1_norm = torch.sigmoid((family1 - band_power_median) / band_power_iqr)
    return torch.cat([family1_norm, concepts_raw[..., N_BAND_POWER_CONCEPTS:]], dim=-1)


class ConceptBottleneckShallowCNN(nn.Module):
    """
    ShallowCNN backbone -> concept bottleneck -> classifier. Same proven
    backbone (temporal_conv -> spatial_conv -> bn -> square -> pool ->
    log -> global_pool) as models/shallow_cnn.py's ShallowConvNet, feeding
    a small head that predicts the 28 clinical concepts, which the
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

        # fixed (non-learnable) hard mask for the 4 confirmed-dead concepts (see
        # DEAD_CONCEPT_INDICES's docstring) -- same mechanism as ConceptBottleneckGNN,
        # kept in sync so the two backbones are a fair apples-to-apples comparison.
        dead_mask = torch.ones(N_CONCEPTS)
        dead_mask[DEAD_CONCEPT_INDICES] = 0.0
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
        # purposes) -- see ConceptBottleneckGNN.forward for the same pattern.
        gated_concepts = concepts * self.dead_mask

        if self.residual:
            resid = F.relu(self.residual_proj(feat))
            classifier_input = torch.cat([gated_concepts, resid], dim=-1)
        else:
            classifier_input = gated_concepts

        logits = self.classifier(classifier_input)
        return logits, concepts


class ConceptBottleneckGNN(nn.Module):
    """
    ConceptBottleneckShallowCNN's concept-routed classification, with two
    targeted architectural fixes informed by that model's real result
    (81.88% accuracy vs. 85.14% plain CNN; mean concept R^2=-0.142, with
    a sharp split between well-predicted alpha/ratio concepts, up to
    R^2=0.511, and completely failed beta-band and asymmetry concepts):

    1. GRAPH-COUPLING MAIN BACKBONE (matching models/graph_coupling_cnn.py's
       ShallowGNN exactly: get_channel_features -> per-sample cosine-
       similarity adjacency -> n_hops rounds of propagation -> collapse)
       instead of a single fixed spatial_conv. This is a necessary
       precondition for the 3 asymmetry concepts (by definition cross-
       channel: F3 vs F4 etc) to be predictable at all -- the old
       spatial_conv architecture collapsed all 21 channels into one fixed
       linear combination BEFORE concept_predictor ever saw the data, so
       "how different is F3 from F4" was structurally unrecoverable by
       any downstream head, regardless of how it was trained. NOTE:
       check_asymmetry_signal.py (an earlier, separate check in this
       project) already found near-zero asymmetry-label correlation even
       WITH per-channel access to the CNN's own features -- so this fixes
       the architectural bottleneck but isn't expected to make asymmetry
       strongly predictive, just no longer impossible by construction.

    2. DEDICATED BETA-BAND BRANCH: every beta-band concept failed
       completely in the real run, consistent with beta (much lower power
       than delta/theta/alpha in raw EEG) being crowded out of the shared
       broadband pooled feature -- nothing in the original architecture
       isolated any one band before the square+log+pool nonlinearity
       mixed everything together, and nothing in the loss specifically
       rewarded preserving beta resolution when delta/theta/alpha already
       dominated. Fixed via fft_bandpass (zero-parameter, deterministic
       frequency-domain isolation, not a learned filter) applied to the
       raw signal for the beta band specifically, feeding its own small,
       separate temporal_conv+spatial_conv+pool branch whose output
       predicts ONLY the 5 beta concepts (one per region) -- the other 23
       concepts still come from the (now graph-coupled) main branch. This
       keeps the two fixes cleanly separable: the beta branch does NOT
       use graph coupling (plain spatial_conv collapse, matching the
       original architecture's simplicity), isolating which of the two
       changes is responsible for any observed improvement.

    3. LEARNABLE PER-CONCEPT GATE -- tried and removed. A sigmoid-bounded
       scalar per concept, multiplying the concept vector before the
       classifier, was meant to let the classification loss decide which
       concepts matter (rather than hand-pruning by offline R^2/Spearman).
       Across every real run, the learned gate values never strayed far
       from the neutral 0.5 start (roughly 0.49-0.57) -- it wasn't
       learning a strong signal, just adding parameters and (per one run)
       a real multi-task side effect: it correctly identified beta
       concepts as most classification-relevant, but the extra
       classification gradient flowing through the branch it favored
       measurably hurt that branch's OWN concept-prediction R^2. Removed;
       only the fixed dead-concept mask (below) remains.

    4. FIXED DEAD-CONCEPT MASK (dead_mask, a non-learnable buffer): hard-
       zeroes the 4 concepts confirmed to have NO statistically
       significant Spearman rank correlation with their true value at
       all (see DEAD_CONCEPT_INDICES's docstring), so the classifier
       never sees them and the concept loss doesn't waste gradient
       trying to fit pure noise for them. Deliberately a much smaller,
       more conservative list than an R^2<0.2 cutoff would give -- several
       concepts that looked dead by R^2 alone turned out to have real,
       significant Spearman correlation, so they stay in and are left to
       the learned gate rather than being hard-excluded.
    """

    def __init__(self, n_channels, n_classes=2, n_filters_time=40, filter_time_length=25,
                 n_filters_spat=40, pool_time_length=75, pool_time_stride=15,
                 n_hops=2, sfreq=100, beta_band=(13, 30), beta_filters=16,
                 dropout=0.5, residual=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_hops = n_hops
        self.sfreq = sfreq
        self.beta_band = beta_band
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.residual = residual

        # main branch: graph-coupling backbone, matching ShallowGNN exactly.
        self.temporal_conv = nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length))
        self.hop_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.01)) for _ in range(n_hops)
        ])
        self.channel_collapse = nn.Linear(n_channels * n_filters_time, n_filters_spat)
        self.bn = nn.BatchNorm1d(n_filters_spat)
        self.dropout = nn.Dropout(dropout)

        # beta branch: small, dedicated, fed by a fixed FFT bandpass isolate.
        self.beta_temporal_conv = nn.Conv2d(1, beta_filters, kernel_size=(1, filter_time_length))
        self.beta_spatial_conv = nn.Conv2d(beta_filters, beta_filters, kernel_size=(n_channels, 1), bias=False)
        self.beta_bn = nn.BatchNorm2d(beta_filters)
        self.beta_pool = nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride))
        self.beta_global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.main_concept_head = nn.Sequential(
            nn.Linear(n_filters_spat, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, len(NON_BETA_CONCEPT_INDICES)),
        )
        self.beta_concept_head = nn.Linear(beta_filters, len(BETA_CONCEPT_INDICES))

        # fixed (non-learnable) hard mask for the 4 confirmed-dead concepts (see
        # DEAD_CONCEPT_INDICES's docstring) -- a learnable per-concept gate was tried
        # here too, but removed after real runs showed it never moved far from neutral
        # and measurably hurt beta's own concept-prediction R^2 (see class docstring).
        dead_mask = torch.ones(N_CONCEPTS)
        dead_mask[DEAD_CONCEPT_INDICES] = 0.0
        self.register_buffer("dead_mask", dead_mask)

        if residual:
            residual_dim = n_filters_spat // 4
            self.residual_proj = nn.Linear(n_filters_spat, residual_dim)
            classifier_input = N_CONCEPTS + residual_dim
        else:
            classifier_input = N_CONCEPTS
        self.classifier = nn.Linear(classifier_input, n_classes)

    def get_channel_features(self, x):
        """Same pattern as ShallowGNN.get_channel_features (models/graph_coupling_cnn.py)."""
        x_4d = x.unsqueeze(1)  # (batch, 1, n_ch, T)
        t_feat = self.temporal_conv(x_4d)
        t_feat = t_feat ** 2
        t_feat = F.avg_pool2d(
            t_feat, kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride),
        )
        t_feat = torch.log(torch.clamp(t_feat, min=1e-6))
        t_feat = t_feat.mean(dim=-1)  # (batch, n_filters_time, n_ch)
        return t_feat.permute(0, 2, 1)  # (batch, n_ch, n_filters_time)

    def per_sample_adjacency(self, h):
        """Cosine-similarity Gram matrix, recomputed fresh at each hop -- see
        ShallowGNN.per_sample_adjacency (models/graph_coupling_cnn.py)."""
        h_norm = F.normalize(h, dim=-1)
        return h_norm @ h_norm.transpose(-2, -1)

    def forward(self, x, lengths=None, intervention=None):
        """
        x: (batch, n_channels, n_samples)
        Returns: logits (batch, n_classes), concepts (batch, N_CONCEPTS)
        """
        # main graph-coupling branch -> 23 non-beta concepts
        h = self.get_channel_features(x)
        for k in range(self.n_hops):
            A = self.per_sample_adjacency(h)
            h = h + self.hop_alphas[k] * torch.einsum("bij,bjf->bif", A, h)
            if k < self.n_hops - 1:
                h = F.elu(h)
        h_flat = h.reshape(h.shape[0], -1)
        main_feat = self.channel_collapse(h_flat)
        main_feat = self.bn(main_feat)
        main_feat = F.relu(main_feat)
        main_feat = self.dropout(main_feat)

        # dedicated beta branch -> 5 beta concepts
        x_beta = fft_bandpass(x, self.sfreq, *self.beta_band)
        xb = x_beta.unsqueeze(1)
        xb = self.beta_temporal_conv(xb)
        xb = self.beta_spatial_conv(xb)
        xb = self.beta_bn(xb)
        xb = xb ** 2
        xb = self.beta_pool(xb)
        xb = torch.log(torch.clamp(xb, min=1e-6))
        beta_feat = self.beta_global_pool(xb).flatten(1)

        main_concepts_raw = self.main_concept_head(main_feat)  # (batch, 23)
        beta_concepts_raw = self.beta_concept_head(beta_feat)  # (batch, 5)
        concepts_unordered = torch.cat([main_concepts_raw, beta_concepts_raw], dim=-1)  # (batch, 28)
        concepts = torch.sigmoid(concepts_unordered[:, INVERSE_PERM.to(x.device)])  # reorder to CONCEPT_NAMES order

        if intervention is not None:
            concepts = concepts.clone()
            for idx, val in intervention.items():
                concepts[:, idx] = val

        # dead-concept mask applies to what the CLASSIFIER sees, not to the returned
        # `concepts` (which stays the model's actual belief, for R^2/Spearman/reporting
        # purposes) -- an intervention on a dead-masked concept should have little/no
        # effect on logits, which is the informative, expected behavior, not a bug.
        gated_concepts = concepts * self.dead_mask

        if self.residual:
            resid = F.relu(self.residual_proj(main_feat))
            classifier_input = torch.cat([gated_concepts, resid], dim=-1)
        else:
            classifier_input = gated_concepts

        logits = self.classifier(classifier_input)
        return logits, concepts
