import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    """
    Minimal reimplementation of the temporal-conv -> spatial-conv -> square
    -> pool -> log EEG decoding architecture (Schirrmeister et al. 2017,
    "ShallowFBCSPNet"). The NMT dataset's own reference pipeline
    (dll-ncai/eeg_pre-diagnostic_screening) benchmarks this same model
    family as "Shallow-CNN" (72% accuracy on NMT) via the braindecode
    library; reimplemented natively here to avoid adding that dependency
    and to run on this project's own data pipeline unchanged.

    Beat every fully-learned cross-channel attention/transformer variant
    tried on real NMT eval accuracy (75.7% vs 69.7% and worse) in this
    project's own controlled comparison. Physics-informed
    constraints are added here directly on top of this proven backbone
    (see spatial_coupling() and ShallowCNNPhysicsLoss below) rather than
    by adding a new attention/graph module, since every attempt at
    fully-learned cross-channel attention so far has underperformed this
    simpler model.

    AdaptiveAvgPool2d at the end (not a fixed-size final conv/linear like
    the original) makes this robust to the batch's padded time length
    varying, since collate_e2e pads to each batch's own max rather than a
    fixed size.
    """

    def __init__(self, n_channels, n_classes=2, n_filters_time=40,
                 filter_time_length=25, n_filters_spat=40,
                 pool_time_length=75, pool_time_stride=15, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_filters_spat = n_filters_spat
        self.n_filters_time = n_filters_time
        self.temporal_conv = nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length))
        self.spatial_conv = nn.Conv2d(n_filters_time, n_filters_spat, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_filters_spat, n_classes)
        # Jansen-Rit-style per-channel excitatory gain estimate, predicted
        # from the same pooled features the classifier sees -- same
        # class-conditional-gain convention used by this project's other
        # physics losses (see ShallowCNNPhysicsLoss below).
        self.jr_estimator = nn.Sequential(
            nn.Linear(n_filters_spat, 32),
            nn.ReLU(),
            nn.Linear(32, n_channels),
        )

    def forward(self, x, lengths=None):
        """
        x: (batch, n_channels, n_samples) -- lengths accepted (unused) so
        this model is drop-in compatible with the same forward_batch-style
        call used elsewhere in this project.

        Returns:
            logits: (batch, n_classes)
            A_pred: (batch, n_channels) -- Jansen-Rit excitatory gain estimates
        """
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_samples)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2  # square nonlinearity -- ShallowFBCSPNet-specific, approximates band-power
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        pooled = self.global_pool(x).flatten(1)  # (batch, n_filters_spat)

        logits = self.classifier(pooled)
        A_pred = 1.5 + 3.0 * torch.sigmoid(self.jr_estimator(pooled))
        return logits, A_pred

    def spatial_coupling(self):
        """
        (n_channels, n_channels) cosine-similarity matrix implied by the
        learned spatial filter -- this is the model's ONLY point of
        cross-channel interaction (spatial_conv spans all 21 channels via
        kernel_size=(n_channels, 1)). Each channel c's weight vector
        across every (spatial filter, temporal filter) pair,
        spatial_conv.weight[:, :, c, 0] flattened, is treated as that
        channel's "role" embedding; channels the model has learned to
        treat similarly across filters get high similarity here.

        Global, not per-sample: spatial_conv's weights are the same for
        every subject (they're model parameters, not a function of the
        input), unlike the attention-based models elsewhere in this
        project where connectivity is computed fresh per input. Called
        directly on the model, not returned from forward().
        """
        W = self.spatial_conv.weight.squeeze(-1)  # (n_filters_spat, n_filters_time, n_channels)
        W_flat = W.permute(2, 0, 1).reshape(self.n_channels, -1)  # (n_channels, n_filters_spat*n_filters_time)
        W_norm = F.normalize(W_flat, dim=-1)
        return W_norm @ W_norm.T  # (n_channels, n_channels), values in [-1, 1]


class ShallowCNNPhysicsLoss(nn.Module):
    """
    Physics/anatomy-informed constraints applied directly to
    ShallowConvNet's existing weights and pooled features -- no new
    attention or graph module, since every fully-learned cross-channel
    attention mechanism tried in this project underperformed this CNN
    backbone on real NMT accuracy.

    1. Electrode-coupling prior: the spatial filter's implied per-channel
       similarity (see ShallowConvNet.spatial_coupling) is encouraged
       toward matching a LEARNABLE per-channel embedding's similarity,
       not a fixed function of physical distance. Measured on real NMT
       data: a fixed exp(-dist/tau) proximity target made accuracy worse
       (73.5% vs 75.7% without it) -- plausibly because it can only
       express "nearby channels should look similar," while EEG
       abnormality often shows up as inter-HEMISPHERIC asymmetry (e.g.
       C3 vs C4, F7 vs F8), a relationship pure physical distance can't
       represent (those pairs are far apart, not close). The learnable
       embedding is initialized from real 10-20 scalp coordinates
       (data/electrode_positions.py) -- so training starts from that same
       physically-grounded prior -- but is a free nn.Parameter, refined
       by gradient descent like any other weight, so the model can learn
       to relate electrode pairs however actually helps classification
       instead of being locked to a proximity-only assumption.
    2. Class-conditional excitatory gain: healthy subjects (label=1,
       this project's "0=pathology first"
       convention) should have higher predicted Jansen-Rit excitatory
       gain than abnormal ones (label=0). Per-sample, unlike the coupling
       term, since it uses the CNN's own per-sample pooled features.
    """

    def __init__(self, channel_positions, lambda_coupling=0.1, lambda_class=0.1):
        super().__init__()
        self.lambda_coupling = lambda_coupling
        self.lambda_class = lambda_class
        # Learnable, not a fixed buffer -- caller must include these
        # parameters in the optimizer (see train_nmt_shallow_cnn.py) or
        # they'll silently never update, same class of bug this project
        # has hit before with untrained auxiliary heads.
        self.channel_embed = nn.Parameter(channel_positions.clone())

    def target_similarity(self):
        emb = F.normalize(self.channel_embed, dim=-1)
        return emb @ emb.T

    def forward(self, spatial_sim, A_pred, labels):
        """
        spatial_sim: (n_channels, n_channels) -- from ShallowConvNet.spatial_coupling()
        A_pred:      (batch, n_channels)
        labels:      (batch,) -- 0=abnormal/pathology, 1=normal/healthy
        """
        l_coupling = F.mse_loss(spatial_sim, self.target_similarity())

        A_mean = A_pred.mean(dim=-1)
        target_A = labels.float() * 3.5 + (1 - labels.float()) * 2.0
        l_class = F.mse_loss(A_mean, target_A)

        return self.lambda_coupling * l_coupling + self.lambda_class * l_class
