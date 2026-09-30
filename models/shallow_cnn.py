import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    """
    Minimal reimplementation of the temporal-conv -> spatial-conv -> square
    -> pool -> log. (Ref: Schirrmeister et al. 2017).

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
        self.jr_estimator = nn.Sequential(
            nn.Linear(n_filters_spat, 32),
            nn.ReLU(),
            nn.Linear(32, n_channels),
        )

    def forward(self, x, lengths=None):

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

