"""Two-branch audio-visual deepfake detector (work in progress)."""

import torch
import torch.nn as nn
import timm

FUSED_DIM = 256


class AudioBranch(nn.Module):
    """Small CNN over log-mel spectrograms (B x 1 x 128 x T)."""

    def __init__(self, out_dim=FUSED_DIM):
        super().__init__()
        # TODO: proper conv blocks, this is a first sketch
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, spec):
        x = self.net(spec).flatten(1)
        return self.fc(x)


class MultimodalDetector(nn.Module):
    """EfficientNet-B0 visual branch + audio CNN, late fusion, two heads."""

    def __init__(self, out_dim=FUSED_DIM):
        super().__init__()
        # timm gives the pre-trained backbone without its classifier head
        self.visual = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.audio = AudioBranch(out_dim)
        # TODO: check feature dims (b0 outputs 1280)
        self.fuse = nn.Sequential(
            nn.Linear(1280 + out_dim, out_dim), nn.ReLU(), nn.Dropout(0.3),
        )
        self.fake_head = nn.Linear(out_dim, 2)      # real / fake
        self.category_head = nn.Linear(out_dim, 4)  # RVRA / FVRA / RVFA / FVFA

    def forward(self, frames, spec):
        v = self.visual(frames)  # (B, 1280)
        a = self.audio(spec)     # (B, out_dim)
        z = self.fuse(torch.cat([v, a], dim=1))
        return self.fake_head(z), self.category_head(z)
