"""Experimental numeric forecasting module; independent of EmberGPT weights.

BSQ follows https://arxiv.org/abs/2406.07548. The supervised forecasting
objective and optional continuous feature path are experimental adaptations.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, bits: int, temperature: float = 10.0):
        super().__init__()
        if not 2 <= bits <= 32:
            raise ValueError("bits must be between 2 and 32")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.bits, self.temperature = bits, temperature

    def forward(self, latent: torch.Tensor):
        if latent.shape[-1] != self.bits:
            raise ValueError("latent width must equal bits")
        unit = F.normalize(latent, dim=-1, eps=1e-6)
        hard = torch.where(unit >= 0, 1.0, -1.0) / math.sqrt(self.bits)
        code = unit + (hard - unit).detach()  # Straight-through gradient.
        commitment = F.mse_loss(unit, hard.detach())
        probability = torch.sigmoid(2 * self.temperature * unit / math.sqrt(self.bits))
        probability = probability.reshape(-1, self.bits)

        def entropy(p):
            p = p.clamp(1e-6, 1 - 1e-6)
            return -(p * p.log() + (1 - p) * (1 - p).log())

        # Encourage decisive codes and use of both signs across the batch.
        # Marginal bit entropy approximates, rather than equals, joint entropy.
        entropy_loss = entropy(probability).mean() - entropy(probability.mean(0)).mean()
        return code, commitment, entropy_loss


class BSQForecaster(nn.Module):
    """Encode each numeric observation, then model its past sequence with a GRU."""

    def __init__(self, feature_count: int, bits: int = 16, hidden_size: int = 32,
                 mode: str = "hybrid"):
        super().__init__()
        if mode not in {"continuous", "bsq", "hybrid"}:
            raise ValueError("unknown representation mode")
        if feature_count < 1 or hidden_size < 1:
            raise ValueError("feature_count and hidden_size must be positive")
        self.feature_count, self.mode = feature_count, mode
        self.encoder = nn.Sequential(nn.Linear(feature_count, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, bits))
        self.quantizer = BinarySphericalQuantizer(bits)
        self.decoder = nn.Linear(bits, feature_count)
        self.temporal = nn.GRU(bits + (feature_count if mode == "hybrid" else 0),
                               hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor):
        if features.ndim != 3 or features.shape[-1] != self.feature_count or features.shape[1] < 1:
            raise ValueError("features must have shape [batch, history, feature_count]")
        latent = self.encoder(features)
        if self.mode == "continuous":
            code = latent
            commitment = entropy_loss = latent.new_zeros(())
        else:
            code, commitment, entropy_loss = self.quantizer(latent)
        reconstruction = F.mse_loss(self.decoder(code), features)
        temporal_input = torch.cat((code, features), dim=-1) if self.mode == "hybrid" else code
        _, state = self.temporal(temporal_input)
        prediction = self.head(state[-1]).squeeze(-1)
        return prediction, {"reconstruction": reconstruction, "commitment": commitment,
                            "entropy": entropy_loss}
