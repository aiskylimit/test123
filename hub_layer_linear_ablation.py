"""No-anchor linear ablation — control for V2c_tail.

Tests whether V2c_tail's cross-lingual gain comes from anchors or just
from the learned linear transform. This block is a single d->d linear
with Identity init (output = x at step 0, drifts with training).

No anchors, no retrieval, no temperature. Just a learned reshaping.
"""

import torch
import torch.nn as nn


class EmbHubLinearAblation(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = 0
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def compute_diagnostics(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            x_f = x.float()
            output = torch.nn.functional.linear(
                x_f, self.linear.weight.float(), self.linear.bias.float()
            )
            diff = output - x_f
            diff_norm = diff.norm(dim=-1)
            x_norm = x_f.norm(dim=-1).clamp(min=1e-8)
            norm_ratio = (diff_norm / x_norm).mean().item()

            # How far has the weight drifted from identity?
            eye = torch.eye(self.embedding_dim, device=self.linear.weight.device)
            weight_drift = (self.linear.weight.float() - eye).norm().item()

        return {
            "norm_ratio": norm_ratio,
            "weight_drift": weight_drift,
            "bias_norm": self.linear.bias.float().norm().item(),
        }
