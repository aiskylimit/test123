"""EmbHub V2c family — top-k anchor concat (V2c, V2c+tail, V2c+buckets).

V2c:         output = Linear([x ; w1*a_i1 ; ... ; wk*a_ik])
V2c+tail:    output = Linear([x ; w1*a_i1 ; ... ; wk*a_ik ; mixture_rest])
V2c+buckets: output = Linear([x ; w1*a_i1 ; ... ; wk*a_ik ; bucket_1 ; ... ; bucket_B])

Top-k by cosine similarity. Hard routing — gradient flows only to selected anchors
(+tail/+buckets variants keep non-top anchors alive via aggregated slots).

Weighting options for top-k slots:
- "raw_softmax" (Option B, default): raw softmax weights (sum < 1, preserves confidence)
- "renormalized" (Option A): renormalized top-k weights (sum = 1)
- "none" (Option C): unweighted (multiplier = 1)

Safe init: Linear weight = [I | 0 | 0 | ...], bias = 0.
NOTE: safe init makes contribution ~0 but does NOT neutralize routing —
top-k selection is hard from step 0. Use +tail to keep non-top anchors alive.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbHubV2TopK(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 1000,
        top_k: int = 10,
        weighting: str = "raw_softmax",
        tail_mode: str = "none",
        num_buckets: int = 10,
        reference_weight: torch.Tensor = None,
    ):
        """
        Args:
            embedding_dim: dimension of input vectors
            num_embeddings: number of anchor vectors (N)
            top_k: number of anchors to select per token
            weighting: "raw_softmax" (B), "renormalized" (A), or "none" (C)
            tail_mode: "none" (V2c), "tail" (V2c+tail), "buckets" (V2c+buckets)
            num_buckets: number of similarity-rank buckets (only for tail_mode="buckets")
            reference_weight: tensor to match init stats from
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.top_k = min(top_k, num_embeddings)
        self.weighting = weighting
        self.tail_mode = tail_mode
        self.num_buckets = num_buckets

        assert weighting in ("raw_softmax", "renormalized", "none"), \
            f"weighting must be 'raw_softmax', 'renormalized', or 'none', got '{weighting}'"
        assert tail_mode in ("none", "tail", "buckets"), \
            f"tail_mode must be 'none', 'tail', or 'buckets', got '{tail_mode}'"

        self.hub_embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))

        if tail_mode == "none":
            num_slots = self.top_k
        elif tail_mode == "tail":
            num_slots = self.top_k + 1
        else:
            num_slots = self.top_k + num_buckets
        self.num_slots = num_slots

        self.linear_out = nn.Linear((1 + num_slots) * embedding_dim, embedding_dim)

        self._init_weights(reference_weight)

    def _init_weights(self, reference_weight: torch.Tensor = None) -> None:
        if reference_weight is not None:
            std = reference_weight.std().item()
            mean = reference_weight.mean().item()
            self.hub_embeddings.data.normal_(mean=mean, std=std)
        else:
            nn.init.xavier_uniform_(self.hub_embeddings.data)

        d = self.embedding_dim
        with torch.no_grad():
            self.linear_out.weight[:, :d].copy_(torch.eye(d))
            self.linear_out.weight[:, d:].zero_()
            self.linear_out.bias.zero_()

    def _compute_selection(self, x: torch.Tensor):
        q = F.normalize(x, dim=-1)
        k = F.normalize(self.hub_embeddings, dim=-1)
        scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)

        topk_weights, topk_indices = weights.topk(self.top_k, dim=-1)

        return weights, logits, topk_weights, topk_indices

    def _build_topk_slots(self, topk_weights, topk_indices):
        """Build weighted top-k anchor slots. Returns (*, top_k * d)."""
        B_seq = topk_weights.shape[:-1]
        anchors = self.hub_embeddings[topk_indices]

        if self.weighting == "raw_softmax":
            weighted = anchors * topk_weights.unsqueeze(-1)
        elif self.weighting == "renormalized":
            renorm = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            weighted = anchors * renorm.unsqueeze(-1)
        else:
            weighted = anchors

        return weighted.reshape(*B_seq, self.top_k * self.embedding_dim)

    def _build_tail_slot(self, weights, topk_indices):
        """Build aggregated rest slot from non-top-k anchors. Returns (*, d)."""
        mask = torch.ones_like(weights, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, False)

        rest_weights = weights * mask.to(weights.dtype)
        rest_sum = rest_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        rest_weights_norm = rest_weights / rest_sum

        return rest_weights_norm @ self.hub_embeddings

    def _build_bucket_slots(self, weights, logits, topk_indices):
        """Build bucket slots from non-top-k anchors ranked by similarity. Returns (*, num_buckets * d)."""
        B_seq = weights.shape[:-1]
        N = self.num_embeddings
        k = self.top_k

        mask = torch.ones_like(weights, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, False)

        rest_weights = weights * mask.to(weights.dtype)
        rest_logits = logits.masked_fill(~mask, float('-inf'))

        sorted_indices = rest_logits.argsort(dim=-1, descending=True)
        num_rest = N - k
        bucket_size = max(1, num_rest // self.num_buckets)

        bucket_slots = []
        for b in range(self.num_buckets):
            start = b * bucket_size
            end = min((b + 1) * bucket_size, num_rest) if b < self.num_buckets - 1 else num_rest
            bucket_idx = sorted_indices[..., start:end]

            bucket_w = torch.gather(rest_weights, -1, bucket_idx)
            bucket_w_sum = bucket_w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            bucket_w_norm = bucket_w / bucket_w_sum

            bucket_anchors = self.hub_embeddings[bucket_idx]
            bucket_vec = (bucket_w_norm.unsqueeze(-1) * bucket_anchors).sum(dim=-2)
            bucket_slots.append(bucket_vec)

        return torch.cat(bucket_slots, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, logits, topk_weights, topk_indices = self._compute_selection(x)

        topk_slots = self._build_topk_slots(topk_weights, topk_indices)

        parts = [x, topk_slots]

        if self.tail_mode == "tail":
            tail = self._build_tail_slot(weights, topk_indices)
            parts.append(tail)
        elif self.tail_mode == "buckets":
            buckets = self._build_bucket_slots(weights, logits, topk_indices)
            parts.append(buckets)

        return self.linear_out(torch.cat(parts, dim=-1))

    def compute_diagnostics(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            x_f = x.float()
            q = F.normalize(x_f, dim=-1)
            k = F.normalize(self.hub_embeddings.float(), dim=-1)
            scale = self.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)

            topk_weights, topk_indices = weights.topk(self.top_k, dim=-1)

            output = self.forward(x)
            contribution = output.float() - x_f

            entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=-1)
            uniform_entropy = math.log(self.num_embeddings)

            anchor_mass = weights.mean(dim=tuple(range(weights.dim() - 1)))
            top10_mass = anchor_mass.topk(min(10, self.num_embeddings)).values.sum().item()
            uniform_mass = 1.0 / self.num_embeddings
            dead_fraction = (anchor_mass < 0.1 * uniform_mass).float().mean().item()

            topk_mass_total = topk_weights.sum(dim=-1).mean().item()

            contrib_norm = contribution.norm(dim=-1)
            token_norm = x_f.norm(dim=-1)
            norm_ratio = (contrib_norm / token_norm.clamp(min=1e-8)).mean().item()

            d = self.embedding_dim
            w_anchor_norm = self.linear_out.weight[:, d:].float().norm().item()

            n_sample = min(100, self.num_embeddings)
            sampled = F.normalize(self.hub_embeddings[:n_sample].float(), dim=-1)
            pairwise_cos = (sampled @ sampled.T).triu(diagonal=1)
            mask_tri = torch.triu(torch.ones_like(pairwise_cos, dtype=torch.bool), diagonal=1)
            mean_pairwise_cos = pairwise_cos[mask_tri].mean().item()

            return {
                "logit_std": logits.std().item(),
                "entropy_mean": entropy.mean().item(),
                "entropy_std": entropy.std().item(),
                "uniform_entropy": uniform_entropy,
                "effective_anchors": math.exp(entropy.mean().item()),
                "max_weight_mean": weights.max(dim=-1).values.mean().item(),
                "logit_scale": scale.item(),
                "top10_anchor_mass": top10_mass,
                "dead_anchor_fraction": dead_fraction,
                "topk_mass_total": topk_mass_total,
                "norm_ratio": norm_ratio,
                "w_anchor_norm": w_anchor_norm,
                "anchor_pairwise_cosine": mean_pairwise_cos,
            }
