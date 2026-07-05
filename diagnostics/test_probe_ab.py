"""Probe A (anchor weight overlap) + Probe B (post-hub embedding cosine gap).

Probe A: For translation pairs, how similar are their anchor-weight
distributions? (JS divergence). Hub-only.

Probe B: For translation pairs, how similar are their post-hub
representations compared to random pairs? Uses the model's actual output
at the anchor layer (hub hook fires normally during forward pass).
For baseline models, this is the raw embedding. Single-token words only.

Usage (standalone):
    python diagnostics/test_probe_ab.py \
        --checkpoint /path/to/checkpoint \
        --output temp/probe_ab_results.json
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from scipy import stats

from diagnostics.test_utils import (
    LANGS,
    NON_EN_LANGS,
    filter_loanwords_per_pair,
    get_hub_input,
    get_representation_at_anchor_layer,
    get_tokenizer,
    load_checkpoint,
    load_translations,
)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = (p * (p.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    kl_qm = (q * (q.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    return (0.5 * (kl_pm + kl_qm)).item()


def topk_jaccard(w1, w2, k=10):
    s1 = set(w1.topk(k).indices.tolist())
    s2 = set(w2.topk(k).indices.tolist())
    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0


def get_anchor_weights_for_word(hub, hub_type, embedding_weight, tokenizer, word, device):
    """Get anchor-weight distribution for a single word.

    Uses raw token embedding (bypasses hub hook) to compute anchor weights.

    Returns:
        (weights_mean, n_tokens, token_emb_mean) or (None, n_tokens, token_emb_mean)
    """
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    token_ids = torch.tensor([ids], device=device)

    with torch.no_grad():
        token_emb = F.embedding(token_ids, embedding_weight).float()
        token_emb_mean = token_emb.squeeze(0).mean(dim=0)

        if hub is None:
            return None, len(ids), token_emb_mean

        x = token_emb  # (1, n_tokens, d)

        if hub_type == "v2_additive":
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)

        elif hub_type == "v3":
            if hub.num_heads == 1:
                q = F.normalize(x, dim=-1)
                k = F.normalize(hub.anchor_keys.float(), dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                weights = (q @ k.T * scale).softmax(dim=-1)
            else:
                B_seq = x.shape[:-1]
                h, d_h = hub.num_heads, hub.head_dim
                N = hub.num_embeddings
                x_h = x.view(*B_seq, h, d_h)
                k_h = hub.anchor_keys.float().view(N, h, d_h)
                q = F.normalize(x_h, dim=-1)
                k = F.normalize(k_h, dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
                weights = logits.softmax(dim=-1).mean(dim=-2)

        elif hub_type in ("v2_concat", "v2_topk"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)

        elif hub_type in ("v6", "v6f"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.anchor_keys.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)

        else:
            return None, len(ids), token_emb_mean

        weights_mean = weights.squeeze(0).mean(dim=0)

    return weights_mean, len(ids), token_emb_mean


def run_probe_ab(model, hub_info, tokenizer, translations, device,
                 single_token_only=True, k=10):
    """Run Probe A (anchor overlap) and Probe B (embedding cosine gap).

    Returns:
        dict with probe_a and probe_b results
    """
    random.seed(42)
    hub = hub_info["hub"]
    hub_type = hub_info["hub_type"]
    has_hub = hub_info["has_hub"]
    embedding_weight = model.get_input_embeddings().weight

    # Collect per-word data
    word_weights = {}   # (tuple_idx, lang) -> anchor weight distribution (Probe A)
    word_reps = {}      # (tuple_idx, lang) -> post-hub representation (Probe B)
    token_counts = {}

    for t_idx, (en_word, trans) in enumerate(translations):
        all_words = {"en": en_word, **trans}
        for lang, word in all_words.items():
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            token_counts[(t_idx, lang)] = len(ids)

            # Probe A: anchor weights from raw embeddings
            w, _, _ = get_anchor_weights_for_word(
                hub, hub_type, embedding_weight, tokenizer, word, device,
            )
            if w is not None:
                word_weights[(t_idx, lang)] = w

            # Probe B: post-hub representation at the anchor layer
            input_ids = torch.tensor([ids], device=device)
            rep = get_representation_at_anchor_layer(model, hub_info, input_ids)
            word_reps[(t_idx, lang)] = rep.squeeze(0).float().mean(dim=0)

    # Filter to single-token if requested
    if single_token_only:
        multi_keys = {key for key, n in token_counts.items() if n > 1}
        for key in multi_keys:
            word_weights.pop(key, None)
            word_reps.pop(key, None)

    n_tuples = len(translations)
    result = {}

    # ---- Probe A: anchor weight overlap (hub only) ----
    if has_hub:
        trans_js = []
        per_pair_a = {}

        for t_idx in range(n_tuples):
            langs_in = [l for l in LANGS if (t_idx, l) in word_weights]
            for i, l1 in enumerate(langs_in):
                for l2 in langs_in[i + 1:]:
                    w1 = word_weights[(t_idx, l1)]
                    w2 = word_weights[(t_idx, l2)]
                    js = 1.0 - js_divergence(w1, w2)
                    trans_js.append(js)

                    pair_key = f"{min(l1, l2)}-{max(l1, l2)}"
                    per_pair_a.setdefault(pair_key, []).append(js)

        # Random control
        random.seed(42)
        rand_js = []
        all_ww_keys = list(word_weights.keys())
        for _ in range(len(trans_js)):
            k1, k2 = random.sample(all_ww_keys, 2)
            if k1[1] == k2[1]:
                continue
            rand_js.append(1.0 - js_divergence(word_weights[k1], word_weights[k2]))

        mean_trans_js = sum(trans_js) / len(trans_js) if trans_js else 0
        mean_rand_js = sum(rand_js) / len(rand_js) if rand_js else 0
        if trans_js and rand_js:
            _, pval = stats.mannwhitneyu(trans_js, rand_js, alternative="greater")
        else:
            pval = 1.0

        result["probe_a"] = {
            "js_sim_translation": mean_trans_js,
            "js_sim_random": mean_rand_js,
            "js_gap": mean_trans_js - mean_rand_js,
            "pvalue": pval,
            "n_translation": len(trans_js),
            "n_random": len(rand_js),
            "per_pair": {
                p: {"mean": sum(v) / len(v), "n": len(v)}
                for p, v in sorted(per_pair_a.items())
            },
        }

    # ---- Probe B: post-hub cosine gap (all models) ----
    trans_cos = []
    per_pair_b = {}

    for t_idx in range(n_tuples):
        langs_in = [l for l in LANGS if (t_idx, l) in word_reps]
        for i, l1 in enumerate(langs_in):
            for l2 in langs_in[i + 1:]:
                rep1 = word_reps[(t_idx, l1)]
                rep2 = word_reps[(t_idx, l2)]
                cos = F.cosine_similarity(
                    rep1.unsqueeze(0), rep2.unsqueeze(0),
                ).item()
                trans_cos.append(cos)

                pair_key = f"{min(l1, l2)}-{max(l1, l2)}"
                per_pair_b.setdefault(pair_key, []).append(cos)

    random.seed(42)
    rand_cos = []
    all_rep_keys = list(word_reps.keys())
    for _ in range(len(trans_cos)):
        k1, k2 = random.sample(all_rep_keys, 2)
        if k1[1] == k2[1]:
            continue
        rep1 = word_reps[k1]
        rep2 = word_reps[k2]
        rand_cos.append(F.cosine_similarity(
            rep1.unsqueeze(0), rep2.unsqueeze(0),
        ).item())

    mean_trans_cos = sum(trans_cos) / len(trans_cos) if trans_cos else 0
    mean_rand_cos = sum(rand_cos) / len(rand_cos) if rand_cos else 0
    if trans_cos and rand_cos:
        _, pval_b = stats.mannwhitneyu(trans_cos, rand_cos, alternative="greater")
    else:
        pval_b = 1.0

    result["probe_b"] = {
        "cos_translation": mean_trans_cos,
        "cos_random": mean_rand_cos,
        "cos_gap": mean_trans_cos - mean_rand_cos,
        "pvalue": pval_b,
        "n_translation": len(trans_cos),
        "n_random": len(rand_cos),
        "per_pair": {
            p: {"mean": sum(v) / len(v), "n": len(v)}
            for p, v in sorted(per_pair_b.items())
        },
    }

    return result


def format_results(result, tag=""):
    lines = []
    if tag:
        lines.append(f"\n### {tag}\n")

    if "probe_a" in result:
        a = result["probe_a"]
        lines.append(f"**Probe A** (anchor JS similarity): "
                     f"trans={a['js_sim_translation']:.4f} rand={a['js_sim_random']:.4f} "
                     f"gap={a['js_gap']:+.4f} p={a['pvalue']:.2e}")

    b = result["probe_b"]
    lines.append(f"**Probe B** (post-hub cosine): "
                 f"trans={b['cos_translation']:.4f} rand={b['cos_random']:.4f} "
                 f"gap={b['cos_gap']:+.4f} p={b['pvalue']:.2e}")

    return "\n".join(lines)
