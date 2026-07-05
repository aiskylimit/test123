"""T3 — Mixture interchange test.

Causal proof that anchors carry shared meaning: swap a foreign word's
anchor mixture with its English translation's mixture, measure how much
the model's predictions degrade. If translation-swap hurts much LESS
than random-swap, the anchors encode shared cross-lingual meaning.

Embedding-layer variants only (V2-V4, V6, V6f). NOT for V5 (mid-layer
mixtures are contextual and cannot be precomputed).

Usage:
    python diagnostics/test_t3_mixture_interchange.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --eval-dir /path/to/eval \
        --output temp/t3_interchange.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets

from diagnostics.test_utils import (
    NON_EN_LANGS,
    filter_loanwords_per_pair,
    get_hub_input,
    get_tokenizer,
    load_checkpoint,
    load_translations,
)


def compute_mixture_from_embedding(hub, hub_type, token_embedding, device):
    """Compute the anchor mixture given a raw token embedding.

    Args:
        hub: the hub module
        hub_type: str
        token_embedding: (hidden_dim,) tensor — raw embedding of one token
        device: device

    Returns:
        mixture: (hidden_dim,) float32 tensor
    """
    x = token_embedding.unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, d)

    with torch.no_grad():
        if hub_type == "v2_additive":
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)
            mixture = weights @ hub.hub_embeddings.float()

        elif hub_type == "v3":
            if hub.num_heads == 1:
                q = F.normalize(x, dim=-1)
                k = F.normalize(hub.anchor_keys.float(), dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                weights = (q @ k.T * scale).softmax(dim=-1)
                mixture = weights @ hub.anchor_values.float()
            else:
                B_seq = x.shape[:-1]
                h, d_h = hub.num_heads, hub.head_dim
                N = hub.num_embeddings
                x_heads = x.view(*B_seq, h, d_h)
                keys_f = hub.anchor_keys.float().view(N, h, d_h)
                values_f = hub.anchor_values.float().view(N, h, d_h)
                q = F.normalize(x_heads, dim=-1)
                k = F.normalize(keys_f, dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
                weights = logits.softmax(dim=-1)
                mixture_heads = torch.einsum("...hn,nhd->...hd", weights, values_f)
                mixture = mixture_heads.reshape(*B_seq, hub.embedding_dim)

        elif hub_type in ("v2_concat", "v2_topk"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)
            mixture = weights @ hub.hub_embeddings.float()

        elif hub_type in ("v6", "v6f"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.anchor_keys.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)
            topk_w, topk_i = weights.topk(hub.top_k, dim=-1)
            w_norm = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            mixture = (w_norm.unsqueeze(-1) * hub.anchor_values.float()[topk_i]).sum(dim=-2)

        else:
            raise ValueError(f"Unsupported hub_type for T3: {hub_type}")

    return mixture.squeeze(0).squeeze(0)


def build_hub_output_with_swap(hub, hub_type, token_embeddings, swap_pos, replacement_mixture, device):
    """Run the hub forward but replace the mixture at swap_pos.

    Args:
        hub: the hub module
        hub_type: str
        token_embeddings: (1, seq_len, d) raw token embeddings
        swap_pos: int — position to swap
        replacement_mixture: (d,) tensor — the replacement mixture
        device: device

    Returns:
        (1, seq_len, d) tensor — hub output with the swapped mixture
    """
    x = token_embeddings.float().to(device)

    with torch.no_grad():
        if hub_type == "v2_additive":
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)
            mixture = weights @ hub.hub_embeddings.float()
            # Swap at position
            mixture[0, swap_pos] = replacement_mixture.float()
            output = x + hub.alpha * mixture

        elif hub_type == "v3":
            if hub.num_heads == 1:
                q = F.normalize(x, dim=-1)
                k = F.normalize(hub.anchor_keys.float(), dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                weights = (q @ k.T * scale).softmax(dim=-1)
                mixture = weights @ hub.anchor_values.float()
            else:
                B_seq = x.shape[:-1]
                h, d_h = hub.num_heads, hub.head_dim
                N = hub.num_embeddings
                x_heads = x.view(*B_seq, h, d_h)
                keys_f = hub.anchor_keys.float().view(N, h, d_h)
                values_f = hub.anchor_values.float().view(N, h, d_h)
                q = F.normalize(x_heads, dim=-1)
                k = F.normalize(keys_f, dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
                weights = logits.softmax(dim=-1)
                mixture_heads = torch.einsum("...hn,nhd->...hd", weights, values_f)
                mixture = mixture_heads.reshape(*B_seq, hub.embedding_dim)

            mixture[0, swap_pos] = replacement_mixture.float()
            update = F.linear(mixture, hub.linear_v.weight.float())
            gate = torch.sigmoid(F.linear(x, hub.linear_g.weight.float(), hub.linear_g.bias.float()))
            output = x + gate * update

        elif hub_type in ("v2_concat", "v2_topk"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)
            mixture = weights @ hub.hub_embeddings.float()
            mixture[0, swap_pos] = replacement_mixture.float()
            if hasattr(hub, 'use_mlp') and hub.use_mlp:
                mixture = hub.mlp(mixture.to(next(hub.mlp.parameters()).dtype)).float()
            output = hub.linear_out(torch.cat([x, mixture], dim=-1).to(hub.linear_out.weight.dtype))
            output = output.float()

        elif hub_type in ("v6", "v6f"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.anchor_keys.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            weights = (q @ k.T * scale).softmax(dim=-1)
            topk_w, topk_i = weights.topk(hub.top_k, dim=-1)
            w_norm = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            concept = (w_norm.unsqueeze(-1) * hub.anchor_values.float()[topk_i]).sum(dim=-2)
            if hub.use_residual_cap:
                # Compute residual BEFORE swap (spec: "leave residual untouched")
                resid = hub._cap_residual(x, concept)
                concept[0, swap_pos] = replacement_mixture.float()
                output = concept + resid
            else:
                concept[0, swap_pos] = replacement_mixture.float()
                output = x + concept

        else:
            raise ValueError(f"Unsupported hub_type: {hub_type}")

    return output


def compute_logprobs_after_position(model, hub, hub_type, input_ids, swap_pos,
                                     replacement_mixture, device, min_tokens_after=10):
    """Run model with a swapped mixture and return summed log-prob of tokens AFTER swap_pos.

    Args:
        model: the HF model (with hub hook — we'll temporarily bypass it)
        hub: the hub module
        hub_type: str
        input_ids: (1, seq_len) tensor
        swap_pos: position of the swapped word
        replacement_mixture: (d,) tensor, or None for clean run (no swap)
        device: device
        min_tokens_after: require at least this many tokens after swap_pos

    Returns:
        float: summed log-prob of tokens at positions [swap_pos+1, ..., end],
               or None if not enough tokens after swap_pos
    """
    seq_len = input_ids.shape[1]
    if seq_len - swap_pos - 1 < min_tokens_after:
        return None

    input_ids = input_ids.to(device)

    # Get raw token embeddings (bypass hub hook)
    raw_emb = F.embedding(input_ids, model.get_input_embeddings().weight)

    if replacement_mixture is not None:
        # Build hub output with the swap
        hub_output = build_hub_output_with_swap(
            hub, hub_type, raw_emb, swap_pos, replacement_mixture, device
        )
    else:
        # Clean run: normal hub forward
        with torch.no_grad():
            hub_output = hub(raw_emb.to(hub_output_dtype(hub))).float()

    # Replace the embedding hook output with our custom hub_output
    # by running the rest of the model manually
    hook_handle = model._embhub_hook_handle
    hook_handle.remove()

    # Temporarily replace embedding output
    captured_output = [hub_output.to(next(model.parameters()).dtype)]

    def inject_hook(mod, inp, out):
        return captured_output[0]

    emb_layer = model.get_input_embeddings()
    inject_handle = emb_layer.register_forward_hook(inject_hook)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    inject_handle.remove()

    # Re-attach the original hub hook
    def hub_hook(mod, inp, out):
        return hub(out)
    model._embhub_hook_handle = emb_layer.register_forward_hook(hub_hook)

    # Compute log-probs for tokens AFTER swap_pos
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Token at position t is predicted by logits at position t-1
    total_logprob = 0.0
    for t in range(swap_pos + 1, seq_len):
        target_token = input_ids[0, t]
        total_logprob += log_probs[0, t - 1, target_token].item()

    return total_logprob


def hub_output_dtype(hub):
    """Get the dtype of the hub's parameters."""
    return next(hub.parameters()).dtype


def compute_clean_logprobs(model, input_ids, swap_pos, device, min_tokens_after=10):
    """Run model normally (no swap) and return summed log-prob after swap_pos."""
    seq_len = input_ids.shape[1]
    if seq_len - swap_pos - 1 < min_tokens_after:
        return None

    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits.float(), dim=-1)
    total_logprob = 0.0
    for t in range(swap_pos + 1, seq_len):
        target_token = input_ids[0, t]
        total_logprob += log_probs[0, t - 1, target_token].item()

    return total_logprob


def load_eval_sequences(eval_dir, tokenizer, lang, max_sequences=500, block_size=128):
    """Load tokenized eval sequences for a language."""
    lang_path = os.path.join(eval_dir, lang)
    if not os.path.isdir(lang_path):
        return []

    shard_dirs = sorted(
        os.path.join(lang_path, d) for d in os.listdir(lang_path)
        if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d))
    )
    if shard_dirs:
        ds = concatenate_datasets([load_from_disk(sd) for sd in shard_dirs])
    else:
        ds = load_from_disk(lang_path)

    sequences = []
    token_buffer = []
    for example in ds:
        if len(sequences) >= max_sequences:
            break
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(tokens)
        while len(token_buffer) >= block_size and len(sequences) < max_sequences:
            sequences.append(torch.tensor(token_buffer[:block_size], dtype=torch.long))
            token_buffer = token_buffer[block_size:]

    return sequences


def run_interchange(model, hub_info, tokenizer, translations, eval_dir, device,
                    max_pairs=200, max_sequences=500, min_tokens_after=10):
    """Run the mixture interchange test.

    Returns:
        dict with per-language damage_translation vs damage_random
    """
    hub = hub_info["hub"]
    hub_type = hub_info["hub_type"]

    if hub is None:
        return {"error": "No hub found"}
    if hub_info["placement"] != "embedding":
        return {"error": "T3 only works for embedding-layer hubs (not mid-layer/V5)"}

    emb_weight = model.get_input_embeddings().weight

    results = {}
    for lang in NON_EN_LANGS:
        print(f"    {lang}:", end=" ", flush=True)

        # Get eval sequences for this language
        eval_seqs = load_eval_sequences(eval_dir, tokenizer, lang, max_sequences)
        if not eval_seqs:
            print("no eval data")
            continue

        damages_trans = []
        damages_random = []
        n_tested = 0

        for en_word, trans in translations:
            if lang not in trans:
                continue
            tgt_word = trans[lang]

            # Both must be single-token
            en_ids = tokenizer(en_word, add_special_tokens=False)["input_ids"]
            tgt_ids = tokenizer(tgt_word, add_special_tokens=False)["input_ids"]
            if len(en_ids) != 1 or len(tgt_ids) != 1:
                continue
            if en_word.lower() == tgt_word.lower():
                continue

            en_token_id = en_ids[0]
            tgt_token_id = tgt_ids[0]

            # Precompute mixtures
            en_emb = emb_weight[en_token_id].detach()
            tgt_emb = emb_weight[tgt_token_id].detach()
            en_mixture = compute_mixture_from_embedding(hub, hub_type, en_emb, device)

            # Find a frequency-matched random English word for control
            random_mixture = None
            for other_en, _ in translations:
                if other_en == en_word:
                    continue
                other_ids = tokenizer(other_en, add_special_tokens=False)["input_ids"]
                if len(other_ids) == 1:
                    other_emb = emb_weight[other_ids[0]].detach()
                    random_mixture = compute_mixture_from_embedding(hub, hub_type, other_emb, device)
                    break

            if random_mixture is None:
                continue

            # Find occurrences of tgt_word in eval sequences
            for seq in eval_seqs:
                positions = (seq == tgt_token_id).nonzero(as_tuple=True)[0]
                for pos in positions:
                    pos = pos.item()
                    if len(seq) - pos - 1 < min_tokens_after:
                        continue

                    input_ids = seq.unsqueeze(0)

                    # Clean run
                    L_clean = compute_clean_logprobs(
                        model, input_ids, pos, device, min_tokens_after
                    )
                    if L_clean is None:
                        continue

                    # Translation swap
                    L_trans = compute_logprobs_after_position(
                        model, hub, hub_type, input_ids, pos,
                        en_mixture, device, min_tokens_after
                    )

                    # Random swap
                    L_random = compute_logprobs_after_position(
                        model, hub, hub_type, input_ids, pos,
                        random_mixture, device, min_tokens_after
                    )

                    if L_trans is not None and L_random is not None:
                        damages_trans.append(L_clean - L_trans)
                        damages_random.append(L_clean - L_random)
                        n_tested += 1

                    if n_tested >= max_pairs:
                        break
                if n_tested >= max_pairs:
                    break
            if n_tested >= max_pairs:
                break

        if damages_trans:
            mean_d_trans = sum(damages_trans) / len(damages_trans)
            mean_d_random = sum(damages_random) / len(damages_random)
            results[f"en-{lang}"] = {
                "damage_translation": mean_d_trans,
                "damage_random": mean_d_random,
                "damage_ratio": mean_d_trans / max(abs(mean_d_random), 1e-8),
                "n_pairs": len(damages_trans),
            }
            print(f"d_trans={mean_d_trans:.4f} d_rand={mean_d_random:.4f} "
                  f"ratio={mean_d_trans / max(abs(mean_d_random), 1e-8):.4f} (n={len(damages_trans)})")
        else:
            print("no valid pairs found")

    return results


def format_results(results, checkpoint_name=""):
    """Format results as markdown."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")

    if "error" in results:
        lines.append(f"Error: {results['error']}")
        return "\n".join(lines)

    lines.append("| Lang Pair | Damage Trans | Damage Random | Ratio | N |")
    lines.append("|-----------|-------------|---------------|-------|---|")

    for pair in [f"en-{l}" for l in NON_EN_LANGS]:
        r = results.get(pair, {})
        if not r:
            continue
        lines.append(
            f"| {pair} | {r['damage_translation']:.4f} | {r['damage_random']:.4f} | "
            f"{r['damage_ratio']:.4f} | {r['n_pairs']} |"
        )

    lines.append("")
    lines.append("Ratio < 1 = translation swap hurts less than random swap = anchors share meaning.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T3: Mixture interchange test")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Hub checkpoint paths")
    parser.add_argument("--eval-dir", required=True,
                        help="Path to eval directory (per-language subdirs)")
    parser.add_argument("--translations", default=None)
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="Max interchange pairs per language (default: 200)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="temp/t3_interchange.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(args.checkpoint[0])
    translations = load_translations(path=args.translations, single_token_only=False)
    translations = filter_loanwords_per_pair(translations)
    print(f"Translation tuples: {len(translations)}")

    all_results = {"checkpoints": []}
    all_tables = []

    for ckpt_path in args.checkpoint:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nLoading: {ckpt_path}")
        model, hub_info = load_checkpoint(ckpt_path, device=device)
        print(f"  hub_type={hub_info['hub_type']}, placement={hub_info['placement']}")

        print("  Running interchange test...")
        results = run_interchange(
            model, hub_info, tokenizer, translations, args.eval_dir, device,
            max_pairs=args.max_pairs,
        )

        results["checkpoint"] = ckpt_path
        results["hub_type"] = hub_info["hub_type"]
        all_results["checkpoints"].append(results)

        table = format_results(results, ckpt_name)
        all_tables.append(table)
        print(table)

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# T3 — Mixture Interchange Results\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
