"""T4 — Hub contribution share per token.

Measures how much the hub contributes relative to the token representation.
Answers: is the model actually USING the hub, or ignoring it?

Metric per hub type:
  V3/V4/V5:     ||gate * update|| / ||x||  (clean — residual block)
  V2-concat:    ||output - x|| / ||x||
  V2-topk:      ||output - x|| / ||x||
  V6/V6f:       ||concept|| / ||tok_emb||  (inference mode = concept + resid)
  V2-additive:  ||alpha * contribution|| / ||x||

Reports mean, p50, p95 per language. Also reports mean gate value for V3/V4/V5.

Usage:
    python diagnostics/test_t4_contribution_share.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --eval-dir /path/to/eval \
        --output temp/t4_contribution.json

    # Multiple checkpoints:
    python diagnostics/test_t4_contribution_share.py \
        --checkpoint /path/to/ckpt-1500 /path/to/ckpt-3250 /path/to/ckpt-6500 \
        --eval-dir /path/to/eval \
        --output temp/t4_contribution.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets

from diagnostics.test_utils import (
    LANGS,
    get_hub_input,
    get_tokenizer,
    load_checkpoint,
)


def measure_contribution_v3(hub, x):
    """V3/V4/V5: ||gate * update|| / ||x||, plus gate mean."""
    x_f = x.float()
    if hub.num_heads == 1:
        q = F.normalize(x_f, dim=-1)
        k = F.normalize(hub.anchor_keys.float(), dim=-1)
        scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
        mixture = weights @ hub.anchor_values.float()
    else:
        B_seq = x_f.shape[:-1]
        h, d_h = hub.num_heads, hub.head_dim
        N = hub.num_embeddings
        x_heads = x_f.view(*B_seq, h, d_h)
        keys_f = hub.anchor_keys.float().view(N, h, d_h)
        values_f = hub.anchor_values.float().view(N, h, d_h)
        q = F.normalize(x_heads, dim=-1)
        k = F.normalize(keys_f, dim=-1)
        scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
        logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
        weights = logits.softmax(dim=-1)
        mixture_heads = torch.einsum("...hn,nhd->...hd", weights, values_f)
        mixture = mixture_heads.reshape(*B_seq, hub.embedding_dim)

    update = F.linear(mixture, hub.linear_v.weight.float())
    gate = torch.sigmoid(F.linear(x_f, hub.linear_g.weight.float(), hub.linear_g.bias.float()))
    contribution = gate * update

    contrib_norm = contribution.norm(dim=-1)
    x_norm = x_f.norm(dim=-1).clamp(min=1e-8)
    ratio = contrib_norm / x_norm

    return ratio, gate.mean(dim=-1)


def measure_contribution_v2_additive(hub, x):
    """V2 additive: ||alpha * contribution|| / ||x||."""
    x_f = x.float()
    q = F.normalize(x_f, dim=-1)
    k = F.normalize(hub.hub_embeddings.float(), dim=-1)
    scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
    logits = (q @ k.T) * scale
    weights = logits.softmax(dim=-1)
    contribution = weights @ hub.hub_embeddings.float()

    contrib_norm = (hub.alpha * contribution).norm(dim=-1)
    x_norm = x_f.norm(dim=-1).clamp(min=1e-8)
    ratio = contrib_norm / x_norm

    return ratio, None


def measure_contribution_v2_concat(hub, x):
    """V2-concat / V2-topk: ||output - x|| / ||x||."""
    x_f = x.float()
    with torch.no_grad():
        output = hub(x).float()
    diff = output - x_f
    diff_norm = diff.norm(dim=-1)
    x_norm = x_f.norm(dim=-1).clamp(min=1e-8)
    ratio = diff_norm / x_norm

    return ratio, None


def measure_contribution_v6(hub, x):
    """V6/V6f: ||concept|| / ||tok_emb|| in inference mode."""
    x_f = x.float()
    q = F.normalize(x_f, dim=-1)
    k = F.normalize(hub.anchor_keys.float(), dim=-1)
    scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
    logits = (q @ k.T) * scale
    weights = logits.softmax(dim=-1)

    topk_weights, topk_indices = weights.topk(hub.top_k, dim=-1)
    w_norm = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    concept = (w_norm.unsqueeze(-1) * hub.anchor_values.float()[topk_indices]).sum(dim=-2)

    concept_norm = concept.norm(dim=-1)
    x_norm = x_f.norm(dim=-1).clamp(min=1e-8)
    ratio = concept_norm / x_norm

    return ratio, None


def load_eval_tokens(eval_dir, tokenizer, lang, n_tokens=5000, block_size=128):
    """Load eval data and return tokenized sequences for a language."""
    lang_path = os.path.join(eval_dir, lang)
    if not os.path.isdir(lang_path):
        return None

    shard_dirs = sorted(
        os.path.join(lang_path, d) for d in os.listdir(lang_path)
        if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d))
    )
    if shard_dirs:
        ds = concatenate_datasets([load_from_disk(sd) for sd in shard_dirs])
    else:
        ds = load_from_disk(lang_path)

    n_sequences = (n_tokens + block_size - 1) // block_size
    token_buffer = []
    sequences = []

    for example in ds:
        if len(sequences) >= n_sequences:
            break
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(tokens)
        while len(token_buffer) >= block_size and len(sequences) < n_sequences:
            sequences.append(token_buffer[:block_size])
            token_buffer = token_buffer[block_size:]

    if not sequences:
        return None
    return torch.tensor(sequences, dtype=torch.long)


def run_contribution_share(model, hub_info, tokenizer, eval_dir, device,
                           tokens_per_lang=5000, block_size=128):
    """Measure hub contribution share per language.

    Returns:
        dict with per-language stats and overall summary
    """
    hub = hub_info["hub"]
    if hub is None:
        return {"error": "No hub found in checkpoint"}

    hub_type = hub_info["hub_type"]

    # Select the right measurement function
    if hub_type == "v3":
        measure_fn = measure_contribution_v3
    elif hub_type == "v2_additive":
        measure_fn = measure_contribution_v2_additive
    elif hub_type in ("v2_concat", "v2_topk"):
        measure_fn = measure_contribution_v2_concat
    elif hub_type in ("v6", "v6f"):
        measure_fn = measure_contribution_v6
    else:
        return {"error": f"Unknown hub_type: {hub_type}"}

    per_lang = {}
    all_ratios = []
    all_gates = []

    print("  Measuring contribution share per language...")
    for lang in LANGS:
        input_ids = load_eval_tokens(eval_dir, tokenizer, lang, tokens_per_lang, block_size)
        if input_ids is None:
            print(f"    {lang}: no eval data, skipping")
            continue

        lang_ratios = []
        lang_gates = []
        batch_size = 8

        for start in range(0, len(input_ids), batch_size):
            batch = input_ids[start:start + batch_size]
            with torch.no_grad():
                x = get_hub_input(model, hub_info, batch, device)
                ratio, gate_vals = measure_fn(hub, x)

            lang_ratios.append(ratio.reshape(-1).cpu())
            if gate_vals is not None:
                lang_gates.append(gate_vals.reshape(-1).cpu())

        ratios = torch.cat(lang_ratios)[:tokens_per_lang]
        stats = {
            "mean": ratios.mean().item(),
            "median": ratios.median().item(),
            "p95": ratios.quantile(0.95).item(),
            "n_tokens": len(ratios),
        }
        if lang_gates:
            gates = torch.cat(lang_gates)[:tokens_per_lang]
            stats["gate_mean"] = gates.mean().item()

        per_lang[lang] = stats
        all_ratios.append(ratios)
        print(f"    {lang}: mean={stats['mean']:.4f} median={stats['median']:.4f} "
              f"p95={stats['p95']:.4f}" +
              (f" gate={stats.get('gate_mean', 0):.4f}" if "gate_mean" in stats else ""))

    if not all_ratios:
        return {"error": "No eval data found"}

    combined = torch.cat(all_ratios)
    summary = {
        "overall_mean": combined.mean().item(),
        "overall_median": combined.median().item(),
        "overall_p95": combined.quantile(0.95).item(),
    }

    return {"per_language": per_lang, "summary": summary}


def format_results(results, checkpoint_name=""):
    """Format results as readable text."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")

    if "error" in results:
        lines.append(f"Error: {results['error']}")
        return "\n".join(lines)

    has_gate = any("gate_mean" in s for s in results["per_language"].values())

    header = "| Language | Mean | Median | P95 |"
    divider = "|----------|------|--------|-----|"
    if has_gate:
        header += " Gate Mean |"
        divider += "-----------|"

    lines.append(header)
    lines.append(divider)

    for lang in LANGS:
        if lang not in results["per_language"]:
            continue
        s = results["per_language"][lang]
        row = f"| {lang} | {s['mean']:.4f} | {s['median']:.4f} | {s['p95']:.4f} |"
        if has_gate:
            row += f" {s.get('gate_mean', 0):.4f} |"
        lines.append(row)

    summary = results["summary"]
    lines.append("")
    lines.append(f"**Overall:** mean={summary['overall_mean']:.4f} "
                 f"median={summary['overall_median']:.4f} p95={summary['overall_p95']:.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T4: Hub contribution share")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more hub checkpoint paths")
    parser.add_argument("--eval-dir", required=True,
                        help="Path to eval directory (per-language subdirs)")
    parser.add_argument("--tokens-per-lang", type=int, default=5000,
                        help="Tokens per language (default: 5000)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="temp/t4_contribution.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(args.checkpoint[0])

    all_results = {"checkpoints": []}
    all_tables = []

    for ckpt_path in args.checkpoint:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nLoading: {ckpt_path}")
        model, hub_info = load_checkpoint(ckpt_path, device=device)
        hub_type = hub_info["hub_type"]
        print(f"  hub_type={hub_type}, placement={hub_info['placement']}")

        results = run_contribution_share(
            model, hub_info, tokenizer, args.eval_dir, device,
            tokens_per_lang=args.tokens_per_lang,
        )
        results["checkpoint"] = ckpt_path
        results["hub_type"] = hub_type
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
        f.write("# T4 — Hub Contribution Share Results\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
