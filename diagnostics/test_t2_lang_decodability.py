"""T2 — Language decodability from anchor weights.

Trains a logistic regression classifier to predict a token's language from
its anchor-weight vector (the softmax distribution over all N anchors).
High accuracy = anchors encode language identity (partition failure).

Intended reading: decodability FALLING over training while Test B RISES
= shared structure replacing language partition. Target is NOT zero —
some language-specific content is legitimate.

Usage:
    python diagnostics/test_t2_lang_decodability.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --eval-dir /path/to/eval \
        --output temp/t2_decodability.json

    # Multiple checkpoints (training progression):
    python diagnostics/test_t2_lang_decodability.py \
        --checkpoint /path/to/ckpt-1500 /path/to/ckpt-3250 /path/to/ckpt-6500 \
        --eval-dir /path/to/eval \
        --output temp/t2_decodability.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from diagnostics.test_utils import (
    LANGS,
    get_hub_input,
    get_tokenizer,
    load_checkpoint,
)


def get_anchor_weights_for_tokens(model, hub_info, input_ids, device):
    """Compute anchor-weight vectors for each token in a batch.

    Uses get_hub_input() to get the pre-hub representation (raw embeddings
    for embedding-layer hubs, or the transformer layer output before the
    hook for mid-layer hubs), then computes the softmax anchor weights.

    Args:
        model: the model with hub injected
        hub_info: dict from load_checkpoint
        input_ids: (batch, seq_len) tensor

    Returns:
        (batch * seq_len, N_anchors) float32 tensor, or None if no hub
    """
    hub = hub_info["hub"]
    if hub is None:
        return None

    with torch.no_grad():
        x = get_hub_input(model, hub_info, input_ids, device).float()

        hub_type = hub_info["hub_type"]

        if hub_type == "v2_additive":
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)

        elif hub_type == "v3":
            if hub.num_heads == 1:
                q = F.normalize(x, dim=-1)
                k = F.normalize(hub.anchor_keys.float(), dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                logits = (q @ k.T) * scale
                weights = logits.softmax(dim=-1)
            else:
                B_seq = x.shape[:-1]
                h, d_h = hub.num_heads, hub.head_dim
                N = hub.num_embeddings
                x_heads = x.view(*B_seq, h, d_h)
                keys_f = hub.anchor_keys.float().view(N, h, d_h)
                q = F.normalize(x_heads, dim=-1)
                k = F.normalize(keys_f, dim=-1)
                scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
                logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
                weights = logits.softmax(dim=-1)
                weights = weights.mean(dim=-2)

        elif hub_type in ("v2_concat", "v2_topk"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.hub_embeddings.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)

        elif hub_type in ("v6", "v6f"):
            q = F.normalize(x, dim=-1)
            k = F.normalize(hub.anchor_keys.float(), dim=-1)
            scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)

        else:
            raise ValueError(f"Unknown hub_type: {hub_type}")

    return weights.reshape(-1, weights.shape[-1]).cpu()


def load_eval_tokens(eval_dir, tokenizer, lang, n_tokens=5000, block_size=128):
    """Load eval data and return tokenized sequences for a language.

    Returns:
        input_ids tensor (n_sequences, block_size), or None if no data
    """
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


def run_decodability(model, hub_info, tokenizer, eval_dir, device,
                     samples_per_lang=5000, block_size=128):
    """Run language decodability test.

    Returns:
        dict with accuracy, chance, per-language counts
    """
    all_weights = []
    all_labels = []
    lang_to_idx = {lang: i for i, lang in enumerate(LANGS)}

    print("  Collecting anchor weights per language...")
    for lang in LANGS:
        input_ids = load_eval_tokens(eval_dir, tokenizer, lang, samples_per_lang, block_size)
        if input_ids is None:
            print(f"    {lang}: no eval data, skipping")
            continue

        # Get anchor weights in batches
        weights_list = []
        batch_size = 8
        for start in range(0, len(input_ids), batch_size):
            batch = input_ids[start:start + batch_size]
            w = get_anchor_weights_for_tokens(model, hub_info, batch, device)
            if w is None:
                return {"error": "No hub found in checkpoint"}
            weights_list.append(w)

        weights = torch.cat(weights_list, dim=0)
        # Take exactly samples_per_lang tokens (trim excess from block rounding)
        weights = weights[:samples_per_lang]
        labels = torch.full((len(weights),), lang_to_idx[lang], dtype=torch.long)

        all_weights.append(weights)
        all_labels.append(labels)
        print(f"    {lang}: {len(weights)} tokens")

    if not all_weights:
        return {"error": "No eval data found"}

    X = torch.cat(all_weights, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    print(f"  Total: {len(X)} samples, {len(set(y))} languages")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic regression
    print("  Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    n_classes = len(set(y))
    chance = 1.0 / n_classes

    # Per-language accuracy
    per_lang = {}
    for lang, idx in lang_to_idx.items():
        mask = y_test == idx
        if mask.sum() > 0:
            per_lang[lang] = accuracy_score(y_test[mask], y_pred[mask])

    return {
        "accuracy": accuracy,
        "chance": chance,
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "per_language_accuracy": per_lang,
    }


def format_results(results, checkpoint_name=""):
    """Format results as readable text."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")

    if "error" in results:
        lines.append(f"Error: {results['error']}")
        return "\n".join(lines)

    acc = results["accuracy"]
    chance = results["chance"]
    lines.append(f"Accuracy: **{acc:.4f}** (chance: {chance:.4f}, lift: {acc - chance:+.4f})")
    lines.append(f"Features: {results['n_features']} anchors, "
                 f"Train: {results['n_train']}, Test: {results['n_test']}")
    lines.append("")
    lines.append("| Language | Accuracy |")
    lines.append("|----------|----------|")
    for lang, lang_acc in sorted(results["per_language_accuracy"].items()):
        lines.append(f"| {lang} | {lang_acc:.4f} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T2: Language decodability from anchor weights")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more hub checkpoint paths")
    parser.add_argument("--eval-dir", required=True,
                        help="Path to eval directory (per-language subdirs)")
    parser.add_argument("--samples-per-lang", type=int, default=5000,
                        help="Tokens per language (default: 5000)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="temp/t2_decodability.json")
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
        print(f"  hub_type={hub_info['hub_type']}, N={hub_info['hub'].num_embeddings if hub_info['hub'] else 'N/A'}")

        results = run_decodability(
            model, hub_info, tokenizer, args.eval_dir, device,
            samples_per_lang=args.samples_per_lang,
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
        f.write("# T2 — Language Decodability Results\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
