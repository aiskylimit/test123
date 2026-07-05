"""T1 — Zero-shot cross-lingual transfer probe (XNLI).

Freezes the model, trains a linear classifier on English XNLI using
mean-pooled hidden states at chosen layers, then tests zero-shot on
other languages. Compares hub vs baseline.

Pair features: [u; v; |u-v|; u*v] where u=premise, v=hypothesis.
Trains on English train, early-stops on English dev, evaluates on all
languages' test sets.

Usage:
    python diagnostics/test_t1_transfer_probe.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --baseline /path/to/baseline/checkpoint-6500 \
        --layers 10 14 27 \
        --output temp/t1_transfer.json

    # Multiple seeds for significance:
    python diagnostics/test_t1_transfer_probe.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --baseline /path/to/baseline/checkpoint-6500 \
        --layers 14 27 \
        --seeds 42 43 44 \
        --output temp/t1_transfer.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from diagnostics.test_utils import (
    NON_EN_LANGS,
    get_tokenizer,
    load_checkpoint,
)

XNLI_LANGS = ["en", "vi", "zh", "ru", "de", "ar"]
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def mean_pool_at_layer(model, tokenizer, texts, layer_idx, device,
                       batch_size=32, max_length=128):
    """Mean-pool hidden states at a specific layer for a list of texts.

    Args:
        model: frozen HF model
        tokenizer: tokenizer
        texts: list of strings
        layer_idx: which layer (None = embedding, 0..N-1 = transformer layers)
        device: device
        batch_size: texts per forward pass
        max_length: max token length

    Returns:
        np.array of shape (len(texts), hidden_dim), float32
    """
    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if layer_idx is None:
            hs = outputs.hidden_states[0]
        else:
            hs = outputs.hidden_states[layer_idx + 1]

        # Mean-pool over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embs.append(pooled.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def build_pair_features(premise_embs, hypothesis_embs):
    """Build NLI pair features: [u; v; |u-v|; u*v].

    Args:
        premise_embs: (N, d) array
        hypothesis_embs: (N, d) array

    Returns:
        (N, 4*d) array
    """
    diff = np.abs(premise_embs - hypothesis_embs)
    prod = premise_embs * hypothesis_embs
    return np.concatenate([premise_embs, hypothesis_embs, diff, prod], axis=1)


def load_xnli(lang, split, max_examples=None):
    """Load XNLI data for a language and split.

    Returns:
        list of (premise, hypothesis, label_int) tuples
    """
    if split == "train":
        ds = load_dataset("facebook/xnli", lang, split="train")
    elif split == "dev":
        ds = load_dataset("facebook/xnli", lang, split="validation")
    else:
        ds = load_dataset("facebook/xnli", lang, split="test")

    examples = []
    for row in ds:
        label = row["label"]
        if isinstance(label, str):
            label = LABEL_MAP.get(label, label)
        examples.append((row["premise"], row["hypothesis"], label))
        if max_examples and len(examples) >= max_examples:
            break

    return examples


def encode_examples(model, tokenizer, examples, layer_idx, device,
                    batch_size=32, max_length=128):
    """Encode XNLI examples into pair features at a given layer.

    Returns:
        X: (N, 4*d) feature array
        y: (N,) label array
    """
    premises = [ex[0] for ex in examples]
    hypotheses = [ex[1] for ex in examples]
    labels = np.array([ex[2] for ex in examples])

    premise_embs = mean_pool_at_layer(
        model, tokenizer, premises, layer_idx, device, batch_size, max_length
    )
    hypothesis_embs = mean_pool_at_layer(
        model, tokenizer, hypotheses, layer_idx, device, batch_size, max_length
    )

    X = build_pair_features(premise_embs, hypothesis_embs)
    return X, labels


def run_transfer_probe(model, tokenizer, layer_idx, device, seeds,
                       max_train=50000, batch_size=32, max_length=128):
    """Run the transfer probe at one layer across all seeds.

    Encodes data ONCE (the expensive part), then trains a classifier
    per seed (the cheap part).

    Returns:
        list of per-seed result dicts
    """
    # Load data (cached by HF datasets after first download)
    print(f"    Loading XNLI data...")
    train_examples = load_xnli("en", "train", max_examples=max_train)
    dev_examples = load_xnli("en", "dev")

    test_data = {}
    for lang in XNLI_LANGS:
        test_data[lang] = load_xnli(lang, "test")

    # Encode ONCE for all seeds
    print(f"    Encoding train ({len(train_examples)})...")
    X_train, y_train = encode_examples(
        model, tokenizer, train_examples, layer_idx, device, batch_size, max_length
    )
    print(f"    Encoding dev ({len(dev_examples)})...")
    X_dev, y_dev = encode_examples(
        model, tokenizer, dev_examples, layer_idx, device, batch_size, max_length
    )

    X_tests, y_tests = {}, {}
    for lang in XNLI_LANGS:
        print(f"    Encoding {lang} test ({len(test_data[lang])})...", end=" ", flush=True)
        X_tests[lang], y_tests[lang] = encode_examples(
            model, tokenizer, test_data[lang], layer_idx, device, batch_size, max_length
        )
        print("done")

    # Train classifier per seed (cheap — no forward passes)
    seed_results = []
    for seed in seeds:
        print(f"    Training classifier (seed={seed})...")
        clf = LogisticRegression(
            max_iter=1000, random_state=seed, C=1.0,
        )
        clf.fit(X_train, y_train)

        train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
        dev_acc = float(accuracy_score(y_dev, clf.predict(X_dev)))
        print(f"      en: train={train_acc:.4f}, dev={dev_acc:.4f}")

        per_lang = {}
        for lang in XNLI_LANGS:
            acc = float(accuracy_score(y_tests[lang], clf.predict(X_tests[lang])))
            per_lang[lang] = acc

        seed_results.append({
            "en_train_acc": train_acc,
            "en_dev_acc": dev_acc,
            "per_lang": per_lang,
        })

    return seed_results


def format_results(all_layer_results, checkpoint_name="", baseline_results=None):
    """Format results as markdown."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")

    for layer_name, layer_data in sorted(all_layer_results.items()):
        lines.append(f"\n**{layer_name}** (en_dev={layer_data['mean']['en_dev_acc']:.4f})\n")

        header = "| Language | Hub Acc |"
        divider = "|----------|---------|"
        if baseline_results and layer_name in baseline_results:
            header += " Base Acc | Delta |"
            divider += "----------|-------|"
        if "std" in layer_data:
            header = header.replace("Hub Acc", "Hub Acc (±std)")

        lines.append(header)
        lines.append(divider)

        for lang in XNLI_LANGS:
            hub_acc = layer_data["mean"]["per_lang"].get(lang, 0)
            hub_std = layer_data.get("std", {}).get("per_lang", {}).get(lang, 0)
            row = f"| {lang} | {hub_acc:.4f}"
            if hub_std > 0:
                row += f" ±{hub_std:.4f}"
            row += " |"

            if baseline_results and layer_name in baseline_results:
                base_acc = baseline_results[layer_name]["mean"]["per_lang"].get(lang, 0)
                delta = hub_acc - base_acc
                sign = "+" if delta >= 0 else ""
                row += f" {base_acc:.4f} | {sign}{delta:.4f} |"

            lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T1: Zero-shot transfer probe (XNLI)")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more hub checkpoint paths")
    parser.add_argument("--baseline", required=True,
                        help="Matched no-hub baseline checkpoint path")
    parser.add_argument("--layers", nargs="+", type=int, default=[14, 27],
                        help="Layer indices to probe (default: 14 27)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds for the classifier (default: 42)")
    parser.add_argument("--max-train", type=int, default=50000,
                        help="Max English training examples (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoding (default: 32)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length (default: 128)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="temp/t1_transfer.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(args.checkpoint[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Run baseline ---
    print(f"\nLoading baseline: {args.baseline}")
    base_model, _ = load_checkpoint(args.baseline, device=device, baseline=True)

    baseline_all = {}
    for layer_idx in args.layers:
        layer_name = f"layer_{layer_idx}"
        print(f"\n  Baseline @ {layer_name}:")
        seed_results = run_transfer_probe(
            base_model, tokenizer, layer_idx, device, seeds=args.seeds,
            max_train=args.max_train, batch_size=args.batch_size,
            max_length=args.max_length,
        )
        baseline_all[layer_name] = _aggregate_seeds(seed_results)

    del base_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # --- Run hub checkpoints ---
    all_results = {"baseline": baseline_all, "checkpoints": []}
    all_tables = []

    for ckpt_path in args.checkpoint:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nLoading hub: {ckpt_path}")
        model, hub_info = load_checkpoint(ckpt_path, device=device)
        print(f"  hub_type={hub_info['hub_type']}, placement={hub_info['placement']}")

        hub_all = {}
        for layer_idx in args.layers:
            layer_name = f"layer_{layer_idx}"
            print(f"\n  Hub @ {layer_name}:")
            seed_results = run_transfer_probe(
                model, tokenizer, layer_idx, device, seeds=args.seeds,
                max_train=args.max_train, batch_size=args.batch_size,
                max_length=args.max_length,
            )
            hub_all[layer_name] = _aggregate_seeds(seed_results)

        ckpt_result = {
            "checkpoint": ckpt_path,
            "hub_type": hub_info["hub_type"],
            "layers": hub_all,
        }
        all_results["checkpoints"].append(ckpt_result)

        table = format_results(hub_all, ckpt_name, baseline_all)
        all_tables.append(table)
        print(table)

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# T1 — Zero-Shot Transfer Probe Results (XNLI)\n\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Layers: {args.layers}\n")
        f.write(f"Baseline: {args.baseline}\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown saved to {md_path}")


def _aggregate_seeds(seed_results):
    """Aggregate results across seeds into mean ± std."""
    if len(seed_results) == 1:
        return {"mean": seed_results[0]}

    mean_result = {
        "en_train_acc": float(np.mean([r["en_train_acc"] for r in seed_results])),
        "en_dev_acc": float(np.mean([r["en_dev_acc"] for r in seed_results])),
        "per_lang": {},
    }
    std_result = {
        "en_train_acc": float(np.std([r["en_train_acc"] for r in seed_results])),
        "en_dev_acc": float(np.std([r["en_dev_acc"] for r in seed_results])),
        "per_lang": {},
    }

    for lang in XNLI_LANGS:
        accs = [r["per_lang"].get(lang, 0) for r in seed_results]
        mean_result["per_lang"][lang] = float(np.mean(accs))
        std_result["per_lang"][lang] = float(np.std(accs))

    return {"mean": mean_result, "std": std_result}


if __name__ == "__main__":
    main()
