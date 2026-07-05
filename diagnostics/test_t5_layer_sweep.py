"""T5 — Per-layer cross-lingual similarity sweep.

For each transformer layer, measures whether translation pairs have more
similar hidden states than frequency-matched random pairs. The layer with
the highest gap is where cross-lingual structure naturally lives — the
optimal placement for V5's anchor block, and the baseline numbers V5 must beat.

Runs on a BASELINE checkpoint (no hub needed). Uses real in-context
occurrences (deeper layers are contextual, so isolated tokens are wrong).

Usage:
    python diagnostics/test_t5_layer_sweep.py \
        --checkpoint /path/to/baseline/checkpoint-6500 \
        --eval-dir /path/to/eval \
        --output temp/t5_layer_sweep.json

    # Optionally specify which layers to measure (default: all)
    python diagnostics/test_t5_layer_sweep.py \
        --checkpoint /path/to/baseline/checkpoint-6500 \
        --eval-dir /path/to/eval \
        --layers 0 5 10 14 20 27 \
        --output temp/t5_layer_sweep.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

from diagnostics.test_utils import (
    NON_EN_LANGS,
    filter_loanwords_per_pair,
    get_tokenizer,
    load_checkpoint,
    load_translations,
)


def find_word_occurrences(token_id, input_ids_list, max_occurrences=50):
    """Find positions where a token occurs across eval sequences.

    Args:
        token_id: the token ID to search for
        input_ids_list: list of 1D tensors (each a tokenized sequence)
        max_occurrences: cap to avoid dominating by one word

    Returns:
        list of (seq_idx, position) tuples
    """
    occurrences = []
    for seq_idx, ids in enumerate(input_ids_list):
        positions = (ids == token_id).nonzero(as_tuple=True)[0]
        for pos in positions:
            occurrences.append((seq_idx, pos.item()))
            if len(occurrences) >= max_occurrences:
                return occurrences
    return occurrences


def get_hidden_states_at_positions(model, input_ids_batch, positions_batch, layers, device):
    """Run forward pass and extract hidden states at specific positions.

    Args:
        model: the HF model
        input_ids_batch: (batch, seq_len) tensor
        positions_batch: list of positions (one per batch item) to extract
        layers: list of layer indices to return (None in list = embedding output)
        device: target device

    Returns:
        dict: {layer_idx: tensor of shape (batch, hidden_dim)}
        where layer_idx=None means embedding output
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_batch.to(device),
            output_hidden_states=True,
            return_dict=True,
        )

    result = {}
    for layer_idx in layers:
        if layer_idx is None:
            hs = outputs.hidden_states[0]
        else:
            hs = outputs.hidden_states[layer_idx + 1]
        # Extract the specific position for each batch item
        extracted = torch.stack([
            hs[i, pos] for i, pos in enumerate(positions_batch)
        ])
        result[layer_idx] = extracted.float()

    return result


def collect_word_representations(
    model, tokenizer, word, eval_sequences, layers, device,
    max_occurrences=50, context_len=128,
):
    """Get mean-pooled hidden states for a word across its in-context occurrences.

    Args:
        model: the HF model
        tokenizer: tokenizer
        word: the word to find
        eval_sequences: list of tokenized sequences (1D tensors)
        layers: list of layer indices
        device: device
        max_occurrences: max occurrences to use
        context_len: context window around the occurrence

    Returns:
        dict: {layer_idx: mean_vector (hidden_dim,)} or None if no occurrences found
    """
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        return None
    token_id = ids[0]

    occurrences = find_word_occurrences(token_id, eval_sequences, max_occurrences)
    if not occurrences:
        return None

    # Process occurrences in batches
    all_states = {l: [] for l in layers}
    batch_size = 16

    for batch_start in range(0, len(occurrences), batch_size):
        batch_occ = occurrences[batch_start:batch_start + batch_size]

        # Build context windows around each occurrence
        batch_inputs = []
        batch_positions = []
        for seq_idx, pos in batch_occ:
            seq = eval_sequences[seq_idx]
            # Take a window of context_len tokens centered on the occurrence
            start = max(0, pos - context_len // 2)
            end = min(len(seq), start + context_len)
            start = max(0, end - context_len)
            window = seq[start:end]
            local_pos = pos - start
            batch_inputs.append(window)
            batch_positions.append(local_pos)

        # Pad to same length
        max_len = max(len(x) for x in batch_inputs)
        padded = torch.zeros(len(batch_inputs), max_len, dtype=torch.long)
        for i, inp in enumerate(batch_inputs):
            padded[i, :len(inp)] = inp

        states = get_hidden_states_at_positions(
            model, padded, batch_positions, layers, device
        )
        for l in layers:
            all_states[l].append(states[l])

    # Mean-pool across all occurrences
    result = {}
    for l in layers:
        stacked = torch.cat(all_states[l], dim=0)
        result[l] = stacked.mean(dim=0)
    return result


def load_eval_sequences(eval_dir, tokenizer, lang, max_sequences=2000, block_size=128):
    """Load and tokenize eval data for a language, returning sequence tensors.

    Args:
        eval_dir: path to eval directory (with per-language subdirs)
        tokenizer: tokenizer
        lang: language code
        max_sequences: max number of sequences to load
        block_size: token length per sequence

    Returns:
        list of 1D tensors (tokenized sequences)
    """
    lang_path = os.path.join(eval_dir, lang)
    if not os.path.isdir(lang_path):
        return []

    # Check for sharded layout (shard_* subdirs inside the lang dir)
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

    for i, example in enumerate(ds):
        if len(sequences) >= max_sequences:
            break
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(tokens)

        while len(token_buffer) >= block_size:
            seq = torch.tensor(token_buffer[:block_size], dtype=torch.long)
            sequences.append(seq)
            token_buffer = token_buffer[block_size:]
            if len(sequences) >= max_sequences:
                break

    return sequences


def run_layer_sweep(
    model, tokenizer, translations, eval_dir, layers, device,
    max_occurrences=50, max_sequences=2000, context_len=128,
):
    """Run the full layer sweep: for each layer, compute translation vs random gap.

    Returns:
        dict with per-layer results and summary
    """
    # Load eval sequences for English (source for finding en word occurrences)
    print("  Loading eval sequences...")
    eval_seqs = {}
    for lang in ["en"] + NON_EN_LANGS:
        seqs = load_eval_sequences(eval_dir, tokenizer, lang, max_sequences, context_len)
        if seqs:
            eval_seqs[lang] = seqs
            print(f"    {lang}: {len(seqs)} sequences")
        else:
            print(f"    {lang}: no eval data found, skipping")

    if "en" not in eval_seqs:
        raise ValueError(f"No English eval data found in {eval_dir}")

    # Collect representations for all words at all layers
    print("  Collecting word representations...")
    word_reps = {}  # (en_word_or_trans, lang) -> {layer: vector}
    n_found = 0
    n_total = 0

    for en_word, trans in translations:
        # English word
        if "en" in eval_seqs:
            n_total += 1
            reps = collect_word_representations(
                model, tokenizer, en_word, eval_seqs["en"],
                layers, device, max_occurrences, context_len,
            )
            if reps is not None:
                word_reps[(en_word, "en")] = reps
                n_found += 1

        # Target language words
        for lang, word in trans.items():
            if lang not in eval_seqs:
                continue
            n_total += 1
            reps = collect_word_representations(
                model, tokenizer, word, eval_seqs[lang],
                layers, device, max_occurrences, context_len,
            )
            if reps is not None:
                word_reps[(word, lang)] = reps
                n_found += 1

    print(f"    Found representations for {n_found}/{n_total} word-language pairs")

    # Compute per-layer translation vs random cosine gaps
    print("  Computing per-layer gaps...")
    per_layer = {}

    for layer_idx in layers:
        trans_cosines = []
        for en_word, trans in translations:
            en_key = (en_word, "en")
            if en_key not in word_reps:
                continue
            en_vec = word_reps[en_key][layer_idx]

            for lang, word in trans.items():
                tgt_key = (word, lang)
                if tgt_key not in word_reps:
                    continue
                tgt_vec = word_reps[tgt_key][layer_idx]
                cos = F.cosine_similarity(
                    en_vec.unsqueeze(0), tgt_vec.unsqueeze(0)
                ).item()
                trans_cosines.append(cos)

        # Random control: random en-X pairs (matching the translation metric's structure)
        random.seed(42)
        rand_cosines = []
        en_keys = [k for k in word_reps if k[1] == "en"]
        non_en_keys = [k for k in word_reps if k[1] != "en"]
        if en_keys and non_en_keys:
            for _ in range(len(trans_cosines)):
                k1 = random.choice(en_keys)
                k2 = random.choice(non_en_keys)
                v1 = word_reps[k1][layer_idx]
                v2 = word_reps[k2][layer_idx]
                cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                rand_cosines.append(cos)

        if trans_cosines and rand_cosines:
            mean_trans = sum(trans_cosines) / len(trans_cosines)
            mean_rand = sum(rand_cosines) / len(rand_cosines)
            gap = mean_trans - mean_rand
        else:
            mean_trans = mean_rand = gap = 0.0

        layer_name = "embedding" if layer_idx is None else f"layer_{layer_idx}"
        per_layer[layer_name] = {
            "layer_idx": layer_idx,
            "translation_cos_mean": mean_trans,
            "random_cos_mean": mean_rand,
            "gap": gap,
            "n_translation_pairs": len(trans_cosines),
            "n_random_pairs": len(rand_cosines),
        }

    return per_layer


def format_results(per_layer):
    """Format results as a markdown table and identify the peak layer."""
    # Find the peak layer first
    best_layer = None
    best_gap = -float("inf")
    for layer_name, res in per_layer.items():
        if res["gap"] > best_gap:
            best_gap = res["gap"]
            best_layer = layer_name

    lines = []
    lines.append("| Layer | Trans Cos | Random Cos | Gap | N pairs |")
    lines.append("|-------|-----------|------------|-----|---------|")

    for layer_name, res in sorted(per_layer.items(), key=lambda x: x[1]["layer_idx"] if x[1]["layer_idx"] is not None else -1):
        gap = res["gap"]
        marker = " ←" if layer_name == best_layer else ""
        lines.append(
            f"| {layer_name} | {res['translation_cos_mean']:.4f} | "
            f"{res['random_cos_mean']:.4f} | {gap:+.4f}{marker} | "
            f"{res['n_translation_pairs']} |"
        )

    lines.append("")
    lines.append(f"**Peak layer:** {best_layer} (gap = {best_gap:+.4f})")
    lines.append("")
    lines.append("The peak layer is where V5's anchor block should sit.")
    lines.append("These per-layer baseline gaps are the numbers V5 must beat.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T5: Per-layer cross-lingual similarity sweep")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Baseline checkpoint path (no hub)"
    )
    parser.add_argument(
        "--eval-dir", required=True,
        help="Path to eval directory (per-language subdirs)"
    )
    parser.add_argument(
        "--translations", default=None,
        help="Path to frequent_translations_llm.json (default: resources/)"
    )
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Layer indices to measure (default: all layers + embedding)"
    )
    parser.add_argument(
        "--max-occurrences", type=int, default=50,
        help="Max occurrences per word (default: 50)"
    )
    parser.add_argument(
        "--max-sequences", type=int, default=2000,
        help="Max eval sequences per language (default: 2000)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--output", default="temp/t5_layer_sweep.json",
        help="Output JSON path"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading baseline: {args.checkpoint}")
    model, hub_info = load_checkpoint(args.checkpoint, device=device, baseline=True)
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} transformer layers")

    # Determine which layers to measure
    if args.layers is not None:
        layers = [None] + args.layers  # always include embedding
    else:
        layers = [None] + list(range(n_layers))
    print(f"  Measuring {len(layers)} layers (embedding + {len(layers)-1} transformer layers)")

    # Load tokenizer and translations
    tokenizer = get_tokenizer(args.checkpoint)
    translations = load_translations(path=args.translations, single_token_only=False)
    translations = filter_loanwords_per_pair(translations)
    print(f"  Translation tuples: {len(translations)}")

    # Run the sweep
    print("\nRunning layer sweep...")
    per_layer = run_layer_sweep(
        model, tokenizer, translations, args.eval_dir, layers, device,
        max_occurrences=args.max_occurrences,
        max_sequences=args.max_sequences,
    )

    # Format and print results
    table = format_results(per_layer)
    print(f"\n{table}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"per_layer": per_layer, "checkpoint": args.checkpoint}, f, indent=2)
    print(f"\nResults saved to {args.output}")

    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# T5 — Layer Sweep Results\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write(table + "\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
