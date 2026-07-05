"""T6 — Translation Retrieval (Bilingual Lexicon Induction) via CSLS.

For each English word, score all single-token target-language words by CSLS
and check whether the true translation is the nearest neighbor.

Reports P@1 and P@5 per language pair, hub vs baseline.

Usage:
    python diagnostics/test_t6_bli_retrieval.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --baseline /path/to/baseline/checkpoint-6500 \
        --output temp/t6_bli_results.json

    # Multiple checkpoints (e.g. training progression):
    python diagnostics/test_t6_bli_retrieval.py \
        --checkpoint /path/to/ckpt-1500 /path/to/ckpt-3250 /path/to/ckpt-6500 \
        --baseline /path/to/baseline/ckpt-6500 \
        --output temp/t6_bli_results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from diagnostics.test_utils import (
    NON_EN_LANGS,
    compute_csls_scores,
    filter_loanwords_per_pair,
    get_representation_at_anchor_layer,
    get_tokenizer,
    load_checkpoint,
    load_translations,
)


def build_word_embeddings(model, hub_info, tokenizer, words, device, batch_size=256):
    """Batch-compute anchor-layer representations for a list of single-token words.

    Args:
        model: the loaded model
        hub_info: dict from load_checkpoint
        tokenizer: the tokenizer
        words: list of words (all assumed single-token)
        device: target device
        batch_size: how many words per forward pass

    Returns:
        (embeddings_tensor, valid_words, valid_indices)
        embeddings_tensor: (N_valid, hidden_dim) float32 tensor, L2-normalized
        valid_words: list of words that were actually single-token
        valid_indices: original indices of valid words
    """
    # First, tokenize all words and filter to single-token
    all_ids = []
    valid_words = []
    valid_indices = []
    for i, word in enumerate(words):
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            all_ids.append(ids[0])
            valid_words.append(word)
            valid_indices.append(i)

    if not all_ids:
        return torch.empty(0, model.config.hidden_size), [], []

    # Batch forward passes
    all_reps = []
    for start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[start:start + batch_size]
        # Each word is 1 token, so input shape is (batch, 1)
        input_ids = torch.tensor(batch_ids, device=device).unsqueeze(1)
        reps = get_representation_at_anchor_layer(model, hub_info, input_ids)
        # reps: (batch, 1, hidden_dim) -> (batch, hidden_dim)
        all_reps.append(reps.squeeze(1).float())

    embeddings = torch.cat(all_reps, dim=0)
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings, valid_words, valid_indices


def run_bli_for_lang_pair(
    model, hub_info, tokenizer, translations, target_lang, device, csls_k=10
):
    """Run BLI retrieval for one language pair (en -> target_lang).

    Args:
        model: loaded model
        hub_info: dict from load_checkpoint
        tokenizer: tokenizer
        translations: list of (en_word, {lang: word}) tuples
        target_lang: target language code (e.g. "vi")
        device: device
        csls_k: k for CSLS neighbor computation

    Returns:
        dict with keys: p_at_1, p_at_5, n_pairs, n_candidates
    """
    # Collect valid pairs: en word must be single-token, target word must be single-token
    en_words = []
    tgt_words = []
    pair_en_indices = []  # index into en_words for each pair
    pair_tgt_indices = []  # index into tgt_words for each pair

    # Build unique word lists
    en_word_set = {}  # word -> index
    tgt_word_set = {}  # word -> index

    for en_word, trans in translations:
        if target_lang not in trans:
            continue
        tgt_word = trans[target_lang]

        # Check single-token
        en_ids = tokenizer(en_word, add_special_tokens=False)["input_ids"]
        tgt_ids = tokenizer(tgt_word, add_special_tokens=False)["input_ids"]
        if len(en_ids) != 1 or len(tgt_ids) != 1:
            continue

        # Skip if en and target are identical (loanword)
        if en_word.lower() == tgt_word.lower():
            continue

        if en_word not in en_word_set:
            en_word_set[en_word] = len(en_words)
            en_words.append(en_word)
        if tgt_word not in tgt_word_set:
            tgt_word_set[tgt_word] = len(tgt_words)
            tgt_words.append(tgt_word)

        pair_en_indices.append(en_word_set[en_word])
        pair_tgt_indices.append(tgt_word_set[tgt_word])

    if not pair_en_indices:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "n_pairs": 0, "n_candidates": 0}

    # Build embeddings for all source and target words
    en_embs, en_valid, _ = build_word_embeddings(model, hub_info, tokenizer, en_words, device)
    tgt_embs, tgt_valid, _ = build_word_embeddings(model, hub_info, tokenizer, tgt_words, device)

    # Map from original indices to embedding indices
    en_emb_idx = {word: i for i, word in enumerate(en_valid)}
    tgt_emb_idx = {word: i for i, word in enumerate(tgt_valid)}

    # Compute CSLS scores: (N_en, N_tgt)
    csls_scores = compute_csls_scores(en_embs, tgt_embs, k=csls_k)

    # Evaluate P@1 and P@5
    hits_1 = 0
    hits_5 = 0
    n_evaluated = 0

    for pair_idx in range(len(pair_en_indices)):
        en_idx_orig = pair_en_indices[pair_idx]
        tgt_idx_orig = pair_tgt_indices[pair_idx]
        en_word = en_words[en_idx_orig]
        tgt_word = tgt_words[tgt_idx_orig]

        if en_word not in en_emb_idx or tgt_word not in tgt_emb_idx:
            continue

        en_emb_i = en_emb_idx[en_word]
        tgt_emb_i = tgt_emb_idx[tgt_word]

        # Get the row of CSLS scores for this English word
        scores = csls_scores[en_emb_i]  # (N_tgt,)

        # Rank all target candidates
        ranked = scores.argsort(descending=True)

        # Find the rank of the true translation
        rank = (ranked == tgt_emb_i).nonzero(as_tuple=True)[0]
        if len(rank) == 0:
            continue
        rank = rank[0].item()

        n_evaluated += 1
        if rank == 0:
            hits_1 += 1
        if rank < 5:
            hits_5 += 1

    if n_evaluated == 0:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "n_pairs": 0, "n_candidates": len(tgt_valid)}

    return {
        "p_at_1": hits_1 / n_evaluated,
        "p_at_5": hits_5 / n_evaluated,
        "n_pairs": n_evaluated,
        "n_candidates": len(tgt_valid),
    }


def run_bli_all_langs(model, hub_info, tokenizer, translations, device, csls_k=10):
    """Run BLI for all language pairs (en -> each non-English lang).

    Returns:
        dict of {lang_pair: {p_at_1, p_at_5, n_pairs, n_candidates}}
    """
    results = {}
    for lang in NON_EN_LANGS:
        result = run_bli_for_lang_pair(
            model, hub_info, tokenizer, translations, lang, device, csls_k
        )
        results[f"en-{lang}"] = result
    return results


def summarize_results(results_per_lang):
    """Compute summary statistics: mean P@1/P@5, related vs distant."""
    related = ["en-de", "en-vi"]
    distant = ["en-zh", "en-ar", "en-ru"]

    all_p1 = []
    all_p5 = []
    related_p1 = []
    distant_p1 = []

    for pair, res in results_per_lang.items():
        if res["n_pairs"] == 0:
            continue
        all_p1.append(res["p_at_1"])
        all_p5.append(res["p_at_5"])
        if pair in related:
            related_p1.append(res["p_at_1"])
        elif pair in distant:
            distant_p1.append(res["p_at_1"])

    return {
        "mean_p1": sum(all_p1) / len(all_p1) if all_p1 else 0.0,
        "mean_p5": sum(all_p5) / len(all_p5) if all_p5 else 0.0,
        "related_mean_p1": sum(related_p1) / len(related_p1) if related_p1 else 0.0,
        "distant_mean_p1": sum(distant_p1) / len(distant_p1) if distant_p1 else 0.0,
    }


def format_results_table(hub_results, baseline_results, checkpoint_name=""):
    """Format results as a markdown table."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")
    lines.append("| Lang Pair | Hub P@1 | Hub P@5 | Base P@1 | Base P@5 | Delta P@1 | N pairs |")
    lines.append("|-----------|---------|---------|----------|----------|-----------|---------|")

    for pair in ["en-vi", "en-zh", "en-ru", "en-de", "en-ar"]:
        h = hub_results.get(pair, {})
        b = baseline_results.get(pair, {})
        hp1 = h.get("p_at_1", 0)
        hp5 = h.get("p_at_5", 0)
        bp1 = b.get("p_at_1", 0)
        bp5 = b.get("p_at_5", 0)
        delta = hp1 - bp1
        n = h.get("n_pairs", 0)
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {pair} | {hp1:.4f} | {hp5:.4f} | {bp1:.4f} | {bp5:.4f} | {sign}{delta:.4f} | {n} |"
        )

    h_sum = summarize_results(hub_results)
    b_sum = summarize_results(baseline_results)
    lines.append("")
    lines.append(f"**Hub mean P@1:** {h_sum['mean_p1']:.4f} "
                 f"(related: {h_sum['related_mean_p1']:.4f}, distant: {h_sum['distant_mean_p1']:.4f})")
    lines.append(f"**Baseline mean P@1:** {b_sum['mean_p1']:.4f} "
                 f"(related: {b_sum['related_mean_p1']:.4f}, distant: {b_sum['distant_mean_p1']:.4f})")
    lines.append(f"**Delta mean P@1:** {h_sum['mean_p1'] - b_sum['mean_p1']:+.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T6: BLI retrieval via CSLS")
    parser.add_argument(
        "--checkpoint", nargs="+", required=True,
        help="One or more hub checkpoint paths"
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Matched no-hub baseline checkpoint path"
    )
    parser.add_argument(
        "--translations", default=None,
        help="Path to frequent_translations_llm.json (default: resources/)"
    )
    parser.add_argument(
        "--csls-k", type=int, default=10,
        help="k for CSLS neighbor computation (default: 10)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--output", default="temp/t6_bli_results.json",
        help="Output JSON path"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer from the first checkpoint
    tokenizer = get_tokenizer(args.checkpoint[0])
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load translations with per-pair loanword filtering (removes only the specific
    # language entry where translation == English word, keeps all other pairs intact).
    translations = load_translations(path=args.translations, single_token_only=False)
    translations = filter_loanwords_per_pair(translations)
    print(f"Translation tuples (per-pair loanword filtered): {len(translations)}")

    # Determine measurement layer from the first hub checkpoint config (without
    # loading the full model). All checkpoints in one run should share the same
    # placement — if they don't, we re-run baseline per unique layer.
    hub_layer_configs = []
    for ckpt_path in args.checkpoint:
        v3_cfg_path = os.path.join(ckpt_path, "embhub_v3_config.json")
        v2_cfg_path = os.path.join(ckpt_path, "embhub_config.json")
        if os.path.isfile(v3_cfg_path):
            with open(v3_cfg_path) as f:
                cfg = json.load(f)
            hub_layer_configs.append((cfg.get("placement", "embedding"), cfg.get("layer_idx", 0)))
        elif os.path.isfile(v2_cfg_path):
            hub_layer_configs.append(("embedding", 0))
        else:
            hub_layer_configs.append(("embedding", 0))

    # Get unique measurement layers needed
    unique_layers = list(dict.fromkeys(hub_layer_configs))

    # Run baseline at each needed measurement layer
    print(f"\nLoading baseline: {args.baseline}")
    base_model, base_info = load_checkpoint(args.baseline, device=device, baseline=True)

    baseline_results_by_layer = {}
    for placement, layer_idx in unique_layers:
        # Override base_info to measure at the hub's anchor layer
        measure_info = dict(base_info)
        measure_info["placement"] = placement
        measure_info["layer_idx"] = layer_idx
        layer_desc = f"layer {layer_idx}" if placement == "mid" else "embedding"
        print(f"  Running BLI at {layer_desc}...")
        results = run_bli_all_langs(
            base_model, measure_info, tokenizer, translations, device, args.csls_k
        )
        baseline_results_by_layer[(placement, layer_idx)] = results

    del base_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # Run each hub checkpoint
    all_results = {
        "baseline": {
            "checkpoint": args.baseline,
            "per_layer": {
                f"{p}_{l}": {"per_lang": r, "summary": summarize_results(r)}
                for (p, l), r in baseline_results_by_layer.items()
            },
        },
        "checkpoints": [],
    }
    all_tables = []

    for ckpt_idx, ckpt_path in enumerate(args.checkpoint):
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nLoading hub checkpoint: {ckpt_path}")
        model, hub_info = load_checkpoint(ckpt_path, device=device)
        print(f"  hub_type={hub_info['hub_type']}, placement={hub_info['placement']}, "
              f"layer_idx={hub_info['layer_idx']}")

        if not hub_info["has_hub"]:
            print(f"  WARNING: no hub found in {ckpt_path}, treating as baseline")

        print(f"  Running BLI...")
        hub_results = run_bli_all_langs(
            model, hub_info, tokenizer, translations, device, args.csls_k
        )

        # Use baseline measured at the SAME layer as this hub checkpoint
        layer_key = hub_layer_configs[ckpt_idx]
        baseline_results = baseline_results_by_layer[layer_key]

        ckpt_result = {
            "checkpoint": ckpt_path,
            "hub_type": hub_info["hub_type"],
            "placement": hub_info["placement"],
            "layer_idx": hub_info["layer_idx"],
            "per_lang": hub_results,
            "summary": summarize_results(hub_results),
        }
        all_results["checkpoints"].append(ckpt_result)

        table = format_results_table(hub_results, baseline_results, ckpt_name)
        all_tables.append(table)
        print(table)

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Also save markdown summary
    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# T6 — BLI Retrieval Results (CSLS)\n\n")
        f.write(f"CSLS k={args.csls_k}\n")
        f.write(f"Baseline: {args.baseline}\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown summary saved to {md_path}")


if __name__ == "__main__":
    main()
