"""T8 — MEXA (Multilingual Evaluation via Cross-lingual Alignment).

Reimplements the core MEXA metric from Kargaran et al. (arXiv 2410.05873).
For each layer, builds an N×N cosine similarity matrix between parallel
English and target-language sentence embeddings (position-weighted average),
then checks what fraction of parallel pairs are mutual nearest neighbors.

Reports per-language-pair MEXA scores at each layer, plus mean-over-layers.
Compares hub vs baseline.

Data: FLORES-200 devtest, first 100 lines per language pair.

Usage:
    # With pre-downloaded FLORES text files:
    python diagnostics/test_t8_mexa.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --baseline /path/to/baseline/checkpoint-6500 \
        --flores-dir resources/flores200 \
        --output temp/t8_mexa_results.json

    # Download FLORES from HuggingFace (requires gated access):
    python diagnostics/test_t8_mexa.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --baseline /path/to/baseline/checkpoint-6500 \
        --hf-token YOUR_TOKEN \
        --output temp/t8_mexa_results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from diagnostics.test_utils import (
    NON_EN_LANGS,
    get_tokenizer,
    load_checkpoint,
)

# FLORES-200 language codes for our 6 languages
LANG_TO_FLORES = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "ar": "arb_Arab",
}


def load_flores_from_dir(flores_dir, lang, n_sentences=100):
    """Load FLORES sentences from a text file (one sentence per line).

    Tries multiple filename patterns:
      {flores_code}.txt, {lang}.txt, {flores_code}.devtest
    """
    flores_code = LANG_TO_FLORES.get(lang, lang)
    candidates = [
        os.path.join(flores_dir, f"{flores_code}.txt"),
        os.path.join(flores_dir, f"{lang}.txt"),
        os.path.join(flores_dir, f"{flores_code}.devtest"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines[:n_sentences]
    return None


def download_flores_hf(output_dir, hf_token, n_sentences=100):
    """Download FLORES-200 devtest from HuggingFace and save as text files."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    for lang, flores_code in LANG_TO_FLORES.items():
        out_path = os.path.join(output_dir, f"{flores_code}.txt")
        if os.path.isfile(out_path):
            continue

        try:
            pair = f"eng_Latn-{flores_code}" if lang != "en" else "eng_Latn-vie_Latn"
            ds = load_dataset("facebook/flores", pair, split="devtest", token=hf_token)

            sentences = []
            col = "sentence_" + flores_code if "sentence_" + flores_code in ds.column_names else flores_code
            if col not in ds.column_names:
                for c in ds.column_names:
                    if flores_code.lower() in c.lower():
                        col = c
                        break

            for i, row in enumerate(ds):
                if i >= n_sentences:
                    break
                sentences.append(row[col])

            with open(out_path, "w") as f:
                for s in sentences:
                    f.write(s.strip() + "\n")
            print(f"  Downloaded {flores_code}: {len(sentences)} sentences")

        except Exception as e:
            print(f"  Failed to download {flores_code}: {e}")


def compute_sentence_embeddings(model, tokenizer, sentences, device):
    """Compute position-weighted sentence embeddings at all layers.

    Matches the official MEXA implementation (embed_extractor.py):
    - Tokenization with padding=True (handles attention_mask correctly)
    - Position weights: w_t = t * attention_mask[t] (1-indexed, zero for padding)
    - Embedding = sum(h_lt * w_t) / sum(w_t)
    - Output in float64 for cosine precision (matching official compute_mexa.py)

    Args:
        model: HF causal LM
        tokenizer: tokenizer
        sentences: list of strings
        device: device

    Returns:
        dict: {layer_idx: np.array of shape (n_sentences, hidden_dim)}
        layer_idx 0 = embedding output, 1..N = after transformer layer 0..N-1
    """
    all_layer_embs = {}

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", padding=True).to(device)
        attention_mask = inputs["attention_mask"]
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Position weights masked by attention_mask (official formula)
        positions = torch.arange(
            start=1, end=seq_len + 1, device=device
        ).unsqueeze(0)  # (1, seq_len)
        weights = attention_mask * positions  # (1, seq_len)
        weight_sum = weights.sum(dim=-1).unsqueeze(-1)  # (1, 1)

        for layer_idx, hidden_state in enumerate(outputs["hidden_states"]):
            # hidden_state: (1, seq_len, hidden_dim)
            weighted_sum = torch.sum(
                hidden_state * weights.unsqueeze(-1), dim=1
            )  # (1, hidden_dim)
            emb = (weighted_sum / weight_sum).squeeze(0)
            emb = emb.to(torch.float32).cpu().numpy().astype(np.float64)

            if layer_idx not in all_layer_embs:
                all_layer_embs[layer_idx] = []
            all_layer_embs[layer_idx].append(emb)

    return {k: np.stack(v) for k, v in all_layer_embs.items()}


def compute_mexa_score(pivot_embs, target_embs):
    """Compute the MEXA score between pivot and target embeddings.

    MEXA = fraction of sentence pairs where the parallel pair's cosine
    similarity is strictly greater than all other entries in BOTH its
    row and column of the similarity matrix.

    Args:
        pivot_embs: (N, d) numpy array (float64)
        target_embs: (N, d) numpy array (float64)

    Returns:
        float: MEXA score in [0, 1]
    """
    N = pivot_embs.shape[0]
    assert target_embs.shape[0] == N

    # Normalize for cosine similarity
    pivot_norm = pivot_embs / np.linalg.norm(pivot_embs, axis=1, keepdims=True)
    target_norm = target_embs / np.linalg.norm(target_embs, axis=1, keepdims=True)

    # Cosine similarity matrix: (N, N)
    sim_matrix = pivot_norm @ target_norm.T

    correct = 0
    for i in range(N):
        diag = sim_matrix[i, i]
        # Must be strictly greater than all off-diagonal in BOTH row and column
        row_max = max(sim_matrix[i, j] for j in range(N) if j != i) if N > 1 else -float("inf")
        col_max = max(sim_matrix[j, i] for j in range(N) if j != i) if N > 1 else -float("inf")
        if diag > row_max and diag > col_max:
            correct += 1

    return correct / N


def run_mexa(model, hub_info, tokenizer, flores_sentences, device, layers=None):
    """Run MEXA for all language pairs.

    Args:
        model: the model
        hub_info: dict from load_checkpoint
        tokenizer: tokenizer
        flores_sentences: dict {lang: [sentences]}
        device: device
        layers: list of layer indices to report (None = all)

    Returns:
        dict with per-lang-pair, per-layer MEXA scores
    """
    if "en" not in flores_sentences:
        return {"error": "No English FLORES sentences"}

    # Compute embeddings for all languages
    print("  Computing sentence embeddings...")
    lang_embs = {}
    for lang, sentences in flores_sentences.items():
        print(f"    {lang}: {len(sentences)} sentences...", end=" ", flush=True)
        embs = compute_sentence_embeddings(model, tokenizer, sentences, device)
        lang_embs[lang] = embs
        n_layers = len(embs)
        print(f"{n_layers} layers")

    if layers is None:
        layers = sorted(lang_embs["en"].keys())

    # Compute MEXA per language pair, per layer
    results = {}
    for lang in NON_EN_LANGS:
        if lang not in lang_embs:
            continue

        pair_key = f"en-{lang}"
        per_layer = {}
        layer_scores = []

        for layer_idx in layers:
            en_embs = lang_embs["en"][layer_idx]
            tgt_embs = lang_embs[lang][layer_idx]

            n = min(len(en_embs), len(tgt_embs))
            score = compute_mexa_score(en_embs[:n], tgt_embs[:n])
            per_layer[layer_idx] = score
            layer_scores.append(score)

        mean_score = sum(layer_scores) / len(layer_scores) if layer_scores else 0.0
        best_layer = max(per_layer, key=per_layer.get) if per_layer else None
        best_score = per_layer.get(best_layer, 0.0)

        results[pair_key] = {
            "per_layer": per_layer,
            "mean_over_layers": mean_score,
            "best_layer": best_layer,
            "best_score": best_score,
            "n_sentences": n,
        }

    return results


def format_results(hub_results, baseline_results, checkpoint_name=""):
    """Format MEXA results as markdown."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")

    if "error" in hub_results:
        lines.append(f"Error: {hub_results['error']}")
        return "\n".join(lines)

    lines.append("| Lang Pair | Hub Mean | Hub Best | Base Mean | Base Best | Delta Mean | N |")
    lines.append("|-----------|----------|----------|-----------|-----------|------------|---|")

    hub_means = []
    base_means = []

    for pair in [f"en-{l}" for l in NON_EN_LANGS]:
        h = hub_results.get(pair, {})
        b = baseline_results.get(pair, {})
        hm = h.get("mean_over_layers", 0)
        hb = h.get("best_score", 0)
        bm = b.get("mean_over_layers", 0)
        bb = b.get("best_score", 0)
        n = h.get("n_sentences", 0)
        delta = hm - bm
        sign = "+" if delta >= 0 else ""

        if pair in hub_results:
            hub_means.append(hm)
        if pair in baseline_results:
            base_means.append(bm)

        hl = h.get("best_layer", "-")
        bl = b.get("best_layer", "-")
        lines.append(
            f"| {pair} | {hm:.4f} | {hb:.4f} (L{hl}) | {bm:.4f} | {bb:.4f} (L{bl}) | {sign}{delta:.4f} | {n} |"
        )

    avg_hub = sum(hub_means) / len(hub_means) if hub_means else 0
    avg_base = sum(base_means) / len(base_means) if base_means else 0
    lines.append("")
    lines.append(f"**Average MEXA (mean-over-layers):** Hub={avg_hub:.4f}, Baseline={avg_base:.4f}, "
                 f"Delta={avg_hub - avg_base:+.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T8: MEXA sentence-level alignment")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more hub checkpoint paths")
    parser.add_argument("--baseline", required=True,
                        help="Matched no-hub baseline checkpoint path")
    parser.add_argument("--flores-dir", default=None,
                        help="Directory with FLORES text files (one per language)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for downloading gated FLORES dataset")
    parser.add_argument("--n-sentences", type=int, default=100,
                        help="Number of parallel sentences (default: 100)")
    parser.add_argument("--layers", nargs="*", type=int, default=None,
                        help="Layer indices to report (default: all)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="temp/t8_mexa_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve FLORES data
    flores_dir = args.flores_dir
    if flores_dir is None:
        flores_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources", "flores200",
        )

    # Try loading FLORES; if missing, auto-download using --hf-token or HF_TOKEN env var
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Check if FLORES files already exist
    flores_missing = not os.path.isfile(
        os.path.join(flores_dir, f"{LANG_TO_FLORES['en']}.txt")
    )

    if flores_missing and hf_token:
        print(f"Downloading FLORES to {flores_dir}...")
        download_flores_hf(flores_dir, hf_token, args.n_sentences)
    elif flores_missing:
        print(f"FLORES files not found in {flores_dir} and no HF token available.")
        print(f"Set HF_TOKEN env var or pass --hf-token to download automatically.")

    # Load FLORES sentences
    print("Loading FLORES sentences...")
    flores_sentences = {}
    for lang in ["en"] + NON_EN_LANGS:
        sentences = load_flores_from_dir(flores_dir, lang, args.n_sentences)
        if sentences:
            flores_sentences[lang] = sentences
            print(f"  {lang}: {len(sentences)} sentences")
        else:
            print(f"  {lang}: not found in {flores_dir}")

    if "en" not in flores_sentences:
        print(f"ERROR: No English FLORES data found. Provide --flores-dir or --hf-token.")
        sys.exit(1)

    tokenizer = get_tokenizer(args.checkpoint[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run baseline
    print(f"\nLoading baseline: {args.baseline}")
    base_model, base_info = load_checkpoint(args.baseline, device=device, baseline=True)
    print("  Running MEXA...")
    baseline_results = run_mexa(
        base_model, base_info, tokenizer, flores_sentences, device, args.layers
    )
    del base_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # Run hub checkpoints
    all_results = {
        "baseline": {"checkpoint": args.baseline, **baseline_results},
        "checkpoints": [],
    }
    all_tables = []

    for ckpt_path in args.checkpoint:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nLoading hub: {ckpt_path}")
        model, hub_info = load_checkpoint(ckpt_path, device=device)
        print(f"  hub_type={hub_info['hub_type']}, placement={hub_info['placement']}")

        print("  Running MEXA...")
        hub_results = run_mexa(
            model, hub_info, tokenizer, flores_sentences, device, args.layers
        )

        ckpt_result = {
            "checkpoint": ckpt_path,
            "hub_type": hub_info["hub_type"],
            **hub_results,
        }
        all_results["checkpoints"].append(ckpt_result)

        table = format_results(hub_results, baseline_results, ckpt_name)
        all_tables.append(table)
        print(table)

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# T8 — MEXA Results\n\n")
        f.write(f"N sentences: {args.n_sentences}\n")
        f.write(f"Baseline: {args.baseline}\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
