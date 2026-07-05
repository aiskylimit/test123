"""T7 — Generative code-switch continuation test.

Tests whether the model's own output (through the LM head) behaves more
cross-lingually with the hub. Builds minimal prompts whose correct next
token is a known translation, then measures log P(correct) - log P(random).

Usage:
    python diagnostics/test_t7_generative_codeswitch.py \
        --checkpoint /path/to/hub/checkpoint-6500 \
        --baseline /path/to/baseline/checkpoint-6500 \
        --output temp/t7_codeswitch.json
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
    filter_loanwords_per_pair,
    get_tokenizer,
    load_checkpoint,
    load_translations,
)

LANG_NAMES = {
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ru": "Russian",
    "de": "German",
    "ar": "Arabic",
}

TEMPLATES_EN_TO_X = [
    '"{word}" in {lang_name} is',
    'the {lang_name} word for "{word}" is',
    '"{word}" translates to {lang_name} as',
]

TEMPLATES_X_TO_EN = [
    '"{word}" in English is',
    '"{word}" =',
    '"{word}" means',
]


def build_prompts(en_word, tgt_word, lang, tokenizer):
    """Build prompt-target pairs for one translation pair.

    Returns:
        list of (prompt_text, target_token_id, direction) tuples
        Only includes prompts where the target is a single token.
    """
    lang_name = LANG_NAMES.get(lang, lang)
    prompts = []

    # en -> X: prompt with English word, target is foreign word
    tgt_ids = tokenizer(tgt_word, add_special_tokens=False)["input_ids"]
    if len(tgt_ids) == 1:
        for template in TEMPLATES_EN_TO_X:
            prompt = template.format(word=en_word, lang_name=lang_name)
            prompts.append((prompt, tgt_ids[0], "en_to_x"))

    # X -> en: prompt with foreign word, target is English word
    en_ids = tokenizer(en_word, add_special_tokens=False)["input_ids"]
    if len(en_ids) == 1:
        for template in TEMPLATES_X_TO_EN:
            prompt = template.format(word=tgt_word)
            prompts.append((prompt, en_ids[0], "x_to_en"))

    return prompts


def get_next_token_logprob(model, tokenizer, prompt, target_token_id, device):
    """Get log P(target_token | prompt).

    Returns:
        float: log probability of the target token given the prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # The next-token prediction is at the last position
    last_logits = logits[0, -1].float()  # (vocab_size,)
    log_probs = F.log_softmax(last_logits, dim=-1)

    return log_probs[target_token_id].item()


def find_random_control_token(exclude_word, answer_lang, translations, tokenizer):
    """Find a random single-token word in answer_lang for the control.

    Args:
        exclude_word: word to exclude (the correct answer)
        answer_lang: language of the answer ("en" for X→en, target lang for en→X)
        translations: list of (en_word, {lang: word}) tuples
        tokenizer: tokenizer

    Returns the token ID of a random word in answer_lang, or None if not found.
    """
    for other_en, other_trans in translations:
        if answer_lang == "en":
            # Looking for a random English word
            if other_en == exclude_word:
                continue
            other_ids = tokenizer(other_en, add_special_tokens=False)["input_ids"]
            if len(other_ids) == 1:
                return other_ids[0]
        else:
            # Looking for a random word in target language
            if answer_lang not in other_trans:
                continue
            other_word = other_trans[answer_lang]
            if other_word == exclude_word:
                continue
            if other_word.lower() == other_en.lower():
                continue
            other_ids = tokenizer(other_word, add_special_tokens=False)["input_ids"]
            if len(other_ids) == 1:
                return other_ids[0]
    return None


def run_codeswitch(model, hub_info, tokenizer, translations, device, max_pairs=500):
    """Run the generative code-switch test.

    Returns:
        dict with per-language and per-direction results
    """
    results = {}

    for lang in NON_EN_LANGS:
        print(f"    {lang}:", end=" ", flush=True)

        margins = []  # log P(correct) - log P(random)
        margins_en_to_x = []
        margins_x_to_en = []
        n_tested = 0

        for en_word, trans in translations:
            if lang not in trans:
                continue
            tgt_word = trans[lang]

            # Skip loanwords
            if en_word.lower() == tgt_word.lower():
                continue

            # Build prompts
            prompts = build_prompts(en_word, tgt_word, lang, tokenizer)
            if not prompts:
                continue

            # Find random controls per direction
            # en→X: random control is a random target-language word
            # X→en: random control is a random English word
            rand_en_to_x = find_random_control_token(tgt_word, lang, translations, tokenizer)
            rand_x_to_en = find_random_control_token(en_word, "en", translations, tokenizer)

            for prompt_text, target_id, direction in prompts:
                if direction == "en_to_x":
                    random_token_id = rand_en_to_x
                else:
                    random_token_id = rand_x_to_en

                if random_token_id is None:
                    continue

                log_p_correct = get_next_token_logprob(
                    model, tokenizer, prompt_text, target_id, device
                )
                log_p_random = get_next_token_logprob(
                    model, tokenizer, prompt_text, random_token_id, device
                )

                margin = log_p_correct - log_p_random
                margins.append(margin)
                if direction == "en_to_x":
                    margins_en_to_x.append(margin)
                else:
                    margins_x_to_en.append(margin)

                n_tested += 1

            if n_tested >= max_pairs:
                break

        if margins:
            results[f"en-{lang}"] = {
                "mean_margin": sum(margins) / len(margins),
                "mean_margin_en_to_x": sum(margins_en_to_x) / len(margins_en_to_x) if margins_en_to_x else 0,
                "mean_margin_x_to_en": sum(margins_x_to_en) / len(margins_x_to_en) if margins_x_to_en else 0,
                "n_prompts": len(margins),
                "n_en_to_x": len(margins_en_to_x),
                "n_x_to_en": len(margins_x_to_en),
            }
            print(f"margin={sum(margins)/len(margins):.4f} (n={len(margins)})")
        else:
            print("no valid prompts")

    return results


def format_results(hub_results, baseline_results, checkpoint_name=""):
    """Format results as markdown."""
    lines = []
    if checkpoint_name:
        lines.append(f"\n### {checkpoint_name}\n")

    if "error" in hub_results:
        lines.append(f"Error: {hub_results['error']}")
        return "\n".join(lines)

    lines.append("| Lang Pair | Hub Margin | Base Margin | Delta | N prompts |")
    lines.append("|-----------|-----------|-------------|-------|-----------|")

    for pair in [f"en-{l}" for l in NON_EN_LANGS]:
        h = hub_results.get(pair, {})
        b = baseline_results.get(pair, {})
        hm = h.get("mean_margin", 0)
        bm = b.get("mean_margin", 0)
        n = h.get("n_prompts", 0)
        delta = hm - bm
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {pair} | {hm:.4f} | {bm:.4f} | {sign}{delta:.4f} | {n} |")

    hub_margins = [r["mean_margin"] for r in hub_results.values() if isinstance(r, dict) and "mean_margin" in r]
    base_margins = [r["mean_margin"] for r in baseline_results.values() if isinstance(r, dict) and "mean_margin" in r]
    avg_hub = sum(hub_margins) / len(hub_margins) if hub_margins else 0
    avg_base = sum(base_margins) / len(base_margins) if base_margins else 0

    lines.append("")
    lines.append(f"**Average margin:** Hub={avg_hub:.4f}, Baseline={avg_base:.4f}, "
                 f"Delta={avg_hub - avg_base:+.4f}")
    lines.append("")
    lines.append("Margin = log P(correct translation) - log P(random word). "
                 "Higher = model prefers the correct translation more strongly.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T7: Generative code-switch continuation")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more hub checkpoint paths")
    parser.add_argument("--baseline", required=True,
                        help="Matched no-hub baseline checkpoint path")
    parser.add_argument("--translations", default=None)
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="Max prompt pairs per language (default: 500)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="temp/t7_codeswitch.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(args.checkpoint[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    translations = load_translations(path=args.translations, single_token_only=False)
    translations = filter_loanwords_per_pair(translations)
    print(f"Translation tuples: {len(translations)}")

    # Baseline
    print(f"\nLoading baseline: {args.baseline}")
    base_model, base_info = load_checkpoint(args.baseline, device=device, baseline=True)
    print("  Running code-switch test...")
    baseline_results = run_codeswitch(
        base_model, base_info, tokenizer, translations, device, args.max_pairs
    )
    del base_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # Hub checkpoints
    all_results = {"baseline": baseline_results, "checkpoints": []}
    all_tables = []

    for ckpt_path in args.checkpoint:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nLoading hub: {ckpt_path}")
        model, hub_info = load_checkpoint(ckpt_path, device=device)
        print(f"  hub_type={hub_info['hub_type']}, placement={hub_info['placement']}")

        print("  Running code-switch test...")
        hub_results = run_codeswitch(
            model, hub_info, tokenizer, translations, device, args.max_pairs
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
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# T7 — Generative Code-Switch Results\n\n")
        f.write(f"Baseline: {args.baseline}\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
