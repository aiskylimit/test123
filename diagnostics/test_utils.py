"""Shared utilities for EmbHub evaluation tests (T1-T8).

Provides:
- Universal checkpoint loading (V2 additive / V3+ variants)
- Token-level and layer-level representation extraction
- Translation pair loading and filtering
- CSLS similarity computation
- Frequency-matched random pair generation
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model_wrapper_v2 import (
    EMBHUB_CONFIG_NAME,
    EMBHUB_WEIGHTS_NAME,
    inject_embhub,
)
from model_wrapper_v3 import (
    EMBHUB_V3_CONFIG_NAME,
    EMBHUB_V3_WEIGHTS_NAME,
    inject_embhub_v3,
)

LANGS = ["en", "vi", "zh", "ru", "de", "ar"]
NON_EN_LANGS = ["vi", "zh", "ru", "de", "ar"]


def load_checkpoint(checkpoint_path, device="cpu", dtype=torch.bfloat16, baseline=False):
    """Load a checkpoint, auto-detecting V2 vs V3+ hub type.

    Args:
        checkpoint_path: path to a HF checkpoint directory
        device: target device
        dtype: model dtype (default bfloat16)
        baseline: if True, load as a plain model (no hub)

    Returns:
        (model, hub_info) where hub_info is a dict:
        {
            "hub": nn.Module or None,
            "hub_type": str ("v2_additive"|"v3"|"v2_concat"|"v2_topk"|"v6"|"v6f"|None),
            "placement": str ("embedding"|"mid"),
            "layer_idx": int,
            "has_hub": bool,
        }
    """
    config = AutoConfig.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, torch_dtype=dtype
    )

    hub_info = {
        "hub": None,
        "hub_type": None,
        "placement": "embedding",
        "layer_idx": 0,
        "has_hub": False,
    }

    if baseline:
        model.to(device)
        model.eval()
        return model, hub_info

    v3_cfg_path = os.path.join(checkpoint_path, EMBHUB_V3_CONFIG_NAME)
    v3_wt_path = os.path.join(checkpoint_path, EMBHUB_V3_WEIGHTS_NAME)
    v2_cfg_path = os.path.join(checkpoint_path, EMBHUB_CONFIG_NAME)
    v2_wt_path = os.path.join(checkpoint_path, EMBHUB_WEIGHTS_NAME)

    if os.path.isfile(v3_cfg_path) and os.path.isfile(v3_wt_path):
        with open(v3_cfg_path) as f:
            hub_cfg = json.load(f)
        hub = inject_embhub_v3(
            model,
            hub_type=hub_cfg.get("hub_type", "v3"),
            num_embeddings=hub_cfg["num_embeddings"],
            num_heads=hub_cfg.get("num_heads", 1),
            use_mlp=hub_cfg.get("use_mlp", False),
            top_k=hub_cfg.get("top_k", 10),
            weighting=hub_cfg.get("weighting", "raw_softmax"),
            tail_mode=hub_cfg.get("tail_mode", "none"),
            num_buckets=hub_cfg.get("num_buckets", 10),
            r_budget=hub_cfg.get("r_budget", 0.3),
            p_only=hub_cfg.get("p_only", 0.10),
            p_both=hub_cfg.get("p_both", 0.40),
            anneal_steps=hub_cfg.get("anneal_steps", 2000),
            placement=hub_cfg.get("placement", "embedding"),
            layer_idx=hub_cfg.get("layer_idx", 10),
        )
        state_dict = torch.load(v3_wt_path, map_location="cpu", weights_only=True)
        hub.load_state_dict(state_dict)
        hub_info = {
            "hub": hub,
            "hub_type": hub_cfg.get("hub_type", "v3"),
            "placement": hub_cfg.get("placement", "embedding"),
            "layer_idx": hub_cfg.get("layer_idx", 0),
            "has_hub": True,
        }

    elif os.path.isfile(v2_cfg_path) and os.path.isfile(v2_wt_path):
        with open(v2_cfg_path) as f:
            hub_cfg = json.load(f)
        hub = inject_embhub(
            model,
            num_embeddings=hub_cfg["num_embeddings"],
            alpha=hub_cfg["alpha"],
        )
        state_dict = torch.load(v2_wt_path, map_location="cpu", weights_only=True)
        hub.load_state_dict(state_dict)
        hub_info = {
            "hub": hub,
            "hub_type": "v2_additive",
            "placement": "embedding",
            "layer_idx": 0,
            "has_hub": True,
        }

    model.to(device)
    model.eval()
    return model, hub_info


def get_tokenizer(checkpoint_path):
    """Load the tokenizer from a checkpoint or fall back to Qwen/Qwen3-0.6B."""
    try:
        return AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def get_embedding_layer(model):
    """Get the token embedding layer from a HF model."""
    return model.get_input_embeddings()


def get_token_embedding(model, token_ids, device=None):
    """Get raw token embeddings (before any hub processing).

    Args:
        model: the HF model
        token_ids: tensor of shape (*, ) with token IDs
        device: move token_ids to this device if given

    Returns:
        tensor of shape (*, embedding_dim)
    """
    if device is not None:
        token_ids = token_ids.to(device)
    embedding = get_embedding_layer(model)
    with torch.no_grad():
        return embedding(token_ids)


def get_representations_at_layer(model, input_ids, layer_idx=None):
    """Get hidden states at a specific layer via a forward pass.

    Args:
        model: the HF model (with or without hub — hub hook runs automatically)
        input_ids: (batch, seq_len) tensor
        layer_idx: which layer to return. None = embedding output (layer 0 input).
                   0..N-1 = after transformer layer i.

    Returns:
        hidden_states tensor of shape (batch, seq_len, hidden_dim)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors:
    # [embedding_output, after_layer_0, after_layer_1, ..., after_layer_N-1]
    if layer_idx is None:
        return outputs.hidden_states[0]
    return outputs.hidden_states[layer_idx + 1]


def get_all_layer_representations(model, input_ids):
    """Get hidden states at ALL layers via a single forward pass.

    Returns:
        tuple of tensors: (embedding_output, after_layer_0, ..., after_layer_N-1)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    return outputs.hidden_states


def get_anchor_layer_idx(hub_info):
    """Get the layer index where the hub operates.

    For embedding-layer hubs, returns None (embedding output = hidden_states[0]).
    For mid-layer hubs, returns layer_idx (hidden_states[layer_idx + 1]).
    """
    if hub_info["placement"] == "mid":
        return hub_info["layer_idx"]
    return None


def get_representation_at_anchor_layer(model, hub_info, input_ids):
    """Get representations at the layer where the hub operates.

    This is the correct layer for Test B / T6 comparisons.
    NOTE: Returns POST-hub representations (the hub hook fires during the
    forward pass). For PRE-hub representations, use get_hub_input().
    """
    layer_idx = get_anchor_layer_idx(hub_info)
    return get_representations_at_layer(model, input_ids, layer_idx)


def get_hub_input(model, hub_info, input_ids, device):
    """Get the input tensor that the hub operates on (BEFORE hub processing).

    For embedding-layer hubs: raw token embeddings (bypasses the hub hook).
    For mid-layer hubs: hidden state at the hub's layer output, before the
        hub hook modifies it (temporarily removes and re-attaches the hook).

    This is the correct input for computing anchor weights, contribution
    share, etc. — anywhere you need the representation the hub SEES, not
    what it PRODUCES.
    """
    from model_wrapper_v3 import _get_transformer_layers

    input_ids = input_ids.to(device)

    if hub_info["placement"] == "embedding":
        with torch.no_grad():
            return F.embedding(input_ids, model.get_input_embeddings().weight)

    # Mid-layer: capture the transformer layer's output BEFORE the hub hook.
    layer_idx = hub_info["layer_idx"]
    layers = _get_transformer_layers(model)
    target_layer = layers[layer_idx]
    hub = model.embhub

    hook_handle = model._embhub_hook_handle
    hook_handle.remove()

    captured = {}
    def capture_hook(mod, inp, out):
        if isinstance(out, tuple):
            captured["hidden"] = out[0].detach()
        else:
            captured["hidden"] = out.detach()

    handle = target_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    # Re-attach the hub hook
    def mid_layer_hook(mod, inp, out):
        if isinstance(out, tuple):
            modified = hub(out[0])
            return (modified,) + out[1:]
        return hub(out)
    model._embhub_hook_handle = target_layer.register_forward_hook(mid_layer_hook)

    return captured["hidden"]


# ---------------------------------------------------------------------------
# Translation pairs
# ---------------------------------------------------------------------------

def load_translations(path=None, single_token_only=False, tokenizer=None):
    """Load translation tuples from the LLM-generated translations file.

    Args:
        path: path to frequent_translations_llm.json. If None, uses the
              committed resources/frequent_translations_llm.json.
        single_token_only: if True, filter to tuples where ALL words are single-token.
        tokenizer: required if single_token_only=True.

    Returns:
        list of (en_word, {lang: word}) tuples
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "frequent_translations_llm.json",
        )

    with open(path) as f:
        data = json.load(f)

    tuples = []
    for en_word, translations in data["all_five_tuples"].items():
        tuples.append((en_word, translations))

    if single_token_only:
        if tokenizer is None:
            raise ValueError("tokenizer required when single_token_only=True")
        tuples = _filter_single_token(tuples, tokenizer)

    return tuples


def _filter_single_token(tuples, tokenizer):
    """Keep only tuples where every word (en + all translations) is a single token."""
    filtered = []
    for en_word, translations in tuples:
        ids_en = tokenizer(en_word, add_special_tokens=False)["input_ids"]
        if len(ids_en) != 1:
            continue
        all_single = True
        for lang, word in translations.items():
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                all_single = False
                break
        if all_single:
            filtered.append((en_word, translations))
    return filtered


def filter_loanwords(tuples):
    """Remove tuples where any non-English translation is identical to the English word.

    NOTE: This is the strict/legacy version — removes the ENTIRE tuple if any
    single language is a loanword. Prefer filter_loanwords_per_pair() which only
    removes the specific loanword pair and keeps valid pairs from the same tuple.
    """
    filtered = []
    for en_word, translations in tuples:
        is_loan = any(w.lower() == en_word.lower() for w in translations.values())
        if not is_loan:
            filtered.append((en_word, translations))
    return filtered


def filter_loanwords_per_pair(tuples):
    """Remove only the specific language pairs where translation == English word.

    Keeps the tuple and all other language pairs intact.

    Example:
        ("bus", {"vi": "bus", "zh": "公共汽车", "de": "Bus", "ar": "حافلة"})
      → ("bus", {"zh": "公共汽车", "ar": "حافلة"})
        # en-vi and en-de removed (loanwords), en-zh and en-ar kept

    Tuples with no remaining translations after filtering are dropped entirely.

    Works with the (en_word, {lang: word}) format used by load_translations().
    For the old probe's flat dict format, use filter_loanwords_per_pair_flat().
    """
    filtered = []
    for en_word, translations in tuples:
        clean_trans = {
            lang: word for lang, word in translations.items()
            if word.lower() != en_word.lower()
        }
        if clean_trans:
            filtered.append((en_word, clean_trans))
    return filtered


def filter_loanwords_per_pair_flat(tuples):
    """Per-pair loanword filter for the old probe's flat dict format.

    Input format:  [{"en": "bus", "vi": "bus", "zh": "公共汽车", "de": "Bus", "ar": "حافلة"}, ...]
    Output format: [{"en": "bus", "zh": "公共汽车", "ar": "حافلة"}, ...]

    Removes language entries where the word is identical to the English word
    (case-insensitive). Keeps the "en" key always. Drops tuples with no
    remaining non-English entries.

    Drop-in replacement for the strict filter in anchor_probe2_muse_no_loan_word.py.
    """
    filtered = []
    for tup in tuples:
        en_word = tup.get("en", "")
        clean = {"en": en_word}
        for lang, word in tup.items():
            if lang == "en":
                continue
            if word.lower() != en_word.lower():
                clean[lang] = word
        if len(clean) > 1:
            filtered.append(clean)
    return filtered


def get_word_representation(model, hub_info, tokenizer, word, device):
    """Get the representation of a word at the anchor layer.

    For single-token words, returns the representation directly.
    For multi-token words, returns the mean-pool over sub-tokens.

    Returns:
        (representation_vector, num_tokens)
    """
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    reps = get_representation_at_anchor_layer(model, hub_info, ids)
    # reps shape: (1, num_tokens, hidden_dim)
    rep_mean = reps.squeeze(0).float().mean(dim=0)
    return rep_mean, ids.shape[1]


# ---------------------------------------------------------------------------
# CSLS (Cross-domain Similarity Local Scaling)
# ---------------------------------------------------------------------------

def compute_csls_scores(source_embs, target_embs, k=10):
    """Compute CSLS scores between source and target embeddings.

    CSLS(e, x) = 2*cos(e, x) - r(e) - r(x)
    where r(e) = mean cosine of e to its k nearest TARGET neighbors
    and   r(x) = mean cosine of x to its k nearest SOURCE neighbors

    Args:
        source_embs: (N_src, d) tensor, L2-normalized
        target_embs: (N_tgt, d) tensor, L2-normalized
        k: number of neighbors for local scaling

    Returns:
        (N_src, N_tgt) tensor of CSLS scores
    """
    # Cosine similarity matrix (inputs must be normalized)
    cos_matrix = source_embs @ target_embs.T  # (N_src, N_tgt)

    # r(e): for each source, mean cos to its k nearest targets
    k_src = min(k, target_embs.shape[0])
    r_source = cos_matrix.topk(k_src, dim=1).values.mean(dim=1)  # (N_src,)

    # r(x): for each target, mean cos to its k nearest sources
    k_tgt = min(k, source_embs.shape[0])
    r_target = cos_matrix.T.topk(k_tgt, dim=1).values.mean(dim=1)  # (N_tgt,)

    # CSLS = 2*cos - r_source - r_target
    csls = 2 * cos_matrix - r_source.unsqueeze(1) - r_target.unsqueeze(0)
    return csls


# ---------------------------------------------------------------------------
# Frequency matching for random controls
# ---------------------------------------------------------------------------

def build_frequency_ranks(tokenizer, translations):
    """Build approximate frequency ranks for translation words.

    Uses token ID as a rough proxy for frequency (lower ID = more frequent
    in most tokenizers). Returns a dict: {lang: [(word, rank), ...]} sorted by rank.
    """
    ranks = {}
    for lang in LANGS:
        words = set()
        for en_word, trans in translations:
            if lang == "en":
                words.add(en_word)
            elif lang in trans:
                words.add(trans[lang])

        word_ranks = []
        for word in words:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                word_ranks.append((word, ids[0]))
        word_ranks.sort(key=lambda x: x[1])
        ranks[lang] = word_ranks
    return ranks


def get_frequency_matched_random(word, lang, freq_ranks, n=1, exclude=None):
    """Get n frequency-matched random words from the same language.

    Finds words whose token ID rank is close to the target word's rank.
    Returns a list of words.
    """
    if lang not in freq_ranks or not freq_ranks[lang]:
        return []

    word_list = freq_ranks[lang]
    exclude_set = set(exclude) if exclude else set()

    target_rank = None
    for w, r in word_list:
        if w == word:
            target_rank = r
            break

    if target_rank is None:
        return []

    candidates = [(w, abs(r - target_rank)) for w, r in word_list
                  if w != word and w not in exclude_set]
    candidates.sort(key=lambda x: x[1])

    return [c[0] for c in candidates[:n]]
