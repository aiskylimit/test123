"""Cross-variant tests: all hub types x all placements through the unified wrapper."""

import json
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_wrapper_v3 import (
    inject_embhub_v3, remove_embhub_v3, save_embhub_v3,
    load_model_with_embhub_v3, EMBHUB_V3_CONFIG_NAME,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

@pytest.fixture(scope="module")
def input_ids(tokenizer):
    return tokenizer("Hello world test", return_tensors="pt")["input_ids"]


VARIANTS = [
    {"hub_type": "v3", "placement": "embedding", "label": "V3-emb"},
    {"hub_type": "v3", "placement": "mid", "layer_idx": 5, "label": "V5-mid5"},
    {"hub_type": "v3", "num_heads": 4, "placement": "embedding", "label": "V4-emb"},
    {"hub_type": "v3", "num_heads": 4, "placement": "mid", "layer_idx": 5, "label": "V4-mid5"},
    {"hub_type": "v2_concat", "placement": "embedding", "label": "V2-emb"},
    {"hub_type": "v2_concat", "placement": "mid", "layer_idx": 5, "label": "V2-mid5"},
    {"hub_type": "v2_concat", "use_mlp": True, "placement": "embedding", "label": "V2b-emb"},
    {"hub_type": "v2_concat", "use_mlp": True, "placement": "mid", "layer_idx": 5, "label": "V2b-mid5"},
]


class TestSafeInitAllVariants:

    @pytest.mark.parametrize("cfg", VARIANTS, ids=[v["label"] for v in VARIANTS])
    def test_pass_through(self, input_ids, cfg):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        original = model(input_ids=input_ids).logits
        kwargs = {k: v for k, v in cfg.items() if k != "label"}
        inject_embhub_v3(model, num_embeddings=32, **kwargs)
        hub_out = model(input_ids=input_ids).logits
        assert torch.allclose(hub_out, original, atol=1e-4), f"{cfg['label']} failed pass-through"


class TestRemoveAllVariants:

    @pytest.mark.parametrize("cfg", VARIANTS, ids=[v["label"] for v in VARIANTS])
    def test_remove_restores(self, input_ids, cfg):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        original = model(input_ids=input_ids).logits
        kwargs = {k: v for k, v in cfg.items() if k != "label"}
        inject_embhub_v3(model, num_embeddings=32, **kwargs)
        remove_embhub_v3(model)
        restored = model(input_ids=input_ids).logits
        assert torch.allclose(restored, original, atol=1e-5), f"{cfg['label']} failed restore"
        assert not hasattr(model, "embhub")


class TestGradientAllVariants:

    @pytest.mark.parametrize("cfg", VARIANTS, ids=[v["label"] for v in VARIANTS])
    def test_gradient_after_bootstrap(self, input_ids, cfg):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        kwargs = {k: v for k, v in cfg.items() if k != "label"}
        hub = inject_embhub_v3(model, num_embeddings=32, freeze_base=True, **kwargs)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{cfg['label']}: {name} has no gradient after bootstrap"


class TestSaveLoadAllVariants:

    @pytest.mark.parametrize("cfg", VARIANTS, ids=[v["label"] for v in VARIANTS])
    def test_save_load_config(self, input_ids, cfg, tmp_path):
        save_dir = str(tmp_path / cfg["label"])
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        kwargs = {k: v for k, v in cfg.items() if k != "label"}
        hub = inject_embhub_v3(model, num_embeddings=32, **kwargs)
        # Train a bit
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        model.save_pretrained(save_dir)
        save_embhub_v3(model, save_dir)

        with open(f"{save_dir}/{EMBHUB_V3_CONFIG_NAME}") as f:
            saved_cfg = json.load(f)
        assert saved_cfg["hub_type"] == cfg["hub_type"]
        assert saved_cfg["placement"] == cfg.get("placement", "embedding")
        if cfg["hub_type"] == "v3":
            assert saved_cfg["num_heads"] == cfg.get("num_heads", 1)
        elif cfg["hub_type"] == "v2_concat":
            assert saved_cfg["use_mlp"] == cfg.get("use_mlp", False)


class TestDiagnosticsAllVariants:

    @pytest.mark.parametrize("cfg", VARIANTS, ids=[v["label"] for v in VARIANTS])
    def test_diagnostics_at_init(self, cfg):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        kwargs = {k: v for k, v in cfg.items() if k != "label"}
        hub = inject_embhub_v3(model, num_embeddings=32, **kwargs)
        x = torch.randn(1, 5, model.config.hidden_size)
        diag = hub.compute_diagnostics(x)
        assert diag["norm_ratio"] < 1e-4, f"{cfg['label']} norm_ratio too high at init"
        assert abs(diag["logit_scale"] - 14.0) < 0.01, f"{cfg['label']} logit_scale wrong"
