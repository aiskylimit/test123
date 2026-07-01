"""Integration tests for hub_layer_v3 + model_wrapper_v3 with a real HF model."""

import torch
import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer

from hub_layer_v3 import EmbHubV3
from model_wrapper_v3 import (
    inject_embhub_v3,
    remove_embhub_v3,
    save_embhub_v3,
    load_model_with_embhub_v3,
    EMBHUB_V3_WEIGHTS_NAME,
    EMBHUB_V3_CONFIG_NAME,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def dummy_input(tokenizer):
    return tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")


# ---------------------------------------------------------------------------
# V3: Embedding placement
# ---------------------------------------------------------------------------

class TestV3Embedding:

    def test_forward_shape(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        hub_out = model(input_ids=input_ids)
        assert hub_out.logits.shape == original_out.logits.shape

    def test_safe_init_matches_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        hub_out = model(input_ids=input_ids)
        assert torch.allclose(hub_out.logits, original_out.logits, atol=1e-5)

    def test_remove_restores_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        remove_embhub_v3(model)
        restored_out = model(input_ids=input_ids)
        assert torch.allclose(restored_out.logits, original_out.logits, atol=1e-6)
        assert not hasattr(model, "embhub")
        assert not hasattr(model, "_embhub_hook_handle")

    def test_double_inject_raises(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub_v3(model, num_embeddings=64)
        with pytest.raises(ValueError, match="already has"):
            inject_embhub_v3(model, num_embeddings=64)

    def test_freeze_base(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub_v3(model, num_embeddings=64, placement="embedding", freeze_base=True)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) == 6
        assert all("embhub" in n for n in trainable)

    def test_backward_pass(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="embedding", freeze_base=True)
        input_ids = dummy_input["input_ids"]
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        assert hub.linear_v.weight.grad is not None
        assert hub.linear_v.weight.grad.abs().sum() > 0

    def test_training_step_updates_hub(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="embedding", freeze_base=True)
        input_ids = dummy_input["input_ids"]
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        weights_before = hub.linear_v.weight.data.clone()
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        assert not torch.equal(hub.linear_v.weight.data, weights_before)

    def test_state_dict_keys(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        original_keys = set(model.state_dict().keys())
        inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        new_keys = set(model.state_dict().keys())
        added = new_keys - original_keys
        expected_added = {
            "embhub.anchor_keys", "embhub.anchor_values", "embhub.log_logit_scale",
            "embhub.linear_v.weight", "embhub.linear_g.weight", "embhub.linear_g.bias",
        }
        assert added == expected_added

    def test_dtype_matches_model(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        assert hub.anchor_keys.dtype == torch.float16
        assert hub.linear_v.weight.dtype == torch.float16


# ---------------------------------------------------------------------------
# V4: Multi-head at embedding
# ---------------------------------------------------------------------------

class TestV4MultiHead:

    def test_safe_init_matches_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, num_heads=4, placement="embedding")
        hub_out = model(input_ids=input_ids)
        assert torch.allclose(hub_out.logits, original_out.logits, atol=1e-5)

    def test_backward_multi_head(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, num_heads=4, freeze_base=True)
        input_ids = dummy_input["input_ids"]
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        assert hub.linear_v.weight.abs().max().item() > 0


# ---------------------------------------------------------------------------
# V5: Mid-layer placement
# ---------------------------------------------------------------------------

class TestV5MidLayer:

    def test_safe_init_matches_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5)
        hub_out = model(input_ids=input_ids)
        assert torch.allclose(hub_out.logits, original_out.logits, atol=1e-4)

    def test_remove_restores_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5)
        remove_embhub_v3(model)
        restored_out = model(input_ids=input_ids)
        assert torch.allclose(restored_out.logits, original_out.logits, atol=1e-6)

    def test_placement_stored(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5)
        assert model._embhub_placement == "mid"
        assert model._embhub_layer_idx == 5

    def test_invalid_layer_idx(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        with pytest.raises(ValueError, match="out of range"):
            inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=999)

    def test_backward_mid_layer(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5, freeze_base=True)
        input_ids = dummy_input["input_ids"]
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        assert hub.linear_v.weight.grad is not None
        assert hub.linear_v.weight.grad.abs().sum() > 0

    def test_freeze_base_mid_layer(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5, freeze_base=True)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) == 6
        assert all("embhub" in n for n in trainable)

    def test_training_updates_hub_mid_layer(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5, freeze_base=True)
        input_ids = dummy_input["input_ids"]
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        weights_before = hub.linear_v.weight.data.clone()
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        assert not torch.equal(hub.linear_v.weight.data, weights_before)

    def test_v5_multi_head(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]
        original_out = model(input_ids=input_ids)
        inject_embhub_v3(model, num_embeddings=64, num_heads=4, placement="mid", layer_idx=5)
        hub_out = model(input_ids=input_ids)
        assert hub_out.logits.shape == original_out.logits.shape
        assert torch.allclose(hub_out.logits, original_out.logits, atol=1e-4)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_files(self, tmp_path):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        save_embhub_v3(model, str(tmp_path))
        assert (tmp_path / EMBHUB_V3_WEIGHTS_NAME).exists()
        assert (tmp_path / EMBHUB_V3_CONFIG_NAME).exists()

    def test_save_config_content(self, tmp_path):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub_v3(model, num_embeddings=64, num_heads=2, placement="mid", layer_idx=5)
        save_embhub_v3(model, str(tmp_path))
        import json
        with open(tmp_path / EMBHUB_V3_CONFIG_NAME) as f:
            cfg = json.load(f)
        assert cfg["hub_type"] == "v3"
        assert cfg["num_embeddings"] == 64
        assert cfg["num_heads"] == 2
        assert cfg["placement"] == "mid"
        assert cfg["layer_idx"] == 5

    def test_save_weights_float32(self, tmp_path):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        save_embhub_v3(model, str(tmp_path))
        weights = torch.load(tmp_path / EMBHUB_V3_WEIGHTS_NAME, weights_only=True)
        for k, v in weights.items():
            assert v.dtype == torch.float32, f"{k} saved as {v.dtype}"

    def test_save_without_hub_raises(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        with pytest.raises(ValueError, match="does not have"):
            save_embhub_v3(model, "/tmp/should_not_exist")

    def test_load_embedding(self, dummy_input, tmp_path):
        save_dir = str(tmp_path / "ckpt")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        input_ids = dummy_input["input_ids"]
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        trained_weights = hub.anchor_keys.data.clone()
        model.save_pretrained(save_dir)
        save_embhub_v3(model, save_dir)

        loaded, loaded_hub = load_model_with_embhub_v3(save_dir, torch_dtype=torch.float32)
        assert torch.equal(loaded_hub.anchor_keys.data, trained_weights)
        assert loaded._embhub_placement == "embedding"

    def test_load_mid_layer(self, dummy_input, tmp_path):
        save_dir = str(tmp_path / "ckpt")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        input_ids = dummy_input["input_ids"]
        for _ in range(3):
            optimizer.zero_grad()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            optimizer.step()
        trained_weights = hub.anchor_keys.data.clone()
        model.save_pretrained(save_dir)
        save_embhub_v3(model, save_dir)

        loaded, loaded_hub = load_model_with_embhub_v3(save_dir, torch_dtype=torch.float32)
        assert torch.equal(loaded_hub.anchor_keys.data, trained_weights)
        assert loaded._embhub_placement == "mid"
        assert loaded._embhub_layer_idx == 5

    def test_load_without_hub_files(self, tmp_path):
        save_dir = str(tmp_path / "base")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.save_pretrained(save_dir)
        loaded, loaded_hub = load_model_with_embhub_v3(
            save_dir, num_embeddings=32, placement="mid", layer_idx=3,
            torch_dtype=torch.float32,
        )
        assert loaded_hub.num_embeddings == 32
        assert loaded._embhub_placement == "mid"
        assert loaded._embhub_layer_idx == 3


# ---------------------------------------------------------------------------
# Diagnostics with real model
# ---------------------------------------------------------------------------

class TestDiagnosticsIntegration:

    def test_diagnostics_embedding(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="embedding")
        embedding = model.get_input_embeddings()
        with torch.no_grad():
            token_emb = embedding(dummy_input["input_ids"])
        diag = hub.compute_diagnostics(token_emb)
        assert diag["gate_mean"] < 0.02
        assert diag["norm_ratio"] < 1e-6
        assert abs(diag["logit_scale"] - 14.0) < 0.01

    def test_diagnostics_mid_layer(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub_v3(model, num_embeddings=64, placement="mid", layer_idx=5)
        hidden = torch.randn(1, 5, model.config.hidden_size)
        diag = hub.compute_diagnostics(hidden)
        assert diag["gate_mean"] < 0.02
        assert diag["norm_ratio"] < 1e-6
