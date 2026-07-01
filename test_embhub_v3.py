import math

import torch
import pytest

from hub_layer_v3 import EmbHubV3


# ---------------------------------------------------------------------------
# Safe init tests
# ---------------------------------------------------------------------------

class TestSafeInit:

    def test_pass_through_single_head(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_pass_through_multi_head(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_linear_v_is_zero(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        assert torch.equal(hub.linear_v.weight, torch.zeros_like(hub.linear_v.weight))

    def test_gate_bias_default(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        expected = torch.full_like(hub.linear_g.bias, -5.0)
        assert torch.equal(hub.linear_g.bias, expected)

    def test_gate_bias_custom(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, gate_bias_init=-3.0)
        expected = torch.full_like(hub.linear_g.bias, -3.0)
        assert torch.equal(hub.linear_g.bias, expected)

    def test_gate_near_zero_at_init(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        gate = torch.sigmoid(hub.linear_g(x))
        assert gate.mean().item() < 0.02


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:

    def test_output_shape_2d_batch(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)

    def test_output_shape_single_seq(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(1, 5, 64)
        assert hub(x).shape == (1, 5, 64)

    def test_output_shape_large_batch(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(8, 100, 64)
        assert hub(x).shape == (8, 100, 64)

    def test_output_shape_multi_head(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, num_heads=4)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)

    def test_num_heads_must_divide_dim(self):
        with pytest.raises(AssertionError):
            EmbHubV3(embedding_dim=64, num_embeddings=32, num_heads=3)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestArchitecture:

    def test_decoupled_keys_values(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        assert hub.anchor_keys.shape == (32, 64)
        assert hub.anchor_values.shape == (32, 64)
        assert not torch.equal(hub.anchor_keys, hub.anchor_values)

    def test_no_alpha_parameter(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        assert not hasattr(hub, "alpha")

    def test_linear_v_no_bias(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        assert hub.linear_v.bias is None

    def test_linear_g_has_bias(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        assert hub.linear_g.bias is not None

    def test_deterministic_output(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        out1 = hub(x)
        out2 = hub(x)
        assert torch.equal(out1, out2)

    def test_reference_weight_init(self):
        ref = torch.randn(1000, 64) * 0.02
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, reference_weight=ref)
        ref_std = ref.std().item()
        keys_std = hub.anchor_keys.std().item()
        values_std = hub.anchor_values.std().item()
        assert abs(keys_std - ref_std) < 0.01
        assert abs(values_std - ref_std) < 0.01


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    def test_step0_only_linear_v_gets_gradient(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        hub(x).sum().backward()
        assert hub.linear_v.weight.grad is not None
        assert hub.linear_v.weight.grad.abs().sum() > 0
        assert hub.anchor_keys.grad.abs().sum() == 0
        assert hub.anchor_values.grad.abs().sum() == 0
        assert hub.linear_g.weight.grad.abs().sum() == 0

    def test_all_params_get_gradient_after_bootstrap(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(2, 10, 64)
            hub(x).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} has no gradient after bootstrap"

    def test_gradient_flow_multi_head(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, num_heads=4)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} has no gradient after bootstrap (multi-head)"


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:

    def test_contribution_grows_over_steps(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        x = torch.randn(2, 10, 64)
        norms = []
        for _ in range(10):
            optimizer.zero_grad()
            out = hub(x)
            diff = (out - x).norm().item()
            norms.append(diff)
            out.sum().backward()
            optimizer.step()
        assert norms[0] < 1e-4, "Step 0 contribution should be ~0"
        assert norms[-1] > norms[0], "Contribution should grow over steps"

    def test_linear_v_moves_from_zero(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        assert hub.linear_v.weight.abs().max().item() == 0
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        assert hub.linear_v.weight.abs().max().item() > 0

    def test_gate_opens_over_steps(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        x = torch.randn(2, 10, 64)
        gate_init = torch.sigmoid(hub.linear_g(x)).mean().item()
        for _ in range(20):
            optimizer.zero_grad()
            hub(x).sum().backward()
            optimizer.step()
        gate_after = torch.sigmoid(hub.linear_g(x)).mean().item()
        assert gate_init < 0.02
        assert gate_after > gate_init


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_all_keys_present(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        expected_keys = [
            "logit_std", "entropy_mean", "entropy_std", "uniform_entropy",
            "effective_anchors", "max_weight_mean", "logit_scale",
            "top10_anchor_mass", "dead_anchor_fraction", "norm_ratio",
            "gate_mean", "anchor_pairwise_cosine",
        ]
        for k in expected_keys:
            assert k in diag, f"Missing key: {k}"

    def test_gate_mean_at_init(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["gate_mean"] < 0.02

    def test_norm_ratio_zero_at_init(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["norm_ratio"] < 1e-6

    def test_logit_scale_at_init(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert abs(diag["logit_scale"] - 14.0) < 0.01

    def test_diagnostics_multi_head(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, num_heads=4)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert "gate_mean" in diag
        assert diag["gate_mean"] < 0.02
        assert diag["norm_ratio"] < 1e-6

    def test_diagnostics_with_bf16_model(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert "gate_mean" in diag
        assert isinstance(diag["gate_mean"], float)

    def test_diagnostics_multi_head_bf16(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32, num_heads=4).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert "gate_mean" in diag
        assert isinstance(diag["gate_mean"], float)


# ---------------------------------------------------------------------------
# State dict tests
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_state_dict_keys(self):
        hub = EmbHubV3(embedding_dim=64, num_embeddings=32)
        keys = set(hub.state_dict().keys())
        expected = {
            "anchor_keys", "anchor_values", "log_logit_scale",
            "linear_v.weight", "linear_g.weight", "linear_g.bias",
        }
        assert keys == expected

    def test_save_and_load(self):
        hub1 = EmbHubV3(embedding_dim=64, num_embeddings=32)
        optimizer = torch.optim.Adam(hub1.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            hub1(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()

        state = hub1.state_dict()

        hub2 = EmbHubV3(embedding_dim=64, num_embeddings=32)
        hub2.load_state_dict(state)

        x = torch.randn(2, 10, 64)
        assert torch.equal(hub1(x), hub2(x))

    def test_param_count(self):
        d, N = 64, 32
        hub = EmbHubV3(embedding_dim=d, num_embeddings=N)
        total = sum(p.numel() for p in hub.parameters())
        expected = (N * d) + (N * d) + 1 + (d * d) + (d * d) + d
        assert total == expected
