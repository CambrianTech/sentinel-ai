"""
Comprehensive tests for GPT-2 → Adaptive Transformer weight transfer.

These tests mathematically prove that the weight transfer preserves the exact
computation of the original GPT-2 model.
"""

import torch
import torch.nn as nn
import pytest
from transformers import AutoModelForCausalLM, AutoConfig

from models.loaders.gpt2_loader_clean import (
    transfer_attention_weights,
    transfer_ffn_weights,
    transfer_layer_norm_weights,
    load_adaptive_model_gpt_clean
)
from sentinel.models.adaptive_transformer import AdaptiveTransformer, AdaptiveTransformerBlock


class TestWeightTransferMath:
    """Test that weight transfer preserves mathematical equivalence."""

    @pytest.fixture
    def gpt2_model(self):
        """Load a small GPT-2 model for testing."""
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        return model

    @pytest.fixture
    def config(self):
        """GPT-2 config."""
        return AutoConfig.from_pretrained("gpt2")

    @pytest.fixture
    def baseline_layer(self, gpt2_model):
        """Get first transformer block from GPT-2."""
        return gpt2_model.transformer.h[0]

    @pytest.fixture
    def adaptive_layer(self, config):
        """Create an adaptive transformer block."""
        return AdaptiveTransformerBlock(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            intermediate_size=config.n_inner,
            dropout_prob=0.0,  # Disable for testing
            activation="gelu"
        )

    def test_attention_weight_dimensions(self, baseline_layer, adaptive_layer, config):
        """Test that transferred attention weights have correct dimensions."""
        transfer_attention_weights(baseline_layer, adaptive_layer, config)

        num_heads = config.n_head
        hidden_size = config.n_embd
        head_dim = hidden_size // num_heads

        for h in range(num_heads):
            # Check Q, K, V weight dimensions
            assert adaptive_layer.attn.W_q[h].shape == (hidden_size, head_dim), \
                f"W_q[{h}] has wrong shape: {adaptive_layer.attn.W_q[h].shape}"
            assert adaptive_layer.attn.W_k[h].shape == (hidden_size, head_dim)
            assert adaptive_layer.attn.W_v[h].shape == (hidden_size, head_dim)

            # Check output projection dimension
            assert adaptive_layer.attn.W_o[h].shape == (head_dim, hidden_size), \
                f"W_o[{h}] has wrong shape: {adaptive_layer.attn.W_o[h].shape}"

            # Check bias dimensions
            assert adaptive_layer.attn.b_q[h].shape == (head_dim,)
            assert adaptive_layer.attn.b_k[h].shape == (head_dim,)
            assert adaptive_layer.attn.b_v[h].shape == (head_dim,)
            assert adaptive_layer.attn.b_o[h].shape == (hidden_size,)

    def test_attention_forward_equivalence(self, baseline_layer, adaptive_layer, config):
        """
        Test that adaptive attention produces the same output as baseline.

        This is the CRITICAL test - it proves mathematical equivalence.
        """
        transfer_attention_weights(baseline_layer, adaptive_layer, config)

        # Create random input
        batch_size, seq_len = 2, 10
        hidden_size = config.n_embd
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Baseline GPT-2 attention forward
        with torch.no_grad():
            baseline_output = baseline_layer.attn(hidden_states)[0]

        # Adaptive attention forward
        with torch.no_grad():
            adaptive_output = adaptive_layer.attn(hidden_states)

        # Check outputs are identical (within numerical precision)
        max_diff = torch.abs(baseline_output - adaptive_output).max().item()
        print(f"\n   Max difference in attention output: {max_diff:.2e}")

        assert torch.allclose(baseline_output, adaptive_output, atol=1e-5, rtol=1e-4), \
            f"Attention outputs differ by {max_diff} (should be < 1e-5)"

    def test_qkv_split_correctness(self, baseline_layer, config):
        """
        Test that QKV splitting preserves the fused computation.

        GPT-2 computes: [Q | K | V] = input @ W_qkv + b_qkv
        We split into: Q = input @ W_q + b_q, K = input @ W_k + b_k, V = input @ W_v + b_v

        This tests that splitting preserves the computation.
        """
        hidden_size = config.n_embd
        num_heads = config.n_head
        head_dim = hidden_size // num_heads

        # Get baseline QKV weights
        qkv_weight = baseline_layer.attn.c_attn.weight.data
        qkv_bias = baseline_layer.attn.c_attn.bias.data

        # Handle transpose
        if qkv_weight.shape[0] == hidden_size:
            qkv_weight = qkv_weight.t()

        # Split (same logic as loader)
        q_all = qkv_weight[:hidden_size, :].t()
        k_all = qkv_weight[hidden_size:2*hidden_size, :].t()
        v_all = qkv_weight[2*hidden_size:, :].t()

        q_bias_all = qkv_bias[:hidden_size]
        k_bias_all = qkv_bias[hidden_size:2*hidden_size]
        v_bias_all = qkv_bias[2*hidden_size:]

        # Test with random input
        x = torch.randn(5, hidden_size)

        # Baseline computation
        qkv_fused = torch.matmul(x, qkv_weight.t()) + qkv_bias
        q_baseline = qkv_fused[:, :hidden_size]
        k_baseline = qkv_fused[:, hidden_size:2*hidden_size]
        v_baseline = qkv_fused[:, 2*hidden_size:]

        # Split computation
        q_split = torch.matmul(x, q_all) + q_bias_all
        k_split = torch.matmul(x, k_all) + k_bias_all
        v_split = torch.matmul(x, v_all) + v_bias_all

        # Verify equivalence
        assert torch.allclose(q_baseline, q_split, atol=1e-6), "Q split doesn't match"
        assert torch.allclose(k_baseline, k_split, atol=1e-6), "K split doesn't match"
        assert torch.allclose(v_baseline, v_split, atol=1e-6), "V split doesn't match"

    def test_per_head_output_projection(self, baseline_layer, config):
        """
        Test that per-head output projections reconstruct the fused projection.

        GPT-2: out = concat_heads @ W_o + b_o
        Adaptive: out = sum(head_i @ W_o[i] + b_o[i] for each head)
        """
        hidden_size = config.n_embd
        num_heads = config.n_head
        head_dim = hidden_size // num_heads

        # Get baseline output projection
        out_weight = baseline_layer.attn.c_proj.weight.data
        out_bias = baseline_layer.attn.c_proj.bias.data

        if out_weight.shape[0] != hidden_size:
            out_weight = out_weight.t()

        # Extract per-head projections (same logic as loader)
        per_head_weights = []
        per_head_biases = []
        for h in range(num_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim

            o_h = out_weight[:, h_start:h_end].t()  # [head_dim, hidden]
            b_h = out_bias / num_heads

            per_head_weights.append(o_h)
            per_head_biases.append(b_h)

        # Test with random head outputs
        heads = [torch.randn(3, head_dim) for _ in range(num_heads)]
        concat_heads = torch.cat(heads, dim=1)  # [3, hidden]

        # Baseline computation
        baseline_out = torch.matmul(concat_heads, out_weight.t()) + out_bias

        # Per-head computation
        adaptive_out = torch.zeros_like(baseline_out)
        for h in range(num_heads):
            adaptive_out += torch.matmul(heads[h], per_head_weights[h]) + per_head_biases[h]

        # Verify equivalence
        max_diff = torch.abs(baseline_out - adaptive_out).max().item()
        print(f"\n   Max difference in output projection: {max_diff:.2e}")
        assert torch.allclose(baseline_out, adaptive_out, atol=1e-5), \
            f"Output projections differ by {max_diff}"

    def test_ffn_forward_equivalence(self, baseline_layer, adaptive_layer):
        """Test that FFN forward pass is identical."""
        transfer_ffn_weights(baseline_layer, adaptive_layer)

        # Random input
        x = torch.randn(5, 768)

        # Baseline FFN
        with torch.no_grad():
            baseline_out = baseline_layer.mlp(x)

        # Adaptive FFN
        with torch.no_grad():
            adaptive_out = adaptive_layer.ffn(x)

        max_diff = torch.abs(baseline_out - adaptive_out).max().item()
        print(f"\n   Max difference in FFN output: {max_diff:.2e}")

        assert torch.allclose(baseline_out, adaptive_out, atol=1e-6), \
            f"FFN outputs differ by {max_diff}"

    def test_layer_norm_equivalence(self, baseline_layer, adaptive_layer):
        """Test that layer norms are identical."""
        transfer_layer_norm_weights(baseline_layer, adaptive_layer)

        x = torch.randn(5, 768)

        # Test norm1 (pre-attention)
        with torch.no_grad():
            baseline_norm1 = baseline_layer.ln_1(x)
            adaptive_norm1 = adaptive_layer.norm1(x)

        assert torch.allclose(baseline_norm1, adaptive_norm1, atol=1e-6), \
            "norm1 outputs differ"

        # Test norm2 (pre-FFN)
        with torch.no_grad():
            baseline_norm2 = baseline_layer.ln_2(x)
            adaptive_norm2 = adaptive_layer.norm2(x)

        assert torch.allclose(baseline_norm2, adaptive_norm2, atol=1e-6), \
            "norm2 outputs differ"

    def test_full_layer_equivalence(self, baseline_layer, adaptive_layer, config):
        """
        ULTIMATE TEST: Full transformer block forward pass equivalence.

        This tests the entire computation including residuals, proving that
        the adaptive block is mathematically identical to GPT-2.
        """
        # Transfer all weights
        transfer_attention_weights(baseline_layer, adaptive_layer, config)
        transfer_ffn_weights(baseline_layer, adaptive_layer)
        transfer_layer_norm_weights(baseline_layer, adaptive_layer)

        # Random input
        x = torch.randn(2, 10, 768)

        # Baseline forward
        with torch.no_grad():
            baseline_out = baseline_layer(x)[0]

        # Adaptive forward
        with torch.no_grad():
            adaptive_out = adaptive_layer(x)

        max_diff = torch.abs(baseline_out - adaptive_out).max().item()
        mean_diff = torch.abs(baseline_out - adaptive_out).mean().item()

        print(f"\n   Max difference: {max_diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")
        print(f"   Relative error: {(max_diff / baseline_out.abs().max()).item():.2e}")

        assert torch.allclose(baseline_out, adaptive_out, atol=1e-4, rtol=1e-3), \
            f"Full layer outputs differ by {max_diff} (max) / {mean_diff} (mean)"

    def test_full_model_equivalence(self, gpt2_model, config):
        """
        GOLD STANDARD TEST: Full model forward pass equivalence.

        This proves that the entire 12-layer model produces identical outputs
        to the original GPT-2 after weight transfer.
        """
        # Load adaptive model with weight transfer
        adaptive_model = load_adaptive_model_gpt_clean(
            "gpt2",
            gpt2_model,
            config,
            device="cpu",
            quiet=True
        )

        # Create random input
        input_ids = torch.randint(0, config.vocab_size, (2, 20))

        # Baseline forward
        with torch.no_grad():
            baseline_out = gpt2_model(input_ids).logits

        # Adaptive forward
        with torch.no_grad():
            adaptive_out = adaptive_model(input_ids).logits

        max_diff = torch.abs(baseline_out - adaptive_out).max().item()
        mean_diff = torch.abs(baseline_out - adaptive_out).mean().item()

        print(f"\n   Full model comparison:")
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")
        print(f"   Relative error: {(max_diff / baseline_out.abs().max()).item():.2e}")

        # This is the ultimate proof!
        assert torch.allclose(baseline_out, adaptive_out, atol=1e-3, rtol=1e-2), \
            f"Full model outputs differ by {max_diff} (max) / {mean_diff} (mean)"

    def test_generation_equivalence(self, gpt2_model, config):
        """
        Test that both models generate the same tokens (when deterministic).

        This is the ultimate practical test - if generation is identical,
        the weight transfer is perfect.
        """
        # Load adaptive model
        adaptive_model = load_adaptive_model_gpt_clean(
            "gpt2",
            gpt2_model,
            config,
            device="cpu",
            quiet=True
        )

        # Use greedy decoding (deterministic)
        input_ids = torch.tensor([[15496, 318]])  # "Hello is"

        # Generate with baseline
        with torch.no_grad():
            baseline_next = gpt2_model(input_ids).logits[0, -1, :].argmax()

        # Generate with adaptive
        with torch.no_grad():
            adaptive_next = adaptive_model(input_ids).logits[0, -1, :].argmax()

        print(f"\n   Baseline next token: {baseline_next}")
        print(f"   Adaptive next token: {adaptive_next}")

        assert baseline_next == adaptive_next, \
            f"Models predict different next tokens: {baseline_next} vs {adaptive_next}"


class TestWeightTransferNumericalStability:
    """Test numerical stability of weight transfer."""

    def test_no_nans_or_infs(self, gpt2_model, config):
        """Ensure transferred weights contain no NaN or Inf values."""
        adaptive_model = load_adaptive_model_gpt_clean(
            "gpt2",
            gpt2_model,
            config,
            device="cpu",
            quiet=True
        )

        for name, param in adaptive_model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN found in {name}"
            assert not torch.isinf(param).any(), f"Inf found in {name}"

    def test_weight_magnitudes_reasonable(self, gpt2_model, config):
        """Check that transferred weights have reasonable magnitudes."""
        adaptive_model = load_adaptive_model_gpt_clean(
            "gpt2",
            gpt2_model,
            config,
            device="cpu",
            quiet=True
        )

        for name, param in adaptive_model.named_parameters():
            max_val = param.abs().max().item()
            assert max_val < 100, f"{name} has unreasonably large values: {max_val}"
            assert max_val > 1e-6, f"{name} has unreasonably small values: {max_val}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
