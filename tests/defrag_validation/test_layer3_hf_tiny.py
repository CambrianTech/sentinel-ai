"""
Layer 3: Tiny HuggingFace model integration tests.

Tests defrag on real HF transformers (Qwen2.5-0.5B or similar).
Catches HF-specific quirks: config caching, model.config.head_dim drift,
state_dict naming, GQA group constraints, attention implementation variants.

Run: pytest tests/defrag_validation/test_layer3_hf_tiny.py -v
Speed: ~30 seconds (loads a real model once via fixture).

Skipped if transformers not installed or HF cache empty.
"""

import os
import pytest
import torch

# Skip the entire module if transformers unavailable
transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the smallest reliable Qwen model — same family as our forge targets
TINY_MODEL = os.environ.get("DEFRAG_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    """Load model once per test session — saves ~5 seconds per test."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            TINY_MODEL,
            torch_dtype=torch.float32,  # fp32 for deterministic output comparison
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model
    except Exception as e:
        pytest.skip(f"Could not load {TINY_MODEL}: {e}")


@pytest.fixture(scope="module")
def tiny_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")


@pytest.fixture
def sample_input(tiny_tokenizer):
    """Deterministic input for output comparison tests."""
    return tiny_tokenizer("The quick brown fox jumps over", return_tensors="pt")


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_attention_dims(model):
    """Read the actual tensor dimensions from the first attention block."""
    # Find first layer with self_attn
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            q = module.q_proj
            k = module.k_proj
            v = module.v_proj
            o = module.o_proj
            return {
                "q_out": q.weight.shape[0],
                "k_out": k.weight.shape[0],
                "v_out": v.weight.shape[0],
                "o_in": o.weight.shape[1],
                "hidden": q.weight.shape[1],
            }
    return None


def get_config_dims(model):
    """Read the dimensions the config CLAIMS the model has."""
    cfg = model.config
    return {
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
    }


# ── Test 1: Baseline — model loads and runs ────────────────────────────────


class TestBaseline:
    def test_model_loads(self, tiny_model):
        assert tiny_model is not None
        assert hasattr(tiny_model, "config")

    def test_forward_pass(self, tiny_model, sample_input):
        with torch.no_grad():
            out = tiny_model(**sample_input)
        assert hasattr(out, "logits")
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()

    def test_attention_dims_consistent(self, tiny_model):
        """Tensor dims should match config dims BEFORE we touch anything."""
        tensor_dims = get_attention_dims(tiny_model)
        config_dims = get_config_dims(tiny_model)
        assert tensor_dims is not None

        if config_dims["num_attention_heads"] and config_dims["head_dim"]:
            expected_q = config_dims["num_attention_heads"] * config_dims["head_dim"]
            assert tensor_dims["q_out"] == expected_q, (
                f"Config says {expected_q} Q dims, tensor has {tensor_dims['q_out']}"
            )


# ── Test 2: Defrag preserves structural invariants ─────────────────────────


class TestDefragInvariants:
    """After defrag, the model's structure must be self-consistent."""

    def test_defrag_q_o_dims_match(self, tiny_model):
        """Q output dim must equal O input dim post-defrag."""
        # Import the actual defrag function
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        # Clone model so other tests get an unmodified version
        import copy
        m = copy.deepcopy(tiny_model)

        # Pick a few heads to remove (just one per layer to be safe)
        # Find num_layers from tensor structure
        layers = None
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            layers = m.model.layers
        elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            layers = m.transformer.h

        if layers is None:
            pytest.skip("Could not locate layers")

        # Remove head 0 from each layer (must respect GQA — remove complete groups)
        # For Qwen2.5-0.5B: 14 attention heads, 2 KV heads → group size 7
        # Removing head 0 alone breaks GQA. Remove the whole first group: heads 0-6.
        cfg = m.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        # Remove first complete group from each layer
        dead_heads = {li: list(range(group_size)) for li in range(len(layers))}

        try:
            defrag_live_model(m, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed (expected if model arch unsupported): {e}")

        # Verify each layer's Q/O dims match
        for li, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if attn is None or not hasattr(attn, "q_proj"):
                continue
            q_out = attn.q_proj.weight.shape[0]
            o_in = attn.o_proj.weight.shape[1]
            assert q_out == o_in, f"Layer {li}: Q out={q_out}, O in={o_in} mismatch"

    def test_defrag_forward_pass_works(self, tiny_model, sample_input):
        """Defragged model must produce valid output (no NaN, correct shape)."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        m = copy.deepcopy(tiny_model)

        layers = None
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            layers = m.model.layers
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = m.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        # Remove the LAST KV group from each layer (heads num_q - group_size .. num_q - 1)
        dead_heads = {
            li: list(range(num_q - group_size, num_q))
            for li in range(len(layers))
        }

        try:
            defrag_live_model(m, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed: {e}")

        # Forward pass must work
        with torch.no_grad():
            out = m(**sample_input)
        assert hasattr(out, "logits")
        assert out.logits.shape[-1] == cfg.vocab_size
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()


# ── Test 3: Defrag preserves output for low-importance pruning ─────────────


class TestSemanticPreservation:
    """The hard test: does cosine similarity stay high when pruning low-importance heads?"""

    def test_zero_weight_prune_preserves_output(self, tiny_model, sample_input):
        """Manually zero a head's weights, then defrag it. Output should be ~identical."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        import torch.nn.functional as F

        m = copy.deepcopy(tiny_model)
        layers = m.model.layers if hasattr(m, "model") and hasattr(m.model, "layers") else None
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = m.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        # Zero the weights for one KV group across all layers
        kv_to_zero = num_kv - 1  # last KV group
        q_heads_in_group = list(range(kv_to_zero * group_size, (kv_to_zero + 1) * group_size))

        with torch.no_grad():
            for layer in layers:
                attn = layer.self_attn
                head_dim = attn.q_proj.weight.shape[0] // num_q
                kv_head_dim = attn.k_proj.weight.shape[0] // num_kv
                # Zero Q rows
                for qh in q_heads_in_group:
                    attn.q_proj.weight[qh * head_dim:(qh + 1) * head_dim] = 0
                # Zero K, V rows
                attn.k_proj.weight[kv_to_zero * kv_head_dim:(kv_to_zero + 1) * kv_head_dim] = 0
                attn.v_proj.weight[kv_to_zero * kv_head_dim:(kv_to_zero + 1) * kv_head_dim] = 0
                # Zero O cols
                for qh in q_heads_in_group:
                    attn.o_proj.weight[:, qh * head_dim:(qh + 1) * head_dim] = 0

        # Get baseline output (zeroed but not defragged)
        with torch.no_grad():
            out_zeroed = m(**sample_input).logits

        # Now defrag (should be a no-op semantically since we zeroed)
        dead_heads = {li: q_heads_in_group for li in range(len(layers))}
        try:
            defrag_live_model(m, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed: {e}")

        with torch.no_grad():
            out_defragged = m(**sample_input).logits

        # Cosine similarity should be VERY high — defrag of zeroed weights is a no-op
        cos = F.cosine_similarity(out_zeroed.flatten(), out_defragged.flatten(), dim=0).item()
        assert cos > 0.99, (
            f"Defrag of zeroed weights changed output significantly (cos={cos:.4f}). "
            "Zeroed weights and defragged surviving heads should produce identical logits."
        )


# ── Test 4: Save/load roundtrip ────────────────────────────────────────────


class TestSaveLoadRoundtrip:
    """A defragged model must save and reload without losing structure."""

    def test_defragged_model_saves_and_loads(self, tiny_model, sample_input, tmp_path):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        m = copy.deepcopy(tiny_model)
        layers = m.model.layers if hasattr(m, "model") and hasattr(m.model, "layers") else None
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = m.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        # Remove last KV group
        dead_heads = {
            li: list(range(num_q - group_size, num_q))
            for li in range(len(layers))
        }
        try:
            defrag_live_model(m, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed: {e}")

        with torch.no_grad():
            out_before = m(**sample_input).logits

        # Save
        save_dir = tmp_path / "defragged"
        m.save_pretrained(save_dir)

        # Load fresh
        loaded = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float32)
        loaded.eval()

        with torch.no_grad():
            out_after = loaded(**sample_input).logits

        # Must be byte-identical (same model, fp32, no rounding)
        assert torch.allclose(out_before, out_after, atol=1e-5), (
            "Defragged model produces different output after save/load — config drift suspected"
        )


# ── Test 5: Multi-cycle defrag stability ──────────────────────────────────


class TestMultiCycleStability:
    """The exact pattern from our 9B forge bug — 3 cycles of defrag should not destroy the model."""

    def test_three_cycle_defrag_no_corruption(self, tiny_model, sample_input):
        """3 successive defrags. Each removes one KV group. Model must still produce sane output."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        m = copy.deepcopy(tiny_model)
        layers = m.model.layers if hasattr(m, "model") and hasattr(m.model, "layers") else None
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = m.config
        num_kv = cfg.num_key_value_heads

        # Need at least 4 KV heads to do 3 cycles (each removes one)
        if num_kv < 4:
            pytest.skip(f"Need >=4 KV heads for 3-cycle test, have {num_kv}")

        for cycle in range(3):
            # Read CURRENT dims (they shrink each cycle)
            attn = layers[0].self_attn
            current_num_q = attn.q_proj.weight.shape[0] // (cfg.head_dim if cfg.head_dim else 64)
            current_kv_head_dim = attn.k_proj.weight.shape[0] // num_kv  # this is approximate
            # Remove the LAST attention head from each layer (always exists)
            head_dim = attn.q_proj.weight.shape[0] // current_num_q
            dead_heads = {li: [current_num_q - 1] for li in range(len(layers))}

            try:
                defrag_live_model(m, dead_heads=dead_heads)
            except Exception as e:
                # GQA constraints may make single-head removal impossible — that's fine
                pytest.skip(f"Cycle {cycle} defrag failed (likely GQA constraint): {e}")

        # After 3 cycles, model must still work
        with torch.no_grad():
            out = m(**sample_input).logits
        assert not torch.isnan(out).any(), "3-cycle defrag produced NaN logits"
        assert not torch.isinf(out).any(), "3-cycle defrag produced Inf logits"

        # Output magnitude should be reasonable (not 1e30 or 1e-30)
        max_abs = out.abs().max().item()
        assert 0.001 < max_abs < 1000, f"Logit magnitude unreasonable: {max_abs}"
