"""
Layer 1: Pure tensor surgery unit tests.

No models. No HuggingFace. Just nn.Linear and torch ops.
Tests that the math of removing attention heads from weight matrices is correct.

Run: pytest tests/defrag_validation/test_layer1_tensor_surgery.py -v
Speed: All tests should complete in under 1 second total.
"""

import pytest
import torch
import torch.nn as nn


# ── Helpers ──────────────────────────────────────────────────────────────────


def slice_q_proj(q_proj: nn.Linear, num_heads: int, head_dim: int, surviving_heads: list[int]) -> nn.Linear:
    """Remove rows for pruned heads from a Q projection.

    q_proj.weight shape: (num_heads * head_dim, hidden_dim)
    Each head occupies head_dim consecutive rows.
    """
    rows_to_keep = []
    for h in surviving_heads:
        start = h * head_dim
        rows_to_keep.extend(range(start, start + head_dim))

    new_out = len(rows_to_keep)
    new_proj = nn.Linear(q_proj.in_features, new_out, bias=q_proj.bias is not None)
    with torch.no_grad():
        new_proj.weight.copy_(q_proj.weight[rows_to_keep])
        if q_proj.bias is not None:
            new_proj.bias.copy_(q_proj.bias[rows_to_keep])
    return new_proj


def slice_o_proj(o_proj: nn.Linear, num_heads: int, head_dim: int, surviving_heads: list[int]) -> nn.Linear:
    """Remove columns for pruned heads from an O projection.

    o_proj.weight shape: (hidden_dim, num_heads * head_dim)
    Each head occupies head_dim consecutive columns.
    """
    cols_to_keep = []
    for h in surviving_heads:
        start = h * head_dim
        cols_to_keep.extend(range(start, start + head_dim))

    new_in = len(cols_to_keep)
    new_proj = nn.Linear(new_in, o_proj.out_features, bias=o_proj.bias is not None)
    with torch.no_grad():
        new_proj.weight.copy_(o_proj.weight[:, cols_to_keep])
        if o_proj.bias is not None:
            new_proj.bias.copy_(o_proj.bias)
    return new_proj


# ── Test 1: Basic Q projection slicing ──────────────────────────────────────


class TestQProjectionSlicing:
    """Q projection: removing heads = removing groups of consecutive rows."""

    def test_remove_one_head_shape(self):
        """4 heads, dim 4. Remove head 1. Result: 3 heads, dim 4 = 12 rows."""
        q = nn.Linear(8, 16)  # hidden=8, num_heads=4, head_dim=4 → 16 rows
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 2, 3])
        assert new_q.weight.shape == (12, 8)
        assert new_q.in_features == 8
        assert new_q.out_features == 12

    def test_preserves_surviving_rows(self):
        """The remaining rows should be byte-identical to the original."""
        q = nn.Linear(8, 16)
        # Set distinctive values per head
        with torch.no_grad():
            for h in range(4):
                q.weight[h * 4:(h + 1) * 4] = float(h + 1)  # head 0 = 1.0, head 1 = 2.0, etc.

        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 2, 3])
        # New head 0 = old head 0 (all 1.0)
        assert torch.all(new_q.weight[0:4] == 1.0)
        # New head 1 = old head 2 (all 3.0)
        assert torch.all(new_q.weight[4:8] == 3.0)
        # New head 2 = old head 3 (all 4.0)
        assert torch.all(new_q.weight[8:12] == 4.0)

    def test_remove_all_but_one(self):
        """Aggressive prune: keep only head 2."""
        q = nn.Linear(8, 16)
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[2])
        assert new_q.weight.shape == (4, 8)

    def test_remove_first_head(self):
        """Edge case: remove head 0."""
        q = nn.Linear(8, 16)
        with torch.no_grad():
            q.weight[0:4] = -99.0  # mark head 0
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[1, 2, 3])
        assert new_q.weight.shape == (12, 8)
        # Head 0's distinctive value should be gone
        assert -99.0 not in new_q.weight

    def test_remove_last_head(self):
        """Edge case: remove final head."""
        q = nn.Linear(8, 16)
        with torch.no_grad():
            q.weight[12:16] = -99.0  # mark head 3
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 1, 2])
        assert new_q.weight.shape == (12, 8)
        assert -99.0 not in new_q.weight

    def test_bias_preserved(self):
        """Bias rows should be sliced too."""
        q = nn.Linear(8, 16, bias=True)
        with torch.no_grad():
            for h in range(4):
                q.bias[h * 4:(h + 1) * 4] = float(h + 1)
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[1, 3])
        assert new_q.bias.shape == (8,)
        assert torch.all(new_q.bias[0:4] == 2.0)
        assert torch.all(new_q.bias[4:8] == 4.0)


# ── Test 2: Output projection slicing ───────────────────────────────────────


class TestOProjectionSlicing:
    """O projection: removing heads = removing groups of consecutive columns."""

    def test_remove_one_head_shape(self):
        """4 heads, dim 4 → 16 input cols. Remove head 1 → 12 cols."""
        o = nn.Linear(16, 8)  # input=16 (4 heads × 4 dim), output=hidden=8
        new_o = slice_o_proj(o, num_heads=4, head_dim=4, surviving_heads=[0, 2, 3])
        assert new_o.weight.shape == (8, 12)
        assert new_o.in_features == 12
        assert new_o.out_features == 8

    def test_preserves_surviving_columns(self):
        """The remaining columns should be byte-identical to the original."""
        o = nn.Linear(16, 8)
        with torch.no_grad():
            for h in range(4):
                o.weight[:, h * 4:(h + 1) * 4] = float(h + 1)

        new_o = slice_o_proj(o, num_heads=4, head_dim=4, surviving_heads=[0, 2, 3])
        assert torch.all(new_o.weight[:, 0:4] == 1.0)
        assert torch.all(new_o.weight[:, 4:8] == 3.0)
        assert torch.all(new_o.weight[:, 8:12] == 4.0)

    def test_bias_unchanged(self):
        """O projection bias is per-output, not per-head — should NOT change."""
        o = nn.Linear(16, 8, bias=True)
        with torch.no_grad():
            o.bias[:] = torch.arange(8).float()
        new_o = slice_o_proj(o, num_heads=4, head_dim=4, surviving_heads=[0, 2])
        assert torch.equal(new_o.bias, torch.arange(8).float())


# ── Test 3: Q + O round-trip preserves matrix multiply structure ────────────


class TestQOConsistency:
    """The key invariant: Q@input → attention → O must remain dimensionally consistent."""

    def test_qo_dimensions_match_after_prune(self):
        """After removing heads, q_proj output dim must equal o_proj input dim."""
        hidden = 8
        num_heads = 4
        head_dim = 4

        q = nn.Linear(hidden, num_heads * head_dim)
        o = nn.Linear(num_heads * head_dim, hidden)

        surviving = [0, 2]  # remove heads 1 and 3
        new_q = slice_q_proj(q, num_heads, head_dim, surviving)
        new_o = slice_o_proj(o, num_heads, head_dim, surviving)

        # The key invariant
        assert new_q.out_features == new_o.in_features
        assert new_q.out_features == len(surviving) * head_dim

    def test_qo_forward_pass_works(self):
        """A simulated forward pass through pruned Q and O should not error."""
        hidden = 8
        num_heads = 4
        head_dim = 4

        q = nn.Linear(hidden, num_heads * head_dim)
        o = nn.Linear(num_heads * head_dim, hidden)

        surviving = [0, 2, 3]
        new_q = slice_q_proj(q, num_heads, head_dim, surviving)
        new_o = slice_o_proj(o, num_heads, head_dim, surviving)

        # Simulated input: batch=2, seq=5, hidden=8
        x = torch.randn(2, 5, hidden)
        q_out = new_q(x)
        # In real attention, q_out would be reshaped to (batch, seq, num_heads, head_dim),
        # softmax(QK^T)V applied, then reshaped back. We skip that and just feed to O.
        out = new_o(q_out)

        assert out.shape == (2, 5, hidden)
        assert not torch.isnan(out).any()


# ── Test 4: GQA invariant (grouped query attention) ─────────────────────────


class TestGQAInvariants:
    """Grouped Query Attention: num_heads must be divisible by num_kv_heads."""

    def test_gqa_constraint_holds_after_prune(self):
        """If we have 8 Q heads and 2 KV heads (group size 4), pruning must remove
        complete groups of 4 Q heads at a time."""
        num_q_heads = 8
        num_kv_heads = 2
        group_size = num_q_heads // num_kv_heads  # 4

        # Valid prune: remove KV head 0 → remove Q heads 0-3
        # New: 4 Q heads, 1 KV head, group_size=4 ✓
        surviving_kv = [1]
        surviving_q = []
        for kv in surviving_kv:
            for q_in_group in range(group_size):
                surviving_q.append(kv * group_size + q_in_group)

        assert len(surviving_q) % len(surviving_kv) == 0
        assert len(surviving_q) // len(surviving_kv) == group_size

    def test_partial_group_prune_violates_gqa(self):
        """Removing just one Q head from a group of 4 breaks the GQA invariant.
        This test documents WHY group-aware pruning is required."""
        num_q_heads = 8
        num_kv_heads = 2
        group_size = num_q_heads // num_kv_heads  # 4

        # Naive: remove just Q head 0
        surviving_q = [1, 2, 3, 4, 5, 6, 7]  # 7 Q heads
        surviving_kv = [0, 1]  # 2 KV heads

        # 7 % 2 != 0 — broken GQA
        assert len(surviving_q) % len(surviving_kv) != 0


# ── Test 5: Save/load roundtrip on bare tensors ─────────────────────────────


class TestSaveLoadRoundtrip:
    """Sliced layers must serialize and deserialize without losing data."""

    def test_state_dict_roundtrip(self, tmp_path):
        """Slice a layer, save state_dict, load into a new layer, verify identical."""
        q = nn.Linear(8, 16)
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 2])

        path = tmp_path / "q_proj.pt"
        torch.save(new_q.state_dict(), path)

        # Create a fresh layer with the new shape and load
        loaded = nn.Linear(8, 8)  # 2 heads × 4 dim = 8
        loaded.load_state_dict(torch.load(path))

        assert torch.equal(loaded.weight, new_q.weight)
        assert torch.equal(loaded.bias, new_q.bias)

    def test_load_into_wrong_shape_fails(self, tmp_path):
        """Loading a sliced state_dict into a full-size layer should fail (catches config drift)."""
        q = nn.Linear(8, 16)
        new_q = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 2])

        path = tmp_path / "q_proj.pt"
        torch.save(new_q.state_dict(), path)

        wrong_size = nn.Linear(8, 16)  # full size, not pruned
        with pytest.raises(RuntimeError, match="size mismatch"):
            wrong_size.load_state_dict(torch.load(path))


# ── Test 6: Math arithmetic for head counting ───────────────────────────────


class TestHeadArithmetic:
    """The metadata math: num_heads, head_dim, hidden_size must stay consistent."""

    @pytest.mark.parametrize("num_heads,head_dim,prune_count", [
        (4, 4, 1),
        (8, 8, 3),
        (16, 64, 4),
        (32, 128, 8),
        (40, 128, 8),  # Qwen3 9B style
    ])
    def test_dimensions_after_prune(self, num_heads, head_dim, prune_count):
        """For any model size, removing N heads should give exact expected dims."""
        hidden = num_heads * head_dim
        q = nn.Linear(hidden, hidden)

        surviving = list(range(prune_count, num_heads))
        new_q = slice_q_proj(q, num_heads, head_dim, surviving)

        expected_out = (num_heads - prune_count) * head_dim
        assert new_q.weight.shape == (expected_out, hidden)

    def test_hidden_dim_unchanged(self):
        """Pruning heads should NOT change input dimension (hidden_size)."""
        q = nn.Linear(512, 16 * 64)  # hidden=512, 16 heads of dim 64
        new_q = slice_q_proj(q, num_heads=16, head_dim=64, surviving_heads=list(range(8)))
        assert new_q.in_features == 512  # unchanged
        assert new_q.out_features == 8 * 64  # halved


# ── Test 7: Determinism — same prune produces same result ───────────────────


class TestDeterminism:
    """Repeated pruning of the same layer with the same surviving list must be byte-identical."""

    def test_repeated_prune_identical(self):
        torch.manual_seed(42)
        q = nn.Linear(8, 16)

        new_q1 = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 2, 3])
        new_q2 = slice_q_proj(q, num_heads=4, head_dim=4, surviving_heads=[0, 2, 3])

        assert torch.equal(new_q1.weight, new_q2.weight)
        assert torch.equal(new_q1.bias, new_q2.bias)

    def test_order_of_surviving_matters(self):
        """[0, 2] vs [2, 0] should produce DIFFERENT results — order is preserved."""
        q = nn.Linear(8, 16)
        with torch.no_grad():
            for h in range(4):
                q.weight[h * 4:(h + 1) * 4] = float(h)

        new_a = slice_q_proj(q, 4, 4, [0, 2])
        new_b = slice_q_proj(q, 4, 4, [2, 0])

        # In new_a, "head 0" is old head 0 (zeros)
        assert torch.all(new_a.weight[0:4] == 0.0)
        # In new_b, "head 0" is old head 2 (twos)
        assert torch.all(new_b.weight[0:4] == 2.0)
