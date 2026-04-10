"""TDD spec for DeepSeekV2Adapter — deepseek-routed-shared layout.

DeepSeek-V2 / DeepSeek-V2-Lite has TWO expert pathways per MoE layer:

    Routed experts (pruning target):
        model.layers.{L}.mlp.gate.weight                       ← router
        model.layers.{L}.mlp.experts.{K}.{gate,up,down}_proj.weight

    Shared experts (PRESERVE BIT-EXACT — fires on every token):
        model.layers.{L}.mlp.shared_experts.gate_proj.weight
        model.layers.{L}.mlp.shared_experts.up_proj.weight
        model.layers.{L}.mlp.shared_experts.down_proj.weight

Plus a dense first layer (layer 0 has no MoE, just .mlp.{gate,up,down}_proj).

Pruning rules:
  1. Routed experts: top-K per layer, slice gate, drop dropped, renumber survivors
  2. Shared experts: passthrough bit-exact (NOT optional — they carry the
     always-fires capability the model relies on)
  3. Dense layer 0: passthrough bit-exact
  4. config.json: update n_routed_experts (NOT num_local_experts; DeepSeek
     uses a different field name)

The routed-expert tensor name pattern is IDENTICAL to QWEN3_MOE_LAYOUT
(same `mlp.experts.{e}.{proj}_proj.weight` shape). The shared_experts
tensors do NOT match the routed regex (no digit after `shared_experts.`),
so they fall through to the passthrough branch in the streaming rewriter
and are copied bit-exact. We verify that as part of this spec.

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Layout spec ─────────────────────────────────────────────────────────────


def test_deepseek_v2_layout_constant_exists():
    """DEEPSEEK_V2_LAYOUT MUST exist as a module-level constant."""
    from cpu_expert_prune_v2 import DEEPSEEK_V2_LAYOUT, LayoutSpec
    assert isinstance(DEEPSEEK_V2_LAYOUT, LayoutSpec)
    assert DEEPSEEK_V2_LAYOUT.family_name == "deepseek_v2"


def test_deepseek_v2_layout_matches_routed_expert_names():
    """DEEPSEEK_V2_LAYOUT must match real DeepSeek-V2 routed expert tensor names."""
    from cpu_expert_prune_v2 import DEEPSEEK_V2_LAYOUT
    routed = [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.27.mlp.experts.63.down_proj.weight",
    ]
    for name in routed:
        m = DEEPSEEK_V2_LAYOUT.expert_re().match(name)
        assert m is not None, f"DEEPSEEK_V2_LAYOUT.expert_re does not match {name!r}"
        assert m.group(3) in ("gate_proj", "up_proj", "down_proj")

    gm = DEEPSEEK_V2_LAYOUT.gate_re().match("model.layers.1.mlp.gate.weight")
    assert gm is not None
    assert gm.group(1) == "1"


def test_deepseek_v2_layout_does_NOT_match_shared_experts():
    """CRITICAL: shared_experts tensors must NOT be matched by the routed
    regex — they must fall through to passthrough so they're preserved
    bit-exact. If the regex grabs them, we'd accidentally prune the
    always-fires pathway and break the model."""
    from cpu_expert_prune_v2 import DEEPSEEK_V2_LAYOUT
    shared = [
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
    ]
    for name in shared:
        assert DEEPSEEK_V2_LAYOUT.expert_re().match(name) is None, (
            f"DEEPSEEK_V2_LAYOUT.expert_re must NOT match shared expert {name!r} "
            f"— shared experts must passthrough bit-exact"
        )


def test_deepseek_v2_layout_does_NOT_match_dense_layer():
    """Dense first-layer mlp tensors (layer 0 in DeepSeek-V2) must passthrough."""
    from cpu_expert_prune_v2 import DEEPSEEK_V2_LAYOUT
    dense = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    for name in dense:
        assert DEEPSEEK_V2_LAYOUT.expert_re().match(name) is None
        assert DEEPSEEK_V2_LAYOUT.gate_re().match(name) is None


# ── Adapter ──────────────────────────────────────────────────────────────────


def test_deepseek_v2_adapter_no_longer_stub():
    """DeepSeekV2Adapter.expert_prune body MUST be real (call prune_experts
    with DEEPSEEK_V2_LAYOUT) — not the _stub_expert_prune_raise stub."""
    import inspect
    from adapters.sota_moe import DeepSeekV2Adapter
    src = inspect.getsource(DeepSeekV2Adapter.expert_prune)
    assert "prune_experts" in src
    assert "DEEPSEEK_V2_LAYOUT" in src
    assert "_stub_expert_prune_raise" not in src


def test_deepseek_v2_dispatch_only_path_short_circuits():
    from adapters.sota_moe import DeepSeekV2Adapter
    from dataclasses import dataclass

    @dataclass
    class _MockCtx:
        model: object = None
        tokenizer: object = None
        output_dir: Path = Path("/tmp/test-deepseek")
        alloy: dict = None

    ctx = _MockCtx(alloy={"source": {"architecture": "deepseek_v2"}})
    adapter = DeepSeekV2Adapter()
    result = adapter.expert_prune(
        ctx,
        keepExpertsPerLayer=32,
        originalExpertsPerLayer=64,
        strategy="calibration-aware-activation-count",
        expertTensorLayout="deepseek-routed-shared",
    )
    assert result is ctx


# ── End-to-end: shared experts + dense layer preserved bit-exact ────────────


def test_deepseek_v2_synthetic_end_to_end_preserves_shared_and_dense(tmp_path):
    """End-to-end synthetic fixture: 3 layers (layer 0 dense, layers 1-2 MoE
    with 4 routed experts + shared experts), prune routed to 2/layer, verify:
      - routed experts renumbered to {0,1}
      - shared_experts tensors present in output BIT-EXACT (same data)
      - dense layer 0 tensors present BIT-EXACT
      - config.json n_routed_experts updated to 2
    """
    try:
        import torch
        from safetensors.torch import save_file
        from safetensors import safe_open
    except ImportError:
        pytest.skip("torch / safetensors not installed")

    from cpu_expert_prune_v2 import prune_experts, DEEPSEEK_V2_LAYOUT

    src = tmp_path / "synthetic-deepseek"
    src.mkdir()

    hidden = 8
    inter = 16
    moe_inter = 12
    n_routed = 4

    tensors: dict[str, "torch.Tensor"] = {}

    # Layer 0: dense (no MoE) — must passthrough bit-exact
    tensors["model.layers.0.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
    tensors["model.layers.0.mlp.up_proj.weight"] = torch.randn(inter, hidden)
    tensors["model.layers.0.mlp.down_proj.weight"] = torch.randn(hidden, inter)

    # Layers 1-2: MoE with routed + shared experts
    for L in (1, 2):
        tensors[f"model.layers.{L}.mlp.gate.weight"] = torch.randn(n_routed, hidden)
        for E in range(n_routed):
            tensors[f"model.layers.{L}.mlp.experts.{E}.gate_proj.weight"] = torch.randn(moe_inter, hidden)
            tensors[f"model.layers.{L}.mlp.experts.{E}.up_proj.weight"] = torch.randn(moe_inter, hidden)
            tensors[f"model.layers.{L}.mlp.experts.{E}.down_proj.weight"] = torch.randn(hidden, moe_inter)
        # Shared experts (preserve bit-exact)
        tensors[f"model.layers.{L}.mlp.shared_experts.gate_proj.weight"] = torch.randn(moe_inter, hidden)
        tensors[f"model.layers.{L}.mlp.shared_experts.up_proj.weight"] = torch.randn(moe_inter, hidden)
        tensors[f"model.layers.{L}.mlp.shared_experts.down_proj.weight"] = torch.randn(hidden, moe_inter)

    tensors["model.embed_tokens.weight"] = torch.randn(100, hidden)
    tensors["lm_head.weight"] = torch.randn(100, hidden)

    save_file(tensors, str(src / "model.safetensors"))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * 4 for t in tensors.values())},
        "weight_map": {n: "model.safetensors" for n in tensors},
    }, indent=2))
    (src / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "num_hidden_layers": 3,
        "first_k_dense_replace": 1,
        "n_routed_experts": n_routed,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "vocab_size": 100,
    }, indent=2))

    out = tmp_path / "synthetic-deepseek-pruned"
    prune_experts(
        model_dir=src,
        out_dir=out,
        keep_experts=2,
        layout=DEEPSEEK_V2_LAYOUT,
    )

    # Read back
    out_files = list(out.glob("*.safetensors"))
    assert out_files
    out_tensors: dict[str, "torch.Tensor"] = {}
    for sf in out_files:
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                out_tensors[k] = f.get_tensor(k)

    # Routed experts renumbered to {0, 1} on layers 1, 2
    routed_indices_per_layer: dict[int, set[int]] = {}
    for name in out_tensors:
        m = DEEPSEEK_V2_LAYOUT.expert_re().match(name)
        if m:
            routed_indices_per_layer.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    assert routed_indices_per_layer == {1: {0, 1}, 2: {0, 1}}, (
        f"expected {{1: {{0,1}}, 2: {{0,1}}}}, got {routed_indices_per_layer}"
    )

    # Shared experts bit-exact preservation (CRITICAL)
    for L in (1, 2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{L}.mlp.shared_experts.{proj}.weight"
            assert name in out_tensors, f"shared expert {name} missing from output"
            assert torch.equal(out_tensors[name], tensors[name]), (
                f"shared expert {name} was modified — must be bit-exact"
            )

    # Dense layer 0 bit-exact preservation
    for proj in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.0.mlp.{proj}.weight"
        assert name in out_tensors
        assert torch.equal(out_tensors[name], tensors[name])

    # Router gate sliced from 4 → 2 rows
    for L in (1, 2):
        gate = out_tensors[f"model.layers.{L}.mlp.gate.weight"]
        assert gate.shape == (2, hidden), f"layer {L} gate shape {gate.shape}"

    # Config: n_routed_experts updated
    new_cfg = json.loads((out / "config.json").read_text())
    assert new_cfg["n_routed_experts"] == 2, (
        f"n_routed_experts not updated; got {new_cfg.get('n_routed_experts')}"
    )
