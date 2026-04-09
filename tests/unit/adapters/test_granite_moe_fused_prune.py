"""TDD spec for GraniteMoE fused-tensor pruning.

GraniteMoE is structurally distinct from the unfused MoE families
(Mixtral, Qwen3MoE, OLMoE, DeepSeek-V2). Instead of one tensor per
expert per projection, all experts in a layer share THREE big tensors
along an expert axis:

    model.layers.{L}.block_sparse_moe.input_linear.weight
        shape [num_experts, 2 * intermediate, hidden]
        (fuses gate_proj and up_proj for all experts)
    model.layers.{L}.block_sparse_moe.output_linear.weight
        shape [num_experts, hidden, intermediate]
        (down_proj for all experts)
    model.layers.{L}.block_sparse_moe.router.layer.weight
        shape [num_experts, hidden]

To prune k of n experts you SLICE these tensors along axis=0 (expert
axis) instead of dropping/renaming named param entries. The same Pass
1 (importance reading) and Pass 2 (streaming rewrite) shape applies,
but the per-tensor handling is fundamentally different — it's a
slice, not a delete-and-rename.

This is the granite-moe-fused layout. New pruner function:
cpu_expert_prune_v2.prune_experts_fused(layout=GRANITE_MOE_LAYOUT, ...)

GraniteMoEAdapter.expert_prune graduates from a stub to a real body
that calls prune_experts_fused. expert_activation_profile also
graduates because the profiler now accepts a gate_attr_path
parameter (Granite passes 'block_sparse_moe.router.layer' instead
of the unfused default 'mlp.gate').

Verified from HF:
    ibm-granite/granite-3.0-3b-a800m-instruct
    model_type: granitemoe, 32 layers × 40 experts, 8 active per token
    hidden 1536, intermediate 512
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Layout spec for the fused case ──────────────────────────────────────────


def test_granite_moe_layout_constant_exists():
    from cpu_expert_prune_v2 import GRANITE_MOE_LAYOUT, FusedLayoutSpec
    assert isinstance(GRANITE_MOE_LAYOUT, FusedLayoutSpec)
    assert GRANITE_MOE_LAYOUT.family_name == "granitemoe"


def test_granite_moe_layout_matches_real_tensor_names():
    from cpu_expert_prune_v2 import GRANITE_MOE_LAYOUT
    real = [
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "model.layers.0.block_sparse_moe.router.layer.weight",
        "model.layers.31.block_sparse_moe.input_linear.weight",
    ]
    for name in real:
        m = GRANITE_MOE_LAYOUT.match_re().match(name)
        assert m is not None, f"GRANITE_MOE_LAYOUT.match_re does not match {name!r}"
    # The layout exposes which tensors hold the per-expert axis
    assert "input_linear" in GRANITE_MOE_LAYOUT.fused_tensor_names
    assert "output_linear" in GRANITE_MOE_LAYOUT.fused_tensor_names
    assert "router" in GRANITE_MOE_LAYOUT.gate_tensor_names \
        or any("router" in n for n in GRANITE_MOE_LAYOUT.gate_tensor_names)


# ── prune_experts_fused: end-to-end synthetic ──────────────────────────────


def test_prune_experts_fused_slices_synthetic_granite_fixture(tmp_path):
    """Build a tiny synthetic granitemoe-shaped directory, prune 4 of 8
    experts, verify the fused tensors are sliced along axis 0 to length 4."""
    try:
        import torch
        from safetensors.torch import save_file
        from safetensors import safe_open
    except ImportError:
        pytest.skip("torch / safetensors not installed")

    from cpu_expert_prune_v2 import prune_experts_fused, GRANITE_MOE_LAYOUT

    src = tmp_path / "synthetic-granite"
    src.mkdir()

    num_layers = 3
    num_experts = 8
    num_experts_per_tok = 2
    hidden = 16
    intermediate = 8

    tensors: dict[str, "torch.Tensor"] = {}
    for L in range(num_layers):
        # input_linear: [num_experts, 2 * intermediate, hidden]
        tensors[f"model.layers.{L}.block_sparse_moe.input_linear.weight"] = torch.randn(
            num_experts, 2 * intermediate, hidden
        )
        # output_linear: [num_experts, hidden, intermediate]
        tensors[f"model.layers.{L}.block_sparse_moe.output_linear.weight"] = torch.randn(
            num_experts, hidden, intermediate
        )
        # router: [num_experts, hidden]
        tensors[f"model.layers.{L}.block_sparse_moe.router.layer.weight"] = torch.randn(
            num_experts, hidden
        )
    tensors["model.embed_tokens.weight"] = torch.randn(100, hidden)
    tensors["lm_head.weight"] = torch.randn(100, hidden)

    save_file(tensors, str(src / "model.safetensors"))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * 4 for t in tensors.values())},
        "weight_map": {n: "model.safetensors" for n in tensors},
    }, indent=2))
    (src / "config.json").write_text(json.dumps({
        "architectures": ["GraniteMoeForCausalLM"],
        "model_type": "granitemoe",
        "num_hidden_layers": num_layers,
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": 100,
    }, indent=2))

    out = tmp_path / "synthetic-granite-pruned"
    metadata = prune_experts_fused(
        model_dir=src,
        out_dir=out,
        keep_experts=4,
        layout=GRANITE_MOE_LAYOUT,
    )

    # Read back and verify the sliced shapes
    out_files = list(out.glob("*.safetensors"))
    assert out_files
    out_tensors: dict[str, "torch.Tensor"] = {}
    for sf in out_files:
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                out_tensors[k] = f.get_tensor(k)

    for L in range(num_layers):
        il = out_tensors[f"model.layers.{L}.block_sparse_moe.input_linear.weight"]
        ol = out_tensors[f"model.layers.{L}.block_sparse_moe.output_linear.weight"]
        rt = out_tensors[f"model.layers.{L}.block_sparse_moe.router.layer.weight"]
        assert il.shape == (4, 2 * intermediate, hidden), f"layer {L} input_linear shape {il.shape}"
        assert ol.shape == (4, hidden, intermediate), f"layer {L} output_linear shape {ol.shape}"
        assert rt.shape == (4, hidden), f"layer {L} router shape {rt.shape}"

    # Embed/lm_head pass through bit-exact
    assert out_tensors["model.embed_tokens.weight"].shape == (100, hidden)

    # config updated
    new_cfg = json.loads((out / "config.json").read_text())
    assert new_cfg["num_local_experts"] == 4

    # Sidecar metadata layout_family
    sidecar = json.loads((out / "expert_prune.metadata.v1.json").read_text())
    assert sidecar["selection"]["layout_family"] == "granitemoe"


# ── Adapter graduation ──────────────────────────────────────────────────────


def test_granite_moe_adapter_no_longer_stub():
    import inspect
    from adapters.sota_moe import GraniteMoEAdapter
    src = inspect.getsource(GraniteMoEAdapter.expert_prune)
    assert "prune_experts_fused" in src
    assert "GRANITE_MOE_LAYOUT" in src
    assert "_stub_expert_prune_raise" not in src


def test_granite_moe_adapter_dispatch_only_path_short_circuits():
    from adapters.sota_moe import GraniteMoEAdapter
    from dataclasses import dataclass

    @dataclass
    class _MockCtx:
        model: object = None
        tokenizer: object = None
        output_dir: Path = Path("/tmp/test-granite")
        alloy: dict = None

    ctx = _MockCtx(alloy={"source": {"architecture": "granitemoe"}})
    adapter = GraniteMoEAdapter()
    result = adapter.expert_prune(
        ctx,
        keepExpertsPerLayer=20,
        originalExpertsPerLayer=40,
        strategy="calibration-aware-activation-count",
        expertTensorLayout="granite-moe-fused",
    )
    assert result is ctx


def test_granite_moe_adapter_resolves_via_dispatch():
    from adapters import resolve_family_adapter
    from adapters.sota_moe import GraniteMoEAdapter
    a = resolve_family_adapter("granitemoe")
    assert isinstance(a, GraniteMoEAdapter)


# ── Profiler family-awareness ───────────────────────────────────────────────


def test_profile_experts_accepts_gate_attr_path():
    """The profiler must accept a gate_attr_path parameter so the
    family adapter can pass the family-specific module path. Default
    'mlp.gate' for unfused Qwen3MoE / OLMoE backwards compat."""
    import inspect
    from expert_activation_profile import profile_experts
    sig = inspect.signature(profile_experts)
    assert "gate_attr_path" in sig.parameters, (
        "profile_experts must accept gate_attr_path so family adapters "
        "can pass family-specific router paths (Granite uses "
        "'block_sparse_moe.router.layer' instead of 'mlp.gate')"
    )
    # Default value preserves backwards compat
    assert sig.parameters["gate_attr_path"].default == "mlp.gate"
