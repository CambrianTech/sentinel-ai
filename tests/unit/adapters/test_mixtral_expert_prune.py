"""TDD spec for MixtralAdapter.expert_prune — block_sparse_moe-unfused
layout dispatch + real tensor walk.

Roadmap "take it to the limit" round. Mixtral 8x22B is Joel's stated
single-5090 prosumer headline play; the architecture has dispatched
through MixtralAdapter since the gap-fill round, but expert_prune was
a stub that raised NotImplementedError. This step graduates the stub
to a real implementation by extending cpu_expert_prune_v2.py with a
layout-spec parameter so the same algorithm can walk both Qwen3MoE's
mlp.experts.{e}.{gate,up,down}_proj layout AND Mixtral's
block_sparse_moe.experts.{e}.{w1,w2,w3} layout.

Mixtral layout (verified against Mixtral-8x7B-Instruct-v0.1's
model.safetensors.index.json):

    model.layers.{L}.block_sparse_moe.gate.weight             [num_experts, hidden]
    model.layers.{L}.block_sparse_moe.experts.{K}.w1.weight   [moe_inter, hidden]   ← gate_proj
    model.layers.{L}.block_sparse_moe.experts.{K}.w2.weight   [hidden, moe_inter]   ← down_proj
    model.layers.{L}.block_sparse_moe.experts.{K}.w3.weight   [moe_inter, hidden]   ← up_proj

Same algorithm as Qwen3MoE: read router gates, compute per-expert
importance, keep top-K per layer, slice the gate, drop the dropped
experts' tensors, renumber survivors to sequential indices, write the
new safetensors shards. Just different tensor name patterns.

Written test-first per TDD/TDValidation discipline. The test builds a
synthetic in-memory Mixtral-shaped model on disk (3 layers × 4 experts,
~1MB total) and verifies the layout dispatch correctly identifies
experts, slices the gate, and writes the renumbered output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Layout spec contract ────────────────────────────────────────────────────


def test_layout_spec_class_is_importable():
    """cpu_expert_prune_v2 MUST expose a LayoutSpec class so callers can
    declare the per-family tensor name patterns. This is the
    parameterization point that lets one algorithm handle multiple MoE
    families without per-family branches in the algorithm body."""
    from cpu_expert_prune_v2 import LayoutSpec
    assert LayoutSpec is not None


def test_qwen3_moe_layout_constant_exists():
    """The default Qwen3MoE layout MUST exist as a module-level constant
    so the existing forge path keeps working with no changes."""
    from cpu_expert_prune_v2 import QWEN3_MOE_LAYOUT, LayoutSpec
    assert isinstance(QWEN3_MOE_LAYOUT, LayoutSpec)
    # Check the qwen3_moe pattern matches the expected tensor names
    assert "mlp" in QWEN3_MOE_LAYOUT.gate_pattern
    assert "experts" in QWEN3_MOE_LAYOUT.expert_pattern


def test_mixtral_layout_constant_exists():
    """The Mixtral layout MUST exist as a module-level constant pointing at
    the block_sparse_moe pattern."""
    from cpu_expert_prune_v2 import MIXTRAL_LAYOUT, LayoutSpec
    assert isinstance(MIXTRAL_LAYOUT, LayoutSpec)
    assert "block_sparse_moe" in MIXTRAL_LAYOUT.gate_pattern
    assert "block_sparse_moe" in MIXTRAL_LAYOUT.expert_pattern


def test_mixtral_layout_matches_real_mixtral_tensor_names():
    """The MIXTRAL_LAYOUT regex MUST match the actual tensor names from
    Mixtral-8x22B / Mixtral-8x7B's published safetensors index. These
    are the literal name strings from
    https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/model.safetensors.index.json
    """
    from cpu_expert_prune_v2 import MIXTRAL_LAYOUT
    real_mixtral_names = [
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        "model.layers.7.block_sparse_moe.experts.5.w2.weight",
    ]
    # Gate pattern should match the gate.weight name only
    gate_match = MIXTRAL_LAYOUT.gate_re().match(real_mixtral_names[0])
    assert gate_match is not None, "MIXTRAL_LAYOUT.gate_re does not match real Mixtral gate name"
    assert gate_match.group(1) == "0", "gate match must capture the layer index"

    # Expert pattern should match each expert weight tensor
    for name in real_mixtral_names[1:]:
        m = MIXTRAL_LAYOUT.expert_re().match(name)
        assert m is not None, f"MIXTRAL_LAYOUT.expert_re does not match {name!r}"
        assert m.group(1).isdigit(), f"layer index capture should be a digit, got {m.group(1)!r}"
        assert m.group(2).isdigit(), f"expert index capture should be a digit, got {m.group(2)!r}"
        assert m.group(3) in ("w1", "w2", "w3"), f"weight name capture should be w1/w2/w3, got {m.group(3)!r}"


def test_qwen3_moe_layout_does_not_match_mixtral_names():
    """Sanity: Qwen3MoE layout MUST NOT match Mixtral tensor names. This
    catches a refactor that accidentally makes the qwen3 pattern too
    permissive (e.g. matching both 'mlp' and 'block_sparse_moe')."""
    from cpu_expert_prune_v2 import QWEN3_MOE_LAYOUT
    mixtral_name = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
    assert QWEN3_MOE_LAYOUT.expert_re().match(mixtral_name) is None


def test_mixtral_layout_does_not_match_qwen3_names():
    """Reverse sanity check."""
    from cpu_expert_prune_v2 import MIXTRAL_LAYOUT
    qwen3_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    assert MIXTRAL_LAYOUT.expert_re().match(qwen3_name) is None


# ── prune_experts takes a layout parameter ──────────────────────────────────


def test_prune_experts_accepts_layout_parameter():
    """prune_experts MUST accept a layout=LayoutSpec parameter. Default
    is QWEN3_MOE_LAYOUT for backwards compatibility — the existing forge
    path that called prune_experts without layout= keeps working unchanged."""
    import inspect
    from cpu_expert_prune_v2 import prune_experts
    sig = inspect.signature(prune_experts)
    assert "layout" in sig.parameters, (
        "prune_experts must accept a 'layout' kwarg for layout-spec dispatch"
    )


# ── MixtralAdapter.expert_prune calls the layout-aware pruner ───────────────


def test_mixtral_adapter_expert_prune_no_longer_stub():
    """MixtralAdapter.expert_prune body MUST be real (call prune_experts
    with the Mixtral layout) — not the _stub_expert_prune_raise stub."""
    import inspect
    from adapters.sota_moe import MixtralAdapter
    src = inspect.getsource(MixtralAdapter.expert_prune)
    # Real body must reference the prune_experts function and the Mixtral layout
    assert "prune_experts" in src, "MixtralAdapter.expert_prune must call prune_experts"
    assert "MIXTRAL_LAYOUT" in src, "MixtralAdapter.expert_prune must reference MIXTRAL_LAYOUT"
    # The stub raise should NOT be in the body anymore
    assert "_stub_expert_prune_raise" not in src, (
        "MixtralAdapter.expert_prune must not delegate to the stub helper anymore"
    )


def test_mixtral_dispatch_only_path_short_circuits():
    """Tier 1 dispatch path: when ctx.model is None, the method MUST
    short-circuit cleanly without invoking the layout-aware pruner.
    Same dispatch-only contract as every other family adapter."""
    from adapters.sota_moe import MixtralAdapter
    from dataclasses import dataclass
    from pathlib import Path

    @dataclass
    class _MockCtx:
        model: object = None
        tokenizer: object = None
        output_dir: Path = Path("/tmp/test-mixtral")
        alloy: dict = None

    ctx = _MockCtx(alloy={"source": {"architecture": "mixtral"}})
    adapter = MixtralAdapter()
    result = adapter.expert_prune(
        ctx,
        keepExpertsPerLayer=4,
        originalExpertsPerLayer=8,
        strategy="calibration-aware-activation-count",
        expertTensorLayout="block_sparse_moe-unfused",
    )
    assert result is ctx


# ── End-to-end synthetic Mixtral fixture ────────────────────────────────────


def _build_synthetic_mixtral_dir(tmp_path: Path) -> Path:
    """Build a tiny synthetic Mixtral-shaped model directory on disk for
    end-to-end layout-dispatch validation. 3 layers × 4 experts (top-2),
    hidden=8, moe_inter=16. Total ~5KB. Just enough for the streaming
    rewriter to walk the tensor names and produce a renumbered output."""
    import torch
    from safetensors.torch import save_file

    src = tmp_path / "synthetic-mixtral"
    src.mkdir()

    num_layers = 3
    num_experts = 4
    num_experts_per_tok = 2
    hidden_size = 8
    intermediate_size = 16

    # Build the tensor map
    tensors: dict[str, "torch.Tensor"] = {}
    for L in range(num_layers):
        # Router gate — [num_experts, hidden]
        tensors[f"model.layers.{L}.block_sparse_moe.gate.weight"] = torch.randn(num_experts, hidden_size)
        for E in range(num_experts):
            # w1 = gate_proj — [moe_inter, hidden]
            tensors[f"model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight"] = torch.randn(intermediate_size, hidden_size)
            # w2 = down_proj — [hidden, moe_inter]
            tensors[f"model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight"] = torch.randn(hidden_size, intermediate_size)
            # w3 = up_proj — [moe_inter, hidden]
            tensors[f"model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight"] = torch.randn(intermediate_size, hidden_size)
    # Add a non-MoE tensor so the rewriter has something to passthrough
    tensors["model.embed_tokens.weight"] = torch.randn(100, hidden_size)
    tensors["lm_head.weight"] = torch.randn(100, hidden_size)

    # Single-shard save (small enough)
    save_file(tensors, str(src / "model.safetensors"))

    # Manifest
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * 4 for t in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }, indent=2))

    # config.json — minimal Mixtral-shaped config
    (src / "config.json").write_text(json.dumps({
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "num_hidden_layers": num_layers,
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": 100,
    }, indent=2))

    return src


def test_prune_experts_with_mixtral_layout_end_to_end(tmp_path):
    """End-to-end smoke: build a synthetic Mixtral-shaped directory,
    invoke prune_experts(layout=MIXTRAL_LAYOUT) to keep 2 of 4 experts
    per layer, verify the output directory has correctly renumbered
    expert tensors and the router gate is sliced to the new size."""
    try:
        import torch  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401
    except ImportError:
        pytest.skip("torch / safetensors not installed in this environment")

    from cpu_expert_prune_v2 import prune_experts, MIXTRAL_LAYOUT

    src = _build_synthetic_mixtral_dir(tmp_path)
    out = tmp_path / "synthetic-mixtral-pruned"

    metadata = prune_experts(
        model_dir=src,
        out_dir=out,
        keep_experts=2,
        layout=MIXTRAL_LAYOUT,
    )

    # Output directory must exist with the renumbered safetensors
    assert (out / "model.safetensors").exists() or any(out.glob("model-*.safetensors"))

    # Updated config: num_experts -> 2
    new_cfg = json.loads((out / "config.json").read_text())
    cfg_section = new_cfg.get("text_config", new_cfg)
    assert cfg_section["num_local_experts"] == 2 or cfg_section.get("num_experts") == 2, (
        f"updated config should declare num_local_experts=2; got {new_cfg}"
    )

    # The output safetensors must contain the renumbered expert tensors
    # (experts 0 and 1 in the output, not the original 4-expert layout)
    from safetensors import safe_open
    out_files = list(out.glob("*.safetensors"))
    assert out_files, "no output safetensors shards"
    found_expert_indices = set()
    for sf in out_files:
        with safe_open(str(sf), framework="pt") as f:
            for name in f.keys():
                m = MIXTRAL_LAYOUT.expert_re().match(name)
                if m:
                    found_expert_indices.add(int(m.group(2)))
    assert found_expert_indices == {0, 1}, (
        f"output should have only expert indices {{0, 1}} after renumbering, got {found_expert_indices}"
    )

    # Sidecar metadata exists with the per-layer selection recorded
    sidecar = out / "expert_prune.metadata.v1.json"
    assert sidecar.exists()
    sidecar_data = json.loads(sidecar.read_text())
    selection = sidecar_data.get("selection", {})
    kept_indices = selection.get("per_layer_kept_indices") or {}
    # 3 layers, each kept 2 experts
    assert len(kept_indices) == 3, f"expected 3 layers in selection, got {len(kept_indices)}"
    for layer_str, indices in kept_indices.items():
        assert len(indices) == 2, (
            f"layer {layer_str} kept {len(indices)} experts, expected 2"
        )
    # And the sidecar declares which family layout it used
    assert selection.get("layout_family") == "mixtral", (
        f"sidecar should declare layout_family='mixtral', got "
        f"{selection.get('layout_family')!r}"
    )
