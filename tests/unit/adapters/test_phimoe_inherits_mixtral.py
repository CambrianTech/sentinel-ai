"""TDD spec for PhiMoEAdapter — second block_sparse_moe-unfused family.

Phi-3.5-MoE shares the EXACT same tensor name pattern with Mixtral
(verified against Phi-3.5-MoE-instruct's published safetensors index):

    model.layers.{L}.block_sparse_moe.gate.weight
    model.layers.{L}.block_sparse_moe.experts.{K}.{w1,w2,w3}.weight

Only differences: 16 experts/layer (vs Mixtral's 8) and different
hidden / intermediate dimensions. The pruner doesn't care about
geometry — it reads num_experts from config.json.

The OOP move: PhiMoEAdapter inherits from MixtralAdapter. Inheritance
is the degenerate form of base extraction; when a third
block_sparse_moe-unfused family ships, the right move is to rename
MixtralAdapter to BlockSparseMoEUnfusedBase and have all three siblings
inherit from it. For now, since there are only two, the parent name
can be Mixtral and the child name is PhiMoE.

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_phimoe_adapter_is_registered():
    """Resolving 'phimoe' MUST yield a PhiMoEAdapter instance."""
    from adapters import resolve_family_adapter
    from adapters.sota_moe import PhiMoEAdapter
    a = resolve_family_adapter("phimoe")
    assert isinstance(a, PhiMoEAdapter)


def test_phimoe_inherits_from_mixtral():
    """PhiMoEAdapter MUST inherit from MixtralAdapter — they share the
    block_sparse_moe-unfused layout, the algorithm, and the entire
    expert_prune / expert_activation_profile code path. Inheritance
    is how the architecture handles a second sibling without code
    duplication."""
    from adapters.sota_moe import MixtralAdapter, PhiMoEAdapter
    assert issubclass(PhiMoEAdapter, MixtralAdapter), (
        "PhiMoEAdapter must inherit from MixtralAdapter — both families "
        "share the block_sparse_moe-unfused layout exactly. Adding a "
        "second sibling that duplicates the body is the OOP smell the "
        "never-branch rule prohibits."
    )


def test_phimoe_inherits_expert_prune_from_mixtral():
    """PhiMoEAdapter.expert_prune MUST be the inherited Mixtral version
    (which calls prune_experts with MIXTRAL_LAYOUT) — NOT the stub from
    sota_stubs that just raises NotImplementedError."""
    from adapters.sota_moe import MixtralAdapter, PhiMoEAdapter
    from adapters.base import FamilyAdapter
    # PhiMoEAdapter.expert_prune resolves up the MRO to MixtralAdapter.expert_prune
    # because PhiMoE doesn't override it. Both classes' attribute lookups
    # return the SAME function object.
    assert PhiMoEAdapter.expert_prune is MixtralAdapter.expert_prune, (
        "PhiMoEAdapter.expert_prune must be inherited from MixtralAdapter "
        "(same function object). If you defined a separate body, that's "
        "the duplication the architecture exists to prevent — delete it "
        "and let inheritance handle it."
    )
    # And the inherited method must NOT be the FamilyAdapter base stub
    assert PhiMoEAdapter.expert_prune is not FamilyAdapter.expert_prune


def test_phimoe_inherits_expert_activation_profile_from_mixtral():
    """Same shape for expert_activation_profile."""
    from adapters.sota_moe import MixtralAdapter, PhiMoEAdapter
    assert PhiMoEAdapter.expert_activation_profile is MixtralAdapter.expert_activation_profile


def test_phimoe_dispatch_only_path_short_circuits():
    """When ctx.model is None (Tier 1 dispatch path), the inherited
    expert_prune body short-circuits cleanly without invoking the pruner."""
    from adapters.sota_moe import PhiMoEAdapter
    from dataclasses import dataclass

    @dataclass
    class _MockCtx:
        model: object = None
        tokenizer: object = None
        output_dir: Path = Path("/tmp/test-phimoe")
        alloy: dict = None

    ctx = _MockCtx(alloy={"source": {"architecture": "phimoe"}})
    adapter = PhiMoEAdapter()
    result = adapter.expert_prune(
        ctx,
        keepExpertsPerLayer=8,
        originalExpertsPerLayer=16,
        strategy="calibration-aware-activation-count",
        expertTensorLayout="block_sparse_moe-unfused",
    )
    assert result is ctx


def test_phimoe_synthetic_end_to_end():
    """Same end-to-end fixture as the Mixtral test, but with the Phi-3.5
    expert geometry (16 experts/layer instead of 8). The layout is
    identical so the synthetic Mixtral fixture builder works directly."""
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("torch / safetensors not installed in this environment")

    from cpu_expert_prune_v2 import prune_experts, MIXTRAL_LAYOUT
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src = tmpdir / "synthetic-phimoe"
        src.mkdir()

        num_layers = 2
        num_experts = 16  # Phi-3.5-MoE geometry
        num_experts_per_tok = 2
        hidden = 8
        inter = 16

        tensors = {}
        for L in range(num_layers):
            tensors[f"model.layers.{L}.block_sparse_moe.gate.weight"] = torch.randn(num_experts, hidden)
            for E in range(num_experts):
                tensors[f"model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight"] = torch.randn(inter, hidden)
                tensors[f"model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight"] = torch.randn(hidden, inter)
                tensors[f"model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight"] = torch.randn(inter, hidden)
        tensors["model.embed_tokens.weight"] = torch.randn(100, hidden)
        tensors["lm_head.weight"] = torch.randn(100, hidden)

        save_file(tensors, str(src / "model.safetensors"))
        (src / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(t.numel() * 4 for t in tensors.values())},
            "weight_map": {n: "model.safetensors" for n in tensors},
        }, indent=2))
        (src / "config.json").write_text(json.dumps({
            "architectures": ["PhiMoEForCausalLM"],
            "model_type": "phimoe",
            "num_hidden_layers": num_layers,
            "num_local_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "hidden_size": hidden,
            "intermediate_size": inter,
            "vocab_size": 100,
        }, indent=2))

        out = tmpdir / "synthetic-phimoe-pruned"
        metadata = prune_experts(
            model_dir=src,
            out_dir=out,
            keep_experts=8,  # Phi-3.5: keep half the experts
            layout=MIXTRAL_LAYOUT,  # Same layout as Mixtral
        )

        # Verify the output has the renumbered experts {0..7}
        from safetensors import safe_open
        out_files = list(out.glob("*.safetensors"))
        found = set()
        for sf in out_files:
            with safe_open(str(sf), framework="pt") as f:
                for name in f.keys():
                    m = MIXTRAL_LAYOUT.expert_re().match(name)
                    if m:
                        found.add(int(m.group(2)))
        assert found == set(range(8)), (
            f"expected renumbered expert indices 0..7, got {sorted(found)}"
        )
        # Updated config: num_local_experts=8
        new_cfg = json.loads((out / "config.json").read_text())
        assert new_cfg.get("num_local_experts") == 8
