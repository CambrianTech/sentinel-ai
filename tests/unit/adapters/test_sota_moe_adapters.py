"""TDD spec for the SOTA MoE family adapters — Mixtral, Phi-MoE,
GraniteMoE, DeepSeek-V2.

Roadmap "fill the gaps" follow-up. The MoEUnfusedExpertsBase from
roadmap step 2 handles the unfused-experts layout that Qwen3MoE and
OLMoE share. But the SOTA MoE families that Kash mapped use DIFFERENT
module-tree layouts and need their own family adapters per the
never-branch rule:

  Mixtral 8x22B / Mixtral 8x7B
      arch:   mixtral
      layout: model.layers.{i}.block_sparse_moe.experts.{e}.w[123]
      base:   MoE-block_sparse_moe layout

  Phi-3.5-MoE
      arch:   phimoe
      layout: model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}
      base:   shares block_sparse_moe with Mixtral but smaller experts

  Granite-MoE / Granite-3.1-3b-a800m-instruct
      arch:   granitemoe
      layout: model.layers.{i}.block_sparse_moe.input_linear / output_linear
              (FUSED experts — all experts share one big tensor)
      base:   structurally distinct from unfused; needs its own walk

  DeepSeek-V2 / DeepSeek-V2-Lite
      arch:   deepseek_v2
      layout: model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}
              + model.layers.{i}.mlp.shared_experts.* (routed + shared)
      base:   superset of unfused with shared-expert pathway

This file gates the existence + dispatch correctness of each. The
families are structurally novel enough that each one's expert_prune
will need its own tensor walk, but for now the test asserts:

  - Each architecture string resolves to a registered adapter
  - The adapter is registered in the singleton
  - The dispatch test routes a synthetic alloy through it cleanly

Each family's REAL expert_prune body is a follow-up TDD cycle once the
first forge of that family runs. The stub raises NotImplementedError
loudly with a pointer to the layout doc, mirroring the SOTA eval-runner
stubs pattern.

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


SOTA_MOE_ARCHITECTURES = [
    ("mixtral",     "MixtralAdapter",     "block_sparse_moe-unfused"),
    ("phimoe",      "PhiMoEAdapter",      "block_sparse_moe-unfused"),
    ("granitemoe",  "GraniteMoEAdapter",  "granite-moe-fused"),
    ("deepseek_v2", "DeepSeekV2Adapter",  "deepseek-routed-shared"),
]


@pytest.mark.parametrize("arch,class_name,layout", SOTA_MOE_ARCHITECTURES)
def test_sota_moe_adapter_is_registered(arch: str, class_name: str, layout: str):
    """Each SOTA MoE architecture string resolves to its registered adapter."""
    from adapters import resolve_family_adapter
    from adapters.sota_moe import (
        MixtralAdapter, PhiMoEAdapter, GraniteMoEAdapter, DeepSeekV2Adapter,
    )
    by_name = {
        "MixtralAdapter":     MixtralAdapter,
        "PhiMoEAdapter":      PhiMoEAdapter,
        "GraniteMoEAdapter":  GraniteMoEAdapter,
        "DeepSeekV2Adapter":  DeepSeekV2Adapter,
    }
    expected = by_name[class_name]
    a = resolve_family_adapter(arch)
    assert isinstance(a, expected)


@pytest.mark.parametrize("arch,class_name,layout", SOTA_MOE_ARCHITECTURES)
def test_sota_moe_adapter_inherits_from_family_adapter(arch, class_name, layout):
    """Each SOTA MoE adapter inherits from the FamilyAdapter ABC. Whether
    it inherits from MoEUnfusedExpertsBase OR from FamilyAdapter directly
    depends on whether its layout is compatible with the unfused base —
    Mixtral and Phi-MoE share the block_sparse_moe-unfused pattern with
    each other (so they could share a sub-base later); GraniteMoE-fused
    and DeepSeek-V2 routed+shared are structurally distinct."""
    from adapters import resolve_family_adapter
    from adapters.base import FamilyAdapter
    a = resolve_family_adapter(arch)
    assert isinstance(a, FamilyAdapter)


@pytest.mark.parametrize("arch,class_name,layout", SOTA_MOE_ARCHITECTURES)
def test_sota_moe_dispatch_resolves(arch, class_name, layout):
    """A synthetic MoE alloy with the architecture string resolves through
    the right adapter."""
    from adapters import resolve_adapter_chain
    synthetic = {
        "name": f"synthetic-{arch}-test",
        "version": "0.0.1-test",
        "source": {
            "baseModel": f"some-org/{arch}-model",
            "architecture": arch,
            "isMoE": True,
        },
        "stages": [
            {
                "type": "expert-prune",
                "strategy": "calibration-aware-activation-count",
                "keepExpertsPerLayer": 8,
                "originalExpertsPerLayer": 16,
                "expertTensorLayout": layout,
            },
        ],
        "cycles": 1,
    }
    chain = resolve_adapter_chain(synthetic)
    assert len(chain) == 1
    assert chain[0].method_name == "expert_prune"
    family_class = type(chain[0].family_adapter).__name__
    assert family_class == class_name


def test_sota_moe_layout_dispatch_inside_expert_prune():
    """Each adapter MUST raise loudly with a layout-specific message when
    expert_prune is called against a real model — the body is a stub
    until the first forge of that family runs, but the stub's
    NotImplementedError MUST name the expected layout so a future
    implementer knows what tensor walk to write.

    This is the same pattern as the SOTA eval-runner stubs: registered
    so dispatch resolves, raises loudly so calling without an
    implementation fails LOUDLY at the runner site, never silently."""
    from adapters import resolve_family_adapter
    import inspect
    for arch, class_name, layout in SOTA_MOE_ARCHITECTURES:
        adapter = resolve_family_adapter(arch)
        src = inspect.getsource(type(adapter).expert_prune)
        # The body must mention the layout discriminator string so an
        # implementer reading the file knows which tensor walk to write.
        assert layout in src or layout.replace("-", "_") in src, (
            f"{class_name}.expert_prune must reference its layout discriminator "
            f"{layout!r} so an implementer reading the file knows which "
            f"tensor walk to write. Got: {src[:200]}"
        )
