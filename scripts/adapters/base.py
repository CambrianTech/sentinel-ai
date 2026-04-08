"""FamilyAdapter — abstract base for per-architecture forge implementations.

A family adapter is the per-model-family implementation of forge concerns.
It is the SECOND axis of dispatch in the forge pipeline:

    Axis 1 (already exists in scripts/stages/): stage type → StageExecutor
        prune → PruneExecutor
        train → TrainExecutor
        expert-prune → ExpertPruneExecutor
        ...

    Axis 2 (this module): source.architecture → FamilyAdapter
        qwen3_5            → Qwen3DenseAdapter
        qwen3_moe          → Qwen3MoEAdapter
        olmoe              → OlmoeAdapter
        mixtral            → MixtralAdapter
        ...

A StageExecutor's execute() method now reads its stage params, looks up the
FamilyAdapter for ctx.model's architecture, and delegates the actual model-
touching work to the family adapter. The StageExecutor never reaches into
torch.nn.Module trees; the family adapter does.

Why: per the never-branch rule, new model families must NEVER be handled by
adding `if architectures[0] == "..."` branches to existing scripts. New family
= new FamilyAdapter subclass in scripts/adapters/, registered in registry.py.
Old adapters stay frozen so older alloys keep reproducing bit-identically.

Reproducibility guarantee: every published continuum-ai/* alloy declares its
source.architecture. The dispatcher looks that up and gets back THE EXACT
adapter that originally forged the artifact. Old alloys never drift.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid runtime torch import — Tier 1 dispatch resolution must work on a
    # plain Mac with no torch installed. Real adapter execution imports torch
    # lazily inside methods, never at module import time.
    from scripts.stages.base import ForgeContext


@dataclass
class AdapterCall:
    """One step in a resolved adapter chain.

    Returned by resolve_adapter_chain(). Tier 1 (dispatch resolution only)
    produces a list of these without ever loading the model. Tier 2 actually
    invokes them in order on a ForgeContext.
    """
    stage_type: str                     # "prune", "train", "expert-prune", ...
    stage_index: int                    # position in alloy.stages[]
    family_adapter: "FamilyAdapter"     # the resolved family adapter for the alloy's source.architecture
    method_name: str                    # which adapter method handles this stage type
    params: dict[str, Any] = field(default_factory=dict)  # the stage's own config dict from the alloy

    def __repr__(self) -> str:
        return (
            f"AdapterCall(stage[{self.stage_index}].{self.stage_type} → "
            f"{type(self.family_adapter).__name__}.{self.method_name}, "
            f"params={sorted(self.params.keys())})"
        )


class FamilyAdapter(ABC):
    """Abstract base for one model family's forge implementation.

    Concrete subclasses implement only the stage-handler methods that apply
    to their family. Methods that don't apply (e.g. expert-prune on a dense
    model) raise NotImplementedError with a clear "this family does not
    support stage X" message — that's how the dispatcher catches alloy/family
    mismatches at the contract layer rather than failing deep in tensor code.

    Each method takes (ctx, **stage_params) and mutates ctx in place,
    returning the mutated ctx for chaining. The exact param contract for each
    stage type is defined by the forge-alloy llm-forge domain extension —
    when that lands, this base class's method signatures will validate
    against it.
    """

    # Subclass MUST set this — the source.architecture string(s) it handles.
    # registry.register_family_adapter() reads this to populate the lookup map.
    architectures: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Human-readable adapter name for logs and dispatch reports."""
        return type(self).__name__

    def log(self, msg: str) -> None:
        """Adapter log helper. Mirrors StageExecutor.log so adapter methods
        and stage executors produce visually consistent output."""
        print(f"  [{self.name}] {msg}")

    # ── Stage handlers — subclasses override only what applies ──────────────

    def prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Dense head pruning. Override in dense families.

        Params (from the alloy's prune stage):
            strategy: str  — entropy | magnitude | gradient | activation-magnitude | ...
            level: float   — fraction of heads to remove (0.0..0.9)
            (additional adapter-specific params via the stage's extras)
        """
        raise NotImplementedError(
            f"{self.name} does not implement dense head pruning. "
            f"If this is a dense model family, override .prune() in the subclass. "
            f"If this is a MoE family, the alloy should use 'expert-prune' not 'prune'."
        )

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Recovery / fine-tuning training. Override in families that train.

        Params (from the alloy's train or lora stage):
            domain: str        — code | general | math | ...
            dataset: str       — HF dataset id
            steps: int
            learningRate: str
            batchSize: int
            (additional adapter-specific params via the stage's extras)
        """
        raise NotImplementedError(
            f"{self.name} does not implement training. Override .train() in the subclass."
        )

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """MoE expert pruning. Override in MoE families."""
        raise NotImplementedError(
            f"{self.name} does not implement expert pruning. "
            f"This is a MoE-only stage; the alloy's source.architecture must declare a MoE family."
        )

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§4.1.3.4 calibration-aware MoE expert importance profiling."""
        raise NotImplementedError(
            f"{self.name} does not implement expert-activation-profile. MoE-only stage."
        )

    def compensation_lora(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§4.1.3.3 KL-distillation-against-teacher compensation LoRA."""
        raise NotImplementedError(
            f"{self.name} does not implement compensation-lora."
        )

    def context_extend(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Context-window extension via YaRN / NTK / etc."""
        raise NotImplementedError(
            f"{self.name} does not implement context-extend."
        )

    def modality(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Modality addition (vision / audio encoder attach)."""
        raise NotImplementedError(
            f"{self.name} does not implement modality stages."
        )

    def source_config(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Front-bookend stage. Default no-op — most families just declare capabilities."""
        return ctx

    # ── Output / packaging stages ───────────────────────────────────────────
    #
    # These are family-agnostic by default — quantization, evaluation, and
    # packaging are mostly the same regardless of model family. Concrete
    # adapters override only when the family has family-specific concerns
    # (e.g. MoE quant tier selection differs from dense quant tier selection).
    # The default is "supported but no-op at the family-adapter layer" —
    # the actual work lives in the existing scripts/stages/output_stages.py
    # executors that the dispatcher can call directly. The family adapter
    # is here so dispatch resolution succeeds and Tier 2 wiring has a clear
    # extension point if a family ever needs to override.

    def quant(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Quantization to GGUF / MLX / safetensors / ONNX. Default: no-op
        at the family layer — the existing QuantExecutor handles it."""
        return ctx

    def eval(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Benchmark evaluation. Default: no-op at the family layer — the
        existing EvalExecutor handles it via eval_with_calibration.py."""
        return ctx

    def publish(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Publish artifact to HF (or other registry). Default: no-op at
        the family layer — PublishExecutor / publish_model.py handles it."""
        return ctx

    def package(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """End-bookend packaging stage. Default: no-op at the family layer."""
        return ctx

    def deploy(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Deploy to a grid node. Default: no-op at the family layer."""
        return ctx

    def deliver(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Delivery stage (alias for publish in some legacy alloys)."""
        return ctx

    # ── Stage type → method name mapping ────────────────────────────────────

    # Used by the dispatcher to route an alloy stage to the right adapter
    # method. Stages whose names contain hyphens map to snake_case methods.
    # Methods that MUST be overridden by a concrete family adapter for any
    # alloy that includes the corresponding stage type. The dispatch test
    # uses this to fail loudly if a family adapter inherits a NotImplementedError
    # stub instead of providing real behavior. Output / bookend stages are
    # family-agnostic by default and are NOT in this set.
    REQUIRES_FAMILY_OVERRIDE: frozenset[str] = frozenset({
        "prune",
        "train",
        "expert_prune",
        "expert_activation_profile",
        "compensation_lora",
        "context_extend",
        "modality",
    })

    STAGE_METHOD_MAP: dict[str, str] = {
        # Input stages
        "source-config":             "source_config",
        "context-extend":            "context_extend",
        "modality":                  "modality",
        # Transform stages
        "prune":                     "prune",
        "train":                     "train",
        "lora":                      "train",   # LoRA is a training variant — same handler for now
        "expert-prune":              "expert_prune",
        "expert-activation-profile": "expert_activation_profile",
        "compensation-lora":         "compensation_lora",
        # Output / bookend stages
        "quant":                     "quant",
        "eval":                      "eval",
        "publish":                   "publish",
        "package":                   "package",
        "deploy":                    "deploy",
        "deliver":                   "deliver",
    }
