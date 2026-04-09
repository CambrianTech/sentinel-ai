"""Transform stages — middle of the pipeline, cycled.

These executors are now THIN dispatchers that delegate the actual model-
touching work to a family adapter resolved from ctx.alloy['source']['architecture'].
The per-family tensor walks live in scripts/adapters/<family>.py.

Why: a new model family must NEVER be handled by editing one of these
executors with an `if architectures[0] == ...` branch. New family = new
adapter file under scripts/adapters/. See the never-branch rule in the
project memory at feedback_adapters_not_branches.md.

Each executor below:
  1. Reads the alloy source.architecture from ctx.alloy.
  2. Looks up the family adapter via scripts.adapters.resolve_family_adapter.
  3. Calls the adapter's matching method with the stage's config dict.
  4. Returns the mutated ctx.
"""

import sys
from pathlib import Path
from .base import StageExecutor, ForgeContext

# Resolve the adapter package one level up. Done at module load so the
# import error surfaces immediately, not on the first stage execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import resolve_family_adapter, DispatchError


def _resolve_family_for_ctx(ctx: ForgeContext, stage_label: str):
    """Look up the family adapter for the alloy currently in ctx.

    Raises DispatchError with a clear message if ctx has no alloy or the
    alloy lacks source.architecture — both indicate a wiring bug upstream
    in alloy_executor.execute_alloy that needs fixing, not a per-family
    branch added here.
    """
    alloy = getattr(ctx, "alloy", None) or {}
    source = alloy.get("source") or {}
    arch = source.get("architecture")
    if not arch:
        raise DispatchError(
            f"{stage_label}: ctx.alloy['source']['architecture'] is missing — cannot "
            f"resolve a family adapter. The alloy_executor must populate ctx.alloy "
            f"before invoking transform stages."
        )
    return resolve_family_adapter(arch)


class PruneExecutor(StageExecutor):
    """Dense head pruning — delegates to family_adapter.prune()."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        family = _resolve_family_for_ctx(ctx, "PruneExecutor")
        # Pass every stage param except 'type' as kwargs. The family adapter's
        # prune() method owns the param contract (level, strategy, plus any
        # adapter-specific extras passed through via **params).
        params = {k: v for k, v in self.config.items() if k != "type"}
        return family.prune(ctx, **params)


class TrainExecutor(StageExecutor):
    """Recovery / fine-tuning — delegates to family_adapter.train().

    Same executor handles both 'train' and 'lora' stages because LoRA is
    a training variant; the registry maps both stage types here. The family
    adapter's train() method decides whether to apply LoRA based on the
    stage params it receives.
    """

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        family = _resolve_family_for_ctx(ctx, "TrainExecutor")
        params = {k: v for k, v in self.config.items() if k != "type"}
        return family.train(ctx, **params)


class ExpertActivationProfileExecutor(StageExecutor):
    """§4.1.3.4 calibration-aware MoE expert importance profiling.

    Delegates to family_adapter.expert_activation_profile() which loads
    the calibration corpus, registers forward hooks on the family-specific
    router gate path, runs the corpus through inference, and writes the
    importance JSON to ctx.importance_json_path. The downstream
    expert-prune stage reads that path to make calibration-aware
    selection decisions.

    Without this executor registered, the alloy_executor's stage
    dispatch would silently SKIP the stage (logging a warning), and
    the expert-prune stage would loud-fail because importance_json_path
    is not set. This executor IS the wire that makes calibration-aware
    pruning the default §4.1.3.4 path.
    """

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        family = _resolve_family_for_ctx(ctx, "ExpertActivationProfileExecutor")
        params = {k: v for k, v in self.config.items() if k != "type"}
        return family.expert_activation_profile(ctx, **params)


class ExpertPruneExecutor(StageExecutor):
    """MoE expert pruning — delegates to family_adapter.expert_prune().

    The actual MoE expert removal lives in scripts/cpu_expert_prune_v2.py
    and is invoked from concrete MoE family adapters (Qwen3MoEAdapter,
    OlmoeAdapter, MixtralAdapter, etc.) — never from this executor directly.
    Each family handles its own module-tree layout (fused router vs unfused,
    block_sparse_moe vs mlp.experts naming, granite-fused vs deepseek-routed-shared).

    Dense families' expert_prune() raises NotImplementedError on the base —
    that's how dispatch catches alloy/family mismatches at the contract layer.
    """

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        family = _resolve_family_for_ctx(ctx, "ExpertPruneExecutor")
        params = {k: v for k, v in self.config.items() if k != "type"}
        return family.expert_prune(ctx, **params)
