"""QwenVLAdapter — family adapter for Qwen2.5-VL / Qwen3.5-VL with vision-tower preservation.

Inherits from QwenDenseBase. Overrides prune() / train() / modality() to
consult scripts/vision_safety.py before any tensor walk that could touch
the vision tower or merger params. The whole point of this adapter is
that vision-tower preservation is NOT a per-stage flag — it is a
contractual property of the family. A VL forge that drops a vision tower
silently destroys the visual modality of the artifact, which is exactly
the brand-integrity gap the morning's audit caught for the legacy
Qwen3.5 catalog (8 published artifacts that lost vision capability that
could have been preserved bit-exact for free).

Tensor layout: Qwen2.5-VL is dense at the text-decoder layer with a
vision encoder + merger attached. The text-decoder uses the same
module-tree layout as Qwen3.5 dense (model.layers.{i}.{self_attn,mlp}.*),
so QwenDenseBase.prune + QwenDenseBase.train work for the text side
unchanged. The VISION-side parameters live under model.visual.* (the
vision tower) and the projector/merger lives under model.visual.merger.*
or similar; vision_safety.py enumerates these as the "untouchable
whitelist" the adapter consults before allowing any prune / LoRA / quant
operation to attach to a parameter.

Forge concerns this family supports:
    - prune (text-side only — vision-tower heads are filtered out)
    - train (recovery LoRA on text-side projections only — vision
      target_modules are filtered out via vision_safety.filter_target_modules)
    - modality (real method that handles the modality stage by asserting
      vision_config is present and the vision token ids match — refuses
      to forge a model whose VL config is broken)
    - context-extend (inherited from base — works for VL too)

Forge concerns this family does NOT support:
    - expert-prune (Qwen2.5-VL is dense, not MoE)
    - expert-activation-profile (MoE-only)

When Qwen3.5-VL ships (different release, same vision tower architecture
pattern), this adapter handles it without code changes — both architecture
strings are in the .architectures tuple.

Reproducibility contract: this adapter MUST stay frozen against any
published continuum-ai/* VL artifact (none today; future re-forges of
the legacy Qwen3.5 catalog with vision_safety integrated will be the
first VL artifacts in the catalog). Methodology improvements ship as
NEW adapters with NEW discriminators, never as edits to this file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .qwen_dense_base import QwenDenseBase
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


@register_family_adapter
class QwenVLAdapter(QwenDenseBase):
    """Family adapter for Qwen2.5-VL / Qwen3.5-VL — dense text decoder with
    vision tower preservation enforced via scripts/vision_safety.py."""

    architectures = ("qwen2_5_vl", "qwen3_5_vl")

    # ── prune ────────────────────────────────────────────────────────────────

    def prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Dense head pruning with vision-tower preservation.

        The text-decoder pruning logic is identical to QwenDenseBase.prune
        (the underlying forge_model.prune call walks model.layers.{i}.self_attn.*
        which is text-side only). The VL-specific addition is the
        vision_safety whitelist build BEFORE pruning + the
        verify_bit_exact_preservation check AFTER. If anything in the
        vision tower or merger changes, the post-prune assertion raises
        loudly and the forge halts — better than silently shipping a
        broken vision pathway.
        """
        self.log(f"Pruning with vision-tower preservation (Qwen VL family)")

        if ctx.model is None:
            self.log("  No model loaded — VL prune deferred (dispatch-only path)")
            return ctx

        # Lazy import — Tier 1 dispatch must work without torch / transformers.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vision_safety import (
            build_whitelist_from_model,
            verify_bit_exact_preservation,
        )

        # 1. Build the whitelist BEFORE pruning so the post-prune check
        #    can compare against the pre-prune sha256 of the vision tower
        #    and merger param subsets.
        whitelist = build_whitelist_from_model(ctx.model)
        self.log(
            f"  vision-safety whitelist: {len(whitelist.untouchable_param_names)} "
            f"untouchable params, {len(whitelist.untouchable_vocab_indices)} vocab indices"
        )

        # Stash for downstream stages that may need to consult it (train,
        # quant). Single source of truth — every stage in this alloy that
        # touches the model uses the SAME whitelist that was built
        # immediately after model load.
        ctx.vl_whitelist = whitelist

        # 2. Run the inherited dense prune. forge_model.prune walks the
        #    text-side attention modules; the whitelist is consulted by
        #    the post-prune verify step below to assert nothing on the
        #    vision side moved.
        ctx = super().prune(ctx, **params)

        # 3. Post-prune verification: vision tower + merger sha256 must
        #    still match the pre-prune values. If they don't, prune
        #    accidentally touched a vision-side param and the artifact
        #    is corrupted.
        verify_bit_exact_preservation(ctx.model, whitelist)
        self.log("  vision-safety post-prune verification PASSED — vision tower bit-exact")

        return ctx

    # ── train ────────────────────────────────────────────────────────────────

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """LoRA training with vision-tower preservation.

        Same dispatch shape as QwenDenseBase.train (recovery vs compensation
        based on the 'teacher' field), but the LoRA target_modules pass is
        filtered through vision_safety.filter_target_modules() to drop
        any vision-side projection that happens to share a name with a
        text-side one. Without this filter, a recovery LoRA could attach
        to model.visual.merger.linear_fc1 because its name suffix matches
        a text target_module like 'fc1', and merge_and_unload would then
        modify the vision tower.

        Post-train verification: same bit-exact check as prune.
        """
        # Tier 1 short-circuit — must come before any model-touching code.
        if ctx.model is None:
            self.log("  No model loaded — VL train deferred (dispatch-only path)")
            return ctx

        # Lazy import — Tier 1 dispatch must work without torch / transformers.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vision_safety import (
            build_whitelist_from_model,
            filter_target_modules,
            verify_bit_exact_preservation,
        )

        # Reuse the whitelist if a previous stage in this alloy already
        # built it (consistent provenance across stages); build fresh
        # otherwise.
        whitelist = getattr(ctx, "vl_whitelist", None)
        if whitelist is None:
            whitelist = build_whitelist_from_model(ctx.model)
            ctx.vl_whitelist = whitelist

        # If the alloy declared targetModules explicitly (compensation
        # distillation case), filter the list against the whitelist
        # before passing it through. The base class doesn't know about
        # filter_target_modules; we filter the params dict in place
        # before delegation.
        target_modules = params.get("targetModules")
        if target_modules:
            all_module_names = [name for name, _ in ctx.model.named_modules()]
            safe_targets = filter_target_modules(
                target_modules=target_modules,
                whitelist=whitelist,
                all_module_names=all_module_names,
            )
            dropped = set(target_modules) - set(safe_targets)
            if dropped:
                self.log(
                    f"  vision-safety filtered LoRA target_modules: dropped "
                    f"{len(dropped)} vision-side targets {sorted(dropped)[:5]}"
                )
            params = {**params, "targetModules": safe_targets}

        self.log(f"Training with vision-tower preservation (Qwen VL family)")

        # 2. Run the inherited dense train (recovery or compensation
        #    based on 'teacher' field).
        ctx = super().train(ctx, **params)

        # 3. Post-train verification: vision tower + merger bit-exact.
        verify_bit_exact_preservation(ctx.model, whitelist)
        self.log("  vision-safety post-train verification PASSED — vision tower bit-exact")

        return ctx

    # ── modality ─────────────────────────────────────────────────────────────

    def modality(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Modality stage handler for VL alloys.

        Param contract (from the alloy's modality stage):
            modality: 'vision' | 'audio' | 'multimodal'
            encoderModel: HF id or path of the vision encoder
                          (e.g. 'google/siglip-so400m-patch14-384' for the
                          standard Qwen2.5-VL setup)
            projectionArch: 'mlp' | 'cross-attention' | 'linear'
            freezeBase: bool
            freezeEncoder: bool
            trainingDataset: optional
            trainingSteps: optional
            projectionDim: optional

        For Qwen VL family models the vision encoder is ALREADY attached
        in the base model. The modality stage in a VL alloy is therefore
        a DECLARATION + INVARIANT CHECK, not an attach operation: it
        asserts vision_config is present, the vision token ids match the
        published config, and the deepstack injection setup is the
        validated single-level pattern. If anything is off, raises with
        a clear message.

        For a future case where the alloy declares a DIFFERENT encoder
        than the one already attached (e.g. swapping SigLIP for InternViT),
        the modality stage handler would detach + re-attach. That's not a
        case any published continuum-ai/* artifact uses today, so the
        body is just the invariant check.
        """
        modality_kind = params.get("modality", "vision")
        encoder = params.get("encoderModel", "<inherited>")
        self.log(f"modality {modality_kind} — encoder={encoder} (VL family invariant check)")

        if ctx.model is None:
            self.log("  No model loaded — modality check deferred (dispatch-only path)")
            return ctx

        # Lazy import — same pattern as prune / train.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vision_safety import assert_vl_config, build_whitelist_from_model

        # Hard preconditions: the model MUST already have vision_config
        # and the published vision token ids. assert_vl_config raises
        # loudly if any precondition fails.
        assert_vl_config(ctx.model.config)
        self.log("  vision_config preconditions PASSED")

        # Build the whitelist now if it hasn't been built already, so
        # downstream stages in the same alloy share it.
        if not hasattr(ctx, "vl_whitelist") or ctx.vl_whitelist is None:
            ctx.vl_whitelist = build_whitelist_from_model(ctx.model)
            self.log(
                f"  vision-safety whitelist built: "
                f"{len(ctx.vl_whitelist.untouchable_param_names)} untouchable params"
            )

        return ctx
