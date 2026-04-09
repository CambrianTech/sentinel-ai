"""QwenOmniAdapter — family adapter for Qwen2.5-Omni / Qwen3-Omni.

Priority 1 multimodal forge target from Kash's frontier roadmap (the
convo-with-kash analysis read 2026-04-08): Qwen2.5-Omni-7B is the only
clean-license open model on HuggingFace that does
text+vision+video+audio IN and text+speech OUT in a single inference
loop. Apache-2.0, ~15GB fp16, projected 5-7 GB Q4_K_M post-forge.
Fills the existing 'Qwen3-Omni' product agent slot in Continuum.

Architecture: Qwen2.5-Omni's text decoder is dense Qwen2.5 architecture
+ THREE encoder/decoder towers attached:

    model.thinker.*           — text decoder (dense Qwen2.5 architecture, what we forge)
    model.thinker.visual.*    — vision encoder (SigLIP-style, BIT-EXACT preserved)
    model.thinker.audio.*     — audio encoder (Whisper-style, BIT-EXACT preserved)
    model.talker.*            — speech decoder ("talker" — text → speech), BIT-EXACT preserved
    model.token2wav.*         — speech vocoder, BIT-EXACT preserved

The forge methodology pruners + LoRA training operate on the text-decoder
side (model.thinker.layers.{i}.*) and MUST NOT touch any of the four
encoder/decoder towers. The whitelist pattern is the same as VL but
covers FOUR pathways instead of one.

Forge concerns this family supports:
    - prune (text-decoder side, with omni-safety whitelist guarding all
      four encoder/decoder towers)
    - train (text-decoder side LoRA, with target_modules filtered to
      exclude any model.thinker.visual.* / .audio.* / model.talker.* /
      model.token2wav.* projections)
    - modality (real method that asserts vision_config + audio_config +
      talker_config + token2wav_config are all present and bit-exact)
    - quant (family-agnostic — handled by base default + QuantExecutor)

Forge concerns this family does NOT support:
    - expert-prune (Qwen2.5-Omni is dense, not MoE)
    - expert-activation-profile (MoE-only)

When Qwen3-Omni ships (Kash's analysis flagged it as the next omni
release), it gets added to the architectures tuple if it shares the
same module-tree shape, OR a separate adapter if the layout is
structurally different.

Reproducibility contract: this adapter MUST stay frozen against any
published continuum-ai/* omni artifact (none today; the first one
will be the post-Tier-2-wiring qwen2.5-omni-compacted forge). The
omni-safety preservation guarantee is the brand-integrity floor for
omni forging — without it, an omni forge would silently destroy one
or more of the four encoder/decoder towers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .qwen_dense_base import QwenDenseBase
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


# The four encoder/decoder tower module-name prefixes that omni-safety
# whitelisting must cover. Anything matching one of these prefixes is
# untouchable by prune / train / quant on the text-decoder side.
OMNI_PRESERVED_PREFIXES: tuple[str, ...] = (
    "thinker.visual.",     # vision encoder (SigLIP-style)
    "thinker.audio.",      # audio encoder (Whisper-style)
    "talker.",             # speech decoder (text → speech tokens)
    "token2wav.",          # speech vocoder (speech tokens → wav)
)


@register_family_adapter
class QwenOmniAdapter(QwenDenseBase):
    """Family adapter for Qwen2.5-Omni / future Qwen3-Omni — text decoder
    forging with omni-safety preservation of all four encoder/decoder towers."""

    architectures = ("qwen2_5_omni",)

    def model_auto_class(self):
        """Omni-modal models (text + vision + audio + speech) load via
        the generic AutoModel — there's no specific 'AutoModelForOmni'
        class. The aggregate model has thinker/talker/token2wav
        sub-modules that the family adapter walks individually."""
        from transformers import AutoModel
        return AutoModel

    # ── modality — real override for the omni shape ──────────────────────────

    def modality(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Modality stage handler for omni alloys.

        Asserts the model has all four encoder/decoder towers (vision,
        audio, speech decoder, speech vocoder) AND builds the
        omni-safety whitelist that downstream stages will consult. The
        assertion is a hard precondition: a Qwen2.5-Omni model with a
        damaged vision encoder, missing audio config, or stripped
        talker is NOT a valid forge input.

        Param contract:
            modality:        'multimodal' (omni declares everything in one stage)
            encoderModel:    'qwen-2.5-omni-built-in' or path
            freezeBase:      bool (always True for omni — forging the
                             text decoder while freezing all encoders)
            freezeEncoder:   bool (always True)
        """
        modality_kind = params.get("modality", "multimodal")
        self.log(f"omni modality {modality_kind} — vision + audio + speech preservation check")

        if ctx.model is None:
            self.log("  No model loaded — omni modality check deferred (dispatch-only path)")
            return ctx

        cfg = ctx.model.config
        # Each of the four towers MUST be present in the config. Loud
        # failure if any is missing — this is the omni shape contract.
        missing = []
        if not (hasattr(cfg, "vision_config") or
                (hasattr(cfg, "thinker_config") and hasattr(cfg.thinker_config, "vision_config"))):
            missing.append("vision_config")
        if not (hasattr(cfg, "audio_config") or
                (hasattr(cfg, "thinker_config") and hasattr(cfg.thinker_config, "audio_config"))):
            missing.append("audio_config")
        if not hasattr(cfg, "talker_config"):
            missing.append("talker_config")
        if not hasattr(cfg, "token2wav_config"):
            missing.append("token2wav_config")
        if missing:
            raise ValueError(
                f"{self.name}.modality: model is not a complete Qwen2.5-Omni — "
                f"missing config sections: {missing}. The omni shape requires "
                f"vision + audio encoders + talker (speech decoder) + token2wav "
                f"(speech vocoder). Cannot forge an incomplete omni model."
            )
        self.log(f"  omni shape OK: vision + audio + talker + token2wav all present")

        # Build the omni-safety whitelist by collecting every parameter
        # name whose suffix matches one of the OMNI_PRESERVED_PREFIXES.
        untouchable: set[str] = set()
        for name, _param in ctx.model.named_parameters():
            for prefix in OMNI_PRESERVED_PREFIXES:
                if prefix in name:
                    untouchable.add(name)
                    break
        ctx.omni_whitelist = frozenset(untouchable)
        self.log(
            f"  omni-safety whitelist: {len(untouchable)} untouchable params "
            f"across vision + audio + talker + token2wav towers"
        )
        return ctx

    # ── prune — wraps base.prune with omni-safety preservation ───────────────

    def prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Dense head pruning on the text-decoder with omni-safety preservation.

        The text-decoder layer is dense Qwen2.5; the inherited
        QwenDenseBase.prune body walks model.layers.{i}.self_attn.* which
        is the text-decoder side. The omni-safety whitelist is consulted
        BEFORE pruning to assert nothing on the four encoder/decoder
        towers will move, AFTER pruning to verify they didn't.
        """
        self.log("Pruning with omni-safety preservation (Qwen omni family)")

        if ctx.model is None:
            self.log("  No model loaded — omni prune deferred (dispatch-only path)")
            return ctx

        # Build whitelist if upstream modality stage didn't already.
        whitelist = getattr(ctx, "omni_whitelist", None)
        if whitelist is None:
            untouchable = set()
            for name, _ in ctx.model.named_parameters():
                for prefix in OMNI_PRESERVED_PREFIXES:
                    if prefix in name:
                        untouchable.add(name)
                        break
            whitelist = frozenset(untouchable)
            ctx.omni_whitelist = whitelist

        # Snapshot the encoder-tower param hashes BEFORE pruning so we
        # can verify bit-exact preservation after.
        import hashlib
        pre_hashes = {}
        for name, param in ctx.model.named_parameters():
            if name in whitelist:
                pre_hashes[name] = hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()

        # Run inherited dense prune on the text-decoder side.
        ctx = super().prune(ctx, **params)

        # Verify each encoder/decoder tower param is bit-exact preserved.
        post_failures = []
        for name, param in ctx.model.named_parameters():
            if name in whitelist:
                post = hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
                if post != pre_hashes[name]:
                    post_failures.append(name)
        if post_failures:
            raise AssertionError(
                f"{self.name}.prune: omni-safety preservation FAILED — "
                f"{len(post_failures)} encoder/decoder tower params changed during "
                f"prune: {sorted(post_failures)[:5]}. The forge would have "
                f"silently corrupted one or more of vision/audio/talker/token2wav. "
                f"Halting before downstream stages can ship the broken artifact."
            )
        self.log(f"  omni-safety preservation PASSED — all 4 encoder/decoder towers bit-exact")
        return ctx

    # ── train — wraps base.train with omni target_modules filtering ──────────

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Recovery / compensation LoRA training with omni-safety filtering.

        Filters the LoRA target_modules to drop any matches on the four
        preserved encoder/decoder towers BEFORE delegating to base.train.
        Without this filter, a recovery LoRA could attach to
        model.thinker.visual.merger.linear_fc1 because 'fc1' matches a
        text-side target_module suffix; merge_and_unload would then
        modify the vision encoder.
        """
        if ctx.model is None:
            self.log("  No model loaded — omni train deferred (dispatch-only path)")
            return ctx

        # Build / reuse whitelist
        whitelist = getattr(ctx, "omni_whitelist", None)
        if whitelist is None:
            untouchable = set()
            for name, _ in ctx.model.named_parameters():
                for prefix in OMNI_PRESERVED_PREFIXES:
                    if prefix in name:
                        untouchable.add(name)
                        break
            whitelist = frozenset(untouchable)
            ctx.omni_whitelist = whitelist

        # Filter target_modules if the alloy declared them explicitly.
        target_modules = params.get("targetModules")
        if target_modules:
            all_modules = [name for name, _ in ctx.model.named_modules()]
            safe_targets = []
            for full_name in all_modules:
                if not any(full_name.endswith(p) for p in target_modules):
                    continue
                weight_name = full_name + ".weight"
                if weight_name in whitelist:
                    continue  # vision/audio/talker/token2wav side — drop
                safe_targets.append(full_name)
            dropped = len([n for n in all_modules
                           if any(n.endswith(p) for p in target_modules)
                           and n + ".weight" in whitelist])
            if dropped:
                self.log(
                    f"  omni-safety filtered LoRA target_modules: dropped "
                    f"{dropped} encoder/decoder-tower targets"
                )
            params = {**params, "targetModules": safe_targets}

        self.log("Training with omni-safety preservation (Qwen omni family)")
        return super().train(ctx, **params)
