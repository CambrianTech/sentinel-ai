"""Input stages — front of the pipeline.

These configure what the model will be capable of:
- SourceConfig: context window, modalities, target devices
- ContextExtend: RoPE rescaling for longer context
- Modality: bolt-on vision/audio/video encoders
"""

from .base import StageExecutor, ForgeContext


class SourceConfigExecutor(StageExecutor):
    """Reads source-config stage and configures the forge context."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        context_length = self.config.get("contextLength")
        modalities = self.config.get("inputModalities", ["text"])
        devices = self.config.get("targetDevices", [])

        self.log(f"Context: {context_length or 'default'}, Modalities: {modalities}, Devices: {devices}")

        ctx.source_config = {
            "contextLength": context_length,
            "inputModalities": modalities,
            "targetDevices": devices,
        }
        return ctx


class ContextExtendExecutor(StageExecutor):
    """Extends context window via RoPE rescaling."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        target = self.config.get("targetLength", 32768)
        method = self.config.get("method", "yarn")
        self.log(f"Extending context to {target} tokens via {method}")

        if ctx.model is None:
            self.log("WARNING: No model loaded — context extension deferred")
            return ctx

        config = ctx.model.config
        # Use original_max_position_embeddings if available (the actual training context),
        # NOT max_position_embeddings (which can be the theoretical maximum, e.g., 262K).
        # Qwen3.5-4B has max_position_embeddings=262144 but was trained on ~32K.
        # Using 262K gives factor=0.5 which COMPRESSES positions and breaks the model.
        rope_cfg = getattr(config, 'rope_scaling', None) or {}
        original_ctx = (
            rope_cfg.get('original_max_position_embeddings') or
            getattr(config, 'original_max_position_embeddings', None) or
            getattr(config, 'max_position_embeddings', 4096)
        )
        # Sanity: if original_ctx is suspiciously large (>64K), it's probably the theoretical
        # max, not training context. Default to 32K for Qwen models.
        if original_ctx > 65536:
            self.log(f"WARNING: original context {original_ctx} seems too large, using 32768")
            original_ctx = 32768
        factor = target / original_ctx

        # Validate: factor must be >= 1.0 (extending, not compressing)
        if factor < 1.0:
            self.log(f"ERROR: factor {factor:.2f}x would COMPRESS context ({original_ctx} → {target}). Aborting.")
            raise ValueError(f"Context extend factor {factor:.2f} < 1.0 — would compress, not extend")
        if factor > 16.0:
            self.log(f"WARNING: factor {factor:.1f}x is very aggressive (>16x). Quality may degrade.")

        scaling_type = {
            "yarn": "yarn",
            "ntk": "dynamic",
            "dynamic-ntk": "dynamic",
            "linear": "linear",
        }.get(method, "yarn")

        # Qwen3.5 uses rope_parameters with 'rope_type' key
        # Other models use rope_scaling with 'type' key
        # Set both to be safe
        rope_cfg_new = {"type": scaling_type, "factor": factor,
                        "original_max_position_embeddings": original_ctx}
        if hasattr(config, "rope_parameters"):
            # Qwen3.5 format: merge into existing rope_parameters
            existing = getattr(config, "rope_parameters", {}) or {}
            existing["rope_type"] = scaling_type
            existing["factor"] = factor
            existing["original_max_position_embeddings"] = original_ctx
            config.rope_parameters = existing
            self.log(f"Applied {method} via rope_parameters: {factor:.1f}x ({original_ctx} → {target})")
        if hasattr(config, "rope_scaling"):
            config.rope_scaling = {**rope_cfg_new}
            if not hasattr(config, "rope_parameters"):
                self.log(f"Applied {method} via rope_scaling: {factor:.1f}x ({original_ctx} → {target})")
        if not hasattr(config, "rope_scaling") and not hasattr(config, "rope_parameters"):
            self.log(f"WARNING: model has no rope_scaling or rope_parameters — extension may not work")

        config.max_position_embeddings = target
        return ctx


class ModalityExecutor(StageExecutor):
    """Adds vision/audio/video encoder to the base model.

    This is the highest-impact stage — gives a text model new senses.
    Implementation phases:
    1. Record intent (now) — the alloy captures what should happen
    2. Download encoder + create projection (next)
    3. Train projection on modality dataset (next)
    4. Wire into base model forward pass (next)
    """

    # Known encoder models per modality
    RECOMMENDED_ENCODERS = {
        "vision": {
            "encoder": "openai/clip-vit-large-patch14",
            "datasets": ["liuhaotian/LLaVA-Instruct-150K", "ShareGPT4V/ShareGPT4V"],
        },
        "audio": {
            "encoder": "openai/whisper-large-v3",
            "datasets": ["librispeech_asr", "mozilla-foundation/common_voice_17_0"],
        },
        "video": {
            "encoder": "openai/clip-vit-large-patch14",  # Frame-based initially
            "datasets": ["MBZUAI/VideoInstruct-100K"],
        },
    }

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        modality = self.config.get("modality", "vision")
        encoder = self.config.get("encoderModel", "")
        proj_arch = self.config.get("projectionArch", "mlp")
        freeze_base = self.config.get("freezeBase", True)
        freeze_encoder = self.config.get("freezeEncoder", True)
        training_steps = self.config.get("trainingSteps", 5000)
        dataset = self.config.get("trainingDataset", "")

        # Auto-recommend encoder and dataset if not specified
        if not encoder and modality in self.RECOMMENDED_ENCODERS:
            encoder = self.RECOMMENDED_ENCODERS[modality]["encoder"]
            self.log(f"Auto-selected encoder: {encoder}")
        if not dataset and modality in self.RECOMMENDED_ENCODERS:
            dataset = self.RECOMMENDED_ENCODERS[modality]["datasets"][0]
            self.log(f"Auto-selected dataset: {dataset}")

        self.log(f"Adding {modality} via {encoder}")
        self.log(f"  Projection: {proj_arch}, Steps: {training_steps}")
        self.log(f"  Freeze base: {freeze_base}, Freeze encoder: {freeze_encoder}")
        self.log(f"  Dataset: {dataset}")

        # TODO: Phase 2 implementation
        # 1. from transformers import CLIPModel, WhisperModel
        # 2. encoder_model = CLIPModel.from_pretrained(encoder)
        # 3. projection = nn.Linear(encoder_dim, model_dim) or MLP
        # 4. Train projection on dataset with frozen base + encoder
        # 5. Save combined model

        self.log(f"RECORDING INTENT — modality attachment not yet implemented")
        self.log(f"  Issue: CambrianTech/sentinel-ai#120")

        return ctx
