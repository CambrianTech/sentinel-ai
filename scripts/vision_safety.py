"""
vision_safety.py — VL whitelist generator for the forge pipeline.

Pure read-only module. Given a Qwen3.5-VL (or compatible VL) model or config,
produces the set of "untouchable" parameter names, embedding vocab indices,
and config keys that the forge pipeline must preserve bit-exact when forging
a vision-language model.

Consumed by:
    - compensation_lora_vl.py — to filter LoRA target_modules
    - cpu_expert_prune_vl.py — to filter expert-prune regex matches
    - forge_model.py (after Phase 4 refactor) — to filter prune/defrag targets

Design doc: docs/VL-FORGE-DESIGN.md

Hard preconditions enforced (per the no-fallback discipline):
    1. config has vision_config (else this is not a VL model — caller is wrong)
    2. config.vision_config.deepstack_visual_indexes is empty
       (non-empty multi-level injection is not validated)
    3. all five vision token ids are present in config
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig, PreTrainedModel


# Vision token id field names that must be present on a Qwen3.5-VL config.
# Order matters — these are exact field names from the published config.json.
VISION_TOKEN_ID_FIELDS: tuple[str, ...] = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)

# M-RoPE config fields that encode spatial structure for vision grounding.
# These must match the source model exactly in the saved config.json after
# any forge stage. Any tool that rewrites RoPE fails the post-forge validation.
MROPE_CONFIG_FIELDS: tuple[str, ...] = (
    "mrope_interleaved",
    "mrope_section",
    "partial_rotary_factor",
    "rope_theta",
)

# Module path prefix for the vision tower in Qwen-VL family models.
# Convention: model.visual.* covers patch_embed, blocks[i], merger, etc.
VISION_TOWER_PREFIX: str = "model.visual."

# The merger / projector module under the vision tower.
# Single point of failure between modalities.
MERGER_MODULE_NAME: str = "model.visual.merger"


@dataclass(frozen=True)
class VisionSafetyWhitelist:
    """Immutable record of every untouchable element of a VL model.

    Consumed by every forge stage to filter prune/LoRA/defrag targets.
    """

    # Parameter names (full dotted path) that must not be modified.
    # Includes vision tower, merger, and any other vision-side parameters.
    untouchable_param_names: frozenset[str]

    # Embedding row indices that must survive vocab defrag bit-exact.
    # Includes the four vision token ids plus any tokenizer special tokens.
    untouchable_vocab_indices: frozenset[int]

    # Config keys (top-level or under text_config / vision_config) whose
    # values must match the source model exactly in the saved config.json.
    untouchable_config_keys: frozenset[str]

    # SHA256 over the source vision tower state dict, for post-forge bit-exact
    # verification. None if the whitelist was generated from config alone.
    vision_tower_sha256: str | None

    # SHA256 over the source merger state dict.
    merger_sha256: str | None


def assert_vl_config(config: PretrainedConfig) -> None:
    """Hard precondition: config must be a recognizable VL config.

    Raises a loud error rather than silently treating it as text-only.
    """
    if not hasattr(config, "vision_config"):
        raise AssertionError(
            f"vision_safety: config has no vision_config; this is not a VL model. "
            f"Use the text-only forge path. config class: {type(config).__name__}"
        )

    vc = config.vision_config
    if not hasattr(vc, "deepstack_visual_indexes"):
        raise AssertionError(
            f"vision_safety: vision_config has no deepstack_visual_indexes field. "
            f"Either this is an older Qwen2-VL config (not validated against this "
            f"forge), or the field has been renamed. Inspect the source config.json "
            f"and update VISION_TOKEN_ID_FIELDS / MROPE_CONFIG_FIELDS in vision_safety.py."
        )

    deepstack = vc.deepstack_visual_indexes
    if deepstack:
        raise AssertionError(
            f"vision_safety: deepstack_visual_indexes is non-empty: {deepstack}. "
            f"Multi-level visual injection into the text decoder is not validated "
            f"against this forge. Layer-deletion safety is unverified for this "
            f"checkpoint. Refusing to proceed; update the design doc and re-validate "
            f"before forging this model."
        )

    for field in VISION_TOKEN_ID_FIELDS:
        if not hasattr(config, field):
            raise AssertionError(
                f"vision_safety: config missing required field {field!r}. "
                f"This may not be a Qwen3.5-VL family config. Inspect config.json "
                f"and update VISION_TOKEN_ID_FIELDS in vision_safety.py if the "
                f"field has been renamed."
            )


def collect_vision_token_indices(config: PretrainedConfig) -> frozenset[int]:
    """Extract the vision token vocab indices from a VL config.

    Returns the union of image_token_id, video_token_id, vision_start_token_id,
    vision_end_token_id. Eos and other special tokens are NOT included here —
    those come from the tokenizer's special_tokens_map at calibration time.
    """
    indices: set[int] = set()
    for field in VISION_TOKEN_ID_FIELDS:
        value = getattr(config, field)
        if not isinstance(value, int):
            raise AssertionError(
                f"vision_safety: config.{field} is not an int (got {type(value).__name__}). "
                f"Vision token ids must be integers."
            )
        indices.add(value)
    return frozenset(indices)


def collect_untouchable_param_names(model: PreTrainedModel) -> frozenset[str]:
    """Walk the model module tree and collect all parameter names under the
    vision tower (including the merger). Pure read; no model modification.

    Returns full dotted parameter names matching what model.named_parameters()
    yields, so the caller can compare directly.
    """
    names: set[str] = set()
    for name, _param in model.named_parameters():
        if name.startswith(VISION_TOWER_PREFIX):
            names.add(name)
    if not names:
        raise AssertionError(
            f"vision_safety: no parameters found under prefix {VISION_TOWER_PREFIX!r}. "
            f"Either the model has no vision tower, or the Qwen-VL convention has "
            f"changed and the vision tower lives elsewhere. Inspect "
            f"model.named_parameters() and update VISION_TOWER_PREFIX in vision_safety.py."
        )
    return frozenset(names)


def hash_param_subset(model: PreTrainedModel, param_names: frozenset[str]) -> str:
    """SHA256 over a deterministic ordering of the named parameters' raw bytes.

    Used to verify bit-exact preservation of the vision tower / merger across
    forge stages. The hash covers tensor data only, not metadata.
    """
    h = hashlib.sha256()
    for name in sorted(param_names):
        param = dict(model.named_parameters())[name]
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        # Use contiguous bytes; .cpu() to ensure host-side hashing
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def collect_merger_param_names(model: PreTrainedModel) -> frozenset[str]:
    """Just the merger MLP parameter names. Subset of vision-tower untouchables,
    surfaced separately because Phase 2's gradient-flow check needs to assert
    the merger is frozen but receives gradient through it."""
    merger_prefix = MERGER_MODULE_NAME + "."
    names = {
        name for name, _p in model.named_parameters()
        if name.startswith(merger_prefix)
    }
    if not names:
        raise AssertionError(
            f"vision_safety: no merger parameters found under {merger_prefix!r}. "
            f"Either the merger lives at a different attribute path on this "
            f"checkpoint, or the model class is non-standard. Inspect "
            f"model.visual.named_modules() and update MERGER_MODULE_NAME."
        )
    return frozenset(names)


def build_whitelist_from_model(model: PreTrainedModel) -> VisionSafetyWhitelist:
    """Full whitelist construction from a loaded VL model.

    Computes SHA256 over the vision tower and merger so post-forge stages
    can assert bit-exact preservation. Use build_whitelist_from_config()
    if you only have the config (e.g. for unit tests without the weights).
    """
    config = model.config
    assert_vl_config(config)

    untouchable_params = collect_untouchable_param_names(model)
    merger_params = collect_merger_param_names(model)
    vocab_indices = collect_vision_token_indices(config)

    config_keys: set[str] = set(VISION_TOKEN_ID_FIELDS)
    for field in MROPE_CONFIG_FIELDS:
        if hasattr(config, field):
            config_keys.add(field)

    return VisionSafetyWhitelist(
        untouchable_param_names=untouchable_params,
        untouchable_vocab_indices=vocab_indices,
        untouchable_config_keys=frozenset(config_keys),
        vision_tower_sha256=hash_param_subset(model, untouchable_params),
        merger_sha256=hash_param_subset(model, merger_params),
    )


def build_whitelist_from_config(config: PretrainedConfig) -> VisionSafetyWhitelist:
    """Whitelist construction from config alone, without loaded weights.

    Used by unit tests and dry-run validators. SHA256 fields are None;
    untouchable_param_names is empty (cannot enumerate without the model).
    The vocab indices and config keys are still populated.
    """
    assert_vl_config(config)
    vocab_indices = collect_vision_token_indices(config)

    config_keys: set[str] = set(VISION_TOKEN_ID_FIELDS)
    for field in MROPE_CONFIG_FIELDS:
        if hasattr(config, field):
            config_keys.add(field)

    return VisionSafetyWhitelist(
        untouchable_param_names=frozenset(),
        untouchable_vocab_indices=vocab_indices,
        untouchable_config_keys=frozenset(config_keys),
        vision_tower_sha256=None,
        merger_sha256=None,
    )


def assert_param_not_in_whitelist(
    param_name: str,
    whitelist: VisionSafetyWhitelist,
    operation: str,
) -> None:
    """Assert a parameter is safe to modify. Call this before any prune/LoRA
    target is added. Operation is a human-readable label for the error message.
    """
    if param_name in whitelist.untouchable_param_names:
        raise AssertionError(
            f"vision_safety: refusing to {operation} parameter {param_name!r} — "
            f"it is in the vision-safety whitelist (vision tower / merger). "
            f"Modifying it would break the vision pathway."
        )


def filter_target_modules(
    target_modules: list[str],
    whitelist: VisionSafetyWhitelist,
    all_module_names: list[str],
) -> list[str]:
    """Given a LoRA target_modules pattern list (e.g. ['q_proj', 'k_proj']),
    return the subset of full module names that match the patterns AND are
    NOT in the vision-safety whitelist.

    Used by compensation_lora_vl.py to ensure LoRA never targets vision-side
    projections that happen to share a name with text-side ones.
    """
    safe_targets: list[str] = []
    for full_name in all_module_names:
        # Match if any pattern is a suffix of the module name (PEFT semantics)
        if not any(full_name.endswith(pattern) for pattern in target_modules):
            continue
        # Check the corresponding weight parameter is not whitelisted
        weight_name = full_name + ".weight"
        if weight_name in whitelist.untouchable_param_names:
            continue
        safe_targets.append(full_name)
    return safe_targets


def verify_bit_exact_preservation(
    model: PreTrainedModel,
    whitelist: VisionSafetyWhitelist,
) -> None:
    """Post-forge assertion: the vision tower and merger have not changed.

    Computes the current SHA256 over the same parameter set as the original
    whitelist and asserts equality. Call this after every forge stage.
    """
    if whitelist.vision_tower_sha256 is None:
        raise AssertionError(
            "vision_safety: cannot verify bit-exact preservation against a "
            "config-only whitelist. Build the whitelist from the loaded model "
            "before any modification."
        )

    current_vision_hash = hash_param_subset(model, whitelist.untouchable_param_names)
    if current_vision_hash != whitelist.vision_tower_sha256:
        raise AssertionError(
            f"vision_safety: vision tower hash changed after forge stage. "
            f"Expected {whitelist.vision_tower_sha256}, got {current_vision_hash}. "
            f"Some forge operation modified parameters under {VISION_TOWER_PREFIX}. "
            f"This is a bug — the vision pathway has been corrupted."
        )

    merger_param_names = collect_merger_param_names(model)
    current_merger_hash = hash_param_subset(model, merger_param_names)
    if current_merger_hash != whitelist.merger_sha256:
        raise AssertionError(
            f"vision_safety: merger hash changed after forge stage. "
            f"Expected {whitelist.merger_sha256}, got {current_merger_hash}. "
            f"The merger MLP was modified. This severs the vision pathway."
        )
