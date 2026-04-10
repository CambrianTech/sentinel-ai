"""
test_vision_safety.py — CPU smoke test for vision_safety.py.

Validates:
    1. assert_vl_config rejects text-only configs (loud failure)
    2. assert_vl_config rejects configs with non-empty deepstack_visual_indexes
    3. assert_vl_config rejects configs missing vision token id fields
    4. collect_vision_token_indices returns the right four ids
    5. build_whitelist_from_config produces correct config-level whitelist
    6. assert_param_not_in_whitelist fails loudly on whitelisted params
    7. filter_target_modules excludes vision-side projections by name
    8. (when a real VL model is reachable) build_whitelist_from_model and
       verify_bit_exact_preservation work end-to-end

This script does NOT download a real VL model. Steps 1-7 are pure unit tests
against fabricated configs. Step 8 is gated on QWEN3_5_VL_PATH env var
pointing to a local checkpoint, and is skipped otherwise.

Usage::

    python test_vision_safety.py

To enable the optional integration test (downloads ~4GB+ if not cached)::

    QWEN3_5_VL_PATH=Qwen/Qwen2-VL-2B-Instruct python test_vision_safety.py

The integration test uses Qwen2-VL-2B because it shares the scatter-injection
pathway and module layout with Qwen3.5-VL but is small enough for a smoke
test. It will catch any divergence between our whitelist generator and the
real Qwen-VL module tree convention.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from transformers import PretrainedConfig

# Local import — same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_safety import (
    MROPE_CONFIG_FIELDS,
    VISION_TOKEN_ID_FIELDS,
    VISION_TOWER_PREFIX,
    VisionSafetyWhitelist,
    assert_param_not_in_whitelist,
    assert_vl_config,
    build_whitelist_from_config,
    collect_vision_token_indices,
    filter_target_modules,
)


def make_fake_vl_config(
    deepstack_indices: list[int] | None = None,
    omit_vision_token: str | None = None,
    omit_vision_config: bool = False,
    omit_deepstack_field: bool = False,
) -> PretrainedConfig:
    """Build a minimal fabricated config that mimics Qwen3.5-VL structure.

    Used to exercise the assertions without downloading a real model.
    Knobs let each test inject a specific failure mode.
    """
    cfg = PretrainedConfig()

    # Vision token ids — match the Qwen3.5-VL published values
    cfg.image_token_id = 248056
    cfg.video_token_id = 248057
    cfg.vision_start_token_id = 248053
    cfg.vision_end_token_id = 248054

    if omit_vision_token is not None:
        delattr(cfg, omit_vision_token)

    # M-RoPE config (top-level on Qwen3.5-VL)
    cfg.mrope_interleaved = True
    cfg.mrope_section = [11, 11, 10]
    cfg.partial_rotary_factor = 0.25
    cfg.rope_theta = 1e7

    # Vision config — fabricated SimpleNamespace mimicking the real one
    if not omit_vision_config:
        vc = SimpleNamespace()
        vc.depth = 27
        vc.hidden_size = 1152
        vc.intermediate_size = 4304
        vc.num_heads = 16
        vc.patch_size = 16
        if not omit_deepstack_field:
            vc.deepstack_visual_indexes = deepstack_indices if deepstack_indices is not None else []
        cfg.vision_config = vc

    return cfg


def test_assert_vl_config_rejects_text_only() -> None:
    """A text-only config (no vision_config) must fail loudly."""
    cfg = make_fake_vl_config(omit_vision_config=True)
    try:
        assert_vl_config(cfg)
    except AssertionError as e:
        assert "no vision_config" in str(e), f"wrong error message: {e}"
        print("[smoke] ✓ rejects text-only config (no vision_config)")
        return
    raise AssertionError("expected AssertionError on text-only config")


def test_assert_vl_config_rejects_nonempty_deepstack() -> None:
    """A config with deepstack_visual_indexes != [] must fail loudly."""
    cfg = make_fake_vl_config(deepstack_indices=[8, 16, 24])
    try:
        assert_vl_config(cfg)
    except AssertionError as e:
        assert "deepstack_visual_indexes is non-empty" in str(e), f"wrong error message: {e}"
        print("[smoke] ✓ rejects non-empty deepstack_visual_indexes")
        return
    raise AssertionError("expected AssertionError on non-empty deepstack")


def test_assert_vl_config_rejects_missing_deepstack_field() -> None:
    """A vision_config without the deepstack field at all must fail loudly
    (older Qwen2-VL or renamed schema)."""
    cfg = make_fake_vl_config(omit_deepstack_field=True)
    try:
        assert_vl_config(cfg)
    except AssertionError as e:
        assert "no deepstack_visual_indexes field" in str(e), f"wrong error message: {e}"
        print("[smoke] ✓ rejects vision_config without deepstack_visual_indexes field")
        return
    raise AssertionError("expected AssertionError on missing deepstack field")


def test_assert_vl_config_rejects_missing_vision_token() -> None:
    """A config missing any of the four vision token id fields must fail."""
    for field in VISION_TOKEN_ID_FIELDS:
        cfg = make_fake_vl_config(omit_vision_token=field)
        try:
            assert_vl_config(cfg)
        except AssertionError as e:
            assert f"missing required field {field!r}" in str(e), f"wrong error message: {e}"
            continue
        raise AssertionError(f"expected AssertionError on missing {field}")
    print(f"[smoke] ✓ rejects configs missing any of {len(VISION_TOKEN_ID_FIELDS)} vision token fields")


def test_collect_vision_token_indices() -> None:
    """The four published Qwen3.5-VL vision token ids must round-trip."""
    cfg = make_fake_vl_config()
    indices = collect_vision_token_indices(cfg)
    expected = frozenset({248053, 248054, 248056, 248057})
    assert indices == expected, f"expected {expected}, got {indices}"
    print(f"[smoke] ✓ collects vision token indices: {sorted(indices)}")


def test_build_whitelist_from_config() -> None:
    """Config-only whitelist has empty param names but populated indices/keys."""
    cfg = make_fake_vl_config()
    wl = build_whitelist_from_config(cfg)
    assert wl.untouchable_param_names == frozenset()
    assert wl.vision_tower_sha256 is None
    assert wl.merger_sha256 is None
    assert wl.untouchable_vocab_indices == frozenset({248053, 248054, 248056, 248057})
    # Config keys should include all four token id fields plus all four mrope fields
    expected_keys = set(VISION_TOKEN_ID_FIELDS) | set(MROPE_CONFIG_FIELDS)
    assert wl.untouchable_config_keys == frozenset(expected_keys), \
        f"expected {expected_keys}, got {wl.untouchable_config_keys}"
    print(f"[smoke] ✓ config-only whitelist: {len(wl.untouchable_config_keys)} config keys, "
          f"{len(wl.untouchable_vocab_indices)} vocab indices")


def test_assert_param_not_in_whitelist() -> None:
    """Passing a whitelisted param must raise; passing a free one must not."""
    fake_wl = VisionSafetyWhitelist(
        untouchable_param_names=frozenset({
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.merger.0.weight",
        }),
        untouchable_vocab_indices=frozenset({248053, 248054, 248056, 248057}),
        untouchable_config_keys=frozenset(),
        vision_tower_sha256="dummy",
        merger_sha256="dummy",
    )

    # Free param — should pass silently
    assert_param_not_in_whitelist(
        "model.layers.0.self_attn.q_proj.weight",
        fake_wl,
        operation="prune",
    )

    # Whitelisted param — must raise
    try:
        assert_param_not_in_whitelist(
            "model.visual.merger.0.weight",
            fake_wl,
            operation="prune",
        )
    except AssertionError as e:
        assert "refusing to prune" in str(e)
        print("[smoke] ✓ assert_param_not_in_whitelist fails loudly on vision params")
        return
    raise AssertionError("expected AssertionError on whitelisted param")


def test_filter_target_modules() -> None:
    """LoRA target patterns must exclude any vision-side modules with matching
    suffixes (e.g. vision tower attention's qkv)."""
    fake_wl = VisionSafetyWhitelist(
        untouchable_param_names=frozenset({
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.blocks.0.attn.proj.weight",
            "model.visual.merger.0.weight",
            "model.visual.merger.2.weight",
        }),
        untouchable_vocab_indices=frozenset(),
        untouchable_config_keys=frozenset(),
        vision_tower_sha256="dummy",
        merger_sha256="dummy",
    )

    # Simulated module list — text decoder has q_proj/k_proj/v_proj/o_proj at
    # multiple layers; vision tower has qkv/proj
    all_modules = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.self_attn.q_proj",
        "model.visual.blocks.0.attn.qkv",   # vision side, would match if pattern was 'qkv'
        "model.visual.blocks.0.attn.proj",  # vision side, would match if pattern was 'proj'
        "model.visual.merger.0",            # vision side merger
    ]

    # LoRA targets q/k/v/o_proj
    targets = filter_target_modules(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        whitelist=fake_wl,
        all_module_names=all_modules,
    )
    expected = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.self_attn.q_proj",
    ]
    assert targets == expected, f"expected {expected}, got {targets}"
    print(f"[smoke] ✓ filter_target_modules excludes vision-side modules: kept {len(targets)} text targets")

    # Now try a pattern that WOULD match vision side ('proj' suffix matches both
    # text o_proj and vision attn.proj). Vision side must still be excluded.
    targets_proj = filter_target_modules(
        target_modules=["proj"],
        whitelist=fake_wl,
        all_module_names=all_modules,
    )
    # All text *_proj modules should match; vision .attn.proj and .merger.0 should not
    # (note: 'merger.0' doesn't end in 'proj', so the suffix match excludes it naturally;
    # the vision attn.proj is whitelisted by name)
    assert "model.visual.blocks.0.attn.proj" not in targets_proj
    assert "model.layers.0.self_attn.q_proj" in targets_proj
    print(f"[smoke] ✓ filter_target_modules with risky 'proj' suffix still excludes vision attn.proj")


def test_optional_integration_with_real_vl_model() -> None:
    """Optional: if QWEN3_5_VL_PATH is set, load a real VL model and exercise
    the model-level whitelist building. Skipped otherwise."""
    path = os.environ.get("QWEN3_5_VL_PATH")
    if not path:
        print("[smoke] - skipping integration test (set QWEN3_5_VL_PATH to enable)")
        return

    print(f"[smoke] loading {path} for integration test...")
    from transformers import AutoModelForCausalLM
    from vision_safety import (
        build_whitelist_from_model,
        verify_bit_exact_preservation,
    )

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    wl = build_whitelist_from_model(model)
    print(f"[smoke] ✓ built whitelist from real VL model: "
          f"{len(wl.untouchable_param_names)} untouchable params, "
          f"vision_tower_sha256={wl.vision_tower_sha256[:16]}..., "
          f"merger_sha256={wl.merger_sha256[:16]}...")

    # Verify bit-exact preservation passes against the unmodified model
    verify_bit_exact_preservation(model, wl)
    print("[smoke] ✓ verify_bit_exact_preservation passes on unmodified model")

    # Sanity: every param under model.visual.* must be in the whitelist
    visual_params = {
        name for name, _p in model.named_parameters()
        if name.startswith(VISION_TOWER_PREFIX)
    }
    missing = visual_params - wl.untouchable_param_names
    if missing:
        raise AssertionError(
            f"vision params not whitelisted: {sorted(missing)[:5]}... "
            f"({len(missing)} total)"
        )
    print(f"[smoke] ✓ all {len(visual_params)} vision tower params are whitelisted")


def main() -> None:
    print("=" * 72)
    print("vision_safety smoke test")
    print("=" * 72)

    test_assert_vl_config_rejects_text_only()
    test_assert_vl_config_rejects_nonempty_deepstack()
    test_assert_vl_config_rejects_missing_deepstack_field()
    test_assert_vl_config_rejects_missing_vision_token()
    test_collect_vision_token_indices()
    test_build_whitelist_from_config()
    test_assert_param_not_in_whitelist()
    test_filter_target_modules()
    test_optional_integration_with_real_vl_model()

    print()
    print("=" * 72)
    print("vision_safety smoke test PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
