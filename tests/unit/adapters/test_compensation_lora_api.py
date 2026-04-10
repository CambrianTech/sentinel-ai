"""TDD spec for the compensation_lora.py importable API.

Roadmap step 3.5 from docs/PLUGIN-SPRINT.md: refactor compensation_lora.py
to expose a callable Python function alongside its existing CLI wrapper,
so QwenDenseBase._train_compensation can wire to it via lazy import
without subprocess shells, NotImplementedError stubs, or any other
contract-breaking surface.

Written test-first per TDD/TDValidation discipline. The contract this
test asserts IS the spec the refactor must satisfy. If any assertion
fails, the refactor is wrong.

These tests do NOT load any model — they verify only the function
signature, docstring presence, and structured-error behavior. The actual
distillation execution requires a 5090 + a real teacher and is
exercised at the Tier 2 reproducibility level (separate test, separate
run).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── compensate_lora — adapter entry point (caller provides loaded student) ──


def test_compensate_lora_is_importable():
    """The function MUST be importable from the script module."""
    from compensation_lora import compensate_lora
    assert callable(compensate_lora)


def test_compensate_lora_signature():
    """compensate_lora MUST accept the kwargs the family-adapter wiring needs.

    Required kwargs:
        student              — pre-loaded student model object
        student_tokenizer    — pre-loaded tokenizer
        teacher_path         — path or HF id of unmodified teacher (this
                               function loads the teacher itself in the
                               requested quant; the adapter does NOT preload
                               the teacher because it's GB-class and only
                               needed for the compensation step)
        teacher_quant        — '8bit' | '4bit'
        calibration_data     — JSONL path
        output               — output dir for the compensated student
        steps                — int
        lora_rank            — int
        lora_alpha           — int
        learning_rate        — float
        loss_type            — 'mse_hidden' | 'kl_logits' | 'both'
        target_modules       — list[str]
        max_length           — int

    All kwargs are keyword-only so callers can't accidentally swap
    positional argument order.
    """
    from compensation_lora import compensate_lora
    sig = inspect.signature(compensate_lora)
    required = {
        "student", "student_tokenizer", "teacher_path", "teacher_quant",
        "calibration_data", "output", "steps", "lora_rank", "lora_alpha",
        "learning_rate", "loss_type", "target_modules", "max_length",
    }
    actual = set(sig.parameters.keys())
    missing = required - actual
    assert not missing, f"compensate_lora missing required kwargs: {sorted(missing)}"
    # All parameters MUST be keyword-only — no positional accidents.
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"compensate_lora parameter {name!r} is {param.kind.name}, "
            f"must be KEYWORD_ONLY"
        )


def test_compensate_lora_raises_on_missing_calibration_corpus():
    """Loud failure when the calibration corpus path doesn't exist.
    The §4.1.3.4.1 discipline gate requires the calibration corpus to
    be present and hash-pinned; a missing file is a contract violation,
    not a thing to silently route around."""
    from compensation_lora import compensate_lora
    with pytest.raises((FileNotFoundError, ValueError)) as exc_info:
        compensate_lora(
            student=object(),  # dummy — function should fail before touching it
            student_tokenizer=object(),
            teacher_path="dummy",
            teacher_quant="8bit",
            calibration_data="/tmp/does-not-exist.jsonl",
            output="/tmp/out",
            steps=1,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=1e-4,
            loss_type="kl_logits",
            target_modules=["q_proj"],
            max_length=1024,
        )
    assert "calibration" in str(exc_info.value).lower() or "exist" in str(exc_info.value).lower(), (
        f"error message should mention the missing calibration corpus, got: {exc_info.value}"
    )


def test_compensate_lora_raises_on_invalid_loss_type():
    """Loss type must be one of the canonical values. Anything else is a
    typo / contract bug and should fail loudly at the entry point."""
    from compensation_lora import compensate_lora
    with pytest.raises(ValueError) as exc_info:
        compensate_lora(
            student=object(),
            student_tokenizer=object(),
            teacher_path="dummy",
            teacher_quant="8bit",
            calibration_data="/tmp/whatever.jsonl",
            output="/tmp/out",
            steps=1,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=1e-4,
            loss_type="not_a_real_loss",
            target_modules=["q_proj"],
            max_length=1024,
        )
    assert "loss_type" in str(exc_info.value) or "loss" in str(exc_info.value).lower(), (
        f"error message should name the loss_type field, got: {exc_info.value}"
    )


# ── compensate_lora_from_paths — CLI entry point ────────────────────────────


def test_compensate_lora_from_paths_is_importable():
    """The path-based entry point MUST be importable for the CLI wrapper."""
    from compensation_lora import compensate_lora_from_paths
    assert callable(compensate_lora_from_paths)


def test_compensate_lora_from_paths_signature():
    """The path-based entry point MUST accept the same param contract as
    the adapter-based one, but with teacher_path / student_path / student_quant
    in place of pre-loaded objects."""
    from compensation_lora import compensate_lora_from_paths
    sig = inspect.signature(compensate_lora_from_paths)
    required = {
        "teacher_path", "student_path", "student_quant",
        "calibration_data", "output", "steps", "lora_rank", "lora_alpha",
        "learning_rate", "loss_type", "target_modules", "max_length",
        "teacher_quant",
    }
    actual = set(sig.parameters.keys())
    missing = required - actual
    assert not missing, f"compensate_lora_from_paths missing required kwargs: {sorted(missing)}"


# ── CLI wrapper still exists and is callable ────────────────────────────────


def test_main_still_exists():
    """The CLI wrapper main() MUST still exist so existing forge pipelines
    that invoke `python scripts/compensation_lora.py ...` keep working.
    The refactor extracts the body into compensate_lora_from_paths but
    leaves the argparse + main() as the thin wrapper."""
    from compensation_lora import main
    assert callable(main)


# ── QwenDenseBase wiring ────────────────────────────────────────────────────


def test_qwen_dense_base_train_compensation_calls_real_function():
    """QwenDenseBase._train_compensation MUST call compensation_lora.compensate_lora
    when ctx.model is non-None — not raise NotImplementedError, not log a warning,
    not subprocess-shell-out, not return ctx unchanged.

    The dispatch path is: QwenDenseBase.train(ctx, **params) sees a 'teacher'
    field, calls _train_compensation(ctx, **params), which lazy-imports
    compensate_lora and calls it.
    """
    from adapters.qwen_dense_base import QwenDenseBase
    import inspect as _inspect
    src = _inspect.getsource(QwenDenseBase._train_compensation)
    # The body MUST contain the import of the real function and a call to it.
    assert "from compensation_lora import compensate_lora" in src or \
           "compensation_lora" in src and "compensate_lora" in src, (
        "QwenDenseBase._train_compensation must lazy-import compensation_lora.compensate_lora"
    )
    # The body MUST NOT contain a NotImplementedError raise — that's the
    # stub state that this commit is supposed to retire.
    assert "raise NotImplementedError" not in src, (
        "QwenDenseBase._train_compensation must NOT raise NotImplementedError. "
        "The Tier 2 wiring is required, not deferred."
    )


def test_qwen_dense_base_train_compensation_short_circuits_on_dispatch_path():
    """When ctx.model is None (dispatch-only / Tier 1 path), the method
    MUST short-circuit cleanly and return ctx without invoking the real
    compensation function. This is what makes the Tier 1 dispatch test
    Mac-safe — torch + compensation_lora never get imported in the
    dispatch-only path."""
    from adapters.qwen2_dense import Qwen2DenseAdapter
    from dataclasses import dataclass

    @dataclass
    class _MockCtx:
        model: object = None
        tokenizer: object = None
        output_dir: Path = Path("/tmp/test")
        alloy: dict = None

    adapter = Qwen2DenseAdapter()
    ctx = _MockCtx(alloy={"source": {"architecture": "qwen2"}})

    # Should NOT raise. Should return the ctx.
    result = adapter.train(
        ctx,
        teacher="dummy/teacher",
        loraRank=16,
        loraAlpha=32,
        kdTemperature=2.0,
        lossType="kl_logits",
        steps=10,
        learningRate="1e-4",
        domain="code",
    )
    assert result is ctx, "dispatch-only path must return ctx unchanged"
