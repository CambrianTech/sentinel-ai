"""TDD spec for the Open VLM Leaderboard runner pack.

Open VLM Leaderboard scores vision-language models against a standard
suite via `lmms-eval` (the vision-aware fork of `lm-eval-harness`).
The four headline benchmarks every modern VL forge reports against:

    mmmu_val            — Massive Multi-discipline Multimodal Understanding
    chartqa             — visual QA on charts/graphs
    docvqa_val          — document visual QA (OCR + reasoning)
    ai2d                — Allen Institute Diagrams (science diagrams)

All four route through lmms-eval, so the right architecture is ONE
base class (LmmsEvalHarnessRunner) that takes the harness task name +
metric extractor, plus four thin subclasses that just declare the
task name and metric key. Mirror image of the Open LLM Leaderboard v2
pack, except the harness module is `lmms_eval` instead of `lm_eval`.

Key OOP move: LmmsEvalHarnessRunner INHERITS from LmEvalHarnessRunner.
The score() method (parses results JSON) is identical because lmms-eval
emits the same results format. Only evaluate() is overridden to call
lmms_eval.simple_evaluate instead of lm_eval.simple_evaluate. Zero
duplicated body — same OOP rule the family adapters use.

Critical: 4 of the 12 alloys in the seed catalog are VL forges
(qwen3-vl-8b, qwen3-vl-30b-a3b, qwen2-5-vl-7b, qwen2-5-vl-32b). If
these runners stay stubs, those 4 alloys loud-fail at the eval stage
and land in rework/. Graduating them is a hard prerequisite for
shipping any VL forge.

Written test-first per TDD discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Base class — inherits from LmEvalHarnessRunner ──────────────────────────


def test_lmms_eval_harness_base_importable():
    from eval_runners.lmms_eval_harness_base import LmmsEvalHarnessRunner
    from eval_runners.lm_eval_harness_base import LmEvalHarnessRunner
    assert issubclass(LmmsEvalHarnessRunner, LmEvalHarnessRunner)


def test_lmms_eval_inherits_score_from_lm_eval_base():
    """The score() method MUST be inherited from LmEvalHarnessRunner —
    lmms-eval emits the same results JSON format, so re-implementing
    score() would be the duplication the never-branch rule prohibits."""
    from eval_runners.lmms_eval_harness_base import LmmsEvalHarnessRunner
    from eval_runners.lm_eval_harness_base import LmEvalHarnessRunner
    assert LmmsEvalHarnessRunner.score is LmEvalHarnessRunner.score


# ── Runner registrations (4 VL benchmarks) ──────────────────────────────────


OPEN_VLM_RUNNERS = [
    ("mmmu",     "mmmu",     "mmmu_val",    "mmmu_acc,none"),
    ("chartqa",  "chartqa",  "chartqa",     "relaxed_overall,none"),
    ("docvqa",   "docvqa",   "docvqa_val",  "anls,none"),
    ("ai2d",     "ai2d",     "ai2d",        "exact_match,none"),
]


@pytest.mark.parametrize("module_stem,registered_name,task_name,metric_key", OPEN_VLM_RUNNERS)
def test_runner_module_imports_with_correct_class_attrs(module_stem, registered_name, task_name, metric_key):
    mod = __import__(f"eval_runners.{module_stem}", fromlist=[module_stem])
    from eval_runners.base import BenchmarkRunner
    runner_classes = [
        v for v in vars(mod).values()
        if isinstance(v, type) and issubclass(v, BenchmarkRunner) and v is not BenchmarkRunner
    ]
    assert runner_classes, f"no BenchmarkRunner subclass in eval_runners/{module_stem}.py"
    cls = next(c for c in runner_classes if getattr(c, "name", None) == registered_name)
    assert cls.task_name == task_name
    assert cls.metric_key == metric_key


@pytest.mark.parametrize("module_stem,registered_name,task_name,metric_key", OPEN_VLM_RUNNERS)
def test_runner_is_registered_in_global_registry(module_stem, registered_name, task_name, metric_key):
    from eval_runners import resolve_runner
    runner = resolve_runner(registered_name)
    assert runner.name == registered_name
    assert runner.task_name == task_name


# ── score() parses lmms-eval results JSON correctly ─────────────────────────


@pytest.mark.parametrize("module_stem,registered_name,task_name,metric_key", OPEN_VLM_RUNNERS)
def test_runner_score_parses_synthetic_results_json(
    tmp_path, module_stem, registered_name, task_name, metric_key
):
    """Build a synthetic lmms-eval results JSON in the canonical format
    and verify the runner extracts the right metric."""
    from eval_runners import resolve_runner

    fake_score = 0.4271
    results = {
        "results": {
            task_name: {
                metric_key: fake_score,
                "alias": task_name,
            }
        },
        "n-samples": {task_name: {"original": 100, "effective": 100}},
        "config": {"model": "synthetic"},
    }
    results_path = tmp_path / f"results_{registered_name}.json"
    results_path.write_text(json.dumps(results))

    runner = resolve_runner(registered_name)
    result = runner.score(results_path)
    assert result.benchmark_name == registered_name
    assert abs(result.pass_at_1 - fake_score) < 1e-9
    assert result.extras.get("task_name") == task_name


# ── Stubs got graduated ─────────────────────────────────────────────────────


def test_graduated_vl_runners_no_longer_in_sota_stubs():
    """The 4 VL runners used to be stubs in sota_stubs. After graduation
    they MUST not be in REGISTRATIONS — registry would raise on import
    if both versions were registered."""
    from eval_runners import sota_stubs
    graduated = {"mmmu", "chartqa", "docvqa", "ai2d"}
    stub_names = {getattr(c, "name", "") for c in sota_stubs.REGISTRATIONS}
    overlap = graduated & stub_names
    assert not overlap, (
        f"these VL runners graduated but are still registered as stubs: {overlap}. "
        f"Remove them from sota_stubs.REGISTRATIONS."
    )
