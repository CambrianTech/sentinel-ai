"""LiveCodeBenchV6Runner — the canonical scorer for LiveCodeBench v6.

The hard prerequisite Kash's frontier-target analysis identified: every
modern frontier coder card (Qwen3-Coder-30B, Qwen3-Coder-480B,
DeepSeek-V3.1, Mixtral 8x22B, GPT-4) reports against LiveCodeBench v6
instead of HumanEval, because LCB v6 is the contamination-free
"problems published after a fixed cutoff" successor that hasn't been
in any model's training set. The §4.1.4.1 anchor-reproduction
discipline gate cannot run on any frontier coder forge until the
calibrated eval pipeline supports LCB v6.

This module is the SCORING-ONLY half of the LCB integration: given a
samples JSONL produced by lcb_runner's codegen scenario (or any
LCB-format JSONL), it invokes lcb_runner's canonical
compute_code_generation_metrics function and returns a ScoreResult
with pass@1 + per-task details.

The CODEGEN half (running an actual model against LCB v6 to PRODUCE
the samples) lives in scripts/eval_with_calibration.py::run_livecodebench_v6
and shells out to lcb_runner.runner.main with the right scenario flags.
That function is called by the §4.1.4.1 discipline gate during a forge
run; this runner is what Tier 4 reproducibility tests call to RE-SCORE
an existing JSONL without re-generating samples.

Why a separate scoring runner: the same shape as humaneval.py — the
codegen step is upstream + heavyweight (loads a model, generates samples),
the scoring step is downstream + lightweight (scores existing samples
against the dataset's test cases). Splitting them lets the publish
pipeline hash the samples JSONL once and re-score it any number of
times via the registry without re-running inference.

Lazy import: lcb_runner is NOT imported at module load time so this
module is importable on machines without lcb_runner installed (Tier 1
dispatch path on a Mac, the contract test layer). The lazy import
fires inside score() and raises a clear ImportError pointing at the
install path if lcb_runner isn't available.

Reproducibility contract: this runner MUST stay frozen against the
LCB v6 release_v6 dataset version. If LCB ships a v7, that gets a new
file (livecodebench_v7.py) and a new registry name; old alloys keep
resolving to v6 forever.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .base import BenchmarkRunner, ScoreResult
from .registry import BenchmarkRunnerRegistry


class LiveCodeBenchV6Runner(BenchmarkRunner):
    """Score an existing LCB v6 samples JSONL via lcb_runner's canonical
    codegen_metrics function.

    The samples JSONL must be in lcb_runner's codegeneration scenario
    output format (the format that
    `python -m lcb_runner.runner.main --scenario codegeneration` writes).
    Each line is a problem with the model's generated code completion(s).
    """

    name = "livecodebench_v6"

    def score(self, samples_path: str | Path) -> ScoreResult:
        """Re-score an LCB v6 samples JSONL using lcb_runner's canonical scorer.

        Args:
            samples_path: path to a JSONL produced by lcb_runner's
                          codegeneration scenario, OR a JSON file in the
                          format lcb_runner.runner.main writes
                          (output/{model_repr}/codegeneration_{n}_{temp}.json).

        Returns:
            ScoreResult with benchmark_name='livecodebench_v6', metric='pass@1',
            pass_at_1 as the canonical LCB v6 pass@1 fraction (0..1), and
            extras carrying per-difficulty breakdowns when available.

        Raises:
            ImportError: if lcb_runner is not installed in the active Python
                environment. The error message names the install command:
                `pip install livecodebench` (or its equivalent for the
                pinned LCB release).
            FileNotFoundError: if samples_path does not exist.
            ValueError: if samples_path doesn't contain LCB-format records.
        """
        samples_path = Path(samples_path)
        if not samples_path.exists():
            raise FileNotFoundError(
                f"LCB v6 samples file does not exist: {samples_path}. "
                f"The file must be produced by lcb_runner's codegeneration "
                f"scenario or carry the same shape (per-task records with "
                f"'task_id', 'output_list', and the LCB problem id format)."
            )

        # Lazy import — Tier 1 dispatch must work without lcb_runner installed.
        try:
            from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
            from lcb_runner.benchmarks.code_generation import (
                load_code_generation_dataset,
                CodeGenerationProblem,
            )
        except ImportError as e:
            raise ImportError(
                f"LiveCodeBenchV6Runner requires lcb_runner to be installed. "
                f"Run `pip install livecodebench` to get the canonical "
                f"scoring harness. The runner is registered against name "
                f"'livecodebench_v6' so dispatch resolves on every machine, "
                f"but actual scoring requires the harness present locally. "
                f"Underlying ImportError: {e}"
            ) from e

        # Load the LCB v6 dataset (release_v6 — pinned). The harness
        # caches the dataset to ~/.cache/livecodebench/ on first call.
        problems: list[CodeGenerationProblem] = load_code_generation_dataset(
            release_version="release_v6",
        )

        # Parse the samples file. Two accepted formats:
        #   1. JSONL: one JSON object per line, each with task_id + output_list
        #   2. Single JSON file: lcb_runner's
        #      output/{model_repr}/codegeneration_{n}_{temp}.json shape
        samples_by_task: dict[str, list[str]] = {}
        text = samples_path.read_text().strip()
        if samples_path.suffix == ".json" and text.startswith("["):
            # lcb_runner output JSON format — list of dicts
            records = json.loads(text)
        else:
            # JSONL — one record per line
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for record in records:
            task_id = record.get("task_id") or record.get("question_id")
            outputs = record.get("output_list") or [record.get("output")] if record.get("output") else []
            if task_id and outputs:
                samples_by_task[task_id] = outputs

        if not samples_by_task:
            raise ValueError(
                f"LCB v6 samples file {samples_path} contains no valid records. "
                f"Expected each record to carry 'task_id' (or 'question_id') + "
                f"'output_list' (or 'output')."
            )

        # Run the canonical scorer. lcb_runner.compute_code_generation_metrics
        # takes (samples_list, problems_list, k_list) and returns a metrics
        # dict with 'pass@1', 'pass@5', 'pass@10' keys. We use k=[1] for the
        # canonical headline pass@1.
        samples_list = []
        for problem in problems:
            outputs = samples_by_task.get(problem.task_id) or samples_by_task.get(problem.question_id)
            if outputs is None:
                outputs = []
            samples_list.append({"task_id": problem.task_id, "output_list": outputs})

        metrics, details = codegen_metrics(
            samples=samples_list,
            problems=problems,
            k_list=[1],
        )
        pass_at_1 = float(metrics.get("pass@1", 0.0))
        if pass_at_1 > 1.0:
            # lcb_runner's metric is sometimes 0..100 instead of 0..1
            pass_at_1 = pass_at_1 / 100.0

        passed_count = sum(
            1 for d in details.values() if d.get("pass@1", 0) > 0
        ) if isinstance(details, dict) else None
        total = len(problems)

        return ScoreResult(
            benchmark_name=self.name,
            pass_at_1=pass_at_1,
            passed=passed_count,
            total=total,
            metric="pass@1",
            extras={
                "release_version": "release_v6",
                "k_list": [1],
                "problem_count": total,
            },
            samples_path=str(samples_path),
        )


def register(reg: BenchmarkRunnerRegistry) -> None:
    """Register this runner with a registry instance. Called at module
    import time from scripts/eval_runners/__init__.py."""
    reg.register(LiveCodeBenchV6Runner)
