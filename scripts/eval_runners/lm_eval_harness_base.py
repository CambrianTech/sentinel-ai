"""LmEvalHarnessRunner — shared base for benchmarks scored via lm-eval-harness.

The HuggingFace Open LLM Leaderboard v2 is the most-watched general-purpose
LLM leaderboard and scores against six benchmarks via lm-eval-harness:

    leaderboard_ifeval        — instruction following
    leaderboard_bbh           — Big-Bench Hard
    leaderboard_math_hard     — MATH level 5
    leaderboard_gpqa          — GPQA Diamond
    leaderboard_mmlu_pro      — MMLU-Pro
    leaderboard_musr          — Multistep Soft Reasoning

All six route through the same harness with different task names and metric
keys. Per the never-branch rule the right shape is ONE base class that does
the harness wiring once and SIX thin subclasses that just declare the task
name + metric key. The base does the lazy import, the model invocation, the
results JSON parsing, and the ScoreResult assembly. Subclasses add zero
behavior — only data.

Two methods, mirroring the humaneval / livecodebench_v6 split:

    score(samples_path)  — score an existing lm-eval-harness results JSON
                            without re-running inference. Lightweight,
                            no GPU needed. Used by Tier 4 reproducibility
                            and the publish pipeline's re-score step.
    evaluate(model_dir,  — run lm-eval-harness end-to-end against a model
             output_dir)   directory: load the model, generate completions,
                            score them, write the results JSON. Heavyweight,
                            requires lm_eval installed and (typically) a GPU.

The lazy import for `lm_eval` lives inside evaluate(), not score() —
re-scoring an existing JSON works on a Mac without harness installed.

Reproducibility contract: each subclass MUST stay frozen against its
declared task_name and metric_key. If lm-eval-harness ships a new task
version (e.g. ifeval_v2), that gets a new file with a new registry name;
old alloys keep resolving to the original runner forever.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkRunner, ScoreResult


class LmEvalHarnessRunner(BenchmarkRunner):
    """Shared base for benchmarks scored via lm-eval-harness.

    Subclasses MUST declare:
        name        — the registry name (and the alloy benchmark.name field)
        task_name   — the lm-eval-harness task identifier
                      (e.g. 'leaderboard_ifeval', 'leaderboard_mmlu_pro')
        metric_key  — the metric to extract from the results JSON
                      (e.g. 'acc,none', 'inst_level_strict_acc,none',
                       'exact_match,none')

    Subclasses MAY override _normalize_score() to apply benchmark-specific
    score normalization, but the default (clamp 0..1, divide by 100 if >1)
    is correct for every Open LLM Leaderboard v2 benchmark.
    """

    task_name: str = ""   # subclass overrides
    metric_key: str = ""  # subclass overrides

    def _check_subclass_attrs(self) -> None:
        if not self.task_name:
            raise NotImplementedError(
                f"{type(self).__name__} must declare task_name as a class "
                f"attribute (the lm-eval-harness task identifier this runner "
                f"scores against, e.g. 'leaderboard_ifeval')."
            )
        if not self.metric_key:
            raise NotImplementedError(
                f"{type(self).__name__} must declare metric_key as a class "
                f"attribute (the key to extract from the harness results JSON, "
                f"e.g. 'acc,none' or 'inst_level_strict_acc,none')."
            )

    def score(self, samples_path: str | Path) -> ScoreResult:
        """Re-score an existing lm-eval-harness results JSON file.

        Args:
            samples_path: path to a results JSON file written by
                          `lm_eval --tasks <task_name> --output_path <dir>`.
                          The file's results dict must contain an entry
                          for self.task_name with self.metric_key inside.

        Returns:
            ScoreResult with benchmark_name=self.name, pass_at_1 = the
            extracted metric (clamped to 0..1), samples_path = the input
            results JSON path, and extras carrying task_name + metric_key
            for the chain of custody.

        Raises:
            FileNotFoundError: if samples_path does not exist.
            ValueError: if the results JSON does not contain self.task_name
                        or self.metric_key — loud failure naming what's
                        missing, never silent substitution.
        """
        self._check_subclass_attrs()
        samples_path = Path(samples_path)
        if not samples_path.exists():
            raise FileNotFoundError(
                f"lm-eval-harness results file does not exist: {samples_path}. "
                f"Run `lm_eval --tasks {self.task_name} --output_path <dir>` "
                f"first, or pass the path to an existing harness results JSON."
            )

        try:
            data = json.loads(samples_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"lm-eval-harness results file {samples_path} is not valid JSON: {e}"
            ) from e

        results = data.get("results")
        if not isinstance(results, dict):
            raise ValueError(
                f"results file {samples_path} has no 'results' dict at the top "
                f"level — does not match lm-eval-harness output format."
            )
        task_results = results.get(self.task_name)
        if task_results is None:
            registered = sorted(results.keys())
            raise ValueError(
                f"results file {samples_path} does not contain task "
                f"{self.task_name!r}. Found tasks: {registered}. Make sure "
                f"the harness was invoked with `--tasks {self.task_name}`."
            )
        if self.metric_key not in task_results:
            available = sorted(k for k in task_results.keys() if "," in k)
            raise ValueError(
                f"results for {self.task_name!r} do not contain metric "
                f"{self.metric_key!r}. Available metrics: {available}. "
                f"Either the harness task version changed (open a new runner "
                f"file) or the metric_key declaration in {type(self).__name__} "
                f"is wrong."
            )

        raw_score = float(task_results[self.metric_key])
        normalized = self._normalize_score(raw_score)

        n_samples = data.get("n-samples", {}).get(self.task_name, {})
        total = n_samples.get("effective") or n_samples.get("original")

        return ScoreResult(
            benchmark_name=self.name,
            pass_at_1=normalized,
            passed=int(round(normalized * total)) if total else None,
            total=total,
            metric=self.metric_key.split(",")[0],
            extras={
                "task_name": self.task_name,
                "metric_key": self.metric_key,
                "raw_score": raw_score,
                "harness": "lm-eval-harness",
            },
            samples_path=str(samples_path),
        )

    def _normalize_score(self, raw: float) -> float:
        """Clamp the raw harness score to 0..1.

        lm-eval-harness usually emits 0..1 fractions but some tasks emit
        0..100 percentages. Divide by 100 if >1, then clamp to [0, 1].
        Subclasses can override for benchmark-specific normalization.
        """
        if raw > 1.0:
            raw = raw / 100.0
        return max(0.0, min(1.0, raw))

    def evaluate(
        self,
        model_dir: str | Path,
        output_dir: str | Path,
        *,
        batch_size: int = 1,
        device: str = "cuda:0",
        num_fewshot: int | None = None,
        limit: int | None = None,
        extra_model_args: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Run lm-eval-harness end-to-end against a model directory.

        This is the codegen-equivalent for lm-eval benchmarks: it loads the
        model, generates completions for the harness task, writes the
        canonical results JSON to output_dir, and returns a ScoreResult
        scored from that JSON via self.score(). The samples_path on the
        returned ScoreResult points at the harness results JSON for
        re-scoring without re-running inference.

        Lazy import: lm_eval is NOT imported until this method is called,
        so Tier 1 dispatch (instantiating the runner, calling score on an
        existing JSON) works on machines without harness installed.

        Args:
            model_dir: path to the local model directory (HuggingFace format).
                       The harness loads it via `pretrained=<model_dir>`.
            output_dir: where to write the results JSON.
            batch_size: harness batch size (1 is safest for big models).
            device: cuda device string.
            num_fewshot: override the task's default few-shot count.
                          Open LLM Leaderboard v2 uses the task default,
                          so leave None unless you know why.
            limit: limit the number of examples (for fast smoke runs).
                   Production scoring must NOT pass limit — the leaderboard
                   numbers are computed on the full split.
            extra_model_args: additional kwargs forwarded to the harness
                              model loader (e.g. dtype, attention impl).

        Returns:
            Path to the results JSON file. Pass this to self.score() to
            extract the metric, or to publish_model.py for the alloy
            benchmark.samplesPath field.

        Raises:
            ImportError: if lm_eval is not installed in the active env.
            FileNotFoundError: if model_dir doesn't exist.
        """
        self._check_subclass_attrs()
        model_dir = Path(model_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"model directory does not exist: {model_dir}"
            )

        try:
            from lm_eval import simple_evaluate
            from lm_eval.tasks import TaskManager
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__}.evaluate requires lm-eval-harness. "
                f"Install with `pip install lm-eval[api]` (or the pinned "
                f"version your forge env requires). The runner is registered "
                f"so dispatch resolves on every machine, but actual evaluation "
                f"requires the harness present locally. Underlying ImportError: {e}"
            ) from e

        model_args = {
            "pretrained": str(model_dir),
            "trust_remote_code": True,
        }
        if extra_model_args:
            model_args.update(extra_model_args)

        task_manager = TaskManager()
        results = simple_evaluate(
            model="hf",
            model_args=",".join(f"{k}={v}" for k, v in model_args.items()),
            tasks=[self.task_name],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            task_manager=task_manager,
        )

        results_path = output_dir / f"results_{self.name}.json"
        # The harness's results dict carries non-JSON-serializable bits in
        # 'samples'; strip them and keep the canonical scoring core.
        serializable = {
            "results": results.get("results", {}),
            "n-samples": results.get("n-samples", {}),
            "configs": results.get("configs", {}),
            "config": results.get("config", {}),
            "versions": results.get("versions", {}),
        }
        results_path.write_text(json.dumps(serializable, indent=2))
        # Score the freshly-written results JSON via self.score() so the
        # caller gets a canonical ScoreResult uniformly with every other
        # runner. eval_with_calibration.run_benchmark depends on this
        # uniform contract — every runner.evaluate returns ScoreResult.
        return self.score(results_path)
