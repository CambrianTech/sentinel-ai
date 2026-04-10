"""LmmsEvalHarnessRunner — shared base for vision-language benchmarks scored via lmms-eval.

The Open VLM Leaderboard scores VL models against a standard suite via
`lmms-eval` (the vision-aware fork of `lm-eval-harness`). The headline
benchmarks every modern VL forge reports against:

    mmmu_val            — Massive Multi-discipline Multimodal Understanding
    chartqa             — visual QA on charts/graphs
    docvqa_val          — document visual QA (OCR + reasoning)
    ai2d                — Allen Institute Diagrams (science diagrams)

All four route through lmms-eval, so per the never-branch rule the right
shape is ONE base class that does the harness wiring once and FOUR thin
subclasses that just declare task name + metric key.

KEY OOP MOVE: this base INHERITS from LmEvalHarnessRunner. The score()
method (parses results JSON) is identical because lmms-eval emits the
same results format as lm-eval-harness. Only evaluate() is overridden
to call lmms_eval.simple_evaluate instead of lm_eval.simple_evaluate.
Zero duplicated body — the same OOP rule the family adapters use
(PhiMoEAdapter inherits MixtralAdapter; LmmsEvalHarnessRunner inherits
LmEvalHarnessRunner).

Lazy import of lmms_eval — Tier 1 dispatch path on a Mac without harness
installed must keep working. The lazy import fires inside evaluate(),
not score() (score reads JSON, no harness needed for re-scoring).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import ScoreResult
from .lm_eval_harness_base import LmEvalHarnessRunner


class LmmsEvalHarnessRunner(LmEvalHarnessRunner):
    """Shared base for VL benchmarks scored via lmms-eval.

    Inherits score() unchanged — lmms-eval results JSON is identical
    to lm-eval-harness results JSON, so the parser is the same. Only
    overrides evaluate() to swap the harness module.

    Subclasses MUST declare:
        name        — registry name (and alloy benchmark.name field)
        task_name   — lmms-eval task identifier (e.g. 'mmmu_val')
        metric_key  — metric key in the results JSON (e.g. 'mmmu_acc,none')
    """

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
        **kwargs: Any,
    ) -> ScoreResult:
        """Run lmms-eval end-to-end against a VL model directory.

        VL models load with the multimodal `lmms-eval` model wrapper
        (typically 'qwen2_vl', 'qwen2_5_vl', 'qwen3_vl', etc. depending
        on the model_type in config.json). The wrapper handles the
        image-token preprocessing the harness needs.

        Lazy import of lmms_eval — Tier 1 dispatch on a Mac without
        the harness installed keeps working; only the actual evaluate()
        call fires the import.
        """
        self._check_subclass_attrs()
        model_dir = Path(model_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not model_dir.exists():
            raise FileNotFoundError(f"model directory does not exist: {model_dir}")

        try:
            from lmms_eval import simple_evaluate  # type: ignore
            from lmms_eval.tasks import TaskManager  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__}.evaluate requires lmms-eval. "
                f"Install with `pip install lmms-eval`. The runner is "
                f"registered so dispatch resolves on every machine, but "
                f"actual VL evaluation requires the harness present "
                f"locally. Underlying ImportError: {e}"
            ) from e

        # The lmms-eval model name is family-specific. The forge passes
        # the model_type from config.json via extra_model_args; default
        # to 'qwen2_vl' which covers the majority of Qwen-family VL
        # forges (including Qwen3-VL via shape compatibility).
        model_name = (extra_model_args or {}).pop("model", "qwen2_vl")
        model_args = {
            "pretrained": str(model_dir),
            "trust_remote_code": True,
        }
        if extra_model_args:
            model_args.update(extra_model_args)

        task_manager = TaskManager()
        results = simple_evaluate(
            model=model_name,
            model_args=",".join(f"{k}={v}" for k, v in model_args.items()),
            tasks=[self.task_name],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            task_manager=task_manager,
        )

        results_path = output_dir / f"results_{self.name}.json"
        serializable = {
            "results": results.get("results", {}),
            "n-samples": results.get("n-samples", {}),
            "configs": results.get("configs", {}),
            "config": results.get("config", {}),
            "versions": results.get("versions", {}),
        }
        results_path.write_text(json.dumps(serializable, indent=2))
        # Score the freshly-written results JSON via the inherited
        # score() method — uniform contract with every other runner.
        return self.score(results_path)
