"""DocVQARunner — document visual QA via lmms-eval.

DocVQA (Mathew et al. 2021) tests reading text embedded in document
images: PDFs, scanned forms, receipts, business documents. The standard
VL benchmark for OCR + reasoning. lmms-eval task: `docvqa_val` (the
validation split, the published-anchor convention). Metric: `anls,none`
(Average Normalized Levenshtein Similarity — DocVQA's official metric,
lenient on minor OCR-style typos).
"""

from __future__ import annotations

from .lmms_eval_harness_base import LmmsEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class DocVQARunner(LmmsEvalHarnessRunner):
    name = "docvqa"
    task_name = "docvqa_val"
    metric_key = "anls,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(DocVQARunner)
