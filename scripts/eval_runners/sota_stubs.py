"""SOTA benchmark runner stubs — frontier-target eval suite.

Each runner here is REGISTERED but its score() method raises
NotImplementedError with a clear message naming the benchmark and the
documented protocol source. The architectural contract: alloy_executor
can dispatch a frontier forge's eval stage to a named runner without
any other code change. When a real implementation lands for any of
these (e.g. the first Qwen3-Coder-480B forge runs on BigMama and needs
SWE-Bench Verified scoring), the corresponding class gets its real
score() body in a focused commit gated by its own TDD test.

NotImplementedError here is NOT the f-word stub pattern — there's no
"correct architecture" code path being silently substituted. The runner
EXISTS so dispatch resolves; calling it before the real implementation
lands fails LOUDLY at the runner site, which is exactly the
deterministic-rock signal a developer needs to know which file to fill in.

When you implement a real runner:
  1. Move the class out of this file into its own scripts/eval_runners/<name>.py
     (the one-file-per-runner pattern from humaneval.py + humaneval_plus.py)
  2. Implement score() with the official harness for that benchmark
  3. Add a TDD test that scores a known-published JSONL and asserts
     the result matches the published value (the same shape as
     test_humaneval_runner_scores_a_real_published_jsonl in
     test_eval_runner_registry.py)
  4. Remove the entry from this file's REGISTRATIONS list
  5. The test in test_sota_eval_runners.py should be updated to assert
     the real behavior instead of the NotImplementedError

This file is the bridge between "the architecture knows about the
benchmark name" and "the architecture can actually score the benchmark."
The bridge exists so frontier forge alloys can be DECLARED today via
the dispatch path, even before the scoring is wired.
"""

from __future__ import annotations

from pathlib import Path

from .base import BenchmarkRunner, ScoreResult
from .registry import BenchmarkRunnerRegistry


def _stub_score_raise(benchmark: str, protocol_source: str, samples_path) -> ScoreResult:
    raise NotImplementedError(
        f"BenchmarkRunner for {benchmark!r} is registered but not yet implemented. "
        f"To wire it: read the {protocol_source}, implement score(samples_path) "
        f"in scripts/eval_runners/{benchmark}.py (move the class out of "
        f"scripts/eval_runners/sota_stubs.py), and add a TDD test in "
        f"tests/unit/adapters/test_eval_runner_registry.py asserting the score "
        f"reproduces a known-published JSONL value. Called with samples_path={samples_path!r}."
    )


# ── Code benchmarks (frontier coder targets) ────────────────────────────────


class SWEBenchVerifiedRunner(BenchmarkRunner):
    """SWE-Bench Verified — the frontier coder benchmark Qwen3-Coder-480B,
    DeepSeek-V3.1, Claude, GPT-4 etc. all report against. 500 hand-verified
    GitHub issues; the model writes a patch that must apply cleanly AND
    pass the upstream test suite. Protocol source: https://www.swebench.com/

    This is the headline benchmark for the Qwen3-Coder-480B forge target
    Kash mapped in the frontier-quadrant analysis. When the first 480B
    forge runs, this runner gets a real score() body that invokes the
    SWE-Bench Verified harness (Docker per task, runs the patched repo's
    test suite, reports % verified)."""
    name = "swe_bench_verified"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "SWE-Bench Verified protocol at https://www.swebench.com/", samples_path)


# LiveCodeBenchV6Runner moved out to its own file with a real body —
# see scripts/eval_runners/livecodebench_v6.py. The frontier coder cards
# (Qwen3-Coder-30B, Qwen3-Coder-480B, DeepSeek-V3.1, Mixtral 8x22B) all
# report against LCB v6, so this is the first SOTA stub to graduate to a
# real implementation per the §4.1.4.1 anchor-reproduction discipline gate.


class AiderPolyglotRunner(BenchmarkRunner):
    """Aider Polyglot — multi-language code editing benchmark covering
    Python, JavaScript, Rust, Go, C++, Java. Tests whether the model can
    edit existing codebases (the SWE-Bench cousin for non-Python).
    Protocol: https://aider.chat/2024/12/21/polyglot.html"""
    name = "aider_polyglot"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "Aider Polyglot protocol", samples_path)


class MBPPPlusRunner(BenchmarkRunner):
    """MBPP+ — the evalplus extension to MBPP (the "extra-tests" version,
    same author as HumanEval+). Already importable via evalplus; the
    real implementation here is essentially the same shape as
    HumanEvalPlusRunner but pointed at the mbpp dataset."""
    name = "mbpp_plus"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "evalplus MBPP+ protocol (same harness as HumanEval+)", samples_path)


# ── General-purpose benchmarks (frontier general targets) ───────────────────


# MMLUProRunner, GPQADiamondRunner, IFEvalRunner moved out to their own
# files (mmlu_pro.py, gpqa.py, ifeval.py) with real lm-eval-harness bodies
# via LmEvalHarnessRunner. Open LLM Leaderboard v2 runner pack — see also
# bbh.py, math_hard.py, musr.py for the other three v2 benchmarks.


class GSM8KRunner(BenchmarkRunner):
    """GSM8K — grade-school math word problems with chain-of-thought.
    Standard math reasoning benchmark. Runs via lm-eval-harness."""
    name = "gsm8k"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "GSM8K protocol via lm-eval-harness", samples_path)


class AIME2024Runner(BenchmarkRunner):
    """AIME 2024 — American Invitational Mathematics Examination 2024
    problems (30 questions, integer answers). The hardest math benchmark
    in the frontier reasoning suite. DeepSeek-V3.1 reports here."""
    name = "aime_2024"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "AIME 2024 problem set", samples_path)


# ── Vision benchmarks (Qwen2.5-VL / Qwen3.5-VL targets) ─────────────────────


class MMMURunner(BenchmarkRunner):
    """MMMU — Massive Multi-discipline Multimodal Understanding. The
    headline VL benchmark for frontier VL targets (Qwen2.5-VL, future
    Qwen3.5-VL re-forges, GPT-4V, Claude). 11K college-exam-style
    multimodal questions across art / business / science / medicine /
    humanities / tech. Protocol: https://mmmu-benchmark.github.io/"""
    name = "mmmu"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "MMMU protocol at https://mmmu-benchmark.github.io/", samples_path)


class ChartQARunner(BenchmarkRunner):
    """ChartQA — visual question answering on charts/graphs. Tests whether
    the VL model can read structured visual data (bar charts, line graphs,
    pie charts). Standard VL benchmark. Protocol:
    https://github.com/vis-nlp/ChartQA"""
    name = "chartqa"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "ChartQA protocol", samples_path)


class DocVQARunner(BenchmarkRunner):
    """DocVQA — document visual question answering. Tests reading text
    embedded in document images (PDFs, scanned forms, receipts). Standard
    VL benchmark for OCR + reasoning. Protocol: https://www.docvqa.org/"""
    name = "docvqa"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "DocVQA protocol", samples_path)


class AI2DRunner(BenchmarkRunner):
    """AI2D — Allen Institute Diagrams. Multiple-choice VQA on science
    diagrams from grade-school textbooks. Standard VL benchmark for
    diagram understanding. Protocol:
    https://allenai.org/data/diagrams"""
    name = "ai2d"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "AI2D protocol", samples_path)


# ── Audio benchmarks (Qwen2.5-Omni target) ──────────────────────────────────


class CoVoST2Runner(BenchmarkRunner):
    """COVOST 2 — speech translation from 21 languages to English (and
    English to 15 languages). Tests omni / audio models on
    multilingual speech understanding. Protocol:
    https://github.com/facebookresearch/covost"""
    name = "covost2"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "COVOST 2 protocol", samples_path)


class LibriSpeechRunner(BenchmarkRunner):
    """LibriSpeech — automatic speech recognition on audiobook readings.
    The standard ASR benchmark; reports word error rate (WER) on the
    test-clean and test-other splits. Protocol: https://www.openslr.org/12"""
    name = "librispeech"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "LibriSpeech ASR protocol (WER on test-clean / test-other)", samples_path)


class GTZANRunner(BenchmarkRunner):
    """GTZAN — music genre classification (10 genres, 100 30-second clips
    each). Tests whether the audio model can categorize music. Used by
    Qwen2.5-Omni's audio modality evaluation."""
    name = "gtzan"
    def score(self, samples_path):
        return _stub_score_raise(self.name, "GTZAN music genre classification protocol", samples_path)


# ── Registration ────────────────────────────────────────────────────────────


REGISTRATIONS = [
    SWEBenchVerifiedRunner,
    AiderPolyglotRunner,
    MBPPPlusRunner,
    GSM8KRunner,
    AIME2024Runner,
    MMMURunner,
    ChartQARunner,
    DocVQARunner,
    AI2DRunner,
    CoVoST2Runner,
    LibriSpeechRunner,
    GTZANRunner,
]


def register(reg: BenchmarkRunnerRegistry) -> None:
    """Register every SOTA stub runner with a registry instance. Called
    at module import time from scripts/eval_runners/__init__.py."""
    for cls in REGISTRATIONS:
        reg.register(cls)
