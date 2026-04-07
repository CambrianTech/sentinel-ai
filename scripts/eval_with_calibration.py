"""
EvalPlus runner with mandatory calibration anchor.

The forge-side discipline that the manual EvalPlus runs got wrong tonight:
you do NOT report a number from the eval pipeline until the pipeline has
reproduced a known-good third-party number within tolerance. This script
enforces that discipline as a one-shot wrapper.

For any model the forge wants to evaluate, this script:

1. Runs the model under test through patched evalplus.codegen + vllm
2. Runs the configured CALIBRATION ANCHOR model through the same pipeline
3. Compares the calibration result against its published (third-party) number
4. If the calibration check fails (delta > tolerance), HALTS and refuses to
   report the model-under-test number, because the pipeline is not trusted
5. If the calibration check passes, reports both numbers and the calibration
   delta as one combined results JSON for the paper

The whole point: a model-under-test EvalPlus number is meaningless without a
calibration anchor measured on the same pipeline. This script makes that
non-optional.

Calibration anchors are model-card-published HumanEval / HumanEval+ pass@1
numbers from the model authors themselves. The script ships with anchors for
the models we currently care about (Qwen2.5-Coder-{0.5,1.5,3,7,14,32}B base,
from the Qwen2.5-Coder Technical Report Table 5, arXiv:2409.12186). New
anchors must include a citation in the dictionary so the source is auditable.

Usage:
    python -m scripts.eval_with_calibration <model_dir> \\
        --anchor-model <hf_repo> \\
        [--anchor-dir <local_dir>] \\
        [--out <results_dir>] \\
        [--tolerance 3.0]

The anchor-model must be a key in PUBLISHED_ANCHORS below. If anchor-dir is
given, that local path is used instead of downloading from HF.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# Published calibration anchors. Citations are required so the source is
# auditable when the table appears in a paper.
# ────────────────────────────────────────────────────────────────────────────
#
# Schema: model_id -> {
#     "benchmarks": {
#         <benchmark_name>: {"score": float, "metric": str, "source": str},
#         ...
#     }
# }
#
# Each benchmark entry gives one published score on one benchmark from a
# specific source. Multiple benchmarks per model are supported so the same
# model can be calibrated on humaneval, livecodebench_v6, mmlu_pro, etc.
#
# Add new anchors only after verifying the number against the published
# source listed in 'source'. Do not eyeball numbers from blog posts or
# third-party leaderboards without confirming against the model authors'
# own paper or model card.

_QWEN_TECH_REPORT_T5 = "Qwen2.5-Coder Technical Report Table 5, arXiv:2409.12186"
_QWEN35_35B_CARD = "Qwen/Qwen3.5-35B-A3B model card on HuggingFace, March 2026"


def _hf_humaneval_only(he: float, hep: float, src: str) -> dict:
    """Helper for legacy two-benchmark anchors (HumanEval + HumanEval+ from Qwen2.5-Coder Table 5)."""
    return {"benchmarks": {
        "humaneval": {"score": he, "metric": "pass@1", "source": src},
        "humaneval_plus": {"score": hep, "metric": "pass@1", "source": src},
    }}


PUBLISHED_ANCHORS = {
    # Qwen2.5-Coder base series — primary forge targets for Qwen2.5 generation
    "Qwen/Qwen2.5-Coder-0.5B":     _hf_humaneval_only(28.0, 23.8, _QWEN_TECH_REPORT_T5),
    "Qwen/Qwen2.5-Coder-1.5B":     _hf_humaneval_only(43.9, 36.6, _QWEN_TECH_REPORT_T5),
    "Qwen/Qwen2.5-Coder-3B":       _hf_humaneval_only(52.4, 42.7, _QWEN_TECH_REPORT_T5),
    "Qwen/Qwen2.5-Coder-7B":       _hf_humaneval_only(61.6, 53.0, _QWEN_TECH_REPORT_T5),
    "Qwen/Qwen2.5-Coder-14B":      _hf_humaneval_only(64.0, 57.9, _QWEN_TECH_REPORT_T5),
    "Qwen/Qwen2.5-Coder-32B":      _hf_humaneval_only(65.9, 60.4, _QWEN_TECH_REPORT_T5),

    # DeepSeek-Coder series — comparison targets
    "deepseek-ai/deepseek-coder-1.3b-base": _hf_humaneval_only(34.8, 26.8, _QWEN_TECH_REPORT_T5),
    "deepseek-ai/deepseek-coder-6.7b-base": _hf_humaneval_only(47.6, 39.6, _QWEN_TECH_REPORT_T5),
    "deepseek-ai/deepseek-coder-33b-base":  _hf_humaneval_only(54.9, 47.6, _QWEN_TECH_REPORT_T5),
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base": _hf_humaneval_only(40.9, 34.1, _QWEN_TECH_REPORT_T5),
    "deepseek-ai/DeepSeek-Coder-V2-Base":   _hf_humaneval_only(50.0, 43.3, _QWEN_TECH_REPORT_T5),

    # StarCoder2 series — comparison targets
    "bigcode/starcoder2-3b":  _hf_humaneval_only(31.7, 27.4, _QWEN_TECH_REPORT_T5),
    "bigcode/starcoder2-7b":  _hf_humaneval_only(35.4, 29.9, _QWEN_TECH_REPORT_T5),
    "bigcode/starcoder2-15b": _hf_humaneval_only(46.3, 37.8, _QWEN_TECH_REPORT_T5),

    # CodeQwen1.5 — comparison target
    "Qwen/CodeQwen1.5-7B": _hf_humaneval_only(51.8, 45.7, _QWEN_TECH_REPORT_T5),

    # Qwen3.5-35B-A3B — target A for the moonshot. Multi-benchmark anchors
    # extracted from the model card on HuggingFace. NOTE: Qwen3.5 is general-
    # purpose, NOT a Coder variant — its evaluation focuses on reasoning,
    # math, instruction following, multilingual, multimodal — NOT HumanEval.
    # The closest code-generation anchor is LiveCodeBench v6.
    "Qwen/Qwen3.5-35B-A3B": {"benchmarks": {
        # Reasoning
        "mmlu_pro":         {"score": 85.3, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        "mmlu_redux":       {"score": 93.3, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        "gpqa_diamond":     {"score": 84.2, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        # Instruction following
        "ifeval":           {"score": 91.9, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        # Math
        "hmmt_feb_25":      {"score": 89.0, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        "polymath":         {"score": 64.4, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        "mathvision":       {"score": 83.9, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        # Coding (the closest substitutes for HumanEval — Qwen3.5 doesn't
        # report HumanEval; LiveCodeBench v6 is the agentic-coding analog)
        "livecodebench_v6": {"score": 74.6, "metric": "pass@1", "source": _QWEN35_35B_CARD},
        "swe_bench_verified": {"score": 69.2, "metric": "resolved%", "source": _QWEN35_35B_CARD},
        # Multilingual
        "mmmlu":            {"score": 85.2, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        # Multimodal
        "mmmu":             {"score": 81.4, "metric": "accuracy", "source": _QWEN35_35B_CARD},
        "realworldqa":      {"score": 84.1, "metric": "accuracy", "source": _QWEN35_35B_CARD},
    }},

    # NOTE: Llama-3 and Mistral are NOT in PUBLISHED_ANCHORS. Their HumanEval
    # numbers require separate sourcing from the Llama 3 / Mistral papers
    # respectively. Add them here only with explicit citation to the source
    # paper — do not eyeball numbers from blog posts or third-party
    # leaderboards.
}


# Benchmark runners. Each maps a benchmark name to (1) the runner function
# that produces a score and (2) the metric the runner outputs. New benchmarks
# get registered here when their runner is implemented. Unimplemented
# benchmarks are noted in NOT_YET_IMPLEMENTED so the script can fail loud
# instead of silently picking the wrong default.

NOT_YET_IMPLEMENTED = {
    "mmlu_pro": "needs lm-evaluation-harness or equivalent integration — required for Qwen3.5+ reasoning anchor",
    "ifeval": "needs ifeval harness integration",
    "swe_bench_verified": "needs SWE-bench Verified harness integration (very expensive, requires Docker sandboxes)",
    "gpqa_diamond": "needs lm-evaluation-harness integration",
    "hmmt_feb_25": "needs HMMT harness integration",
    "polymath": "needs PolyMATH harness integration",
    "mathvision": "needs MathVision harness integration",
    "mmmlu": "needs MMMLU integration",
    "mmmu": "needs multimodal harness integration",
    "realworldqa": "needs multimodal harness integration",
    "mmlu_redux": "needs lm-evaluation-harness integration",
}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def run_benchmark(benchmark: str, model_dir: Path, out_dir: Path, *, force_base_prompt: bool) -> dict:
    """Dispatch to the right runner for the requested benchmark.

    Returns a dict with at least:
        {"benchmark": str, "scores": {<benchmark_subname>: float, ...}}

    For HumanEval, the runner produces both humaneval and humaneval_plus
    scores in one pass (since evalplus.evaluate emits both). The scores
    dict will contain both keys, and the calibration check will use the
    one matching the requested benchmark name.
    """
    if benchmark in ("humaneval", "humaneval_plus"):
        return run_humaneval(model_dir, out_dir, force_base_prompt=force_base_prompt)
    if benchmark == "livecodebench_v6":
        return run_livecodebench_v6(model_dir, out_dir, force_base_prompt=force_base_prompt)
    if benchmark in NOT_YET_IMPLEMENTED:
        raise NotImplementedError(
            f"benchmark {benchmark!r} is not yet wired into eval_with_calibration.py. "
            f"Reason: {NOT_YET_IMPLEMENTED[benchmark]}. "
            f"Implement the runner in eval_with_calibration.py and register it in run_benchmark()."
        )
    raise ValueError(
        f"unknown benchmark {benchmark!r}. Known: humaneval, humaneval_plus, livecodebench_v6, "
        f"plus the not-yet-implemented set: {sorted(NOT_YET_IMPLEMENTED.keys())}"
    )


def run_livecodebench_v6(model_dir: Path, out_dir: Path, *, force_base_prompt: bool) -> dict:
    """Run LiveCodeBench v6 codegen+evaluate via subprocess to lcb_runner.

    LCB requires (a) a registered model name in lcb_runner/lm_styles.py to
    determine the prompt format, and (b) optionally --local_model_path to
    use a local model instead of downloading the registered name. We use
    a proxy model name to control the prompt format and pass our local
    forge output via --local_model_path.

    For force_base_prompt=True (base models, the v2 forge case), we use
    a GenericBase-style proxy. For False (instruct models), we use a
    Qwen-instruct proxy.

    LCB writes its output to `output/{model_repr}/codegeneration_{n}_{temp}.json`
    relative to the CWD. We set CWD to out_dir so the output lands there.
    The final pass@1 is printed to stdout by lcb_runner.scenario_router.

    Default LCB methodology: n=10 samples, temperature=0.2 (sampling, not
    greedy). This is what the LiveCodeBench paper uses and what published
    numbers are measured against.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "lcb.log"

    # Proxy model name. Choice determines the prompt template LCB uses.
    # For base models we use a GenericBase entry; for instruct/chat models
    # we use the Qwen2.5 instruct entry (same chat template format Qwen3
    # and Qwen3.5 use too).
    if force_base_prompt:
        proxy_model = "meta-llama/Meta-Llama-3-8B"  # GenericBase prompt format
    else:
        proxy_model = "Qwen/Qwen2.5-7B-Instruct"  # CodeQwenInstruct prompt format

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", proxy_model,
        "--local_model_path", str(model_dir),
        "--scenario", "codegeneration",
        "--release_version", "release_v6",
        "--evaluate",
        "--n", "10",
        "--temperature", "0.2",
        "--dtype", "float16",
        "--tensor_parallel_size", "1",
        "--enable_prefix_caching",
        "--trust_remote_code",
    ]

    _log(f"  lcb cmd: {' '.join(cmd)}")
    _log(f"  lcb cwd: {out_dir}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(out_dir), check=True)

    # LCB prints `metrics[0]["pass@1"]` as the last numeric line in stdout.
    log_text = log_path.read_text()
    # Find all standalone float lines (LCB just prints the float)
    matches = re.findall(r"^\s*(\d+\.\d+)\s*$", log_text, re.MULTILINE)
    if not matches:
        raise RuntimeError(
            f"could not parse LiveCodeBench v6 pass@1 from output. "
            f"LCB log: {log_path}. "
            f"Last 20 log lines:\n" + "\n".join(log_text.splitlines()[-20:])
        )
    # The last numeric line is the final pass@1
    pass_at_1_fraction = float(matches[-1])
    pass_at_1 = pass_at_1_fraction * 100  # convert 0.X → X%

    # Find the LCB output JSON for the per-task results (in case downstream
    # wants to inspect them)
    output_jsons = sorted((out_dir / "output").rglob("codegeneration_10_0.2.json"))
    samples_path = str(output_jsons[0]) if output_jsons else None

    return {
        "benchmark": "livecodebench_v6",
        "scores": {"livecodebench_v6": round(pass_at_1, 2)},
        "log_path": str(log_path),
        "samples_path": samples_path,
        "lcb_proxy_model": proxy_model,
        "lcb_release_version": "release_v6",
        "lcb_n": 10,
        "lcb_temperature": 0.2,
    }


def run_humaneval(model_dir: Path, out_dir: Path, *, force_base_prompt: bool) -> dict:
    """Run evalplus.codegen + sanitize + evaluate on a model. Returns
    {'benchmark': 'humaneval', 'scores': {'humaneval': X, 'humaneval_plus': Y}, ...}.

    NO try/except wrapping. If any step fails, the exception propagates and the
    caller halts. Silent failure here is the bug class the harness paper is
    about — never re-introduce it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "evalplus.log"

    cmd = [
        sys.executable, "-m", "evalplus.codegen",
        "--model", str(model_dir),
        "--dataset", "humaneval",
        "--backend", "vllm",
        "--greedy",
        "--bs", "1",
        "--root", str(out_dir),
        "--dtype", "float16",
        "--trust_remote_code",
    ]
    if force_base_prompt:
        cmd.append("--force-base-prompt")

    _log(f"  codegen: {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)

    # Find the generated samples file
    humaneval_dir = out_dir / "humaneval"
    samples = sorted(humaneval_dir.glob("*_temp_0.0.jsonl"))
    if not samples:
        raise RuntimeError(
            f"evalplus.codegen produced no samples file in {humaneval_dir}. "
            f"See log at {log_path}."
        )
    samples_path = samples[0]

    _log(f"  sanitize: {samples_path.name}")
    subprocess.run(
        [sys.executable, "-m", "evalplus.sanitize", "--samples", str(samples_path)],
        check=True,
    )

    sanitized = samples_path.with_name(samples_path.stem + "-sanitized.jsonl")
    if not sanitized.exists():
        raise RuntimeError(f"sanitized file not produced at {sanitized}")

    _log(f"  evaluate: {sanitized.name}")
    eval_proc = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
         "--samples", str(sanitized)],
        check=True, capture_output=True, text=True,
    )
    eval_output = eval_proc.stdout + eval_proc.stderr

    # Parse the two pass@1 numbers. Format from evalplus is:
    #   humaneval (base tests)
    #   pass@1:	0.622
    #   humaneval+ (base + extra tests)
    #   pass@1:	0.537
    base_match = re.search(
        r"humaneval \(base tests\)\s*\n?\s*pass@1:\s*(\d+\.\d+)",
        eval_output,
    )
    plus_match = re.search(
        r"humaneval\+ \(base \+ extra tests\)\s*\n?\s*pass@1:\s*(\d+\.\d+)",
        eval_output,
    )
    if not base_match or not plus_match:
        raise RuntimeError(
            f"could not parse pass@1 from evalplus.evaluate output. "
            f"raw output:\n{eval_output}"
        )
    humaneval_pass1 = float(base_match.group(1)) * 100
    humanevalplus_pass1 = float(plus_match.group(1)) * 100

    return {
        "benchmark": "humaneval",
        "scores": {
            "humaneval": round(humaneval_pass1, 2),
            "humaneval_plus": round(humanevalplus_pass1, 2),
        },
        "samples_path": str(samples_path),
        "sanitized_path": str(sanitized),
        "eval_output": eval_output,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Path to the model under test (forge output)")
    ap.add_argument("--anchor-model", required=True,
                    help="HF repo id of the calibration anchor model. Must be in PUBLISHED_ANCHORS.")
    ap.add_argument("--anchor-dir", default=None,
                    help="Local path to the anchor model. If omitted, uses HF cache.")
    ap.add_argument("--benchmark", default="humaneval",
                    help="Benchmark to run. Default 'humaneval'. The anchor model must have "
                         "a published score for this benchmark in PUBLISHED_ANCHORS. Other "
                         "benchmarks (livecodebench_v6, mmlu_pro, ifeval, etc.) require their "
                         "runners to be implemented in run_benchmark() first.")
    ap.add_argument("--out", default="eval_with_calibration_results",
                    help="Output directory for results")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="Maximum absolute pass@1 delta tolerated for calibration (default 3.0)")
    ap.add_argument("--anchor-force-base-prompt", action="store_true", default=True,
                    help="Whether to pass --force-base-prompt to the anchor (default True; disable for instruct anchors)")
    ap.add_argument("--model-force-base-prompt", action="store_true", default=True,
                    help="Whether to pass --force-base-prompt to the model under test (default True)")
    args = ap.parse_args()

    if args.anchor_model not in PUBLISHED_ANCHORS:
        sys.exit(
            f"FATAL: anchor-model {args.anchor_model!r} is not in PUBLISHED_ANCHORS. "
            f"Add it with a citation before using it as a calibration anchor. "
            f"Known anchors: {sorted(PUBLISHED_ANCHORS.keys())}"
        )

    anchor_meta = PUBLISHED_ANCHORS[args.anchor_model]
    anchor_benchmarks = anchor_meta.get("benchmarks", {})
    if args.benchmark not in anchor_benchmarks:
        sys.exit(
            f"FATAL: anchor {args.anchor_model!r} has no published score for "
            f"benchmark {args.benchmark!r}. Available benchmarks for this anchor: "
            f"{sorted(anchor_benchmarks.keys())}. Add the score to PUBLISHED_ANCHORS "
            f"with a citation, or pick a different --benchmark."
        )
    anchor_published = anchor_benchmarks[args.benchmark]
    anchor_published_score = anchor_published["score"]

    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else Path(args.anchor_model)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Calibration anchor: {args.anchor_model}")
    _log(f"  benchmark: {args.benchmark} ({anchor_published['metric']})")
    _log(f"  published score: {anchor_published_score}")
    _log(f"  source: {anchor_published['source']}")
    _log(f"  tolerance: ±{args.tolerance} points")

    _log("=" * 60)
    _log("Stage 1/2: anchor calibration run")
    _log("=" * 60)
    anchor_result = run_benchmark(
        args.benchmark, anchor_dir, out_dir / "anchor",
        force_base_prompt=args.anchor_force_base_prompt,
    )
    if args.benchmark not in anchor_result["scores"]:
        sys.exit(
            f"FATAL: run_benchmark({args.benchmark!r}) returned scores "
            f"{sorted(anchor_result['scores'].keys())} but the requested benchmark "
            f"{args.benchmark!r} is not in the result. Runner contract violation."
        )
    anchor_measured_score = anchor_result["scores"][args.benchmark]
    _log(f"anchor measured: {args.benchmark}={anchor_measured_score}")

    delta = anchor_measured_score - anchor_published_score
    _log(f"anchor delta:    {args.benchmark}={delta:+.2f}")

    if abs(delta) > args.tolerance:
        halt_path = out_dir / "CALIBRATION_HALT.json"
        halt_data = {
            "status": "calibration_failed",
            "anchor_model": args.anchor_model,
            "benchmark": args.benchmark,
            "anchor_published": anchor_published,
            "anchor_measured": anchor_result["scores"],
            "delta": round(delta, 2),
            "tolerance": args.tolerance,
            "reason": (
                "Calibration anchor delta exceeded tolerance. The eval pipeline "
                "is not trusted; the model-under-test was NOT evaluated. Diagnose "
                "the pipeline before re-running."
            ),
        }
        halt_path.write_text(json.dumps(halt_data, indent=2))
        _log("=" * 60)
        _log(f"HALT: calibration delta exceeded tolerance ({args.tolerance})")
        _log(f"  delta: {delta:+.2f} (allowed ±{args.tolerance})")
        _log(f"State dumped to {halt_path}")
        _log("=" * 60)
        sys.exit(2)

    _log("anchor calibration PASSED")
    _log("=" * 60)
    _log("Stage 2/2: model under test")
    _log("=" * 60)
    model_result = run_benchmark(
        args.benchmark, model_dir, out_dir / "model",
        force_base_prompt=args.model_force_base_prompt,
    )
    model_score = model_result["scores"][args.benchmark]
    _log(f"model measured: {args.benchmark}={model_score}")

    # Build the final results JSON
    results = {
        "status": "complete",
        "benchmark": args.benchmark,
        "anchor": {
            "model": args.anchor_model,
            "published": anchor_published,
            "measured_scores": anchor_result["scores"],
            "delta": round(delta, 2),
        },
        "model_under_test": {
            "path": str(model_dir),
            "measured_scores": model_result["scores"],
        },
        "tolerance": args.tolerance,
        "calibration_passed": True,
    }
    results_path = out_dir / "calibrated_eval_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    _log("=" * 60)
    _log(f"DONE. results: {results_path}")
    _log(f"  benchmark: {args.benchmark}")
    _log(f"  anchor measured: {anchor_result['scores']}")
    _log(f"  model measured:  {model_result['scores']}")
    _log(f"  delta vs anchor: {model_score - anchor_measured_score:+.2f}")
    _log("=" * 60)


if __name__ == "__main__":
    main()
