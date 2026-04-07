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
# Format: model_id -> {humaneval_pass1, humanevalplus_pass1, source}
#
# Add new anchors only after verifying the number against the published
# source listed in 'source'. Do not eyeball numbers from blog posts or
# leaderboards without confirming against the model authors' own paper or
# model card.

_QWEN_TECH_REPORT_T5 = "Qwen2.5-Coder Technical Report Table 5, arXiv:2409.12186"

PUBLISHED_ANCHORS = {
    # Qwen2.5-Coder base series — primary forge targets
    "Qwen/Qwen2.5-Coder-0.5B":     {"humaneval_pass1": 28.0, "humanevalplus_pass1": 23.8, "source": _QWEN_TECH_REPORT_T5},
    "Qwen/Qwen2.5-Coder-1.5B":     {"humaneval_pass1": 43.9, "humanevalplus_pass1": 36.6, "source": _QWEN_TECH_REPORT_T5},
    "Qwen/Qwen2.5-Coder-3B":       {"humaneval_pass1": 52.4, "humanevalplus_pass1": 42.7, "source": _QWEN_TECH_REPORT_T5},
    "Qwen/Qwen2.5-Coder-7B":       {"humaneval_pass1": 61.6, "humanevalplus_pass1": 53.0, "source": _QWEN_TECH_REPORT_T5},
    "Qwen/Qwen2.5-Coder-14B":      {"humaneval_pass1": 64.0, "humanevalplus_pass1": 57.9, "source": _QWEN_TECH_REPORT_T5},
    "Qwen/Qwen2.5-Coder-32B":      {"humaneval_pass1": 65.9, "humanevalplus_pass1": 60.4, "source": _QWEN_TECH_REPORT_T5},

    # DeepSeek-Coder series — comparison targets
    "deepseek-ai/deepseek-coder-1.3b-base": {"humaneval_pass1": 34.8, "humanevalplus_pass1": 26.8, "source": _QWEN_TECH_REPORT_T5},
    "deepseek-ai/deepseek-coder-6.7b-base": {"humaneval_pass1": 47.6, "humanevalplus_pass1": 39.6, "source": _QWEN_TECH_REPORT_T5},
    "deepseek-ai/deepseek-coder-33b-base":  {"humaneval_pass1": 54.9, "humanevalplus_pass1": 47.6, "source": _QWEN_TECH_REPORT_T5},
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base": {"humaneval_pass1": 40.9, "humanevalplus_pass1": 34.1, "source": _QWEN_TECH_REPORT_T5},
    "deepseek-ai/DeepSeek-Coder-V2-Base":   {"humaneval_pass1": 50.0, "humanevalplus_pass1": 43.3, "source": _QWEN_TECH_REPORT_T5},

    # StarCoder2 series — comparison targets
    "bigcode/starcoder2-3b":  {"humaneval_pass1": 31.7, "humanevalplus_pass1": 27.4, "source": _QWEN_TECH_REPORT_T5},
    "bigcode/starcoder2-7b":  {"humaneval_pass1": 35.4, "humanevalplus_pass1": 29.9, "source": _QWEN_TECH_REPORT_T5},
    "bigcode/starcoder2-15b": {"humaneval_pass1": 46.3, "humanevalplus_pass1": 37.8, "source": _QWEN_TECH_REPORT_T5},

    # CodeQwen1.5 — comparison target
    "Qwen/CodeQwen1.5-7B": {"humaneval_pass1": 51.8, "humanevalplus_pass1": 45.7, "source": _QWEN_TECH_REPORT_T5},

    # NOTE: Llama-3 and Mistral are NOT in this table. Their HumanEval numbers
    # require separate sourcing from the Llama 3 / Mistral papers respectively.
    # Add them here only with explicit citation to the Meta / Mistral source —
    # do not eyeball numbers from blog posts or third-party leaderboards.
}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def run_evalplus(model_dir: Path, out_dir: Path, *, force_base_prompt: bool) -> dict:
    """Run evalplus.codegen + sanitize + evaluate on a model. Returns
    {'humaneval_pass1': float, 'humanevalplus_pass1': float, 'codegen_log': str}.

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
        "humaneval_pass1": round(humaneval_pass1, 2),
        "humanevalplus_pass1": round(humanevalplus_pass1, 2),
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
    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else Path(args.anchor_model)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Calibration anchor: {args.anchor_model}")
    _log(f"  published HumanEval pass@1:  {anchor_meta['humaneval_pass1']}")
    _log(f"  published HumanEval+ pass@1: {anchor_meta['humanevalplus_pass1']}")
    _log(f"  source: {anchor_meta['source']}")
    _log(f"  tolerance: ±{args.tolerance} points")

    _log("=" * 60)
    _log("Stage 1/2: anchor calibration run")
    _log("=" * 60)
    anchor_result = run_evalplus(
        anchor_dir, out_dir / "anchor",
        force_base_prompt=args.anchor_force_base_prompt,
    )
    _log(f"anchor measured: HE={anchor_result['humaneval_pass1']} HE+={anchor_result['humanevalplus_pass1']}")

    he_delta = anchor_result["humaneval_pass1"] - anchor_meta["humaneval_pass1"]
    plus_delta = anchor_result["humanevalplus_pass1"] - anchor_meta["humanevalplus_pass1"]
    _log(f"anchor delta:   HE={he_delta:+.2f}  HE+={plus_delta:+.2f}")

    if abs(he_delta) > args.tolerance or abs(plus_delta) > args.tolerance:
        # Halt and dump state. The model-under-test number does NOT get reported.
        halt_path = out_dir / "CALIBRATION_HALT.json"
        halt_data = {
            "status": "calibration_failed",
            "anchor_model": args.anchor_model,
            "anchor_published": anchor_meta,
            "anchor_measured": anchor_result,
            "anchor_delta": {
                "humaneval": he_delta,
                "humanevalplus": plus_delta,
            },
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
        _log(f"  HE delta:  {he_delta:+.2f} (allowed ±{args.tolerance})")
        _log(f"  HE+ delta: {plus_delta:+.2f} (allowed ±{args.tolerance})")
        _log(f"State dumped to {halt_path}")
        _log("=" * 60)
        sys.exit(2)

    _log("anchor calibration PASSED")
    _log("=" * 60)
    _log("Stage 2/2: model under test")
    _log("=" * 60)
    model_result = run_evalplus(
        model_dir, out_dir / "model",
        force_base_prompt=args.model_force_base_prompt,
    )
    _log(f"model measured: HE={model_result['humaneval_pass1']} HE+={model_result['humanevalplus_pass1']}")

    # Build the final results JSON
    results = {
        "status": "complete",
        "anchor": {
            "model": args.anchor_model,
            "published": anchor_meta,
            "measured": {
                "humaneval_pass1": anchor_result["humaneval_pass1"],
                "humanevalplus_pass1": anchor_result["humanevalplus_pass1"],
            },
            "delta": {
                "humaneval": round(he_delta, 2),
                "humanevalplus": round(plus_delta, 2),
            },
        },
        "model_under_test": {
            "path": str(model_dir),
            "measured": {
                "humaneval_pass1": model_result["humaneval_pass1"],
                "humanevalplus_pass1": model_result["humanevalplus_pass1"],
            },
        },
        "tolerance": args.tolerance,
        "calibration_passed": True,
    }
    results_path = out_dir / "calibrated_eval_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    _log("=" * 60)
    _log(f"DONE. results: {results_path}")
    _log(f"  anchor measured: {anchor_result['humaneval_pass1']} / {anchor_result['humanevalplus_pass1']}")
    _log(f"  model measured:  {model_result['humaneval_pass1']} / {model_result['humanevalplus_pass1']}")
    _log("=" * 60)


if __name__ == "__main__":
    main()
