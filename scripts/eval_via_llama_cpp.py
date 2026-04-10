#!/usr/bin/env python3
"""eval_via_llama_cpp.py — Run perplexity eval via llama.cpp, bypassing BnB.

The BnB 0.49.2 + transformers + accelerate stack has too many meta-tensor
bugs for 4-bit loading of pruned MoE safetensors. llama.cpp handles MoE
models natively, uses mmap for memory efficiency, and has been running
Mixtral variants since day one.

This script:
1. Converts safetensors → GGUF (if not already done)
2. Optionally quantizes to Q4_K_M / Q5_K_M / Q8_0
3. Runs llama-perplexity on the GGUF
4. Outputs the perplexity number for the model card

Usage:
    # Full pipeline: convert + quantize + eval
    python scripts/eval_via_llama_cpp.py \\
        --model-dir /mnt/cold/factory-work/.../pruned \\
        --llama-cpp ~/llama.cpp \\
        --quant Q4_K_M \\
        --eval-dataset wikitext-2-raw

    # Eval only (GGUF already exists)
    python scripts/eval_via_llama_cpp.py \\
        --gguf /mnt/cold/mixtral-8x7b-compacted-q4km.gguf \\
        --llama-cpp ~/llama.cpp \\
        --eval-dataset wikitext-2-raw

    # Just convert, no eval
    python scripts/eval_via_llama_cpp.py \\
        --model-dir /mnt/cold/factory-work/.../pruned \\
        --llama-cpp ~/llama.cpp \\
        --convert-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_llama_binary(llama_cpp_dir: Path, name: str) -> Path:
    """Find a llama.cpp binary in the build directory."""
    candidates = [
        llama_cpp_dir / "build" / "bin" / name,
        llama_cpp_dir / name,
        Path(f"/usr/local/bin/{name}"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        f"Cannot find {name} in {llama_cpp_dir}. "
        f"Build llama.cpp first: cd {llama_cpp_dir} && cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
    )


def convert_to_gguf(
    model_dir: Path,
    llama_cpp_dir: Path,
    output: Path,
    dtype: str = "f16",
) -> Path:
    """Convert HuggingFace safetensors to GGUF via llama.cpp's converter."""
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Converter not found at {converter}")

    print(f"Converting {model_dir} → {output} (dtype={dtype})")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(converter), str(model_dir),
         "--outfile", str(output), "--outtype", dtype],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"CONVERT FAILED (exit {result.returncode}):")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        raise RuntimeError("GGUF conversion failed")

    size_gb = output.stat().st_size / 1e9
    print(f"Converted in {elapsed:.0f}s → {output} ({size_gb:.1f} GB)")
    return output


def quantize_gguf(
    input_gguf: Path,
    output_gguf: Path,
    llama_cpp_dir: Path,
    quant_type: str = "Q4_K_M",
) -> Path:
    """Quantize a GGUF file via llama-quantize."""
    quantize_bin = find_llama_binary(llama_cpp_dir, "llama-quantize")

    print(f"Quantizing {input_gguf} → {quant_type}")
    t0 = time.time()
    result = subprocess.run(
        [str(quantize_bin), str(input_gguf), str(output_gguf), quant_type],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"QUANTIZE FAILED (exit {result.returncode}):")
        print(result.stderr[-2000:])
        raise RuntimeError("GGUF quantization failed")

    size_gb = output_gguf.stat().st_size / 1e9
    print(f"Quantized in {elapsed:.0f}s → {output_gguf} ({size_gb:.1f} GB)")
    return output_gguf


def run_perplexity(
    gguf_path: Path,
    llama_cpp_dir: Path,
    dataset: str = "wikitext-2-raw",
    context_size: int = 2048,
    batch_size: int = 512,
    n_gpu_layers: int = -1,  # -1 = offload all to GPU
) -> dict:
    """Run llama-perplexity and return the result."""
    perplexity_bin = find_llama_binary(llama_cpp_dir, "llama-perplexity")

    # llama-perplexity expects a text file for the dataset
    # For wikitext-2-raw, we use the standard test split
    dataset_path = Path(f"/tmp/{dataset}-test.txt")
    if not dataset_path.exists():
        print(f"Downloading {dataset} test split...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n".join(t for t in ds["text"] if t.strip())
            dataset_path.write_text(text)
            print(f"Saved {len(text)} chars to {dataset_path}")
        except ImportError:
            raise RuntimeError(
                "datasets library not installed. Install with: pip install datasets"
            )

    print(f"Running perplexity eval on {gguf_path.name}")
    print(f"  dataset: {dataset_path}")
    print(f"  context: {context_size}, batch: {batch_size}, gpu_layers: {n_gpu_layers}")

    t0 = time.time()
    result = subprocess.run(
        [
            str(perplexity_bin),
            "-m", str(gguf_path),
            "-f", str(dataset_path),
            "-c", str(context_size),
            "-b", str(batch_size),
            "-ngl", str(n_gpu_layers),
        ],
        capture_output=True, text=True,
        timeout=7200,  # 2 hour max
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"PERPLEXITY FAILED (exit {result.returncode}):")
        print(result.stderr[-2000:])
        raise RuntimeError("Perplexity evaluation failed")

    # Parse the final perplexity from llama-perplexity output
    # Format: "Final estimate: PPL = 8.1234 +/- 0.5678"
    ppl = None
    for line in result.stdout.splitlines():
        if "Final estimate" in line and "PPL" in line:
            parts = line.split("PPL")
            if len(parts) >= 2:
                try:
                    ppl = float(parts[1].strip().split()[1])
                except (IndexError, ValueError):
                    pass
        # Also check for per-chunk PPL lines
        if "perplexity" in line.lower() and "=" in line:
            try:
                val = float(line.split("=")[-1].strip().split()[0])
                ppl = val  # keep updating — last value is the final
            except (ValueError, IndexError):
                pass

    print(f"\nPerplexity eval complete in {elapsed:.0f}s")
    if ppl is not None:
        print(f"  PPL = {ppl:.4f}")
    else:
        print("  WARNING: could not parse PPL from output")
        print("  Last 10 lines of output:")
        for line in result.stdout.splitlines()[-10:]:
            print(f"    {line}")

    return {
        "perplexity": ppl,
        "elapsed_seconds": elapsed,
        "dataset": dataset,
        "context_size": context_size,
        "gguf": str(gguf_path),
        "stdout_tail": result.stdout[-500:],
    }


def main():
    parser = argparse.ArgumentParser(description="Eval via llama.cpp (bypasses BnB)")
    parser.add_argument("--model-dir", type=str, help="Path to HF-format model dir (safetensors)")
    parser.add_argument("--gguf", type=str, help="Path to existing GGUF file (skip conversion)")
    parser.add_argument("--llama-cpp", type=str, default=os.path.expanduser("~/llama.cpp"),
                        help="Path to llama.cpp directory")
    parser.add_argument("--quant", type=str, default=None,
                        help="Quantization type (Q4_K_M, Q5_K_M, Q8_0). If omitted, eval on F16 GGUF")
    parser.add_argument("--output-dir", type=str, default="/mnt/cold",
                        help="Where to write GGUF files")
    parser.add_argument("--convert-only", action="store_true",
                        help="Just convert to GGUF, don't run eval")
    parser.add_argument("--eval-dataset", type=str, default="wikitext-2-raw",
                        help="Dataset for perplexity evaluation")
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--result-json", type=str, default=None,
                        help="Write eval results to this JSON file")
    args = parser.parse_args()

    llama_cpp = Path(args.llama_cpp)
    output_dir = Path(args.output_dir)

    # Step 1: Get or create GGUF
    if args.gguf:
        gguf_path = Path(args.gguf)
        if not gguf_path.exists():
            print(f"GGUF file not found: {gguf_path}")
            sys.exit(1)
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            print(f"Model dir not found: {model_dir}")
            sys.exit(1)
        model_name = model_dir.parent.name if model_dir.name == "pruned" else model_dir.name
        f16_gguf = output_dir / f"{model_name}-f16.gguf"
        if not f16_gguf.exists():
            convert_to_gguf(model_dir, llama_cpp, f16_gguf)
        else:
            print(f"F16 GGUF already exists: {f16_gguf}")
        gguf_path = f16_gguf
    else:
        print("Either --model-dir or --gguf is required")
        sys.exit(1)

    # Step 2: Quantize if requested
    if args.quant:
        quant_name = gguf_path.stem.replace("-f16", "") + f"-{args.quant}.gguf"
        quant_path = output_dir / quant_name
        if not quant_path.exists():
            quantize_gguf(gguf_path, quant_path, llama_cpp, args.quant)
        else:
            print(f"Quantized GGUF already exists: {quant_path}")
        gguf_path = quant_path

    if args.convert_only:
        print(f"Conversion complete. GGUF at: {gguf_path}")
        return

    # Step 3: Run perplexity
    result = run_perplexity(
        gguf_path, llama_cpp,
        dataset=args.eval_dataset,
        context_size=args.context_size,
        batch_size=args.batch_size,
    )

    # Step 4: Write results
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2))
        print(f"Results written to {result_path}")

    return result


if __name__ == "__main__":
    main()
