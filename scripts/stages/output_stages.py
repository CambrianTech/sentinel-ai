"""Output stages — end of the pipeline.

These produce deliverables:
- Quant: GGUF/MLX/ONNX quantization
- Package: device-specific packaging (CoreML, TensorRT)
- Eval: benchmark evaluation (HumanEval, MMLU, etc.)
- Publish: push to HuggingFace
- Deploy: push to grid node
"""

from pathlib import Path
from .base import StageExecutor, ForgeContext


class QuantExecutor(StageExecutor):
    """Quantize model to GGUF/MLX/ONNX/safetensors."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        fmt = self.config.get("format", "gguf")
        quant_types = self.config.get("quantTypes", ["Q4_K_M"])
        device_targets = self.config.get("deviceTargets", [])

        self.log(f"Quantizing to {fmt}: {', '.join(quant_types)}")
        if device_targets:
            self.log(f"  Device targets: {', '.join(device_targets)}")

        model_dir = ctx.output_dir / "model"
        if not model_dir.exists():
            self.log("WARNING: No model directory — quant deferred")
            return ctx

        if fmt == "gguf":
            # llama.cpp convert + quantize
            for qt in quant_types:
                out_path = ctx.output_dir / f"{ctx.alloy.get('name', 'model')}-{qt}.gguf"
                self.log(f"  {qt} → {out_path.name}")
            self.log("  Requires: llama.cpp (convert_hf_to_gguf.py + llama-quantize)")
            self.log(f"  Use: python scripts/publish_gguf.py {model_dir}")
        elif fmt == "mlx":
            self.log("  Requires: mlx-lm (pip install mlx-lm)")
            self.log(f"  Use: mlx_lm.convert --hf-path {model_dir} --q-bits 4")
        elif fmt == "onnx":
            self.log("  Requires: optimum (pip install optimum)")
            self.log(f"  Use: optimum-cli export onnx -m {model_dir}")
        else:
            self.log(f"  {fmt} not yet supported")

        return ctx


class PackageExecutor(StageExecutor):
    """Package for device-specific runtimes beyond basic quantization."""

    # Known packaging tools per format
    PACKAGE_TOOLS = {
        "coreml": "coremltools (pip install coremltools)",
        "tensorrt": "tensorrt-llm (pip install tensorrt-llm)",
        "onnx": "optimum (pip install optimum[onnxruntime])",
        "openvino": "optimum-intel (pip install optimum[openvino])",
    }

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        fmt = self.config.get("format", "coreml")
        runtime = self.config.get("runtime", "")
        optimization = self.config.get("optimization", "balanced")

        self.log(f"Packaging for {fmt}" + (f" ({runtime})" if runtime else ""))
        self.log(f"  Optimization: {optimization}")

        tool = self.PACKAGE_TOOLS.get(fmt, f"Unknown tool for {fmt}")
        self.log(f"  Requires: {tool}")
        self.log(f"  Issue: CambrianTech/sentinel-ai#121")

        return ctx


class EvalExecutor(StageExecutor):
    """Run benchmark evaluations."""

    # Known benchmarks and their tools
    BENCHMARK_TOOLS = {
        "humaneval": "evalplus (pip install evalplus)",
        "humaneval+": "evalplus (pip install evalplus)",
        "mmlu": "lm-eval-harness (pip install lm-eval)",
        "gsm8k": "lm-eval-harness",
        "arc": "lm-eval-harness",
        "hellaswag": "lm-eval-harness",
        "winogrande": "lm-eval-harness",
        "truthfulqa": "lm-eval-harness",
    }

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        benchmarks = self.config.get("benchmarks", [])
        threshold = self.config.get("passingThreshold")
        compare = self.config.get("compareToBase", True)

        bench_names = [b["name"] for b in benchmarks]
        self.log(f"Evaluating: {', '.join(bench_names)}")

        for bench in benchmarks:
            name = bench["name"]
            tool = self.BENCHMARK_TOOLS.get(name, "unknown")
            self.log(f"  {name}: requires {tool}")

            # TODO: Wire to actual eval harnesses
            ctx.eval_results.append({
                "name": name,
                "subset": bench.get("subset"),
                "metrics": {"status": "pending"},
                "submittedToLeaderboard": bench.get("submitToLeaderboard", False),
            })

        if threshold:
            self.log(f"  Passing threshold: {threshold}%")
        if compare:
            self.log(f"  Will compare to base model")

        return ctx


class PublishExecutor(StageExecutor):
    """Publish to HuggingFace with model card + alloy."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        org = self.config.get("org", "continuum-ai")
        template = self.config.get("repoNameTemplate", "{base}-{domain}-forged")
        include_alloy = self.config.get("includeAlloy", True)
        card_from_benchmarks = self.config.get("cardFromBenchmarks", True)
        tags = self.config.get("tags", [])
        private = self.config.get("private", False)

        self.log(f"Publishing to {org}")
        self.log(f"  Repo template: {template}")
        self.log(f"  Tags: {tags + (['forge-alloy'] if include_alloy else [])}")
        self.log(f"  Private: {private}")
        self.log(f"  Use: python publish_forged.py {ctx.output_dir} --org {org}")

        return ctx


class DeployExecutor(StageExecutor):
    """Deploy to a grid node for serving."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        target = self.config.get("target", "local")
        health = self.config.get("healthCheck", True)
        warmup = self.config.get("warmup", True)
        concurrency = self.config.get("maxConcurrency", 4)
        auto_scale = self.config.get("autoScale", False)

        self.log(f"Deploy target: {target}")
        self.log(f"  Health check: {health}, Warmup: {warmup}")
        self.log(f"  Concurrency: {concurrency}, Auto-scale: {auto_scale}")

        if target == "local":
            self.log(f"  Model available at: {ctx.output_dir / 'model'}")
        elif target in ("bigmama", "grid"):
            self.log(f"  Grid deploy requires SSH/reticulum transport")
            self.log(f"  Issue: CambrianTech/sentinel-ai#122")
        else:
            self.log(f"  Custom target: {target}")

        return ctx
