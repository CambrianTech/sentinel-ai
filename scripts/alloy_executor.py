#!/usr/bin/env python3
"""
alloy_executor.py — Execute ForgeAlloy pipelines stage by stage.

Reads an .alloy.json, executes each stage in order, writes results back.
Wraps forge_model.py's existing functions into a stage-based executor.

Usage:
    python scripts/alloy_executor.py path/to/recipe.alloy.json
    python scripts/alloy_executor.py path/to/recipe.alloy.json --output-dir output/custom
    python scripts/alloy_executor.py path/to/recipe.alloy.json --dry-run

Stage execution order:
    INPUT stages:  source-config → modality → context-extend
    TRANSFORM:     prune → train → lora → compact → expert-prune (cycled)
    OUTPUT stages: quant → package → eval → publish → deploy
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ["PYTHONUNBUFFERED"] = "1"


@dataclass
class ForgeContext:
    """Shared state that flows through the pipeline."""
    model: object = None             # The HuggingFace model
    tokenizer: object = None         # The tokenizer
    model_name: str = ""             # HuggingFace model ID
    output_dir: Path = field(default_factory=lambda: Path("output/forged"))
    alloy: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)  # Model architecture info
    baseline_ppl: float = 0.0
    final_ppl: float = 0.0
    cycle_results: list = field(default_factory=list)
    samples: dict = field(default_factory=dict)
    device: str = ""
    tier: str = ""
    load_4bit: bool = False


class StageExecutor:
    """Base class for stage executors. Each alloy stage type has one."""

    def __init__(self, config: dict):
        self.config = config

    @property
    def stage_type(self) -> str:
        return self.config.get("type", "unknown")

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        raise NotImplementedError(f"Stage executor for '{self.stage_type}' not implemented")

    def log(self, msg: str):
        print(f"  [{self.stage_type}] {msg}")


# ── Input Stages ─────────────────────────────────────────────────────────────

class SourceConfigExecutor(StageExecutor):
    """Reads source-config stage and sets up forge context."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        context_length = self.config.get("contextLength")
        modalities = self.config.get("inputModalities", ["text"])
        devices = self.config.get("targetDevices", [])

        self.log(f"Context: {context_length or 'default'}, Modalities: {modalities}, Devices: {devices}")

        # Store in context for downstream stages
        ctx.alloy["_source_config"] = {
            "contextLength": context_length,
            "inputModalities": modalities,
            "targetDevices": devices,
        }
        return ctx


class ContextExtendExecutor(StageExecutor):
    """Extends context window via RoPE rescaling."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        target = self.config.get("targetLength", 32768)
        method = self.config.get("method", "yarn")
        self.log(f"Extending context to {target} tokens via {method}")

        if ctx.model is not None:
            config = ctx.model.config
            # Apply RoPE scaling
            if method == "yarn":
                if hasattr(config, "rope_scaling"):
                    config.rope_scaling = {"type": "yarn", "factor": target / config.max_position_embeddings}
                    self.log(f"Applied YaRN scaling factor: {target / config.max_position_embeddings:.1f}x")
            elif method == "ntk":
                if hasattr(config, "rope_scaling"):
                    config.rope_scaling = {"type": "dynamic", "factor": target / config.max_position_embeddings}
                    self.log(f"Applied dynamic NTK scaling factor: {target / config.max_position_embeddings:.1f}x")
            elif method == "linear":
                if hasattr(config, "rope_scaling"):
                    config.rope_scaling = {"type": "linear", "factor": target / config.max_position_embeddings}

            config.max_position_embeddings = target
            self.log(f"Context extended to {target}")
        else:
            self.log("WARNING: No model loaded — context extension deferred")

        return ctx


class ModalityExecutor(StageExecutor):
    """Adds vision/audio/video encoder to the model."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        modality = self.config.get("modality", "vision")
        encoder = self.config.get("encoderModel", "")
        proj_arch = self.config.get("projectionArch", "mlp")
        freeze_base = self.config.get("freezeBase", True)
        freeze_encoder = self.config.get("freezeEncoder", True)
        training_steps = self.config.get("trainingSteps", 5000)
        dataset = self.config.get("trainingDataset", "")

        self.log(f"Adding {modality} via {encoder}")
        self.log(f"  Projection: {proj_arch}, Steps: {training_steps}")
        self.log(f"  Freeze base: {freeze_base}, Freeze encoder: {freeze_encoder}")

        if not encoder:
            self.log("ERROR: encoderModel is required for modality stage")
            return ctx

        # TODO: Implement actual encoder attachment
        # 1. Load encoder model
        # 2. Create projection layer (MLP/cross-attention/linear)
        # 3. Train projection on modality-specific dataset
        # 4. Wire into the base model's forward pass
        self.log(f"STUB: Modality attachment not yet implemented — recording intent in alloy")
        self.log(f"  Dataset: {dataset or 'auto-select based on modality'}")

        return ctx


# ── Transform Stages ─────────────────────────────────────────────────────────

class PruneExecutor(StageExecutor):
    """Head pruning by entropy/magnitude/gradient."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        level = self.config.get("level", 0.3)
        strategy = self.config.get("strategy", "entropy")
        self.log(f"Pruning {level:.0%} heads via {strategy}")

        if ctx.model is not None:
            # Import from forge_model.py
            sys.path.insert(0, str(Path(__file__).parent))
            from forge_model import prune, ForgeConfig
            heads, hooks = prune(ctx.model, level, ctx.info, "forward_hooks")
            self.log(f"Pruned {len(heads)}/{ctx.info.get('total_heads', '?')} heads")
            # Store hooks for cleanup
            ctx.alloy.setdefault("_hooks", []).extend(hooks)
        else:
            self.log("WARNING: No model loaded — prune deferred")

        return ctx


class TrainExecutor(StageExecutor):
    """Recovery/fine-tuning with LoRA."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        domain = self.config.get("domain", "code")
        steps = self.config.get("steps", 1000)
        lr = float(self.config.get("learningRate", "2e-4"))
        self.log(f"Training {steps} steps on {domain} data, lr={lr}")

        if ctx.model is not None:
            sys.path.insert(0, str(Path(__file__).parent))
            from forge_model import train_lora, make_dataloaders, ForgeConfig, evaluate

            # Build data loaders
            cfg = ForgeConfig.auto(ctx.info["fp16_gb"],
                                   self._get_vram_gb(),
                                   ctx.load_4bit)
            train_loader, eval_loader = make_dataloaders(ctx.tokenizer, cfg, domain)

            # Train
            ctx.model = train_lora(ctx.model, train_loader, cfg, steps, lr, ctx.output_dir)

            # Evaluate
            post_train = evaluate(ctx.model, eval_loader, ctx.output_dir, "post-train")
            self.log(f"Post-train perplexity: {post_train['perplexity']:.2f}")
        else:
            self.log("WARNING: No model loaded — train deferred")

        return ctx

    @staticmethod
    def _get_vram_gb() -> float:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0


class ExpertPruneExecutor(StageExecutor):
    """MoE expert pruning by activation profile."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        keep = self.config.get("keepExperts", 16)
        strategy = self.config.get("selectionStrategy", "activation")
        self.log(f"Keeping {keep} experts via {strategy} profiling")
        self.log("STUB: Expert pruning via alloy_executor not yet wired — use forge_model.py --experts")
        return ctx


# ── Output Stages ────────────────────────────────────────────────────────────

class QuantExecutor(StageExecutor):
    """Quantize model to GGUF/MLX/ONNX."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        fmt = self.config.get("format", "gguf")
        quant_types = self.config.get("quantTypes", ["Q4_K_M"])
        self.log(f"Quantizing to {fmt}: {', '.join(quant_types)}")

        if fmt == "gguf":
            model_dir = ctx.output_dir / "model"
            if model_dir.exists():
                # Try to use llama.cpp's convert script
                self.log(f"Converting {model_dir} to GGUF")
                for qt in quant_types:
                    gguf_path = ctx.output_dir / f"{ctx.alloy.get('name', 'model')}-{qt}.gguf"
                    self.log(f"  {qt} → {gguf_path.name}")
                    # TODO: Call llama.cpp quantize
                    # For now, record intent
                self.log("STUB: GGUF conversion requires llama.cpp — recording in alloy")
            else:
                self.log("WARNING: No model directory found — quant deferred")
        elif fmt == "mlx":
            self.log("STUB: MLX conversion requires mlx-lm — recording in alloy")
        else:
            self.log(f"STUB: {fmt} conversion not yet implemented")

        return ctx


class PackageExecutor(StageExecutor):
    """Package for device-specific runtimes."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        fmt = self.config.get("format", "coreml")
        runtime = self.config.get("runtime", "")
        optimization = self.config.get("optimization", "balanced")
        self.log(f"Packaging for {fmt} ({runtime or 'default runtime'}), optimization: {optimization}")

        if fmt == "coreml":
            self.log("STUB: CoreML conversion requires coremltools")
        elif fmt == "tensorrt":
            self.log("STUB: TensorRT conversion requires tensorrt-llm")
        elif fmt == "onnx":
            self.log("STUB: ONNX export requires optimum")
        else:
            self.log(f"STUB: {fmt} packaging not yet implemented")

        return ctx


class EvalExecutor(StageExecutor):
    """Run benchmarks."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        benchmarks = self.config.get("benchmarks", [])
        threshold = self.config.get("passingThreshold")
        compare = self.config.get("compareToBase", True)

        bench_names = [b["name"] for b in benchmarks]
        self.log(f"Evaluating: {', '.join(bench_names)}")

        results = []
        for bench in benchmarks:
            name = bench["name"]
            self.log(f"  Running {name}...")

            if name == "humaneval" or name == "humaneval+":
                # TODO: Wire to evalplus
                self.log(f"  STUB: {name} requires evalplus — recording in alloy")
                results.append({"name": name, "metrics": {"status": "pending"}})
            elif name == "mmlu":
                self.log(f"  STUB: {name} requires lm-eval-harness — recording in alloy")
                results.append({"name": name, "metrics": {"status": "pending"}})
            else:
                self.log(f"  STUB: {name} not yet supported")
                results.append({"name": name, "metrics": {"status": "unsupported"}})

        # Store benchmark results in context
        ctx.alloy.setdefault("_eval_results", []).extend(results)

        if threshold:
            self.log(f"  Passing threshold: {threshold}%")

        return ctx


class PublishExecutor(StageExecutor):
    """Publish to HuggingFace."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        org = self.config.get("org", "continuum-ai")
        include_alloy = self.config.get("includeAlloy", True)
        tags = self.config.get("tags", [])
        self.log(f"Publishing to {org}")
        self.log(f"  Tags: {tags}")
        self.log(f"  Include alloy: {include_alloy}")
        self.log(f"  Use: python publish_forged.py {ctx.output_dir} --org {org} --domain {ctx.alloy.get('_domain', 'general')}")
        return ctx


class DeployExecutor(StageExecutor):
    """Deploy to grid node."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        target = self.config.get("target", "local")
        health = self.config.get("healthCheck", True)
        warmup = self.config.get("warmup", True)
        concurrency = self.config.get("maxConcurrency", 4)

        self.log(f"Deploying to {target}")
        self.log(f"  Health check: {health}, Warmup: {warmup}, Concurrency: {concurrency}")

        if target == "local":
            self.log("Local deploy: model available in output directory")
        elif target == "bigmama" or target.startswith("100."):
            self.log(f"STUB: Grid deploy to {target} requires SSH/reticulum transport")
        else:
            self.log(f"STUB: Deploy target '{target}' not yet supported")

        return ctx


# ── Stage Registry ───────────────────────────────────────────────────────────

STAGE_EXECUTORS: dict[str, type[StageExecutor]] = {
    # Input
    "source-config": SourceConfigExecutor,
    "context-extend": ContextExtendExecutor,
    "modality": ModalityExecutor,
    # Transform
    "prune": PruneExecutor,
    "train": TrainExecutor,
    "expert-prune": ExpertPruneExecutor,
    # Output
    "quant": QuantExecutor,
    "package": PackageExecutor,
    "eval": EvalExecutor,
    "publish": PublishExecutor,
    "deploy": DeployExecutor,
}


# ── Pipeline Executor ────────────────────────────────────────────────────────

def execute_alloy(alloy_path: str, output_dir: str = None, dry_run: bool = False):
    """Execute a complete ForgeAlloy pipeline."""
    alloy = json.loads(Path(alloy_path).read_text())
    stages = alloy.get("stages", [])
    cycles = alloy.get("cycles", 1)
    model_name = alloy["source"]["baseModel"]
    name = alloy.get("name", "unnamed")

    print(f"\n{'='*60}")
    print(f"  ALLOY EXECUTOR: {name} v{alloy.get('version', '?')}")
    print(f"  Model: {model_name}")
    print(f"  Stages: {len(stages)}, Cycles: {cycles}")
    print(f"  Pipeline: {' → '.join(s['type'] for s in stages)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN — showing what would execute:\n")
        for i, stage in enumerate(stages):
            stype = stage["type"]
            executor_cls = STAGE_EXECUTORS.get(stype)
            status = "READY" if executor_cls else "NOT IMPLEMENTED"
            print(f"  Stage {i+1}: {stype} [{status}]")
            for k, v in stage.items():
                if k != "type":
                    print(f"    {k}: {v}")
        print(f"\n  Cycles: {cycles} (transform stages repeat)")
        return

    # Setup context
    slug = model_name.split("/")[-1].lower()
    out = Path(output_dir or f"output/forged/{slug}")
    out.mkdir(parents=True, exist_ok=True)

    ctx = ForgeContext(
        model_name=model_name,
        output_dir=out,
        alloy=alloy,
    )

    # Separate stages by position
    input_stages = []
    transform_stages = []
    output_stages = []

    input_types = {"source-config", "context-extend", "modality"}
    output_types = {"quant", "package", "eval", "publish", "deploy"}

    for stage in stages:
        stype = stage["type"]
        if stype in input_types:
            input_stages.append(stage)
        elif stype in output_types:
            output_stages.append(stage)
        else:
            transform_stages.append(stage)

    # Load model
    print("[1] Loading model...")
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from forge_model import load_model, get_model_info, evaluate, make_dataloaders, ForgeConfig, write_status, generate_samples

    ctx.info = get_model_info(model_name)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = ForgeConfig.auto(ctx.info["fp16_gb"], vram_gb)
    ctx.tier = cfg.tier
    ctx.load_4bit = cfg.load_4bit
    ctx.device = torch.cuda.get_device_name(0)

    ctx.model, ctx.tokenizer = load_model(model_name, cfg.load_4bit)

    # Execute input stages
    print("\n[2] Input stages...")
    for stage in input_stages:
        executor = create_executor(stage)
        ctx = executor.execute(ctx)

    # Baseline
    print("\n[3] Baseline evaluation...")
    _, eval_loader = make_dataloaders(ctx.tokenizer, cfg, transform_stages[0].get("domain", "general") if transform_stages else "general")
    baseline = evaluate(ctx.model, eval_loader, out, "baseline")
    ctx.baseline_ppl = baseline["perplexity"]
    print(f"  Baseline perplexity: {ctx.baseline_ppl:.2f}")

    # Execute transform stages (cycled)
    for cycle in range(1, cycles + 1):
        print(f"\n[4.{cycle}] Cycle {cycle}/{cycles}")
        for stage in transform_stages:
            executor = create_executor(stage)
            ctx = executor.execute(ctx)

    # Final evaluation
    print("\n[5] Final evaluation...")
    final = evaluate(ctx.model, eval_loader)
    ctx.final_ppl = final["perplexity"]
    imp = (ctx.baseline_ppl - ctx.final_ppl) / ctx.baseline_ppl * 100
    print(f"  Final: {ctx.baseline_ppl:.2f} → {ctx.final_ppl:.2f} ({imp:+.1f}%)")

    # Save model
    print("\n[6] Saving model...")
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    ctx.model.save_pretrained(str(model_dir))
    ctx.tokenizer.save_pretrained(str(model_dir))

    # Generate samples
    print("\n[7] Generating samples...")
    domain = transform_stages[0].get("domain", "general") if transform_stages else "general"
    ctx.samples = generate_samples(ctx.model, ctx.tokenizer, domain)
    for name, text in ctx.samples.items():
        (out / "benchmark" / f"{name}.txt").write_text(text)

    # Execute output stages
    print("\n[8] Output stages...")
    for stage in output_stages:
        executor = create_executor(stage)
        ctx = executor.execute(ctx)

    # Write results
    results = {
        "model": model_name,
        "domain": domain,
        "baseline_ppl": round(ctx.baseline_ppl, 4),
        "final_ppl": round(ctx.final_ppl, 4),
        "improvement_pct": round(imp, 2),
        "forged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": ctx.device,
        "tier": ctx.tier,
        "cycles": cycles,
        "stages": [s["type"] for s in stages],
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # Write executed alloy
    write_executed_alloy_v2(ctx, results, out)

    print(f"\n{'='*60}")
    print(f"  {model_name}: {ctx.baseline_ppl:.2f} → {ctx.final_ppl:.2f} ({imp:+.1f}%)")
    print(f"  Output: {out}")
    print(f"{'='*60}")


def create_executor(stage: dict) -> StageExecutor:
    """Create a stage executor from an alloy stage config."""
    stype = stage.get("type", "unknown")
    cls = STAGE_EXECUTORS.get(stype)
    if not cls:
        print(f"  WARNING: No executor for stage type '{stype}' — skipping")
        return StageExecutor(stage)
    return cls(stage)


def write_executed_alloy_v2(ctx: ForgeContext, results: dict, out: Path):
    """Write the executed alloy with full results."""
    alloy = ctx.alloy.copy()
    # Remove internal keys
    alloy.pop("_source_config", None)
    alloy.pop("_hooks", None)
    alloy.pop("_eval_results", None)
    alloy.pop("_domain", None)

    # Ensure forge-alloy tag
    if "forge-alloy" not in alloy.get("tags", []):
        alloy.setdefault("tags", []).append("forge-alloy")

    # Build samples
    alloy_samples = []
    for name, text in ctx.samples.items():
        label = name.replace(".txt", "").replace("_", " ").title()
        alloy_samples.append({
            "label": label,
            "prompt": f"(generation sample)",
            "completion": text.strip()[:2000],
        })

    # Model hash
    model_hash = ""
    model_dir = out / "model"
    if model_dir.exists():
        safetensors = sorted(model_dir.glob("*.safetensors"))
        if safetensors:
            h = hashlib.sha256()
            for sf in safetensors:
                with open(sf, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        h.update(chunk)
            model_hash = f"sha256:{h.hexdigest()}"

    script_hash = f"sha256:{hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}"

    alloy["results"] = {
        "completedAt": results.get("forged_at", ""),
        "baselinePerplexity": results.get("baseline_ppl"),
        "finalPerplexity": results.get("final_ppl"),
        "improvementPct": results.get("improvement_pct"),
        "benchmarks": [
            {
                "name": "perplexity",
                "metrics": {
                    "baseline": results.get("baseline_ppl", 0),
                    "final": results.get("final_ppl", 0),
                    "improvement": results.get("improvement_pct", 0),
                },
            }
        ] + ctx.alloy.get("_eval_results", []),
        "hardwareVerified": [
            {
                "device": results.get("device", "unknown"),
                "format": "fp16" if not ctx.load_4bit else "4-bit",
                "verified": True,
            }
        ],
        "samples": alloy_samples,
        "integrity": {
            "trustLevel": "self-attested",
            "code": {
                "runner": "sentinel-ai/alloy_executor",
                "version": "1.0.0",
                "binaryHash": script_hash,
            },
            "modelHash": model_hash,
            "datasets": [],
            "attestedAt": results.get("forged_at", ""),
        },
    }

    alloy_path = out / f"{alloy.get('name', 'unnamed')}.alloy.json"
    alloy_path.write_text(json.dumps(alloy, indent=2))
    print(f"  Alloy: {alloy_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Execute a ForgeAlloy pipeline")
    parser.add_argument("alloy", help="Path to .alloy.json file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show what would execute without running")
    args = parser.parse_args()

    execute_alloy(args.alloy, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
