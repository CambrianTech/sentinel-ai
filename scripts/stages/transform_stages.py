"""Transform stages — middle of the pipeline, cycled.

These modify the model's architecture and weights:
- Prune: remove attention heads
- Train: recovery/fine-tuning with LoRA
- ExpertPrune: MoE expert selection
"""

import sys
from pathlib import Path
from .base import StageExecutor, ForgeContext


class PruneExecutor(StageExecutor):
    """Head pruning by entropy/magnitude/gradient."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        level = self.config.get("level", 0.3)
        strategy = self.config.get("strategy", "entropy")
        self.log(f"Pruning {level:.0%} heads via {strategy}")

        if ctx.model is None:
            self.log("WARNING: No model loaded — prune deferred")
            return ctx

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from forge_model import prune
        heads, hooks = prune(ctx.model, level, ctx.info, "forward_hooks")
        ctx.hooks.extend(hooks)
        self.log(f"Pruned {len(heads)} head groups")

        return ctx


class TrainExecutor(StageExecutor):
    """Recovery/fine-tuning with LoRA on domain data."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        domain = self.config.get("domain", "code")
        steps = self.config.get("steps", 1000)
        lr = float(self.config.get("learningRate", "2e-4"))
        batch_size = self.config.get("batchSize", 4)
        self.log(f"Training {steps} steps on {domain} data, lr={lr}")

        if ctx.model is None:
            self.log("WARNING: No model loaded — train deferred")
            return ctx

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from forge_model import train_lora, make_dataloaders, ForgeConfig, evaluate
        import torch

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        cfg = ForgeConfig.auto(ctx.info.get("fp16_gb", 8), vram_gb, ctx.load_4bit)

        train_loader, eval_loader = make_dataloaders(ctx.tokenizer, cfg, domain)
        ctx.model = train_lora(ctx.model, train_loader, cfg, steps, lr, ctx.output_dir)

        post_train = evaluate(ctx.model, eval_loader, ctx.output_dir, "post-train")
        self.log(f"Post-train perplexity: {post_train['perplexity']:.2f}")

        return ctx


class ExpertPruneExecutor(StageExecutor):
    """MoE expert pruning by activation profile.

    Reduces a MoE model by keeping only the most-activated experts.
    Uses cpu_expert_prune.py for the actual pruning.
    """

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        keep = self.config.get("keepExperts", 16)
        strategy = self.config.get("selectionStrategy", "activation")
        profile_steps = self.config.get("profileSteps", 100)

        self.log(f"Keeping {keep} experts via {strategy} profiling ({profile_steps} steps)")

        # TODO: Wire to cpu_expert_prune.py
        # from cpu_expert_prune import prune_experts
        # ctx.model = prune_experts(ctx.model, keep_experts=keep, strategy=strategy)

        self.log(f"STUB — use: python scripts/cpu_expert_prune.py {ctx.model_name} --keep-experts {keep}")
        self.log(f"  Issue: CambrianTech/sentinel-ai#119")

        return ctx
