"""Many-Worlds stage executors — substrate training + adapter training.

Wires the Many-Worlds primitives (scripts/many_worlds/) into the
forge-alloy stage executor pattern. The alloy executor dispatches
to these when it encounters 'many-worlds-substrate' or
'many-worlds-adapter' stage types.
"""

import json
import sys
from pathlib import Path

from .base import StageExecutor, ForgeContext

# Add many_worlds package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "many_worlds"))


class ManyWorldsSubstrateExecutor(StageExecutor):
    """Phase A: train the shared Gaussian substrate across multiple models."""

    stage_type = "many-worlds-substrate"

    def execute(self, ctx: ForgeContext, params: dict) -> ForgeContext:
        from many_worlds.train_substrate import train_substrate

        source_models = params.get("sourceModels", [])
        corpus = params.get("calibrationCorpus", "")
        substrate_dim = params.get("substrateDim", 256)
        num_gaussians = params.get("numGaussians", 128)
        steps = params.get("trainingSteps", 1000)
        lr = float(params.get("learningRate", "1e-4"))
        loss_type = params.get("lossType", "both")

        output_dir = str(ctx.output_dir / "substrate")

        self.log(f"Many-Worlds substrate training:")
        self.log(f"  Models: {source_models}")
        self.log(f"  Substrate: dim={substrate_dim}, gaussians={num_gaussians}")
        self.log(f"  Steps: {steps}, LR: {lr}")

        metadata = train_substrate(
            model_names=source_models,
            corpus_path=corpus,
            substrate_dim=substrate_dim,
            num_gaussians=num_gaussians,
            steps=steps,
            learning_rate=lr,
            output_dir=output_dir,
        )

        # Store substrate path on context for downstream adapter stages
        ctx.substrate_path = str(Path(output_dir) / "substrate.pt")
        ctx.substrate_metadata = metadata

        self.log(f"  Substrate trained: loss={metadata.get('final_loss', '?'):.4f}")
        return ctx


class ManyWorldsAdapterExecutor(StageExecutor):
    """Phase B: train per-model Project/Read adapter against fixed substrate."""

    stage_type = "many-worlds-adapter"

    def execute(self, ctx: ForgeContext, params: dict) -> ForgeContext:
        # TODO: implement adapter-specific training
        # For v0, the substrate training already creates adapters for all models
        # This stage is for ADDITIONAL adapter training or fine-tuning

        target_model = params.get("targetModel", "")
        substrate_path = params.get("substratePath") or getattr(ctx, "substrate_path", None)
        adapter_rank = params.get("adapterRank", 64)
        steps = params.get("trainingSteps", 500)

        self.log(f"Many-Worlds adapter training:")
        self.log(f"  Target: {target_model}")
        self.log(f"  Substrate: {substrate_path}")
        self.log(f"  Rank: {adapter_rank}, Steps: {steps}")

        # For v0, the adapter was already trained in the substrate stage
        # This stage verifies it exists and records metadata
        safe_name = target_model.replace("/", "_")
        adapter_path = str(Path(substrate_path).parent / f"adapter_{safe_name}.pt")

        if Path(adapter_path).exists():
            self.log(f"  Adapter exists: {adapter_path}")
        else:
            self.log(f"  WARNING: adapter not found at {adapter_path}")
            # TODO: train adapter independently against fixed substrate

        return ctx
