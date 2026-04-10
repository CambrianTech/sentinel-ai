"""stages.py — forge-alloy stage executors for Many-Worlds training.

These stage executors plug the Many-Worlds primitives (substrate,
project_read, framework, losses) into the sentinel-ai forge pipeline
by exposing three new stage types:

  1. `substrate-train`     — Phase A: train the substrate against a
                             population of frozen base models
  2. `adapter-train`       — Phase B: train per-model Project/Read
                             adapters against a frozen substrate
  3. `many-worlds-eval`    — the §VII five-condition comparison
                             (text-bottleneck, substrate-transfer,
                             random-substrate, FuseLLM, single-MoE)

Each executor mirrors the existing forge-alloy stage executor pattern
in scripts/stages/ — it takes a ForgeContext + stage params, mutates
the context, and returns it. The forge-alloy schema additions needed
to make these stages recognized live in a separate forge-alloy PR
(see `continuum/docs/papers/MANY-WORLDS-ABSTRACT.md` §V.6.6 for the
recipe sketch and §V.6.5 for the leverage from existing infrastructure).

**Scaffolding note (2026-04-10)**: these executors are written as
skeleton implementations with the full algorithm documented inline
but the actual torch training loop body stubbed as `NotImplementedError`
with a clear TODO. The reason: the v0 validation protocol (§VII) needs
the substrate + adapter primitives to exist in importable form BEFORE
the training loop can be written, because the training loop imports
them. This file is the contract layer between forge-alloy stage
dispatch and the Many-Worlds primitives.

The training loops themselves live in:
  - `scripts/many_worlds/train_substrate.py` (Phase A, TODO)
  - `scripts/many_worlds/train_adapters.py` (Phase B, TODO)
  - `scripts/many_worlds/eval_v0.py` (the §VII five-condition driver, TODO)

These TODO files are written incrementally — start with substrate.py,
project_read.py, framework.py, losses.py (all done as of this commit),
then add the training loops as the next layer up.

**Recipe shape for v0 validation**:

```jsonc
{
  "name": "many-worlds-v0-tiny",
  "version": "0.1.0",
  "workloadType": "forge",
  "source": {
    "population": [
      { "baseModel": "Qwen/Qwen2.5-1.5B-Instruct", "architecture": "qwen2_5_dense" },
      { "baseModel": "meta-llama/Llama-3.2-1B-Instruct", "architecture": "llama3" }
    ]
  },
  "substrate": {
    "dimensionality": 128,
    "num_bases": 1024,
    "init_strategy": "orthogonal"
  },
  "stages": [
    {
      "type": "substrate-train",
      "calibrationCorpus": "many-worlds-v0-mixed-1k",
      "trainingStepsK": 50,
      "loss": {
        "alpha_contrastive": 1.0,
        "beta_round_trip": 1.0,
        "contrastive_temperature": 0.07
      }
    },
    {
      "type": "adapter-train",
      "loraRank": 64,
      "trainingStepsKPerMember": 20,
      "loss": {
        "gamma_round_trip": 1.0,
        "delta_cross_model": 1.0,
        "epsilon_native_preservation": 0.1
      }
    },
    {
      "type": "many-worlds-eval",
      "conditions": ["text_bottleneck", "substrate_transfer", "random_substrate"],
      "heldout_corpus": "many-worlds-v0-heldout-100",
      "predictedOutcomes": {
        "substrate_transfer_beats_text_bottleneck": true,
        "substrate_transfer_beats_random_substrate_margin": 0.10
      }
    },
    {
      "type": "publish",
      "destination": "huggingface",
      "org": "continuum-ai",
      "repoName": "many-worlds-v0-tiny"
    }
  ]
}
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


# ── Substrate training (Phase A) ───────────────────────────────────────


@dataclass
class SubstrateTrainParams:
    """Parsed stage params for the substrate-train stage."""

    calibration_corpus: str
    calibration_corpus_file: Optional[str] = None
    training_steps: int = 50_000
    batch_size: int = 16
    learning_rate: float = 1e-3
    alpha_contrastive: float = 1.0
    beta_round_trip: float = 1.0
    contrastive_temperature: float = 0.07
    round_trip_loss_type: str = "mse"
    eval_interval: int = 1000
    save_interval: int = 5000
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "SubstrateTrainParams":
        loss = d.get("loss") or {}
        return cls(
            calibration_corpus=d["calibrationCorpus"],
            calibration_corpus_file=d.get("calibrationCorpusFile"),
            training_steps=int(d.get("trainingSteps", d.get("trainingStepsK", 50)) * (
                1000 if "trainingStepsK" in d else 1
            )),
            batch_size=int(d.get("batchSize", 16)),
            learning_rate=float(d.get("learningRate", 1e-3)),
            alpha_contrastive=float(loss.get("alpha_contrastive", 1.0)),
            beta_round_trip=float(loss.get("beta_round_trip", 1.0)),
            contrastive_temperature=float(loss.get("contrastive_temperature", 0.07)),
            round_trip_loss_type=loss.get("round_trip_loss_type", "mse"),
            eval_interval=int(d.get("evalInterval", 1000)),
            save_interval=int(d.get("saveInterval", 5000)),
            seed=int(d.get("seed", 42)),
        )


class SubstrateTrainExecutor:
    """Phase A executor: train the substrate against a frozen population.

    Inputs (from ctx.alloy):
      - source.population[] — list of base model + architecture pairs
      - substrate.* — SubstrateConfig fields (dimensionality, num_bases, etc.)
      - Current stage's params — training hyperparameters + loss weights

    Outputs (mutations to ctx):
      - ctx.many_worlds_framework: a freshly constructed ManyWorldsFramework
        with the substrate trained and population members declared but
        their adapters NOT YET trained (that happens in the next stage)
      - ctx.alloy['results']['substrate_phase_a'] = metrics dict
      - ctx.alloy['results']['priorMetricBaselines'] may be appended
        with a zero-length baseline placeholder (filled by the eval stage)

    The substrate is trained against the CALIBRATION CORPUS declared in
    the stage params. Each batch pulls the same inputs through every
    population member (with the base models frozen) and captures the
    chosen-layer residual from each. These residuals are the training
    data: Project → write → read → Read round-trip losses plus the
    contrastive alignment across members are the two-term objective.
    """

    def execute(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Scaffold — full implementation is TODO.

        The algorithm, fully specified:

        1. Construct the SubstrateVectorSpace from ctx.alloy['substrate']
           config. Fresh init, not trained yet.
        2. For each population member in ctx.alloy['source']['population']:
           a. Load the base model via the family adapter's load path
              (reuse existing load_model infrastructure including the
              streaming-load path for big models)
           b. Freeze all base model parameters
           c. Query the model's config for residual_hidden_size and
              num_hidden_layers; compute layer_idx from default_layer_fraction
           d. Construct an AdapterPair for this member with the substrate's
              dimensionality and the specified lora_rank
           e. Add the member to a ManyWorldsFramework instance
        3. Load the calibration corpus from calibration_corpus_file. Tokenize
           for each population member separately (they have different
           tokenizers — that's the whole point of heterogeneous populations).
        4. Training loop:
           for step in range(training_steps):
             batch = next_batch_across_all_members()  # paired inputs
             # Forward through each member, capture residual at layer_idx
             for member in population:
                 with torch.no_grad():
                     residual = run_model_up_to_layer(member.model, batch[member.name], member.layer_idx)
                 mu, log_var = member.adapter.project(residual)
                 substrate_vec = substrate.read(mu, log_var)
                 reconstructed = member.adapter.read(substrate_vec)

                 projections[member.name] = mu.mean(dim=1)  # pool for contrastive
                 original_residuals[member.name] = residual
                 reconstructed_residuals[member.name] = reconstructed

             loss, metrics = phase_a_loss(
                 projections, original_residuals, reconstructed_residuals,
                 config=PhaseALossConfig(...)
             )
             loss.backward()
             optimizer.step()
             optimizer.zero_grad()

             if step % eval_interval == 0:
                 log(metrics)
             if step % save_interval == 0:
                 substrate.save(intermediate checkpoint)
        5. Mark substrate as trained via substrate.mark_trained()
        6. Attach the framework to ctx for downstream stages
        7. Record final metrics in ctx.alloy['results']

        The stage is deliberately self-contained — by the time it returns,
        the framework has its substrate trained and every member's adapter
        pair instantiated (but those adapters still have zero-init output
        heads; Phase B trains them).
        """
        raise NotImplementedError(
            "SubstrateTrainExecutor.execute is a scaffold. The full "
            "training loop is documented inline but not yet implemented. "
            "The v0 driver in scripts/many_worlds/train_substrate.py is "
            "the next file to write; this executor becomes a thin wrapper "
            "around it once that lands. See continuum/docs/papers/"
            "MANY-WORLDS-ABSTRACT.md §VII for the validation protocol."
        )


# ── Per-model adapter training (Phase B) ───────────────────────────────


@dataclass
class AdapterTrainParams:
    """Parsed stage params for the adapter-train stage."""

    lora_rank: int = 64
    training_steps_per_member: int = 20_000
    batch_size: int = 16
    learning_rate: float = 5e-4
    gamma_round_trip: float = 1.0
    delta_cross_model: float = 1.0
    epsilon_native_preservation: float = 0.1
    transfer_rollout_length: int = 50
    eval_interval: int = 500
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterTrainParams":
        loss = d.get("loss") or {}
        steps = d.get("trainingStepsPerMember") or d.get("trainingStepsKPerMember", 20) * 1000
        return cls(
            lora_rank=int(d.get("loraRank", 64)),
            training_steps_per_member=int(steps),
            batch_size=int(d.get("batchSize", 16)),
            learning_rate=float(d.get("learningRate", 5e-4)),
            gamma_round_trip=float(loss.get("gamma_round_trip", 1.0)),
            delta_cross_model=float(loss.get("delta_cross_model", 1.0)),
            epsilon_native_preservation=float(loss.get("epsilon_native_preservation", 0.1)),
            transfer_rollout_length=int(loss.get("transfer_rollout_length", 50)),
            eval_interval=int(d.get("evalInterval", 500)),
            seed=int(d.get("seed", 42)),
        )


class AdapterTrainExecutor:
    """Phase B executor: train per-model adapters against a frozen substrate.

    Prerequisite: ctx.many_worlds_framework must be populated by a
    prior substrate-train stage. The substrate is frozen during this
    phase; only the per-model AdapterPair parameters are updated.

    Training happens ONE MEMBER AT A TIME (not all at once) to keep
    the training loop simple and memory-efficient. Each member's
    adapter is trained against:
      1. Its own round-trip loss (Project → substrate → Read should
         reconstruct the original residual)
      2. Cross-model transfer loss (Project this member's residual,
         Read into a peer member, measure the peer's continuation
         quality under a reference model)
      3. Native preservation regularization (adapter output_scale
         penalty)

    All three loss terms are summed with the configured weights and
    backpropagated through the adapter's parameters only. The base
    models stay frozen; the substrate stays frozen; the other members'
    adapters stay frozen.

    Output mutations:
      - Each member in ctx.many_worlds_framework now has a trained
        AdapterPair attached
      - ctx.alloy['results']['adapter_phase_b'] = per-member metrics
    """

    def execute(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Scaffold — full implementation is TODO (see SubstrateTrainExecutor).

        The per-member training loop structure:

        for member in ctx.many_worlds_framework.population:
            # Fresh optimizer for this member's adapter only
            optimizer = AdamW(member.adapter.parameters(), lr=...)

            for step in range(training_steps_per_member):
                batch = next_batch_for_this_member()

                # Round-trip: own adapter only
                with torch.no_grad():
                    residual = run_model_up_to_layer(member.model, batch, member.layer_idx)
                mu, log_var = member.adapter.project(residual)
                substrate_vec = substrate.read(mu, log_var)
                reconstructed = member.adapter.read(substrate_vec)

                # Cross-model transfer: project this member, read into a
                # peer, continue peer's inference, measure peer's ppl under
                # a reference model
                peer = pick_random_peer(ctx.many_worlds_framework, exclude=member)
                peer_residual = peer.adapter.read(substrate_vec)
                peer_continuation = continue_peer_inference(
                    peer.model, peer.layer_idx, peer_residual,
                    length=transfer_rollout_length
                )
                cross_model_loss_val = measure_continuation_quality(
                    peer_continuation, reference_model
                )

                loss, metrics = phase_b_loss(
                    reconstructed, residual, cross_model_loss_val,
                    member.adapter.project.module.output_scale,
                    config=PhaseBLossConfig(...)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        After this loop, every member has a trained adapter attached and
        the framework is ready for the many-worlds-eval stage.
        """
        raise NotImplementedError(
            "AdapterTrainExecutor.execute is a scaffold. The training "
            "loop is documented inline but not yet implemented. See "
            "scripts/many_worlds/train_adapters.py (TODO) for the "
            "real implementation."
        )


# ── Five-condition evaluation (§VII validation) ────────────────────────


@dataclass
class ManyWorldsEvalParams:
    """Parsed stage params for the many-worlds-eval stage."""

    conditions: list[str]
    heldout_corpus: str
    heldout_corpus_file: Optional[str] = None
    metric: str = "continuation_perplexity"
    judge_model: Optional[str] = None
    fuseLLM_baseline_path: Optional[str] = None
    single_moe_baseline_repo: Optional[str] = None
    sample_count: int = 200
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "ManyWorldsEvalParams":
        return cls(
            conditions=d.get("conditions", ["text_bottleneck", "substrate_transfer", "random_substrate"]),
            heldout_corpus=d["heldoutCorpus"],
            heldout_corpus_file=d.get("heldoutCorpusFile"),
            metric=d.get("metric", "continuation_perplexity"),
            judge_model=d.get("judgeModel"),
            fuseLLM_baseline_path=d.get("fuseLLMBaselinePath"),
            single_moe_baseline_repo=d.get("singleMoEBaselineRepo"),
            sample_count=int(d.get("sampleCount", 200)),
            seed=int(d.get("seed", 42)),
        )


class ManyWorldsEvalExecutor:
    """Run the §VII five-condition comparison.

    Conditions (all configurable in params.conditions):

      A - text_bottleneck      : baseline. Disable all Many-Worlds adapters,
                                 force cross-model communication through
                                 text serialization. Each model's thought
                                 is serialized to 50 tokens of text, the
                                 next model reads those tokens, continues.
                                 Measure the continuation's quality.

      B - substrate_transfer   : the Many-Worlds path. Enable all adapters,
                                 Project from source model, Read into
                                 target model via the trained substrate.
                                 Measure continuation quality.

      C - random_substrate     : structurally-paired negative baseline
                                 (Kash's §4.1.3.4 discipline anchor).
                                 Same as B but with the substrate's
                                 weights randomly re-initialized before
                                 the eval (not the trained substrate).
                                 If substrate transfer beats text bottleneck
                                 (B > A) AND substrate transfer beats random
                                 substrate (B > C) by a clear margin, the
                                 paper proceeds. Otherwise the design is
                                 refined and re-tested.

      D - fuseLLM_headtohead   : OPTIONAL comparison against FuseLLM
                                 (Wan et al. 2024) at equal compute.
                                 The closest direct prior art. Run
                                 FuseLLM's fusion on the same population
                                 at matched training cost; compare on
                                 the same held-out task suite.

      E - single_moe_baseline  : OPTIONAL comparison against a single
                                 same-size MoE (e.g. DeepSeek-V2-Lite,
                                 OLMoE) at equal compute. The dominant
                                 alternative architecture. If Many-Worlds
                                 population beats a single MoE at equal
                                 training cost, the economic argument is
                                 empirically validated.

    Output mutations:
      - ctx.alloy['results']['many_worlds_eval'] = {condition: metrics}
      - ctx.alloy['results']['priorMetricBaselines'].append(
            random_substrate_result  # the negative baseline
        )
      - ctx.alloy['results']['decision'] = "proceed" | "refine" | "fail"
        based on whether the predicted outcomes hold
    """

    def execute(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Scaffold — full implementation is TODO.

        The eval protocol for each condition:

        1. Load the held-out calibration corpus split
        2. For each sample in the split:
             - Run Condition A (text-bottleneck) and record metric
             - Run Condition B (substrate-transfer) and record metric
             - Run Condition C (random-substrate) and record metric
             - (optionally D and E)
        3. Aggregate per-condition metrics (mean, stdev, CI)
        4. Compare against predicted outcomes:
             - B > A on continuation quality?
             - B > C by at least 2x the noise floor?
        5. Set ctx.alloy['results']['decision'] based on the comparison
        6. Append the random_substrate result to priorMetricBaselines[]
           with the canonical §4.1.3.4 format so the daemon's
           propagation code (from commit e299d3c) carries it through
           to result.json.

        The v0 tiny-scale validation uses only conditions A, B, C
        (the structurally-paired discipline anchor). Conditions D
        (FuseLLM) and E (single same-size MoE) are added in v1
        production-scale runs once the tiny-scale validation passes.
        """
        raise NotImplementedError(
            "ManyWorldsEvalExecutor.execute is a scaffold. The five-condition "
            "evaluation driver is the next file to write after the training "
            "loops. See scripts/many_worlds/eval_v0.py (TODO) for the real "
            "implementation."
        )


# ── Registration ───────────────────────────────────────────────────────

# The actual stage registration with forge-alloy dispatch happens in
# scripts/stages/__init__.py (or wherever the stage registry is) by
# mapping the stage type strings to these executor classes. That
# registration is a separate commit because it requires coordinating
# with the forge-alloy schema addition for the new stage types.
# For now, these executors exist but are not yet dispatched from any
# alloy recipe — they're importable and testable in isolation.

MANY_WORLDS_STAGE_EXECUTORS = {
    "substrate-train": SubstrateTrainExecutor,
    "adapter-train": AdapterTrainExecutor,
    "many-worlds-eval": ManyWorldsEvalExecutor,
}
