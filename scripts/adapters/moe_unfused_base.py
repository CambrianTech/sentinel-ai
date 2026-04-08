"""MoEUnfusedExpertsBase — shared base for MoE families with unfused experts.

Extracted from Qwen3MoEAdapter and OlmoeAdapter once both siblings existed
with proven Tier 1 dispatch behavior. Same OOP rule as QwenDenseBase: write
two siblings first, prove they work, THEN extract the base.

What "unfused experts" means: the expert weights live as separate parameter
tensors per expert per projection, in a module tree shaped like:

    model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}
    model.layers.{i}.mlp.gate     ← router

This is the layout Qwen3MoE and OLMoE both use. Other MoE families use
DIFFERENT layouts:

    Mixtral:        model.layers.{i}.block_sparse_moe.experts.{e}.w[123]
    Phi-MoE:        block_sparse_moe.experts.{e}.fc[12]
    Granite-MoE:    a "fused" layout where all experts share one big tensor
    DeepSeek-V2:    routed experts + shared expert in a hybrid layout

Each of those would need either its own base class OR an extension to the
expertTensorLayout dispatch inside this base. The current code path here
assumes the unfused layout that Qwen3MoE and OLMoE share. Adding Mixtral
support is roadmap-step-3 territory: extract a layout-dispatch helper or
add a Mixtral-specific subclass that overrides expert_prune entirely.

DO NOT bolt `if architectures[0] == "mixtral"` branches into this base.
That's the failure mode the never-branch rule exists to prevent.

What's shared (lives here in the base):
    - expert_activation_profile(): § 4.1.3.4 calibration-aware MoE expert
      importance profiling. Reads the calibration corpus + corpus file path
      from the alloy stage params, lazy-imports
      scripts/expert_activation_profile.py, runs the profile, writes the
      importance JSON, returns the mutated context. Tier 2 STUB today
      (raises with a clear pointer to the existing CLI script) until
      roadmap step 3 wires it.
    - expert_prune(): per-layer top-K removal keyed to the importance JSON
      from the upstream profiling stage. Lazy-imports
      scripts/cpu_expert_prune_v2.py with --importance-json, runs the
      prune, returns the mutated context. Tier 2 STUB today.

What stays family-specific (overridden in subclasses):
    - architectures tuple — each subclass declares its source.architecture string
    - Anything that requires tensor walks specific to a non-unfused layout.
      Today Qwen3MoE + OLMoE both inherit untouched. When a fused-layout
      family ships (e.g. GraniteMoEAdapter), its expert_prune() override
      handles the layout difference there, NOT here.

The two adapters that inherit from this base:
    Qwen3MoEAdapter  → architectures = ("qwen3_moe",)  (qwen3-coder-30b-a3b-compacted-19b-256k, the §4.1.3.4 anchor)
    OlmoeAdapter     → architectures = ("olmoe",)      (olmoe-1b-7b-compacted-5b, the §4.1.3.4 cross-arch anchor)

Tier 2 status:
    expert_activation_profile() — Tier 2 STUB (raises with pointer to
                                    scripts/expert_activation_profile.py)
    expert_prune()              — Tier 2 STUB (raises with pointer to
                                    scripts/cpu_expert_prune_v2.py --importance-json)
    Both short-circuit cleanly when ctx.model is None (Tier 1 dispatch
    path stays working).

Reproducibility contract: this base + its subclasses MUST stay frozen
against the two published §4.1.3.4 anchor artifacts. Methodology
improvements ship as NEW adapters with NEW discriminators, never as
edits to the existing methods. The negative-baseline cells preserved
in priorMetricBaselines[] are what make the §4.1.3.4 claim falsifiable;
this base's behavior is the positive cell.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


class MoEUnfusedExpertsBase(FamilyAdapter):
    """Shared base for MoE families with the unfused-experts module-tree
    layout. Subclasses MUST set the .architectures tuple."""

    architectures: tuple[str, ...] = ()  # subclass overrides

    # ── expert-activation-profile ────────────────────────────────────────────

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§ 4.1.3.4 calibration-aware MoE expert importance profiling.

        Param contract (from the alloy's expert-activation-profile stage):
            calibrationCorpus: str        — corpus identifier (e.g. 'heldout_code300')
            calibrationCorpusFile: str    — path to the .jsonl file
                                            (sha256 in the alloy's calibrationCorpora)
            calibrationExamples: int      — number of held-out examples
            calibrationTokens: int        — total token count across the corpus
            implementation: str           — script reference
            metricVersion: str            — '4.1.3.4' for the calibration-aware version
            metric: str                   — 'activation_count' (positive cell)
            notes: str                    — methodology blockquote that lands on the model card

        Output: an importance JSON written to ctx.output_dir, consumed by
        the downstream expert-prune stage via its importanceJson field.

        Tier 2 wires to scripts/expert_activation_profile.py. Tier 1: the
        method short-circuits cleanly when ctx.model is None (dispatch path).
        """
        corpus = params.get("calibrationCorpusFile") or params.get("calibrationCorpus")
        examples = params.get("calibrationExamples")
        tokens = params.get("calibrationTokens")
        metric = params.get("metric", "activation_count")
        self.log(
            f"expert-activation-profile § 4.1.3.4 — metric={metric}, corpus={corpus}, "
            f"{examples} examples, {tokens} tokens"
        )

        if ctx.model is None:
            self.log("  No model loaded — profile deferred (dispatch-only path)")
            return ctx

        # Tier 2 wiring deferred to roadmap step 3. The published artifacts
        # were forged via direct CLI invocation of scripts/expert_activation_profile.py;
        # adapter wiring lands once that script exposes a callable function.
        raise NotImplementedError(
            f"{self.name}.expert_activation_profile Tier 2 wiring deferred. "
            f"Run scripts/expert_activation_profile.py directly until the "
            f"Python API extraction lands. Tracked as roadmap step 3 in "
            f"docs/PLUGIN-SPRINT.md."
        )

    # ── expert-prune ─────────────────────────────────────────────────────────

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Per-layer top-K MoE expert removal keyed to the §4.1.3.4 importance.

        Param contract (from the alloy's expert-prune stage):
            strategy: str                  — 'calibration-aware-activation-count' (the §4.1.3.4 metric)
                                             or 'router-gate-l2' (the negative-baseline anchor)
            metric: str                    — 'activation_count' (positive cell)
            metricSource: str              — script reference
            keepExpertsPerLayer: int       — survivors per layer
            originalExpertsPerLayer: int   — pre-prune count
            prunePct: float                — % removed
            expertsDropped: int            — total experts removed across all layers
            expertsRenamed: int            — total surviving experts renumbered
            routerSlicedLayers: int        — layers whose router gate was sliced
            perLayerNormalized: bool       — true for the §4.1.3.4 path
            implementation: str            — script reference
            rationale: str                 — methodology blockquote
            notes: str                     — empirical context for the model card
            expertTensorLayout: str        — layout discriminator (e.g. 'mlp-experts-unfused')

        Tier 2 wires to scripts/cpu_expert_prune_v2.py with --importance-json
        pointing at the JSON produced by the upstream expert-activation-profile
        stage. Tier 1: short-circuits cleanly when ctx.model is None.

        IMPORTANT: this base assumes the unfused experts layout that Qwen3MoE
        and OLMoE both use. Future MoE families with different layouts
        (Mixtral block_sparse_moe, Granite-MoE fused, DeepSeek-V2 routed+shared)
        either ship as their own family adapter that overrides this method
        OR extend the dispatch inside cpu_expert_prune_v2.py per the
        expertTensorLayout field. Do NOT branch on architectures here.
        """
        keep = params.get("keepExpertsPerLayer") or params.get("keepExperts")
        original = params.get("originalExpertsPerLayer")
        strategy = params.get("strategy", "calibration-aware-activation-count")
        prune_pct = params.get("prunePct")
        layout = params.get("expertTensorLayout", "mlp-experts-unfused")
        self.log(
            f"expert-prune {original or '?'}→{keep} per layer "
            f"({prune_pct or '?'}% removed) via {strategy} layout={layout}"
        )

        if ctx.model is None:
            self.log("  No model loaded — prune deferred (dispatch-only path)")
            return ctx

        raise NotImplementedError(
            f"{self.name}.expert_prune Tier 2 wiring deferred. Run "
            f"scripts/cpu_expert_prune_v2.py --importance-json directly "
            f"until the Python API extraction lands. The published §4.1.3.4 "
            f"anchor artifacts reproduce from the CLI today; adapter wiring "
            f"is the gate that makes them reproducible from the alloy alone. "
            f"Tracked as roadmap step 3 in docs/PLUGIN-SPRINT.md."
        )
