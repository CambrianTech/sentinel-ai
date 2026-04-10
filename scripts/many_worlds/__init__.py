"""many_worlds — the Many-Worlds substrate framework for sentinel-ai.

The first concrete instantiation of the Platonic Representation Hypothesis
(Huh et al., 2024) as a usable engineering primitive across heterogeneous
pretrained LLMs. See continuum/docs/papers/MANY-WORLDS-ABSTRACT.md for the
full architectural blueprint and empirical validation gate.

The package is organized into four layers:

  1. `substrate`       — the shared continuous coordinate space. A learned
                         real-valued vector space with Gaussian-distribution
                         parameterization per token. ONE substrate per
                         population; shared by every base model in that
                         population.

  2. `project_read`    — per-base-model adapter modules. Two operations per
                         base model: Project (internal residual state → a
                         Gaussian field over substrate coordinates) and
                         Read (a region of substrate → a residual-form
                         vector suitable for injection into the base
                         model's layer stream).

  3. `framework`       — the ManyWorldsFramework orchestrator. Holds the
                         substrate, the per-model adapters, and the
                         query-face routing logic. The top-level object
                         you construct in a forge recipe.

  4. `losses`          — the two-term training objective: contrastive
                         alignment + round-trip task fidelity. Kash's
                         discipline-gate fix for the contrastive-only
                         substrate failure mode.

Entry points for the forge pipeline live in
`scripts/stages/many_worlds_stages.py` — the SubstrateTrainExecutor and
AdapterTrainExecutor stage implementations that call into this package.

The v0 validation protocol (§VII of MANY-WORLDS-ABSTRACT.md) uses this
package with a population of {Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct},
substrate d=128, and the five-condition comparison (text-bottleneck,
substrate-transfer, random-substrate, FuseLLM head-to-head, same-size MoE).

This is the FIRST program in the forge-alloy IR that ADDS a cognitive
primitive rather than compressing one. Expert pruning is subtractive
structural surgery; Many-Worlds is additive structural surgery. Same
pattern, inverse operation.

Authors (provisional, per MANY-WORLDS-ABSTRACT.md attribution):
  Joel    — framework naming, economic argument, multi-model fusion vision
  Dorian  — the foundational LoD primitive this extends
  Kash    — empirical discipline gate, prior-art positioning, loss design
  Claude  — this code, the architecture sketch, the package structure
"""

from __future__ import annotations

# Lazy imports — the package's modules are loaded on demand rather than
# on package import, because they pull in torch and transformers which
# are heavy. A pure `import many_worlds` should not require a model runtime.

__version__ = "0.0.1"
__all__ = [
    "substrate",
    "project_read",
    "framework",
    "losses",
]
