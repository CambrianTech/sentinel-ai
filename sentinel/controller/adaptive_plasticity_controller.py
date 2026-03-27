"""
Self-Directed Plasticity Controller

Eliminates human-specified hyperparameters from the plasticity cycle.
The controller observes model state and decides:
  - How much to prune (based on head redundancy)
  - Which strategy to use (based on past recovery success)
  - When to stop retraining (loss plateau detection)
  - Whether to cycle again (based on remaining redundancy)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlasticityState:
    """Observable state of the model's attention architecture."""
    entropy_per_head: np.ndarray          # [layers, heads] — Shannon entropy
    gate_values: Optional[np.ndarray]     # [layers, heads] — gate magnitudes (if gated)
    eval_loss: float                      # current evaluation loss
    eval_perplexity: float                # current perplexity
    total_heads: int                      # total head count
    active_heads: int                     # heads with gate > threshold
    layer_count: int
    heads_per_layer: int


@dataclass
class PlasticityAction:
    """Controller output: what to do next in the plasticity cycle."""
    should_prune: bool
    pruning_ratio: float                  # 0.0 to 1.0
    strategy: str                         # "entropy", "gradient", "random", "combined"
    max_training_steps: int               # upper bound (plateau detection may stop earlier)
    reason: str                           # human-readable explanation of the decision


@dataclass
class CycleRecord:
    """Record of a completed plasticity cycle for learning."""
    strategy: str
    pruning_ratio: float
    training_steps_used: int
    baseline_ppl: float
    post_prune_ppl: float
    post_train_ppl: float
    recovery_ratio: float                 # (post_prune - post_train) / (post_prune - baseline)


class AdaptivePlasticityController:
    """
    Self-directed plasticity controller.

    Observes model state and decides plasticity actions without
    human-specified hyperparameters.
    """

    def __init__(
        self,
        min_pruning_ratio: float = 0.05,
        max_pruning_ratio: float = 0.50,
        redundancy_threshold: float = 0.15,
        plateau_patience: int = 50,
        plateau_min_delta: float = 0.001,
        max_cycles: int = 10,
    ):
        self.min_pruning_ratio = min_pruning_ratio
        self.max_pruning_ratio = max_pruning_ratio
        self.redundancy_threshold = redundancy_threshold
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.max_cycles = max_cycles

        self.history: list[CycleRecord] = []
        self.cycle_count = 0

    def assess_redundancy(self, state: PlasticityState) -> float:
        """
        Measure head redundancy from entropy distribution.

        High redundancy = many heads with similar entropy values.
        Returns a ratio from 0.0 (no redundancy) to 1.0 (all identical).
        """
        entropy = state.entropy_per_head.flatten()
        if len(entropy) < 2:
            return 0.0

        # Coefficient of variation — low CV means heads are similar
        mean_entropy = np.mean(entropy)
        if mean_entropy < 1e-8:
            return 0.0

        cv = np.std(entropy) / mean_entropy

        # Also check pairwise similarity within layers
        layer_redundancies = []
        for layer_idx in range(state.layer_count):
            layer_entropy = state.entropy_per_head[layer_idx]
            if len(layer_entropy) < 2:
                continue

            # What fraction of heads are within redundancy_threshold of another?
            redundant = 0
            for i in range(len(layer_entropy)):
                for j in range(i + 1, len(layer_entropy)):
                    if abs(layer_entropy[i] - layer_entropy[j]) < self.redundancy_threshold:
                        redundant += 1

            total_pairs = len(layer_entropy) * (len(layer_entropy) - 1) / 2
            layer_redundancies.append(redundant / total_pairs if total_pairs > 0 else 0.0)

        # Combine: high pairwise redundancy + low CV = high overall redundancy
        pairwise_redundancy = np.mean(layer_redundancies) if layer_redundancies else 0.0

        # Normalize CV to 0-1 range (CV of 0 = redundant, CV of 1+ = diverse)
        cv_score = max(0.0, 1.0 - cv)

        return 0.6 * pairwise_redundancy + 0.4 * cv_score

    def select_pruning_ratio(self, redundancy: float) -> float:
        """
        Determine how much to prune based on measured redundancy.

        More redundancy → more aggressive pruning.
        """
        # Linear interpolation between min and max, clamped
        ratio = self.min_pruning_ratio + redundancy * (self.max_pruning_ratio - self.min_pruning_ratio)
        return round(min(max(ratio, self.min_pruning_ratio), self.max_pruning_ratio), 2)

    def select_strategy(self) -> str:
        """
        Choose pruning strategy based on past cycle performance.

        If no history, default to entropy (our best performer).
        Otherwise, pick the strategy with best recovery ratio.
        """
        if not self.history:
            return "entropy"

        # Group by strategy, compute mean recovery ratio
        strategy_scores: dict[str, list[float]] = {}
        for record in self.history:
            if record.strategy not in strategy_scores:
                strategy_scores[record.strategy] = []
            strategy_scores[record.strategy].append(record.recovery_ratio)

        best_strategy = "entropy"
        best_score = -float("inf")
        for strategy, scores in strategy_scores.items():
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_strategy = strategy

        return best_strategy

    def estimate_training_budget(self, pruning_ratio: float) -> int:
        """
        Estimate training steps needed based on pruning aggressiveness.

        More aggressive pruning → more training needed for recovery.
        """
        # Base: 200 steps per 10% pruning
        base_steps = int(pruning_ratio * 2000)

        # If we have history, adjust based on observed plateau timing
        if self.history:
            avg_steps = np.mean([r.training_steps_used for r in self.history])
            # Blend historical average with estimate
            base_steps = int(0.5 * base_steps + 0.5 * avg_steps * (pruning_ratio / 0.3))

        return max(200, min(5000, base_steps))

    def decide(self, state: PlasticityState) -> PlasticityAction:
        """
        Main decision function. Observes model state, returns action.

        Uses three stopping criteria:
        1. Max cycles reached
        2. Redundancy too low (model is efficient)
        3. Recovery ratio declining (model is being damaged faster than it heals)

        Adapts pruning ratio based on recent recovery performance.
        """
        self.cycle_count += 1

        if self.cycle_count > self.max_cycles:
            return PlasticityAction(
                should_prune=False,
                pruning_ratio=0.0,
                strategy="none",
                max_training_steps=0,
                reason=f"Maximum cycle count ({self.max_cycles}) reached. Stopping."
            )

        # Quality-aware stopping: check if recent cycles are degrading
        if len(self.history) >= 2:
            recent = self.history[-2:]
            avg_recovery = np.mean([r.recovery_ratio for r in recent])

            if avg_recovery < 0.3:
                return PlasticityAction(
                    should_prune=False,
                    pruning_ratio=0.0,
                    strategy="none",
                    max_training_steps=0,
                    reason=f"Recovery ratio declining ({avg_recovery:.1%} avg over last 2 cycles). "
                           f"Model is at structural limit. Stopping."
                )

            # Also stop if post-train PPL is getting worse cycle over cycle
            if len(self.history) >= 3:
                ppl_trend = [r.post_train_ppl for r in self.history[-3:]]
                if all(ppl_trend[i] > ppl_trend[i-1] for i in range(1, len(ppl_trend))):
                    return PlasticityAction(
                        should_prune=False,
                        pruning_ratio=0.0,
                        strategy="none",
                        max_training_steps=0,
                        reason=f"Perplexity degrading for 3 consecutive cycles "
                               f"({ppl_trend[0]:.2f} → {ppl_trend[1]:.2f} → {ppl_trend[2]:.2f}). Stopping."
                    )

        # 1. Assess redundancy
        redundancy = self.assess_redundancy(state)

        if redundancy < 0.05:
            return PlasticityAction(
                should_prune=False,
                pruning_ratio=0.0,
                strategy="none",
                max_training_steps=0,
                reason=f"Redundancy too low ({redundancy:.3f}). Model is already efficient."
            )

        # 2. Determine pruning ratio — adapt based on recent recovery
        base_ratio = self.select_pruning_ratio(redundancy)

        if self.history:
            last_recovery = self.history[-1].recovery_ratio
            if last_recovery < 0.5:
                # Recovery is weak — back off on pruning
                base_ratio = max(self.min_pruning_ratio, base_ratio * 0.5)
            elif last_recovery < 0.75:
                # Recovery is moderate — slight reduction
                base_ratio = max(self.min_pruning_ratio, base_ratio * 0.75)
            # Recovery > 0.75 — keep current ratio (model is handling it)

        pruning_ratio = round(base_ratio, 2)

        # 3. Select best strategy from history
        strategy = self.select_strategy()

        # 4. Estimate training budget
        max_steps = self.estimate_training_budget(pruning_ratio)

        reason = (
            f"Cycle {self.cycle_count}: "
            f"redundancy={redundancy:.3f} → prune {pruning_ratio:.0%} "
            f"using {strategy} strategy, "
            f"up to {max_steps} training steps"
        )
        if self.history:
            reason += f" (last recovery: {self.history[-1].recovery_ratio:.1%})"

        return PlasticityAction(
            should_prune=True,
            pruning_ratio=pruning_ratio,
            strategy=strategy,
            max_training_steps=max_steps,
            reason=reason
        )

    def record_cycle(
        self,
        strategy: str,
        pruning_ratio: float,
        training_steps_used: int,
        baseline_ppl: float,
        post_prune_ppl: float,
        post_train_ppl: float,
    ):
        """Record a completed cycle for future decision-making."""
        gap = post_prune_ppl - baseline_ppl
        recovery = (post_prune_ppl - post_train_ppl) / gap if gap > 0 else 0.0

        record = CycleRecord(
            strategy=strategy,
            pruning_ratio=pruning_ratio,
            training_steps_used=training_steps_used,
            baseline_ppl=baseline_ppl,
            post_prune_ppl=post_prune_ppl,
            post_train_ppl=post_train_ppl,
            recovery_ratio=recovery,
        )
        self.history.append(record)

    def summary(self) -> str:
        """Human-readable summary of controller state and history."""
        lines = [f"AdaptivePlasticityController — {len(self.history)} cycles completed"]
        for i, record in enumerate(self.history):
            lines.append(
                f"  Cycle {i+1}: {record.strategy} @ {record.pruning_ratio:.0%}, "
                f"{record.training_steps_used} steps, "
                f"ppl {record.baseline_ppl:.2f} → {record.post_prune_ppl:.2f} → {record.post_train_ppl:.2f} "
                f"(recovery: {record.recovery_ratio:.1%})"
            )
        return "\n".join(lines)


class PlateauDetector:
    """
    Detect loss plateau to stop training early.

    Replaces fixed step counts with adaptive stopping.
    """

    def __init__(self, patience: int = 50, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.steps_without_improvement = 0

    def step(self, eval_loss: float) -> bool:
        """
        Record a loss value. Returns True if training should stop.
        """
        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement >= self.patience

    def reset(self):
        """Reset for a new training phase."""
        self.best_loss = float("inf")
        self.steps_without_improvement = 0
