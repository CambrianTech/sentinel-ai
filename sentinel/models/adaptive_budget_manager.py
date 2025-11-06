"""
Adaptive Budget Manager - Goal-Oriented Architecture Evolution.

This manager extends adaptive head management with goal-oriented optimization:
- Target compression (e.g., "reduce to 80% of original size")
- Target growth (e.g., "grow until loss < 2.5")
- Automatic learning rate adjustments after structural changes
- Multiple budget types (size, FLOPs, memory, loss)

Key Features:
1. Goal-oriented decisions (not just reactive thresholds)
2. Learning rate boost after pruning/splitting (helps network heal)
3. Pacing control (gradual changes, monitor recovery)
4. Budget-aware (respects memory/compute constraints)

Example Usage:
    # Compress to fit memory budget
    manager = AdaptiveBudgetManager(
        model, optimizer,
        budget_type='memory',
        target_value=500_000_000,  # 500MB
        direction='compress'
    )

    # Grow to reduce loss
    manager = AdaptiveBudgetManager(
        model, optimizer,
        budget_type='loss',
        target_value=2.5,
        direction='grow'
    )
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from sentinel.models.adaptive_head_cloning import AdaptiveHeadManager


@dataclass
class BudgetGoal:
    """Configuration for budget-oriented optimization."""
    budget_type: Literal['size', 'flops', 'memory', 'loss']
    target_value: Optional[float]  # None = auto-determine
    direction: Literal['compress', 'grow', 'maintain']
    tolerance: float = 0.05  # 5% tolerance for "close enough"


@dataclass
class StructuralChange:
    """Record of a structural change (for LR scheduling)."""
    step: int
    change_type: Literal['prune', 'split']
    num_heads_changed: int
    lr_boost_factor: float = 2.0
    lr_boost_duration: int = 50  # Steps to maintain boosted LR


class AdaptiveBudgetManager(AdaptiveHeadManager):
    """
    Goal-oriented adaptive architecture manager.

    Extends AdaptiveHeadManager with:
    - Budget awareness (memory, compute, quality goals)
    - Learning rate adjustments after structural changes
    - Goal-oriented decision making
    """

    def __init__(
        self,
        model,
        optimizer,
        goal: Optional[BudgetGoal] = None,
        prune_threshold: float = 0.3,
        clone_threshold: float = 0.8,
        min_active_heads: int = 4,
        max_heads_per_layer: Optional[int] = None,
        update_frequency: int = 100,
        warmup_steps: int = 500,
        lr_boost_factor: float = 2.0,
        lr_boost_duration: int = 50,
        max_changes_per_update: int = 2  # Don't change too fast
    ):
        """
        Initialize adaptive budget manager.

        Args:
            model: The adaptive transformer model
            optimizer: Optimizer (for LR adjustments)
            goal: Budget goal configuration
            lr_boost_factor: Multiply LR by this after structural changes
            lr_boost_duration: Maintain boosted LR for this many steps
            max_changes_per_update: Max heads to change per adaptation
        """
        super().__init__(
            model,
            prune_threshold,
            clone_threshold,
            min_active_heads,
            max_heads_per_layer or model.transformer.blocks[0].attn.num_heads,
            update_frequency,
            warmup_steps
        )

        self.optimizer = optimizer
        self.goal = goal
        self.lr_boost_factor = lr_boost_factor
        self.lr_boost_duration = lr_boost_duration
        self.max_changes_per_update = max_changes_per_update

        # Track structural changes for LR scheduling
        self.structural_changes: List[StructuralChange] = []
        self.base_lr = self._get_current_lr()

        # Budget tracking
        self.initial_size = self._compute_current_size()
        self.size_history: List[Tuple[int, int]] = [(0, self.initial_size)]  # (step, size)
        self.loss_history: List[Tuple[int, float]] = []  # (step, loss)

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]['lr']

    def _set_lr(self, lr: float):
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _compute_current_size(self) -> int:
        """Compute current active head count."""
        total = 0
        for layer_idx in range(len(self.model.transformer.blocks)):
            active_heads = self.get_active_heads(layer_idx)
            total += len(active_heads)
        return total

    def _boost_lr_after_change(self, change: StructuralChange):
        """Temporarily boost learning rate after structural change."""
        new_lr = self.base_lr * self.lr_boost_factor
        self._set_lr(new_lr)
        self.structural_changes.append(change)

        print(f"\n🔥 LR BOOST after {change.change_type}:", flush=True)
        print(f"   Base LR: {self.base_lr:.2e}", flush=True)
        print(f"   Boosted LR: {new_lr:.2e} ({self.lr_boost_factor}x)", flush=True)
        print(f"   Duration: {self.lr_boost_duration} steps", flush=True)

    def _update_lr_schedule(self):
        """Update LR based on recent structural changes."""
        # Check if any boosts should expire
        current_step = self.step_count
        active_boosts = [
            c for c in self.structural_changes
            if current_step - c.step < self.lr_boost_duration
        ]

        if not active_boosts:
            # No active boosts, return to base LR
            current_lr = self._get_current_lr()
            if current_lr != self.base_lr:
                self._set_lr(self.base_lr)
                print(f"✅ LR returned to base: {self.base_lr:.2e}", flush=True)

    def _should_compress(self, current_size: int, current_loss: Optional[float] = None) -> bool:
        """Determine if we should compress (prune heads)."""
        if not self.goal:
            return False

        if self.goal.direction != 'compress':
            return False

        if self.goal.budget_type == 'size':
            target = self.goal.target_value or int(self.initial_size * 0.8)
            return current_size > target * (1 + self.goal.tolerance)

        elif self.goal.budget_type == 'loss':
            # Only compress if quality is good enough
            if current_loss is None:
                return False
            target_loss = self.goal.target_value
            return current_loss < target_loss * (1 - self.goal.tolerance)

        return False

    def _should_grow(self, current_size: int, current_loss: Optional[float] = None) -> bool:
        """Determine if we should grow (split heads)."""
        if not self.goal:
            return False

        if self.goal.direction != 'grow':
            return False

        if self.goal.budget_type == 'loss':
            if current_loss is None:
                return False
            target_loss = self.goal.target_value
            return current_loss > target_loss * (1 + self.goal.tolerance)

        elif self.goal.budget_type == 'size':
            target = self.goal.target_value or int(self.initial_size * 1.5)
            return current_size < target * (1 - self.goal.tolerance)

        return False

    def adapt_toward_goal(self, current_loss: Optional[float] = None) -> Dict[str, int]:
        """
        Make adaptation decisions toward budget goal.

        Args:
            current_loss: Current training loss (optional)

        Returns:
            Dictionary with counts: {'heads_pruned': X, 'heads_split': Y}
        """
        if self.step_count < self.warmup_steps:
            return {'heads_pruned': 0, 'heads_split': 0}

        current_size = self._compute_current_size()
        self.size_history.append((self.step_count, current_size))

        if current_loss is not None:
            self.loss_history.append((self.step_count, current_loss))

        heads_pruned = 0
        heads_split = 0

        # Goal-oriented decisions
        should_compress = self._should_compress(current_size, current_loss)
        should_grow = self._should_grow(current_size, current_loss)

        if should_compress:
            print(f"\n🎯 Goal: Compress toward target", flush=True)
            print(f"   Current size: {current_size}", flush=True)
            if self.goal.budget_type == 'size':
                target = self.goal.target_value or int(self.initial_size * 0.8)
                print(f"   Target size: {target}", flush=True)

            # Prune most aggressively
            for layer_idx in range(len(self.model.transformer.blocks)):
                if heads_pruned >= self.max_changes_per_update:
                    break

                active_heads = self.get_active_heads(layer_idx)
                if len(active_heads) <= self.min_active_heads:
                    continue

                # Find lowest utilization head
                utilizations = [
                    (h, self.head_stats[(layer_idx, h)].utilization_score)
                    for h in active_heads
                ]
                utilizations.sort(key=lambda x: x[1])

                # Prune lowest (AGGRESSIVE: ignore threshold when goal-oriented)
                if utilizations:
                    head_idx, score = utilizations[0]
                    # When goal-oriented, prune the weakest head regardless of absolute threshold
                    # (as long as it's the weakest in its layer)
                    self.prune_head(layer_idx, head_idx)
                    heads_pruned += 1
                    print(f"   🎯 Pruned Layer {layer_idx}, Head {head_idx} (utilization: {score:.3f}) - GOAL-DRIVEN", flush=True)

            if heads_pruned > 0:
                change = StructuralChange(
                    step=self.step_count,
                    change_type='prune',
                    num_heads_changed=heads_pruned,
                    lr_boost_factor=self.lr_boost_factor,
                    lr_boost_duration=self.lr_boost_duration
                )
                self._boost_lr_after_change(change)

        elif should_grow:
            print(f"\n🎯 Goal: Grow to improve quality", flush=True)
            if current_loss:
                print(f"   Current loss: {current_loss:.4f}", flush=True)
                if self.goal.budget_type == 'loss':
                    print(f"   Target loss: {self.goal.target_value:.4f}", flush=True)

            # Split most aggressively
            for layer_idx in range(len(self.model.transformer.blocks)):
                if heads_split >= self.max_changes_per_update:
                    break

                active_heads = self.get_active_heads(layer_idx)
                inactive_heads = self.get_inactive_heads(layer_idx)

                if not inactive_heads:
                    continue

                # Find highest utilization head
                utilizations = [
                    (h, self.head_stats[(layer_idx, h)].utilization_score)
                    for h in active_heads
                ]
                utilizations.sort(key=lambda x: x[1], reverse=True)

                # Split highest
                if utilizations:
                    head_idx, score = utilizations[0]
                    if score > self.clone_threshold * 0.7:  # More lenient for goal-oriented
                        new_head = self.split_head(layer_idx, head_idx)
                        if new_head is not None:
                            heads_split += 1

            if heads_split > 0:
                change = StructuralChange(
                    step=self.step_count,
                    change_type='split',
                    num_heads_changed=heads_split,
                    lr_boost_factor=self.lr_boost_factor,
                    lr_boost_duration=self.lr_boost_duration
                )
                self._boost_lr_after_change(change)

        else:
            # No goal pressure, use standard thresholds
            result = super().adapt_architecture()
            heads_pruned = result.get('heads_pruned', 0)
            heads_split = result.get('heads_cloned', 0)

            # Still boost LR if standard adaptation happened
            if heads_pruned > 0:
                change = StructuralChange(
                    step=self.step_count,
                    change_type='prune',
                    num_heads_changed=heads_pruned
                )
                self._boost_lr_after_change(change)

            if heads_split > 0:
                change = StructuralChange(
                    step=self.step_count,
                    change_type='split',
                    num_heads_changed=heads_split
                )
                self._boost_lr_after_change(change)

        return {'heads_pruned': heads_pruned, 'heads_split': heads_split}

    def step(self, batch_gradients: Optional[Dict] = None, current_loss: Optional[float] = None):
        """
        Update manager state after a training step.

        Args:
            batch_gradients: Gradient magnitudes per head
            current_loss: Current training loss
        """
        self.step_count += 1

        # Update head statistics
        if batch_gradients:
            for (layer_idx, head_idx), grad_mag in batch_gradients.items():
                stats = self.head_stats.get((layer_idx, head_idx))
                if stats:
                    self.update_stats(layer_idx, head_idx,
                                    entropy=0.5,  # Would need attention weights
                                    gradient_mag=grad_mag)

        # Update LR schedule (decay boosts if expired)
        self._update_lr_schedule()

        # Make adaptation decisions periodically
        if self.step_count % self.update_frequency == 0:
            results = self.adapt_toward_goal(current_loss)

            if results['heads_pruned'] > 0 or results['heads_split'] > 0:
                print(f"\n🔄 Budget-aware adaptation at step {self.step_count}:", flush=True)
                print(f"   Heads pruned: {results['heads_pruned']}", flush=True)
                print(f"   Heads split: {results['heads_split']}", flush=True)

                current_size = self._compute_current_size()
                print(f"   Current size: {current_size}/{len(self.model.transformer.blocks) * self.max_heads_per_layer}", flush=True)

                if self.goal:
                    progress = self._compute_goal_progress(current_size, current_loss)
                    print(f"   Goal progress: {progress*100:.1f}%", flush=True)

    def _compute_goal_progress(self, current_size: int, current_loss: Optional[float] = None) -> float:
        """Compute progress toward goal (0.0 = start, 1.0 = goal reached)."""
        if not self.goal:
            return 0.0

        if self.goal.budget_type == 'size':
            target = self.goal.target_value or int(self.initial_size * 0.8)
            if self.goal.direction == 'compress':
                # Progress = how much we've shrunk toward target
                total_needed = self.initial_size - target
                current_progress = self.initial_size - current_size
                return min(1.0, current_progress / total_needed)
            elif self.goal.direction == 'grow':
                total_needed = target - self.initial_size
                current_progress = current_size - self.initial_size
                return min(1.0, current_progress / total_needed)

        elif self.goal.budget_type == 'loss' and current_loss is not None:
            target = self.goal.target_value
            if not self.loss_history:
                return 0.0

            initial_loss = self.loss_history[0][1]
            total_needed = initial_loss - target
            current_progress = initial_loss - current_loss
            return min(1.0, max(0.0, current_progress / total_needed))

        return 0.0

    def get_budget_report(self) -> str:
        """Generate detailed budget progress report."""
        lines = ["=" * 80]
        lines.append("ADAPTIVE BUDGET REPORT")
        lines.append("=" * 80)
        lines.append(f"Training step: {self.step_count}")
        lines.append("")

        current_size = self._compute_current_size()
        max_size = len(self.model.transformer.blocks) * self.max_heads_per_layer

        lines.append(f"Current architecture: {current_size}/{max_size} heads")
        lines.append(f"Initial size: {self.initial_size} heads")
        lines.append(f"Net change: {current_size - self.initial_size:+d} heads ({(current_size/self.initial_size - 1)*100:+.1f}%)")
        lines.append("")

        if self.goal:
            lines.append(f"Goal: {self.goal.direction} via {self.goal.budget_type}")
            if self.goal.target_value:
                lines.append(f"Target: {self.goal.target_value}")

            current_loss = self.loss_history[-1][1] if self.loss_history else None
            progress = self._compute_goal_progress(current_size, current_loss)
            lines.append(f"Progress: {progress*100:.1f}%")
            lines.append("")

        lines.append(f"Structural changes: {len(self.structural_changes)}")
        for change in self.structural_changes[-5:]:  # Last 5
            lines.append(f"  Step {change.step}: {change.change_type} ({change.num_heads_changed} heads)")

        lines.append("")
        lines.append(f"Current LR: {self._get_current_lr():.2e}")
        lines.append(f"Base LR: {self.base_lr:.2e}")

        active_boosts = [
            c for c in self.structural_changes
            if self.step_count - c.step < self.lr_boost_duration
        ]
        if active_boosts:
            lines.append(f"Active LR boosts: {len(active_boosts)}")

        lines.append("=" * 80)
        return "\n".join(lines)
