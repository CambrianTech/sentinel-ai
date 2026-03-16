"""
Adaptive Head Cloning and Pruning for Transformer Models.

This module implements bidirectional adaptation:
- Prune under-utilized heads to save computation
- Clone over-utilized heads to increase capacity dynamically

The system tracks head utilization during training and automatically
adjusts the architecture for optimal performance.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HeadStats:
    """Statistics for tracking head utilization."""
    attention_entropy: float  # How diverse the attention pattern is
    gradient_magnitude: float  # How much the head affects the loss
    utilization_score: float  # Combined metric for pruning/cloning decisions
    num_times_cloned: int = 0  # Track clone lineage


class AdaptiveHeadManager:
    """
    Manages dynamic pruning and cloning of attention heads.

    Key features:
    - Tracks head utilization during training
    - Prunes under-utilized heads
    - Clones over-utilized heads
    - Maintains architectural constraints (max heads, min heads)
    """

    def __init__(
        self,
        model,
        prune_threshold: float = 0.3,  # Below this = prune
        clone_threshold: float = 0.8,  # Above this = clone
        min_active_heads: int = 4,     # Never go below this per layer
        max_heads_per_layer: int = 16, # Architecture limit
        update_frequency: int = 100,    # Update decisions every N batches
        warmup_steps: int = 500         # Don't adapt during warmup
    ):
        self.model = model
        self.prune_threshold = prune_threshold
        self.clone_threshold = clone_threshold
        self.min_active_heads = min_active_heads
        self.max_heads_per_layer = max_heads_per_layer
        self.update_frequency = update_frequency
        self.warmup_steps = warmup_steps

        self.step_count = 0
        self.head_stats: Dict[Tuple[int, int], HeadStats] = {}

        # Track head lineage (which heads were cloned from which)
        self.head_lineage: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

        # Initialize stats for all heads
        self._initialize_stats()

    def _initialize_stats(self):
        """Initialize tracking for all heads."""
        for layer_idx, layer in enumerate(self.model.transformer.blocks):
            for head_idx in range(layer.attn.num_heads):
                self.head_stats[(layer_idx, head_idx)] = HeadStats(
                    attention_entropy=0.5,  # Neutral start
                    gradient_magnitude=0.5,
                    utilization_score=0.5
                )
                self.head_lineage[(layer_idx, head_idx)] = None  # Original head

    def update_stats(self, layer_idx: int, head_idx: int,
                     gate_value: float, gradient_mag: float):
        """Update statistics for a specific head.

        Args:
            layer_idx: Layer index
            head_idx: Head index
            gate_value: Current gate value (raw, not normalized)
            gradient_mag: Gradient magnitude for this head's gate
        """
        # Exponential moving average
        alpha = 0.1
        stats = self.head_stats[(layer_idx, head_idx)]

        stats.attention_entropy = (1 - alpha) * stats.attention_entropy + alpha * gate_value
        stats.gradient_magnitude = (1 - alpha) * stats.gradient_magnitude + alpha * gradient_mag

        # Utilization score: gate value IS the primary signal
        # Gate > 1.0 means the head is growing beyond its initial capacity
        # Gate near 0 means the head is dying
        # Gradient magnitude is secondary — high gradient = head matters to loss
        stats.utilization_score = 0.8 * stats.attention_entropy + 0.2 * stats.gradient_magnitude

    def get_active_heads(self, layer_idx: int) -> List[int]:
        """Get list of active (non-pruned) heads in a layer."""
        layer = self.model.transformer.blocks[layer_idx]
        return [h for h in range(layer.attn.num_heads)
                if layer.attn.gate.data[h] > 0.1]

    def get_inactive_heads(self, layer_idx: int) -> List[int]:
        """Get list of inactive (pruned) heads in a layer."""
        layer = self.model.transformer.blocks[layer_idx]
        return [h for h in range(layer.attn.num_heads)
                if layer.attn.gate.data[h] <= 0.1]

    def split_head(self, layer_idx: int, parent_head_idx: int) -> Optional[int]:
        """
        Split an over-utilized head via TRUE mitosis (cell division).

        The parent head is divided in HALF:
        - Parent weights → 0.5 * original
        - New head weights → 0.5 * original
        - Combined output = original (continuous)
        - Both heads now have 2x headroom to grow

        This is the real "division of labor" - both heads start identical
        but with spare capacity, then diverge during training.

        Args:
            layer_idx: Layer containing the overloaded head
            parent_head_idx: Index of head to split

        Returns:
            Index of new split head, or None if no slots available
        """
        layer = self.model.transformer.blocks[layer_idx]
        inactive_heads = self.get_inactive_heads(layer_idx)

        if not inactive_heads:
            print(f"⚠️  No inactive head slots in layer {layer_idx} for splitting", flush=True)
            return None

        new_head_idx = inactive_heads[0]
        head_dim = layer.attn.head_dim
        embed_dim = layer.attn.embed_dim

        with torch.no_grad():
            # Calculate slice positions in fused QKV projection
            parent_q_start = parent_head_idx * head_dim
            parent_k_start = parent_q_start + embed_dim
            parent_v_start = parent_k_start + embed_dim

            new_q_start = new_head_idx * head_dim
            new_k_start = new_q_start + embed_dim
            new_v_start = new_k_start + embed_dim

            # CLONE QKV weights to new head (identical copy — NOT halved!)
            # QKV must stay intact because softmax(Q@K^T) is nonlinear:
            # softmax(0.5*scores) ≠ softmax(scores), so halving QKV breaks output continuity.
            # NOTE: c_attn is nn.Linear → weight shape is [out_features, in_features] = [3*embed_dim, embed_dim]
            # Per-head weights are in ROWS (dim 0), not columns.
            layer.attn.c_attn.weight.data[new_q_start:new_q_start+head_dim, :] = \
                layer.attn.c_attn.weight.data[parent_q_start:parent_q_start+head_dim, :].clone()
            layer.attn.c_attn.bias.data[new_q_start:new_q_start+head_dim] = \
                layer.attn.c_attn.bias.data[parent_q_start:parent_q_start+head_dim].clone()

            layer.attn.c_attn.weight.data[new_k_start:new_k_start+head_dim, :] = \
                layer.attn.c_attn.weight.data[parent_k_start:parent_k_start+head_dim, :].clone()
            layer.attn.c_attn.bias.data[new_k_start:new_k_start+head_dim] = \
                layer.attn.c_attn.bias.data[parent_k_start:parent_k_start+head_dim].clone()

            layer.attn.c_attn.weight.data[new_v_start:new_v_start+head_dim, :] = \
                layer.attn.c_attn.weight.data[parent_v_start:parent_v_start+head_dim, :].clone()
            layer.attn.c_attn.bias.data[new_v_start:new_v_start+head_dim] = \
                layer.attn.c_attn.bias.data[parent_v_start:parent_v_start+head_dim].clone()

            # Clone output projection, then HALVE both (the mitosis step!)
            # Only output weights get halved: parent contributes 0.5, child contributes 0.5,
            # combined output = original (continuous). Both start identical then diverge.
            layer.attn.W_o[new_head_idx].data = \
                layer.attn.W_o[parent_head_idx].data.clone()
            layer.attn.b_o[new_head_idx].data = \
                layer.attn.b_o[parent_head_idx].data.clone()

            layer.attn.W_o[parent_head_idx].data *= 0.5
            layer.attn.b_o[parent_head_idx].data *= 0.5
            layer.attn.W_o[new_head_idx].data *= 0.5
            layer.attn.b_o[new_head_idx].data *= 0.5

            # Both heads start at FULL gate strength (they combine to original output)
            layer.attn.gate.data[new_head_idx] = layer.attn.gate.data[parent_head_idx]

            # Make gates trainable
            layer.attn.gate.requires_grad = True

        # Update tracking - this is a true split (mitosis)
        self.head_lineage[(layer_idx, new_head_idx)] = (layer_idx, parent_head_idx)
        parent_stats = self.head_stats[(layer_idx, parent_head_idx)]

        # Both heads inherit the parent's utilization (they're identical initially)
        self.head_stats[(layer_idx, new_head_idx)] = HeadStats(
            attention_entropy=parent_stats.attention_entropy,
            gradient_magnitude=parent_stats.gradient_magnitude,
            utilization_score=parent_stats.utilization_score,
            num_times_cloned=0
        )
        parent_stats.num_times_cloned += 1

        print(f"🧬 Split head {parent_head_idx} → [{parent_head_idx}, {new_head_idx}] in layer {layer_idx}", flush=True)
        print(f"   Original utilization: {parent_stats.utilization_score:.3f}", flush=True)
        print(f"   Both heads now at 50% capacity → 2x headroom to diverge!", flush=True)
        print(f"   Output remains continuous (0.5 + 0.5 = 1.0)", flush=True)

        return new_head_idx

    def prune_head(self, layer_idx: int, head_idx: int):
        """Prune an under-utilized head."""
        layer = self.model.transformer.blocks[layer_idx]

        # Check minimum constraint
        active_heads = self.get_active_heads(layer_idx)
        if len(active_heads) <= self.min_active_heads:
            print(f"⚠️  Cannot prune - already at minimum {self.min_active_heads} heads in layer {layer_idx}", flush=True)
            return

        # Set gate to 0 (disables the head)
        with torch.no_grad():
            layer.attn.gate.data[head_idx] = 0.0

        stats = self.head_stats[(layer_idx, head_idx)]
        print(f"✂️  Pruned head {head_idx} in layer {layer_idx}", flush=True)
        print(f"   Utilization: {stats.utilization_score:.3f} (threshold: {self.prune_threshold})", flush=True)

    def adapt_architecture(self) -> Dict[str, int]:
        """
        Make pruning/cloning decisions based on current head utilization.

        Returns:
            Dictionary with counts: {'heads_pruned': X, 'heads_cloned': Y}
        """
        if self.step_count < self.warmup_steps:
            return {'heads_pruned': 0, 'heads_cloned': 0}

        heads_pruned = 0
        heads_cloned = 0

        for layer_idx, layer in enumerate(self.model.transformer.blocks):
            active_heads = self.get_active_heads(layer_idx)

            # Collect utilization scores for active heads
            utilization = [(h, self.head_stats[(layer_idx, h)].utilization_score)
                          for h in active_heads]
            utilization.sort(key=lambda x: x[1])  # Sort by utilization (ascending)

            # Prune lowest utilization head if below threshold
            if len(utilization) > self.min_active_heads:
                head_idx, score = utilization[0]
                if score < self.prune_threshold:
                    self.prune_head(layer_idx, head_idx)
                    heads_pruned += 1

            # Split highest utilization head if above threshold and slots available
            if len(utilization) > 0:
                head_idx, score = utilization[-1]
                if score > self.clone_threshold:
                    inactive_heads = self.get_inactive_heads(layer_idx)
                    if inactive_heads and len(active_heads) < self.max_heads_per_layer:
                        new_head_idx = self.split_head(layer_idx, head_idx)
                        if new_head_idx is not None:
                            heads_cloned += 1

        return {'heads_pruned': heads_pruned, 'heads_cloned': heads_cloned}

    def step(self, batch_gradients: Optional[Dict] = None):
        """
        Update manager state after a training step.

        Args:
            batch_gradients: Optional dictionary of gradient magnitudes per head
        """
        self.step_count += 1

        # Update statistics from gradients AND gate values
        if batch_gradients:
            for (layer_idx, head_idx), grad_mag in batch_gradients.items():
                stats = self.head_stats.get((layer_idx, head_idx))
                if stats:
                    # Use actual gate value as utilization signal
                    # Gate > 1.0 = high utilization (growing), gate near 0 = low (dying)
                    layer = self.model.transformer.blocks[layer_idx]
                    gate_value = max(layer.attn.gate.data[head_idx].item(), 0.0)
                    self.update_stats(layer_idx, head_idx,
                                    gate_value=gate_value,
                                    gradient_mag=grad_mag)

        # Make adaptation decisions periodically
        if self.step_count % self.update_frequency == 0:
            results = self.adapt_architecture()
            if results['heads_pruned'] > 0 or results['heads_cloned'] > 0:
                print(f"\n🔄 Architecture adapted at step {self.step_count}:", flush=True)
                print(f"   Heads pruned: {results['heads_pruned']}", flush=True)
                print(f"   Heads cloned: {results['heads_cloned']}", flush=True)
                self._print_summary()

    def _print_summary(self):
        """Print current architecture summary."""
        for layer_idx, layer in enumerate(self.model.transformer.blocks):
            active_heads = self.get_active_heads(layer_idx)
            total_heads = layer.attn.num_heads
            print(f"   Layer {layer_idx}: {len(active_heads)}/{total_heads} heads active", flush=True)

    def get_architecture_report(self) -> str:
        """Generate detailed report of current architecture."""
        lines = ["=" * 80]
        lines.append("ADAPTIVE ARCHITECTURE REPORT")
        lines.append("=" * 80)
        lines.append(f"Training step: {self.step_count}")
        lines.append(f"Warmup: {'Complete' if self.step_count >= self.warmup_steps else f'{self.step_count}/{self.warmup_steps}'}")
        lines.append("")

        for layer_idx, layer in enumerate(self.model.transformer.blocks):
            active_heads = self.get_active_heads(layer_idx)
            lines.append(f"Layer {layer_idx}:")
            lines.append(f"  Active heads: {len(active_heads)}/{layer.attn.num_heads}")

            # Show utilization scores for active heads
            for head_idx in active_heads:
                stats = self.head_stats[(layer_idx, head_idx)]
                lineage = self.head_lineage.get((layer_idx, head_idx))
                lineage_str = ""
                if lineage:
                    parent_layer, parent_head = lineage
                    lineage_str = f" (cloned from head {parent_head})"

                lines.append(f"    Head {head_idx}: util={stats.utilization_score:.3f}, "
                           f"cloned={stats.num_times_cloned}x{lineage_str}")

        lines.append("=" * 80)
        return "\n".join(lines)


# Example integration with training loop
def example_training_with_adaptation():
    """Example showing how to integrate adaptive head management."""
    # Assume model, dataloader, optimizer are set up
    manager = AdaptiveHeadManager(
        model,
        prune_threshold=0.3,
        clone_threshold=0.8,
        update_frequency=100
    )

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(batch['input_ids'], labels=batch['labels'])
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Collect gradient magnitudes for each head
            batch_gradients = {}
            for layer_idx, layer in enumerate(model.transformer.blocks):
                if layer.attn.gate.grad is not None:
                    for head_idx in range(layer.attn.num_heads):
                        grad_mag = abs(layer.attn.gate.grad[head_idx].item())
                        batch_gradients[(layer_idx, head_idx)] = grad_mag

            optimizer.step()

            # Update adaptive manager
            manager.step(batch_gradients)

        # Print report after each epoch
        print(manager.get_architecture_report())
