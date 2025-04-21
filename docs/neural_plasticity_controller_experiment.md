# Neural Plasticity with ANN Controller

## Overview

This document describes a comprehensive experiment that combines neural plasticity with our ANN controller for intelligent pruning of transformer attention heads, specifically targeting GPT-2 models. The implementation intelligently refines the model through multiple phases of training, pruning, and fine-tuning, with detailed visualizations powered by our multi-phase dashboard.

Version: v1.0.0 (2025-04-20 22:30:00)

## Key Features

### 1. Intelligent Pruning with ANN Controller

The experiment uses the Adaptive Neural Network Controller (ANN Controller) as described in our research to dynamically gate attention heads based on feedback metrics:

- **Entropy-Based Analysis**: Identifies and prunes heads with high entropy (unfocused attention)
- **Dynamic Gate Mechanism**: Maintains learnable gate parameters for each attention head
- **Gradient-Aware Updates**: Considers both attention entropy and gradient magnitudes
- **L1 Regularization**: Encourages natural sparsity in head utilization

### 2. Multi-Phase Training Process

The experiment follows a structured, multi-phase approach to maximize model performance:

1. **Warmup Phase**: Initial training to establish baseline behavior
2. **Analysis Phase**: Collecting attention distributions and performance metrics
3. **Pruning Phase**: Strategic removal of attention heads with entropy-based criteria
4. **Fine-tuning Phase**: Recovery training to adapt to pruned architecture
5. **Evaluation Phase**: Comprehensive assessment comparing to baseline

### 3. Multi-Cycle Pruning Strategy

Rather than pruning all at once, the implementation uses multiple cycles of pruning and fine-tuning:

- **Progressive Sparsity**: Gradually increases model sparsity across cycles
- **Adaptation Periods**: Allows model to recover between pruning operations
- **Continuous Feedback**: Updates pruning strategy based on evolving metrics

### 4. Comprehensive Visualization

The experiment integrates our multi-phase dashboard for detailed tracking and visualization:

- **Timeline View**: Tracks the complete training process with phase transitions
- **Metric Tracking**: Monitors loss, perplexity, and sparsity throughout training
- **Pruning Events**: Highlights when and which heads are pruned
- **Comparative Analysis**: Visualizes baseline vs. pruned performance

## Technical Implementation

### Adaptive GPT-2 Model

The implementation wraps a standard GPT-2 model with adaptive attention mechanisms:

```python
class AdaptiveGPT2(torch.nn.Module):
    def __init__(self, model_name="gpt2", controller_config=None):
        super().__init__()
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize ANN controller
        self.controller = ANNController(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            config=controller_config or {}
        )
        
        # Register hooks for adaptive attention
        self._register_attention_hooks()
```

The model uses forward hooks to apply controller gates to attention heads:

```python
def _attention_hook(self, module, inputs, outputs):
    """Apply controller gates to attention heads output."""
    layer_idx = getattr(module, "layer_idx")
    
    # Get gates for this layer from controller
    gates = self.controller()[layer_idx].view(1, -1, 1, 1)
    
    # Apply gates to attention weights
    if isinstance(outputs, tuple):
        attn_output, attn_weights = outputs
        return (attn_output * gates, attn_weights)
    else:
        return outputs * gates
```

### Controller Integration

The ANN Controller maintains learnable gate parameters and updates them based on metrics:

```python
def update_controller(self):
    """Update controller gates based on collected metrics."""
    metrics = self.collect_metrics()
    self.controller.update_gates(metrics)
    
    # Return active head count for monitoring
    gate_values = torch.sigmoid(self.controller.gate_logits)
    active_heads = (gate_values > 0.5).sum().item()
    total_heads = self.num_layers * self.num_heads
    return active_heads, total_heads
```

The metrics include attention entropy and gradient norms for each head:

```python
def collect_metrics(self):
    """Collect metrics for controller updates."""
    # Compute gradient norms if gradients exist
    for layer_idx, layer in enumerate(self.model.transformer.h):
        for head_idx in range(self.num_heads):
            # Find weights corresponding to this attention head
            # [...]
            
            # Compute gradient norm
            grad_norm = 0.0
            for param in head_params:
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
            
            self.head_grad_norm[layer_idx, head_idx] = grad_norm
    
    # Prepare metrics dictionary
    metrics_dict = {
        "entropy": self.head_entropy,
        "grad_norm": self.head_grad_norm,
        "controller_lr": torch.tensor(0.01)
    }
    
    return metrics_dict
```

### Entropy-Based Pruning

In addition to the controller's dynamic gating, the experiment applies explicit pruning:

```python
# Collect attention distributions
attn_distributions = collect_attention_distributions(
    model, 
    eval_dataloader,
    num_batches=5,
    device=device
)

# Perform entropy-based pruning
pruned_heads = entropy_based_pruning(model, attn_distributions, pruning_ratio)
```

### Multi-Phase Dashboard Integration

The experiment records detailed metrics at each step for visualization:

```python
# Record metrics
metrics = {
    "loss": loss,
    "phase": current_phase,
    "sparsity": sparsity,
    "cycle": cycle + 1,
    "perplexity": perplexity
}

# Log to dashboard
dashboard.record_step(metrics, global_step)

# Record phase transitions
dashboard.record_phase_transition("pruning", global_step)

# Record pruning events
dashboard.record_pruning_event(pruning_info, global_step)
```

## Running the Experiment

### Basic Usage

```bash
python scripts/neural_plasticity_controller_experiment.py --model_name gpt2 --cycles 3 --pruning_level 0.3
```

### Full Options

```
usage: neural_plasticity_controller_experiment.py [-h] [--model_name MODEL_NAME]
                                                 [--device DEVICE] [--batch_size BATCH_SIZE]
                                                 [--max_length MAX_LENGTH]
                                                 [--learning_rate LEARNING_RATE]
                                                 [--pruning_level PRUNING_LEVEL]
                                                 [--cycles CYCLES]
                                                 [--controller_reg_weight CONTROLLER_REG_WEIGHT]
                                                 [--controller_update_freq CONTROLLER_UPDATE_FREQ]
                                                 [--warmup_steps WARMUP_STEPS]
                                                 [--finetuning_steps FINETUNING_STEPS]
                                                 [--training_steps TRAINING_STEPS]
                                                 [--eval_freq EVAL_FREQ]
                                                 [--stability_window STABILITY_WINDOW]
                                                 [--output_dir OUTPUT_DIR] [--no_wandb]
```

Key parameters:
- `--model_name`: Hugging Face model name (default: "gpt2")
- `--pruning_level`: Overall pruning level from 0.0-1.0 (default: 0.3)
- `--cycles`: Number of pruning cycles (default: 3)
- `--controller_reg_weight`: L1 regularization weight (default: 1e-4)

## Expected Results

The experiment should result in:

1. **Performance Maintenance**: Similar or better perplexity compared to baseline
2. **Model Sparsity**: Approximately 30% reduction in active attention heads
3. **Visualization Dashboard**: Comprehensive visualization of the entire process
4. **Comparison Samples**: Text generation samples from both baseline and pruned models

## Visualization Examples

The experiment produces detailed visualizations including:

1. **Complete Process Timeline**: Shows all phases with color coding
   - Blue regions: Warmup phase
   - Red regions: Pruning phases
   - Green regions: Fine-tuning phases
   - Vertical markers: Pruning events

2. **Metrics Tracking**: Loss, perplexity, and sparsity over time
   - Training loss curves
   - Perplexity evaluation points
   - Gradually increasing sparsity steps

3. **Multi-Cycle Analysis**: Per-cycle breakdown
   - Comparative performance across cycles
   - Pruning impact visualization
   - Recovery patterns after each pruning event

## References

1. Sentinel-AI Internal Documentation on Adaptive Transformers
2. "Neural Plasticity in Transformer Models" (our research paper)
3. "Entropy-Based Pruning for Transformer Models" (our methodology paper)
4. "ANNS: Adaptive Neural Network Supervision" (related controller work)

## Future Directions

1. **Per-Head Learning Rates**: Individualized optimization for each attention head
2. **Task-Specific Pruning**: Adapting pruning strategy to specific downstream tasks
3. **Automatic Cycle Determination**: Intelligently determining optimal cycle count
4. **Attention Head Regrowth**: Allowing pruned heads to be reactivated if needed

## Conclusion

This experiment demonstrates Sentinel-AI's commitment to developing efficient transformer models through neural plasticity techniques. By combining our ANN controller with entropy-based pruning across multiple cycles, we create models that are both more efficient and maintain or improve performance.