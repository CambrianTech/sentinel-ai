# Pruning Visualization Utilities

This directory contains visualization utilities for pruning experiments and neural plasticity demonstrations.

## Overview

The visualization utilities provide functions for:

- Creating visualizations of pruning experiment results
- Comparing different pruning strategies
- Visualizing attention patterns and entropy
- Tracking neural plasticity metrics
- Analyzing gradient norms and head importance

## Modules

### Pruning Experiment Visualizations (`visualization.py`)

General-purpose visualizations for pruning results and comparisons:

- `plot_experiment_summary()`: Create a comprehensive visualization of experiment results
- `plot_strategy_comparison()`: Bar chart comparing different pruning strategies
- `plot_recovery_comparison()`: Compare recovery or improvement percentages
- `visualize_head_importance()`: Attention head importance scores by strategy
- `plot_head_gradients_with_overlays()`: Gradient norms with pruning/revival markers

### Neural Plasticity Visualizations (`visualization_additions.py`)

Specialized visualizations for neural plasticity demonstrations:

- `visualize_gradient_norms()`: Visualize gradient norms with markers for pruned/revived heads
- `visualize_attention_matrix()`: Show attention patterns for specific layers/heads
- `visualize_entropy_heatmap()`: Display entropy values across layers and heads
- `visualize_normalized_entropy()`: Show entropy as percentage of maximum possible
- `visualize_entropy_vs_gradient()`: Scatter plot of entropy vs gradient relationship
- `visualize_training_progress()`: Training metrics including loss, pruning and sparsity

## Usage Examples

### Basic Experiment Visualization

```python
from utils.pruning.visualization import plot_experiment_summary

# Load experiment results
results_df = load_results_from_file("experiment_results.csv")

# Create experiment summary
fig = plot_experiment_summary(results_df)
fig.savefig("experiment_summary.png")
```

### Neural Plasticity Visualizations

```python
from utils.pruning.visualization_additions import (
    visualize_gradient_norms,
    visualize_attention_matrix,
    visualize_entropy_heatmap
)

# Visualize gradient norms with pruning information
grad_norms = model.get_gradient_norms()
pruned_heads = [(0, 1), (1, 3)]  # (layer, head) tuples
fig = visualize_gradient_norms(
    grad_norms,
    pruned_heads=pruned_heads,
    title="Gradient Norms with Pruned Heads"
)
fig.savefig("gradient_norms.png")

# Visualize attention patterns
attention = model_outputs.attentions[0]  # First layer attention
fig = visualize_attention_matrix(
    attention,
    layer_idx=0,
    head_idx=0,
    title="Attention Pattern for Head 0"
)
fig.savefig("attention_pattern.png")

# Visualize entropy heatmap
entropy_values = calculate_entropy(model_outputs.attentions)
fig = visualize_entropy_heatmap(
    entropy_values,
    title="Attention Entropy Across Layers and Heads"
)
fig.savefig("entropy_heatmap.png")
```

### Training Progress Visualization

```python
from utils.pruning.visualization_additions import visualize_training_progress

# Metrics history collected during training
metrics_history = {
    "step": steps,
    "train_loss": train_losses,
    "eval_loss": eval_losses,
    "pruned_heads": pruned_head_counts,
    "revived_heads": revived_head_counts,
    "sparsity": sparsity_values,
    "epoch": epochs,
    "perplexity": perplexities
}

# Create training progress visualization
fig = visualize_training_progress(
    metrics_history,
    max_display_points=100  # Downsample if there are too many steps
)
fig.savefig("training_progress.png")
```

## Testing

To test the visualization utilities:

```bash
# Run the basic visualization tests
python temp_fix/test_visualization_utils.py

# Run full test suite including visualization
pytest utils/pruning
```