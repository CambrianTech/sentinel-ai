# Neural Plasticity Module

This module provides utilities for implementing neural plasticity in transformer models. Neural plasticity allows models to dynamically modify their structure during training by pruning and potentially growing attention heads based on their utility.

## Core Features

- Head importance calculation using entropy and gradient metrics
- Pruning of low-utility attention heads
- Differential learning rates for pruned vs. unpruned components
- Comprehensive visualization of attention patterns and model adaptation
- Specialized training loops for neural plasticity experiments

## Module Structure

- `core.py`: Core neural plasticity algorithms and metrics
- `visualization.py`: Visualization utilities for neural plasticity
- `training.py`: Training utilities with differential learning rates

## Usage Example

```python
from utils.neural_plasticity.core import calculate_head_gradients, generate_pruning_mask, apply_pruning_mask
from utils.neural_plasticity.training import run_plasticity_loop
from utils.neural_plasticity.visualization import visualize_head_gradients, visualize_training_metrics

# Load model and data
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
train_dataloader, eval_dataloader = get_dataloaders()

# Run complete neural plasticity loop
results = run_plasticity_loop(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    pruning_level=0.2,
    strategy="gradient",
    learning_rate=5e-5,
    training_steps=500
)

# Visualize results
visualize_head_gradients(
    results["grad_norm_values"],
    pruned_heads=results["pruned_heads"],
    title="Head Gradient Norms with Pruned Heads"
)

visualize_training_metrics(
    results["training_metrics"],
    title="Training Progress"
)
```

## Example Script

See the `examples/neural_plasticity_example.py` script for a complete demonstration of using these utilities.

## Integration with Colab Notebooks

This module is designed to integrate smoothly with Colab notebooks, providing:

1. Safe tensor visualization with automatic CPU conversion
2. Progress tracking and metrics visualization
3. Memory-efficient training loops

See the `colab_notebooks/NeuralPlasticityDemo.ipynb` notebook for a detailed demonstration.