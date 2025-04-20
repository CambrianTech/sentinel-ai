# Neural Plasticity Module

The Neural Plasticity module provides utilities for implementing neural plasticity in transformer models, enabling dynamic pruning and regrowth of neural connections during training.

## Overview

Neural plasticity allows transformer models to adapt their structure over time by:
- Identifying and pruning less useful attention heads
- Selectively reviving heads that show potential
- Using differential learning rates for pruned vs. unpruned heads
- Visualizing plasticity patterns over time

This mimics how biological brains form efficient neural pathways by pruning unused connections and strengthening useful ones.

## Structure

The module is organized into three main components:

### Core (`core.py`)

Core algorithms for neural plasticity:
- `calculate_head_entropy()`: Calculate entropy of attention patterns
- `calculate_head_gradients()`: Calculate gradient magnitudes for attention heads
- `generate_pruning_mask()`: Create pruning masks based on different strategies
- `apply_pruning_mask()`: Apply pruning to a model based on a mask
- `evaluate_model()`: Evaluate model performance on a dataset

### Visualization (`visualization.py`)

Visualization utilities for neural plasticity:
- `visualize_head_entropy()`: Visualize entropy values across layers and heads
- `visualize_head_gradients()`: Show gradient norms with pruned/revived head markers
- `visualize_pruning_decisions()`: Highlight pruning decisions based on metrics
- `visualize_training_metrics()`: Plot training progress and metrics
- `visualize_attention_patterns()`: Display attention patterns for selected heads

### Training (`training.py`)

Training utilities with differential learning rates:
- `create_plasticity_trainer()`: Configure training with neural plasticity
- `run_plasticity_loop()`: Complete neural plasticity workflow
- `train_with_plasticity()`: Train a pruned model with differential learning rates
- `get_plasticity_optimizer()`: Create optimizer with head-specific learning rates

## Usage Example

```python
from utils.neural_plasticity.core import calculate_head_gradients, generate_pruning_mask
from utils.neural_plasticity.training import run_plasticity_loop
from utils.neural_plasticity.visualization import visualize_head_gradients

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
```

## GPU Compatibility

All tensor handling in this module is specifically designed to work with both CPU and GPU environments. Proper tensor detachment and CPU conversion is used for all visualization functions:

```python
# Safe tensor visualization
def safe_tensor_imshow(tensor, title=None):
    """Safely display a tensor as an image, handling GPU tensors properly."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return plt.imshow(tensor.numpy())
```

## Notebook Integration

The `NeuralPlasticityDemo.ipynb` notebook in the `colab_notebooks` directory demonstrates the use of these modules. The notebook was refactored to use modularized components instead of inline function definitions.

## Testing

Unit tests for the neural plasticity modules are available in the `tests/unit/utils/neural_plasticity` directory.