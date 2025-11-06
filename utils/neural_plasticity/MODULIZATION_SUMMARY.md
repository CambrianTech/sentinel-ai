# Neural Plasticity Modularization Summary

This document summarizes the modularization of the neural plasticity functionality, including the architecture, key components, and cross-platform compatibility features.

## Modularization Goals

1. **Cross-Platform Compatibility**: Ensure the code runs reliably on Apple Silicon (M1/M2/M3), standard CPUs, and GPU environments
2. **Consistent API**: Provide a unified API for all neural plasticity operations
3. **Environment Awareness**: Automatically detect and adapt to the execution environment
4. **Tensor Safety**: Implement robust tensor operations with multiple fallback mechanisms
5. **Maintainable Architecture**: Organize code into logical modules with clear responsibilities

## Module Structure

The neural plasticity functionality is organized into the following modules:

- **core.py**: Environment detection, tensor operations, pruning logic
- **visualization.py**: Visualization functions for attention patterns, entropy, gradients
- **training.py**: Training loops for pruned models, differential learning rates
- **experiment.py**: Experiment runners for complete neural plasticity cycles
- **\_\_init\_\_.py**: Public API, enums, and high-level class

## API Design

The API is designed with two levels of abstraction:

1. **Direct Function Imports**: For users who need fine-grained control
2. **High-Level NeuralPlasticity Class**: For users who want a simplified interface

```python
# Direct function imports
from utils.neural_plasticity import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask
)

# High-level class
from utils.neural_plasticity import NeuralPlasticity
```

## Environment Detection

Environment detection automatically identifies the execution environment:

- **IS_APPLE_SILICON**: True if running on Apple ARM chips
- **IS_COLAB**: True if running in Google Colab
- **HAS_GPU**: True if GPU acceleration is available

Based on these flags, the module applies appropriate optimizations:

- **Apple Silicon**: Force CPU usage, disable threading for BLAS operations
- **Colab with GPU**: Ensure tensors are on GPU for maximum performance
- **Standard CPU**: Regular operation with standard PyTorch settings

## Tensor Safety

To prevent crashes on Apple Silicon, the module implements a robust tensor handling system:

1. **Safe Matrix Multiplication**: `safe_matmul` function with multiple fallback mechanisms:
   - NumPy-based matrix multiplication (bypasses BLAS)
   - Manual Python implementation for small matrices
   - Protected single-threaded PyTorch operations

2. **Safe Tensor Conversion**: `safe_tensor_to_numpy` for device-aware tensor conversion

## Visualization Enhancements

Visualization functions are enhanced for cross-platform compatibility:

- Safe tensor conversion before visualization
- Device-aware tensor handling
- Matplotlib backend switching for Apple Silicon
- Proper figure sizing and saving

## Testing and Validation

Testing utilities ensure the module works across platforms:

- **Minimal Test**: Quick validation of core functionality
- **Adapted Notebook**: Full notebook using the modular API
- **Runnable Notebook**: End-to-end test with execution

## Scripts for Testing

The following scripts help test the modular functionality:

- **create_minimal_test.py**: Creates a minimal test notebook
- **run_neural_plasticity_minimal.py**: Runs the minimal test
- **adapt_neural_plasticity_notebook.py**: Creates a notebook using the modular API
- **run_neural_plasticity_notebook.py**: Runs the adapted notebook

## Usage Examples

### Basic Environment Detection

```python
from utils.neural_plasticity import NeuralPlasticity

# Get environment information
env_info = NeuralPlasticity.get_environment_info()
print(f"Platform: {env_info['platform']}")
print(f"Apple Silicon: {env_info['is_apple_silicon']}")
print(f"GPU Available: {env_info['has_gpu']}")
print(f"Device: {env_info['device']}")
```

### Analyzing Attention Patterns

```python
# Analyze attention patterns
attention_data = NeuralPlasticity.analyze_attention_patterns(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask
)

# Extract data
attention_tensors = attention_data['attention_tensors']
entropy_values = attention_data['entropy_values']
```

### Pruning Model

```python
from utils.neural_plasticity import (
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    PruningStrategy
)

# Calculate gradients
grad_norms = calculate_head_gradients(
    model=model,
    dataloader=train_dataloader,
    num_batches=2,
    device=device
)

# Generate pruning mask
pruning_mask = generate_pruning_mask(
    grad_norm_values=grad_norms,
    entropy_values=entropy_values[0],
    prune_percent=0.2,
    strategy=PruningStrategy.COMBINED
)

# Apply pruning
pruned_heads = apply_pruning_mask(
    model=model,
    pruning_mask=pruning_mask,
    mode="zero_weights"
)
```

### Visualization

```python
from utils.neural_plasticity import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions
)

# Visualize entropy
entropy_fig = visualize_head_entropy(
    entropy_values=entropy_values,
    title="Attention Entropy Heatmap"
)

# Visualize gradients
grad_fig = visualize_head_gradients(
    grad_norm_values=grad_norms,
    title="Head Gradient Norms"
)

# Visualize pruning decisions
mask_fig = visualize_pruning_decisions(
    grad_norm_values=grad_norms,
    pruning_mask=pruning_mask,
    title="Pruning Decisions"
)
```

## Complete Pruning Cycle

```python
results = NeuralPlasticity.run_pruning_cycle(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    pruning_level=0.2,
    strategy=PruningStrategy.COMBINED,
    learning_rate=5e-5,
    training_steps=200
)
```