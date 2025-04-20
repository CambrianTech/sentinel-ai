# Neural Plasticity Module

This module provides a comprehensive framework for applying neural plasticity principles to transformer models, allowing dynamic pruning and regrowth of attention heads based on utility metrics.

## Overview

Neural plasticity in transformer models mimics how biological brains optimize neural pathways by:

1. Tracking attention head metrics (entropy, gradients)
2. Pruning unfocused or less useful heads
3. Optionally regrowing promising heads during training
4. Visualizing the "brain dynamics" over time

This creates more efficient models that focus computational resources on the most useful attention mechanisms.

## Cross-Platform Compatibility

The neural plasticity module automatically detects and adapts to different execution environments:

- **Colab:** Uses GPU acceleration when available for maximum performance
- **Apple Silicon:** Applies safeguards against BLAS/libtorch crashes that commonly occur on M1/M2/M3 Macs
- **Standard Hardware:** Operates normally with GPU acceleration when available

No manual configuration is required - the module automatically optimizes for your environment.

## API Usage

The API exposes functionality through both direct function imports and a high-level `NeuralPlasticity` class:

```python
# High-level API
from utils.neural_plasticity import NeuralPlasticity

# Get environment information
env_info = NeuralPlasticity.get_environment_info()
device = env_info['device']  # Automatically selects appropriate device

# Analyze attention patterns
analysis = NeuralPlasticity.analyze_attention_patterns(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask
)

# Access results
attention_tensors = analysis['attention_tensors']
entropy_values = analysis['entropy_values']

# Run a complete pruning cycle
results = NeuralPlasticity.run_pruning_cycle(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    pruning_level=0.2,       # Prune 20% of heads
    strategy="combined",     # Use both entropy and gradient info
    learning_rate=5e-5,
    training_steps=200
)
```

## Core Components

### Environment Detection

The module automatically detects your execution environment and applies appropriate optimizations:

```python
from utils.neural_plasticity import IS_APPLE_SILICON, IS_COLAB, HAS_GPU

# Check environment
if IS_APPLE_SILICON:
    print("Running on Apple Silicon with BLAS crash prevention")
if IS_COLAB and HAS_GPU:
    print("Running in Colab with GPU acceleration")
```

### Analysis Functions

Functions for analyzing transformer attention patterns:

```python
from utils.neural_plasticity import (
    calculate_head_entropy,
    calculate_head_gradients
)

# Calculate entropy for attention maps
entropy_values = calculate_head_entropy(attention_maps)

# Calculate gradient norms for each head
grad_norms = calculate_head_gradients(
    model=model,
    dataloader=train_dataloader,
    num_batches=2,
    device=device
)
```

### Pruning Functions

Functions for generating and applying pruning masks:

```python
from utils.neural_plasticity import (
    generate_pruning_mask,
    apply_pruning_mask,
    PruningStrategy,
    PruningMode
)

# Generate pruning mask
pruning_mask = generate_pruning_mask(
    grad_norm_values=grad_norms,
    entropy_values=entropy_values,
    prune_percent=0.2,
    strategy=PruningStrategy.COMBINED  # Use both entropy and gradients
)

# Apply pruning
pruned_heads = apply_pruning_mask(
    model=model,
    pruning_mask=pruning_mask,
    mode="zero_weights"  # Other options: "mask_forward", "gate"
)
```

### Visualization Functions

Functions for visualizing attention patterns and pruning decisions:

```python
from utils.neural_plasticity import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_attention_patterns
)

# Visualize entropy heatmap
entropy_fig = visualize_head_entropy(
    entropy_values=entropy_values,
    title="Attention Entropy Heatmap",
    min_value=0.0,
    annotate=True
)

# Visualize gradient norms
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

# Visualize attention patterns
attn_fig = visualize_attention_patterns(
    attention_maps=attention_tensors,
    layer_idx=0,
    head_idx=0,
    title="Attention Pattern"
)
```

## Example Notebooks

The repository includes several notebooks demonstrating neural plasticity:

1. **NeuralPlasticityDemo.ipynb**: Full demonstration of neural plasticity
2. **neural_plasticity_minimal_test.ipynb**: Minimal test of the module functionality
3. **neural_plasticity_runnable.ipynb**: Simplified runnable version for quick tests

## Creating and Running Notebooks

The scripts directory includes utilities for creating and running neural plasticity notebooks:

```bash
# Create an adapted notebook with the modular API
python scripts/adapt_neural_plasticity_notebook.py

# Create and run a minimal test
python scripts/run_neural_plasticity_minimal.py --run

# Create a minimal test without running it
python scripts/create_minimal_test.py
```

## Tensor Safety

The module implements multiple fallback mechanisms for matrix operations to ensure compatibility across platforms:

1. **Primary**: NumPy-based matrix multiplication (bypasses BLAS)  
2. **Secondary**: Manual Python implementation for small matrices
3. **Tertiary**: Protected single-threaded PyTorch operations

This prevents crashes on Apple Silicon while maintaining performance on other platforms.

## Module Structure

- `utils/neural_plasticity/core.py`: Core tensor operations and pruning logic
- `utils/neural_plasticity/visualization.py`: Visualization utilities
- `utils/neural_plasticity/training.py`: Training loops for pruned models
- `utils/neural_plasticity/experiment.py`: Experiment runners
- `utils/neural_plasticity/__init__.py`: API definition and enums