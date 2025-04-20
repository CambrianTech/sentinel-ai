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

Functions for visualizing attention patterns, pruning decisions, and the complete neural plasticity process:

```python
from utils.neural_plasticity import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_attention_patterns
)

from utils.neural_plasticity.visualization import (
    visualize_warmup_dashboard,
    VisualizationReporter
)

from utils.colab.visualizations import visualize_complete_training_process

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

# Create comprehensive warmup phase dashboard
warmup_dashboard = visualize_warmup_dashboard(
    warmup_results=experiment.warmup_results,
    title="Neural Plasticity Warmup Dashboard",
    figsize=(12, 10),
    save_path="warmup_dashboard.png"
)

# Visualize the complete neural plasticity process
complete_process_fig = visualize_complete_training_process(
    experiment=experiment,
    title="Complete Neural Plasticity Training Process",
    show_plot=True,
    show_quote=True
)

# Using the VisualizationReporter for comprehensive reporting
reporter = VisualizationReporter(
    model=model,
    tokenizer=tokenizer,
    output_dir="visualizations",
    save_visualizations=True
)

# Display warmup results with enhanced visualization
reporter.display_warmup_results(experiment.warmup_results)

# Display pruning results with visualizations
reporter.display_pruning_results(experiment.pruning_results)

# Display complete training process
reporter.display_complete_training_process(experiment)
```

### Dashboard Generation

The module includes a comprehensive dashboard generator that can create visualizations for all phases of the neural plasticity process:

```python
from scripts.neural_plasticity_dashboard import generate_dashboards

# Generate all dashboards and save to output directory
dashboards = generate_dashboards(
    experiment=experiment,
    output_dir="./visualizations"
)

# Access individual dashboards
warmup_dashboard = dashboards["warmup"]
complete_process = dashboards["complete"]
pruning_visualizations = dashboards["pruning"]

# Generate only the complete process dashboard
from scripts.neural_plasticity_dashboard import generate_complete_process_dashboard

process_fig = generate_complete_process_dashboard(
    experiment=experiment,
    output_dir="./visualizations/complete"
)
```

You can also use the dashboard generator from the command line:

```bash
# Generate all dashboards from a saved experiment
python scripts/neural_plasticity_dashboard.py --experiment_file=experiment.pkl --output_dir=./visualizations

# Show dashboards without saving
python scripts/neural_plasticity_dashboard.py --no_show --output_dir=./visualizations
```

#### Using with the VisualizationReporter

The `VisualizationReporter` now includes methods for generating comprehensive dashboards:

```python
# Create a visualization reporter
reporter = VisualizationReporter(
    model=model,
    tokenizer=tokenizer,
    output_dir="visualizations",
    save_visualizations=True
)

# Generate all dashboards for an experiment
reporter.generate_comprehensive_dashboard(
    experiment=experiment,
    output_dir="dashboards"
)

# Display just the complete process visualization
reporter.display_complete_training_process(experiment)
```

## Example Notebooks

The repository includes several notebooks demonstrating neural plasticity:

1. **NeuralPlasticityDemo.ipynb**: Full demonstration of neural plasticity with comprehensive visualizations
2. **neural_plasticity_minimal_test.ipynb**: Minimal test of the module functionality
3. **neural_plasticity_runnable.ipynb**: Simplified runnable version for quick tests

### Visualization Features in Notebooks

The neural plasticity notebooks include comprehensive visualizations that help you understand the entire process:

1. **Warmup Dashboard**: Shows loss curves, stabilization detection, and polynomial curve fitting
2. **Complete Process Visualization**: Displays the entire neural plasticity cycle with clear phase markers
3. **Attention Pattern Visualization**: Visualizes attention patterns of specific heads
4. **Entropy Heatmaps**: Shows entropy values across all layers and heads
5. **Pruning Decision Visualization**: Highlights which heads were pruned and why
6. **Training Metrics**: Tracks loss, perplexity, and sparsity throughout training

#### Using the Complete Process Dashboard in Colab

To add the complete neural plasticity process visualization to any Colab notebook, copy the contents of `utils/colab/neural_plasticity_dashboard_cell.py` into a notebook cell, then use:

```python
# Create visualization for experiment results
experiment_results = {
    'warmup': warmup_results,
    'pruning': pruning_results,
    'fine_tuning': fine_tuning_results
}

# Display the comprehensive dashboard
display_neural_plasticity_dashboard(
    experiment=experiment_results,
    output_dir="neural_plasticity_output"
)
```

This will generate a multi-panel visualization showing:
- The complete training process across all phases
- Detailed view of each phase (warmup, pruning, fine-tuning)
- Training metrics including perplexity and pruning statistics
- Clear markers for stabilization points and phase transitions
- Summary statistics for the entire process

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