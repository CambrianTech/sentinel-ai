# Sentinel Pruning API

This module provides a modular, reusable API for pruning transformer models. It separates the core functionality from presentation layers like notebooks and scripts.

## Overview

The Sentinel Pruning API enables:

- Loading and managing transformer models
- Computing head importance using various strategies
- Pruning less important attention heads
- Fine-tuning pruned models to recover or exceed original performance
- Evaluating model performance
- Generating text with pruned and fine-tuned models
- Tracking and visualizing experiment progress

## Module Structure

- `experiment_runner.py` - Main experiment runner for pruning and fine-tuning
- `model_manager.py` - Model loading, saving, and management utilities
- `text_generator.py` - Text generation utilities
- `visualization.py` - Progress tracking and visualization utilities
- `entropy_magnitude.py` - Implementation of entropy and magnitude pruning strategies
- `strategies/` - Additional pruning strategies
- `fine_tuning/` - Fine-tuning implementations
- `growth/` - Implementations for growing new heads

## Usage

Here's a simple example of how to use the API:

```python
from sentinel.pruning.experiment_runner import run_experiment, ExperimentConfig

# Create experiment configuration
config = ExperimentConfig(
    model_name="distilgpt2",
    pruning_percent=0.3,
    num_epochs=3,
    batch_size=4,
    device="cuda",
    output_dir="pruning_results"
)

# Run the experiment
model, tokenizer, summary = run_experiment(config)

# Generate text with the pruned and fine-tuned model
from sentinel.pruning.text_generator import generate_text
text = generate_text(model, tokenizer, "Once upon a time")
print(text)
```

## Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Testability**: Functions can be tested independently
3. **Configurability**: Behavior is controlled through a clear configuration system
4. **Reusability**: Components can be used in different contexts (notebooks, scripts, etc.)
5. **Local-first**: Development and testing can be done locally before using in Colab

## Components

### ExperimentConfig

Configuration object for pruning experiments, which includes:

- Model name
- Pruning percentage
- Number of fine-tuning epochs
- Batch size
- Learning rate
- Device (CPU/GPU)
- Output directory
- Test mode flag

### ProgressTracker

Tracks metrics throughout the pruning and fine-tuning process:

- Loss
- Perplexity
- Pruning information
- Generated text samples

### Text Generation

Utilities for generating text from pruned and fine-tuned models:

- Basic text generation
- Interactive text generation
- Batch text generation