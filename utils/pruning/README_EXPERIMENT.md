# Modular Experiment Framework for Pruning and Fine-tuning

This module provides a modular framework for running pruning and fine-tuning experiments with language models. It supports different model architectures, pruning strategies, and evaluation methods.

## Overview

The experiment framework consists of two main classes:

1. `PruningExperiment`: Base class for individual pruning experiments
2. `PruningFineTuningExperiment`: Extended class for running multiple experiments with different configurations

This framework offers significant improvements over the previous approach:

- **Modularity**: Separate components for environment detection, pruning, fine-tuning, and visualization
- **Stability**: Built-in NaN detection and prevention, especially for OPT models
- **Memory Management**: Dynamic parameter optimization based on model size and hardware
- **Visualization**: Comprehensive plotting capabilities for analyzing results
- **Error Handling**: Graceful failure recovery and detailed logging

## Key Features

- **Environment Detection**: Automatically detects available hardware and adjusts parameters
- **Model Selection**: Intelligently selects suitable models based on available memory
- **NaN Prevention**: Handles numerical instability in models like OPT
- **Visualization**: Creates detailed plots of experiment results
- **Serialization**: Saves experiment results to disk for later analysis
- **Progress Tracking**: Shows realtime progress during experiments

## Usage Examples

### Basic Single Experiment

```python
from utils.pruning.experiment import PruningExperiment

# Create an experiment
experiment = PruningExperiment(results_dir="pruning_results")

# Run a single experiment
result = experiment.run_single_experiment(
    model="gpt2",
    strategy="attention",
    pruning_level=0.3,
    prompt="Artificial intelligence will",
    fine_tuning_epochs=1
)

# Plot results
experiment.plot_results()
```

### Multiple Experiments

```python
from utils.pruning.experiment import PruningFineTuningExperiment

# Create multi-experiment object
experiment = PruningFineTuningExperiment(results_dir="multi_experiment_results")

# Run multiple experiments
results = experiment.run_experiment(
    strategies=["random", "attention", "entropy"],
    pruning_levels=[0.1, 0.3, 0.5],
    prompt="Artificial intelligence will",
    fine_tuning_epochs=1,
    max_runtime=3600,  # 1 hour max runtime
    models=["distilgpt2", "gpt2"]
)

# Plot comprehensive results
experiment.plot_results()
```

## Testing

To test the experiment framework, use the provided test script:

```bash
python scripts/test_modular_experiment.py --model distilgpt2 --strategy random --pruning_level 0.3

# For a more comprehensive test
python scripts/test_modular_experiment.py --full_experiment
```

## Integration with main.py

The modular experiment framework is designed to be compatible with the main.py script. After running experiments, you can:

1. Extract the fine-tuned model parameters
2. Convert them to the format expected by main.py
3. Run inference using the pruned and fine-tuned model

Example:

```python
# After running an experiment
result = experiment.run_single_experiment(...)

# Extract fine-tuned parameters
fine_tuned_params = result["stages"]["fine_tuned"]["params"]

# Save to a format compatible with main.py
# (Implementation depends on specific requirements)
```

## Advanced Features

- **Safety Mechanisms**: Detects and prevents NaN values in loss functions and gradients
- **Memory Optimization**: Adjusts batch size and sequence length based on model size and GPU memory
- **Hardware Adaptation**: Different strategies for different hardware configurations
- **Error Analysis**: Detailed metrics on training issues and recovery rates
- **Visualizations**: Multiple charts for analyzing performance across experiments

## Future Improvements

- Distributed experiments across multiple GPUs
- Integration with hyperparameter search
- Extended support for more model architectures
- Enhanced pruning strategies with neural architecture search