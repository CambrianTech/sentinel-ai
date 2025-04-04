# Model Improvement Platform for Google Colab

## Overview

This Colab notebook provides a modular, UI-based framework for configuring and running model improvement experiments. It enables users to easily configure parameters for model pruning, fine-tuning, and adaptive plasticity through an interactive interface rather than hard-coded values.

## Features

- **Interactive UI**: Dropdown menus, sliders, and checkboxes for all parameters
- **Environment Detection**: Automatically detects and configures for available hardware
- **Auto-Optimization**: Smart parameter tuning based on model size and hardware
- **Experiment Management**: Save and load experiment configurations
- **Multiple Model Support**: Works with various model architectures (GPT-2, OPT, etc.)
- **Visualization**: Real-time experiment progress and results visualization
- **Adaptive Plasticity**: Optional neural plasticity for dynamic architecture improvement

## Usage

1. Upload `ModelImprovementColab.ipynb` to Google Colab
2. Run the setup cells to install dependencies
3. Configure experiment parameters using the interactive UI
4. Click "Auto-Optimize Parameters" to optimize for your environment
5. Run the experiment with "Run Experiment"
6. View results and save configurations for future use

## UI Sections

The UI is organized into tabs for easy navigation:

1. **Model**: Select model architecture and size
2. **Pruning**: Configure pruning strategy and level
3. **Fine-tuning**: Set up fine-tuning parameters
4. **Adaptive**: Configure adaptive plasticity features
5. **Advanced**: Adjust memory, stability, and performance settings
6. **Experiment**: Set prompt, runtime, and output location

## Parameter Descriptions

### Model Parameters
- **Model Size**: Categorizes models by parameter count (tiny to xl)
- **Model**: Specific model architecture to use

### Pruning Parameters
- **Enable Pruning**: Toggle pruning on/off
- **Pruning Strategy**: Methodology for identifying heads to prune (entropy, magnitude, random)
- **Pruning Level**: Percentage of attention heads to prune (0.1-0.9)

### Fine-tuning Parameters
- **Enable Fine-tuning**: Toggle fine-tuning on/off
- **Epochs**: Number of training epochs
- **Learning Rate**: Step size for optimization

### Adaptive Plasticity Parameters
- **Enable Adaptive Plasticity**: Toggle dynamic architecture features
- **Plasticity Level**: How readily the model adapts (0.1-1.0)
- **Growth Rate**: Rate of potential architecture expansion (0.0-0.5)

### Advanced Parameters
- **Stability Level**: Numerical stability safeguards (1-3)
- **Batch Size**: Number of samples per training batch
- **Sequence Length**: Token context length for training
- **Optimize Memory**: Enables memory optimization

### Experiment Parameters
- **Prompt**: Text prompt for evaluation
- **Max Runtime**: Maximum experiment duration
- **Results Directory**: Where outputs are saved

## Example Configurations

### Balanced Performance/Quality
```json
{
  "model": "gpt2",
  "pruning_strategy": "entropy",
  "pruning_level": 0.3,
  "fine_tuning_epochs": 2,
  "stability_level": 2
}
```

### Maximum Speed
```json
{
  "model": "distilgpt2",
  "pruning_strategy": "magnitude",
  "pruning_level": 0.7,
  "fine_tuning_epochs": 1,
  "stability_level": 1
}
```

### Adaptive Architecture
```json
{
  "model": "gpt2-medium",
  "pruning_level": 0.3,
  "enable_adaptive_plasticity": true,
  "plasticity_level": 0.6,
  "growth_rate": 0.2
}
```

## Extending the Framework

The ModularExperimentRunner class is designed to be extensible. To add new experiment types or parameters:

1. Add new fields to the config dictionary in `__init__()`
2. Create UI elements for the new parameters in `create_ui()`
3. Update the experiment creation logic in `create_experiment()`
4. Modify the experiment running code in `run_experiment()`

## Version History

- v1.0.0 (April 2025): Initial release with full parameter configuration and adaptive plasticity support