# Neural Plasticity Cycle Implementation

## Overview

The Neural Plasticity Cycle is a biomimetic approach to transformer model optimization that simulates neural plasticity mechanisms found in biological brains. The implementation follows a complete cycle: train → prune → measure → grow → learn, allowing transformer models to adapt their architecture dynamically based on task requirements.

## Implementation Components

### Core Script

- **`scripts/neural_plasticity_cycle.py`**: Orchestrates the complete neural plasticity cycle with command-line interface for configuration parameters.

### Supporting Modules

- **`utils/pruning/pruning_module.py`**: Core implementation for loading models and pruning attention heads.
- **`utils/pruning/strategies.py`**: Implements different pruning strategies (entropy, magnitude, random).
- **`utils/pruning/growth.py`**: Implements head growth strategies (gradient sensitivity, entropy gap, balanced, random).
- **`utils/pruning/head_lr_manager.py`**: Manages differential learning rates for newly added heads.
- **`utils/train_utils.py`**: Provides fine-tuning capabilities with support for differential learning rates.
- **`utils/charting.py`**: Visualization utilities for the neural plasticity process.
- **`utils/metrics_logger.py`**: Logging utilities for tracking metrics throughout the cycle.

## Neural Plasticity Cycle Phases

1. **Training**: Initial training phase to establish baseline performance.
2. **Pruning**: Removal of less important attention heads based on selected strategy.
3. **Measurement**: Evaluation of model performance after pruning.
4. **Growth**: Strategic addition of new attention heads in optimal positions.
5. **Learning**: Fine-tuning with differential learning rates to integrate new heads.

## Available Strategies

### Pruning Strategies

- **Entropy**: Removes heads with lowest attention entropy (least information content).
- **Magnitude**: Removes heads with smallest weight magnitudes.
- **Random**: Removes heads randomly (useful as a baseline).

### Growth Strategies

- **Gradient Sensitivity**: Adds heads in positions where they would have the most impact based on gradient measurements.
- **Entropy Gap**: Adds heads where there's a significant entropy gap in the attention patterns.
- **Balanced**: Ensures heads are distributed evenly across layers.
- **Random**: Adds heads randomly (useful as a baseline).

## Usage Examples

```bash
# Basic usage with default parameters
python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --dataset tiny_shakespeare

# Customized pruning and growth strategies
python scripts/neural_plasticity_cycle.py --model_name gpt2 --pruning_strategy entropy --growth_strategy gradient_sensitivity

# Multiple cycles of plasticity
python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --cycles 3 --initial_pruning 0.3 --growth_ratio 0.33

# Fine-tuning with differential learning rates
python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --learning_rate 5e-5 --new_head_lr_multiplier 5.0
```

## Outputs and Visualizations

The implementation generates comprehensive outputs including:

- **Performance Metrics**: Perplexity measurements before/after each phase
- **Head Distribution Visualizations**: Maps showing active heads at each stage
- **Metrics Comparisons**: Charts comparing model performance across stages
- **Cycle Summaries**: For multi-cycle experiments, summaries of improvements over cycles

## Future Work

- Integration with more model architectures beyond GPT-2 family
- Additional metrics for head importance evaluation
- Implementation of adaptive pruning levels based on performance feedback
- Online learning capabilities with continuous plasticity