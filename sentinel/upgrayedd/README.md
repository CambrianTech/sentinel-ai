# Upgrayedd - Adaptive Optimization for Transformer Models

This package provides a modular framework for optimizing transformer models through pruning, fine-tuning, and adaptive learning.

## Core Components

- **AdaptiveOptimizer**: The main entry point for model optimization
- **Pruning Strategies**: Implementation of different pruning methods (entropy, magnitude, random)
- **Metrics & Visualization**: Progress tracking and visualization tools
- **Utilities**: Data loading, training, and generation helpers

## Getting Started

### CLI Usage

The simplest way to use Upgrayedd is through the CLI:

```bash
# Run with default settings (distilgpt2, WikiText, 30% pruning)
python scripts/run_experiment.py

# Run with custom model and settings
python scripts/run_experiment.py --model_name distilgpt2 --dataset wikitext --pruning_ratio 0.3

# Run with a config file
python scripts/run_experiment.py --config configs/entropy_pruning.json
```

### Python API Usage

You can also use Upgrayedd directly in your Python code:

```python
from sentinel.upgrayedd.optimizer import AdaptiveOptimizer, AdaptiveOptimizerConfig

# Create config
config = AdaptiveOptimizerConfig(
    model_name="distilgpt2",
    pruning_ratio=0.3,
    strategy="entropy",
    epochs_per_cycle=3
)

# Create optimizer
optimizer = AdaptiveOptimizer(config)

# Run optimization
results = optimizer.run_continuous_optimization(max_cycles=1)

# Access results
print(f"Improvement: {results['improvement']:.2f}%")
```

## Pruning Strategies

Upgrayedd supports multiple pruning strategies:

- **Entropy-based**: Prunes heads with the highest attention entropy (least focused attention)
- **Magnitude-based**: Prunes heads with the smallest weight magnitudes 
- **Random**: Randomly selects heads to prune (useful as a baseline)

## Advanced Features

- **Differential Learning Rates**: Apply different learning rates to different model components
- **Continuous Optimization**: Run multiple cycles of pruning and fine-tuning
- **Visualization**: Track metrics and pruning patterns visually