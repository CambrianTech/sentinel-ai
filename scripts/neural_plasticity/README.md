# Neural Plasticity Experiments

This directory contains scripts and utilities for running neural plasticity experiments with the Sentinel-AI framework. Neural plasticity refers to the system's ability to dynamically prune and regrow attention heads in transformer models, optimizing for both performance and efficiency.

## Overview

Neural plasticity is a biologically-inspired approach that allows transformer models to adapt their structure during training. The system uses entropy and gradient-based metrics to identify and prune less important attention heads, then applies focused training to recover or improve performance with the reduced structure.

## Key Features

- **Dynamic Head Pruning**: Identifies and removes underperforming attention heads
- **Differential Learning Rates**: Applies higher learning rates to areas that need to adapt after pruning
- **Multiple Pruning Strategies**: Supports entropy-based, gradient-based, combined, and random pruning
- **Performance Visualization**: Comprehensive dashboards to analyze pruning decisions and performance
- **Interactive Dashboards**: HTML dashboards with visualizations of attention patterns and metrics
- **Attention Analysis**: Detailed visualization of attention head behavior before and after pruning

## Running Experiments

The main entry point is `scripts/run_neural_plasticity.py`, which provides a comprehensive CLI for running experiments.

### Basic Usage

```bash
# Run a basic experiment with default settings
python scripts/run_neural_plasticity.py

# Run a quick test with minimal data
python scripts/run_neural_plasticity.py --quick_test

# Run with a specific model and pruning strategy
python scripts/run_neural_plasticity.py --model_name distilgpt2 --pruning_strategy entropy --pruning_level 0.2
```

### Advanced Usage

```bash
# Run multiple pruning cycles
python scripts/run_neural_plasticity.py --cycles 3 --training_steps 200

# Compare different pruning strategies
python scripts/run_neural_plasticity.py --compare_strategies

# Generate interactive dashboard
python scripts/run_neural_plasticity.py --use_dashboard

# Save the pruned model
python scripts/run_neural_plasticity.py --save_model
```

### Full Options

Run `python scripts/run_neural_plasticity.py --help` for a complete list of options:

- **Model Configuration**:
  - `--model_name`: Model name or path (default: "distilgpt2")
  - `--device`: Device to run on (cpu, cuda, auto)

- **Dataset Configuration**:
  - `--dataset`: Dataset name (default: "wikitext")
  - `--dataset_config`: Dataset configuration (default: "wikitext-2-raw-v1")
  - `--batch_size`: Batch size
  - `--max_length`: Maximum sequence length (default: 128)

- **Pruning Configuration**:
  - `--pruning_strategy`: Pruning strategy (entropy, magnitude, random, combined)
  - `--pruning_level`: Pruning level (0.0 to 1.0) (default: 0.2)
  - `--learning_rate`: Learning rate (default: 5e-5)
  - `--cycles`: Number of pruning cycles (default: 1)
  - `--training_steps`: Training steps per cycle (default: 100)

- **Experiment Mode**:
  - `--quick_test`: Run quick test with minimal data
  - `--compare_strategies`: Compare multiple pruning strategies

- **Output Configuration**:
  - `--output_dir`: Output directory
  - `--save_model`: Save model after experiment
  - `--no_visualize`: Disable visualization generation
  - `--use_dashboard`: Generate interactive HTML dashboard
  - `--verbose`: Enable verbose output

## Output Structure

All experiment outputs are saved to the `/output/neural_plasticity_TIMESTAMP` directory, organized as follows:

```
/output/neural_plasticity_YYYYMMDD_HHMMSS/
├── dashboards/               # Interactive HTML dashboards
├── logs/                     # Experiment logs
├── models/                   # Saved model checkpoints
│   └── pruned_model/         # Final pruned model
└── visualizations/           # Generated visualizations
    ├── metrics_dashboard.png # Summary dashboard
    ├── entropy_heatmap.png   # Attention entropy visualization
    ├── pruning_decisions.png # Pruning decision visualization
    └── ...
```

## Advanced Topics

### Custom Pruning Strategies

The system supports multiple pruning strategies:

- **Entropy**: Prunes heads with high entropy (unfocused attention)
- **Magnitude**: Prunes heads with low gradient magnitudes (less learning)
- **Combined**: Uses a weighted combination of entropy and gradient metrics
- **Random**: Randomly selects heads to prune (useful as a baseline)

### Multi-cycle Pruning

Running multiple pruning cycles allows for incremental pruning and fine-tuning, often leading to better results than aggressive single-cycle pruning:

```bash
python scripts/run_neural_plasticity.py --cycles 3 --pruning_level 0.1
```

### Strategy Comparison

To identify the best pruning strategy for your specific model and task:

```bash
python scripts/run_neural_plasticity.py --compare_strategies --output_dir output/strategy_comparison
```

This will run experiments with different combinations of strategies and pruning levels, producing comparative visualizations.

## Extending the System

The neural plasticity framework is designed to be extensible:

1. **New Pruning Strategies**: Add new strategies in `utils/neural_plasticity/core.py`
2. **Custom Metrics**: Implement new importance metrics in the same file
3. **Visualization Extensions**: Extend the dashboard in `utils/neural_plasticity/dashboard.py`
4. **New Models**: The system automatically detects model structure for most HuggingFace models

## References

For more information on the theoretical background:

- Entropy-based attention pruning: See `docs/pruning/entropy_magnitude_pruning.md`
- Gradient-based head importance: See `docs/pruning/methods_magnitude.md`
- Neural plasticity principles: See `docs/neural_plasticity.md`
EOF < /dev/null