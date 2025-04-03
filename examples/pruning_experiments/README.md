# Pruning Experiments

This directory contains examples, tests, and utilities for running pruning and fine-tuning experiments using the modular experiment framework.

## Directory Structure

```
pruning_experiments/
├── README.md               # This file
├── notebooks/              # Jupyter notebooks for interactive experiments
│   └── PruningStrategiesComparison.ipynb  # Notebook comparing pruning strategies
├── scripts/                # Standalone scripts for running experiments
│   └── run_experiment.py   # Command-line script for running experiments
└── tests/                  # Unit tests for the experiment framework
    └── test_pruning_experiment.py  # Tests for the PruningExperiment classes
```

## Command-line Script Usage

The `run_experiment.py` script provides a command-line interface for running pruning and fine-tuning experiments:

```bash
# Run a basic experiment with default settings
python examples/pruning_experiments/scripts/run_experiment.py

# Specify a model and strategies
python examples/pruning_experiments/scripts/run_experiment.py --model distilgpt2 --strategies random entropy --pruning_levels 0.1 0.3 0.5

# Run a longer experiment with more fine-tuning epochs
python examples/pruning_experiments/scripts/run_experiment.py --epochs 5 --runtime 3600 --output_dir my_results
```

### Command-line Arguments

- `--model`: Specific model to test (e.g., "distilgpt2", "gpt2", "facebook/opt-125m")
- `--models`: List of models to test (alternative to --model)
- `--strategies`: List of pruning strategies to use (default: "random entropy") 
- `--pruning_levels`: List of pruning levels to use (default: 0.1 0.3 0.5)
- `--epochs`: Number of fine-tuning epochs (default: 1)
- `--prompt`: Text prompt for evaluation
- `--runtime`: Maximum runtime in seconds (default: 3600 - 1 hour)
- `--output_dir`: Directory to save results and plots (default: "experiment_results")
- `--debug`: Enable debug output

## Interactive Notebook

The `PruningStrategiesComparison.ipynb` notebook provides an interactive environment for comparing different pruning strategies on the same model. It includes visualizations and analysis tools.

To use in Google Colab:
1. Upload the notebook to Colab
2. Run the cells in order
3. The notebook will automatically clone the repository and set up the environment

## Running Tests

To run the unit tests:

```bash
# Run basic tests
python -m unittest examples/pruning_experiments/tests/test_pruning_experiment.py

# Run intensive tests (requires model download)
RUN_INTENSIVE_TESTS=1 python -m unittest examples/pruning_experiments/tests/test_pruning_experiment.py
```

The tests verify that the `PruningExperiment` and `PruningFineTuningExperiment` classes work correctly with various configurations.

## Integration with Main Code

These examples use the modular experiment framework from `utils/pruning/experiment.py`, which provides:

- Environment detection for optimal hardware utilization
- NaN prevention for stable training
- Memory management for large models
- Comprehensive result visualization
- Multi-experiment management

The examples demonstrate best practices for using the framework in different contexts.