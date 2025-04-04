# Experiments

This directory contains modular experiments for the SentinelAI project. The structure is designed
to promote organization and reusability across different experiment types.

## Directory Structure

- `configs/` - Configuration files for experiments
- `examples/` - Example code snippets and demonstrations
- `notebooks/` - Jupyter notebooks for interactive experiments
- `results/` - Output directories for experiment results (gitignored)
  - `modular_experiment/` - Results from modular experiments
  - `adaptive_plasticity/` - Results from adaptive plasticity experiments
- `scripts/` - Scripts for running experiments

## Usage

The ModularExperimentRunner in `utils/colab/experiment_ui.py` provides a unified interface
for running experiments with configurable parameters. This can be used both in the Colab
environment and in local development.

### Running Experiments

```python
from utils.colab.experiment_ui import ModularExperimentRunner

# Create a runner
runner = ModularExperimentRunner()

# Configure experiment
runner.update_config(
    model="distilgpt2",
    pruning_strategy="entropy",
    pruning_level=0.3,
    enable_fine_tuning=True,
    fine_tuning_epochs=2,
    results_dir="experiments/results/my_experiment"
)

# Run experiment
runner.create_experiment()
results = runner.run_experiment()
```

### Colab Integration

For Google Colab integration, use the provided notebook in `colab_notebooks/ModelImprovementColab.ipynb`.
This notebook includes a user interface for configuring and running experiments.

## Adding New Experiment Types

To add a new experiment type:

1. Create a new class that inherits from the base experiment class
2. Implement the required methods (setup, run, evaluate)
3. Register it in the ModularExperimentRunner
4. Add configuration parameters to the UI

See existing experiment types for examples.