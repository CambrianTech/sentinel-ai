# Colab Integration Utilities

This directory contains utilities for integrating SentinelAI with Google Colab.

## Modules

- `helpers.py` - Helper functions for setting up the Colab environment
- `experiment_ui.py` - Modular experiment UI framework

## ModularExperimentRunner

The `ModularExperimentRunner` class in `experiment_ui.py` provides a unified interface for
configuring and running experiments with a graphical UI in Colab environments.

Key features:
- Interactive UI with dropdowns, sliders, and checkboxes
- Integration with pruning and fine-tuning experiments
- Support for adaptive plasticity experiments
- Auto-optimization of parameters based on environment
- Result visualization and serialization

## Usage in Notebooks

```python
from utils.colab.experiment_ui import launch_experiment_ui

# Launch the UI
launch_experiment_ui()
```

For programmatic usage:

```python
from utils.colab.experiment_ui import ModularExperimentRunner

# Create a runner
runner = ModularExperimentRunner()

# Configure
runner.update_config(
    model="distilgpt2",
    pruning_strategy="entropy",
    pruning_level=0.3
)

# Run
runner.create_experiment()
runner.run_experiment()
```

## Environment Detection and Optimization

The `setup_colab_environment` function in `helpers.py` detects and configures the Colab
environment, including:

- GPU availability
- Memory constraints
- Optimal batch sizes and sequence lengths
- JAX/PyTorch configuration

## Extending the UI

To add new parameters to the UI:
1. Add defaults to the `config` dictionary in `ModularExperimentRunner.__init__`
2. Add UI widgets in `create_ui` method
3. Update parameter mapping in the run button handler
4. Ensure the new parameters are passed to the experiment setup

For adding completely new experiment types, see the experiment framework in
`utils/pruning/experiment.py`.
