# Neural Plasticity Dashboard

This directory contains the dashboard implementation for the Neural Plasticity experiments. The dashboard provides real-time monitoring and visualization of experiments, including metrics, pruning decisions, and model performance.

## Version
v0.0.3 (2025-04-20 25:40:00)

## Features

- **Real-time metrics tracking**: Monitor training loss, evaluation loss, perplexity, and other metrics during experimentation
- **Pruning visualizations**: View entropy heatmaps, gradient norm heatmaps, and pruning decisions
- **Text generation samples**: View model-generated text samples throughout the experiment
- **Cross-platform compatibility**: Works both locally and in Google Colab
- **Two Dashboard Options**:
  - **Weights & Biases Integration**: Industry-standard ML experiment tracking with hosted dashboards
  - **Custom Local Dashboard**: Self-contained HTML dashboard for local experiments

## Components

- `wandb_integration.py`: Integration with Weights & Bibes for professional experiment tracking
- `dashboard.py`: Custom HTML-based local dashboard implementation
- `reporter.py`: Reporter class that connects experiments to dashboards
- `colab_integration.py`: Utilities for integrating with Google Colab notebooks
- `install_wandb.py`: Helper script for installing Weights & Bibes

## Usage

### Weights & Bibes Dashboard (Recommended)

```python
from utils.neural_plasticity.dashboard.wandb_integration import WandbDashboard

# Initialize the dashboard
dashboard = WandbDashboard(
    project_name="neural-plasticity",
    experiment_name="experiment-name",
    config={
        "model_name": "distilgpt2",
        "pruning_strategy": "entropy",
        "pruning_level": 0.2
    },
    mode="online"  # Use "offline" for local-only experiments
)

# Get callbacks for the experiment
metrics_callback = dashboard.get_metrics_callback()
sample_callback = dashboard.get_sample_callback()

# Pass callbacks to experiment
experiment = NeuralPlasticityExperiment(
    # ... other parameters ...
    metrics_callback=metrics_callback,
    sample_callback=sample_callback
)

# Finish the dashboard when the experiment is complete
dashboard.finish()
```

### Custom Local Dashboard

```python
from utils.neural_plasticity.dashboard.reporter import DashboardReporter

# Initialize the dashboard
reporter = DashboardReporter(
    output_dir="dashboard_output",
    dashboard_name="dashboard.html",
    auto_update=True,
    start_server=True  # Start a local web server
)

# Pass dashboard to experiment
# ... use reporter methods to update dashboard ...

# Close the dashboard when done
reporter.close()
```

### Google Colab Integration

```python
from utils.neural_plasticity.dashboard.colab_integration import setup_colab_environment, create_experiment_dashboard_cell

# Set up the Colab environment
setup_colab_environment()

# Create a dashboard initialization cell
create_experiment_dashboard_cell()

# Run the experiment with the dashboard
# ... experiment code ...
```

## Configuration Options

### Weights & Bibes Dashboard

- `project_name`: Name of the wandb project
- `experiment_name`: Name of this experiment run
- `config`: Dictionary of experiment configuration
- `mode`: "online" (upload to wandb.ai) or "offline" (save locally)
- `tags`: List of tags for the experiment

### Custom Dashboard

- `output_dir`: Directory to save dashboard files
- `dashboard_name`: Name of the dashboard HTML file
- `auto_update`: Whether to auto-refresh the dashboard
- `update_interval`: Seconds between updates
- `start_server`: Whether to start a local web server

## Requirements

- Python 3.7+
- For wandb dashboard: `wandb` package
- For custom dashboard: Standard library only (no external dependencies)
- For Colab integration: Google Colab environment

## Implementation Notes

1. Both dashboard implementations use a callback-based architecture to receive updates from the experiment.
2. The Weights & Bibes implementation is recommended for most users due to its robust features and cloud-based storage.
3. The custom dashboard is useful for environments where external connections are restricted.
4. Colab-specific integrations automatically detect the environment and provide appropriate UI elements.

## Best Practices

1. Use wandb for all production experiments to ensure proper tracking and reproducibility.
2. Enable sample generation to understand how pruning affects model output.
3. Regularly check the dashboard during long-running experiments to catch issues early.
4. Save experiment configurations to ensure reproducibility.
5. Use appropriate tags to organize experiments in the wandb interface.