# Neural Plasticity Visualization

This directory contains visualization tools for neural plasticity experiments, including dashboard generators, visualization utilities, and templates.

## Dashboard Generators

- `enhanced_dashboard.py`: Creates a rich, interactive dashboard with dynamic visualizations and tabbed navigation. Uses matplotlib for generating visualizations from experiment data.
- `simple_dashboard.py`: A minimal dashboard generator for quickly viewing experiment results.
- `dashboard_generator.py`: The original dashboard generator used by the neural plasticity experiment runner.
- `generate_dashboard.py`: A standalone dashboard generator with simple styling.

## Usage

### Enhanced Dashboard Generator

The enhanced dashboard generator creates a rich, interactive HTML dashboard with visualizations and tabbed navigation.

```bash
python scripts/neural_plasticity/visualization/enhanced_dashboard.py \
    --experiment_dir /path/to/experiment_output \
    --output_path /path/to/save/dashboard.html
```

### Simple Dashboard Generator

The simple dashboard generator creates a basic HTML dashboard with minimal styling.

```bash
python scripts/neural_plasticity/visualization/simple_dashboard.py \
    --experiment_dir /path/to/experiment_output \
    --output_path /path/to/save/dashboard.html
```

### Original Dashboard Generator

The original dashboard generator is used by the neural plasticity experiment runner and creates a comprehensive dashboard with advanced visualizations.

```bash
python scripts/neural_plasticity/visualization/dashboard_generator.py \
    --experiment_dir /path/to/experiment_output \
    --output_file /path/to/save/dashboard.html
```

## Templates

The `templates` directory contains HTML templates used by the dashboard generators.

## Integration

The dashboard generators are integrated with the neural plasticity experiment runner. After an experiment is completed, dashboards are automatically generated in the experiment output directory.

## Accessing Dashboards

Generated dashboards can be found in the `dashboards` directory within each experiment output directory. For convenience, a symlink to the latest dashboard is available at:

```
/Users/joel/Development/sentinel-ai/dashboards/latest_dashboard.html
```

## Version History

- v0.1.0 (2025-04-20): Initial implementation with enhanced, simple, and original dashboard generators