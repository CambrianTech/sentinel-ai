# Neural Plasticity Dashboard - Multi-Phase Tracking System

## Overview
This document describes the multi-phase dashboard system for neural plasticity experiments, providing comprehensive visualization and tracking capabilities across different phases and multiple pruning cycles.

Version: v1.0.0 (2025-04-20 21:30:00)

## Key Features

### 1. Phase-Based Tracking
The dashboard tracks experiments through distinct phases:
- **Warmup**: Initial training before pruning starts
- **Analysis**: Period for measuring and analyzing model attributes
- **Pruning**: Active pruning of attention heads
- **Fine-tuning**: Post-pruning recovery and optimization
- **Evaluation**: Comprehensive assessment of pruned model

### 2. Multi-Cycle Support
- Tracks multiple pruning cycles in a single experiment
- Each cycle can have its own pruning strategy and target
- Visualizes cycle boundaries and transitions
- Provides per-cycle performance metrics

### 3. Comprehensive Visualizations
- **Complete Process Timeline**: Unified view of the entire training process
- **Per-Phase Details**: Focused visualizations of each experimental phase
- **Metric Tracking**: Loss, perplexity, and sparsity over time
- **Event Markers**: Pruning events, stabilization points, and phase transitions
- **Multi-Cycle Analysis**: Detailed breakdown of performance across cycles

### 4. Integration Points
- **Experiment Runner**: Direct integration with `run_neural_plasticity.py`
- **WandB Support**: Online tracking with Weights & Biases
- **Standalone Mode**: Self-contained HTML dashboard without external dependencies
- **Browser Integration**: Automatic dashboard opening for immediate viewing

## Dashboard Components

### Timeline Visualization
The top section displays a unified timeline showing:
- Color-coded phase regions
- Pruning event markers
- Stabilization points
- Loss curves throughout the process

### Metrics Tracking
Several metrics are tracked and visualized:
- **Loss**: Training loss over time
- **Perplexity**: Language modeling perplexity
- **Sparsity**: Percentage of pruned heads
- **Performance Impact**: Changes in model quality

### Phase Details
Each phase has dedicated visualizations:
- **Warmup**: Stabilization curves and convergence
- **Pruning**: Head importance scores, pruning decisions
- **Fine-tuning**: Recovery metrics after pruning

### Multi-Cycle Analysis
For experiments with multiple pruning cycles:
- Cycle-by-cycle breakdown
- Comparative performance between cycles
- Cumulative pruning impact
- Per-cycle optimization statistics

## Usage

### Running with Multi-Phase Dashboard
```python
python scripts/run_neural_plasticity.py --model_name gpt2 --pruning_strategy entropy --pruning_level 0.2 --cycles 3 --use_dashboard
```

### Command-line Options
- `--cycles`: Number of pruning cycles (>1 enables multi-cycle view)
- `--use_dashboard`: Enable visualization dashboard
- `--output_dir`: Output directory for dashboard files
- `--no_visualize`: Disable visualization generation

### Programmatic Usage
```python
from utils.neural_plasticity.dashboard.multi_phase_dashboard import MultiPhaseDashboard

# Initialize dashboard
dashboard = MultiPhaseDashboard(
    project_name="neural-plasticity",
    experiment_name="my-experiment",
    output_dir="output/my_experiment",
    config=experiment_config
)

# Record phase transitions
dashboard.record_phase_transition("warmup", 0)

# Record metrics for each step
dashboard.record_step({
    "loss": 2.5,
    "perplexity": 12.1,
    "sparsity": 0.0,
    "phase": "warmup"
}, step=10)

# Record pruning events
dashboard.record_pruning_event({
    "strategy": "entropy",
    "pruning_level": 0.2,
    "pruned_heads": [(0, 2), (1, 3), (2, 1)],
    "cycle": 1
}, step=100)

# Generate visualizations
dashboard.visualize_complete_process("output/complete_process.png")
dashboard.generate_multi_cycle_dashboard("output/multi_cycle.png")
dashboard.generate_standalone_dashboard("output/dashboard")
```

## Technical Implementation
The dashboard is implemented as a Python class `MultiPhaseDashboard` that extends the base `WandbDashboard`:

- **Data Structures**: Metrics stored in phase-specific arrays
- **Visualization Engine**: Matplotlib for generating plots
- **HTML Generation**: Self-contained dashboard template
- **Event Tracking**: Phase transitions, pruning events, stabilization

## Example Output
The dashboard generates multiple visualizations:

1. **Complete Process**: `complete_process.png`
2. **Multi-Cycle Analysis**: `multi_cycle_process.png`
3. **HTML Dashboard**: `dashboard.html`

## Future Enhancements
- Interactive JavaScript-based visualizations
- Comparative visualizations between different experiments
- Attention head importance heatmaps
- Embedding space visualizations
- Automated optimal cycle determination