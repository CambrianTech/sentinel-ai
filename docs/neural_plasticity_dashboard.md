# Neural Plasticity Dashboard - Multi-Phase Tracking System

## Overview
This document describes the multi-phase dashboard system for neural plasticity experiments, providing comprehensive visualization and tracking capabilities across different phases and multiple pruning cycles, with special support for ANN controller integration.

Version: v1.1.0 (2025-04-20 23:15:00)

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
- **Head Metrics Analysis**: Detailed visualization of head-level metrics
- **Controller Timeline**: Tracking of controller decisions over time
- **Head Activity Heatmap**: Visualization of head activity patterns
- **Head Recovery Tracking**: Analysis of pruned head recovery patterns

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
- Controller decisions and interventions

### Metrics Tracking
Several metrics are tracked and visualized:
- **Loss**: Training loss over time
- **Perplexity**: Language modeling perplexity
- **Sparsity**: Percentage of pruned heads
- **Performance Impact**: Changes in model quality
- **Controller Activities**: Number of active heads managed by controller
- **Head Importance**: Metrics of head significance over time
- **Entropy and Magnitude**: Key head performance indicators

### Phase Details
Each phase has dedicated visualizations:
- **Warmup**: Stabilization curves and convergence
- **Pruning**: Head importance scores, pruning decisions
- **Fine-tuning**: Recovery metrics after pruning
- **Analysis**: Detailed head-level metric distributions

### Multi-Cycle Analysis
For experiments with multiple pruning cycles:
- Cycle-by-cycle breakdown
- Comparative performance between cycles
- Cumulative pruning impact
- Per-cycle optimization statistics
- Head recovery patterns across cycles

### Controller Integration Visualizations
For experiments with ANN controller:
- **Controller Decision Timeline**: Tracking of gating decisions
- **Head Activity Heatmap**: Visualization of which heads are active
- **Head Importance Heatmap**: Visualization of head importance scores
- **Recovery Tracking**: Analysis of pruned head reactivation
- **Layer-wise Metrics**: Performance tracking across model layers

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
# Import from the Sentinel framework
from sentinel.plasticity.visualization.multi_phase.dashboard import MultiPhaseDashboard

# Initialize dashboard
dashboard = MultiPhaseDashboard(
    output_dir="output/my_experiment",
    experiment_name="my-experiment",
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

# Record controller decisions and head metrics (for ANN controller integration)
dashboard.record_controller_decision({
    "active_heads": [(0, 1), (1, 0), (2, 3)],
    "controller_loss": 0.15
}, step=50)

dashboard.record_head_metrics({
    "0.1": {"entropy": 1.2, "magnitude": 0.8, "importance": 0.75},
    "1.0": {"entropy": 0.9, "magnitude": 1.2, "importance": 0.85}
}, step=50)

dashboard.record_attention_entropy({
    "0.1": 1.2,
    "1.0": 0.9
}, step=50)

dashboard.record_head_importance({
    "0.1": 0.75,
    "1.0": 0.85
}, step=50)

# Record head recovery for controller-managed heads
dashboard.record_head_recovery({
    "0.2": 0.8,  # Head (0,2) has recovered with score 0.8
    "1.3": 0.65  # Head (1,3) has recovered with score 0.65
}, step=150)

# Generate visualizations
dashboard.visualize_complete_process("output/complete_process.png")
dashboard.generate_multi_cycle_dashboard("output/multi_cycle.png")
dashboard.visualize_head_metrics("output/head_metrics.png")
dashboard.generate_controller_dashboard("output/controller_dashboard.png")
dashboard.generate_standalone_dashboard("output/dashboard")
dashboard.save_dashboard_data("output/dashboard_data")
```

## Technical Implementation
The dashboard is implemented as a Python class `MultiPhaseDashboard` in the Sentinel framework:

- **Data Structures**: 
  - Phase-specific metrics arrays (warmup, pruning, fine-tuning)
  - Head-level metrics dictionaries for detailed tracking
  - Controller decisions history
  - Active/pruned head tracking sets
  - Attention entropy and magnitude history

- **Visualization Engine**: 
  - Matplotlib for generating static visualizations
  - Custom color schemes for different phases and events
  - Interactive components in HTML dashboard
  - Heatmap visualizations for head-level metrics

- **HTML Generation**: 
  - Self-contained dashboard template with CSS and JavaScript
  - Tabbed interface for different visualization categories
  - Interactive elements for exploring detailed metrics
  - Responsive design adapting to different screen sizes

- **Event Tracking**: 
  - Phase transitions with timestamps
  - Pruning events with head identifiers
  - Controller decisions with gating information
  - Stabilization points with detection criteria
  - Head recovery events with scores

## Example Output
The dashboard generates multiple visualizations:

1. **Complete Process**: `complete_process.png`
2. **Multi-Cycle Analysis**: `multi_cycle_process.png`
3. **Controller Dashboard**: `controller_dashboard.png`
4. **Head Metrics**: `head_metrics.png`
5. **HTML Dashboard**: `dashboard.html` with tabbed sections:
   - Process Overview
   - Pruning Cycles
   - Controller Metrics
   - Head Metrics
6. **Dashboard Data**: JSON files in `dashboard_data/` for further analysis

## Future Enhancements
- Fully interactive JavaScript-based visualizations with D3.js
- Comparative visualizations between different experiments
- Attention head importance heatmaps with detailed cross-references
- Embedding space visualizations for semantic understanding
- Automated optimal cycle determination
- Real-time controller decision visualization
- Animated phase transitions and pruning events
- 3D visualizations of head activity across layers
- Integration with model registry for experiment comparison
- Customizable dashboard layouts for different research focuses
- Export to publication-ready figures for papers
- Extended video generation for presentations
- Compatibility with other pruning frameworks for comparison