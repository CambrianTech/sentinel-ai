# Adaptive Neural Plasticity System

The Adaptive Neural Plasticity System is a self-improving neural network architecture that dynamically optimizes transformer models by:

1. **Continuous Quality Monitoring**: Evaluates model outputs with comprehensive degeneration detection
2. **Neural Plasticity Cycles**: Implements cycles of pruning → measuring → growing → learning
3. **Strategy Adaptation**: Learns which pruning and growth strategies work best
4. **Memory-Based Optimization**: Remembers successful transformations and reuses them
5. **Visualization Tools**: Provides comprehensive visualizations of the optimization process

## Key Features

- **Automated Model Optimization**: Reduces model size while maintaining or improving performance
- **Degeneration Detection**: Identifies and penalizes poor quality outputs
- **Adaptive Strategies**: Self-adjusts pruning and growth strategies based on historical success
- **Differential Learning Rates**: Applies higher learning rates to newly grown attention heads
- **Comprehensive Visualization**: Visualizes gradient patterns, model structure evolution, and optimization metrics

## Main Components

### AdaptivePlasticitySystem

The core system that manages the entire optimization process:

```python
system = AdaptivePlasticitySystem(
    model_name="distilgpt2",
    dataset=dataset,
    output_dir="./output/adaptive_plasticity",
    device="cuda",
    verbose=True
)

# Run optimization
results = system.run_adaptive_optimization(
    max_cycles=10,
    initial_pruning_level=0.2,
    initial_growth_ratio=0.5
)
```

### Neural Plasticity Cycle

Each plasticity cycle consists of four phases:

1. **Pruning**: Identifies and removes less useful attention heads
2. **Measurement**: Evaluates the impact of pruning on model quality
3. **Growth**: Strategically grows new attention heads where they're most useful
4. **Learning**: Fine-tunes the model with differential learning rates

### Visualization

The system provides comprehensive visualization tools:

- **Head gradient visualizations**: Shows gradient activity with pruned/revived head overlays
- **Cycle comparisons**: Visualizes metrics across optimization cycles
- **Strategy effectiveness**: Compares the performance of different strategies
- **Dashboard generation**: Creates a comprehensive HTML dashboard of optimization results

## Usage

### Running the System

Use the `run_adaptive_plasticity.py` script:

```bash
python scripts/run_adaptive_plasticity.py --model_name distilgpt2 --dataset wikitext
```

### Testing the System

For a quick test of the system:

```bash
python scripts/test_adaptive_plasticity.py --model_name distilgpt2 --cycles 2
```

### Visualizing Results

To visualize results after optimization:

```bash
python scripts/visualize_plasticity_results.py --results_dir ./output/adaptive_plasticity/run_20250401-120000 --dashboard
```

## Key Metrics

The system optimizes for these metrics:

- **Perplexity**: How well the model predicts the next token (lower is better)
- **Degeneration Score**: Measures repetition, low diversity, and other quality issues
- **Head Count**: Number of active attention heads (lower is better)
- **Efficiency**: Perplexity per head (lower is better)

## Integration with Other Components

The adaptive plasticity system integrates with other components of Sentinel AI:

- **Pruning Module**: For head pruning and growth operations
- **Fine-Tuner**: For efficient fine-tuning with differential learning rates
- **Metrics Logger**: For tracking optimization metrics over time
- **Visualization Tools**: For creating visualizations of model structure and performance