# Neural Plasticity for Transformer Models

This document describes the neural plasticity cycle implementation in Sentinel AI, which enables transformer models to dynamically adapt their architecture through pruning and strategic growth of attention heads.

## Overview

Neural plasticity enables transformer models to:
1. **Reduce computational cost** by pruning underutilized attention heads
2. **Maintain performance** through strategic regrowth of heads where they're most needed
3. **Adapt to new tasks** by reorganizing model structure based on data patterns
4. **Transfer knowledge** through U-Net-like skip connections between pruned and new heads

The neural plasticity cycle follows four stages:

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│   PRUNE  ├────►  MEASURE  ├────►   GROW   ├────►   LEARN  │
└─────────┘     └──────────┘     └─────────┘     └─────────┘
     ▲                                                │
     └────────────────────────────────────────────────┘
```

1. **Prune**: Remove underutilized attention heads based on entropy, magnitude, or other metrics
2. **Measure**: Evaluate model performance and identify areas for improvement
3. **Grow**: Strategically add new attention heads where they would be most beneficial
4. **Learn**: Fine-tune the model with higher learning rates for newly added heads

## Implementation

The neural plasticity cycle is implemented through the following components:

### 1. Pruning Module (`utils/pruning/pruning_module.py`)

The core module for manipulating transformer model attention heads, supporting:
- Loading and initializing transformer models
- Zeroing out attention head weights
- Evaluating model performance after modifications

### 2. Pruning Strategies (`utils/pruning/strategies.py`)

Different strategies for determining which heads to prune:
- `RandomStrategy`: Random pruning (baseline)
- `MagnitudeStrategy`: Prune heads with smallest weight magnitudes
- `AttentionEntropyStrategy`: Prune heads with most deterministic attention patterns

### 3. Growth Strategies (`utils/pruning/growth.py`)

Strategies for determining where to add new attention heads:
- `GradientSensitivityStrategy`: Add heads where they would have the most impact based on gradients
- `EntropyGapStrategy`: Add heads where there's a significant entropy gap in attention patterns
- `BalancedStrategy`: Ensure even distribution of heads across layers
- `RandomStrategy`: Randomly select positions for new heads

### 4. Head Learning Rate Management (`utils/head_lr_manager.py`)

Manages differential learning rates to accelerate adaptation of new heads:
- Higher learning rates for newly added heads
- Lower learning rates for existing heads to maintain stability

## Usage

### Basic Neural Plasticity Cycle

To perform the full neural plasticity cycle, use the provided scripts:

```bash
# 1. Prune a model
python scripts/prune_heads.py --model_name distilgpt2 --pruning_level 0.3 --strategy entropy --output_path ./pruned_model.pth

# 2. Grow new heads in strategic locations
python scripts/expand_heads.py --model_path ./pruned_model.pth --model_name distilgpt2 --growth_percentage 0.1 --growth_strategy gradient_sensitivity --output_path ./grown_model.pth

# 3. Fine-tune the model
python scripts/fine_tune_model.py --model_path ./grown_model.pth --model_name distilgpt2 --learning_rate 5e-5 --new_head_lr_multiplier 5.0 --epochs 3 --output_path ./fine_tuned_model.pth
```

### Complete Experiment

To run a complete neural plasticity experiment with detailed metrics and visualization:

```bash
python scripts/neural_plasticity_experiment.py --model_name distilgpt2 --pruning_levels 0.1,0.3,0.5 --growth_percentages 0.05,0.1,0.2 --save_visualizations
```

### Unit Testing

To verify the functionality of the head growth implementation:

```bash
python scripts/test_head_growth_unit.py
```

### Colab Notebook

For an interactive demonstration of the neural plasticity cycle, run the Colab notebook:

```
notebooks/NeuralPlasticityDemo.ipynb
```

## Growth Strategies in Detail

### Gradient Sensitivity Strategy

This strategy identifies positions where adding a new attention head would have the most impact on the model's gradients. It works by:

1. Calculating the sensitivity of model outputs to potential new heads
2. Ranking potential head positions by gradient magnitude
3. Adding heads where gradients indicate they would be most influential

```python
# Sample usage
grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
    pruning_module,
    params=pruned_params,
    active_heads=pruned_active_heads,
    growth_percentage=0.1,
    strategy="gradient_sensitivity",
    initial_scale=0.01
)
```

### Entropy Gap Strategy

This strategy adds heads where there's a significant entropy gap in attention patterns, indicating areas where more diverse attention could be beneficial:

1. Calculate attention entropy for each layer
2. Identify layers with high entropy gaps (difference between optimal and actual entropy)
3. Add heads where entropy gaps are largest

### Balanced Strategy

This simpler strategy ensures heads are distributed evenly across layers:

1. Count active heads per layer
2. Add heads to layers with fewest active heads
3. Maintains architectural balance across the model

### Warmup Schedule

To prevent performance collapse when adding new heads, we use a gradual warmup schedule:

1. Initialize new heads with small weights (controlled by `initial_scale`)
2. Gradually increase their influence over `warmup_steps` steps
3. This allows the model to smoothly integrate new heads

```python
# The warmup schedule function returned by grow_attention_heads_gradually
scale = warmup_schedule(step)  # Scale increases from initial_scale to 1.0
```

## Performance Considerations

### Head Pruning Impact

Typically, models can tolerate pruning of 30-50% of attention heads with minimal performance degradation (less than 5% increase in perplexity). This results in significant computation savings.

### Growth Efficiency

Strategically growing back just 10-20% of the pruned heads can recover most of the performance loss, leading to net efficiency gains of 20-40% with negligible performance impact.

### Differential Learning Rates

Using higher learning rates for new heads (3-5x the base rate) accelerates their integration and improves overall model adaptation.

## Future Directions

1. **Iterative Plasticity Cycles**: Multiple cycles of pruning and growth for progressive adaptation
2. **Cross-Modal Transfers**: Adapting models across different modalities while preserving knowledge
3. **Dynamic Adaptive Inference**: Runtime adjustment of active heads based on input complexity
4. **Structural Knowledge Distillation**: Using neural plasticity for more efficient knowledge transfer
5. **Meta-Plasticity**: Learning the optimal pruning and growth strategies from data

## References

- "Analyzing and Improving the Training Dynamics of Pruned Transformer Heads" (Michel et al., 2019)
- "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, 2019)
- "Movement Pruning: Adaptive Sparsity by Fine-Tuning" (Malladi et al., 2020)
- "Structured Pruning of Large Language Models" (Fan et al., 2020)
- "What Does BERT Look At? An Analysis of BERT's Attention" (Clark et al., 2019)