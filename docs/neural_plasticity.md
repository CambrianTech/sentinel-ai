# Neural Plasticity in Transformer Models

## Introduction

Neural plasticity is a biomimetic approach to transformer model optimization that draws inspiration from the brain's ability to form and reorganize neural connections. This document details the implementation of the neural plasticity cycle in Sentinel-AI, which follows four key phases: train → prune → measure → grow → learn.

The plasticity system enables transformer models to dynamically adapt their architecture by removing less important attention heads and strategically growing new ones in positions where they can have the greatest impact. This approach offers several benefits:

1. **Performance Efficiency**: Models can maintain or improve performance with fewer parameters
2. **Specialization**: New heads can specialize in patterns that weren't captured by the original model
3. **Continuous Adaptation**: Models can undergo multiple cycles of plasticity for iterative refinement

## Core Concepts

### 1. Pruning

Pruning is the selective removal of attention heads based on importance metrics. Several strategies are available:

- **Entropy**: Removes heads with lowest attention entropy (least information content)
- **Magnitude**: Removes heads with smallest weight magnitudes
- **Random**: Removes heads randomly (useful as a baseline)

Pruning is performed by zeroing out weights associated with specific attention heads, effectively removing them from computation while preserving the overall model structure.

### 2. Measurement

After pruning, model performance is measured to understand the impact of head removal. Metrics include:

- **Perplexity**: Primary metric for assessing language modeling quality
- **Generation quality**: Sample texts are generated to evaluate model capabilities
- **Active head count**: Tracking how many heads remain active

### 3. Growth

Growth is the strategic addition of new attention heads to replace a portion of those pruned. Available strategies:

- **Gradient Sensitivity**: Adds heads where they would have most impact based on gradient measurements
- **Entropy Gap**: Adds heads where there's a significant entropy gap in attention patterns
- **Balanced**: Ensures heads are distributed evenly across layers
- **Random**: Adds heads randomly (useful as a baseline)

New heads are initialized with small weight values to prevent disruption to existing functionality.

### 4. Learning

The learning phase fine-tunes the model with a focus on integrating the newly added heads. Key features:

- **Differential Learning Rates**: Higher learning rates for new heads to accelerate their specialization
- **Warmup Schedule**: Gradually increases the influence of new heads during training
- **Parameter Tracking**: Monitors the convergence of new head parameters

## Implementation Architecture

The neural plasticity system is implemented across several modules:

### Core Script

- `scripts/neural_plasticity_cycle.py`: Orchestrates the complete plasticity cycle with command-line interface

### Supporting Modules

- `utils/pruning/pruning_module.py`: Core implementation for model loading and head pruning
- `utils/pruning/strategies.py`: Pruning strategies implementation
- `utils/pruning/growth.py`: Head growth strategies implementation
- `utils/pruning/head_lr_manager.py`: Differential learning rate management
- `utils/train_utils.py`: Training implementation with plasticity-aware features
- `utils/charting.py`: Visualization utilities for neural plasticity
- `utils/metrics_logger.py`: Logging utilities for tracking plasticity metrics

## Detailed Process Flow

The neural plasticity cycle involves the following detailed steps:

1. **Initial Training**
   - Load pre-trained model or train from scratch
   - Evaluate initial performance as baseline
   - Store active head distribution for comparison

2. **Pruning Phase**
   - Calculate head importance using selected strategy
   - Sort heads by importance
   - Prune least important heads up to specified threshold
   - Evaluate model performance post-pruning

3. **Measurement Phase**
   - Analyze performance impact of pruning
   - Calculate perplexity change
   - Generate sample texts with pruned model
   - Log metrics and visualize remaining head distribution

4. **Growth Phase**
   - Identify optimal positions for new heads using selected strategy
   - Initialize new heads with small weight values
   - Create warmup schedule for gradual integration
   - Evaluate model with newly added heads

5. **Learning Phase**
   - Configure differential learning rates for new vs. existing heads
   - Fine-tune the model with emphasis on new head integration
   - Evaluate final model performance
   - Compare with baseline and pruned metrics

6. **Cycle Completion**
   - Generate visualizations and metrics reports
   - Save model checkpoints if requested
   - Prepare for potential next plasticity cycle

## Usage Examples

### Basic Usage

```bash
python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --dataset tiny_shakespeare
```

### Customized Configuration

```bash
python scripts/neural_plasticity_cycle.py \
  --model_name gpt2 \
  --dataset tiny_shakespeare \
  --initial_training_steps 500 \
  --initial_pruning 0.3 \
  --growth_ratio 0.33 \
  --learning_steps 300 \
  --pruning_strategy entropy \
  --growth_strategy gradient_sensitivity \
  --new_head_lr_multiplier 5.0 \
  --save_visualizations
```

### Multiple Plasticity Cycles

```bash
python scripts/neural_plasticity_cycle.py \
  --model_name distilgpt2 \
  --cycles 3 \
  --save_model \
  --experiment_name plasticity_multi_cycle_experiment
```

## Visualization Outputs

The neural plasticity system generates several visualizations:

1. **Head Distribution Maps**: Heat maps showing active head distribution at different stages
2. **Metrics Comparison Charts**: Bar charts comparing performance metrics across stages
3. **Cycle Comparison Plots**: Line charts showing trends across multiple cycles
4. **Growth Pattern Visualizations**: Visualizations highlighting which heads were added and removed

## Results and Analysis

Early experiments with the neural plasticity cycle have demonstrated:

1. Models can maintain performance while reducing active head count by 20-40%
2. Strategic head growth outperforms random growth by 30-70% in perplexity recovery
3. Multiple cycles show progressive improvement with diminishing returns after 3-4 cycles
4. Newly grown heads often specialize in different patterns than the ones they replace

## Future Directions

The neural plasticity system can be enhanced in several ways:

1. **Continuous Plasticity**: Implementing on-the-fly pruning and growth during training
2. **Cross-Architecture Support**: Extending to a wider range of model architectures
3. **Improved Growth Heuristics**: Developing more sophisticated strategies for head positioning
4. **Theoretical Analysis**: Deeper investigation of why certain patterns of plasticity emerge
5. **Domain-Specific Recipes**: Creating tuned configurations for specific tasks and domains

## References

1. Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.
2. Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.
3. Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One?
4. Bau, A., et al. (2020). Identifying and Controlling Important Neurons in Neural Machine Translation.

## Appendix

### Command Line Arguments Reference

- `--model_name`: Model name (default: distilgpt2)
- `--dataset`: Dataset name (default: tiny_shakespeare)
- `--cycles`: Number of complete plasticity cycles to run (default: 1)
- `--initial_training_steps`: Initial training steps before first pruning (default: 500)
- `--initial_pruning`: Initial pruning level as fraction of total heads (default: 0.3)
- `--growth_ratio`: Ratio of pruned heads to grow back (default: 0.33)
- `--learning_steps`: Learning steps after each growth phase (default: 300)
- `--pruning_strategy`: Strategy for pruning (default: entropy)
- `--growth_strategy`: Strategy for growth (default: gradient_sensitivity)
- `--new_head_lr_multiplier`: Learning rate multiplier for new heads (default: 5.0)
- `--save_visualizations`: Save visualizations as PNG files
- `--save_model`: Save model checkpoints at each stage

### Compatibility Notes

The current implementation has been tested with the following models:
- DistilGPT-2
- GPT-2 (small)
- GPT-2 (medium)
- OPT-125M
- OPT-350M

Support for other architectures is planned in future updates.