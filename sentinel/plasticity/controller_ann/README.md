# Adaptive Neural Plasticity with ANN Controller

## Overview

This module implements a comprehensive neural plasticity workflow integrating the ANN controller for intelligent pruning and fine-tuning of transformer models. It provides a full experiment pipeline that:

1. Uses the ANN controller to dynamically gate attention heads
2. Visualizes the complete plasticity process across multiple phases
3. Supports multi-cycle pruning and fine-tuning for progressive model refinement
4. Generates detailed dashboards and visualizations

## Key Components

- `adaptive_experiment.py`: Core experiment implementation with ANN controller
- `model_wrapper.py`: Model wrapper for dynamic attention head control
- `run_experiment.py`: Command-line runner for adaptive experiments
- `visualization.py`: Integration with multi-phase dashboard

## Usage

### Basic Usage

```bash
cd /path/to/sentinel-ai
source .venv/bin/activate
python -m sentinel.plasticity.controller_ann.run_experiment --model_name gpt2
```

### Advanced Usage

```bash
python -m sentinel.plasticity.controller_ann.run_experiment \
  --model_name gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --batch_size 8 \
  --warmup_steps 200 \
  --finetuning_steps 200 \
  --cycles 3 \
  --pruning_level 0.2 \
  --entropy_threshold 1.5 \
  --controller_lr 0.01
```

## Implementation Details

The adaptive experiment integrates multiple components:

1. **ANN Controller**: Uses `sentinel.controller.controller_ann.ANNController` for dynamic attention gating

2. **Plasticity Integration**: Extends the neural plasticity framework with controller-based pruning

3. **Multi-Phase Dashboard**: Provides comprehensive visualizations of the entire process

4. **Dashboard HTML Generation**: Creates standalone interactive dashboards for experiment review

## Experiment Flow

1. **Setup**: Initialize model, controller, and dashboard
2. **Warmup Phase**: Initial training to establish baseline behavior
3. **Analysis Phase**: Collect metrics for pruning decisions (entropy, gradients)
4. **Pruning Phase**: Identify and prune less important attention heads
5. **Fine-tuning Phase**: Recover and improve performance after pruning
6. **Repeat**: Run multiple cycles of analysis → prune → fine-tune
7. **Evaluation**: Compare baseline and pruned model performance
8. **Visualization**: Generate dashboards and visualizations

## Multi-Phase Dashboard

The experiment generates comprehensive visualizations including:

- **Complete Process Timeline**: Shows all phases with color coding
- **Perplexity Tracking**: Perplexity changes throughout training
- **Sparsity Tracking**: Model sparsity over time
- **Attention Head Analysis**: Heatmaps showing head activity and importance
- **Phase Details**: Detailed metrics for each training phase

## References

- ANN Controller: `sentinel/controller/controller_ann.py`
- Neural Plasticity Base: `sentinel/plasticity/neural_plasticity_experiment.py`
- Project Documentation: `docs/neural_plasticity_controller_experiment.md`