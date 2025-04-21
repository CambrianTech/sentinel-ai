# Adaptive Neural Plasticity with ANN Controller

## Overview

This experiment implements a complete neural plasticity workflow that intelligently refines a GPT-2 model through pruning and fine-tuning using the ANN controller. It provides comprehensive visualizations of the training process across multiple phases (warmup, analysis, pruning, fine-tuning) and cycles.

Version: v1.0.0 (2025-04-20)

## Features

- **ANN Controller Integration**: Uses the ANN-based Dynamic Controller to make pruning decisions
- **Multi-Phase Tracking**: Visualizes the complete plasticity process across phases
- **Multi-Cycle Support**: Runs multiple cycles of pruning and fine-tuning for progressive refinement
- **Comprehensive Visualization**: Generates detailed dashboards with metrics and visualizations
- **Attention Heatmaps**: Visualizes head importance and activity for interpretability
- **Performance Metrics**: Tracks perplexity, loss, and sparsity throughout training
- **Intelligent Pruning**: Uses entropy and gradient-based metrics to identify prunable heads
- **Fine-Tuning Recovery**: Recovers and improves performance after pruning

## Usage

### Basic Usage

```bash
cd /path/to/sentinel-ai
source .venv/bin/activate
python scripts/adaptive_neural_plasticity.py --model_name gpt2
```

### Advanced Usage

```bash
python scripts/adaptive_neural_plasticity.py \
  --model_name gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --batch_size 8 \
  --warmup_steps 200 \
  --finetuning_steps 200 \
  --cycles 3 \
  --pruning_level 0.2 \
  --learning_rate 5e-5 \
  --entropy_threshold 1.5 \
  --importance_threshold 0.7
```

### Available Parameters

#### Model Configuration
- `--model_name`: Model name or path (e.g., gpt2, distilgpt2) (default: gpt2)
- `--device`: Device to run on (cpu, cuda, auto) (default: auto-detect)

#### Dataset Configuration
- `--dataset`: Dataset name (default: wikitext)
- `--dataset_config`: Dataset configuration (default: wikitext-2-raw-v1)
- `--batch_size`: Batch size (default: 8)
- `--max_length`: Maximum sequence length (default: 128)

#### Experiment Configuration
- `--warmup_steps`: Number of warmup steps (default: 200)
- `--finetuning_steps`: Number of fine-tuning steps per cycle (default: 200)
- `--analyze_steps`: Number of steps for analysis phase (default: 10)
- `--cycles`: Number of pruning cycles (default: 3)
- `--pruning_level`: Pruning level (0.0 to 1.0) (default: 0.2)
- `--learning_rate`: Learning rate for training (default: 5e-5)

#### Controller Configuration
- `--entropy_threshold`: Entropy threshold for pruning (default: 1.5)
- `--importance_threshold`: Importance threshold for pruning (default: 0.7)
- `--controller_lr`: Controller learning rate (default: 0.01)

#### Output Configuration
- `--output_dir`: Output directory (default: experiment_output/neural_plasticity/run_[timestamp])
- `--verbose`: Enable verbose output (default: True)

## Experiment Flow

1. **Setup**: Initialize model, controller, and dashboard
2. **Warmup Phase**: Initial training to establish baseline behavior
3. **Analysis Phase**: Collect metrics for pruning decisions (entropy, gradients)
4. **Pruning Phase**: Identify and prune less important attention heads
5. **Fine-tuning Phase**: Recover and improve performance after pruning
6. **Repeat**: Run multiple cycles of analysis → prune → fine-tune
7. **Evaluation**: Compare baseline and pruned model performance
8. **Visualization**: Generate dashboards and visualizations

## Output

The experiment produces the following outputs in the specified output directory:

- `dashboards/`: Interactive HTML dashboard and visualizations
  - `dashboard.html`: Main dashboard with metrics and visualizations
  - `complete_process.png`: Complete process visualization
  - `multi_cycle_process.png`: Multi-cycle visualization (if applicable)
  - `attention_heatmaps.png`: Attention head activity and importance

- `pruned_model/`: Saved pruned model checkpoint
- `controller.pt`: Saved ANN controller state
- `metrics.json`: Recorded metrics across all phases
- `pruning_events.json`: Details of all pruning events

## Dashboard Features

The interactive dashboard provides:

- **Experiment Summary**: Key metrics and results
- **Complete Process Visualization**: Training curves with phase highlighting
- **Perplexity Tracking**: Perplexity changes throughout training
- **Sparsity Tracking**: Model sparsity over time
- **Attention Head Analysis**: Heatmaps showing head activity and importance
- **Text Generation Samples**: Comparison between baseline and pruned models
- **Phase Details**: Detailed metrics for each training phase

## Visualization Examples

The dashboard includes visualizations similar to the reference examples:
- Dynamic Neural Plasticity training loss with pruning events
- Complete Neural Plasticity Training Process with phase transitions
- Warmup loss curves with stabilization detection
- Attention Head Activity and Importance heatmaps

## Requirements

- PyTorch
- Transformers
- Matplotlib
- NumPy

## References

- ANN Controller: `sentinel/controller/controller_ann.py`
- Dashboard Implementation: `utils/neural_plasticity/dashboard/multi_phase_dashboard.py`
- Neural Plasticity Experiment: `utils/neural_plasticity/experiment.py`