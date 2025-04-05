# Sentinel AI Scripts

This directory contains utility scripts for various functions of the Sentinel AI system.

## Overview

These scripts serve multiple purposes including training, evaluation, benchmark generation, 
publication-ready figure generation, and experimentation with new features.

## Key Scripts

- `train.py`: Main training script for all model types
- `main.py`: Primary entry point for inference and text generation
- `benchmark.py`: Run standard benchmarks on model configurations
- `benchmark_with_metrics.py`: Comprehensive benchmarking with metrics collection, pruning analysis, and fine-tuning
- `generate_publication_figures.py`: Create figures for papers and presentations
- `inference.py`: Run text generation inference with loaded models

## Neural Plasticity Scripts

- `neural_plasticity_cycle.py`: Run the complete neural plasticity cycle (train → prune → measure → grow → learn)
- `neural_plasticity_experiment.py`: Run comprehensive experiments with multiple pruning/growth configurations
- `test_head_growth.py`: Test and validate head growth functionality
- `test_head_growth_unit.py`: Unit tests for head growth implementation

## Pruning Scripts

- `prune_heads.py`: Apply pruning to attention heads using various strategies
- `analyze_heads.py`: Analyze importance of attention heads in models
- `benchmark_pruning.py`: Compare the performance impact of different pruning strategies
- `pruning_impact_analyzer.py`: Analyze the impact of pruning on different model metrics
- `finetune_pruned_model.py`: Fine-tune models after pruning to recover performance
- `expand_heads.py`: Grow new heads in strategically selected positions
- `inference_with_pruning.py`: Test inference with various pruning strategies

## Agency Scripts

- `benchmark_agency.py`: Run benchmarks on agency specialization techniques
- `run_agency_validation.py`: Validate agency behavior against requirements
- `runtime_specialization.py`: Test dynamic specialization capabilities

## Optimization Scripts

- `profile_full_model.py`: Profile performance of the full model
- `profile_attention_optimization.py`: Profile optimized attention mechanism
- `test_optimization_levels.py`: Test different optimization levels
- `benchmark_optimization.py`: Benchmark optimization techniques

## Usage Examples

### Neural Plasticity Cycle

```bash
# Run a basic neural plasticity cycle
python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --dataset tiny_shakespeare

# Run multiple cycles with visualization
python scripts/neural_plasticity_cycle.py --model_name gpt2 --dataset wikitext --cycles 3 --save_visualizations

# Comprehensive experiment with multiple configurations
python scripts/neural_plasticity_experiment.py --model_name distilgpt2 \
    --pruning_levels 0.1,0.3,0.5 \
    --growth_percentages 0.05,0.1,0.2 \
    --save_visualizations
```

### Pruning Operations

```bash
# Prune a model and test inference
python scripts/prune_heads.py --model_name distilgpt2 --pruning_level 0.3 --strategy entropy
python scripts/inference_with_pruning.py --model_name distilgpt2 --strategy entropy --pruning_level 0.3

# Analyze head importance
python scripts/analyze_heads.py --model_name gpt2 --output_path ./head_analysis.json

# Comprehensive benchmarking with metrics collection
python scripts/benchmark_with_metrics.py --model_name distilgpt2 \
  --pruning_strategies "random,entropy,magnitude" \
  --pruning_levels "0.1,0.3,0.5" \
  --eval_dataset "gutenberg" \
  --use_real_data
```

### Benchmarking with Real Data

```bash
# Run benchmark with Project Gutenberg books
python scripts/benchmark_with_metrics.py \
  --model_name distilgpt2 \
  --output_dir ./benchmark_results \
  --pruning_strategies "random,entropy,magnitude" \
  --pruning_levels "0.1,0.3,0.5" \
  --learning_steps 100 \
  --learning_rate 2e-5 \
  --eval_dataset "gutenberg" \
  --use_real_data \
  --use_adaptive_lr

# Use specific books for more targeted benchmarks
python scripts/benchmark_with_metrics.py \
  --model_name distilgpt2 \
  --eval_dataset "sherlock" \
  --use_real_data

# Use pre-processed datasets from previous runs
python scripts/benchmark_with_metrics.py \
  --model_name distilgpt2 \
  --eval_dataset "processed" \
  --use_real_data
```

## Expanding Functionality

When adding new scripts, please:
1. Follow the project coding standards
2. Add clear documentation and help text
3. Include a brief description in this README
4. Add comprehensive error handling