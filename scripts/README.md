# Scripts

This directory contains utility scripts for working with Sentinel-AI.

## Training and Evaluation

- **`train_colab.py`**: Streamlined training script optimized for Google Colab
- **`eval_colab.py`**: Evaluation script for Google Colab environments

## Analysis Tools

- **`analyze_heads.py`**: Analyze attention head activity and gate values
- **`benchmark.py`**: Performance benchmarking across model configurations
- **`benchmark_pruning.py`**: Comprehensive analysis of different pruning strategies and levels
- **`profile_full_model.py`**: Detailed model profiling to identify bottlenecks
- **`profile_attention_optimization.py`**: Profile attention mechanism optimizations
- **`pruning_impact_analyzer.py`**: Focused analysis of pruning impact on model performance
- **`prune_heads.py`**: Manually prune specific attention heads for testing
- **`expand_heads.py`**: Test head expansion for specific model structures
- **`inference.py`**: Stand-alone inference script for deployment scenarios

## Usage Examples

### Training on Colab

```bash
python scripts/train_colab.py \
  --model gpt2 \
  --dataset wikitext \
  --epochs 3 \
  --batch_size 4 \
  --enable_controller \
  --enable_unet
```

### Analyzing Head Behavior

```bash
python scripts/analyze_heads.py \
  --model_path checkpoints/model.pth \
  --output_dir analysis_results \
  --plot_gates \
  --compute_entropy
```

### Benchmarking

```bash
# General performance benchmarking
python scripts/benchmark.py \
  --models gpt2,distilgpt2 \
  --pruning_levels 0.0,0.3,0.5,0.7 \
  --batch_sizes 1,4,8 \
  --measure_latency \
  --measure_memory

# Comprehensive pruning strategy analysis
python scripts/benchmark_pruning.py \
  --model gpt2 \
  --pruning_levels 0.0 0.1 0.3 0.5 0.7 \
  --strategies entropy gradient random combined \
  --test_perplexity \
  --test_speed \
  --test_memory

# Quick pruning impact visualization
python scripts/pruning_impact_analyzer.py \
  --model gpt2 \
  --pruning_levels 0.0 0.1 0.3 0.5 0.7 0.9 \
  --metric perplexity
```

### Profiling

```bash
# Profile full model execution
python scripts/profile_full_model.py \
  --model_name gpt2 \
  --device cuda \
  --profile_mode all \
  --pruning_levels 0,30,50,70 \
  --optimization_level 2 \
  --visualize

# Compare model architectures
./scripts/run_multi_model_profile.sh \
  --models "gpt2,gpt2-medium,facebook/opt-125m,bigscience/bloom-560m" \
  --device cuda \
  --iterations 3

# Test integration optimizations
python scripts/profile_full_model.py \
  --model_name gpt2 \
  --test_integration_points \
  --pruning_levels 0,50 \
  --optimization_level 3 \
  --visualize
```

### Running Inference

```bash
python scripts/inference.py \
  --model_path checkpoints/model.pth \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.7
```