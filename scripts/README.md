# Scripts

This directory contains utility scripts for working with Sentinel-AI.

## Training and Evaluation

- **`train_colab.py`**: Streamlined training script optimized for Google Colab
- **`eval_colab.py`**: Evaluation script for Google Colab environments

## Analysis Tools

- **`analyze_heads.py`**: Analyze attention head activity and gate values
- **`benchmark.py`**: Performance benchmarking across model configurations
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
python scripts/benchmark.py \
  --models gpt2,distilgpt2 \
  --pruning_levels 0.0,0.3,0.5,0.7 \
  --batch_sizes 1,4,8 \
  --measure_latency \
  --measure_memory
```

### Running Inference

```bash
python scripts/inference.py \
  --model_path checkpoints/model.pth \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.7
```