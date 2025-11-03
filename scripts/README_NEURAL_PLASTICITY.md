# Neural Plasticity Experiment Runner

This directory contains scripts for running neural plasticity experiments, which dynamically prune and grow attention heads in transformer models to optimize their structure.

## Key Script: run_real_neural_plasticity.py

This script provides an end-to-end implementation of neural plasticity, including:

1. Warmup training with stabilization detection
2. Attention entropy and gradient analysis
3. Dynamic pruning based on head importance
4. Fine-tuning after pruning
5. Comprehensive visualization and HTML dashboard generation

### Running the script

First, activate your virtual environment:

```bash
source .venv/bin/activate
```

#### Basic usage:

```bash
python scripts/run_real_neural_plasticity.py
```

This will run with default settings (distilgpt2 model on wikitext dataset).

#### Advanced usage:

```bash
python scripts/run_real_neural_plasticity.py \
  --model_name gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --batch_size 4 \
  --warmup_patience 20 \
  --pruning_level 0.2 \
  --pruning_strategy combined \
  --output_dir experiment_results/my_experiment
```

### Parameters

- `--model_name`: The model to use (default: distilgpt2)
- `--dataset`: The dataset name (default: wikitext)
- `--dataset_config`: Dataset configuration name (default: wikitext-2-raw-v1)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size for training (default: 4)
- `--num_epochs`: Number of epochs for warmup (default: 1)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--warmup_patience`: Steps to wait for stabilization (default: 20)
- `--pruning_level`: Percentage of heads to prune (default: 0.2)
- `--pruning_strategy`: Strategy (gradient, entropy, random, combined) (default: combined)
- `--output_dir`: Directory to save results (default: experiment_results/run_<timestamp>)
- `--training_steps`: Steps per phase (default: 100)
- `--eval_steps`: Evaluation steps (default: 10)
- `--sample_texts`: Number of sample texts to generate (default: 3)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (cuda or cpu, default: auto-detect)

### Running in Google Colab

Upload the script to Colab and run:

```python
!pip install transformers datasets torch matplotlib numpy tqdm
!python run_real_neural_plasticity.py --device cuda --output_dir /content/experiment_results
```

## Output

The script generates a comprehensive output directory with:

- `warmup/`: Warmup phase data and visualizations
- `pruning/`: Pruning decisions and visualizations
- `fine_tuning/`: Fine-tuning metrics and visualizations 
- `inference/`: Text samples generated at each phase
- `metrics/`: JSON files with all experiment metrics
- `html/`: HTML dashboards and reports
- `model/`: Saved model and tokenizer
- `logs/`: Experiment logs
- `visualizations/`: Comprehensive visualizations

## HTML Dashboard

The script generates an interactive HTML dashboard that shows:

1. Complete training process with all phases
2. Pruning decisions with detailed visualizations
3. Text generation examples from each phase
4. Before/after comparisons of model performance
5. Interactive visualizations of attention head behavior

## Local vs. Colab Compatibility

The script works identically in both local and Colab environments, with automatic environment detection and adaptation.

## Notes

- When running on Apple Silicon, the script enforces CPU-only operation to avoid numerical stability issues
- For larger models, adjust batch size and max_length according to available memory
- Set training_steps and eval_steps to lower values for quicker runs
- If you encounter memory issues, consider using a smaller model (like distilgpt2) or reducing batch size

## Troubleshooting

### Script Execution Timeout

If the script appears to time out during execution:

1. This is normal! The script continues running in the background and will generate results
2. Check the output directory for logs and results
3. For a minimal test run, use these extremely conservative parameters:

```bash
python scripts/run_real_neural_plasticity.py \
  --model_name distilgpt2 \
  --batch_size 2 \
  --training_steps 5 \
  --warmup_patience 3 \
  --sample_texts 1 \
  --eval_steps 2 \
  --output_dir experiment_results/minimal_test
```

### Memory Issues

If you encounter memory issues:

1. Reduce batch_size to 1 or 2
2. Use a smaller model like distilgpt2 instead of gpt2
3. Reduce max_length parameter to 64 or 32
4. Set environment variable `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` on Mac to disable MPS memory caching

### Apple Silicon Issues

On Apple Silicon Macs, the script automatically:

1. Forces CPU-only execution to avoid numerical stability issues
2. Disables BLAS threading to prevent known crashes
3. Applies PyTorch safeguards for improved stability
4. Uses matplotlib's Agg backend to avoid UI issues

These safeguards ensure the script runs reliably on Apple Silicon, though at reduced performance.

Version: v0.0.68 (2025-04-20)