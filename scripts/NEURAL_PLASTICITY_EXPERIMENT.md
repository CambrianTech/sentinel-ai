# Neural Plasticity Experiment

This document describes how to run the neural plasticity experiment that dynamically prunes attention heads in transformer models to optimize their structure while maintaining performance.

## Dynamic Neural Plasticity Experiment

The core script `run_dynamic_neural_plasticity.py` runs a complete neural plasticity experiment that demonstrates the entire process from warmup through pruning to fine-tuning.

### Key Features

- Uses **real HuggingFace models**, not simulated models
- Processes **real datasets**, not simulated data
- Makes **dynamic decisions** on when to prune based on mathematical stabilization detection, not fixed schedules
- Performs actual pruning by modifying model weights
- Evaluates models with real text generation at each phase
- Creates comprehensive visualizations and HTML dashboards
- Works identically in both local and Google Colab environments

### Mathematical Stabilization Detection

The system uses **polynomial curve fitting** and other mathematical techniques to determine when the model's loss has stabilized during warmup, rather than using fixed step counts. This mimics how biological systems adapt based on performance feedback rather than predetermined schedules.

This is done using several techniques:
1. **Polynomial fitting** of recent loss values to detect curve flattening
2. **Trend analysis** across multiple time windows
3. **Relative improvement analysis** comparing recent vs. historical progress

### Running the Experiment

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default parameters (distilgpt2 on wikitext)
python scripts/run_dynamic_neural_plasticity.py

# Run with custom parameters
python scripts/run_dynamic_neural_plasticity.py \
  --model_name gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --batch_size 4 \
  --pruning_strategy combined \
  --pruning_level 0.2 \
  --output_dir experiment_results/my_experiment
```

### Parameters

- `--model_name`: Model name or path (default: distilgpt2)
- `--dataset`: Dataset name (default: wikitext)
- `--dataset_config`: Dataset configuration (default: wikitext-2-raw-v1)
- `--max_length`: Max sequence length (default: 128)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--warmup_patience`: Patience for stabilization detection (default: 20)
- `--pruning_strategy`: Strategy (gradient, entropy, random, combined)
- `--pruning_level`: Pruning level (default: 0.2)
- `--max_warmup_steps`: Maximum warmup steps (default: 300)
- `--training_steps`: Training steps after pruning (default: 200)
- `--output_dir`: Directory to save results
- `--device`: Device to use (cuda or cpu, default: auto-detect)

### Running in Google Colab

```python
!pip install transformers datasets torch matplotlib numpy tqdm
!python run_dynamic_neural_plasticity.py --device cuda --output_dir /content/experiment_results
```

## Experiment Outputs

The script generates a comprehensive output directory structure:

```
experiment_results/
├── warmup/               # Warmup phase data and visualizations
├── pruning/              # Pruning decisions and visualizations
├── fine_tuning/          # Fine-tuning metrics
├── visualizations/       # Comprehensive visualizations 
├── inference/            # Text samples generated at each phase
├── metrics/              # JSON files with experiment metrics
├── html/                 # HTML dashboards showing the process
├── model/                # Saved model and tokenizer
├── logs/                 # Experiment logs
└── parameters.json       # Experiment parameters
```

### HTML Dashboard

The experiment generates an HTML dashboard that shows:

1. Complete training process with all phases
2. Pruning decisions with detailed visualizations 
3. Text generation samples from each phase
4. Before/after comparisons of model performance
5. Entropy and gradient heatmaps

### Troubleshooting

If the script appears to time out or runs out of memory:

1. Reduce batch_size to 2
2. Use a smaller model like distilgpt2 instead of gpt2
3. Reduce max_length parameter to 64
4. Set max_warmup_steps and training_steps to lower values
5. On Apple Silicon, the script will automatically use CPU to avoid stability issues

Version: v0.0.70 (2025-04-20)