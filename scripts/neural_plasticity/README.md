# Neural Plasticity Experiment Scripts

This directory contains the main scripts for running neural plasticity experiments with real models and data.

## Available Scripts

1. **simple_experiment.py** - A simplified neural plasticity experiment runner
   - Runs a complete experiment with minimal parameters
   - Useful for quick testing and demonstrations
   - Supports all core neural plasticity features
   - Example usage: `python scripts/neural_plasticity/simple_experiment.py`

2. **full_experiment.py** - Comprehensive neural plasticity experiment runner
   - Provides extensive command-line options and customization
   - Includes detailed logging and error handling
   - Generates comprehensive visualizations and HTML dashboards
   - Example usage: `python scripts/neural_plasticity/full_experiment.py --model_name gpt2 --dataset gutenberg`

## Running Experiments

Both scripts use the modular neural plasticity implementation from `utils/neural_plasticity/` and provide a complete experimental workflow:

1. Warmup training with dynamic stabilization detection
2. Attention head analysis using entropy and gradient metrics
3. Pruning of low-importance heads based on mathematical criteria
4. Fine-tuning with differential learning rates
5. Comprehensive visualization and evaluation

### Simple Example

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run a quick experiment with small parameters
python scripts/neural_plasticity/simple_experiment.py
```

### Full Example with Customization

```bash
# Full experiment with custom parameters
python scripts/neural_plasticity/full_experiment.py \
  --model_name distilgpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --batch_size 4 \
  --pruning_level 0.15 \
  --pruning_strategy combined \
  --max_warmup_steps 100 \
  --training_steps 100 \
  --output_dir experiment_results/custom_run
```

## Key Features

- **Real Models & Data**: Uses actual HuggingFace models and datasets, not simulated data
- **Mathematical Decision Making**: Uses entropy and gradient metrics for pruning decisions
- **Dynamic Stabilization**: Detects training stabilization using polynomial curve fitting
- **HTML Dashboards**: Generates interactive dashboards showing the entire process
- **Text Evaluation**: Generates text samples at each phase to evaluate model quality
- **Environment Awareness**: Works seamlessly in both local and Google Colab environments

## Configuration Options

Both scripts support various command-line arguments, with sensible defaults. Key options include:

- `--model_name`: HuggingFace model name/path (default: "distilgpt2")
- `--dataset`: Dataset name (default: "wikitext")
- `--dataset_config`: Dataset configuration (default: "wikitext-2-raw-v1") 
- `--pruning_level`: Percentage of heads to prune (default: 0.15)
- `--pruning_strategy`: Strategy to use (options: "gradient", "entropy", "random", "combined")
- `--device`: Device to use for computation (default: auto-detect)

## Output Structure

Both scripts create a structured output directory containing:

- `/warmup` - Warmup phase results and visualizations
- `/pruning` - Pruning phase data and head selection information
- `/fine_tuning` - Fine-tuning metrics and checkpoints
- `/visualizations` - Comprehensive visualizations and plots
- `/inference` - Generated text samples at each phase
- `/html` - Interactive HTML dashboards
- `/metrics` - JSON files with all experiment metrics
- `/model` - Saved model and tokenizer
- `/logs` - Detailed log files