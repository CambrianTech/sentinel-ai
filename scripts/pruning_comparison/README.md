# Pruning Comparison Framework

This directory contains scripts for empirically measuring and visualizing the benefits of agency-enabled transformer models compared to baseline models when subjected to progressive pruning.

**Note:** These scripts are designed to work with the main branch of the repository. They automatically pull the latest code from the main branch when run.

## Key Features

- **Comprehensive Testing**: Compare baseline and agency-enabled models at multiple pruning levels
- **Multiple Metrics**: Measure speed (tokens/sec), quality (perplexity), diversity, first token latency
- **Statistical Significance**: Run multiple iterations and compute error bars
- **Visualization Suite**: Generate comprehensive charts showing the benefits of agency
- **Optimized for Colab**: Ready to run on Google Colab with T4 GPU

## Background

The pruning comparison framework empirically demonstrates the key benefit of agency-enabled attention: resilience to pruning. Agency-enabled models can maintain performance even with 50%+ of attention heads pruned, while baseline models degrade significantly.

## Available Scripts

- `pruning_agency_comparison.py`: Main script implementing the comparison framework
- `run_pruning_cpu_test.py`: Run a quick test on CPU to verify functionality
- `run_pruning_comparison_colab.py`: Run the full comparison on Google Colab with GPU

## Quick Start

### Step 1: Run a CPU Test

First, verify that the framework works properly on your local machine:

```bash
python scripts/pruning_comparison/run_pruning_cpu_test.py
```

For an even faster test (useful while debugging):

```bash
python scripts/pruning_comparison/run_pruning_cpu_test.py --debug
```

### Step 2: Run on Google Colab

For the full comparison with GPU acceleration:

1. Upload `run_pruning_comparison_colab.py` to Google Colab
2. Change runtime to GPU (Runtime > Change runtime type > GPU)
3. Run the notebook cells
4. Results will be automatically saved to Google Drive if mounted

## Interpreting Results

The framework generates several visualizations to help understand the results:

1. **Speed Comparison**: Shows tokens/sec for both models across pruning levels
2. **Quality Metrics**: Visualizes perplexity, diversity, and repetition scores
3. **Relative Improvement**: Demonstrates the percentage improvement from agency
4. **Comprehensive Summary**: Multi-panel visualization showing all metrics together
5. **Radar Chart**: For comparing multi-dimensional performance at max pruning level

## Usage Options

The main script supports many options for customizing the experiment:

```bash
python pruning_agency_comparison.py \
  --model_name=gpt2 \
  --device=cuda \
  --precision=float16 \
  --pruning_levels=0,20,40,60,80 \
  --pruning_method=entropy \
  --num_tokens=100 \
  --temperatures=0.7,1.0 \
  --max_prompts=5 \
  --iterations=3 \
  --batch_size=1 \
  --output_dir=validation_results/pruning_agency \
  --save_outputs \
  --memory_logging
```

## Key Parameters

- `--model_name`: Base model to use (gpt2, distilgpt2)
- `--device`: Device to run on (cpu or cuda)
- `--precision`: Precision for model weights (float32, float16, bfloat16)
- `--pruning_levels`: Comma-separated pruning percentages to evaluate
- `--pruning_method`: Method to select heads for pruning (entropy, random, magnitude)
- `--temperatures`: Comma-separated temperatures to test
- `--num_tokens`: Number of tokens to generate for each prompt
- `--iterations`: Number of iterations for statistical significance
- `--max_prompts`: Maximum number of prompts to evaluate
- `--batch_size`: Batch size for generation
- `--save_outputs`: Save generated text outputs
- `--memory_logging`: Log memory usage during evaluation

## Expected Results

When running the comparison with default parameters, you should expect to see:

1. **Speed Improvement**: Agency-enabled models typically maintain or improve speed as pruning increases, while baseline models slow down
2. **Quality Preservation**: Agency models maintain lower perplexity at high pruning levels
3. **Diversity Advantage**: Agency models produce more diverse text even under aggressive pruning
4. **Responsiveness**: First token latency remains lower for agency-enabled models

These results empirically validate that agency mechanisms provide tangible benefits for model efficiency and quality when resources are constrained.

## Interpreting Results

The experiment allows us to answer several key questions:

1. **Does agency improve resilience to pruning?** - By comparing the slope of performance decline
2. **What is the optimal pruning level?** - Where performance gains are maximized
3. **How does temperature affect pruned models?** - Via cross-temperature comparisons
4. **Is agency more beneficial under resource constraints?** - By comparing relative gains at high pruning levels
5. **What types of quality metrics benefit most?** - Through multi-metric radar visualizations