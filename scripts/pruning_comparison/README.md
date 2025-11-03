# Pruning Comparison with Agency

This directory contains scripts for evaluating the effectiveness of agency-enabled models compared to baseline models when subjected to progressive levels of pruning.

## Overview

The experiments measure how agency mechanisms help models maintain performance under aggressive pruning conditions. The scripts test:

- Different pruning levels (0% to 70%)
- Multiple pruning strategies (entropy, random, magnitude-based)
- Various temperatures and model configurations
- Statistical significance with multiple iterations

## Available Scripts

- `pruning_agency_comparison.py` - Main script for running pruning comparison experiments
- `run_pruning_comparison_colab.py` - Wrapper for running on Google Colab with T4 GPU

## Usage

### Local Execution

```bash
# Basic usage
python pruning_agency_comparison.py

# Specific configuration with additional options
python pruning_agency_comparison.py \
  --model_name=gpt2 \
  --device=cuda \
  --pruning_levels=0,30,50,70 \
  --pruning_method=entropy \
  --temperatures=0.7,1.0 \
  --num_tokens=100 \
  --iterations=3 \
  --max_prompts=10 \
  --save_outputs \
  --memory_logging
```

### Google Colab

To run on Google Colab with T4 GPU:

1. Upload `run_pruning_comparison_colab.py` to Google Colab
2. Change runtime to use GPU (Runtime > Change runtime type > GPU)
3. Run the notebook
4. Optionally save results to Google Drive

## Key Parameters

- `--model_name`: Base model to use (gpt2, distilgpt2)
- `--device`: Device to run on (cpu or cuda)
- `--pruning_levels`: Comma-separated pruning percentages to evaluate
- `--pruning_method`: Method to select heads for pruning (entropy, random, magnitude)
- `--temperatures`: Comma-separated temperatures to test
- `--num_tokens`: Number of tokens to generate for each prompt
- `--iterations`: Number of iterations for statistical significance
- `--max_prompts`: Maximum number of prompts to evaluate
- `--save_outputs`: Save generated text outputs
- `--memory_logging`: Log memory usage during evaluation

## Output

The script generates comprehensive visualizations and statistics:

- Generation speed comparison with error bars
- Perplexity comparison across pruning levels
- Lexical diversity measurements
- First token latency analysis
- Heatmaps of improvement at different temperatures
- Radar charts comparing overall performance
- Comprehensive statistical summaries

Results are saved to `validation_results/pruning_agency/run_TIMESTAMP/` with:
- JSON results file with all metrics
- Visualization PNGs
- Text outputs if requested
- Statistical summaries with standard deviations

## Example Visualization

![Comprehensive Summary](../../docs/assets/figures/pruning_summary_example.png)

## Interpreting Results

The experiment allows us to answer several key questions:

1. **Does agency improve resilience to pruning?** - By comparing the slope of performance decline
2. **What is the optimal pruning level?** - Where performance gains are maximized
3. **How does temperature affect pruned models?** - Via cross-temperature comparisons
4. **Is agency more beneficial under resource constraints?** - By comparing relative gains at high pruning levels
5. **What types of quality metrics benefit most?** - Through multi-metric radar visualizations