# Colab Notebooks

This directory contains Jupyter notebooks designed to run in Google Colab.

## PruningAndFineTuningColab.ipynb (v1.0.0)

Notebook for demonstrating and benchmarking transformer model pruning and fine-tuning recovery.

### Features
- Supports multiple model architectures (GPT-2, OPT)
- Implements three pruning strategies (random, magnitude, entropy)
- Fine-tunes pruned models to recover performance
- Real-time visualization of experiments
- Memory-efficient implementation for Colab environments

### Usage
1. Upload to Colab with File > Upload notebook
2. Select GPU or TPU runtime
3. Run cells sequentially 
4. Run the visualize_ongoing_experiments() cell anytime to see progress

### Version History
- v1.0.0 (April 2025): Initial release with OPT model support and chart improvements