# Colab Notebooks

This directory contains Jupyter notebooks designed to run in Google Colab.

## PruningAndFineTuningColab.ipynb (v0.0.37)

Notebook for demonstrating and benchmarking transformer model pruning and fine-tuning recovery.

### Features
- Minimal and reliable implementation that works in all environments
- Focuses on GPT-2 models (distilgpt2, gpt2, gpt2-medium)
- Simple head masking approach to pruning
- Fine-tunes pruned models to recover performance
- Real-time visualization of experiments
- Uses real-world Wikitext data for training and evaluation
- Comprehensive metrics and visualizations
- Interactive text generation with the fine-tuned model

### Usage
1. Upload to Colab with File > Upload notebook
2. Select GPU runtime (recommended)
3. Run cells sequentially 
4. Use the interactive generation cell to test your model

### Version History
- v0.0.37 (April 2025): Complete rewrite with minimal dependencies for reliability
- v0.0.36 (April 2025): Simplified pruning implementation for better reliability
- v0.0.35 (April 2025): Fixed in-place operation error in apply_head_pruning function
- v0.0.34 (April 2025): Fixed undefined variable error, visualization issues and enhanced CUDA error handling
- v0.0.33 (April 2025): Fixed visualization issues, improved model compatibility and enhanced error handling
- v0.0.32 (April 2025): Added CUDA error handling for Colab compatibility and memory management
- v0.0.31 (April 2025): Fixed get_strategy parameters issue and improved Colab compatibility
- v0.0.30 (April 2025): Added OPT model support and chart improvements