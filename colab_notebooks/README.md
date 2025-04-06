# Colab Notebooks

This directory contains Jupyter notebooks designed to run in Google Colab.

## PruningAndFineTuningColab.ipynb (v0.0.34)

Notebook for demonstrating and benchmarking transformer model pruning and fine-tuning recovery.

### Features
- Supports multiple model architectures (GPT-2, OPT, Pythia)
- Implements three pruning strategies (random, magnitude, entropy)
- Fine-tunes pruned models to recover performance
- Real-time visualization of experiments with improved readability
- Memory-efficient implementation for Colab environments
- Uses real-world Wikitext data (not tiny Shakespeare)
- Robust CUDA error handling for reliable Colab execution
- Advanced model compatibility detection and attribute handling

### Usage
1. Upload to Colab with File > Upload notebook
2. Select GPU or TPU runtime
3. Run cells sequentially 
4. Run the visualize_ongoing_experiments() cell anytime to see progress

### Version History
- v0.0.34 (April 2025): Fixed undefined variable error, visualization issues and enhanced CUDA error handling
- v0.0.33 (April 2025): Fixed visualization issues, improved model compatibility and enhanced error handling
- v0.0.32 (April 2025): Added CUDA error handling for Colab compatibility and memory management
- v0.0.31 (April 2025): Fixed get_strategy parameters issue and improved Colab compatibility
- v0.0.30 (April 2025): Added OPT model support and chart improvements