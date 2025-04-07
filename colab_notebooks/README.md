# Colab Notebooks

This directory contains Jupyter notebooks designed to run in Google Colab.

## PruningAndFineTuningColab.ipynb (v0.0.48)

Notebook for demonstrating and benchmarking transformer model pruning and fine-tuning recovery.

### Features
- Minimal and reliable implementation that works in all environments
- Uses modular API components from the repository
- Focuses on GPT-2 models (distilgpt2, gpt2, gpt2-medium)
- Uses entropy-based pruning to identify less important attention heads
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
- v0.0.48 (April 2025): Add customizable text prompt and fix metrics handling
- v0.0.47 (April 2025): Fix data preparation and improve error handling
- v0.0.46 (April 2025): Simplified notebook to use modular repository API
- v0.0.45 (April 2025): Made notebook self-contained without requiring complex imports
- v0.0.44 (April 2025): Fixed Colab repository URL and branch selection for reliable execution
- v0.0.43 (April 2025): Fixed entropy pruning implementation to handle API availability gracefully
- v0.0.42 (April 2025): Added super_simple test mode and improved error handling
- v0.0.41 (April 2025): Modularized code using sentinel.pruning package
- v0.0.40 (April 2025): Improve robustness for different model architectures
- v0.0.39 (April 2025): Fix TypeError in run_experiment function call
- v0.0.38 (April 2025): Fix ValueError in generate_text function
- v0.0.37 (April 2025): Complete rewrite with minimal dependencies for reliability