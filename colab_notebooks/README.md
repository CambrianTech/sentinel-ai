# Colab Notebooks

This directory contains Jupyter notebooks designed to run in Google Colab.

## ModelImprovementColab.ipynb (v1.0.0)

Interactive notebook with UI components for configuring and running model improvement experiments.

### Features
- Interactive UI with dropdowns, sliders, and checkboxes for all parameters
- Auto-optimization of parameters based on model size and hardware
- Support for pruning, fine-tuning, and adaptive plasticity
- Real-time visualization of experiment results
- Save/load experiment configurations

### Usage
1. Upload to Colab with File > Upload notebook
2. Select GPU or TPU runtime
3. Run all cells to initialize the UI
4. Configure parameters using the interactive interface
5. Click "Auto-Optimize Parameters" then "Run Experiment"

### Version History
- v1.0.0 (April 2025): Initial release with parameter UI and adaptive plasticity support

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