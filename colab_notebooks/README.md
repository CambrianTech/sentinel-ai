# Colab Notebooks

This directory contains Jupyter notebooks designed to run in Google Colab.

## NeuralPlasticityDemo.ipynb (v0.0.52 (2025-04-19 18:29:23))

Demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

### Features
- Dynamic pruning of attention heads based on entropy and gradient metrics
- Visualization of pruning decisions and model performance
- Adaptive recovery of pruned heads when needed
- Complete training loop with metrics tracking

### Latest Improvements in v0.0.52 (2025-04-19)
- Fixed GPU tensor visualization errors
- Ensured proper tensor detachment and CPU conversion for visualization
- Integrated with utils.colab.visualizations module
- Improved notebook stability for both CPU and GPU environments
- Added %matplotlib inline for Colab compatibility
- Added system dependency checks
- Improved error handling in training loop
- Deduplicated import statements
- Fixed cell execution counts for better notebook flow

### Usage
1. Upload to Colab with File > Upload notebook
2. Select GPU runtime (recommended) 
3. Run cells sequentially
4. Observe how the model prunes and regrows attention heads during training
5. Review the generated text samples to see how pruning affects model output

## PruningAndFineTuningColab.ipynb (v0.0.55)

Notebook for demonstrating and benchmarking transformer model pruning and fine-tuning recovery.

### Features
- Minimal and reliable implementation that works in all environments
- Clear key parameters at the top with meaningful values (100 epochs, proper learning rate)
- Simple customizable text generation prompt
- Uses modular API components from the repository
- Focuses on GPT-2 models (distilgpt2, gpt2, gpt2-medium)
- Uses entropy-based pruning to identify less important attention heads
- Fine-tunes pruned models to recover performance
- Real-time visualization of experiments
- Uses real-world Wikitext data for training and evaluation
- Comprehensive metrics and visualizations with text examples
- Text generation examples at each stage (baseline, pruned, fine-tuned)
- Interactive text generation with the fine-tuned model

### Usage
1. Upload to Colab with File > Upload notebook
2. Select GPU runtime (recommended)
3. Review and modify key parameters at the top if needed
4. Edit the generation prompt variable if desired
5. Run cells sequentially 
6. Wait for the model to be pruned and fine-tuned
7. Review the perplexity metrics and bar chart
8. Use the interactive generation cell to test your model

### Version History
- v0.0.55 (April 2025): Fix pruning implementation to properly zero out attention head weights
- v0.0.52 (NeuralPlasticityDemo - 2025-04-19): Add timestamps to versioning and fix all Colab compatibility issues
- v0.0.51 (NeuralPlasticityDemo - April 2025): Add Colab compatibility improvements and fix execution flow
- v0.0.50 (NeuralPlasticityDemo - April 2025): Fix GPU tensor detach/visualization issues and use utils.colab visualizations
- v0.0.54 (April 2025): Add warmup fine-tuning phase for more realistic baseline metrics
- v0.0.53 (April 2025): Improve robustness for partial and interrupted runs
- v0.0.52 (April 2025): Add text generation examples at each stage and per-epoch metrics
- v0.0.51 (April 2025): Visualization and perplexity values
- v0.0.50 (April 2025): Add key parameters at top and use meaningful values
- v0.0.49 (April 2025): Remove start button and simplify notebook
- v0.0.48 (April 2025): Add interactive text prompt widget and fix metrics handling
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