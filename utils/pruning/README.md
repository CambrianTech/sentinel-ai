# JAX/Flax Pruning Library

This library provides a stable pruning and fine-tuning implementation using JAX/Flax that works reliably across all platforms, including M1/M2 Macs which experienced BLAS crashes with PyTorch.

## Key Components

1. **Environment:** Auto-detects and configures the runtime environment (Mac/Colab/TPU/GPU/CPU)
2. **PruningModule:** Core implementation for loading models and applying pruning
3. **Strategies:** Different approaches to selecting heads for pruning (random, magnitude, entropy)
4. **FineTuner:** Fine-tunes pruned models to recover or improve performance
5. **ResultsManager:** Handles storage and visualization of results
6. **PruningBenchmark:** High-level benchmark controller

## Usage

### Jupyter Notebook

The easiest way to get started is to use the `PruningAndFineTuning.ipynb` notebook in the `notebooks/` directory. This notebook will:

1. Detect your environment capabilities (Mac/TPU/GPU/CPU)
2. Select suitable models for your hardware
3. Run pruning experiments with different strategies and levels
4. Fine-tune pruned models to recover performance
5. Visualize the results in real-time

### Command Line

For more control, you can use the command-line script:

```bash
python scripts/pruning_and_finetuning.py --strategies random magnitude entropy --pruning_levels 0.1 0.3 0.5 --max_runtime 6h
```

Command line options:
- `--models`: List of specific models to test (if not provided, will auto-select suitable models)
- `--strategies`: Pruning strategies to test (random, magnitude, entropy)
- `--pruning_levels`: Pruning levels to test (0.1 = 10% pruning, 0.5 = 50% pruning, etc.)
- `--fine_tuning_epochs`: Number of fine-tuning epochs per model
- `--dataset`: Dataset to use for fine-tuning (default: wikitext)
- `--prompt`: Prompt to use for evaluation
- `--results_dir`: Directory to save results
- `--max_runtime`: Maximum runtime in format like '6h', '30m', '3600s', etc.

### Programmatic API

You can also use the library programmatically:

```python
from utils.pruning import PruningModule, get_strategy, FineTuner

# Initialize pruning module with model
pruning_module = PruningModule("distilgpt2")
pruning_module.load_model()

# Get original params
original_params = pruning_module.original_params

# Select pruning strategy
strategy = get_strategy("magnitude", pruning_module)

# Calculate head importance
importance = strategy.get_head_importance(original_params)
importance.sort(key=lambda x: x[2])  # Sort by importance (ascending)

# Determine heads to prune
total_heads = pruning_module.num_layers * pruning_module.num_heads
pruning_level = 0.3  # 30% pruning
heads_to_prune = int(total_heads * pruning_level)
head_indices = [(l, h) for l, h, _ in importance[:heads_to_prune]]

# Prune heads
pruned_params = strategy.prune_heads(original_params, head_indices)

# Fine-tune pruned model
fine_tuner = FineTuner(pruning_module, dataset_name="wikitext")
tuned_params, metrics = fine_tuner.fine_tune(pruned_params, num_epochs=2)

# Evaluate all model versions
prompt = "Artificial intelligence will transform"
print(f"Original: {pruning_module.evaluate_perplexity(original_params, prompt):.2f}")
print(f"Pruned: {pruning_module.evaluate_perplexity(pruned_params, prompt):.2f}")
print(f"Fine-tuned: {pruning_module.evaluate_perplexity(tuned_params, prompt):.2f}")
```

## Model Support

The library supports a variety of models from the Hugging Face Transformers library:

- GPT-2 family (distilgpt2, gpt2, gpt2-medium, etc.)
- OPT family (facebook/opt-125m, facebook/opt-350m, etc.)
- Pythia family (EleutherAI/pythia-160m, EleutherAI/pythia-410m, etc.)

Models are automatically selected based on your hardware capabilities to ensure reliable performance.

### Model Compatibility Features

We've added comprehensive compatibility features to handle architecture-specific differences:

1. **GPT-2 Models**
   - Native JAX/Flax support with efficient loading
   - Optimized parameter access patterns

2. **OPT Models**
   - Special handling for shape mismatches during text generation
   - Custom generation parameters for maximum stability
   - Shape-aware perplexity calculation with tensor broadcasting fixes

3. **Pythia Models** 
   - PyTorch to JAX/Flax conversion bridge
   - GPT-2 compatible configuration mapping
   - Architecture compatibility layer for attention mechanisms

4. **Testing Framework**
   - Comprehensive model compatibility testing suite
   - Validation across different model architectures
   - Cross-architecture comparison of pruning effects
   - Unit tests for metrics handling (tuple vs dictionary format)
   - Automatic regression detection for common issues

## Running Tests

To run the test suite for the pruning API:

```bash
# Run the comprehensive test suite
cd utils/pruning/api/tests
python run_tests.py

# Run the simplified metrics handling test
python utils/pruning/tests_metrics.py
```

The simplified metrics test is specifically designed to verify the fix for the tuple vs dictionary metrics issue we encountered. It ensures proper handling of different metric formats to prevent regressions.

## Platform Support

This library works reliably on:

- M1/M2 Macs (uses optimized JAX configuration to avoid BLAS crashes)
- Google Colab (auto-detects and utilizes TPU/GPU when available)
- Standard Linux/Windows environments

## Results Analysis

Results are automatically saved and can be analyzed using the visualization tools provided by the `ResultsManager` class. Key metrics tracked:

- Perplexity change after pruning
- Perplexity change after fine-tuning
- Recovery percentage (how much of the pruning-induced performance loss was recovered)
- Token generation examples before/after pruning/fine-tuning

## FineTuner Consolidation Plan

Currently, the codebase has two fine-tuner implementations:
- `FineTuner`: Basic implementation for fine-tuning pruned models
- `ImprovedFineTuner`: Enhanced implementation with stability features for large models

### Planned Consolidation

We will consolidate these implementations into a single robust `FineTuner` class that:

1. Incorporates all stability enhancements from `ImprovedFineTuner`
2. Provides backward compatibility
3. Offers configurable stability levels

### New FineTuner Class Structure

```python
class FineTuner:
    def __init__(
        self,
        pruning_module,
        dataset_name="openwebtext",
        dataset_config=None,
        batch_size=4,
        stability_level=1,  # 0: basic, 1: standard, 2: high
        use_synthetic_data=False,
        model_specific_optimizations=True
    ):
        # Initialize with configurable stability features
        pass
```

### Key Features

1. **Stability Levels**
   - Level 0: Basic operation with minimal safety features (legacy mode)
   - Level 1: Standard stability enhancements (default, recommended)
   - Level 2: High stability for challenging models (OPT, large models)

2. **Model-Specific Optimizations**
   - Automatic detection and application of model-specific settings
   - Special handling for OPT, large models, etc.

3. **Enhanced Dataset Handling**
   - Improved loading with better error recovery
   - Automatic fallback to synthetic data
   - Support for various dataset structures

4. **NaN Prevention**
   - Comprehensive NaN detection in inputs, gradients, and losses
   - Safe computation patterns to avoid division by zero
   - Adaptive learning rate and batch size adjustment

5. **Memory Optimization**
   - Automatic batch size adjustment based on model size
   - Sequence length optimization
   - Garbage collection during training