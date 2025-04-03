# Colab Utilities for Sentinel-AI

This module provides helper functions and utilities to enhance the usage of Sentinel-AI in Google Colab notebooks. The utilities help with environment setup, hardware detection, and memory optimization.

## Features

### Environment Setup

- Auto-detection of Google Colab environment
- Automatic selection of GPU acceleration (when available)
- Hardware detection and status reporting

### Memory Management

- GPU memory monitoring and reporting
- Memory optimization recommendations based on model size
- Adaptive parameters for different hardware configurations

### Optimization Utilities

- Batch size recommendations based on available memory
- Sequence length adjustments for memory efficiency
- Gradient accumulation configuration
- Mixed precision enablement when appropriate

## Usage Example

```python
from utils.colab import setup_colab_environment, check_gpu_status, optimize_for_colab

# Set up the Colab environment with GPU preference
env_info = setup_colab_environment(prefer_gpu=True)

# Check the current GPU status
gpu_info = check_gpu_status()

# Get optimized parameters for a specific model size
params = optimize_for_colab(
    model_size="large",      # Options: tiny, small, medium, large, xl
    prefer_stability=True    # Prefer stability over speed
)

# Use the optimized parameters
batch_size = params["batch_size"]
sequence_length = params["sequence_length"]
stability_level = params["stability_level"]
```

## Notebook Integration

These utilities are designed to be integrated into all Sentinel-AI notebooks. To add them to a notebook, include the following at the beginning:

```python
# Set up Colab environment
try:
    from utils.colab import setup_colab_environment, optimize_for_colab
    env_info = setup_colab_environment(prefer_gpu=True)
    
    # Get optimized parameters for experiment
    params = optimize_for_colab(model_size="medium")  # Adjust size as needed
    
    # Extract parameters for use
    batch_size = params["batch_size"]
    sequence_length = params["sequence_length"]
    stability_level = params["stability_level"]
except ImportError:
    print("Colab utilities not available, using default parameters")
    batch_size = 4
    sequence_length = 128
    stability_level = 1
```

## Notes on Parameter Optimization

The `optimize_for_colab` function provides recommendations based on:

1. **Model Size**: Different parameters for varying model sizes:
   - tiny: DistilGPT2, smallest models
   - small: GPT2, OPT-125M
   - medium: GPT2-Medium, OPT-350M
   - large: GPT2-Large, OPT-1.3B
   - xl: GPT2-XL, larger models

2. **Available Memory**: Parameters are scaled based on available GPU memory.

3. **Stability Preference**: 
   - When `prefer_stability=True`, more conservative parameters are used
   - When `prefer_stability=False`, more aggressive parameters maximize utilization

4. **Mixed Precision**: FP16 is enabled when appropriate based on model size and memory availability.