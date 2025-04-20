# Neural Plasticity Module

## Overview

The Neural Plasticity module provides functionality for transformer models to dynamically adapt their structure through:
- Entropy-based head importance measurement
- Gradient-based pruning of attention heads
- Targeted head revitalization during training
- Visualization of model brain dynamics

## Recent Fixes (v0.0.55 - 2025-04-19 21:15:00)

We've implemented comprehensive fixes for BLAS/libtorch crashes on Apple Silicon (M1/M2/M3):

1. **Enhanced Apple Silicon Support**: Improved detection and safeguards for Apple Silicon architecture
2. **Environment Variables**: Automatically set `OMP_NUM_THREADS=1` and other threading vars on Apple Silicon
3. **Thread Management**: Added `torch.set_num_threads(1)` to prevent parallel BLAS operations
4. **Memory Management**: Added stricter tensor memory layout handling with `.contiguous()` calls
5. **Tensor Device Control**: Forced CPU tensors for all BLAS operations on Apple Silicon
6. **Matplotlib Backend**: Switched to 'Agg' backend on Apple Silicon for visualization stability
7. **Gradient Handling**: Added extra `.detach()` calls to prevent autograd-related crashes

## Running the Notebook

### Option 1: Local Execution

For running locally, use our optimized environment settings:

```bash
# Activate virtual environment
source .venv/bin/activate

# Set optimized environment variables
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run with minimal settings for testing
python scripts/run_neural_plasticity_notebook.py --minimal
```

### Option 2: Google Colab (Recommended)

For running in Colab, follow these steps:

1. Open [NeuralPlasticityDemo.ipynb](https://colab.research.google.com/github/CambrianTech/sentinel-ai/blob/feature/implement-adaptive-plasticity/colab_notebooks/NeuralPlasticityDemo.ipynb) in Google Colab
2. Select Runtime > Change runtime type > Hardware accelerator: GPU
3. Run the notebook cells in sequence

## Testing Individual Components

If you just want to verify that the tensor operations work correctly:

```bash
# Test only the tensor handling
python scripts/test_tensor_handling.py
```

This will test:
- Attention tensor creation
- Entropy calculation
- Visualization functionality
- Safe tensor display utilities

## Implementation Details

The modular architecture includes:

- **Core Module**: `utils/neural_plasticity/core.py`
  - Contains fundamental algorithms and tensor operations

- **Visualization Module**: `utils/neural_plasticity/visualization.py`
  - Provides visualization utilities for entropy, gradients, etc.

- **Training Module**: `utils/neural_plasticity/training.py`
  - Implements differential learning for pruned vs. active heads

## Troubleshooting

If you encounter issues:

### Apple Silicon (M1/M2/M3) Users

The module now includes automatic detection and fixes for Apple Silicon, but if you still encounter BLAS crashes:

1. **Force CPU Usage**: Add `device="cpu"` to model loading and all tensor operations
2. **Install JAX**: Make sure you have JAX installed with `pip install jax jaxlib`
3. **Single-Threaded BLAS**: Set the environment variables shown in the Local Execution section
4. **Visualization Issues**: If you see rendering errors, try forcing `matplotlib.use('Agg')`

### General Troubleshooting

1. **BLAS Errors**: Set single-threaded environment variables as shown above
2. **Memory Errors**: Reduce batch size and sequence length in the configuration cell
3. **Visualization Errors**: Make sure matplotlib is properly configured
4. **GPU Errors**: Try running on CPU if GPU errors persist

## Next Steps

Further improvements planned:

1. Make tensor operations more memory-efficient
2. Add more visualization options for head dynamics
3. Implement adaptive pruning schedules
4. Add support for more model architectures