# Test Scripts

This directory contains test scripts for various components of the Sentinel AI project, including some specialized workarounds for platform-specific issues.

## PyTorch BLAS Issues on M1/M2 Macs

Some scripts in this directory specifically address BLAS crashes that can occur on Apple Silicon (M1/M2) Macs when using PyTorch for certain operations, particularly with transformer attention mechanisms.

### Available Solutions

1. **Environment Variable Fixes**
   - `minimal_pruning.py` - Basic PyTorch fix using environment variables
   - `mac_pytorch_fix.py` - More comprehensive PyTorch threading controls

   These scripts set important environment variables:
   ```python
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   os.environ["OPENBLAS_NUM_THREADS"] = "1"
   os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
   ```

2. **JAX/Flax Implementation**
   - `flax_pruning.py` - Complete pruning implementation using JAX/Flax
   - `flax_workaround.py` - Minimal demo of JAX/Flax functionality

   These scripts completely bypass PyTorch BLAS issues by using Google's JAX library and Flax for transformer models.

## Usage

### PyTorch with Environment Variables

If you want to try fixing PyTorch:

```bash
python minimal_pruning.py
```

### JAX/Flax Implementation (Recommended)

For reliable pruning on M1/M2 Macs:

```bash
python flax_pruning.py --strategy random --pruning_level 0.1 --prompt "Artificial intelligence will"
```

This script includes automatic installation of JAX if needed, and works on both macOS and Google Colab.

## Notebooks

For Jupyter Notebook versions of these solutions, see:

- `/notebooks/JaxPruningBenchmark.ipynb` - JAX/Flax implementation
- `/notebooks/StablePruningBenchmark.ipynb` - PyTorch implementation with fixes