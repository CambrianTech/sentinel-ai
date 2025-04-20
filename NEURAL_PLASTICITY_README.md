# Neural Plasticity Cross-Platform Guide

This guide provides instructions for running the Neural Plasticity notebook across different platforms, with special focus on resolving issues on Apple Silicon (M1/M2/M3) chips and ensuring cross-platform compatibility.

## Overview

The Neural Plasticity Demo showcases how transformer models can adapt their structure through dynamic pruning and regrowth of attention heads. This mimics how biological brains form efficient neural pathways by strengthening useful connections and pruning unused ones.

## Key Features

- **Attention Head Entropy Analysis**: Identify unfocused/dispersed attention patterns
- **Gradient-Based Importance**: Measure each head's contribution to the model's output
- **Dynamic Pruning**: Selectively remove heads with high entropy and low gradient impact
- **Adaptive Recovery**: Optionally restore pruned heads that become more useful
- **Visualization**: Track and visualize the pruning decisions and model performance

## Platform-Specific Optimizations

The notebook automatically detects your environment and applies appropriate optimizations:

### Apple Silicon (M1/M2/M3)

On Apple Silicon, the code:
- Detects Apple chips using platform module
- Forces CPU execution to avoid BLAS/libtorch crashes
- Disables multi-threading for BLAS operations
- Uses safer tensor operations with multiple fallback mechanisms
- Implements safe matrix multiplication that avoids native BLAS issues
- Adds special handling for visualization operations

### Google Colab

In Colab environments, the code:
- Detects GPU availability and optimizes operations accordingly
- Utilizes GPU acceleration when available for improved performance
- Handles proper tensor placement between CPU/GPU for visualizations
- Pre-installs required dependencies and resolves import conflicts

### Standard Hardware

On regular systems (Intel/AMD), the code:
- Utilizes GPU if available while preserving CPU compatibility
- Uses standard PyTorch operations without special handling
- Takes advantage of multi-threading for better performance

## Running the Notebook

### Setting Up Environment

First, ensure you're in a virtual environment:

```bash
# Activate the existing virtual environment
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows
```

### Installing Dependencies

Next, install all required dependencies:

```bash
# Automatically install all required dependencies
python scripts/install_neural_plasticity_deps.py
```

This will install:
- PyTorch: Base deep learning framework
- Transformers: For loading and using transformer models
- Datasets: For data loading and processing
- Matplotlib & Seaborn: For visualizations
- nbformat & jupyter: For notebook processing

### Fixing the Dataset Import Conflict

The notebook requires the `datasets` module, but there's a known circular import issue. We've created scripts to fix this problem:

```bash
# Option 1: Fix dataset imports only
python scripts/fix_neural_plasticity_datasets.py

# Option 2: Fix all issues (imports, Apple Silicon compatibility, etc.)
python scripts/fix_neural_plasticity_imports.py

# Option 3: Fix and execute the notebook end-to-end
python scripts/run_neural_plasticity_notebook_e2e.py
```

### Running on Apple Silicon (M1/M2/M3)

```bash
# Method 1: Run with fixes and safety measures
python scripts/run_neural_plasticity_notebook_e2e.py

# Method 2: Fix dataset imports first
python scripts/fix_neural_plasticity_datasets.py
# Then open and run the fixed notebook in JupyterLab
jupyter lab notebooks/NeuralPlasticityDemo_datasets_fixed.ipynb
```

### Running in Google Colab

1. Upload the notebook to Colab
2. Ensure you've selected a GPU runtime (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Run all cells - the first few cells will automatically clone the repository and set up the environment

### Running on Standard Hardware

```bash
# Fix and run the notebook
python scripts/run_neural_plasticity_notebook_e2e.py
```

## Troubleshooting

### Apple Silicon Issues

**Problem**: BLAS/libtorch crashes during matrix operations
**Solution**: We've implemented a comprehensive solution with multiple fallbacks:
1. First attempt: Safely convert tensors and use NumPy's matrix multiplication
2. Second attempt: Use a Python-based implementation for small matrices
3. Last resort: Use PyTorch with single-threading and safety measures

### Dataset Import Conflicts

**Problem**: Circular imports with the `datasets` module
**Solution**: We pre-import the module and inject the necessary functions to avoid circular references

### GPU Memory Issues

**Problem**: Out of memory errors on GPU
**Solution**: Reduce batch size or sequence length in the notebook configuration section

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/fix_neural_plasticity_datasets.py` | Fixes dataset import conflicts |
| `scripts/fix_neural_plasticity_imports.py` | Comprehensive fixes for imports and cross-platform compatibility |
| `scripts/run_neural_plasticity_notebook_e2e.py` | Fixes and executes the notebook end-to-end |
| `scripts/run_neural_plasticity_minimal.py` | Runs a minimal version of the notebook |
| `scripts/fix_neural_plasticity_local.py` | Applies local environment fixes |

## Core Modules

The neural plasticity functionality is implemented in these core modules:

- `utils/neural_plasticity/core.py`: Core algorithms for entropy calculation, gradient analysis, and pruning
- `utils/neural_plasticity/visualization.py`: Visualization utilities for attention patterns and metrics
- `utils/colab/helpers.py`: Colab-specific utilities for environment detection and tensor handling

## Conclusion

With the fixes and optimizations we've implemented, the Neural Plasticity Demo should now run reliably across all platforms, including Apple Silicon Macs that previously experienced BLAS/libtorch crashes.