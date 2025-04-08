# Visualization Code Refactoring

This directory contains scripts for refactoring the visualization code in the NeuralPlasticityDemo notebook into proper utility functions.

## Overview

The goal is to make the visualization code:
1. More maintainable and reusable
2. Properly typed and documented
3. Testable independently from the notebook
4. Easier to update and improve

## Files

- `update_notebook_to_use_utility_viz.py`: Updates the NeuralPlasticityDemo.ipynb to use the refactored utility functions
- `test_visualization_utils.py`: Tests for the visualization utility functions
- `/utils/pruning/visualization_additions.py`: The refactored visualization utility functions
- `/utils/pruning/visualization_README.md`: Documentation for the visualization utilities

## Usage

### Update Notebook to Use Utilities

```bash
# Update the notebook to use utility functions
python temp_fix/update_notebook_to_use_utility_viz.py
```

This script will:
1. Add an import statement for the visualization functions
2. Remove inline function definitions that are now imported
3. Update function references if necessary
4. Increment the version number and add a changelog entry

### Test Visualization Utilities

```bash
# Run tests for the visualization utilities
python temp_fix/test_visualization_utils.py
```

## Implementation Details

### Visualization Functions

The following functions have been refactored:

1. `visualize_gradient_norms()`: Show gradient norms with markers for pruned/revived heads
2. `visualize_attention_matrix()`: Visualize attention patterns for specific layers/heads
3. `visualize_entropy_heatmap()`: Display entropy values across layers and heads
4. `visualize_normalized_entropy()`: Show entropy as percentage of maximum possible
5. `visualize_entropy_vs_gradient()`: Create scatter plot of entropy vs gradient values
6. `visualize_training_progress()`: Show training metrics including loss, pruning and sparsity

Each function:
- Has proper type hints
- Includes detailed docstrings
- Returns a matplotlib Figure object
- Handles both torch.Tensor and numpy.ndarray inputs
- Has proper error handling
- Maintains consistent style and interface

### Benefits

Refactoring the visualization code into utility functions provides several benefits:

1. **Code Reusability**: The functions can be used in other notebooks and scripts
2. **Maintainability**: Easier to fix bugs or make improvements in one place
3. **Testability**: Can be tested independently from the notebook
4. **Consistency**: Ensures consistent visualization style and behavior
5. **Documentation**: Better documentation for how to use visualization functions

This approach follows best practices for notebook development, keeping core functionality in importable modules while using notebooks for interactive exploration and demonstration.