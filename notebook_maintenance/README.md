# Notebook Maintenance Tools

This directory contains tools for maintaining Jupyter notebooks in the Sentinel AI project, following the guidelines specified in CLAUDE.md.

## Tools

### `comprehensive_notebook_validation.py`

Performs extensive validation of Jupyter notebooks, checking for common issues and best practices.

**Usage:**
```bash
python comprehensive_notebook_validation.py [notebook_path]
```

**Checks performed:**
- Cell execution order
- Import statements
- Tensor operations
- Variable dependencies
- Device handling
- Colab compatibility
- Error handling

### `fix_colab_compatibility.py`

Fixes common Colab compatibility issues in notebooks.

**Usage:**
```bash
python fix_colab_compatibility.py [notebook_path] [output_path]
```

**Fixes applied:**
- Adds %matplotlib inline
- Adds system dependency checks
- Improves error handling
- Deduplicates import statements
- Fixes cell execution counts

### `fix_neural_plasticity_demo.py`

Fixes GPU tensor visualization issues and improves utilities integration in the NeuralPlasticityDemo.ipynb notebook.

**Usage:**
```bash
python fix_neural_plasticity_demo.py [input_notebook] [output_notebook]
```

**Fixes applied:**
- Corrects improper tensor detach calls (`.detach(.detach().cpu().numpy())`)
- Fixes redundant CPU/numpy conversions (`.cpu().numpy().cpu().numpy()`)
- Fixes visualization imports and pruning monitor integration
- Ensures proper tensor handling for matplotlib visualizations

### `backup_and_fix_notebook.py`

Creates a timestamped backup of a notebook before applying fixes.

**Usage:**
```bash
python backup_and_fix_notebook.py [notebook_path]
```

### `increment_notebook_version.py`

Increments the version number in a notebook's markdown header and adds notes about the changes.

**Usage:**
```bash
python increment_notebook_version.py [notebook_path] [output_path]
```

### `validate_neural_plasticity_notebook.py`

Validates a notebook for common issues and errors.

**Usage:**
```bash
python validate_neural_plasticity_notebook.py [notebook_path]
```

## Development Guidelines

As specified in CLAUDE.md, follow these guidelines when modifying notebooks:

1. **NEVER** use sed, awk, or regex tools to edit .ipynb files directly
2. Always use `nbformat` for all modifications to code cells, metadata, or outputs
3. Create a backup before making significant changes
4. Always validate notebooks after making changes
5. Increment version numbers when fixing bugs
6. Modularize code into utilities where possible

## Example Workflow

```bash
# Create a backup and fix issues
python backup_and_fix_notebook.py colab_notebooks/NeuralPlasticityDemo.ipynb

# Validate the notebook
python validate_neural_plasticity_notebook.py colab_notebooks/NeuralPlasticityDemo.ipynb

# Increment the version number
python increment_notebook_version.py colab_notebooks/NeuralPlasticityDemo.ipynb
```