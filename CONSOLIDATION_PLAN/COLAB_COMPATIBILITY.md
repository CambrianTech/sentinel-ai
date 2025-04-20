# Colab Notebook Compatibility Plan

This document outlines the strategy for ensuring that our neural plasticity implementation works seamlessly both in local environments and in Google Colab notebooks.

## Current Notebook Status

1. **NeuralPlasticityDemo.ipynb**
   - Main demonstration notebook for neural plasticity
   - Contains various visualizations and interactive elements
   - Needs to be updated to use consolidated implementation

2. **PruningAndFineTuningColab.ipynb**
   - Focused on pruning and fine-tuning experiments
   - Demonstrates dual-mode pruning approach
   - Needs compatibility updates for cross-platform support

## Integration Challenges

1. **Code Duplication**
   - Notebooks contain duplicated code that should be imported from modules
   - Visualizations are often reimplemented in notebooks
   - Core algorithm implementations sometimes differ

2. **Environment Differences**
   - Colab uses T4 GPU with different memory constraints
   - File paths and storage are different (Google Drive vs. local)
   - Installation and dependency management varies

3. **Notebook Cell Structure**
   - Notebooks need to display progress and results inline
   - Visualizations must work within notebook cells
   - User interaction requires different approaches

## Integration Strategy

### 1. Modular Structure for Notebooks

```
scripts/neural_plasticity/colab/
├── __init__.py
├── integration.py        # Core Colab integration utilities
├── visualization.py      # Notebook-specific visualization
├── installation.py       # Dependency management
├── progress.py           # Progress tracking
└── interactive.py        # Interactive widgets
```

### 2. Notebook Refactoring Approach

1. **Import Structure**
   - Replace inline code with imports from our modular packages
   - Add installation cell that installs from the GitHub repository

```python
# Installation cell
!pip install -q "git+https://github.com/CambrianTech/sentinel-ai.git@feature/implement-adaptive-plasticity"
```

2. **Core Logic Separation**
   - Move all core algorithm logic to the package
   - Keep only visualization and interaction in the notebook

```python
# Before refactoring
def compute_entropy(attn_probs):
    # Complex implementation...
    return entropy

# After refactoring
from sentinel.pruning.entropy_magnitude import compute_attention_entropy
```

3. **Visualization Abstraction**
   - Create visualization functions that work both in notebooks and standalone
   - Use environment detection to adjust visualization approach

```python
from scripts.neural_plasticity.colab.visualization import plot_entropy_heatmap

# Will work both in notebook and standalone
plot_entropy_heatmap(entropy_values, layer_idx=0)
```

### 3. Data Storage and Loading

1. **Unified Experiment Output Structure**
   - Use consistent directory structure across environments
   - Implement adaptable path handling for Colab/local differences

```python
from scripts.neural_plasticity.colab.integration import get_output_dir

# Will return appropriate path for Colab or local
output_dir = get_output_dir("neural_plasticity_experiment")
```

2. **Results Saving and Loading**
   - Create utilities for saving results in a portable format
   - Implement functions to load results from either environment

```python
from scripts.neural_plasticity.colab.integration import save_experiment_results, load_experiment_results

# Save results (handles Colab/local differences)
save_experiment_results(results, "experiment_1")

# Load results (works from either environment)
results = load_experiment_results("experiment_1")
```

### 4. Progress Tracking and UI

1. **Colab-Specific Progress Tracking**
   - Implement progress bars and status indicators for Colab
   - Display real-time metrics during training

```python
from scripts.neural_plasticity.colab.progress import ProgressTracker

tracker = ProgressTracker("Training model")
for i in range(100):
    # Do work
    tracker.update(i, {"loss": current_loss})
```

2. **Interactive Widgets**
   - Create widgets for parameter selection and visualization control
   - Ensure widgets work in Colab environment

```python
from scripts.neural_plasticity.colab.interactive import PruningStrategySelector

selector = PruningStrategySelector()
strategy = selector.get_selected_strategy()
```

## Implementation Plan

### 1. Create Colab Integration Package

```python
# scripts/neural_plasticity/colab/integration.py
import os
import platform
import sys

def is_colab():
    """Detect if code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_output_dir(experiment_name):
    """Get appropriate output directory for current environment."""
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        base_dir = "/content/drive/MyDrive/neural_plasticity_experiments"
    else:
        base_dir = "experiment_output/neural_plasticity"
    
    # Create timestamp-based directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir
```

### 2. Create Visualization Module

```python
# scripts/neural_plasticity/colab/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from scripts.neural_plasticity.colab.integration import is_colab

def initialize_visualization():
    """Set up visualization environment."""
    if is_colab():
        # Colab-specific setup
        %matplotlib inline
        from google.colab import output
        output.enable_custom_widget_manager()
    else:
        # Local setup
        import platform
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            # Use Agg backend for Apple Silicon to avoid crashes
            import matplotlib
            matplotlib.use('Agg')

def plot_entropy_heatmap(entropy_values, title="Attention Entropy", layer_idx=None):
    """Plot entropy heatmap that works in both environments."""
    fig, ax = plt.figure(figsize=(10, 6))
    
    # Generate heatmap
    im = ax.imshow(entropy_values, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set title with layer info if provided
    if layer_idx is not None:
        ax.set_title(f"{title} (Layer {layer_idx})")
    else:
        ax.set_title(title)
    
    # Axis labels
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index" if layer_idx is None else "Attention Position")
    
    # Show plot
    if is_colab():
        # In Colab, display inline
        plt.show()
    else:
        # In local env, either show or return figure
        return fig
```

### 3. Refactor Notebook

1. **Add Installation Cell**
```python
# Installation and setup
!pip install -q git+https://github.com/CambrianTech/sentinel-ai.git@feature/implement-adaptive-plasticity
!pip install -q matplotlib numpy torch transformers datasets tqdm
```

2. **Import Core Modules**
```python
# Import core modules
from scripts.neural_plasticity.run_experiment import main as run_neural_plasticity
from scripts.neural_plasticity.colab.integration import get_output_dir, is_colab
from scripts.neural_plasticity.colab.visualization import initialize_visualization, plot_entropy_heatmap
```

3. **Setup Environment**
```python
# Setup environment
initialize_visualization()
output_dir = get_output_dir("neural_plasticity_demo")
print(f"Results will be saved to: {output_dir}")
```

4. **Run Experiment**
```python
# Configure experiment
experiment_args = {
    "model_name": "distilgpt2",
    "pruning_strategy": "entropy",
    "pruning_level": 0.2,
    "output_dir": output_dir,
    "quick_test": True  # For demo purposes
}

# Run the experiment
results = run_neural_plasticity(**experiment_args)
```

5. **Visualize Results**
```python
# Generate visualizations
from scripts.neural_plasticity.visualization.dashboard_generator import generate_dashboard

# Create dashboard
dashboard_path = os.path.join(output_dir, "dashboard.html")
generate_dashboard(
    experiment_dir=output_dir,
    output_path=dashboard_path,
    model_name=experiment_args["model_name"],
    pruning_strategy=experiment_args["pruning_strategy"],
    pruning_level=experiment_args["pruning_level"]
)

# Display link to dashboard
from IPython.display import HTML, display
if is_colab():
    display(HTML(f'<a href="{dashboard_path}" target="_blank">View Dashboard</a>'))
else:
    print(f"Dashboard generated at: {dashboard_path}")
```

## Testing Strategy

1. **Test Matrix Approach**
   - Test in Colab with T4 GPU enabled
   - Test in Colab with CPU only
   - Test locally with the same notebook

2. **Automated Testing**
   - Create CI tests for notebook functionality
   - Implement test fixtures for Colab environment simulation
   - Test core functionality from notebooks

3. **Result Validation**
   - Verify experiment results match between environments
   - Confirm visualizations render correctly in both
   - Ensure dashboards are generated with the same content

## Implementation Roadmap

1. **Create Core Colab Integration Package** (scripts/neural_plasticity/colab/)
2. **Update Visualization Code to be Cross-Platform**
3. **Refactor NeuralPlasticityDemo.ipynb**
4. **Refactor PruningAndFineTuningColab.ipynb**
5. **Create Unit Tests for Colab Integration**
6. **Test End-to-End in Both Environments**
7. **Create Documentation for Using Notebooks**