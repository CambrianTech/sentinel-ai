"""
Neural Plasticity Integration Module for PruningAndFineTuningColab

This module provides integration with Sentinel AI's neural plasticity system,
allowing dynamic pruning and revival of attention heads during training.

Usage:
    from PlasticityIntegration import setup_plasticity, apply_plasticity, visualize_plasticity
    
    # Create controller
    controller = setup_plasticity(model, mode="adaptive")
    
    # During training loop, periodically call:
    pruned, revived = apply_plasticity(controller, dataloader)
    
    # After training, visualize the results:
    visualize_plasticity(controller)
"""

import torch
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets

# Try to import from sentinel package
try:
    from sentinel.pruning.dual_mode_pruning import PruningMode
    from sentinel.pruning.plasticity_controller import create_plasticity_controller
    PLASTICITY_AVAILABLE = True
except ImportError:
    # Fallback - try to clone repository
    import subprocess
    import os
    import sys
    
    if not os.path.exists("/content/sentinel-ai"):
        print("Cloning repository for plasticity support...")
        subprocess.check_call([
            "git", "clone", "-b", "feature/implement-adaptive-plasticity", 
            "https://github.com/CambrianTech/sentinel-ai.git", "/content/sentinel-ai"
        ])
        
        # Add to Python path
        if "/content/sentinel-ai" not in sys.path:
            sys.path.insert(0, "/content/sentinel-ai")
    
    try:
        from sentinel.pruning.dual_mode_pruning import PruningMode
        from sentinel.pruning.plasticity_controller import create_plasticity_controller
        PLASTICITY_AVAILABLE = True
    except ImportError:
        print("Could not import plasticity components. Plasticity will be disabled.")
        PLASTICITY_AVAILABLE = False


def create_plasticity_ui():
    """Create a user interface widget for configuring neural plasticity."""
    if not PLASTICITY_AVAILABLE:
        return display(widgets.HTML(
            "<div style='background-color: #ffe6e6; padding: 10px; border-radius: 5px;'>"
            "<b>Neural Plasticity not available.</b> Please make sure you have the latest version "
            "of the Sentinel AI repository with feature/implement-adaptive-plasticity branch."
            "</div>"
        ))
    
    # Create widgets
    enable_plasticity = widgets.Checkbox(
        value=True,
        description='Enable Neural Plasticity',
        disabled=False
    )
    
    plasticity_mode = widgets.RadioButtons(
        options=[
            ('Adaptive (allows recovery)', 'adaptive'),
            ('Compressed (permanent)', 'compressed')
        ],
        value='adaptive',
        description='Plasticity Mode:',
        disabled=False
    )
    
    high_entropy_threshold = widgets.FloatSlider(
        value=0.8,
        min=0.5,
        max=1.0,
        step=0.05,
        description='High Entropy Threshold:',
        disabled=False,
        continuous_update=False,
        readout=True,
        readout_format='.2f'
    )
    
    low_entropy_threshold = widgets.FloatSlider(
        value=0.4,
        min=0.1,
        max=0.7,
        step=0.05,
        description='Low Entropy Threshold:',
        disabled=False,
        continuous_update=False,
        readout=True,
        readout_format='.2f'
    )
    
    grad_threshold = widgets.FloatLogSlider(
        value=1e-4,
        base=10,
        min=-6,  # 10^-6
        max=-2,  # 10^-2
        step=0.2,
        description='Gradient Threshold:',
        disabled=False,
        continuous_update=False,
        readout=True,
        readout_format='.2e'
    )
    
    min_zero_epochs = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description='Min Zero Epochs:',
        disabled=False,
        continuous_update=False,
        readout=True
    )
    
    eval_interval = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description='Eval Interval (steps):',
        disabled=False,
        continuous_update=False,
        readout=True
    )
    
    # Create description widgets
    adaptive_desc = widgets.HTML(
        value="""<div style="margin-left: 20px; margin-bottom: 10px; color: #555;">
            <b>Adaptive Mode</b>: Temporarily zeros weights, allows recovery during fine-tuning.<br>
            <small>• Better for maximizing quality</small><br>
            <small>• Allows heads to recover if needed</small><br>
            <small>• More flexible neural plasticity</small>
        </div>"""
    )
    
    compressed_desc = widgets.HTML(
        value="""<div style="margin-left: 20px; margin-bottom: 20px; color: #555;">
            <b>Compressed Mode</b>: Permanently zeros weights, prevents recovery.<br>
            <small>• Better for deployment/efficiency</small><br>
            <small>• Creates true sparsity in the model</small><br>
            <small>• Can be exported as a smaller model</small>
        </div>"""
    )
    
    threshold_desc = widgets.HTML(
        value="""<div style="margin-top: 10px; margin-bottom: 10px; color: #555;">
            <b>Threshold Explanation:</b><br>
            <small>• High Entropy Threshold: Heads with entropy above this are candidates for pruning</small><br>
            <small>• Low Entropy Threshold: Pruned heads with entropy below this are candidates for revival</small><br>
            <small>• Gradient Threshold: Heads with gradient norm below this are candidates for pruning</small><br>
            <small>• Min Zero Epochs: Minimum epochs a head should remain pruned before revival consideration</small>
        </div>"""
    )
    
    # Layout the widgets
    header = widgets.HTML("<h3>Neural Plasticity Configuration</h3>")
    
    # Create a container for all the widgets
    container = widgets.VBox([
        header,
        enable_plasticity,
        widgets.HBox([widgets.VBox([plasticity_mode]), widgets.VBox([adaptive_desc, compressed_desc])]),
        threshold_desc,
        widgets.HBox([
            widgets.VBox([high_entropy_threshold, low_entropy_threshold]),
            widgets.VBox([grad_threshold, min_zero_epochs, eval_interval])
        ])
    ])
    
    # Display the widget
    display(container)
    
    # Return the widget values in a dictionary
    return {
        'enable_plasticity': enable_plasticity,
        'plasticity_mode': plasticity_mode,
        'high_entropy_threshold': high_entropy_threshold,
        'low_entropy_threshold': low_entropy_threshold,
        'grad_threshold': grad_threshold,
        'min_zero_epochs': min_zero_epochs,
        'eval_interval': eval_interval
    }


def setup_plasticity(model, widget_values=None, mode="adaptive", 
                    high_entropy_threshold=0.8, low_entropy_threshold=0.4,
                    grad_threshold=1e-4, min_zero_epochs=3):
    """
    Set up the neural plasticity controller.
    
    Args:
        model: The transformer model
        widget_values: Values from the plasticity UI widget (optional)
        mode: Plasticity mode ('adaptive' or 'compressed')
        high_entropy_threshold: Threshold for pruning heads
        low_entropy_threshold: Threshold for reviving heads
        grad_threshold: Gradient norm threshold
        min_zero_epochs: Minimum epochs to keep head zeroed
        
    Returns:
        Plasticity controller or None if not available
    """
    if not PLASTICITY_AVAILABLE:
        print("Neural plasticity is not available. Skipping plasticity setup.")
        return None
    
    # Check if we have widget values
    if widget_values is not None:
        # Check if plasticity is enabled
        if not widget_values['enable_plasticity'].value:
            print("Neural plasticity is disabled in settings. Skipping plasticity setup.")
            return None
        
        # Get values from widgets
        mode = widget_values['plasticity_mode'].value
        high_entropy_threshold = widget_values['high_entropy_threshold'].value
        low_entropy_threshold = widget_values['low_entropy_threshold'].value
        grad_threshold = widget_values['grad_threshold'].value
        min_zero_epochs = widget_values['min_zero_epochs'].value
    
    # Set pruning mode
    pruning_mode = PruningMode.ADAPTIVE if mode == "adaptive" else PruningMode.COMPRESSED
    
    # Create controller
    try:
        controller = create_plasticity_controller(
            model=model,
            mode=pruning_mode,
            high_entropy_threshold=high_entropy_threshold,
            low_entropy_threshold=low_entropy_threshold,
            grad_threshold=grad_threshold,
            min_zero_epochs=min_zero_epochs
        )
        
        print(f"Plasticity controller created with mode: {pruning_mode.value}")
        print(f"Total layers: {controller.total_layers}, Heads per layer: {controller.heads_per_layer}")
        
        return controller
    except Exception as e:
        print(f"Error creating plasticity controller: {e}")
        return None


def apply_plasticity(controller, dataloader, num_batches=2, verbose=True):
    """
    Apply neural plasticity (pruning/revival) based on current metrics.
    
    Args:
        controller: The plasticity controller
        dataloader: DataLoader for evaluation
        num_batches: Number of batches to process
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (pruned_heads, revived_heads) or (None, None) if controller is None
    """
    if controller is None:
        return None, None
    
    try:
        # Collect metrics and apply plasticity
        pruned_heads, revived_heads, metrics = controller.step(
            dataloader, 
            num_batches=num_batches,
            verbose=verbose
        )
        
        # Return pruned and revived heads
        return pruned_heads, revived_heads
    except Exception as e:
        print(f"Error applying plasticity: {e}")
        return None, None


def visualize_plasticity(controller, save_path=None):
    """
    Visualize neural plasticity metrics and head dynamics.
    
    Args:
        controller: The plasticity controller
        save_path: Optional path to save visualizations
        
    Returns:
        None
    """
    if controller is None:
        print("No plasticity controller available for visualization.")
        return
    
    try:
        # Get controller summary
        summary = controller.get_summary()
        
        # Display summary
        print("Neural Plasticity Summary:")
        print(f"  Total heads: {summary['total_heads']}")
        print(f"  Pruned heads: {summary['pruned_heads']} ({summary['pruning_rate']:.2%})")
        print(f"  Model sparsity: {summary['sparsity']:.4f}")
        
        # Create visualizations
        fig1 = controller.visualize_head_dynamics(metric='entropy', save_path=save_path)
        plt.title("Head Entropy Dynamics")
        plt.show()
        
        fig2 = controller.visualize_head_dynamics(metric='decision', save_path=save_path)
        plt.title("Head Pruning Decisions")
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing plasticity: {e}")


# Example of how to use this in the PruningAndFineTuningColab:
if __name__ == "__main__":
    # This code only runs when this module is executed directly
    print("Neural Plasticity Integration Module")
    print("To use, import this module in PruningAndFineTuningColab.py")
    
    # Create UI widget
    widget_values = create_plasticity_ui()