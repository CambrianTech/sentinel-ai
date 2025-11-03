"""
Widget for selecting pruning mode in Colab notebooks.

Usage:
    from PruningModeSelectorWidget import create_pruning_widget, PruningMode
    
    mode_selector, pruning_mode = create_pruning_widget()
    # pruning_mode will be updated when the user changes the selection
"""

from ipywidgets import widgets
from IPython.display import display
from enum import Enum


class PruningMode(str, Enum):
    """Pruning modes for Sentinel-AI."""
    ADAPTIVE = "adaptive"    # Allows heads to recover during fine-tuning
    COMPRESSED = "compressed"  # Permanently zeros weights


def create_pruning_widget():
    """
    Create a widget for selecting pruning mode in Colab notebooks.
    
    Returns:
        Tuple containing (widget, pruning_mode)
    """
    # Create shared pruning mode state
    pruning_mode = PruningMode.ADAPTIVE
    
    # Create widget
    mode_selector = widgets.RadioButtons(
        options=[
            ('Adaptive (quality focus)', PruningMode.ADAPTIVE), 
            ('Compressed (size/speed focus)', PruningMode.COMPRESSED)
        ],
        description='Pruning Mode:',
        disabled=False,
        layout=widgets.Layout(width='auto', margin='10px')
    )
    
    # Add descriptions
    adaptive_desc = widgets.HTML(
        value="""<div style="margin-left: 20px; margin-bottom: 10px; color: #555;">
            <b>Adaptive</b>: Temporarily zeros weights, allows recovery during fine-tuning.<br>
            <small>• Better for maximizing quality</small><br>
            <small>• Allows heads to recover if needed</small><br>
            <small>• Does not reduce model size</small>
        </div>"""
    )
    
    compressed_desc = widgets.HTML(
        value="""<div style="margin-left: 20px; margin-bottom: 20px; color: #555;">
            <b>Compressed</b>: Permanently zeros weights, prevents recovery.<br>
            <small>• Better for deployment/efficiency</small><br>
            <small>• Maintains true sparsity during training</small><br>
            <small>• Can be exported as smaller model</small>
        </div>"""
    )
    
    # Define callback when mode changes
    def on_mode_change(change):
        nonlocal pruning_mode
        if change['type'] == 'change' and change['name'] == 'value':
            pruning_mode = change['new']
            print(f"Pruning mode set to: {pruning_mode}")
    
    # Register callback
    mode_selector.observe(on_mode_change, names='value')
    
    # Display widget with descriptions
    container = widgets.VBox([
        widgets.HTML("<h3>Pruning Mode Selection</h3>"),
        mode_selector,
        adaptive_desc,
        compressed_desc
    ])
    
    return container, pruning_mode