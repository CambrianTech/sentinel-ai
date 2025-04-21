"""
Interactive widgets for neural plasticity in Colab.

This module provides interactive widgets that work in Colab notebooks,
allowing users to select parameters, configure experiments, and interact
with visualizations.
"""

import os
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

# Local imports
from scripts.neural_plasticity.colab.integration import is_colab

def ensure_ipywidgets():
    """
    Ensure ipywidgets is installed and properly configured.
    
    This is necessary for interactive widgets in Colab.
    """
    if not is_colab():
        # In local environment, just check if ipywidgets is available
        try:
            import ipywidgets
            return True
        except ImportError:
            print("ipywidgets not available. Install with: pip install ipywidgets")
            return False
    
    # In Colab, try to set up widgets
    try:
        # Check if ipywidgets is available
        import ipywidgets
        
        # Initialize Colab notebook widgets
        from google.colab import output
        output.enable_custom_widget_manager()
        
        return True
    except ImportError:
        # Install ipywidgets if not available
        import subprocess
        import sys
        
        print("Installing ipywidgets...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "ipywidgets"
        ])
        
        # Initialize Colab widgets
        try:
            import ipywidgets
            from google.colab import output
            output.enable_custom_widget_manager()
            return True
        except Exception as e:
            print(f"Failed to initialize widgets: {e}")
            return False

class PruningStrategySelector:
    """Widget for selecting pruning strategy and parameters."""
    
    def __init__(self, on_change: Optional[Callable] = None):
        """
        Initialize pruning strategy selector.
        
        Args:
            on_change: Callback function when selection changes
        """
        self.on_change = on_change
        self.widget = None
        self.description = None
        
        # Check if widgets are available
        if not ensure_ipywidgets():
            print("WARNING: Interactive widgets not available. Using default parameters.")
            self.strategy = "entropy"
            self.level = 0.2
            return
        
        # Import widgets
        import ipywidgets as widgets
        from IPython.display import display
        
        # Create strategy dropdown
        self.strategy_dropdown = widgets.Dropdown(
            options=[
                ('Entropy-based Pruning', 'entropy'),
                ('Magnitude-based Pruning', 'magnitude'),
                ('Combined (Entropy + Magnitude)', 'combined')
            ],
            description='Strategy:',
            value='entropy',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        # Create level slider
        self.level_slider = widgets.FloatSlider(
            value=0.2,
            min=0.05,
            max=0.5,
            step=0.05,
            description='Pruning Level:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        # Create weight slider for combined strategy
        self.weight_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Entropy Weight:',
            disabled=self.strategy_dropdown.value != 'combined',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        # Create description area
        self.description = widgets.HTML(
            value=self._get_strategy_description(self.strategy_dropdown.value),
            layout=widgets.Layout(width='500px', margin='10px 0')
        )
        
        # Register callbacks
        self.strategy_dropdown.observe(self._on_strategy_change, names='value')
        self.level_slider.observe(self._on_param_change, names='value')
        self.weight_slider.observe(self._on_param_change, names='value')
        
        # Create main widget container
        self.widget = widgets.VBox([
            widgets.HBox([
                widgets.VBox([
                    self.strategy_dropdown,
                    self.level_slider,
                    self.weight_slider
                ]),
                self.description
            ]),
            widgets.HTML(value="<hr style='margin: 10px 0'>")
        ])
        
        # Display the widget
        display(self.widget)
    
    def _get_strategy_description(self, strategy):
        """Get HTML description for a pruning strategy."""
        if strategy == 'entropy':
            return """
            <p><b>Entropy-based Pruning</b> removes attention heads that have high entropy (unfocused attention).</p>
            <ul>
                <li>Identifies heads that distribute attention broadly (high entropy)</li>
                <li>Preserves heads with focused attention patterns</li>
                <li>Good for improving focus on relevant tokens</li>
            </ul>
            """
        elif strategy == 'magnitude':
            return """
            <p><b>Magnitude-based Pruning</b> removes attention heads with small weight magnitudes.</p>
            <ul>
                <li>Identifies heads with small weight values</li>
                <li>Preserves heads with larger weights (more influence)</li>
                <li>Good for reducing model size with minimal impact</li>
            </ul>
            """
        elif strategy == 'combined':
            return """
            <p><b>Combined Pruning</b> uses both entropy and magnitude scores.</p>
            <ul>
                <li>Balances between entropy and magnitude metrics</li>
                <li>Adjustable weight determines the relative importance</li>
                <li>Good for finding optimal balance between focus and influence</li>
            </ul>
            """
        return ""
    
    def _on_strategy_change(self, change):
        """Handle change in pruning strategy."""
        # Update description
        self.description.value = self._get_strategy_description(change['new'])
        
        # Enable/disable weight slider for combined strategy
        self.weight_slider.disabled = change['new'] != 'combined'
        
        # Call user callback if provided
        if self.on_change:
            self.on_change(self.get_config())
    
    def _on_param_change(self, change):
        """Handle change in parameters."""
        # Call user callback if provided
        if self.on_change:
            self.on_change(self.get_config())
    
    def get_strategy(self):
        """Get the selected pruning strategy."""
        if hasattr(self, 'strategy_dropdown'):
            return self.strategy_dropdown.value
        return self.strategy
    
    def get_level(self):
        """Get the selected pruning level."""
        if hasattr(self, 'level_slider'):
            return self.level_slider.value
        return self.level
    
    def get_weight(self):
        """Get the selected weight for combined strategy."""
        if hasattr(self, 'weight_slider'):
            return self.weight_slider.value
        return 0.5
    
    def get_config(self):
        """Get the complete configuration."""
        config = {
            "strategy": self.get_strategy(),
            "level": self.get_level()
        }
        
        if config["strategy"] == "combined":
            config["entropy_weight"] = self.get_weight()
            config["magnitude_weight"] = 1.0 - self.get_weight()
        
        return config

class ModelSelector:
    """Widget for selecting model architecture and size."""
    
    def __init__(self, on_change: Optional[Callable] = None):
        """
        Initialize model selector.
        
        Args:
            on_change: Callback function when selection changes
        """
        self.on_change = on_change
        self.widget = None
        self.info = None
        
        # Check if widgets are available
        if not ensure_ipywidgets():
            print("WARNING: Interactive widgets not available. Using default model.")
            self.model_type = "gpt2"
            self.model_size = "small"
            self.model_path = "distilgpt2"
            return
        
        # Import widgets
        import ipywidgets as widgets
        from IPython.display import display
        
        # Define model options
        self.model_options = {
            "gpt2": {
                "description": "OpenAI GPT-2 transformer model",
                "sizes": {
                    "tiny": {"path": "distilgpt2", "params": "82M", "description": "DistilGPT-2 (82M parameters)"},
                    "small": {"path": "gpt2", "params": "124M", "description": "GPT-2 Small (124M parameters)"},
                    "medium": {"path": "gpt2-medium", "params": "355M", "description": "GPT-2 Medium (355M parameters)"},
                    "large": {"path": "gpt2-large", "params": "774M", "description": "GPT-2 Large (774M parameters)"}
                }
            },
            "bloom": {
                "description": "BigScience BLOOM transformer model",
                "sizes": {
                    "tiny": {"path": "bigscience/bloom-560m", "params": "560M", "description": "BLOOM 560M"},
                    "small": {"path": "bigscience/bloom-1b1", "params": "1.1B", "description": "BLOOM 1.1B"},
                    "medium": {"path": "bigscience/bloom-3b", "params": "3B", "description": "BLOOM 3B"},
                    "large": {"path": "bigscience/bloom-7b1", "params": "7.1B", "description": "BLOOM 7.1B"}
                }
            },
            "opt": {
                "description": "Facebook OPT transformer model",
                "sizes": {
                    "tiny": {"path": "facebook/opt-125m", "params": "125M", "description": "OPT 125M"},
                    "small": {"path": "facebook/opt-350m", "params": "350M", "description": "OPT 350M"},
                    "medium": {"path": "facebook/opt-1.3b", "params": "1.3B", "description": "OPT 1.3B"},
                    "large": {"path": "facebook/opt-2.7b", "params": "2.7B", "description": "OPT 2.7B"}
                }
            }
        }
        
        # Create model type dropdown
        self.model_type_dropdown = widgets.Dropdown(
            options=[(name, name) for name in self.model_options.keys()],
            description='Model Architecture:',
            value="gpt2",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        # Create model size dropdown
        size_options = [(size_info["description"], size) 
                        for size, size_info in self.model_options["gpt2"]["sizes"].items()]
        
        self.model_size_dropdown = widgets.Dropdown(
            options=size_options,
            description='Model Size:',
            value="small",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        # Create info area
        self.info = widgets.HTML(
            value=self._get_model_info("gpt2", "small"),
            layout=widgets.Layout(width='500px', margin='10px 0')
        )
        
        # Register callbacks
        self.model_type_dropdown.observe(self._on_model_type_change, names='value')
        self.model_size_dropdown.observe(self._on_model_size_change, names='value')
        
        # Create warning for large models
        self.warning = widgets.HTML(
            value="",
            layout=widgets.Layout(width='500px', margin='10px 0')
        )
        self._update_warning("small")
        
        # Create main widget container
        self.widget = widgets.VBox([
            widgets.HBox([
                widgets.VBox([
                    self.model_type_dropdown,
                    self.model_size_dropdown
                ]),
                widgets.VBox([
                    self.info,
                    self.warning
                ])
            ]),
            widgets.HTML(value="<hr style='margin: 10px 0'>")
        ])
        
        # Display the widget
        display(self.widget)
    
    def _get_model_info(self, model_type, model_size):
        """Get HTML info for a model."""
        model_data = self.model_options[model_type]
        size_data = model_data["sizes"][model_size]
        
        return f"""
        <p><b>{model_type.upper()}</b>: {model_data["description"]}</p>
        <ul>
            <li><b>Size</b>: {size_data["params"]} parameters</li>
            <li><b>HuggingFace path</b>: <code>{size_data["path"]}</code></li>
        </ul>
        """
    
    def _update_warning(self, model_size):
        """Update warning based on model size."""
        if model_size in ["medium", "large"]:
            self.warning.value = f"""
            <div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                <b>⚠️ Warning:</b> {model_size.capitalize()} models require significant memory and may cause Colab to crash.
                Consider using a smaller model or enabling GPU runtime.
            </div>
            """
        else:
            self.warning.value = ""
    
    def _on_model_type_change(self, change):
        """Handle change in model type."""
        model_type = change['new']
        
        # Update size dropdown options
        size_options = [(size_info["description"], size) 
                         for size, size_info in self.model_options[model_type]["sizes"].items()]
        
        # Save current size if possible
        current_size = self.model_size_dropdown.value
        if current_size not in [size for _, size in size_options]:
            current_size = "small"  # Default to small if current size not available
        
        # Update dropdown
        self.model_size_dropdown.options = size_options
        self.model_size_dropdown.value = current_size
        
        # Update info
        self.info.value = self._get_model_info(model_type, current_size)
        
        # Call user callback if provided
        if self.on_change:
            self.on_change(self.get_config())
    
    def _on_model_size_change(self, change):
        """Handle change in model size."""
        model_size = change['new']
        model_type = self.model_type_dropdown.value
        
        # Update info
        self.info.value = self._get_model_info(model_type, model_size)
        
        # Update warning
        self._update_warning(model_size)
        
        # Call user callback if provided
        if self.on_change:
            self.on_change(self.get_config())
    
    def get_model_type(self):
        """Get the selected model type."""
        if hasattr(self, 'model_type_dropdown'):
            return self.model_type_dropdown.value
        return self.model_type
    
    def get_model_size(self):
        """Get the selected model size."""
        if hasattr(self, 'model_size_dropdown'):
            return self.model_size_dropdown.value
        return self.model_size
    
    def get_model_path(self):
        """Get the HuggingFace path for the selected model."""
        if hasattr(self, 'model_type_dropdown') and hasattr(self, 'model_size_dropdown'):
            model_type = self.model_type_dropdown.value
            model_size = self.model_size_dropdown.value
            return self.model_options[model_type]["sizes"][model_size]["path"]
        return self.model_path
    
    def get_config(self):
        """Get the complete configuration."""
        return {
            "model_type": self.get_model_type(),
            "model_size": self.get_model_size(),
            "model_path": self.get_model_path()
        }

def create_experiment_config_widget(on_change: Optional[Callable] = None):
    """
    Create a complete experiment configuration widget.
    
    Args:
        on_change: Callback function when configuration changes
        
    Returns:
        Dictionary with the model and pruning widgets
    """
    # Check if widgets are available
    if not ensure_ipywidgets():
        print("WARNING: Interactive widgets not available. Using default configuration.")
        return {
            "model_path": "distilgpt2",
            "pruning_strategy": "entropy",
            "pruning_level": 0.2
        }
    
    # Import widgets
    import ipywidgets as widgets
    from IPython.display import display
    
    # Create configuration holder
    config = {
        "model_path": "distilgpt2",
        "pruning_strategy": "entropy",
        "pruning_level": 0.2
    }
    
    # Define update callback
    def update_config(new_values):
        config.update(new_values)
        if on_change:
            on_change(config)
    
    # Create widgets
    print("📋 Configure Neural Plasticity Experiment:")
    
    # Create model selector
    model_selector = ModelSelector(on_change=lambda model_config: update_config({
        "model_path": model_config["model_path"]
    }))
    
    # Create pruning strategy selector
    pruning_selector = PruningStrategySelector(on_change=lambda pruning_config: update_config({
        "pruning_strategy": pruning_config["strategy"],
        "pruning_level": pruning_config["level"]
    }))
    
    # Create run button
    run_button = widgets.Button(
        description='🚀 Run Experiment',
        button_style='success',
        tooltip='Start the neural plasticity experiment',
        icon='rocket'
    )
    
    display(run_button)
    
    # Return widgets and config
    return {
        "config": config,
        "model_selector": model_selector,
        "pruning_selector": pruning_selector,
        "run_button": run_button
    }