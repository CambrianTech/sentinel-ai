"""
Agency Visualization Utilities

This module provides tools for visualizing attention head agency states,
gate values, learning rates, and state transitions over time.

These visualizations help understand the dynamics of agency-aware attention
and how it interacts with the controller system.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches


class AgencyVisualizer:
    """
    Visualization dashboard for agency states, gate values, and learning rates.
    
    This class creates a unified visualization showing:
    - Active/withdrawn state
    - Gate value
    - Learning rate
    - Specialization role
    - State transitions over time
    """
    
    def __init__(self, model, controller_manager=None, head_lr_manager=None):
        """
        Initialize the visualizer with model and optional managers.
        
        Args:
            model: The adaptive transformer model
            controller_manager: Optional controller manager
            head_lr_manager: Optional head learning rate manager
        """
        self.model = model
        self.controller_manager = controller_manager
        self.head_lr_manager = head_lr_manager
        
        # Configure figure and plotting parameters
        plt.style.use('ggplot')
        
        # Define color maps for different visualizations
        self.state_cmap = {
            "active": "#4CAF50",       # Green
            "overloaded": "#FF9800",   # Orange
            "misaligned": "#2196F3",   # Blue
            "withdrawn": "#F44336"     # Red
        }
        
        # Create custom colormap for gate values
        self.gate_cmap = LinearSegmentedColormap.from_list(
            'gate_colormap', ['#FFFFFF', '#8BC34A', '#4CAF50', '#2E7D32']
        )
        
        # Create custom colormap for learning rates
        self.lr_cmap = LinearSegmentedColormap.from_list(
            'lr_colormap', ['#FFFFFF', '#FFC107', '#FF9800', '#FF5722']
        )
        
        # Initialize history tracking
        self.history = {
            "states": [],
            "gates": [],
            "learning_rates": [],
            "timestamps": []
        }
    
    def get_agency_data(self):
        """
        Extract current agency data from the model.
        
        Returns:
            Dictionary with agency state information
        """
        num_layers = len(self.model.blocks)
        num_heads = self.model.blocks[0]["attn"].num_heads
        
        # Initialize data arrays
        states = np.full((num_layers, num_heads), "active", dtype=object)
        consents = np.ones((num_layers, num_heads), dtype=bool)
        utilizations = np.zeros((num_layers, num_heads))
        gate_values = np.zeros((num_layers, num_heads))
        lr_multipliers = np.ones((num_layers, num_heads))
        
        # Extract data from model
        for layer_idx in range(num_layers):
            attn = self.model.blocks[layer_idx]["attn"]
            
            # Get gate values
            gate_values[layer_idx] = attn.gate.detach().cpu().numpy()
            
            # Get agency signals if available
            if hasattr(attn, "agency_signals"):
                for head_idx in range(num_heads):
                    if head_idx in attn.agency_signals:
                        signals = attn.agency_signals[head_idx]
                        states[layer_idx, head_idx] = signals.get("state", "active")
                        consents[layer_idx, head_idx] = signals.get("consent", True)
                        utilizations[layer_idx, head_idx] = signals.get("utilization", 0.0)
        
        # Get learning rate multipliers if available
        if self.head_lr_manager is not None:
            lr_multipliers = self.head_lr_manager.get_lr_multipliers()
        
        return {
            "states": states,
            "consents": consents,
            "utilizations": utilizations,
            "gate_values": gate_values,
            "lr_multipliers": lr_multipliers
        }
    
    def update_history(self, timestamp=None):
        """
        Update history with current model state.
        
        Args:
            timestamp: Optional timestamp for the history entry
        """
        # Get current data
        data = self.get_agency_data()
        
        # Add to history
        self.history["states"].append(data["states"])
        self.history["gates"].append(data["gate_values"])
        self.history["learning_rates"].append(data["lr_multipliers"])
        self.history["timestamps"].append(timestamp or len(self.history["states"]))
    
    def plot_current_state(self, figsize=(14, 10)):
        """
        Plot current agency state visualization.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            matplotlib figure
        """
        data = self.get_agency_data()
        num_layers, num_heads = data["states"].shape
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 3)
        
        # Create axes for different visualizations
        ax_states = fig.add_subplot(gs[0, :])
        ax_gates = fig.add_subplot(gs[1, :])
        ax_lr = fig.add_subplot(gs[2, :])
        ax_history = fig.add_subplot(gs[3, :])
        
        # Plot head states
        self._plot_states(ax_states, data["states"])
        
        # Plot gate values
        self._plot_gate_values(ax_gates, data["gate_values"])
        
        # Plot learning rates
        self._plot_learning_rates(ax_lr, data["lr_multipliers"])
        
        # Plot state history if available
        if len(self.history["states"]) > 1:
            self._plot_state_history(ax_history)
        else:
            ax_history.text(0.5, 0.5, "State history not available yet",
                          ha='center', va='center', fontsize=12)
            ax_history.set_axis_off()
        
        # Set title and layout
        fig.suptitle("Attention Head Agency Dashboard", fontsize=16)
        fig.tight_layout()
        return fig
    
    def _plot_states(self, ax, states):
        """Plot head states as a grid with color-coded states."""
        num_layers, num_heads = states.shape
        
        # Create a numerical representation for states
        state_values = {
            "active": 3,
            "misaligned": 2,
            "overloaded": 1,
            "withdrawn": 0
        }
        
        state_array = np.zeros((num_layers, num_heads))
        for i in range(num_layers):
            for j in range(num_heads):
                state_array[i, j] = state_values.get(states[i, j], 3)
        
        # Plot as a heatmap
        im = ax.imshow(state_array, cmap='viridis', aspect='auto')
        
        # Configure axes
        ax.set_title("Attention Head States")
        ax.set_ylabel("Layer")
        ax.set_xlabel("Head")
        ax.set_yticks(np.arange(num_layers))
        ax.set_xticks(np.arange(num_heads))
        
        # Add state labels
        for i in range(num_layers):
            for j in range(num_heads):
                state = states[i, j]
                state_abbr = state[:2].upper()
                ax.text(j, i, state_abbr, ha="center", va="center", 
                       color="white" if state_array[i, j] < 2 else "black", fontsize=8)
        
        # Add a legend
        patches = [
            mpatches.Patch(color='#2E7D32', label='Active'),
            mpatches.Patch(color='#2196F3', label='Misaligned'),
            mpatches.Patch(color='#FF9800', label='Overloaded'),
            mpatches.Patch(color='#F44336', label='Withdrawn')
        ]
        ax.legend(handles=patches, loc='upper right', ncol=4)
    
    def _plot_gate_values(self, ax, gate_values):
        """Plot gate values as a heatmap."""
        # Plot as a heatmap
        im = ax.imshow(gate_values, cmap=self.gate_cmap, aspect='auto', vmin=0, vmax=1)
        
        # Configure axes
        ax.set_title("Gate Values")
        ax.set_ylabel("Layer")
        ax.set_xlabel("Head")
        ax.set_yticks(np.arange(gate_values.shape[0]))
        ax.set_xticks(np.arange(gate_values.shape[1]))
        
        # Add gate value labels
        for i in range(gate_values.shape[0]):
            for j in range(gate_values.shape[1]):
                value = gate_values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                       color="black" if value > 0.5 else "black", fontsize=8)
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Gate Value')
    
    def _plot_learning_rates(self, ax, lr_multipliers):
        """Plot learning rate multipliers as a heatmap."""
        # Plot as a heatmap
        im = ax.imshow(lr_multipliers, cmap=self.lr_cmap, aspect='auto', vmin=0, vmax=5)
        
        # Configure axes
        ax.set_title("Learning Rate Multipliers")
        ax.set_ylabel("Layer")
        ax.set_xlabel("Head")
        ax.set_yticks(np.arange(lr_multipliers.shape[0]))
        ax.set_xticks(np.arange(lr_multipliers.shape[1]))
        
        # Add lr multiplier labels
        for i in range(lr_multipliers.shape[0]):
            for j in range(lr_multipliers.shape[1]):
                value = lr_multipliers[i, j]
                # Skip zeroes (withdrawn)
                if value < 0.01:
                    text = "0"
                else:
                    text = f"{value:.1f}"
                ax.text(j, i, text, ha="center", va="center", 
                       color="black", fontsize=8)
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('LR Multiplier')
    
    def _plot_state_history(self, ax):
        """Plot state changes over time."""
        # Convert state history to numerical values for plotting
        state_values = {
            "active": 3,
            "misaligned": 2,
            "overloaded": 1,
            "withdrawn": 0
        }
        
        # Count states over time
        timestamps = self.history["timestamps"]
        active_counts = []
        overloaded_counts = []
        misaligned_counts = []
        withdrawn_counts = []
        
        for states in self.history["states"]:
            # Flatten and count
            flat_states = states.flatten()
            active_counts.append(np.sum(flat_states == "active"))
            overloaded_counts.append(np.sum(flat_states == "overloaded"))
            misaligned_counts.append(np.sum(flat_states == "misaligned"))
            withdrawn_counts.append(np.sum(flat_states == "withdrawn"))
        
        # Plot as a stacked area chart
        ax.stackplot(timestamps, 
                    [active_counts, misaligned_counts, overloaded_counts, withdrawn_counts],
                    labels=['Active', 'Misaligned', 'Overloaded', 'Withdrawn'],
                    colors=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
        
        # Configure axes
        ax.set_title("Head State Distribution Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Head Count")
        ax.legend(loc='upper right', ncol=4)
        
        # Set integer ticks for x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    def plot_state_transitions(self, figsize=(10, 8)):
        """
        Plot head state transitions over time.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            matplotlib figure
        """
        if len(self.history["states"]) < 2:
            # Not enough history for transitions
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "State transition history not available yet",
                  ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        # Count transitions between states
        transitions = {
            "active_to_overloaded": [],
            "active_to_misaligned": [],
            "active_to_withdrawn": [],
            "overloaded_to_active": [],
            "misaligned_to_active": [],
            "withdrawn_to_active": []
        }
        
        # Analyze transitions between consecutive time steps
        for i in range(1, len(self.history["states"])):
            prev_states = self.history["states"][i-1]
            curr_states = self.history["states"][i]
            
            # Compare state changes
            for transition_type, count in [
                ("active_to_overloaded", np.sum((prev_states == "active") & (curr_states == "overloaded"))),
                ("active_to_misaligned", np.sum((prev_states == "active") & (curr_states == "misaligned"))),
                ("active_to_withdrawn", np.sum((prev_states == "active") & (curr_states == "withdrawn"))),
                ("overloaded_to_active", np.sum((prev_states == "overloaded") & (curr_states == "active"))),
                ("misaligned_to_active", np.sum((prev_states == "misaligned") & (curr_states == "active"))),
                ("withdrawn_to_active", np.sum((prev_states == "withdrawn") & (curr_states == "active")))
            ]:
                transitions[transition_type].append(count)
        
        # Create a figure for transition plots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Plot transitions
        timestamps = self.history["timestamps"][1:]
        for i, (transition_type, counts) in enumerate(transitions.items()):
            ax = axes[i]
            ax.plot(timestamps, counts, marker='o')
            ax.set_title(transition_type.replace('_', ' → ').title())
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Transition Count")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set overall title and layout
        fig.suptitle("Head State Transitions Over Time", fontsize=16)
        fig.tight_layout()
        return fig
    
    def create_dashboard(self, save_path=None):
        """
        Create a comprehensive dashboard with current state and history.
        
        Args:
            save_path: Optional path to save the dashboard
            
        Returns:
            Dictionary of created figures
        """
        # Create current state plot
        current_fig = self.plot_current_state()
        
        # Create transition plot if history is available
        if len(self.history["states"]) > 1:
            transition_fig = self.plot_state_transitions()
        else:
            transition_fig = None
        
        # Save figures if path is provided
        if save_path:
            current_fig.savefig(f"{save_path}/current_state.png", dpi=150)
            if transition_fig:
                transition_fig.savefig(f"{save_path}/state_transitions.png", dpi=150)
        
        return {
            "current_state": current_fig,
            "state_transitions": transition_fig
        }


# Utility function for quick visualization
def visualize_agency(model, controller_manager=None, head_lr_manager=None, history_length=10):
    """
    Create and display agency visualization for a model.
    
    Args:
        model: The adaptive transformer model
        controller_manager: Optional controller manager
        head_lr_manager: Optional head learning rate manager
        history_length: Number of history entries to simulate
        
    Returns:
        AgencyVisualizer instance
    """
    visualizer = AgencyVisualizer(model, controller_manager, head_lr_manager)
    
    # Add some history entries for demonstration
    for i in range(history_length):
        visualizer.update_history(i)
    
    # Display the dashboard
    visualizer.create_dashboard()
    plt.show()
    
    return visualizer