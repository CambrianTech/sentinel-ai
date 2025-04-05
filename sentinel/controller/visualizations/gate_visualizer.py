import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class GateVisualizer:
    """
    Visualization tool for analyzing and visualizing gate activity in the
    Adaptive Transformer model.
    
    This class provides methods to:
    - Create heatmaps of gate values
    - Track gate evolution during training
    - Compare gate patterns across different models
    - Visualize attention patterns of pruned vs. active heads
    """
    
    def __init__(self, output_dir="visualizations"):
        """
        Initialize the visualizer with an output directory.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Custom colormap for gates (from blue for 0 to red for 1)
        colors = [(0, 0, 0.8), (0.8, 0.8, 0.8), (0.8, 0, 0)]
        self.gate_cmap = LinearSegmentedColormap.from_list("gate_cmap", colors, N=100)
    
    def visualize_gate_heatmap(self, model, title=None, save_path=None):
        """
        Create a heatmap visualization of gate values.
        
        Args:
            model: The adaptive transformer model
            title: Optional title for the plot
            save_path: Path to save the visualization (if None, will be automatically generated)
        
        Returns:
            Path to the saved visualization
        """
        if not hasattr(model, "blocks"):
            raise ValueError("Model does not have blocks attribute (not an adaptive model)")
        
        # Extract gate values
        num_layers = len(model.blocks)
        num_heads = model.blocks[0]["attn"].num_heads
        
        # Create gate value matrix
        gate_values = np.zeros((num_layers, num_heads))
        
        for layer_idx, block in enumerate(model.blocks):
            for head_idx in range(num_heads):
                gate_values[layer_idx, head_idx] = block["attn"].gate[head_idx].item()
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            gate_values, 
            annot=True, 
            fmt=".2f", 
            cmap=self.gate_cmap,
            vmin=0.0, 
            vmax=1.0,
            linewidths=0.5
        )
        
        # Set labels and title
        plt.xlabel("Attention Head")
        plt.ylabel("Layer")
        if title:
            plt.title(title)
        else:
            plt.title("Gate Values Heatmap")
        
        # Set tick labels
        plt.yticks(np.arange(num_layers) + 0.5, np.arange(num_layers))
        plt.xticks(np.arange(num_heads) + 0.5, np.arange(num_heads))
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "gate_heatmap.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def visualize_gate_history(self, gate_history, title=None, save_path=None):
        """
        Visualize the evolution of gate values over time.
        
        Args:
            gate_history: List of gate activity dictionaries from controller
            title: Optional title for the plot
            save_path: Path to save the visualization (if None, will be automatically generated)
        
        Returns:
            Path to the saved visualization
        """
        if not gate_history:
            raise ValueError("Gate history is empty")
        
        # Count active heads per layer over time
        num_layers = len(gate_history[0])
        time_steps = len(gate_history)
        
        # Create matrix of active head counts [time, layer]
        active_heads = np.zeros((time_steps, num_layers))
        
        for t, gate_activity in enumerate(gate_history):
            for layer_idx, heads in gate_activity.items():
                active_heads[t, layer_idx] = len(heads)
        
        # Plot active heads over time
        plt.figure(figsize=(12, 6))
        
        # Line plot for each layer
        for layer_idx in range(num_layers):
            plt.plot(
                active_heads[:, layer_idx], 
                label=f"Layer {layer_idx}",
                marker='o', 
                markersize=4, 
                alpha=0.7
            )
        
        # Plot overall pruning level
        total_heads_per_layer = max([len(heads) for activity in gate_history for heads in activity.values()])
        plt.axhline(y=total_heads_per_layer, color='gray', linestyle='--', label="Max heads")
        
        # Add labels and title
        plt.xlabel("Training Step")
        plt.ylabel("Active Attention Heads")
        if title:
            plt.title(title)
        else:
            plt.title("Gate Activity Evolution During Training")
        
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid(alpha=0.3)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "gate_history.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def visualize_attention_patterns(self, model, tokenizer, prompt, layer_idx, head_idx, 
                               save_path=None, show_gates=True):
        """
        Visualize attention patterns for specific heads.
        
        Args:
            model: The adaptive transformer model
            tokenizer: The tokenizer
            prompt: Text prompt to use for attention visualization
            layer_idx: Layer index to visualize
            head_idx: Head index to visualize (can be a list)
            save_path: Path to save the visualization
            show_gates: Whether to show gate values in the title
        
        Returns:
            Path to the saved visualization
        """
        if not hasattr(model, "blocks"):
            raise ValueError("Model does not have blocks attribute (not an adaptive model)")
        
        # Convert to list if single head index
        if isinstance(head_idx, int):
            head_idx = [head_idx]
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Get attention patterns
        with torch.no_grad():
            # Forward pass
            outputs = model(inputs.input_ids)
            
            # Check if attention weights were captured
            if not hasattr(model.blocks[layer_idx]["attn"], "attention_weights"):
                raise ValueError("Attention weights not captured. Run a forward pass with storing_weights=True")
            
            # Get attention weights for selected heads
            attention_weights = model.blocks[layer_idx]["attn"].attention_weights
            
            # Get token ids and decode them
            token_ids = inputs.input_ids[0].tolist()
            tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
            
            # Create figure
            num_heads = len(head_idx)
            fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
            if num_heads == 1:
                axes = [axes]
            
            # Plot attention patterns for each head
            for i, h_idx in enumerate(head_idx):
                ax = axes[i]
                
                # Get gate value
                gate_value = model.blocks[layer_idx]["attn"].gate[h_idx].item()
                
                # Get attention weights for this head
                if h_idx in attention_weights:
                    attn = attention_weights[h_idx][0].cpu().numpy()
                else:
                    # If head was pruned, show zeros
                    attn = np.zeros((len(tokens), len(tokens)))
                
                # Create heatmap
                sns.heatmap(
                    attn, 
                    ax=ax,
                    cmap="viridis", 
                    xticklabels=tokens, 
                    yticklabels=tokens
                )
                
                # Set title
                if show_gates:
                    ax.set_title(f"Layer {layer_idx}, Head {h_idx}\nGate Value: {gate_value:.2f}")
                else:
                    ax.set_title(f"Layer {layer_idx}, Head {h_idx}")
                
                # Rotate x labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            
            # Save figure
            if save_path is None:
                head_str = "-".join(map(str, head_idx))
                save_path = os.path.join(
                    self.output_dir, 
                    f"attention_l{layer_idx}_h{head_str}.png"
                )
            
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        
        return save_path
    
    def visualize_pruning_impact(self, loss_impact_data, save_path=None):
        """
        Visualize the impact of pruning on model performance.
        
        Args:
            loss_impact_data: Dictionary mapping pruned head percentages to loss deltas
            save_path: Path to save the visualization
        
        Returns:
            Path to the saved visualization
        """
        # Sort data by pruning percentage
        pruning_pcts = sorted(loss_impact_data.keys())
        loss_deltas = [loss_impact_data[pct] for pct in pruning_pcts]
        
        plt.figure(figsize=(10, 6))
        plt.plot(pruning_pcts, loss_deltas, 'o-', linewidth=2, markersize=8)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Highlight regions
        plt.axvspan(0, 30, alpha=0.2, color='green', label="Safe pruning")
        plt.axvspan(30, 70, alpha=0.2, color='yellow', label="Moderate impact")
        plt.axvspan(70, 100, alpha=0.2, color='red', label="Severe impact")
        
        # Add labels and title
        plt.xlabel("Pruned Heads (%)")
        plt.ylabel("Loss Increase (%)")
        plt.title("Impact of Head Pruning on Model Performance")
        
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "pruning_impact.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def create_layer_importance_chart(self, model, metrics, save_path=None):
        """
        Create a visualization showing the importance of each layer based on metrics.
        
        Args:
            model: The adaptive transformer model
            metrics: Dictionary with metrics by layer (keys: 'entropy', 'importance', etc.)
            save_path: Path to save the visualization
        
        Returns:
            Path to the saved visualization
        """
        if not hasattr(model, "blocks"):
            raise ValueError("Model does not have blocks attribute (not an adaptive model)")
        
        num_layers = len(model.blocks)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(
            len(metrics), 1, 
            figsize=(12, 3 * len(metrics)), 
            sharex=True
        )
        
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]
            
            # Ensure values is a list or array of the right length
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            
            if len(values) != num_layers:
                raise ValueError(f"Metric '{metric_name}' has {len(values)} values, but model has {num_layers} layers")
            
            # Plot bar chart
            ax.bar(range(num_layers), values, alpha=0.7)
            
            # Add labels and title
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f"{metric_name.capitalize()} by Layer")
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
        
        # Set common x-axis label
        plt.xlabel("Layer")
        plt.xticks(range(num_layers))
        
        # Add overall title
        plt.suptitle("Layer-wise Metrics Analysis", fontsize=16)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "layer_importance.png")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return save_path