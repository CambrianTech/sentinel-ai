"""
Neural Plasticity Visualization

This module provides visualization utilities for neural plasticity experiments.
It visualizes head entropy, gradients, pruning decisions, training metrics,
and attention patterns.

Version: v0.0.64 (2025-04-20 23:15:00)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from utils.colab.helpers import safe_tensor_imshow, get_colab_type

# Initialize environment variables
IS_APPLE_SILICON = False
IS_COLAB = False
HAS_GPU = False

# Detect if we're running in Google Colab
try:
    import google.colab
    IS_COLAB = True
    print("🌐 Running in Google Colab environment")
    
    # Check for GPU in Colab
    colab_info = get_colab_type(verbose=False)
    if colab_info.get("hardware") == "GPU":
        HAS_GPU = True
        print(f"✅ CUDA GPU detected in Colab: {colab_info.get('gpu_name', 'Unknown GPU')}")
        print("🚀 Visualizations will be optimized for GPU acceleration")
        
        # Display GPU info for confirmation
        try:
            import subprocess
            subprocess.run(["nvidia-smi"], check=False)
        except Exception:
            pass
    else:
        print("⚠️ No GPU detected in Colab - using CPU for visualizations")
except (ImportError, ModuleNotFoundError):
    pass

# Detect Apple Silicon and apply optimizations if needed
try:
    if platform.system() == "Darwin" and platform.processor() == "arm":
        IS_APPLE_SILICON = True
        print("🍎 Apple Silicon detected - enabling visualization crash prevention")
        
        # Skip Apple Silicon optimizations if running in Colab (shouldn't happen, but just in case)
        if not IS_COLAB:
            # Force single-threaded image processing
            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            
            # Switch matplotlib backend to Agg (non-interactive) on Apple Silicon
            # This helps prevent some rendering issues
            try:
                import matplotlib
                matplotlib.use('Agg')
                print("🎨 Switching to Agg matplotlib backend for improved stability")
            except (ImportError, RuntimeError):
                pass
                
            # Configure PyTorch for Apple Silicon if available
            try:
                import torch
                # Disable parallel CPU operations
                torch.set_num_threads(1)
            except (ImportError, AttributeError):
                pass
except (ImportError, AttributeError):
    pass


def visualize_head_entropy(
    entropy_values: torch.Tensor,
    title: str = "Attention Entropy Heatmap",
    min_value: float = 0.0,
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize entropy values across all layers and heads as a heatmap.
    
    Args:
        entropy_values: Tensor of entropy values with shape [layers, heads]
                        or dictionary of layers to entropy tensors
        title: Title for the plot
        min_value: Minimum value for the colormap scale
        annotate: Whether to add value annotations to the cells
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Handle dictionary of layers to entropy values
    if isinstance(entropy_values, dict):
        # Convert dictionary to tensor [layers, heads]
        layers = sorted(list(entropy_values.keys()))
        if len(layers) > 0:
            # Check if the first value is a tensor with multiple dimensions
            first_val = entropy_values[layers[0]]
            if isinstance(first_val, torch.Tensor) and first_val.dim() > 1:
                # Take mean across additional dimensions
                entropy_stacked = []
                for layer in layers:
                    layer_entropy = entropy_values[layer]
                    if layer_entropy.dim() > 1:
                        # Reduce to 1D by taking mean across extra dimensions
                        layer_entropy = layer_entropy.mean(dim=tuple(range(1, layer_entropy.dim())))
                    entropy_stacked.append(layer_entropy)
                entropy_values = torch.stack(entropy_stacked)
            else:
                # Standard case, just stack the tensors
                entropy_values = torch.stack([entropy_values[layer] for layer in layers])
    
    # Handle tensors with extra dimensions
    if isinstance(entropy_values, torch.Tensor) and entropy_values.dim() > 2:
        # Reduce extra dimensions by taking the mean
        # Keep only the first two dimensions [layers, heads]
        entropy_values = entropy_values.mean(dim=tuple(range(2, entropy_values.dim())))
    
    # Convert to numpy if tensor
    if isinstance(entropy_values, torch.Tensor):
        entropy_data = entropy_values.detach().cpu().numpy()
    else:
        entropy_data = entropy_values
    
    # Create figure (only once)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap using imshow directly since entropy_data is already numpy
    im = ax.imshow(entropy_data, cmap=cmap)
    ax.set_title(title)
    
    # Set proper colormap limits with non-zero range
    im.set_clim(min_value, max(0.1, np.max(entropy_data)))
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Entropy')
    
    # Add labels
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text annotations if requested
    if annotate:
        for i in range(entropy_data.shape[0]):
            for j in range(entropy_data.shape[1]):
                ax.text(j, i, f'{entropy_data[i, j]:.2f}',
                        ha="center", va="center", color="w")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_head_gradients(
    grad_norm_values: torch.Tensor,
    pruned_heads: Optional[List[Tuple[int, int]]] = None,
    revived_heads: Optional[List[Tuple[int, int]]] = None,
    title: str = "Gradient Norms",
    figsize: Tuple[int, int] = (10, 5),
    cmap: str = "plasma",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization of gradient norms with markers for pruned/revived heads.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        pruned_heads: List of (layer, head) tuples for pruned heads
        revived_heads: List of (layer, head) tuples for revived heads
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(grad_norm_values, torch.Tensor):
        grad_data = grad_norm_values.detach().cpu().numpy()
    else:
        grad_data = grad_norm_values
        
    # Create figure (only once)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap directly
    im = ax.imshow(grad_data, cmap=cmap)
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label="Gradient Norm")
    
    # Mark pruned heads with 'P'
    if pruned_heads:
        for layer, head in pruned_heads:
            ax.text(head, layer, "P", ha="center", va="center",
                   color="white", weight="bold", bbox=dict(facecolor='red', alpha=0.5))
    
    # Mark revived heads with 'R'
    if revived_heads:
        for layer, head in revived_heads:
            ax.text(head, layer, "R", ha="center", va="center",
                   color="white", weight="bold", bbox=dict(facecolor='green', alpha=0.5))
    
    # Set labels
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_pruning_decisions(
    grad_norm_values: torch.Tensor,
    pruning_mask: torch.Tensor,
    title: str = "Pruning Decisions",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization highlighting pruning decisions.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        pruning_mask: Boolean tensor where True indicates a head should be pruned
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy arrays
    if isinstance(grad_norm_values, torch.Tensor):
        grad_data = grad_norm_values.detach().cpu().numpy()
    else:
        grad_data = grad_norm_values
        
    if isinstance(pruning_mask, torch.Tensor):
        mask_data = pruning_mask.detach().cpu().numpy()
    else:
        mask_data = pruning_mask
    
    # Create figure (only once)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Base plot with all gradient values
    im = ax.imshow(grad_data, cmap="YlOrRd")
    ax.set_title(title)
    
    # Create a masked array where pruned heads are highlighted
    masked_grads = np.ma.array(grad_data, mask=~mask_data)
    
    # Overlay plot with pruned heads highlighted
    plt.imshow(
        masked_grads, 
        cmap='Reds', 
        alpha=0.7,
        aspect='auto'
    )
    
    # Add colorbar
    plt.colorbar(im, label='Gradient Norm')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_training_metrics(
    metrics_history: Dict[str, List[float]],
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize training metrics over time.
    
    Args:
        metrics_history: Dictionary with metrics history
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Extract step information
    steps = metrics_history.get("step", list(range(len(next(iter(metrics_history.values()))))))
    
    # Create subplot layout based on available metrics
    num_plots = 0
    if any(m in metrics_history for m in ["train_loss", "eval_loss"]):
        num_plots += 1
    if "perplexity" in metrics_history:
        num_plots += 1
    if any(m in metrics_history for m in ["pruned_heads", "revived_heads", "sparsity"]):
        num_plots += 1
        
    num_plots = max(num_plots, 1)  # Ensure at least one plot
    
    # Create figure
    fig, axs = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    
    # Convert to list if only one subplot
    if num_plots == 1:
        axs = [axs]
    
    plot_idx = 0
    
    # Plot loss metrics
    if any(m in metrics_history for m in ["train_loss", "eval_loss"]):
        ax = axs[plot_idx]
        if "train_loss" in metrics_history:
            ax.plot(steps, metrics_history["train_loss"], label="Train Loss")
        if "eval_loss" in metrics_history:
            ax.plot(steps, metrics_history["eval_loss"], label="Eval Loss")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot perplexity
    if "perplexity" in metrics_history:
        ax = axs[plot_idx]
        ax.plot(steps, metrics_history["perplexity"], label="Perplexity")
        ax.set_ylabel("Perplexity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot pruning metrics
    if any(m in metrics_history for m in ["pruned_heads", "revived_heads", "sparsity"]):
        ax = axs[plot_idx]
        if "pruned_heads" in metrics_history:
            ax.plot(steps, metrics_history["pruned_heads"], label="Pruned Heads", color="red")
        if "revived_heads" in metrics_history:
            ax.plot(steps, metrics_history["revived_heads"], label="Revived Heads", color="green")
        if "sparsity" in metrics_history:
            # Plot on secondary axis
            ax2 = ax.twinx()
            ax2.plot(steps, metrics_history["sparsity"], label="Sparsity", color="blue", linestyle="--")
            ax2.set_ylabel("Sparsity")
            
        ax.set_ylabel("Head Count")
        ax.legend(loc="upper left")
        if "sparsity" in metrics_history:
            ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Set common x-axis label
    axs[-1].set_xlabel("Steps")
    
    # Set overall title
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


class VisualizationReporter:
    """
    Reporter for neural plasticity visualizations and metrics.
    
    This class provides methods to display, save, and summarize
    neural plasticity experiment results consistently across different
    environments and contexts.
    """
    
    def __init__(self, model=None, tokenizer=None, output_dir=None, save_visualizations=False, verbose=True):
        """
        Initialize the visualization reporter.
        
        Args:
            model: Optional model for evaluation and generation
            tokenizer: Optional tokenizer for text generation
            output_dir: Directory to save visualizations (created if it doesn't exist)
            save_visualizations: Whether to save visualizations to disk
            verbose: Whether to print detailed information
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_visualizations = save_visualizations
        self.verbose = verbose
        self.viz_paths = {}
        
        # Create output directory if needed
        if self.save_visualizations and self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
    def display_warmup_results(self, warmup_results):
        """
        Display warmup results including visualizations and metrics.
        
        Args:
            warmup_results: Dictionary with warmup metrics from run_warmup_phase
            
        Returns:
            Dictionary with additional display information
        """
        # Extract key metrics
        warmup_losses = warmup_results.get("losses", [])
        
        # Show the visualization if available
        warmup_visualization = warmup_results.get("visualization")
        if warmup_visualization:
            plt.figure(warmup_visualization.number)
            plt.show()

        # Display segment analysis
        segment_analysis = warmup_results.get("segment_analysis", {})
        if segment_analysis and segment_analysis.get("segment_size", 0) > 0:
            print(f"\nWarm-up Segment Analysis:")
            print(f"First segment average loss: {segment_analysis['first_segment_avg']:.4f}")
            print(f"Last segment average loss: {segment_analysis['last_segment_avg']:.4f}")
            print(f"Improvement during warm-up: {segment_analysis['improvement']:.1f}%")
            print(f"Is model still significantly improving? {'Yes' if segment_analysis.get('still_improving', False) else 'No'}")

        # Print warm-up summary
        print(f"\nWarm-up completed with {len(warmup_losses)} steps across {len(warmup_results.get('epochs', []))}")
        print(f"Initial loss: {warmup_results.get('initial_loss', 0):.4f}")
        print(f"Final loss: {warmup_results.get('final_loss', 0):.4f}")
        
        # Print improvement percentage if available
        if 'improvement_percent' in warmup_results:
            print(f"Overall loss reduction: {warmup_results['improvement_percent']:.1f}%")

        # Display saved visualization paths if available
        if "visualization_paths" in warmup_results and warmup_results["visualization_paths"]:
            print("\nSaved visualizations:")
            for name, path in warmup_results["visualization_paths"].items():
                print(f"- {name}: {path}")
                
        return warmup_results
    
    def display_pruning_results(self, pruning_results):
        """
        Display pruning results including visualizations and metrics.
        
        Args:
            pruning_results: Dictionary with pruning metrics from run_pruning_cycle
            
        Returns:
            Dictionary with additional display information
        """
        # Extract key metrics
        baseline_metrics = pruning_results.get("baseline_metrics", {})
        pruned_metrics = pruning_results.get("pruned_metrics", {})
        final_metrics = pruning_results.get("final_metrics", {})
        
        # Print baseline, pruned and final metrics
        print("\nMetrics Comparison:")
        print(f"  Baseline:  Loss = {baseline_metrics.get('loss', 0):.4f}, Perplexity = {baseline_metrics.get('perplexity', 0):.2f}")
        print(f"  After Pruning: Loss = {pruned_metrics.get('loss', 0):.4f}, Perplexity = {pruned_metrics.get('perplexity', 0):.2f}")
        print(f"  After Training: Loss = {final_metrics.get('loss', 0):.4f}, Perplexity = {final_metrics.get('perplexity', 0):.2f}")
        
        # Print improvement metrics
        if 'perplexity_improvement' in pruning_results:
            print(f"\nPerplexity improvement: {pruning_results['perplexity_improvement']*100:.2f}%")
            
        if 'recovery_rate' in pruning_results:
            print(f"Recovery rate: {pruning_results['recovery_rate']*100:.2f}%")
        
        # Print pruning summary
        pruned_heads = pruning_results.get("pruned_heads", [])
        print(f"\nPruning Summary:")
        print(f"  Pruned {len(pruned_heads)} out of {pruning_results.get('total_heads', 0)} heads")
        print(f"  Pruning strategy: {pruning_results.get('strategy', 'unknown')}")
        print(f"  Pruning level: {pruning_results.get('pruning_level', 0)*100:.1f}%")
        
        # Display visualizations
        training_metrics = pruning_results.get("training_metrics", {})
        if training_metrics:
            try:
                metrics_fig = visualize_training_metrics(
                    metrics_history=training_metrics,
                    title="Training After Pruning"
                )
                plt.figure(metrics_fig.number)
                plt.show()
            except Exception as e:
                if self.verbose:
                    print(f"Could not display training metrics: {e}")
        
        # Display visualization directory if available
        if "visualization_dir" in pruning_results:
            print(f"\nVisualization directory: {pruning_results['visualization_dir']}")
            
        return pruning_results
    
    def report_model_stats(self, model, sparsity=None, pruned_heads=None):
        """
        Display summary statistics for a model.
        
        Args:
            model: The transformer model
            sparsity: Optional sparsity value to display
            pruned_heads: Optional list of pruned heads to display
            
        Returns:
            Dictionary with model statistics
        """
        # Estimate model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Print model statistics
        print(f"\nModel Statistics:")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Parameters: {total_params:,}")
        
        if sparsity is not None:
            print(f"  Sparsity: {sparsity:.2%}")
            
        if pruned_heads is not None:
            print(f"  Pruned heads: {len(pruned_heads)}")
            
        return {
            "model_size_mb": model_size_mb,
            "total_params": total_params,
            "sparsity": sparsity,
            "pruned_heads_count": len(pruned_heads) if pruned_heads else 0
        }
    
    def save_figure(self, fig, name, subfolder=None):
        """
        Save a figure to disk.
        
        Args:
            fig: Matplotlib figure to save
            name: Name for the saved figure file
            subfolder: Optional subfolder within output_dir
            
        Returns:
            Path to saved figure or None if not saved
        """
        if not self.save_visualizations or not self.output_dir:
            return None
            
        try:
            import os
            
            # Create subfolder if needed
            if subfolder:
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = self.output_dir
                
            # Ensure filename has .png extension
            if not name.endswith(".png"):
                name = f"{name}.png"
                
            # Save figure
            save_path = os.path.join(save_dir, name)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
            # Store path in visualization paths
            self.viz_paths[name] = save_path
            
            if self.verbose:
                print(f"✅ Saved visualization to {save_path}")
                
            return save_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving visualization: {e}")
            return None
            
    def get_visualization_paths(self):
        """
        Get dictionary of all saved visualization paths.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        return self.viz_paths
        
    def evaluate_model(self, model=None, dataloader=None, device=None):
        """
        Evaluate model on the given dataloader.
        
        Args:
            model: Model to evaluate (defaults to self.model)
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on (defaults to model's device)
            
        Returns:
            Tuple of (loss, perplexity)
        """
        # Use model from instance if not provided
        if model is None:
            if self.model is None:
                raise ValueError("No model provided for evaluation")
            model = self.model
            
        # Import directly from core to avoid circular imports
        from .core import evaluate_model
        
        # Determine device if not provided
        if device is None:
            device = next(model.parameters()).device
            
        # Evaluate the model directly
        eval_results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device
        )
        
        # Print results if verbose
        if self.verbose:
            print(f"Evaluation: Loss = {eval_results['loss']:.4f}, Perplexity = {eval_results['perplexity']:.2f}")
            
        return eval_results["loss"], eval_results["perplexity"]
    
    def generate_text(self, prompt, model=None, tokenizer=None, max_length=100, 
                     temperature=0.7, top_k=50, top_p=0.95, device=None, 
                     save_to_file=None):
        """
        Generate text from the model.
        
        Args:
            prompt: Text prompt to generate from
            model: Model to use (defaults to self.model)
            tokenizer: Tokenizer to use (defaults to self.tokenizer)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            device: Device to run generation on
            save_to_file: Optional file path to save generated text
            
        Returns:
            Generated text string
        """
        # Use model and tokenizer from instance if not provided
        if model is None:
            if self.model is None:
                raise ValueError("No model provided for text generation")
            model = self.model
            
        if tokenizer is None:
            if self.tokenizer is None:
                raise ValueError("No tokenizer provided for text generation")
            tokenizer = self.tokenizer
            
        # Determine device if not provided
        if device is None:
            device = next(model.parameters()).device
            
        # Set model to evaluation mode
        model.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Save to file if requested
        if save_to_file:
            try:
                import os
                
                # Create parent directory if needed
                os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
                
                with open(save_to_file, 'w') as f:
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Generated text:\n{generated_text}")
                    
                if self.verbose:
                    print(f"✅ Generated text saved to {save_to_file}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error saving generated text: {e}")
        
        return generated_text
        
    def run_text_generation_suite(self, prompts, model=None, tokenizer=None, 
                                 max_length=100, save_to_file=True, subfolder="generation"):
        """
        Run a suite of text generation examples and optionally save them.
        
        Args:
            prompts: List of prompts or dictionary of {name: prompt}
            model: Model to use (defaults to self.model)
            tokenizer: Tokenizer to use (defaults to self.tokenizer)
            max_length: Maximum length of generated text
            save_to_file: Whether to save generations to files
            subfolder: Subfolder within output_dir to save generations
            
        Returns:
            Dictionary of prompt to generated text
        """
        # Convert list of prompts to dictionary if needed
        if isinstance(prompts, list):
            prompts_dict = {f"prompt_{i}": prompt for i, prompt in enumerate(prompts)}
        else:
            prompts_dict = prompts
            
        results = {}
        
        # Print header
        if self.verbose:
            print("\n=== Running Text Generation Suite ===\n")
            
        # Run generation for each prompt
        for name, prompt in prompts_dict.items():
            if self.verbose:
                print(f"Prompt: {prompt}")
                
            # Determine save path
            save_path = None
            if save_to_file and self.save_visualizations and self.output_dir:
                import os
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{name}.txt")
                
            # Generate text
            generated = self.generate_text(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                save_to_file=save_path
            )
            
            # Store result
            results[prompt] = generated
            
            # Print result
            if self.verbose:
                print(f"Generated: {generated[:200]}...")
                if len(generated) > 200:
                    print("...")
                print("-" * 40)
                
        return results
        
    def save_metrics_to_csv(self, metrics, filename="metrics.csv", subfolder=None):
        """
        Save metrics dictionary to CSV file.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Name for the CSV file
            subfolder: Optional subfolder within output_dir
            
        Returns:
            Path to saved CSV file or None if not saved
        """
        if not self.save_visualizations or not self.output_dir:
            return None
            
        try:
            import os
            import csv
            
            # Create subfolder if needed
            if subfolder:
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = self.output_dir
                
            # Ensure filename has .csv extension
            if not filename.endswith(".csv"):
                filename = f"{filename}.csv"
                
            # Save CSV
            save_path = os.path.join(save_dir, filename)
            
            # Prepare data for CSV
            # Assume each value in metrics is a list of the same length
            if not metrics:
                return None
                
            # Get list of metric names
            metric_names = list(metrics.keys())
            
            # Get length of first metric array
            first_metric = next(iter(metrics.values()))
            rows = len(first_metric) if isinstance(first_metric, list) else 1
            
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(metric_names)
                
                # Write each row of data
                if rows > 1:
                    for i in range(rows):
                        writer.writerow([metrics[name][i] if i < len(metrics[name]) else "" for name in metric_names])
                else:
                    # Just one row of data
                    writer.writerow([metrics[name] for name in metric_names])
            
            # Store path
            self.viz_paths[filename] = save_path
            
            if self.verbose:
                print(f"✅ Saved metrics to {save_path}")
                
            return save_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving metrics: {e}")
            return None


def create_pruning_state_heatmap(
    model: torch.nn.Module,
    cumulative_pruned: List[Tuple[int, int]],
    newly_pruned: List[Tuple[int, int]] = None,
    title: str = "Cumulative Pruning State",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap showing the cumulative state of pruned heads.
    
    Args:
        model: The transformer model
        cumulative_pruned: List of (layer, head) tuples of all pruned heads
        newly_pruned: List of (layer, head) tuples of newly pruned heads
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Determine model structure
    # Attempt to detect model structure directly
    num_layers = 0
    num_heads = 0
    
    # Check for HF transformer models
    if hasattr(model, 'config'):
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
            
        if hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
    
    # If we couldn't detect from config, try counting transformer blocks
    if num_layers == 0 or num_heads == 0:
        # Try to detect from model architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            num_layers = len(model.transformer.h)
            if num_layers > 0 and hasattr(model.transformer.h[0].attn, 'num_heads'):
                num_heads = model.transformer.h[0].attn.num_heads
    
    # Fallback values if detection fails
    if num_layers == 0:
        num_layers = len(cumulative_pruned) // 2 if cumulative_pruned else 6
    if num_heads == 0:
        num_heads = 12  # Common default
    
    # Create empty matrix for pruning state
    pruning_state = np.zeros((num_layers, num_heads))
    
    # Fill in cumulative pruned heads with 1
    for layer, head in cumulative_pruned:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            pruning_state[layer, head] = 1
    
    # Fill in newly pruned heads with 2 (for different color)
    if newly_pruned:
        for layer, head in newly_pruned:
            if 0 <= layer < num_layers and 0 <= head < num_heads:
                pruning_state[layer, head] = 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap to distinguish newly pruned and previously pruned
    import matplotlib.colors as mcolors
    colors = [(1,1,1), (0.8,0.2,0.2), (0.2,0.8,0.2)]  # white, red, green
    cmap_custom = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_custom.N)
    
    # Plot heatmap
    im = ax.imshow(pruning_state, cmap=cmap_custom, norm=norm)
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Active', 'Previously Pruned', 'Newly Pruned'])
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text annotations showing pruned head counts
    previously_pruned_count = len(cumulative_pruned) - len(newly_pruned) if newly_pruned else len(cumulative_pruned)
    newly_pruned_count = len(newly_pruned) if newly_pruned else 0
    total_heads = num_layers * num_heads
    
    pruned_percent = (len(cumulative_pruned) / total_heads) * 100
    ax.text(0.05, -0.15, f"Total pruned: {len(cumulative_pruned)}/{total_heads} heads ({pruned_percent:.1f}%)",
            transform=ax.transAxes, fontsize=10)
            
    if newly_pruned:
        ax.text(0.05, -0.20, f"Newly pruned this cycle: {newly_pruned_count} heads",
                transform=ax.transAxes, fontsize=10)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_attention_patterns(
    attention_maps: Union[torch.Tensor, List[torch.Tensor]],
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    num_heads: int = 4,
    save_path: Optional[str] = None,
    use_safe_tensor: bool = True
) -> plt.Figure:
    """
    Visualize attention patterns for one or more heads with cross-platform compatibility.
    
    Args:
        attention_maps: Attention tensor with shape [batch, heads, seq_len, seq_len]
                       or list of attention tensors per layer
        layer_idx: Index of layer to visualize
        head_idx: Index of attention head to visualize (if None, shows multiple heads)
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        num_heads: Number of heads to visualize if head_idx is None
        save_path: Optional path to save the visualization
        use_safe_tensor: Whether to use the safe_tensor_imshow function (recommended for
                        cross-platform compatibility)
        
    Returns:
        matplotlib Figure object
    """
    from utils.colab.helpers import safe_tensor_imshow
    
    # Handle tensor if it's a list of layers
    if isinstance(attention_maps, list):
        if layer_idx >= len(attention_maps):
            print(f"Warning: Layer {layer_idx} out of bounds (max: {len(attention_maps)-1})")
            layer_idx = min(layer_idx, len(attention_maps)-1)
        attention_maps = attention_maps[layer_idx]
    
    # Check tensor dimensions
    if isinstance(attention_maps, torch.Tensor):
        if attention_maps.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D attention tensor, got shape {attention_maps.shape}")
        
        # Handle 3D tensor (missing batch dimension)
        if attention_maps.dim() == 3:
            attention_maps = attention_maps.unsqueeze(0)
    
    # If using safe_tensor_imshow (recommended for cross-platform)
    if use_safe_tensor:
        if head_idx is not None:
            # Single head visualization
            # Extract the head attention tensor
            if isinstance(attention_maps, torch.Tensor):
                head_attention = attention_maps[0, head_idx]
            else:
                head_attention = attention_maps[0][head_idx]
                
            # Use safe_tensor_imshow for cross-platform visualization
            head_title = title or f'Attention Pattern (Layer {layer_idx}, Head {head_idx})'
            fig = safe_tensor_imshow(
                tensor=head_attention,
                title=head_title,
                cmap='viridis',
                figsize=figsize,
                save_path=save_path,
                vmin=0.0,
                vmax=1.0
            )
            
            # Access the created axis for additional customization
            ax = fig.axes[0]
            ax.set_xlabel('Sequence Position (To)')
            ax.set_ylabel('Sequence Position (From)')
            
            return fig
        else:
            # Multiple heads visualization
            heads_to_show = min(num_heads, attention_maps.shape[1])
            fig, axs = plt.subplots(1, heads_to_show, figsize=figsize)
            
            # Adjust title
            main_title = title or f'Attention Patterns (Layer {layer_idx})'
            fig.suptitle(main_title, fontsize=14)
            
            # Ensure axs is always an array even with single subplot
            if heads_to_show == 1:
                axs = [axs]
            
            # Plot each head
            for i in range(heads_to_show):
                # Extract head attention tensor
                head_attention = attention_maps[0, i]
                
                # Use numpy-safe processing that works across platforms
                if isinstance(head_attention, torch.Tensor):
                    # Process safely
                    try:
                        if head_attention.requires_grad:
                            head_attention = head_attention.detach()
                        if head_attention.is_cuda:
                            head_attention = head_attention.cpu()
                        head_data = head_attention.numpy()
                    except Exception:
                        # Fallback for any conversion issues
                        head_data = np.zeros((10, 10))  # Empty placeholder
                else:
                    head_data = head_attention
                
                # Clean up NaN/Inf values
                head_data = np.nan_to_num(head_data, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Plot attention pattern
                im = axs[i].imshow(head_data, cmap='viridis', vmin=0.0, vmax=1.0)
                
                # Add labels only to the first subplot for cleaner display
                if i == 0:
                    axs[i].set_ylabel('From Position')
                
                axs[i].set_xlabel('To Position')
                axs[i].set_title(f'Head {i}')
            
            # Add colorbar to the rightmost subplot
            plt.colorbar(im, ax=axs[-1], label='Attention Probability')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            return fig
    else:
        # Legacy implementation without safe_tensor_imshow (less reliable across platforms)
        # Ensure tensor is properly prepared for visualization
        if isinstance(attention_maps, torch.Tensor):
            # Handle environment-specific processing
            if IS_APPLE_SILICON:
                # Force to CPU on Apple Silicon to avoid BLAS crashes
                if attention_maps.is_cuda:
                    attention_maps = attention_maps.detach().cpu()
                # Detach for safety
                if attention_maps.requires_grad:
                    attention_maps = attention_maps.detach()
            elif IS_COLAB and HAS_GPU:
                # In Colab with GPU, move to CPU for visualization
                if attention_maps.is_cuda:
                    # Only detach if needed
                    if attention_maps.requires_grad:
                        attention_maps = attention_maps.detach().cpu()
                    else:
                        attention_maps = attention_maps.cpu()
            else:
                # Standard environment handling
                if attention_maps.requires_grad:
                    attention_maps = attention_maps.detach()
                if attention_maps.is_cuda:
                    attention_maps = attention_maps.cpu()
            
            # Always ensure contiguous memory layout for efficient conversion
            if not attention_maps.is_contiguous():
                attention_maps = attention_maps.contiguous()
                
            attn = attention_maps
        else:
            attn = attention_maps
        
        # Create figure
        if head_idx is not None:
            # Single head visualization
            fig, ax = plt.subplots(figsize=figsize)
            
            # Extract the specific head attention and safely convert to numpy
            if isinstance(attn, torch.Tensor):
                # Handle potential NaN/Inf values during conversion
                try:
                    head_attention = attn[0, head_idx].cpu().numpy()
                    if np.isnan(head_attention).any() or np.isinf(head_attention).any():
                        head_attention = np.nan_to_num(head_attention, nan=0.0, posinf=1.0, neginf=0.0)
                except Exception as e:
                    print(f"⚠️ Warning: Error converting attention tensor: {e}")
                    head_attention = np.zeros((10, 10))  # Empty placeholder
            else:
                head_attention = attn[0, head_idx]
            
            # Plot attention pattern
            im = ax.imshow(head_attention, cmap='viridis')
            ax.set_title(title or f'Attention pattern (layer {layer_idx}, head {head_idx})')
            
            # Set proper limits for attention values (0 to 1)
            im.set_clim(0, 1.0)
            
            # Add colorbar
            plt.colorbar(im, label='Attention probability')
            
            # Add labels
            plt.xlabel('Sequence position (to)')
            plt.ylabel('Sequence position (from)')
            
        else:
            # Multiple heads visualization
            heads_to_show = min(num_heads, attn.shape[1])
            fig, axs = plt.subplots(1, heads_to_show, figsize=figsize)
            
            # Ensure axs is always an array even with single subplot
            if heads_to_show == 1:
                axs = [axs]
                
            # Adjust title
            if title:
                fig.suptitle(title, fontsize=14)
            else:
                fig.suptitle(f'Attention patterns (layer {layer_idx})', fontsize=14)
            
            # Plot each head
            for i in range(heads_to_show):
                # Extract head attention and safely convert to numpy
                if isinstance(attn, torch.Tensor):
                    try:
                        head_attention = attn[0, i].cpu().numpy()
                        # Check for NaN/Inf values
                        if np.isnan(head_attention).any() or np.isinf(head_attention).any():
                            head_attention = np.nan_to_num(head_attention, nan=0.0, posinf=1.0, neginf=0.0)
                    except Exception as e:
                        print(f"⚠️ Warning: Error converting attention tensor for head {i}: {e}")
                        head_attention = np.zeros((10, 10))  # Empty placeholder
                else:
                    head_attention = attn[0, i]
                    
                # Plot attention pattern
                im = axs[i].imshow(head_attention, cmap='viridis')
                
                # Set proper limits for attention values (0 to 1)
                im.set_clim(0, 1.0)
                
                # Add labels only to the first subplot for cleaner display
                if i == 0:
                    axs[i].set_ylabel('From position')
                
                axs[i].set_xlabel('To position')
                axs[i].set_title(f'Head {i}')
            
            # Add colorbar to the right of the last subplot
            fig.colorbar(im, ax=axs[-1], label='Attention probability')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig