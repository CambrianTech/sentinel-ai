"""
Neural Plasticity Utilities (v0.0.60 2025-04-20)

This module provides a comprehensive modular implementation of neural plasticity
for transformer models, enabling dynamic pruning and regrowth of attention heads.

Key features:
- Environment-aware tensor operations that work across platforms
- Detection and adaptation for Apple Silicon, CPU, and GPU environments
- Dynamic pruning based on entropy and gradient metrics
- Visualization tools for attention patterns and pruning decisions
- Training loops with differential learning rates
- Complete experiment runners for end-to-end testing

These utilities enable transformer models to become more efficient through
adaptively modifying their structure during training.
"""

__version__ = "0.0.60"
__date__ = "2025-04-20"

# Import environment detection and core tensor operations
from .core import (
    # Core tensor operations
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model,
    safe_matmul,
    
    # Environment detection
    IS_APPLE_SILICON,
    IS_COLAB,
    HAS_GPU,
    
    # Utilities
    detect_model_structure,
    extract_head_gradient
)

# Import visualization functions
from .visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    visualize_attention_patterns
)

# Import training utilities
from .training import (
    PlasticityTrainer,
    create_plasticity_trainer,
    run_plasticity_loop,
    train_with_plasticity,
    get_plasticity_optimizer
)

# Import experiment runner
from .experiment import (
    run_neural_plasticity_experiment,
    create_model_and_entropy_baseline,
    get_dataloader_builder,
    plot_baseline_entropy
)

# Define an enum for pruning strategies
class PruningStrategy:
    """Pruning strategy options for neural plasticity."""
    GRADIENT = "gradient"
    ENTROPY = "entropy"
    RANDOM = "random"
    COMBINED = "combined"

# Define an enum for pruning modes
class PruningMode:
    """Pruning mode options (adaptive vs permanent)."""
    ADAPTIVE = "adaptive"  # Allows recovery/regrowth
    COMPRESSED = "compressed"  # Permanent pruning

# Create a simple API for notebook usage
class NeuralPlasticity:
    """
    High-level API for neural plasticity in transformer models.
    
    This class provides a simplified interface for notebooks to use
    neural plasticity functionality without having to import multiple
    modules or handle environment-specific code.
    """
    
    @staticmethod
    def get_environment_info():
        """Get information about the current execution environment."""
        import torch
        import platform
        
        return {
            "is_apple_silicon": IS_APPLE_SILICON,
            "is_colab": IS_COLAB,
            "has_gpu": HAS_GPU,
            "device": "cuda" if HAS_GPU and not IS_APPLE_SILICON else "cpu",
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }
    
    @staticmethod
    def detect_attention_heads(model):
        """
        Detect number of layers and heads in a transformer model.
        
        Args:
            model: The transformer model
            
        Returns:
            tuple of (num_layers, num_heads)
        """
        return detect_model_structure(model)
    
    @staticmethod
    def analyze_attention_patterns(model, input_ids, attention_mask=None):
        """
        Analyze attention patterns in a model for given inputs.
        
        Args:
            model: The transformer model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing entropy values and attention tensors
        """
        import torch
        
        # Ensure model is in eval mode
        model.eval()
        
        # Prepare inputs
        device = next(model.parameters()).device
        if attention_mask is None and hasattr(input_ids, 'shape'):
            attention_mask = torch.ones(input_ids.shape, device=device)
        
        # Get attention patterns
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True
            )
            
        # Extract attention tensors
        attention_tensors = outputs.attentions
        
        # Calculate entropy for each attention head
        entropy_values = {}
        for layer_idx, layer_attention in enumerate(attention_tensors):
            layer_entropy = calculate_head_entropy(layer_attention)
            entropy_values[layer_idx] = layer_entropy
            
        return {
            "attention_tensors": attention_tensors,
            "entropy_values": entropy_values
        }
    
    @staticmethod
    def run_pruning_cycle(model, train_dataloader, eval_dataloader, 
                          pruning_level=0.2, strategy="combined", 
                          learning_rate=5e-5, training_steps=100,
                          callback=None):
        """
        Run a complete pruning cycle: analyze → prune → train → evaluate.
        
        Args:
            model: The transformer model
            train_dataloader: DataLoader for training
            eval_dataloader: DataLoader for evaluation
            pruning_level: Percentage of heads to prune (0-1)
            strategy: Pruning strategy (gradient, entropy, random, combined)
            learning_rate: Learning rate for fine-tuning
            training_steps: Number of training steps
            callback: Optional callback function for progress tracking
            
        Returns:
            Dictionary with pruning results and metrics
        """
        return run_plasticity_loop(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            pruning_level=pruning_level,
            strategy=strategy,
            learning_rate=learning_rate,
            training_steps=training_steps,
            callback=callback
        )
    
    @staticmethod
    def run_full_experiment(model_name="distilgpt2", output_dir="plasticity_results",
                            pruning_strategy="entropy", prune_percent=0.2,
                            batch_size=4, learning_rate=5e-5):
        """
        Run a complete neural plasticity experiment from scratch.
        
        Args:
            model_name: Name of the model to use
            output_dir: Directory to save results
            pruning_strategy: Strategy for pruning
            prune_percent: Percentage of heads to prune (0-1)
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Dictionary with experiment results
        """
        # Determine appropriate device
        device = "cuda" if HAS_GPU and not IS_APPLE_SILICON else "cpu"
        
        return run_neural_plasticity_experiment(
            model_name=model_name,
            device=device,
            output_dir=output_dir,
            pruning_strategy=pruning_strategy,
            prune_percent=prune_percent,
            batch_size=batch_size,
            learning_rate=learning_rate
        )