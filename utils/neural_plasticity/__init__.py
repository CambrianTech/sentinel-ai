"""
Neural Plasticity Utilities (v0.0.61 2025-04-20)

This module provides a comprehensive modular implementation of neural plasticity
for transformer models, enabling dynamic pruning and regrowth of attention heads.

Key features:
- Environment-aware tensor operations that work across platforms
- Detection and adaptation for Apple Silicon, CPU, and GPU environments
- Dynamic pruning based on entropy and gradient metrics
- Improved entropy calculation with detailed diagnostics
- Apple Silicon compatibility with safe tensor operations
- Visualization tools for attention patterns and pruning decisions
- Training loops with differential learning rates
- Complete experiment runners for end-to-end testing

These utilities enable transformer models to become more efficient through
adaptively modifying their structure during training.
"""

__version__ = "0.0.61"
__date__ = "2025-04-20"

# Import environment detection and core tensor operations
from .core import (
    # Core tensor operations
    calculate_head_entropy,
    compute_improved_entropy,  # New function exposed
    calculate_head_gradients,
    generate_pruning_mask,
    gradient_based_pruning,
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
    get_plasticity_optimizer,
    run_warmup_phase
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
    def calculate_head_importance(model, dataloader, num_batches=2, mode="combined"):
        """
        Calculate importance scores for each attention head.
        
        Args:
            model: The transformer model
            dataloader: DataLoader for input data
            num_batches: Number of batches to process
            mode: Importance metric mode - "entropy", "gradient", or "combined"
            
        Returns:
            Dictionary with importance metrics for each head
        """
        import torch
        
        # Get device
        device = next(model.parameters()).device
        
        # Calculate entropy values
        with torch.no_grad():
            batch = next(iter(dataloader))
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass with attention outputs
            outputs = model(**inputs, output_attentions=True)
            
            # Extract attention maps
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Calculate entropy for each layer
                entropy_values = torch.stack([
                    calculate_head_entropy(layer_attn) 
                    for layer_attn in outputs.attentions
                ])
            else:
                raise ValueError("Model does not output attention maps")
        
        # Calculate gradient norms
        grad_norm_values = calculate_head_gradients(model, dataloader, num_batches=num_batches)
        
        # Combine metrics if requested
        if mode == "combined":
            # Normalize entropy (higher is worse)
            norm_entropy = (entropy_values - entropy_values.min()) / (entropy_values.max() - entropy_values.min() + 1e-8)
            
            # Normalize gradient norms (lower is worse)
            norm_grad = 1.0 - (grad_norm_values - grad_norm_values.min()) / (grad_norm_values.max() - grad_norm_values.min() + 1e-8)
            
            # Combine metrics (higher score = less important head)
            importance = 1.0 - (norm_entropy * 0.6 + norm_grad * 0.4)
        elif mode == "entropy":
            # Lower entropy = more important (focused attention)
            max_entropy = entropy_values.max()
            importance = 1.0 - (entropy_values / max_entropy)
        elif mode == "gradient":
            # Higher gradient = more important (more learning)
            max_grad = grad_norm_values.max()
            importance = grad_norm_values / max_grad
        else:
            raise ValueError(f"Unknown importance mode: {mode}")
        
        return {
            "importance": importance,
            "entropy": entropy_values,
            "gradients": grad_norm_values
        }
    
    @staticmethod
    def create_plasticity_trainer(model, learning_rate=5e-5, use_differential_lr=True):
        """
        Create a plasticity trainer for fine-tuning models after pruning.
        
        Args:
            model: The transformer model
            learning_rate: Base learning rate
            use_differential_lr: Whether to use different learning rates for pruned layers
            
        Returns:
            Plasticity trainer object
        """
        return create_plasticity_trainer(
            model=model,
            learning_rate=learning_rate,
            use_differential_lr=use_differential_lr
        )
    
    @staticmethod
    def prune_model_heads(model, pruning_mask, mode="zero_weights"):
        """
        Apply pruning to a model based on a mask.
        
        Args:
            model: The transformer model
            pruning_mask: Boolean tensor where True indicates heads to prune
            mode: Pruning mode - "zero_weights", "mask_forward", or "gate"
            
        Returns:
            List of (layer, head) tuples of pruned heads
        """
        return apply_pruning_mask(model, pruning_mask, mode)
    
    @staticmethod
    def evaluate_model_performance(model, dataloader, device=None, max_eval_steps=10):
        """
        Evaluate model performance.
        
        Args:
            model: The transformer model
            dataloader: DataLoader for evaluation
            device: Device to use (defaults to model's device)
            max_eval_steps: Maximum eval steps
            
        Returns:
            Dictionary with evaluation metrics
        """
        return evaluate_model(model, dataloader, device, max_eval_steps)
    
    @staticmethod
    def train_pruned_model(model, train_dataloader, eval_dataloader, 
                           pruned_heads=None, learning_rate=5e-5, steps=100):
        """
        Fine-tune a pruned model to recover performance.
        
        Args:
            model: The pruned model
            train_dataloader: DataLoader for training
            eval_dataloader: DataLoader for evaluation
            pruned_heads: List of (layer, head) tuples of pruned heads
            learning_rate: Learning rate for training
            steps: Number of training steps
            
        Returns:
            Dictionary with training metrics
        """
        if pruned_heads is None:
            pruned_heads = []
            
        return train_with_plasticity(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            pruned_heads=pruned_heads,
            learning_rate=learning_rate,
            training_steps=steps
        )
    
    @staticmethod
    def generate_pruning_mask(grad_values, entropy_values=None, prune_percent=0.2, 
                             strategy="combined", random_seed=None):
        """
        Generate a pruning mask based on head importance metrics.
        
        Args:
            grad_values: Gradient norm values tensor
            entropy_values: Optional entropy values tensor
            prune_percent: Percentage of heads to prune (0-1)
            strategy: Pruning strategy - "gradient", "entropy", "random", or "combined"
            random_seed: Optional seed for reproducibility
            
        Returns:
            Boolean tensor where True indicates heads to prune
        """
        return generate_pruning_mask(
            grad_norm_values=grad_values,
            entropy_values=entropy_values,
            prune_percent=prune_percent,
            strategy=strategy,
            random_seed=random_seed
        )
    
    @staticmethod
    def visualize_head_metrics(entropy_values, grad_norm_values, pruned_heads=None):
        """
        Create visualizations for head metrics and pruning decisions.
        
        Args:
            entropy_values: Entropy values tensor
            grad_norm_values: Gradient norm values tensor
            pruned_heads: Optional list of (layer, head) tuples of pruned heads
            
        Returns:
            Dictionary of visualization figures
        """
        # Create entropy visualization
        entropy_fig = visualize_head_entropy(
            entropy_values=entropy_values,
            title="Attention Head Entropy",
            annotate=True
        )
        
        # Create gradient visualization
        grad_fig = visualize_head_gradients(
            grad_norm_values=grad_norm_values,
            pruned_heads=pruned_heads,
            title="Attention Head Gradient Norms"
        )
        
        return {
            "entropy": entropy_fig,
            "gradients": grad_fig
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
    def compute_entropy_with_diagnostics(attention_maps, eps=1e-8, debug=True):
        """
        Compute entropy with detailed diagnostics for debugging.
        
        This method is useful for diagnosing numerical stability issues 
        in entropy calculations on different hardware platforms.
        
        Args:
            attention_maps: Attention tensor [batch, heads, seq_len, seq_len]
            eps: Small epsilon value for numerical stability
            debug: Whether to print detailed diagnostic information
            
        Returns:
            Tensor with entropy values
        """
        return compute_improved_entropy(attention_maps, eps=eps, debug=debug)
        
    @staticmethod
    def create_gradient_pruning_mask(grad_norm_values, prune_percent=0.1):
        """
        Generate a pruning mask based purely on gradient norms.
        
        This targets heads with the lowest gradients for pruning.
        
        Args:
            grad_norm_values: Tensor of gradient norm values
            prune_percent: Percentage of heads to prune (0-1)
            
        Returns:
            Boolean tensor where True indicates heads to prune
        """
        return gradient_based_pruning(grad_norm_values, prune_percent)
    
    @staticmethod
    def run_warmup_training(model, train_dataloader, max_epochs=1, 
                           learning_rate=5e-5, patience=15,
                           device=None, verbose=True):
        """
        Run a warmup phase until loss stabilizes.
        
        This pre-trains the model until metrics stabilize.
        
        Args:
            model: The model to warm up
            train_dataloader: DataLoader for training data
            max_epochs: Maximum epochs to run
            learning_rate: Learning rate
            patience: Steps without improvement to consider stable
            device: Device to run on
            verbose: Whether to print progress info
            
        Returns:
            Dictionary with warmup metrics
        """
        return run_warmup_phase(
            model=model,
            train_dataloader=train_dataloader,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            patience=patience,
            device=device,
            verbose=verbose
        )
    
    @staticmethod
    def diagnose_attention_patterns(model, inputs, device=None):
        """
        Diagnose attention patterns with detailed metrics.
        
        Args:
            model: The model to diagnose
            inputs: Input data batch
            device: Device to run on
            
        Returns:
            Dictionary with diagnostic information
        """
        # Determine device
        if device is None:
            if IS_APPLE_SILICON:
                device = torch.device('cpu')
            else:
                device = next(model.parameters()).device
                
        # Move inputs to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        else:
            inputs = {"input_ids": inputs[0].to(device)}
            if len(inputs) > 1:
                inputs["attention_mask"] = inputs[1].to(device)
        
        # Run model to get attention values
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        # Extract and analyze attention tensors
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            return {"error": "Model does not output attention maps"}
            
        attn_tensors = outputs.attentions
        
        # Analyze attention statistics
        layer_stats = []
        total_entropy = 0
        
        for layer_idx, layer_attn in enumerate(attn_tensors):
            # Basic tensor statistics
            has_nan = torch.isnan(layer_attn).any().item()
            has_inf = torch.isinf(layer_attn).any().item()
            
            # Check row sum = 1
            row_sums = layer_attn.sum(dim=-1)
            valid_sum = torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3)
            
            # Calculate entropy for this layer
            layer_entropy = calculate_head_entropy(layer_attn)
            total_entropy += layer_entropy.mean().item()
            
            # Store layer statistics
            layer_stats.append({
                "layer_idx": layer_idx,
                "shape": list(layer_attn.shape),
                "min": layer_attn.min().item(),
                "max": layer_attn.max().item(),
                "mean": layer_attn.mean().item(),
                "has_nan": has_nan,
                "has_inf": has_inf, 
                "valid_sum": valid_sum,
                "entropy": layer_entropy.detach().cpu().numpy().tolist()
            })
        
        # Return comprehensive diagnostic information
        return {
            "layer_count": len(attn_tensors),
            "layer_stats": layer_stats,
            "total_entropy": total_entropy / len(attn_tensors),
            "attention_tensors": [attn.detach().cpu() for attn in attn_tensors]
        }
        
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