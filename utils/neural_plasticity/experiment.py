"""
Neural Plasticity Experiment

This module provides experiment utilities for running end-to-end neural
plasticity experiments, including baseline creation, data preparation,
and results analysis.

Version: v0.0.69 (2025-04-20 24:25:00)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from datetime import datetime
import importlib.util

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
)

# Fix for scheduler import
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    # Provide a simple implementation if transformers version is missing it
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr.
        This is a simple fallback implementation for when transformers is not available.
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
from torch.utils.data import DataLoader

# Safely import datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets package not available. Limited functionality.")
    # Provide a dummy function for testing
    def load_dataset(*args, **kwargs):
        raise ImportError("datasets package is required for loading datasets")

from .core import (
    calculate_head_entropy,
    calculate_head_gradients,
    detect_model_structure,
    evaluate_model,
    generate_pruning_mask,
    apply_pruning_mask,
    IS_APPLE_SILICON,
    IS_COLAB,
    HAS_GPU
)

from .training import run_plasticity_loop, run_warmup_phase
from .visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    visualize_attention_patterns,
    VisualizationReporter
)

# Try to import dashboard utilities
try:
    from .dashboard import DashboardReporter
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


def get_dataloader_builder(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    model_name: str = "distilgpt2",
    max_length: int = 128,
    batch_size: int = 4
) -> Callable[[], Tuple[DataLoader, DataLoader]]:
    """
    Create a function that builds dataloaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset to use
        dataset_config: Dataset configuration
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for dataloaders

    Returns:
        Function that returns (train_dataloader, eval_dataloader)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
        # Load datasets
        train_dataset = load_dataset(dataset_name, dataset_config, split="train")
        validation_dataset = load_dataset(dataset_name, dataset_config, split="validation")

        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

        # Tokenize datasets
        train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        validation_dataset = validation_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # Add labels for language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        train_dataset = train_dataset.map(add_labels)
        validation_dataset = validation_dataset.map(add_labels)

        # Set format
        train_dataset = train_dataset.with_format("torch")
        validation_dataset = validation_dataset.with_format("torch")

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator
        )

        return train_dataloader, validation_dataloader

    return build_dataloaders


def create_model_and_entropy_baseline(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
    eval_dataloader: Optional[DataLoader] = None,
    dataloader_builder: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create a model and calculate baseline entropy values.

    Args:
        model_name: Name of the model to use
        device: Device to use (auto-detected if None)
        eval_dataloader: Evaluation dataloader
        dataloader_builder: Function to build dataloaders if eval_dataloader is None

    Returns:
        Dictionary with model, entropy, gradient values, and evaluation metrics
    """
    # Determine appropriate device
    if device is None:
        if IS_COLAB and HAS_GPU:
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif IS_APPLE_SILICON:
            device = torch.device("cpu")
            print("Using CPU on Apple Silicon")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Create dataloader if not provided
    if eval_dataloader is None:
        if dataloader_builder is None:
            dataloader_builder = get_dataloader_builder(model_name=model_name)
        
        train_dataloader, eval_dataloader = dataloader_builder()
    
    # Calculate baseline entropy
    print("Calculating baseline entropy...")
    with torch.no_grad():
        batch = next(iter(eval_dataloader))
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
    
    # Calculate baseline gradients
    print("Calculating baseline gradients...")
    grad_norm_values = calculate_head_gradients(model, eval_dataloader)
    
    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(model, eval_dataloader)
    
    return {
        "model": model,
        "device": device,
        "entropy_values": entropy_values,
        "grad_norm_values": grad_norm_values,
        "baseline_metrics": baseline_metrics
    }


def plot_baseline_entropy(
    entropy_values: torch.Tensor,
    grad_norm_values: torch.Tensor,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot baseline entropy and gradient values.

    Args:
        entropy_values: Tensor of entropy values
        grad_norm_values: Tensor of gradient norm values
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot entropy
    entropy_data = entropy_values.detach().cpu().numpy()
    im1 = ax1.imshow(entropy_data, cmap="viridis", aspect="auto")
    fig.colorbar(im1, ax=ax1, label='Entropy')
    ax1.set_title('Head Entropy (Higher = Less Focused)')
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Layer Index')
    
    # Set proper colormap limits with non-zero range
    im1.set_clim(0, max(0.1, entropy_data.max()))
    
    # Plot gradients
    grad_data = grad_norm_values.detach().cpu().numpy()
    im2 = ax2.imshow(grad_data, cmap="plasma", aspect="auto")
    fig.colorbar(im2, ax=ax2, label='Gradient Norm')
    ax2.set_title('Head Gradient Norms (Higher = More Learning)')
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Layer Index')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


class NeuralPlasticityExperiment:
    """
    End-to-end experiment runner for neural plasticity.
    
    This class provides a comprehensive workflow for running neural plasticity
    experiments, including model setup, dataset loading, warmup, pruning, 
    training, and evaluation. It's designed to provide a consistent interface
    that works across different environments (local, remote, Colab, etc.)
    while handling environment-specific optimizations automatically.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        dataset: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        output_dir: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 128,
        pruning_level: float = 0.2,
        pruning_strategy: str = "combined",
        learning_rate: float = 5e-5,
        device: Optional[str] = None,
        verbose: bool = True,
        save_results: bool = True,
        show_samples: bool = False,
        sample_interval: int = 20,
        tokenizer=None,
        use_dashboard: bool = False,
        dashboard_dir: Optional[str] = None,
        dashboard_name: str = "neural_plasticity_dashboard.html",
        metrics_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        sample_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the Neural Plasticity experiment.
        
        Args:
            model_name: Name of the model to use
            dataset: Name of the dataset to use
            dataset_config: Dataset configuration
            output_dir: Directory to save results (auto-created with timestamp if None)
            batch_size: Batch size for training
            max_length: Maximum sequence length
            pruning_level: Percentage of heads to prune (0-1)
            pruning_strategy: Pruning strategy to use
            learning_rate: Learning rate for training
            device: Device to use (auto-detected if None)
            verbose: Whether to print status information
            save_results: Whether to save experiment results
            show_samples: Whether to display sample predictions during training
            sample_interval: Interval for showing sample predictions
            tokenizer: Optional tokenizer (loaded from model_name if None)
            use_dashboard: Whether to generate an interactive HTML dashboard
            dashboard_dir: Directory for dashboard files (uses output_dir if None)
            dashboard_name: Name of the main dashboard HTML file
            metrics_callback: Optional callback function for reporting metrics (step, metrics_dict)
            sample_callback: Optional callback function for reporting sample generations (step, samples_dict)
        """
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Dashboard parameters
        self.use_dashboard = use_dashboard
        self.dashboard_dir = dashboard_dir
        self.dashboard_name = dashboard_name
        
        # Callback functions
        self.metrics_callback = metrics_callback
        self.sample_callback = sample_callback
        self.pruning_level = pruning_level
        self.pruning_strategy = pruning_strategy
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.save_results = save_results
        self.show_samples = show_samples
        self.sample_interval = sample_interval
        self.tokenizer_provided = tokenizer is not None
        self.tokenizer = tokenizer
        
        # Create output directory with timestamp if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"neural_plasticity_output/experiment_{timestamp}"
        else:
            self.output_dir = output_dir
            
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine appropriate device
        if device is None:
            if IS_COLAB and HAS_GPU:
                self.device = torch.device("cuda")
                if self.verbose:
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif IS_APPLE_SILICON:
                self.device = torch.device("cpu")
                if self.verbose:
                    print("Using CPU on Apple Silicon")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.verbose:
                    print(f"Using device: {self.device}")
        else:
            self.device = torch.device(device)
            if self.verbose:
                print(f"Using device: {self.device}")
                
        # Initialize experiment state
        self.model = None
        self.tokenizer = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.reporter = None
        self.baseline_loss = None
        self.baseline_perplexity = None
        self.final_loss = None
        self.final_perplexity = None
        self.cycle_results = []
        self.current_cycle = 0
        self.entropy_values = None
        self.grad_norm_values = None
        self.pruning_mask = None
        self.pruned_heads = []
        
        # Initialize metrics tracking
        self.metrics_history = {
            "step": [], 
            "epoch": [], 
            "train_loss": [], 
            "eval_loss": [],
            "perplexity": [], 
            "pruned_heads": [], 
            "revived_heads": [],
            "sparsity": [], 
            "total_pruned": []
        }
    
    def setup(self):
        """
        Set up the experiment by loading model, tokenizer, and datasets.
        """
        if self.verbose:
            print(f"Setting up experiment with {self.model_name} on {self.dataset}/{self.dataset_config}")
            
        # Load tokenizer if not provided or ensure it's ready for use
        if not self.tokenizer_provided or self.tokenizer is None:
            if self.verbose:
                print(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer_provided = True
        
        # Ensure pad token is set    
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        if self.verbose:
            print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        # Load datasets
        if self.verbose:
            print(f"Loading dataset: {self.dataset}/{self.dataset_config}")
        train_dataset = load_dataset(self.dataset, self.dataset_config, split="train")
        validation_dataset = load_dataset(self.dataset, self.dataset_config, split="validation")

        # Define tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length
            )

        # Tokenize datasets
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Add labels for language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        train_dataset = train_dataset.map(add_labels)
        validation_dataset = validation_dataset.map(add_labels)

        # Set format
        train_dataset = train_dataset.with_format("torch")
        validation_dataset = validation_dataset.with_format("torch")

        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=default_data_collator
        )

        self.validation_dataloader = DataLoader(
            validation_dataset, 
            batch_size=self.batch_size, 
            collate_fn=default_data_collator
        )
        
        if self.verbose:
            print(f"Train dataset size: {len(train_dataset)} examples")
            print(f"Validation dataset size: {len(validation_dataset)} examples")
            
        # Initialize visualization reporter
        self.reporter = VisualizationReporter(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=self.output_dir,
            save_visualizations=self.save_results,
            verbose=self.verbose
        )
        
        # Initialize dashboard if requested
        if self.use_dashboard and DASHBOARD_AVAILABLE:
            # Set dashboard directory to output_dir if not specified
            self.dashboard_dir = self.dashboard_dir or os.path.join(self.output_dir, "dashboard")
            os.makedirs(self.dashboard_dir, exist_ok=True)
            
            if self.verbose:
                print(f"🔍 Dashboard will be available at: {os.path.join(self.dashboard_dir, self.dashboard_name)}")
        
        # Log environment information
        if self.save_results:
            env_info = {
                "model_name": self.model_name,
                "dataset": self.dataset,
                "dataset_config": self.dataset_config,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "pruning_level": self.pruning_level,
                "pruning_strategy": self.pruning_strategy,
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                "is_apple_silicon": IS_APPLE_SILICON,
                "is_colab": IS_COLAB,
                "has_gpu": HAS_GPU,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save environment info
            import json
            with open(os.path.join(self.output_dir, "experiment_config.json"), 'w') as f:
                json.dump(env_info, f, indent=2)
        
        return self
    
    def run_warmup(self, max_epochs=1, patience=15, min_steps=50, max_steps=150):
        """
        Run warmup training to stabilize model before pruning.
        
        Args:
            max_epochs: Maximum number of epochs for warmup
            patience: Number of steps with no decrease to consider loss stabilized
            min_steps: Minimum number of warmup steps
            max_steps: Maximum number of warmup steps per epoch
            
        Returns:
            Dictionary with warmup results
        """
        if self.model is None or self.train_dataloader is None:
            raise ValueError("Model and dataloaders must be set up first. Call setup() method.")
            
        if self.verbose:
            print(f"\n=== Running Warmup Phase (max {max_epochs} epochs) ===")
        
        # If we have a metrics callback, update it with warmup start
        if self.metrics_callback:
            self.metrics_callback(0, {"phase": "warmup_start"})
            
        # Use the modular API's warmup function
        warmup_results = run_warmup_phase(
            model=self.model,
            train_dataloader=self.train_dataloader,
            max_epochs=max_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=100,  # Standard warmup steps for scheduler
            patience=patience,
            min_warmup_steps=min_steps,
            max_warmup_steps=max_steps,
            device=self.device,
            verbose=self.verbose,
            save_visualizations=self.save_results,
            output_dir=os.path.join(self.output_dir, "warmup")
        )
        
        # Use the reporter to display warmup results
        self.reporter.display_warmup_results(warmup_results)
        
        # Get baseline evaluation after warmup
        self.baseline_loss, self.baseline_perplexity = self.reporter.evaluate_model(
            model=self.model,
            dataloader=self.validation_dataloader
        )
        
        # Store the warmup results
        self.warmup_results = warmup_results
        
        # Send metrics to callback
        if self.metrics_callback:
            # Extract all losses for step-by-step metrics
            steps = list(range(len(warmup_results["losses"])))
            for i, (step, loss) in enumerate(zip(steps, warmup_results["losses"])):
                self.metrics_callback(step, {
                    "phase": "warmup",
                    "warmup_loss": loss,
                    "epoch": warmup_results.get("epochs", [0])[0] if warmup_results.get("epochs") else 0,
                    "step": step
                })
            
            # Send final warmup metrics
            self.metrics_callback(steps[-1] if steps else 0, {
                "phase": "warmup_complete",
                "baseline_loss": self.baseline_loss,
                "baseline_perplexity": self.baseline_perplexity,
                "warmup_steps": len(warmup_results["losses"]),
                "is_stable": warmup_results.get("is_stable", False)
            })
        
        # Save warmup metrics to CSV
        if self.save_results:
            self.reporter.save_metrics_to_csv(
                metrics={
                    "step": list(range(len(warmup_results["losses"]))),
                    "loss": warmup_results["losses"],
                    "smoothed_loss": warmup_results["smoothed_losses"] if len(warmup_results["smoothed_losses"]) == len(warmup_results["losses"]) else [None] * len(warmup_results["losses"])
                },
                filename="warmup_metrics.csv",
                subfolder="warmup"
            )
            
        # Save baseline model checkpoint for later comparison
        try:
            baseline_checkpoint_path = os.path.join(self.output_dir, "baseline_checkpoint.pt")
            if self.verbose:
                print(f"Saving baseline model checkpoint to {baseline_checkpoint_path}")
            torch.save(self.model.state_dict(), baseline_checkpoint_path)
            
            # Store baseline metrics for later use
            self.baseline_metrics = {
                "loss": self.baseline_loss,
                "perplexity": self.baseline_perplexity
            }
            
            if self.metrics_callback:
                self.metrics_callback(0, {
                    "message": "Saved baseline model checkpoint",
                    "baseline_checkpoint_path": baseline_checkpoint_path
                })
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to save baseline model checkpoint: {e}")
        
        return warmup_results
    
    def analyze_attention(self):
        """
        Analyze attention patterns of the model.
        
        Calculates entropy and gradient metrics for the current model state.
        
        Returns:
            Dictionary with attention analysis results
        """
        if self.model is None:
            raise ValueError("Model must be set up first. Call setup() method.")
            
        if self.verbose:
            print("\n=== Analyzing Attention Patterns ===")
            
        # Calculate head importance directly using core functions
        # Get a batch from the dataloader
        batch = next(iter(self.validation_dataloader))
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
            # Extract attention maps
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Calculate entropy for each layer and ensure proper shape
                self.entropy_values = torch.stack([
                    calculate_head_entropy(layer_attn) 
                    for layer_attn in outputs.attentions
                ])
                
                # The entropy might have extra dimensions - ensure it matches grad_norm shape
                if len(self.entropy_values.shape) > 2:
                    # Take mean over extra dimensions to get [layers, heads] shape
                    extra_dims = tuple(range(2, len(self.entropy_values.shape)))
                    if extra_dims:
                        self.entropy_values = self.entropy_values.mean(dim=extra_dims)
            else:
                raise ValueError("Model does not output attention maps")
                
        # Calculate gradients
        self.grad_norm_values = calculate_head_gradients(self.model, self.validation_dataloader)
        
        # Store results
        importance_metrics = {
            "entropy": self.entropy_values,
            "gradients": self.grad_norm_values
        }
        
        # Extract metrics
        self.entropy_values = importance_metrics["entropy"]
        self.grad_norm_values = importance_metrics["gradients"]
        
        # Create visualizations
        if self.verbose:
            # Create and display entropy visualization
            entropy_fig = visualize_head_entropy(
                entropy_values=self.entropy_values,
                title="Attention Head Entropy",
                annotate=True
            )
            plt.figure(entropy_fig.number)
            plt.show()
            
            # Create and display gradient visualization
            grad_fig = visualize_head_gradients(
                grad_norm_values=self.grad_norm_values,
                title="Attention Head Gradient Norms"
            )
            plt.figure(grad_fig.number)
            plt.show()
        
        # Save the visualizations
        if self.save_results:
            analysis_dir = os.path.join(self.output_dir, "attention_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            entropy_fig.savefig(os.path.join(analysis_dir, "entropy_heatmap.png"), dpi=100, bbox_inches='tight')
            grad_fig.savefig(os.path.join(analysis_dir, "gradient_heatmap.png"), dpi=100, bbox_inches='tight')
        
        # Return analysis results
        analysis_results = {
            "entropy_values": self.entropy_values,
            "grad_norm_values": self.grad_norm_values,
            "model_structure": detect_model_structure(self.model)
        }
        
        if self.verbose:
            num_layers, num_heads = analysis_results["model_structure"]
            print(f"Model structure: {num_layers} layers with {num_heads} heads each")
            print(f"Total attention heads: {num_layers * num_heads}")
        
        return analysis_results
    
    def run_pruning_cycle(self, training_steps=100, callback=None):
        # Initialize dashboard reporter if requested
        dashboard_reporter = None
        dashboard_path = None
        if self.use_dashboard and DASHBOARD_AVAILABLE:
            # Set dashboard directory to output_dir if not specified
            dashboard_dir = self.dashboard_dir or os.path.join(self.output_dir, "dashboard")
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # Create dashboard reporter
            dashboard_reporter = DashboardReporter(
                output_dir=dashboard_dir,
                dashboard_name=self.dashboard_name,
                auto_update=True,
                update_interval=min(training_steps // 10, 10)  # Update every ~10% of training or 10 steps, whichever is smaller
            )
            if self.verbose:
                print(f"🔍 Interactive dashboard will be available at: {os.path.join(dashboard_dir, self.dashboard_name)}")
        elif self.use_dashboard and not DASHBOARD_AVAILABLE:
            if self.verbose:
                print("⚠️ Dashboard visualization requested but not available. Make sure dashboard.py is accessible.")
        """
        Run a complete pruning cycle: analyze, prune, train, evaluate.
        
        This method represents one full iteration of neural plasticity,
        which includes pruning heads and then retraining the model.
        
        Args:
            training_steps: Number of training steps after pruning
            callback: Optional callback function for custom progress tracking
            
        Returns:
            Dictionary with pruning cycle results
        """
        if self.model is None:
            raise ValueError("Model must be set up first. Call setup() method.")
            
        # Update cycle counter
        self.current_cycle += 1
        
        if self.verbose:
            print(f"\n=== Running Pruning Cycle {self.current_cycle} ===")
        
        # Create cycle-specific directory
        cycle_dir = os.path.join(self.output_dir, f"cycle_{self.current_cycle}")
        if self.save_results:
            os.makedirs(cycle_dir, exist_ok=True)
        
        # Define a wrapper for the callback to track metrics
        def tracking_callback(event, step, metrics):
            # Update our metrics history
            if event == "training" and "train_loss" in metrics:
                self.metrics_history["step"].append(step)
                self.metrics_history["epoch"].append(self.current_cycle)
                self.metrics_history["train_loss"].append(metrics.get("train_loss", 0))
                self.metrics_history["eval_loss"].append(metrics.get("eval_loss", 0))
                self.metrics_history["perplexity"].append(metrics.get("perplexity", 0))
                
                # Update sparsity metrics if available
                if "new_pruned" in metrics:
                    self.metrics_history["pruned_heads"].append(metrics.get("new_pruned", 0))
                if "total_pruned" in metrics:
                    self.metrics_history["total_pruned"].append(metrics.get("total_pruned", 0))
                if "sparsity" in metrics:
                    self.metrics_history["sparsity"].append(metrics.get("sparsity", 0))
            
            # Call the user-provided callback if available
            if callback:
                callback(event, step, metrics)
                
            # Print status if verbose
            if self.verbose and event == "training":
                print(f"  Step {step} - Train loss: {metrics.get('train_loss', 0):.4f}, "
                      f"Eval loss: {metrics.get('eval_loss', 0):.4f}, "
                      f"Perplexity: {metrics.get('perplexity', 0):.2f}")
        
        # Run the pruning cycle (direct implementation)
        
        # 1. Extract the model structure
        model_structure = detect_model_structure(self.model)
        
        # 2. Generate pruning mask using previously calculated metrics
        pruning_mask = generate_pruning_mask(
            grad_norm_values=self.grad_norm_values,
            prune_percent=self.pruning_level,
            strategy=self.pruning_strategy,
            entropy_values=self.entropy_values
        )
        
        # 3. Apply the pruning mask to get pruned heads
        pruned_heads = []
        for layer_idx, layer_mask in enumerate(pruning_mask):
            for head_idx, is_pruned in enumerate(layer_mask):
                if is_pruned:
                    pruned_heads.append((layer_idx, head_idx))
                    
        # 4. Apply pruning to the model (modifies model in-place and returns pruned_heads)
        pruned_heads = apply_pruning_mask(self.model, pruning_mask)
        
        # 5. Evaluate after pruning
        pruned_metrics = evaluate_model(self.model, self.validation_dataloader, self.device)
        
        # 6. Train the pruned model using plasticity trainer
        from .training import PlasticityTrainer
        
        trainer = PlasticityTrainer(
            model=self.model,
            learning_rate=self.learning_rate,
            use_differential_lr=True,
            pruned_head_lr_multiplier=3.0
        )
        
        trainer.prepare_optimizer(
            pruned_heads=pruned_heads,
            warmup_steps=training_steps // 10,
            total_steps=training_steps
        )
        
        # Define a training callback that wraps the tracking callback
        def train_callback(step, metrics):
            if tracking_callback:
                # Add pruning metrics
                metrics["new_pruned"] = len(pruned_heads)
                metrics["total_pruned"] = len(pruned_heads)
                tracking_callback("training", step, metrics)
            
            # Update dashboard with training metrics
            if dashboard_reporter:
                # Add sparsity metric to track pruning level
                dashboard_metrics = metrics.copy()
                if "sparsity" not in dashboard_metrics:
                    # Calculate sparsity as fraction of pruned heads
                    if hasattr(self.model, "config"):
                        total_heads = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
                        dashboard_metrics["sparsity"] = len(pruned_heads) / total_heads
                
                dashboard_reporter.add_metrics(dashboard_metrics, step)
        
        # Define a sample callback if samples are being shown
        def sample_callback(step, sample_data):
            if tracking_callback:
                tracking_callback("sample", step, sample_data)
            
            # Update dashboard with sample data
            if dashboard_reporter and sample_data:
                try:
                    # Transform prediction format for dashboard
                    input_text = sample_data.get("input_text", "")
                    predictions = sample_data.get("predictions", [])
                    
                    if predictions:
                        predicted_tokens = [p["predicted_token"] for p in predictions]
                        predicted_probs = [p["predicted_prob"] for p in predictions]
                        actual_tokens = [p["actual_token"] for p in predictions]
                        actual_probs = [p["actual_prob"] for p in predictions]
                        perplexities = [p["perplexity"] for p in predictions]
                        
                        dashboard_reporter.add_sample(
                            step=step,
                            input_text=input_text,
                            predicted_tokens=predicted_tokens,
                            predicted_probs=predicted_probs,
                            actual_tokens=actual_tokens,
                            actual_probs=actual_probs,
                            perplexities=perplexities
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error adding sample to dashboard: {e}")
        
        # Set up dashboard sample callback
        dashboard_sample_callback = None
        if dashboard_reporter and self.show_samples:
            dashboard_sample_callback = dashboard_reporter.get_sample_callback()
        
        # Train the model with sample display if requested
        training_metrics = trainer.train(
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.validation_dataloader,
            steps=training_steps,
            eval_interval=max(1, training_steps // 10),
            callback=train_callback,
            show_samples=self.show_samples,
            tokenizer=self.tokenizer if self.show_samples else None,
            sample_interval=self.sample_interval,
            sample_callback=sample_callback if tracking_callback else dashboard_sample_callback
        )
            
        # 7. Evaluate final model
        final_metrics = evaluate_model(self.model, self.validation_dataloader, self.device)
        
        # Update dashboard with final metrics
        if dashboard_reporter:
            # Add final metrics and sparsity
            if hasattr(self.model, "config"):
                total_heads = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
                sparsity = len(pruned_heads) / total_heads
            else:
                sparsity = len(pruned_heads) / (grad_norm_values.numel()) if grad_norm_values is not None else 0.0
            
            dashboard_reporter.add_metrics({
                "eval_loss": final_metrics['loss'],
                "perplexity": final_metrics['perplexity'],
                "sparsity": sparsity,
                "train_loss": training_metrics.get("train_loss", [-1])[-1] if "train_loss" in training_metrics else -1
            }, training_steps)
            
            # Add attention visualization if possible
            try:
                # Get attention maps for visualization
                with torch.no_grad():
                    batch = next(iter(self.validation_dataloader))
                    
                    # Move batch to device
                    inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    # Forward pass with attention outputs
                    outputs = self.model(**inputs, output_attentions=True)
                    
                    # Add attention maps to dashboard
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        for layer_idx, layer_attn in enumerate(outputs.attentions):
                            dashboard_reporter.add_attention_map(layer_attn, layer_idx)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error adding attention visualization to dashboard: {e}")
        
        # 8. Create visualizations
        if self.save_results:
            # Create visualizations using our visualization module
            from .visualization import (
                visualize_head_entropy,
                visualize_head_gradients,
                visualize_pruning_decisions,
                visualize_training_metrics
            )
            
            entropy_fig = visualize_head_entropy(
                entropy_values=self.entropy_values,
                title="Head Entropy"
            )
            entropy_fig.savefig(os.path.join(cycle_dir, "entropy_heatmap.png"))
            plt.close(entropy_fig)
            
            grad_fig = visualize_head_gradients(
                grad_norm_values=self.grad_norm_values,
                pruned_heads=pruned_heads,
                title="Gradient Norms"
            )
            grad_fig.savefig(os.path.join(cycle_dir, "gradient_heatmap.png"))
            plt.close(grad_fig)
            
            pruning_fig = visualize_pruning_decisions(
                grad_norm_values=self.grad_norm_values, 
                pruning_mask=pruning_mask,
                title="Pruning Decisions"
            )
            pruning_fig.savefig(os.path.join(cycle_dir, "pruning_decisions.png"))
            plt.close(pruning_fig)
        
        # 9. Prepare results
        pruning_results = {
            "pruned_heads": pruned_heads,
            "pruning_mask": pruning_mask,
            "model_structure": model_structure,
            "entropy_values": self.entropy_values,
            "grad_norm_values": self.grad_norm_values,
            "baseline_metrics": {
                "loss": self.baseline_loss,
                "perplexity": self.baseline_perplexity
            },
            "pruned_metrics": pruned_metrics,
            "final_metrics": final_metrics,
            "training_metrics": training_metrics,
            "strategy": self.pruning_strategy,
            "pruning_level": self.pruning_level,
            "total_heads": model_structure[0] * model_structure[1]
        }
        
        # Use the reporter to display pruning results
        self.reporter.display_pruning_results(pruning_results)
        
        # Store the pruning results
        self.cycle_results.append(pruning_results)
        
        # Update pruned heads information
        self.pruned_heads = pruning_results.get("pruned_heads", [])
        self.pruning_mask = pruning_results.get("pruning_mask", None)
        
        # Get final evaluation after pruning cycle
        self.final_loss, self.final_perplexity = self.reporter.evaluate_model(
            model=self.model,
            dataloader=self.validation_dataloader
        )
        
        # Save metrics to CSV
        if self.save_results:
            self.reporter.save_metrics_to_csv(
                metrics=self.metrics_history,
                filename=f"metrics_cycle{self.current_cycle}.csv",
                subfolder=f"cycle_{self.current_cycle}"
            )
        
        # Generate final dashboard and add path to results
        if dashboard_reporter:
            dashboard_path = dashboard_reporter.update_dashboard()
            if self.verbose:
                print(f"🔍 Final dashboard generated at: {dashboard_path}")
            # Add dashboard path to results
            pruning_results["dashboard_path"] = dashboard_path
        
        return pruning_results
    
    def run_training(self, steps=100, epochs=None):
        """
        Run additional training without pruning.
        
        Args:
            steps: Number of training steps per epoch
            epochs: Number of epochs (default is 1)
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None or self.train_dataloader is None:
            raise ValueError("Model and dataloaders must be set up first. Call setup() method.")
            
        if epochs is None:
            epochs = 1
            
        if self.verbose:
            print(f"\n=== Running Additional Training ({epochs} epochs) ===")
            
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = steps * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10, 
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        training_losses = []
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            # Create training progress bar if verbose
            if self.verbose:
                from tqdm import tqdm
                progress_bar = tqdm(total=steps, desc=f"Epoch {epoch+1}/{epochs}")
            
            # Iterate through batches
            for step, batch in enumerate(self.train_dataloader):
                if step >= steps:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Track loss
                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_steps += 1
                training_losses.append(loss_val)
                global_step += 1
                
                # Update progress bar if verbose
                if self.verbose:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss_val:.4f}")
                
                # Evaluate every 20% of steps
                if step % (steps // 5) == 0:
                    eval_loss, eval_perplexity = self.reporter.evaluate_model(
                        dataloader=self.validation_dataloader
                    )
                    
                    # Update metrics history
                    self.metrics_history["step"].append(global_step)
                    self.metrics_history["epoch"].append(epoch + 1)
                    self.metrics_history["train_loss"].append(loss_val)
                    self.metrics_history["eval_loss"].append(eval_loss)
                    self.metrics_history["perplexity"].append(eval_perplexity)
            
            # Close progress bar if verbose
            if self.verbose:
                progress_bar.close()
                
            # Print epoch summary
            if epoch_steps > 0:
                avg_loss = epoch_loss / epoch_steps
                if self.verbose:
                    print(f"Epoch {epoch+1}/{epochs} completed - Avg loss: {avg_loss:.4f}")
        
        # Final evaluation
        self.final_loss, self.final_perplexity = self.reporter.evaluate_model(
            dataloader=self.validation_dataloader
        )
        
        # Create and save training metrics visualization
        if self.verbose or self.save_results:
            training_fig = visualize_training_metrics(
                metrics_history=self.metrics_history,
                title="Training Metrics"
            )
            
            if self.verbose:
                plt.figure(training_fig.number)
                plt.show()
                
            if self.save_results:
                training_dir = os.path.join(self.output_dir, "training")
                os.makedirs(training_dir, exist_ok=True)
                training_fig.savefig(os.path.join(training_dir, "training_metrics.png"), dpi=100, bbox_inches='tight')
                
                # Save metrics to CSV
                self.reporter.save_metrics_to_csv(
                    metrics=self.metrics_history,
                    filename="training_metrics.csv",
                    subfolder="training"
                )
        
        # Return training metrics
        training_metrics = {
            "losses": training_losses,
            "initial_loss": training_losses[0] if training_losses else 0,
            "final_loss": training_losses[-1] if training_losses else 0,
            "final_eval_loss": self.final_loss,
            "final_perplexity": self.final_perplexity
        }
        
        return training_metrics
    
    def evaluate(self):
        """
        Perform a comprehensive evaluation of the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be set up first. Call setup() method.")
            
        if self.verbose:
            print("\n=== Evaluating Model ===")
        
        # Run full evaluation
        eval_loss, eval_perplexity = self.reporter.evaluate_model(
            dataloader=self.validation_dataloader
        )
        
        # Calculate improvement over baseline
        if self.baseline_perplexity is not None:
            improvement = (self.baseline_perplexity - eval_perplexity) / self.baseline_perplexity * 100
            if self.verbose:
                print(f"Improvement over baseline: {improvement:.2f}%")
        else:
            improvement = 0
            
        # Report model statistics
        if self.verbose:
            self.reporter.report_model_stats(
                model=self.model,
                sparsity=len(self.pruned_heads) / (self.model.config.num_hidden_layers * self.model.config.num_attention_heads) if hasattr(self.model, "config") else 0,
                pruned_heads=self.pruned_heads
            )
            
        # Save evaluation results
        if self.save_results:
            eval_dir = os.path.join(self.output_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Create a results summary file
            with open(os.path.join(eval_dir, "evaluation_summary.txt"), 'w') as f:
                f.write(f"=== Neural Plasticity Experiment Evaluation ===\n\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Dataset: {self.dataset}/{self.dataset_config}\n")
                f.write(f"Pruning Strategy: {self.pruning_strategy}\n")
                f.write(f"Pruning Level: {self.pruning_level:.2f}\n\n")
                
                f.write(f"Baseline Loss: {self.baseline_loss:.4f}\n")
                f.write(f"Baseline Perplexity: {self.baseline_perplexity:.2f}\n\n")
                
                f.write(f"Final Loss: {eval_loss:.4f}\n")
                f.write(f"Final Perplexity: {eval_perplexity:.2f}\n")
                f.write(f"Improvement: {improvement:.2f}%\n\n")
                
                f.write(f"Pruned Heads: {len(self.pruned_heads)}\n")
                f.write(f"Pruned Head Indices: {self.pruned_heads}\n")
        
        # Return evaluation metrics
        eval_metrics = {
            "loss": eval_loss,
            "perplexity": eval_perplexity,
            "baseline_loss": self.baseline_loss,
            "baseline_perplexity": self.baseline_perplexity,
            "improvement_percent": improvement,
            "pruned_heads": self.pruned_heads,
            "num_pruned_heads": len(self.pruned_heads)
        }
        
        return eval_metrics
    
    def generate_examples(self, prompts=None, max_length=100):
        """
        Generate text examples with the model.
        
        Args:
            prompts: Dictionary of prompt names to text, or list of prompts
            max_length: Maximum generation length
            
        Returns:
            Dictionary of generated texts
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set up first. Call setup() method.")
            
        if self.verbose:
            print("\n=== Generating Text Examples ===")
            
        # Use default prompts if none provided
        if prompts is None:
            prompts = {
                "story": "Once upon a time",
                "ai": "The future of artificial intelligence",
                "space": "In a distant galaxy",
                "science": "Scientists recently discovered"
            }
            
        # Run the generation suite
        generated_texts = self.reporter.run_text_generation_suite(
            prompts=prompts,
            max_length=max_length,
            save_to_file=self.save_results,
            subfolder="generation"
        )
        
        # Send generated examples to the sample callback
        if self.sample_callback:
            # If prompts is a dictionary
            if isinstance(prompts, dict):
                for name, prompt in prompts.items():
                    self.sample_callback(0, {
                        "name": name,
                        "input_text": prompt,
                        "predicted_tokens": generated_texts[prompt].split(),  # Simple tokenization for callback
                        "generated_text": generated_texts[prompt],
                        "model_type": "current"
                    })
            # If prompts is a list
            elif isinstance(prompts, list):
                for i, prompt in enumerate(prompts):
                    self.sample_callback(0, {
                        "name": f"example_{i}",
                        "input_text": prompt,
                        "predicted_tokens": generated_texts[prompt].split(),  # Simple tokenization for callback
                        "generated_text": generated_texts[prompt],
                        "model_type": "current"
                    })
        
        return generated_texts
        
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """
        Generate text using the current model (potentially pruned).
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set up first. Call setup() method.")
        
        # Generate text using reporter
        generated_text = self.reporter.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        # Send to callback if available
        if self.sample_callback:
            self.sample_callback(0, {
                "input_text": prompt,
                "generated_text": generated_text,
                "model_type": "pruned"
            })
            
        return generated_text
        
    def generate_baseline_text(self, prompt, max_length=100, temperature=0.7):
        """
        Generate text using the baseline model (before pruning).
        This requires that baseline metrics were captured during warmup.
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set up first. Call setup() method.")
        
        # Check if we have baseline metrics saved
        if not hasattr(self, 'baseline_metrics') or not self.baseline_metrics:
            if self.verbose:
                print("Warning: No baseline metrics found. Using current model for generation.")
            return self.generate_text(prompt, max_length, temperature)
            
        # Create checkpoint for current model state
        checkpoint = self.model.state_dict().copy()
        
        # Try to load the baseline checkpoint if available
        baseline_checkpoint_path = os.path.join(self.output_dir, "baseline_checkpoint.pt")
        if os.path.exists(baseline_checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(baseline_checkpoint_path))
                baseline_text = self.reporter.generate_text(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                
                # Restore current model
                self.model.load_state_dict(checkpoint)
                
                # Send to callback if available
                if self.sample_callback:
                    self.sample_callback(0, {
                        "input_text": prompt,
                        "generated_text": baseline_text,
                        "model_type": "baseline"
                    })
                
                return baseline_text
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load baseline checkpoint: {e}")
                    print("Using current model state for generation.")
                
                # Restore current model just in case
                self.model.load_state_dict(checkpoint)
                
                # Fall back to current model
                return self.generate_text(prompt, max_length, temperature)
        else:
            if self.verbose:
                print("No baseline checkpoint found. Using current model for generation.")
            return self.generate_text(prompt, max_length, temperature)
    
    def get_attention_maps(self, text=None, input_ids=None):
        """
        Get attention maps from both baseline and current models.
        
        Args:
            text: Text to analyze (tokenized if input_ids not provided)
            input_ids: Pre-tokenized input IDs (used if text is None)
            
        Returns:
            Dictionary with attention maps and related data
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set up first. Call setup() method.")
        
        # Tokenize text if provided
        if text is not None:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        elif input_ids is None:
            # Use a sample from validation data
            batch = next(iter(self.validation_dataloader))
            input_ids = batch["input_ids"][:1].to(self.device)
            
        # Get attention from current model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            
        if not hasattr(outputs, 'attentions') or not outputs.attentions:
            if self.verbose:
                print("Model does not output attention maps. Make sure to use output_attentions=True.")
            return None
        
        # Extract current model attention
        current_attentions = [attn.detach().cpu().numpy() for attn in outputs.attentions]
        
        # Store the current model state
        current_state = self.model.state_dict().copy()
        baseline_attentions = None
        
        # Try to load the baseline checkpoint if available
        baseline_checkpoint_path = os.path.join(self.output_dir, "baseline_checkpoint.pt")
        if os.path.exists(baseline_checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(baseline_checkpoint_path))
                
                # Get baseline model attention
                with torch.no_grad():
                    baseline_outputs = self.model(input_ids=input_ids, output_attentions=True)
                    
                if hasattr(baseline_outputs, 'attentions') and baseline_outputs.attentions:
                    baseline_attentions = [attn.detach().cpu().numpy() for attn in baseline_outputs.attentions]
                
                # Restore current model state
                self.model.load_state_dict(current_state)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to get baseline attention maps: {e}")
                # Ensure model is restored
                self.model.load_state_dict(current_state)
        
        # Format the output for visualization
        attention_data = {}
        
        # If we have both baseline and current attention maps, create comparison data
        if baseline_attentions is not None:
            num_layers = min(len(baseline_attentions), len(current_attentions))
            
            for layer_idx in range(num_layers):
                baseline_layer = baseline_attentions[layer_idx]
                current_layer = current_attentions[layer_idx]
                
                num_heads = min(baseline_layer.shape[1], current_layer.shape[1])
                
                for head_idx in range(num_heads):
                    # Skip heads that were pruned
                    if hasattr(self, 'pruned_heads') and (layer_idx, head_idx) in self.pruned_heads:
                        continue
                        
                    key = f"layer{layer_idx}_head{head_idx}"
                    attention_data[key] = {
                        "baseline": baseline_layer[0, head_idx].copy(),  # First batch item
                        "pruned": current_layer[0, head_idx].copy(),     # First batch item
                        "text": self.tokenizer.decode(input_ids[0]) if text is None else text,
                        "layer": layer_idx,
                        "head": head_idx
                    }
        
        return attention_data
    
    def save_model(self, path=None):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model (default: output_dir/model)
            
        Returns:
            Dictionary with save paths
        """
        if self.model is None:
            raise ValueError("Model must be set up first. Call setup() method.")
            
        if not self.save_results:
            if self.verbose:
                print("Skipping model saving because save_results=False")
            return None
            
        if path is None:
            path = os.path.join(self.output_dir, "model")
            
        os.makedirs(path, exist_ok=True)
        
        if self.verbose:
            print(f"\n=== Saving Model to {path} ===")
            
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save a simple model info file
        with open(os.path.join(path, "model_info.txt"), 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pruning Strategy: {self.pruning_strategy}\n")
            f.write(f"Pruning Level: {self.pruning_level:.2f}\n")
            f.write(f"Final Perplexity: {self.final_perplexity:.2f}\n")
            f.write(f"Pruned Heads: {len(self.pruned_heads)}\n")
        
        # Return save paths
        save_paths = {
            "model_dir": path,
            "config": os.path.join(path, "config.json"),
            "pytorch_model": os.path.join(path, "pytorch_model.bin"),
            "tokenizer": os.path.join(path, "tokenizer.json"),
            "model_info": os.path.join(path, "model_info.txt")
        }
        
        return save_paths
    
    def visualize_metrics_dashboard(self, figsize=(15, 10), save_path=None):
        """
        Create a comprehensive dashboard visualization of experiment metrics.
        
        Args:
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        if len(self.metrics_history["step"]) == 0:
            if self.verbose:
                print("No metrics history available for visualization.")
            return None
            
        # Create dashboard layout
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # Plot 1: Training and evaluation loss
        ax_loss = fig.add_subplot(gs[0, 0])
        ax_loss.plot(self.metrics_history["step"], self.metrics_history["train_loss"], 
                   label="Train Loss", color="blue")
        ax_loss.plot(self.metrics_history["step"], self.metrics_history["eval_loss"], 
                   label="Eval Loss", color="red")
        ax_loss.set_title("Training and Evaluation Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Plot 2: Perplexity
        ax_ppl = fig.add_subplot(gs[0, 1])
        ax_ppl.plot(self.metrics_history["step"], self.metrics_history["perplexity"], 
                  label="Perplexity", color="purple")
        ax_ppl.set_title("Perplexity Over Time")
        ax_ppl.set_xlabel("Step")
        ax_ppl.set_ylabel("Perplexity")
        ax_ppl.grid(True, alpha=0.3)
        
        # Plot 3: Pruned Heads Count
        ax_prune = fig.add_subplot(gs[1, 0])
        if "pruned_heads" in self.metrics_history and len(self.metrics_history["pruned_heads"]) > 0:
            ax_prune.plot(self.metrics_history["step"], self.metrics_history["pruned_heads"], 
                        label="Newly Pruned", color="red")
            ax_prune.set_title("Pruned Heads per Step")
            ax_prune.set_xlabel("Step")
            ax_prune.set_ylabel("Count")
            ax_prune.grid(True, alpha=0.3)
            
            # Add cumulative line
            if "total_pruned" in self.metrics_history and len(self.metrics_history["total_pruned"]) > 0:
                ax_prune_twin = ax_prune.twinx()
                ax_prune_twin.plot(self.metrics_history["step"], self.metrics_history["total_pruned"], 
                                 label="Total Pruned", color="orange", linestyle="--")
                ax_prune_twin.set_ylabel("Total Count")
                
                # Add both legends
                lines1, labels1 = ax_prune.get_legend_handles_labels()
                lines2, labels2 = ax_prune_twin.get_legend_handles_labels()
                ax_prune.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax_prune.text(0.5, 0.5, "No pruning data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_prune.transAxes)
        
        # Plot 4: Sparsity
        ax_sparsity = fig.add_subplot(gs[1, 1])
        if "sparsity" in self.metrics_history and len(self.metrics_history["sparsity"]) > 0:
            ax_sparsity.plot(self.metrics_history["step"], self.metrics_history["sparsity"], 
                           label="Model Sparsity", color="green")
            ax_sparsity.set_title("Model Sparsity Over Time")
            ax_sparsity.set_xlabel("Step")
            ax_sparsity.set_ylabel("Sparsity (%)")
            ax_sparsity.set_ylim(0, 100)
            ax_sparsity.grid(True, alpha=0.3)
        else:
            ax_sparsity.text(0.5, 0.5, "No sparsity data available", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_sparsity.transAxes)
        
        # Plot 5: Cycle Comparison (bar chart)
        ax_cycles = fig.add_subplot(gs[2, :])
        if len(self.cycle_results) > 0:
            cycles = list(range(1, len(self.cycle_results) + 1))
            baseline_ppl = [r.get("baseline_metrics", {}).get("perplexity", 0) for r in self.cycle_results]
            pruned_ppl = [r.get("pruned_metrics", {}).get("perplexity", 0) for r in self.cycle_results]
            final_ppl = [r.get("final_metrics", {}).get("perplexity", 0) for r in self.cycle_results]
            
            width = 0.25
            ax_cycles.bar([x - width for x in cycles], baseline_ppl, width, label="Baseline", color="blue")
            ax_cycles.bar(cycles, pruned_ppl, width, label="After Pruning", color="red")
            ax_cycles.bar([x + width for x in cycles], final_ppl, width, label="After Training", color="green")
            
            ax_cycles.set_title("Perplexity Comparison Across Cycles")
            ax_cycles.set_xlabel("Cycle")
            ax_cycles.set_ylabel("Perplexity")
            ax_cycles.set_xticks(cycles)
            ax_cycles.legend()
            ax_cycles.grid(True, alpha=0.3)
        else:
            ax_cycles.text(0.5, 0.5, "No cycle data available", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax_cycles.transAxes)
        
        # Add overall title
        fig.suptitle(f"Neural Plasticity Experiment Dashboard - {self.model_name}", fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)  # Adjust for title
        
        # Save visualization if path provided
        if save_path:
            try:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                if self.verbose:
                    print(f"✅ Saved metrics dashboard to {save_path}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error saving metrics dashboard: {e}")
        
        return fig

    def compare_pruning_strategies(self, strategies=None, pruning_levels=None, cycles=1, 
                                  training_steps=100, save_path=None):
        """
        Compare multiple pruning strategies and levels.
        
        This method runs a mini experiment for each combination of strategy and level,
        and compares their performance.
        
        Args:
            strategies: List of pruning strategies to compare
            pruning_levels: List of pruning levels to compare
            cycles: Number of pruning cycles for each experiment
            training_steps: Number of training steps per cycle
            save_path: Optional path to save the comparison results
            
        Returns:
            Dictionary with comparison results
        """
        if strategies is None:
            strategies = ["entropy", "gradient", "random", "combined"]
            
        if pruning_levels is None:
            pruning_levels = [0.1, 0.2, 0.3]
            
        if self.model is None or self.train_dataloader is None:
            raise ValueError("Model and dataloaders must be set up first. Call setup() method.")
            
        if self.verbose:
            print(f"\n=== Comparing {len(strategies)} Pruning Strategies at {len(pruning_levels)} Levels ===")
            
        # Store original strategy and level
        original_strategy = self.pruning_strategy
        original_level = self.pruning_level
        
        # Dictionary to store results for each configuration
        comparison_results = {}
        
        # Outer loop: strategies
        for strategy in strategies:
            strategy_results = {}
            
            if self.verbose:
                print(f"\nTesting strategy: {strategy}")
                
            # Inner loop: pruning levels
            for level in pruning_levels:
                if self.verbose:
                    print(f"  Testing pruning level: {level}")
                    
                # Set strategy and level
                self.pruning_strategy = strategy
                self.pruning_level = level
                
                # Create specific output directory for this experiment
                if self.save_results:
                    experiment_dir = os.path.join(self.output_dir, f"comparison/{strategy}_level{level}")
                    os.makedirs(experiment_dir, exist_ok=True)
                    
                    # Check if this configuration was already tested (saved results exist)
                    results_path = os.path.join(experiment_dir, "results.json")
                    if os.path.exists(results_path):
                        try:
                            import json
                            with open(results_path, 'r') as f:
                                experiment_results = json.load(f)
                                if self.verbose:
                                    print(f"    Loading cached results (perplexity: {experiment_results.get('final_perplexity', 'N/A')})")
                                strategy_results[level] = experiment_results
                                continue
                        except Exception as e:
                            if self.verbose:
                                print(f"    Could not load cached results: {e}")
                
                # Reset model to initial state
                if self.verbose:
                    print(f"    Reloading model...")
                self.model = None
                self.tokenizer = None
                self.baseline_loss = None
                self.baseline_perplexity = None
                self.setup()
                
                # Run warmup once (shared across all configurations)
                if not hasattr(self, 'warmup_results') or self.warmup_results is None:
                    if self.verbose:
                        print(f"    Running warm-up phase...")
                    self.run_warmup(max_epochs=1)
                else:
                    if self.verbose:
                        print(f"    Using existing warm-up results...")
                
                # Run pruning cycle(s)
                for cycle in range(cycles):
                    if self.verbose:
                        print(f"    Running pruning cycle {cycle+1}/{cycles}...")
                    self.run_pruning_cycle(training_steps=training_steps)
                
                # Evaluate final model
                eval_metrics = self.evaluate()
                
                # Store results for this configuration
                experiment_results = {
                    "strategy": strategy,
                    "pruning_level": level,
                    "baseline_perplexity": self.baseline_perplexity,
                    "final_perplexity": self.final_perplexity,
                    "improvement_percent": eval_metrics["improvement_percent"],
                    "pruned_heads_count": len(self.pruned_heads),
                    "pruned_heads": self.pruned_heads
                }
                
                strategy_results[level] = experiment_results
                
                # Save results for this configuration if requested
                if self.save_results:
                    import json
                    
                    # Use the specific experiment directory
                    experiment_dir = os.path.join(self.output_dir, f"comparison/{strategy}_level{level}")
                    os.makedirs(experiment_dir, exist_ok=True)
                    
                    # Save results as JSON
                    with open(os.path.join(experiment_dir, "results.json"), 'w') as f:
                        json.dump(experiment_results, f, indent=2)
            
            # Store results for this strategy
            comparison_results[strategy] = strategy_results
        
        # Create comparison visualization
        comparison_fig = self._create_strategy_comparison_visualization(comparison_results)
        
        # Save visualization if path provided
        if save_path and comparison_fig is not None:
            try:
                comparison_fig.savefig(save_path, dpi=100, bbox_inches='tight')
                if self.verbose:
                    print(f"✅ Saved strategy comparison to {save_path}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error saving strategy comparison: {e}")
        
        # Restore original strategy and level
        self.pruning_strategy = original_strategy
        self.pruning_level = original_level
        
        return {
            "comparison_results": comparison_results,
            "visualization": comparison_fig
        }
        
    def _create_strategy_comparison_visualization(self, comparison_results):
        """
        Create a visualization comparing different pruning strategies.
        
        Args:
            comparison_results: Dictionary with comparison results
            
        Returns:
            matplotlib Figure object
        """
        if not comparison_results:
            return None
            
        # Extract data for visualization
        strategies = list(comparison_results.keys())
        levels = list(comparison_results[strategies[0]].keys())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Perplexity vs. Pruning Level
        for strategy in strategies:
            x = []
            y = []
            for level in levels:
                x.append(float(level))
                y.append(comparison_results[strategy][level].get("final_perplexity", 0))
            
            ax1.plot(x, y, marker='o', label=strategy)
        
        ax1.set_title("Perplexity vs. Pruning Level")
        ax1.set_xlabel("Pruning Level")
        ax1.set_ylabel("Final Perplexity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement Percentage vs. Pruning Level
        for strategy in strategies:
            x = []
            y = []
            for level in levels:
                x.append(float(level))
                y.append(comparison_results[strategy][level].get("improvement_percent", 0))
            
            ax2.plot(x, y, marker='o', label=strategy)
        
        ax2.set_title("Improvement % vs. Pruning Level")
        ax2.set_xlabel("Pruning Level")
        ax2.set_ylabel("Improvement %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f"Pruning Strategy Comparison - {self.model_name}", fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Adjust for title
        
        return fig

    def save_metadata(self, filename="experiment_metadata.json"):
        """
        Save metadata about the experiment.
        
        Args:
            filename: Filename for the metadata JSON
            
        Returns:
            Path to saved metadata file or None if not saved
        """
        if not self.save_results or not self.output_dir:
            if self.verbose:
                print("Skipping metadata saving because save_results=False or output_dir is None")
            return None
        
        try:
            import time
            from datetime import datetime
            
            metadata = {
                "experiment_version": "1.0.0",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": {
                    "name": self.model_name,
                    "layers": self.model.config.num_hidden_layers if hasattr(self.model, "config") else None,
                    "heads": self.model.config.num_attention_heads if hasattr(self.model, "config") else None,
                    "parameters": sum(p.numel() for p in self.model.parameters())
                },
                "dataset": {
                    "name": self.dataset,
                    "config": self.dataset_config,
                    "batch_size": self.batch_size,
                    "max_length": self.max_length
                },
                "pruning": {
                    "strategy": self.pruning_strategy,
                    "level": self.pruning_level,
                    "pruned_heads_count": len(self.pruned_heads) if hasattr(self, "pruned_heads") else 0
                },
                "training": {
                    "learning_rate": self.learning_rate
                },
                "environment": {
                    "is_apple_silicon": IS_APPLE_SILICON,
                    "is_colab": IS_COLAB,
                    "has_gpu": HAS_GPU,
                    "device": str(self.device),
                    "pytorch_version": torch.__version__
                }
            }
            
            # Save to JSON
            metadata_path = os.path.join(self.output_dir, filename)
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if self.verbose:
                print(f"✅ Saved experiment metadata to {metadata_path}")
                
            return metadata_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving metadata: {e}")
            return None

    def run_multiple_pruning_cycles(self, num_cycles=3, training_steps=100):
        """
        Run multiple pruning cycles with continuous tracking of pruning state.
        
        This method simulates true neural plasticity by running multiple cycles of
        pruning, training, and evaluation, while tracking the cumulative pruning state.
        
        Args:
            num_cycles: Number of pruning cycles to run
            training_steps: Number of training steps per pruning cycle
            
        Returns:
            Dictionary with cycle-by-cycle results and tracking metrics
        """
        # Tracking structures for pruning state
        all_pruning_results = []
        cumulative_pruned_heads = []
        cycle_metrics = {
            "perplexity": [],
            "pruned_heads_count": [],
            "loss": [],
            "newly_pruned_count": []
        }
        
        # Get model structure info
        if hasattr(self.model, 'config'):
            total_heads = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        else:
            # Estimate from first cycle results
            total_heads = None
        
        # Notify metrics callback we're starting pruning phase
        if self.metrics_callback:
            self.metrics_callback(0, {
                "phase": "pruning_start",
                "num_cycles": num_cycles,
                "training_steps": training_steps
            })
        
        for cycle in range(num_cycles):
            if self.verbose:
                print(f"Running pruning cycle {cycle+1}/{num_cycles}")
            
            # Notify cycle start via callback
            if self.metrics_callback:
                self.metrics_callback(cycle, {
                    "phase": "pruning_cycle_start",
                    "cycle": cycle + 1,
                    "total_cycles": num_cycles
                })
            
            # Run a pruning cycle with training
            pruning_results = self.run_pruning_cycle(training_steps=training_steps)
            
            # Store the results for this cycle
            all_pruning_results.append(pruning_results)
            
            # Extract newly pruned heads
            newly_pruned = pruning_results.get("pruned_heads", [])
            
            # Track cumulative pruned heads (avoiding duplicates)
            newly_pruned_count = 0
            for head in newly_pruned:
                if head not in cumulative_pruned_heads:
                    cumulative_pruned_heads.append(head)
                    newly_pruned_count += 1
            
            # Track metrics for this cycle
            cycle_perplexity = pruning_results.get("final_metrics", {}).get("perplexity", 0)
            cycle_loss = pruning_results.get("final_metrics", {}).get("loss", 0)
            pruned_heads_count = len(cumulative_pruned_heads)
            
            cycle_metrics["perplexity"].append(cycle_perplexity)
            cycle_metrics["pruned_heads_count"].append(pruned_heads_count)
            cycle_metrics["loss"].append(cycle_loss)
            cycle_metrics["newly_pruned_count"].append(newly_pruned_count)
            
            # Calculate sparsity if we have total_heads info
            cycle_sparsity = None
            if total_heads is not None:
                if "sparsity" not in cycle_metrics:
                    cycle_metrics["sparsity"] = []
                cycle_sparsity = len(cumulative_pruned_heads) / total_heads * 100
                cycle_metrics["sparsity"].append(cycle_sparsity)
            
            # If we don't have total_heads yet but have results, try to extract it
            if total_heads is None and pruning_results.get("model_structure") is not None:
                num_layers, num_heads = pruning_results.get("model_structure")
                total_heads = num_layers * num_heads
                # Backfill sparsity calculations
                cycle_metrics["sparsity"] = [count / total_heads * 100 for count in cycle_metrics["pruned_heads_count"]]
                cycle_sparsity = cycle_metrics["sparsity"][-1]
            
            # Create visualization showing cumulative pruning state
            from utils.neural_plasticity.visualization import create_pruning_state_heatmap
            heatmap_fig = create_pruning_state_heatmap(
                model=self.model,
                cumulative_pruned=cumulative_pruned_heads,
                newly_pruned=newly_pruned,
                title=f"Neural Plasticity State After Cycle {cycle+1}/{num_cycles}"
            )
            
            # Send cycle results to callback
            if self.metrics_callback:
                # Notify about newly pruned heads and entropy/gradient values
                self.metrics_callback(cycle, {
                    "phase": "pruning_cycle_complete",
                    "cycle": cycle + 1,
                    "perplexity": cycle_perplexity,
                    "loss": cycle_loss,
                    "pruned_heads_count": pruned_heads_count,
                    "newly_pruned_count": newly_pruned_count,
                    "sparsity": cycle_sparsity,
                    "pruned_heads": newly_pruned
                })
                
                # Send entropy and gradient data if available
                if "entropy_values" in pruning_results and isinstance(pruning_results["entropy_values"], torch.Tensor):
                    self.metrics_callback(cycle, {
                        "entropy_values": pruning_results["entropy_values"].detach().cpu().numpy()
                    })
                
                if "grad_norm_values" in pruning_results and isinstance(pruning_results["grad_norm_values"], torch.Tensor):
                    self.metrics_callback(cycle, {
                        "grad_norm_values": pruning_results["grad_norm_values"].detach().cpu().numpy()
                    })
            
            # Save the heatmap
            if self.save_results:
                cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle+1}")
                os.makedirs(cycle_dir, exist_ok=True)
                heatmap_fig.savefig(os.path.join(cycle_dir, "pruning_state_heatmap.png"), dpi=100, bbox_inches='tight')
        
        # Create visualization of metrics evolution across cycles
        if num_cycles > 1:
            evolution_fig = self._create_metrics_evolution_plot(cycle_metrics, num_cycles)
            if self.save_results:
                evolution_fig.savefig(os.path.join(self.output_dir, "metrics_evolution.png"), dpi=100, bbox_inches='tight')
        
        # Notify all pruning cycles complete
        if self.metrics_callback:
            self.metrics_callback(num_cycles, {
                "phase": "pruning_complete",
                "total_pruned": len(cumulative_pruned_heads),
                "sparsity": len(cumulative_pruned_heads) / total_heads * 100 if total_heads else None,
                "final_perplexity": cycle_metrics["perplexity"][-1] if cycle_metrics["perplexity"] else None,
                "final_loss": cycle_metrics["loss"][-1] if cycle_metrics["loss"] else None
            })
        
        # Return comprehensive results
        return {
            "all_pruning_results": all_pruning_results,
            "cumulative_pruned_heads": cumulative_pruned_heads,
            "cycle_metrics": cycle_metrics,
            "num_cycles": num_cycles,
            "total_pruned": len(cumulative_pruned_heads),
            "sparsity": len(cumulative_pruned_heads) / total_heads * 100 if total_heads else None
        }
    
    def _create_metrics_evolution_plot(self, cycle_metrics, num_cycles):
        """
        Create visualization of metrics evolution across pruning cycles.
        
        Args:
            cycle_metrics: Dictionary of metrics tracked per cycle
            num_cycles: Number of cycles run
            
        Returns:
            matplotlib Figure with evolution plots
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Create a 2x2 grid of subplots
        plt.subplot(2, 2, 1)
        plt.plot(range(1, num_cycles+1), cycle_metrics["perplexity"], marker='o')
        plt.title("Perplexity Evolution")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Perplexity")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(range(1, num_cycles+1), cycle_metrics["pruned_heads_count"], marker='o', color='r')
        plt.title("Cumulative Pruned Heads")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Number of Pruned Heads")
        plt.grid(True, alpha=0.3)
        
        # Plot sparsity if available
        if "sparsity" in cycle_metrics:
            plt.subplot(2, 2, 3)
            plt.plot(range(1, num_cycles+1), cycle_metrics["sparsity"], marker='o', color='g')
            plt.title("Model Sparsity")
            plt.xlabel("Pruning Cycle")
            plt.ylabel("Sparsity (%)")
            plt.grid(True, alpha=0.3)
        
        # Plot loss evolution
        plt.subplot(2, 2, 4)
        plt.plot(range(1, num_cycles+1), cycle_metrics["loss"], marker='o', color='purple')
        plt.title("Loss Evolution")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle("Neural Plasticity Evolution Across Cycles", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def run_full_experiment(self, warmup_epochs=1, pruning_cycles=3, training_steps=100):
        """
        Run a complete neural plasticity experiment pipeline.
        
        This is a convenience method that chains together the individual
        experiment steps into a complete workflow.
        
        Args:
            warmup_epochs: Number of warmup epochs
            pruning_cycles: Number of pruning cycles to run
            training_steps: Number of training steps per pruning cycle
            
        Returns:
            Dictionary with full experiment results
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"=== Starting Full Neural Plasticity Experiment ===")
            print(f"Model: {self.model_name}")
            print(f"Dataset: {self.dataset}/{self.dataset_config}")
            print(f"Pruning Strategy: {self.pruning_strategy}")
            print(f"Pruning Level: {self.pruning_level:.2f}")
            print(f"Output Directory: {self.output_dir}")
            print(f"Device: {self.device}")
            
        # Save experiment metadata
        if self.save_results:
            self.save_metadata()
            
        # Run each step of the experiment
        self.setup()
        warmup_results = self.run_warmup(max_epochs=warmup_epochs)
        attention_analysis = self.analyze_attention()
        
        # Run multiple pruning cycles with continuous tracking
        multi_cycle_results = self.run_multiple_pruning_cycles(
            num_cycles=pruning_cycles,
            training_steps=training_steps
        )
        
        # Save final metrics dashboard
        if self.save_results:
            dashboard_path = os.path.join(self.output_dir, "final_metrics_dashboard.png")
            self.visualize_metrics_dashboard(save_path=dashboard_path)
            
        # Final evaluation and generation
        eval_metrics = self.evaluate()
        generated_texts = self.generate_examples()
        
        # Save the model
        if self.save_results:
            save_paths = self.save_model()
            
        # Create a final results structure
        results = {
            "baseline_metrics": {
                "loss": self.baseline_loss,
                "perplexity": self.baseline_perplexity
            },
            "final_metrics": {
                "loss": self.final_loss,
                "perplexity": self.final_perplexity
            },
            "improvement_percent": eval_metrics["improvement_percent"],
            "pruned_heads": self.pruned_heads,
            "cycle_results": self.cycle_results,
            "metrics_history": self.metrics_history,
            "multi_cycle_results": multi_cycle_results,
            "execution_time": time.time() - start_time
        }
        
        # Save final results
        if self.save_results:
            import json
            
            # Convert torch tensors to lists for JSON serialization
            def tensor_to_list(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: tensor_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [tensor_to_list(item) for item in obj]
                else:
                    return obj
            
            # Create a serializable version of the results
            serializable_results = tensor_to_list(results)
            
            # Save to JSON
            with open(os.path.join(self.output_dir, "experiment_results.json"), 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            # Create a summary file with key metrics
            with open(os.path.join(self.output_dir, "experiment_summary.txt"), 'w') as f:
                f.write(f"=== Neural Plasticity Experiment Summary ===\n\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Dataset: {self.dataset}/{self.dataset_config}\n")
                f.write(f"Pruning Strategy: {self.pruning_strategy}\n")
                f.write(f"Pruning Level: {self.pruning_level:.2f}\n\n")
                
                f.write(f"Baseline Loss: {self.baseline_loss:.4f}\n")
                f.write(f"Baseline Perplexity: {self.baseline_perplexity:.2f}\n\n")
                
                f.write(f"Final Loss: {self.final_loss:.4f}\n")
                f.write(f"Final Perplexity: {self.final_perplexity:.2f}\n")
                f.write(f"Improvement: {eval_metrics['improvement_percent']:.2f}%\n\n")
                
                f.write(f"Pruned Heads: {len(self.pruned_heads)}\n")
                f.write(f"Total Cycles Run: {pruning_cycles}\n")
                f.write(f"Execution Time: {(time.time() - start_time)/60:.1f} minutes\n")
                
            if self.verbose:
                print(f"\n✅ Full experiment completed successfully!")
                print(f"Results saved to: {self.output_dir}")
                print(f"Execution time: {(time.time() - start_time)/60:.1f} minutes")
        
        return results


def run_neural_plasticity_experiment(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
    output_dir: str = "plasticity_results",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    pruning_strategy: str = "entropy",
    prune_percent: float = 0.2,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    training_steps: int = 500,
    use_differential_lr: bool = True,
    num_cycles: int = 3,
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Run a complete neural plasticity experiment.

    Args:
        model_name: Name of the model to use
        device: Device to use (auto-detected if None)
        output_dir: Directory to save results
        dataset_name: Name of the dataset to use
        dataset_config: Dataset configuration
        pruning_strategy: Pruning strategy to use
        prune_percent: Percentage of heads to prune (0-1)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        training_steps: Number of training steps per cycle
        use_differential_lr: Whether to use differential learning rates
        num_cycles: Number of plasticity cycles to run
        create_visualizations: Whether to create and save visualizations

    Returns:
        Dictionary with experiment results
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{model_name.split('/')[-1]}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    if create_visualizations:
        visualization_dir = os.path.join(experiment_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Create dataloader builder
    dataloader_builder = get_dataloader_builder(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        model_name=model_name,
        max_length=128,
        batch_size=batch_size
    )
    
    # Create dataloaders
    train_dataloader, eval_dataloader = dataloader_builder()
    
    # Create model and calculate baseline
    baseline = create_model_and_entropy_baseline(
        model_name=model_name,
        device=device,
        eval_dataloader=eval_dataloader
    )
    
    model = baseline["model"]
    device = baseline["device"]
    entropy_values = baseline["entropy_values"]
    grad_norm_values = baseline["grad_norm_values"]
    baseline_metrics = baseline["baseline_metrics"]
    
    # Save baseline metrics
    baseline_file = os.path.join(experiment_dir, "baseline_metrics.txt")
    with open(baseline_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Baseline Loss: {baseline_metrics['loss']:.4f}\n")
        f.write(f"Baseline Perplexity: {baseline_metrics['perplexity']:.2f}\n")
    
    # Create baseline visualization
    if create_visualizations:
        baseline_plot = plot_baseline_entropy(
            entropy_values=entropy_values,
            grad_norm_values=grad_norm_values,
            save_path=os.path.join(visualization_dir, "baseline_metrics.png")
        )
    
    # Run plasticity cycles
    cycle_results = []
    current_metrics = baseline_metrics
    best_metrics = baseline_metrics
    best_model_path = None
    
    for cycle in range(num_cycles):
        print(f"\n--- Plasticity Cycle {cycle+1}/{num_cycles} ---")
        
        # Run plasticity loop
        result = run_plasticity_loop(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            pruning_level=prune_percent,
            strategy=pruning_strategy,
            learning_rate=learning_rate,
            training_steps=training_steps,
            use_differential_lr=use_differential_lr
        )
        
        # Update current metrics
        current_metrics = result["final_metrics"]
        
        # Save cycle results
        cycle_results.append({
            "cycle": cycle + 1,
            "baseline_metrics": baseline_metrics,
            "pruned_metrics": result["pruned_metrics"],
            "final_metrics": result["final_metrics"],
            "pruned_heads": result["pruned_heads"],
            "perplexity_improvement": result["perplexity_improvement"],
            "recovery_rate": result["recovery_rate"]
        })
        
        # Create cycle visualization
        if create_visualizations:
            # Visualize pruning decisions
            pruning_viz = visualize_pruning_decisions(
                grad_norm_values=result["grad_norm_values"],
                pruning_mask=result["pruning_mask"],
                title=f"Cycle {cycle+1} Pruning Decisions ({len(result['pruned_heads'])} heads)",
                save_path=os.path.join(visualization_dir, f"cycle{cycle+1}_pruning.png")
            )
        
        # Check if this is the best model so far
        if current_metrics["perplexity"] < best_metrics["perplexity"]:
            best_metrics = current_metrics
            
            # Save best model
            best_model_path = os.path.join(experiment_dir, f"model_best_cycle{cycle+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            
            print(f"New best model (cycle {cycle+1}) - Perplexity: {best_metrics['perplexity']:.2f}")
    
    # Calculate overall improvement
    overall_improvement = (baseline_metrics["perplexity"] - best_metrics["perplexity"]) / baseline_metrics["perplexity"]
    
    # Save final results
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "pruning_strategy": pruning_strategy,
        "prune_percent": prune_percent,
        "learning_rate": learning_rate,
        "training_steps": training_steps,
        "num_cycles": num_cycles,
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "overall_improvement": overall_improvement,
        "cycle_results": cycle_results,
        "best_model_path": best_model_path
    }
    
    # Print final results
    print("\n=== Experiment Results ===")
    print(f"Baseline Perplexity: {baseline_metrics['perplexity']:.2f}")
    print(f"Final Perplexity: {best_metrics['perplexity']:.2f}")
    print(f"Overall Improvement: {overall_improvement*100:.2f}%")
    print(f"Results saved to: {experiment_dir}")
    
    return results