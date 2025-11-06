"""
AdaptiveOptimizer - Core optimizer for transformer model pruning and regrowth.

This module provides the main entry point for model optimization, handling:
- Model loading and preparation
- Pruning strategy selection and application
- Fine-tuning with adaptive learning rates
- Head regrowth and continuous optimization
- Progress tracking and visualization
"""

import os
import torch
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

from ..metrics.tracker import ProgressTracker
from ..strategies.entropy import entropy_based_pruning
from ..strategies.magnitude import magnitude_based_pruning
from ..strategies.random import random_pruning
from ..utils.model_utils import load_model_and_tokenizer
from ..utils.training import fine_tune_model, evaluate_model
from ..utils.generation import generate_text

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveOptimizerConfig:
    """Configuration for the AdaptiveOptimizer."""
    # Model configuration
    model_name: str = "distilgpt2"
    cache_dir: Optional[str] = None
    
    # Optimization parameters
    pruning_ratio: float = 0.3
    growth_ratio: float = 0.1
    strategy: str = "entropy"  # "entropy", "magnitude", "random"
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation: int = 1
    epochs_per_cycle: int = 1
    max_cycles: Optional[int] = None
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output configuration
    output_dir: str = "./upgrayedd_output"
    save_frequency: int = 1
    eval_frequency: int = 1
    
    # Advanced options
    use_differential_lr: bool = True
    compress_model: bool = False
    compression_type: str = "mask"
    
    # Dataset options
    dataset: str = "wikitext"
    dataset_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for k, v in self.__dict__.items():
            # Convert torch.device to string if needed
            if k == "device" and isinstance(v, torch.device):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdaptiveOptimizerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AdaptiveOptimizerConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class AdaptiveOptimizer:
    """
    Core optimizer for transformer model pruning and regrowth.
    
    This class manages the complete optimization lifecycle, including:
    - Loading and preparing models
    - Pruning based on selected strategies
    - Fine-tuning with differential learning rates
    - Regrowth of pruned heads
    - Visualization and progress tracking
    """
    
    def __init__(
        self, 
        config: Union[Dict[str, Any], AdaptiveOptimizerConfig] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ):
        """
        Initialize the optimizer.
        
        Args:
            config: Configuration for the optimizer
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            **kwargs: Additional config parameters when not providing a config object
        """
        # Set up configuration
        if config is None:
            self.config = AdaptiveOptimizerConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = AdaptiveOptimizerConfig(**config)
        else:
            self.config = config
            
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, "config.json")
        self.config.save(config_path)
        
        # Initialize tracker
        self.tracker = ProgressTracker(output_dir=self.config.output_dir)
        
        # Set up model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        
        # Track optimization state
        self.current_cycle = 0
        self.baseline_metrics = None
        self.current_metrics = None
        self.pruned_heads = []
        self.active_heads = []
        
        # Load data if needed
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Log initialization
        logger.info(f"Initialized AdaptiveOptimizer with {self.config.model_name}")
        logger.info(f"Output directory: {self.config.output_dir}")
        
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                device=self.config.device
            )
        
        return self.model, self.tokenizer
    
    def load_data(self):
        """Load and prepare datasets."""
        if self.train_dataloader is None or self.val_dataloader is None:
            logger.info(f"Loading dataset: {self.config.dataset}")
            
            # Import here to avoid circular imports
            from ..utils.data import load_and_prepare_data
            
            self.train_dataloader, self.val_dataloader = load_and_prepare_data(
                self.config.dataset,
                self.tokenizer,
                batch_size=self.config.batch_size,
                dataset_path=self.config.dataset_path
            )
            
        return self.train_dataloader, self.val_dataloader
    
    def prune_model(self) -> List[Tuple[int, int]]:
        """
        Prune model based on selected strategy.
        
        Returns:
            List of (layer_idx, head_idx) tuples for pruned heads
        """
        logger.info(f"Pruning model using {self.config.strategy} strategy")
        
        # Import strategy modules
        if self.config.strategy == "entropy":
            pruning_fn = entropy_based_pruning
        elif self.config.strategy == "magnitude":
            pruning_fn = magnitude_based_pruning
        elif self.config.strategy == "random":
            pruning_fn = random_pruning
        else:
            raise ValueError(f"Unknown pruning strategy: {self.config.strategy}")
        
        # Apply pruning
        pruned_heads = pruning_fn(
            self.model, 
            self.val_dataloader,
            prune_ratio=self.config.pruning_ratio,
            device=self.config.device
        )
        
        # Save pruned heads
        self.pruned_heads = pruned_heads
        
        # Log pruning results
        logger.info(f"Pruned {len(pruned_heads)} heads")
        
        return pruned_heads
    
    def fine_tune(self) -> Dict[str, Any]:
        """
        Fine-tune the model after pruning.
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Fine-tuning model for {self.config.epochs_per_cycle} epochs")
        
        # Fine-tune the model
        results = fine_tune_model(
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.tokenizer,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.epochs_per_cycle,
            device=self.config.device,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            use_differential_lr=self.config.use_differential_lr,
            progress_tracker=self.tracker
        )
        
        # Update current metrics
        self.current_metrics = {
            "loss": results["final_loss"],
            "perplexity": results["final_perplexity"]
        }
        
        return results
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate current model performance.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Evaluate the model
        loss, perplexity = evaluate_model(
            self.model, 
            self.val_dataloader,
            device=self.config.device
        )
        
        # Generate example text
        prompt = "Artificial intelligence is becoming increasingly important because"
        generated_text = generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_length=100
        )
        
        # Update tracker
        self.tracker.add_generated_text(generated_text, step=self.current_cycle)
        
        # Save metrics
        metrics = {
            "loss": loss,
            "perplexity": perplexity,
            "generated_text": generated_text
        }
        
        return metrics
    
    def save_checkpoint(self, suffix: str = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            suffix: Optional suffix to add to checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        # Determine checkpoint path
        if suffix:
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{suffix}")
        else:
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.current_cycle}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        state = {
            "cycle": self.current_cycle,
            "pruned_heads": self.pruned_heads,
            "baseline_metrics": self.baseline_metrics,
            "current_metrics": self.current_metrics
        }
        
        with open(os.path.join(checkpoint_dir, "optimizer_state.json"), "w") as f:
            json.dump(state, f, indent=2)
            
        # Save tracker state
        self.tracker.save(os.path.join(checkpoint_dir, "tracker_state.json"))
        
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load optimizer state from checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load optimizer state
        state_path = os.path.join(checkpoint_dir, "optimizer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            
            self.current_cycle = state.get("cycle", 0)
            self.pruned_heads = state.get("pruned_heads", [])
            self.baseline_metrics = state.get("baseline_metrics", None)
            self.current_metrics = state.get("current_metrics", None)
        
        # Load tracker state
        tracker_path = os.path.join(checkpoint_dir, "tracker_state.json")
        if os.path.exists(tracker_path):
            self.tracker.load(tracker_path)
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """
        Run a single optimization cycle (prune, fine-tune, evaluate).
        
        Returns:
            Dictionary with cycle results
        """
        self.current_cycle += 1
        logger.info(f"Starting optimization cycle {self.current_cycle}")
        
        # Ensure model and data are loaded
        self.load_model()
        self.load_data()
        
        # Establish baseline if needed
        if self.baseline_metrics is None:
            logger.info("Establishing baseline metrics")
            baseline = self.evaluate()
            self.baseline_metrics = baseline
            self.tracker.add_metrics(
                step=0, 
                loss=baseline["loss"], 
                perplexity=baseline["perplexity"]
            )
        
        # Prune model
        pruned_heads = self.prune_model()
        
        # Evaluate after pruning
        post_pruning = self.evaluate()
        
        # Update tracker with post-pruning metrics
        self.tracker.add_metrics(
            step=self.current_cycle * 10 - 5,  # Halfway point in cycle
            loss=post_pruning["loss"],
            perplexity=post_pruning["perplexity"],
            pruned_heads=pruned_heads
        )
        
        # Fine-tune model
        fine_tune_results = self.fine_tune()
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        # Update tracker with final metrics
        self.tracker.add_metrics(
            step=self.current_cycle * 10,
            loss=final_metrics["loss"],
            perplexity=final_metrics["perplexity"]
        )
        
        # Save checkpoint if needed
        if self.current_cycle % self.config.save_frequency == 0:
            self.save_checkpoint()
        
        # Return cycle results
        cycle_results = {
            "cycle": self.current_cycle,
            "pruned_heads": pruned_heads,
            "pruned_head_count": len(pruned_heads),
            "baseline_metrics": self.baseline_metrics,
            "post_pruning_metrics": post_pruning,
            "final_metrics": final_metrics,
            "fine_tune_results": fine_tune_results
        }
        
        return cycle_results
    
    def run_continuous_optimization(
        self, 
        max_cycles: Optional[int] = None,
        cycle_callback: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Run continuous optimization cycles until stopped or max_cycles reached.
        
        Args:
            max_cycles: Maximum number of cycles to run
            cycle_callback: Callback function after each cycle, should return True to continue
            
        Returns:
            Dictionary with overall results
        """
        # Use config max_cycles if not specified
        if max_cycles is None:
            max_cycles = self.config.max_cycles
        
        # Initialize final results
        results = {
            "cycles": [],
            "baseline_metrics": None,
            "final_metrics": None,
            "improvement": None
        }
        
        try:
            # Run cycles until max_cycles or interrupted
            continue_running = True
            while continue_running:
                if max_cycles is not None and self.current_cycle >= max_cycles:
                    logger.info(f"Reached maximum cycles ({max_cycles}), stopping")
                    break
                
                # Run cycle
                cycle_results = self.run_optimization_cycle()
                
                # Save baseline if first cycle
                if self.current_cycle == 1:
                    results["baseline_metrics"] = self.baseline_metrics
                
                # Add cycle results
                results["cycles"].append(cycle_results)
                
                # Update final metrics
                results["final_metrics"] = cycle_results["final_metrics"]
                
                # Calculate improvement
                baseline_ppl = results["baseline_metrics"]["perplexity"]
                final_ppl = results["final_metrics"]["perplexity"]
                improvement = (baseline_ppl - final_ppl) / baseline_ppl * 100
                results["improvement"] = improvement
                
                logger.info(f"Cycle {self.current_cycle} completed. Current improvement: {improvement:.2f}%")
                
                # Call callback if provided
                if cycle_callback:
                    continue_running = cycle_callback(cycle_results)
        
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        finally:
            # Save final checkpoint
            logger.info("Saving final checkpoint")
            self.save_checkpoint(suffix="final")
            
            # Final visualization
            self.tracker.create_plots()
        
        return results