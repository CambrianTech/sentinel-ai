"""
Neural Plasticity Experiment Module

This module implements the main experiment interface for neural plasticity
studies. It serves as the primary entry point for running pruning and 
fine-tuning experiments while keeping visualization completely separate
from the core functionality.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import internal modules
from .plasticity_tracker import PlasticityTracker
from .fine_tuner import AdaptiveFinetuner
from .pruning import EntropyPruner, MagnitudePruner, PruningStrategy
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class PlasticityExperiment:
    """
    Orchestrates complete plasticity experiments.
    
    This class coordinates:
    1. Model preparation and pruning
    2. Fine-tuning with adaptive learning rates
    3. Analysis of plasticity and regrowth patterns
    4. Result storage
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: str = None,
        adaptive_model: bool = True
    ):
        """
        Initialize a plasticity experiment.
        
        Args:
            model_name: Name of the pre-trained model to use
            output_dir: Directory to save results
            device: Device to run on (auto-detected if None)
            adaptive_model: Whether to use adaptive model wrapper (recommended)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adaptive_model = adaptive_model
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Visualization is handled separately
        self.visualizer = None
        
    def _load_model(self):
        """
        Load the model based on experiment settings.
        
        Returns:
            Loaded model
        """
        if self.adaptive_model:
            # Import inside function to avoid circular imports
            from models.loaders.loader import load_baseline_model, load_adaptive_model
            
            # Load with adaptive wrapper
            baseline_model = load_baseline_model(self.model_name, self.device)
            model = load_adaptive_model(self.model_name, baseline_model, self.device)
            return model
        else:
            # Import inside function to avoid circular imports
            from models.loaders.loader import load_baseline_model
            
            # Load standard model
            return load_baseline_model(self.model_name, self.device)
    
    def run_experiment(
        self,
        pruning_strategy: str,
        pruning_level: float,
        dataloader_builder_fn: Callable,
        fine_tuning_steps: int = 500,
        learning_rate: float = 5e-5,
        use_differential_lr: bool = True,
        output_dir: Optional[str] = None,
        batch_size: int = 4,
        generate_visualizations: bool = False,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete neural plasticity experiment.
        
        Args:
            pruning_strategy: Strategy to use for pruning ('entropy' or 'magnitude')
            pruning_level: Percentage of heads to prune (0.0 to 1.0)
            dataloader_builder_fn: Function that returns (train_dataloader, eval_dataloader)
            fine_tuning_steps: Number of fine-tuning steps
            learning_rate: Base learning rate for fine-tuning
            use_differential_lr: Use higher learning rate for pruned heads
            output_dir: Directory to save results (overrides instance output_dir)
            batch_size: Batch size for training and evaluation
            generate_visualizations: Whether to generate visualizations
            experiment_id: Unique identifier for this experiment
            
        Returns:
            Dictionary containing experiment results
        """
        # Use output_dir if provided, otherwise use instance output_dir
        output_dir = output_dir or str(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{pruning_strategy}_{pruning_level}_{timestamp}"
            
        # Create experiment directory
        experiment_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save experiment parameters
        params = {
            "model_name": self.model_name,
            "pruning_strategy": pruning_strategy,
            "pruning_level": pruning_level,
            "fine_tuning_steps": fine_tuning_steps,
            "learning_rate": learning_rate,
            "use_differential_lr": use_differential_lr,
            "batch_size": batch_size,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "adaptive_model": self.adaptive_model
        }
        
        with open(os.path.join(experiment_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
            
        # Initialize visualization if requested
        if generate_visualizations:
            try:
                # Dynamically import visualizers to avoid forcing the dependency
                from .visualization import (
                    MetricsVisualizer, 
                    EntropyVisualizer, 
                    PruningVisualizer, 
                    HeadRecoveryVisualizer
                )
                
                # Create visualizers
                metrics_viz = MetricsVisualizer(output_dir)
                entropy_viz = EntropyVisualizer(output_dir)
                pruning_viz = PruningVisualizer(output_dir)
                recovery_viz = HeadRecoveryVisualizer(output_dir)
                
                # Store in a dictionary for easy access
                self.visualizer = {
                    "metrics": metrics_viz,
                    "entropy": entropy_viz,
                    "pruning": pruning_viz,
                    "recovery": recovery_viz
                }
            except ImportError:
                logger.warning("Visualization modules not available. Continuing without visualization.")
                generate_visualizations = False
                
        # Load model
        logger.info(f"Loading model: {self.model_name}")
        model = self._load_model()
        
        # Create dataloaders
        logger.info("Creating dataloaders")
        train_dataloader, eval_dataloader = dataloader_builder_fn(batch_size=batch_size)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(device=self.device)
        
        # Evaluate baseline model
        logger.info("Evaluating baseline model")
        baseline_metrics = evaluator.evaluate(model, eval_dataloader)
        logger.info(f"Baseline metrics: {baseline_metrics}")
        
        # Apply pruning
        logger.info(f"Applying {pruning_strategy} pruning with level {pruning_level}")
        pruner = self._create_pruner(pruning_strategy)
        
        # Collect attention distributions for entropy pruning
        if pruning_strategy == "entropy":
            # Collect entropy data
            entropy_data = pruner.collect_distributions(model, eval_dataloader)
            
            # Save pre-pruning entropy
            entropy_dir = os.path.join(experiment_dir, "pre_entropy.json")
            serializable_entropy = {}
            for layer_idx, entropy in entropy_data.items():
                serializable_entropy[str(layer_idx)] = entropy.tolist()
            
            with open(entropy_dir, "w") as f:
                json.dump(serializable_entropy, f)
                
            # Visualize pre-pruning entropy if requested
            if generate_visualizations and self.visualizer is not None:
                self.visualizer["entropy"].visualize_entropy_heatmap(
                    entropy_data, experiment_id, phase="pre_pruning"
                )
            
            # Perform pruning
            pruned_heads = pruner.prune(model, distributions=entropy_data, prune_percent=pruning_level)
        else:
            # Perform magnitude-based pruning
            pruned_heads = pruner.prune(model, prune_percent=pruning_level)
            
        logger.info(f"Pruned {len(pruned_heads)} heads")
        
        # Save pruned heads
        pruned_heads_path = os.path.join(experiment_dir, "pruned_heads.json")
        with open(pruned_heads_path, "w") as f:
            json.dump([(l, h, float(s)) for l, h, s in pruned_heads], f, indent=2)
            
        # Visualize pruned heads if requested
        if generate_visualizations and self.visualizer is not None:
            # Detect model structure for visualization
            num_layers, num_heads = pruner.detect_model_structure(model)
            
            self.visualizer["pruning"].visualize_pruned_heads(
                pruned_heads, num_layers, num_heads, 
                experiment_id, pruning_strategy
            )
            
        # Evaluate model after pruning
        logger.info("Evaluating model after pruning")
        post_pruning_metrics = evaluator.evaluate(model, eval_dataloader)
        logger.info(f"Post-pruning metrics: {post_pruning_metrics}")
        
        # Fine-tune pruned model
        logger.info("Fine-tuning pruned model")
        fine_tuner = AdaptiveFinetuner(
            model=model,
            learning_rate=learning_rate,
            use_differential_lr=use_differential_lr
        )
        
        # Fine-tune and track plasticity
        training_results = fine_tuner.fine_tune(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            pruned_heads=pruned_heads,
            steps=fine_tuning_steps,
            evaluator=evaluator
        )
        
        # Save training results
        plasticity_tracker = fine_tuner.plasticity_tracker
        plasticity_tracker.save_tracking_data(experiment_dir)
        
        # Evaluate model after fine-tuning
        logger.info("Evaluating model after fine-tuning")
        final_metrics = evaluator.evaluate(model, eval_dataloader)
        logger.info(f"Final metrics: {final_metrics}")
        
        # Collect post-pruning entropy
        if pruning_strategy == "entropy":
            post_entropy_data = pruner.collect_distributions(model, eval_dataloader)
            
            # Save post-pruning entropy
            post_entropy_dir = os.path.join(experiment_dir, "post_entropy.json")
            serializable_post_entropy = {}
            for layer_idx, entropy in post_entropy_data.items():
                serializable_post_entropy[str(layer_idx)] = entropy.tolist()
            
            with open(post_entropy_dir, "w") as f:
                json.dump(serializable_post_entropy, f)
                
            # Visualize post-pruning entropy if requested
            if generate_visualizations and self.visualizer is not None:
                self.visualizer["entropy"].visualize_entropy_heatmap(
                    post_entropy_data, experiment_id, phase="post_finetuning"
                )
                
                # Visualize entropy changes
                self.visualizer["entropy"].visualize_entropy_changes(
                    entropy_data, post_entropy_data, experiment_id
                )
                
        # Analyze regrowth patterns
        regrowth_data = plasticity_tracker.analyze_regrowth()
        
        # Calculate improvement metrics
        recovery_metrics = {}
        try:
            # Calculate recovery percentage for each key metric
            for metric in ["loss", "perplexity"]:
                if (metric in baseline_metrics and 
                    metric in post_pruning_metrics and 
                    metric in final_metrics):
                    
                    baseline = baseline_metrics[metric]
                    pruned = post_pruning_metrics[metric]
                    final = final_metrics[metric]
                    
                    # Calculate degradation from baseline to pruned
                    degradation = pruned - baseline
                    
                    # Calculate recovery from pruned to final
                    recovery = pruned - final
                    
                    # Calculate recovery percentage (positive means improvement)
                    # For metrics where lower is better (like loss and perplexity)
                    if abs(degradation) > 1e-6:
                        recovery_pct = (recovery / abs(degradation)) * 100
                        recovery_metrics[f"{metric}_recovery_pct"] = recovery_pct
                        
            # Overall recovery is based on perplexity if available, otherwise loss
            if "perplexity_recovery_pct" in recovery_metrics:
                recovery_metrics["overall_recovery_pct"] = recovery_metrics["perplexity_recovery_pct"]
            elif "loss_recovery_pct" in recovery_metrics:
                recovery_metrics["overall_recovery_pct"] = recovery_metrics["loss_recovery_pct"]
                
        except Exception as e:
            logger.warning(f"Error calculating recovery metrics: {e}")
            
        # Save results
        results = {
            "metrics": {
                "baseline": baseline_metrics,
                "post_pruning": post_pruning_metrics,
                "final": final_metrics
            },
            "pruned_heads": [(l, h, float(s)) for l, h, s in pruned_heads],
            "regrowth_data": {f"{l}_{h}": data for (l, h), data in regrowth_data.items()},
            "recovery_metrics": recovery_metrics
        }
        
        results_path = os.path.join(experiment_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations for recovery analysis if requested
        if generate_visualizations and self.visualizer is not None:
            # Visualize recovery metrics
            self.visualizer["recovery"].visualize_recovery_analysis(
                baseline_metrics, post_pruning_metrics, final_metrics, experiment_id
            )
            
            # Visualize regrown heads if any
            if regrowth_data:
                num_layers, num_heads = pruner.detect_model_structure(model)
                self.visualizer["recovery"].visualize_regrown_heads(
                    regrowth_data, num_layers, num_heads, experiment_id
                )
                
            # Visualize training progress
            self.visualizer["metrics"].visualize_training_progress(
                plasticity_tracker.performance_history, experiment_id, phase="complete"
            )
            
            # Save metrics data
            self.visualizer["metrics"].save_metrics(results, experiment_id)
            
        logger.info(f"Experiment completed successfully. Results saved to {experiment_dir}")
        
        return results
    
    def _create_pruner(self, strategy: str) -> PruningStrategy:
        """
        Create a pruner based on the specified strategy.
        
        Args:
            strategy: Pruning strategy ('entropy' or 'magnitude')
            
        Returns:
            Pruning strategy object
        """
        if strategy.lower() == "entropy":
            return EntropyPruner(device=self.device)
        elif strategy.lower() == "magnitude":
            return MagnitudePruner(device=self.device)
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
            
# Define a function to run experiments without creating a class instance
def run_plasticity_experiment(
    model_name: str,
    pruning_strategy: str = "entropy",
    prune_ratio: float = 0.3,
    learning_rate: float = 5e-6,
    adaptive_lr: bool = True,
    learning_steps: int = 500,
    batch_size: int = 4,
    dataloader_builder_fn = None,
    device: Optional[str] = None,
    output_dir: str = "./output",
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Main plasticity loop experiment.

    1. Load baseline model
    2. Measure pre-finetune entropy
    3. Prune heads
    4. Fine-tune model
    5. Measure post-finetune entropy
    6. Analyze plasticity patterns
    7. Generate visualizations throughout the process

    Args:
        model_name: name of the base model (e.g., 'distilgpt2')
        pruning_strategy: 'entropy' or 'magnitude'
        prune_ratio: ratio of heads to prune
        learning_rate: base learning rate
        adaptive_lr: whether to use higher LR for pruned heads
        learning_steps: number of fine-tuning steps
        batch_size: for training and evaluation
        dataloader_builder_fn: function returning train and val dataloaders
        device: compute device (auto-detected if None)
        output_dir: directory to save results and visualizations
        visualize: whether to generate visualizations during execution
        
    Returns:
        Dictionary containing experiment results
    """
    # Create experiment
    experiment = PlasticityExperiment(
        model_name=model_name,
        output_dir=output_dir,
        device=device,
        adaptive_model=True
    )
    
    # Run experiment
    results = experiment.run_experiment(
        pruning_strategy=pruning_strategy,
        pruning_level=prune_ratio,
        dataloader_builder_fn=dataloader_builder_fn,
        fine_tuning_steps=learning_steps,
        learning_rate=learning_rate,
        use_differential_lr=adaptive_lr,
        batch_size=batch_size,
        output_dir=output_dir,
        generate_visualizations=visualize
    )
    
    return results