"""
Neural Plasticity Experiment Implementation

This module provides a complete implementation of neural plasticity experiments
that is compatible with both local filesystem and Colab environments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import base experiment framework
from .base_experiment import BaseExperiment

# Import core modules
from .plasticity_tracker import PlasticityTracker
from .fine_tuner import AdaptiveFinetuner
from .pruning import EntropyPruner, MagnitudePruner, PruningStrategy
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class NeuralPlasticityExperiment(BaseExperiment):
    """
    Neural Plasticity Experiment that works in both local and Colab environments.
    
    This experiment:
    1. Analyzes attention patterns in transformer models
    2. Prunes least important attention heads
    3. Fine-tunes the pruned model to recover performance
    4. Tracks all metrics and generates visualizations
    """
    
    def __init__(
        self,
        output_dir: str,
        device: Optional[str] = None,
        model_name: str = "distilgpt2",
        adaptive_model: bool = True,
        experiment_name: str = "neural_plasticity"
    ):
        """
        Initialize the neural plasticity experiment.
        
        Args:
            output_dir: Directory to save results
            device: Device to run on (auto-detected if None)
            model_name: Name of the pre-trained model to use
            adaptive_model: Whether to use adaptive model wrapper
            experiment_name: Name of the experiment type
        """
        # Initialize base experiment
        super().__init__(
            output_dir=output_dir,
            device=device,
            experiment_name=experiment_name,
            enable_colab_integration=True
        )
        
        # Neural plasticity specific attributes
        self.model_name = model_name
        self.adaptive_model = adaptive_model
        self.model = None
        self.quick_test = False  # Will be set if --quick_test flag is provided
        self.batch_size = 4      # Default batch size
        self.pruning_strategy = "entropy"  # Default pruning strategy
        self.pruning_level = 0.2           # Default pruning level
        self.fine_tuning_steps = 500       # Default number of fine-tuning steps
        self.learning_rate = 5e-5          # Default learning rate
        self.no_visualize = False          # Default to generating visualizations
        self.save_model = False            # Default to not saving the model
        
        # Initialize visualizers if needed
        self._initialize_visualizers()
    
    def _initialize_visualizers(self):
        """Initialize visualization components if available."""
        try:
            # Import visualizers - only if available
            from .visualization import (
                MetricsVisualizer, 
                EntropyVisualizer, 
                PruningVisualizer, 
                HeadRecoveryVisualizer
            )
            
            # Store in a dictionary for easy access
            self.visualizers = {
                "metrics": MetricsVisualizer(self.output_dir),
                "entropy": EntropyVisualizer(self.output_dir),
                "pruning": PruningVisualizer(self.output_dir),
                "recovery": HeadRecoveryVisualizer(self.output_dir)
            }
            
            logger.info("Initialized visualization components")
        except ImportError:
            logger.info("Visualization components not available - continuing without visualization")
            self.visualizers = None
        
    def _load_model(self):
        """
        Load the model based on experiment settings.
        
        Returns:
            Loaded model
        """
        try:
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
        except ImportError:
            # Fallback to transformers library if custom loaders not available
            logger.warning("Custom model loaders not available, using transformers library")
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model = model.to(self.device)
            return model
            
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
    
    def run_experiment(
        self,
        pruning_strategy: str,
        pruning_level: float,
        dataloader_builder_fn: Callable,
        fine_tuning_steps: int = 500,
        learning_rate: float = 5e-5,
        use_differential_lr: bool = True,
        batch_size: int = 4,
        generate_visualizations: bool = True,
        save_model: bool = False,
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
            batch_size: Batch size for training and evaluation
            generate_visualizations: Whether to generate visualizations
            save_model: Whether to save the model after experiment
            experiment_id: Unique identifier for this experiment
            
        Returns:
            Dictionary containing experiment results
        """
        # Create experiment run if not specified
        if experiment_id is None:
            experiment_id = self.create_experiment_run(
                f"{pruning_strategy}_{pruning_level}"
            )
            
        # Save experiment parameters
        params = {
            "model_name": self.model_name,
            "pruning_strategy": pruning_strategy,
            "pruning_level": pruning_level,
            "fine_tuning_steps": fine_tuning_steps,
            "learning_rate": learning_rate,
            "use_differential_lr": use_differential_lr,
            "batch_size": batch_size,
            "device": self.device,
            "adaptive_model": self.adaptive_model,
            "save_model": save_model
        }
        
        self.save_experiment_params(params, experiment_id)
        
        # Disable visualizations if requested or not available
        if generate_visualizations and self.visualizers is None:
            logger.warning("Visualization components not available - continuing without visualization")
            generate_visualizations = False
            
        # Update progress in Colab
        self.update_colab_progress("Loading model...", 0.05)
            
        # Load model
        logger.info(f"Loading model: {self.model_name}")
        self.model = self._load_model()
        
        # Create dataloaders
        logger.info("Creating dataloaders")
        self.update_colab_progress("Creating dataloaders...", 0.1)
        train_dataloader, eval_dataloader = dataloader_builder_fn(batch_size=batch_size)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(device=self.device)
        
        # Evaluate baseline model
        self.update_colab_progress("Evaluating baseline model...", 0.15)
        logger.info("Evaluating baseline model")
        baseline_metrics = evaluator.evaluate(self.model, eval_dataloader)
        logger.info(f"Baseline metrics: {baseline_metrics}")
        
        # Apply pruning
        self.update_colab_progress(f"Applying {pruning_strategy} pruning...", 0.25)
        logger.info(f"Applying {pruning_strategy} pruning with level {pruning_level}")
        pruner = self._create_pruner(pruning_strategy)
        
        # Collect attention distributions for entropy pruning
        if pruning_strategy == "entropy":
            # Collect entropy data
            entropy_data = pruner.collect_distributions(self.model, eval_dataloader)
            
            # Save pre-pruning entropy
            experiment_dir = self.output_dir / experiment_id
            entropy_path = experiment_dir / "pre_entropy.json"
            self._save_tensor_dict(entropy_data, entropy_path)
                
            # Visualize pre-pruning entropy if requested
            if generate_visualizations:
                self.visualizers["entropy"].visualize_entropy_heatmap(
                    entropy_data, experiment_id, phase="pre_pruning"
                )
            
            # Perform pruning
            pruned_heads = pruner.prune(self.model, distributions=entropy_data, prune_percent=pruning_level)
        else:
            # Perform magnitude-based pruning
            pruned_heads = pruner.prune(self.model, prune_percent=pruning_level)
            
        logger.info(f"Pruned {len(pruned_heads)} heads")
        
        # Save pruned heads
        experiment_dir = self.output_dir / experiment_id
        pruned_heads_path = experiment_dir / "pruned_heads.json"
        with open(pruned_heads_path, "w") as f:
            import json
            # Handle potential NaN values properly
            processed_heads = []
            for l, h, s in pruned_heads:
                # Convert float to None if it's NaN to ensure valid JSON
                score = None if isinstance(s, float) and np.isnan(s) else float(s)
                processed_heads.append([int(l), int(h), score])
            json.dump(processed_heads, f, indent=2)
            
        # Visualize pruned heads if requested
        if generate_visualizations:
            # Detect model structure for visualization
            num_layers, num_heads = pruner.detect_model_structure(self.model)
            
            self.visualizers["pruning"].visualize_pruned_heads(
                pruned_heads, num_layers, num_heads, 
                experiment_id, pruning_strategy
            )
            
        # Evaluate model after pruning
        self.update_colab_progress("Evaluating model after pruning...", 0.4)
        logger.info("Evaluating model after pruning")
        post_pruning_metrics = evaluator.evaluate(self.model, eval_dataloader)
        logger.info(f"Post-pruning metrics: {post_pruning_metrics}")
        
        # Fine-tune pruned model
        self.update_colab_progress("Fine-tuning pruned model...", 0.5)
        logger.info("Fine-tuning pruned model")
        fine_tuner = AdaptiveFinetuner(
            model=self.model,
            learning_rate=learning_rate,
            use_differential_lr=use_differential_lr
        )
        
        # Fine-tune and track plasticity
        # Create progress callback for Colab
        def progress_callback(step, total_steps):
            progress = 0.5 + (step / total_steps) * 0.3
            self.update_colab_progress(
                f"Fine-tuning: {step}/{total_steps} steps", 
                progress
            )
        
        training_results = fine_tuner.fine_tune(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            pruned_heads=pruned_heads,
            steps=fine_tuning_steps,
            evaluator=evaluator,
            progress_callback=progress_callback
        )
        
        # Save training results
        plasticity_tracker = fine_tuner.plasticity_tracker
        plasticity_tracker.save_tracking_data(str(experiment_dir))
        
        # Evaluate model after fine-tuning
        self.update_colab_progress("Evaluating final model...", 0.85)
        logger.info("Evaluating model after fine-tuning")
        final_metrics = evaluator.evaluate(self.model, eval_dataloader)
        logger.info(f"Final metrics: {final_metrics}")
        
        # Collect post-pruning entropy
        if pruning_strategy == "entropy":
            post_entropy_data = pruner.collect_distributions(self.model, eval_dataloader)
            
            # Save post-pruning entropy
            post_entropy_path = experiment_dir / "post_entropy.json"
            self._save_tensor_dict(post_entropy_data, post_entropy_path)
                
            # Visualize post-pruning entropy if requested
            if generate_visualizations:
                self.visualizers["entropy"].visualize_entropy_heatmap(
                    post_entropy_data, experiment_id, phase="post_finetuning"
                )
                
                # Visualize entropy changes
                self.visualizers["entropy"].visualize_entropy_changes(
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
            
        # Save model if requested
        if save_model:
            self.update_colab_progress("Saving model...", 0.9)
            logger.info("Saving model")
            model_path = experiment_dir / "models" / "final_model"
            self.model.save_pretrained(str(model_path))
            logger.info(f"Model saved to {model_path}")
            
        # Prepare results
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
        
        # Generate visualizations for recovery analysis if requested
        if generate_visualizations:
            self.update_colab_progress("Generating visualizations...", 0.95)
            
            # Visualize recovery metrics
            self.visualizers["recovery"].visualize_recovery_analysis(
                baseline_metrics, post_pruning_metrics, final_metrics, experiment_id
            )
            
            # Visualize regrown heads if any
            if regrowth_data:
                num_layers, num_heads = pruner.detect_model_structure(self.model)
                self.visualizers["recovery"].visualize_regrown_heads(
                    regrowth_data, num_layers, num_heads, experiment_id
                )
                
            # Visualize training progress
            self.visualizers["metrics"].visualize_training_progress(
                plasticity_tracker.performance_history, experiment_id, phase="complete"
            )
            
            # Save metrics data
            self.visualizers["metrics"].save_metrics(results, experiment_id)
            
        # Save final results
        self.save_experiment_results(results, experiment_id)
        
        # Final progress update
        self.update_colab_progress("Experiment completed successfully!", 1.0)
        
        logger.info(f"Experiment completed successfully. Results saved to {experiment_dir}")
        
        # If in Colab, display summary visualization
        if self.in_colab:
            self._display_colab_summary(experiment_id, results, generate_visualizations)
        
        return results
    
    def _save_tensor_dict(self, data_dict: Dict[int, torch.Tensor], save_path: Path):
        """
        Save a dictionary of tensors to a JSON file.
        
        Args:
            data_dict: Dictionary mapping integers to tensors
            save_path: Path to save the data
        """
        # Convert to serializable format
        serializable_dict = {}
        for layer_idx, tensor in data_dict.items():
            serializable_dict[str(layer_idx)] = tensor.detach().cpu().numpy().tolist()
        
        # Save as JSON
        import json
        with open(save_path, "w") as f:
            json.dump(serializable_dict, f)
    
    def _display_colab_summary(
        self, 
        experiment_id: str, 
        results: Dict[str, Any],
        has_visualizations: bool
    ):
        """
        Display a summary of the experiment in Colab.
        
        Args:
            experiment_id: Experiment ID
            results: Results dictionary
            has_visualizations: Whether visualizations were generated
        """
        if not self.in_colab:
            return
            
        try:
            from IPython.display import display, HTML
            
            # Prepare metrics for display
            baseline_metrics = results["metrics"]["baseline"]
            post_pruning_metrics = results["metrics"]["post_pruning"]
            final_metrics = results["metrics"]["final"]
            
            # Format perplexity for display
            def format_perplexity(value):
                if value > 1000000:
                    return f"{value/1000000:.2f}M"
                elif value > 1000:
                    return f"{value/1000:.1f}K"
                else:
                    return f"{value:.2f}"
            
            # Create HTML summary
            html = f"""
            <div style="background:#f0f8ff; padding:20px; border-radius:10px; margin:20px 0;">
                <h2>Neural Plasticity Experiment Results</h2>
                <p><b>Experiment ID:</b> {experiment_id}</p>
                <p><b>Model:</b> {self.model_name}</p>
                <p><b>Pruning Strategy:</b> {results.get('pruning_strategy', 'N/A')}</p>
                <p><b>Pruned Heads:</b> {len(results.get('pruned_heads', []))}</p>
                
                <h3>Metrics</h3>
                <table style="width:100%; border-collapse:collapse;">
                    <tr style="background:#4285F4; color:white;">
                        <th style="padding:8px; text-align:left;">Metric</th>
                        <th style="padding:8px; text-align:center;">Baseline</th>
                        <th style="padding:8px; text-align:center;">After Pruning</th>
                        <th style="padding:8px; text-align:center;">After Fine-tuning</th>
                        <th style="padding:8px; text-align:center;">Recovery</th>
                    </tr>
            """
            
            # Add metrics rows
            for metric in ["loss", "perplexity"]:
                if metric in baseline_metrics and metric in post_pruning_metrics and metric in final_metrics:
                    baseline = baseline_metrics[metric]
                    pruned = post_pruning_metrics[metric]
                    final = final_metrics[metric]
                    
                    # Format values
                    if metric == "perplexity":
                        baseline_str = format_perplexity(baseline)
                        pruned_str = format_perplexity(pruned)
                        final_str = format_perplexity(final)
                    else:
                        baseline_str = f"{baseline:.4f}"
                        pruned_str = f"{pruned:.4f}"
                        final_str = f"{final:.4f}"
                    
                    # Calculate recovery percentage
                    recovery_pct = results.get("recovery_metrics", {}).get(f"{metric}_recovery_pct", 0)
                    recovery_str = f"{recovery_pct:.1f}%" if recovery_pct else "N/A"
                    
                    # Add row
                    html += f"""
                    <tr style="border-bottom:1px solid #ddd;">
                        <td style="padding:8px;">{metric.capitalize()}</td>
                        <td style="padding:8px; text-align:center;">{baseline_str}</td>
                        <td style="padding:8px; text-align:center;">{pruned_str}</td>
                        <td style="padding:8px; text-align:center;">{final_str}</td>
                        <td style="padding:8px; text-align:center;">{recovery_str}</td>
                    </tr>
                    """
            
            html += """
                </table>
            """
            
            # Add visualizations if available
            if has_visualizations:
                experiment_dir = self.output_dir / experiment_id
                viz_dir = experiment_dir / "visualizations"
                
                html += """
                <h3>Visualizations</h3>
                <div style="display:flex; flex-wrap:wrap; gap:10px;">
                """
                
                # Add some common visualization files if they exist
                viz_files = [
                    ("Training Progress", "training_progress_complete.png"),
                    ("Pruned Heads", "pruned_heads_entropy.png"),
                    ("Entropy Heatmap", "entropy_heatmap_pre_pruning.png"),
                    ("Recovery Analysis", "recovery_analysis.png")
                ]
                
                for title, filename in viz_files:
                    file_path = viz_dir / filename
                    if file_path.exists():
                        from IPython.display import Image
                        # Display the image with a caption
                        html += f"""
                        <div style="flex:1; min-width:45%;">
                            <p style="text-align:center; font-weight:bold;">{title}</p>
                            <img src="file://{file_path}" style="width:100%; max-width:500px; display:block; margin:0 auto;">
                        </div>
                        """
                
                html += """
                </div>
                """
            
            html += """
            </div>
            """
            
            # Display the HTML
            display(HTML(html))
            
        except Exception as e:
            logger.warning(f"Failed to display Colab summary: {e}")
    
    @classmethod
    def get_argument_parser(cls):
        """
        Get the argument parser for command-line arguments.
        
        Returns:
            Argument parser
        """
        parser = super().get_argument_parser()
        
        # Add neural plasticity specific arguments
        parser.add_argument(
            "--model_name", 
            type=str, 
            default="distilgpt2",
            help="Model name (default: distilgpt2)"
        )
        
        parser.add_argument(
            "--dataset", 
            type=str, 
            default="wikitext",
            help="Dataset name (default: wikitext)"
        )
        
        parser.add_argument(
            "--dataset_config", 
            type=str, 
            default="wikitext-2-raw-v1",
            help="Dataset configuration (default: wikitext-2-raw-v1)"
        )
        
        parser.add_argument(
            "--max_length", 
            type=int, 
            default=128,
            help="Maximum sequence length (default: 128)"
        )
        
        parser.add_argument(
            "--batch_size", 
            type=int, 
            default=4,
            help="Batch size (default: 4)"
        )
        
        parser.add_argument(
            "--learning_rate", 
            type=float, 
            default=5e-5,
            help="Learning rate (default: 5e-5)"
        )
        
        parser.add_argument(
            "--pruning_strategy", 
            type=str, 
            default="entropy",
            choices=["entropy", "magnitude"],
            help="Pruning strategy (default: entropy)"
        )
        
        parser.add_argument(
            "--pruning_level", 
            type=float, 
            default=0.2,
            help="Pruning level (default: 0.2)"
        )
        
        parser.add_argument(
            "--fine_tuning_steps", 
            type=int, 
            default=500,
            help="Number of fine-tuning steps (default: 500)"
        )
        
        parser.add_argument(
            "--adaptive_model", 
            action="store_true", 
            default=True,
            help="Use adaptive model wrapper (default: True)"
        )
        
        parser.add_argument(
            "--save_model", 
            action="store_true", 
            default=False,
            help="Save model after experiment (default: False)"
        )
        
        parser.add_argument(
            "--quick_test", 
            action="store_true", 
            default=False,
            help="Run a quick test with minimal steps (default: False)"
        )
        
        return parser
    
    @classmethod
    def main(cls):
        """
        Main function for running the experiment from command line.
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create experiment instance from command-line arguments
        experiment = cls.from_args()
        
        try:
            from transformers import AutoTokenizer, default_data_collator
            from torch.utils.data import DataLoader
            from datasets import load_dataset
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(experiment.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Create dataloader builder
            def create_dataloader_builder(args, tokenizer, quick_test=False):
                def build_dataloaders(batch_size=args.batch_size):
                    # For quick tests, use a small subset of the real dataset
                    if quick_test:
                        try:
                            # Use wikitext as a small but real dataset for quick testing
                            logger.info("Loading small wikitext sample for quick test...")
                            
                            # Always use real text data, just with a small sample
                            train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
                            eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:20]")
                            
                            # Process datasets
                            def tokenize_function(examples):
                                return tokenizer(
                                    examples["text"], 
                                    padding="max_length", 
                                    truncation=True, 
                                    max_length=args.max_length
                                )
                            
                            train_dataset = train_dataset.map(tokenize_function, batched=True)
                            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
                            
                            # Remove original text columns
                            train_dataset = train_dataset.remove_columns(["text"])
                            eval_dataset = eval_dataset.remove_columns(["text"])
                            
                            # Add labels for language modeling
                            def add_labels(examples):
                                examples["labels"] = examples["input_ids"].copy()
                                return examples
                            
                            train_dataset = train_dataset.map(add_labels)
                            eval_dataset = eval_dataset.map(add_labels)
                            
                            # Set torch format
                            train_dataset = train_dataset.with_format("torch")
                            eval_dataset = eval_dataset.with_format("torch")
                            
                            logger.info("Using small sample of real wikitext data for quick test")
                        except Exception as e:
                            logger.error(f"Error loading wikitext for quick test: {e}")
                            raise
                    else:
                        # Load full dataset
                        logger.info(f"Loading dataset: {args.dataset}/{args.dataset_config}...")
                        try:
                            train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
                            validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
                            
                            # Define tokenization function
                            def tokenize_function(examples):
                                if "text" in examples:
                                    text_key = "text"
                                else:
                                    # Try to find a suitable text column
                                    text_cols = [col for col in examples.keys() 
                                              if col.lower() in ["text", "content", "document", "article"]]
                                    if text_cols:
                                        text_key = text_cols[0]
                                    else:
                                        raise ValueError(f"Could not find text column in dataset. Available columns: {list(examples.keys())}")
                                
                                return tokenizer(
                                    examples[text_key], 
                                    padding="max_length", 
                                    truncation=True, 
                                    max_length=args.max_length
                                )
                            
                            # Process datasets
                            train_dataset = train_dataset.map(tokenize_function, batched=True)
                            validation_dataset = validation_dataset.map(tokenize_function, batched=True)
                            
                            # Remove original text columns
                            text_columns = [col for col in train_dataset.column_names 
                                          if col.lower() in ["text", "content", "document", "article"]]
                            if text_columns:
                                train_dataset = train_dataset.remove_columns(text_columns)
                                validation_dataset = validation_dataset.remove_columns(text_columns)
                            
                            # Add labels for language modeling
                            def add_labels(examples):
                                examples["labels"] = examples["input_ids"].copy()
                                return examples
                            
                            train_dataset = train_dataset.map(add_labels)
                            validation_dataset = validation_dataset.map(add_labels)
                            
                            # Set torch format
                            train_dataset = train_dataset.with_format("torch")
                            validation_dataset = validation_dataset.with_format("torch")
                            
                            # Use validation dataset for eval
                            eval_dataset = validation_dataset
                            
                        except Exception as e:
                            logger.error(f"Error loading dataset: {e}")
                            raise
                    
                    # Create dataloaders
                    train_dataloader = DataLoader(
                        train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        collate_fn=default_data_collator
                    )
                    
                    eval_dataloader = DataLoader(
                        eval_dataset, 
                        batch_size=batch_size, 
                        collate_fn=default_data_collator
                    )
                    
                    return train_dataloader, eval_dataloader
                
                return build_dataloaders
                
            # Create dataloader builder
            dataloader_builder = create_dataloader_builder(
                experiment, tokenizer, quick_test=experiment.quick_test
            )
            
            # Override fine_tuning_steps for quick test
            if experiment.quick_test:
                experiment.fine_tuning_steps = 20
            
            # Run the experiment
            results = experiment.run_experiment(
                pruning_strategy=experiment.pruning_strategy,
                pruning_level=experiment.pruning_level,
                dataloader_builder_fn=dataloader_builder,
                fine_tuning_steps=experiment.fine_tuning_steps,
                learning_rate=experiment.learning_rate,
                use_differential_lr=True,
                batch_size=experiment.batch_size,
                generate_visualizations=not experiment.no_visualize,
                save_model=experiment.save_model
            )
            
            # Print success message
            print(f"\nExperiment completed successfully!")
            print(f"Results saved to: {experiment.output_dir}")
            
            # Print helpful next steps
            print("\nNext steps:")
            print(f"- View log file: {experiment.log_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running experiment: {e}", exc_info=True)
            print(f"Error running experiment: {e}")
            return None