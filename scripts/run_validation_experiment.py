#!/usr/bin/env python
"""
Neural Plasticity Validation Experiment Runner

This script implements the validation plan outlined in docs/reports/validation_plan.md.
It runs controlled experiments to benchmark the neural plasticity system against
various baselines, ensuring scientific rigor and reproducibility.

Usage:
    python run_validation_experiment.py --experiment single_cycle --model distilgpt2 --seeds 5
    python run_validation_experiment.py --experiment multi_cycle --model gpt2 --seeds 3
    python run_validation_experiment.py --experiment task_alternation --model pythia-70m
    python run_validation_experiment.py --experiment cross_architecture --models distilgpt2,gpt2,pythia-70m,opt-125m
"""

import os
import sys
import logging
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm
import random
from scipy import stats
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import model components
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Import experiment components
from sentinel.plasticity.controller.rl_controller import RLController, RLControllerConfig
from sentinel.plasticity.entropy_journal import EntropyJournal, EntropyJournalConfig
from sentinel.plasticity.function_tracking import FunctionTracker, FunctionTrackingConfig
from sentinel.pruning.pruning_module import PruningModule
from sentinel.pruning.strategies import EntropyPruningStrategy, MagnitudePruningStrategy, RandomPruningStrategy

# Import MultiCycleRunner for experiment orchestration
from scripts.multi_cycle_runner import MultiCycleRunner


class ValidationConfig:
    """Configuration for validation experiments"""
    
    def __init__(self, args):
        """Initialize from command line arguments"""
        self.experiment_type = args.experiment
        self.model_names = args.models.split(',') if ',' in args.models else [args.models]
        self.num_seeds = args.seeds
        self.pruning_ratios = [float(r) for r in args.pruning_ratios.split(',')] if args.pruning_ratios else [0.1, 0.2, 0.3, 0.4, 0.5]
        self.num_cycles = args.cycles
        self.steps_per_cycle = args.steps
        self.batch_size = args.batch_size
        self.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = args.output_dir
        self.dataset = args.dataset
        self.rl_enabled = not args.disable_rl
        self.methods = args.methods.split(',') if args.methods else ["random", "magnitude", "entropy", "adaptive"]
        
        # Create experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"validation_{self.experiment_type}_{timestamp}"
        
        # Create output directory
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "config": vars(args),
            "metrics": {},
            "models": {}
        }


class OraclePruningStrategy:
    """
    Oracle pruning strategy that determines optimal heads to prune
    based on full evaluation on a validation set.
    
    This is computationally expensive but serves as an upper bound
    for pruning performance.
    """
    
    def __init__(self, model, dataloader, device, ratio=0.3):
        """
        Initialize oracle pruning strategy.
        
        Args:
            model: The model to analyze
            dataloader: Validation dataloader
            device: Device for computation
            ratio: Pruning ratio
        """
        self.name = "oracle"
        self.ratio = ratio
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    def select_heads(self, model, dataloader):
        """
        Determine optimal heads to prune by trying each head individually.
        
        Args:
            model: The model to analyze
            dataloader: Dataloader for evaluation
            
        Returns:
            List of (layer_idx, head_idx) pairs to prune
        """
        logger.info("Running oracle pruning analysis (this may take a while)...")
        
        # Create baseline performance
        baseline_metrics = self._evaluate_model(model, dataloader)
        baseline_loss = baseline_metrics["loss"]
        
        # Find all attention heads
        head_impacts = []
        
        # For each layer
        for name, module in model.named_modules():
            if hasattr(module, 'num_heads') and hasattr(module, 'head_mask'):
                # Extract layer index if possible
                layer_idx = -1
                if any(c.isdigit() for c in name):
                    layer_idx = int(''.join(filter(str.isdigit, name)))
                
                # Test each head
                for head_idx in range(module.num_heads):
                    # Create temporary pruning mask
                    mask = torch.ones(module.num_heads, device=self.device)
                    mask[head_idx] = 0
                    
                    # Apply mask
                    old_mask = module.head_mask
                    module.head_mask = mask
                    
                    # Evaluate
                    metrics = self._evaluate_model(model, dataloader)
                    
                    # Restore mask
                    module.head_mask = old_mask
                    
                    # Calculate impact
                    impact = metrics["loss"] - baseline_loss
                    head_impacts.append((layer_idx, head_idx, impact))
        
        # Sort by impact (ascending)
        head_impacts.sort(key=lambda x: x[2])
        
        # Select heads with lowest impact (or even slightly beneficial pruning)
        num_to_prune = int(len(head_impacts) * self.ratio)
        heads_to_prune = [(h[0], h[1]) for h in head_impacts[:num_to_prune]]
        
        logger.info(f"Oracle pruning selected {len(heads_to_prune)} heads")
        return heads_to_prune
    
    def _evaluate_model(self, model, dataloader):
        """Evaluate model on dataloader"""
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = {
                        "input_ids": batch[0].to(self.device),
                        "attention_mask": batch[1].to(self.device),
                        "labels": batch[0].to(self.device)
                    }
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Accumulate metrics
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }


class ValidationExperiment:
    """
    Base class for validation experiments.
    
    This class provides common functionality for all validation experiments,
    including model initialization, dataset loading, and metric tracking.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize validation experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = config.device
        self.output_dir = config.experiment_dir
        
        logger.info(f"Initializing validation experiment: {config.experiment_type}")
        logger.info(f"Models: {config.model_names}")
        logger.info(f"Seeds: {config.num_seeds}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Output directory: {config.experiment_dir}")
    
    def _create_datasets(self, model_name, dataset_name=None):
        """
        Create datasets for training and evaluation.
        
        Args:
            model_name: Model name/path for tokenizer
            dataset_name: Dataset name or path
            
        Returns:
            Tuple of (train_dataset, eval_dataset, test_dataset, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset folder
        dataset_dir = os.path.join(self.output_dir, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # If no dataset specified, create synthetic dataset
        if dataset_name is None or dataset_name == "synthetic":
            logger.info("Creating synthetic dataset")
            
            # Create synthetic data
            train_data = [
                "The neural network model processes data through multiple layers of computation.",
                "Artificial intelligence systems can learn from experience and improve over time.",
                "The transformer architecture revolutionized natural language processing tasks.",
                "Self-attention mechanisms enable models to focus on relevant parts of the input.",
                "Deep learning approaches use neural networks with multiple layers to learn hierarchical representations.",
                "Transfer learning allows models to apply knowledge gained from one task to another.",
                "Language models predict the next token based on the context of previous tokens.",
                "Neural plasticity refers to the brain's ability to reorganize itself by forming new connections.",
                "Attention weights determine how much focus to put on different parts of the input sequence.",
                "Gradient descent optimizes model parameters by minimizing the loss function."
            ] * 10  # Repeat for more samples
            
            eval_data = [
                "Reinforcement learning trains agents through trial and error with rewards and penalties.",
                "Unsupervised learning finds patterns in data without explicit labels or guidance.",
                "Generative models can create new content similar to their training examples.",
                "Recurrent neural networks process sequential data using internal memory states.",
                "Fine-tuning adapts pre-trained models to specific tasks with domain-specific data."
            ] * 5
            
            test_data = [
                "Convolutional networks excel at image recognition by applying filters to detect features.",
                "Natural language processing enables computers to understand and generate human language.",
                "Model compression techniques reduce size while preserving important capabilities.",
                "Federated learning allows training across multiple devices while preserving privacy.",
                "Ensemble methods combine multiple models to improve overall performance and robustness."
            ] * 5
            
            # Write to files
            train_path = os.path.join(dataset_dir, "train.txt")
            eval_path = os.path.join(dataset_dir, "eval.txt")
            test_path = os.path.join(dataset_dir, "test.txt")
            
            with open(train_path, 'w') as f:
                f.write("\n".join(train_data))
            
            with open(eval_path, 'w') as f:
                f.write("\n".join(eval_data))
                
            with open(test_path, 'w') as f:
                f.write("\n".join(test_data))
            
            # Create datasets
            train_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=train_path,
                block_size=128
            )
            
            eval_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=eval_path,
                block_size=128
            )
            
            test_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=test_path,
                block_size=128
            )
            
        else:
            # Use specified dataset (e.g., from Hugging Face)
            from datasets import load_dataset
            
            logger.info(f"Loading dataset: {dataset_name}")
            
            if dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
                
                # Create datasets
                def tokenize_function(examples):
                    return tokenizer(examples["text"])
                
                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"]
                )
                
                block_size = 128
                
                def group_texts(examples):
                    # Concatenate all texts and create overlapping blocks
                    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
                    total_length = len(concatenated[list(examples.keys())[0]])
                    
                    result = {
                        k: [t[i:i+block_size] for i in range(0, total_length - block_size + 1, block_size)]
                        for k, t in concatenated.items()
                    }
                    
                    result["labels"] = result["input_ids"].copy()
                    return result
                
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True
                )
                
                train_dataset = tokenized_datasets["train"]
                eval_dataset = tokenized_datasets["validation"]
                test_dataset = tokenized_datasets["test"]
                
            else:
                # Could add support for more datasets here
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return train_dataset, eval_dataset, test_dataset, tokenizer
    
    def _create_data_loader(self, dataset, tokenizer, batch_size=None):
        """
        Create a data loader from a dataset.
        
        Args:
            dataset: The dataset to load
            tokenizer: Tokenizer for batch creation
            batch_size: Batch size (defaults to config)
            
        Returns:
            DataLoader instance
        """
        from torch.utils.data import DataLoader
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Create collate function
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator
        )
        
        return dataloader
    
    def _evaluate_model(self, model, dataloader):
        """
        Evaluate model on a dataloader.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Prepare batch
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = {
                        "input_ids": batch[0].to(self.device),
                        "attention_mask": batch[1].to(self.device) if len(batch) > 1 else None,
                        "labels": batch[0].to(self.device)
                    }
                    
                    if batch["attention_mask"] is None:
                        del batch["attention_mask"]
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Accumulate metrics
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def _fine_tune(self, model, train_dataset, eval_dataset, tokenizer, steps, output_subdir):
        """
        Fine-tune a model for a specific number of steps.
        
        Args:
            model: The model to fine-tune
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for the model
            steps: Number of training steps
            output_subdir: Subdirectory for outputs
            
        Returns:
            Fine-tuned model and evaluation metrics
        """
        # Create output directory
        output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            evaluation_strategy="steps",
            eval_steps=steps // 2,
            save_steps=steps,
            save_total_limit=1,
            max_steps=steps,
            logging_steps=steps // 4,
            logging_dir=os.path.join(output_dir, "logs")
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate the model
        eval_metrics = trainer.evaluate()
        
        return model, eval_metrics
    
    def _create_pruning_strategy(self, strategy_name, ratio=0.3, model=None, dataloader=None):
        """
        Create a pruning strategy.
        
        Args:
            strategy_name: Name of the strategy
            ratio: Pruning ratio
            model: Model for oracle strategy
            dataloader: Dataloader for oracle strategy
            
        Returns:
            Pruning strategy instance
        """
        if strategy_name.lower() == "random":
            return RandomPruningStrategy()
        elif strategy_name.lower() == "magnitude":
            return MagnitudePruningStrategy()
        elif strategy_name.lower() == "entropy":
            return EntropyPruningStrategy()
        elif strategy_name.lower() == "oracle":
            return OraclePruningStrategy(model, dataloader, self.device, ratio)
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy_name}")
    
    def _prune_model(self, model, pruning_strategy, ratio, dataloader):
        """
        Prune a model using the specified strategy.
        
        Args:
            model: The model to prune
            pruning_strategy: Pruning strategy to use
            ratio: Pruning ratio (ignored for oracle)
            dataloader: Dataloader for pruning analysis
            
        Returns:
            Pruned model and pruning info
        """
        # Create pruning module
        pruning_module = PruningModule(model, pruning_strategy)
        
        # For oracle strategy, use its custom head selection
        if hasattr(pruning_strategy, 'name') and pruning_strategy.name == 'oracle':
            pruned_heads = pruning_strategy.select_heads(model, dataloader)
            pruning_mask = pruning_module.apply_mask_to_heads(pruned_heads)
        else:
            # Standard pruning
            pruned_heads = pruning_module.prune(
                pruning_ratio=ratio,
                dataloader=dataloader,
                device=self.device
            )
            
            # Get pruning mask
            pruning_mask = pruning_module.get_pruning_mask()
        
        logger.info(f"Pruned {len(pruned_heads)} attention heads")
        
        # Create pruning info
        pruning_info = {
            "strategy": pruning_strategy.__class__.__name__,
            "ratio": ratio,
            "pruned_heads": pruned_heads,
            "mask": pruning_mask
        }
        
        return model, pruning_info
    
    def _calculate_recovery_rate(self, pre_pruning_loss, post_pruning_loss, post_recovery_loss):
        """
        Calculate recovery rate from losses.
        
        Recovery rate measures how much of the performance drop from pruning
        was recovered through fine-tuning.
        
        Args:
            pre_pruning_loss: Loss before pruning
            post_pruning_loss: Loss after pruning
            post_recovery_loss: Loss after recovery fine-tuning
            
        Returns:
            Recovery rate (0.0-1.0)
        """
        if post_pruning_loss <= pre_pruning_loss:
            # No performance drop from pruning, so recovery is perfect
            return 1.0
            
        # Calculate how much of the drop was recovered
        recovery_rate = (post_pruning_loss - post_recovery_loss) / (post_pruning_loss - pre_pruning_loss)
        
        # Clamp to valid range
        return max(0.0, min(1.0, recovery_rate))
    
    def _save_results(self, results, filename="results.json"):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Process results for JSON serialization
        def process_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = process_dict(value)
                elif isinstance(value, (torch.Tensor, np.ndarray)):
                    d[key] = value.item() if value.size == 1 else value.tolist()
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], tuple):
                    # Convert list of tuples to list of lists
                    d[key] = [list(item) for item in value]
            return d
        
        serializable_results = process_dict(results.copy())
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved results to {output_path}")
    
    def _create_visualizations(self, results, prefix=""):
        """
        Create visualizations from results.
        
        Args:
            results: Results dictionary
            prefix: Prefix for filenames
        """
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Function to extract metrics by method
        def extract_metrics(metric_name, seed_data=True):
            methods = []
            values = []
            errors = []
            
            for method, method_results in results.get("methods", {}).items():
                if not method_results:
                    continue
                    
                methods.append(method)
                
                if seed_data and "seeds" in method_results:
                    # Extract values from all seeds
                    seed_values = []
                    for seed_result in method_results["seeds"]:
                        if metric_name in seed_result:
                            seed_values.append(seed_result[metric_name])
                    
                    # Calculate mean and std
                    if seed_values:
                        values.append(np.mean(seed_values))
                        errors.append(np.std(seed_values))
                    else:
                        values.append(0)
                        errors.append(0)
                else:
                    # Single value
                    if metric_name in method_results:
                        values.append(method_results[metric_name])
                        errors.append(0)
                    else:
                        values.append(0)
                        errors.append(0)
            
            return methods, values, errors
        
        # Plot recovery rates
        methods, recovery_rates, errors = extract_metrics("recovery_rate")
        if methods and any(recovery_rates):
            plt.figure(figsize=(10, 6))
            plt.bar(methods, recovery_rates, yerr=errors)
            plt.xlabel('Method')
            plt.ylabel('Recovery Rate')
            plt.title('Recovery Rate by Pruning Method')
            plt.ylim(0, 1.1)
            
            # Add value labels
            for i, v in enumerate(recovery_rates):
                plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{prefix}recovery_rates.png"))
            plt.close()
        
        # Plot function preservation
        methods, preservation, errors = extract_metrics("function_preservation")
        if methods and any(preservation):
            plt.figure(figsize=(10, 6))
            plt.bar(methods, preservation, yerr=errors)
            plt.xlabel('Method')
            plt.ylabel('Function Preservation Score')
            plt.title('Function Preservation by Pruning Method')
            plt.ylim(0, 1.1)
            
            # Add value labels
            for i, v in enumerate(preservation):
                plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{prefix}function_preservation.png"))
            plt.close()
        
        # Plot perplexity comparison
        methods, pre_ppl, _ = extract_metrics("pre_pruning_perplexity")
        _, post_ppl, _ = extract_metrics("post_pruning_perplexity")
        _, recovery_ppl, _ = extract_metrics("post_recovery_perplexity")
        
        if methods and any(pre_ppl) and any(post_ppl) and any(recovery_ppl):
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(methods))
            width = 0.25
            
            plt.bar(x - width, pre_ppl, width, label='Pre-Pruning')
            plt.bar(x, post_ppl, width, label='Post-Pruning')
            plt.bar(x + width, recovery_ppl, width, label='Post-Recovery')
            
            plt.xlabel('Method')
            plt.ylabel('Perplexity')
            plt.title('Perplexity Across Pruning Stages')
            plt.xticks(x, methods)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{prefix}perplexity_comparison.png"))
            plt.close()
        
        # Plot learning curves if available
        if "learning_curves" in results:
            for metric, curves in results["learning_curves"].items():
                plt.figure(figsize=(10, 6))
                
                for method, values in curves.items():
                    if isinstance(values, list) and values:
                        plt.plot(range(1, len(values) + 1), values, 'o-', label=method)
                
                plt.xlabel('Cycle')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} Over Cycles')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{prefix}learning_curve_{metric}.png"))
                plt.close()
        
        logger.info(f"Created visualizations in {viz_dir}")

    def run(self):
        """Run the experiment (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement run()")


class SingleCycleExperiment(ValidationExperiment):
    """
    Single-cycle pruning and recovery experiment.
    
    This experiment tests pruning and recovery with a single cycle:
    1. Establish baseline performance
    2. Apply pruning with different methods
    3. Fine-tune for recovery
    4. Measure recovery rate and function preservation
    """
    
    def run(self):
        """Run the single-cycle experiment"""
        results = {
            "experiment_type": "single_cycle",
            "models": {},
            "methods": {}
        }
        
        # Run for each model
        for model_name in self.config.model_names:
            logger.info(f"Running experiment for model: {model_name}")
            
            model_results = {
                "methods": {}
            }
            
            # Create datasets
            train_dataset, eval_dataset, test_dataset, tokenizer = self._create_datasets(
                model_name, self.config.dataset
            )
            
            # Create data loaders
            train_loader = self._create_data_loader(train_dataset, tokenizer)
            eval_loader = self._create_data_loader(eval_dataset, tokenizer)
            test_loader = self._create_data_loader(test_dataset, tokenizer)
            
            # Test prompts for function tracking
            test_prompts = [
                "The transformer architecture allows models to",
                "Neural networks consist of layers of",
                "Attention mechanisms help models focus on",
                "Language models predict the next token based on",
                "Deep learning systems can learn from"
            ]
            
            # Set up function tracking
            function_config = FunctionTrackingConfig(
                output_dir=os.path.join(self.output_dir, "function_tracking"),
                experiment_name=f"{model_name}_single_cycle"
            )
            function_tracker = FunctionTracker(function_config, self.device)
            
            # Run for each method and pruning ratio
            for method in self.config.methods:
                logger.info(f"Testing pruning method: {method}")
                
                method_results = {}
                
                # For RL controller, we only have one adaptive ratio
                if method == "adaptive" and not self.config.rl_enabled:
                    logger.info("Skipping adaptive method (RL disabled)")
                    continue
                    
                pruning_ratios = self.config.pruning_ratios if method != "adaptive" else [None]
                
                for ratio in pruning_ratios:
                    ratio_str = f"{ratio:.1f}" if ratio is not None else "adaptive"
                    logger.info(f"Testing pruning ratio: {ratio_str}")
                    
                    ratio_results = {
                        "seeds": []
                    }
                    
                    # Run multiple seeds
                    for seed in range(self.config.num_seeds):
                        logger.info(f"Running with seed {seed}")
                        
                        # Set random seed
                        torch.manual_seed(seed)
                        random.seed(seed)
                        np.random.seed(seed)
                        
                        # Initialize model
                        model = AutoModelForCausalLM.from_pretrained(model_name)
                        model = model.to(self.device)
                        
                        # Store original model for function tracking
                        original_model = AutoModelForCausalLM.from_pretrained(model_name)
                        original_model = original_model.to(self.device)
                        
                        # 1. Evaluate baseline performance
                        logger.info("Evaluating baseline performance")
                        baseline_metrics = self._evaluate_model(model, test_loader)
                        logger.info(f"Baseline perplexity: {baseline_metrics['perplexity']:.4f}")
                        
                        # 2. Apply pruning
                        if method == "adaptive":
                            # Use RL controller for pruning decisions
                            logger.info("Initializing RL controller")
                            rl_config = RLControllerConfig(
                                output_dir=os.path.join(self.output_dir, "rl_controller"),
                                experiment_name=f"{model_name}_seed{seed}"
                            )
                            rl_controller = RLController(rl_config, self.device)
                            
                            # Select action based on initial state
                            initial_state = {
                                "avg_entropy": 0.5,  # Default initial values
                                "entropy_std": 0.2,
                                "perplexity": baseline_metrics["perplexity"],
                                "recovery_rate": 0.0,
                                "function_preservation": 0.0
                            }
                            
                            action = rl_controller.select_action(initial_state, 0, 1)
                            strategy_name = action["strategy"]
                            ratio = action["ratio"]
                            
                            logger.info(f"RL selected strategy: {strategy_name}, ratio: {ratio:.2f}")
                            
                            # Create pruning strategy
                            pruning_strategy = self._create_pruning_strategy(strategy_name)
                            
                        else:
                            # Use specified strategy
                            pruning_strategy = self._create_pruning_strategy(method)
                        
                        # Apply pruning
                        logger.info(f"Applying {method} pruning" + (f" at {ratio:.2f} ratio" if ratio is not None else ""))
                        pruned_model, pruning_info = self._prune_model(model, pruning_strategy, ratio, eval_loader)
                        
                        # 3. Evaluate post-pruning performance
                        logger.info("Evaluating post-pruning performance")
                        post_pruning_metrics = self._evaluate_model(pruned_model, test_loader)
                        logger.info(f"Post-pruning perplexity: {post_pruning_metrics['perplexity']:.4f}")
                        
                        # 4. Fine-tune for recovery
                        logger.info("Fine-tuning for recovery")
                        recovered_model, recovery_metrics = self._fine_tune(
                            pruned_model,
                            train_dataset,
                            eval_dataset,
                            tokenizer,
                            self.config.steps_per_cycle,
                            f"{model_name}/{method}/{ratio_str}/seed{seed}"
                        )
                        
                        # 5. Evaluate post-recovery performance
                        logger.info("Evaluating post-recovery performance")
                        post_recovery_metrics = self._evaluate_model(recovered_model, test_loader)
                        logger.info(f"Post-recovery perplexity: {post_recovery_metrics['perplexity']:.4f}")
                        
                        # 6. Track function preservation
                        logger.info("Tracking function preservation")
                        function_results = function_tracker.track_function(
                            original_model,
                            recovered_model,
                            test_prompts,
                            tokenizer,
                            cycle_idx=0,
                            cycle_name=f"{method}_{ratio_str}_seed{seed}"
                        )
                        
                        # Calculate recovery rate
                        recovery_rate = self._calculate_recovery_rate(
                            baseline_metrics["loss"],
                            post_pruning_metrics["loss"],
                            post_recovery_metrics["loss"]
                        )
                        logger.info(f"Recovery rate: {recovery_rate:.4f}")
                        
                        # 7. If using RL controller, provide reward
                        if method == "adaptive":
                            # Calculate reward
                            reward_metrics = {
                                "recovery_rate": recovery_rate,
                                "function_preservation": function_results["summary"].get("overall_preservation_score", 0.0),
                                "perplexity_change": (post_recovery_metrics["perplexity"] - baseline_metrics["perplexity"]) / baseline_metrics["perplexity"]
                            }
                            
                            rl_controller.observe_reward(reward_metrics, 0, 1, done=True)
                        
                        # Store seed results
                        seed_result = {
                            "seed": seed,
                            "pre_pruning_perplexity": baseline_metrics["perplexity"],
                            "post_pruning_perplexity": post_pruning_metrics["perplexity"],
                            "post_recovery_perplexity": post_recovery_metrics["perplexity"],
                            "pre_pruning_loss": baseline_metrics["loss"],
                            "post_pruning_loss": post_pruning_metrics["loss"],
                            "post_recovery_loss": post_recovery_metrics["loss"],
                            "recovery_rate": recovery_rate,
                            "function_preservation": function_results["summary"].get("overall_preservation_score", 0.0),
                            "pruned_heads_count": len(pruning_info["pruned_heads"]),
                            "strategy": pruning_strategy.__class__.__name__,
                            "ratio": ratio
                        }
                        
                        ratio_results["seeds"].append(seed_result)
                    
                    # Calculate aggregate metrics for this ratio
                    if ratio_results["seeds"]:
                        recovery_rates = [r["recovery_rate"] for r in ratio_results["seeds"]]
                        function_scores = [r["function_preservation"] for r in ratio_results["seeds"]]
                        
                        ratio_results["avg_recovery_rate"] = np.mean(recovery_rates)
                        ratio_results["std_recovery_rate"] = np.std(recovery_rates)
                        ratio_results["avg_function_preservation"] = np.mean(function_scores)
                        ratio_results["std_function_preservation"] = np.std(function_scores)
                    
                    # Store ratio results
                    method_results[f"ratio_{ratio}" if ratio is not None else "adaptive"] = ratio_results
                
                # Find best ratio for this method
                best_ratio = None
                best_recovery = -1
                
                for ratio_key, ratio_data in method_results.items():
                    if "avg_recovery_rate" in ratio_data and ratio_data["avg_recovery_rate"] > best_recovery:
                        best_recovery = ratio_data["avg_recovery_rate"]
                        best_ratio = ratio_key
                
                if best_ratio:
                    method_results["best_ratio"] = best_ratio
                    method_results["best_recovery_rate"] = best_recovery
                
                # Store method results
                model_results["methods"][method] = method_results
                
                # Also store in the global method results
                if method not in results["methods"]:
                    results["methods"][method] = {"models": {}}
                
                results["methods"][method]["models"][model_name] = method_results
            
            # Calculate best method for this model
            best_method = None
            best_recovery = -1
            
            for method, method_data in model_results["methods"].items():
                if "best_recovery_rate" in method_data and method_data["best_recovery_rate"] > best_recovery:
                    best_recovery = method_data["best_recovery_rate"]
                    best_method = method
            
            if best_method:
                model_results["best_method"] = best_method
                model_results["best_recovery_rate"] = best_recovery
            
            # Store model results
            results["models"][model_name] = model_results
        
        # Calculate overall best method
        if results["methods"]:
            for method, method_data in results["methods"].items():
                # Average across models
                recovery_rates = []
                function_scores = []
                
                for model, model_data in method_data["models"].items():
                    if "best_recovery_rate" in model_data:
                        recovery_rates.append(model_data["best_recovery_rate"])
                    
                    best_ratio = model_data.get("best_ratio")
                    if best_ratio and best_ratio in model_data:
                        ratio_data = model_data[best_ratio]
                        if "avg_function_preservation" in ratio_data:
                            function_scores.append(ratio_data["avg_function_preservation"])
                
                if recovery_rates:
                    method_data["avg_recovery_rate"] = np.mean(recovery_rates)
                    method_data["std_recovery_rate"] = np.std(recovery_rates)
                
                if function_scores:
                    method_data["avg_function_preservation"] = np.mean(function_scores)
                    method_data["std_function_preservation"] = np.std(function_scores)
            
            # Find best overall method
            best_method = None
            best_recovery = -1
            
            for method, method_data in results["methods"].items():
                if "avg_recovery_rate" in method_data and method_data["avg_recovery_rate"] > best_recovery:
                    best_recovery = method_data["avg_recovery_rate"]
                    best_method = method
            
            if best_method:
                results["best_method"] = best_method
                results["best_recovery_rate"] = best_recovery
        
        # Save results
        self._save_results(results)
        
        # Create visualizations
        self._create_visualizations(results)
        
        # Statistical significance testing
        if "adaptive" in results["methods"] and len(results["methods"]) > 1:
            p_values = {}
            
            # Get adaptive results
            adaptive_rates = []
            for model, model_data in results["methods"]["adaptive"]["models"].items():
                best_ratio = model_data.get("best_ratio")
                if best_ratio and best_ratio in model_data:
                    for seed_result in model_data[best_ratio]["seeds"]:
                        adaptive_rates.append(seed_result["recovery_rate"])
            
            # Compare with other methods
            for method in results["methods"]:
                if method == "adaptive":
                    continue
                
                method_rates = []
                for model, model_data in results["methods"][method]["models"].items():
                    best_ratio = model_data.get("best_ratio")
                    if best_ratio and best_ratio in model_data:
                        for seed_result in model_data[best_ratio]["seeds"]:
                            method_rates.append(seed_result["recovery_rate"])
                
                if adaptive_rates and method_rates:
                    # Perform statistical test
                    stat, p_value = stats.ttest_ind(adaptive_rates, method_rates)
                    p_values[method] = p_value
            
            results["statistical_tests"] = {
                "p_values": p_values
            }
            
            self._save_results(results)
        
        return results


class MultiCycleExperiment(ValidationExperiment):
    """
    Multi-cycle plasticity experiment.
    
    This experiment tests adaptation over multiple plasticity cycles:
    1. Run 5 complete plasticity cycles
    2. Track metrics across all cycles
    3. Measure cumulative effects of repeated pruning and recovery
    4. Analyze entropy patterns and function preservation over time
    """
    
    def run(self):
        """Run the multi-cycle experiment"""
        results = {
            "experiment_type": "multi_cycle",
            "models": {},
            "methods": {},
            "learning_curves": {
                "recovery_rate": {},
                "function_preservation": {},
                "perplexity": {}
            }
        }
        
        # Run for each model
        for model_name in self.config.model_names:
            logger.info(f"Running experiment for model: {model_name}")
            
            model_results = {
                "methods": {}
            }
            
            # Run for each method
            for method in self.config.methods:
                logger.info(f"Testing method: {method}")
                
                # For RL controller, we only have one adaptive ratio
                if method == "adaptive" and not self.config.rl_enabled:
                    logger.info("Skipping adaptive method (RL disabled)")
                    continue
                
                # For static methods, use best ratio from validation
                if method != "adaptive":
                    ratios = self.config.pruning_ratios
                    if len(ratios) > 1:
                        ratio = sum(ratios) / len(ratios)  # Use average as default
                    else:
                        ratio = ratios[0]
                else:
                    ratio = None
                
                method_learning_curves = {
                    "recovery_rate": [],
                    "function_preservation": [],
                    "perplexity": []
                }
                
                # Run multi-cycle for this method
                method_dir = os.path.join(self.output_dir, f"{model_name}/{method}")
                
                # Create MultiCycleRunner for this method
                if method == "adaptive":
                    # Use RL controller
                    runner = MultiCycleRunner(
                        model_name=model_name,
                        output_dir=method_dir,
                        num_cycles=self.config.num_cycles,
                        pruning_strategy="entropy",  # Initial strategy, will be overridden
                        pruning_ratio=0.3,  # Initial ratio, will be overridden
                        steps_per_cycle=self.config.steps_per_cycle,
                        batch_size=self.config.batch_size,
                        device=self.device,
                        experiment_name=f"multi_cycle_{method}"
                    )
                    
                    # Initialize RL controller
                    rl_config = RLControllerConfig(
                        output_dir=os.path.join(method_dir, "rl_controller"),
                        experiment_name=f"{model_name}_multi_cycle"
                    )
                    rl_controller = RLController(rl_config, self.device)
                    
                    # Create datasets and prepare for experiment
                    datasets = runner._create_datasets()
                    train_dataset, train_collator = datasets["train"]
                    eval_dataset, eval_collator = datasets["eval"]
                    test_prompts = datasets["test_prompts"]
                    
                    # Create data loaders
                    train_loader = runner._create_data_loader(train_dataset)
                    eval_loader = runner._create_data_loader(eval_dataset)
                    
                    # Record initial state
                    logger.info("Recording initial model state")
                    initial_metrics = runner._evaluate(runner.model, eval_loader)
                    runner.entropy_journal.record_model_state(
                        runner.model, 
                        eval_loader, 
                        cycle_idx=0, 
                        cycle_name="Initial",
                        metadata={"metrics": initial_metrics}
                    )
                    
                    # Store model version
                    model_versions = {
                        "initial": runner.model.state_dict().copy()
                    }
                    
                    # Store baseline result
                    cycle_results = [{
                        "cycle": 0,
                        "phase": "initial",
                        "metrics": initial_metrics,
                        "pruning": None
                    }]
                    
                    # Run plasticity cycles
                    for cycle in range(1, self.config.num_cycles + 1):
                        logger.info(f"Running cycle {cycle}")
                        
                        # Prepare metrics for controller
                        controller_metrics = self._prepare_controller_metrics(cycle_results)
                        
                        # Select action from RL controller
                        action = rl_controller.select_action(
                            controller_metrics, 
                            cycle, 
                            self.config.num_cycles
                        )
                        
                        # Override runner settings with controller action
                        pruning_strategy = action["strategy"]
                        pruning_ratio = action["ratio"]
                        
                        logger.info(f"RL selected strategy: {pruning_strategy}, ratio: {pruning_ratio:.2f}")
                        
                        # Create pruning strategy
                        strategy = self._create_pruning_strategy(pruning_strategy)
                        
                        # 1. Prune the model
                        pruned_model, pruning_info = runner._prune_model(
                            runner.model, train_loader, cycle
                        )
                        cycle_result = {
                            "cycle": cycle,
                            "pruning": pruning_info
                        }
                        
                        # 2. Record post-pruning state
                        logger.info(f"Recording post-pruning state for cycle {cycle}")
                        post_pruning_metrics = runner._evaluate(pruned_model, eval_loader)
                        runner.entropy_journal.record_model_state(
                            pruned_model, 
                            eval_loader, 
                            cycle_idx=cycle, 
                            cycle_name=f"Cycle_{cycle}_PostPruning",
                            metadata={
                                "phase": "post_pruning",
                                "metrics": post_pruning_metrics,
                                "pruning": {
                                    "strategy": pruning_strategy,
                                    "ratio": pruning_ratio,
                                    "pruned_heads_count": len(pruning_info["pruned_heads"])
                                }
                            }
                        )
                        
                        # Store post-pruning metrics
                        cycle_result["post_pruning"] = {
                            "metrics": post_pruning_metrics
                        }
                        
                        # 3. Fine-tune the model
                        logger.info(f"Fine-tuning model for cycle {cycle}")
                        fine_tuned_model = runner._fine_tune(
                            pruned_model, train_dataset, train_collator, cycle
                        )
                        
                        # Store model version
                        model_versions[f"cycle_{cycle}"] = fine_tuned_model.state_dict().copy()
                        
                        # 4. Record post-fine-tuning state
                        logger.info(f"Recording post-fine-tuning state for cycle {cycle}")
                        post_ft_metrics = runner._evaluate(fine_tuned_model, eval_loader)
                        
                        # Get entropy journal metrics
                        entropy_state = runner.entropy_journal.record_model_state(
                            fine_tuned_model, 
                            eval_loader, 
                            cycle_idx=cycle, 
                            cycle_name=f"Cycle_{cycle}_PostFineTuning",
                            metadata={
                                "phase": "post_fine_tuning",
                                "metrics": post_ft_metrics,
                            }
                        )
                        
                        # Store post-fine-tuning metrics
                        cycle_result["post_fine_tuning"] = {
                            "metrics": post_ft_metrics
                        }
                        
                        # Extract entropy stats
                        cycle_result["entropy_journal"] = {
                            "avg_entropy": np.mean([
                                np.mean(v) for v in entropy_state.get("entropy", {}).values()
                            ]),
                            "entropy_std": np.mean([
                                np.std(v) for v in entropy_state.get("entropy", {}).values()
                            ])
                        }
                        
                        # 5. Track function preservation (compare with previous cycle)
                        if cycle > 1:
                            logger.info(f"Tracking function preservation for cycle {cycle}")
                            
                            # Create previous model
                            prev_model = AutoModelForCausalLM.from_pretrained(model_name)
                            prev_model.load_state_dict(model_versions[f"cycle_{cycle-1}"])
                            prev_model = prev_model.to(self.device)
                            
                            # Track function
                            function_results = runner.function_tracker.track_function(
                                prev_model,
                                fine_tuned_model,
                                test_prompts,
                                runner.tokenizer,
                                cycle_idx=cycle,
                                cycle_name=f"Cycle_{cycle}"
                            )
                            
                            # Store function tracking results
                            cycle_result["function_tracking"] = {
                                "overall_score": function_results["summary"].get("overall_preservation_score", None),
                                "output_similarity": function_results["summary"].get("output", {}).get("avg_cosine_similarity", None)
                            }
                        
                        # Calculate recovery rate
                        recovery_rate = self._calculate_recovery_rate(
                            cycle_results[-1]["metrics"]["loss"],
                            post_pruning_metrics["loss"],
                            post_ft_metrics["loss"]
                        )
                        cycle_result["recovery_rate"] = recovery_rate
                        logger.info(f"Recovery rate: {recovery_rate:.4f}")
                        
                        # Store cycle result
                        cycle_results.append(cycle_result)
                        
                        # Update learning curves
                        method_learning_curves["recovery_rate"].append(recovery_rate)
                        method_learning_curves["perplexity"].append(post_ft_metrics["perplexity"])
                        
                        if "function_tracking" in cycle_result:
                            func_score = cycle_result["function_tracking"]["overall_score"]
                            method_learning_curves["function_preservation"].append(func_score)
                        
                        # Calculate reward and update controller
                        reward_metrics = {
                            "recovery_rate": recovery_rate,
                            "function_preservation": cycle_result.get("function_tracking", {}).get("overall_score", 0.0),
                            "perplexity_change": (post_ft_metrics["perplexity"] - cycle_results[-2]["metrics"]["perplexity"]) / cycle_results[-2]["metrics"]["perplexity"],
                            "entropy_change": cycle_result["entropy_journal"]["avg_entropy"] - cycle_results[-2].get("entropy_journal", {}).get("avg_entropy", 0.5)
                        }
                        
                        rl_controller.observe_reward(
                            reward_metrics,
                            cycle,
                            self.config.num_cycles,
                            done=(cycle == self.config.num_cycles)
                        )
                        
                        # Update model for next cycle
                        runner.model = fine_tuned_model
                    
                    # Create summary visualizations
                    logger.info("Creating summary visualizations")
                    runner._create_summary_visualizations()
                    
                    # Create summary report
                    logger.info("Creating summary report")
                    runner._create_summary_report()
                    
                    # Create entropy visualizations
                    logger.info("Creating entropy visualizations")
                    runner.entropy_journal.visualize_entropy_evolution()
                    runner.entropy_journal.visualize_gate_evolution()
                    runner.entropy_journal.create_summary_report()
                    
                    # Create function tracking summary
                    logger.info("Creating function tracking summary")
                    runner.function_tracker.create_summary_report()
                    
                    # Get RL controller decisions
                    rl_decisions = self._extract_rl_decisions(rl_controller)
                    
                    # Store multi-cycle results
                    method_results = {
                        "cycle_results": cycle_results,
                        "learning_curves": method_learning_curves,
                        "rl_decisions": rl_decisions,
                        "final_perplexity": post_ft_metrics["perplexity"],
                        "avg_recovery_rate": np.mean(method_learning_curves["recovery_rate"]) if method_learning_curves["recovery_rate"] else 0.0,
                        "final_recovery_rate": recovery_rate
                    }
                    
                else:
                    # Use fixed strategy
                    runner = MultiCycleRunner(
                        model_name=model_name,
                        output_dir=method_dir,
                        num_cycles=self.config.num_cycles,
                        pruning_strategy=method,
                        pruning_ratio=ratio,
                        steps_per_cycle=self.config.steps_per_cycle,
                        batch_size=self.config.batch_size,
                        device=self.device,
                        experiment_name=f"multi_cycle_{method}"
                    )
                    
                    # Run the experiment
                    runner_results = runner.run_experiment()
                    
                    # Extract learning curves
                    for cycle_idx, cycle_result in enumerate(runner_results.get("cycle_results", [])):
                        if cycle_idx == 0:
                            continue  # Skip initial cycle
                            
                        # Recovery rate
                        if "recovery" in cycle_result:
                            recovery_metrics = cycle_result["recovery"]
                            recovery_rate = recovery_metrics.get("recovery_rate", 0.0)
                            method_learning_curves["recovery_rate"].append(recovery_rate)
                        
                        # Perplexity
                        if "post_fine_tuning" in cycle_result and "metrics" in cycle_result["post_fine_tuning"]:
                            perplexity = cycle_result["post_fine_tuning"]["metrics"].get("perplexity", 0.0)
                            method_learning_curves["perplexity"].append(perplexity)
                        
                        # Function preservation
                        if "function_tracking" in cycle_result:
                            func_score = cycle_result["function_tracking"].get("overall_score", 0.0)
                            method_learning_curves["function_preservation"].append(func_score)
                    
                    # Calculate summary metrics
                    method_results = {
                        "cycle_results": runner_results.get("cycle_results", []),
                        "learning_curves": method_learning_curves,
                        "final_perplexity": method_learning_curves["perplexity"][-1] if method_learning_curves["perplexity"] else 0.0,
                        "avg_recovery_rate": np.mean(method_learning_curves["recovery_rate"]) if method_learning_curves["recovery_rate"] else 0.0,
                        "final_recovery_rate": method_learning_curves["recovery_rate"][-1] if method_learning_curves["recovery_rate"] else 0.0
                    }
                
                # Store method results
                model_results["methods"][method] = method_results
                
                # Store learning curves
                for metric, values in method_learning_curves.items():
                    if values:
                        if method not in results["learning_curves"][metric]:
                            results["learning_curves"][metric][method] = []
                        results["learning_curves"][metric][method].extend(values)
                
                # Also store in the global method results
                if method not in results["methods"]:
                    results["methods"][method] = {"models": {}}
                
                results["methods"][method]["models"][model_name] = method_results
            
            # Calculate best method for this model
            best_method = None
            best_avg_recovery = -1
            
            for method, method_data in model_results["methods"].items():
                if "avg_recovery_rate" in method_data and method_data["avg_recovery_rate"] > best_avg_recovery:
                    best_avg_recovery = method_data["avg_recovery_rate"]
                    best_method = method
            
            if best_method:
                model_results["best_method"] = best_method
                model_results["best_avg_recovery_rate"] = best_avg_recovery
            
            # Store model results
            results["models"][model_name] = model_results
        
        # Calculate overall best method
        if results["methods"]:
            for method, method_data in results["methods"].items():
                # Average across models
                avg_recovery_rates = []
                
                for model, model_data in method_data["models"].items():
                    if "avg_recovery_rate" in model_data:
                        avg_recovery_rates.append(model_data["avg_recovery_rate"])
                
                if avg_recovery_rates:
                    method_data["avg_recovery_rate"] = np.mean(avg_recovery_rates)
            
            # Find best overall method
            best_method = None
            best_avg_recovery = -1
            
            for method, method_data in results["methods"].items():
                if "avg_recovery_rate" in method_data and method_data["avg_recovery_rate"] > best_avg_recovery:
                    best_avg_recovery = method_data["avg_recovery_rate"]
                    best_method = method
            
            if best_method:
                results["best_method"] = best_method
                results["best_avg_recovery_rate"] = best_avg_recovery
        
        # Save results
        self._save_results(results)
        
        # Create visualizations
        self._create_visualizations(results)
        
        return results
    
    def _prepare_controller_metrics(self, cycle_results):
        """
        Prepare metrics from cycle results for the RL controller.
        
        Args:
            cycle_results: List of results from plasticity cycles
            
        Returns:
            Dictionary of metrics suitable for the RL controller
        """
        metrics = {}
        
        # Handle case with only one cycle (initial state)
        if len(cycle_results) <= 1:
            return {
                "avg_entropy": 0.5,
                "entropy_change": 0.0,
                "entropy_std": 0.0,
                "perplexity": cycle_results[0]["metrics"]["perplexity"],
                "perplexity_change": 0.0,
                "recovery_rate": 0.0,
                "function_preservation": 0.0,
                "function_change": 0.0,
                "last_pruning_ratio": 0.3,
                "last_strategy": "entropy"
            }
        
        # Get current and previous cycle results
        current_cycle = cycle_results[-1]
        prev_cycle = cycle_results[-2]
        
        # Calculate metrics
        
        # 1. Entropy metrics
        if "entropy_journal" in current_cycle:
            entropy_data = current_cycle["entropy_journal"]
            prev_entropy_data = prev_cycle.get("entropy_journal", {})
            
            metrics["avg_entropy"] = entropy_data.get("avg_entropy", 0.5)
            
            # Calculate entropy change
            if "avg_entropy" in prev_entropy_data:
                metrics["entropy_change"] = (
                    entropy_data.get("avg_entropy", 0.5) - 
                    prev_entropy_data.get("avg_entropy", 0.5)
                )
            else:
                metrics["entropy_change"] = 0.0
            
            metrics["entropy_std"] = entropy_data.get("entropy_std", 0.0)
        
        # 2. Performance metrics
        metrics["perplexity"] = current_cycle.get("metrics", {}).get("perplexity", 0.0)
        
        # Calculate perplexity change
        prev_perplexity = prev_cycle.get("metrics", {}).get("perplexity", 0.0)
        if prev_perplexity > 0:
            metrics["perplexity_change"] = (metrics["perplexity"] - prev_perplexity) / prev_perplexity
        else:
            metrics["perplexity_change"] = 0.0
        
        # Recovery rate from current cycle if available
        metrics["recovery_rate"] = current_cycle.get("recovery_rate", 0.0)
        
        # 3. Function preservation metrics
        if "function_tracking" in current_cycle:
            metrics["function_preservation"] = current_cycle["function_tracking"].get("overall_score", 0.0)
            
            # Calculate function change
            if "function_tracking" in prev_cycle:
                metrics["function_change"] = (
                    current_cycle["function_tracking"].get("overall_score", 0.0) - 
                    prev_cycle["function_tracking"].get("overall_score", 0.0)
                )
            else:
                metrics["function_change"] = 0.0
        
        # 4. Pruning history
        if "pruning" in current_cycle:
            metrics["last_pruning_ratio"] = current_cycle["pruning"].get("ratio", 0.3)
            metrics["last_strategy"] = current_cycle["pruning"].get("strategy", "entropy")
        
        return metrics
    
    def _extract_rl_decisions(self, rl_controller):
        """
        Extract decision history from RL controller.
        
        Args:
            rl_controller: The RL controller
            
        Returns:
            Dictionary with decision history
        """
        decisions = {
            "strategies": [],
            "ratios": []
        }
        
        for i, action in enumerate(rl_controller.action_history):
            decisions["strategies"].append(action["strategy"])
            decisions["ratios"].append(action["ratio"])
        
        return decisions


class TaskAlternationExperiment(ValidationExperiment):
    """
    Task alternation stress test.
    
    This experiment tests adaptation to task changes:
    1. Alternate between language modeling and classification tasks
    2. Apply pruning between task switches
    3. Measure adaptation to task changes
    4. Compare RL controller vs fixed strategies
    """
    
    def run(self):
        """Run the task alternation experiment"""
        # Skip for now - requires more setup for classification tasks
        logger.info("Task alternation experiment not yet implemented")
        
        results = {
            "experiment_type": "task_alternation",
            "status": "not_implemented"
        }
        
        self._save_results(results)
        
        return results


class CrossArchitectureExperiment(ValidationExperiment):
    """
    Cross-architecture validation.
    
    This experiment tests whether findings generalize across model architectures:
    1. Apply best methods from other experiments to all model architectures
    2. Analyze architecture-specific behaviors and limitations
    3. Identify common patterns in plasticity across architectures
    """
    
    def run(self):
        """Run the cross-architecture experiment"""
        # This should run a subset of the single-cycle experiment on multiple architectures
        # For now, we'll defer to the single-cycle experiment with multiple models
        experiment = SingleCycleExperiment(self.config)
        results = experiment.run()
        
        # Update experiment type
        results["experiment_type"] = "cross_architecture"
        
        self._save_results(results)
        
        return results


def run_validation_experiment(args):
    """
    Run validation experiment based on command line arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        Experiment results
    """
    # Create configuration
    config = ValidationConfig(args)
    
    # Run the appropriate experiment
    if args.experiment == "single_cycle":
        experiment = SingleCycleExperiment(config)
    elif args.experiment == "multi_cycle":
        experiment = MultiCycleExperiment(config)
    elif args.experiment == "task_alternation":
        experiment = TaskAlternationExperiment(config)
    elif args.experiment == "cross_architecture":
        experiment = CrossArchitectureExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")
    
    # Run the experiment
    results = experiment.run()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Plasticity Validation Experiment Runner")
    parser.add_argument("--experiment", type=str, default="single_cycle", 
                        choices=["single_cycle", "multi_cycle", "task_alternation", "cross_architecture"],
                        help="Type of experiment to run")
    parser.add_argument("--models", type=str, default="distilgpt2", 
                        help="Model names (comma-separated for multiple)")
    parser.add_argument("--output_dir", type=str, default="./output/validation", 
                        help="Output directory")
    parser.add_argument("--seeds", type=int, default=3, 
                        help="Number of random seeds to run")
    parser.add_argument("--pruning_ratios", type=str, default="0.1,0.3,0.5", 
                        help="Pruning ratios to test (comma-separated)")
    parser.add_argument("--cycles", type=int, default=5, 
                        help="Number of cycles for multi-cycle experiment")
    parser.add_argument("--steps", type=int, default=50, 
                        help="Training steps per cycle")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Dataset to use (leave empty for synthetic)")
    parser.add_argument("--disable_rl", action="store_true", 
                        help="Disable RL controller")
    parser.add_argument("--methods", type=str, default=None, 
                        help="Methods to test (comma-separated)")
    
    args = parser.parse_args()
    
    run_validation_experiment(args)