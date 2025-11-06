#!/usr/bin/env python
"""
Multi-Cycle Neural Plasticity Experiment Runner

This script runs automated multi-cycle plasticity experiments that:
1. Prune → fine-tune → entropy measure → visualize → repeat
2. Track entropy, function preservation, and recovery
3. Generate comprehensive reports and visualizations

This enables longitudinal studies of neural adaptation, entropy collapse,
regrowth patterns, and resilience over multiple plasticity cycles.
"""

import os
import sys
import logging
import argparse
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

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
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Import plasticity components
from sentinel.plasticity.entropy_journal import EntropyJournal, EntropyJournalConfig
from sentinel.plasticity.function_tracking import FunctionTracker, FunctionTrackingConfig
from sentinel.plasticity.stress_protocols import (
    TaskAlternationProtocol, StressProtocolConfig
)
from sentinel.visualization.entropy_rhythm_plot import (
    plot_entropy_rhythm, create_animated_entropy_rhythm, create_entropy_delta_heatmap
)

# Import pruning components
from sentinel.pruning.pruning_module import PruningModule
from sentinel.pruning.strategies import EntropyPruningStrategy, MagnitudePruningStrategy, RandomPruningStrategy


class MultiCycleRunner:
    """
    Runs automated multi-cycle neural plasticity experiments.
    
    This class orchestrates complete plasticity cycles including pruning,
    fine-tuning, entropy tracking, and function preservation measurement.
    It generates comprehensive reports and visualizations showing adaptation
    patterns over time.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        num_cycles: int = 5,
        pruning_strategy: str = "entropy",
        pruning_ratio: float = 0.3,
        steps_per_cycle: int = 100,
        batch_size: int = 4,
        device: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the multi-cycle runner.
        
        Args:
            model_name: Hugging Face model name
            output_dir: Directory to save results
            num_cycles: Number of plasticity cycles to run
            pruning_strategy: Pruning strategy ("entropy", "magnitude", "random")
            pruning_ratio: Ratio of heads to prune (0.0-1.0)
            steps_per_cycle: Training steps per cycle
            batch_size: Batch size for training
            device: Device for computation (cpu/cuda)
            experiment_name: Custom name for the experiment
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_cycles = num_cycles
        self.pruning_strategy = pruning_strategy
        self.pruning_ratio = pruning_ratio
        self.steps_per_cycle = steps_per_cycle
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"multi_cycle_{pruning_strategy}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize tracking systems
        self._init_tracking_systems()
        
        # Initialize results storage
        self.cycle_results = []
        
        logger.info(f"Initialized MultiCycleRunner (experiment: {self.experiment_name})")
        logger.info(f"Output directory: {self.experiment_dir}")
    
    def _init_tracking_systems(self):
        """Initialize entropy journal and function tracker"""
        # Entropy journal
        entropy_dir = os.path.join(self.experiment_dir, "entropy_journal")
        entropy_config = EntropyJournalConfig(
            output_dir=entropy_dir,
            experiment_name=self.experiment_name,
            create_visualizations=True
        )
        self.entropy_journal = EntropyJournal(entropy_config, self.device)
        
        # Function tracker
        function_dir = os.path.join(self.experiment_dir, "function_tracking")
        function_config = FunctionTrackingConfig(
            output_dir=function_dir,
            experiment_name=self.experiment_name
        )
        self.function_tracker = FunctionTracker(function_config, self.device)
    
    def _create_pruning_strategy(self):
        """Create the pruning strategy based on configuration"""
        if self.pruning_strategy == "entropy":
            return EntropyPruningStrategy()
        elif self.pruning_strategy == "magnitude":
            return MagnitudePruningStrategy()
        elif self.pruning_strategy == "random":
            return RandomPruningStrategy()
        else:
            logger.warning(f"Unknown pruning strategy: {self.pruning_strategy}, using entropy")
            return EntropyPruningStrategy()
    
    def _create_datasets(self):
        """Create datasets for training and evaluation"""
        # Training text
        train_texts = [
            "The transformer architecture has revolutionized natural language processing by allowing models to process sequences in parallel.",
            "Self-attention mechanisms enable each position in a sequence to attend to all other positions, capturing dependencies across the entire sequence.",
            "The field of artificial intelligence combines computer science, mathematics, and cognitive science to create systems capable of learning and problem-solving.",
            "Transformer models have become the foundation for various language tasks, including translation, summarization, and text generation.",
            "Machine learning algorithms analyze data to identify patterns and make predictions without explicit programming rules.",
            "Deep learning approaches use neural networks with multiple layers to learn hierarchical representations of data.",
            "Transfer learning allows models to apply knowledge gained from one task to improve performance on a related task.",
            "Attention mechanisms help models focus on relevant parts of the input when generating outputs.",
            "Natural language processing aims to enable computers to understand, interpret, and generate human language.",
            "Generative AI models can create new content, such as text, images, or music, based on patterns learned from training data."
        ] * 10  # Repeat to get more samples
        
        # Evaluation text (different distribution)
        eval_texts = [
            "Quantum mechanics describes the behavior of matter and energy at the smallest scales.",
            "Neural plasticity refers to the brain's ability to reorganize itself by forming new connections throughout life.",
            "The principles of thermodynamics govern energy transfer and transformation processes.",
            "Cognitive science investigates the nature of the mind and its relationship to the physical brain.",
            "Statistical methods help researchers quantify uncertainty and make inferences from data."
        ] * 5
        
        # Create output directories
        train_dir = os.path.join(self.experiment_dir, "datasets", "train")
        eval_dir = os.path.join(self.experiment_dir, "datasets", "eval")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create datasets
        train_dataset, train_collator = self._create_text_dataset(train_texts, train_dir)
        eval_dataset, eval_collator = self._create_text_dataset(eval_texts, eval_dir)
        
        # Create test prompts for function tracking
        test_prompts = [
            "The transformer architecture allows",
            "Neural networks consist of",
            "Machine learning algorithms can",
            "Attention mechanisms help models",
            "Natural language processing aims to"
        ]
        
        return {
            "train": (train_dataset, train_collator),
            "eval": (eval_dataset, eval_collator),
            "test_prompts": test_prompts
        }
    
    def _create_text_dataset(self, texts, output_dir, block_size=128):
        """Create a text dataset from a list of texts"""
        # Write texts to a file
        dataset_path = os.path.join(output_dir, "text.txt")
        with open(dataset_path, "w") as f:
            f.write("\n".join(texts))
        
        # Create dataset
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=dataset_path,
            block_size=block_size
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return dataset, data_collator
    
    def _create_data_loader(self, dataset, batch_size=None):
        """Create a data loader from a dataset"""
        from torch.utils.data import DataLoader
        
        if batch_size is None:
            batch_size = self.batch_size
        
        def collate_fn(examples):
            # Convert examples to tensors
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                padding="longest"
            )
            batch["labels"] = batch["input_ids"].clone()
            return batch
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
    
    def _fine_tune(self, model, dataset, data_collator, cycle_idx):
        """Fine-tune the model for a cycle"""
        fine_tune_dir = os.path.join(self.experiment_dir, f"finetune_cycle_{cycle_idx}")
        os.makedirs(fine_tune_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=fine_tune_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=self.batch_size,
            save_steps=self.steps_per_cycle,
            save_total_limit=1,
            max_steps=self.steps_per_cycle
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )
        
        trainer.train()
        
        return model
    
    def _evaluate(self, model, dataloader):
        """Evaluate the model on a dataloader"""
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Track loss
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def _prune_model(self, model, dataloader, cycle_idx):
        """Prune the model using the specified strategy"""
        logger.info(f"Pruning model (cycle {cycle_idx}, strategy: {self.pruning_strategy}, ratio: {self.pruning_ratio})")
        
        # Create pruning module
        strategy = self._create_pruning_strategy()
        pruning_module = PruningModule(model, strategy)
        
        # Prune the model
        pruned_heads = pruning_module.prune(
            pruning_ratio=self.pruning_ratio,
            dataloader=dataloader,
            device=self.device
        )
        
        # Get the pruning mask
        pruning_mask = pruning_module.get_pruning_mask()
        
        logger.info(f"Pruned {len(pruned_heads)} attention heads")
        
        # Save pruning information
        pruning_info = {
            "cycle": cycle_idx,
            "strategy": self.pruning_strategy,
            "ratio": self.pruning_ratio,
            "pruned_heads": pruned_heads,
            "mask": pruning_mask
        }
        
        pruning_file = os.path.join(self.experiment_dir, f"pruning_cycle_{cycle_idx}.json")
        with open(pruning_file, 'w') as f:
            # Convert to serializable format
            serializable_info = {
                "cycle": pruning_info["cycle"],
                "strategy": pruning_info["strategy"],
                "ratio": pruning_info["ratio"],
                "pruned_heads": [{"layer": h[0], "head": h[1]} for h in pruned_heads],
                "mask": {
                    str(layer): mask.cpu().numpy().tolist() 
                    for layer, mask in pruning_mask.items()
                }
            }
            json.dump(serializable_info, f, indent=2)
        
        return model, pruning_info
    
    def run_experiment(self):
        """
        Run the complete multi-cycle experiment.
        
        This method orchestrates the entire experiment:
        1. Creates datasets and baselines
        2. For each cycle:
           a. Prunes the model
           b. Records post-pruning state
           c. Fine-tunes the model
           d. Records post-fine-tuning state
           e. Tracks function preservation
        3. Creates summary reports and visualizations
        
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting multi-cycle experiment: {self.experiment_name}")
        
        # Create datasets
        logger.info("Creating datasets")
        datasets = self._create_datasets()
        train_dataset, train_collator = datasets["train"]
        eval_dataset, eval_collator = datasets["eval"]
        test_prompts = datasets["test_prompts"]
        
        # Create data loaders
        train_loader = self._create_data_loader(train_dataset)
        eval_loader = self._create_data_loader(eval_dataset)
        
        # Store model versions
        model_versions = {
            "initial": self.model.state_dict().copy()
        }
        
        # Record initial state
        logger.info("Recording initial model state")
        initial_metrics = self._evaluate(self.model, eval_loader)
        self.entropy_journal.record_model_state(
            self.model, 
            eval_loader, 
            cycle_idx=0, 
            cycle_name="Initial",
            metadata={"metrics": initial_metrics}
        )
        
        # Store baseline result
        self.cycle_results.append({
            "cycle": 0,
            "phase": "initial",
            "metrics": initial_metrics,
            "pruning": None
        })
        
        # Run plasticity cycles
        for cycle in range(1, self.num_cycles + 1):
            logger.info(f"Starting plasticity cycle {cycle}")
            cycle_results = {"cycle": cycle}
            
            # 1. Prune the model
            pruned_model, pruning_info = self._prune_model(
                self.model, train_loader, cycle
            )
            cycle_results["pruning"] = pruning_info
            
            # 2. Record post-pruning state
            logger.info(f"Recording post-pruning state for cycle {cycle}")
            post_pruning_metrics = self._evaluate(pruned_model, eval_loader)
            self.entropy_journal.record_model_state(
                pruned_model, 
                eval_loader, 
                cycle_idx=cycle, 
                cycle_name=f"Cycle_{cycle}_PostPruning",
                metadata={
                    "phase": "post_pruning",
                    "metrics": post_pruning_metrics,
                    "pruning": {
                        "strategy": self.pruning_strategy,
                        "ratio": self.pruning_ratio,
                        "pruned_heads_count": len(pruning_info["pruned_heads"])
                    }
                }
            )
            
            # Store post-pruning metrics
            cycle_results["post_pruning"] = {
                "metrics": post_pruning_metrics
            }
            
            # 3. Fine-tune the model
            logger.info(f"Fine-tuning model for cycle {cycle}")
            fine_tuned_model = self._fine_tune(
                pruned_model, train_dataset, train_collator, cycle
            )
            
            # Store model version
            model_versions[f"cycle_{cycle}"] = fine_tuned_model.state_dict().copy()
            
            # 4. Record post-fine-tuning state
            logger.info(f"Recording post-fine-tuning state for cycle {cycle}")
            post_ft_metrics = self._evaluate(fine_tuned_model, eval_loader)
            self.entropy_journal.record_model_state(
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
            cycle_results["post_fine_tuning"] = {
                "metrics": post_ft_metrics
            }
            
            # 5. Track function preservation (compare with previous cycle)
            if cycle > 1:
                logger.info(f"Tracking function preservation for cycle {cycle}")
                
                # Create previous model
                prev_model = AutoModelForCausalLM.from_pretrained(self.model_name)
                prev_model.load_state_dict(model_versions[f"cycle_{cycle-1}"])
                prev_model = prev_model.to(self.device)
                
                # Track function
                function_results = self.function_tracker.track_function(
                    prev_model,
                    fine_tuned_model,
                    test_prompts,
                    self.tokenizer,
                    cycle_idx=cycle,
                    cycle_name=f"Cycle_{cycle}"
                )
                
                # Store function tracking results
                cycle_results["function_tracking"] = {
                    "overall_score": function_results["summary"].get("overall_preservation_score", None),
                    "output_similarity": function_results["summary"].get("output", {}).get("avg_cosine_similarity", None)
                }
            
            # Store cycle results
            self.cycle_results.append(cycle_results)
            
            # Update model for next cycle
            self.model = fine_tuned_model
        
        # Create summary visualizations
        logger.info("Creating summary visualizations")
        self._create_summary_visualizations()
        
        # Create summary report
        logger.info("Creating summary report")
        self._create_summary_report()
        
        # Create entropy visualizations
        logger.info("Creating entropy visualizations")
        self.entropy_journal.visualize_entropy_evolution()
        self.entropy_journal.visualize_gate_evolution()
        self.entropy_journal.create_summary_report()
        
        # Create function tracking summary
        logger.info("Creating function tracking summary")
        self.function_tracker.create_summary_report()
        
        logger.info(f"Experiment completed successfully. Results saved to {self.experiment_dir}")
        
        return {
            "experiment_name": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "num_cycles": self.num_cycles,
            "pruning_strategy": self.pruning_strategy,
            "pruning_ratio": self.pruning_ratio,
            "cycle_results": self.cycle_results
        }
    
    def _create_summary_visualizations(self):
        """Create summary visualizations from experiment results"""
        # Create visualization directory
        viz_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extract metrics from cycle results
        cycles = [r["cycle"] for r in self.cycle_results]
        
        # Performance metrics
        perplexity_initial = [r.get("metrics", {}).get("perplexity", None) for r in self.cycle_results]
        perplexity_pruned = [r.get("post_pruning", {}).get("metrics", {}).get("perplexity", None) 
                          for r in self.cycle_results[1:]]
        perplexity_pruned = [None] + perplexity_pruned  # Add None for initial cycle
        
        perplexity_ft = [r.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", None) 
                       for r in self.cycle_results[1:]]
        perplexity_ft = [None] + perplexity_ft  # Add None for initial cycle
        
        # Plot perplexity evolution
        plt.figure(figsize=(12, 6))
        plt.plot(cycles, perplexity_initial, 'o-', label='Initial/Previous Cycle')
        plt.plot(cycles[1:], perplexity_pruned[1:], 'x--', label='Post-Pruning')
        plt.plot(cycles[1:], perplexity_ft[1:], 's-', label='Post-Fine-Tuning')
        
        plt.xlabel('Cycle')
        plt.ylabel('Perplexity')
        plt.title('Model Performance Across Plasticity Cycles')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(cycles)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "perplexity_evolution.png"))
        plt.close()
        
        # Plot function preservation if available
        if any("function_tracking" in r for r in self.cycle_results):
            plt.figure(figsize=(12, 6))
            
            # Extract function tracking scores
            function_cycles = [r["cycle"] for r in self.cycle_results if "function_tracking" in r]
            overall_scores = [r["function_tracking"]["overall_score"] for r in self.cycle_results 
                            if "function_tracking" in r]
            output_scores = [r["function_tracking"]["output_similarity"] for r in self.cycle_results 
                           if "function_tracking" in r]
            
            plt.plot(function_cycles, overall_scores, 'o-', label='Overall Preservation')
            plt.plot(function_cycles, output_scores, 's--', label='Output Similarity')
            
            plt.xlabel('Cycle')
            plt.ylabel('Similarity Score')
            plt.title('Function Preservation Across Plasticity Cycles')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)
            plt.xticks(function_cycles)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "function_preservation.png"))
            plt.close()
        
        # Create entropy rhythm plots
        try:
            journal_path = os.path.join(
                self.experiment_dir, 
                "entropy_journal", 
                self.experiment_name, 
                "entropy_journal.jsonl"
            )
            
            if os.path.exists(journal_path):
                # Static rhythm plot
                rhythm_plot_path = os.path.join(viz_dir, "entropy_rhythm.png")
                plot_entropy_rhythm_from_file(
                    journal_path,
                    rhythm_plot_path,
                    normalize=True,
                    title="Entropy Rhythm Across Plasticity Cycles"
                )
                
                # Animated rhythm plot
                animated_path = os.path.join(viz_dir, "entropy_rhythm_animated.mp4")
                create_animated_entropy_rhythm_from_file(
                    journal_path,
                    animated_path,
                    title="Entropy Evolution Across Plasticity Cycles"
                )
                
                # Delta heatmap
                delta_path = os.path.join(viz_dir, "entropy_delta.png")
                create_entropy_delta_heatmap_from_file(
                    journal_path,
                    delta_path,
                    title="Entropy Changes Between Cycles"
                )
        except Exception as e:
            logger.warning(f"Error creating rhythm plots: {str(e)}")
    
    def _create_summary_report(self):
        """Create a summary report of the experiment"""
        report_file = os.path.join(self.experiment_dir, "experiment_summary.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# Multi-Cycle Plasticity Experiment Summary\n\n")
            f.write(f"Experiment: {self.experiment_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Experiment Configuration\n\n")
            f.write(f"- Model: {self.model_name}\n")
            f.write(f"- Number of cycles: {self.num_cycles}\n")
            f.write(f"- Pruning strategy: {self.pruning_strategy}\n")
            f.write(f"- Pruning ratio: {self.pruning_ratio}\n")
            f.write(f"- Steps per cycle: {self.steps_per_cycle}\n")
            f.write(f"- Batch size: {self.batch_size}\n")
            f.write(f"- Device: {self.device}\n\n")
            
            f.write(f"## Cycle Results\n\n")
            
            # Create table for performance metrics
            f.write(f"### Performance Metrics\n\n")
            f.write(f"| Cycle | Initial Perplexity | Post-Pruning Perplexity | Post-Fine-Tuning Perplexity | Recovery Rate |\n")
            f.write(f"|-------|-------------------|--------------------------|------------------------------|---------------|\n")
            
            for i, result in enumerate(self.cycle_results):
                cycle = result["cycle"]
                
                # Initial metrics
                initial_ppl = result.get("metrics", {}).get("perplexity", "N/A")
                if isinstance(initial_ppl, float):
                    initial_ppl = f"{initial_ppl:.2f}"
                
                # Post-pruning metrics (if available)
                if i > 0:  # Skip initial cycle
                    post_pruning_ppl = result.get("post_pruning", {}).get("metrics", {}).get("perplexity", "N/A")
                    if isinstance(post_pruning_ppl, float):
                        post_pruning_ppl = f"{post_pruning_ppl:.2f}"
                        
                    post_ft_ppl = result.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", "N/A")
                    if isinstance(post_ft_ppl, float):
                        post_ft_ppl = f"{post_ft_ppl:.2f}"
                    
                    # Calculate recovery rate if all metrics are available
                    if (isinstance(result.get("metrics", {}).get("perplexity", None), float) and
                        isinstance(result.get("post_pruning", {}).get("metrics", {}).get("perplexity", None), float) and
                        isinstance(result.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", None), float)):
                        
                        init_ppl = result["metrics"]["perplexity"]
                        pruned_ppl = result["post_pruning"]["metrics"]["perplexity"]
                        ft_ppl = result["post_fine_tuning"]["metrics"]["perplexity"]
                        
                        # Calculate how much of the performance drop was recovered
                        if pruned_ppl > init_ppl:  # Performance degraded after pruning
                            recovery = (pruned_ppl - ft_ppl) / (pruned_ppl - init_ppl)
                            recovery = max(0.0, min(1.0, recovery))  # Clamp to [0, 1]
                            recovery_rate = f"{recovery:.2%}"
                        else:
                            recovery_rate = "N/A (No degradation)"
                    else:
                        recovery_rate = "N/A"
                else:
                    post_pruning_ppl = "N/A"
                    post_ft_ppl = "N/A"
                    recovery_rate = "N/A"
                
                f.write(f"| {cycle} | {initial_ppl} | {post_pruning_ppl} | {post_ft_ppl} | {recovery_rate} |\n")
            
            # Function preservation table if available
            if any("function_tracking" in r for r in self.cycle_results):
                f.write(f"\n### Function Preservation\n\n")
                f.write(f"| Cycle | Overall Score | Output Similarity |\n")
                f.write(f"|-------|---------------|-------------------|\n")
                
                for result in self.cycle_results:
                    if "function_tracking" in result:
                        cycle = result["cycle"]
                        overall = result["function_tracking"].get("overall_score", "N/A")
                        if isinstance(overall, float):
                            overall = f"{overall:.4f}"
                            
                        output = result["function_tracking"].get("output_similarity", "N/A")
                        if isinstance(output, float):
                            output = f"{output:.4f}"
                            
                        f.write(f"| {cycle} | {overall} | {output} |\n")
            
            # Pruning information
            f.write(f"\n## Pruning Statistics\n\n")
            f.write(f"| Cycle | Pruned Heads | % of Model |\n")
            f.write(f"|-------|--------------|------------|\n")
            
            for result in self.cycle_results[1:]:  # Skip initial cycle
                cycle = result["cycle"]
                if "pruning" in result and "pruned_heads" in result["pruning"]:
                    pruned_count = len(result["pruning"]["pruned_heads"])
                    
                    # Estimate total heads
                    total_heads = int(pruned_count / self.pruning_ratio) if self.pruning_ratio > 0 else 0
                    percentage = f"{(pruned_count / total_heads * 100) if total_heads > 0 else 0:.1f}%"
                    
                    f.write(f"| {cycle} | {pruned_count} | {percentage} |\n")
            
            # Key observations
            f.write(f"\n## Key Observations\n\n")
            
            # Performance trends
            if len(self.cycle_results) > 1:
                first_ppl = self.cycle_results[0].get("metrics", {}).get("perplexity", None)
                last_ppl = self.cycle_results[-1].get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", None)
                
                if isinstance(first_ppl, float) and isinstance(last_ppl, float):
                    ppl_change = (last_ppl - first_ppl) / first_ppl * 100
                    if ppl_change < -10:
                        f.write(f"- **Performance improved significantly** ({ppl_change:.1f}%) over the course of plasticity cycles.\n")
                    elif ppl_change < 0:
                        f.write(f"- **Performance improved slightly** ({ppl_change:.1f}%) over the course of plasticity cycles.\n")
                    elif ppl_change < 10:
                        f.write(f"- **Performance was stable** with minimal degradation ({ppl_change:.1f}%) despite repeated pruning.\n")
                    else:
                        f.write(f"- **Performance degraded** ({ppl_change:.1f}%) over the course of plasticity cycles.\n")
            
            # Recovery capability
            if len(self.cycle_results) > 1:
                recovery_rates = []
                
                for result in self.cycle_results[1:]:  # Skip initial cycle
                    if (isinstance(result.get("metrics", {}).get("perplexity", None), float) and
                        isinstance(result.get("post_pruning", {}).get("metrics", {}).get("perplexity", None), float) and
                        isinstance(result.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", None), float)):
                        
                        init_ppl = result["metrics"]["perplexity"]
                        pruned_ppl = result["post_pruning"]["metrics"]["perplexity"]
                        ft_ppl = result["post_fine_tuning"]["metrics"]["perplexity"]
                        
                        if pruned_ppl > init_ppl:  # Performance degraded after pruning
                            recovery = (pruned_ppl - ft_ppl) / (pruned_ppl - init_ppl)
                            recovery = max(0.0, min(1.0, recovery))
                            recovery_rates.append(recovery)
                
                if recovery_rates:
                    avg_recovery = sum(recovery_rates) / len(recovery_rates)
                    if avg_recovery > 0.9:
                        f.write(f"- **Excellent recovery capability** ({avg_recovery:.1%} on average), suggesting high neural plasticity.\n")
                    elif avg_recovery > 0.7:
                        f.write(f"- **Good recovery capability** ({avg_recovery:.1%} on average), showing resilient neural plasticity.\n")
                    elif avg_recovery > 0.5:
                        f.write(f"- **Moderate recovery capability** ({avg_recovery:.1%} on average), with partial adaptation to structural changes.\n")
                    else:
                        f.write(f"- **Limited recovery capability** ({avg_recovery:.1%} on average), suggesting challenges in adaptation.\n")
                    
                    # Look for trends in recovery
                    if len(recovery_rates) > 2:
                        first_recovery = recovery_rates[0]
                        last_recovery = recovery_rates[-1]
                        
                        if last_recovery > first_recovery * 1.1:
                            f.write(f"- **Recovery capability improved** over time, from {first_recovery:.1%} to {last_recovery:.1%}, suggesting meta-adaptation to pruning.\n")
                        elif last_recovery < first_recovery * 0.9:
                            f.write(f"- **Recovery capability declined** over time, from {first_recovery:.1%} to {last_recovery:.1%}, suggesting cumulative structural stress.\n")
            
            # Function preservation insights
            function_scores = [r.get("function_tracking", {}).get("overall_score", None) 
                              for r in self.cycle_results if "function_tracking" in r]
            
            if function_scores and all(isinstance(score, float) for score in function_scores):
                avg_score = sum(function_scores) / len(function_scores)
                if avg_score > 0.9:
                    f.write(f"- **High function preservation** ({avg_score:.1%} on average), maintaining capabilities despite structural changes.\n")
                elif avg_score > 0.7:
                    f.write(f"- **Good function preservation** ({avg_score:.1%} on average), with minimal functional drift.\n")
                elif avg_score > 0.5:
                    f.write(f"- **Moderate function preservation** ({avg_score:.1%} on average), with some functional drift.\n")
                else:
                    f.write(f"- **Low function preservation** ({avg_score:.1%} on average), suggesting significant functional changes.\n")
            
            f.write(f"\n## Conclusions\n\n")
            f.write("*Add your interpretation of these results here.*\n")
            
            f.write(f"\n## Additional Resources\n\n")
            f.write(f"- Detailed entropy journal: `entropy_journal/{self.experiment_name}/`\n")
            f.write(f"- Function tracking data: `function_tracking/{self.experiment_name}/`\n")
            f.write(f"- Visualizations: `visualizations/`\n")


def plot_entropy_rhythm_from_file(journal_path, save_path, normalize=True, smooth_window=1, title=None):
    """Load entropy journal from file and create rhythm plot"""
    df = load_entropy_journal(journal_path)
    return plot_entropy_rhythm(
        df, save_path, normalize, smooth_window, title
    )


def create_animated_entropy_rhythm_from_file(journal_path, save_path, fps=10, normalize=True, title=None):
    """Load entropy journal from file and create animated rhythm plot"""
    df = load_entropy_journal(journal_path)
    return create_animated_entropy_rhythm(
        df, save_path, fps, normalize, title
    )


def create_entropy_delta_heatmap_from_file(journal_path, save_path, title=None):
    """Load entropy journal from file and create delta heatmap"""
    df = load_entropy_journal(journal_path)
    return create_entropy_delta_heatmap(
        df, save_path, title=title
    )


def load_entropy_journal(journal_path):
    """Load entropy journal from JSONL file"""
    import pandas as pd
    import json
    
    entries = []
    with open(journal_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    return pd.DataFrame(entries)


def run_multi_cycle_experiment(args):
    """Run multi-cycle experiment with command line arguments"""
    # Create runner
    runner = MultiCycleRunner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_cycles=args.cycles,
        pruning_strategy=args.pruning_strategy,
        pruning_ratio=args.pruning_ratio,
        steps_per_cycle=args.steps_per_cycle,
        batch_size=args.batch_size,
        device=args.device,
        experiment_name=args.experiment_name
    )
    
    # Run experiment
    results = runner.run_experiment()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Cycle Neural Plasticity Experiment Runner")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--cycles", type=int, default=5, help="Number of plasticity cycles")
    parser.add_argument("--pruning_strategy", type=str, default="entropy", 
                        choices=["entropy", "magnitude", "random"], help="Pruning strategy")
    parser.add_argument("--pruning_ratio", type=float, default=0.3, help="Ratio of heads to prune (0.0-1.0)")
    parser.add_argument("--steps_per_cycle", type=int, default=100, help="Training steps per cycle")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom name for the experiment")
    
    args = parser.parse_args()
    
    run_multi_cycle_experiment(args)