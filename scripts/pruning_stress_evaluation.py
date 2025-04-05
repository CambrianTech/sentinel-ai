#!/usr/bin/env python
"""
Pruning Stress Evaluation

This script runs the task alternation protocol on pruned models to evaluate
how pruning affects neural plasticity, recovery, and adaptation under stress.
It integrates the pruning module with the stress protocols and tracks metrics
using the entropy journal and function tracker.
"""

import os
import sys
import argparse
import logging
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try importing with aliasing to avoid conflicts
    import datasets as hf_datasets
except ImportError:
    print("Warning: Failed to import HuggingFace datasets. Some functionality may be limited.")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Sentinel modules
from sentinel.plasticity.stress_protocols import (
    TaskSuite, TaskConfig, TaskExample,
    create_diverse_task_suite, create_memory_stress_task, create_conflicting_tasks,
    TaskAlternationConfig, TaskAlternationProtocol
)

from sentinel.plasticity import (
    EntropyJournal, EntropyJournalConfig, record_entropy,
    ModelProbe, FunctionTracker, track_function
)

from sentinel.pruning.pruning_module import (
    prune_model, 
    get_pruning_method
)

from sentinel.utils.metrics import (
    calculate_perplexity,
    evaluate_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pruning_stress_eval.log")
    ]
)
logger = logging.getLogger(__name__)


class PruningStressEvaluator:
    """
    Evaluates how pruned models respond to stress test protocols.
    
    This class conducts experiments to measure:
    1. How pruning affects a model's ability to adapt to new tasks
    2. Recovery rates after pruning under different task protocols
    3. The relationship between pruning strategies and plasticity
    """
    
    def __init__(
        self,
        model_name: str,
        pruning_strategies: List[str],
        pruning_levels: List[float],
        output_dir: str,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name or path of the model to evaluate
            pruning_strategies: List of pruning strategies to test
            pruning_levels: List of pruning ratios to test
            output_dir: Directory to save results
            device: Device to run on
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.pruning_strategies = pruning_strategies
        self.pruning_levels = pruning_levels
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tracking containers
        self.results = {}
        self.experiment_summaries = []
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        logger.info(f"Initialized PruningStressEvaluator for model {model_name}")
        logger.info(f"Testing {len(pruning_strategies)} strategies × {len(pruning_levels)} levels")
        logger.info(f"Device: {self.device}, Output: {self.output_dir}")
        
    def run_evaluation(
        self,
        protocol_type: str = "diverse",
        cycles: int = 3,
        epochs_per_task: int = 1,
        track_entropy: bool = True,
        track_function_preservation: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full evaluation experiment.
        
        Args:
            protocol_type: Type of stress protocol to use ('diverse', 'memory', 'conflict')
            cycles: Number of task alternation cycles
            epochs_per_task: Number of epochs to train on each task
            track_entropy: Whether to track attention entropy during the experiment
            track_function_preservation: Whether to track function preservation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation with {protocol_type} protocol")
        
        # Create results container for this run
        results_key = f"{protocol_type}_{cycles}_{epochs_per_task}"
        self.results[results_key] = {
            "config": {
                "protocol": protocol_type,
                "cycles": cycles,
                "epochs_per_task": epochs_per_task,
                "track_entropy": track_entropy,
                "track_function_preservation": track_function_preservation
            },
            "pruning_results": {}
        }
        
        # Run experiments for each pruning configuration
        for strategy in self.pruning_strategies:
            for level in self.pruning_levels:
                config_name = f"{strategy}_{level}"
                logger.info(f"Evaluating pruning configuration: {config_name}")
                
                # Create experiment directory
                exp_dir = self.output_dir / f"{protocol_type}_{config_name}"
                os.makedirs(exp_dir, exist_ok=True)
                
                # Run experiment
                experiment_results = self._run_experiment(
                    strategy=strategy,
                    level=level,
                    protocol_type=protocol_type,
                    cycles=cycles,
                    epochs_per_task=epochs_per_task,
                    output_dir=str(exp_dir),
                    track_entropy=track_entropy,
                    track_function_preservation=track_function_preservation
                )
                
                # Store results
                self.results[results_key]["pruning_results"][config_name] = experiment_results
                
                # Add to summaries for comparative analysis
                self.experiment_summaries.append({
                    "protocol": protocol_type,
                    "strategy": strategy,
                    "level": level,
                    "cycles": cycles,
                    "recovery_rate": experiment_results.get("avg_recovery_rate", 0),
                    "forgetting_rate": experiment_results.get("avg_forgetting_rate", 0),
                    "final_performance": experiment_results.get("final_performance", 0)
                })
        
        # Generate comparative visualizations
        self._generate_comparative_visuals(results_key)
        
        # Save overall results
        self._save_results()
        
        return self.results[results_key]
    
    def _run_experiment(
        self,
        strategy: str,
        level: float,
        protocol_type: str,
        cycles: int,
        epochs_per_task: int,
        output_dir: str,
        track_entropy: bool = True,
        track_function_preservation: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single experiment with a specific pruning configuration.
        
        Args:
            strategy: Pruning strategy
            level: Pruning level (0-1)
            protocol_type: Type of stress protocol
            cycles: Number of task alternation cycles
            epochs_per_task: Epochs per task
            output_dir: Directory to save results
            track_entropy: Whether to track entropy
            track_function_preservation: Whether to track function preservation
            
        Returns:
            Dictionary with experiment results
        """
        # Load the base model and tokenizer
        model, tokenizer = self._load_model(self.model_name)
        
        # Create task suite based on protocol type
        if protocol_type == "diverse":
            task_suite = create_diverse_task_suite()
            task_names = ["commonsense_qa", "summarization", "code_completion"]
        elif protocol_type == "memory":
            task_suite = create_memory_stress_task()
            task_names = ["long_context", "key_value_recall"]
        elif protocol_type == "conflict":
            task_suite = create_conflicting_tasks()
            task_names = ["standard_completion", "reversed_completion", "literal_task", "idiomatic_task"]
        else:
            raise ValueError(f"Unknown protocol type: {protocol_type}")
        
        # Create tracking tools if requested
        entropy_journal = None
        function_tracker = None
        
        if track_entropy:
            entropy_config = EntropyJournalConfig(
                output_dir=os.path.join(output_dir, "entropy_tracking"),
                experiment_name=f"{strategy}_{level}_{protocol_type}"
            )
            entropy_journal = EntropyJournal(entropy_config, device=self.device)
            
        if track_function_preservation:
            function_tracker = FunctionTracker(
                output_dir=os.path.join(output_dir, "function_tracking")
            )
            
            # Store baseline function signature
            sample_dataloader = task_suite.create_dataloader(task_names[0], tokenizer, batch_size=4)
            function_tracker.record_baseline(model, sample_dataloader, tokenizer)
        
        # Prune the model
        logger.info(f"Pruning model with strategy={strategy}, level={level}")
        pruning_method = get_pruning_method(strategy)
        pruned_model = prune_model(model, level=level, method=pruning_method)
        
        # Record post-pruning entropy if tracking
        if entropy_journal:
            for task_name in task_names:
                task_dataloader = task_suite.create_dataloader(task_name, tokenizer, batch_size=4)
                entropy_journal.record_model_state(
                    pruned_model, 
                    task_dataloader, 
                    cycle_idx=0, 
                    cycle_name="Post-Pruning",
                    metadata={"task": task_name, "pruning_strategy": strategy, "pruning_level": level}
                )
                
        # Track function preservation after pruning if requested
        if function_tracker:
            function_tracker.track_function_preservation(
                pruned_model, 
                sample_dataloader, 
                tokenizer,
                step_name=f"post_pruning_{strategy}_{level}"
            )
        
        # Configure task alternation
        task_config = TaskAlternationConfig(
            tasks=task_names,
            cycles=cycles,
            epochs_per_task=epochs_per_task,
            output_dir=output_dir,
            eval_intervals=[1]  # Evaluate after each task
        )
        
        # Create protocol
        protocol = TaskAlternationProtocol(task_suite, task_config)
        
        # Custom fine-tuning function that integrates tracking
        def tracked_fine_tuning(model, dataloader, task_name, epochs):
            """Custom fine-tuning with integrated tracking"""
            # Default fine-tuning
            metrics = protocol._default_fine_tuning(
                model, dataloader, task_name, epochs, device=self.device
            )
            
            # Track entropy if enabled
            if entropy_journal:
                entropy_journal.record_model_state(
                    model, 
                    dataloader, 
                    cycle_idx=epochs,  # Use epoch as cycle index
                    cycle_name=f"{task_name}_epoch{epochs}",
                    metadata={"task": task_name, "active_task": task_name}
                )
                
            # Track function preservation if enabled
            if function_tracker:
                function_tracker.track_function_preservation(
                    model, 
                    dataloader, 
                    tokenizer,
                    step_name=f"{task_name}_epoch{epochs}"
                )
            
            return metrics
        
        # Run the protocol
        logger.info(f"Running task alternation protocol")
        results = protocol.run_protocol(
            pruned_model, 
            tokenizer, 
            fine_tuning_fn=tracked_fine_tuning if (track_entropy or track_function_preservation) else None,
            device=self.device
        )
        
        # Calculate summary metrics
        final_metrics = results.get("final_metrics", {})
        
        # Calculate average recovery rate
        recovery_rates = []
        for task_metrics in results.get("metrics_history", {}).get("recovery_rates", {}).values():
            if task_metrics:
                recovery_rates.extend([m["recovery_rate"] for m in task_metrics])
        
        avg_recovery_rate = sum(recovery_rates) / len(recovery_rates) if recovery_rates else 0
        
        # Calculate average forgetting rate
        forgetting_rates = []
        for task_metrics in results.get("metrics_history", {}).get("forgetting_rates", {}).values():
            if task_metrics:
                forgetting_rates.extend([m["forgetting_rate"] for m in task_metrics])
        
        avg_forgetting_rate = sum(forgetting_rates) / len(forgetting_rates) if forgetting_rates else 0
        
        # Calculate final average performance
        final_performances = [m["score"] for m in final_metrics.values()]
        final_performance = sum(final_performances) / len(final_performances) if final_performances else 0
        
        # Compile summary results
        summary = {
            "strategy": strategy,
            "level": level,
            "avg_recovery_rate": avg_recovery_rate,
            "avg_forgetting_rate": avg_forgetting_rate,
            "final_performance": final_performance,
            "task_metrics": {task: metrics for task, metrics in final_metrics.items()},
            "protocol_results_path": output_dir
        }
        
        # Generate entropy visualizations if tracking
        if entropy_journal:
            entropy_journal.visualize_entropy_evolution()
            if hasattr(entropy_journal, "visualize_gate_evolution"):
                entropy_journal.visualize_gate_evolution()
            entropy_journal.create_summary_report()
            summary["entropy_journal_path"] = entropy_journal.output_dir
            
        # Generate function preservation visualizations if tracking
        if function_tracker and hasattr(function_tracker, "visualize_preservation"):
            function_tracker.visualize_preservation()
            summary["function_tracker_path"] = function_tracker.output_dir
        
        return summary
    
    def _load_model(self, model_name: str) -> Tuple[torch.nn.Module, Any]:
        """Load model and tokenizer"""
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(self.device)
        
        return model, tokenizer
    
    def _generate_comparative_visuals(self, results_key: str) -> None:
        """Generate comparative visualizations for the experiments"""
        results = self.results[results_key]
        protocol = results["config"]["protocol"]
        
        vis_dir = self.output_dir / f"{protocol}_comparison"
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Compare recovery rates across pruning strategies and levels
        strategies = sorted(set(exp["strategy"] for exp in self.experiment_summaries 
                               if exp["protocol"] == protocol))
        levels = sorted(set(exp["level"] for exp in self.experiment_summaries
                           if exp["protocol"] == protocol))
        
        # Create a matrix of recovery rates
        recovery_matrix = np.zeros((len(strategies), len(levels)))
        forgetting_matrix = np.zeros((len(strategies), len(levels)))
        performance_matrix = np.zeros((len(strategies), len(levels)))
        
        for exp in self.experiment_summaries:
            if exp["protocol"] != protocol:
                continue
                
            i = strategies.index(exp["strategy"])
            j = levels.index(exp["level"])
            
            recovery_matrix[i, j] = exp["recovery_rate"]
            forgetting_matrix[i, j] = exp["forgetting_rate"]
            performance_matrix[i, j] = exp["final_performance"]
        
        # Plot recovery rate heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(recovery_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Recovery Rate')
        plt.xticks(range(len(levels)), [f"{level:.1f}" for level in levels])
        plt.yticks(range(len(strategies)), strategies)
        plt.xlabel('Pruning Level')
        plt.ylabel('Pruning Strategy')
        plt.title(f'Recovery Rate by Pruning Strategy and Level - {protocol.capitalize()} Protocol')
        
        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(levels)):
                plt.text(j, i, f"{recovery_matrix[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if recovery_matrix[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"recovery_rate_comparison.png")
        plt.close()
        
        # Plot forgetting rate heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(forgetting_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Forgetting Rate')
        plt.xticks(range(len(levels)), [f"{level:.1f}" for level in levels])
        plt.yticks(range(len(strategies)), strategies)
        plt.xlabel('Pruning Level')
        plt.ylabel('Pruning Strategy')
        plt.title(f'Forgetting Rate by Pruning Strategy and Level - {protocol.capitalize()} Protocol')
        
        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(levels)):
                plt.text(j, i, f"{forgetting_matrix[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if forgetting_matrix[i, j] > 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"forgetting_rate_comparison.png")
        plt.close()
        
        # Plot final performance heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(performance_matrix, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Performance Score')
        plt.xticks(range(len(levels)), [f"{level:.1f}" for level in levels])
        plt.yticks(range(len(strategies)), strategies)
        plt.xlabel('Pruning Level')
        plt.ylabel('Pruning Strategy')
        plt.title(f'Final Performance by Pruning Strategy and Level - {protocol.capitalize()} Protocol')
        
        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(levels)):
                plt.text(j, i, f"{performance_matrix[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if performance_matrix[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"performance_comparison.png")
        plt.close()
        
        # Create radar chart comparing strategies
        if len(strategies) > 1:
            plt.figure(figsize=(10, 10))
            
            # Set up the radar chart
            attributes = ['Recovery Rate', 'Retention (1-Forgetting)', 'Performance']
            angles = np.linspace(0, 2*np.pi, len(attributes), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Create subplot with polar projection
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], attributes)
            
            # Plot each strategy (averaged across pruning levels)
            for strategy in strategies:
                strategy_exps = [exp for exp in self.experiment_summaries 
                                if exp["protocol"] == protocol and exp["strategy"] == strategy]
                
                avg_recovery = sum(exp["recovery_rate"] for exp in strategy_exps) / len(strategy_exps)
                avg_retention = 1 - sum(exp["forgetting_rate"] for exp in strategy_exps) / len(strategy_exps)
                avg_performance = sum(exp["final_performance"] for exp in strategy_exps) / len(strategy_exps)
                
                values = [avg_recovery, avg_retention, avg_performance]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=strategy)
                ax.fill(angles, values, alpha=0.25)
            
            plt.legend(loc='upper right')
            plt.title(f'Pruning Strategy Comparison - {protocol.capitalize()} Protocol')
            plt.savefig(vis_dir / f"strategy_radar_comparison.png")
            plt.close()
    
    def _save_results(self) -> None:
        """Save all results to disk"""
        results_file = self.output_dir / "pruning_stress_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        summaries_file = self.output_dir / "experiment_summaries.json"
        
        with open(summaries_file, 'w') as f:
            json.dump(self.experiment_summaries, f, indent=2)
            
        logger.info(f"Saved results to {results_file} and {summaries_file}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate neural plasticity of pruned models under stress testing"
    )
    
    parser.add_argument(
        "--model", type=str, default="distilgpt2",
        help="Model to test (default: distilgpt2)"
    )
    parser.add_argument(
        "--strategies", type=str, default="entropy,random",
        help="Comma-separated list of pruning strategies to test"
    )
    parser.add_argument(
        "--levels", type=str, default="0.1,0.3,0.5",
        help="Comma-separated list of pruning levels to test"
    )
    parser.add_argument(
        "--protocol", type=str, default="diverse", choices=["diverse", "memory", "conflict"],
        help="Stress protocol to use for testing"
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Number of task alternation cycles (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Epochs per task (default: 1)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/pruning_stress_eval",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_entropy", action="store_true",
        help="Disable entropy tracking"
    )
    parser.add_argument(
        "--no_function", action="store_true",
        help="Disable function preservation tracking"
    )
    
    args = parser.parse_args()
    
    # Process arguments
    strategies = [s.strip() for s in args.strategies.split(",")]
    levels = [float(l.strip()) for l in args.levels.split(",")]
    
    # Create evaluator
    evaluator = PruningStressEvaluator(
        model_name=args.model,
        pruning_strategies=strategies,
        pruning_levels=levels,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    # Run evaluation
    evaluator.run_evaluation(
        protocol_type=args.protocol,
        cycles=args.cycles,
        epochs_per_task=args.epochs,
        track_entropy=not args.no_entropy,
        track_function_preservation=not args.no_function
    )
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()