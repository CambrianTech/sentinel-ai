#!/usr/bin/env python
"""
Task Alternation Protocol

This module implements the Task Alternation Protocol for testing neural plasticity.
It alternates between different tasks from a TaskSuite, measuring how models adapt
to changing task demands and how well they preserve previous capabilities.
"""

import os
import torch
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Sequence, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from torch.utils.data import DataLoader

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel, Trainer, TrainingArguments

from sentinel.plasticity.stress_protocols.task_suite import TaskSuite, TaskConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TaskAlternationConfig:
    """Configuration for Task Alternation Protocol"""
    
    # Task configuration
    tasks: List[str]  # List of task names to alternate between
    cycles: int = 5   # Number of alternation cycles
    epochs_per_task: int = 1  # Number of epochs to train on each task
    
    # Evaluation configuration
    eval_previous_tasks: bool = True  # Whether to evaluate on all previous tasks
    eval_intervals: List[int] = field(default_factory=lambda: [1])  # Eval after every N task switches
    
    # Output configuration
    output_dir: Optional[str] = None
    track_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    
    # Advanced options
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    use_wandb: bool = False
    seed: int = 42


class TaskAlternationProtocol:
    """
    Protocol that alternates between different tasks to test neural plasticity.
    
    This protocol:
    1. Trains on task A for N epochs
    2. Evaluates on task A to establish baseline
    3. Switches to task B for N epochs
    4. Evaluates on both task A and B to measure:
       - How well the model learns task B
       - How much of task A capability is preserved
    5. Repeats with additional tasks or cycles
    
    The protocol quantifies:
    - Learning efficiency per task
    - Catastrophic forgetting effects
    - Recovery rate when returning to previous tasks
    - Function preservation across task boundaries
    """
    
    def __init__(
        self,
        task_suite: TaskSuite,
        config: TaskAlternationConfig
    ):
        """
        Initialize the Task Alternation Protocol.
        
        Args:
            task_suite: The TaskSuite containing the tasks to alternate between
            config: Configuration parameters for the protocol
        """
        self.task_suite = task_suite
        self.config = config
        
        # Validate task names
        for task_name in config.tasks:
            if task_name not in task_suite.get_task_names():
                available_tasks = ", ".join(task_suite.get_task_names())
                raise ValueError(f"Task {task_name} not found in task suite. Available tasks: {available_tasks}")
        
        # Set up output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"outputs/task_alternation_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = {
            "task_sequence": [],
            "task_metrics": {},
            "forgetting_rates": {},
            "recovery_rates": {},
        }
        
        # Set random seed for reproducibility
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        logger.info(f"Initialized TaskAlternationProtocol with {len(config.tasks)} tasks over {config.cycles} cycles")
    
    def run_protocol(
        self,
        model: "Any",  # PreTrainedModel
        tokenizer: "Any",  # PreTrainedTokenizer
        fine_tuning_fn: Optional[Callable] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the task alternation protocol.
        
        Args:
            model: The model to test neural plasticity on
            tokenizer: Tokenizer for the model
            fine_tuning_fn: Optional custom fine-tuning function, if None uses default
            device: Device to run on
            
        Returns:
            Dictionary with metrics and results
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Set up dataloaders for each task
        dataloaders = {}
        for task_name in self.config.tasks:
            dataloaders[task_name] = self.task_suite.create_dataloader(
                task_name,
                tokenizer,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Initialize metrics tracking for this task
            self.metrics_history["task_metrics"][task_name] = []
        
        # Baseline evaluation on all tasks
        baseline_metrics = self._evaluate_all_tasks(model, tokenizer, self.config.tasks)
        
        # Log baseline metrics
        logger.info("Baseline metrics:")
        for task_name, metrics in baseline_metrics.items():
            logger.info(f"  {task_name}: {metrics}")
            
            # Store baseline metrics
            self.metrics_history["task_metrics"][task_name].append({
                "cycle": 0,
                "epoch": 0,
                "metrics": metrics,
                "active_task": None
            })
        
        total_iterations = self.config.cycles * len(self.config.tasks)
        current_iteration = 0
        
        # Run through cycles
        for cycle in range(self.config.cycles):
            logger.info(f"Starting cycle {cycle+1}/{self.config.cycles}")
            
            # Go through each task in the cycle
            for task_idx, task_name in enumerate(self.config.tasks):
                current_iteration += 1
                logger.info(f"Switching to task: {task_name} ({current_iteration}/{total_iterations})")
                
                # Record task sequence
                self.metrics_history["task_sequence"].append(task_name)
                
                # Fine-tune on this task
                if fine_tuning_fn:
                    # Use custom fine-tuning function if provided
                    fine_tuning_metrics = fine_tuning_fn(
                        model,
                        dataloaders[task_name],
                        task_name=task_name,
                        epochs=self.config.epochs_per_task
                    )
                else:
                    # Use default fine-tuning
                    fine_tuning_metrics = self._default_fine_tuning(
                        model,
                        dataloaders[task_name],
                        task_name,
                        epochs=self.config.epochs_per_task,
                        device=device
                    )
                
                # Evaluate on the current task
                current_task_metrics = self.task_suite.evaluate(task_name, model, tokenizer, device)
                logger.info(f"Performance on current task {task_name}: {current_task_metrics}")
                
                # Store current task metrics
                self.metrics_history["task_metrics"][task_name].append({
                    "cycle": cycle + 1,
                    "epoch": task_idx + 1,
                    "metrics": current_task_metrics,
                    "active_task": task_name
                })
                
                # Evaluate on previous tasks (measure forgetting)
                if self.config.eval_previous_tasks and current_iteration % self.config.eval_intervals[0] == 0:
                    # Get tasks to evaluate (all except current)
                    if current_iteration > 1:  # Only if we've seen at least one other task
                        eval_tasks = [t for t in self.config.tasks if t != task_name]
                        eval_metrics = self._evaluate_all_tasks(model, tokenizer, eval_tasks)
                        
                        logger.info("Performance on previous tasks:")
                        for eval_task, metrics in eval_metrics.items():
                            logger.info(f"  {eval_task}: {metrics}")
                            
                            # Store evaluation metrics
                            self.metrics_history["task_metrics"][eval_task].append({
                                "cycle": cycle + 1,
                                "epoch": task_idx + 1,
                                "metrics": metrics,
                                "active_task": task_name  # The active task during evaluation
                            })
                            
                            # Calculate forgetting rate
                            self._calculate_forgetting_rate(eval_task, cycle, task_idx)
                
                # Save checkpoint
                self._save_checkpoint(model, tokenizer, f"checkpoint_cycle{cycle+1}_task{task_idx+1}")
                
                # Save metrics
                self._save_metrics()
        
        # Final evaluation on all tasks
        final_metrics = self._evaluate_all_tasks(model, tokenizer, self.config.tasks)
        
        logger.info("Final metrics:")
        for task_name, metrics in final_metrics.items():
            logger.info(f"  {task_name}: {metrics}")
        
        # Calculate recovery rates
        for task_name in self.config.tasks:
            self._calculate_recovery_rates(task_name)
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save final metrics
        self._save_metrics()
        
        return {
            "metrics_history": self.metrics_history,
            "final_metrics": final_metrics,
            "config": self.config.__dict__
        }
    
    def _default_fine_tuning(
        self,
        model: "Any",  # PreTrainedModel
        dataloader: DataLoader,
        task_name: str,
        epochs: int = 1,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Default fine-tuning implementation if no custom function is provided.
        
        Args:
            model: Model to fine-tune
            dataloader: DataLoader for the task
            task_name: Name of the task
            epochs: Number of epochs to train
            device: Device to train on
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        metrics = {
            "train_loss": [],
            "epoch_loss": []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Track metrics
                metrics["train_loss"].append(loss.item() * self.config.gradient_accumulation_steps)
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                batch_count += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / batch_count
            metrics["epoch_loss"].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_epoch_loss:.4f}")
        
        return metrics
    
    def _evaluate_all_tasks(
        self,
        model: "Any",  # PreTrainedModel
        tokenizer: "Any",  # PreTrainedTokenizer
        task_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model on multiple tasks.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            task_names: List of task names to evaluate on
            
        Returns:
            Dictionary mapping task names to evaluation metrics
        """
        model.eval()
        results = {}
        
        for task_name in task_names:
            metrics = self.task_suite.evaluate(task_name, model, tokenizer)
            results[task_name] = metrics
        
        return results
    
    def _calculate_forgetting_rate(self, task_name: str, cycle: int, task_idx: int) -> float:
        """
        Calculate forgetting rate for a task.
        
        Args:
            task_name: Name of the task to calculate forgetting for
            cycle: Current cycle number
            task_idx: Current task index within the cycle
            
        Returns:
            Forgetting rate (0-1, higher means more forgetting)
        """
        task_metrics = self.metrics_history["task_metrics"][task_name]
        
        # Find the last time this was the active task
        last_active_idx = None
        for i, metrics_point in enumerate(reversed(task_metrics)):
            if metrics_point["active_task"] == task_name:
                last_active_idx = len(task_metrics) - 1 - i
                break
        
        # If we've never trained on this task or we only have one datapoint, return 0
        if last_active_idx is None or last_active_idx >= len(task_metrics) - 1:
            return 0.0
        
        # Get the score when the task was last active vs. current score
        last_active_score = task_metrics[last_active_idx]["metrics"]["score"]
        current_score = task_metrics[-1]["metrics"]["score"]
        
        # Calculate forgetting (normalized decrease in performance)
        if task_metrics[last_active_idx]["metrics"]["metric"] == "perplexity":
            # For perplexity, lower is better
            if last_active_score == 0:  # Avoid division by zero
                forgetting = 0.0
            else:
                forgetting = max(0, (current_score - last_active_score) / last_active_score)
        else:
            # For accuracy and other metrics, higher is better
            if last_active_score == 0:  # Avoid division by zero
                forgetting = 0.0
            else:
                forgetting = max(0, (last_active_score - current_score) / last_active_score)
        
        # Store the forgetting rate
        if task_name not in self.metrics_history["forgetting_rates"]:
            self.metrics_history["forgetting_rates"][task_name] = []
        
        self.metrics_history["forgetting_rates"][task_name].append({
            "cycle": cycle + 1,
            "epoch": task_idx + 1,
            "forgetting_rate": forgetting,
            "last_active_score": last_active_score,
            "current_score": current_score
        })
        
        return forgetting
    
    def _calculate_recovery_rates(self, task_name: str) -> None:
        """
        Calculate recovery rates for a task across cycles.
        
        Recovery rate measures how quickly a model recovers performance on a task
        after coming back to it from other tasks.
        
        Args:
            task_name: Name of the task to calculate recovery for
        """
        task_metrics = self.metrics_history["task_metrics"][task_name]
        recovery_rates = []
        
        # Find all instances where the task becomes active again
        for i in range(1, len(task_metrics)):
            current_point = task_metrics[i]
            previous_point = task_metrics[i-1]
            
            # Check if this is a transition back to this task
            if current_point["active_task"] == task_name and previous_point["active_task"] != task_name:
                # Find the last time it was active before this
                last_active_idx = None
                for j in range(i-1, -1, -1):
                    if task_metrics[j]["active_task"] == task_name:
                        last_active_idx = j
                        break
                
                # Calculate recovery rate
                if last_active_idx is not None:
                    last_active_score = task_metrics[last_active_idx]["metrics"]["score"]
                    before_recovery_score = previous_point["metrics"]["score"]
                    after_recovery_score = current_point["metrics"]["score"]
                    
                    if task_metrics[i]["metrics"]["metric"] == "perplexity":
                        # For perplexity, lower is better
                        if last_active_score == before_recovery_score:  # No forgetting
                            recovery_rate = 1.0
                        elif last_active_score == after_recovery_score:  # Full recovery
                            recovery_rate = 1.0
                        else:
                            # Calculate how much of the forgetting was recovered
                            forgetting = max(0, before_recovery_score - last_active_score)
                            recovery = max(0, before_recovery_score - after_recovery_score)
                            recovery_rate = recovery / forgetting if forgetting > 0 else 1.0
                    else:
                        # For accuracy and other metrics, higher is better
                        if last_active_score == before_recovery_score:  # No forgetting
                            recovery_rate = 1.0
                        elif last_active_score == after_recovery_score:  # Full recovery
                            recovery_rate = 1.0
                        else:
                            # Calculate how much of the forgetting was recovered
                            forgetting = max(0, last_active_score - before_recovery_score)
                            recovery = max(0, after_recovery_score - before_recovery_score)
                            recovery_rate = recovery / forgetting if forgetting > 0 else 1.0
                    
                    recovery_rates.append({
                        "cycle": current_point["cycle"],
                        "recovery_rate": recovery_rate,
                        "before_score": before_recovery_score,
                        "after_score": after_recovery_score,
                        "last_active_score": last_active_score
                    })
        
        # Store recovery rates
        self.metrics_history["recovery_rates"][task_name] = recovery_rates
    
    def _save_checkpoint(
        self,
        model: "Any",  # PreTrainedModel
        tokenizer: "Any",  # PreTrainedTokenizer
        checkpoint_name: str
    ) -> None:
        """
        Save model and tokenizer checkpoint.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _save_metrics(self) -> None:
        """Save metrics to a JSON file."""
        metrics_file = self.output_dir / "metrics_history.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")
    
    def _generate_visualizations(self) -> None:
        """Generate visualizations of metrics."""
        # Create visualization directory
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Task performance over time
        self._plot_task_performance(viz_dir / "task_performance.png")
        
        # 2. Forgetting rates
        if any(self.metrics_history["forgetting_rates"]):
            self._plot_forgetting_rates(viz_dir / "forgetting_rates.png")
        
        # 3. Recovery rates
        if any(self.metrics_history["recovery_rates"]):
            self._plot_recovery_rates(viz_dir / "recovery_rates.png")
        
        logger.info(f"Generated visualizations in {viz_dir}")
    
    def _plot_task_performance(self, save_path: Path) -> None:
        """
        Plot performance on each task over time.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for task_name, metrics_points in self.metrics_history["task_metrics"].items():
            # Extract x and y values (assuming sequential evaluation)
            x = list(range(len(metrics_points)))
            
            # For perplexity, lower is better, so we invert the scale
            if len(metrics_points) > 0 and metrics_points[0]["metrics"]["metric"] == "perplexity":
                max_score = max([p["metrics"]["score"] for p in metrics_points]) * 1.1
                y = [max_score - p["metrics"]["score"] for p in metrics_points]
                label = f"{task_name} (inverted perplexity)"
            else:
                y = [p["metrics"]["score"] for p in metrics_points]
                label = f"{task_name}"
            
            # Plot line
            plt.plot(x, y, label=label, marker='o')
            
            # Highlight points where this was the active task
            active_x = [i for i, p in enumerate(metrics_points) if p["active_task"] == task_name]
            active_y = [y[i] for i in active_x]
            plt.scatter(active_x, active_y, color='red', s=100, zorder=10)
        
        # Add task switches as vertical lines
        task_sequence = self.metrics_history["task_sequence"]
        if task_sequence:
            for i in range(1, len(task_sequence)):
                plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
        
        plt.title("Task Performance Over Time")
        plt.xlabel("Evaluation Point")
        plt.ylabel("Performance Score")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_forgetting_rates(self, save_path: Path) -> None:
        """
        Plot forgetting rates for each task.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for task_name, forgetting_points in self.metrics_history["forgetting_rates"].items():
            if not forgetting_points:
                continue
                
            # Extract x and y values
            x = list(range(len(forgetting_points)))
            y = [p["forgetting_rate"] for p in forgetting_points]
            
            # Plot line
            plt.plot(x, y, label=f"{task_name}", marker='o')
        
        plt.title("Forgetting Rates Over Time")
        plt.xlabel("Evaluation Point")
        plt.ylabel("Forgetting Rate (0-1)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_recovery_rates(self, save_path: Path) -> None:
        """
        Plot recovery rates for each task.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for task_name, recovery_points in self.metrics_history["recovery_rates"].items():
            if not recovery_points:
                continue
                
            # Extract x and y values
            x = [p["cycle"] for p in recovery_points]
            y = [p["recovery_rate"] for p in recovery_points]
            
            # Plot line
            plt.plot(x, y, label=f"{task_name}", marker='o')
        
        plt.title("Recovery Rates Across Cycles")
        plt.xlabel("Cycle")
        plt.ylabel("Recovery Rate (0-1)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def run_diverse_task_alternation(
    model: "Any",  # PreTrainedModel
    tokenizer: "Any",  # PreTrainedTokenizer
    output_dir: Optional[str] = None,
    cycles: int = 3,
    epochs_per_task: int = 1
) -> Dict[str, Any]:
    """
    Run task alternation with diverse tasks.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer for the model
        output_dir: Directory to save outputs
        cycles: Number of task alternation cycles
        epochs_per_task: Epochs to train on each task
        
    Returns:
        Dictionary with protocol results
    """
    from sentinel.plasticity.stress_protocols.task_suite import create_diverse_task_suite
    
    # Create diverse task suite
    task_suite = create_diverse_task_suite()
    
    # Configure task alternation
    config = TaskAlternationConfig(
        tasks=["commonsense_qa", "summarization", "code_completion", "nli"],
        cycles=cycles,
        epochs_per_task=epochs_per_task,
        output_dir=output_dir
    )
    
    # Create and run protocol
    protocol = TaskAlternationProtocol(task_suite, config)
    results = protocol.run_protocol(model, tokenizer)
    
    return results


def run_conflicting_task_alternation(
    model: "Any",  # PreTrainedModel
    tokenizer: "Any",  # PreTrainedTokenizer
    output_dir: Optional[str] = None,
    cycles: int = 3,
    epochs_per_task: int = 1
) -> Dict[str, Any]:
    """
    Run task alternation with conflicting tasks.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer for the model
        output_dir: Directory to save outputs
        cycles: Number of task alternation cycles
        epochs_per_task: Epochs to train on each task
        
    Returns:
        Dictionary with protocol results
    """
    from sentinel.plasticity.stress_protocols.task_suite import create_conflicting_tasks
    
    # Create conflicting task suite
    task_suite = create_conflicting_tasks()
    
    # Configure task alternation
    config = TaskAlternationConfig(
        tasks=["standard_completion", "reversed_completion", "literal_task", "idiomatic_task"],
        cycles=cycles,
        epochs_per_task=epochs_per_task,
        output_dir=output_dir
    )
    
    # Create and run protocol
    protocol = TaskAlternationProtocol(task_suite, config)
    results = protocol.run_protocol(model, tokenizer)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Loading required modules...")
    # Only import when actually running the script
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model and tokenizer
        model_name = "distilgpt2"
        print(f"Loading model and tokenizer: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Run diverse task alternation
        print("Running diverse task alternation...")
        results = run_diverse_task_alternation(
            model=model,
            tokenizer=tokenizer,
            output_dir="outputs/diverse_task_alternation",
            cycles=2,
            epochs_per_task=1
        )
        
        print(f"Task alternation completed. Results saved to {results['config']['output_dir']}")
    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("This script requires transformers and datasets to be installed.")