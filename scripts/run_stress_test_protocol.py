#!/usr/bin/env python
"""
Run Stress Test Protocol

This script runs the neural plasticity stress test protocols on different models,
tracking how they adapt to changing tasks and maintain functionality over time.
It supports diverse task alternation, memory stress testing, and task conflict testing.
"""

import os
import sys
import logging
import argparse
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentinel.plasticity.stress_protocols import (
    TaskSuite, create_diverse_task_suite, create_memory_stress_task, create_conflicting_tasks
)
from sentinel.plasticity.stress_protocols.task_alternation import (
    TaskAlternationConfig, TaskAlternationProtocol,
    run_diverse_task_alternation, run_conflicting_task_alternation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def setup_output_dir(base_dir: str, model_name: str, protocol_name: str) -> Path:
    """
    Set up output directory for stress test results.
    
    Args:
        base_dir: Base output directory
        model_name: Name of the model being tested
        protocol_name: Name of the protocol being run
        
    Returns:
        Path object for the output directory
    """
    # Clean model name for path
    model_name_clean = model_name.replace('/', '_')
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{protocol_name}_{model_name_clean}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def load_model_and_tokenizer(model_name: str, device: Optional[str] = None) -> tuple:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Name/path of the model to load
        device: Device to load model on (default: auto-detect)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device if device == 'auto' else None,
    )
    
    if device != 'auto':
        model = model.to(device)
    
    logger.info(f"Model loaded on {device}")
    return model, tokenizer


def run_multi_model_stress_test(
    model_names: List[str],
    protocol_name: str,
    output_dir: str,
    cycles: int = 3,
    epochs_per_task: int = 1,
    device: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run stress tests on multiple models with the specified protocol.
    
    Args:
        model_names: List of model names/paths to test
        protocol_name: Name of protocol to run ('diverse', 'memory', 'conflict')
        output_dir: Base directory for outputs
        cycles: Number of task alternation cycles
        epochs_per_task: Epochs to train on each task
        device: Device to run on
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results summary
    """
    results_summary = {
        "protocol": protocol_name,
        "models": {},
        "seed": seed,
        "cycles": cycles,
        "epochs_per_task": epochs_per_task,
        "timestamp": datetime.now().isoformat()
    }
    
    for model_name in model_names:
        logger.info(f"Running {protocol_name} protocol on model: {model_name}")
        
        # Set up model-specific output directory
        model_output_dir = setup_output_dir(output_dir, model_name, protocol_name)
        
        try:
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(model_name, device)
            
            # Run appropriate protocol
            if protocol_name == 'diverse':
                results = run_diverse_task_alternation(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=str(model_output_dir),
                    cycles=cycles,
                    epochs_per_task=epochs_per_task
                )
            elif protocol_name == 'conflict':
                results = run_conflicting_task_alternation(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=str(model_output_dir),
                    cycles=cycles,
                    epochs_per_task=epochs_per_task
                )
            else:
                raise ValueError(f"Unknown protocol: {protocol_name}")
            
            # Extract key metrics
            final_metrics = results.get('final_metrics', {})
            forgetting_rates = {}
            recovery_rates = {}
            
            for task, rates in results.get('metrics_history', {}).get('forgetting_rates', {}).items():
                if rates:
                    forgetting_rates[task] = sum(r['forgetting_rate'] for r in rates) / len(rates)
            
            for task, rates in results.get('metrics_history', {}).get('recovery_rates', {}).items():
                if rates:
                    recovery_rates[task] = sum(r['recovery_rate'] for r in rates) / len(rates)
            
            # Store results summary
            results_summary["models"][model_name] = {
                "final_metrics": final_metrics,
                "avg_forgetting_rates": forgetting_rates,
                "avg_recovery_rates": recovery_rates,
                "output_dir": str(model_output_dir)
            }
            
            logger.info(f"Completed {protocol_name} protocol for {model_name}")
            
        except Exception as e:
            logger.error(f"Error running protocol for {model_name}: {str(e)}")
            results_summary["models"][model_name] = {
                "error": str(e)
            }
    
    # Save summary results
    summary_file = Path(output_dir) / f"{protocol_name}_multi_model_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Saved multi-model summary to {summary_file}")
    return results_summary


def generate_comparison_visualizations(results_summary: Dict[str, Any], output_dir: str) -> None:
    """
    Generate visualizations comparing models on stress tests.
    
    Args:
        results_summary: Results from multi-model stress test
        output_dir: Directory to save visualizations
    """
    viz_dir = Path(output_dir) / "comparisons"
    os.makedirs(viz_dir, exist_ok=True)
    
    protocol = results_summary["protocol"]
    model_names = list(results_summary["models"].keys())
    
    # 1. Compare forgetting rates
    plt.figure(figsize=(12, 8))
    
    # Collect forgetting data
    task_names = set()
    for model_data in results_summary["models"].values():
        if "avg_forgetting_rates" in model_data:
            task_names.update(model_data["avg_forgetting_rates"].keys())
    
    task_names = sorted(list(task_names))
    
    # Create grouped bar chart
    x = list(range(len(task_names)))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        model_data = results_summary["models"][model_name]
        if "avg_forgetting_rates" not in model_data:
            continue
            
        model_name_short = model_name.split('/')[-1]
        forgetting_rates = [model_data["avg_forgetting_rates"].get(task, 0) for task in task_names]
        
        plt.bar([pos + i * width - 0.4 + width/2 for pos in x], forgetting_rates, width, label=model_name_short)
    
    plt.xlabel("Task")
    plt.ylabel("Average Forgetting Rate (lower is better)")
    plt.title(f"Forgetting Rate Comparison - {protocol.capitalize()} Protocol")
    plt.xticks(x, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(viz_dir / f"{protocol}_forgetting_comparison.png")
    plt.close()
    
    # 2. Compare recovery rates
    plt.figure(figsize=(12, 8))
    
    # Collect recovery data
    task_names = set()
    for model_data in results_summary["models"].values():
        if "avg_recovery_rates" in model_data:
            task_names.update(model_data["avg_recovery_rates"].keys())
    
    task_names = sorted(list(task_names))
    
    if task_names:
        # Create grouped bar chart
        x = list(range(len(task_names)))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            model_data = results_summary["models"][model_name]
            if "avg_recovery_rates" not in model_data:
                continue
                
            model_name_short = model_name.split('/')[-1]
            recovery_rates = [model_data["avg_recovery_rates"].get(task, 0) for task in task_names]
            
            plt.bar([pos + i * width - 0.4 + width/2 for pos in x], recovery_rates, width, label=model_name_short)
        
        plt.xlabel("Task")
        plt.ylabel("Average Recovery Rate (higher is better)")
        plt.title(f"Recovery Rate Comparison - {protocol.capitalize()} Protocol")
        plt.xticks(x, task_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(viz_dir / f"{protocol}_recovery_comparison.png")
        plt.close()
    
    # 3. Compare final metrics (if applicable)
    task_scores = {}
    
    for model_name in model_names:
        model_data = results_summary["models"][model_name]
        if "final_metrics" not in model_data:
            continue
            
        for task_name, metrics in model_data["final_metrics"].items():
            if task_name not in task_scores:
                task_scores[task_name] = []
            
            task_scores[task_name].append((model_name, metrics["score"]))
    
    for task_name, scores in task_scores.items():
        plt.figure(figsize=(10, 6))
        
        model_names_short = [model_name.split('/')[-1] for model_name, _ in scores]
        task_scores_values = [score for _, score in scores]
        
        plt.bar(model_names_short, task_scores_values)
        plt.xlabel("Model")
        plt.ylabel("Final Score")
        plt.title(f"Final Performance on {task_name} - {protocol.capitalize()} Protocol")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(viz_dir / f"{protocol}_{task_name}_comparison.png")
        plt.close()
    
    logger.info(f"Generated comparison visualizations in {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run neural plasticity stress test protocols")
    
    parser.add_argument(
        "--models", type=str, required=True,
        help="Comma-separated list of model names"
    )
    parser.add_argument(
        "--protocol", type=str, default="diverse", choices=["diverse", "conflict"],
        help="Stress test protocol to run"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/stress_tests",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Number of task alternation cycles"
    )
    parser.add_argument(
        "--epochs_per_task", type=int, default=1,
        help="Epochs to train on each task"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Parse model names
    model_names = [name.strip() for name in args.models.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run stress tests
    results_summary = run_multi_model_stress_test(
        model_names=model_names,
        protocol_name=args.protocol,
        output_dir=args.output_dir,
        cycles=args.cycles,
        epochs_per_task=args.epochs_per_task,
        device=args.device,
        seed=args.seed
    )
    
    # Generate comparison visualizations
    generate_comparison_visualizations(results_summary, args.output_dir)
    
    logger.info(f"Stress test protocol completed. Results in {args.output_dir}")


if __name__ == "__main__":
    main()