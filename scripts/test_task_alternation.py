#!/usr/bin/env python
"""
Test Task Alternation Protocol

This script provides a simple test of the task alternation protocol
with a small model to verify functionality.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer

from sentinel.plasticity.stress_protocols import (
    create_diverse_task_suite,
    create_conflicting_tasks,
    TaskAlternationConfig,
    TaskAlternationProtocol
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_diverse_tasks(model_name: str, output_dir: str = "outputs/task_alternation_test"):
    """
    Test the task alternation protocol with diverse tasks.
    
    Args:
        model_name: Name of the model to test
        output_dir: Directory to save outputs
    """
    logger.info(f"Testing diverse task alternation with {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create task suite
    task_suite = create_diverse_task_suite()
    
    # Configure task alternation with minimal settings for testing
    config = TaskAlternationConfig(
        tasks=["commonsense_qa", "summarization"],  # Use only 2 tasks for quick testing
        cycles=2,  # Only run 2 cycles
        epochs_per_task=1,
        output_dir=output_dir,
        batch_size=2,  # Small batch size for testing
        eval_intervals=[1]  # Evaluate after each task
    )
    
    # Create and run protocol
    protocol = TaskAlternationProtocol(task_suite, config)
    results = protocol.run_protocol(model, tokenizer)
    
    logger.info(f"Test completed. Results saved to {output_dir}")
    
    # Print summary
    print("\n=== Task Alternation Test Results ===")
    for task_name, metrics_points in results['metrics_history']['task_metrics'].items():
        if metrics_points:
            initial_score = metrics_points[0]['metrics']['score']
            final_score = metrics_points[-1]['metrics']['score']
            print(f"{task_name}: Initial score: {initial_score:.4f}, Final score: {final_score:.4f}")
    
    # Check if forgetting rates were calculated
    if results['metrics_history']['forgetting_rates']:
        print("\nAverage Forgetting Rates:")
        for task_name, forgetting_points in results['metrics_history']['forgetting_rates'].items():
            if forgetting_points:
                avg_forgetting = sum(p['forgetting_rate'] for p in forgetting_points) / len(forgetting_points)
                print(f"{task_name}: {avg_forgetting:.4f}")
    
    return results


def test_conflicting_tasks(model_name: str, output_dir: str = "outputs/task_alternation_conflict_test"):
    """
    Test the task alternation protocol with conflicting tasks.
    
    Args:
        model_name: Name of the model to test
        output_dir: Directory to save outputs
    """
    logger.info(f"Testing conflicting task alternation with {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create task suite
    task_suite = create_conflicting_tasks()
    
    # Configure task alternation with minimal settings for testing
    config = TaskAlternationConfig(
        tasks=["standard_completion", "reversed_completion"],  # Use only 2 tasks for quick testing
        cycles=2,  # Only run 2 cycles
        epochs_per_task=1,
        output_dir=output_dir,
        batch_size=2,  # Small batch size for testing
        eval_intervals=[1]  # Evaluate after each task
    )
    
    # Create and run protocol
    protocol = TaskAlternationProtocol(task_suite, config)
    results = protocol.run_protocol(model, tokenizer)
    
    logger.info(f"Test completed. Results saved to {output_dir}")
    
    # Print summary
    print("\n=== Conflicting Task Alternation Test Results ===")
    for task_name, metrics_points in results['metrics_history']['task_metrics'].items():
        if metrics_points:
            initial_score = metrics_points[0]['metrics']['score']
            final_score = metrics_points[-1]['metrics']['score']
            print(f"{task_name}: Initial score: {initial_score:.4f}, Final score: {final_score:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Task Alternation Protocol")
    
    parser.add_argument(
        "--model", type=str, default="distilgpt2",
        help="Model to test with (default: distilgpt2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/task_alternation_test",
        help="Directory to save test outputs"
    )
    parser.add_argument(
        "--test_type", type=str, default="diverse", choices=["diverse", "conflict", "both"],
        help="Type of test to run (diverse, conflict, or both)"
    )
    
    args = parser.parse_args()
    
    if args.test_type in ["diverse", "both"]:
        diverse_output_dir = os.path.join(args.output_dir, "diverse")
        test_diverse_tasks(args.model, diverse_output_dir)
    
    if args.test_type in ["conflict", "both"]:
        conflict_output_dir = os.path.join(args.output_dir, "conflict")
        test_conflicting_tasks(args.model, conflict_output_dir)
    
    logger.info("All tests completed.")


if __name__ == "__main__":
    main()