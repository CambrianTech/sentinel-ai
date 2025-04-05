#!/usr/bin/env python
"""
Controller-Plasticity Integration

This script implements the integration between the Sentinel-AI neural controller system
and the Adaptive Plasticity System, creating a feedback loop where:

1. The controller guides the plasticity decisions (pruning and growth)
2. The plasticity system's metrics inform the controller's learning
3. Dynamic feedback loops allow for continuous self-optimization

Usage:
    python controller_plasticity_integration.py --model_name distilgpt2 --dataset tiny_shakespeare
"""

import os
import sys
import time
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

# Import controller system
from controller.controller_manager import ControllerManager
from controller.controller_ann import ANNController
from controller.metrics.head_metrics import collect_head_metrics

# Import plasticity system
from utils.adaptive.adaptive_plasticity import AdaptivePlasticitySystem, run_adaptive_system

# Import dataset handling
from sentinel_data.dataset_loader import load_dataset  # Updated import path

# Import utility functions
from utils.metrics_logger import MetricsLogger

class ControllerPlasticityIntegration:
    """
    Integration class that bridges the Controller and Adaptive Plasticity systems,
    allowing them to work together in a feedback loop to optimize model structure
    and performance.
    """
    
    def __init__(
        self,
        model,
        dataset: Any,
        output_dir: str = "./output/controller_plasticity",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_cycles: int = 10,
        controller_config: Dict = None,
        plasticity_config: Dict = None,
        verbose: bool = False
    ):
        """
        Initialize the integration system.
        
        Args:
            model: The model to optimize (either a model instance or model name)
            dataset: Dataset for training and evaluation
            output_dir: Directory to save results
            device: Device to use for computation
            max_cycles: Maximum number of plasticity cycles to run
            controller_config: Configuration for the controller
            plasticity_config: Configuration for the plasticity system
            verbose: Whether to print verbose output
        """
        # Handle both model name strings and model instances
        self.model_name = model if isinstance(model, str) else model.__class__.__name__
        self.model = None if isinstance(model, str) else model
        self.model_path = model if isinstance(model, str) else None
        
        self.dataset = dataset
        self.device = device
        self.max_cycles = max_cycles
        self.verbose = verbose
        
        # Set up output directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"ctrl_plast_{self.model_name.split('/')[-1]}_{self.run_timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_name)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.visualizations_dir = os.path.join(self.run_dir, "visualizations")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Initialize configuration
        self.controller_config = controller_config or {}
        self.plasticity_config = plasticity_config or {}
        
        # Set up metrics logging
        self.metrics_file = os.path.join(self.metrics_dir, "integration_metrics.jsonl")
        self.metrics_logger = MetricsLogger(self.metrics_file)
        
        # Log basic configuration
        self.config = {
            "model_name": self.model_name,
            "device": device,
            "max_cycles": max_cycles,
            "controller_config": self.controller_config,
            "plasticity_config": self.plasticity_config,
            "run_timestamp": self.run_timestamp,
            "run_dir": self.run_dir
        }
        
        # Save configuration
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # Initialize systems
        self.plasticity_system = None
        self.controller_manager = None
        
        # Initialize metrics and state tracking
        self.cycle_metrics = []
        self.best_perplexity = float('inf')
        self.best_params = None
        self.best_cycle = -1
        
        # Print initialization message
        if verbose:
            print(f"=== Controller-Plasticity Integration Initialized ===")
            print(f"Model: {self.model_name}")
            print(f"Output directory: {self.run_dir}")
            print(f"Device: {device}")
            print(f"Max cycles: {max_cycles}")
    
    def initialize_systems(self):
        """
        Initialize the plasticity and controller systems.
        """
        # Initialize the adaptive plasticity system first
        print(f"Initializing Adaptive Plasticity System...")
        
        # Handle different initialization paths based on whether we have a model or model name
        if self.model is None and self.model_path is not None:
            self.plasticity_system = AdaptivePlasticitySystem(
                model_name=self.model_path,
                dataset=self.dataset,
                output_dir=self.run_dir,
                device=self.device,
                max_degeneration_score=self.plasticity_config.get("max_degeneration_score", 3.0),
                max_perplexity_increase=self.plasticity_config.get("max_perplexity_increase", 0.15),
                learning_rate=self.plasticity_config.get("learning_rate", 5e-5),
                memory_capacity=self.plasticity_config.get("memory_capacity", 10),
                verbose=self.verbose
            )
        else:
            # If we already have a model instance, use it
            self.plasticity_system = AdaptivePlasticitySystem(
                model=self.model,
                dataset=self.dataset,
                output_dir=self.run_dir,
                device=self.device,
                max_degeneration_score=self.plasticity_config.get("max_degeneration_score", 3.0),
                max_perplexity_increase=self.plasticity_config.get("max_perplexity_increase", 0.15),
                learning_rate=self.plasticity_config.get("learning_rate", 5e-5),
                memory_capacity=self.plasticity_config.get("memory_capacity", 10),
                verbose=self.verbose
            )
        
        # Get the model from the plasticity system
        self.model = self.plasticity_system.model
        
        # Now initialize the controller manager with the model
        print(f"Initializing Controller Manager...")
        self.controller_manager = ControllerManager(
            model=self.model, 
            config=self.controller_config
        )
        
        # Log initialization completion
        print(f"Both systems initialized successfully.")
        return True
    
    def run_integrated_optimization(self):
        """
        Run the integrated optimization process using both the controller
        and plasticity systems in a feedback loop.
        
        Returns:
            Dictionary with optimization results
        """
        print(f"\n=== Starting Integrated Optimization ===")
        start_time = time.time()
        
        # Initialize systems if not already done
        if self.plasticity_system is None or self.controller_manager is None:
            self.initialize_systems()
        
        # Initial evaluation to establish baseline
        print(f"Running baseline evaluation...")
        baseline_eval = self.plasticity_system.evaluate_model(
            params=self.plasticity_system.current_params,
            num_samples=5,
            generate_length=50,
            show_generations=True
        )
        
        # Save baseline metrics
        baseline_perplexity = baseline_eval["average_perplexity"]
        self.best_perplexity = baseline_perplexity
        self.best_params = self.plasticity_system.current_params
        
        # Log baseline metrics
        active_heads = determine_active_heads(
            self.plasticity_system.pruning_module, 
            self.plasticity_system.current_params
        )
        
        baseline_head_count = len(active_heads)
        
        self.metrics_logger.log({
            "phase": "baseline",
            "perplexity": baseline_perplexity,
            "active_heads": baseline_head_count,
            "total_heads": self.plasticity_system.pruning_module.num_layers * 
                          self.plasticity_system.pruning_module.num_heads
        })
        
        print(f"Baseline - Perplexity: {baseline_perplexity:.2f}, " +
              f"Active heads: {baseline_head_count}")
        
        # Run integrated optimization cycles
        for cycle in range(self.max_cycles):
            print(f"\n--- Integrated Plasticity Cycle {cycle+1}/{self.max_cycles} ---")
            cycle_start_time = time.time()
            
            # 1. Controller Guidance Phase
            # The controller provides guidance on pruning and growth decisions
            print("1. Controller Guidance Phase")
            
            # Collect metrics for controller
            metrics_dict = collect_head_metrics(
                self.model,
                dataloader=self.plasticity_system.dataloader,
                device=self.device
            )
            
            # Get controller recommendations
            controller_update = self.controller_manager.step(metrics_dict=metrics_dict)
            active_gates = controller_update["active_gates"]
            pruned_percent = controller_update["pruned_percent"]
            
            # Map controller recommendations to plasticity parameters
            pruning_level = min(0.8, pruned_percent / 100 + 0.1)  # Add margin
            growth_ratio = 0.5  # Default starting value
            
            # Adaptive growth based on controller learning
            if cycle > 0 and len(self.cycle_metrics) > 0:
                previous_success = self.cycle_metrics[-1]["success"]
                if previous_success:
                    # If previous cycle was successful, be more conservative with growth
                    growth_ratio = 0.3
                else:
                    # If previous cycle failed, be more aggressive with growth
                    growth_ratio = 0.7
            
            # 2. Plasticity Cycle with Controller Guidance
            print(f"2. Plasticity Cycle (pruning_level={pruning_level:.2f}, growth_ratio={growth_ratio:.2f})")
            
            # Run the plasticity cycle with controller-guided parameters
            cycle_result = self.plasticity_system.run_plasticity_cycle(
                pruning_level=pruning_level,
                growth_ratio=growth_ratio,
                training_steps=self.plasticity_config.get("training_steps", 100),
                use_memory=True
            )
            
            # 3. Feedback Integration Phase
            print("3. Feedback Integration Phase")
            
            # Extract results from plasticity cycle
            cycle_success = cycle_result["success"]
            perplexity_improvement = cycle_result["perplexity_improvement"]
            final_perplexity = cycle_result["final"]["perplexity"]
            head_reduction = cycle_result["head_reduction"]
            
            # Create integrated metrics for controller learning
            integrated_metrics = {
                "perplexity_improvement": torch.tensor(perplexity_improvement, device=self.device),
                "head_reduction": torch.tensor(head_reduction, device=self.device),
                "cycle_success": torch.tensor(1.0 if cycle_success else 0.0, device=self.device),
                "controller_lr": torch.tensor(0.01 * (0.9 ** cycle), device=self.device)
            }
            
            # Update controller with plasticity results
            self.controller_manager.step(metrics_dict=integrated_metrics)
            
            # Track best results
            if final_perplexity < self.best_perplexity:
                self.best_perplexity = final_perplexity
                self.best_params = self.plasticity_system.current_params
                self.best_cycle = cycle + 1
                
                # Save best model checkpoint
                best_model_path = os.path.join(self.checkpoints_dir, "model_best.pt")
                torch.save(self.best_params, best_model_path)
                print(f"New best model saved (perplexity: {self.best_perplexity:.2f})")
            
            # Log cycle metrics
            active_heads = determine_active_heads(
                self.plasticity_system.pruning_module, 
                self.plasticity_system.current_params
            )
            
            cycle_metrics = {
                "cycle": cycle + 1,
                "success": cycle_success,
                "pruning_level": pruning_level,
                "growth_ratio": growth_ratio,
                "initial_perplexity": cycle_result["initial"]["perplexity"],
                "pruned_perplexity": cycle_result["pruned"]["perplexity"],
                "grown_perplexity": cycle_result.get("grown", {}).get("perplexity", None),
                "final_perplexity": final_perplexity,
                "perplexity_improvement": perplexity_improvement,
                "active_heads": len(active_heads),
                "head_reduction": head_reduction,
                "duration_seconds": time.time() - cycle_start_time
            }
            
            self.cycle_metrics.append(cycle_metrics)
            self.metrics_logger.log({
                "phase": "cycle_complete",
                **cycle_metrics
            })
            
            # Print cycle results
            print(f"\nCycle {cycle+1} results:")
            print(f"  Success: {'✓' if cycle_success else '✗'}")
            print(f"  Perplexity: {cycle_result['initial']['perplexity']:.2f} → {final_perplexity:.2f} " +
                  f"({perplexity_improvement*100:+.1f}%)")
            print(f"  Active heads: {cycle_result['initial']['head_count']} → {len(active_heads)} " +
                  f"({head_reduction*100:.1f}% reduction)")
            print(f"  Duration: {time.time() - cycle_start_time:.1f}s")
            
            # Save checkpoint for this cycle
            checkpoint_path = os.path.join(self.checkpoints_dir, f"model_cycle{cycle+1}.pt")
            torch.save(self.plasticity_system.current_params, checkpoint_path)
            
            # Early stopping if we've reached diminishing returns
            if cycle >= 2:
                last_three_improvements = [self.cycle_metrics[i]["perplexity_improvement"] 
                                         for i in range(max(0, cycle-2), cycle+1)]
                avg_improvement = sum(last_three_improvements) / len(last_three_improvements)
                
                if avg_improvement < 0.01 and self.cycle_metrics[cycle]["perplexity_improvement"] < 0.005:
                    print(f"\n⚠️ Early stopping: Diminishing returns detected (avg improvement: {avg_improvement:.3f})")
                    break
        
        # Final evaluation with best parameters
        print(f"\n=== Final Evaluation ===")
        print(f"Best model from cycle {self.best_cycle}")
        print(f"Best perplexity: {self.best_perplexity:.2f}")
        
        final_eval = self.plasticity_system.evaluate_model(
            params=self.best_params,
            num_samples=5,
            generate_length=100,
            show_generations=True
        )
        
        # Calculate optimization results
        total_duration = time.time() - start_time
        improvement = (baseline_perplexity - self.best_perplexity) / baseline_perplexity
        
        print(f"\n=== Optimization Complete ===")
        print(f"Total time: {total_duration:.1f}s")
        print(f"Perplexity improvement: {improvement*100:.1f}%")
        print(f"Results saved to: {self.run_dir}")
        
        # Save results summary
        results = {
            "baseline_perplexity": baseline_perplexity,
            "best_perplexity": self.best_perplexity,
            "best_cycle": self.best_cycle,
            "improvement": improvement,
            "total_duration": total_duration,
            "cycles_completed": len(self.cycle_metrics),
            "cycle_metrics": self.cycle_metrics
        }
        
        results_path = os.path.join(self.run_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def determine_active_heads(pruning_module, params):
    """
    Helper function to determine active attention heads based on current parameters.
    
    Args:
        pruning_module: The pruning module handling the model
        params: Current model parameters
        
    Returns:
        List of active heads as (layer_idx, head_idx) tuples
    """
    active_heads = []
    
    # Extract head mask parameters from the model params
    head_mask_pattern = "attention_mask"
    
    for param_name, param_value in params.items():
        if head_mask_pattern in param_name and isinstance(param_value, torch.Tensor):
            # Parse layer and head indices from parameter name
            # Example pattern: "transformer.h.0.attn.attention_mask"
            parts = param_name.split('.')
            
            # Find layer index
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'h' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            
            if layer_idx is not None:
                # Get mask values and identify active heads
                mask = param_value.detach().cpu().numpy()
                if mask.ndim == 1:
                    # Single dimension masks
                    for head_idx, value in enumerate(mask):
                        if value > 0.5:  # Consider heads active if mask value > 0.5
                            active_heads.append((layer_idx, head_idx))
                else:
                    # Multi-dimensional masks (might need adaptation based on specific model)
                    for head_idx in range(mask.shape[0]):
                        if mask[head_idx].mean() > 0.5:
                            active_heads.append((layer_idx, head_idx))
    
    # If no heads were found through mask parameters, use a fallback approach
    if not active_heads and hasattr(pruning_module, 'get_active_heads'):
        # Use the pruning module's built-in method if available
        active_heads = pruning_module.get_active_heads(params)
    
    return active_heads


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Controller-Plasticity Integration")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name (default: tiny_shakespeare)")
    parser.add_argument("--output_dir", type=str, default="./output/controller_plasticity",
                      help="Directory to save results (default: ./output/controller_plasticity)")
    
    # Optimization parameters
    parser.add_argument("--max_cycles", type=int, default=10,
                      help="Maximum number of plasticity cycles to run (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Base learning rate for plasticity training (default: 5e-5)")
    parser.add_argument("--controller_lr", type=float, default=0.01,
                      help="Learning rate for controller updates (default: 0.01)")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--sequence_length", type=int, default=128,
                      help="Sequence length for training (default: 128)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Controller-Plasticity Integration ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Max cycles: {args.max_cycles}")
    print(f"Device: {args.device}")
    
    # Load tokenizer first
    print(f"Loading tokenizer for {args.model_name}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load the dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_length=args.sequence_length
    )
    
    # Define controller configuration
    controller_config = {
        "controller_type": "ann",
        "controller_lr": args.controller_lr,
        "controller_lr_decay": 0.9,
        "update_frequency": 50,
        "warmup_steps": 100,
        "controller_config": {
            "init_value": 2.0,
            "reg_weight": 1e-4
        }
    }
    
    # Define plasticity configuration
    plasticity_config = {
        "learning_rate": args.learning_rate,
        "max_degeneration_score": 3.0,
        "max_perplexity_increase": 0.15,
        "memory_capacity": 5,
        "training_steps": 100
    }
    
    # Initialize and run the integrated optimization
    integration = ControllerPlasticityIntegration(
        model=args.model_name,
        dataset=dataset,
        output_dir=args.output_dir,
        device=args.device,
        max_cycles=args.max_cycles,
        controller_config=controller_config,
        plasticity_config=plasticity_config,
        verbose=args.verbose
    )
    
    # Run the integrated optimization
    results = integration.run_integrated_optimization()
    
    # Print summary
    print(f"\n=== Final Results ===")
    print(f"Baseline perplexity: {results['baseline_perplexity']:.2f}")
    print(f"Best perplexity: {results['best_perplexity']:.2f}")
    print(f"Improvement: {results['improvement']*100:.1f}%")
    print(f"Best cycle: {results['best_cycle']}")
    print(f"Total duration: {results['total_duration']/60:.1f} minutes")


if __name__ == "__main__":
    main()