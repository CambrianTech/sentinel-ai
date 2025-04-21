#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Neural Plasticity with ANN Controller

This script implements a full neural plasticity experiment that integrates
the ANN controller to intelligently refine a GPT-2 model through pruning 
and fine-tuning. It provides comprehensive visualizations of the complete 
training process, including warmup, pruning, and fine-tuning phases.

Version: v0.1.0 (2025-04-20 22:30:00)
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Configure paths and environment
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("adaptive_plasticity")

# Import required modules
try:
    from sentinel.controller.controller_ann import ANNController
    from sentinel.controller.controller_manager import ControllerManager
    from utils.neural_plasticity.dashboard.multi_phase_dashboard import MultiPhaseDashboard
    from utils.neural_plasticity.experiment import NeuralPlasticityExperiment
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all dependencies are installed correctly")
    sys.exit(1)

class AdaptiveNeuralPlasticityExperiment:
    """
    Integrated experiment that combines neural plasticity with ANN controller
    for intelligent model refinement with dynamic pruning and fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        dataset: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        output_dir: str = None,
        device: str = None,
        pruning_strategy: str = "entropy",
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        max_length: int = 128,
        entropy_threshold: float = 1.5,
        importance_threshold: float = 0.7,
        controller_config: dict = None,
        verbose: bool = True,
    ):
        """Initialize adaptive neural plasticity experiment with ANN controller."""
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_config = dataset_config
        
        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            self.output_dir = os.path.join(project_root, "experiment_output", "neural_plasticity", f"run_{timestamp}")
        else:
            self.output_dir = output_dir
        
        # Create necessary subdirectories
        self.dashboard_dir = os.path.join(self.output_dir, "dashboards")
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Set up configuration parameters
        self.pruning_strategy = pruning_strategy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.entropy_threshold = entropy_threshold
        self.importance_threshold = importance_threshold
        self.verbose = verbose
        
        # Initialize controller config
        self.controller_config = controller_config or {
            "init_value": 3.0,
            "reg_weight": 1e-4
        }
        
        # Initialize experiment components
        self.model = None
        self.tokenizer = None
        self.controller = None
        self.controller_manager = None
        self.dashboard = None
        self.experiment = None
        
        # Metrics tracking
        self.metrics = {
            "warmup": {"loss": [], "perplexity": [], "steps": []},
            "pruning": {"loss": [], "perplexity": [], "steps": [], "sparsity": []},
            "finetuning": {"loss": [], "perplexity": [], "steps": [], "sparsity": []},
            "evaluation": {"perplexity": [], "steps": []},
            "global": {"loss": [], "perplexity": [], "steps": [], "sparsity": []}
        }
        self.pruning_events = []
        self.stabilization_points = []
        self.current_phase = "setup"
        self.current_step = 0
        self.current_cycle = 0
        
        logger.info(f"Experiment initialized with model {model_name} on {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup(self):
        """Set up all components including model, controller, and dashboard."""
        logger.info("Setting up experiment components...")
        
        # Initialize model and tokenizer
        try:
            logger.info(f"Loading model {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully: {self.model.__class__.__name__}")
            
            # Get model architecture details
            if hasattr(self.model.config, "n_layer"):
                self.num_layers = self.model.config.n_layer
                self.num_heads = self.model.config.n_head
            elif hasattr(self.model.config, "num_hidden_layers"):
                self.num_layers = self.model.config.num_hidden_layers
                self.num_heads = self.model.config.num_attention_heads
            else:
                # Fallback for other model types
                self.num_layers = 12
                self.num_heads = 12
                logger.warning(f"Could not determine model architecture, using defaults: {self.num_layers} layers, {self.num_heads} heads")
            
            # Initialize ANN controller
            logger.info(f"Initializing ANN controller for {self.num_layers} layers with {self.num_heads} heads...")
            self.controller = ANNController(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                config=self.controller_config
            )
            
            # Initialize controller manager (integrates controller with model)
            self.controller_manager = ControllerManager(
                model=self.model,
                controller=self.controller,
                device=self.device
            )
            
            # Initialize dashboard for visualization
            logger.info("Initializing multi-phase dashboard...")
            self.dashboard = MultiPhaseDashboard(
                project_name="adaptive-plasticity",
                experiment_name=f"adaptive-{self.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                output_dir=self.dashboard_dir,
                config={
                    "model_name": self.model_name,
                    "dataset": f"{self.dataset}/{self.dataset_config}",
                    "pruning_strategy": self.pruning_strategy,
                    "learning_rate": self.learning_rate,
                    "entropy_threshold": self.entropy_threshold,
                    "controller_config": self.controller_config
                },
                mode="offline"  # Use offline mode for this experiment
            )
            
            # Initialize NeuralPlasticityExperiment for the actual training logic
            logger.info("Initializing neural plasticity experiment...")
            def metrics_callback(metrics_dict):
                """Callback to track metrics during experiment."""
                # Add current phase
                metrics_dict["phase"] = self.current_phase
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.record_step(metrics_dict, self.current_step)
                
                # Save metrics for our own tracking
                phase = self.current_phase
                if phase in self.metrics:
                    if "loss" in metrics_dict:
                        self.metrics[phase]["loss"].append(metrics_dict["loss"])
                    if "perplexity" in metrics_dict:
                        self.metrics[phase]["perplexity"].append(metrics_dict["perplexity"])
                    if "sparsity" in metrics_dict:
                        self.metrics[phase]["sparsity"].append(metrics_dict["sparsity"])
                    self.metrics[phase]["steps"].append(self.current_step)
                
                # Track global metrics
                if "loss" in metrics_dict:
                    self.metrics["global"]["loss"].append(metrics_dict["loss"])
                if "perplexity" in metrics_dict:
                    self.metrics["global"]["perplexity"].append(metrics_dict["perplexity"])
                if "sparsity" in metrics_dict:
                    self.metrics["global"]["sparsity"].append(metrics_dict["sparsity"])
                self.metrics["global"]["steps"].append(self.current_step)
                
                # Increment step counter
                self.current_step += 1
                
                return metrics_dict
            
            # Initialize the experiment
            self.experiment = NeuralPlasticityExperiment(
                model_name=self.model_name,
                device=self.device,
                dataset=self.dataset,
                dataset_config=self.dataset_config,
                batch_size=self.batch_size,
                max_length=self.max_length,
                pruning_strategy=self.pruning_strategy,
                learning_rate=self.learning_rate,
                output_dir=self.output_dir,
                metrics_callback=metrics_callback,
                verbose=self.verbose
            )
            
            logger.info("Experiment setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}", exc_info=True)
            return False
    
    def run_warmup_phase(self, warmup_steps=200):
        """Run warmup phase to establish baseline model behavior."""
        logger.info(f"Starting warmup phase with {warmup_steps} steps...")
        
        # Set phase
        self.current_phase = "warmup"
        if self.dashboard:
            self.dashboard.set_phase("warmup")
            self.dashboard.record_phase_transition("warmup", self.current_step)
        
        # Save the original model state for comparison
        baseline_checkpoint_path = os.path.join(self.output_dir, "baseline_checkpoint.pt")
        torch.save(self.model.state_dict(), baseline_checkpoint_path)
        logger.info(f"Saved baseline model checkpoint to {baseline_checkpoint_path}")
        
        # Run warmup training
        try:
            logger.info("Running warmup training...")
            warmup_results = self.experiment.run_warmup(steps=warmup_steps)
            
            # Check for stabilization
            if "stabilization_step" in warmup_results:
                stabilization_step = warmup_results["stabilization_step"]
                self.stabilization_points.append((stabilization_step, "warmup"))
                logger.info(f"Warmup stabilized at step {stabilization_step}")
                
                # Record stabilization in dashboard
                if self.dashboard:
                    self.dashboard.stabilization_points.append((stabilization_step, "warmup"))
            
            # Return warmup results
            return warmup_results
        
        except Exception as e:
            logger.error(f"Error during warmup phase: {e}", exc_info=True)
            return None
    
    def run_analyze_phase(self, analyze_steps=10):
        """Analyze model to collect metrics for pruning decisions."""
        logger.info("Starting analysis phase...")
        
        # Set phase
        self.current_phase = "analysis"
        if self.dashboard:
            self.dashboard.set_phase("analysis")
            self.dashboard.record_phase_transition("analysis", self.current_step)
        
        try:
            # Collect attention entropy metrics
            logger.info("Collecting attention entropy metrics...")
            entropy_metrics = self.experiment.collect_attention_metrics(steps=analyze_steps)
            
            # Collect gradient-based metrics
            logger.info("Collecting gradient-based importance metrics...")
            gradient_metrics = self.experiment.collect_gradient_metrics(steps=analyze_steps)
            
            # Combine metrics for controller
            metrics_dict = {
                "entropy": entropy_metrics["entropy"],
                "grad_norm": gradient_metrics["gradient_norm"],
                "head_importance": gradient_metrics["head_importance"],
                "entropy_threshold": self.entropy_threshold,
                "importance_threshold": self.importance_threshold,
                "controller_lr": 0.01
            }
            
            # Update controller based on metrics
            logger.info("Updating controller gates based on collected metrics...")
            self.controller.update_gates(metrics_dict)
            
            # Get current gate values
            gate_values = self.controller.forward().detach().cpu().numpy()
            
            # Return analysis results
            analysis_results = {
                "entropy": entropy_metrics["entropy"],
                "gradient_norm": gradient_metrics["gradient_norm"],
                "head_importance": gradient_metrics["head_importance"],
                "gate_values": gate_values
            }
            
            logger.info("Analysis phase completed")
            return analysis_results
        
        except Exception as e:
            logger.error(f"Error during analysis phase: {e}", exc_info=True)
            return None
    
    def run_pruning_phase(self, pruning_cycle=1, pruning_level=0.2):
        """Run pruning phase using controller and metrics."""
        logger.info(f"Starting pruning phase (cycle {pruning_cycle}) with target level {pruning_level}...")
        
        # Set phase and cycle
        self.current_phase = "pruning"
        self.current_cycle = pruning_cycle
        if self.dashboard:
            self.dashboard.set_phase("pruning")
            self.dashboard.record_phase_transition("pruning", self.current_step)
        
        try:
            # Get current gate values from controller
            gate_values = self.controller.forward().detach().cpu().numpy()
            
            # Identify heads to prune based on gate values and pruning level
            heads_to_prune = []
            for l in range(self.num_layers):
                for h in range(self.num_heads):
                    if gate_values[l, h] < pruning_level:
                        heads_to_prune.append((l, h))
            
            # Ensure we don't prune too many heads
            max_heads_to_prune = int(self.num_layers * self.num_heads * pruning_level)
            if len(heads_to_prune) > max_heads_to_prune:
                # Sort by gate value (ascending) and take only max_heads_to_prune
                head_scores = [(l, h, gate_values[l, h]) for l, h in heads_to_prune]
                head_scores.sort(key=lambda x: x[2])
                heads_to_prune = [(l, h) for l, h, _ in head_scores[:max_heads_to_prune]]
            
            # Apply pruning
            logger.info(f"Pruning {len(heads_to_prune)} attention heads: {heads_to_prune}")
            pruning_info = {
                "strategy": "controller",
                "pruning_level": pruning_level,
                "pruned_heads": heads_to_prune,
                "cycle": pruning_cycle
            }
            
            # Track pruning event
            pruning_step = self.current_step
            self.pruning_events.append((pruning_step, pruning_info))
            
            # Record pruning event in dashboard
            if self.dashboard:
                self.dashboard.record_pruning_event(pruning_info, pruning_step)
            
            # Apply pruning to model via experiment
            logger.info("Applying pruning to model...")
            pruning_results = self.experiment.prune_heads(heads_to_prune)
            
            # Calculate new sparsity
            total_heads = self.num_layers * self.num_heads
            pruned_heads = len(heads_to_prune)
            sparsity = pruned_heads / total_heads
            
            # Update metrics with sparsity
            if self.dashboard:
                self.dashboard.record_step({
                    "phase": "pruning",
                    "sparsity": sparsity,
                    "pruned_heads": pruned_heads,
                    "total_heads": total_heads
                }, self.current_step)
            
            logger.info(f"Pruning phase completed with sparsity {sparsity:.2f}")
            return pruning_results
            
        except Exception as e:
            logger.error(f"Error during pruning phase: {e}", exc_info=True)
            return None
    
    def run_finetuning_phase(self, finetuning_steps=200):
        """Run fine-tuning phase to recover performance after pruning."""
        logger.info(f"Starting fine-tuning phase with {finetuning_steps} steps...")
        
        # Set phase
        self.current_phase = "finetuning"
        if self.dashboard:
            self.dashboard.set_phase("finetuning")
            self.dashboard.record_phase_transition("finetuning", self.current_step)
        
        try:
            # Run fine-tuning
            logger.info("Running fine-tuning to recover performance...")
            finetuning_results = self.experiment.run_finetuning(steps=finetuning_steps)
            
            logger.info("Fine-tuning phase completed")
            return finetuning_results
        
        except Exception as e:
            logger.error(f"Error during fine-tuning phase: {e}", exc_info=True)
            return None
    
    def run_evaluation_phase(self):
        """Evaluate the model after all phases are complete."""
        logger.info("Starting evaluation phase...")
        
        # Set phase
        self.current_phase = "evaluation"
        if self.dashboard:
            self.dashboard.set_phase("evaluation")
            self.dashboard.record_phase_transition("evaluation", self.current_step)
        
        try:
            # Generate sample texts with both baseline and pruned models
            prompts = [
                "The future of artificial intelligence seems to be",
                "Neural networks have revolutionized how we approach",
                "The key challenge in machine learning today is"
            ]
            
            # Generate samples from baseline and pruned models
            generation_samples = {}
            
            # Get baseline model samples
            logger.info("Generating text samples for comparison...")
            for i, prompt in enumerate(prompts):
                baseline_text = self.experiment.generate_baseline_text(prompt, max_length=100)
                pruned_text = self.experiment.generate_text(prompt, max_length=100)
                
                # Store samples
                sample_id = f"sample_{i+1}"
                generation_samples[sample_id] = {
                    "prompt": prompt,
                    "baseline": baseline_text,
                    "pruned": pruned_text
                }
                
                # Log samples to dashboard
                if self.dashboard:
                    self.dashboard.log_text_sample(prompt, baseline_text, model_type="baseline")
                    self.dashboard.log_text_sample(prompt, pruned_text, model_type="pruned")
                    
                    # Create comparison visualization
                    self.dashboard.log_inference_comparison(
                        prompt, baseline_text, pruned_text,
                        metrics={
                            "baseline_perplexity": self.metrics["warmup"]["perplexity"][-1] if self.metrics["warmup"]["perplexity"] else None,
                            "pruned_perplexity": self.metrics["finetuning"]["perplexity"][-1] if self.metrics["finetuning"]["perplexity"] else None
                        }
                    )
            
            # Get attention visualization data if available
            attention_data = self.experiment.get_attention_maps(prompts[0])
            
            # Calculate final metrics
            total_heads = self.num_layers * self.num_heads
            pruned_heads = 0
            for _, info in self.pruning_events:
                pruned_heads += len(info["pruned_heads"])
            
            sparsity = pruned_heads / total_heads * 100  # As percentage
            
            # Calculate perplexity improvement
            baseline_perplexity = self.metrics["warmup"]["perplexity"][-1] if self.metrics["warmup"]["perplexity"] else 0
            final_perplexity = self.metrics["finetuning"]["perplexity"][-1] if self.metrics["finetuning"]["perplexity"] else 0
            
            if baseline_perplexity > 0:
                improvement_percent = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
            else:
                improvement_percent = 0
            
            # Record evaluation results
            eval_results = {
                "baseline_perplexity": baseline_perplexity,
                "final_perplexity": final_perplexity,
                "improvement_percent": improvement_percent,
                "sparsity_percent": sparsity,
                "pruned_heads": pruned_heads,
                "total_heads": total_heads,
                "generation_samples": generation_samples
            }
            
            logger.info(f"Evaluation completed: sparsity={sparsity:.2f}%, improvement={improvement_percent:.2f}%")
            return eval_results
        
        except Exception as e:
            logger.error(f"Error during evaluation phase: {e}", exc_info=True)
            return None
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations of the experiment."""
        logger.info("Generating visualizations...")
        
        try:
            # Create complete process visualization with multi-phase dashboard
            if self.dashboard:
                logger.info("Generating complete process visualization...")
                complete_process_path = os.path.join(self.dashboard_dir, "complete_process.png")
                self.dashboard.visualize_complete_process(save_path=complete_process_path)
                
                # Generate multi-cycle visualization if we have multiple cycles
                if self.current_cycle > 1:
                    logger.info("Generating multi-cycle visualization...")
                    multi_cycle_path = os.path.join(self.dashboard_dir, "multi_cycle_process.png")
                    self.dashboard.generate_multi_cycle_dashboard(save_path=multi_cycle_path)
                
                # Generate standalone HTML dashboard
                logger.info("Generating standalone HTML dashboard...")
                html_path = self.dashboard.generate_standalone_dashboard(self.dashboard_dir)
                
                # Try to open the dashboard in a browser
                try:
                    import webbrowser
                    logger.info(f"Opening dashboard in browser: file://{os.path.abspath(html_path)}")
                    webbrowser.open(f"file://{os.path.abspath(html_path)}")
                except Exception as e:
                    logger.warning(f"Could not open browser: {e}")
            
            # Generate custom attention head heatmaps
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Attention Head Activity
            logger.info("Generating attention head activity heatmap...")
            
            # Get current gate values from controller
            gate_values = self.controller.forward().detach().cpu().numpy()
            
            # Plot gate values as activity heatmap
            im0 = axes[0].imshow(gate_values, cmap='RdYlGn', vmin=0, vmax=1)
            axes[0].set_title("Attention Head Activity (Controller Gates)")
            axes[0].set_xlabel("Head Index")
            axes[0].set_ylabel("Layer Index")
            plt.colorbar(im0, ax=axes[0], label="Active (1) / Pruned (0)")
            
            # Construct the entropy heatmap if metrics are available
            if hasattr(self.experiment, "entropy_metrics") and self.experiment.entropy_metrics is not None:
                entropy = self.experiment.entropy_metrics["entropy"]
                # Make sure it's a numpy array
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.detach().cpu().numpy()
                
                # Plot entropy heatmap
                im1 = axes[1].imshow(entropy, cmap='viridis')
                axes[1].set_title("Head Importance (Entropy × Gradient)")
                axes[1].set_xlabel("Head Index")
                axes[1].set_ylabel("Layer Index")
                plt.colorbar(im1, ax=axes[1], label="Importance Score")
            else:
                axes[1].text(0.5, 0.5, "Entropy data not available", 
                           horizontalalignment='center', verticalalignment='center')
                axes[1].set_title("Head Importance")
            
            # Save the heatmap figure
            heatmap_path = os.path.join(self.dashboard_dir, "attention_heatmaps.png")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=120, bbox_inches="tight")
            logger.info(f"Attention heatmaps saved to {heatmap_path}")
            
            # Close the figure to free memory
            plt.close(fig)
            
            logger.info("Visualizations generated successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            return False
    
    def save_experiment_results(self):
        """Save experiment results including model checkpoints and metrics."""
        logger.info("Saving experiment results...")
        
        try:
            # Save pruned model
            model_path = os.path.join(self.output_dir, "pruned_model")
            self.experiment.save_model(path=model_path)
            logger.info(f"Pruned model saved to {model_path}")
            
            # Save controller state
            controller_path = os.path.join(self.output_dir, "controller.pt")
            torch.save(self.controller.state_dict(), controller_path)
            logger.info(f"Controller state saved to {controller_path}")
            
            # Save metrics as JSON
            import json
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            
            # Convert any numpy arrays or tensors to lists
            serializable_metrics = {}
            for phase, metrics in self.metrics.items():
                serializable_metrics[phase] = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        serializable_metrics[phase][key] = value.tolist()
                    else:
                        serializable_metrics[phase][key] = value
            
            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
            
            # Save pruning events
            pruning_path = os.path.join(self.output_dir, "pruning_events.json")
            
            # Convert pruning events to serializable format
            serializable_events = []
            for step, info in self.pruning_events:
                event = {
                    "step": int(step),
                    "cycle": int(info["cycle"]),
                    "strategy": info["strategy"],
                    "pruning_level": float(info["pruning_level"]),
                    "pruned_heads": [list(head) for head in info["pruned_heads"]]
                }
                serializable_events.append(event)
            
            # Save pruning events
            with open(pruning_path, 'w') as f:
                json.dump(serializable_events, f, indent=2)
            logger.info(f"Pruning events saved to {pruning_path}")
            
            logger.info("Experiment results saved successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error saving experiment results: {e}", exc_info=True)
            return False
    
    def run_full_experiment(self, warmup_steps=200, analyze_steps=10, finetuning_steps=200, pruning_cycles=3, pruning_level=0.2):
        """Run the complete adaptive neural plasticity experiment."""
        logger.info(f"Starting full adaptive neural plasticity experiment with {pruning_cycles} cycles...")
        logger.info(f"Stages: warmup({warmup_steps}) → analyze({analyze_steps}) → prune → finetune({finetuning_steps}) → repeat{pruning_cycles}×")
        
        results = {}
        
        try:
            # 1. Setup the experiment
            if not self.setup():
                return {"status": "failed", "message": "Experiment setup failed"}
            
            # 2. Run warmup phase
            warmup_results = self.run_warmup_phase(warmup_steps=warmup_steps)
            if warmup_results is None:
                return {"status": "failed", "message": "Warmup phase failed"}
            results["warmup"] = warmup_results
            
            # 3. Run multi-cycle pruning and fine-tuning
            for cycle in range(1, pruning_cycles + 1):
                logger.info(f"Starting pruning cycle {cycle}/{pruning_cycles}")
                
                # 3a. Run analysis phase
                analysis_results = self.run_analyze_phase(analyze_steps=analyze_steps)
                if analysis_results is None:
                    return {"status": "failed", "message": f"Analysis phase failed in cycle {cycle}"}
                results[f"analysis_{cycle}"] = analysis_results
                
                # 3b. Run pruning phase
                pruning_results = self.run_pruning_phase(pruning_cycle=cycle, pruning_level=pruning_level)
                if pruning_results is None:
                    return {"status": "failed", "message": f"Pruning phase failed in cycle {cycle}"}
                results[f"pruning_{cycle}"] = pruning_results
                
                # 3c. Run fine-tuning phase
                finetuning_results = self.run_finetuning_phase(finetuning_steps=finetuning_steps)
                if finetuning_results is None:
                    return {"status": "failed", "message": f"Fine-tuning phase failed in cycle {cycle}"}
                results[f"finetuning_{cycle}"] = finetuning_results
            
            # 4. Run evaluation phase
            eval_results = self.run_evaluation_phase()
            if eval_results is None:
                return {"status": "failed", "message": "Evaluation phase failed"}
            results["evaluation"] = eval_results
            
            # 5. Generate visualizations
            self.generate_visualizations()
            
            # 6. Save experiment results
            self.save_experiment_results()
            
            # Calculate summary metrics for output
            pruned_heads = 0
            for _, info in self.pruning_events:
                pruned_heads += len(info["pruned_heads"])
            
            sparsity = pruned_heads / (self.num_layers * self.num_heads) * 100
            
            if "warmup" in self.metrics and self.metrics["warmup"]["perplexity"]:
                baseline_perplexity = self.metrics["warmup"]["perplexity"][-1]
            else:
                baseline_perplexity = 0
                
            if "finetuning" in self.metrics and self.metrics["finetuning"]["perplexity"]:
                final_perplexity = self.metrics["finetuning"]["perplexity"][-1] 
            else:
                final_perplexity = 0
            
            if baseline_perplexity > 0:
                improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
            else:
                improvement = 0
            
            # Add summary to results
            results["summary"] = {
                "status": "success",
                "pruning_cycles": pruning_cycles,
                "pruned_heads": pruned_heads,
                "sparsity_percent": sparsity,
                "baseline_perplexity": baseline_perplexity,
                "final_perplexity": final_perplexity,
                "improvement_percent": improvement,
                "output_dir": self.output_dir,
                "dashboard_dir": self.dashboard_dir
            }
            
            logger.info(f"Experiment completed successfully with {pruned_heads} pruned heads ({sparsity:.2f}% sparsity)")
            logger.info(f"Perplexity improvement: {improvement:.2f}% (from {baseline_perplexity:.2f} to {final_perplexity:.2f})")
            
            return results
        
        except KeyboardInterrupt:
            logger.warning("Experiment interrupted by user")
            return {"status": "interrupted", "partial_results": results}
            
        except Exception as e:
            logger.error(f"Error during experiment: {e}", exc_info=True)
            return {"status": "failed", "message": str(e), "partial_results": results}
        
        finally:
            # Clean up resources
            if self.dashboard:
                try:
                    self.dashboard.finish()
                except:
                    pass
            
            logger.info(f"Experiment session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run adaptive neural plasticity experiment with ANN controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_name", type=str, default="gpt2", 
                          help="Model name or path (e.g., gpt2, distilgpt2, facebook/opt-125m)")
    model_group.add_argument("--device", type=str, default=None, 
                          help="Device to run on (cpu, cuda, auto). Auto-detected if not specified.")
    
    # Dataset configuration group
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument("--dataset", type=str, default="wikitext",
                         help="Dataset name (e.g., wikitext, cnn_dailymail)")
    data_group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                         help="Dataset configuration")
    data_group.add_argument("--batch_size", type=int, default=8,
                         help="Batch size")
    data_group.add_argument("--max_length", type=int, default=128,
                         help="Maximum sequence length")
    
    # Experiment configuration group
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument("--warmup_steps", type=int, default=200,
                        help="Number of warmup steps")
    exp_group.add_argument("--finetuning_steps", type=int, default=200,
                        help="Number of fine-tuning steps per cycle")
    exp_group.add_argument("--analyze_steps", type=int, default=10,
                        help="Number of steps for analysis phase")
    exp_group.add_argument("--cycles", type=int, default=3,
                        help="Number of pruning cycles")
    exp_group.add_argument("--pruning_level", type=float, default=0.2,
                        help="Pruning level (0.0 to 1.0)")
    exp_group.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    
    # Controller configuration group
    controller_group = parser.add_argument_group("Controller Configuration")
    controller_group.add_argument("--entropy_threshold", type=float, default=1.5,
                               help="Entropy threshold for pruning")
    controller_group.add_argument("--importance_threshold", type=float, default=0.7,
                               help="Importance threshold for pruning")
    controller_group.add_argument("--controller_lr", type=float, default=0.01,
                               help="Controller learning rate")
    
    # Output configuration group
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output_dir", type=str, default=None, 
                           help="Output directory for experiment results")
    output_group.add_argument("--verbose", action="store_true", default=True,
                           help="Enable verbose output")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Setup controller config
    controller_config = {
        "init_value": 3.0,
        "reg_weight": 1e-4,
        "controller_lr": args.controller_lr
    }
    
    # Initialize experiment
    experiment = AdaptiveNeuralPlasticityExperiment(
        model_name=args.model_name,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        device=args.device,
        pruning_strategy="entropy",  # Using entropy for controller
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        entropy_threshold=args.entropy_threshold,
        importance_threshold=args.importance_threshold,
        controller_config=controller_config,
        verbose=args.verbose
    )
    
    # Run the experiment
    results = experiment.run_full_experiment(
        warmup_steps=args.warmup_steps,
        analyze_steps=args.analyze_steps,
        finetuning_steps=args.finetuning_steps,
        pruning_cycles=args.cycles,
        pruning_level=args.pruning_level
    )
    
    # Check if experiment was successful
    if results["status"] == "success":
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to {results['summary']['output_dir']}")
        logger.info(f"Dashboard available at {results['summary']['dashboard_dir']}")
        
        # Print summary metrics
        logger.info("Summary Metrics:")
        logger.info(f"  Pruning cycles: {results['summary']['pruning_cycles']}")
        logger.info(f"  Pruned heads: {results['summary']['pruned_heads']}")
        logger.info(f"  Sparsity: {results['summary']['sparsity_percent']:.2f}%")
        logger.info(f"  Baseline perplexity: {results['summary']['baseline_perplexity']:.2f}")
        logger.info(f"  Final perplexity: {results['summary']['final_perplexity']:.2f}")
        logger.info(f"  Improvement: {results['summary']['improvement_percent']:.2f}%")
    else:
        logger.error(f"Experiment failed: {results.get('message', 'Unknown error')}")
        sys.exit(1)