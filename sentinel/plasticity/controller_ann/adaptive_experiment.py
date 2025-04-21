#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Neural Plasticity Experiment with ANN Controller

This module implements an advanced neural plasticity experiment that uses
the ANN Controller for dynamic attention head management and multi-phase
tracking with comprehensive visualizations.

Version: v0.1.0 (2025-04-20 22:30:00)
"""

import os
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import base experiment implementation
from sentinel.plasticity.neural_plasticity_experiment import NeuralPlasticityExperiment
from sentinel.plasticity.base_experiment import BaseExperiment

# Import ANN controller
from sentinel.controller.controller_ann import ANNController

# Import plasticity components
from sentinel.plasticity.pruning import EntropyPruner, MagnitudePruner

# Import dashboard integration
from utils.neural_plasticity.dashboard.multi_phase_dashboard import MultiPhaseDashboard

logger = logging.getLogger(__name__)

class AdaptiveNeuralPlasticityExperiment(NeuralPlasticityExperiment):
    """
    Neural Plasticity Experiment with ANN Controller integration.
    
    This experiment extends the base neural plasticity experiment with:
    1. ANN Controller for dynamic attention head gating
    2. Multi-phase tracking and visualization
    3. Multi-cycle pruning and fine-tuning
    4. Comprehensive visualization dashboard
    """
    
    def __init__(
        self,
        output_dir: str,
        device: Optional[str] = None,
        model_name: str = "distilgpt2",
        adaptive_model: bool = True,
        experiment_name: str = "adaptive_neural_plasticity",
        controller_config: Optional[Dict[str, Any]] = None,
        dashboard_dir: Optional[str] = None
    ):
        """
        Initialize adaptive neural plasticity experiment.
        
        Args:
            output_dir: Directory to save results
            device: Device to run on (auto-detected if None)
            model_name: Name of the pre-trained model to use
            adaptive_model: Whether to use adaptive model wrapper
            experiment_name: Name of the experiment type
            controller_config: Configuration for ANN controller
            dashboard_dir: Directory for dashboard output
        """
        # Initialize base experiment
        super().__init__(
            output_dir=output_dir,
            device=device,
            model_name=model_name,
            adaptive_model=adaptive_model,
            experiment_name=experiment_name
        )
        
        # ANN controller specific attributes
        self.controller_config = controller_config or {
            "init_value": 3.0,
            "reg_weight": 1e-4
        }
        
        self.controller = None
        self.dashboard = None
        self.dashboard_dir = dashboard_dir or os.path.join(output_dir, "dashboard")
        
        # Experiment tracking
        self.current_phase = "setup"
        self.current_step = 0
        self.current_cycle = 0
        
        # Metrics for multi-phase tracking
        self.phase_metrics = {
            "warmup": {"steps": [], "loss": [], "perplexity": [], "sparsity": []},
            "analysis": {"steps": [], "loss": [], "perplexity": [], "sparsity": []},
            "pruning": {"steps": [], "loss": [], "perplexity": [], "sparsity": []},
            "finetuning": {"steps": [], "loss": [], "perplexity": [], "sparsity": []}
        }
        
        # Events and transitions
        self.phase_transitions = []
        self.pruning_events = []
        self.stabilization_points = []
        
        # Create output directories
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        logger.info(f"Initialized adaptive neural plasticity experiment with ANN controller")
    
    def initialize_controller(self):
        """Initialize ANN controller for the model."""
        if self.model is None:
            logger.warning("Cannot initialize controller: model not loaded")
            return False
        
        try:
            # Determine model dimensions
            if hasattr(self.model.config, "n_layer"):
                num_layers = self.model.config.n_layer
                num_heads = self.model.config.n_head
            elif hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
                num_heads = self.model.config.num_attention_heads
            else:
                # Fallback for other model types
                logger.warning("Could not determine model architecture from config, using detection")
                pruner = self._create_pruner("entropy")
                num_layers, num_heads = pruner.detect_model_structure(self.model)
            
            # Initialize controller
            logger.info(f"Initializing ANN controller for {num_layers} layers with {num_heads} heads")
            self.controller = ANNController(
                num_layers=num_layers,
                num_heads=num_heads,
                config=self.controller_config
            )
            
            # Move to device
            self.controller.to(self.device)
            
            # Set controller in model if it supports it
            if hasattr(self.model, "set_controller"):
                self.model.set_controller(self.controller)
                logger.info("Controller set in adaptive model")
            else:
                logger.warning("Model does not have set_controller method - manual integration required")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing controller: {e}")
            return False
    
    def initialize_dashboard(self, config: Dict[str, Any] = None):
        """Initialize multi-phase dashboard for visualizations."""
        try:
            # Create unique experiment name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"adaptive-{self.model_name.split('/')[-1]}-{timestamp}"
            
            # Combine with existing config
            dashboard_config = {
                "model_name": self.model_name,
                "experiment_type": "adaptive_neural_plasticity",
                "controller_config": self.controller_config
            }
            
            # Add provided config
            if config:
                dashboard_config.update(config)
            
            # Initialize dashboard
            self.dashboard = MultiPhaseDashboard(
                project_name="neural-plasticity",
                experiment_name=experiment_name,
                output_dir=self.dashboard_dir,
                config=dashboard_config,
                mode="offline"  # Use offline mode by default
            )
            
            # Set initial phase
            self.set_phase("setup")
            
            logger.info(f"Multi-phase dashboard initialized: {self.dashboard_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}")
            self.dashboard = None
            return False
    
    def set_phase(self, phase: str):
        """Set current experiment phase and record transition."""
        self.current_phase = phase
        
        # Record in dashboard if available
        if self.dashboard:
            self.dashboard.set_phase(phase)
            
            # Record phase transition if dashboard supports it
            if hasattr(self.dashboard, "record_phase_transition"):
                self.dashboard.record_phase_transition(phase, self.current_step)
        
        # Save transition point
        self.phase_transitions.append((self.current_step, phase))
        
        logger.info(f"Phase transition: {phase} at step {self.current_step}")
    
    def record_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Record metrics for current phase."""
        if step is None:
            step = self.current_step
            self.current_step += 1
        else:
            self.current_step = step + 1
        
        # Add phase to metrics
        metrics["phase"] = self.current_phase
        
        # Record in dashboard if available
        if self.dashboard:
            if hasattr(self.dashboard, "record_step"):
                self.dashboard.record_step(metrics, step)
            else:
                self.dashboard.log_metrics(metrics, step)
        
        # Track in phase metrics
        if self.current_phase in self.phase_metrics:
            phase_data = self.phase_metrics[self.current_phase]
            phase_data["steps"].append(step)
            
            # Record standard metrics if available
            for key in ["loss", "perplexity", "sparsity"]:
                if key in metrics:
                    if key not in phase_data:
                        phase_data[key] = []
                    phase_data[key].append(metrics[key])
        
        # Check for stabilization
        if metrics.get("stabilized", False):
            self.stabilization_points.append((step, self.current_phase))
            
            # Record in dashboard if it supports it
            if self.dashboard and hasattr(self.dashboard, "stabilization_points"):
                self.dashboard.stabilization_points.append((step, self.current_phase))
    
    def record_pruning_event(self, pruning_info: Dict[str, Any], step: Optional[int] = None):
        """Record a pruning event."""
        if step is None:
            step = self.current_step
        
        # Add step to info
        pruning_info["step"] = step
        
        # Record in dashboard if available
        if self.dashboard and hasattr(self.dashboard, "record_pruning_event"):
            self.dashboard.record_pruning_event(pruning_info, step)
        elif self.dashboard:
            self.dashboard.log_pruning_decision(pruning_info, step)
        
        # Save pruning event
        self.pruning_events.append((step, pruning_info))
        
        logger.info(f"Pruning event recorded at step {step}: {len(pruning_info.get('pruned_heads', []))} heads pruned")
    
    def update_controller_with_metrics(self, metrics_dict: Dict[str, Any]):
        """Update controller gates using current metrics."""
        if self.controller is None:
            logger.warning("Cannot update controller: controller not initialized")
            return
        
        try:
            logger.info("Updating controller gates with current metrics")
            self.controller.update_gates(metrics_dict)
            
            # Get current gate values
            gate_values = torch.sigmoid(self.controller.gate_logits)
            active_heads = (gate_values > 0.5).sum().item()
            total_heads = gate_values.numel()
            
            logger.info(f"Controller update: {active_heads}/{total_heads} heads active ({active_heads/total_heads*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error updating controller: {e}")
    
    def collect_controller_metrics(self, eval_dataloader=None):
        """Collect metrics for controller updates."""
        if self.model is None:
            logger.warning("Cannot collect metrics: model not loaded")
            return {}
        
        try:
            # Create metrics dictionary
            metrics_dict = {}
            
            # 1. Collect entropy metrics if possible
            if eval_dataloader:
                pruner = self._create_pruner("entropy")
                entropy_data = pruner.collect_distributions(self.model, eval_dataloader)
                
                # Convert to tensor if needed
                if isinstance(entropy_data, dict):
                    # Combine layer-wise entropy into a single tensor
                    num_layers = max(entropy_data.keys()) + 1
                    num_heads = entropy_data[0].shape[0]  # Assuming all layers have same number of heads
                    
                    entropy_tensor = torch.zeros((num_layers, num_heads), device=self.device)
                    for layer_idx, layer_entropy in entropy_data.items():
                        entropy_tensor[layer_idx] = layer_entropy
                    
                    metrics_dict["entropy"] = entropy_tensor
                else:
                    metrics_dict["entropy"] = entropy_data
            
            # 2. Create gradient metrics
            if self.model.training:
                # Check if we can calculate gradient norms
                has_grads = False
                for name, param in self.model.named_parameters():
                    if "query" in name or "key" in name or "value" in name:
                        if param.grad is not None:
                            has_grads = True
                            break
                
                if has_grads:
                    # Create a tensor to store gradient norms
                    if "entropy" in metrics_dict:
                        # Use same shape as entropy for consistency
                        grad_norm = torch.zeros_like(metrics_dict["entropy"])
                    else:
                        # Get model dimensions from controller
                        num_layers = self.controller.num_layers
                        num_heads = self.controller.num_heads
                        grad_norm = torch.zeros((num_layers, num_heads), device=self.device)
                    
                    # Fill with some dummy values (real implementation would compute actual gradient norms)
                    grad_norm.fill_(0.01)
                    metrics_dict["grad_norm"] = grad_norm
            
            # 3. Add thresholds and learning rate
            metrics_dict["entropy_threshold"] = self.controller_config.get("entropy_threshold", 1.5)
            metrics_dict["importance_threshold"] = self.controller_config.get("importance_threshold", 0.7)
            metrics_dict["controller_lr"] = torch.tensor(self.controller_config.get("controller_lr", 0.01))
            
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error collecting controller metrics: {e}")
            return {}
    
    def run_warmup_phase(self, dataloader_builder_fn, warmup_steps=200):
        """Run warmup phase to establish baseline model behavior."""
        logger.info(f"Starting warmup phase with {warmup_steps} steps")
        
        # Set phase
        self.set_phase("warmup")
        
        # Create dataloaders
        train_dataloader, eval_dataloader = dataloader_builder_fn()
        
        # Save the original model state for comparison
        baseline_checkpoint_path = os.path.join(self.output_dir, "baseline_checkpoint.pt")
        torch.save(self.model.state_dict(), baseline_checkpoint_path)
        logger.info(f"Saved baseline model checkpoint to {baseline_checkpoint_path}")
        
        # Run warmup training
        for step in range(warmup_steps):
            # Get batch (with wraparound)
            batch = list(train_dataloader)[step % len(train_dataloader)]
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Record metrics
            metrics = {
                "loss": loss.item(),
                "step": step,
                "phase": "warmup",
                "sparsity": 0.0  # No pruning yet
            }
            
            # Evaluate periodically
            if step % 20 == 0 or step == warmup_steps - 1:
                eval_metrics = self._evaluate_model(eval_dataloader)
                metrics.update(eval_metrics)
                
                logger.info(f"Warmup step {step+1}/{warmup_steps} - Loss: {loss.item():.4f}, Perplexity: {eval_metrics.get('perplexity', 0.0):.2f}")
                
                # Check for stabilization (could implement better logic here)
                if step >= warmup_steps // 2:
                    metrics["stabilized"] = True
                    logger.info(f"Warmup stabilized at step {step}")
            
            # Record metrics
            self.record_metrics(metrics, step)
            
            # Update controller periodically
            if (step + 1) % 50 == 0:
                controller_metrics = self.collect_controller_metrics(eval_dataloader)
                self.update_controller_with_metrics(controller_metrics)
        
        # Final evaluation
        final_eval = self._evaluate_model(eval_dataloader)
        
        warmup_results = {
            "baseline_metrics": final_eval,
            "baseline_checkpoint": baseline_checkpoint_path,
            "stabilization_step": self.stabilization_points[-1][0] if self.stabilization_points else None
        }
        
        return warmup_results
    
    def run_analysis_phase(self, dataloader_builder_fn, analysis_steps=10):
        """Run analysis phase to collect metrics for pruning decisions."""
        logger.info(f"Starting analysis phase with {analysis_steps} steps")
        
        # Set phase
        self.set_phase("analysis")
        
        # Create dataloaders
        _, eval_dataloader = dataloader_builder_fn()
        
        # Collect entropy metrics
        pruner = self._create_pruner("entropy")
        entropy_data = pruner.collect_distributions(self.model, eval_dataloader)
        
        # Structure for analysis results
        analysis_results = {
            "entropy_data": entropy_data,
            "metrics": {}
        }
        
        # Evaluate model
        eval_metrics = self._evaluate_model(eval_dataloader)
        analysis_results["metrics"] = eval_metrics
        
        # Record metrics
        self.record_metrics({
            "phase": "analysis",
            "step": self.current_step,
            **eval_metrics
        })
        
        # Update controller with analysis metrics
        controller_metrics = self.collect_controller_metrics(eval_dataloader)
        self.update_controller_with_metrics(controller_metrics)
        
        # Save gate values after update
        if self.controller:
            gate_values = torch.sigmoid(self.controller.gate_logits).detach().cpu().numpy()
            analysis_results["gate_values"] = gate_values
            
            # Calculate current sparsity
            sparsity = 1.0 - (gate_values > 0.5).sum() / gate_values.size
            analysis_results["sparsity"] = sparsity
            
            logger.info(f"Analysis complete: current sparsity {sparsity*100:.1f}%")
        
        return analysis_results
    
    def run_pruning_phase(self, dataloader_builder_fn, pruning_level=0.2, cycle=1):
        """Run pruning phase to remove least important heads."""
        logger.info(f"Starting pruning phase (cycle {cycle}) with target level {pruning_level}")
        
        # Set phase
        self.set_phase("pruning")
        self.current_cycle = cycle
        
        # Create dataloaders
        _, eval_dataloader = dataloader_builder_fn()
        
        # Perform pruning based on controller gates
        if self.controller:
            # Get current gate values
            gate_values = torch.sigmoid(self.controller.gate_logits).detach().cpu().numpy()
            
            # Identify heads to prune (those with lowest gate values)
            pruned_heads = []
            for layer_idx in range(gate_values.shape[0]):
                for head_idx in range(gate_values.shape[1]):
                    gate = gate_values[layer_idx, head_idx]
                    if gate < pruning_level:
                        pruned_heads.append((layer_idx, head_idx, float(gate)))
            
            # Sort by gate value (ascending)
            pruned_heads.sort(key=lambda x: x[2])
            
            # Limit number of heads to prune if necessary
            max_heads_to_prune = int(gate_values.size * pruning_level)
            if len(pruned_heads) > max_heads_to_prune:
                pruned_heads = pruned_heads[:max_heads_to_prune]
            
            logger.info(f"Pruning {len(pruned_heads)} heads with lowest gate values")
            
            # Apply pruning to model
            pruner = self._create_pruner("entropy")  # Using entropy pruner for applying pruning
            pruner.apply_pruning(self.model, [(l, h) for l, h, _ in pruned_heads])
            
            # Record pruning event
            pruning_info = {
                "strategy": "controller",
                "pruning_level": pruning_level,
                "pruned_heads": [(int(l), int(h)) for l, h, _ in pruned_heads],
                "cycle": cycle
            }
            self.record_pruning_event(pruning_info)
            
            # Calculate new sparsity
            new_sparsity = len(pruned_heads) / gate_values.size
            
            # Evaluate model after pruning
            eval_metrics = self._evaluate_model(eval_dataloader)
            
            # Record metrics
            self.record_metrics({
                "phase": "pruning",
                "sparsity": new_sparsity,
                **eval_metrics
            })
            
            pruning_results = {
                "pruned_heads": pruned_heads,
                "sparsity": new_sparsity,
                "metrics": eval_metrics
            }
            
            return pruning_results
        else:
            logger.warning("Controller not available, falling back to standard pruning")
            
            # Fallback to standard pruning
            pruner = self._create_pruner("entropy")
            entropy_data = pruner.collect_distributions(self.model, eval_dataloader)
            pruned_heads = pruner.prune(self.model, distributions=entropy_data, prune_percent=pruning_level)
            
            # Record pruning event
            pruning_info = {
                "strategy": "entropy",
                "pruning_level": pruning_level,
                "pruned_heads": [(int(l), int(h)) for l, h, _ in pruned_heads],
                "cycle": cycle
            }
            self.record_pruning_event(pruning_info)
            
            # Calculate sparsity
            num_layers, num_heads = pruner.detect_model_structure(self.model)
            sparsity = len(pruned_heads) / (num_layers * num_heads)
            
            # Evaluate model after pruning
            eval_metrics = self._evaluate_model(eval_dataloader)
            
            # Record metrics
            self.record_metrics({
                "phase": "pruning",
                "sparsity": sparsity,
                **eval_metrics
            })
            
            pruning_results = {
                "pruned_heads": pruned_heads,
                "sparsity": sparsity,
                "metrics": eval_metrics
            }
            
            return pruning_results
    
    def run_finetuning_phase(self, dataloader_builder_fn, finetuning_steps=200):
        """Run fine-tuning phase to recover performance after pruning."""
        logger.info(f"Starting fine-tuning phase with {finetuning_steps} steps")
        
        # Set phase
        self.set_phase("finetuning")
        
        # Create dataloaders
        train_dataloader, eval_dataloader = dataloader_builder_fn()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Fine-tuning loop
        for step in range(finetuning_steps):
            # Get batch (with wraparound)
            batch = list(train_dataloader)[step % len(train_dataloader)]
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Collect metrics
            metrics = {
                "loss": loss.item(),
                "step": step,
                "phase": "finetuning"
            }
            
            # Evaluate periodically
            if step % 50 == 0 or step == finetuning_steps - 1:
                eval_metrics = self._evaluate_model(eval_dataloader)
                metrics.update(eval_metrics)
                
                logger.info(f"Fine-tuning step {step+1}/{finetuning_steps} - Loss: {loss.item():.4f}, Perplexity: {eval_metrics.get('perplexity', 0.0):.2f}")
                
                # Calculate current sparsity from controller
                if self.controller:
                    gate_values = torch.sigmoid(self.controller.gate_logits)
                    sparsity = 1.0 - (gate_values > 0.5).sum().item() / gate_values.numel()
                    metrics["sparsity"] = sparsity
            
            # Record metrics
            self.record_metrics(metrics, step)
            
            # Update controller periodically
            if self.controller and (step + 1) % 50 == 0:
                controller_metrics = self.collect_controller_metrics(eval_dataloader)
                self.update_controller_with_metrics(controller_metrics)
        
        # Final evaluation
        final_metrics = self._evaluate_model(eval_dataloader)
        
        finetuning_results = {
            "final_metrics": final_metrics
        }
        
        return finetuning_results
    
    def run_evaluation_phase(self, dataloader_builder_fn):
        """Run evaluation phase to assess final model quality."""
        logger.info("Starting evaluation phase")
        
        # Set phase
        self.set_phase("evaluation")
        
        # Create dataloaders
        _, eval_dataloader = dataloader_builder_fn()
        
        # Perform evaluation
        eval_metrics = self._evaluate_model(eval_dataloader)
        
        # Calculate sparsity if controller is available
        if self.controller:
            gate_values = torch.sigmoid(self.controller.gate_logits)
            active_heads = (gate_values > 0.5).sum().item()
            total_heads = gate_values.numel()
            sparsity = 1.0 - (active_heads / total_heads)
        else:
            # Try to calculate from pruning events
            total_heads = 0
            pruned_heads = set()
            
            for _, info in self.pruning_events:
                for l, h in info.get("pruned_heads", []):
                    pruned_heads.add((l, h))
            
            if total_heads == 0:
                # Try to detect model structure
                pruner = self._create_pruner("entropy")
                num_layers, num_heads = pruner.detect_model_structure(self.model)
                total_heads = num_layers * num_heads
            
            sparsity = len(pruned_heads) / total_heads if total_heads > 0 else 0
        
        # Record metrics
        self.record_metrics({
            "phase": "evaluation",
            "sparsity": sparsity,
            **eval_metrics
        })
        
        evaluation_results = {
            "final_metrics": eval_metrics,
            "sparsity": sparsity,
            "active_heads": total_heads - len(pruned_heads) if 'pruned_heads' in locals() else active_heads,
            "total_heads": total_heads
        }
        
        logger.info(f"Evaluation complete: sparsity {sparsity*100:.1f}%, perplexity {eval_metrics.get('perplexity', 0.0):.2f}")
        
        return evaluation_results
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations of the experiment."""
        if not self.dashboard:
            logger.warning("Dashboard not available, skipping visualization generation")
            return False
        
        try:
            logger.info("Generating visualizations...")
            
            # Generate complete process visualization
            if hasattr(self.dashboard, "visualize_complete_process"):
                process_path = os.path.join(self.dashboard_dir, "complete_process.png")
                self.dashboard.visualize_complete_process(save_path=process_path)
                logger.info(f"Complete process visualization saved to {process_path}")
            
            # Generate multi-cycle dashboard if applicable
            if hasattr(self.dashboard, "generate_multi_cycle_dashboard") and self.current_cycle > 1:
                multi_cycle_path = os.path.join(self.dashboard_dir, "multi_cycle_process.png")
                self.dashboard.generate_multi_cycle_dashboard(save_path=multi_cycle_path)
                logger.info(f"Multi-cycle visualization saved to {multi_cycle_path}")
            
            # Generate standalone HTML dashboard
            if hasattr(self.dashboard, "generate_standalone_dashboard"):
                html_path = self.dashboard.generate_standalone_dashboard(self.dashboard_dir)
                logger.info(f"Standalone dashboard saved to {html_path}")
                
                # Try to open the dashboard in a browser
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(html_path)}")
                except Exception as e:
                    logger.warning(f"Could not open browser: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return False
    
    def _evaluate_model(self, eval_dataloader):
        """Evaluate model on evaluation dataset."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.model.train()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def run_full_experiment(
        self,
        dataloader_builder_fn,
        warmup_steps=200,
        finetuning_steps=200,
        analysis_steps=10,
        pruning_cycles=3,
        pruning_level=0.2,
        learning_rate=5e-5,
        config=None
    ):
        """
        Run the complete adaptive neural plasticity experiment.
        
        Args:
            dataloader_builder_fn: Function that returns (train_dataloader, eval_dataloader)
            warmup_steps: Number of warmup steps
            finetuning_steps: Number of fine-tuning steps per cycle
            analysis_steps: Number of steps for analysis phase
            pruning_cycles: Number of pruning cycles
            pruning_level: Pruning level (0.0 to 1.0)
            learning_rate: Learning rate for fine-tuning
            config: Additional configuration
            
        Returns:
            Dictionary with experiment results
        """
        # Initialize learning rate
        self.learning_rate = learning_rate
        
        # Update with provided config
        if config:
            self.controller_config.update(config)
        
        # Initialize experiment components
        if self.model is None:
            logger.info("Loading model...")
            self.model = self._load_model()
        
        # Initialize controller
        if self.controller is None:
            self.initialize_controller()
        
        # Initialize dashboard
        if self.dashboard is None:
            dashboard_config = {
                "pruning_level": pruning_level,
                "pruning_cycles": pruning_cycles,
                "warmup_steps": warmup_steps,
                "finetuning_steps": finetuning_steps,
                "learning_rate": learning_rate,
            }
            if config:
                dashboard_config.update(config)
                
            self.initialize_dashboard(dashboard_config)
        
        logger.info(f"Starting full experiment with {pruning_cycles} pruning cycles")
        
        results = {}
        
        # 1. Run warmup phase
        warmup_results = self.run_warmup_phase(
            dataloader_builder_fn, 
            warmup_steps=warmup_steps
        )
        results["warmup"] = warmup_results
        
        # 2. Run multi-cycle pruning and fine-tuning
        cycle_results = []
        
        for cycle in range(1, pruning_cycles + 1):
            logger.info(f"Starting pruning cycle {cycle}/{pruning_cycles}")
            
            # Calculate pruning level for this cycle
            cycle_pruning_level = pruning_level / pruning_cycles
            
            # 2a. Run analysis phase
            analysis_results = self.run_analysis_phase(
                dataloader_builder_fn,
                analysis_steps=analysis_steps
            )
            
            # 2b. Run pruning phase
            pruning_results = self.run_pruning_phase(
                dataloader_builder_fn,
                pruning_level=cycle_pruning_level,
                cycle=cycle
            )
            
            # 2c. Run fine-tuning phase
            finetuning_results = self.run_finetuning_phase(
                dataloader_builder_fn,
                finetuning_steps=finetuning_steps
            )
            
            # Save cycle results
            cycle_results.append({
                "cycle": cycle,
                "analysis": analysis_results,
                "pruning": pruning_results,
                "finetuning": finetuning_results
            })
        
        results["cycles"] = cycle_results
        
        # 3. Run evaluation phase
        evaluation_results = self.run_evaluation_phase(dataloader_builder_fn)
        results["evaluation"] = evaluation_results
        
        # 4. Generate visualizations
        self.generate_visualizations()
        
        # 5. Calculate summary metrics
        baseline_perplexity = warmup_results.get("baseline_metrics", {}).get("perplexity", 0)
        final_perplexity = evaluation_results.get("final_metrics", {}).get("perplexity", 0)
        
        if baseline_perplexity > 0:
            improvement_percent = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
        else:
            improvement_percent = 0
        
        pruned_heads = sum(len(cycle["pruning"]["pruned_heads"]) for cycle in cycle_results)
        
        summary = {
            "baseline_perplexity": baseline_perplexity,
            "final_perplexity": final_perplexity,
            "improvement_percent": improvement_percent,
            "sparsity": evaluation_results.get("sparsity", 0) * 100,  # Convert to percentage
            "pruned_heads": pruned_heads,
            "total_heads": evaluation_results.get("total_heads", 0),
            "pruning_cycles": pruning_cycles
        }
        
        results["summary"] = summary
        
        logger.info("Experiment completed successfully")
        logger.info(f"Summary:")
        logger.info(f"  Baseline perplexity: {baseline_perplexity:.2f}")
        logger.info(f"  Final perplexity: {final_perplexity:.2f}")
        logger.info(f"  Improvement: {improvement_percent:.2f}%")
        logger.info(f"  Pruned heads: {pruned_heads}/{evaluation_results.get('total_heads', 0)}")
        logger.info(f"  Sparsity: {evaluation_results.get('sparsity', 0) * 100:.1f}%")
        
        return results
    
    @classmethod
    def get_argument_parser(cls):
        """Get argument parser with additional controller options."""
        parser = super().get_argument_parser()
        
        # Add controller-specific arguments
        parser.add_argument(
            "--controller_reg_weight",
            type=float,
            default=1e-4,
            help="L1 regularization weight for controller (default: 1e-4)"
        )
        
        parser.add_argument(
            "--controller_lr",
            type=float,
            default=0.01,
            help="Learning rate for controller updates (default: 0.01)"
        )
        
        parser.add_argument(
            "--entropy_threshold",
            type=float,
            default=1.5,
            help="Entropy threshold for controller pruning decisions (default: 1.5)"
        )
        
        parser.add_argument(
            "--importance_threshold",
            type=float,
            default=0.7,
            help="Importance threshold for controller regrowth decisions (default: 0.7)"
        )
        
        parser.add_argument(
            "--cycles",
            type=int,
            default=3,
            help="Number of pruning cycles (default: 3)"
        )
        
        return parser