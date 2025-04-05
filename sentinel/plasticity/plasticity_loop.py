"""
Plasticity Loop: Prune → Fine-tune → Measure → Regrow

This module enables the study of transformer plasticity by combining:
- Entropy/magnitude-based head pruning
- Adaptive fine-tuning
- Post-finetune attention entropy analysis
- Head regrowth simulation
"""

import torch
import os
import numpy as np
import json
import logging
from typing import Optional, List, Dict, Tuple, Callable, Union, Any
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

# Import from models modules
# Need to use the legacy loaders for now as they have the needed functions
from models.loaders.loader import load_baseline_model, load_adaptive_model
from sentinel.pruning.entropy_magnitude import (
    collect_attention_distributions,
    compute_attention_entropy,
    entropy_based_pruning,
    magnitude_based_pruning,
    update_mask
)

logger = logging.getLogger(__name__)


class PlasticityTracker:
    """
    Tracks changes in model parameters and behavior through the adaptation process.
    
    This class is responsible for:
    1. Recording gate activities before and after fine-tuning
    2. Tracking entropy changes for each attention head
    3. Monitoring performance metrics during adaptation
    4. Analyzing which pruned heads exhibit regrowth tendencies
    """
    
    def __init__(self, model, tracking_frequency: int = 10):
        """
        Initialize the plasticity tracker.
        
        Args:
            model: The model to track
            tracking_frequency: How often to record metrics during fine-tuning
        """
        self.model = model
        self.tracking_frequency = tracking_frequency
        self.gate_history = {}
        self.attention_history = {}
        self.entropy_history = {}
        self.performance_history = {}
        self.pruned_heads = []
        
    def initialize_tracking(self, pruned_heads: List[Tuple[int, int, float]]):
        """
        Record the initial state after pruning.
        
        Args:
            pruned_heads: List of (layer_idx, head_idx, score) tuples of pruned heads
        """
        self.pruned_heads = pruned_heads
        self.snapshot_gates(step=0)
        
    def snapshot_gates(self, step: int = None):
        """
        Record current state of all gate values.
        
        Args:
            step: Current training step (if None, uses the length of history)
        """
        if step is None:
            step = len(self.gate_history)
            
        self.gate_history[step] = {}
        
        if hasattr(self.model, 'blocks'):
            # Direct blocks access (Sentinel-AI adaptive model)
            for layer_idx, layer in enumerate(self.model.blocks):
                if hasattr(layer.attn, 'gate'):
                    # Deep copy to avoid reference issues
                    self.gate_history[step][layer_idx] = layer.attn.gate.detach().clone()
        else:
            # Try to access through transformer attribute
            transformer = None
            if hasattr(self.model, 'transformer'):
                transformer = self.model.transformer
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'transformer'):
                transformer = self.model.model.transformer
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'transformer'):
                transformer = self.model.base_model.transformer
                
            if transformer and hasattr(transformer, 'h'):
                for layer_idx, layer in enumerate(transformer.h):
                    if hasattr(layer.attn, 'gate'):
                        # Deep copy to avoid reference issues
                        self.gate_history[step][layer_idx] = layer.attn.gate.detach().clone()
    
    def record_entropy(self, step: int, distributions: Dict[int, torch.Tensor]):
        """
        Record entropy values for all heads.
        
        Args:
            step: Current training step
            distributions: Dictionary mapping layer indices to attention distributions
        """
        self.entropy_history[step] = {}
        
        for layer_idx, dist in distributions.items():
            entropy = compute_attention_entropy(dist)
            self.entropy_history[step][layer_idx] = entropy.detach().cpu()
    
    def record_attention(self, step: int, distributions: Dict[int, torch.Tensor]):
        """
        Record attention distributions.
        
        Args:
            step: Current training step
            distributions: Dictionary mapping layer indices to attention distributions
        """
        self.attention_history[step] = {}
        
        for layer_idx, dist in distributions.items():
            # Store a down-sampled version to save memory
            # Take first batch item and average across sequence positions
            if dist.dim() >= 4:  # batch, heads, seq, seq
                avg_dist = dist[0].mean(dim=1)  # Average across sequence positions
                self.attention_history[step][layer_idx] = avg_dist.detach().cpu()
            else:
                self.attention_history[step][layer_idx] = dist[0].detach().cpu()
    
    def record_performance(self, step: int, metrics: Dict[str, float]):
        """
        Record performance metrics.
        
        Args:
            step: Current training step
            metrics: Dictionary of performance metrics
        """
        self.performance_history[step] = metrics
    
    def track_step(self, step: int, eval_data=None, evaluator=None):
        """
        Track model changes at the current training step.
        
        Args:
            step: Current training step
            eval_data: Evaluation data for recording attention patterns (optional)
            evaluator: Evaluator object for performance metrics (optional)
        """
        if step % self.tracking_frequency != 0:
            return
            
        # Record gate values
        self.snapshot_gates(step)
        
        # Optionally record attention patterns and entropy
        if eval_data is not None:
            try:
                distributions = collect_attention_distributions(
                    self.model, 
                    eval_data, 
                    num_batches=1
                )
                self.record_entropy(step, distributions)
                self.record_attention(step, distributions)
            except Exception as e:
                logger.warning(f"Failed to collect attention distributions: {e}")
            
        # Record performance metrics
        if eval_data is not None and evaluator is not None:
            try:
                metrics = evaluator.evaluate(self.model, eval_data)
                self.record_performance(step, metrics)
            except Exception as e:
                logger.warning(f"Failed to evaluate model: {e}")
    
    def analyze_regrowth(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Analyze which pruned heads have regrown.
        
        Returns:
            Dictionary mapping (layer_idx, head_idx) to regrowth metrics
        """
        regrowth_data = {}
        
        # Get initial and final states
        if not self.gate_history:
            logger.warning("No gate history recorded")
            return regrowth_data
            
        initial_gates = self.gate_history[min(self.gate_history.keys())]
        final_gates = self.gate_history[max(self.gate_history.keys())]
        
        # Identify regrown heads
        for layer_idx, head_idx, _ in self.pruned_heads:
            if layer_idx in initial_gates and layer_idx in final_gates:
                initial_value = initial_gates[layer_idx][head_idx].item()
                final_value = final_gates[layer_idx][head_idx].item()
                
                # Check if the head has significantly regrown
                if initial_value < 0.1 and final_value > 0.3:
                    regrowth_data[(layer_idx, head_idx)] = {
                        'initial_value': initial_value,
                        'final_value': final_value,
                        'regrowth_ratio': final_value / (initial_value + 1e-10)
                    }
                    
                    # Add entropy change if available
                    if self.entropy_history and min(self.entropy_history.keys()) in self.entropy_history and max(self.entropy_history.keys()) in self.entropy_history:
                        if layer_idx in self.entropy_history[min(self.entropy_history.keys())] and layer_idx in self.entropy_history[max(self.entropy_history.keys())]:
                            initial_entropy = self.entropy_history[min(self.entropy_history.keys())][layer_idx][head_idx].item()
                            final_entropy = self.entropy_history[max(self.entropy_history.keys())][layer_idx][head_idx].item()
                            entropy_change = final_entropy - initial_entropy
                            regrowth_data[(layer_idx, head_idx)]['initial_entropy'] = initial_entropy
                            regrowth_data[(layer_idx, head_idx)]['final_entropy'] = final_entropy
                            regrowth_data[(layer_idx, head_idx)]['entropy_change'] = entropy_change
        
        return regrowth_data
    
    def get_entropy_deltas(self) -> Dict[int, torch.Tensor]:
        """
        Calculate entropy changes for each head.
        
        Returns:
            Dictionary mapping layer indices to entropy change tensors
        """
        if not self.entropy_history:
            logger.warning("No entropy history recorded")
            return {}
            
        initial_step = min(self.entropy_history.keys())
        final_step = max(self.entropy_history.keys())
        
        entropy_deltas = {}
        for layer_idx in self.entropy_history[initial_step]:
            if layer_idx in self.entropy_history[final_step]:
                entropy_deltas[layer_idx] = (
                    self.entropy_history[final_step][layer_idx] - 
                    self.entropy_history[initial_step][layer_idx]
                )
        
        return entropy_deltas
    
    def save_tracking_data(self, output_dir: str):
        """
        Save tracking data to disk.
        
        Args:
            output_dir: Directory to save data to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_gate_history = {}
        for step, layers in self.gate_history.items():
            serializable_gate_history[str(step)] = {}
            for layer_idx, gate in layers.items():
                serializable_gate_history[str(step)][str(layer_idx)] = gate.tolist()
        
        serializable_entropy_history = {}
        for step, layers in self.entropy_history.items():
            serializable_entropy_history[str(step)] = {}
            for layer_idx, entropy in layers.items():
                serializable_entropy_history[str(step)][str(layer_idx)] = entropy.tolist()
        
        # Save data
        with open(os.path.join(output_dir, 'gate_history.json'), 'w') as f:
            json.dump(serializable_gate_history, f, indent=2)
        
        with open(os.path.join(output_dir, 'entropy_history.json'), 'w') as f:
            json.dump(serializable_entropy_history, f, indent=2)
        
        with open(os.path.join(output_dir, 'performance_history.json'), 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        # Save pruned heads
        pruned_heads_data = [(l, h, float(s)) for l, h, s in self.pruned_heads]
        with open(os.path.join(output_dir, 'pruned_heads.json'), 'w') as f:
            json.dump(pruned_heads_data, f, indent=2)
        
        # Save regrowth analysis
        regrowth_data = self.analyze_regrowth()
        serializable_regrowth = {f"{l}_{h}": data for (l, h), data in regrowth_data.items()}
        with open(os.path.join(output_dir, 'regrowth_analysis.json'), 'w') as f:
            json.dump(serializable_regrowth, f, indent=2)


class AdaptiveFinetuner:
    """
    Fine-tunes models with differential learning rates based on pruning status.
    
    This class implements:
    1. Adaptive learning rate assignment based on pruned layers
    2. Integration with PlasticityTracker for monitoring adaptation
    3. Training loop with scheduled evaluation
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 2e-5,
        use_differential_lr: bool = True,
        tracking_frequency: int = 10
    ):
        """
        Initialize the adaptive fine-tuner.
        
        Args:
            model: The model to fine-tune
            learning_rate: Base learning rate
            use_differential_lr: Whether to use different learning rates for pruned layers
            tracking_frequency: How often to record metrics during fine-tuning
        """
        self.model = model
        self.base_lr = learning_rate
        self.use_differential_lr = use_differential_lr
        self.plasticity_tracker = PlasticityTracker(model, tracking_frequency)
        
    def prepare_optimizer(self, pruned_heads: List[Tuple[int, int, float]]):
        """
        Create optimizer with adaptive learning rates.
        
        Args:
            pruned_heads: List of (layer_idx, head_idx, score) tuples of pruned heads
            
        Returns:
            Optimizer with layer-specific learning rates
        """
        if not self.use_differential_lr:
            return torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
            
        # Extract pruned layers
        pruned_layers = set(layer_idx for layer_idx, _, _ in pruned_heads)
        
        # Access model blocks based on architecture
        blocks = None
        if hasattr(self.model, 'blocks'):
            blocks = self.model.blocks
        else:
            # Try to access through transformer attribute
            transformer = None
            if hasattr(self.model, 'transformer'):
                transformer = self.model.transformer
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'transformer'):
                transformer = self.model.model.transformer
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'transformer'):
                transformer = self.model.base_model.transformer
                
            if transformer and hasattr(transformer, 'h'):
                blocks = transformer.h
        
        if blocks is None:
            logger.warning("Could not identify model blocks, using default learning rate")
            return torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
        
        # Group parameters with different learning rates
        pruned_layer_params = []
        other_params = []
        
        for layer_idx, layer in enumerate(blocks):
            # Check if this layer had pruned heads
            layer_pruned = layer_idx in pruned_layers
            
            for name, param in layer.named_parameters():
                if layer_pruned and 'attn' in name:
                    # Higher learning rate for attention in pruned layers
                    pruned_layer_params.append(param)
                else:
                    other_params.append(param)
        
        # Add any parameters not in the blocks
        for name, param in self.model.named_parameters():
            if not any(param is p for p in pruned_layer_params + other_params):
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': pruned_layer_params, 'lr': self.base_lr * 3.0},
            {'params': other_params, 'lr': self.base_lr}
        ]
        
        logger.info(f"Using adaptive learning rates: {self.base_lr * 3.0} for pruned layers, {self.base_lr} for others")
        return torch.optim.AdamW(param_groups)
        
    def fine_tune(
        self,
        train_dataloader,
        eval_dataloader,
        pruned_heads: List[Tuple[int, int, float]],
        steps: int = 1000,
        evaluator = None,
        eval_interval: int = 50
    ) -> Dict[str, Any]:
        """
        Fine-tune the model with plasticity tracking.
        
        Args:
            train_dataloader: DataLoader for training
            eval_dataloader: DataLoader for evaluation
            pruned_heads: List of (layer_idx, head_idx, score) tuples of pruned heads
            steps: Number of training steps
            evaluator: Evaluator object for performance metrics (optional)
            eval_interval: How often to evaluate the model
            
        Returns:
            Dictionary containing training results and tracking data
        """
        # Initialize tracking
        self.plasticity_tracker.initialize_tracking(pruned_heads)
        
        # Prepare optimizer and scheduler
        optimizer = self.prepare_optimizer(pruned_heads)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=self.base_lr * 0.1
        )
        
        # Set up loss function
        device = next(self.model.parameters()).device
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initialize eval data for tracking
        eval_batch = next(iter(eval_dataloader))
        
        # Training loop
        self.model.train()
        global_step = 0
        train_iterator = iter(train_dataloader)
        progress_bar = tqdm(total=steps, desc="Fine-tuning")
        
        while global_step < steps:
            # Get next batch (with cycling)
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            # Move to device
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                # Assume batch is a tuple/list with input_ids first
                input_ids = batch[0].to(device)
                inputs = {"input_ids": input_ids}
                if len(batch) > 1:
                    inputs["attention_mask"] = batch[1].to(device)
                if len(batch) > 2:
                    inputs["labels"] = batch[2].to(device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Calculate loss
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
                
            if "labels" in inputs:
                labels = inputs["labels"]
            elif len(batch) > 2:
                labels = batch[2].to(device)
            else:
                # For causal language modeling, shift labels
                labels = inputs["input_ids"].clone()
                labels = labels[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            
            # Track plasticity
            if global_step % eval_interval == 0:
                # Switch to eval mode temporarily
                self.model.eval()
                self.plasticity_tracker.track_step(
                    global_step, 
                    eval_data=eval_dataloader,
                    evaluator=evaluator
                )
                self.model.train()
                
            global_step += 1
        
        progress_bar.close()
        
        # Final evaluation in eval mode
        self.model.eval()
        if evaluator:
            final_metrics = evaluator.evaluate(self.model, eval_dataloader)
        else:
            final_metrics = {"loss": loss.item()}
            
        # Final tracking snapshot
        self.plasticity_tracker.track_step(
            global_step, 
            eval_data=eval_dataloader,
            evaluator=evaluator
        )
        
        # Analyze regrowth patterns
        regrowth_data = self.plasticity_tracker.analyze_regrowth()
        
        return {
            'final_metrics': final_metrics,
            'regrowth_data': regrowth_data,
            'gate_history': self.plasticity_tracker.gate_history,
            'entropy_history': self.plasticity_tracker.entropy_history,
            'performance_history': self.plasticity_tracker.performance_history,
            'pruned_heads': pruned_heads
        }


class PlasticityExperiment:
    """
    Orchestrates complete plasticity experiments.
    
    This class coordinates:
    1. Model preparation and pruning
    2. Fine-tuning with adaptive learning rates
    3. Analysis of plasticity and regrowth patterns
    4. Result visualization and storage
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: str = None,
        adaptive_model: bool = True
    ):
        """
        Initialize a plasticity experiment.
        
        Args:
            model_name: Name of the pre-trained model to use
            output_dir: Directory to save results
            device: Device to run on (auto-detected if None)
            adaptive_model: Whether to use adaptive model wrapper (recommended)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adaptive_model = adaptive_model
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_model(self):
        """
        Load the model based on experiment settings.
        
        Returns:
            Loaded model
        """
        if self.adaptive_model:
            # Load with adaptive wrapper
            baseline_model = load_baseline_model(self.model_name, self.device)
            model = load_adaptive_model(self.model_name, baseline_model, self.device)
            return model
        else:
            # Load standard model
            return load_baseline_model(self.model_name, self.device)
    
    def _apply_pruning(
        self,
        model,
        strategy: str,
        prune_ratio: float,
        eval_dataloader
    ) -> List[Tuple[int, int, float]]:
        """
        Apply pruning to the model.
        
        Args:
            model: Model to prune
            strategy: Pruning strategy ('entropy' or 'magnitude')
            prune_ratio: Ratio of heads to prune
            eval_dataloader: DataLoader for collecting attention distributions
            
        Returns:
            List of (layer_idx, head_idx, score) tuples of pruned heads
        """
        if strategy == "entropy":
            # Collect attention distributions for entropy pruning
            distributions = collect_attention_distributions(
                model,
                eval_dataloader,
                num_batches=5
            )
            
            # Apply entropy-based pruning
            pruned_heads = entropy_based_pruning(
                model,
                distributions,
                prune_ratio
            )
            
        elif strategy == "magnitude":
            # Apply magnitude-based pruning
            pruned_heads = magnitude_based_pruning(
                model,
                prune_ratio
            )
            
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
            
        return pruned_heads
    
    def _evaluate_model(self, model, dataloader, evaluator=None):
        """
        Evaluate the model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation
            evaluator: Optional custom evaluator
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # If no evaluator provided, use a simple perplexity evaluator
        if evaluator is None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            losses = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move to device
                    if isinstance(batch, dict):
                        inputs = {k: v.to(device) for k, v in batch.items()}
                    else:
                        # Assume batch is a tuple/list with input_ids first
                        input_ids = batch[0].to(device)
                        inputs = {"input_ids": input_ids}
                        if len(batch) > 1:
                            inputs["attention_mask"] = batch[1].to(device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    
                    # Calculate loss
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs
                        
                    # For causal language modeling, shift labels
                    labels = inputs["input_ids"].clone()
                    labels = labels[:, 1:].contiguous()
                    shift_logits = logits[:, :-1, :].contiguous()
                    
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                    losses.append(loss.item())
            
            avg_loss = sum(losses) / len(losses)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            return {
                "loss": avg_loss,
                "perplexity": perplexity
            }
        else:
            # Use provided evaluator
            return evaluator.evaluate(model, dataloader)
    
    def run_experiment(
        self,
        pruning_strategy: str,
        pruning_level: float,
        dataloader_builder_fn: Callable,
        fine_tuning_steps: int = 500,
        learning_rate: float = 2e-5,
        use_differential_lr: bool = True,
        evaluator=None,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Run a complete plasticity experiment.
        
        Args:
            pruning_strategy: Pruning strategy ('entropy' or 'magnitude')
            pruning_level: Ratio of heads to prune
            dataloader_builder_fn: Function that returns (train_dataloader, eval_dataloader)
            fine_tuning_steps: Number of fine-tuning steps
            learning_rate: Base learning rate for fine-tuning
            use_differential_lr: Whether to use different learning rates for pruned layers
            evaluator: Optional custom evaluator
            batch_size: Batch size for dataloaders
            
        Returns:
            Dictionary containing experimental results
        """
        experiment_id = f"{pruning_strategy}_{pruning_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = self.output_dir / experiment_id
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Log experiment parameters
        params = {
            "model_name": self.model_name,
            "pruning_strategy": pruning_strategy,
            "pruning_level": pruning_level,
            "fine_tuning_steps": fine_tuning_steps,
            "learning_rate": learning_rate,
            "use_differential_lr": use_differential_lr,
            "batch_size": batch_size,
            "device": str(self.device),
            "experiment_id": experiment_id
        }
        
        with open(experiment_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        
        # Load model
        logger.info(f"Loading model: {self.model_name}")
        model = self._load_model()
        
        # Create dataloaders
        logger.info("Preparing data")
        train_dataloader, eval_dataloader = dataloader_builder_fn(batch_size=batch_size)
        
        # Baseline evaluation
        logger.info("Running baseline evaluation")
        baseline_metrics = self._evaluate_model(model, eval_dataloader, evaluator)
        logger.info(f"Baseline metrics: {baseline_metrics}")
        
        # Collect initial entropy
        logger.info("Collecting initial attention distributions")
        pre_distributions = collect_attention_distributions(model, eval_dataloader, num_batches=5)
        pre_entropy = {
            layer: compute_attention_entropy(dist).detach().cpu()
            for layer, dist in pre_distributions.items()
        }
        
        # Apply pruning
        logger.info(f"Applying {pruning_strategy} pruning at level {pruning_level}")
        pruned_heads = self._apply_pruning(model, pruning_strategy, pruning_level, eval_dataloader)
        
        # Post-pruning evaluation
        logger.info("Evaluating model after pruning")
        post_pruning_metrics = self._evaluate_model(model, eval_dataloader, evaluator)
        logger.info(f"Post-pruning metrics: {post_pruning_metrics}")
        
        # Fine-tuning
        logger.info("Fine-tuning model")
        fine_tuner = AdaptiveFinetuner(
            model,
            learning_rate=learning_rate,
            use_differential_lr=use_differential_lr
        )
        
        fine_tuning_results = fine_tuner.fine_tune(
            train_dataloader,
            eval_dataloader,
            pruned_heads,
            steps=fine_tuning_steps,
            evaluator=evaluator
        )
        
        # Final evaluation
        logger.info("Performing final evaluation")
        final_metrics = self._evaluate_model(model, eval_dataloader, evaluator)
        logger.info(f"Final metrics: {final_metrics}")
        
        # Calculate recovery rate
        if "perplexity" in baseline_metrics and "perplexity" in post_pruning_metrics and "perplexity" in final_metrics:
            pruning_impact = post_pruning_metrics["perplexity"] - baseline_metrics["perplexity"]
            recovery = post_pruning_metrics["perplexity"] - final_metrics["perplexity"]
            recovery_rate = recovery / (pruning_impact + 1e-10) if pruning_impact > 0 else 0.0
            
            logger.info(f"Recovery rate: {recovery_rate:.2%}")
            fine_tuning_results["recovery_rate"] = recovery_rate
        
        # Collect final entropy
        logger.info("Collecting final attention distributions")
        post_distributions = collect_attention_distributions(model, eval_dataloader, num_batches=5)
        post_entropy = {
            layer: compute_attention_entropy(dist).detach().cpu()
            for layer, dist in post_distributions.items()
        }
        
        # Calculate entropy deltas
        entropy_deltas = {}
        for layer in pre_entropy:
            if layer in post_entropy:
                entropy_deltas[layer] = post_entropy[layer] - pre_entropy[layer]
        
        # Save results
        logger.info("Saving results")
        fine_tuner.plasticity_tracker.save_tracking_data(experiment_dir)
        
        # Save entropy data
        serializable_pre_entropy = {str(k): v.tolist() for k, v in pre_entropy.items()}
        serializable_post_entropy = {str(k): v.tolist() for k, v in post_entropy.items()}
        serializable_entropy_deltas = {str(k): v.tolist() for k, v in entropy_deltas.items()}
        
        with open(experiment_dir / "pre_entropy.json", "w") as f:
            json.dump(serializable_pre_entropy, f, indent=2)
            
        with open(experiment_dir / "post_entropy.json", "w") as f:
            json.dump(serializable_post_entropy, f, indent=2)
            
        with open(experiment_dir / "entropy_deltas.json", "w") as f:
            json.dump(serializable_entropy_deltas, f, indent=2)
        
        # Save metrics
        metrics = {
            "baseline": baseline_metrics,
            "post_pruning": post_pruning_metrics,
            "final": final_metrics
        }
        
        with open(experiment_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create result summary
        results = {
            "params": params,
            "metrics": metrics,
            "regrowth_data": fine_tuning_results["regrowth_data"],
            "recovery_rate": fine_tuning_results.get("recovery_rate", 0.0),
            "pruned_heads": [(l, h, float(s)) for l, h, s in pruned_heads],
            "entropy_change_summary": {
                "layers": len(entropy_deltas),
                "heads_with_decreased_entropy": sum(
                    (v < 0).sum().item() for v in entropy_deltas.values()
                ),
                "heads_with_increased_entropy": sum(
                    (v > 0).sum().item() for v in entropy_deltas.values()
                )
            }
        }
        
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment completed successfully. Results saved to {experiment_dir}")
        
        return results


def run_plasticity_experiment(
    model_name: str,
    pruning_strategy: str = "entropy",
    prune_ratio: float = 0.3,
    learning_rate: float = 5e-6,
    adaptive_lr: bool = True,
    learning_steps: int = 500,
    batch_size: int = 4,
    dataloader_builder_fn = None,
    device: Optional[str] = None,
    output_dir: str = "./plasticity_results"
) -> Dict[str, Any]:
    """
    Main plasticity loop experiment.

    1. Load baseline model
    2. Measure pre-finetune entropy
    3. Prune heads
    4. Fine-tune model
    5. Measure post-finetune entropy
    6. Analyze plasticity patterns

    Args:
        model_name: name of the base model (e.g., 'distilgpt2')
        pruning_strategy: 'entropy' or 'magnitude'
        prune_ratio: ratio of heads to prune
        learning_rate: base learning rate
        adaptive_lr: whether to use higher LR for pruned heads
        learning_steps: number of fine-tuning steps
        batch_size: for training and evaluation
        dataloader_builder_fn: function returning train and val dataloaders
        device: compute device (auto-detected if None)
        output_dir: directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    # Create experiment
    experiment = PlasticityExperiment(
        model_name=model_name,
        output_dir=output_dir,
        device=device,
        adaptive_model=True
    )
    
    # Run experiment
    results = experiment.run_experiment(
        pruning_strategy=pruning_strategy,
        pruning_level=prune_ratio,
        dataloader_builder_fn=dataloader_builder_fn,
        fine_tuning_steps=learning_steps,
        learning_rate=learning_rate,
        use_differential_lr=adaptive_lr,
        batch_size=batch_size
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a neural plasticity experiment")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--pruning_strategy", type=str, default="entropy", choices=["entropy", "magnitude"], help="Pruning strategy")
    parser.add_argument("--prune_ratio", type=float, default=0.3, help="Pruning ratio")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--no_adaptive_lr", action="store_true", help="Disable adaptive learning rates")
    parser.add_argument("--learning_steps", type=int, default=500, help="Number of fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./plasticity_results", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Compute device")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Setup dataloaders
    try:
        from sentinel.datasets.loader import get_gutenberg_dataloaders
        dataloader_builder = lambda batch_size: get_gutenberg_dataloaders(batch_size=batch_size)
    except ImportError:
        logger.warning("Gutenberg dataset loader not found. Using fallback dataloader.")
        
        # Fallback dataset loader
        def get_fallback_dataloaders(batch_size=4):
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create synthetic data
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "In a world where technology dominates, humans seek connection.",
                "Once upon a time, there lived a wise king who ruled with compassion.",
                "The history of artificial intelligence dates back to ancient myths.",
                "Climate change is affecting ecosystems worldwide, leading to rising sea levels.",
            ] * 10  # Repeat to create more samples
            
            # Tokenize
            from torch.utils.data import TensorDataset, DataLoader
            
            encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            dataset = TensorDataset(input_ids, attention_mask)
            
            # Split into train and eval
            train_size = int(0.8 * len(dataset))
            eval_size = len(dataset) - train_size
            
            train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
            
            return train_dataloader, eval_dataloader
        
        dataloader_builder = get_fallback_dataloaders
    
    # Run experiment
    results = run_plasticity_experiment(
        model_name=args.model_name,
        pruning_strategy=args.pruning_strategy,
        prune_ratio=args.prune_ratio,
        learning_rate=args.learning_rate,
        adaptive_lr=not args.no_adaptive_lr,
        learning_steps=args.learning_steps,
        batch_size=args.batch_size,
        dataloader_builder_fn=dataloader_builder,
        device=args.device,
        output_dir=args.output_dir
    )
    
    logger.info(f"Experiment complete. Results:")
    logger.info(f"Baseline metrics: {results['metrics']['baseline']}")
    logger.info(f"Post-pruning metrics: {results['metrics']['post_pruning']}")
    logger.info(f"Final metrics: {results['metrics']['final']}")
    logger.info(f"Recovery rate: {results.get('recovery_rate', 0.0):.2%}")
    logger.info(f"Regrown heads: {len(results['regrowth_data'])}")