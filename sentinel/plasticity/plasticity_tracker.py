"""
Plasticity Tracker for Neural Plasticity Experiments

This module provides tracking and analysis capabilities for neural plasticity
experiments, recording performance metrics and head activation patterns.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class PlasticityTracker:
    """
    Tracks neural plasticity metrics during experiments.
    
    This class is responsible for:
    1. Recording performance metrics during training
    2. Tracking head activations and contributions
    3. Analyzing recovery patterns post-pruning
    4. Storing and loading tracking data
    """
    
    def __init__(self):
        """Initialize the plasticity tracker."""
        # Performance metrics
        self.performance_history = {
            "loss": [],
            "perplexity": [],
            "step": []
        }
        
        # Head activations and gates
        self.head_gates = {}
        self.entropy_history = {}
        self.gate_history = {}
        
    def record_performance(self, step: int, loss: float, perplexity: Optional[float] = None):
        """
        Record performance metrics at a given step.
        
        Args:
            step: Current training step
            loss: Loss value
            perplexity: Optional perplexity value
        """
        self.performance_history["step"].append(step)
        self.performance_history["loss"].append(loss)
        
        if perplexity is not None:
            self.performance_history["perplexity"].append(perplexity)
        else:
            # Calculate perplexity from loss if not provided
            self.performance_history["perplexity"].append(float(np.exp(loss)))
            
    def record_head_gates(self, layer_idx: int, head_gates: torch.Tensor):
        """
        Record attention head gate values.
        
        Args:
            layer_idx: Layer index
            head_gates: Tensor of head gate values
        """
        if layer_idx not in self.head_gates:
            self.head_gates[layer_idx] = []
            
        # Convert tensor to list for storage
        if isinstance(head_gates, torch.Tensor):
            gate_values = head_gates.detach().cpu().tolist()
        else:
            gate_values = head_gates
            
        self.head_gates[layer_idx].append(gate_values)
        
    def record_entropy(self, step: int, entropy_by_layer: Dict[int, torch.Tensor]):
        """
        Record entropy values by layer at a given step.
        
        Args:
            step: Current training step
            entropy_by_layer: Dictionary mapping layer indices to entropy tensors
        """
        # Initialize step entry if not exists
        if step not in self.entropy_history:
            self.entropy_history[step] = {}
            
        # Record entropy for each layer
        for layer_idx, entropy in entropy_by_layer.items():
            if isinstance(entropy, torch.Tensor):
                entropy_values = entropy.detach().cpu().tolist()
            else:
                entropy_values = entropy
                
            self.entropy_history[step][layer_idx] = entropy_values
            
    def record_gate_values(self, step: int, gate_by_layer: Dict[int, torch.Tensor]):
        """
        Record gate values by layer at a given step.
        
        Args:
            step: Current training step
            gate_by_layer: Dictionary mapping layer indices to gate tensors
        """
        # Initialize step entry if not exists
        if step not in self.gate_history:
            self.gate_history[step] = {}
            
        # Record gate values for each layer
        for layer_idx, gates in gate_by_layer.items():
            if isinstance(gates, torch.Tensor):
                gate_values = gates.detach().cpu().tolist()
            else:
                gate_values = gates
                
            self.gate_history[step][layer_idx] = gate_values
            
    def analyze_regrowth(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Analyze regrowth patterns in pruned heads.
        
        Returns:
            Dictionary mapping (layer, head) tuples to regrowth analysis data
        """
        regrowth_data = {}
        
        # Analyze only if we have gate history
        if not self.gate_history:
            logger.warning("No gate history available for regrowth analysis")
            return regrowth_data
            
        # Get all steps in order
        steps = sorted(self.gate_history.keys())
        if len(steps) < 2:
            logger.warning("Insufficient gate history for regrowth analysis")
            return regrowth_data
            
        # Get initial gates (after pruning) and final gates
        initial_gates = self.gate_history[steps[0]]
        final_gates = self.gate_history[steps[-1]]
        
        # Calculate regrowth for each layer and head
        for layer_idx in initial_gates.keys():
            if layer_idx not in final_gates:
                continue
                
            initial_layer_gates = initial_gates[layer_idx]
            final_layer_gates = final_gates[layer_idx]
            
            # Skip if dimensions don't match
            if len(initial_layer_gates) != len(final_layer_gates):
                continue
                
            # Check each head in the layer
            for head_idx, (initial_gate, final_gate) in enumerate(
                zip(initial_layer_gates, final_layer_gates)
            ):
                # Calculate growth
                growth = final_gate - initial_gate
                
                # Consider significant regrowth if gate increased by more than threshold
                if growth > 0.05:  # 5% threshold for significant regrowth
                    regrowth_data[(layer_idx, head_idx)] = {
                        "initial_gate": initial_gate,
                        "final_gate": final_gate,
                        "growth": growth,
                        "growth_percent": (growth / max(0.001, initial_gate)) * 100
                    }
                    
        return regrowth_data
        
    def save_tracking_data(self, output_dir: str):
        """
        Save tracking data to output directory.
        
        Args:
            output_dir: Directory to save tracking data
        """
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Save performance history
        with open(output_path / "performance_history.json", "w") as f:
            json.dump(self.performance_history, f, indent=2)
            
        # Save gate history
        with open(output_path / "gate_history.json", "w") as f:
            json.dump(self.gate_history, f, indent=2)
            
        # Save entropy history
        with open(output_path / "entropy_history.json", "w") as f:
            json.dump(self.entropy_history, f, indent=2)
            
        # Calculate and save regrowth analysis
        regrowth_data = self.analyze_regrowth()
        
        # Convert tuple keys to strings for JSON serialization
        serializable_regrowth = {}
        for (layer, head), data in regrowth_data.items():
            serializable_regrowth[f"{layer}_{head}"] = data
            
        with open(output_path / "regrowth_analysis.json", "w") as f:
            json.dump(serializable_regrowth, f, indent=2)
            
        logger.info(f"Saved plasticity tracking data to {output_dir}")
        
    def load_tracking_data(self, input_dir: str) -> bool:
        """
        Load tracking data from input directory.
        
        Args:
            input_dir: Directory to load tracking data from
            
        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_dir)
        
        try:
            # Load performance history
            with open(input_path / "performance_history.json", "r") as f:
                self.performance_history = json.load(f)
                
            # Load gate history
            with open(input_path / "gate_history.json", "r") as f:
                self.gate_history = json.load(f)
                
            # Load entropy history if available
            entropy_path = input_path / "entropy_history.json"
            if entropy_path.exists():
                with open(entropy_path, "r") as f:
                    self.entropy_history = json.load(f)
                    
            logger.info(f"Loaded plasticity tracking data from {input_dir}")
            return True
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load tracking data: {e}")
            return False