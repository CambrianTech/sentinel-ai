#!/usr/bin/env python
"""
Sleep Cycle for Transformer Models

This module implements the sleep cycle concept for transformer models,
alternating between active learning phases and maintenance phases
in a way that mimics biological sleep-wake cycles.

During the active phase, the model learns normally through training.
During the maintenance phase, the model undergoes neural defragmentation,
pruning, and reorganization to consolidate knowledge and optimize
internal representations.
"""

import os
import torch
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from .defrag_heads import HeadDefragmenter, DefragConfiguration
from ..pruning.strategies.entropy import EntropyPruningStrategy
from ..pruning.strategies.magnitude import MagnitudePruningStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Phases of the sleep-wake cycle."""
    ACTIVE = "active"  # Active learning/inference
    MAINTENANCE = "maintenance"  # "Sleep" phase - defrag, prune, consolidate


@dataclass
class SleepCycleConfig:
    """Configuration for the transformer sleep cycle."""
    
    # Cycle timing
    active_steps: int = 1000  # Steps to stay in active phase
    maintenance_steps: int = 200  # Steps for maintenance phase
    maintenance_batch_size: int = 8  # Batch size during maintenance
    
    # Maintenance operations to perform
    enable_defrag: bool = True
    enable_pruning: bool = True  
    enable_regrowth: bool = True
    
    # Defragmentation settings
    defrag_config: Optional[DefragConfiguration] = None
    
    # Pruning settings
    pruning_strategy: str = "entropy"  # "entropy" or "magnitude"
    pruning_level: float = 0.1  # Percentage of heads to prune
    
    # Regrowth settings
    regrowth_percentage: float = 0.05  # Percentage of heads to regrow
    
    # Learning rate adjustments after maintenance
    lr_boost_new_heads: float = 5.0  # Learning rate multiplier for new heads
    
    # Monitoring and metrics
    log_cycle_metrics: bool = True
    visualization_dir: str = "./output/sleep_cycle"
    

class TransformerSleepCycle:
    """
    Implements sleep-wake cycles for transformer models.
    
    This class manages the alternation between active learning and 
    maintenance phases, implementing the neural defragmentation and
    consolidation processes during the "sleep" phase.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader_fn: Callable[[], torch.utils.data.DataLoader],
        config: Optional[SleepCycleConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the sleep cycle manager.
        
        Args:
            model: The transformer model to manage
            dataloader_fn: Function that returns a dataloader for maintenance phase
            config: Sleep cycle configuration
            device: Device to run computations on
        """
        self.model = model
        self.dataloader_fn = dataloader_fn
        self.config = config or SleepCycleConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize state
        self.current_phase = CyclePhase.ACTIVE
        self.steps_in_phase = 0
        self.total_active_steps = 0
        self.total_maintenance_steps = 0
        self.cycle_count = 0
        
        # Metrics tracking
        self.metrics_history = {
            'active_perplexity': [],
            'maintenance_perplexity': [],
            'head_count': [],
            'entropy': [],
            'pruning_events': [],
            'defrag_events': [],
            'regrowth_events': [],
        }
        
        # Create output directory if needed
        if self.config.log_cycle_metrics:
            os.makedirs(self.config.visualization_dir, exist_ok=True)
        
        logger.info(f"Initialized TransformerSleepCycle (device: {self.device})")
        logger.info(f"Initial phase: {self.current_phase.value}")
    
    def step(self) -> CyclePhase:
        """
        Step the sleep cycle forward and determine if phase should change.
        
        Returns:
            Current phase after the step
        """
        self.steps_in_phase += 1
        
        if self.current_phase == CyclePhase.ACTIVE:
            self.total_active_steps += 1
            
            # Check if we should transition to maintenance
            if self.steps_in_phase >= self.config.active_steps:
                self._transition_to_maintenance()
                
        elif self.current_phase == CyclePhase.MAINTENANCE:
            self.total_maintenance_steps += 1
            
            # Check if we should transition to active
            if self.steps_in_phase >= self.config.maintenance_steps:
                self._transition_to_active()
        
        return self.current_phase
    
    def _transition_to_maintenance(self) -> None:
        """Transition from active to maintenance phase."""
        logger.info(f"Transitioning to MAINTENANCE phase after {self.steps_in_phase} active steps")
        
        # Reset step counter
        self.steps_in_phase = 0
        
        # Change phase
        self.current_phase = CyclePhase.MAINTENANCE
        
        # Perform maintenance operations
        self._perform_maintenance()
        
        # Increment cycle count
        self.cycle_count += 1
        
        logger.info(f"Completed maintenance cycle #{self.cycle_count}")
    
    def _transition_to_active(self) -> None:
        """Transition from maintenance to active phase."""
        logger.info(f"Transitioning to ACTIVE phase after {self.steps_in_phase} maintenance steps")
        
        # Reset step counter
        self.steps_in_phase = 0
        
        # Change phase
        self.current_phase = CyclePhase.ACTIVE
        
        # Any post-maintenance adjustments can go here
        
        logger.info(f"Beginning active learning phase #{self.cycle_count + 1}")
    
    def _perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform maintenance operations during sleep phase.
        
        Returns:
            Dictionary of maintenance metrics
        """
        logger.info("Beginning maintenance operations")
        
        maintenance_results = {}
        
        # Get dataloader for maintenance operations
        dataloader = self.dataloader_fn()
        
        # 1. Perform neural defragmentation if enabled
        if self.config.enable_defrag:
            defrag_results = self._perform_defragmentation(dataloader)
            maintenance_results['defrag'] = defrag_results
            
            # Record metrics
            self.metrics_history['defrag_events'].append({
                'cycle': self.cycle_count,
                'redundant_heads': defrag_results.get('redundant_heads', 0),
                'dead_heads': defrag_results.get('dead_heads', 0),
                'merged_heads': defrag_results.get('merged_heads', 0),
                'reinitialized_heads': defrag_results.get('reinitialized_heads', 0),
            })
        
        # 2. Perform pruning if enabled
        if self.config.enable_pruning:
            pruning_results = self._perform_pruning(dataloader)
            maintenance_results['pruning'] = pruning_results
            
            # Record metrics
            self.metrics_history['pruning_events'].append({
                'cycle': self.cycle_count,
                'pruned_heads': pruning_results.get('pruned_heads', 0),
                'strategy': self.config.pruning_strategy,
                'level': self.config.pruning_level,
            })
        
        # 3. Perform head regrowth if enabled
        if self.config.enable_regrowth:
            regrowth_results = self._perform_regrowth(dataloader)
            maintenance_results['regrowth'] = regrowth_results
            
            # Record metrics
            self.metrics_history['regrowth_events'].append({
                'cycle': self.cycle_count,
                'regrown_heads': regrowth_results.get('regrown_heads', 0),
                'percentage': self.config.regrowth_percentage,
            })
        
        # 4. Update metrics history with current model state
        self._update_metrics_history()
        
        # 5. Create visualizations if enabled
        if self.config.log_cycle_metrics:
            self._create_cycle_visualizations()
        
        return maintenance_results
    
    def _perform_defragmentation(
        self, 
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Perform neural defragmentation.
        
        Args:
            dataloader: DataLoader for analysis
            
        Returns:
            Dictionary of defragmentation metrics
        """
        logger.info("Performing neural defragmentation")
        
        # Use the configured defrag settings or defaults
        defrag_config = self.config.defrag_config or DefragConfiguration(
            visualization_path=os.path.join(self.config.visualization_dir, f"defrag_cycle_{self.cycle_count}")
        )
        
        # Create defragmenter and run process
        defragmenter = HeadDefragmenter(self.model, defrag_config, self.device)
        defrag_results = defragmenter.defragment(dataloader)
        
        logger.info(f"Defragmentation complete: merged {defrag_results['merged_heads']} heads, "
                   f"reinitialized {defrag_results['reinitialized_heads']} heads")
        
        return defrag_results
    
    def _perform_pruning(
        self, 
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Perform head pruning using the configured strategy.
        
        Args:
            dataloader: DataLoader for analysis
            
        Returns:
            Dictionary of pruning metrics
        """
        logger.info(f"Performing pruning with {self.config.pruning_strategy} strategy "
                   f"at {self.config.pruning_level:.1%} level")
        
        # This is a placeholder - actual implementation would depend on your pruning module
        # For a complete implementation, integrate with your pruning system
        
        # Example pseudocode:
        pruned_heads = 0
        try:
            if self.config.pruning_strategy == "entropy":
                strategy = EntropyPruningStrategy()
            elif self.config.pruning_strategy == "magnitude":
                strategy = MagnitudePruningStrategy()
            else:
                raise ValueError(f"Unknown pruning strategy: {self.config.pruning_strategy}")
            
            # Calculate head importance
            head_importance = strategy.get_head_importance(self.model)
            
            # Sort heads by importance (ascending)
            sorted_heads = sorted(head_importance, key=lambda x: x[2])
            
            # Calculate how many heads to prune
            total_heads = len(sorted_heads)
            heads_to_prune = int(total_heads * self.config.pruning_level)
            
            # Get heads to prune (least important first)
            heads_to_prune = sorted_heads[:heads_to_prune]
            
            # Perform pruning
            for layer_idx, head_idx, _ in heads_to_prune:
                # This needs to be customized to your model's pruning implementation
                self._prune_head(layer_idx, head_idx)
                pruned_heads += 1
                
        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
        
        logger.info(f"Pruned {pruned_heads} heads")
        
        return {
            'pruned_heads': pruned_heads,
            'strategy': self.config.pruning_strategy,
            'level': self.config.pruning_level,
        }
    
    def _prune_head(self, layer_idx: int, head_idx: int) -> None:
        """
        Prune a specific attention head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index within the layer
        """
        # This is a placeholder - implement according to your model architecture
        try:
            # Example for a transformer with a gate mechanism
            # This assumes your model has a mechanism to disable heads with gates
            attn_module = self._get_attention_module(layer_idx)
            if attn_module and hasattr(attn_module, "head_gates"):
                # Set gate value to 0 to disable the head
                attn_module.head_gates.data[head_idx] = 0.0
                logger.info(f"Pruned head ({layer_idx}, {head_idx})")
        except Exception as e:
            logger.error(f"Error pruning head ({layer_idx}, {head_idx}): {str(e)}")
    
    def _perform_regrowth(
        self, 
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Perform head regrowth to replace pruned capacity.
        
        Args:
            dataloader: DataLoader for analysis
            
        Returns:
            Dictionary of regrowth metrics
        """
        logger.info(f"Performing head regrowth at {self.config.regrowth_percentage:.1%} level")
        
        # This is a placeholder - actual implementation would depend on your regrowth module
        # For a complete implementation, integrate with your existing head growth system
        
        # Example pseudocode:
        regrown_heads = 0
        try:
            # Find pruned heads that could be regrown
            pruned_heads = self._find_pruned_heads()
            
            # Calculate how many heads to regrow
            total_heads = self._count_total_heads()
            heads_to_regrow = int(total_heads * self.config.regrowth_percentage)
            heads_to_regrow = min(heads_to_regrow, len(pruned_heads))
            
            # Select heads to regrow
            # In a real implementation, you might want to use a more sophisticated strategy
            heads_to_regrow = pruned_heads[:heads_to_regrow]
            
            # Regrow selected heads
            for layer_idx, head_idx in heads_to_regrow:
                self._regrow_head(layer_idx, head_idx)
                regrown_heads += 1
                
        except Exception as e:
            logger.error(f"Error during head regrowth: {str(e)}")
        
        logger.info(f"Regrew {regrown_heads} heads")
        
        return {
            'regrown_heads': regrown_heads,
            'percentage': self.config.regrowth_percentage,
        }
    
    def _find_pruned_heads(self) -> List[Tuple[int, int]]:
        """
        Find heads that are currently pruned and could be regrown.
        
        Returns:
            List of (layer_idx, head_idx) tuples for pruned heads
        """
        # This is a placeholder - implement according to your model architecture
        pruned_heads = []
        
        try:
            # Example: Iterate through all layers and find disabled heads
            num_layers = self._get_num_layers()
            for layer_idx in range(num_layers):
                attn_module = self._get_attention_module(layer_idx)
                if attn_module and hasattr(attn_module, "head_gates"):
                    head_gates = attn_module.head_gates.data
                    for head_idx, gate in enumerate(head_gates):
                        if gate.item() < 0.01:  # Head is effectively disabled
                            pruned_heads.append((layer_idx, head_idx))
        except Exception as e:
            logger.error(f"Error finding pruned heads: {str(e)}")
        
        return pruned_heads
    
    def _regrow_head(self, layer_idx: int, head_idx: int) -> None:
        """
        Regrow a previously pruned attention head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index within the layer
        """
        # This is a placeholder - implement according to your model architecture
        try:
            # Example for a transformer with a gate mechanism
            attn_module = self._get_attention_module(layer_idx)
            if attn_module and hasattr(attn_module, "head_gates"):
                # Reinitialize the head weights with small random values
                query_weights = self._get_head_weights(attn_module, "query", head_idx)
                key_weights = self._get_head_weights(attn_module, "key", head_idx)
                value_weights = self._get_head_weights(attn_module, "value", head_idx)
                
                if query_weights is not None:
                    self._set_head_weights(attn_module, "query", head_idx, 
                                          torch.randn_like(query_weights) * 0.01)
                if key_weights is not None:
                    self._set_head_weights(attn_module, "key", head_idx, 
                                          torch.randn_like(key_weights) * 0.01)
                if value_weights is not None:
                    self._set_head_weights(attn_module, "value", head_idx, 
                                          torch.randn_like(value_weights) * 0.01)
                
                # Set gate value to small positive value to enable the head
                attn_module.head_gates.data[head_idx] = 0.5
                logger.info(f"Regrew head ({layer_idx}, {head_idx})")
        except Exception as e:
            logger.error(f"Error regrowing head ({layer_idx}, {head_idx}): {str(e)}")
    
    def _update_metrics_history(self) -> None:
        """Update metrics history with current model state."""
        try:
            # Count active heads
            active_heads = self._count_active_heads()
            self.metrics_history['head_count'].append({
                'cycle': self.cycle_count,
                'active_heads': active_heads,
                'total_heads': self._count_total_heads(),
            })
            
            # Store average entropy if available
            if hasattr(self, '_calculate_avg_entropy'):
                avg_entropy = self._calculate_avg_entropy()
                self.metrics_history['entropy'].append({
                    'cycle': self.cycle_count,
                    'avg_entropy': avg_entropy,
                })
        except Exception as e:
            logger.error(f"Error updating metrics history: {str(e)}")
    
    def _create_cycle_visualizations(self) -> None:
        """Create visualizations of sleep cycle metrics."""
        if not self.config.log_cycle_metrics:
            return
            
        try:
            # Create cycle-specific directory
            cycle_dir = os.path.join(self.config.visualization_dir, f"cycle_{self.cycle_count}")
            os.makedirs(cycle_dir, exist_ok=True)
            
            # 1. Plot head count over time
            if self.metrics_history['head_count']:
                cycles = [d['cycle'] for d in self.metrics_history['head_count']]
                active_heads = [d['active_heads'] for d in self.metrics_history['head_count']]
                total_heads = [d['total_heads'] for d in self.metrics_history['head_count']]
                
                plt.figure(figsize=(10, 6))
                plt.plot(cycles, active_heads, 'b-o', label='Active Heads')
                plt.plot(cycles, total_heads, 'k--', label='Total Heads')
                plt.xlabel('Sleep Cycle')
                plt.ylabel('Head Count')
                plt.title('Active Heads Over Sleep Cycles')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(cycle_dir, 'head_count.png'))
                plt.close()
            
            # 2. Plot pruning and regrowth events
            if self.metrics_history['pruning_events'] and self.metrics_history['regrowth_events']:
                prune_cycles = [d['cycle'] for d in self.metrics_history['pruning_events']]
                pruned_heads = [d['pruned_heads'] for d in self.metrics_history['pruning_events']]
                
                regrow_cycles = [d['cycle'] for d in self.metrics_history['regrowth_events']]
                regrown_heads = [d['regrown_heads'] for d in self.metrics_history['regrowth_events']]
                
                plt.figure(figsize=(10, 6))
                plt.bar(prune_cycles, pruned_heads, color='r', alpha=0.7, label='Pruned Heads')
                plt.bar(regrow_cycles, regrown_heads, color='g', alpha=0.7, label='Regrown Heads')
                plt.xlabel('Sleep Cycle')
                plt.ylabel('Head Count')
                plt.title('Pruning and Regrowth Events')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(cycle_dir, 'prune_regrow_events.png'))
                plt.close()
            
            # 3. Plot defragmentation metrics if available
            if self.metrics_history['defrag_events']:
                defrag_cycles = [d['cycle'] for d in self.metrics_history['defrag_events']]
                redundant_heads = [d['redundant_heads'] for d in self.metrics_history['defrag_events']]
                dead_heads = [d['dead_heads'] for d in self.metrics_history['defrag_events']]
                merged_heads = [d['merged_heads'] for d in self.metrics_history['defrag_events']]
                reinitialized_heads = [d['reinitialized_heads'] for d in self.metrics_history['defrag_events']]
                
                plt.figure(figsize=(12, 6))
                width = 0.2
                x = np.arange(len(defrag_cycles))
                
                plt.bar(x - width*1.5, redundant_heads, width, label='Redundant', color='orange')
                plt.bar(x - width*0.5, dead_heads, width, label='Dead', color='red')
                plt.bar(x + width*0.5, merged_heads, width, label='Merged', color='blue')
                plt.bar(x + width*1.5, reinitialized_heads, width, label='Reinitialized', color='green')
                
                plt.xlabel('Sleep Cycle')
                plt.ylabel('Head Count')
                plt.title('Defragmentation Metrics by Cycle')
                plt.xticks(x, defrag_cycles)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(cycle_dir, 'defrag_metrics.png'))
                plt.close()
                
            # 4. Create summary of this cycle
            self._create_cycle_summary(cycle_dir)
            
        except Exception as e:
            logger.error(f"Error creating cycle visualizations: {str(e)}")
    
    def _create_cycle_summary(self, cycle_dir: str) -> None:
        """
        Create a text summary of the current cycle.
        
        Args:
            cycle_dir: Directory to save the summary
        """
        try:
            summary_path = os.path.join(cycle_dir, "cycle_summary.txt")
            
            with open(summary_path, "w") as f:
                f.write(f"=== Sleep Cycle {self.cycle_count} Summary ===\n\n")
                
                f.write(f"Active steps: {self.config.active_steps}\n")
                f.write(f"Maintenance steps: {self.config.maintenance_steps}\n\n")
                
                f.write("--- Defragmentation ---\n")
                if self.metrics_history['defrag_events'] and self.cycle_count < len(self.metrics_history['defrag_events']):
                    defrag = self.metrics_history['defrag_events'][self.cycle_count]
                    f.write(f"Redundant heads detected: {defrag['redundant_heads']}\n")
                    f.write(f"Dead heads detected: {defrag['dead_heads']}\n")
                    f.write(f"Heads merged: {defrag['merged_heads']}\n")
                    f.write(f"Heads reinitialized: {defrag['reinitialized_heads']}\n")
                else:
                    f.write("No defragmentation data available\n")
                    
                f.write("\n--- Pruning ---\n")
                if self.metrics_history['pruning_events'] and self.cycle_count < len(self.metrics_history['pruning_events']):
                    pruning = self.metrics_history['pruning_events'][self.cycle_count]
                    f.write(f"Strategy: {pruning['strategy']}\n")
                    f.write(f"Pruning level: {pruning['level']:.1%}\n")
                    f.write(f"Heads pruned: {pruning['pruned_heads']}\n")
                else:
                    f.write("No pruning data available\n")
                    
                f.write("\n--- Regrowth ---\n")
                if self.metrics_history['regrowth_events'] and self.cycle_count < len(self.metrics_history['regrowth_events']):
                    regrowth = self.metrics_history['regrowth_events'][self.cycle_count]
                    f.write(f"Regrowth percentage: {regrowth['percentage']:.1%}\n")
                    f.write(f"Heads regrown: {regrowth['regrown_heads']}\n")
                else:
                    f.write("No regrowth data available\n")
                    
                f.write("\n--- Current Model State ---\n")
                if self.metrics_history['head_count'] and self.cycle_count < len(self.metrics_history['head_count']):
                    head_count = self.metrics_history['head_count'][self.cycle_count]
                    f.write(f"Active heads: {head_count['active_heads']} / {head_count['total_heads']}\n")
                    f.write(f"Utilization: {head_count['active_heads']/head_count['total_heads']:.1%}\n")
                else:
                    f.write("No head count data available\n")
                    
                f.write("\n=== End of Summary ===\n")
                
            logger.info(f"Created cycle summary at {summary_path}")
            
        except Exception as e:
            logger.error(f"Error creating cycle summary: {str(e)}")
    
    # Helper methods for accessing model internals
    # These need to be customized for your specific model architecture
    
    def _get_attention_module(self, layer_idx: int) -> Optional[torch.nn.Module]:
        """Get the attention module for a specific layer."""
        # This is a placeholder - implement according to your model architecture
        try:
            # Example for a typical transformer
            return self.model.transformer.layers[layer_idx].attention
        except (AttributeError, IndexError):
            return None
    
    def _get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        # This is a placeholder - implement according to your model architecture
        try:
            # Example for a typical transformer
            return len(self.model.transformer.layers)
        except AttributeError:
            return 0
    
    def _count_total_heads(self) -> int:
        """Count the total number of attention heads in the model."""
        # This is a placeholder - implement according to your model architecture
        try:
            num_layers = self._get_num_layers()
            if num_layers == 0:
                return 0
            
            # Assume all layers have the same number of heads
            first_layer = self._get_attention_module(0)
            if first_layer and hasattr(first_layer, "num_attention_heads"):
                return num_layers * first_layer.num_attention_heads
            
            return 0
        except Exception:
            return 0
    
    def _count_active_heads(self) -> int:
        """Count the number of active (non-pruned) attention heads."""
        # This is a placeholder - implement according to your model architecture
        try:
            active_count = 0
            num_layers = self._get_num_layers()
            
            for layer_idx in range(num_layers):
                attn_module = self._get_attention_module(layer_idx)
                if attn_module and hasattr(attn_module, "head_gates"):
                    head_gates = attn_module.head_gates.data
                    active_count += torch.sum(head_gates > 0.01).item()
            
            return active_count
        except Exception:
            return 0
    
    def _get_head_weights(
        self, 
        module: torch.nn.Module, 
        param_name: str, 
        head_idx: int
    ) -> Optional[torch.Tensor]:
        """Get weights for a specific attention head."""
        # This is a placeholder - implement according to your model architecture
        try:
            # Example for a typical transformer with multi-head attention
            param = getattr(module, f"{param_name}_proj.weight")
            hidden_size = param.shape[0] // module.num_attention_heads
            start_idx = head_idx * hidden_size
            end_idx = (head_idx + 1) * hidden_size
            return param[start_idx:end_idx, :]
        except (AttributeError, IndexError):
            return None
    
    def _set_head_weights(
        self, 
        module: torch.nn.Module, 
        param_name: str, 
        head_idx: int, 
        new_weights: torch.Tensor
    ) -> None:
        """Set weights for a specific attention head."""
        # This is a placeholder - implement according to your model architecture
        try:
            # Example for a typical transformer with multi-head attention
            param = getattr(module, f"{param_name}_proj.weight")
            hidden_size = param.shape[0] // module.num_attention_heads
            start_idx = head_idx * hidden_size
            end_idx = (head_idx + 1) * hidden_size
            param.data[start_idx:end_idx, :] = new_weights
        except (AttributeError, IndexError):
            pass


def create_sleep_cycle(
    model: torch.nn.Module,
    dataloader_fn: Callable[[], torch.utils.data.DataLoader],
    config: Optional[SleepCycleConfig] = None,
    device: Optional[str] = None
) -> TransformerSleepCycle:
    """
    Convenience function to create a sleep cycle manager.
    
    Args:
        model: Transformer model to manage
        dataloader_fn: Function that returns a dataloader for maintenance
        config: Sleep cycle configuration
        device: Device to run on
        
    Returns:
        Initialized TransformerSleepCycle manager
    """
    return TransformerSleepCycle(model, dataloader_fn, config, device)


if __name__ == "__main__":
    # Example usage
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    
    parser = argparse.ArgumentParser(description="Sleep Cycle for Transformer Models")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name or path")
    parser.add_argument("--active_steps", type=int, default=1000, help="Steps in active phase")
    parser.add_argument("--maintenance_steps", type=int, default=200, help="Steps in maintenance phase")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for maintenance")
    parser.add_argument("--pruning_level", type=float, default=0.1, help="Pruning level")
    parser.add_argument("--regrowth_percentage", type=float, default=0.05, help="Regrowth percentage")
    parser.add_argument("--output_dir", type=str, default="./output/sleep_cycle", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    # Create sample dataset
    class SampleTextDataset(Dataset):
        def __init__(self, tokenizer, text_samples=None, block_size=128):
            if text_samples is None:
                text_samples = [
                    "The transformer architecture has revolutionized natural language processing.",
                    "Neural networks can learn complex patterns from data.",
                    "Attention mechanisms allow models to focus on relevant parts of the input.",
                    "Language models predict the next token based on context.",
                    "Training deep learning models requires large amounts of data."
                ] * 10
            
            encodings = tokenizer(text_samples, truncation=True, padding="max_length", 
                                 max_length=block_size, return_tensors="pt")
            self.examples = encodings
            
        def __len__(self):
            return len(self.examples["input_ids"])
            
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.examples.items()}
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataset and dataloader function
    dataset = SampleTextDataset(tokenizer)
    
    def get_dataloader():
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Configure sleep cycle
    config = SleepCycleConfig(
        active_steps=args.active_steps,
        maintenance_steps=args.maintenance_steps,
        maintenance_batch_size=args.batch_size,
        pruning_level=args.pruning_level,
        regrowth_percentage=args.regrowth_percentage,
        visualization_dir=args.output_dir
    )
    
    # Create sleep cycle manager
    print("Initializing sleep cycle manager...")
    sleep_cycle = create_sleep_cycle(model, get_dataloader, config, device)
    
    # Simulate a few cycles
    print("Simulating sleep cycles...")
    num_cycles = 3
    for cycle in range(num_cycles):
        print(f"\n--- Starting Cycle {cycle+1}/{num_cycles} ---")
        
        # Simulate active phase
        print(f"Active phase: {args.active_steps} steps")
        for _ in range(args.active_steps):
            sleep_cycle.step()
            
        # Simulate maintenance phase
        print(f"Maintenance phase: {args.maintenance_steps} steps")
        for _ in range(args.maintenance_steps):
            sleep_cycle.step()
    
    print(f"\nCompleted {num_cycles} sleep cycles")
    print(f"Visualizations saved to: {args.output_dir}")