#!/usr/bin/env python
"""
Entropy Journal for Transformer Models

This module tracks attention entropy per head over time throughout
the neural plasticity cycle. It enables the study of how entropy
patterns evolve as the model adapts, learns, and reorganizes.

The entropy journal can be used to:
1. Track specialization of attention heads across time
2. Identify patterns of entropy collapse during learning
3. Observe the emergence of new functional structures
4. Study the relationship between entropy and model performance
"""

import os
import torch
import numpy as np
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Import modules for entropy calculations
from sentinel.pruning.entropy_magnitude import (
    collect_attention_distributions,
    compute_attention_entropy
)

logger = logging.getLogger(__name__)


@dataclass
class EntropyJournalConfig:
    """Configuration for entropy tracking"""
    
    # Output configuration
    output_dir: str = "./output/entropy_journal"
    experiment_name: Optional[str] = None
    
    # Data collection settings
    num_samples: int = 50  # Number of samples to use for entropy calculation
    batch_size: int = 4
    
    # Analysis settings
    track_gate_values: bool = True  # Also track gate values if available
    track_delta: bool = True  # Track entropy changes between cycles
    
    # Visualization settings
    create_visualizations: bool = True
    plot_format: str = "png"  # png, pdf, svg
    
    def __post_init__(self):
        """Set experiment name if not provided"""
        if self.experiment_name is None:
            self.experiment_name = f"entropy_journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class EntropyJournal:
    """
    Tracks and analyzes attention entropy over time.
    
    This class provides methods to track the attention entropy of each head
    across multiple cycles of neural plasticity, enabling the study of how
    attention patterns evolve and stabilize over time.
    """
    
    def __init__(
        self,
        config: Optional[EntropyJournalConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize entropy journal.
        
        Args:
            config: Configuration for entropy tracking
            device: Device to run computations on
        """
        self.config = config or EntropyJournalConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage
        self.journal = []  # List of records
        self.cycle_states = {}  # Maps cycle_idx to model state metrics
        
        logger.info(f"Initialized EntropyJournal (device: {self.device})")
        logger.info(f"Output directory: {self.output_dir}")
        
    def record_model_state(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        cycle_idx: int,
        cycle_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record entropy and gate values for a model at a specific cycle.
        
        Args:
            model: The model to analyze
            dataloader: DataLoader with samples for analysis
            cycle_idx: Index of the current cycle
            cycle_name: Optional name for the cycle
            metadata: Optional additional information to log
            
        Returns:
            Dictionary with recorded metrics
        """
        logger.info(f"Recording model state for cycle {cycle_idx}")
        
        # Set model to eval mode
        model.eval()
        
        # Collect attention distributions
        distributions = collect_attention_distributions(
            model, 
            dataloader,
            num_batches=min(self.config.num_samples // self.config.batch_size, 10),
            device=self.device
        )
        
        # Compute entropy for each layer and head
        entropy_per_layer = {}
        for layer_idx, attn_dists in distributions.items():
            entropy_per_layer[layer_idx] = compute_attention_entropy(attn_dists)
        
        # Get gate values if available and requested
        gate_values = {}
        if self.config.track_gate_values:
            try:
                for name, module in model.named_modules():
                    if hasattr(module, 'head_gates'):
                        layer_idx = int(''.join(filter(str.isdigit, name))) if any(c.isdigit() for c in name) else -1
                        if layer_idx >= 0:  # Valid layer index extracted
                            gate_values[layer_idx] = module.head_gates.detach().cpu()
            except Exception as e:
                logger.warning(f"Error collecting gate values: {str(e)}")
        
        # Create state record
        state_record = {
            "cycle_idx": cycle_idx,
            "cycle_name": cycle_name or f"cycle_{cycle_idx}",
            "timestamp": datetime.now().isoformat(),
            "entropy": {str(k): v.cpu().numpy().tolist() for k, v in entropy_per_layer.items()},
            "gate_values": {str(k): v.cpu().numpy().tolist() for k, v in gate_values.items()} if gate_values else None,
            "metadata": metadata or {}
        }
        
        # Store the record
        self.cycle_states[cycle_idx] = state_record
        
        # Save to disk
        self._save_state(state_record)
        
        # Compute delta if we have previous cycles and it's requested
        if self.config.track_delta and cycle_idx > 0 and cycle_idx - 1 in self.cycle_states:
            delta_record = self._compute_entropy_delta(cycle_idx - 1, cycle_idx)
            # Save delta separately
            self._save_delta(delta_record)
        
        # Create visualizations if requested
        if self.config.create_visualizations:
            self._visualize_cycle(cycle_idx)
        
        return state_record
    
    def _compute_entropy_delta(
        self, 
        prev_cycle_idx: int, 
        curr_cycle_idx: int
    ) -> Dict[str, Any]:
        """
        Compute entropy changes between two cycles.
        
        Args:
            prev_cycle_idx: Previous cycle index
            curr_cycle_idx: Current cycle index
            
        Returns:
            Dictionary with entropy delta information
        """
        prev_state = self.cycle_states[prev_cycle_idx]
        curr_state = self.cycle_states[curr_cycle_idx]
        
        delta_entropy = {}
        
        # Process each layer that exists in both states
        for layer_idx in prev_state["entropy"]:
            if layer_idx in curr_state["entropy"]:
                prev_entropy = np.array(prev_state["entropy"][layer_idx])
                curr_entropy = np.array(curr_state["entropy"][layer_idx])
                
                # Make sure arrays are the same length
                min_length = min(len(prev_entropy), len(curr_entropy))
                delta = curr_entropy[:min_length] - prev_entropy[:min_length]
                delta_entropy[layer_idx] = delta.tolist()
        
        # Create delta record
        delta_record = {
            "prev_cycle_idx": prev_cycle_idx,
            "curr_cycle_idx": curr_cycle_idx,
            "prev_cycle_name": prev_state["cycle_name"],
            "curr_cycle_name": curr_state["cycle_name"],
            "delta_entropy": delta_entropy,
            "timestamp": datetime.now().isoformat()
        }
        
        return delta_record
    
    def _save_state(self, state_record: Dict[str, Any]) -> None:
        """Save state record to disk"""
        cycle_idx = state_record["cycle_idx"]
        state_file = self.output_dir / f"cycle_{cycle_idx}_state.json"
        
        with open(state_file, 'w') as f:
            json.dump(state_record, f, indent=2)
            
        # Also save to main journal file (condensed format for easier analysis)
        journal_entries = []
        
        # Flatten entropy data
        for layer_idx, entropy_values in state_record["entropy"].items():
            for head_idx, entropy in enumerate(entropy_values):
                entry = {
                    "cycle_idx": cycle_idx,
                    "cycle_name": state_record["cycle_name"],
                    "layer_idx": int(layer_idx),
                    "head_idx": head_idx,
                    "entropy": entropy,
                    "gate_value": state_record["gate_values"][layer_idx][head_idx] if state_record["gate_values"] and layer_idx in state_record["gate_values"] else None,
                    "timestamp": state_record["timestamp"]
                }
                # Add metadata
                for k, v in state_record.get("metadata", {}).items():
                    entry[f"meta_{k}"] = v
                    
                journal_entries.append(entry)
        
        # Save to JSONL for easier processing
        journal_file = self.output_dir / "entropy_journal.jsonl"
        mode = 'a' if journal_file.exists() else 'w'
        
        with open(journal_file, mode) as f:
            for entry in journal_entries:
                f.write(json.dumps(entry) + '\n')
                
        logger.info(f"Saved state for cycle {cycle_idx} to {state_file}")
    
    def _save_delta(self, delta_record: Dict[str, Any]) -> None:
        """Save delta record to disk"""
        prev_cycle = delta_record["prev_cycle_idx"]
        curr_cycle = delta_record["curr_cycle_idx"]
        delta_file = self.output_dir / f"delta_{prev_cycle}_to_{curr_cycle}.json"
        
        with open(delta_file, 'w') as f:
            json.dump(delta_record, f, indent=2)
            
        # Also save to delta journal file (condensed format)
        delta_entries = []
        
        # Flatten delta data
        for layer_idx, delta_values in delta_record["delta_entropy"].items():
            for head_idx, delta in enumerate(delta_values):
                entry = {
                    "prev_cycle_idx": prev_cycle,
                    "curr_cycle_idx": curr_cycle,
                    "prev_cycle_name": delta_record["prev_cycle_name"],
                    "curr_cycle_name": delta_record["curr_cycle_name"],
                    "layer_idx": int(layer_idx),
                    "head_idx": head_idx,
                    "entropy_delta": delta,
                    "timestamp": delta_record["timestamp"]
                }
                delta_entries.append(entry)
        
        # Save to JSONL for easier processing
        delta_file = self.output_dir / "entropy_deltas.jsonl"
        mode = 'a' if delta_file.exists() else 'w'
        
        with open(delta_file, mode) as f:
            for entry in delta_entries:
                f.write(json.dumps(entry) + '\n')
                
        logger.info(f"Saved delta from cycle {prev_cycle} to {curr_cycle}")
    
    def _visualize_cycle(self, cycle_idx: int) -> None:
        """Create visualizations for a specific cycle"""
        if cycle_idx not in self.cycle_states:
            logger.warning(f"Cannot visualize cycle {cycle_idx}: not found in journal")
            return
            
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        state = self.cycle_states[cycle_idx]
        
        # Prepare entropy data for heatmap
        layers = sorted([int(k) for k in state["entropy"].keys()])
        max_heads = max(len(state["entropy"][str(layer)]) for layer in layers)
        
        entropy_array = np.zeros((len(layers), max_heads))
        entropy_array.fill(np.nan)  # Fill with NaN for missing values
        
        for i, layer in enumerate(layers):
            entropy_values = np.array(state["entropy"][str(layer)])
            entropy_array[i, :len(entropy_values)] = entropy_values
        
        # Create entropy heatmap
        plt.figure(figsize=(12, 8))
        mask = np.isnan(entropy_array)
        heatmap = plt.pcolormesh(
            np.arange(max_heads+1),
            np.arange(len(layers)+1),
            entropy_array,
            cmap='viridis',
            vmin=0.0
        )
        plt.colorbar(heatmap, label='Entropy')
        plt.title(f"Attention Entropy - {state['cycle_name']}")
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        plt.yticks(np.arange(0.5, len(layers) + 0.5), layers)
        plt.xticks(np.arange(0.5, max_heads + 0.5), np.arange(max_heads))
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(max_heads):
                if not np.isnan(entropy_array[i, j]):
                    # Choose text color based on background darkness
                    threshold = 0.5 * (plt.cm.viridis.max() + plt.cm.viridis.min())
                    normalized_value = entropy_array[i, j] / np.nanmax(entropy_array)
                    text_color = 'white' if normalized_value > threshold else 'black'
                    
                    plt.text(j + 0.5, i + 0.5, f"{entropy_array[i, j]:.2f}",
                            ha="center", va="center", color=text_color, fontsize=8)
                    
        plt.tight_layout()
        plt.savefig(viz_dir / f"entropy_heatmap_cycle_{cycle_idx}.{self.config.plot_format}")
        plt.close()
        
        # Create gate value visualization if available
        if state["gate_values"]:
            gate_array = np.zeros((len(layers), max_heads))
            gate_array.fill(np.nan)
            
            for i, layer in enumerate(layers):
                if str(layer) in state["gate_values"]:
                    gate_values = np.array(state["gate_values"][str(layer)])
                    gate_array[i, :len(gate_values)] = gate_values
            
            # Create gate heatmap
            plt.figure(figsize=(12, 8))
            mask = np.isnan(gate_array)
            heatmap = plt.pcolormesh(
                np.arange(max_heads+1),
                np.arange(len(layers)+1),
                gate_array,
                cmap='Reds',
                vmin=0.0,
                vmax=1.0
            )
            plt.colorbar(heatmap, label='Gate Value')
            plt.title(f"Attention Gate Values - {state['cycle_name']}")
            plt.xlabel('Head Index')
            plt.ylabel('Layer Index')
            plt.yticks(np.arange(0.5, len(layers) + 0.5), layers)
            plt.xticks(np.arange(0.5, max_heads + 0.5), np.arange(max_heads))
            
            # Add text annotations
            for i in range(len(layers)):
                for j in range(max_heads):
                    if not np.isnan(gate_array[i, j]):
                        # Choose text color based on background darkness
                        text_color = 'white' if gate_array[i, j] > 0.5 else 'black'
                        plt.text(j + 0.5, i + 0.5, f"{gate_array[i, j]:.2f}",
                                ha="center", va="center", color=text_color, fontsize=8)
                        
            plt.tight_layout()
            plt.savefig(viz_dir / f"gate_values_cycle_{cycle_idx}.{self.config.plot_format}")
            plt.close()
            
        logger.info(f"Created visualizations for cycle {cycle_idx}")
    
    def visualize_entropy_evolution(self) -> None:
        """
        Create visualization showing entropy evolution across cycles.
        
        This creates a time-series view of entropy changes for each head.
        """
        if not self.cycle_states:
            logger.warning("Cannot visualize entropy evolution: no data in journal")
            return
            
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert journal to DataFrame for easier processing
        journal_file = self.output_dir / "entropy_journal.jsonl"
        if not journal_file.exists():
            logger.warning(f"Cannot visualize entropy evolution: {journal_file} not found")
            return
            
        data = []
        with open(journal_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        # Create a visualization for each layer
        for layer in df['layer_idx'].unique():
            layer_df = df[df['layer_idx'] == layer]
            
            plt.figure(figsize=(12, 8))
            
            # Plot entropy for each head over time
            for head in layer_df['head_idx'].unique():
                head_df = layer_df[layer_df['head_idx'] == head]
                head_df = head_df.sort_values('cycle_idx')
                
                plt.plot(
                    head_df['cycle_idx'], 
                    head_df['entropy'],
                    'o-',
                    label=f"Head {head}"
                )
            
            plt.title(f"Entropy Evolution - Layer {layer}")
            plt.xlabel('Cycle')
            plt.ylabel('Entropy')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Show legend only if not too many heads
            if len(layer_df['head_idx'].unique()) <= 12:
                plt.legend()
                
            plt.tight_layout()
            plt.savefig(viz_dir / f"entropy_evolution_layer_{layer}.{self.config.plot_format}")
            plt.close()
        
        logger.info(f"Created entropy evolution visualizations in {viz_dir}")
    
    def visualize_gate_evolution(self) -> None:
        """
        Create visualization showing gate value evolution across cycles.
        
        This creates a time-series view of gate value changes for each head.
        """
        if not self.cycle_states:
            logger.warning("Cannot visualize gate evolution: no data in journal")
            return
            
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert journal to DataFrame for easier processing
        journal_file = self.output_dir / "entropy_journal.jsonl"
        if not journal_file.exists():
            logger.warning(f"Cannot visualize gate evolution: {journal_file} not found")
            return
            
        data = []
        with open(journal_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Only include entries with gate values
                if entry.get('gate_value') is not None:
                    data.append(entry)
        
        if not data:
            logger.warning("No gate values found in journal")
            return
            
        df = pd.DataFrame(data)
        
        # Create a visualization for each layer
        for layer in df['layer_idx'].unique():
            layer_df = df[df['layer_idx'] == layer]
            
            plt.figure(figsize=(12, 8))
            
            # Plot gate values for each head over time
            for head in layer_df['head_idx'].unique():
                head_df = layer_df[layer_df['head_idx'] == head]
                head_df = head_df.sort_values('cycle_idx')
                
                plt.plot(
                    head_df['cycle_idx'], 
                    head_df['gate_value'],
                    'o-',
                    label=f"Head {head}"
                )
            
            plt.title(f"Gate Value Evolution - Layer {layer}")
            plt.xlabel('Cycle')
            plt.ylabel('Gate Value')
            plt.ylim(0, 1.05)  # Gate values are between 0 and 1
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Show legend only if not too many heads
            if len(layer_df['head_idx'].unique()) <= 12:
                plt.legend()
                
            plt.tight_layout()
            plt.savefig(viz_dir / f"gate_evolution_layer_{layer}.{self.config.plot_format}")
            plt.close()
        
        logger.info(f"Created gate evolution visualizations in {viz_dir}")
    
    def create_summary_report(self) -> None:
        """
        Create a summary report of the entropy journal.
        
        This includes key metrics and findings from the entropy tracking.
        """
        if not self.cycle_states:
            logger.warning("Cannot create summary report: no data in journal")
            return
        
        report_file = self.output_dir / "entropy_journal_summary.md"
        
        # Load journal data
        journal_file = self.output_dir / "entropy_journal.jsonl"
        if not journal_file.exists():
            logger.warning(f"Cannot create summary report: {journal_file} not found")
            return
            
        data = []
        with open(journal_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        # Calculate statistics
        cycles = sorted(df['cycle_idx'].unique())
        num_cycles = len(cycles)
        
        # Create report
        with open(report_file, 'w') as f:
            f.write(f"# Entropy Journal Summary\n\n")
            f.write(f"Experiment: {self.config.experiment_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Overview\n\n")
            f.write(f"- Total cycles: {num_cycles}\n")
            f.write(f"- Layers analyzed: {len(df['layer_idx'].unique())}\n")
            f.write(f"- Heads per layer: {len(df[df['layer_idx'] == df['layer_idx'].iloc[0]]['head_idx'].unique())}\n\n")
            
            # Calculate entropy statistics by cycle
            f.write(f"## Entropy Statistics by Cycle\n\n")
            f.write(f"| Cycle | Avg Entropy | Min Entropy | Max Entropy | StdDev |\n")
            f.write(f"|-------|------------|------------|------------|--------|\n")
            
            for cycle in cycles:
                cycle_df = df[df['cycle_idx'] == cycle]
                avg = cycle_df['entropy'].mean()
                min_val = cycle_df['entropy'].min()
                max_val = cycle_df['entropy'].max()
                std = cycle_df['entropy'].std()
                
                f.write(f"| {cycle} | {avg:.4f} | {min_val:.4f} | {max_val:.4f} | {std:.4f} |\n")
            
            f.write(f"\n## Entropy Change Analysis\n\n")
            
            # Calculate overall entropy trend
            if len(cycles) >= 2:
                first_cycle = cycles[0]
                last_cycle = cycles[-1]
                
                first_avg = df[df['cycle_idx'] == first_cycle]['entropy'].mean()
                last_avg = df[df['cycle_idx'] == last_cycle]['entropy'].mean()
                
                change = last_avg - first_avg
                percent_change = (change / first_avg) * 100 if first_avg != 0 else float('inf')
                
                f.write(f"- Overall entropy change: {change:.4f} ({percent_change:+.2f}%)\n")
                
                if change < 0:
                    f.write(f"- Overall entropy **decreased** across cycles, indicating increased attention focus\n")
                else:
                    f.write(f"- Overall entropy **increased** across cycles, indicating more distributed attention\n")
            
            # Add information about gate values if available
            if 'gate_value' in df.columns and not df['gate_value'].isna().all():
                f.write(f"\n## Gate Activity Analysis\n\n")
                
                # Calculate how many heads are active (gate > 0.5) per cycle
                f.write(f"| Cycle | Active Heads | % Active |\n")
                f.write(f"|-------|-------------|----------|\n")
                
                for cycle in cycles:
                    cycle_df = df[(df['cycle_idx'] == cycle) & (df['gate_value'].notna())]
                    if len(cycle_df) > 0:
                        active = (cycle_df['gate_value'] > 0.5).sum()
                        total = len(cycle_df)
                        percent = (active / total) * 100
                        
                        f.write(f"| {cycle} | {active}/{total} | {percent:.1f}% |\n")
            
            f.write(f"\n## Key Observations\n\n")
            
            # Add some automatic observations
            if len(cycles) >= 2:
                # Find heads with largest entropy change
                delta_df = pd.DataFrame()
                delta_df['layer'] = df[df['cycle_idx'] == last_cycle]['layer_idx'].values
                delta_df['head'] = df[df['cycle_idx'] == last_cycle]['head_idx'].values
                delta_df['first_entropy'] = df[df['cycle_idx'] == first_cycle].sort_values(['layer_idx', 'head_idx'])['entropy'].values
                delta_df['last_entropy'] = df[df['cycle_idx'] == last_cycle].sort_values(['layer_idx', 'head_idx'])['entropy'].values
                delta_df['delta'] = delta_df['last_entropy'] - delta_df['first_entropy']
                
                # Top 5 decreasing entropy
                top_dec = delta_df.sort_values('delta').head(5)
                f.write(f"### Top 5 Heads with Decreasing Entropy\n\n")
                f.write(f"| Layer | Head | Initial | Final | Change |\n")
                f.write(f"|-------|------|---------|-------|--------|\n")
                
                for _, row in top_dec.iterrows():
                    f.write(f"| {int(row['layer'])} | {int(row['head'])} | {row['first_entropy']:.4f} | {row['last_entropy']:.4f} | {row['delta']:.4f} |\n")
                
                # Top 5 increasing entropy
                top_inc = delta_df.sort_values('delta', ascending=False).head(5)
                f.write(f"\n### Top 5 Heads with Increasing Entropy\n\n")
                f.write(f"| Layer | Head | Initial | Final | Change |\n")
                f.write(f"|-------|------|---------|-------|--------|\n")
                
                for _, row in top_inc.iterrows():
                    f.write(f"| {int(row['layer'])} | {int(row['head'])} | {row['first_entropy']:.4f} | {row['last_entropy']:.4f} | {row['delta']:.4f} |\n")
            
            f.write(f"\n## Visualizations\n\n")
            f.write(f"- Entropy heatmaps for each cycle are available in the `visualizations` directory\n")
            f.write(f"- Evolution plots showing how entropy changes over time are also available\n")
            
            f.write(f"\n## Conclusions\n\n")
            f.write(f"*Add your interpretation of these results here.*\n")
        
        logger.info(f"Created summary report at {report_file}")


def record_entropy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cycle_idx: int,
    output_path: str,
    device: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to record entropy for a model at a given cycle.
    
    Args:
        model: The model to analyze
        dataloader: DataLoader with samples for analysis
        cycle_idx: Index of the current cycle
        output_path: Directory to save results
        device: Device to run computations on
        metadata: Optional additional information to log
        
    Returns:
        Dictionary with recorded metrics
    """
    config = EntropyJournalConfig(output_dir=output_path)
    journal = EntropyJournal(config, device)
    return journal.record_model_state(model, dataloader, cycle_idx, metadata=metadata)


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Entropy Journal for Transformer Models")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./output/entropy_journal", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for entropy calculation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load model and tokenizer
    logger.info(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Create a simple dataset for demonstration
    texts = [
        "The neural network model processes data through multiple layers of computation.",
        "Artificial intelligence systems can learn from experience and improve over time.",
        "The transformer architecture revolutionized natural language processing tasks.",
        "Self-attention mechanisms enable models to focus on relevant parts of the input.",
        "Neural plasticity allows models to adapt their structure during training."
    ] * 10  # Repeat to get more samples
    
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.encodings = tokenizer(texts, padding=True, truncation=True, 
                                      max_length=max_length, return_tensors="pt")
            
        def __len__(self):
            return len(self.encodings.input_ids)
            
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}
    
    # Create dataset and dataloader
    dataset = SimpleDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create entropy journal configuration
    config = EntropyJournalConfig(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        create_visualizations=True
    )
    
    # Initialize entropy journal
    journal = EntropyJournal(config, device)
    
    # Record initial state
    logger.info("Recording initial state")
    journal.record_model_state(model, dataloader, cycle_idx=0, cycle_name="Initial")
    
    # Simulate some changes to the model (e.g., prune some heads)
    logger.info("Simulating model changes")
    # This is just for demonstration - in a real scenario, you would apply actual changes
    
    # Record state after changes
    logger.info("Recording state after changes")
    journal.record_model_state(model, dataloader, cycle_idx=1, cycle_name="After Changes")
    
    # Create summary visualizations
    logger.info("Creating summary visualizations")
    journal.visualize_entropy_evolution()
    journal.visualize_gate_evolution()
    
    # Create summary report
    logger.info("Creating summary report")
    journal.create_summary_report()
    
    logger.info(f"Complete! Results saved to {args.output_dir}")