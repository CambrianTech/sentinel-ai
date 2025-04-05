#!/usr/bin/env python
"""
Function Tracking for Transformer Models

This module implements tools to track functional preservation across structural changes
in transformer models. It can measure how well function is preserved despite changes
to the underlying architecture during neural plasticity cycles.

Key applications:
1. Measuring functional similarity between models at different stages of plasticity
2. Tracking how outputs evolve despite structural reorganization
3. Testing for functional identity persistence despite architectural changes
4. Evaluating how pruning and regrowth affect specific capabilities
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
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn.functional as F
# We'll implement our own cosine similarity function instead of depending on sklearn

logger = logging.getLogger(__name__)


@dataclass
class FunctionTrackingConfig:
    """Configuration for function tracking"""
    
    # Output configuration
    output_dir: str = "./output/function_tracking"
    experiment_name: Optional[str] = None
    
    # Tracking methods
    track_embeddings: bool = True  # Track intermediate layer embeddings
    track_attention: bool = True   # Track attention patterns
    track_output_logits: bool = True  # Track final output logits
    
    # Probe configuration
    num_probes: int = 10  # Number of probing points in the model
    probe_methods: List[str] = None  # Methods to use for probing
    
    # Test prompt configuration
    num_prompt_tokens: int = 50  # Maximum tokens per prompt
    
    # Visualization settings
    create_visualizations: bool = True
    
    def __post_init__(self):
        """Set default probe methods and experiment name"""
        if self.probe_methods is None:
            self.probe_methods = ["cosine", "kl_divergence"]
            
        if self.experiment_name is None:
            self.experiment_name = f"function_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class ModelProbe:
    """
    A probe for capturing internal model states.
    
    This class allows capturing intermediate activations from a model
    to compare functional behavior across model versions.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize a model probe.
        
        Args:
            model: The model to probe
        """
        self.model = model
        self.hooks = []
        self.activations = {}
        self.attention_patterns = {}
        
    def attach_embedding_probes(self):
        """Attach hooks to capture embeddings from transformer blocks"""
        for name, module in self.model.named_modules():
            # Look for output of transformer blocks (after attention and FFN)
            if any(x in name.lower() for x in ['block', 'layer']) and 'output' in name.lower():
                # Extract layer index if available
                try:
                    layer_idx = int(''.join(filter(str.isdigit, name)))
                except ValueError:
                    layer_idx = len(self.hooks)  # Use hook count as fallback
                
                # Create hook function for this layer
                def get_hook_fn(layer_id):
                    def hook_fn(module, input, output):
                        self.activations[f"layer_{layer_id}"] = output.detach()
                    return hook_fn
                
                # Register the hook
                hook = module.register_forward_hook(get_hook_fn(layer_idx))
                self.hooks.append(hook)
                logger.debug(f"Attached embedding probe to: {name} (layer {layer_idx})")
    
    def attach_attention_probes(self):
        """Attach hooks to capture attention patterns"""
        for name, module in self.model.named_modules():
            # Look for attention modules
            if 'attention' in name.lower() and not any(x in name.lower() for x in ['output', 'dropout', 'layer_norm']):
                # Extract layer index if available
                try:
                    layer_idx = int(''.join(filter(str.isdigit, name)))
                except ValueError:
                    layer_idx = len(self.hooks)  # Use hook count as fallback
                
                # Create hook function for this layer's attention
                def get_hook_fn(layer_id):
                    def hook_fn(module, input, output):
                        # Handle different output formats from different model types
                        # Most return (output, attention_weights) or just output
                        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                            self.attention_patterns[f"layer_{layer_id}"] = output[1].detach()
                    return hook_fn
                
                # Register the hook
                hook = module.register_forward_hook(get_hook_fn(layer_idx))
                self.hooks.append(hook)
                logger.debug(f"Attached attention probe to: {name} (layer {layer_idx})")
    
    def remove_hooks(self):
        """Remove all attached hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("Removed all hooks")
    
    def clear_activations(self):
        """Clear stored activations"""
        self.activations = {}
        self.attention_patterns = {}
    
    def __del__(self):
        """Ensure hooks are removed when the object is deleted"""
        self.remove_hooks()


class FunctionTracker:
    """
    Tracks function preservation across model versions.
    
    This class compares how different model versions respond to the
    same inputs, tracking similarity in internal representations and outputs.
    """
    
    def __init__(
        self,
        config: Optional[FunctionTrackingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize function tracker.
        
        Args:
            config: Configuration for function tracking
            device: Device to run computations on
        """
        self.config = config or FunctionTrackingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage
        self.tracking_results = {}
        
        logger.info(f"Initialized FunctionTracker (device: {self.device})")
        logger.info(f"Output directory: {self.output_dir}")
    
    def track_function(
        self,
        model_before: torch.nn.Module,
        model_after: torch.nn.Module,
        prompts: List[str],
        tokenizer,
        cycle_idx: Optional[int] = None,
        cycle_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track function preservation between two model versions.
        
        Args:
            model_before: Model before changes
            model_after: Model after changes
            prompts: List of text prompts for testing
            tokenizer: Tokenizer for processing prompts
            cycle_idx: Optional cycle index for tracking
            cycle_name: Optional name for the cycle
            
        Returns:
            Dictionary with tracking results
        """
        logger.info(f"Tracking function preservation between model versions")
        
        # Set models to eval mode
        model_before.eval()
        model_after.eval()
        
        # Create probes
        before_probe = ModelProbe(model_before)
        after_probe = ModelProbe(model_after)
        
        # Attach probes
        if self.config.track_embeddings:
            before_probe.attach_embedding_probes()
            after_probe.attach_embedding_probes()
            
        if self.config.track_attention:
            before_probe.attach_attention_probes()
            after_probe.attach_attention_probes()
        
        # Process prompts
        results = {
            "cycle_idx": cycle_idx,
            "cycle_name": cycle_name or f"cycle_{cycle_idx}" if cycle_idx is not None else "unknown",
            "timestamp": datetime.now().isoformat(),
            "embedding_similarity": {},
            "attention_similarity": {},
            "output_similarity": {},
            "prompts": prompts,
            "summary": {}
        }
        
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
                # Tokenize prompt
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=self.config.num_prompt_tokens,
                    truncation=True
                ).to(self.device)
                
                # Clear previous activations
                before_probe.clear_activations()
                after_probe.clear_activations()
                
                # Forward pass through both models
                outputs_before = model_before(**inputs)
                outputs_after = model_after(**inputs)
                
                # Compare output logits
                if self.config.track_output_logits:
                    logits_before = outputs_before.logits
                    logits_after = outputs_after.logits
                    
                    # Compute cosine similarity between output distributions
                    cos_sim = self._compute_cosine_similarity(
                        logits_before.view(logits_before.size(0), -1),
                        logits_after.view(logits_after.size(0), -1)
                    )
                    
                    # Compute KL divergence between output distributions
                    kl_div = self._compute_kl_divergence(
                        F.log_softmax(logits_before, dim=-1),
                        F.softmax(logits_after, dim=-1)
                    )
                    
                    results["output_similarity"][f"prompt_{prompt_idx}"] = {
                        "cosine_similarity": cos_sim.item(),
                        "kl_divergence": kl_div.item()
                    }
                
                # Compare embeddings
                if self.config.track_embeddings:
                    for layer_key in set(before_probe.activations.keys()) & set(after_probe.activations.keys()):
                        embed_before = before_probe.activations[layer_key]
                        embed_after = after_probe.activations[layer_key]
                        
                        # Ensure embeddings have the same shape for comparison
                        min_len = min(embed_before.size(1), embed_after.size(1))
                        embed_before = embed_before[:, :min_len, :]
                        embed_after = embed_after[:, :min_len, :]
                        
                        # Compute similarity
                        cos_sim = self._compute_cosine_similarity(
                            embed_before.view(embed_before.size(0), -1),
                            embed_after.view(embed_after.size(0), -1)
                        )
                        
                        # Store result
                        if layer_key not in results["embedding_similarity"]:
                            results["embedding_similarity"][layer_key] = {}
                            
                        results["embedding_similarity"][layer_key][f"prompt_{prompt_idx}"] = {
                            "cosine_similarity": cos_sim.item()
                        }
                
                # Compare attention patterns
                if self.config.track_attention:
                    for layer_key in set(before_probe.attention_patterns.keys()) & set(after_probe.attention_patterns.keys()):
                        attn_before = before_probe.attention_patterns[layer_key]
                        attn_after = after_probe.attention_patterns[layer_key]
                        
                        # Ensure attention patterns have compatible shapes
                        if attn_before.shape == attn_after.shape:
                            # Compute similarity
                            cos_sim = self._compute_cosine_similarity(
                                attn_before.view(attn_before.size(0), -1),
                                attn_after.view(attn_after.size(0), -1)
                            )
                            
                            # Store result
                            if layer_key not in results["attention_similarity"]:
                                results["attention_similarity"][layer_key] = {}
                                
                            results["attention_similarity"][layer_key][f"prompt_{prompt_idx}"] = {
                                "cosine_similarity": cos_sim.item()
                            }
        
        # Remove hooks
        before_probe.remove_hooks()
        after_probe.remove_hooks()
        
        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results)
        
        # Store results
        self.tracking_results[results["cycle_name"]] = results
        
        # Save results
        self._save_results(results)
        
        # Create visualizations if enabled
        if self.config.create_visualizations:
            self._create_visualizations(results)
        
        return results
    
    def _compute_cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two tensors"""
        return F.cosine_similarity(tensor1, tensor2, dim=1).mean()
    
    def _compute_kl_divergence(self, log_probs1: torch.Tensor, probs2: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between two distributions"""
        return F.kl_div(log_probs1, probs2, reduction='batchmean')
    
    def _calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from tracking results"""
        summary = {}
        
        # Output similarity summary
        if results["output_similarity"]:
            cos_sims = [data["cosine_similarity"] for data in results["output_similarity"].values()]
            kl_divs = [data["kl_divergence"] for data in results["output_similarity"].values()]
            
            summary["output"] = {
                "avg_cosine_similarity": np.mean(cos_sims),
                "std_cosine_similarity": np.std(cos_sims),
                "avg_kl_divergence": np.mean(kl_divs),
                "std_kl_divergence": np.std(kl_divs)
            }
        
        # Embedding similarity summary
        if results["embedding_similarity"]:
            summary["embedding"] = {}
            
            for layer_key, prompts in results["embedding_similarity"].items():
                cos_sims = [data["cosine_similarity"] for data in prompts.values()]
                
                summary["embedding"][layer_key] = {
                    "avg_cosine_similarity": np.mean(cos_sims),
                    "std_cosine_similarity": np.std(cos_sims)
                }
        
        # Attention similarity summary
        if results["attention_similarity"]:
            summary["attention"] = {}
            
            for layer_key, prompts in results["attention_similarity"].items():
                cos_sims = [data["cosine_similarity"] for data in prompts.values()]
                
                summary["attention"][layer_key] = {
                    "avg_cosine_similarity": np.mean(cos_sims),
                    "std_cosine_similarity": np.std(cos_sims)
                }
        
        # Overall function preservation score (simple weighted average)
        scores = []
        
        # Output contributes 50%
        if "output" in summary:
            scores.append(summary["output"]["avg_cosine_similarity"] * 0.5)
        
        # Embedding contributes 25%
        if "embedding" in summary:
            avg_embed_sim = np.mean([layer["avg_cosine_similarity"] for layer in summary["embedding"].values()])
            scores.append(avg_embed_sim * 0.25)
        
        # Attention contributes 25%
        if "attention" in summary:
            avg_attn_sim = np.mean([layer["avg_cosine_similarity"] for layer in summary["attention"].values()])
            scores.append(avg_attn_sim * 0.25)
        
        if scores:
            summary["overall_preservation_score"] = sum(scores) / sum([0.5 if "output" in summary else 0,
                                                                    0.25 if "embedding" in summary else 0,
                                                                    0.25 if "attention" in summary else 0])
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save tracking results to disk"""
        cycle_name = results["cycle_name"]
        results_file = self.output_dir / f"{cycle_name}_tracking.json"
        
        # Convert any tensors to values for JSON serialization
        def process_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = process_dict(value)
                elif isinstance(value, (torch.Tensor, np.ndarray)):
                    d[key] = value.item() if value.size == 1 else value.tolist()
            return d
        
        processed_results = process_dict(results.copy())
        
        with open(results_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
            
        logger.info(f"Saved function tracking results to {results_file}")
    
    def _create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create visualizations from tracking results"""
        cycle_name = results["cycle_name"]
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create visualization for output similarity
        if results["output_similarity"]:
            plt.figure(figsize=(10, 6))
            
            prompts = list(results["output_similarity"].keys())
            cos_sims = [results["output_similarity"][p]["cosine_similarity"] for p in prompts]
            
            plt.bar(range(len(prompts)), cos_sims)
            plt.xlabel("Prompt Index")
            plt.ylabel("Cosine Similarity")
            plt.title(f"Output Similarity - {cycle_name}")
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add average line
            avg_sim = np.mean(cos_sims)
            plt.axhline(y=avg_sim, color='r', linestyle='-', label=f"Avg: {avg_sim:.4f}")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{cycle_name}_output_similarity.png")
            plt.close()
        
        # Create visualization for embedding similarity across layers
        if results["embedding_similarity"]:
            layers = sorted(results["embedding_similarity"].keys())
            
            plt.figure(figsize=(12, 8))
            
            for i, layer in enumerate(layers):
                prompts = list(results["embedding_similarity"][layer].keys())
                cos_sims = [results["embedding_similarity"][layer][p]["cosine_similarity"] for p in prompts]
                avg_sim = np.mean(cos_sims)
                
                plt.bar(
                    [i], 
                    [avg_sim], 
                    yerr=[np.std(cos_sims)],
                    label=layer if i == 0 else None  # Only add label for first bar
                )
            
            plt.xlabel("Layer")
            plt.ylabel("Average Cosine Similarity")
            plt.title(f"Embedding Similarity Across Layers - {cycle_name}")
            plt.xticks(range(len(layers)), [l.replace('layer_', '') for l in layers])
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{cycle_name}_embedding_similarity.png")
            plt.close()
        
        # Create visualization for attention similarity across layers
        if results["attention_similarity"]:
            layers = sorted(results["attention_similarity"].keys())
            
            plt.figure(figsize=(12, 8))
            
            for i, layer in enumerate(layers):
                prompts = list(results["attention_similarity"][layer].keys())
                cos_sims = [results["attention_similarity"][layer][p]["cosine_similarity"] for p in prompts]
                avg_sim = np.mean(cos_sims)
                
                plt.bar(
                    [i], 
                    [avg_sim], 
                    yerr=[np.std(cos_sims)],
                    label=layer if i == 0 else None  # Only add label for first bar
                )
            
            plt.xlabel("Layer")
            plt.ylabel("Average Cosine Similarity")
            plt.title(f"Attention Pattern Similarity Across Layers - {cycle_name}")
            plt.xticks(range(len(layers)), [l.replace('layer_', '') for l in layers])
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{cycle_name}_attention_similarity.png")
            plt.close()
        
        # Create summary visualization
        if "overall_preservation_score" in results["summary"]:
            plt.figure(figsize=(10, 6))
            
            # Create categories and scores
            categories = []
            scores = []
            
            # Overall score
            categories.append("Overall")
            scores.append(results["summary"]["overall_preservation_score"])
            
            # Output score if available
            if "output" in results["summary"]:
                categories.append("Output")
                scores.append(results["summary"]["output"]["avg_cosine_similarity"])
            
            # Embedding score if available
            if "embedding" in results["summary"]:
                categories.append("Embedding")
                avg_embed_sim = np.mean([layer["avg_cosine_similarity"] for layer in results["summary"]["embedding"].values()])
                scores.append(avg_embed_sim)
            
            # Attention score if available
            if "attention" in results["summary"]:
                categories.append("Attention")
                avg_attn_sim = np.mean([layer["avg_cosine_similarity"] for layer in results["summary"]["attention"].values()])
                scores.append(avg_attn_sim)
            
            # Create bar chart
            plt.bar(range(len(categories)), scores)
            plt.xlabel("Category")
            plt.ylabel("Similarity Score")
            plt.title(f"Function Preservation Summary - {cycle_name}")
            plt.xticks(range(len(categories)), categories)
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add score values
            for i, score in enumerate(scores):
                plt.text(i, score + 0.02, f"{score:.4f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{cycle_name}_function_preservation_summary.png")
            plt.close()
            
        logger.info(f"Created visualizations for {cycle_name} in {viz_dir}")
    
    def create_summary_report(self) -> None:
        """Create a summary report of function preservation across cycles"""
        if not self.tracking_results:
            logger.warning("Cannot create summary report: no tracking results")
            return
            
        report_file = self.output_dir / "function_tracking_summary.md"
        
        # Create cycles list
        cycles = sorted(
            self.tracking_results.keys(), 
            key=lambda x: self.tracking_results[x].get("cycle_idx", float('inf'))
        )
        
        with open(report_file, 'w') as f:
            f.write(f"# Function Tracking Summary\n\n")
            f.write(f"Experiment: {self.config.experiment_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Overview\n\n")
            f.write(f"- Total cycles analyzed: {len(cycles)}\n")
            
            # Overall scores by cycle
            f.write(f"\n## Function Preservation Scores\n\n")
            f.write(f"| Cycle | Overall Score | Output | Embedding | Attention |\n")
            f.write(f"|-------|--------------|--------|-----------|----------|\n")
            
            for cycle in cycles:
                results = self.tracking_results[cycle]
                summary = results["summary"]
                
                overall = summary.get("overall_preservation_score", "N/A")
                if not isinstance(overall, str):
                    overall = f"{overall:.4f}"
                
                output = summary.get("output", {}).get("avg_cosine_similarity", "N/A")
                if not isinstance(output, str):
                    output = f"{output:.4f}"
                
                if "embedding" in summary:
                    embedding = np.mean([layer["avg_cosine_similarity"] for layer in summary["embedding"].values()])
                    embedding = f"{embedding:.4f}"
                else:
                    embedding = "N/A"
                
                if "attention" in summary:
                    attention = np.mean([layer["avg_cosine_similarity"] for layer in summary["attention"].values()])
                    attention = f"{attention:.4f}"
                else:
                    attention = "N/A"
                
                f.write(f"| {cycle} | {overall} | {output} | {embedding} | {attention} |\n")
            
            # Layer-specific analysis if available
            if any("embedding" in self.tracking_results[cycle]["summary"] for cycle in cycles):
                f.write(f"\n## Embedding Similarity by Layer\n\n")
                
                # Get all unique layers across cycles
                all_layers = set()
                for cycle in cycles:
                    if "embedding" in self.tracking_results[cycle]["summary"]:
                        all_layers.update(self.tracking_results[cycle]["summary"]["embedding"].keys())
                
                all_layers = sorted(all_layers)
                
                # Create table header
                f.write(f"| Layer | " + " | ".join(cycles) + " |\n")
                f.write(f"|-------|" + "|".join(["-------" for _ in cycles]) + "|\n")
                
                # Fill table with data
                for layer in all_layers:
                    row = [layer.replace('layer_', '')]
                    
                    for cycle in cycles:
                        if ("embedding" in self.tracking_results[cycle]["summary"] and 
                            layer in self.tracking_results[cycle]["summary"]["embedding"]):
                            sim = self.tracking_results[cycle]["summary"]["embedding"][layer]["avg_cosine_similarity"]
                            row.append(f"{sim:.4f}")
                        else:
                            row.append("N/A")
                    
                    f.write(f"| {' | '.join(row)} |\n")
            
            # Interpretation section
            f.write(f"\n## Interpretation\n\n")
            
            if len(cycles) >= 2:
                # Get first and last cycle overall scores
                first_cycle = cycles[0]
                last_cycle = cycles[-1]
                
                first_score = self.tracking_results[first_cycle]["summary"].get("overall_preservation_score", None)
                last_score = self.tracking_results[last_cycle]["summary"].get("overall_preservation_score", None)
                
                if first_score is not None and last_score is not None:
                    if last_score > 0.9:
                        f.write("- **High function preservation**: The model has maintained over 90% functional similarity despite structural changes.\n")
                    elif last_score > 0.7:
                        f.write("- **Moderate function preservation**: The model has maintained good functional similarity, with some divergence in behavior.\n")
                    else:
                        f.write("- **Low function preservation**: Structural changes have resulted in significant behavioral differences.\n")
                    
                    if last_score > first_score:
                        f.write("- Function preservation has **improved** over time, suggesting beneficial adaptation.\n")
                    elif last_score < first_score:
                        f.write("- Function preservation has **declined** over time, suggesting potential drift in capabilities.\n")
                    else:
                        f.write("- Function preservation has remained **stable** over time.\n")
            
            f.write("\n## Conclusions\n\n")
            f.write("*Add your interpretation of these results here.*\n")
        
        logger.info(f"Created summary report at {report_file}")
        
    def compare_multiple_cycles(self, reference_cycle: str) -> Dict[str, Any]:
        """
        Compare function preservation across multiple cycles against a reference.
        
        Args:
            reference_cycle: Name of the reference cycle
            
        Returns:
            Dictionary with comparison results
        """
        if reference_cycle not in self.tracking_results:
            logger.warning(f"Reference cycle {reference_cycle} not found in tracking results")
            return {}
            
        if len(self.tracking_results) < 2:
            logger.warning("Need at least two cycles for comparison")
            return {}
        
        # Get all cycles except reference
        other_cycles = [c for c in self.tracking_results.keys() if c != reference_cycle]
        
        # Create comparison data
        comparison = {
            "reference_cycle": reference_cycle,
            "compared_cycles": other_cycles,
            "overall_scores": {},
            "output_scores": {},
            "embedding_scores": {},
            "attention_scores": {}
        }
        
        # Extract scores
        for cycle in other_cycles:
            summary = self.tracking_results[cycle]["summary"]
            
            # Overall score
            if "overall_preservation_score" in summary:
                comparison["overall_scores"][cycle] = summary["overall_preservation_score"]
            
            # Output score
            if "output" in summary:
                comparison["output_scores"][cycle] = summary["output"]["avg_cosine_similarity"]
            
            # Embedding score (average across layers)
            if "embedding" in summary:
                avg_embed_sim = np.mean([layer["avg_cosine_similarity"] for layer in summary["embedding"].values()])
                comparison["embedding_scores"][cycle] = avg_embed_sim
            
            # Attention score (average across layers)
            if "attention" in summary:
                avg_attn_sim = np.mean([layer["avg_cosine_similarity"] for layer in summary["attention"].values()])
                comparison["attention_scores"][cycle] = avg_attn_sim
        
        # Create visualization
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        if comparison["overall_scores"]:
            plt.figure(figsize=(12, 8))
            
            cycles = sorted(comparison["overall_scores"].keys())
            scores = [comparison["overall_scores"][c] for c in cycles]
            
            plt.bar(range(len(cycles)), scores)
            plt.xlabel("Cycle")
            plt.ylabel("Overall Function Preservation Score")
            plt.title(f"Function Preservation vs. Reference ({reference_cycle})")
            plt.xticks(range(len(cycles)), cycles, rotation=45)
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add score values
            for i, score in enumerate(scores):
                plt.text(i, score + 0.02, f"{score:.4f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"comparison_vs_{reference_cycle}.png")
            plt.close()
        
        return comparison


def track_function(
    model_before: torch.nn.Module,
    model_after: torch.nn.Module,
    prompts: List[str],
    tokenizer,
    cycle_idx: Optional[int] = None,
    cycle_name: Optional[str] = None,
    output_dir: str = "./output/function_tracking",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to track function preservation between model versions.
    
    Args:
        model_before: Model before changes
        model_after: Model after changes
        prompts: List of text prompts for testing
        tokenizer: Tokenizer for processing prompts
        cycle_idx: Optional cycle index for tracking
        cycle_name: Optional name for the cycle
        output_dir: Directory to save results
        device: Device to run computations on
        
    Returns:
        Dictionary with tracking results
    """
    config = FunctionTrackingConfig(output_dir=output_dir)
    tracker = FunctionTracker(config, device)
    return tracker.track_function(model_before, model_after, prompts, tokenizer, cycle_idx, cycle_name)


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Function Tracking for Transformer Models")
    parser.add_argument("--model_before", type=str, required=True, help="Path to model before changes")
    parser.add_argument("--model_after", type=str, required=True, help="Path to model after changes")
    parser.add_argument("--output_dir", type=str, default="./output/function_tracking", help="Output directory")
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of test prompts")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load models and tokenizer
    logger.info(f"Loading models and tokenizer")
    
    # Load first model
    model_before = AutoModelForCausalLM.from_pretrained(args.model_before)
    
    # Use same tokenizer for both models
    tokenizer = AutoTokenizer.from_pretrained(args.model_before)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load second model
    model_after = AutoModelForCausalLM.from_pretrained(args.model_after)
    
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_before = model_before.to(device)
    model_after = model_after.to(device)
    
    # Create test prompts
    test_prompts = [
        "The transformer architecture allows models to process",
        "Artificial intelligence systems can learn from",
        "Neural networks consist of layers of",
        "Language models predict the next token based on",
        "Attention mechanisms enable the model to focus on",
        "The fine-tuning process adapts pre-trained models to",
        "Model compression techniques reduce the size of",
        "Natural language processing has evolved with",
        "Deep learning algorithms require large amounts of",
        "Transfer learning leverages knowledge from one task to"
    ]
    
    if args.num_prompts < len(test_prompts):
        test_prompts = test_prompts[:args.num_prompts]
    
    # Create tracker and run comparison
    config = FunctionTrackingConfig(output_dir=args.output_dir)
    tracker = FunctionTracker(config, device)
    
    logger.info("Tracking function preservation")
    results = tracker.track_function(
        model_before,
        model_after,
        test_prompts,
        tokenizer,
        cycle_idx=0,
        cycle_name="TestComparison"
    )
    
    # Create summary report
    logger.info("Creating summary report")
    tracker.create_summary_report()
    
    # Print overall score
    overall_score = results["summary"].get("overall_preservation_score", 0.0)
    logger.info(f"Overall function preservation score: {overall_score:.4f}")
    
    logger.info(f"Complete! Results saved to {args.output_dir}")