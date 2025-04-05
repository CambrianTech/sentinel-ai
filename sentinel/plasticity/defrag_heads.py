#!/usr/bin/env python
"""
Neural Defragmentation System for Transformer Models

This module implements a brain-inspired "defragging" system for transformer models that:
1. Detects redundant attention heads by analyzing activation patterns
2. Merges similar heads to consolidate functionality
3. Reinitializes dead or unused heads to provide capacity for new learning
4. Reorganizes internal representations for more efficient computation

This represents a sleep-inspired maintenance phase for neural networks,
akin to memory consolidation in biological systems.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DefragConfiguration:
    """Configuration for the neural defragmentation process."""
    
    # Thresholds for identifying head states
    dead_head_threshold: float = 0.1  # Activation below this is considered "dead"
    redundancy_threshold: float = 0.85  # Cosine similarity above this marks redundancy
    
    # Defrag operation controls
    merge_redundant_heads: bool = True
    reinitialize_dead_heads: bool = True
    consolidate_weights: bool = True
    
    # Initialization parameters for reborn heads
    reinit_scale: float = 0.01  # Small initialization scale for reborn heads
    warmup_steps: int = 100  # Gradual warmup for reinitialized heads
    
    # Logging and visualization
    log_defrag_metrics: bool = True
    create_visualizations: bool = True
    visualization_path: str = "./output/defrag_visualizations"


class HeadDefragmenter:
    """
    Implements neural defragmentation for transformer models.
    
    This class analyzes and reorganizes attention heads based on their activity patterns,
    merging redundant heads and reinitializing unused ones to create a more efficient
    neural architecture while preserving learned capabilities.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        config: Optional[DefragConfiguration] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the head defragmenter with a model and configuration.
        
        Args:
            model: The transformer model to defragment
            config: Defragmentation configuration parameters
            device: Device to run computations on (cpu/cuda)
        """
        self.model = model
        self.config = config or DefragConfiguration()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Internal state
        self.head_maps = {}  # Maps original head positions to new positions
        self.activation_history = {}  # Tracks activation patterns during analysis
        self.similarity_matrix = None  # Stores head similarity measurements
        self.merged_heads = []  # List of heads that were merged
        self.reinitialized_heads = []  # List of heads that were reinitialized
        
        # Initialize metrics collection
        self.metrics = {
            'redundant_heads': 0,
            'dead_heads': 0,
            'merged_heads': 0,
            'reinitialized_heads': 0,
            'memory_savings': 0.0,
            'entropy_before': 0.0,
            'entropy_after': 0.0,
        }
        
        logger.info(f"Initialized HeadDefragmenter (device: {self.device})")
    
    def analyze_head_activations(
        self, 
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Analyze head activations across batches of data to identify patterns.
        
        Args:
            dataloader: DataLoader with sample inputs
            num_batches: Number of batches to process for analysis
            
        Returns:
            Dictionary mapping (layer, head) to activation statistics
        """
        logger.info(f"Analyzing head activations across {num_batches} batches")
        
        # Put model in eval mode during analysis
        self.model.eval()
        
        # Store original values for later analysis
        activation_patterns = {}
        entropy_values = {}
        
        # Register hooks to capture activations
        hooks = []
        activation_maps = {}
        
        # Helper function for hook registration
        def get_activation_hook(layer_idx, head_idx):
            def hook(module, input, output):
                # Capture attention weights
                if isinstance(output, tuple):
                    # Most transformer models return attention weights as part of output
                    if len(output) > 1 and output[1] is not None:
                        attn_weights = output[1]  # [batch_size, num_heads, seq_len, seq_len]
                        
                        # Store for this specific head
                        if layer_idx not in activation_maps:
                            activation_maps[layer_idx] = {}
                        if head_idx not in activation_maps[layer_idx]:
                            activation_maps[layer_idx][head_idx] = []
                            
                        # Extract weights for this head
                        head_weights = attn_weights[:, head_idx, :, :]
                        activation_maps[layer_idx][head_idx].append(head_weights.detach())
            return hook
        
        # Register hooks for all attention modules
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and "output" not in name.lower():
                # Parse layer and head indices from name
                # This will need to be adapted to the specific model architecture
                parts = name.split(".")
                for part in parts:
                    if "layer" in part:
                        try:
                            layer_idx = int(part.replace("layer", ""))
                            for head_idx in range(module.num_attention_heads if hasattr(module, "num_attention_heads") else 12):
                                hook = module.register_forward_hook(get_activation_hook(layer_idx, head_idx))
                                hooks.append(hook)
                        except ValueError:
                            continue
        
        # Process batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                # Prepare inputs
                if isinstance(batch, dict):
                    inputs = batch
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                
                # Forward pass
                outputs = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process collected activations
        for layer_idx, heads in activation_maps.items():
            for head_idx, activations in heads.items():
                if activations:
                    # Stack all captured activations
                    stacked_activations = torch.cat(activations, dim=0)
                    
                    # Calculate statistics
                    mean_activation = torch.mean(stacked_activations, dim=0)
                    activation_variance = torch.var(stacked_activations, dim=0)
                    
                    # Calculate entropy of attention patterns
                    # Entropy measures how diffuse or focused the attention is
                    avg_attention = torch.mean(stacked_activations, dim=0)
                    # Add small epsilon to prevent log(0)
                    normalized_attention = avg_attention / (torch.sum(avg_attention, dim=-1, keepdim=True) + 1e-10)
                    entropy = -torch.sum(normalized_attention * torch.log(normalized_attention + 1e-10), dim=-1)
                    avg_entropy = torch.mean(entropy)
                    
                    # Store results
                    activation_patterns[(layer_idx, head_idx)] = {
                        'mean': mean_activation,
                        'variance': activation_variance,
                        'activity_level': torch.mean(torch.abs(mean_activation)),
                        'entropy': avg_entropy
                    }
                    
                    entropy_values[(layer_idx, head_idx)] = avg_entropy.item()
        
        # Store for internal use
        self.activation_history = activation_patterns
        self.metrics['entropy_before'] = np.mean(list(entropy_values.values()))
        
        logger.info(f"Analyzed {len(activation_patterns)} attention heads")
        return activation_patterns
    
    def compute_head_similarity(self) -> torch.Tensor:
        """
        Compute similarity matrix between heads based on their activation patterns.
        
        Returns:
            Tensor containing pairwise cosine similarity between heads
        """
        logger.info("Computing head similarity matrix")
        
        # Extract heads and their mean activations
        heads = list(self.activation_history.keys())
        
        if not heads:
            logger.warning("No head activations found. Run analyze_head_activations first.")
            return torch.zeros((0, 0))
        
        # Create matrix of head representations for similarity computation
        # We flatten the mean activation patterns to create a vector representation
        head_vectors = []
        for head in heads:
            mean_activation = self.activation_history[head]['mean']
            # Flatten to vector
            vector = mean_activation.flatten()
            head_vectors.append(vector)
        
        # Stack all vectors
        if head_vectors:
            head_matrix = torch.stack(head_vectors)
            
            # Normalize vectors for cosine similarity
            norms = torch.norm(head_matrix, dim=1, keepdim=True)
            normalized_matrix = head_matrix / (norms + 1e-8)
            
            # Compute similarity matrix
            similarity = torch.mm(normalized_matrix, normalized_matrix.t())
            
            # Store for internal use
            self.similarity_matrix = similarity
            self.heads = heads
            
            return similarity
        else:
            logger.warning("No valid head vectors found for similarity computation")
            return torch.zeros((0, 0))
    
    def identify_redundant_heads(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Identify pairs of heads that are redundant based on similarity.
        
        Returns:
            List of redundant head pairs [(head1, head2), ...]
        """
        logger.info("Identifying redundant heads")
        
        if self.similarity_matrix is None:
            self.compute_head_similarity()
            
        if self.similarity_matrix.shape[0] == 0:
            return []
        
        # Get upper triangular part of similarity matrix to avoid duplicates
        upper_tri = torch.triu(self.similarity_matrix, diagonal=1)
        
        # Find highly similar pairs
        redundant_pairs = []
        for i in range(upper_tri.shape[0]):
            for j in range(i+1, upper_tri.shape[1]):
                similarity = upper_tri[i, j].item()
                if similarity > self.config.redundancy_threshold:
                    # Get the actual head identifiers
                    head1 = self.heads[i]
                    head2 = self.heads[j]
                    redundant_pairs.append((head1, head2, similarity))
        
        # Sort by similarity (highest first)
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Remove the similarity value for the return
        redundant_heads = [(h1, h2) for h1, h2, _ in redundant_pairs]
        
        self.metrics['redundant_heads'] = len(redundant_heads)
        logger.info(f"Found {len(redundant_heads)} redundant head pairs")
        
        return redundant_heads
    
    def identify_dead_heads(self) -> List[Tuple[int, int]]:
        """
        Identify heads with very low activation (effectively dead).
        
        Returns:
            List of dead heads [(layer_idx, head_idx), ...]
        """
        logger.info("Identifying dead or inactive heads")
        
        dead_heads = []
        for head, stats in self.activation_history.items():
            # Check if activity level is below threshold
            if stats['activity_level'].item() < self.config.dead_head_threshold:
                dead_heads.append(head)
        
        self.metrics['dead_heads'] = len(dead_heads)
        logger.info(f"Found {len(dead_heads)} dead or inactive heads")
        
        return dead_heads
    
    def merge_heads(
        self, 
        redundant_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> None:
        """
        Merge redundant heads by averaging their weights and zeroing out one.
        
        Args:
            redundant_pairs: List of redundant head pairs to merge
        """
        if not self.config.merge_redundant_heads:
            logger.info("Head merging disabled in configuration")
            return
            
        logger.info(f"Merging {len(redundant_pairs)} redundant head pairs")
        
        # Track which heads have already been merged
        merged = set()
        
        for head1, head2 in redundant_pairs:
            # Skip if either head has already been merged
            if head1 in merged or head2 in merged:
                continue
                
            # Merge the heads by averaging their parameters
            self._merge_head_pair(head1, head2)
            
            # Mark as merged
            merged.add(head2)  # We keep head1, merge head2 into it
            self.merged_heads.append((head1, head2))
        
        self.metrics['merged_heads'] = len(self.merged_heads)
        logger.info(f"Successfully merged {len(self.merged_heads)} head pairs")
    
    def _merge_head_pair(self, head1: Tuple[int, int], head2: Tuple[int, int]) -> None:
        """
        Merge a pair of attention heads by averaging weights and zeroing one out.
        
        Args:
            head1: First head (layer_idx, head_idx)
            head2: Second head (layer_idx, head_idx)
        """
        layer1, idx1 = head1
        layer2, idx2 = head2
        
        # This implementation is model-specific and needs to be adapted
        # to the structure of the particular transformer model
        
        # For demonstration, we'll show a generic approach that would
        # need to be customized to the actual model architecture
        
        try:
            # Example for accessing attention modules (adapt to your model)
            attn_module1 = self._get_attention_module(layer1)
            attn_module2 = self._get_attention_module(layer2)
            
            if attn_module1 is None or attn_module2 is None:
                logger.warning(f"Could not find attention modules for heads {head1} and {head2}")
                return
            
            # Merge query, key, value projection weights
            for param_name in ['query', 'key', 'value']:
                if hasattr(attn_module1, f"{param_name}"):
                    # Get the weights for each head
                    weight1 = self._get_head_weights(attn_module1, param_name, idx1)
                    weight2 = self._get_head_weights(attn_module2, param_name, idx2)
                    
                    if weight1 is not None and weight2 is not None:
                        # Average the weights
                        avg_weight = (weight1 + weight2) / 2.0
                        
                        # Update head1 with average weights
                        self._set_head_weights(attn_module1, param_name, idx1, avg_weight)
                        
                        # Zero out head2 (effectively disabling it)
                        self._set_head_weights(attn_module2, param_name, idx2, torch.zeros_like(weight2))
            
            logger.info(f"Merged heads {head1} and {head2}")
            
        except Exception as e:
            logger.error(f"Error merging heads {head1} and {head2}: {str(e)}")
    
    def reinitialize_dead_heads(
        self, 
        dead_heads: List[Tuple[int, int]]
    ) -> None:
        """
        Reinitialize dead heads with small random weights to allow regrowth.
        
        Args:
            dead_heads: List of heads to reinitialize
        """
        if not self.config.reinitialize_dead_heads:
            logger.info("Head reinitialization disabled in configuration")
            return
            
        logger.info(f"Reinitializing {len(dead_heads)} dead heads")
        
        for head in dead_heads:
            layer_idx, head_idx = head
            
            try:
                # Get the attention module for this layer
                attn_module = self._get_attention_module(layer_idx)
                
                if attn_module is None:
                    continue
                
                # Reinitialize query, key, value projection weights for this head
                for param_name in ['query', 'key', 'value']:
                    if hasattr(attn_module, f"{param_name}"):
                        # Get the original weight shape
                        weight = self._get_head_weights(attn_module, param_name, head_idx)
                        
                        if weight is not None:
                            # Create new small random weights
                            new_weight = torch.randn_like(weight) * self.config.reinit_scale
                            
                            # Update with new weights
                            self._set_head_weights(attn_module, param_name, head_idx, new_weight)
                
                # Track reinitialized head
                self.reinitialized_heads.append(head)
                logger.info(f"Reinitialized head {head}")
                
            except Exception as e:
                logger.error(f"Error reinitializing head {head}: {str(e)}")
        
        self.metrics['reinitialized_heads'] = len(self.reinitialized_heads)
        logger.info(f"Successfully reinitialized {len(self.reinitialized_heads)} heads")
    
    def defragment(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_analysis_batches: int = 10
    ) -> Dict[str, Any]:
        """
        Run the full defragmentation process.
        
        Args:
            dataloader: DataLoader with sample inputs
            num_analysis_batches: Number of batches to process for analysis
            
        Returns:
            Dictionary of defragmentation metrics and results
        """
        logger.info("Starting neural defragmentation process")
        
        # 1. Analyze head activations
        self.analyze_head_activations(dataloader, num_analysis_batches)
        
        # 2. Compute head similarity
        self.compute_head_similarity()
        
        # 3. Identify redundant and dead heads
        redundant_pairs = self.identify_redundant_heads()
        dead_heads = self.identify_dead_heads()
        
        # 4. Merge redundant heads
        self.merge_heads(redundant_pairs)
        
        # 5. Reinitialize dead heads
        self.reinitialize_dead_heads(dead_heads)
        
        # 6. Measure results and create visualizations
        self._measure_defrag_results(dataloader)
        
        # 7. Create visualizations if enabled
        if self.config.create_visualizations:
            self._create_visualizations()
        
        return self.metrics
    
    def _measure_defrag_results(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Measure the results of defragmentation.
        
        Args:
            dataloader: DataLoader for measuring results
        """
        logger.info("Measuring defragmentation results")
        
        # Run another head activation analysis to get post-defrag metrics
        post_activation_patterns = self.analyze_head_activations(dataloader, num_batches=5)
        
        # Calculate entropy after defragmentation
        entropy_values = [stats['entropy'].item() for stats in post_activation_patterns.values()]
        self.metrics['entropy_after'] = np.mean(entropy_values) if entropy_values else 0.0
        
        # Calculate memory savings
        total_heads = len(self.activation_history)
        merged_heads = len(self.merged_heads)
        self.metrics['memory_savings'] = merged_heads / total_heads if total_heads > 0 else 0.0
        
        logger.info(f"Defragmentation complete: {merged_heads} heads merged, "
                   f"{len(self.reinitialized_heads)} heads reinitialized")
    
    def _create_visualizations(self) -> None:
        """Create visualizations of the defragmentation process."""
        if not self.config.create_visualizations:
            return
            
        logger.info("Creating defragmentation visualizations")
        
        # Create output directory if needed
        os.makedirs(self.config.visualization_path, exist_ok=True)
        
        # 1. Plot similarity matrix
        if self.similarity_matrix is not None and self.similarity_matrix.shape[0] > 0:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.similarity_matrix.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Cosine Similarity')
            plt.title(f'Head Similarity Matrix')
            plt.xlabel('Head Index')
            plt.ylabel('Head Index')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.visualization_path, 'head_similarity.png'))
            plt.close()
        
        # 2. Plot head activity levels
        if self.activation_history:
            activity_levels = []
            head_labels = []
            
            for (layer, head), stats in sorted(self.activation_history.items()):
                activity_levels.append(stats['activity_level'].item())
                head_labels.append(f"{layer}-{head}")
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(activity_levels)), activity_levels)
            
            # Color bars based on dead/merged status
            for i, (label, bar) in enumerate(zip(head_labels, bars)):
                layer, head = map(int, label.split('-'))
                if (layer, head) in self.reinitialized_heads:
                    bar.set_color('red')  # Dead, reinitialized
                elif any((layer, head) == h2 for _, h2 in self.merged_heads):
                    bar.set_color('orange')  # Merged into another
                elif any((layer, head) == h1 for h1, _ in self.merged_heads):
                    bar.set_color('green')  # Target of merge
                else:
                    bar.set_color('blue')  # Unchanged
            
            plt.xlabel('Head')
            plt.ylabel('Activity Level')
            plt.title('Head Activity Levels')
            plt.xticks(range(len(head_labels)), head_labels, rotation=90)
            plt.axhline(y=self.config.dead_head_threshold, color='r', linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.visualization_path, 'head_activity.png'))
            plt.close()
        
        # 3. Plot metrics summary
        metrics_to_plot = ['redundant_heads', 'dead_heads', 'merged_heads', 'reinitialized_heads']
        values = [self.metrics[key] for key in metrics_to_plot]
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_to_plot, values)
        plt.ylabel('Count')
        plt.title('Defragmentation Metrics')
        for i, v in enumerate(values):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.visualization_path, 'defrag_metrics.png'))
        plt.close()
        
        # 4. Plot entropy before/after
        entropy_labels = ['Before Defrag', 'After Defrag']
        entropy_values = [self.metrics['entropy_before'], self.metrics['entropy_after']]
        
        plt.figure(figsize=(8, 6))
        plt.bar(entropy_labels, entropy_values)
        plt.ylabel('Average Entropy')
        plt.title('Attention Entropy Before vs After Defragmentation')
        for i, v in enumerate(entropy_values):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.visualization_path, 'entropy_comparison.png'))
        plt.close()
        
        logger.info(f"Visualizations saved to {self.config.visualization_path}")
    
    # Helper methods for accessing model internals - these need to be customized for specific models
    
    def _get_attention_module(self, layer_idx: int) -> Optional[torch.nn.Module]:
        """Get the attention module for a specific layer."""
        # This is a placeholder - needs to be customized for your model architecture
        try:
            # Example for a typical transformer architecture
            return self.model.transformer.layers[layer_idx].attention
        except (AttributeError, IndexError):
            logger.warning(f"Could not find attention module for layer {layer_idx}")
            return None
    
    def _get_head_weights(
        self, 
        module: torch.nn.Module, 
        param_name: str, 
        head_idx: int
    ) -> Optional[torch.Tensor]:
        """Get weights for a specific attention head."""
        # This is a placeholder - needs to be customized for your model architecture
        try:
            # Example for a typical transformer with multi-head attention
            param = getattr(module, f"{param_name}_proj.weight")
            hidden_size = param.shape[0] // module.num_attention_heads
            start_idx = head_idx * hidden_size
            end_idx = (head_idx + 1) * hidden_size
            return param[start_idx:end_idx, :]
        except (AttributeError, IndexError):
            logger.warning(f"Could not get weights for {param_name} head {head_idx}")
            return None
    
    def _set_head_weights(
        self, 
        module: torch.nn.Module, 
        param_name: str, 
        head_idx: int, 
        new_weights: torch.Tensor
    ) -> None:
        """Set weights for a specific attention head."""
        # This is a placeholder - needs to be customized for your model architecture
        try:
            # Example for a typical transformer with multi-head attention
            param = getattr(module, f"{param_name}_proj.weight")
            hidden_size = param.shape[0] // module.num_attention_heads
            start_idx = head_idx * hidden_size
            end_idx = (head_idx + 1) * hidden_size
            param.data[start_idx:end_idx, :] = new_weights
        except (AttributeError, IndexError):
            logger.warning(f"Could not set weights for {param_name} head {head_idx}")


def defrag_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Optional[DefragConfiguration] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to defragment a model in one call.
    
    Args:
        model: The transformer model to defragment
        dataloader: DataLoader with sample inputs for analysis
        config: Optional defragmentation configuration
        device: Device to run computations on
        
    Returns:
        Dictionary of defragmentation metrics and results
    """
    defragmenter = HeadDefragmenter(model, config, device)
    return defragmenter.defragment(dataloader)


if __name__ == "__main__":
    # Example usage
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
    from torch.utils.data import DataLoader, Dataset
    
    parser = argparse.ArgumentParser(description="Neural Defragmentation for Transformer Models")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for analysis")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches for analysis")
    parser.add_argument("--output_dir", type=str, default="./output/defrag", help="Output directory")
    parser.add_argument("--dead_threshold", type=float, default=0.1, help="Threshold for dead heads")
    parser.add_argument("--redundancy_threshold", type=float, default=0.85, help="Threshold for redundant heads")
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
    
    # Create dataset and dataloader
    dataset = SampleTextDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Configure defragmentation
    config = DefragConfiguration(
        dead_head_threshold=args.dead_threshold,
        redundancy_threshold=args.redundancy_threshold,
        visualization_path=args.output_dir
    )
    
    # Run defragmentation
    print("Starting neural defragmentation process...")
    metrics = defrag_model(model, dataloader, config, device)
    
    # Display results
    print("\nDefragmentation Results:")
    print(f"- Found {metrics['redundant_heads']} redundant head pairs")
    print(f"- Found {metrics['dead_heads']} dead heads")
    print(f"- Merged {metrics['merged_heads']} head pairs")
    print(f"- Reinitialized {metrics['reinitialized_heads']} heads")
    print(f"- Memory savings: {metrics['memory_savings']*100:.2f}%")
    print(f"- Entropy change: {metrics['entropy_before']:.4f} → {metrics['entropy_after']:.4f}")
    
    print(f"\nVisualizations saved to: {args.output_dir}")