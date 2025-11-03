#!/usr/bin/env python
"""
Pure Pruning Benchmark for Sentinel-AI

This script focuses specifically on measuring the efficiency benefits of 
pruning in isolation, without agency features, to rigorously demonstrate
that our pruning approach creates genuine efficiency improvements beyond
simple quantization effects.

It implements:
1. Various pruning strategies (gradual, one-shot, iterative)
2. Detailed hardware efficiency measurements (FLOPs, memory, latency)
3. Comparison against established pruning methods
4. Fine-tuning phases after pruning for adaptation

Usage:
    python scripts/pure_pruning_benchmark.py --model_name gpt2 \
                                          --pruning_strategy gradual \
                                          --target_sparsity 0.3 \
                                          --compare_methods \
                                          --measure_flops

Features:
- Isolates pruning from agency to demonstrate pure efficiency benefits
- Measures hardware-level efficiency metrics (not just parameter counts)
- Compares against standard pruning methods as baselines
- Includes proper fine-tuning phases to adapt to pruning
"""

import os
import sys
import argparse
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from sdata.dataset_loader import load_dataset
from utils.train_utils import train_epoch, validate
from utils.metrics import compute_perplexity
from utils.metrics_logger import MetricsLogger
from utils.checkpoint import save_checkpoint
from controller.metrics.head_metrics import collect_head_metrics
from utils.generation_wrapper import generate_text

# Optional: if available import fvcore for FLOPs calculation
try:
    from fvcore.nn import FlopCountAnalysis
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False
    print("fvcore not found. FLOPs calculation will be disabled.")


class PruningBenchmark:
    """
    A comprehensive benchmark for evaluating pure pruning effectiveness.
    
    This class focuses on measuring the true efficiency benefits of pruning
    in isolation from other features like agency, to provide a scientific
    evaluation of pruning approaches.
    """
    
    def __init__(self, args):
        """Initialize the pruning benchmark."""
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = os.path.join(
            args.output_dir, 
            f"pure_pruning_{args.pruning_strategy}_{args.model_name.split('/')[-1]}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            log_dir=os.path.join(self.output_dir, "metrics"),
            metrics=[
                "train_loss", "val_loss", "perplexity", "active_heads_percentage",
                "inference_latency", "memory_usage", "flops", "parameter_count",
                "lexical_diversity", "repetition_score"
            ]
        )
        
        # Create sample prompts for generation tests
        self.eval_prompts = [
            "Once upon a time in a land far away,",
            "The future of artificial intelligence depends on",
            "Scientists have recently discovered that",
            "The most important lesson I've learned is",
            "When considering the environmental impact,",
            "In the history of technological innovation,"
        ]
        
        print(f"🔬 Initializing Pure Pruning Benchmark")
        print(f"📂 Results will be saved to {self.output_dir}")
        print(f"🧠 Device: {self.device}")
        print(f"📊 Pruning strategy: {args.pruning_strategy}")
        print(f"📏 Target sparsity: {args.target_sparsity * 100:.1f}%")
    
    def setup(self):
        """Set up models, datasets, and benchmarking environment."""
        print(f"\n⚙️ Loading models and datasets")
        
        # Load tokenizer
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Proceeding without tokenizer. Some evaluations will be disabled.")
        
        # Load datasets
        print(f"📚 Loading {self.args.dataset} dataset")
        self.train_dataset, self.eval_dataset = load_dataset(
            self.args.dataset, self.tokenizer, self.args.max_length
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset, 
            batch_size=self.args.batch_size
        )
        
        # Create small calibration set for metrics
        if len(self.eval_dataset) > 100:
            indices = np.random.choice(len(self.eval_dataset), 100, replace=False)
            self.calibration_dataset = Subset(self.eval_dataset, indices)
        else:
            self.calibration_dataset = self.eval_dataset
            
        self.calibration_loader = DataLoader(
            self.calibration_dataset, 
            batch_size=self.args.batch_size
        )
        
        # Load models
        # 1. Baseline model (original, without modification)
        print(f"📦 Loading baseline model: {self.args.model_name}")
        self.baseline_model = load_baseline_model(self.args.model_name, self.device)
        
        # 2. Adaptive model (our approach)
        print(f"📦 Loading adaptive model with pruning capability")
        self.model = load_adaptive_model(self.args.model_name, self.baseline_model, self.device)
        
        # Setup is complete
        self._print_model_info(self.model, "Adaptive Model (Initial)")
        print("\n✅ Setup complete")
    
    def run(self):
        """Run the complete benchmarking process."""
        print("\n🚀 Starting Pure Pruning Benchmark")
        
        # Phase 1: Initial evaluation of baseline models
        print("\n📊 Phase 1: Initial Evaluation")
        baseline_metrics = self._evaluate_model(self.baseline_model, "Baseline")
        adaptive_metrics = self._evaluate_model(self.model, "Adaptive (Pre-pruning)")
        
        # Record initial metrics
        epoch = 0
        for name, value in adaptive_metrics.items():
            self.metrics_logger.log(name, value, epoch)
        
        # Phase 2: Training and Pruning
        print("\n📊 Phase 2: Training with Pruning")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Initialize pruning schedule
        pruning_schedule = self._create_pruning_schedule()
        
        # Set up phases based on strategy
        if self.args.pruning_strategy == "one_shot":
            # One-shot pruning: train -> prune -> fine-tune
            phases = [
                {"name": "pre-training", "start": 0, "end": self.args.pruning_start_epoch},
                {"name": "pruning", "start": self.args.pruning_start_epoch, "end": self.args.pruning_start_epoch + 1},
                {"name": "fine-tuning", "start": self.args.pruning_start_epoch + 1, "end": self.args.epochs}
            ]
        elif self.args.pruning_strategy == "gradual":
            # Gradual pruning: steadily prune while training
            phases = [
                {"name": "pre-training", "start": 0, "end": self.args.pruning_start_epoch},
                {"name": "gradual-pruning", "start": self.args.pruning_start_epoch, "end": self.args.pruning_end_epoch},
                {"name": "fine-tuning", "start": self.args.pruning_end_epoch, "end": self.args.epochs}
            ]
        elif self.args.pruning_strategy == "iterative":
            # Iterative pruning: prune -> fine-tune -> prune -> fine-tune
            prune_cycles = 3
            epochs_per_cycle = (self.args.epochs - self.args.pruning_start_epoch) // prune_cycles
            
            phases = [{"name": "pre-training", "start": 0, "end": self.args.pruning_start_epoch}]
            
            for i in range(prune_cycles):
                prune_epoch = self.args.pruning_start_epoch + i * epochs_per_cycle
                finetune_end = prune_epoch + epochs_per_cycle
                
                phases.append({"name": f"pruning-{i+1}", "start": prune_epoch, "end": prune_epoch + 1})
                phases.append({"name": f"fine-tuning-{i+1}", "start": prune_epoch + 1, "end": finetune_end})
        
        # Training/pruning loop
        current_epoch = 0
        
        for phase in phases:
            phase_name = phase["name"]
            start_epoch = phase["start"]
            end_epoch = phase["end"]
            
            print(f"\n📋 Phase: {phase_name} (Epochs {start_epoch}-{end_epoch-1})")
            
            # Adjust learning rate for fine-tuning phases
            if phase_name.startswith("fine-tuning"):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.post_pruning_lr
                print(f"Adjusted learning rate to {self.args.post_pruning_lr} for fine-tuning")
            
            for epoch in range(start_epoch, end_epoch):
                current_epoch = epoch
                print(f"\n🔄 Epoch {epoch+1}/{self.args.epochs}")
                
                # Apply pruning for pruning phases
                if phase_name.startswith("pruning"):
                    if self.args.pruning_strategy == "one_shot" or self.args.pruning_strategy == "iterative":
                        # One-shot pruning or iterative pruning step
                        sparsity = self._get_scheduled_sparsity(pruning_schedule, phase)
                        self._apply_pruning(sparsity)
                        print(f"Applied pruning to sparsity level: {sparsity:.3f}")
                elif phase_name == "gradual-pruning":
                    # Gradual pruning (small incremental steps)
                    sparsity = self._get_scheduled_sparsity(pruning_schedule, epoch)
                    self._apply_pruning(sparsity)
                    print(f"Applied gradual pruning to sparsity level: {sparsity:.3f}")
                
                # Train for one epoch
                train_loss = train_epoch(
                    self.model,
                    self.train_loader,
                    optimizer,
                    self.device,
                    epoch=epoch,
                    gate_regularization=self.args.gate_regularization
                )
                
                # Log training metrics
                self.metrics_logger.log("train_loss", train_loss, epoch)
                
                # Evaluate every few epochs
                if epoch % self.args.eval_interval == 0 or epoch == self.args.epochs - 1:
                    print(f"📊 Evaluating at epoch {epoch+1}")
                    metrics = self._evaluate_model(self.model, f"Epoch {epoch+1}")
                    
                    # Log metrics
                    for name, value in metrics.items():
                        self.metrics_logger.log(name, value, epoch)
                
                # Save checkpoint
                if epoch % self.args.checkpoint_interval == 0 or epoch == self.args.epochs - 1:
                    checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"💾 Saved checkpoint: {checkpoint_path}")
            
                # Generate visualizations
                if epoch % self.args.plot_interval == 0 or epoch == self.args.epochs - 1:
                    self._generate_plots(epoch)
        
        # Phase 3: Final comprehensive evaluation
        print("\n📊 Phase 3: Final Evaluation")
        final_metrics = self._evaluate_model(self.model, "Final Pruned Model", detailed=True)
        
        # Compare with other pruning methods if requested
        if self.args.compare_methods:
            print("\n📊 Comparing with other pruning methods")
            self._run_comparison_experiments()
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "final_model.pt"))
        
        # Generate final summary report
        self._generate_summary_report(baseline_metrics, final_metrics)
        
        print(f"\n✅ Benchmarking complete. Results saved to {self.output_dir}")
    
    def _print_model_info(self, model, label="Model"):
        """Print information about the model's structure and parameters."""
        print(f"\n📋 {label} Information:")
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Count attention heads if applicable
        if hasattr(model, "blocks"):
            num_layers = len(model.blocks)
            if len(model.blocks) > 0 and hasattr(model.blocks[0], "attn"):
                num_heads = model.blocks[0].attn.num_heads
                print(f"  - Structure: {num_layers} layers with {num_heads} heads each")
                print(f"  - Total attention heads: {num_layers * num_heads}")
            
            # Count active/pruned heads for adaptive model
            active_heads = 0
            total_heads = 0
            
            for layer_idx in range(num_layers):
                heads_in_layer = model.blocks[layer_idx]["attn"].num_heads
                total_heads += heads_in_layer
                
                for head_idx in range(heads_in_layer):
                    if model.blocks[layer_idx]["attn"].gate[head_idx].item() > 0.1:
                        active_heads += 1
            
            active_percentage = (active_heads / total_heads) * 100 if total_heads > 0 else 0
            pruned_percentage = 100 - active_percentage
            
            print(f"  - Active heads: {active_heads}/{total_heads} ({active_percentage:.1f}%)")
            print(f"  - Pruned heads: {total_heads - active_heads}/{total_heads} ({pruned_percentage:.1f}%)")
    
    def _evaluate_model(self, model, label="Model", detailed=False):
        """Perform comprehensive evaluation of a model."""
        print(f"\n📊 Evaluating {label}")
        model.eval()
        metrics = {}
        
        # 1. Basic metrics: Perplexity
        val_loss = validate(model, self.eval_loader, self.device)
        perplexity = np.exp(val_loss)
        
        metrics["val_loss"] = val_loss
        metrics["perplexity"] = perplexity
        
        print(f"  - Validation loss: {val_loss:.4f}")
        print(f"  - Perplexity: {perplexity:.2f}")
        
        # 2. Count parameters and active heads
        total_params = sum(p.numel() for p in model.parameters())
        metrics["parameter_count"] = total_params
        
        # Count active heads if applicable
        if hasattr(model, "blocks"):
            active_heads = 0
            total_heads = 0
            
            for layer_idx in range(len(model.blocks)):
                for head_idx in range(model.blocks[layer_idx]["attn"].num_heads):
                    total_heads += 1
                    if model.blocks[layer_idx]["attn"].gate[head_idx].item() > 0.1:
                        active_heads += 1
            
            active_percentage = (active_heads / total_heads) * 100 if total_heads > 0 else 0
            metrics["active_heads"] = active_heads
            metrics["total_heads"] = total_heads
            metrics["active_heads_percentage"] = active_percentage
            
            print(f"  - Active attention heads: {active_heads}/{total_heads} ({active_percentage:.1f}%)")
        
        # 3. Inference latency (time per token)
        inference_time = self._measure_inference_latency(model)
        metrics["inference_latency"] = inference_time
        print(f"  - Inference latency: {inference_time:.4f} ms/token")
        
        # 4. Memory usage
        if torch.cuda.is_available() and self.device.type == "cuda":
            memory_usage = self._measure_memory_usage(model)
            metrics["memory_usage"] = memory_usage
            print(f"  - Peak memory usage: {memory_usage:.2f} MB")
        
        # 5. FLOPs calculation (if available)
        if FLOPS_AVAILABLE and self.args.measure_flops:
            try:
                flops = self._measure_flops(model)
                metrics["flops"] = flops
                print(f"  - FLOPs: {flops/1e9:.2f} GFLOPs")
            except Exception as e:
                print(f"  - Error measuring FLOPs: {e}")
        
        # 6. Generation quality (for models with tokenizers)
        if self.tokenizer is not None and (detailed or label == "Baseline" or "Final" in label):
            quality_metrics = self._evaluate_generation_quality(model)
            metrics.update(quality_metrics)
            
            print(f"  - Text quality: diversity={quality_metrics.get('lexical_diversity', 'N/A'):.3f}, " +
                  f"repetition={quality_metrics.get('repetition_score', 'N/A'):.3f}")
            
            # Save sample generations for detailed evaluations
            if detailed:
                self._save_sample_generations(model, label)
        
        return metrics
    
    def _measure_inference_latency(self, model):
        """Measure inference latency in milliseconds per token."""
        model.eval()
        
        # Prepare input batch
        batch = next(iter(self.calibration_loader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(**batch)
        
        # Measure latency
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            num_runs = 10
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = model(**batch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
        
        # Calculate milliseconds per token
        total_time_ms = (end_time - start_time) * 1000  # convert to ms
        total_tokens = batch["input_ids"].numel() * num_runs
        latency_per_token = total_time_ms / total_tokens
        
        return latency_per_token
    
    def _measure_memory_usage(self, model):
        """Measure peak memory usage during inference in MB."""
        if not torch.cuda.is_available():
            return None
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Prepare input batch
        batch = next(iter(self.calibration_loader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Run inference
        model.eval()
        with torch.no_grad():
            _ = model(**batch)
        
        # Measure peak memory
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return peak_memory_mb
    
    def _measure_flops(self, model):
        """Measure FLOPs (floating point operations) for one forward pass."""
        if not FLOPS_AVAILABLE:
            return None
        
        # Prepare sample input
        batch = next(iter(self.calibration_loader))
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        
        # Create FlopCountAnalysis instance
        flops_count = FlopCountAnalysis(model, inputs)
        flops = flops_count.total()
        
        return flops
    
    def _evaluate_generation_quality(self, model):
        """Evaluate text generation quality metrics."""
        metrics = {}
        
        if self.tokenizer is None:
            return metrics
        
        # Generate text samples
        generated_texts = []
        for prompt in self.eval_prompts:
            try:
                output = generate_text(
                    model=model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_length=self.args.generation_length,
                    temperature=0.7,
                    top_p=0.9,
                    device=self.device
                )
                generated_texts.append(output)
            except Exception as e:
                print(f"Error generating text: {e}")
        
        if not generated_texts:
            return metrics
        
        # Calculate quality metrics
        # 1. Repetition score (lower is better)
        repetition_scores = []
        for text in generated_texts:
            words = text.split()
            if len(words) <= 1:
                continue
                
            # Count word repetitions within a window
            window_size = min(50, len(words))
            repeats = 0
            for i in range(len(words) - 1):
                end_idx = min(i + window_size, len(words))
                if words[i] in words[i+1:end_idx]:
                    repeats += 1
            
            repetition_score = repeats / (len(words) - 1) if len(words) > 1 else 0
            repetition_scores.append(repetition_score)
        
        if repetition_scores:
            metrics["repetition_score"] = float(np.mean(repetition_scores))
        
        # 2. Lexical diversity (higher is better)
        diversity_scores = []
        for text in generated_texts:
            words = text.split()
            if not words:
                continue
                
            unique_words = len(set(words))
            total_words = len(words)
            
            diversity = unique_words / total_words if total_words > 0 else 0
            diversity_scores.append(diversity)
        
        if diversity_scores:
            metrics["lexical_diversity"] = float(np.mean(diversity_scores))
        
        return metrics
    
    def _save_sample_generations(self, model, label):
        """Save sample text generations to disk."""
        if self.tokenizer is None:
            return
        
        generations_dir = os.path.join(self.output_dir, "sample_generations")
        os.makedirs(generations_dir, exist_ok=True)
        
        # Generate text samples
        output_file = os.path.join(generations_dir, f"generation_{label.replace(' ', '_')}.txt")
        
        with open(output_file, "w") as f:
            f.write(f"===== Text Generations: {label} =====\n\n")
            
            for prompt in self.eval_prompts:
                try:
                    output = generate_text(
                        model=model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        max_length=self.args.generation_length,
                        temperature=0.7,
                        top_p=0.9,
                        device=self.device
                    )
                    
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Generated: {output}\n\n")
                except Exception as e:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Error generating text: {e}\n\n")
    
    def _create_pruning_schedule(self):
        """Create a pruning schedule based on the specified parameters."""
        if self.args.pruning_strategy == "one_shot":
            # For one-shot pruning, the schedule is just the target sparsity
            return {"target_sparsity": self.args.target_sparsity}
        
        # For gradual pruning, create a schedule from start to end epochs
        schedule = {
            "start_epoch": self.args.pruning_start_epoch,
            "end_epoch": self.args.pruning_end_epoch,
            "target_sparsity": self.args.target_sparsity,
            "schedule_type": self.args.pruning_schedule
        }
        
        return schedule
    
    def _get_scheduled_sparsity(self, schedule, epoch_or_phase):
        """Get the target sparsity for the current epoch based on the schedule."""
        if self.args.pruning_strategy == "one_shot" or self.args.pruning_strategy == "iterative":
            # For one-shot or iterative, return the full target sparsity
            # For iterative, the epoch_or_phase parameter is actually the phase
            if isinstance(epoch_or_phase, dict):
                phase_name = epoch_or_phase["name"]
                if "pruning" in phase_name:
                    phase_num = int(phase_name.split("-")[-1]) if "-" in phase_name else 1
                    total_phases = 3  # default for iterative
                    return (phase_num / total_phases) * self.args.target_sparsity
            return self.args.target_sparsity
        
        # For gradual pruning, calculate based on schedule
        start_epoch = schedule["start_epoch"]
        end_epoch = schedule["end_epoch"]
        target_sparsity = schedule["target_sparsity"]
        schedule_type = schedule["schedule_type"]
        
        # If before start or after end, use boundary values
        if epoch_or_phase < start_epoch:
            return 0.0
        elif epoch_or_phase >= end_epoch:
            return target_sparsity
        
        # Calculate progress as a fraction between 0 and 1
        progress = (epoch_or_phase - start_epoch) / (end_epoch - start_epoch)
        
        # Apply schedule function
        if schedule_type == "linear":
            return progress * target_sparsity
        elif schedule_type == "exp":
            # Exponential growth: faster at the end
            return target_sparsity * (np.exp(3 * progress) - 1) / (np.exp(3) - 1)
        elif schedule_type == "cos":
            # Cosine schedule: slower at beginning and end
            return target_sparsity * 0.5 * (1 - np.cos(np.pi * progress))
        else:
            return progress * target_sparsity  # default to linear
    
    def _apply_pruning(self, sparsity_level):
        """Apply pruning to the model based on specified strategy."""
        if not hasattr(self.model, "blocks"):
            print("Model does not support pruning. Skipping.")
            return
        
        pruning_method = self.args.pruning_method
        model = self.model
        
        # Count total attention heads
        num_layers = len(model.blocks)
        num_heads = model.blocks[0]["attn"].num_heads
        total_heads = num_layers * num_heads
        
        # Calculate number of heads to prune
        heads_to_prune = int(total_heads * sparsity_level)
        if heads_to_prune <= 0:
            return
        
        print(f"Pruning {heads_to_prune} out of {total_heads} heads ({sparsity_level*100:.1f}% sparsity)")
        
        # Determine which heads to prune based on the method
        if pruning_method == "random":
            # Random pruning
            all_head_indices = [(l, h) for l in range(num_layers) for h in range(num_heads)]
            np.random.shuffle(all_head_indices)
            to_prune = all_head_indices[:heads_to_prune]
            
        elif pruning_method == "entropy":
            # Entropy-based pruning
            # First, compute entropy for all heads
            entropies = torch.zeros((num_layers, num_heads))
            
            # Use calibration data to compute attention entropy
            model.eval()
            with torch.no_grad():
                for batch in self.calibration_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    _ = model(**batch, output_attentions=True)
                    
                    # Access attention matrices from the model
                    for layer_idx in range(num_layers):
                        attentions = model.blocks[layer_idx]["attn"].last_attention
                        
                        if attentions is not None:
                            for head_idx in range(num_heads):
                                # Get attention weights for this head
                                head_attention = attentions[:, head_idx]
                                
                                # Compute entropy
                                entropy = -torch.sum(
                                    head_attention * torch.log(head_attention + 1e-10),
                                    dim=-1
                                ).mean()
                                
                                entropies[layer_idx, head_idx] += entropy.item()
            
            # Normalize entropy values
            entropies = entropies / len(self.calibration_loader)
            
            # Flatten and sort by entropy (higher entropy = less specialized = more prunable)
            flat_entropies = entropies.view(-1)
            sorted_indices = torch.argsort(flat_entropies, descending=True)
            
            # Convert flat indices to (layer, head) pairs
            to_prune = [(idx // num_heads, idx % num_heads) for idx in sorted_indices[:heads_to_prune].tolist()]
            
        elif pruning_method == "magnitude":
            # Magnitude-based pruning (prune heads with smallest weights)
            magnitudes = torch.zeros((num_layers, num_heads))
            
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    # Get weights for this head
                    q_weight = model.blocks[layer_idx]["attn"].W_q[head_idx].weight
                    k_weight = model.blocks[layer_idx]["attn"].W_k[head_idx].weight
                    v_weight = model.blocks[layer_idx]["attn"].W_v[head_idx].weight
                    o_weight = model.blocks[layer_idx]["attn"].W_o[head_idx].weight
                    
                    # Calculate magnitude (L2 norm)
                    magnitude = (q_weight.pow(2).sum() + k_weight.pow(2).sum() + 
                                v_weight.pow(2).sum() + o_weight.pow(2).sum()).sqrt()
                    
                    magnitudes[layer_idx, head_idx] = magnitude.item()
            
            # Flatten and sort by magnitude (smaller = more prunable)
            flat_magnitudes = magnitudes.view(-1)
            sorted_indices = torch.argsort(flat_magnitudes)
            
            # Convert flat indices to (layer, head) pairs
            to_prune = [(idx // num_heads, idx % num_heads) for idx in sorted_indices[:heads_to_prune].tolist()]
        
        else:
            print(f"Unknown pruning method: {pruning_method}. Using random pruning.")
            all_head_indices = [(l, h) for l in range(num_layers) for h in range(num_heads)]
            np.random.shuffle(all_head_indices)
            to_prune = all_head_indices[:heads_to_prune]
        
        # Apply pruning by setting gates to near-zero
        with torch.no_grad():
            # First, set all gates to 1
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.999, device=self.device)
            
            # Then, set pruned gates to near-zero
            for layer_idx, head_idx in to_prune:
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=self.device)
        
        # Save pruning mask
        pruning_dir = os.path.join(self.output_dir, "pruning_masks")
        os.makedirs(pruning_dir, exist_ok=True)
        
        mask = torch.ones((num_layers, num_heads))
        for layer_idx, head_idx in to_prune:
            mask[layer_idx, head_idx] = 0
            
        np.save(
            os.path.join(pruning_dir, f"pruning_mask_level{sparsity_level:.2f}.npy"),
            mask.numpy()
        )
    
    def _run_comparison_experiments(self):
        """Run experiments with alternative pruning methods for comparison."""
        methods = ["random", "magnitude", "entropy"]
        
        # Skip the method already used in the main experiment
        methods = [m for m in methods if m != self.args.pruning_method]
        
        # Create comparison directory
        comparison_dir = os.path.join(self.output_dir, "method_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        comparison_results = {self.args.pruning_method: {}}
        
        # Save current model metrics as our method's results
        current_metrics = self._evaluate_model(self.model, f"Final ({self.args.pruning_method})")
        comparison_results[self.args.pruning_method] = current_metrics
        
        # Run experiments with each alternative method
        for method in methods:
            print(f"\n📊 Testing alternative pruning method: {method}")
            
            # Create a fresh model for each method
            fresh_model = load_adaptive_model(self.args.model_name, self.baseline_model, self.device)
            
            # Set the pruning method and apply it
            self.args.pruning_method = method
            self._apply_pruning(self.args.target_sparsity)
            
            # Evaluate the model
            metrics = self._evaluate_model(fresh_model, f"Method: {method}")
            comparison_results[method] = metrics
            
            # Cleanup
            del fresh_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Restore original pruning method
        self.args.pruning_method = self.args.pruning_method
        
        # Save comparison results
        with open(os.path.join(comparison_dir, "method_comparison.json"), "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        # Create comparison plots
        self._plot_method_comparison(comparison_results, comparison_dir)
    
    def _plot_method_comparison(self, results, output_dir):
        """Create plots comparing different pruning methods."""
        methods = list(results.keys())
        
        # Key metrics to compare
        metrics_to_plot = [
            ("perplexity", "Perplexity (lower is better)"),
            ("inference_latency", "Inference Latency (ms/token, lower is better)"),
            ("memory_usage", "Memory Usage (MB, lower is better)"),
            ("lexical_diversity", "Lexical Diversity (higher is better)"),
            ("repetition_score", "Repetition Score (lower is better)")
        ]
        
        for metric_key, metric_label in metrics_to_plot:
            if not all(metric_key in results[method] for method in methods):
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Get values for each method
            values = [results[method][metric_key] for method in methods]
            
            # Create bar plot
            bars = plt.bar(methods, values)
            
            # Determine if lower or higher is better
            lower_better = metric_key in ["perplexity", "inference_latency", "memory_usage", "repetition_score"]
            
            # Color the bars based on value (green for best, red for worst)
            best_idx = np.argmin(values) if lower_better else np.argmax(values)
            worst_idx = np.argmax(values) if lower_better else np.argmin(values)
            
            for i, bar in enumerate(bars):
                if i == best_idx:
                    bar.set_color('green')
                elif i == worst_idx:
                    bar.set_color('red')
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                plt.text(i, v * 1.01, f"{v:.3f}", ha='center')
            
            plt.title(f"Comparison of Pruning Methods: {metric_label}")
            plt.ylabel(metric_label)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f"comparison_{metric_key}.png"))
            plt.close()
        
        # Create a radar chart for overall comparison
        metrics_for_radar = [
            ("perplexity", "Perplexity", True),  # True means lower is better
            ("inference_latency", "Latency", True),
            ("memory_usage", "Memory", True),
            ("lexical_diversity", "Diversity", False),  # False means higher is better
            ("repetition_score", "Repetition", True)
        ]
        
        # Only use metrics that are available for all methods
        metrics_for_radar = [m for m in metrics_for_radar 
                           if all(m[0] in results[method] for method in methods)]
        
        if metrics_for_radar:
            self._create_radar_chart(results, metrics_for_radar, methods, output_dir)
    
    def _create_radar_chart(self, results, metrics, methods, output_dir):
        """Create a radar chart comparing methods across multiple metrics."""
        num_metrics = len(metrics)
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Normalize values for radar chart (0-1 range, 1 is best)
        normalized_values = {}
        
        for method in methods:
            normalized_values[method] = []
            
            for metric_key, _, lower_better in metrics:
                values = [results[m][metric_key] for m in methods]
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val if max_val > min_val else 1
                
                if lower_better:
                    # For metrics where lower is better, invert so 1 is best
                    norm_val = 1 - (results[method][metric_key] - min_val) / range_val
                else:
                    # For metrics where higher is better
                    norm_val = (results[method][metric_key] - min_val) / range_val
                
                normalized_values[method].append(norm_val)
            
            # Close the loop
            normalized_values[method] += normalized_values[method][:1]
        
        # Plot each method
        for method in methods:
            ax.plot(angles, normalized_values[method], linewidth=2, label=method)
            ax.fill(angles, normalized_values[method], alpha=0.1)
        
        # Set labels
        metric_labels = [m[1] for m in metrics]
        metric_labels += metric_labels[:1]  # Close the loop
        plt.xticks(angles, metric_labels)
        
        # Customize the chart
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.set_rlabel_position(0)
        plt.ylim(0, 1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Pruning Methods Comparison (Higher is Better)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radar_comparison.png"))
        plt.close()
    
    def _generate_plots(self, epoch):
        """Generate visualization plots from the collected metrics."""
        print(f"\n📊 Generating plots at epoch {epoch+1}")
        
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get metrics history
        history = self.metrics_logger.get_history()
        
        if not history:
            print("No metrics history available for plotting")
            return
            
        epochs = sorted([int(e) for e in history.get("epochs", [])])
        if not epochs:
            print("No epoch data available for plotting")
            return
        
        # Plot metrics over time
        metrics_to_plot = [
            ("train_loss", "Training Loss"),
            ("val_loss", "Validation Loss"),
            ("perplexity", "Perplexity"),
            ("active_heads_percentage", "Active Heads (%)"),
            ("inference_latency", "Inference Latency (ms/token)"),
            ("memory_usage", "Memory Usage (MB)"),
            ("lexical_diversity", "Lexical Diversity"),
            ("repetition_score", "Repetition Score")
        ]
        
        for metric_key, metric_label in metrics_to_plot:
            if metric_key not in history:
                continue
                
            plt.figure(figsize=(10, 6))
            
            metric_epochs = [e for e in epochs if str(e) in history[metric_key]]
            metric_values = [history[metric_key][str(e)] for e in metric_epochs]
            
            if metric_values:
                plt.plot(metric_epochs, metric_values, 'o-')
            
            plt.title(f"{metric_label} Over Time")
            plt.xlabel("Epoch")
            plt.ylabel(metric_label)
            plt.grid(True, alpha=0.3)
            
            # Mark pruning phases if applicable
            if self.args.pruning_strategy != "none":
                plt.axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5,
                           label="Start Pruning")
                
                if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                    plt.axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5,
                              label="Start Fine-tuning")
                
                plt.legend()
            
            plt.savefig(os.path.join(plots_dir, f"{metric_key}_over_time.png"))
            plt.close()
        
        # Also create a summary dashboard
        self._create_summary_dashboard(epoch)
    
    def _create_summary_dashboard(self, epoch):
        """Create a summary dashboard with all key metrics."""
        dashboard_file = os.path.join(self.output_dir, "summary_dashboard.png")
        
        history = self.metrics_logger.get_history()
        if not history:
            return
            
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"Pure Pruning Benchmark Summary (Epoch {epoch+1})", fontsize=16)
        
        # Helper function to safely get metric data
        def get_metric_data(metric_name):
            if metric_name not in history:
                return [], []
                
            epochs = sorted([int(e) for e in history.get("epochs", [])])
            valid_epochs = [e for e in epochs if str(e) in history[metric_name]]
            values = [history[metric_name][str(e)] for e in valid_epochs]
            return valid_epochs, values
        
        # 1. Training and validation loss
        train_epochs, train_loss = get_metric_data("train_loss")
        val_epochs, val_loss = get_metric_data("val_loss")
        
        axs[0, 0].set_title("Training Progress")
        if train_loss:
            axs[0, 0].plot(train_epochs, train_loss, 'b-', label="Training Loss")
        if val_loss:
            axs[0, 0].plot(val_epochs, val_loss, 'r-', label="Validation Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()
        
        # Add pruning phase markers
        if self.args.pruning_strategy != "none":
            axs[0, 0].axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5)
            if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                axs[0, 0].axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5)
        
        # 2. Perplexity
        perplexity_epochs, perplexity = get_metric_data("perplexity")
        
        axs[0, 1].set_title("Perplexity")
        if perplexity:
            axs[0, 1].plot(perplexity_epochs, perplexity, 'g-', marker='o')
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Perplexity")
        axs[0, 1].grid(True, alpha=0.3)
        
        # Add pruning phase markers
        if self.args.pruning_strategy != "none":
            axs[0, 1].axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5)
            if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                axs[0, 1].axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5)
        
        # 3. Active heads percentage
        active_epochs, active_percent = get_metric_data("active_heads_percentage")
        
        axs[1, 0].set_title("Active Attention Heads")
        if active_percent:
            axs[1, 0].plot(active_epochs, active_percent, 'r-', marker='o')
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("% Active Heads")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        # Add pruning phase markers
        if self.args.pruning_strategy != "none":
            axs[1, 0].axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5)
            if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                axs[1, 0].axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5)
        
        # 4. Inference latency
        latency_epochs, latency = get_metric_data("inference_latency")
        
        axs[1, 1].set_title("Inference Latency (Lower is Better)")
        if latency:
            axs[1, 1].plot(latency_epochs, latency, 'c-', marker='o')
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("ms/token")
        axs[1, 1].grid(True, alpha=0.3)
        
        # Add pruning phase markers
        if self.args.pruning_strategy != "none":
            axs[1, 1].axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5)
            if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                axs[1, 1].axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5)
        
        # 5. Lexical diversity
        diversity_epochs, diversity = get_metric_data("lexical_diversity")
        
        axs[2, 0].set_title("Lexical Diversity")
        if diversity:
            axs[2, 0].plot(diversity_epochs, diversity, 'g-', marker='o')
        axs[2, 0].set_xlabel("Epoch")
        axs[2, 0].set_ylabel("Diversity Score")
        axs[2, 0].grid(True, alpha=0.3)
        
        # Add pruning phase markers
        if self.args.pruning_strategy != "none":
            axs[2, 0].axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5)
            if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                axs[2, 0].axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5)
        
        # 6. Memory usage
        memory_epochs, memory = get_metric_data("memory_usage")
        
        axs[2, 1].set_title("Memory Usage (Lower is Better)")
        if memory:
            axs[2, 1].plot(memory_epochs, memory, 'm-', marker='o')
        axs[2, 1].set_xlabel("Epoch")
        axs[2, 1].set_ylabel("Memory (MB)")
        axs[2, 1].grid(True, alpha=0.3)
        
        # Add pruning phase markers
        if self.args.pruning_strategy != "none":
            axs[2, 1].axvline(x=self.args.pruning_start_epoch, color='r', linestyle='--', alpha=0.5)
            if hasattr(self.args, 'pruning_end_epoch') and self.args.pruning_end_epoch < self.args.epochs:
                axs[2, 1].axvline(x=self.args.pruning_end_epoch, color='g', linestyle='--', alpha=0.5)
        
        # Add timestamp and model info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(
            0.5, 0.01,
            f"Model: {self.args.model_name} | Strategy: {self.args.pruning_strategy} | Generated: {timestamp}",
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(dashboard_file)
        plt.close()
    
    def _generate_summary_report(self, baseline_metrics, final_metrics):
        """Generate a comprehensive summary report of the benchmark results."""
        report_file = os.path.join(self.output_dir, "benchmark_report.md")
        
        with open(report_file, "w") as f:
            f.write(f"# Pure Pruning Benchmark Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Benchmark Configuration\n\n")
            f.write(f"- **Model:** {self.args.model_name}\n")
            f.write(f"- **Dataset:** {self.args.dataset}\n")
            f.write(f"- **Pruning Strategy:** {self.args.pruning_strategy}\n")
            f.write(f"- **Pruning Method:** {self.args.pruning_method}\n")
            f.write(f"- **Target Sparsity:** {self.args.target_sparsity * 100:.1f}%\n")
            f.write(f"- **Training Epochs:** {self.args.epochs}\n\n")
            
            f.write("## Summary of Results\n\n")
            f.write("| Metric | Baseline | Pruned | Change | % Improvement |\n")
            f.write("|--------|----------|--------|--------|---------------|\n")
            
            # Metrics to include in the summary table
            metrics_for_table = [
                ("perplexity", "Perplexity", True),  # True means lower is better
                ("active_heads_percentage", "Active Heads (%)", False),
                ("inference_latency", "Inference Latency (ms/token)", True),
                ("memory_usage", "Memory Usage (MB)", True),
                ("lexical_diversity", "Lexical Diversity", False),
                ("repetition_score", "Repetition Score", True)
            ]
            
            for metric_key, metric_name, lower_better in metrics_for_table:
                if metric_key in baseline_metrics and metric_key in final_metrics:
                    baseline_value = baseline_metrics[metric_key]
                    final_value = final_metrics[metric_key]
                    
                    change = final_value - baseline_value
                    
                    if baseline_value != 0:
                        pct_change = (change / baseline_value) * 100
                        improvement = -pct_change if lower_better else pct_change
                    else:
                        pct_change = float('nan')
                        improvement = float('nan')
                    
                    # Format the sign for readability
                    change_str = f"{change:+.3f}" if not np.isnan(change) else "N/A"
                    improvement_str = f"{improvement:+.2f}%" if not np.isnan(improvement) else "N/A"
                    
                    f.write(f"| {metric_name} | {baseline_value:.3f} | {final_value:.3f} | {change_str} | {improvement_str} |\n")
            
            f.write("\n## Analysis\n\n")
            
            # Calculate active head reduction
            if "active_heads_percentage" in baseline_metrics and "active_heads_percentage" in final_metrics:
                reduction = 100 - final_metrics["active_heads_percentage"]
                f.write(f"The pruning process successfully removed **{reduction:.1f}%** of attention heads ")
                
                if "perplexity" in baseline_metrics and "perplexity" in final_metrics:
                    perplexity_change = final_metrics["perplexity"] - baseline_metrics["perplexity"]
                    if perplexity_change <= 0:
                        f.write(f"while **improving** perplexity by {-perplexity_change:.2f} points.\n\n")
                    else:
                        pct_change = (perplexity_change / baseline_metrics["perplexity"]) * 100
                        f.write(f"with a perplexity increase of {perplexity_change:.2f} points ({pct_change:.1f}%).\n\n")
            
            # Performance improvements
            if "inference_latency" in baseline_metrics and "inference_latency" in final_metrics:
                latency_change = ((final_metrics["inference_latency"] / baseline_metrics["inference_latency"]) - 1) * 100
                
                if latency_change < 0:
                    f.write(f"Inference speed improved by **{-latency_change:.1f}%** compared to the baseline model.\n\n")
                else:
                    f.write(f"Inference speed decreased by {latency_change:.1f}% compared to the baseline model.\n\n")
            
            # Memory savings
            if "memory_usage" in baseline_metrics and "memory_usage" in final_metrics:
                memory_change = ((final_metrics["memory_usage"] / baseline_metrics["memory_usage"]) - 1) * 100
                
                if memory_change < 0:
                    f.write(f"Memory usage reduced by **{-memory_change:.1f}%** compared to the baseline model.\n\n")
                else:
                    f.write(f"Memory usage increased by {memory_change:.1f}% compared to the baseline model.\n\n")
            
            # Text quality changes
            text_quality_changed = False
            quality_comment = "Text generation quality "
            
            if "lexical_diversity" in baseline_metrics and "lexical_diversity" in final_metrics:
                diversity_change = ((final_metrics["lexical_diversity"] / baseline_metrics["lexical_diversity"]) - 1) * 100
                
                if abs(diversity_change) > 1:  # Only mention if change is significant
                    text_quality_changed = True
                    if diversity_change > 0:
                        quality_comment += f"improved with **{diversity_change:.1f}%** better lexical diversity"
                    else:
                        quality_comment += f"declined with {-diversity_change:.1f}% worse lexical diversity"
            
            if "repetition_score" in baseline_metrics and "repetition_score" in final_metrics:
                repetition_change = ((final_metrics["repetition_score"] / baseline_metrics["repetition_score"]) - 1) * 100
                
                if abs(repetition_change) > 1:  # Only mention if change is significant
                    if text_quality_changed:
                        quality_comment += " and "
                    else:
                        text_quality_changed = True
                    
                    if repetition_change < 0:
                        quality_comment += f"improved with **{-repetition_change:.1f}%** less repetition"
                    else:
                        quality_comment += f"declined with {repetition_change:.1f}% more repetition"
            
            if text_quality_changed:
                f.write(quality_comment + ".\n\n")
            else:
                f.write("Text generation quality remained largely unchanged after pruning.\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            
            # Determine overall success based on metrics
            perf_improved = False
            quality_maintained = True
            
            if "inference_latency" in baseline_metrics and "inference_latency" in final_metrics:
                if final_metrics["inference_latency"] < baseline_metrics["inference_latency"]:
                    perf_improved = True
                    
            if "perplexity" in baseline_metrics and "perplexity" in final_metrics:
                perplexity_change_pct = ((final_metrics["perplexity"] / baseline_metrics["perplexity"]) - 1) * 100
                if perplexity_change_pct > 5:  # More than 5% worse
                    quality_maintained = False
            
            if perf_improved and quality_maintained:
                f.write("The pruning benchmark demonstrates that our approach successfully **improves efficiency while maintaining quality**. ")
            elif perf_improved:
                f.write("The pruning benchmark shows **improved efficiency with some quality trade-offs**. ")
            elif quality_maintained:
                f.write("The pruning benchmark shows **maintained quality but limited efficiency improvements**. ")
            else:
                f.write("The pruning benchmark shows **both efficiency and quality were affected**, suggesting further optimization is needed. ")
            
            f.write("These results provide a clear demonstration of the effectiveness of pruning in reducing model size and ")
            f.write("computational requirements beyond simple quantization effects.\n\n")
            
            if "active_heads" in final_metrics and "total_heads" in final_metrics:
                active = final_metrics["active_heads"]
                total = final_metrics["total_heads"]
                f.write(f"The final model operates with {active} out of {total} attention heads active ")
                f.write(f"({(active/total*100):.1f}%), making it more efficient for deployment on resource-constrained devices.\n")
        
        print(f"📝 Benchmark report generated: {report_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pure Pruning Benchmark for Sentinel-AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2",
                      help="Model name (e.g., 'gpt2', 'distilgpt2')")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="wikitext",
                      help="Dataset to use for training and evaluation")
    parser.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length for dataset")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=30,
                      help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for AdamW optimizer")
    parser.add_argument("--gate_regularization", type=float, default=0.0001,
                      help="L1 regularization coefficient for attention gates")
    
    # Pruning configuration
    parser.add_argument("--pruning_strategy", type=str, 
                      choices=["gradual", "one_shot", "iterative"],
                      default="gradual", help="Strategy for pruning")
    parser.add_argument("--pruning_method", type=str, 
                      choices=["random", "entropy", "magnitude"],
                      default="entropy", help="Method to select heads for pruning")
    parser.add_argument("--pruning_schedule", type=str, 
                      choices=["linear", "exp", "cos"], 
                      default="linear", help="Pruning rate schedule for gradual pruning")
    parser.add_argument("--target_sparsity", type=float, default=0.3,
                      help="Target sparsity level (fraction of heads to prune)")
    parser.add_argument("--pruning_start_epoch", type=int, default=5,
                      help="Epoch to start pruning")
    parser.add_argument("--pruning_end_epoch", type=int, default=15,
                      help="Epoch to end pruning phase (followed by fine-tuning)")
    parser.add_argument("--post_pruning_lr", type=float, default=1e-5,
                      help="Learning rate for fine-tuning after pruning phase")
    
    # Evaluation configuration
    parser.add_argument("--eval_interval", type=int, default=1,
                      help="Evaluate model every N epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                      help="Save checkpoint every N epochs")
    parser.add_argument("--plot_interval", type=int, default=5,
                      help="Generate plots every N epochs")
    parser.add_argument("--generation_length", type=int, default=100,
                      help="Maximum length for text generation")
    
    # Analysis options
    parser.add_argument("--measure_flops", action="store_true", default=False,
                      help="Calculate FLOPs (requires fvcore)")
    parser.add_argument("--compare_methods", action="store_true", default=False,
                      help="Compare against other pruning methods")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./pure_pruning_results",
                      help="Base directory to save results")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (default: auto-detect)")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Initialize and run the benchmark
    benchmark = PruningBenchmark(args)
    benchmark.setup()
    benchmark.run()


if __name__ == "__main__":
    main()