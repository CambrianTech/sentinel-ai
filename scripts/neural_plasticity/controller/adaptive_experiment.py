#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment with ANN Controller and Multi-Cycle Pruning

This script implements a comprehensive neural plasticity experiment combining:
1. Multi-phase training with warmup, pruning, and fine-tuning
2. ANN Controller for dynamic head management
3. Entropy-based pruning with multiple cycles
4. Comprehensive visualization dashboard

Version: v0.1.0 (2025-04-20 22:30:00)
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neural_plasticity")

# Import required modules
try:
    from sentinel.controller.controller_ann import ANNController
    from sentinel.pruning.entropy_magnitude import entropy_based_pruning, collect_attention_distributions
    from utils.neural_plasticity.dashboard.multi_phase_dashboard import MultiPhaseDashboard
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all required modules are installed")
    sys.exit(1)


class TextDataset(Dataset):
    """Simple text dataset for language model training."""
    
    def __init__(self, tokenizer, texts, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", 
                                  max_length=max_length, return_tensors="pt")
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Autoregressive language modeling (shift labels for causal LM)
        item["labels"] = item["input_ids"].clone()
        return item
        
    def __len__(self):
        return len(self.encodings["input_ids"])


class AdaptiveGPT2(torch.nn.Module):
    """GPT-2 with adaptive attention via ANNController."""
    
    def __init__(self, model_name="gpt2", controller_config=None):
        """Initialize model with controller."""
        super().__init__()
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Determine model dimensions
        self.config = self.model.config
        self.num_layers = self.config.n_layer
        self.num_heads = self.config.n_head
        
        # Initialize ANN controller
        self.controller = ANNController(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            config=controller_config or {}
        )
        
        # Register hooks for adaptive attention
        self._register_attention_hooks()
        
        # Create head indices tensor for vectorized operations
        self.head_indices = torch.arange(self.num_heads).view(1, -1, 1, 1)
        
        # Placeholder for attention entropy metrics
        self.head_entropy = torch.zeros(self.num_layers, self.num_heads)
        self.head_grad_norm = torch.zeros(self.num_layers, self.num_heads)
        
    def _register_attention_hooks(self):
        """Register forward hooks to apply gates to attention heads."""
        for layer_idx, layer in enumerate(self.model.transformer.h):
            layer.attn.layer_idx = layer_idx
            layer.attn.register_forward_hook(self._attention_hook)
    
    def _attention_hook(self, module, inputs, outputs):
        """Apply controller gates to attention heads output."""
        layer_idx = getattr(module, "layer_idx")
        
        # Get gates for this layer from controller
        gates = self.controller()[layer_idx].view(1, -1, 1, 1)
        
        # Handle different output types
        if isinstance(outputs, tuple):
            # Some GPT models return tuple (attention_output, attention_weights)
            attn_output, attn_weights = outputs
            
            # Apply gates to attention weights for entropy computation
            if self.training:
                # Store attention weights for entropy calculation
                with torch.no_grad():
                    self._compute_entropy(attn_weights, layer_idx)
            
            # Return gated attention output
            return (attn_output * gates, attn_weights)
        else:
            # Direct attention output
            return outputs * gates
    
    def _compute_entropy(self, attn_probs, layer_idx):
        """Compute attention entropy for metrics."""
        # Compute entropy of attention distribution
        log_attn = torch.log(attn_probs + 1e-10)
        entropy = -torch.sum(attn_probs * log_attn, dim=-1)  # (batch, heads, seq_len)
        
        # Average over batch and sequence dimensions
        avg_entropy = entropy.mean(dim=(0, 2))  # (heads,)
        
        # Store in metrics tensor
        self.head_entropy[layer_idx] = avg_entropy.detach()
    
    def collect_metrics(self):
        """Collect metrics for controller updates."""
        # Compute gradient norms if gradients exist
        for layer_idx, layer in enumerate(self.model.transformer.h):
            for head_idx in range(self.num_heads):
                # Get QKV weights for this head
                head_params = []
                
                # Find weights corresponding to this attention head
                if hasattr(layer.attn, "c_attn"):
                    # GPT-2 style attention
                    qkv_weight = layer.attn.c_attn.weight
                    head_dim = qkv_weight.shape[0] // (3 * self.num_heads)
                    
                    for i in range(3):  # Q, K, V
                        start_idx = (i * self.num_heads + head_idx) * head_dim
                        end_idx = (i * self.num_heads + head_idx + 1) * head_dim
                        head_params.append(qkv_weight[start_idx:end_idx])
                    
                    # Add output projection weight for this head
                    proj_weight = layer.attn.c_proj.weight
                    head_params.append(proj_weight[:, head_idx * head_dim:(head_idx + 1) * head_dim])
                
                # Compute gradient norm if gradients exist
                grad_norm = 0.0
                for param in head_params:
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                
                self.head_grad_norm[layer_idx, head_idx] = grad_norm
        
        # Prepare metrics for controller
        metrics_dict = {
            "entropy": self.head_entropy,
            "grad_norm": self.head_grad_norm,
            "controller_lr": torch.tensor(0.01),  # Adjust as needed
            # Could add more metrics like head importance, utilization, etc.
        }
        
        return metrics_dict
    
    def update_controller(self):
        """Update controller gates based on collected metrics."""
        metrics = self.collect_metrics()
        self.controller.update_gates(metrics)
        
        # Return active head count for monitoring
        gate_values = torch.sigmoid(self.controller.gate_logits)
        active_heads = (gate_values > 0.5).sum().item()
        total_heads = self.num_layers * self.num_heads
        return active_heads, total_heads
    
    def forward(self, **inputs):
        """Forward pass with controller-gated attention."""
        # Run model with gated attention (applied via hooks)
        outputs = self.model(**inputs)
        
        # Add regularization loss from controller
        if self.training and hasattr(outputs, "loss"):
            reg_loss = self.controller.regularization_loss() * self.controller.reg_weight
            outputs.loss = outputs.loss + reg_loss
        
        return outputs
    
    def generate(self, **kwargs):
        """Text generation using the gated model."""
        return self.model.generate(**kwargs)


def load_sample_data(tokenizer, max_length=512):
    """Load sample text data for experimentation."""
    # Sample texts from various domains for diverse training
    sample_texts = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of 'intelligent agents': any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
        "The neural network is a computer system modeled after the human brain. The model makes use of interconnected nodes, replicating the biological neurons in human brains. These neurons function as computational units, processing and transmitting information in the form of electrical signals.",
        "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.",
        "Transformer models have revolutionized natural language processing and other fields by using self-attention mechanisms to process input data in parallel, rather than sequentially as done by recurrent neural networks.",
        "The attention mechanism allows neural networks to focus on specific parts of the input when producing an output token. This enables the model to handle long-range dependencies, something that was challenging for previous architectures.",
        "Neural pruning is a technique used to reduce the size of neural networks by removing weights, neurons, or entire layers that contribute least to the network's performance. This results in more efficient models that require less computation and memory.",
        "Deep learning has enabled many practical applications of machine learning and by extension the overall field of AI. Deep learning breaks down tasks in ways that make all kinds of machine assistance seem possible, even likely.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
        "The Turing test, developed by Alan Turing in 1950, is a test of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.",
    ]
    
    # Create dataset
    dataset = TextDataset(tokenizer, sample_texts, max_length=max_length)
    
    return dataset


def compute_perplexity(model, dataloader, device="cuda"):
    """Compute perplexity on evaluation data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get loss and calculate perplexity
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
    
    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def train_step(model, optimizer, batch, device, scheduler=None):
    """Perform a single training step."""
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update learning rate scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    return loss.item()


def run_experiment(args):
    """Run the full neural plasticity experiment."""
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"neural_plasticity_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dashboard
    dashboard = MultiPhaseDashboard(
        project_name="neural-plasticity",
        experiment_name=f"adaptive-gpt2-{timestamp}",
        output_dir=output_dir,
        config={
            "model_name": args.model_name,
            "pruning_strategy": "entropy",
            "pruning_level": args.pruning_level,
            "cycles": args.cycles,
            "controller_reg_weight": args.controller_reg_weight,
        },
        mode="offline" if args.no_wandb else "online",
        tags=["sentinel-ai", "ann-controller", "multi-phase"]
    )
    
    # Record initial phase
    dashboard.record_phase_transition("setup", 0)
    logger.info("Initializing experiment...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize adaptive GPT-2 with controller
    controller_config = {
        "init_value": 3.0,  # Start with active heads
        "reg_weight": args.controller_reg_weight
    }
    model = AdaptiveGPT2(args.model_name, controller_config)
    model.to(device)
    
    # Prepare datasets
    train_dataset = load_sample_data(tokenizer, max_length=args.max_length)
    eval_dataset = load_sample_data(tokenizer, max_length=args.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Save initial model for comparison
    baseline_path = os.path.join(output_dir, "baseline_model.pt")
    torch.save(model.state_dict(), baseline_path)
    
    # Compute initial perplexity
    initial_perplexity = compute_perplexity(model, eval_dataloader, device)
    logger.info(f"Initial perplexity: {initial_perplexity:.2f}")
    
    # Track metrics
    global_step = 0
    best_perplexity = initial_perplexity
    active_heads_history = []
    perplexity_history = []
    
    # Multi-cycle training
    for cycle in range(args.cycles):
        logger.info(f"Starting cycle {cycle+1}/{args.cycles}")
        
        # Warmup phase
        dashboard.record_phase_transition("warmup", global_step)
        logger.info(f"Cycle {cycle+1}: Starting warmup phase")
        
        # Initialize learning rate scheduler
        num_warmup_steps = int(args.warmup_steps * len(train_dataloader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=args.training_steps
        )
        
        warmup_losses = []
        model.train()
        
        for step in range(args.warmup_steps):
            # Get batch (with wraparound if needed)
            batch_idx = step % len(train_dataloader)
            batch = list(train_dataloader)[batch_idx]
            
            # Train step
            loss = train_step(model, optimizer, batch, device, scheduler)
            warmup_losses.append(loss)
            
            # Update controller periodically
            if (step + 1) % args.controller_update_freq == 0:
                active_heads, total_heads = model.update_controller()
                active_head_percent = (active_heads / total_heads) * 100
                active_heads_history.append((global_step, active_head_percent))
                logger.info(f"Step {global_step}: {active_heads}/{total_heads} heads active ({active_head_percent:.1f}%)")
            
            # Record metrics
            metrics = {
                "loss": loss,
                "phase": "warmup",
                "sparsity": 0.0,  # No pruning yet during warmup
                "cycle": cycle + 1,
            }
            
            # Evaluate periodically
            if (step + 1) % args.eval_freq == 0:
                perplexity = compute_perplexity(model, eval_dataloader, device)
                perplexity_history.append((global_step, perplexity))
                metrics["perplexity"] = perplexity
                metrics["eval_perplexity"] = perplexity
                
                logger.info(f"Warmup step {step+1}/{args.warmup_steps} - Loss: {loss:.4f}, Perplexity: {perplexity:.2f}")
                
                # Check for stabilization (no significant improvement in last steps)
                if step >= args.warmup_steps // 2:
                    recent_losses = warmup_losses[-args.stability_window:]
                    loss_std = np.std(recent_losses)
                    
                    if loss_std < 0.05:  # Very small standard deviation indicates stabilization
                        metrics["stabilized"] = True
                        logger.info(f"Warmup loss has stabilized with std={loss_std:.4f}")
                        break
            
            # Log to dashboard
            dashboard.record_step(metrics, global_step)
            global_step += 1
        
        # Pruning phase
        dashboard.record_phase_transition("pruning", global_step)
        logger.info(f"Cycle {cycle+1}: Starting pruning phase")
        
        # Collect attention distributions for entropy-based pruning
        model.eval()
        logger.info("Collecting attention distributions for entropy-based pruning...")
        
        # Compute pruning ratio for this cycle
        pruning_ratio = args.pruning_level / args.cycles
        logger.info(f"Pruning ratio for cycle {cycle+1}: {pruning_ratio:.4f}")
        
        # Collect attention distributions
        attn_distributions = collect_attention_distributions(
            model, 
            eval_dataloader,
            num_batches=5,
            device=device
        )
        
        # Perform entropy-based pruning
        pruned_heads = entropy_based_pruning(model, attn_distributions, pruning_ratio)
        logger.info(f"Pruned {len(pruned_heads)} heads based on entropy")
        
        # Update sparsity metrics based on pruned heads
        total_heads = model.num_layers * model.num_heads
        sparsity = len(pruned_heads) / total_heads
        
        # Log pruning event
        pruning_info = {
            "strategy": "entropy",
            "pruning_level": pruning_ratio,
            "pruned_heads": pruned_heads,
            "cycle": cycle + 1,
            "step": global_step
        }
        dashboard.record_pruning_event(pruning_info, global_step)
        
        # Record metrics after pruning
        metrics = {
            "phase": "pruning",
            "sparsity": sparsity,
            "cycle": cycle + 1,
        }
        
        # Evaluate model after pruning
        perplexity = compute_perplexity(model, eval_dataloader, device)
        metrics["perplexity"] = perplexity
        metrics["eval_perplexity"] = perplexity
        logger.info(f"Perplexity after pruning: {perplexity:.2f}")
        
        # Record pruning metrics
        dashboard.record_step(metrics, global_step)
        perplexity_history.append((global_step, perplexity))
        global_step += 1
        
        # Finetuning phase
        dashboard.record_phase_transition("finetuning", global_step)
        logger.info(f"Cycle {cycle+1}: Starting fine-tuning phase")
        
        model.train()
        for step in range(args.finetuning_steps):
            # Get batch (with wraparound if needed)
            batch_idx = step % len(train_dataloader)
            batch = list(train_dataloader)[batch_idx]
            
            # Train step
            loss = train_step(model, optimizer, batch, device)
            
            # Update controller periodically
            if (step + 1) % args.controller_update_freq == 0:
                active_heads, total_heads = model.update_controller()
                active_head_percent = (active_heads / total_heads) * 100
                active_heads_history.append((global_step, active_head_percent))
            
            # Record metrics
            metrics = {
                "loss": loss,
                "phase": "finetuning",
                "sparsity": sparsity,  # Maintain sparsity from pruning
                "cycle": cycle + 1,
            }
            
            # Evaluate periodically
            if (step + 1) % args.eval_freq == 0:
                perplexity = compute_perplexity(model, eval_dataloader, device)
                perplexity_history.append((global_step, perplexity))
                metrics["perplexity"] = perplexity
                metrics["eval_perplexity"] = perplexity
                
                logger.info(f"Fine-tuning step {step+1}/{args.finetuning_steps} - Loss: {loss:.4f}, Perplexity: {perplexity:.2f}")
                
                # Track best model
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    # Save best model
                    best_model_path = os.path.join(output_dir, f"best_model_cycle_{cycle+1}.pt")
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"New best perplexity: {perplexity:.2f}, model saved to {best_model_path}")
            
            # Log to dashboard
            dashboard.record_step(metrics, global_step)
            global_step += 1
    
    # Evaluation phase
    dashboard.record_phase_transition("evaluation", global_step)
    logger.info("Starting final evaluation phase")
    
    # Load original (baseline) model for comparison
    baseline_model = AdaptiveGPT2(args.model_name, controller_config)
    baseline_model.load_state_dict(torch.load(baseline_path))
    baseline_model.to(device)
    
    # Generate text samples from both models
    logger.info("Generating text samples for comparison...")
    prompt = "Artificial intelligence is"
    
    # Generate from baseline model
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    baseline_output = baseline_model.generate(
        input_ids, 
        max_length=100, 
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    
    # Generate from pruned model
    pruned_output = model.generate(
        input_ids, 
        max_length=100, 
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    pruned_text = tokenizer.decode(pruned_output[0], skip_special_tokens=True)
    
    # Compute perplexity on evaluation data for both models
    baseline_perplexity = compute_perplexity(baseline_model, eval_dataloader, device)
    final_perplexity = compute_perplexity(model, eval_dataloader, device)
    
    # Calculate improvement
    perplexity_change = ((baseline_perplexity - final_perplexity) / baseline_perplexity) * 100
    active_heads, total_heads = model.update_controller()
    final_sparsity = 1.0 - (active_heads / total_heads)
    
    # Log comparison results
    logger.info(f"Baseline Perplexity: {baseline_perplexity:.2f}")
    logger.info(f"Final Perplexity: {final_perplexity:.2f}")
    logger.info(f"Perplexity Change: {perplexity_change:.2f}%")
    logger.info(f"Final Sparsity: {final_sparsity*100:.1f}%")
    
    # Record final metrics
    final_metrics = {
        "baseline_perplexity": baseline_perplexity,
        "final_perplexity": final_perplexity,
        "perplexity_change": perplexity_change,
        "final_sparsity": final_sparsity,
        "active_heads": active_heads,
        "total_heads": total_heads,
        "phase": "evaluation"
    }
    dashboard.record_step(final_metrics, global_step)
    
    # Create visualization dashboard
    logger.info("Generating visualizations...")
    dashboard_dir = os.path.join(output_dir, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Generate complete process visualization
    process_vis_path = os.path.join(dashboard_dir, "complete_process.png")
    dashboard.visualize_complete_process(process_vis_path)
    
    # Generate multi-cycle dashboard
    if args.cycles > 1:
        multi_cycle_path = os.path.join(dashboard_dir, "multi_cycle.png")
        dashboard.generate_multi_cycle_dashboard(multi_cycle_path)
    
    # Generate standalone HTML dashboard
    html_path = dashboard.generate_standalone_dashboard(dashboard_dir)
    logger.info(f"Dashboard generated at: {html_path}")
    
    # Try to open dashboard in browser
    try:
        import webbrowser
        logger.info(f"Opening dashboard in browser: file://{os.path.abspath(html_path)}")
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception as e:
        logger.warning(f"Could not open browser: {e}")
    
    # Save experiment summary
    summary = {
        "start_time": timestamp,
        "model_name": args.model_name,
        "cycles": args.cycles,
        "pruning_level": args.pruning_level,
        "baseline_perplexity": float(baseline_perplexity),
        "final_perplexity": float(final_perplexity),
        "perplexity_change": float(perplexity_change),
        "final_sparsity": float(final_sparsity),
        "active_heads": int(active_heads),
        "total_heads": int(total_heads),
        "baseline_sample": baseline_text,
        "pruned_sample": pruned_text,
    }
    
    # Write summary to file
    import json
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment completed successfully. Results saved to {output_dir}")
    return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run neural plasticity experiment with ANN controller and multi-cycle pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Hugging Face model name (e.g., gpt2, distilgpt2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda, cpu, or None for auto-detection)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    
    # Pruning configuration
    parser.add_argument("--pruning_level", type=float, default=0.3,
                        help="Overall pruning level (0.0-1.0)")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of pruning cycles")
    
    # Controller configuration
    parser.add_argument("--controller_reg_weight", type=float, default=1e-4,
                        help="L1 regularization weight for controller")
    parser.add_argument("--controller_update_freq", type=int, default=10,
                        help="Steps between controller updates")
    
    # Training procedure
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Steps in warmup phase per cycle")
    parser.add_argument("--finetuning_steps", type=int, default=300,
                        help="Steps in fine-tuning phase per cycle")
    parser.add_argument("--training_steps", type=int, default=1000,
                        help="Total training steps across all phases")
    parser.add_argument("--eval_freq", type=int, default=25,
                        help="Steps between evaluation")
    parser.add_argument("--stability_window", type=int, default=20,
                        help="Window for checking loss stabilization")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for experiment results")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)