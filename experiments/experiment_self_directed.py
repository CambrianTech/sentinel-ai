#!/usr/bin/env python3
"""
Self-Directed Plasticity Experiment

Instead of specifying pruning ratio, strategy, and training steps,
this experiment lets the AdaptivePlasticityController decide everything
based on the model's observed state.

Usage:
    python experiments/experiment_self_directed.py --model_name gpt2-medium
    python experiments/experiment_self_directed.py --model_name Qwen/Qwen2.5-3B
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from datasets import load_dataset
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
)
from sentinel.controller.adaptive_plasticity_controller import (
    AdaptivePlasticityController,
    PlasticityState,
    PlateauDetector,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1
            if count >= 50:
                break
    import math
    avg_loss = total_loss / max(count, 1)
    return avg_loss, math.exp(avg_loss)


def get_entropy_state(model, dataloader, device, config):
    """Extract PlasticityState from model's current attention patterns."""
    model.eval()
    batch = next(iter(dataloader))
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise ValueError("Model does not return attention maps. Set config.output_attentions = True.")

    entropy_per_layer = []
    for layer_attn in outputs.attentions:
        layer_entropy = calculate_head_entropy(layer_attn)
        entropy_per_layer.append(layer_entropy.cpu().float().numpy())

    entropy_array = np.array(entropy_per_layer)

    # Handle shape: may be [layers, batch, heads] or [layers, heads]
    if entropy_array.ndim == 3:
        entropy_array = entropy_array.mean(axis=1)  # average over batch

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    eval_loss, eval_ppl = evaluate(model, dataloader, device)

    return PlasticityState(
        entropy_per_head=entropy_array,
        gate_values=None,
        eval_loss=eval_loss,
        eval_perplexity=eval_ppl,
        total_heads=num_layers * num_heads,
        active_heads=num_layers * num_heads,
        layer_count=num_layers,
        heads_per_layer=num_heads,
    )


def train_with_plateau_detection(model, dataloader, eval_dataloader, device, max_steps, plateau_detector):
    """Train until plateau or max_steps, whichever comes first."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    step = 0
    data_iter = iter(dataloader)

    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check plateau every 50 steps
        if (step + 1) % 50 == 0:
            eval_loss, eval_ppl = evaluate(model, eval_dataloader, device)
            print(f"  Step {step+1}/{max_steps}: eval_loss={eval_loss:.4f}, ppl={eval_ppl:.2f}")

            if plateau_detector.step(eval_loss):
                print(f"  Plateau detected at step {step+1} (no improvement for {plateau_detector.patience} evals)")
                return step + 1

    return max_steps


def main():
    parser = argparse.ArgumentParser(description="Self-directed plasticity experiment")
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_cycles", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"=== SELF-DIRECTED PLASTICITY EXPERIMENT ===")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/self_directed_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with attention outputs
    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name)
    config.output_attentions = True
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading dataset...")
    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    eval_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)

    # Initialize controller
    controller = AdaptivePlasticityController(max_cycles=args.max_cycles)

    # Initial evaluation
    baseline_loss, baseline_ppl = evaluate(model, eval_loader, device)
    print(f"Baseline: loss={baseline_loss:.4f}, perplexity={baseline_ppl:.2f}")
    print()

    start_time = time.time()
    cycle_results = []

    while True:
        # Get model state
        print(f"--- Assessing model state ---")
        state = get_entropy_state(model, eval_loader, device, config)

        # Ask controller what to do
        action = controller.decide(state)
        print(f"Controller: {action.reason}")

        if not action.should_prune:
            print("Controller decided to stop. Model is efficient enough.")
            break

        # Execute the action
        print(f"\nPruning {action.pruning_ratio:.0%} using {action.strategy} strategy...")

        # Get entropy and gradients for pruning
        batch = next(iter(eval_loader))
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        entropy_values = torch.stack([calculate_head_entropy(a) for a in outputs.attentions])
        if entropy_values.ndim > 2:
            entropy_values = entropy_values.mean(dim=list(range(1, entropy_values.ndim - 1)))

        grad_norms = calculate_head_gradients(model, eval_loader)

        pruning_mask = generate_pruning_mask(
            entropy_values, grad_norms,
            pruning_level=action.pruning_ratio,
            strategy=action.strategy,
        )

        # Measure pre-train post-prune quality
        pre_prune_loss, pre_prune_ppl = state.eval_loss, state.eval_perplexity
        apply_pruning_mask(model, pruning_mask)
        post_prune_loss, post_prune_ppl = evaluate(model, eval_loader, device)
        pruned_count = int((~pruning_mask.bool()).sum().item())
        print(f"Pruned {pruned_count} heads. Post-prune ppl: {post_prune_ppl:.2f} (was {pre_prune_ppl:.2f})")

        # Train with plateau detection
        print(f"\nTraining (max {action.max_training_steps} steps, plateau detection active)...")
        plateau_detector = PlateauDetector(patience=50, min_delta=0.001)
        steps_used = train_with_plateau_detection(
            model, train_loader, eval_loader, device,
            action.max_training_steps, plateau_detector,
        )

        # Post-train evaluation
        post_train_loss, post_train_ppl = evaluate(model, eval_loader, device)
        print(f"\nPost-training: ppl={post_train_ppl:.2f} (baseline={baseline_ppl:.2f})")

        # Record for controller learning
        controller.record_cycle(
            strategy=action.strategy,
            pruning_ratio=action.pruning_ratio,
            training_steps_used=steps_used,
            baseline_ppl=pre_prune_ppl,
            post_prune_ppl=post_prune_ppl,
            post_train_ppl=post_train_ppl,
        )

        cycle_results.append({
            "cycle": controller.cycle_count,
            "strategy": action.strategy,
            "pruning_ratio": action.pruning_ratio,
            "steps_used": steps_used,
            "max_steps": action.max_training_steps,
            "pre_prune_ppl": round(pre_prune_ppl, 4),
            "post_prune_ppl": round(post_prune_ppl, 4),
            "post_train_ppl": round(post_train_ppl, 4),
            "heads_pruned": pruned_count,
            "reason": action.reason,
        })

        print()

    # Final summary
    elapsed = time.time() - start_time
    final_loss, final_ppl = evaluate(model, eval_loader, device)

    print(f"\n{'='*60}")
    print(f"SELF-DIRECTED PLASTICITY RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"Change: {((baseline_ppl - final_ppl) / baseline_ppl) * 100:+.2f}%")
    print(f"Cycles: {controller.cycle_count}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()
    print(controller.summary())

    # Save results
    results = {
        "model_name": args.model_name,
        "baseline_ppl": round(baseline_ppl, 4),
        "final_ppl": round(final_ppl, 4),
        "improvement_pct": round(((baseline_ppl - final_ppl) / baseline_ppl) * 100, 2),
        "total_cycles": controller.cycle_count,
        "elapsed_seconds": round(elapsed, 1),
        "cycles": cycle_results,
        "device": str(device),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save model
    model_dir = output_dir / "model"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
