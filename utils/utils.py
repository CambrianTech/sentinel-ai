import torch
import math
import matplotlib.pyplot as plt
import os

def count_active_heads(model, threshold=1e-2):
    return sum(float(g) > threshold for block in model.blocks for g in block["attn"].gate)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_training_curves(train_losses, val_losses, perplexities, baseline_perplexities, active_heads_history, param_count_history, eval_interval):
    steps = list(range(0, len(train_losses), eval_interval))

    plt.figure(figsize=(8,5))
    plt.plot(range(len(train_losses)), train_losses, label="Train loss")
    plt.plot(steps, val_losses, label="Val loss")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(steps, perplexities, label="Adaptive Model Perplexity")
    plt.plot(steps, baseline_perplexities, label="Baseline Perplexity", linestyle='--')
    plt.xlabel("Training steps")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title("Validation Perplexity vs Baseline")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(range(len(active_heads_history)), active_heads_history, label="Active Heads")
    plt.xlabel("Training steps")
    plt.ylabel("Number of Active Heads")
    plt.title("Active Attention Heads Over Time")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(range(len(param_count_history)), param_count_history, label="Trainable Parameters")
    plt.xlabel("Training steps")
    plt.ylabel("Parameter count")
    plt.title("Parameter Count Over Time")
    plt.show()

def save_checkpoint(path, model, optimizer, head_lr_multipliers, step, epoch):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "head_lr_multipliers": head_lr_multipliers,
        "train_step": step,
        "epoch": epoch
    }, path)
    print(f"[Checkpoint] Saved to {path}")

def load_checkpoint_if_available(path, model, optimizer, head_lr_multipliers, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        head_lr_multipliers.update(checkpoint.get("head_lr_multipliers", {}))
        start_epoch = checkpoint.get("epoch", 0)
        start_step = checkpoint.get("train_step", 0)
        print(f"[Checkpoint] Loaded from {path}, resuming from epoch {start_epoch}, step {start_step}")
        return start_epoch, start_step
    else:
        print("[Checkpoint] None found, starting from scratch")
        return 0, 0
