# utils/checkpoint.py
import os
import torch

def save_checkpoint(model, optimizer, step, epoch, head_lr_multipliers, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "head_lr_multipliers": head_lr_multipliers,
        "train_step": step,
        "epoch": epoch,
    }, path)
    print(f"[Checkpoint] Saved at {path}")

def load_checkpoint(model, optimizer, path, device, head_lr_multipliers):
    if not os.path.exists(path):
        print("[Checkpoint] No checkpoint found, starting fresh.")
        return 0, 0, head_lr_multipliers

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint.get("train_step", 0)
    epoch = checkpoint.get("epoch", 0)
    head_lr_multipliers = checkpoint.get("head_lr_multipliers", head_lr_multipliers)
    print(f"[Checkpoint] Loaded from {path}, resuming at epoch {epoch}, step {step}")
    return step, epoch, head_lr_multipliers
