import torch
import math

def compute_perplexity(loss):
    return math.exp(loss)

def save_checkpoint(model, optimizer, head_lr_multipliers, step, epoch, path):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "head_lr_multipliers": head_lr_multipliers,
        "train_step": step,
        "epoch": epoch
    }
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, head_lr_multipliers, path, device):
    if not os.path.exists(path):
        print("No checkpoint found, starting fresh.")
        return model, optimizer, head_lr_multipliers, 0, 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    head_lr_multipliers = checkpoint.get("head_lr_multipliers", head_lr_multipliers)
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("train_step", 0)
    print(f"Loaded checkpoint from {path}, resuming at epoch {epoch}, step {step}")
    return model, optimizer, head_lr_multipliers, step, epoch
