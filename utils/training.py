# utils/training.py

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .train_utils import compute_loss, evaluate_model, adjust_learning_rates, prune_inactive_heads, count_active_heads, count_trainable_params
from .checkpoint import save_checkpoint

def train_model(
    model, tokenizer, train_ids, val_ids, device,
    epochs=1, batch_size=8, lr=5e-5, baseline_model=None,
    checkpoint_path="checkpoint.pth"
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    head_lr_multipliers = {(layer_idx, head_idx): 1.0 for layer_idx, block in enumerate(model.blocks)
                           for head_idx in range(block["attn"].num_heads)}

    train_dataset = TensorDataset(torch.tensor(train_ids))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_input_ids = torch.tensor(val_ids).to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)

            logits = model(input_ids)
            loss = compute_loss(logits, input_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        val_loss, val_perplexity, baseline_perplexity = evaluate_model(
            model, val_input_ids, val_input_ids, baseline_model=baseline_model
        )

        active_heads = count_active_heads(model)
        param_count = count_trainable_params(model)

        print(f"Epoch {epoch+1}/{epochs} completed:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Perplexity: {val_perplexity:.2f}")
        if baseline_perplexity:
            print(f"  Baseline Perplexity: {baseline_perplexity:.2f}")
        print(f"  Active Heads: {active_heads}")
        print(f"  Trainable Parameters: {param_count}")

        adjust_learning_rates(optimizer, head_lr_multipliers, model, lr)
        prune_inactive_heads(model)

        save_checkpoint(checkpoint_path, model, optimizer, head_lr_multipliers, epoch+1, 0)

    print("Training complete! 🎉")


def compute_loss(logits, targets):
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def evaluate_model(model, input_ids, targets, baseline_model=None):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        val_loss = compute_loss(logits, targets)
        val_perplexity = math.exp(val_loss.item())

        baseline_perplexity = None
        if baseline_model:
            baseline_logits = baseline_model(input_ids).logits
            baseline_loss = compute_loss(baseline_logits, targets)
            baseline_perplexity = math.exp(baseline_loss.item())

    model.train()
    return val_loss.item(), val_perplexity, baseline_perplexity

def count_active_heads(model, threshold=1e-2):
    active_heads = 0
    for block in model.blocks:
        attn = block["attn"]
        active_heads += sum(float(g) > threshold for g in attn.gate)
    return active_heads

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_learning_rates(optimizer, head_lr_multipliers, model, base_lr):
    for layer_idx, block in enumerate(model.blocks):
        attn = block["attn"]
        for head_idx in range(attn.num_heads):
            g = float(attn.gate[head_idx].item())
            key = (layer_idx, head_idx)
            if g < 0.1:
                head_lr_multipliers[key] = min(head_lr_multipliers[key] * 1.2, 5.0)
            elif g > 0.9:
                head_lr_multipliers[key] = max(head_lr_multipliers[key] * 0.8, 0.1)
            group_index = layer_idx * attn.num_heads + head_idx
            optimizer.param_groups[group_index]['lr'] = base_lr * head_lr_multipliers[key]

def prune_inactive_heads(model, threshold=1e-3):
    for block in model.blocks:
        attn = block["attn"]
        for head_idx in range(attn.num_heads):
            if attn.gate[head_idx] < threshold:
                attn.gate.data[head_idx] = 0.0
                for param in attn.W_q[head_idx].parameters(): param.requires_grad = False
                for param in attn.W_k[head_idx].parameters(): param.requires_grad = False
                for param in attn.W_v[head_idx].parameters(): param.requires_grad = False
                for param in attn.W_o[head_idx].parameters(): param.requires_grad = False
