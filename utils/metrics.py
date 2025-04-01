import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

def perplexity_from_loss(loss):
    """Compute perplexity from loss value"""
    return math.exp(loss)

def compute_perplexity(model, dataloader, device):
    """
    Compute perplexity of a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        
    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Get logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            
            # Calculate loss (cross entropy)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Count tokens (excluding padding tokens)
            num_tokens = (shift_labels != -100).sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return perplexity

def gate_statistics(model):
    stats = {}
    for layer_idx, block in enumerate(model.blocks):
        attn = block["attn"]
        gates = attn.gate.detach().cpu()
        stats[f"layer_{layer_idx}"] = {
            "mean_gate": gates.mean().item(),
            "min_gate": gates.min().item(),
            "max_gate": gates.max().item(),
            "num_active": (gates > 0.01).sum().item(),
        }
    return stats

def attention_entropy(attn_weights):
    # attn_weights: (batch, heads, seq_len, seq_len)
    eps = 1e-8
    p = attn_weights + eps  # avoid log(0)
    entropy = -torch.sum(p * torch.log(p), dim=-1)  # (batch, heads, seq_len)
    return entropy.mean(dim=-1).mean(dim=0)  # mean over seq and batch per head

def log_gate_stats(model):
    print("\n[Sentinel Gate Statistics]")
    stats = gate_statistics(model)
    for layer, s in stats.items():
        print(f"{layer}: mean={s['mean_gate']:.3f} | min={s['min_gate']:.3f} | max={s['max_gate']:.3f} | active={s['num_active']}")
    print()
    return stats

def compute_text_statistics(text):
    """
    Compute various quality metrics for generated text.
    
    Args:
        text: The generated text to analyze
        
    Returns:
        Dictionary with text quality metrics
    """
    if not text:
        return {}
        
    words = text.split()
    if not words:
        return {}
        
    results = {}
    
    # 1. Lexical diversity (higher is better)
    unique_words = len(set(words))
    total_words = len(words)
    results["lexical_diversity"] = unique_words / total_words if total_words > 0 else 0
    
    # 2. Repetition score (lower is better)
    if len(words) <= 1:
        results["repetition_score"] = 0.0
    else:
        window_size = min(50, len(words))
        repeats = 0
        for i in range(len(words) - 1):
            end_idx = min(i + window_size, len(words))
            if words[i] in words[i+1:end_idx]:
                repeats += 1
                
        results["repetition_score"] = repeats / (len(words) - 1)
    
    # 3. Average word length
    avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
    results["avg_word_length"] = avg_word_length
    
    return results