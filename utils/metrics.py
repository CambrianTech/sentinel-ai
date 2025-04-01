import math
import re
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
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

def calculate_perplexity(generated_texts, prompts):
    """
    Calculate an approximate perplexity measure for generated texts.
    
    This is a simplified version for validation that doesn't require the model.
    For more accurate perplexity, use compute_perplexity with a language model.
    
    Args:
        generated_texts: List of generated texts
        prompts: List of corresponding prompts
        
    Returns:
        Approximate perplexity score
    """
    if not generated_texts or not prompts:
        return float('nan')
    
    # Extract only the generated portions (exclude the prompts)
    generated_portions = []
    for i, text in enumerate(generated_texts):
        if i < len(prompts) and text.startswith(prompts[i]):
            # Remove the prompt from the beginning
            generated_portion = text[len(prompts[i]):].strip()
            generated_portions.append(generated_portion)
        else:
            generated_portions.append(text)
    
    # Calculate token-level entropy using frequency distribution
    all_tokens = []
    for text in generated_portions:
        # Simple tokenization (split by whitespace and punctuation)
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return float('nan')
    
    # Calculate token frequencies
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    # Calculate entropy
    entropy = 0
    for token, count in token_counts.items():
        prob = count / total_tokens
        entropy -= prob * math.log(prob)
    
    # Convert entropy to perplexity
    perplexity = math.exp(entropy)
    
    return perplexity

def diversity_metrics(texts):
    """
    Calculate diversity metrics for a list of texts.
    
    Args:
        texts: List of generated texts to analyze
        
    Returns:
        Dictionary with diversity metrics
    """
    if not texts:
        return {
            "lexical_diversity": 0.0,
            "unique_token_ratio": 0.0,
            "vocabulary_size": 0
        }
    
    all_words = []
    unique_words_per_text = []
    
    for text in texts:
        # Simple word tokenization
        words = re.findall(r'\w+', text.lower())
        all_words.extend(words)
        unique_words_per_text.append(len(set(words)))
    
    # Calculate vocabulary size
    vocabulary_size = len(set(all_words))
    
    # Calculate unique token ratio
    total_words = len(all_words)
    unique_token_ratio = vocabulary_size / total_words if total_words > 0 else 0
    
    # Calculate average lexical diversity across texts
    lexical_diversity = 0
    for i, text in enumerate(texts):
        words = re.findall(r'\w+', text.lower())
        if words:
            text_diversity = len(set(words)) / len(words)
            lexical_diversity += text_diversity
    
    lexical_diversity = lexical_diversity / len(texts) if texts else 0
    
    return {
        "lexical_diversity": lexical_diversity,
        "unique_token_ratio": unique_token_ratio,
        "vocabulary_size": vocabulary_size
    }

def repetition_metrics(texts):
    """
    Calculate repetition metrics for a list of texts.
    
    Args:
        texts: List of generated texts to analyze
        
    Returns:
        Dictionary with repetition metrics
    """
    if not texts:
        return {
            "repetition_score": 0.0,
            "avg_repetition_length": 0.0
        }
    
    repetition_scores = []
    repetition_lengths = []
    
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        
        if len(words) <= 1:
            repetition_scores.append(0.0)
            repetition_lengths.append(0.0)
            continue
        
        # Find repeated sequences
        repeated = 0
        repeated_lengths = []
        
        for i in range(len(words)):
            for length in range(2, min(10, len(words) - i)):
                sequence = tuple(words[i:i+length])
                rest_of_text = words[i+length:]
                
                if sequence in [tuple(rest_of_text[j:j+length]) for j in range(len(rest_of_text)-length+1)]:
                    repeated += 1
                    repeated_lengths.append(length)
                    break
        
        repetition_score = repeated / len(words) if words else 0
        repetition_scores.append(repetition_score)
        
        if repeated_lengths:
            repetition_lengths.append(sum(repeated_lengths) / len(repeated_lengths))
        else:
            repetition_lengths.append(0.0)
    
    return {
        "repetition_score": sum(repetition_scores) / len(repetition_scores) if repetition_scores else 0,
        "avg_repetition_length": sum(repetition_lengths) / len(repetition_lengths) if repetition_lengths else 0
    }