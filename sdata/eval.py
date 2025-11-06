# sdata/eval.py

import torch
from sdata.dataset_loader import load_dataset

def evaluate_model(model, tokenizer, dataset_name=None, device="cpu"):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset_name: Name of dataset to evaluate on
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Get dataset
    dataset_wrapper = load_dataset(dataset_name, tokenizer)
    eval_dataset = dataset_wrapper.eval_dataset
    
    # Create eval dataloader
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False
    )
    
    # Evaluate
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Track loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_tokens += batch["input_ids"].size(0)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }

def calculate_perplexity(model, tokenizer, text, device="cpu"):
    """
    Calculate perplexity on a text.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        text: Text to calculate perplexity on
        device: Device to run evaluation on
        
    Returns:
        Perplexity value
    """
    model.eval()
    
    # Tokenize text
    encodings = tokenizer(text, return_tensors="pt")
    
    # Move to device
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    # Calculate loss
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    
    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    
    return perplexity

def load_eval_prompts(dataset_name=None, num_prompts=5):
    """Load evaluation prompts for the specified dataset"""
    # Simple prompts for evaluation
    default_prompts = [
        "The transformer model processes data through multiple layers of computation.",
        "Artificial intelligence systems can learn from experience and improve over time.",
        "The neural network was trained to recognize patterns in complex datasets.",
        "Language models predict the next token based on previous context.",
        "The attention mechanism allows the model to focus on relevant parts of the input."
    ]
    
    return default_prompts[:num_prompts]