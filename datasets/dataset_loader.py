# datasets/dataset_loader.py

import os
import torch
import datasets as ds  # Use alias to avoid confusion with our module
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def load_and_tokenize_dataset(model_name: str, dataset_name="tiny_shakespeare", block_size=128, 
                             limit_train=1000, limit_val=200):
    """
    Loads a dataset and tokenizes it into blocks of input IDs using the tokenizer for `model_name`.
    Returns tokenized torch tensors for train and validation splits.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensures compatibility with models needing padding

    # Load dataset
    if dataset_name == "tiny_shakespeare":
        dataset = ds.load_dataset("tiny_shakespeare")
    elif dataset_name == "wikitext":
        dataset = ds.load_dataset("wikitext", "wikitext-2-raw-v1")
    elif dataset_name == "openwebtext":
        dataset = ds.load_dataset("openwebtext")
    elif dataset_name == "tiny_stories":
        dataset = ds.load_dataset("roneneldan/TinyStories")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def tokenize(example):
        return tokenizer(example["text"], return_special_tokens_mask=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        result = {
            "input_ids": [concatenated[i:i+block_size] for i in range(0, total_len, block_size)]
        }
        return result

    grouped = tokenized.map(group_texts, batched=True)

    # Limit samples for small-scale training/debugging
    train_ids = grouped["train"]["input_ids"][:limit_train]
    val_ids = grouped["validation"]["input_ids"][:limit_val] if "validation" in grouped else grouped["train"]["input_ids"][-limit_val:]

    return train_ids, val_ids


class TokenizedDataset(Dataset):
    """Dataset for language modeling with prepared input and target tokens."""
    
    def __init__(self, input_ids, tokenizer, max_length=None):
        self.input_ids = input_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        # Get sequence of tokens
        tokens = self.input_ids[idx]
        
        # Truncate if necessary
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input_ids and labels (shifted for next token prediction)
        input_ids = torch.tensor(tokens)
        
        # For causal language modeling, labels are the same as inputs
        # (shifted by 1 during loss calculation)
        labels = input_ids.clone()
        
        # Create attention mask (all tokens are attended to)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_dataset(dataset_name, tokenizer, max_length=None):
    """
    Load and prepare datasets for training and evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length (None for no limit)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Set appropriate limits based on dataset size
    if dataset_name == "tiny_shakespeare":
        train_limit, val_limit = 5000, 500
    elif dataset_name == "wikitext":
        train_limit, val_limit = 2000, 200
    elif dataset_name in ["openwebtext", "c4"]:
        train_limit, val_limit = 1000, 100
    else:
        train_limit, val_limit = 1000, 100
    
    # Load and tokenize the dataset
    train_ids, val_ids = load_and_tokenize_dataset(
        tokenizer.name_or_path,
        dataset_name=dataset_name,
        block_size=max_length or 128,
        limit_train=train_limit,
        limit_val=val_limit
    )
    
    # Create datasets
    train_dataset = TokenizedDataset(train_ids, tokenizer, max_length)
    eval_dataset = TokenizedDataset(val_ids, tokenizer, max_length)
    
    return train_dataset, eval_dataset