# sentinel_data/dataset_loader.py

import os
import torch
# Import Hugging Face datasets with absolute import to avoid confusion
import sys
import importlib.util
spec = importlib.util.find_spec('datasets')
if spec:
    datasets = importlib.import_module('datasets')
else:
    # Mock datasets library if not available
    datasets = None
    print("Warning: Hugging Face datasets library not found.")
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def load_and_tokenize_dataset(model_name: str, dataset_name="tiny_shakespeare", block_size=512, 
                             limit_train=1000, limit_val=200):
    """
    Loads a dataset and tokenizes it into blocks of input IDs using the tokenizer for `model_name`.
    Returns tokenized torch tensors for train and validation splits.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensures compatibility with models needing padding

    # Load dataset with trust_remote_code=True for datasets that require it
    if datasets is None:
        raise ImportError("Hugging Face datasets library is required to load datasets")
        
    if dataset_name == "tiny_shakespeare":
        dataset = datasets.load_dataset("tiny_shakespeare", trust_remote_code=True)
    elif dataset_name == "wikitext":
        dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    elif dataset_name == "openwebtext":
        dataset = datasets.load_dataset("openwebtext", trust_remote_code=True)
    elif dataset_name == "tiny_stories":
        dataset = datasets.load_dataset("roneneldan/TinyStories", trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def tokenize(example):
        return tokenizer(example["text"], return_special_tokens_mask=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        
        # Create input_ids
        input_ids = [concatenated[i:i+block_size] for i in range(0, total_len, block_size)]
        
        # Create attention_mask (all 1s since we're using full sequences)
        attention_mask = [[1] * block_size for _ in range(len(input_ids))]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        return result

    # Process in smaller batches to avoid memory issues
    grouped = tokenized.map(group_texts, batched=True, batch_size=10)

    # Limit samples for small-scale training/debugging
    train_dataset = {
        "input_ids": grouped["train"]["input_ids"][:limit_train],
        "attention_mask": grouped["train"]["attention_mask"][:limit_train]
    }
    
    if "validation" in grouped:
        val_dataset = {
            "input_ids": grouped["validation"]["input_ids"][:limit_val],
            "attention_mask": grouped["validation"]["attention_mask"][:limit_val]
        }
    else:
        # Use last part of train as validation if no validation split exists
        val_dataset = {
            "input_ids": grouped["train"]["input_ids"][-limit_val:],
            "attention_mask": grouped["train"]["attention_mask"][-limit_val:]
        }

    return train_dataset, val_dataset


class TokenizedDataset(Dataset):
    """Dataset for language modeling with prepared input and target tokens."""
    
    def __init__(self, dataset_dict, tokenizer, max_length=None):
        """
        Initialize a tokenized dataset.
        
        Args:
            dataset_dict: Dictionary with 'input_ids' and 'attention_mask' keys
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.input_ids = dataset_dict["input_ids"]
        self.attention_masks = dataset_dict["attention_mask"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        # Get sequence of tokens and attention mask
        tokens = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        
        # Truncate if necessary
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # Create input_ids and labels (shifted for next token prediction)
        input_ids = torch.tensor(tokens)
        attention_mask = torch.tensor(attention_mask)
        
        # For causal language modeling, labels are the same as inputs
        # (shifted by 1 during loss calculation)
        labels = input_ids.clone()
        
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