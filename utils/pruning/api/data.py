"""
Data loading utilities for model pruning.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

def load_wikitext():
    """
    Load Wikitext dataset for training and evaluation.
    
    Returns:
        Tuple of (train_data, val_data) from the dataset
    """
    # Load dataset from Hugging Face
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Access the splits
    train_data = wikitext["train"]
    val_data = wikitext["validation"]
    
    return train_data, val_data

def prepare_data(tokenizer, text_data, max_length=512, batch_size=4):
    """
    Prepare dataset for training/evaluation.
    
    Args:
        tokenizer: Tokenizer for the model
        text_data: Dataset containing text
        max_length: Maximum sequence length
        batch_size: Batch size for dataloader
        
    Returns:
        DataLoader for the processed dataset
    """
    # Get text from dataset
    texts = text_data["text"]
    
    # Remove empty strings
    texts = [t for t in texts if t.strip()]
    
    # Limit dataset size for testing
    if len(texts) > 1000:
        print(f"Limiting dataset from {len(texts)} to 1000 samples for faster processing")
        texts = texts[:1000]
    
    # Tokenize text
    encodings = tokenizer(texts, 
                         truncation=True, 
                         max_length=max_length, 
                         padding="max_length", 
                         return_tensors="pt")
    
    # Create dataset
    dataset = TensorDataset(
        encodings["input_ids"], 
        encodings["attention_mask"]
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return dataloader

def prepare_test_data(tokenizer, max_length=512, batch_size=4, num_samples=10):
    """
    Create a tiny test dataset for quick testing.
    
    Args:
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        batch_size: Batch size for dataloader
        num_samples: Number of samples in the test dataset
        
    Returns:
        Tuple of (train_dataloader, val_dataloader) for testing
    """
    # Create some test data
    texts = [
        "This is a simple test sentence for pruning experiments.",
        "The quick brown fox jumps over the lazy dog.",
        "We need to create a small dataset for testing purposes.",
        "Neural network pruning can make models smaller and faster.",
        "Attention heads in transformers can often be pruned without loss of performance.",
        "Testing with a small dataset helps catch issues quickly.",
        "The cake is a lie, but the pie is delicious.",
        "To be or not to be, that is the question.",
        "In machine learning, we often need to balance speed and accuracy.",
        "Transformers have revolutionized natural language processing."
    ]
    
    # Ensure we have enough samples
    while len(texts) < num_samples:
        texts.extend(texts[:num_samples-len(texts)])
    
    # Split into train and val
    train_texts = texts[:int(0.8*len(texts))]
    val_texts = texts[int(0.8*len(texts)):]
    
    # Tokenize train data
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # Tokenize val data
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        train_encodings["input_ids"], 
        train_encodings["attention_mask"]
    )
    
    val_dataset = TensorDataset(
        val_encodings["input_ids"], 
        val_encodings["attention_mask"]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"Created test dataset with {len(train_texts)} training samples and {len(val_texts)} validation samples")
    
    return train_dataloader, val_dataloader