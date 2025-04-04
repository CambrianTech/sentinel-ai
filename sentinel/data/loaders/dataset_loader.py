"""
Dataset loading utilities for adaptive transformers.

This module provides utilities for loading various datasets for the
adaptive transformer training procedure. It supports common text datasets
and provides a consistent interface for working with them.
"""

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import random
import numpy as np
from transformers import AutoTokenizer
import os
import logging

logger = logging.getLogger(__name__)

# Common dataset names and their HuggingFace dataset IDs
DATASET_MAPPING = {
    "wikitext": "wikitext-103-v1",
    "c4": "c4",
    "bookcorpus": "bookcorpus",
    "openwebtext": "openwebtext",
    "wikipedia": "wikipedia",
    "squad": "squad",
    "glue": "glue",
}

def get_dataset_id(dataset_name):
    """Get the HuggingFace dataset ID for a given dataset name."""
    if dataset_name in DATASET_MAPPING:
        return DATASET_MAPPING[dataset_name]
    return dataset_name

class TextDataset(Dataset):
    """
    Dataset for adaptive transformer training with efficient chunking.
    
    This dataset preprocesses text data into chunks of a specified sequence length,
    with optional overlap between chunks to improve learning of long-range dependencies.
    """
    def __init__(self, texts, tokenizer, seq_length=128, stride=None):
        """
        Initialize the text dataset.
        
        Args:
            texts: List of text documents
            tokenizer: Tokenizer to use for encoding
            seq_length: Maximum sequence length for each chunk
            stride: Stride between chunks (if None, no overlap)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        
        # Tokenize all texts and create chunks
        self.examples = []
        
        for text in texts:
            tokenized = tokenizer(text, truncation=False, return_attention_mask=False)
            input_ids = tokenized["input_ids"]
            
            # Create chunks with optional overlap
            for i in range(0, len(input_ids) - seq_length + 1, self.stride):
                self.examples.append(input_ids[i:i + seq_length])
        
        print(f"Created {len(self.examples)} chunks from {len(texts)} documents")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Convert to tensor and get attention mask
        item = torch.tensor(self.examples[idx])
        mask = torch.ones_like(item)
        
        return {
            "input_ids": item,
            "attention_mask": mask,
            "labels": item
        }

def load_dataset(dataset_name, tokenizer, seq_length=128, stride=None, 
                small_subset=False, subset_size=1000, split="train", 
                dry_run=False, max_documents=None):
    """
    Load and prepare a dataset for training.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for tokenization
        seq_length: Maximum sequence length for each training example
        stride: Stride between chunks (if None, uses seq_length)
        small_subset: Whether to use a small subset for testing
        subset_size: Size of the subset if small_subset is True
        split: Dataset split to use ('train', 'validation', 'test')
        dry_run: If True, only loads a tiny subset for testing
        max_documents: Maximum number of documents to load (None for all)
        
    Returns:
        TextDataset containing the tokenized examples
    """
    # For dry runs, load a very tiny subset
    if dry_run:
        small_subset = True
        subset_size = min(100, subset_size)
        logger.info(f"Dry run enabled, using tiny subset of {subset_size} examples")
    
    dataset_id = get_dataset_id(dataset_name)
    logger.info(f"Loading dataset: {dataset_name} (ID: {dataset_id})")
    
    try:
        # Download and load the dataset from HuggingFace
        raw_dataset = hf_load_dataset(dataset_id, split=split)
        
        # Extract text field based on dataset
        if dataset_name == "wikitext" or "wikitext" in dataset_id:
            texts = raw_dataset["text"]
            # Filter out empty or very short texts
            texts = [text for text in texts if len(text.strip()) > 50]
        elif dataset_name == "c4" or "c4" in dataset_id:
            texts = raw_dataset["text"]
        elif dataset_name in ["bookcorpus", "openwebtext"]:
            texts = raw_dataset["text"]
        else:
            # Use the first text field we can find
            for key in raw_dataset.features.keys():
                if raw_dataset.features[key].dtype == 'string':
                    texts = raw_dataset[key]
                    logger.info(f"Using field '{key}' as text")
                    break
            else:
                raise ValueError(f"Could not find text field in dataset {dataset_name}")
        
        # Limit the number of documents if specified
        if max_documents is not None:
            texts = texts[:max_documents]
            logger.info(f"Limited to {max_documents} documents")
        
        # Use a small subset for testing or dry runs
        if small_subset:
            if len(texts) > subset_size:
                texts = texts[:subset_size]
                logger.info(f"Using small subset of {subset_size} documents")
        
        # Create the dataset
        dataset = TextDataset(texts, tokenizer, seq_length, stride)
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise

def load_dataset_for_testing(tokenizer, seq_length=128):
    """
    Create a tiny synthetic dataset for testing.
    
    Useful for quickly testing training loops and model functionality
    without downloading large datasets.
    
    Args:
        tokenizer: Tokenizer to use
        seq_length: Sequence length for examples
        
    Returns:
        A small TensorDataset with synthetic examples
    """
    # Create a few random sequences
    random_texts = [
        "The quick brown fox jumps over the lazy dog. This is a test. " * 10,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10,
        "Machine learning models can process natural language effectively. " * 10
    ]
    
    # Tokenize them
    examples = []
    for text in random_texts:
        tokenized = tokenizer(text, truncation=True, max_length=seq_length, 
                             padding="max_length", return_tensors="pt")
        examples.append({
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0]
        })
    
    # Create tensors
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples])
    labels = torch.stack([ex["labels"] for ex in examples])
    
    return TensorDataset(input_ids, attention_mask, labels)

def create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading
        
    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )