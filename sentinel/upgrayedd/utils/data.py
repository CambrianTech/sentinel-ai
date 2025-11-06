"""
Data loading utilities for transformer models.

This module provides functions for loading and preparing datasets for training
and evaluation of transformer models.
"""

import torch
import logging
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def load_dataset(*args, **kwargs):
    """
    Load a dataset from the datasets library, handling circular imports.
    
    This is a wrapper around the datasets.load_dataset function that handles
    circular import issues by using a fallback approach when necessary.
    """
    try:
        # Check if datasets module is already imported
        if 'datasets' in sys.modules:
            # If it is, just get the load_dataset function from it
            return sys.modules['datasets'].load_dataset(*args, **kwargs)
        else:
            # First try a direct import
            try:
                from datasets import load_dataset as hf_load_dataset
                return hf_load_dataset(*args, **kwargs)
            except ImportError:
                # If that fails due to circular imports, create a more robust mock
                import types
                import importlib.util
                
                logger.warning("Using mock datasets module to avoid circular imports")
                
                # Check if datasets package is installed
                datasets_spec = importlib.util.find_spec('datasets')
                
                if datasets_spec:
                    # If datasets package exists, load it with a custom loader
                    # that breaks the circular import chain
                    
                    # First create a basic mock module
                    mock_datasets = types.ModuleType('datasets')
                    mock_datasets.ArrowBasedBuilder = type('ArrowBasedBuilder', (), {})
                    mock_datasets.GeneratorBasedBuilder = type('GeneratorBasedBuilder', (), {})
                    mock_datasets.Value = lambda *args, **kwargs: None
                    mock_datasets.Features = lambda *args, **kwargs: {}
                    mock_datasets.__path__ = []
                    
                    # Install the mock module
                    sys.modules['datasets'] = mock_datasets
                    
                    # Now try to load core functionality only
                    try:
                        # Import key functions directly
                        from datasets.load import load_dataset as real_load_dataset
                        mock_datasets.load_dataset = real_load_dataset
                        return real_load_dataset(*args, **kwargs)
                    except ImportError as e:
                        logger.error(f"Failed to import load_dataset: {e}")
                        raise
                else:
                    logger.error("datasets package not found in the environment")
                    raise ImportError("datasets package not found")
    except Exception as e:
        logger.error(f"Cannot import datasets: {e}")
        raise

def load_and_prepare_data(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    dataset_path: Optional[str] = None,
    split_ratio: float = 0.9
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load and prepare dataset for training and evaluation.
    
    Args:
        dataset_name: Name of the dataset to use (wikitext, tiny_shakespeare, etc.)
        tokenizer: Tokenizer to use for encoding text
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length to use
        dataset_path: Path to dataset (if using a local dataset)
        split_ratio: Ratio of training data (rest is used for validation)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Try to load the dataset, with fallback for testing
    try:
        # Load dataset based on name
        if dataset_name.lower() == "wikitext":
            # Load WikiText dataset
            raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
            
            # Split training data if no validation set
            train_data = raw_datasets["train"]
            val_data = raw_datasets["validation"]
            
        elif dataset_name.lower() == "tiny_shakespeare":
            # Load Tiny Shakespeare
            if dataset_path:
                with open(dataset_path, 'r') as f:
                    text = f.read()
            else:
                # Use a small sample if no path provided
                text = """
                ROMEO: What lady is that, which doth enrich the hand
                Of yonder knight?
                SERVANT: I know not, sir.
                ROMEO: O, she doth teach the torches to burn bright!
                It seems she hangs upon the cheek of night
                Like a rich jewel in an Ethiope's ear;
                Beauty too rich for use, for earth too dear!
                """
            
            # Split into train/val
            split_index = int(len(text) * split_ratio)
            train_text = text[:split_index]
            val_text = text[split_index:]
            
            # Create simple datasets
            train_data = [{"text": train_text}]
            val_data = [{"text": val_text}]
            
        elif dataset_name.lower() in ["cnn_dailymail", "cnn-dailymail"]:
            # Load CNN/DailyMail dataset
            raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
            
            # Use article field
            train_data = [{"text": d["article"]} for d in raw_datasets["train"]]
            val_data = [{"text": d["article"]} for d in raw_datasets["validation"]]
            
        elif dataset_name.lower() == "custom" and dataset_path:
            # Load custom dataset from file
            with open(dataset_path, 'r') as f:
                text = f.read()
                
            # Split into train/val
            split_index = int(len(text) * split_ratio)
            train_text = text[:split_index]
            val_text = text[split_index:]
            
            # Create simple datasets
            train_data = [{"text": train_text}]
            val_data = [{"text": val_text}]
            
        else:
            # Try loading as a HuggingFace dataset
            try:
                raw_datasets = load_dataset(dataset_name)
                
                # Try to find the right text field
                text_field = None
                for field in ["text", "content", "document", "article"]:
                    if field in raw_datasets["train"].features:
                        text_field = field
                        break
                
                if text_field is None:
                    # If no standard text field found, use the first string field
                    for name, field in raw_datasets["train"].features.items():
                        if field.dtype == "string":
                            text_field = name
                            break
                
                if text_field is None:
                    logger.warning(f"No text field found in dataset {dataset_name}")
                    return _create_dummy_dataloaders(tokenizer, batch_size)
                
                # Extract text data
                train_data = [{"text": d[text_field]} for d in raw_datasets["train"]]
                val_data = [{"text": d[text_field]} for d in raw_datasets["validation" if "validation" in raw_datasets else "test"]]
                
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {e}")
                return _create_dummy_dataloaders(tokenizer, batch_size)
    
    except Exception as e:
        logger.error(f"Error preparing dataset {dataset_name}: {e}")
        return _create_dummy_dataloaders(tokenizer, batch_size)
    
    # Tokenize data
    train_encodings = _tokenize_data(train_data, tokenizer, max_length)
    val_encodings = _tokenize_data(val_data, tokenizer, max_length)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"]
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"]
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader

def _tokenize_data(
    data: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Tokenize data for training.
    
    Args:
        data: List of dictionaries with text data
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized data
    """
    # Extract text from data
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "text" in data[0]:
        texts = [item["text"] for item in data]
    elif isinstance(data, list) and all(isinstance(item, str) for item in data):
        texts = data
    else:
        # Handle the case where data is a single string (e.g., from a file)
        if len(data) == 1 and isinstance(data[0], dict) and "text" in data[0]:
            text = data[0]["text"]
            
            # Split long text into chunks of max_length tokens
            tokenized = tokenizer(text, return_tensors="pt")
            input_ids = tokenized["input_ids"].squeeze()
            
            # Handling very long texts: chunk into sequences
            num_tokens = input_ids.size(0)
            num_chunks = (num_tokens - 1) // max_length + 1
            
            chunked_input_ids = []
            chunked_attention_mask = []
            
            for i in range(num_chunks):
                start_idx = i * max_length
                end_idx = min((i + 1) * max_length, num_tokens)
                
                chunk = input_ids[start_idx:end_idx]
                if len(chunk) < 10:  # Skip very small chunks
                    continue
                    
                # Pad if necessary
                if len(chunk) < max_length:
                    pad_length = max_length - len(chunk)
                    chunk = torch.cat([chunk, torch.full((pad_length,), tokenizer.pad_token_id)])
                
                chunked_input_ids.append(chunk)
                chunked_attention_mask.append(torch.ones_like(chunk))
            
            # Stack chunks
            return {
                "input_ids": torch.stack(chunked_input_ids),
                "attention_mask": torch.stack(chunked_attention_mask)
            }
        else:
            # Fallback
            texts = ["Test data for tokenization."]
    
    # Standard tokenization for list of texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

def _create_dummy_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create dummy dataloaders for testing.
    
    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.warning("Creating dummy dataloaders for testing")
    
    # Create dummy data
    dummy_texts = [
        "This is a test sentence for training.",
        "Another example for the training dataset.",
        "Machine learning models need training data."
    ] * 5
    
    # Tokenize
    encodings = tokenizer(
        dummy_texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"][:5],
        encodings["attention_mask"][:5]
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader