"""
Upgrayedd dataset loader module - Handles loading and preparation of datasets
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Callable
from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger("Upgrayedd")

def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer: Any,
    max_length: int = 512,
    custom_dataset_path: Optional[str] = None
) -> DatasetDict:
    """
    Load and prepare a dataset for Upgrayedd training.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length for tokenization
        custom_dataset_path: Path to custom dataset (if dataset_name is "custom")
        
    Returns:
        DatasetDict: Tokenized dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "tiny_shakespeare":
        return load_shakespeare(tokenizer, max_length)
    elif dataset_name == "wikitext":
        return load_wikitext(tokenizer, max_length)
    elif dataset_name == "gutenberg":
        return load_gutenberg(tokenizer, max_length)
    elif dataset_name == "custom":
        return load_custom_dataset(custom_dataset_path, tokenizer, max_length)
    else:
        try:
            # Try to load as a Hugging Face dataset
            return load_hf_dataset(dataset_name, tokenizer, max_length)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            logger.error("Falling back to tiny_shakespeare dataset")
            return load_shakespeare(tokenizer, max_length)

def tokenize_function(tokenizer, max_length):
    """Create a tokenization function with fixed parameters."""
    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    return _tokenize

def load_shakespeare(tokenizer, max_length=512):
    """Load and tokenize the tiny_shakespeare dataset."""
    try:
        # Use trust_remote_code to avoid prompts
        dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function(tokenizer, max_length),
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info("Successfully loaded tiny_shakespeare dataset")
        return tokenized_dataset
    
    except Exception as e:
        logger.error(f"Error loading tiny_shakespeare: {e}")
        # Try fallback to wikitext
        logger.info("Falling back to wikitext dataset")
        return load_wikitext(tokenizer, max_length)

def load_wikitext(tokenizer, max_length=512):
    """Load and tokenize the wikitext dataset."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function(tokenizer, max_length),
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info("Successfully loaded wikitext dataset")
        return tokenized_dataset
    
    except Exception as e:
        logger.error(f"Error loading wikitext: {e}")
        raise

def load_gutenberg(tokenizer, max_length=512):
    """Load and tokenize a selection of Gutenberg books."""
    try:
        # List of Gutenberg book IDs to use
        book_ids = [
            "84", # Frankenstein
            "1342", # Pride and Prejudice
            "98", # A Tale of Two Cities
            "1661", # The Adventures of Sherlock Holmes
            "2701", # Moby Dick
        ]
        
        # Load each book
        all_text = []
        for book_id in book_ids:
            try:
                book_dataset = load_dataset("gutenberg", book_id)
                all_text.extend(book_dataset["train"]["text"])
            except Exception as e:
                logger.warning(f"Failed to load book {book_id}: {e}")
        
        # Create a new dataset with all the text
        dataset = Dataset.from_dict({"text": all_text})
        
        # Split into train and validation
        dataset = dataset.train_test_split(test_size=0.1)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function(tokenizer, max_length),
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info("Successfully loaded gutenberg dataset")
        return tokenized_dataset
    
    except Exception as e:
        logger.error(f"Error loading gutenberg: {e}")
        raise

def load_custom_dataset(path, tokenizer, max_length=512):
    """Load and tokenize a custom dataset from file."""
    try:
        if not os.path.exists(path):
            logger.error(f"Custom dataset path does not exist: {path}")
            raise FileNotFoundError(f"Custom dataset path does not exist: {path}")
        
        # Determine file type
        _, ext = os.path.splitext(path)
        
        if ext.lower() in ['.txt', '.text']:
            # Load as text file
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into chunks (simple approach)
            chunk_size = 1000
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            # Create dataset
            dataset = Dataset.from_dict({"text": text_chunks})
            
        elif ext.lower() in ['.csv']:
            # Load as CSV
            dataset = load_dataset('csv', data_files=path)
            
            # Check for 'text' column
            if 'text' not in dataset['train'].column_names:
                logger.error(f"CSV file must have a 'text' column")
                raise ValueError(f"CSV file must have a 'text' column")
            
        elif ext.lower() in ['.json', '.jsonl']:
            # Load as JSON
            dataset = load_dataset('json', data_files=path)
            
            # Check for 'text' column
            if 'text' not in dataset['train'].column_names:
                logger.error(f"JSON file must have a 'text' field")
                raise ValueError(f"JSON file must have a 'text' field")
            
        else:
            logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Split into train and validation if needed
        if 'validation' not in dataset:
            dataset = dataset['train'].train_test_split(test_size=0.1)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function(tokenizer, max_length),
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"Successfully loaded custom dataset from {path}")
        return tokenized_dataset
    
    except Exception as e:
        logger.error(f"Error loading custom dataset: {e}")
        raise

def load_hf_dataset(dataset_name, tokenizer, max_length=512):
    """Load and tokenize a dataset from Hugging Face."""
    try:
        # Try to load the dataset
        dataset = load_dataset(dataset_name)
        
        # Check if there's a 'text' column
        text_column = None
        if 'train' in dataset:
            columns = dataset['train'].column_names
            if 'text' in columns:
                text_column = 'text'
            else:
                # Try to find a suitable text column
                for col in columns:
                    if 'text' in col.lower() or 'content' in col.lower():
                        text_column = col
                        break
        
        if text_column is None:
            logger.error(f"Could not find a text column in dataset {dataset_name}")
            raise ValueError(f"Could not find a text column in dataset {dataset_name}")
        
        # Create tokenization function for this column
        def _tokenize(examples):
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            _tokenize,
            batched=True,
            remove_columns=[text_column]
        )
        
        logger.info(f"Successfully loaded {dataset_name} dataset")
        return tokenized_dataset
    
    except Exception as e:
        logger.error(f"Error loading {dataset_name}: {e}")
        raise