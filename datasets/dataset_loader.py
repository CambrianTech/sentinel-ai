# datasets/dataset_loader.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_dataset(model_name: str, dataset_name="tiny_shakespeare", block_size=128, limit_train=1000, limit_val=200):
    """
    Loads a dataset and tokenizes it into blocks of input IDs using the tokenizer for `model_name`.
    Returns tokenized torch tensors for train and validation splits.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensures compatibility with models needing padding

    # Load dataset
    if dataset_name == "tiny_shakespeare":
        dataset = load_dataset("tiny_shakespeare")
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif dataset_name == "tiny_stories":
        dataset = load_dataset("roneneldan/TinyStories")
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
