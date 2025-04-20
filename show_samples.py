#!/usr/bin/env python
"""
Script to demonstrate the sample display feature in neural plasticity.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.neural_plasticity.training import PlasticityTrainer

class SimpleDataset(Dataset):
    def __init__(self, input_ids, attention_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        item = {"input_ids": self.input_ids[idx]}
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        item["labels"] = self.input_ids[idx].clone()
        return item

def main():
    print("\n=== Testing Sample Display Feature ===\n")
    
    # Load model and tokenizer
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be trained on various datasets.",
        "Neural networks are a subset of machine learning.",
        "Deep learning has revolutionized artificial intelligence.",
        "The transformer architecture introduced attention mechanisms."
    ]
    
    # Tokenize texts
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    
    # Create dataset and dataloader
    dataset = SimpleDataset(encodings["input_ids"], encodings["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create trainer with sample display enabled
    trainer = PlasticityTrainer(model=model, learning_rate=5e-5)
    trainer.prepare_optimizer(pruned_heads=[], warmup_steps=2, total_steps=10)
    
    # Run a few training steps with sample display
    print("Running training with sample display (displaying every step):")
    print("-" * 80)
    
    for step in range(5):
        # Get batch
        batch = next(iter(dataloader))
        
        # Train with sample display
        result = trainer.train_step(
            batch, 
            show_samples=True,
            tokenizer=tokenizer,
            sample_step=1
        )
        
        # Display the step results
        print(f"\nStep {step} loss: {result['loss']:.4f}")
        
        # If there's sample data, display it using the trainer's method
        if result.get("sample") is not None:
            trainer._display_sample(step, result["sample"])
        
        print("-" * 80)
    
    print("\nSample display test completed successfully!")
    
if __name__ == "__main__":
    main()