#!/usr/bin/env python
"""
Neural Plasticity Test Script

This script tests the text sample display functionality added to the neural plasticity module.
"""

import os
import sys
import torch
from pathlib import Path
import argparse

# Add parent directory to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.neural_plasticity.experiment import NeuralPlasticityExperiment
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Test neural plasticity with text sample display")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--pruning_level", type=float, default=0.3, help="Pruning level")
    parser.add_argument("--strategy", type=str, default="entropy", help="Pruning strategy")
    parser.add_argument("--sample_interval", type=int, default=10, help="Sample interval")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Set pad token for GPT-2 style models which don't have one by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Create tiny dataset for testing
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be trained on various datasets.",
        "Neural networks are a subset of machine learning.",
        "Deep learning has revolutionized artificial intelligence.",
        "The transformer architecture introduced attention mechanisms."
    ]
    
    # Create encodings
    train_encodings = tokenizer(
        train_texts, 
        padding=True, 
        truncation=True, 
        max_length=64, 
        return_tensors="pt"
    )
    
    # Simple dataset class
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item
            
        def __len__(self):
            return len(self.encodings.input_ids)
    
    # Create dataset and dataloader
    train_dataset = TextDataset(train_encodings)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True
    )
    eval_dataloader = train_dataloader  # Use same for eval in this test
    
    # Create experiment
    experiment = NeuralPlasticityExperiment(
        model_name=args.model,
        pruning_level=args.pruning_level,
        pruning_strategy=args.strategy,
        show_samples=True,
        sample_interval=args.sample_interval,
        tokenizer=tokenizer
    )
    
    # Setup the experiment with our own model and dataloaders
    experiment.model = model
    experiment.train_dataloader = train_dataloader
    experiment.validation_dataloader = eval_dataloader  # Note the correct attribute name
    experiment.device = next(model.parameters()).device
    
    # Analyze attention and run pruning cycle
    experiment.analyze_attention()
    
    # Run pruning cycle with sample display
    results = experiment.run_pruning_cycle(
        training_steps=50
    )
    
    print("\nExperiment complete!")
    print(f"Final perplexity: {results.get('final_metrics', {}).get('perplexity', 0):.2f}")
    
    # Calculate improvement
    baseline = experiment.baseline_perplexity or 0
    final = results.get('final_metrics', {}).get('perplexity', 0)
    improvement = ((baseline - final) / baseline) * 100 if baseline else 0
    
    print(f"Perplexity improvement: {improvement:.2f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())