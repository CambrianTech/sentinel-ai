#!/usr/bin/env python
"""
Test script for run_experiment.py.

This script tests run_experiment.py with a controlled environment
to avoid circular imports and minimize dependencies.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main test function."""
    logger.info("Testing run_experiment.py with simplified setup")
    
    # Create output directory
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")
    
    # Import the optimizer directly
    from sentinel.upgrayedd.optimizer.adaptive_optimizer import AdaptiveOptimizer, AdaptiveOptimizerConfig
    
    # Create a minimal config
    config = AdaptiveOptimizerConfig(
        model_name="distilgpt2",
        pruning_ratio=0.2,
        strategy="random",
        epochs_per_cycle=1,
        max_cycles=1,
        device="cpu",
        output_dir=output_dir,
        batch_size=2,
        dataset="wikitext"
    )
    
    # Log the configuration
    logger.info(f"Testing with config: {config.to_dict()}")
    
    try:
        # Create the optimizer
        logger.info("Creating AdaptiveOptimizer...")
        optimizer = AdaptiveOptimizer(config)
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = optimizer.load_model()
        
        # Set padding token for GPT-2 models
        if tokenizer.pad_token is None:
            logger.info("Setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token
            # Also set it at the model level
            model.config.pad_token_id = model.config.eos_token_id
            
        # Set TOKENIZERS_PARALLELISM environment variable to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
        # Create dummy data
        logger.info("Creating dummy dataloaders...")
        
        # Prepare some simple text samples
        texts = [
            "This is a test sentence for training.",
            "Another example for the training dataset.",
            "Transformer models use attention mechanisms."
        ] * 5
        
        # Tokenize the texts with explicit attention mask
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        
        # Explicitly set the attention mask if pad_token == eos_token
        if tokenizer.pad_token == tokenizer.eos_token:
            attention_mask = torch.ones_like(encodings["input_ids"])
            for i, seq in enumerate(encodings["input_ids"]):
                # Find position of first pad/eos token
                pad_pos = (seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(pad_pos) > 0:
                    # Set attention mask to 0 after the first pad/eos token
                    attention_mask[i, pad_pos[0]+1:] = 0
            encodings["attention_mask"] = attention_mask
        
        # Create a simple dataset
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
            batch_size=config.batch_size,
            shuffle=True
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Set the dataloaders in the optimizer
        optimizer.train_dataloader = train_dataloader
        optimizer.val_dataloader = val_dataloader
        
        # Run one optimization cycle
        logger.info("Running optimization cycle...")
        results = optimizer.run_optimization_cycle()
        
        # Log results
        logger.info(f"Cycle completed. Pruned {len(results['pruned_heads'])} heads.")
        logger.info(f"Final perplexity: {results['final_metrics']['perplexity']:.2f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)