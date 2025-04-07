"""
Text Generator for Pruning Experiments

This module handles text generation with transformer models, providing utilities
for generating and evaluating text from the models.
"""

import torch
from typing import Any, Optional


def generate_text(model: torch.nn.Module, tokenizer: Any, prompt: str, max_length: int = 100) -> str:
    """
    Generate text from a model.
    
    Args:
        model: The model to generate text from
        tokenizer: The tokenizer to use
        prompt: The prompt to generate from
        max_length: The maximum length of the generated text
        
    Returns:
        The generated text
    """
    # Get the device
    device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate - don't pass attention_mask to generate method to avoid issues
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def interactive_generate(model: torch.nn.Module, tokenizer: Any, prompt: Optional[str] = None, max_length: int = 100) -> str:
    """
    Interactive text generation, allowing user to provide prompt.
    
    Args:
        model: The model to generate text from
        tokenizer: The tokenizer to use
        prompt: Optional prompt to generate from (if None, will prompt user)
        max_length: The maximum length of the generated text
        
    Returns:
        The generated text
    """
    # If no prompt is provided, ask the user
    if prompt is None:
        prompt = input("Enter a prompt: ")
    
    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, max_length)
    
    # Print the result
    print(f"\nGenerated text:\n{generated_text}")
    
    return generated_text


def batch_generate(model: torch.nn.Module, tokenizer: Any, prompts: list, max_length: int = 100) -> list:
    """
    Generate text for a batch of prompts.
    
    Args:
        model: The model to generate text from
        tokenizer: The tokenizer to use
        prompts: List of prompts to generate from
        max_length: The maximum length of the generated text
        
    Returns:
        List of generated texts
    """
    results = []
    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, max_length)
        results.append(text)
    return results