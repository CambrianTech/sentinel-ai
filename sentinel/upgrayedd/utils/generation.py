"""
Text generation utilities for transformer models.

This module provides functions for generating text from transformer models,
including interactive generation and batch generation.
"""

import torch
import logging
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    device: Optional[str] = None
) -> str:
    """
    Generate text from a transformer model.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer to use
        prompt: The prompt to generate from
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_return_sequences: Number of sequences to generate
        device: Device to use (inferred from model if None)
        
    Returns:
        Generated text
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with error handling
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                # For backward compatibility with older HF versions:
                no_repeat_ngram_size=3
            )
    except Exception as e:
        logger.warning(f"Error in generate with advanced parameters: {e}")
        # Fallback to simpler generate call
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    do_sample=True
                )
        except Exception as e2:
            logger.error(f"Fallback generation also failed: {e2}")
            return f"{prompt} [ERROR: Generation failed]"
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def batch_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    batch_size: int = 4,
    device: Optional[str] = None
) -> List[str]:
    """
    Generate text for multiple prompts in batches.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer to use
        prompts: List of prompts to generate from
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        batch_size: Number of prompts to process at once
        device: Device to use (inferred from model if None)
        
    Returns:
        List of generated texts
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    # Process prompts in batches
    generated_texts = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Encode prompts
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        # Generate
        try:
            with torch.no_grad():
                batch_outputs = model.generate(
                    batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
        except Exception as e:
            logger.warning(f"Error in batch generate: {e}")
            # Fall back to one-by-one generation
            batch_outputs = []
            for prompt in batch_prompts:
                single_input = tokenizer(prompt, return_tensors="pt").to(device)
                try:
                    with torch.no_grad():
                        output = model.generate(
                            single_input["input_ids"],
                            max_length=max_length,
                            do_sample=True
                        )
                    batch_outputs.append(output[0])
                except Exception:
                    # If even single generation fails, return the prompt with error
                    dummy_output = tokenizer.encode(f"{prompt} [ERROR: Generation failed]", return_tensors="pt")[0]
                    batch_outputs.append(dummy_output)
        
        # Decode outputs
        for output in batch_outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
    
    return generated_texts

def interactive_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: Optional[str] = None,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[str] = None
) -> str:
    """
    Interactive text generation with optional prompt.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer to use
        prompt: Optional prompt (if None, will ask for input)
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to use (inferred from model if None)
        
    Returns:
        Generated text
    """
    # Get prompt if not provided
    if prompt is None:
        prompt = input("Enter a prompt: ")
    
    # Generate text
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        device=device
    )
    
    # Print the result
    print(f"\nGenerated text:\n{generated}")
    
    return generated