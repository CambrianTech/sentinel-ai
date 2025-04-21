"""Text generation utilities for pruned models

This module provides text generation functions for interacting with pruned models.
"""

import torch

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, num_return_sequences=1):
    """Generate text from a model given a prompt.
    
    Args:
        model: A HuggingFace model
        tokenizer: A HuggingFace tokenizer
        prompt: The text prompt to start generation from
        max_length: Maximum length of the generated text
        temperature: Controls randomness in generation (higher = more random)
        num_return_sequences: Number of sequences to return
        
    Returns:
        String with generated text if num_return_sequences=1,
        otherwise a list of generated text strings
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    try:
        # Generate text with safety checks
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated sequences
        generated_texts = []
        for seq in output_sequences:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            generated_texts.append(text)
        
        # Return a single string or a list based on num_return_sequences
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
            
    except Exception as e:
        print(f"Error generating text: {e}")
        # Provide a fallback response
        return prompt + " [Error during text generation]"


def interactive_generate(model, tokenizer, prompt=None, max_length=100):
    """Generate text interactively, prompting the user for input if needed.
    
    Args:
        model: A HuggingFace model
        tokenizer: A HuggingFace tokenizer
        prompt: Optional initial prompt, or None to prompt the user
        max_length: Maximum length of the generated text
        
    Returns:
        The generated text
    """
    # Get prompt from user if not provided
    if prompt is None:
        prompt = input("Enter a prompt (or leave empty for default): ")
        if not prompt:
            prompt = "Once upon a time"
    
    # Generate text
    generated_text = generate_text(
        model, 
        tokenizer, 
        prompt, 
        max_length=max_length, 
        temperature=0.7
    )
    
    # Print and return
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    return generated_text
