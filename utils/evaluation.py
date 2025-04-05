"""
Evaluation utilities for language models.

This module provides functions for evaluating language models,
including text generation, coherence assessment, and benchmark utilities.
"""

import os
import torch
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from tqdm import tqdm

def generate_text_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str] = None,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_samples: int = 1,
    device: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Generate text samples from a language model for evaluation.
    
    Args:
        model: The language model to generate text with
        tokenizer: The tokenizer for the model
        prompts: List of prompts to generate from (if None, uses default prompts)
        max_length: Maximum length of generated text
        temperature: Temperature for sampling (higher = more random)
        top_p: Top-p sampling parameter (nucleus sampling)
        num_samples: Number of samples to generate per prompt
        device: Device to run generation on (defaults to model's device)
        
    Returns:
        List of dictionaries containing prompts and generated text
    """
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The scientists discovered",
            "In a world where technology",
            "The history of literature",
            "When considering the implications"
        ]
    
    # Set model to eval mode for generation
    model.eval()
    
    # Determine device if not specified
    if device is None:
        device = next(model.parameters()).device
    
    generated_samples = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                # Tokenize prompt with attention mask
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                # Generate continuations
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode and add to samples
                for i, output in enumerate(outputs):
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    generated_samples.append({
                        "prompt": prompt,
                        "generated": generated_text,
                        "sample_idx": i
                    })
                
            except Exception as e:
                generated_samples.append({
                    "prompt": prompt,
                    "error": str(e)
                })
    
    return generated_samples

def save_generated_samples(
    samples_dict: Dict[str, List[Dict[str, str]]],
    output_path: str,
    title: str = "GENERATED TEXT SAMPLES"
) -> None:
    """
    Save generated text samples to a readable text file.
    
    Args:
        samples_dict: Dictionary mapping configuration names to lists of sample dicts
        output_path: Path to save the output file
        title: Title for the output file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        for config, samples in samples_dict.items():
            f.write(f"Configuration: {config}\n")
            f.write("-" * 50 + "\n")
            
            for sample in samples:
                f.write(f"Prompt: {sample['prompt']}\n\n")
                
                if "generated" in sample:
                    f.write(f"{sample['generated']}\n\n")
                elif "error" in sample:
                    f.write(f"ERROR: {sample['error']}\n\n")
                
                if "metrics" in sample:
                    f.write("Metrics:\n")
                    for metric, value in sample["metrics"].items():
                        f.write(f"- {metric}: {value}\n")
                    f.write("\n")
                
                f.write("-" * 30 + "\n")
            
            f.write("\n\n")
    
    print(f"Generated samples saved to {output_path}")

def evaluate_text_coherence(
    samples: List[Dict[str, str]],
    methods: List[str] = ["length", "repetition", "prompt_relevance"]
) -> List[Dict[str, Any]]:
    """
    Evaluate the coherence of generated text using various metrics.
    
    Args:
        samples: List of dictionaries containing generated text samples
        methods: List of evaluation methods to use
        
    Returns:
        List of dictionaries with original samples plus added metrics
    """
    evaluated_samples = []
    
    for sample in samples:
        # Skip samples with errors
        if "error" in sample:
            evaluated_samples.append(sample)
            continue
            
        if "generated" not in sample:
            sample["error"] = "No generated text found"
            evaluated_samples.append(sample)
            continue
            
        text = sample["generated"]
        prompt = sample["prompt"]
        
        # Create metrics dictionary
        metrics = {}
        
        # Calculate metrics based on requested methods
        if "length" in methods:
            metrics["length"] = len(text)
            metrics["tokens"] = len(text.split())
            
        if "repetition" in methods:
            # Count repeated n-grams
            words = text.split()
            repetition_2gram = 0
            repetition_3gram = 0
            
            if len(words) > 2:
                # Count bigram repetitions
                bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
                unique_bigrams = set(bigrams)
                repetition_2gram = 1.0 - (len(unique_bigrams) / max(1, len(bigrams)))
                
            if len(words) > 3:
                # Count trigram repetitions
                trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                unique_trigrams = set(trigrams)
                repetition_3gram = 1.0 - (len(unique_trigrams) / max(1, len(trigrams)))
                
            metrics["repetition_2gram"] = repetition_2gram
            metrics["repetition_3gram"] = repetition_3gram
            
        if "prompt_relevance" in methods:
            # Simple prompt relevance - check if prompt words appear in generation
            prompt_words = set(prompt.lower().split())
            gen_words = set(text.lower().split())
            overlap = len(prompt_words.intersection(gen_words))
            relevance = overlap / max(1, len(prompt_words))
            metrics["prompt_relevance"] = relevance
        
        # Add metrics to the sample
        sample_with_metrics = sample.copy()
        sample_with_metrics["metrics"] = metrics
        evaluated_samples.append(sample_with_metrics)
    
    return evaluated_samples

def batch_evaluate_models(
    models_dict: Dict[str, Tuple[torch.nn.Module, Any]],
    eval_prompts: List[str] = None,
    output_path: Optional[str] = None,
    max_length: int = 100,
    temperature: float = 0.7,
    include_metrics: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch evaluate multiple models and save their generated text samples.
    
    Args:
        models_dict: Dictionary mapping configuration names to (model, tokenizer) tuples
        eval_prompts: List of prompts to use for evaluation (if None, uses defaults)
        output_path: Path to save results (if None, doesn't save)
        max_length: Maximum generation length
        temperature: Temperature for generation
        include_metrics: Whether to include coherence metrics
        
    Returns:
        Dictionary of all generated samples with metrics
    """
    all_samples = {}
    
    for config_name, (model, tokenizer) in tqdm(models_dict.items(), desc="Evaluating models"):
        print(f"\nGenerating samples for configuration: {config_name}")
        
        # Generate samples
        samples = generate_text_samples(
            model=model,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            max_length=max_length,
            temperature=temperature
        )
        
        # Add metrics if requested
        if include_metrics:
            samples = evaluate_text_coherence(samples)
        
        # Print a sample
        for sample in samples[:1]:  # Print just the first sample
            if "generated" in sample:
                print(f"\nSample for '{sample['prompt']}':\n{sample['generated'][:200]}...\n")
        
        all_samples[config_name] = samples
    
    # Save if output path provided
    if output_path is not None:
        save_generated_samples(all_samples, output_path)
    
    return all_samples

if __name__ == "__main__":
    # Example usage (for testing)
    try:
        # Use mock data for testing without requiring model loading
        print("Testing with mock data instead of loading real model")
        
        # Create mock samples
        mock_samples = [
            {
                "prompt": "Once upon a time",
                "generated": "Once upon a time there was a kingdom far away where dragons and humans lived in harmony. The king was wise and just, ruling with both compassion and strength."
            },
            {
                "prompt": "The scientists discovered",
                "generated": "The scientists discovered a new species of deep sea creatures living near hydrothermal vents. These organisms had evolved unique adaptations to survive in extreme conditions."
            }
        ]
        
        # Test evaluation
        print("Testing text evaluation...")
        evaluated = evaluate_text_coherence(mock_samples)
        
        # Print a sample with metrics
        if evaluated and "generated" in evaluated[0]:
            sample = evaluated[0]
            print(f"\nGenerated (truncated): {sample['generated'][:100]}...")
            if "metrics" in sample:
                print("\nMetrics:")
                for metric, value in sample["metrics"].items():
                    print(f"- {metric}: {value:.4f}" if isinstance(value, float) else f"- {metric}: {value}")
        
        # Test saving (to a tmp file)
        print("\nTesting sample saving...")
        output_file = "/tmp/test_generated_samples.txt"
        test_dict = {"test_model": evaluated}
        save_generated_samples(test_dict, output_file)
        print(f"Saved test samples to {output_file}")
        
        print("\nAll tests completed successfully!")
    
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()