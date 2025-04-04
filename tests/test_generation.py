#!/usr/bin/env python
"""
Simple test script to verify model generation quality.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test model text generation")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt text")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    print(f"Generating text from prompt: '{args.prompt}'")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    generation_config = {
        "max_length": args.max_length,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if args.do_sample:
        generation_config.update({
            "do_sample": True,
            "temperature": args.temperature,
            "num_return_sequences": args.num_return_sequences,
        })
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return
    
    # Print each generated sequence
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\nGenerated text {i+1}:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Also show character codes to check for stop symbols
        print("\nFirst 30 token IDs:")
        print(output[:30])
        
        # Show token strings for better debugging
        tokens = [tokenizer.decode([t]) for t in output[:30]]
        print("\nFirst 30 tokens as strings:")
        print(tokens)

if __name__ == "__main__":
    main()