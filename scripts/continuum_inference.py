#!/usr/bin/env python3
"""
Continuum Inference Bridge - Load pre-trained models and generate responses.

This script provides a bridge between Continuum (TypeScript) and Sentinel-AI (Python).
It loads pre-trained models (TinyLlama, Phi-2, etc.) and generates text responses.

Usage:
    python scripts/continuum_inference.py --model tinyllama-chat --prompt "Hello, how are you?"

    # With conversation context
    python scripts/continuum_inference.py \
        --model tinyllama-chat \
        --messages '[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello!"}]'
"""
import sys
sys.path.insert(0, '/Volumes/FlashGordon/cambrian/sentinel-ai')

import argparse
import json
import torch
from sentinel.models.model_zoo import load_base_model, list_models

def format_chat_prompt(messages: list[dict], model_type: str = 'llama') -> str:
    """
    Format messages into a prompt string appropriate for the model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_type: Type of model (llama, gpt2, etc.)

    Returns:
        Formatted prompt string
    """
    if model_type == 'llama':
        # Llama chat format: <|system|> ... <|user|> ... <|assistant|>
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                prompt_parts.append(f"<|assistant|>\n{content}")

        # End with assistant marker to signal where to generate
        prompt_parts.append("<|assistant|>")
        return "\n".join(prompt_parts)

    elif model_type == 'gpt2':
        # GPT-2: Simple concatenation with newlines
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    else:
        # Default: simple concatenation
        return "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


def generate_response(
    model_key: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 150,
    device: str = 'cpu'
) -> dict:
    """
    Generate a text response using a pre-trained model.

    Args:
        model_key: Model identifier from model zoo
        messages: Conversation history with roles
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        device: 'cpu' or 'cuda'

    Returns:
        Dict with 'text' and 'metadata'
    """
    # Load model
    print(f"Loading model: {model_key}...", file=sys.stderr)
    model, tokenizer, config = load_base_model(model_key, device=device)
    model.eval()

    # Determine model type for prompt formatting
    model_type = getattr(config, 'model_type', 'unknown')

    # Format messages into prompt
    prompt = format_chat_prompt(messages, model_type=model_type)
    print(f"Formatted prompt ({len(prompt)} chars):", file=sys.stderr)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt, file=sys.stderr)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate
    print(f"Generating response (max_tokens={max_tokens}, temperature={temperature})...", file=sys.stderr)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response (remove prompt)
    response = generated_text[len(prompt):].strip()

    # Clean up any remaining special tokens or formatting
    response = response.replace('<|assistant|>', '').replace('<|user|>', '').replace('<|system|>', '').strip()

    return {
        'text': response,
        'metadata': {
            'model': model_key,
            'model_type': model_type,
            'input_tokens': input_ids.shape[1],
            'temperature': temperature,
            'device': device
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Generate text using pre-trained models')
    parser.add_argument('--model', required=True,
                       help='Model key from model zoo (tinyllama-chat, phi-2, distilgpt2, etc.)')
    parser.add_argument('--prompt', type=str,
                       help='Simple prompt (alternative to --messages)')
    parser.add_argument('--messages', type=str,
                       help='JSON array of message objects with role and content')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=150,
                       help='Maximum tokens to generate (default: 150)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("Available models:")
        for key, info in list_models().items():
            print(f"  {key:20s} - {info['size']:6s} - {info['description']}")
        return

    # Parse messages
    if args.messages:
        messages = json.loads(args.messages)
    elif args.prompt:
        # Convert simple prompt to messages format
        messages = [
            {'role': 'system', 'content': 'You are a helpful AI assistant.'},
            {'role': 'user', 'content': args.prompt}
        ]
    else:
        print("Error: Must provide either --prompt or --messages", file=sys.stderr)
        sys.exit(1)

    # Generate response
    try:
        result = generate_response(
            model_key=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            device=args.device
        )

        # Output JSON to stdout (Continuum will parse this)
        print(json.dumps(result, indent=2))

    except Exception as e:
        error_result = {
            'error': str(e),
            'model': args.model
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
