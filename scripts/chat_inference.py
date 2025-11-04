#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lightweight chat inference for Sentinel-AI integration with Continuum.

Uses the WORKING manual generation pattern from /tmp/train_minimal_memory.py
(simple greedy decoding without HuggingFace GenerationMixin bugs).

Usage:
    python chat_inference.py --prompt "Your question here" --model distilgpt2 --max-tokens 150
"""

import sys
import json
import argparse
import torch
import warnings
from pathlib import Path
import os

# CRITICAL: Redirect ALL stdout to stderr EXCEPT our final JSON output
# This prevents debug messages from breaking JSON parsing in TypeScript
_original_stdout = sys.stdout
_original_stderr = sys.stderr

class StderrRedirector:
    """Redirects all print() calls to stderr, preserving only explicit JSON writes to stdout"""
    def write(self, text):
        _original_stderr.write(text)

    def flush(self):
        _original_stderr.flush()

# Redirect stdout to stderr (we'll restore it only for final JSON output)
sys.stdout = StderrRedirector()

# Suppress warnings to stderr (they cause execAsync to throw)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Also suppress stdout at OS level for any C libraries
devnull = os.open(os.devnull, os.O_WRONLY)

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from models.loaders.gpt2_loader_clean import load_adaptive_model_gpt_clean


def generate_chat_response(model_name: str, prompt: str, max_tokens: int = 150, temperature: float = 0.7, device: str = "cpu") -> dict:
    """
    Generate a chat response using Sentinel-AI's adaptive model.

    Uses the WORKING manual generation pattern from /tmp/train_minimal_memory.py (lines 88-117)
    This pattern works because it does simple greedy decoding without calling `.generate()`.

    Args:
        model_name: HuggingFace model name (e.g., "distilgpt2", "gpt2")
        prompt: User's message/question
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.1-1.0) - currently unused (greedy decoding)
        device: "cpu" or "cuda"

    Returns:
        dict with 'response', 'error', 'model_info'
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load config to get context window size
        config = AutoConfig.from_pretrained(model_name)
        max_context = getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 1024))

        # CRITICAL: Validate and truncate prompt if needed BEFORE loading model
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        available_for_prompt = max_context - max_tokens  # Reserve space for generation

        if len(prompt_tokens) > available_for_prompt:
            # Truncate from beginning (keep most recent context)
            prompt_tokens = prompt_tokens[-available_for_prompt:]
            prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            # Ensure it ends with "Assistant:" to prompt response
            if not prompt.strip().endswith("Assistant:"):
                prompt = prompt.rstrip() + "\n\nAssistant:"

        # Load baseline model first
        baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
        baseline_model.eval()

        # Wrap with adaptive wrapper (this is what training code does)
        model = load_adaptive_model_gpt_clean(model_name, baseline_model, config, device=device, quiet=True)

        # Free baseline model memory
        del baseline_model

        # Set model to eval mode
        model.eval()

        # Tokenize input (already validated above)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Manual generation loop with sampling + repetition penalty
        with torch.no_grad():
            for step in range(max_tokens):
                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits

                # Get next token logits
                next_token_logits = logits[0, -1, :].clone()

                # Apply repetition penalty to previously generated tokens
                repetition_penalty = 1.2
                for prev_token_id in set(input_ids[0].tolist()):
                    next_token_logits[prev_token_id] /= repetition_penalty

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-k filtering (k=50)
                top_k = 50
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample from filtered distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Stop if we hit repetition (detect loops early)
                # Check for 3-gram repetition
                if step >= 6:
                    last_tokens = input_ids[0, -6:].tolist()
                    if last_tokens[:3] == last_tokens[3:6]:
                        break

                # Also check for word-level repetition (more aggressive)
                if step >= 10:
                    # Decode last 10 tokens and check for repeated phrases
                    recent_text = tokenizer.decode(input_ids[0, -10:], skip_special_tokens=True)
                    words = recent_text.split()
                    if len(words) >= 4:
                        # Check if last 2 words repeat
                        if words[-2:] == words[-4:-2]:
                            break

        # Decode output
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Extract only the generated part (remove prompt)
        response = generated_text[len(prompt):].strip()

        # Get model stats
        num_params = sum(p.numel() for p in model.parameters())

        # Count active heads (if adaptive model)
        active_heads = 0
        total_heads = 0
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                attn_module = block.attn
                total_heads += attn_module.num_heads
                for head_idx in range(attn_module.num_heads):
                    if attn_module.gate.data[head_idx] > 0.1:
                        active_heads += 1

        return {
            "success": True,
            "response": response,
            "model_info": {
                "name": model_name,
                "total_params": num_params,
                "active_heads": active_heads,
                "total_heads": total_heads,
                "pruning_ratio": f"{100.0 * (total_heads - active_heads) / total_heads:.1f}%" if total_heads > 0 else "N/A",
                "device": device,
                "temperature": temperature,
                "note": "SUCCESS: Using Sentinel adaptive model with real pruning (manual generation)"
            }
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "response": None
        }


def main():
    parser = argparse.ArgumentParser(description="Generate chat responses using Sentinel-AI")
    parser.add_argument("--prompt", type=str, help="User's message/question")
    parser.add_argument("--messages-file", type=str, help="Path to JSON file containing messages")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name (default: distilgpt2)")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (currently unused)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--prompt-format", type=str, default="base", help="Prompt format (base, chatml, llama2, alpaca)")
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    # Get prompt from either --prompt or --messages-file
    if args.messages_file:
        with open(args.messages_file, 'r') as f:
            messages = json.load(f)

        # Format messages as a conversation for base models (not instruction-tuned)
        # Base models like GPT-2 just continue text, so we format as:
        # "System: ...\nUser: ...\nAssistant: ...\nUser: ...\nAssistant:"
        # This way the model knows to continue as "Assistant"
        conversation_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Format role for readability
            if role == 'system':
                conversation_parts.append(f"System: {content}")
            elif role == 'user':
                conversation_parts.append(f"User: {content}")
            elif role == 'assistant':
                conversation_parts.append(f"Assistant: {content}")

        # Add the final "Assistant:" prompt to signal the model should respond
        conversation_parts.append("Assistant:")

        prompt = "\n\n".join(conversation_parts)
    elif args.prompt:
        # For direct prompts, format as simple user/assistant exchange
        prompt = f"User: {args.prompt}\n\nAssistant:"
    else:
        print(json.dumps({"error": "Either --prompt or --messages-file must be provided"}))
        sys.exit(1)

    result = generate_chat_response(
        model_name=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device
    )

    # ALWAYS output JSON for Continuum compatibility
    # Restore stdout ONLY for final JSON output
    sys.stdout = _original_stdout

    if result["success"]:
        # Format for Continuum adapter
        continuum_response = {
            "text": result["response"],
            "metadata": result["model_info"]
        }
        print(json.dumps(continuum_response))
    else:
        error_response = {
            "error": result["error"]
        }
        if "traceback" in result:
            error_response["traceback"] = result["traceback"]
        print(json.dumps(error_response))
        sys.exit(1)


if __name__ == "__main__":
    main()
