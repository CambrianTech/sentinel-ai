import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.loaders.loader import load_baseline_model, load_adaptive_model

SUPPORTED_MODELS = [
    "distilgpt2", 
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-j-6B",
    # Add other HuggingFace-supported models here
]

def generate_text(model, tokenizer, prompt, device, max_length=50, temperature=0.8, 
                  top_k=50, top_p=0.95, repetition_penalty=1.0):
    """Generate text using HuggingFace's generation API"""
    # Set model to eval mode
    model.eval()
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Configure generation parameters
    generation_config = {
        "max_length": max_length,
        "do_sample": True, 
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Generate text
    with torch.no_grad():
        output_sequences = model.generate(
            **inputs,
            **generation_config
        )
        
    # Decode the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Transformer CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Examples:
  python main.py
  python main.py --model_name=gpt2 --prompt="The future of AI is"
  python main.py --model_name=gpt2 --baseline
  python main.py --model_path=checkpoints/model.pth
"""
    )

    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "gpt2"),
                        help="HuggingFace model name (see supported list below).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved adaptive model checkpoint.")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Prompt text for generating output.")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated text (default: 50).")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Compute device to use: 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8). Lower is more deterministic.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (default: 50). Set to 0 to disable.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p nucleus sampling (default: 0.95).")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2). 1.0 means no penalty.")
    parser.add_argument("--baseline", action="store_true",
                        help="Use only the baseline HuggingFace model, skipping adaptive wrapper.")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"🚀 Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the baseline model
    baseline_model = load_baseline_model(args.model_name, device)

    if args.baseline:
        print("⚙️  Running with baseline HuggingFace model only")
        model = baseline_model
    else:
        print("⚙️  Creating adaptive transformer model")
        model = load_adaptive_model(args.model_name, baseline_model, device)
        
        # Load checkpoint if provided
        if args.model_path and os.path.exists(args.model_path):
            from utils.checkpoint import load_checkpoint
            optimizer = torch.optim.AdamW(model.parameters())
            head_lr_multipliers = {}
            model, _, _, _, _ = load_checkpoint(
                model, optimizer, head_lr_multipliers, args.model_path, device)
            print(f"📂 Loaded checkpoint from {args.model_path}")
    
    # Display gate activity for adaptive model
    if hasattr(model, "blocks"):
        print("\n=== GATE ACTIVITY ===")
        for layer_idx, block in enumerate(model.blocks):
            attn_module = block["attn"]
            active_heads = []
            
            for head_idx in range(attn_module.num_heads):
                if attn_module.gate[head_idx].item() > 0.1:
                    active_heads.append(head_idx)
            
            print(f"Layer {layer_idx}: Active heads -> {active_heads}")
    
    # Generate text
    generation_params = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty
    }
    
    print("\n🧠 Prompt:", args.prompt)
    generated_text = generate_text(model, tokenizer, args.prompt, device, **generation_params)
    print("\n[Generated]:", generated_text)

if __name__ == "__main__":
    main()
