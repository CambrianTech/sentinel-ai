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

def generate_sample_output(model, tokenizer, prompt, device, max_length=50, temperature=0.8, top_k=50, top_p=0.95):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Inspect logits of known input
    test = tokenizer("The cat", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**test).logits
        print("Logits for 'The cat':", logits[0, -1].topk(5).indices.tolist())
        print("Top 5 tokens:", [tokenizer.decode([i]) for i in logits[0, -1].topk(5).indices.tolist()])


    print("\n🧠 Prompt:", prompt)
    print("[Generated]:", tokenizer.decode(output[0], skip_special_tokens=True))

    # If adaptive model supports gate introspection
    if hasattr(model, "get_gate_activity"):
        print("\n=== GATE ACTIVITY ===")
        gate_activity = model.get_gate_activity()
        for layer, indices in gate_activity.items():
            print(f"Layer {layer}: Active heads -> {indices}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Transformer CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Examples:
  python main.py
  python main.py --model_name=gpt2 --prompt="The future of AI is"
  python main.py --model_name=gpt2 --baseline_only
"""
    )

    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "gpt2-medium"),
                        help="HuggingFace model name (see supported list below).")
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
    parser.add_argument("--baseline_only", action="store_true",
                        help="Use only the baseline HuggingFace model, skipping adaptive wrapper.")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"🚀 Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    baseline_model = load_baseline_model(args.model_name, device)

    if args.baseline_only:
        print("⚙️  Running with baseline HuggingFace model only")
        model = baseline_model
    else:
        model = load_adaptive_model(args.model_name, baseline_model, device)

    generate_sample_output(
        model,
        tokenizer,
        args.prompt,
        device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

if __name__ == "__main__":
    main()
