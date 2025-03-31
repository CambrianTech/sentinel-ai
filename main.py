import os
import argparse
import torch
from transformers import AutoTokenizer
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

def generate_sample_output(model, tokenizer, prompt, device, max_length=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.8
        )
    print("\n🧠 Prompt:", prompt)
    print("[Generated]:", tokenizer.decode(output[0], skip_special_tokens=True))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Transformer CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Examples:
  # Default inference with distilgpt2
  python main.py

  # GPT-2 with custom prompt
  python main.py --model_name=gpt2 --prompt="The future of AI is"

  # Using GPT-Neo model on CUDA with longer generation
  python main.py --model_name=EleutherAI/gpt-neo-125M --prompt="Science will" --max_length=100 --device=cuda

Supported Models:
  - distilgpt2 (default)
  - gpt2
  - gpt2-medium
  - gpt2-large
  - EleutherAI/gpt-neo-125M
  - EleutherAI/gpt-neo-1.3B
  - EleutherAI/gpt-j-6B

Any Hugging Face causal LM should generally be compatible.
"""
    )

    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "distilgpt2"),
                        help="HuggingFace model name (see supported list below).")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Prompt text for generating output.")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated text (default: 50).")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Compute device to use: 'cpu' or 'cuda' (default: auto-detect).")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"🚀 Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    baseline_model = load_baseline_model(args.model_name, device)
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, device)

    generate_sample_output(adaptive_model, tokenizer, args.prompt, device, args.max_length)

if __name__ == "__main__":
    main()
