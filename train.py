import os
import argparse
import torch
from datasets.dataset_loader import load_and_tokenize_dataset
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.training import train_model

SUPPORTED_MODELS = [
    "distilgpt2",
    "gpt2",
    "gpt2-medium",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    # Add more HuggingFace-supported models here
]

SUPPORTED_DATASETS = [
    "tiny_shakespeare",
    "wikitext",
    "openwebtext",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Transformer Training CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Examples:
  # Train on Tiny Shakespeare with distilgpt2 (default settings)
  python train.py

  # Train using GPT-2 on WikiText dataset
  python train.py --model_name=gpt2 --dataset=wikitext

  # Training with GPT-Neo on OpenWebText using CUDA
  python train.py --model_name=EleutherAI/gpt-neo-125M --dataset=openwebtext --device=cuda --epochs=3

Supported Models:
  - distilgpt2 (default)
  - gpt2
  - gpt2-medium
  - EleutherAI/gpt-neo-125M
  - EleutherAI/gpt-neo-1.3B

Supported Datasets:
  - tiny_shakespeare (default)
  - wikitext
  - openwebtext
"""
    )

    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="HuggingFace model name (see supported list above).")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                        help="Dataset name (see supported list above).")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to train on: 'cpu' or 'cuda' (default: auto-detect).")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"📌 Training with model: {args.model_name}, dataset: {args.dataset}, device: {device}")

    tokenizer = load_baseline_model(args.model_name, device).config.tokenizer_class.from_pretrained(args.model_name)
    baseline_model = load_baseline_model(args.model_name, device)
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, device)

    train_ids, val_ids = load_and_tokenize_dataset(args.model_name, dataset_name=args.dataset)

    train_model(
        adaptive_model, tokenizer, train_ids, val_ids, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

if __name__ == "__main__":
    main()
