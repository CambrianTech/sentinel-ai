# scripts/train_colab.py

import os
import torch
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from custdata.loaders.dataset_loader import load_and_tokenize_dataset
from utils.model_wrapper import wrap_model_for_generation
from utils.training import train_model
from utils.paths import CHECKPOINT_PATH

def run_colab_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    dataset_name = os.getenv("DATASET_NAME", "tiny_shakespeare")

    print(f"🚀 Starting training with model: {model_name}, dataset: {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model = load_baseline_model(model_name, device)
    adaptive_model = load_adaptive_model(model_name, baseline_model, device)
    adaptive_model = wrap_model_for_generation(adaptive_model, tokenizer)

    train_ids, val_ids = load_and_tokenize_dataset(model_name, dataset_name)

    train_model(
        model=adaptive_model,
        tokenizer=tokenizer,
        train_ids=train_ids,
        val_ids=val_ids,
        device=device,
        checkpoint_path=CHECKPOINT_PATH
    )

if __name__ == "__main__":
    run_colab_demo()
