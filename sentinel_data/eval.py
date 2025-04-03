# eval.py

import os
import torch
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.checkpoint import load_checkpoint
from utils.model_wrapper import wrap_model_for_generation
from sentinel_data.dataset_loader import load_and_tokenize_dataset
import torch.nn.functional as F
import math

def evaluate(model, val_ids, tokenizer, device):
    model.eval()
    input_ids = val_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        ppl = math.exp(loss.item())
        print(f"🧪 Validation Loss: {loss.item():.4f} | Perplexity: {ppl:.2f}")
    return loss.item(), ppl

def print_head_gates(model):
    print("📊 Sentinel Gate Values per Layer:")
    for i, block in enumerate(model.blocks):
        gates = block["attn"].gate.detach().cpu().numpy()
        gate_str = " | ".join(f"{g:.2f}" for g in gates)
        print(f"Layer {i}: {gate_str}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    dataset_name = os.getenv("DATASET_NAME", "tiny_shakespeare")
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/last.ckpt")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model = load_baseline_model(model_name, device)
    model = load_adaptive_model(model_name, baseline_model, device)
    model = wrap_model_for_generation(model, tokenizer)

    val_ids = load_and_tokenize_dataset(model_name, dataset_name)[1]
    load_checkpoint(checkpoint_path, model)

    evaluate(model, val_ids, tokenizer, device)
    print_head_gates(model)

    prompt = "Once upon a midnight dreary"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=60, do_sample=True, temperature=0.9)
    print("\n📝 Generated Sample:\n", tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
