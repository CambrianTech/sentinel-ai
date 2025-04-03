import os
import torch
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from data_modules.dataset_loader import load_and_tokenize_dataset
from utils.generation_wrapper import wrap_for_generation
from utils.metrics import compute_perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoint.pt")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline = load_baseline_model(model_name, device)
    adaptive = load_adaptive_model(model_name, baseline, device)

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        adaptive.load_state_dict(state["model_state"])
        print(f"✅ Loaded model weights from {checkpoint_path}")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}. Evaluating untrained model.")

    # Prepare for generation
    model = wrap_for_generation(adaptive)

    # Demo text generation
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=60, temperature=0.7)
    print("\n[Generated]:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

    # Evaluate perplexity
    print("\n📊 Running perplexity eval...")
    train_ids, val_ids = load_and_tokenize_dataset(model_name)
    val_tensor = torch.tensor(val_ids[0]).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        perplexity = compute_perplexity(model, val_tensor)
    print(f"🔍 Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()
