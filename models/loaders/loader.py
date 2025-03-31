import torch
from transformers import AutoConfig, AutoModelForCausalLM
from .gpt2_loader import load_adaptive_model_gpt


def load_baseline_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"✅ Loaded baseline model: {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def load_adaptive_model(model_name, baseline_model, device):
    config = AutoConfig.from_pretrained(model_name)

    # Dispatch by architecture
    model_type = getattr(config, "architectures", [""])[0].lower()
    if "gpt" in model_type:
        return load_adaptive_model_gpt(model_name, baseline_model, config, device)

    raise NotImplementedError(f"Adaptive loader not implemented for architecture: {model_type}")
