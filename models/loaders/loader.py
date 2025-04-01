import torch
from transformers import AutoConfig, AutoModelForCausalLM
try:
    # First try to load the fixed version
    from .fix_gpt2_loader import load_adaptive_model_gpt
    print("✅ Using fixed GPT2 loader")
except ImportError:
    # Fallback to original loader
    from .gpt2_loader import load_adaptive_model_gpt
    print("⚠️ Using original GPT2 loader (consider using the fixed version)")


def load_baseline_model(model_name, device):
    """
    Load a baseline Hugging Face language model.
    
    Args:
        model_name: Name of the model to load (e.g., 'distilgpt2', 'gpt2')
        device: Torch device to load the model onto
    
    Returns:
        Loaded baseline model
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        print(f"✅ Loaded baseline model: {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    except Exception as e:
        print(f"❌ Error loading baseline model: {e}")
        raise


def load_adaptive_model(model_name, baseline_model, device):
    """
    Create an adaptive transformer model initialized from a baseline model.
    
    Args:
        model_name: Name of the model to load (e.g., 'distilgpt2', 'gpt2')
        baseline_model: Loaded baseline model to initialize from
        device: Torch device to load the model onto
    
    Returns:
        Initialized adaptive model
    """
    config = AutoConfig.from_pretrained(model_name)
    
    # Determine model architecture from config and dispatch to appropriate loader
    if hasattr(config, "model_type") and config.model_type.lower() in ["gpt2", "distilgpt2", "gpt_neo", "gptj"]:
        return load_adaptive_model_gpt(model_name, baseline_model, config, device)
    else:
        # Get architecture from first architecture in config.architectures if exists
        model_type = getattr(config, "architectures", [""])[0].lower()
        if "gpt" in model_type:
            return load_adaptive_model_gpt(model_name, baseline_model, config, device)
        
        raise NotImplementedError(f"Adaptive loader not implemented for architecture: {getattr(config, 'model_type', model_type)}")