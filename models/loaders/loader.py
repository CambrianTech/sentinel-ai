import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Import model loaders
try:
    # First try to load the fixed version for GPT-2
    from .fix_gpt2_loader import load_adaptive_model_gpt
except ImportError:
    # Fallback to original loader
    from .gpt2_loader import load_adaptive_model_gpt
    print("⚠️ Using original GPT2 loader (consider using the fixed version)")

# Import loaders for additional model types
try:
    from .opt_loader import load_adaptive_model_opt
except ImportError:
    print("⚠️ OPT loader not available")
    load_adaptive_model_opt = None

try:
    from .gpt_neox_loader import load_adaptive_model_gpt_neox
except ImportError:
    print("⚠️ GPT-NeoX loader not available")
    load_adaptive_model_gpt_neox = None

try:
    from .bloom_loader import load_adaptive_model_bloom
except ImportError:
    print("⚠️ BLOOM loader not available")
    load_adaptive_model_bloom = None

try:
    from .llama_loader import load_adaptive_model_llama
except ImportError:
    print("⚠️ Llama loader not available")
    load_adaptive_model_llama = None


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


def load_adaptive_model(model_name, baseline_model, device, debug=False, quiet=False):
    """
    Create an adaptive transformer model initialized from a baseline model.
    
    Args:
        model_name: Name of the model to load (e.g., 'distilgpt2', 'gpt2', 'facebook/opt-125m', 
                  'EleutherAI/pythia-70m', 'bigscience/bloom-560m', 'meta-llama/Llama-2-7b-hf')
        baseline_model: Loaded baseline model to initialize from
        device: Torch device to load the model onto
        debug: Whether to print debug information
        quiet: If True, suppresses verbose loading messages
    
    Returns:
        Initialized adaptive model
    """
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "").lower()
    
    if not quiet:
        print(f"Model type: {model_type}")
    
    # Dispatch to appropriate loader based on model type
    if model_type in ["gpt2", "distilgpt2", "gpt_neo", "gptj"]:
        return load_adaptive_model_gpt(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
    
    elif model_type == "opt" and load_adaptive_model_opt:
        return load_adaptive_model_opt(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
    
    elif model_type == "gpt_neox" and load_adaptive_model_gpt_neox:
        return load_adaptive_model_gpt_neox(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
    
    elif model_type == "bloom" and load_adaptive_model_bloom:
        return load_adaptive_model_bloom(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
    
    elif model_type == "llama" and load_adaptive_model_llama:
        return load_adaptive_model_llama(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
    
    else:
        # Try to infer model type from architectures if not found directly
        architectures = getattr(config, "architectures", [])
        if architectures:
            arch_name = architectures[0].lower()
            
            if "gpt" in arch_name:
                if not quiet:
                    print(f"Inferred GPT-style model from architecture: {arch_name}")
                return load_adaptive_model_gpt(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
                
            elif "opt" in arch_name and load_adaptive_model_opt:
                if not quiet:
                    print(f"Inferred OPT-style model from architecture: {arch_name}")
                return load_adaptive_model_opt(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
                
            elif "neox" in arch_name and load_adaptive_model_gpt_neox:
                if not quiet:
                    print(f"Inferred GPT-NeoX-style model from architecture: {arch_name}")
                return load_adaptive_model_gpt_neox(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
                
            elif "bloom" in arch_name and load_adaptive_model_bloom:
                if not quiet:
                    print(f"Inferred BLOOM-style model from architecture: {arch_name}")
                return load_adaptive_model_bloom(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
                
            elif "llama" in arch_name and load_adaptive_model_llama:
                if not quiet:
                    print(f"Inferred Llama-style model from architecture: {arch_name}")
                return load_adaptive_model_llama(model_name, baseline_model, config, device, debug=debug, quiet=quiet)
        
        # If we get here, no supported loader was found
        raise NotImplementedError(
            f"No adaptive loader available for model type: {model_type or 'unknown'}, "
            f"architectures: {architectures}. "
            f"Supported types: GPT-2, OPT, GPT-NeoX (Pythia), BLOOM, Llama."
        )