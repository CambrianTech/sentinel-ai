#!/usr/bin/env python
"""
Main script for the Adaptive Transformer with Sentinel Gates model.

This script can be used to:
- Generate text with a trained adaptive model
- Compare against the baseline model
- Analyze and visualize gate activity
- Interactively adjust gate values

The adaptive model can be loaded from a checkpoint or initialized fresh.
"""
import os
import argparse
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.loaders.loader_optimized import load_optimized_adaptive_model
from controller.controller_manager import ControllerManager
import os

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    
    # Configure generation parameters - slightly different for adaptive model
    if hasattr(model, "blocks"):  # Our adaptive model
        generation_config = {
            "max_length": max_length,
            "do_sample": True, 
            "temperature": 0.8,  # Slightly higher temperature for more natural output
            "top_k": 40,
            "top_p": 0.94,
            "repetition_penalty": 2.2,  # Higher repetition penalty for adaptive model
            "pad_token_id": tokenizer.eos_token_id,
            "attention_mask": inputs.attention_mask  # Explicitly provide attention mask
        }
    else:
        # Standard HuggingFace model generation
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
            input_ids=inputs.input_ids,
            **generation_config
        )
        
    # Decode the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

def analyze_gates(model, controller=None):
    """
    Analyze and visualize gate activity in the adaptive model.
    
    Args:
        model: The adaptive transformer model
        controller: Optional controller manager
    """
    # Check if model has gates
    if not hasattr(model, "blocks"):
        print("❌ Model is not an adaptive transformer with gates.")
        return
    
    print("\n=== GATE ACTIVITY ANALYSIS ===")
    
    # Get number of layers and heads
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    
    # Count active and inactive gates
    total_heads = num_layers * num_heads
    active_heads = 0
    inactive_heads = 0
    
    # Track activity by layer
    layer_activity = {}
    
    # Get threshold from controller if available
    threshold = 0.1
    if controller:
        threshold = controller.config.get("active_threshold", 0.1)
    
    # Analyze each layer
    for layer_idx, block in enumerate(model.blocks):
        active_in_layer = []
        inactive_in_layer = []
        
        for head_idx in range(num_heads):
            gate_value = float(block["attn"].gate[head_idx])
            
            if gate_value > threshold:
                active_heads += 1
                active_in_layer.append((head_idx, gate_value))
            else:
                inactive_heads += 1
                inactive_in_layer.append((head_idx, gate_value))
        
        # Sort by gate value
        active_in_layer.sort(key=lambda x: x[1], reverse=True)
        inactive_in_layer.sort(key=lambda x: x[1], reverse=True)
        
        # Store layer activity
        layer_activity[layer_idx] = {
            "active": active_in_layer,
            "inactive": inactive_in_layer
        }
    
    # Print summary
    pruned_percent = 100.0 * inactive_heads / total_heads
    print(f"Total heads: {total_heads}")
    print(f"Active heads: {active_heads} ({100.0 * active_heads / total_heads:.1f}%)")
    print(f"Pruned heads: {inactive_heads} ({pruned_percent:.1f}%)")
    
    # Print detailed layer activity
    print("\nLayer-by-layer breakdown:")
    for layer_idx, activity in layer_activity.items():
        active_heads = activity["active"]
        inactive_heads = activity["inactive"]
        
        print(f"\nLayer {layer_idx}:")
        
        # Print active heads
        if active_heads:
            print(f"  Active heads ({len(active_heads)}):")
            for head_idx, gate_value in active_heads:
                print(f"    Head {head_idx}: {gate_value:.4f}")
        else:
            print("  No active heads")
        
        # Print inactive heads (only if there are not too many)
        if inactive_heads and len(inactive_heads) <= 5:
            print(f"  Inactive heads ({len(inactive_heads)}):")
            for head_idx, gate_value in inactive_heads:
                print(f"    Head {head_idx}: {gate_value:.4f}")
        elif inactive_heads:
            print(f"  Inactive heads: {len(inactive_heads)} heads")
    
    # Print controller info if available
    if controller:
        print("\nController configuration:")
        for key, value in controller.config.items():
            print(f"  {key}: {value}")

def interactive_mode(model, tokenizer, controller, device):
    """
    Interactive mode for experimenting with the model.
    
    Allows the user to:
    - Generate text with different prompts
    - Adjust gate values manually
    - Toggle U-Net skip connections
    - Compare with baseline model
    
    Args:
        model: The adaptive transformer model
        tokenizer: The tokenizer
        controller: The controller manager
        device: The device to use
    """
    print("\n🔄 Entering interactive mode. Type 'exit' to quit.")
    
    # Create baseline model for comparison
    baseline_model = None
    
    while True:
        # Get user input
        print("\nOptions:")
        print("1. Generate text")
        print("2. Analyze gates")
        print("3. Toggle U-Net connections")
        print("4. Adjust gate values")
        print("5. Compare with baseline")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1" or choice.lower() == "generate":
            # Generate text
            prompt = input("Enter prompt: ").strip()
            if not prompt:
                prompt = "Once upon a time"
            
            max_length = input("Max length (default: 100): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 100
            
            print("\nGenerating...")
            generated_text = generate_text(model, tokenizer, prompt, device, max_length=max_length)
            
            print("\n[Generated]:")
            print(generated_text)
        
        elif choice == "2" or choice.lower() == "analyze":
            # Analyze gates
            analyze_gates(model, controller)
        
        elif choice == "3" or choice.lower() == "unet":
            # Toggle U-Net connections
            state = input("Enable U-Net connections? (y/n): ").strip().lower()
            enabled = state in ("y", "yes", "true", "1")
            
            scale = input("Connection scale (default: auto): ").strip()
            if scale and scale.replace(".", "", 1).isdigit():
                scale = float(scale)
            else:
                scale = None
            
            controller.enable_unet_connections(enabled, scale)
            print(f"✅ U-Net connections {'enabled' if enabled else 'disabled'}")
        
        elif choice == "4" or choice.lower() == "adjust":
            # Adjust gate values
            layer = input("Layer (0-11, or 'all'): ").strip()
            head = input("Head (0-11, or 'all'): ").strip()
            value = input("Gate value (0-1): ").strip()
            
            try:
                value = float(value)
                if value < 0 or value > 1:
                    print("❌ Value must be between 0 and 1")
                    continue
                
                if layer.lower() == "all":
                    layers = range(len(model.blocks))
                else:
                    layers = [int(layer)]
                
                if head.lower() == "all":
                    heads = range(model.blocks[0]["attn"].num_heads)
                else:
                    heads = [int(head)]
                
                # Update gate values
                with torch.no_grad():
                    for l in layers:
                        for h in heads:
                            # Convert to logit for sigmoid
                            logit_value = np.log(value / (1 - value))
                            if hasattr(controller, "controller"):
                                controller.controller.gate_logits.data[l, h] = logit_value
                            
                            # Also update model directly
                            model.blocks[l]["attn"].gate[h] = value
                
                print(f"✅ Gate values updated")
            
            except (ValueError, IndexError) as e:
                print(f"❌ Error: {e}")
        
        elif choice == "5" or choice.lower() == "compare":
            # Compare with baseline model
            if baseline_model is None:
                print("Loading baseline model...")
                model_name = args.model_name
                baseline_model = load_baseline_model(model_name, device)
            
            prompt = input("Enter prompt: ").strip()
            if not prompt:
                prompt = "Once upon a time"
            
            max_length = input("Max length (default: 100): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 100
            
            print("\nGenerating with baseline model...")
            baseline_text = generate_text(baseline_model, tokenizer, prompt, device, max_length=max_length)
            
            print("\nGenerating with adaptive model...")
            adaptive_text = generate_text(model, tokenizer, prompt, device, max_length=max_length)
            
            print("\n[Baseline]:")
            print(baseline_text)
            
            print("\n[Adaptive]:")
            print(adaptive_text)
        
        elif choice == "6" or choice.lower() in ("exit", "quit"):
            print("Exiting interactive mode.")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

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
    parser.add_argument("--controller_path", type=str, default=None,
                        help="Path to a saved controller checkpoint.")
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
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze gate activity in the model.")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive mode for experimenting with the model.")
    parser.add_argument("--enable_unet", action="store_true",
                        help="Enable U-Net skip connections (by default they are disabled).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Set for consistent generation results.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--quiet", action="store_true", default=True,
                        help="Reduce verbose loading output (enabled by default)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed loading and gate activity output (disables --quiet)")
    parser.add_argument("--optimization_level", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Optimization level (0-3), where 3 is fully optimized")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed if specified for reproducible results
    if args.seed is not None:
        set_seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
        
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"🚀 Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the baseline model
    baseline_model = load_baseline_model(args.model_name, device)

    if args.baseline:
        print("⚙️  Running with baseline HuggingFace model only")
        model = baseline_model
        controller = None
    else:
        print("⚙️  Creating adaptive transformer model")
        debug_mode = args.debug or os.environ.get("DEBUG", "0") == "1"
        # If verbose flag is set, it overrides quiet mode
        if args.verbose:
            quiet_mode = False
        else:
            quiet_mode = args.quiet or os.environ.get("QUIET", "0") == "1"
            
        # Use optimized implementation if optimization level is specified
        if args.optimization_level is not None:
            # Set environment variable for other components that check it
            os.environ["OPTIMIZATION_LEVEL"] = str(args.optimization_level)
            print(f"⚡ Using optimization level {args.optimization_level}")
            model = load_optimized_adaptive_model(
                args.model_name, 
                baseline_model, 
                device, 
                debug=debug_mode, 
                quiet=quiet_mode,
                optimization_level=args.optimization_level
            )
        else:
            # Use original implementation
            model = load_adaptive_model(args.model_name, baseline_model, device, debug=debug_mode, quiet=quiet_mode)
        
        # Initialize the controller
        controller = ControllerManager(model)
        
        # Load checkpoint if provided
        if args.model_path and os.path.exists(args.model_path):
            from utils.checkpoint import load_checkpoint
            optimizer = torch.optim.AdamW(model.parameters())
            head_lr_multipliers = {}
            model, _, _, _, _ = load_checkpoint(
                model, optimizer, head_lr_multipliers, args.model_path, device)
            print(f"📂 Loaded model checkpoint from {args.model_path}")
        
        # Load controller checkpoint if provided
        if args.controller_path and os.path.exists(args.controller_path):
            controller_state = torch.load(args.controller_path, map_location=device)
            controller.load_state_dict(controller_state)
            print(f"📂 Loaded controller checkpoint from {args.controller_path}")
        
        # Enable U-Net skip connections if requested
        if args.enable_unet:
            print("🔄 Enabling U-Net skip connections")
            controller.enable_unet_connections(True)
    
    # Display gate activity for adaptive model (only if not in quiet mode or if verbose is enabled)
    if hasattr(model, "blocks") and not args.interactive and (args.verbose or not quiet_mode):
        print("\n=== GATE ACTIVITY ===")
        for layer_idx, block in enumerate(model.blocks):
            attn_module = block["attn"]
            active_heads = []
            
            for head_idx in range(attn_module.num_heads):
                if attn_module.gate[head_idx].item() > 0.1:
                    active_heads.append(head_idx)
            
            print(f"Layer {layer_idx}: Active heads -> {active_heads}")
    
    if args.analyze and hasattr(model, "blocks"):
        analyze_gates(model, controller)
    
    if args.interactive:
        interactive_mode(model, tokenizer, controller, device)
    else:
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