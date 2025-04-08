"""
Minimal pruning test script. 
This directly implements the pruning function and tests it with distilgpt2.
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
set_seed(42)

def prune_head(model, layer_idx, head_idx):
    """Prune a specific head in a GPT-2 model by zeroing its weights"""
    # Get the transformer block
    block = model.transformer.h[layer_idx]
    
    # Get the attention module
    attn_module = block.attn
    
    # Get dimensions
    n_head = attn_module.n_head
    hidden_size = attn_module.c_attn.weight.size(0)
    head_size = hidden_size // n_head
    
    # Zero out the weights for this head
    with torch.no_grad():
        # Query weights
        q_start = head_idx * head_size
        q_end = q_start + head_size
        attn_module.c_attn.weight[q_start:q_end, :] = 0.0
        
        # Key weights
        k_start = hidden_size + head_idx * head_size
        k_end = k_start + head_size
        attn_module.c_attn.weight[k_start:k_end, :] = 0.0
        
        # Value weights
        v_start = 2 * hidden_size + head_idx * head_size
        v_end = v_start + head_size
        attn_module.c_attn.weight[v_start:v_end, :] = 0.0
        
        # Zero bias if present
        if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
            attn_module.c_attn.bias[q_start:q_end] = 0.0
            attn_module.c_attn.bias[k_start:k_end] = 0.0
            attn_module.c_attn.bias[v_start:v_end] = 0.0
    
    return True

def test_pruning():
    """Test GPT-2 pruning with direct implementation"""
    print("Loading distilgpt2 model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Generate text before pruning
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    with torch.no_grad():
        outputs_before = model.generate(
            inputs.input_ids, 
            attention_mask=torch.ones_like(inputs.input_ids),
            max_length=20,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Get weight sample before pruning
    layer_idx = 0
    head_idx = 0
    block = model.transformer.h[layer_idx]
    n_head = block.attn.n_head
    hidden_size = block.attn.c_attn.weight.size(0)
    head_size = hidden_size // n_head
    q_start = head_idx * head_size
    
    weights_before = block.attn.c_attn.weight[q_start:q_start+5, 0:5].clone()
    print("Weights before pruning (sample):")
    print(weights_before)
    
    # Prune the head
    print(f"Pruning head {head_idx} in layer {layer_idx}...")
    prune_head(model, layer_idx, head_idx)
    
    # Check weights after pruning
    weights_after = block.attn.c_attn.weight[q_start:q_start+5, 0:5].clone()
    print("Weights after pruning (sample):")
    print(weights_after)
    
    # Check if weights are zero
    is_zeroed = torch.all(weights_after == 0).item()
    print(f"All weights zeroed: {is_zeroed}")
    
    # Generate text after pruning
    with torch.no_grad():
        outputs_after = model.generate(
            inputs.input_ids,
            attention_mask=torch.ones_like(inputs.input_ids),
            max_length=20,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Compare outputs
    text_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    text_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    
    print("\nOutput before pruning:", text_before)
    print("Output after pruning:", text_after)
    print("Outputs different:", text_before != text_after)
    
    # Summary
    if is_zeroed and text_before != text_after:
        print("\n✅ Pruning test PASSED!")
        print("The pruning implementation works as expected.")
    else:
        print("\n❌ Pruning test FAILED!")
        if not is_zeroed:
            print("   - Weights were not properly zeroed")
        if text_before == text_after:
            print("   - Pruning had no effect on model output")

if __name__ == "__main__":
    test_pruning()