"""
Test module for pruning implementation.
This script tests if pruning actually works by checking that:
1. Weights are actually zeroed out
2. The model produces different outputs after pruning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.pruning.api.entropy import prune_head_in_model

def test_gpt2_pruning():
    """Test pruning with GPT-2 model"""
    print("Testing GPT-2 pruning...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Encode a test input
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Get output before pruning
    model.eval()
    with torch.no_grad():
        outputs_before = model.generate(inputs.input_ids, max_length=20)
    
    # Extract a weights sample before pruning
    layer_idx = 0  # First layer
    head_idx = 0   # First head
    
    # Get the c_attn weights for the first head before pruning
    block = model.transformer.h[layer_idx]
    n_heads = block.attn.n_head
    hidden_size = block.attn.c_attn.weight.size(0)
    head_size = hidden_size // n_heads
    
    # Get the starting indices for the head
    q_idx = head_idx * head_size
    
    # Get a sample of the weights before pruning
    weights_before = block.attn.c_attn.weight[q_idx:q_idx+5, 0:5].clone()
    print("Weights before pruning (sample):")
    print(weights_before)
    
    # Prune the first head in the first layer
    result = prune_head_in_model(model, layer_idx, head_idx)
    print(f"Pruning result: {result}")
    
    # Check that weights are actually zeroed
    weights_after = block.attn.c_attn.weight[q_idx:q_idx+5, 0:5].clone()
    print("Weights after pruning (sample):")
    print(weights_after)
    
    # Check if all elements in the pruned section are zero
    is_zeroed = torch.all(weights_after == 0).item()
    print(f"All weights zeroed: {is_zeroed}")
    
    # Get output after pruning
    with torch.no_grad():
        outputs_after = model.generate(inputs.input_ids, max_length=20)
    
    # Check if outputs are different
    text_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    text_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    
    print("Output before pruning:", text_before)
    print("Output after pruning:", text_after)
    print("Outputs different:", text_before != text_after)
    
    if is_zeroed and text_before != text_after:
        print("✅ Pruning test PASSED!")
    else:
        print("❌ Pruning test FAILED!")
        if not is_zeroed:
            print("   - Weights were not properly zeroed")
        if text_before == text_after:
            print("   - Pruning had no effect on model output")

if __name__ == "__main__":
    test_gpt2_pruning()