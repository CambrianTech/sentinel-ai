# Cross-Model Compatibility Guide

## Overview

This document describes the steps taken to make the sentinel-ai adaptive transformer compatible with multiple model architectures (GPT-2, OPT, BLOOM, Pythia, Llama).

## Issue Description

The original implementation had an issue with token-specific boosting in the `AdaptiveCausalLmWrapper.forward` method which caused text degradation when using non-GPT2 models. The boosting logic used GPT-2 specific token IDs that were incompatible with other model tokenizers.

## Solution

The issue was fixed by removing the token-specific boosting code from the `forward` method in `sentinel/models/adaptive/transformer.py`:

```python
def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
    """
    Forward pass handling both initial inference and autoregressive generation.
    """
    # Call the parent forward method to get logits
    logits = super().forward(input_ids, attention_mask=attention_mask)
    
    # Store input_ids for reference in repetition penalty
    self.current_input_ids = input_ids
    
    # DISABLED: All token-specific boosting has been disabled to fix cross-model compatibility
    # This code was causing severe text degradation because it used GPT-2 specific token IDs
    # that were incompatible with other model tokenizers
    
    # Return in the format expected by the generation process
    return CausalLMOutput(logits=logits) if return_dict else logits
```

## Supported Model Architectures

The following model architectures are now supported:

| Architecture | Verified Models | Status |
|--------------|----------------|--------|
| GPT-2 | distilgpt2, gpt2 | ✅ Working |
| OPT | facebook/opt-125m | ✅ Working |
| BLOOM | bigscience/bloom-560m | ✅ Working |
| Pythia (GPT-NeoX) | EleutherAI/pythia-70m | ✅ Working |
| Llama | meta-llama/Llama-2-7b-hf | 🔄 Not Tested |

## Testing Process

We created a comprehensive test script (`scripts/test_model_generation.py`) to validate the fix across different model architectures. The script:

1. Loads a baseline model of the specified architecture
2. Converts it to an adaptive model
3. Generates text with the adaptive model
4. Optionally applies pruning to test the pruning functionality

## Adaptive Model Feature Support

| Feature | Status | Notes |
|---------|--------|-------|
| Text Generation | ✅ Working | All models can generate text |
| Pruning | ✅ Working | All models support dynamic pruning |
| U-Net Skip Connections | ✅ Working | Architecture-agnostic |
| Agency Layer | ✅ Working | Architecture-agnostic |

## Known Limitations

1. **Generation Quality**: Text quality degrades when applying pruning, especially for non-GPT2 models.
2. **Pruning Strategies**: The entropy-based pruning strategy might not be optimal for all model architectures.
3. **Performance Overhead**: The adaptive transformer adds some computational overhead which varies by model.

## Usage Examples

### Basic Usage

```python
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model_name = "facebook/opt-125m"  # Or any supported model
device = torch.device("cpu")  # Or "cuda" for GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load baseline model
baseline_model = load_baseline_model(model_name, device)

# Convert to adaptive model
adaptive_model = load_adaptive_model(model_name, baseline_model, device)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = adaptive_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=30,
    do_sample=True
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Applying Pruning

```python
import torch
import numpy as np

# Apply pruning to 50% of attention heads
pruning_level = 0.5
num_layers = len(adaptive_model.blocks)
num_heads = adaptive_model.blocks[0]["attn"].num_heads
total_heads = num_layers * num_heads
heads_to_prune = int(total_heads * pruning_level)

# Get a flattened list of (layer, head) tuples
all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

# Randomly select heads to prune
pruned_head_indices = np.random.choice(len(all_heads), heads_to_prune, replace=False)

# Set gates to near-zero for pruned heads
with torch.no_grad():
    for idx in pruned_head_indices:
        layer_idx, head_idx = all_heads[idx]
        adaptive_model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=device)

# Generate text with pruned model
outputs = adaptive_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=30,
    do_sample=True
)
pruned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(pruned_text)
```

## Future Improvements

1. Develop model-specific pruning strategies to better maintain text quality
2. Implement a tokenizer-agnostic approach to token boosting if needed
3. Add fine-tuning support after pruning to recover text quality
4. Optimize model loading and inference for larger models