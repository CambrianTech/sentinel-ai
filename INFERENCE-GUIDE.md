# Sentinel-AI: Inference Guide

## Current Status
1. **Weight Loading**: ✅ Successfully loads all parameters (1290/1290) from pretrained GPT-2 models
2. **Forward Pass**: ✅ Forward pass is working for both training and inference
3. **U-Net Skip Connections**: ⚠️ Temporarily disabled for stability, can be gradually enabled later
4. **Text Generation**: ✅ Now working with improved quality using beam search

## Key Fixes for Improved Generation

### Logit Scaling Issues Fixed
- Removed aggressive logit scaling that was distorting output distributions
- Eliminated the double scaling that was occurring in both forward passes

### Gating Mechanism Stabilization
- Implemented proper clamping for gate values to prevent numeric instability
- Lowered pruning threshold to ensure important heads are not skipped
- Fixed gate application to use smooth transition and avoid abrupt changes

### Generation Parameter Tuning
- Switched to beam search for higher quality generation
- Fine-tuned temperature, top_k, and top_p parameters for better coherence
- Reduced token biasing to allow more natural language flow
- Applied milder penalties for repetition to avoid overpenalizing

### Distribution Matching
- Improved the match between adaptive model and baseline model distributions
- Eliminated extreme scaling factors that were causing skewed token probabilities
- Fixed token repetition issues by using more appropriate penalties

## Recommended Usage

### Basic Generation
For basic text generation, the updated `main.py` automatically detects whether you're using the adaptive or baseline model and applies the appropriate generation method:

```bash
python main.py --prompt "Once upon a time" --model_name=gpt2
```

### Advanced Generation
For more control over the generation process, you can use the `utils/generation_wrapper.py` directly:

```python
from utils.generation_wrapper import GenerationWrapper

wrapper = GenerationWrapper(model=model, tokenizer=tokenizer, device=device)
texts = wrapper.generate_text(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.9,
    top_k=40,
    top_p=0.92,
    use_beam_search=True,
    num_beams=3
)
print(texts[0])
```

### Debugging
For debugging generation issues, use the `debug_compare.py` script which allows you to compare baseline and adaptive model outputs:

```bash
# Regular comparison
python debug_compare.py --prompt "Once upon a time" --seed 42

# Disable gates for testing
python debug_compare.py --prompt "Once upon a time" --seed 42 --disable_gates

# Disable logit scaling for testing
python debug_compare.py --prompt "Once upon a time" --seed 42 --disable_logit_scaling
```

## Next Steps

1. **Fine-tuning**: Consider fine-tuning the adaptive model on a small dataset to better adapt the loaded weights
2. **Re-enable U-Net**: Gradually re-enable the U-Net skip connections once basic generation is stable
3. **Controller Development**: Implement controller logic for dynamic head pruning/growth once generation quality is satisfactory
4. **Attention Visualization**: Use the visualization tools in `generation_wrapper.py` to analyze head importance

## Known Limitations

1. **Inference Speed**: The adaptive model is somewhat slower than the baseline due to per-head processing
2. **Memory Usage**: Separate weight matrices per head increase memory requirements
3. **Training Stability**: Initial training may require careful learning rate tuning