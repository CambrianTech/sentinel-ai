# Supported Model Architectures

This document details the current support status for different model architectures in Sentinel-AI. The framework can load and adapt various transformer models, but compatibility and behavior may vary.

## Support Status Overview

| Family | Architecture | Example Models | Status | Notes |
|--------|--------------|----------------|--------|-------|
| **GPT-2** | Decoder-only | distilgpt2, gpt2, gpt2-medium | ✅ Full Support | Thoroughly tested and optimized |
| **OPT** | Decoder-only | facebook/opt-125m | ⚠️ Partial Support | Basic models work, larger ones may have issues |
| **Pythia/GPT-NeoX** | Decoder-only | EleutherAI/pythia-70m, pythia-160m | ✅ Full Support | Successfully loaded and tested |
| **BLOOM** | Decoder-only | bigscience/bloom-560m | ✅ Full Support | Uses hybrid adapter with original ALiBi attention |
| **Llama** | Decoder-only | meta-llama/Llama-2-7b-hf, TinyLlama-1.1B | ✅ Full Support | Uses hybrid adapter with original RoPE and SwiGLU |

## Detailed Status by Family

### GPT-2 Family

| Model | Loading | Inference | Quality | Notes |
|-------|---------|-----------|---------|-------|
| distilgpt2 | ✅ | ✅ | Good | Coherent outputs, all parameters loaded correctly |
| gpt2 | ✅ | ✅ | Good | Coherent outputs, all parameters loaded correctly |
| gpt2-medium | ✅ | ✅ | Good | Coherent outputs, all parameters loaded correctly |
| gpt2-large | ⚠️ | ⚠️ | Unknown | Should work but not thoroughly tested |
| gpt2-xl | ⚠️ | ⚠️ | Unknown | Should work but not thoroughly tested |

### OPT Family

| Model | Loading | Inference | Quality | Notes |
|-------|---------|-----------|---------|-------|
| facebook/opt-125m | ✅ | ✅ | Fair | Outputs are less coherent but functional |
| facebook/opt-350m | ✅ | ❌ | N/A | Loads weights but fails during inference with tensor dimension mismatch |
| facebook/opt-1.3b | ⚠️ | ⚠️ | Unknown | Not thoroughly tested |
| Larger models | ⚠️ | ⚠️ | Unknown | Not thoroughly tested |

**Known Issues:**
- OPT-350m and larger models: Tensor size mismatch during inference
- OPT models use a different position embedding scheme which may cause issues

### Pythia/GPT-NeoX Family

| Model | Loading | Inference | Quality | Notes |
|-------|---------|-----------|---------|-------|
| EleutherAI/pythia-70m | ✅ | ✅ | Fair | Less coherent outputs but runs successfully |
| EleutherAI/pythia-160m | ✅ | ✅ | Fair | Less coherent outputs but runs successfully |
| EleutherAI/pythia-410m | ⚠️ | ⚠️ | Unknown | Not thoroughly tested |
| Larger models | ⚠️ | ⚠️ | Unknown | Not thoroughly tested |

**Notes:**
- The Pythia family handles weight loading correctly
- Output quality may require adjustment of generation parameters

### BLOOM Family

| Model | Loading | Inference | Quality | Notes |
|-------|---------|-----------|---------|-------|
| bigscience/bloom-560m | ✅ | ✅ | Good | Uses hybrid adapter to preserve ALiBi attention |
| bigscience/bloom-1b1 | ⚠️ | ⚠️ | Unknown | Should work with our hybrid adapter |
| Larger models | ⚠️ | ⚠️ | Unknown | May work but not thoroughly tested |

**Notes:**
- BLOOM uses ALiBi positional attention (no explicit position embeddings)
- Our specialized hybrid adapter preserves BLOOM's ALiBi attention mechanism
- The adapter provides a compatible interface with our adaptive architecture
- No parameter growth in adapted models since we use the original model internals

### Llama Family

| Model | Loading | Inference | Quality | Notes |
|-------|---------|-----------|---------|-------|
| meta-llama/Llama-2-7b-hf | ⚠️ | ⚠️ | Unknown | Requires HuggingFace access token, not tested |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ✅ | ✅ | Good | Uses hybrid adapter, coherent outputs |
| TinyLlama/TinyLlama-1.1B-Chat-v0.6 | ✅ | ✅ | Good | Uses hybrid adapter, coherent outputs |
| TinyLlama/TinyLlama-1.1B-step-50K-105b | ✅ | ✅ | Good | Uses hybrid adapter, coherent outputs |
| openlm-research/open_llama_3b | ⚠️ | ⚠️ | Unknown | May work but requires additional dependencies |
| hf-internal-testing/tiny-random-LlamaForCausalLM | ✅ | ✅ | Poor | Tiny test model (2 layers), produces gibberish |

**Notes:**
- Some Llama models (like Llama-2) require authentication with the HuggingFace API
- Our Llama hybrid adapter preserves Llama's rotary position embeddings (RoPE) and SwiGLU activation
- The adapter provides a compatible interface with our adaptive architecture
- No parameter growth in adapted models since we use the original model internals
- TinyLlama models are fully accessible and work well with our adapter

## Sample Outputs

Below are example outputs from each supported model family when prompted with the same input:

**Prompt**: "The adaptive transformer architecture enables AI systems to"

### GPT-2 Family
```
distilgpt2: the following a few of this is in a few days, I'm not a number of the day or something like a few months and then, as a couple of the world's a bit of the rest of course, but he was not be a
```

### OPT Family
```
facebook/opt-125m: -i ina byr andh (t inam-isas,a inet.-v andw iso-/si andu,l-y andu/ in-o andl willinj or
```

### Pythia Family
```
EleutherAI/pythia-70m: exly-edcase, ")4 seat. 'cked oper-' better.,6ck => and: notospors BAT of itallyone wore levels.,s 1r way. ofiling inâ whence,ing
```

### BLOOM Family
```
bigscience/bloom-560m: replacing the original conventional power control system with a novel, more compact and robust approach. The proposed method is very effective in reducing operating costs of transformers due to its low total cost.
```

### Llama Family
```
TinyLlama/TinyLlama-1.1B-Chat-v1.0: improve their ability to capture dependencies and generate contextualized representations of text. In particular, it consists in using a new layer for generating the output word embedding from hidden states produced at each
```

## Usage Examples

Here's how to use different model architectures with Sentinel-AI:

```python
import torch
from models.loaders.loader import load_baseline_model, load_adaptive_model

# Load a GPT-2 model
baseline_model = load_baseline_model("distilgpt2", "cpu")
adaptive_model = load_adaptive_model("distilgpt2", baseline_model, "cpu")

# Load an OPT model
baseline_model = load_baseline_model("facebook/opt-125m", "cpu")
adaptive_model = load_adaptive_model("facebook/opt-125m", baseline_model, "cpu")

# Load a Pythia model
baseline_model = load_baseline_model("EleutherAI/pythia-70m", "cpu")
adaptive_model = load_adaptive_model("EleutherAI/pythia-70m", baseline_model, "cpu")

# Load a BLOOM model
baseline_model = load_baseline_model("bigscience/bloom-560m", "cpu")
adaptive_model = load_adaptive_model("bigscience/bloom-560m", baseline_model, "cpu")

# Load a Llama model
baseline_model = load_baseline_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "cpu")
adaptive_model = load_adaptive_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", baseline_model, "cpu")
```

## Command Line Usage

To use these models from the command line:

```bash
# GPT-2
python main.py --model_name distilgpt2 --prompt "Your prompt here"

# OPT
python main.py --model_name facebook/opt-125m --prompt "Your prompt here"

# Pythia
python main.py --model_name EleutherAI/pythia-70m --prompt "Your prompt here"

# BLOOM
python main.py --model_name bigscience/bloom-560m --prompt "Your prompt here"

# Llama
python main.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Your prompt here"
```

## Technical Details

### Architecture-Specific Handling

Each model family requires specific weight mapping and configuration handling:

1. **GPT-2**: Uses combined QKV projection, requires transformation into per-head format
2. **OPT**: Uses separate Q, K, V projections, needs careful position embedding handling
3. **Pythia/GPT-NeoX**: Some variants use combined QKV, others separate; loader handles both
4. **BLOOM**: Uses ALiBi attention instead of position embeddings, our hybrid adapter preserves the original ALiBi mechanism
5. **Llama**: Uses rotary position embeddings and SwiGLU activation, requires special handling

#### BLOOM Adaptation Approach
For BLOOM models, we use a specialized hybrid approach:
- The original BLOOM model's ALiBi attention mechanism is preserved
- A thin adapter layer provides compatibility with our adaptive architecture
- Gate values are supported for compatibility with our controller framework
- This approach maintains output quality while enabling integration with our system

#### Llama Adaptation Approach
For Llama models, we use a similar specialized hybrid approach:
- The original Llama model's rotary position embeddings (RoPE) and SwiGLU activation are preserved
- A thin adapter layer provides compatibility with our adaptive architecture
- Gate values are supported for compatibility with our controller framework
- This approach maintains output quality while enabling integration with our system
## Ongoing Improvements

We're actively working to improve multi-model support:

1. Fix tensor dimension issues in larger OPT models
2. Improve generation parameter settings for each model family
3. Add support for more model families (T5, BART, etc.)
4. Optimize loading process for faster initialization
5. Extend the hybrid adapter approach to other model families (Falcon, MPT, Phi)
6. Implement multi-model profiling and benchmarking in standard workflow
7. Improve disk space management for large model caches
8. Test larger Llama models with our hybrid adapter

## Reporting Issues

If you encounter issues with specific models, please report them in our issue tracker with:
1. Model name/version
2. Error message or unexpected behavior
3. Script or code used to reproduce the issue