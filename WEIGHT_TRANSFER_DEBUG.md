# Weight Transfer Debugging Status

## Problem Statement

After transferring weights from pre-trained GPT-2 to AdaptiveTransformer, the models produce significantly different outputs (~100 unit differences) and predict completely different tokens, indicating the model has "unlearned" GPT-2's knowledge.

## Fixes Applied ✅

### 1. Pre-norm vs Post-norm Architecture
- **Issue**: Code had `prenorm=False` but GPT-2 uses pre-norm
- **Fix**: Changed to `prenorm=True` in `adaptive_transformer.py:407`
- **GPT-2 Architecture**: LayerNorm → Attention → Residual, then LayerNorm → FFN → Residual

### 2. Active Gates Normalization
- **Issue**: Attention output was being normalized by number of active gates
- **Fix**: Commented out normalization in `adaptive_transformer.py:251-259`
- **Reason**: GPT-2 doesn't do this; only needed during pruning experiments

### 3. Output Projection Weight Slicing
- **Issue**: Incorrect transpose when slicing output projection weights
- **Fix**: Removed `.t()` in `gpt2_loader_clean.py:81`
- **Reason**: Conv1D stores weights as `[IN, OUT]`, not `[OUT, IN]`
- **Correct slicing**: `weight[i*64:(i+1)*64, :]` for head i (no transpose)

## Current Test Results ❌

```
TEST 1: Attention Forward Equivalence
   Max difference: 8.33e+01
   ❌ FAIL: Attention outputs differ by 83.35

TEST 2: FFN Forward Equivalence
   Max difference: 7.29e+01
   ❌ FAIL: FFN outputs differ by 72.90

TEST 3: Full Layer Equivalence
   Max difference: 8.10e+01
   ❌ FAIL: Layer outputs differ by 81.02

TEST 4: Full Model Equivalence (GOLD STANDARD)
   Max difference: 1.04e+02
   ❌ FAIL: Model outputs differ by 104.49

TEST 5: Generation Equivalence
   Baseline predicts token: 257
   Adaptive predicts token: 11
   ❌ FAIL: Models predict different tokens
```

## Root Cause Analysis

### Attention Module is the Source of Divergence

Step-by-step forward pass comparison shows:
- After LayerNorm: Outputs match perfectly ✅
- After Attention: **Divergence begins** (61 unit difference) ❌
- Divergence compounds through FFN and residual connections

### Mathematical Equivalence Question

**GPT-2's Approach**:
```python
QKV = input @ W_qkv + b_qkv  # Fused projection [batch, seq, 2304]
Q, K, V = split(QKV)          # Split into 3 × 768
Q = reshape(Q, [batch, seq, 12, 64])  # Split into heads
# ... attention computation ...
heads_concat = concat(all_heads)      # [batch, seq, 768]
output = heads_concat @ W_proj + b_proj
```

**Our Approach**:
```python
for each head h:
    Q_h = input @ W_q[h] + b_q[h]  # Per-head projection [batch, seq, 64]
    K_h = input @ W_k[h] + b_k[h]
    V_h = input @ W_v[h] + b_v[h]
    # ... attention computation ...
    output_h = context_h @ W_o[h] + b_o[h]  # [batch, seq, 768]
output = sum(output_h for all heads)
```

**These SHOULD be mathematically equivalent if**:
- `W_q[h] = W_qkv[:, h*64:(h+1)*64]` (slice columns for output dim)
- `W_o[h] = W_proj[h*64:(h+1)*64, :]` (slice rows for input dim)
- `b_o[h] = b_proj / 12` (divide bias equally)

## Potential Remaining Issues

### 1. Dropout in Attention Module
- GPT-2's attention has `attn_dropout` and `resid_dropout` INSIDE the module
- Our `GatedMultiHeadSelfAttention` has NO dropout
- Even in `.eval()` mode, this might affect computation graph

### 2. QKV Slicing Correctness
- Need to verify slicing logic for Q, K, V weight extraction
- Conv1D transpose handling may still have subtle bugs

### 3. Attention Score Computation
- GPT-2 may have additional operations we're missing
- Causal mask application might differ
- Softmax numerical stability tricks

### 4. Output Projection Accumulation
- Summing per-head outputs vs concatenating then projecting
- Numerical precision differences
- Bias accumulation (12 additions of bias/12 vs 1 addition of bias)

## Next Steps

### Option 1: Fix Per-Head Decomposition (Mathematically Correct)
- Add dropout to `GatedMultiHeadSelfAttention`
- Debug QKV slicing with explicit print statements
- Test intermediate attention values (Q, K, V, scores, probs)

### Option 2: Hybrid Approach (Pragmatic)
- Keep fused QKV projection (as in GPT-2)
- Only split output projection per-head
- Reduces complexity, may be easier to debug

### Option 3: Accept Approximate Equivalence (Experimental)
- Test if pruning still works despite ~100 unit differences
- Train from scratch with adaptive architecture
- Accept that perfect equivalence isn't achievable

## Test Files

- `tests/validation/test_weight_transfer_comprehensive.py` - Full test suite
- `tests/validation/test_forward_pass_comparison.py` - Step-by-step comparison
- `tests/validation/test_attention_isolation.py` - Attention module only
- `tests/unit/models/test_weight_transfer.py` - Unit tests with pytest

## References

- GPT-2 source: `transformers/models/gpt2/modeling_gpt2.py`
- Our implementation: `sentinel/models/adaptive_transformer.py`
- Weight transfer: `sentinel/models/loaders/gpt2_loader_clean.py`
