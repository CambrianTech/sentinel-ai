# Hybrid Attention Approach

## Current Problem
Per-head Q/K/V projections don't match GPT-2's fused approach (~100 unit differences)

## Hybrid Solution
Keep GPT-2's fused QKV projection, only split output projection per-head

## Architecture Comparison

### Current (Fully Split - Not Working)
```python
for each head h:
    Q_h = input @ W_q[h] + b_q[h]  # Per-head Q
    K_h = input @ W_k[h] + b_k[h]  # Per-head K  
    V_h = input @ W_v[h] + b_v[h]  # Per-head V
    context_h = attention(Q_h, K_h, V_h)
    output_h = context_h @ W_o[h] + b_o[h]  # Per-head output proj
output = sum(output_h)
```

### Hybrid (Fused QKV, Split Output - Should Work)
```python
QKV = input @ W_qkv + b_qkv  # Fused (like GPT-2)
Q, K, V = split_and_reshape(QKV, num_heads=12)  # [batch, heads, seq, 64]

for each head h:
    context_h = attention(Q[h], K[h], V[h])
    output_h = context_h @ W_o[h] + b_o[h]  # Per-head output proj
output = sum(output_h)
```

## Benefits
1. **Exact QKV match**: Fused projection identical to GPT-2
2. **Simpler weight transfer**: No Q/K/V splitting needed
3. **Still enables per-head pruning**: Can still gate each head's output
4. **Easier to debug**: Only one place (output proj) can have issues

## Implementation Plan
1. Modify `GatedMultiHeadSelfAttention.__init__` to use fused QKV projection
2. Update forward pass to split QKV after projection
3. Simplify weight transfer in `gpt2_loader_clean.py`
4. Run tests - should get exact match now!

## Why This Works for Pruning
We can still prune heads because:
- Each head computes attention independently  
- Gate is applied to each head's OUTPUT
- Disabling a head (gate=0) removes its contribution to final output
- We don't need per-head Q/K/V to prune - only per-head gating!

