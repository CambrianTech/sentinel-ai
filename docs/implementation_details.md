# Implementation Details: Adaptive Transformer with Sentinel Gates

This document describes how our implementation aligns with the concepts presented in the research paper "Adaptive Transformer with Sentinel Gates and ANN-based Dynamic Controller".

## Core Architectural Components

### 1. Sentinel Gated Attention (Section 2.1)

**Paper Description:**
> We modify standard multi-head attention by introducing sentinel gates for each head. The gates, parameterized by learnable scalar logits, regulate head contributions:
> 
> Attention_head_i(Q,K,V) = g_i ⋅ softmax(QK^T/√d)V
> 
> Initially, gates are biased towards 1 (active heads), allowing the model to gradually identify and prune less useful heads.

**Implementation:** [`GatedMultiHeadSelfAttention` class in `models/adaptive_transformer.py`]
```python
# Sentinel gates as learnable parameters
self.gate = nn.Parameter(torch.ones(num_heads))

# In forward pass, apply gate multiplier to each head's output
projected = self.W_o[i](output) * self.gate[i]
```

### 2. U-Net Style Skip Connections (Section 2.2)

**Paper Description:**
> Inspired by the U-Net architecture, our model integrates skip-connections between lower ("encoder-like") and upper ("decoder-like") transformer layers:
> 
> - For a Transformer of N layers, layers 1→N/2 act as encoder layers, and N/2+1→N as decoder layers.
> - Skip connections concatenate hidden representations from encoder layer i to decoder layer N-i+1, followed by linear fusion:
>   h'_decoder_N-i+1 = Linear([h_encoder_i; h_decoder_N-i+1])

**Implementation:** [`AdaptiveTransformerModel.forward` method in `models/adaptive_transformer.py`]
```python
# Store encoder outputs for later use
if i < midpoint:
    encoder_outputs[i] = hidden_states.clone()

# Apply skip connections in decoder layers
if i >= midpoint:
    encoder_layer = self.num_layers - i - 1
    if encoder_layer in encoder_outputs:
        enc_out = encoder_outputs[encoder_layer]
        fused = torch.cat([hidden_states, enc_out], dim=-1)
        fusion_output = block["skip_fuse"](fused)
        hidden_states = hidden_states + fusion_output
```

### 3. ANN-based Dynamic Controller (Section 3)

**Paper Description:**
> We propose an ANN-based controller to dynamically manage sentinel gate values based on live metrics during training.
> 
> The controller maintains learnable gate logits per head:
> G = σ(GateLogits), G ∈ ℝ^(L×H)
> 
> We consider metrics like attention entropy and gradient norms, and apply heuristics:
> - Decrease gate logits where entropy is consistently high
> - Slightly reduce gate logits where gradients are consistently small

**Implementation:** [`ANNController` class in `controller/controller_ann.py`]
```python
# Learnable gate logits
self.gate_logits = nn.Parameter(init_val * torch.ones(num_layers, num_heads))

# Gate update based on entropy
if 'entropy' in metrics_dict:
    entropy = metrics_dict['entropy']
    high_entropy = entropy > entropy_threshold
    self.gate_logits.data = torch.where(
        high_entropy & gate_active,
        self.gate_logits.data - 0.1,
        self.gate_logits.data
    )
```

## Training and Optimization Components

### 4. Training with Gate Regularization (Section 4.1)

**Paper Description:**
> We incorporate an L1 regularization penalty on gate values to encourage sparsity and efficient pruning:
> 
> L_total = L_LM + λ_gate * Σ g_l,h

**Implementation:** [`ANNController.regularization_loss` in `controller/controller_ann.py`]
```python
def regularization_loss(self):
    """L1 regularization encouraging smaller gate values (head pruning)."""
    return torch.sum(torch.sigmoid(self.gate_logits))
```

### 5. Dynamic Pruning of Heads (Section 4.2)

**Paper Description:**
> When gate values fall below a threshold, we freeze and effectively prune corresponding heads, reducing computations.

**Implementation:** [`GatedMultiHeadSelfAttention.forward` in `models/adaptive_transformer.py`]
```python
# Skip computation if gate is near zero (pruned head)
if float(self.gate[i]) < 1e-6:
    outputs.append(torch.zeros(B, T, self.embed_dim, device=device))
    continue
```

## Implementation Challenges

While our implementation aligns with the paper's concepts, we faced several challenges:

1. **Weight Transfer from Pretrained Models**: 
   - Adapting weights from standard transformer models to our architecture required careful handling
   - The GPT-2 attention weights needed to be properly sliced and transposed to fit our per-head parameterization

2. **Generation Stability**:
   - The adaptive model sometimes produced less coherent text than the baseline model
   - We applied special handling during generation to improve quality, including token biasing and temperature scaling

3. **Skip Connection Stability**:
   - The U-Net style skip connections could cause instability during early inference
   - We implemented gradual scaling for skip connections to manage this issue

## Modifications from Original Paper

Some implementation details differ slightly from the paper for practical or stability reasons:

1. **Separate Projection Matrices**: 
   - Our model uses separate Q, K, V, O matrices for each head to enable more flexible pruning
   - This differs from the standard transformer implementation but better supports our per-head gating

2. **Gradual Skip Connection Scaling**:
   - We added scaling factors that vary by layer depth to stabilize the U-Net skip connections
   - Later layers in the network receive more conservative skip connection contributions

3. **Additional Regularization**:
   - We incorporated dropout in attention and residual paths for better stability
   - The paper doesn't explicitly mention these standard regularization techniques