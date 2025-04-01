"""
Integration-Optimized Transformer

This module provides a transformer implementation with optimized integration points
between the attention mechanism and other components. The key optimizations focus on:

1. Minimizing data movement between components
2. Reducing CPU-GPU synchronization points
3. Using in-place operations where possible
4. Optimizing the baseline knowledge integration pattern
5. Implementing efficient caching for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import time

from ..optimized_attention import OptimizedGatedMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation.utils import GenerationMixin


class IntegrationOptimizedBlock(nn.Module):
    """
    Transformer block with optimized integration between components.
    
    This implementation focuses on minimizing overhead between components,
    which helps expose the speedups from the optimized attention mechanism.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
        use_baseline_integration: bool = True,
        baseline_fusion_factor: float = 0.3,
    ):
        super().__init__()
        
        # Default FFN dimension to 4x embedding dimension if not specified
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim
        
        # Store dimensions for later use
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        
        # Optimized multi-head attention with agency features
        self.attn = OptimizedGatedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network with fused implementation
        # Using nn.Sequential for more efficient execution
        if activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
            
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization - using a single instance per normalization point
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Configuration attributes for skip connections - always initialize these
        self.use_skip_connection = False
        self.skip_source = -1
        self.skip_scale = 0.1
        
        # Baseline integration
        self.use_baseline_integration = use_baseline_integration
        self.baseline_fusion_factor = baseline_fusion_factor
        
        if use_baseline_integration:
            # Integration with baseline model - optimized implementation
            self.baseline_adapter = nn.Linear(embed_dim, embed_dim)
            self.baseline_gate = nn.Parameter(torch.ones(1) * baseline_fusion_factor)
            self.ln_baseline = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            
            # Skip connection from encoder to decoder (UNet style)
            # Using a single linear projection with bias for more efficient fusion
            self.skip_fuse = nn.Linear(2 * embed_dim, embed_dim, bias=True)
            
            # Cache for intermediate results
            self.register_buffer("_baseline_cache", None, persistent=False)
            self.register_buffer("_norm_cache", None, persistent=False)
            
        # Pruning optimization - track head activity rate for dynamic computations
        self.register_buffer("_active_heads_ratio", torch.ones(1), persistent=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        baseline_states: Optional[torch.Tensor] = None,
        encoder_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        step_count: Optional[int] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optimized integration between components.
        
        This implementation minimizes data movement and reduces synchronization
        points between operations.
        """
        # Cache management for generation
        is_incremental = hidden_states.size(1) == 1 and use_cache
        
        # Update active heads ratio for dynamic optimizations
        if hasattr(self.attn, "combined_activity_mask"):
            active_count = self.attn.combined_activity_mask.sum().item()
            self._active_heads_ratio = torch.tensor([active_count / self.num_heads], 
                                                   device=hidden_states.device)
        
        # Use fast path when most heads are pruned (>70% pruning)
        # Completely skip attention and baseline integration for heavily pruned layers
        use_fast_path = self._active_heads_ratio.item() < 0.3
        
        # 1. Self-attention with pre-normalization
        # Store original for residual
        residual = hidden_states
        
        # For fast path, we still need to normalize but can skip attention
        if use_fast_path:
            # Just apply normalization
            norm_hidden = self.ln1(hidden_states)
            attn_output = torch.zeros_like(hidden_states)
        else:
            # Apply layer norm - cache normalized states during generation
            if is_incremental and self._norm_cache is not None:
                # Use cached normalization for all but the last token
                norm_hidden = self.ln1(hidden_states)
            else:
                norm_hidden = self.ln1(hidden_states)
                if is_incremental:
                    # Cache for next token
                    self._norm_cache = norm_hidden.detach()
                    
            # Apply attention with optimized implementation
            attn_output = self.attn(
                norm_hidden,
                attn_mask=attention_mask,
                step_count=step_count
            )
        
        # Residual connection - use in-place when possible
        if not hidden_states.requires_grad and hidden_states.dtype == attn_output.dtype:
            hidden_states = residual.add_(attn_output)
        else:
            hidden_states = residual + attn_output
        
        # 2. Baseline model integration - optimized implementation
        # Skip integration when using fast path or no baseline states
        if (not use_fast_path and self.use_baseline_integration and 
            baseline_states is not None and self._active_heads_ratio.item() > 0.1):
            
            # Normalize and adapt baseline states
            # Check for cached baseline if in generation mode
            if is_incremental and self._baseline_cache is not None:
                adapted_baseline = self._baseline_cache
            else:
                # Optimize with fused operation
                adapted_baseline = self.baseline_adapter(self.ln_baseline(baseline_states))
                
                # Cache for generation if needed
                if is_incremental:
                    self._baseline_cache = adapted_baseline.detach()
            
            # Dynamic gating with optimized broadcasting
            # Use lower gate values for higher pruning rates
            gate_value = torch.sigmoid(self.baseline_gate) * self._active_heads_ratio
            hidden_states = hidden_states * (1 - gate_value) + adapted_baseline * gate_value
        
        # 3. UNet skip connection - optimized implementation
        # Only use skip connection for layers that need it
        if (not use_fast_path and self.use_skip_connection and 
            encoder_states is not None and self._active_heads_ratio.item() > 0.2):
            
            # Concatenate along embedding dimension
            combined = torch.cat([hidden_states, encoder_states], dim=-1)
            
            # Single fusion operation (linear projection)
            fusion_output = self.skip_fuse(combined)
            
            # Scale the skip connection based on active heads ratio
            # This helps balance the model when many heads are pruned
            # Minimum scale is 0.05 * self.skip_scale
            effective_scale = max(0.05, self._active_heads_ratio.item()) * self.skip_scale
            
            # Add with scaling - use in-place when possible
            if not hidden_states.requires_grad and hidden_states.dtype == fusion_output.dtype:
                hidden_states.add_(fusion_output * effective_scale)
            else:
                hidden_states = hidden_states + fusion_output * effective_scale
        
        # 4. Feed-forward network with pre-normalization
        # Always apply FFN - it's crucial for model quality
        residual = hidden_states
        norm_hidden = self.ln2(hidden_states)
        
        # Apply FFN with optimized sequential implementation
        ffn_output = self.ffn(norm_hidden)
        
        # Residual connection - use in-place when possible
        if not hidden_states.requires_grad and hidden_states.dtype == ffn_output.dtype:
            hidden_states = residual.add_(ffn_output)
        else:
            hidden_states = residual + ffn_output
        
        return hidden_states
    
    def clear_cache(self):
        """Clear cached values for generation."""
        self._baseline_cache = None
        self._norm_cache = None
        # Reset active heads ratio to default
        self._active_heads_ratio = torch.ones(1, device=self._active_heads_ratio.device)


class IntegrationOptimizedTransformer(nn.Module):
    """
    Transformer with optimized integration between components.
    
    This transformer implementation focuses on minimizing overhead between
    components to expose the speedups from the optimized attention mechanism.
    """
    def __init__(
        self,
        config,
        token_embeddings,
        position_embeddings,
        baseline_model=None,
        use_baseline_integration=True,
        debug=False
    ):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        self.num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        self.num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
        self.debug = debug
        self.enable_agency = True
        
        # Determine FFN dimension from config
        self.ffn_dim = (
            getattr(config, 'n_inner', None) or 
            getattr(config, 'intermediate_size', None) or 
            4 * self.embed_dim
        )
        
        # Use provided embeddings
        self.wte = token_embeddings
        self.wpe = position_embeddings
        
        # Store baseline model for knowledge transfer
        self.baseline_model = baseline_model
        self.use_baseline_integration = use_baseline_integration and baseline_model is not None
        
        # Print debug info
        if debug:
            print(f"[Model Init] embed_dim={self.embed_dim}, num_heads={self.num_heads}, ffn_dim={self.ffn_dim}")
        
        # Initialize transformer blocks
        self.blocks = nn.ModuleList()
        
        # Define midpoint for UNet architecture
        self.midpoint = self.num_layers // 2
        
        # Create transformer blocks with optimized integration
        for i in range(self.num_layers):
            # Enable baseline integration only for selected layers
            # Typically more useful in later layers
            layer_use_baseline = (
                self.use_baseline_integration and 
                i >= self.midpoint  # Only in decoder layers
            )
            
            # Create block with optimized implementation
            block = IntegrationOptimizedBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                use_baseline_integration=layer_use_baseline,
                baseline_fusion_factor=0.3 if layer_use_baseline else 0.0
            )
            
            # Configure UNet skip connections
            if i >= self.midpoint:  # Decoder layers
                # Connect to corresponding encoder layer
                encoder_idx = self.num_layers - i - 1
                if encoder_idx >= 0:
                    block.use_skip_connection = True
                    block.skip_source = encoder_idx
                    # Gradually increase skip connection strength in deeper layers
                    block.skip_scale = 0.1 * (1 + (i - self.midpoint) / (self.num_layers - self.midpoint))
            
            self.blocks.append(block)
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output layer
        self.lm_head.weight = self.wte.weight
        
        # Create causal attention mask only once and cache it
        max_pos = min(getattr(config, 'max_position_embeddings', 1024), 1024)
        causal_mask = torch.tril(torch.ones(max_pos, max_pos))
        self.register_buffer("bias", causal_mask.view(1, 1, max_pos, max_pos))
        
        # Cached baseline states for generation
        self._baseline_cache = {}
        
        # Statistics for profiling
        self.stats = {
            "attention_time": 0,
            "baseline_time": 0,
            "ffn_time": 0,
            "total_time": 0
        }
    
    def _get_baseline_states(self, input_ids, attention_mask=None, use_cache=False):
        """
        Get baseline model states with optimized implementation.
        
        This optimized version minimizes data movement and reduces
        CPU-GPU synchronization points.
        """
        # Return cached results if available in generation mode
        if use_cache and input_ids.size(1) == 1 and self._baseline_cache:
            return self._baseline_cache
        
        # For 50% or higher pruning rates, we can skip baseline computation entirely
        # since our experiments show it doesn't significantly affect quality at those levels
        active_heads_count = 0
        for block in self.blocks:
            if hasattr(block.attn, "combined_activity_mask"):
                active_heads_count += block.attn.combined_activity_mask.sum().item()
        
        total_heads = self.num_heads * self.num_layers
        pruning_rate = 1.0 - (active_heads_count / total_heads) if total_heads > 0 else 0
        
        # Skip baseline computation if pruning rate is higher than threshold
        # or if baseline integration is not used by any blocks
        any_block_uses_baseline = any(
            hasattr(block, "use_baseline_integration") and block.use_baseline_integration 
            for block in self.blocks
        )
        
        if pruning_rate >= 0.5 or not any_block_uses_baseline:
            # Return empty dict as placeholder - blocks will handle None values
            return {}
        
        baseline_outputs = {}
        
        with torch.no_grad():  # Don't compute gradients for baseline
            # Process through baseline model based on model type
            if hasattr(self.baseline_model, "transformer"):
                # GPT-2 style models
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                
                # Create position IDs efficiently
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                if batch_size > 1:
                    position_ids = position_ids.expand(batch_size, -1)
                
                # Get embeddings
                baseline_embeds = self.baseline_model.transformer.wte(input_ids)
                baseline_pos = self.baseline_model.transformer.wpe(position_ids)
                baseline_h = baseline_embeds + baseline_pos
                
                # Only process blocks that are actually used by our model
                used_blocks = set()
                for i, block in enumerate(self.blocks):
                    if (hasattr(block, "use_baseline_integration") and 
                        block.use_baseline_integration and 
                        i < len(self.baseline_model.transformer.h)):
                        used_blocks.add(i)
                
                # Process baseline blocks efficiently, only storing what we need
                for i, block in enumerate(self.baseline_model.transformer.h):
                    baseline_h = block(baseline_h)[0]  # GPT2 blocks return a tuple
                    if i in used_blocks:
                        baseline_outputs[i] = baseline_h
                    
                    # If this is the last needed block, we can break early
                    if i >= max(used_blocks) and used_blocks:
                        break
                
                # Only calculate final layer norm and logits if needed
                if "final" in used_blocks or "logits" in used_blocks:
                    baseline_h = self.baseline_model.transformer.ln_f(baseline_h)
                    baseline_outputs["final"] = baseline_h
                    baseline_outputs["logits"] = self.baseline_model.lm_head(baseline_h)
        
        # Cache for generation if in incremental mode
        if use_cache and input_ids.size(1) == 1:
            self._baseline_cache = baseline_outputs
        
        return baseline_outputs
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        step_count=None,
        return_baseline=False,
        past_key_values=None,
        use_cache=False,
        **kwargs
    ):
        """
        Forward pass through the model with optimized integration.
        
        This implementation minimizes overhead between components to reveal
        the performance benefits of the optimized attention mechanism.
        """
        # Track total time if profiling
        if self.debug:
            start_time = time.time()
        
        # Basic setup
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Incremental decoding (generation) mode detection
        is_incremental = seq_len == 1 and past_key_values is not None
        
        # Clear cache if starting new generation
        if not is_incremental:
            self._baseline_cache = {}
            # Clear block caches
            for block in self.blocks:
                if hasattr(block, "clear_cache"):
                    block.clear_cache()
        
        # OPTIMIZATION: Check if we should use the fast path for heavy pruning
        # Count how many heads are active to determine pruning level
        active_heads_count = 0
        total_heads = self.num_heads * self.num_layers
        
        for block in self.blocks:
            if hasattr(block.attn, "combined_activity_mask"):
                active_heads_count += block.attn.combined_activity_mask.sum().item()
        
        # Calculate pruning level (% of heads that are inactive)
        pruning_level = 1.0 - (active_heads_count / total_heads) if total_heads > 0 else 0
        
        # For pruning levels (≥50%), use optimized forward path
        # This dramatically reduces computation by:
        # 1. Skipping baseline model completely 
        # 2. Only using active blocks
        # 3. Simplifying UNet connections
        use_fast_path = pruning_level >= 0.5
        
        # Create position IDs efficiently
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        if batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)
        
        # Get input embeddings
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        
        # Get baseline model states if integration is enabled and we're not using fast path
        baseline_time = 0
        if not use_fast_path and self.use_baseline_integration and self.baseline_model is not None:
            if self.debug:
                baseline_start = time.time()
            
            baseline_outputs = self._get_baseline_states(
                input_ids, 
                attention_mask, 
                use_cache=use_cache
            )
            
            if self.debug:
                baseline_time = time.time() - baseline_start
                self.stats["baseline_time"] += baseline_time
        else:
            baseline_outputs = None
        
        # Storage for UNet skip connections
        encoder_outputs = {}
        
        # Create attention mask efficiently
        attn_mask = None
        if seq_len <= 1024:
            # Use pre-computed causal mask
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
            attn_mask = (1.0 - causal_mask) * -10000.0
            attn_mask = attn_mask.squeeze(0).squeeze(0)
        else:
            # Create mask on-the-fly for longer sequences
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -10000.0, diagonal=1)
        
        # OPTIMIZATION: Fast path for heavily pruned models (≥70% pruning)
        # Only process through blocks that have active heads
        if use_fast_path:
            # Identify which blocks have active heads
            active_blocks = []
            for i, block in enumerate(self.blocks):
                if hasattr(block.attn, "combined_activity_mask"):
                    if block.attn.combined_activity_mask.sum().item() > 0:
                        active_blocks.append(i)
            
            # Process only through active blocks
            for i in active_blocks:
                block = self.blocks[i]
                
                # For UNet connections, only consider active blocks
                encoder_state = None
                if (hasattr(block, "use_skip_connection") and 
                    block.use_skip_connection and 
                    block.skip_source in encoder_outputs):
                    encoder_state = encoder_outputs[block.skip_source]
                
                # Process through the block with optimizations
                # Skip baseline states in fast path
                hidden_states = block(
                    hidden_states,
                    baseline_states=None,
                    encoder_states=encoder_state,
                    attention_mask=attn_mask,
                    step_count=step_count,
                    use_cache=use_cache
                )
                
                # Store for UNet connections only if needed
                if i < self.midpoint and len([j for j in active_blocks if j >= self.midpoint]) > 0:
                    encoder_outputs[i] = hidden_states.clone()
        
        # Standard path for normal execution
        else:
            # Process through transformer blocks
            for i, block in enumerate(self.blocks):
                # In encoder layers, store outputs for later UNet connections
                if i < self.midpoint:
                    encoder_outputs[i] = hidden_states.clone()
                
                # Get baseline hidden states for this layer if available
                baseline_states = None
                if baseline_outputs is not None and i in baseline_outputs:
                    baseline_states = baseline_outputs[i]
                
                # Get encoder states for UNet skip connection if this is a decoder layer
                encoder_states = None
                if (hasattr(block, "use_skip_connection") and 
                    hasattr(block, "skip_source") and 
                    block.use_skip_connection and 
                    block.skip_source in encoder_outputs):
                    encoder_states = encoder_outputs[block.skip_source]
                
                # Process through the block with optimizations
                hidden_states = block(
                    hidden_states,
                    baseline_states=baseline_states,
                    encoder_states=encoder_states,
                    attention_mask=attn_mask,
                    step_count=step_count,
                    use_cache=use_cache
                )
        
        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits through language model head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift for autoregressive loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if self.debug:
            self.stats["total_time"] += time.time() - start_time
        
        # Return appropriate output format
        if return_baseline and baseline_outputs is not None:
            # Return both agency and baseline outputs
            agency_output = CausalLMOutput(
                loss=loss, 
                logits=logits, 
                hidden_states=None,
                attentions=None
            )
            return agency_output, baseline_outputs["logits"]
        elif labels is not None:
            # Return output with loss
            return CausalLMOutput(
                loss=loss, 
                logits=logits, 
                hidden_states=None,
                attentions=None
            )
        else:
            # Return logits only
            return logits
    
    def get_gate_activity(self):
        """Returns a dictionary with gate activity information for analysis."""
        gate_activity = {}
        for layer_idx, block in enumerate(self.blocks):
            # Get gate values from attention module
            if hasattr(block.attn, "gate"):
                gate_values = block.attn.gate.detach().cpu()
                # Get indices of active heads (gate value > threshold)
                active_heads = [i for i, g in enumerate(gate_values) if float(g) > 0.2]
                gate_activity[layer_idx] = active_heads
        return gate_activity
    
    def get_agency_report(self):
        """Generate a comprehensive report on agency status across all layers."""
        layer_reports = {}
        total_violations = 0
        
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, "get_agency_report"):
                report = block.attn.get_agency_report()
                layer_reports[i] = report
                total_violations += report["violation_count"]
        
        # Add UNet connection information
        unet_connections = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block, "use_skip_connection") and block.use_skip_connection:
                unet_connections[i] = {
                    "source": block.skip_source,
                    "scale": float(block.skip_scale)
                }
        
        # Add baseline integration information
        baseline_integration = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block, "use_baseline_integration") and block.use_baseline_integration:
                if hasattr(block, "baseline_gate"):
                    gate_value = float(torch.sigmoid(block.baseline_gate))
                    baseline_integration[i] = gate_value
        
        # Add performance statistics if available
        performance_stats = {}
        if hasattr(self, "stats") and self.stats["total_time"] > 0:
            performance_stats = {
                "attention_time": self.stats["attention_time"],
                "baseline_time": self.stats["baseline_time"],
                "ffn_time": self.stats["ffn_time"],
                "total_time": self.stats["total_time"],
                "attention_percentage": 100 * self.stats["attention_time"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0,
                "baseline_percentage": 100 * self.stats["baseline_time"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0,
                "ffn_percentage": 100 * self.stats["ffn_time"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0
            }
        
        return {
            "layer_reports": layer_reports,
            "unet_connections": unet_connections,
            "baseline_integration": baseline_integration,
            "total_violations": total_violations,
            "num_layers": self.num_layers,
            "performance_stats": performance_stats
        }
    
    def set_head_state(self, layer_idx, head_idx, state, consent=None):
        """Set state and consent for a specific attention head."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return False
            
        block = self.blocks[layer_idx]
        if hasattr(block.attn, "set_head_state"):
            return block.attn.set_head_state(head_idx, state, consent)
        return False
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "attention_time": 0,
            "baseline_time": 0,
            "ffn_time": 0,
            "total_time": 0
        }


class IntegrationOptimizedCausalLmWrapper(IntegrationOptimizedTransformer, GenerationMixin):
    """
    Causal language model wrapper for the integration-optimized transformer.
    
    This wrapper adds generation capabilities to the optimized transformer
    model, making it compatible with the HuggingFace generate API.
    """
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    can_generate = True  # Required for newer transformers versions
    _supports_cache_class = False  # Required for newer transformers versions
    
    def __init__(
        self,
        config,
        token_embeddings,
        position_embeddings,
        baseline_model=None,
        use_baseline_integration=True,
        debug=False
    ):
        super().__init__(
            config,
            token_embeddings,
            position_embeddings,
            baseline_model,
            use_baseline_integration,
            debug
        )
        
        # Set up generation configuration
        from transformers import GenerationConfig
        try:
            self.generation_config = GenerationConfig.from_model_config(config)
        except Exception as e:
            if debug:
                print(f"Warning: Could not create generation config, using defaults: {e}")
            self.generation_config = GenerationConfig()
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        """Prepare inputs for generation process with efficient caching."""
        # Input prep for generation with optimized caching
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True
        }
        
        # We don't use past_key_values for our model, but need to 
        # include a placeholder to be compatible with HF generation
        if past_key_values is not None:
            inputs["past_key_values"] = past_key_values
        
        return inputs
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search if needed."""
        # This model doesn't use key-value cache yet, but this method is required by GenerationMixin
        if self._baseline_cache:
            # Reorder the baseline cache if it exists
            reordered_cache = {}
            for k, v in self._baseline_cache.items():
                if isinstance(v, torch.Tensor):
                    reordered_cache[k] = v.index_select(0, beam_idx)
                else:
                    reordered_cache[k] = v
            self._baseline_cache = reordered_cache
        
        return past_key_values
    
    def can_generate(self):
        """Method to check if the model can generate text."""
        return True
    
    @property
    def device(self):
        """Return device of the first parameter of the model."""
        return next(self.parameters()).device
    
    def get_output_embeddings(self):
        """Get output embeddings for generation."""
        return self.lm_head
    
    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        """Forward pass optimized for generation efficiency."""
        # Call the parent class forward method with optimizations
        outputs = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **kwargs
        )
        
        # If needed, post-process logits for generation
        if isinstance(outputs, torch.Tensor) and kwargs.get("use_cache", False):
            # Apply light temperature scaling to improve output quality
            temperature = 0.9
            outputs = outputs / temperature
        
        # Return in the expected format
        if return_dict and not isinstance(outputs, CausalLMOutput):
            # Create the CausalLMOutput object with the correct field names
            return CausalLMOutput(
                loss=None,
                logits=outputs, 
                hidden_states=None,
                attentions=None
            )
        return outputs


def load_integration_optimized_model(
    baseline_model,
    device="cpu",
    use_baseline_integration=True,
    debug=False
):
    """
    Create an integration-optimized model initialized from a baseline model.
    
    Args:
        baseline_model: The baseline model to initialize from
        device: Device to load the model on
        use_baseline_integration: Whether to use baseline knowledge
        debug: Whether to print debug information
        
    Returns:
        IntegrationOptimizedCausalLmWrapper: The initialized model
    """
    config = baseline_model.config
    
    # Get token and position embeddings from baseline model
    if hasattr(baseline_model, "transformer"):
        # Standard HuggingFace models (GPT-2)
        token_embeddings = baseline_model.transformer.wte
        position_embeddings = baseline_model.transformer.wpe
    else:
        # Try generic approach
        print("Warning: Non-standard model structure. Embeddings might not be correctly initialized.")
        token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    
    # Create and initialize the model
    model = IntegrationOptimizedCausalLmWrapper(
        config=config,
        token_embeddings=token_embeddings,
        position_embeddings=position_embeddings,
        baseline_model=baseline_model if use_baseline_integration else None,
        use_baseline_integration=use_baseline_integration,
        debug=debug
    )
    
    # Move to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    if debug:
        num_params = sum(p.numel() for p in model.parameters())
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"Integration-optimized model initialized with {num_params:,} parameters")
        print(f"Baseline model has {baseline_params:,} parameters")
        print(f"Ratio: {num_params / baseline_params:.2f}x")
    
    return model