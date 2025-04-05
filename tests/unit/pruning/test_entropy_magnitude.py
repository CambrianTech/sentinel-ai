"""
Tests for the entropy and magnitude-based pruning implementations.
This ensures the scientific implementations work correctly.
"""

import os
import sys
import unittest
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sentinel.pruning.entropy_magnitude import (
    compute_attention_entropy,
    collect_attention_distributions,
    entropy_based_pruning,
    magnitude_based_pruning,
    update_mask,
    _apply_pruning
)


class MockAttention(nn.Module):
    """A mock attention module for testing."""
    
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_heads = num_heads  # GPT-2 style attribute
        
        # GPT-2 style weights
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        
        # Register head gates
        self.register_buffer("gate", torch.ones(num_heads))
        
    def forward(self, x, attention_mask=None, output_attentions=False):
        batch_size, seq_len, _ = x.shape
        
        # Simulate attention
        attention_probs = torch.softmax(
            torch.randn(batch_size, self.num_heads, seq_len, seq_len),
            dim=-1
        )
        
        # Apply gate values
        gate_expanded = self.gate.view(1, -1, 1, 1)
        attention_probs = attention_probs * gate_expanded
        
        # Return attention probabilities if requested
        if output_attentions:
            return attention_probs
        
        return attention_probs


class MockTransformerBlock(nn.Module):
    """A mock transformer block for testing."""
    
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = MockAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )


class MockTransformer(nn.Module):
    """A mock transformer model for testing."""
    
    def __init__(self, num_layers=2, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Create transformer blocks
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([
            MockTransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, input_ids=None, attention_mask=None, output_attentions=False, **kwargs):
        """Mock forward pass that returns attention maps for testing."""
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings
        embeddings = torch.randn(batch_size, seq_len, self.embed_dim)
        
        # Process through transformer blocks
        hidden_states = embeddings
        attentions = []
        
        for layer in self.transformer.h:
            # Call attention
            attn_output = layer.attn(hidden_states, output_attentions=output_attentions)
            
            if output_attentions:
                # In real models, attention output might be a tuple or have attentions as a property
                attentions.append(attn_output)
                
            # Update hidden states
            hidden_states = hidden_states + torch.randn_like(hidden_states) * 0.1  # Simulate residual connection

        # Return a tuple with logits and attentions
        logits = torch.randn(batch_size, seq_len, self.embed_dim)
        
        # Mock transformers library output format
        class ModelOutput:
            def __init__(self, logits, attentions=None):
                self.logits = logits
                self.attentions = attentions
                
        return ModelOutput(logits=logits, attentions=attentions if output_attentions else None)


class TestEntropyMagnitudePruning(unittest.TestCase):
    """Test the entropy and magnitude-based pruning implementations."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model = MockTransformer(num_layers=2, embed_dim=64, num_heads=4)
        
        # Create a mock dataloader
        batch_size = 2
        seq_len = 10
        self.input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        self.attention_mask = torch.ones_like(self.input_ids)
        
        # Create a simple dataset with just one batch for testing
        class MockDataLoader:
            def __init__(self, input_ids, attention_mask):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                
            def __iter__(self):
                yield {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
                
            def __len__(self):
                return 1
                
        self.dataloader = MockDataLoader(self.input_ids, self.attention_mask)
        
        # Safe update function for testing
        self.safe_update_fn = lambda tensor, value, index=None: update_mask(tensor, [index] if index is not None else [], value)
    
    def test_compute_attention_entropy(self):
        """Test the computation of attention entropy."""
        # Create a mock attention map: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len = 2, 4, 10
        
        # Create a uniform attention map (maximum entropy)
        uniform_attn = torch.ones(batch_size, num_heads, seq_len, seq_len) / seq_len
        uniform_entropy = compute_attention_entropy(uniform_attn)
        
        # For uniform distribution, entropy should be log(seq_len)
        expected_entropy = np.log(seq_len)
        for head_entropy in uniform_entropy:
            self.assertAlmostEqual(head_entropy.item(), expected_entropy, places=5)
        
        # Create a focused attention map (minimum entropy)
        focused_attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        # Each token attends only to itself
        for i in range(seq_len):
            focused_attn[:, :, i, i] = 1.0
        focused_entropy = compute_attention_entropy(focused_attn)
        
        # For completely focused distribution, entropy should be 0
        for head_entropy in focused_entropy:
            self.assertAlmostEqual(head_entropy.item(), 0.0, places=5)
    
    def test_collect_attention_distributions(self):
        """Test collecting attention distributions from a model."""
        distributions = collect_attention_distributions(
            self.model,
            self.dataloader,
            num_batches=1
        )
        
        # Check that we have distributions for each layer
        self.assertEqual(len(distributions), len(self.model.transformer.h))
        
        # Check the shape of the distributions
        for layer_idx, dist in distributions.items():
            batch_size, num_heads, seq_len = self.input_ids.shape[0], self.model.num_heads, self.input_ids.shape[1]
            expected_shape = (batch_size, num_heads, seq_len, seq_len)
            self.assertEqual(dist.shape, expected_shape)
    
    def test_entropy_based_pruning(self):
        """Test pruning based on attention entropy."""
        # Collect attention distributions
        distributions = collect_attention_distributions(
            self.model,
            self.dataloader,
            num_batches=1
        )
        
        # Initial gate values should all be 1
        for layer in self.model.transformer.h:
            self.assertTrue(torch.all(layer.attn.gate == 1.0))
        
        # Apply entropy-based pruning with 50% ratio
        pruned_heads = entropy_based_pruning(
            self.model,
            distributions,
            prune_ratio=0.5,
            safe_update_tensor_fn=self.safe_update_fn
        )
        
        # Check that the correct number of heads were pruned
        total_heads = len(self.model.transformer.h) * self.model.num_heads
        self.assertEqual(len(pruned_heads), total_heads // 2)
        
        # Check that the gate values were updated for pruned heads
        pruned_count = 0
        for layer_idx, head_idx, _ in pruned_heads:
            gate_value = self.model.transformer.h[layer_idx].attn.gate[head_idx].item()
            self.assertEqual(gate_value, 0.0)
            pruned_count += 1
        
        self.assertEqual(pruned_count, total_heads // 2)
    
    def test_magnitude_based_pruning(self):
        """Test pruning based on weight magnitude."""
        # Initial gate values should all be 1
        for layer in self.model.transformer.h:
            self.assertTrue(torch.all(layer.attn.gate == 1.0))
        
        # Apply magnitude-based pruning with 25% ratio
        pruned_heads = magnitude_based_pruning(
            self.model,
            prune_ratio=0.25,
            safe_update_tensor_fn=self.safe_update_fn
        )
        
        # Check that the correct number of heads were pruned
        total_heads = len(self.model.transformer.h) * self.model.num_heads
        self.assertEqual(len(pruned_heads), total_heads // 4)
        
        # Check that the gate values were updated for pruned heads
        pruned_count = 0
        for layer_idx, head_idx, _ in pruned_heads:
            gate_value = self.model.transformer.h[layer_idx].attn.gate[head_idx].item()
            self.assertEqual(gate_value, 0.0)
            pruned_count += 1
        
        self.assertEqual(pruned_count, total_heads // 4)
    
    def test_integration_with_real_model(self):
        """Test pruning with a real HuggingFace model (optional)."""
        try:
            # Skip this test if torch is not available
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Try to load a small model
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add gate values to attention modules
            for layer in model.transformer.h:
                # Register gate parameters if they don't exist
                if not hasattr(layer.attn, "gate"):
                    num_heads = getattr(layer.attn, "num_heads", 
                                      getattr(layer.attn, "n_head", 
                                            getattr(layer.attn, "n_heads", 12)))
                    layer.attn.register_buffer("gate", torch.ones(num_heads))
            
            # Apply magnitude-based pruning
            pruned_heads = magnitude_based_pruning(
                model,
                prune_ratio=0.1,
                safe_update_tensor_fn=lambda tensor, value, index=None: update_mask(tensor, [index] if index is not None else [], value)
            )
            
            # Check that pruning was applied
            self.assertGreater(len(pruned_heads), 0)
            
            # Verify model can still generate text
            inputs = tokenizer("Hello, I am", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=20,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.assertGreater(len(decoded), len("Hello, I am"))
            
        except (ImportError, OSError, AttributeError) as e:
            # Skip this test if model can't be loaded or has incompatible structure
            self.skipTest(f"Skipping integration test: {e}")


if __name__ == "__main__":
    unittest.main()