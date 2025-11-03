"""
Unit tests for pruning implementation.

To run these tests:
    pytest -xvs utils/pruning/api/tests/test_pruning_impl.py
"""

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.pruning.api.entropy import prune_head_in_model

class TestPruningImplementation:
    @pytest.fixture(scope="class")
    def model(self):
        """Load a GPT-2 model for testing."""
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        return model
    
    def test_single_head_pruning(self, model):
        """Test pruning a single attention head."""
        # Select a head to prune
        layer_idx = 0
        head_idx = 0
        
        # Get block and attention module
        block = model.transformer.h[layer_idx]
        attn = block.attn
        
        # Get dimensions
        if hasattr(attn, 'n_head'):
            n_heads = attn.n_head
        elif hasattr(attn, 'num_heads'):
            n_heads = attn.num_heads
        elif hasattr(attn, 'num_attention_heads'):
            n_heads = attn.num_attention_heads
        else:
            n_heads = 12
        
        hidden_size = attn.c_attn.weight.size(0)
        head_size = hidden_size // n_heads
        
        # Get indices for this head's weights
        q_start = head_idx * head_size
        q_end = q_start + head_size
        k_start = hidden_size + head_idx * head_size
        k_end = k_start + head_size
        v_start = 2 * hidden_size + head_idx * head_size
        v_end = v_start + head_size
        
        # Verify weights are non-zero before pruning
        assert not torch.all(attn.c_attn.weight[q_start:q_end, :] == 0.0).item()
        
        # Prune the head
        result = prune_head_in_model(model, layer_idx, head_idx)
        
        # Check pruning was successful
        assert result is True
        
        # Check weights are now zero
        assert torch.all(attn.c_attn.weight[q_start:q_end, :] == 0.0).item()
        assert torch.all(attn.c_attn.weight[k_start:k_end, :] == 0.0).item()
        assert torch.all(attn.c_attn.weight[v_start:v_end, :] == 0.0).item()
    
    def test_multiple_head_pruning(self, model):
        """Test pruning multiple heads."""
        # Clean model for this test
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        # Heads to prune
        heads_to_prune = [(0, 1), (1, 0), (1, 2)]
        
        # Prune all heads
        pruned_heads = []
        for layer_idx, head_idx in heads_to_prune:
            if prune_head_in_model(model, layer_idx, head_idx):
                pruned_heads.append((layer_idx, head_idx))
        
        # Check all heads were pruned
        assert len(pruned_heads) == len(heads_to_prune)
        
        # Check each head's weights are zero
        for layer_idx, head_idx in pruned_heads:
            block = model.transformer.h[layer_idx]
            attn = block.attn
            
            # Get dimensions
            if hasattr(attn, 'n_head'):
                n_heads = attn.n_head
            elif hasattr(attn, 'num_heads'):
                n_heads = attn.num_heads
            elif hasattr(attn, 'num_attention_heads'):
                n_heads = attn.num_attention_heads
            else:
                n_heads = 12
            
            hidden_size = attn.c_attn.weight.size(0)
            head_size = hidden_size // n_heads
            
            # Get indices
            q_start = head_idx * head_size
            q_end = q_start + head_size
            k_start = hidden_size + head_idx * head_size
            k_end = k_start + head_size
            v_start = 2 * hidden_size + head_idx * head_size
            v_end = v_start + head_size
            
            # Verify weights are zero
            assert torch.all(attn.c_attn.weight[q_start:q_end, :] == 0.0).item()
            assert torch.all(attn.c_attn.weight[k_start:k_end, :] == 0.0).item()
            assert torch.all(attn.c_attn.weight[v_start:v_end, :] == 0.0).item()
    
    def test_model_output_changes(self, model):
        """Test that model output actually changes after pruning."""
        # Clean model for this test
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Input text
        input_text = "Once upon a time"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Generate with original model
        model.eval()
        with torch.no_grad():
            outputs_before = model.generate(
                inputs.input_ids, 
                max_length=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        text_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
        
        # Prune a significant number of heads
        heads_to_prune = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        for layer_idx, head_idx in heads_to_prune:
            prune_head_in_model(model, layer_idx, head_idx)
        
        # Reset seed to ensure same sampling conditions
        torch.manual_seed(42)
        
        # Generate with pruned model
        with torch.no_grad():
            outputs_after = model.generate(
                inputs.input_ids, 
                max_length=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        text_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
        
        # Outputs should be different after pruning
        assert text_before != text_after, f"Model output didn't change after pruning: {text_before} vs {text_after}"