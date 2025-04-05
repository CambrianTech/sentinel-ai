#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the BLOOM model loader in the sentinel package.

Tests both the new import path (sentinel.models.loaders.bloom_loader) and
the compatibility layer (models.loaders.bloom_loader).

Usage:
    python tests/test_bloom_loader.py
"""

import os
import sys
import warnings
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

def test_import_paths():
    """Test both new and legacy import paths."""
    print("\nTesting import paths:")
    print("-" * 50)
    
    # Test new import path
    try:
        from sentinel.models.loaders.bloom_loader import (
            load_adaptive_model_bloom,
            load_bloom_with_adaptive_transformer
        )
        print("✅ Successfully imported from sentinel.models.loaders.bloom_loader")
    except ImportError as e:
        print(f"❌ Failed to import from sentinel.models.loaders.bloom_loader: {e}")
        return False
    
    # Test legacy import path
    try:
        # Temporarily suppress deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from models.loaders.bloom_loader import (
                load_adaptive_model_bloom as legacy_load_adaptive_model_bloom,
                load_bloom_with_adaptive_transformer as legacy_load_bloom_with_adaptive_transformer
            )
            
            # Check if we got a deprecation warning
            deprecation_warning = any(issubclass(warning.category, DeprecationWarning) for warning in w)
            if deprecation_warning:
                print("✅ Successfully imported from models.loaders.bloom_loader (with deprecation warning)")
            else:
                print("⚠️ Successfully imported from models.loaders.bloom_loader (without deprecation warning)")
    except ImportError as e:
        print(f"❌ Failed to import from models.loaders.bloom_loader: {e}")
        return False
    
    return True

def test_load_bloom_model(model_name="bigscience/bloom-560m", device="cpu"):
    """Test loading a BLOOM model with both import paths."""
    print("\nTesting BLOOM model loading:")
    print("-" * 50)
    
    # Install required dependencies if needed
    try:
        import transformers
    except ImportError:
        print("Installing transformers package...")
        os.system("pip install transformers")
        import transformers
    
    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        device = "cpu"
    
    # First try the new import path
    print(f"Testing with sentinel.models.loaders.bloom_loader:")
    try:
        from transformers import AutoModel, AutoConfig
        from sentinel.models.loaders.bloom_loader import load_adaptive_model_bloom
        
        print(f"Loading BLOOM model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        baseline_model = AutoModel.from_pretrained(model_name)
        
        print("Creating adaptive model...")
        adaptive_model = load_adaptive_model_bloom(
            model_name, baseline_model, config, device, debug=True
        )
        
        # Check that the model has the expected structure
        print("Verifying model structure...")
        assert hasattr(adaptive_model, 'blocks'), "Model missing 'blocks' attribute"
        assert hasattr(adaptive_model, 'wte'), "Model missing 'wte' attribute"
        assert hasattr(adaptive_model, 'wpe'), "Model missing 'wpe' attribute"
        assert hasattr(adaptive_model, 'ln_f'), "Model missing 'ln_f' attribute"
        assert hasattr(adaptive_model, 'lm_head'), "Model missing 'lm_head' attribute"
        
        # Check a specific layer's structure
        block0 = adaptive_model.blocks[0]
        assert hasattr(block0, 'attn'), "Block missing 'attn' attribute"
        assert hasattr(block0, 'ffn'), "Block missing 'ffn' attribute"
        assert hasattr(block0, 'ln1'), "Block missing 'ln1' attribute"
        assert hasattr(block0, 'ln2'), "Block missing 'ln2' attribute"
        
        # Check that heads are properly initialized
        attn0 = block0.attn
        num_heads = config.n_head
        assert hasattr(attn0, 'gate'), "Attention missing 'gate' attribute"
        assert attn0.gate.shape[0] == num_heads, f"Expected {num_heads} gates, got {attn0.gate.shape[0]}"
        
        # Check a simple forward pass
        print("Testing forward pass...")
        input_ids = torch.randint(0, config.vocab_size, (1, 16)).to(device)
        with torch.no_grad():
            outputs = adaptive_model(input_ids)
        
        # Handle different output formats (model may return a tuple, dict, or custom object)
        if hasattr(outputs, 'logits'):
            # Transformers' CausalLMOutput object
            logits = outputs.logits
            print(f"Model output logits shape: {logits.shape}")
            expected_shape = (1, 16, config.vocab_size)
            assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        else:
            # Direct tensor output
            print(f"Model output shape: {outputs.shape}")
            expected_shape = (1, 16, config.vocab_size)
            assert outputs.shape == expected_shape, f"Expected shape {expected_shape}, got {outputs.shape}"
        
        print("✅ Successfully loaded and tested BLOOM model with new import path")
        
        # Clean up
        del adaptive_model
        del baseline_model
        torch.cuda.empty_cache() if device == "cuda" else None
        
    except Exception as e:
        print(f"❌ Error with new import path: {e}")
        return False
    
    # Now try the legacy import path
    print(f"\nTesting with models.loaders.bloom_loader:")
    try:
        from transformers import AutoModel, AutoConfig
        
        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from models.loaders.bloom_loader import load_adaptive_model_bloom as legacy_load
        
        print(f"Loading BLOOM model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        baseline_model = AutoModel.from_pretrained(model_name)
        
        print("Creating adaptive model...")
        adaptive_model = legacy_load(
            model_name, baseline_model, config, device, debug=True
        )
        
        # Check a simple forward pass
        print("Testing forward pass...")
        input_ids = torch.randint(0, config.vocab_size, (1, 16)).to(device)
        with torch.no_grad():
            outputs = adaptive_model(input_ids)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            # Transformers' CausalLMOutput object
            logits = outputs.logits
            print(f"Model output logits shape: {logits.shape}")
        else:
            # Direct tensor output
            print(f"Model output shape: {outputs.shape}")
        print("✅ Successfully loaded and tested BLOOM model with legacy import path")
        
        # Clean up
        del adaptive_model
        del baseline_model
        torch.cuda.empty_cache() if device == "cuda" else None
        
        return True
        
    except Exception as e:
        print(f"❌ Error with legacy import path: {e}")
        return False

def main():
    """Run all tests."""
    # Step 1: Test import paths
    if not test_import_paths():
        print("\n❌ Import path tests failed.")
        return 1
    
    # Step 2: Test model loading
    if not test_load_bloom_model():
        print("\n❌ Model loading tests failed.")
        return 1
    
    print("\n✅ All tests passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())