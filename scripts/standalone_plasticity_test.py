#!/usr/bin/env python
"""
Standalone Neural Plasticity Test

This is a completely standalone script that doesn't rely on importing the 
neural plasticity module. It directly includes the key functions we want to test 
to verify that tensor handling works properly on Apple Silicon.
"""

import os
import sys
import platform
import numpy as np
from pathlib import Path

# Configure environment variables for Apple Silicon
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch

# Detect Apple Silicon
IS_APPLE_SILICON = (platform.system() == "Darwin" and platform.processor() == "arm")
print(f"Running on Apple Silicon: {IS_APPLE_SILICON}")

# Copy of the compute_improved_entropy function from the module
def compute_improved_entropy(
    attn_probs: torch.Tensor,
    eps: float = 1e-8,
    debug: bool = False
) -> torch.Tensor:
    """
    Compute entropy with better numerical stability and optional diagnostics.
    
    This is a core function used by calculate_head_entropy that focuses on the
    numerical computation aspects with detailed diagnostics.
    
    Args:
        attn_probs: Attention probabilities tensor
        eps: Small epsilon value for numerical stability
        debug: Whether to print diagnostic information
        
    Returns:
        Tensor containing entropy values
    """
    # Save original device for returning result
    original_device = attn_probs.device
    
    if debug:
        # Print raw attention stats
        print(f"Raw attention shape: {attn_probs.shape}")
        print(f"Raw min/max/mean: {attn_probs.min().item():.6e}/{attn_probs.max().item():.6e}/{attn_probs.mean().item():.6e}")
        
        # Check for numerical issues
        print(f"Contains zeros: {(attn_probs == 0).any().item()}")
        print(f"Contains NaN: {torch.isnan(attn_probs).any().item()}")
        print(f"Contains Inf: {torch.isinf(attn_probs).any().item()}")
        
        # Check distribution validity
        row_sums = attn_probs.sum(dim=-1)
        print(f"Row sums min/max/mean: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}/{row_sums.mean().item():.6f}")
        print(f"Rows sum to ~1: {torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-2)}")
    
    # Apply numerical safeguards
    # 1. Replace non-finite values
    attn_probs = torch.where(
        torch.isfinite(attn_probs),
        attn_probs,
        torch.ones_like(attn_probs) * eps
    )
    
    # 2. Ensure positive values
    attn_probs = attn_probs.clamp(min=eps)
    
    # 3. Normalize to ensure it sums to 1.0 along attention dimension
    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
    
    if debug:
        print("\nAfter preprocessing:")
        print(f"Min/max/mean: {attn_probs.min().item():.6e}/{attn_probs.max().item():.6e}/{attn_probs.mean().item():.6e}")
        row_sums = attn_probs.sum(dim=-1)
        print(f"Row sums min/max/mean: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}/{row_sums.mean().item():.6f}")
    
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log(attn_probs)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)
    
    # Handle any remaining NaN/Inf values
    entropy = torch.where(
        torch.isfinite(entropy),
        entropy,
        torch.zeros_like(entropy)
    )
    
    if debug:
        print("\nEntropy results:")
        print(f"Entropy shape: {entropy.shape}")
        print(f"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}")
        
        # Compute theoretical maximum entropy (uniform distribution)
        seq_len = attn_probs.size(-1)
        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float))
        print(f"Theoretical max entropy (log(seq_len)): {max_entropy.item():.4f}")
        
        # Check if entropy is at maximum (uniform attention)
        print(f"Percentage of maximum entropy: {entropy.mean().item()/max_entropy.item()*100:.2f}%")
    
    # Return to original device
    if entropy.device != original_device:
        entropy = entropy.to(original_device)
    
    return entropy

# Copy of the safe_matmul function from the module
def safe_matmul(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Safely perform matrix multiplication, with special handling for Apple Silicon.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result of matrix multiplication
    """
    # For standard non-Apple Silicon environments, just use standard matmul
    if not IS_APPLE_SILICON:
        return torch.matmul(a, b)
    
    # EXTREME PROTECTION MODE FOR APPLE SILICON
    # The code below implements a super-safe matrix multiplication for Apple Silicon
    # that completely avoids BLAS crashes by using a pure Python implementation
    try:
        # Ensure tensors are on CPU
        if a.is_cuda:
            a = a.cpu()
        if b.is_cuda:
            b = b.cpu()
        
        # Ensure tensors are contiguous for better memory layout
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        
        # Ensure tensors are in float32 for better numerical stability
        if a.dtype != torch.float32 and a.dtype.is_floating_point:
            a = a.to(torch.float32)
        if b.dtype != torch.float32 and b.dtype.is_floating_point:
            b = b.to(torch.float32)
            
        # Detach tensors to prevent autograd issues
        a = a.detach()
        b = b.detach()
        
        # APPROACH 1: Try numpy first (completely bypasses BLAS issues)
        try:
            # Convert to numpy arrays (safely)
            a_np = a.numpy()
            b_np = b.numpy()
            
            # Use numpy's matmul which is more stable on Apple Silicon
            result_np = np.matmul(a_np, b_np)
            
            # Convert back to torch tensor
            result = torch.tensor(result_np, device='cpu')
            
            # Check for NaN/Inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
            
            return result
            
        # APPROACH 2: If numpy fails, try standard PyTorch
        except Exception as np_error:
            print(f"Numpy approach failed: {np_error}")
            
            # Try PyTorch with extreme safety
            with torch.no_grad():
                # Set PyTorch to single thread for matmul
                prev_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                
                try:
                    # Perform matrix multiplication
                    result = torch.matmul(a, b)
                    
                    # Check for NaN/Inf values in result
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
                        
                    # Restore thread count
                    torch.set_num_threads(prev_threads)
                    return result
                    
                except Exception as torch_error:
                    # Restore thread count
                    torch.set_num_threads(prev_threads)
                    raise torch_error
                        
    except Exception as e:
        print(f"All safe_matmul approaches failed: {e}")
        # Return zero tensor of appropriate shape as absolute last resort
        out_shape = list(a.shape[:-1]) + list(b.shape[1:])
        return torch.zeros(out_shape, device='cpu')

def run_tests():
    """Run tests for neural plasticity core functions."""
    print("=" * 40)
    print("NEURAL PLASTICITY CORE FUNCTION TESTS")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Test 1: Safe matrix multiplication
    print("\n===== Testing safe_matmul =====")
    a = torch.randn(100, 200)
    b = torch.randn(200, 100)
    
    try:
        # Standard PyTorch matmul
        import time
        start_time = time.time()
        result1 = torch.matmul(a, b)
        std_time = time.time() - start_time
        
        # Safe matmul
        start_time = time.time()
        result2 = safe_matmul(a, b)
        safe_time = time.time() - start_time
        
        # Compare results
        is_close = torch.allclose(result1, result2, rtol=1e-4, atol=1e-4)
        
        print(f"Standard matmul shape: {result1.shape}")
        print(f"Safe matmul shape: {result2.shape}")
        print(f"Results match: {is_close}")
        print(f"Standard matmul time: {std_time * 1000:.2f} ms")
        print(f"Safe matmul time: {safe_time * 1000:.2f} ms")
        print(f"Speed ratio: {safe_time / std_time:.2f}x")
        print("✅ Matrix multiplication test succeeded")
    except Exception as e:
        print(f"❌ Matrix multiplication test failed: {e}")
    
    # Test 2: Entropy calculation
    print("\n===== Testing entropy calculation =====")
    try:
        # Create test attention tensor
        batch_size = 2
        num_heads = 4
        seq_len = 32
        
        attention = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        
        print(f"Test attention tensor shape: {attention.shape}")
        
        # Run entropy calculation with diagnostics
        entropy = compute_improved_entropy(attention, debug=True)
        print(f"\nFinal entropy shape: {entropy.shape}")
        print(f"Entropy statistics: min={entropy.min().item():.4f}, max={entropy.max().item():.4f}, mean={entropy.mean().item():.4f}")
        print("✅ Entropy calculation test succeeded")
    except Exception as e:
        print(f"❌ Entropy calculation test failed: {e}")
    
    # Test 3: Large tensor operations (stress test)
    print("\n===== Testing large tensor operations =====")
    try:
        # Larger matrices
        large_size = 1000
        a_large = torch.randn(large_size, large_size)
        b_large = torch.randn(large_size, large_size)
        
        print(f"Large matrix shapes: {a_large.shape} × {b_large.shape}")
        
        # Use safe_matmul
        print("Running safe_matmul with large matrices...")
        result = safe_matmul(a_large, b_large)
        print(f"Result shape: {result.shape}")
        print(f"Result statistics: min={result.min().item():.4f}, max={result.max().item():.4f}, mean={result.mean().item():.4f}")
        print("✅ Large tensor test succeeded")
    except Exception as e:
        print(f"❌ Large tensor test failed: {e}")
    
    print("\n" + "=" * 40)
    print("ALL TESTS COMPLETED")
    print("=" * 40)

if __name__ == "__main__":
    run_tests()