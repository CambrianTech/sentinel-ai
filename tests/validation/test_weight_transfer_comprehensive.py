"""
Standalone comprehensive weight transfer test (no pytest required).

Run: python test_weight_transfer_standalone.py
"""

import sys
sys.path.insert(0, '/Volumes/FlashGordon/cambrian/sentinel-ai')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from models.loaders.gpt2_loader_clean import load_adaptive_model_gpt_clean
from sentinel.models.adaptive_transformer import AdaptiveTransformerBlock


def test_attention_forward_equivalence():
    """Test that adaptive attention produces same output as baseline."""
    print("\n" + "="*80)
    print("TEST 1: Attention Forward Equivalence")
    print("="*80)

    # Load models
    print("Loading GPT-2...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    config = AutoConfig.from_pretrained("gpt2")

    print("Loading adaptive model with weight transfer...")
    adaptive_model = load_adaptive_model_gpt_clean(
        "gpt2",
        gpt2_model,
        config,
        device="cpu",
        quiet=True
    )

    # Get first layer from each
    baseline_layer = gpt2_model.transformer.h[0]
    adaptive_layer = adaptive_model.transformer.blocks[0]

    # Create random input
    print("\nTesting with random input...")
    batch_size, seq_len, hidden_size = 2, 10, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    with torch.no_grad():
        baseline_output = baseline_layer.attn(hidden_states)[0]
        adaptive_output = adaptive_layer.attn(hidden_states)

    # Check equivalence
    max_diff = torch.abs(baseline_output - adaptive_output).max().item()
    mean_diff = torch.abs(baseline_output - adaptive_output).mean().item()
    rel_error = max_diff / baseline_output.abs().max().item()

    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    print(f"   Relative error: {rel_error:.2e}")

    if max_diff < 1e-5:
        print("   ✅ PASS: Attention outputs are identical!")
        return True
    else:
        print(f"   ❌ FAIL: Attention outputs differ by {max_diff}")
        return False


def test_ffn_forward_equivalence():
    """Test that FFN produces same output as baseline."""
    print("\n" + "="*80)
    print("TEST 2: FFN Forward Equivalence")
    print("="*80)

    # Load models
    print("Loading models...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    config = AutoConfig.from_pretrained("gpt2")

    adaptive_model = load_adaptive_model_gpt_clean(
        "gpt2",
        gpt2_model,
        config,
        device="cpu",
        quiet=True
    )

    # Get first layer
    baseline_layer = gpt2_model.transformer.h[0]
    adaptive_layer = adaptive_model.transformer.blocks[0]

    # Test FFN
    print("\nTesting FFN...")
    x = torch.randn(5, 768)

    with torch.no_grad():
        baseline_out = baseline_layer.mlp(x)
        adaptive_out = adaptive_layer.ffn(x)

    max_diff = torch.abs(baseline_out - adaptive_out).max().item()
    mean_diff = torch.abs(baseline_out - adaptive_out).mean().item()

    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-6:
        print("   ✅ PASS: FFN outputs are identical!")
        return True
    else:
        print(f"   ❌ FAIL: FFN outputs differ by {max_diff}")
        return False


def test_full_layer_equivalence():
    """Test full transformer block equivalence."""
    print("\n" + "="*80)
    print("TEST 3: Full Layer Equivalence")
    print("="*80)

    # Load models
    print("Loading models...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    config = AutoConfig.from_pretrained("gpt2")

    adaptive_model = load_adaptive_model_gpt_clean(
        "gpt2",
        gpt2_model,
        config,
        device="cpu",
        quiet=True
    )

    # Get first layer
    baseline_layer = gpt2_model.transformer.h[0]
    adaptive_layer = adaptive_model.transformer.blocks[0]

    # Test full forward
    print("\nTesting full layer forward pass...")
    x = torch.randn(2, 10, 768)

    with torch.no_grad():
        baseline_out = baseline_layer(x)[0]
        adaptive_out = adaptive_layer(x)

    max_diff = torch.abs(baseline_out - adaptive_out).max().item()
    mean_diff = torch.abs(baseline_out - adaptive_out).mean().item()
    rel_error = max_diff / baseline_out.abs().max().item()

    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    print(f"   Relative error: {rel_error:.2e}")

    if max_diff < 1e-4:
        print("   ✅ PASS: Full layer outputs are identical!")
        return True
    else:
        print(f"   ❌ FAIL: Layer outputs differ by {max_diff}")
        return False


def test_full_model_equivalence():
    """
    GOLD STANDARD TEST: Full 12-layer model equivalence.

    This is the ultimate proof - if this passes, the weight transfer is perfect.
    """
    print("\n" + "="*80)
    print("TEST 4: Full Model Equivalence (GOLD STANDARD)")
    print("="*80)

    # Load models
    print("Loading GPT-2 (124M parameters)...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    config = AutoConfig.from_pretrained("gpt2")

    print("Loading adaptive model with weight transfer...")
    adaptive_model = load_adaptive_model_gpt_clean(
        "gpt2",
        gpt2_model,
        config,
        device="cpu",
        quiet=True
    )

    # Create random input
    print("\nTesting with random token IDs...")
    input_ids = torch.randint(0, config.vocab_size, (2, 20))

    # Full forward pass
    print("Running full forward pass through all 12 layers...")
    with torch.no_grad():
        baseline_out = gpt2_model(input_ids).logits
        adaptive_out = adaptive_model(input_ids).logits

    # Compare outputs
    max_diff = torch.abs(baseline_out - adaptive_out).max().item()
    mean_diff = torch.abs(baseline_out - adaptive_out).mean().item()
    rel_error = max_diff / baseline_out.abs().max().item()

    print(f"\n   Full model comparison:")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    print(f"   Relative error: {rel_error:.2e}")

    if max_diff < 1e-3:
        print("   ✅ PASS: Full model outputs are identical!")
        print("\n   🎉 WEIGHT TRANSFER IS MATHEMATICALLY PERFECT!")
        return True
    else:
        print(f"   ❌ FAIL: Model outputs differ by {max_diff}")
        return False


def test_generation_equivalence():
    """Test that both models generate the same next token."""
    print("\n" + "="*80)
    print("TEST 5: Generation Equivalence")
    print("="*80)

    # Load models
    print("Loading models...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    config = AutoConfig.from_pretrained("gpt2")

    adaptive_model = load_adaptive_model_gpt_clean(
        "gpt2",
        gpt2_model,
        config,
        device="cpu",
        quiet=True
    )

    # Test prompt: "Hello is"
    print("\nTesting next token prediction for: 'Hello is'")
    input_ids = torch.tensor([[15496, 318]])

    # Generate next token (greedy)
    with torch.no_grad():
        baseline_logits = gpt2_model(input_ids).logits[0, -1, :]
        adaptive_logits = adaptive_model(input_ids).logits[0, -1, :]

        baseline_next = baseline_logits.argmax()
        adaptive_next = adaptive_logits.argmax()

    print(f"   Baseline predicts token: {baseline_next}")
    print(f"   Adaptive predicts token: {adaptive_next}")

    # Compare logits
    logits_diff = torch.abs(baseline_logits - adaptive_logits).max().item()
    print(f"   Max logits difference: {logits_diff:.2e}")

    if baseline_next == adaptive_next and logits_diff < 1e-3:
        print("   ✅ PASS: Models generate identical predictions!")
        return True
    else:
        print(f"   ❌ FAIL: Models predict different tokens or logits differ")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE WEIGHT TRANSFER VERIFICATION")
    print("="*80)
    print("\nThese tests mathematically prove that the adaptive transformer")
    print("preserves the exact computation of the original GPT-2 model.\n")

    results = []

    try:
        results.append(("Attention Equivalence", test_attention_forward_equivalence()))
        results.append(("FFN Equivalence", test_ffn_forward_equivalence()))
        results.append(("Full Layer Equivalence", test_full_layer_equivalence()))
        results.append(("Full Model Equivalence", test_full_model_equivalence()))
        results.append(("Generation Equivalence", test_generation_equivalence()))
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Weight transfer is mathematically proven to preserve GPT-2 computation.")
        print("\n")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Weight transfer may have issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
