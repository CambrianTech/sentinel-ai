#!/usr/bin/env python
"""
Test Neural Plasticity Imports

This script tests that all the imports from the neural_plasticity module work correctly.
"""

import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

def test_imports():
    """Test that all neural plasticity module imports work correctly."""
    print("Testing neural plasticity imports...")
    
    try:
        # Test core imports
        from utils.neural_plasticity.core import (
            calculate_head_entropy,
            calculate_head_gradients,
            generate_pruning_mask,
            apply_pruning_mask,
            evaluate_model
        )
        print("✅ Core module imports successful")
        
        # Test visualization imports
        from utils.neural_plasticity.visualization import (
            visualize_head_entropy,
            visualize_head_gradients,
            visualize_pruning_decisions,
            visualize_training_metrics,
            visualize_attention_patterns
        )
        print("✅ Visualization module imports successful")
        
        # Test training imports
        from utils.neural_plasticity.training import (
            create_plasticity_trainer,
            run_plasticity_loop,
            train_with_plasticity
        )
        print("✅ Training module imports successful")
        
        return True
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)