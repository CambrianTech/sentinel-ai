#!/usr/bin/env python
"""
Test script to verify model compatibility across architectures.

This script tests loading, pruning, and generation for all supported model types
to ensure compatibility and identify issues proactively.
"""

import os
import sys
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_compatibility_test")

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import pruning utilities
from utils.pruning import PruningModule, get_strategy


def test_model_loading(model_name: str) -> Dict[str, Any]:
    """
    Test loading a specific model.
    
    Args:
        model_name: Name of the model to test
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing model loading for {model_name}")
    
    try:
        # Initialize pruning module
        pruning_module = PruningModule(model_name)
        
        # Load model
        start_time = np.datetime64('now')
        load_success = pruning_module.load_model()
        end_time = np.datetime64('now')
        load_time = (end_time - start_time) / np.timedelta64(1, 's')
        
        if load_success:
            logger.info(f"Successfully loaded {model_name} in {load_time:.2f} seconds")
            logger.info(f"  Layers: {pruning_module.num_layers}, Heads per layer: {pruning_module.num_heads}")
            
            # Return success info, but don't include pruning_module in the result dictionary
            # to avoid JSON serialization issues. We'll pass it separately.
            return {
                "model": model_name,
                "success": True,
                "load_time": load_time,
                "num_layers": pruning_module.num_layers,
                "num_heads": pruning_module.num_heads,
                "model_type": pruning_module.model_type,
                "_pruning_module": pruning_module  # Include with underscore to exclude from serialization
            }
        else:
            logger.error(f"Failed to load {model_name}")
            return {
                "model": model_name,
                "success": False,
                "error": "Failed to load model"
            }
    except Exception as e:
        logger.error(f"Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }


def test_model_pruning(pruning_module: PruningModule, strategy: str = "random", pruning_level: float = 0.3) -> Dict[str, Any]:
    """
    Test pruning a model with a specific strategy and level.
    
    Args:
        pruning_module: Initialized PruningModule with loaded model
        strategy: Pruning strategy to use
        pruning_level: Proportion of heads to prune (0.0 to 1.0)
        
    Returns:
        Dictionary with test results
    """
    model_name = pruning_module.model_name
    logger.info(f"Testing pruning for {model_name} with {strategy} strategy at {pruning_level:.2f} level")
    
    try:
        # Create strategy
        pruning_strat = get_strategy(strategy, pruning_module, "Test prompt")
        
        # Calculate importance scores
        start_time = np.datetime64('now')
        all_head_importance = pruning_strat.get_head_importance(pruning_module.original_params)
        importance_time = (np.datetime64('now') - start_time) / np.timedelta64(1, 's')
        
        # Sort by importance (ascending)
        all_head_importance.sort(key=lambda x: x[2])
        
        # Determine number of heads to prune
        total_heads = pruning_module.num_layers * pruning_module.num_heads
        heads_to_prune = int(total_heads * pruning_level)
        
        # Get head indices to prune (least important first)
        head_indices = [(l, h) for l, h, _ in all_head_importance[:heads_to_prune]]
        
        # Prune heads
        start_time = np.datetime64('now')
        pruned_params = pruning_strat.prune_heads(pruning_module.original_params, head_indices)
        pruning_time = (np.datetime64('now') - start_time) / np.timedelta64(1, 's')
        
        logger.info(f"Successfully pruned {heads_to_prune} heads from {model_name} in {pruning_time:.2f} seconds")
        
        # Return success info but store params separately from JSON-serialized data
        # to avoid serialization issues
        return {
            "model": model_name,
            "success": True,
            "strategy": strategy,
            "pruning_level": pruning_level,
            "total_heads": total_heads,
            "pruned_heads": heads_to_prune,
            "importance_time": importance_time,
            "pruning_time": pruning_time,
            "_pruned_params": pruned_params  # Include with underscore to exclude from serialization
        }
    except Exception as e:
        logger.error(f"Error pruning {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": model_name,
            "success": False,
            "strategy": strategy,
            "pruning_level": pruning_level,
            "error": str(e)
        }


def test_model_generation(pruning_module: PruningModule, params: Any, prompt: str = "Artificial intelligence will") -> Dict[str, Any]:
    """
    Test text generation with a model using specific parameters.
    
    Args:
        pruning_module: Initialized PruningModule with loaded model
        params: Model parameters to use for generation
        prompt: Prompt to use for text generation
        
    Returns:
        Dictionary with test results
    """
    model_name = pruning_module.model_name
    logger.info(f"Testing text generation for {model_name}")
    
    try:
        # Try to evaluate perplexity
        start_time = np.datetime64('now')
        perplexity = pruning_module.evaluate_perplexity(params, prompt)
        perplexity_time = (np.datetime64('now') - start_time) / np.timedelta64(1, 's')
        
        # Generate text
        start_time = np.datetime64('now')
        generated_text = pruning_module.generate_text(params, prompt)
        generation_time = (np.datetime64('now') - start_time) / np.timedelta64(1, 's')
        
        logger.info(f"Generated text: {generated_text[:100]}..." if len(generated_text) > 100 else generated_text)
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        # Success check
        generation_success = len(generated_text) > len(prompt) and not generated_text.endswith("...")
        perplexity_success = not np.isnan(perplexity)
        
        # Return results
        return {
            "model": model_name,
            "success": generation_success and perplexity_success,
            "generated_text": generated_text,
            "perplexity": float(perplexity) if perplexity_success else None,
            "generation_time": generation_time,
            "perplexity_time": perplexity_time,
            "generation_success": generation_success,
            "perplexity_success": perplexity_success
        }
    except Exception as e:
        logger.error(f"Error in generation for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }


def test_model_full_pipeline(model_name: str, strategies: List[str] = ["random"], 
                           pruning_levels: List[float] = [0.3], 
                           prompts: List[str] = ["Artificial intelligence will"]) -> Dict[str, Any]:
    """
    Run full testing pipeline on a model: load, prune, generate.
    
    Args:
        model_name: Name of the model to test
        strategies: List of pruning strategies to test
        pruning_levels: List of pruning levels to test
        prompts: List of prompts to test generation with
        
    Returns:
        Dictionary with test results for each stage
    """
    logger.info(f"========== Testing full pipeline for {model_name} ==========")
    
    # Results container
    results = {
        "model": model_name,
        "loading": None,
        "pruning": [],
        "generation": []
    }
    
    # Step 1: Load model
    loading_result = test_model_loading(model_name)
    results["loading"] = loading_result
    
    # If loading failed, skip the rest
    if not loading_result["success"]:
        logger.error(f"Loading {model_name} failed, skipping further tests")
        return results
    
    # Get pruning module (with underscore prefix to exclude from serialization)
    pruning_module = loading_result.pop("_pruning_module", None)
    
    # Step 2: Test baseline generation
    for prompt in prompts:
        generation_result = test_model_generation(
            pruning_module, 
            pruning_module.original_params, 
            prompt
        )
        generation_result["stage"] = "baseline"
        generation_result["prompt"] = prompt
        results["generation"].append(generation_result)
    
    # Step 3: Test pruning and post-pruning generation
    for strategy in strategies:
        for level in pruning_levels:
            # Skip if pruning level is too high (>80% for safety)
            if level > 0.8:
                logger.warning(f"Skipping {level} pruning level for {model_name} - too aggressive")
                continue
                
            # Perform pruning
            pruning_result = test_model_pruning(pruning_module, strategy, level)
            results["pruning"].append(pruning_result)
            
            # If pruning failed, skip generation
            if not pruning_result["success"]:
                logger.error(f"Pruning {model_name} with {strategy} strategy at {level} level failed")
                continue
            
            # Test generation with pruned model
            for prompt in prompts:
                pruned_params = pruning_result.pop("_pruned_params", None)
                if pruned_params is None:
                    logger.error(f"Pruned parameters not found for {model_name}")
                    continue
                    
                generation_result = test_model_generation(
                    pruning_module, 
                    pruned_params, 
                    prompt
                )
                generation_result["stage"] = "pruned"
                generation_result["strategy"] = strategy
                generation_result["pruning_level"] = level
                generation_result["prompt"] = prompt
                results["generation"].append(generation_result)
    
    logger.info(f"========== Completed testing pipeline for {model_name} ==========")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test model compatibility across architectures")
    parser.add_argument("--models", nargs="+", default=None, 
                       help="Models to test (default: predefined test set)")
    parser.add_argument("--skip-gpt2", action="store_true", 
                       help="Skip GPT-2 models (they're usually well supported)")
    parser.add_argument("--skip-opt", action="store_true", 
                       help="Skip OPT models")
    parser.add_argument("--skip-pythia", action="store_true", 
                       help="Skip Pythia models")
    parser.add_argument("--strategies", nargs="+", default=["random", "magnitude", "entropy"], 
                       help="Pruning strategies to test")
    parser.add_argument("--pruning-levels", nargs="+", type=float, default=[0.1, 0.3, 0.5], 
                       help="Pruning levels to test")
    parser.add_argument("--prompts", nargs="+", default=["Artificial intelligence will", 
                                                      "The future of technology is", 
                                                      "Climate change presents"], 
                       help="Prompts to test generation with")
    parser.add_argument("--output", type=str, default="model_compatibility_results.json", 
                       help="Output file for test results")
    
    args = parser.parse_args()
    
    # Define test models if not provided
    if args.models is None:
        test_models = []
        
        # Add GPT-2 models unless skipped
        if not args.skip_gpt2:
            test_models.extend(["distilgpt2", "gpt2"])
        
        # Add OPT models unless skipped
        if not args.skip_opt:
            test_models.extend(["facebook/opt-125m", "facebook/opt-350m"])
        
        # Add Pythia models unless skipped
        if not args.skip_pythia:
            test_models.extend(["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"])
    else:
        test_models = args.models
    
    # Run tests
    all_results = []
    for model in tqdm(test_models, desc="Testing models"):
        result = test_model_full_pipeline(
            model,
            strategies=args.strategies,
            pruning_levels=[float(l) for l in args.pruning_levels],
            prompts=args.prompts
        )
        all_results.append(result)
    
    # Save results
    import json
    
    # Define a custom JSON encoder to handle non-serializable objects
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            # Skip items with leading underscore (our special marker)
            if hasattr(obj, '__dict__'):
                clean_dict = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                return clean_dict
            # For NumPy types, convert to Python native types
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.datetime64)):
                return str(obj)
            # Let the base encoder raise the TypeError for other types
            return super(CustomJSONEncoder, self).default(obj)
    
    # Clean results by removing keys with leading underscore
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() if not k.startswith('_')}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        else:
            return d
    
    clean_results = clean_dict(all_results)
    
    with open(args.output, 'w') as f:
        json.dump(clean_results, f, indent=2, cls=CustomJSONEncoder)
    
    logger.info(f"Test results saved to {args.output}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Tested {len(test_models)} models")
    
    # Loading stats
    successful_loads = sum(1 for r in all_results if r["loading"]["success"])
    print(f"Loading: {successful_loads}/{len(test_models)} successful")
    
    # Pruning stats
    all_pruning_tests = sum(len(r["pruning"]) for r in all_results)
    successful_pruning = sum(1 for r in all_results for p in r["pruning"] if p["success"])
    if all_pruning_tests > 0:
        print(f"Pruning: {successful_pruning}/{all_pruning_tests} successful")
    
    # Generation stats
    all_generation_tests = sum(len(r["generation"]) for r in all_results)
    successful_generation = sum(1 for r in all_results for g in r["generation"] if g["success"])
    if all_generation_tests > 0:
        print(f"Generation: {successful_generation}/{all_generation_tests} successful")
    
    # List failing models
    failing_models = [r["model"] for r in all_results if not r["loading"]["success"]]
    if failing_models:
        print("\nModels with loading failures:")
        for model in failing_models:
            print(f"  - {model}")


if __name__ == "__main__":
    main()