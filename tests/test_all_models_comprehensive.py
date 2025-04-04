#!/usr/bin/env python
"""
Comprehensive test script for all supported models in Sentinel-AI.

This script tests:
1. Loading baseline models from HuggingFace
2. Converting to adaptive models
3. Text generation quality
4. Pruning functionality at various levels
5. Performance benchmarking

Each test is run on all supported models to ensure complete coverage.
Results are saved to a structured report for easy review.
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_testing.log')
    ]
)

# Add more verbose console output
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)

# Import sentinel modules
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

try:
    # First try sentinel package
    from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
    logger.info("Using sentinel package imports")
except ImportError:
    try:
        # Then try direct imports from models directory
        logger.warning("Falling back to direct models imports")
        from models.loaders.loader import load_baseline_model, load_adaptive_model
    except ImportError:
        # Finally try relative imports 
        logger.warning("Using absolute path imports")
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        sys.path.append(os.path.join(ROOT_DIR, 'models', 'loaders'))
        import loader
        load_baseline_model = loader.load_baseline_model
        load_adaptive_model = loader.load_adaptive_model

# Define constants
TEST_PROMPT = "The future of artificial intelligence is"
PRUNING_LEVELS = [0.0, 0.3, 0.5, 0.7]  # 0%, 30%, 50%, 70% pruning
MAX_OUTPUT_LENGTH = 50
GENERATION_PARAMS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
}

# Define supported models to test
SUPPORTED_MODELS = {
    "GPT-2": ["distilgpt2", "gpt2"],
    "OPT": ["facebook/opt-125m"],
    "Pythia": ["EleutherAI/pythia-70m"],
    "BLOOM": ["bigscience/bloom-560m"],
    "Llama": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]  # May need Hugging Face token
}


def setup_device(device_name=None):
    """Set up the device for testing."""
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def apply_pruning(model, pruning_level):
    """
    Apply random pruning to a percentage of attention heads.
    
    Args:
        model: The model to prune
        pruning_level: Percentage of heads to prune (0.0-1.0)
        
    Returns:
        Pruned model
    """
    if pruning_level == 0.0:
        return model
    
    if not hasattr(model, "blocks"):
        logger.error("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    logger.info(f"Pruning {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
    # Get a flattened list of (layer, head) tuples
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    
    # Randomly select heads to prune
    pruned_head_indices = np.random.choice(len(all_heads), heads_to_prune, replace=False)
    
    # Set gates to near-zero for pruned heads
    with torch.no_grad():
        for idx in pruned_head_indices:
            layer_idx, head_idx = all_heads[idx]
            model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=model.device)
    
    return model


def measure_inference_speed(model, tokenizer, prompt, num_tokens=30, num_runs=3):
    """Measure inference speed in tokens per second."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup run
    with torch.no_grad():
        _ = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=len(inputs.input_ids[0]) + 5,
            do_sample=False
        )
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=len(inputs.input_ids[0]) + num_tokens,
                **GENERATION_PARAMS
            )
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    tokens_per_second = num_tokens / avg_time if avg_time > 0 else 0
    
    return tokens_per_second


def evaluate_output_quality(generated_text, prompt):
    """
    Evaluate the quality of generated text.
    
    This function performs basic checks:
    1. Length beyond prompt
    2. Word diversity
    3. Repetition patterns
    
    Returns:
        dictionary with quality metrics
    """
    if not generated_text.startswith(prompt):
        # Handle case where model output doesn't include prompt
        prompt_len = 0
        new_content = generated_text
    else:
        prompt_len = len(prompt)
        new_content = generated_text[prompt_len:]
    
    # Calculate basic metrics
    new_length = len(new_content.strip())
    
    # Word-level diversity
    words = new_content.split()
    unique_words = set(words)
    word_diversity = len(unique_words) / (len(words) + 1e-10)  # Avoid division by zero
    
    # Detect repetition patterns
    repetition_score = 0
    if len(words) > 3:
        for i in range(len(words) - 3):
            if words[i] == words[i+2] and words[i+1] == words[i+3]:
                repetition_score += 1
    
    repetition_ratio = repetition_score / (len(words) + 1e-10)
    
    # Calculate an overall quality score (higher is better)
    # Length and diversity are good, repetition is bad
    quality_score = (0.5 * min(1.0, new_length / 30)) + (0.5 * word_diversity) - repetition_ratio
    quality_score = max(0, min(1, quality_score))  # Clamp between 0 and 1
    
    # Classify the result
    if quality_score > 0.7:
        quality_category = "Good"
    elif quality_score > 0.4:
        quality_category = "Fair"
    else:
        quality_category = "Poor"
    
    return {
        "new_content_length": new_length,
        "word_diversity": word_diversity,
        "repetition_ratio": repetition_ratio,
        "quality_score": quality_score,
        "quality_category": quality_category,
        "generated_text": generated_text
    }


def generate_text(model, tokenizer, prompt, max_length=50):
    """Generate text using the provided model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            **GENERATION_PARAMS
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_model(model_name, device, prompt=TEST_PROMPT, max_length=MAX_OUTPUT_LENGTH):
    """
    Comprehensive test of a model including loading, adaptation, generation and pruning.
    
    Args:
        model_name: Name of the HuggingFace model to test
        device: PyTorch device to use
        prompt: Text prompt for generation
        max_length: Maximum generation length
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"\n{'='*80}\nTesting model: {model_name}\n{'='*80}")
    result = {
        "model_name": model_name,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "baseline": {},
        "adaptive": {},
        "pruned": defaultdict(dict)
    }
    
    try:
        # Step 1: Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        result["tokenizer_loaded"] = True
        
        # Step 2: Load baseline model
        logger.info(f"Loading baseline model: {model_name}")
        start_time = time.time()
        baseline_model = load_baseline_model(model_name, device)
        load_time = time.time() - start_time
        result["baseline"]["load_time"] = load_time
        result["baseline"]["loaded"] = True
        
        # Step 3: Generate text with baseline model
        logger.info("Generating text with baseline model")
        baseline_text = generate_text(baseline_model, tokenizer, prompt, max_length)
        baseline_quality = evaluate_output_quality(baseline_text, prompt)
        baseline_speed = measure_inference_speed(baseline_model, tokenizer, prompt)
        
        result["baseline"]["text"] = baseline_text
        result["baseline"]["quality"] = baseline_quality
        result["baseline"]["speed"] = baseline_speed
        
        # Step 4: Convert to adaptive model
        logger.info("Converting to adaptive model")
        start_time = time.time()
        adaptive_model = load_adaptive_model(model_name, baseline_model, device)
        adaptation_time = time.time() - start_time
        result["adaptive"]["load_time"] = adaptation_time
        result["adaptive"]["loaded"] = True
        
        # Record model structure
        if hasattr(adaptive_model, "blocks"):
            result["adaptive"]["num_layers"] = len(adaptive_model.blocks)
            result["adaptive"]["num_heads"] = adaptive_model.blocks[0]["attn"].num_heads
            result["adaptive"]["embed_dim"] = adaptive_model.embed_dim
        
        # Step 5: Generate text with adaptive model (no pruning)
        logger.info("Generating text with adaptive model")
        adaptive_text = generate_text(adaptive_model, tokenizer, prompt, max_length)
        adaptive_quality = evaluate_output_quality(adaptive_text, prompt)
        adaptive_speed = measure_inference_speed(adaptive_model, tokenizer, prompt)
        
        result["adaptive"]["text"] = adaptive_text
        result["adaptive"]["quality"] = adaptive_quality
        result["adaptive"]["speed"] = adaptive_speed
        
        # Step 6: Test pruning at different levels
        for pruning_level in PRUNING_LEVELS:
            if pruning_level == 0.0:
                continue  # Already tested unpruned model
                
            logger.info(f"Testing with pruning level: {pruning_level}")
            pruned_model = apply_pruning(adaptive_model, pruning_level)
            
            # Generate text with pruned model
            pruned_text = generate_text(pruned_model, tokenizer, prompt, max_length)
            pruned_quality = evaluate_output_quality(pruned_text, prompt)
            pruned_speed = measure_inference_speed(pruned_model, tokenizer, prompt)
            
            # Store results
            pruning_key = f"{int(pruning_level * 100)}%"
            result["pruned"][pruning_key]["text"] = pruned_text
            result["pruned"][pruning_key]["quality"] = pruned_quality
            result["pruned"][pruning_key]["speed"] = pruned_speed
            
            # Reset model by reloading adaptive model for next pruning test
            if pruning_level != PRUNING_LEVELS[-1]:
                adaptive_model = load_adaptive_model(model_name, baseline_model, device)
        
        result["success"] = True
        logger.info(f"Testing completed successfully for {model_name}")
        
    except Exception as e:
        logger.error(f"Error testing {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        result["success"] = False
        result["error"] = str(e)
    
    return result


def format_results_table(results):
    """Format results as a markdown table for easy viewing."""
    markdown = "# Sentinel-AI Model Test Results\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Overall summary table
    markdown += "## Summary\n\n"
    markdown += "| Model | Baseline | Adaptive | 30% Pruned | 50% Pruned | 70% Pruned |\n"
    markdown += "|-------|----------|----------|------------|------------|------------|\n"
    
    for result in results:
        model_name = result["model_name"]
        
        # Get status indicators
        baseline_status = "✅" if result.get("baseline", {}).get("loaded", False) else "❌"
        adaptive_status = "✅" if result.get("adaptive", {}).get("loaded", False) else "❌"
        
        # Get pruned status
        pruned_30 = result.get("pruned", {}).get("30%", {})
        pruned_50 = result.get("pruned", {}).get("50%", {})
        pruned_70 = result.get("pruned", {}).get("70%", {})
        
        pruned_30_status = "✅" if pruned_30 else "❌"
        pruned_50_status = "✅" if pruned_50 else "❌"
        pruned_70_status = "✅" if pruned_70 else "❌"
        
        markdown += f"| {model_name} | {baseline_status} | {adaptive_status} | {pruned_30_status} | {pruned_50_status} | {pruned_70_status} |\n"
    
    # Performance table
    markdown += "\n## Performance (tokens/sec)\n\n"
    markdown += "| Model | Baseline | Adaptive | 30% Pruned | 50% Pruned | 70% Pruned |\n"
    markdown += "|-------|----------|----------|------------|------------|------------|\n"
    
    for result in results:
        if not result.get("success", False):
            continue
            
        model_name = result["model_name"]
        
        # Get speed numbers
        baseline_speed = f"{result.get('baseline', {}).get('speed', 0):.2f}"
        adaptive_speed = f"{result.get('adaptive', {}).get('speed', 0):.2f}"
        
        pruned_30_speed = f"{result.get('pruned', {}).get('30%', {}).get('speed', 0):.2f}"
        pruned_50_speed = f"{result.get('pruned', {}).get('50%', {}).get('speed', 0):.2f}"
        pruned_70_speed = f"{result.get('pruned', {}).get('70%', {}).get('speed', 0):.2f}"
        
        markdown += f"| {model_name} | {baseline_speed} | {adaptive_speed} | {pruned_30_speed} | {pruned_50_speed} | {pruned_70_speed} |\n"
    
    # Quality table
    markdown += "\n## Quality Rating\n\n"
    markdown += "| Model | Baseline | Adaptive | 30% Pruned | 50% Pruned | 70% Pruned |\n"
    markdown += "|-------|----------|----------|------------|------------|------------|\n"
    
    for result in results:
        if not result.get("success", False):
            continue
            
        model_name = result["model_name"]
        
        # Get quality categories
        baseline_quality = result.get("baseline", {}).get("quality", {}).get("quality_category", "N/A")
        adaptive_quality = result.get("adaptive", {}).get("quality", {}).get("quality_category", "N/A")
        
        pruned_30_quality = result.get("pruned", {}).get("30%", {}).get("quality", {}).get("quality_category", "N/A")
        pruned_50_quality = result.get("pruned", {}).get("50%", {}).get("quality", {}).get("quality_category", "N/A")
        pruned_70_quality = result.get("pruned", {}).get("70%", {}).get("quality", {}).get("quality_category", "N/A")
        
        markdown += f"| {model_name} | {baseline_quality} | {adaptive_quality} | {pruned_30_quality} | {pruned_50_quality} | {pruned_70_quality} |\n"
    
    # Generated Text Examples
    markdown += "\n## Generated Text Examples\n\n"
    
    for result in results:
        if not result.get("success", False):
            continue
            
        model_name = result["model_name"]
        prompt = result.get("prompt", "")
        
        markdown += f"### {model_name}\n\n"
        markdown += f"**Prompt**: {prompt}\n\n"
        
        baseline_text = result.get("baseline", {}).get("text", "")
        adaptive_text = result.get("adaptive", {}).get("text", "")
        pruned_50_text = result.get("pruned", {}).get("50%", {}).get("text", "")
        
        markdown += "**Baseline model**:\n```\n" + baseline_text + "\n```\n\n"
        markdown += "**Adaptive model**:\n```\n" + adaptive_text + "\n```\n\n"
        markdown += "**50% Pruned model**:\n```\n" + pruned_50_text + "\n```\n\n"
        
    return markdown


def run_tests(selected_models=None, device_name=None, output_dir="test_results"):
    """
    Run tests on all specified models.
    
    Args:
        selected_models: List of model names to test, or None for all supported models
        device_name: Device to use for testing
        output_dir: Directory to save results
        
    Returns:
        List of test results
    """
    # Setup device
    device = setup_device(device_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten the model list if no specific models are selected
    if selected_models is None:
        selected_models = []
        for family, models in SUPPORTED_MODELS.items():
            selected_models.extend(models)
    
    # Run tests for each model
    results = []
    for model_name in selected_models:
        try:
            result = test_model(model_name, device)
            results.append(result)
            
            # Save individual result
            model_filename = model_name.replace("/", "_").replace("-", "_")
            with open(f"{output_dir}/{model_filename}.json", "w") as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Add failed result
            results.append({
                "model_name": model_name,
                "success": False,
                "error": str(e)
            })
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{output_dir}/all_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate and save markdown report
    markdown_report = format_results_table(results)
    with open(f"{output_dir}/report_{timestamp}.md", "w") as f:
        f.write(markdown_report)
    
    logger.info(f"Testing completed. Results saved to {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test all supported models in Sentinel-AI")
    parser.add_argument("--models", type=str, nargs="+", 
                      help="Specific models to test (default: all supported models)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use for testing")
    parser.add_argument("--output_dir", type=str, default="test_results",
                      help="Directory to save test results")
    parser.add_argument("--family", type=str, choices=SUPPORTED_MODELS.keys(),
                      help="Test only models from a specific family")
    args = parser.parse_args()
    
    # Handle model selection
    selected_models = args.models
    
    # If family is specified, select models from that family
    if args.family and not selected_models:
        selected_models = SUPPORTED_MODELS[args.family]
    
    # Run the tests
    run_tests(selected_models, args.device, args.output_dir)


if __name__ == "__main__":
    main()