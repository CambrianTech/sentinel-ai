import os
import sys
import time
import subprocess
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Script version
VERSION = "1.2.0"  # Updated with optimized performance recommendations

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    print("This script is designed to be run in Google Colab")
    print("For local execution, use the test_optimized_model.py script directly")

# Check for GPU
if IN_COLAB:
    gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if 'T4' in gpu_info:
        print("T4 GPU detected")
    elif 'GPU' in gpu_info:
        print("GPU detected, but not T4. Script is optimized for T4.")
    else:
        print("No GPU detected! This script requires a GPU runtime.")
        print("Go to Runtime > Change runtime type and select GPU")
        raise SystemError("No GPU detected")

# Use branch_name if it's defined in a previous cell, otherwise use default branch
if 'branch_name' in globals():
    print(f"Using branch: {branch_name}")
    use_specific_branch = True
else:
    print("Using default branch")
    use_specific_branch = False

# Clone the repository if needed
if not os.path.exists('sentinel-ai'):
    print("Cloning Sentinel-AI repository...")
    
    # Clone the repo
    !git clone https://github.com/CambrianTech/sentinel-ai.git
    
    # Move into the directory
    %cd sentinel-ai
    
    # Checkout specific branch if requested
    if use_specific_branch:
        !git checkout {branch_name}
else:
    # Move into the directory if not already there
    if 'sentinel-ai' not in os.getcwd():
        %cd sentinel-ai
    
    # Pull latest changes
    if use_specific_branch:
        print(f"Checking out and updating branch: {branch_name}")
        !git checkout {branch_name}
        !git pull
    else:
        print("Pulling latest changes from current branch")
        !git pull

# Install dependencies
print("Installing dependencies...")
!pip install -q torch transformers matplotlib seaborn numpy psutil tqdm

# Add project to Python path
import sys
sys.path.insert(0, os.getcwd())

# @title Configure and Run Performance Test
# @markdown Adjust these parameters for your performance test

# @markdown ### Model Configuration
model_name = "gpt2"  # @param ["gpt2", "distilgpt2"]
device = "cuda"  # @param ["cuda", "cpu"]
precision = "float16"  # @param ["float32", "float16"]

# @markdown ### Test Configuration 
# @markdown #### Pruning levels to test (comma-separated percentages)
# @markdown #### Recommended: Include 0% (no pruning), 30% (balanced), and 70% (speed-focused)
pruning_levels = "0,30,70"  # @param {type:"string"}
iterations = 3  # @param {type:"slider", min:1, max:5, step:1}
num_tokens = 100  # @param {type:"slider", min:10, max:200, step:10}
temperature = 0.7  # @param {type:"slider", min:0.1, max:1.0, step:0.1}
max_prompts = 3  # @param {type:"slider", min:1, max:5, step:1}
verbose = False  # @param {type:"boolean"}
use_warmup = True  # @param {type:"boolean"}

# @markdown ### Optimization Configuration
optimization_level = 3  # @param {type:"slider", min:0, max:3, step:1}
# @markdown #### Level 0: No optimization (agency only)
# @markdown #### Level 1: Basic optimization (recommended for debugging)
# @markdown #### Level 2: Full optimization (recommended for CPU performance)
# @markdown #### Level 3: Aggressive optimization (recommended for GPU / speed focus)
# @markdown Default recommended: 2 for CPU, 3 for GPU when you need agency features
# @markdown For maximum throughput, original model with 70% pruning is fastest

# Set default optimization level based on user selection
os.environ["OPTIMIZATION_LEVEL"] = str(optimization_level)

# Create results directory
os.makedirs("test_results", exist_ok=True)

# Import the modules we need
print("Importing modules...")
from models.loaders.loader import load_baseline_model, load_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning, evaluate_model
from transformers import AutoTokenizer
import torch

# Define some test prompts
def load_prompts():
    """Load or create test prompts."""
    test_prompts = [
        "Write a short summary of artificial intelligence and its applications in modern technology.",
        "Explain how transformer neural networks function in simple terms.",
        "What are the key ethical implications of large language models?",
        "Describe the concept of attention in neural networks and why it's important.",
        "Write a function to calculate the Fibonacci sequence in Python.",
        "Compare and contrast supervised and unsupervised learning approaches in machine learning.",
        "Explain the concept of gradient descent and its role in training neural networks.",
        "Describe how transformers handle long-range dependencies compared to RNNs.",
        "What are some practical applications of natural language processing in business?",
        "Explain the concept of embeddings in the context of language models."
    ]
    return test_prompts

# Run the comparison
def compare_models():
    """Run comparison between original and optimized models."""
    print("\n===== Optimized Model Performance Comparison =====\n")
    
    # Load prompts
    prompts = load_prompts()
    if max_prompts and max_prompts < len(prompts):
        prompts = prompts[:max_prompts]
    
    print(f"Using {len(prompts)} prompts for evaluation")
    
    # Parse pruning levels
    pruning_levels_list = [int(x) for x in pruning_levels.split(",")]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure proper memory cleanup before starting tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB allocated")
    
    # Import gc for more aggressive garbage collection
    try:
        import gc
        gc.collect()
    except ImportError:
        print("Warning: gc module not available for memory management")
    
    # Results dictionary to store all measurements
    results = {
        "original": {},
        "optimized": {},
        "speedup": {},
        "metadata": {
            "model_name": model_name,
            "device": device,
            "precision": precision,
            "pruning_levels": pruning_levels_list,
            "num_tokens": num_tokens,
            "temperature": temperature,
            "iterations": iterations,
            "optimization_level": optimization_level
        }
    }
    
    # For each pruning level
    for level in pruning_levels_list:
        print(f"\n{'-'*50}")
        print(f"Testing pruning level: {level}%")
        print(f"{'-'*50}")
        
        results["original"][level] = {"times": []}
        results["optimized"][level] = {"times": []}
        
        # Run multiple iterations for statistical significance
        for iteration in range(iterations):
            print(f"\nIteration {iteration+1}/{iterations}")
            
            # Test original model
            print("\nTesting original model...")
            # Force use of original model
            os.environ["USE_OPTIMIZED_MODEL"] = "0"
            
            # Load baseline model first
            baseline_model = load_baseline_model(model_name, device)
            
            # Then load original adaptive model
            start_time = time.time()
            original_model = load_adaptive_model(
                model_name, 
                baseline_model, 
                device,
                debug=False,
                quiet=not verbose
            )
            load_time_original = time.time() - start_time
            
            # Apply pruning
            original_model, pruned_count, _ = apply_pruning(
                original_model, 
                level, 
                verbose=verbose,
                quiet=not verbose
            )
            
            # Warmup run if requested
            if use_warmup:
                print("Performing warmup runs...")
                # First run with short generation to initialize caches
                _ = evaluate_model(
                    original_model,
                    tokenizer,
                    prompts[:1],
                    20,  # Short generation to warm up caches
                    temperature=temperature,
                    device=device,
                    quiet=True
                )
                # Second run with actual token count
                _ = evaluate_model(
                    original_model,
                    tokenizer,
                    prompts[:1],
                    num_tokens,
                    temperature=temperature,
                    device=device,
                    quiet=True
                )
            
            # Time evaluation
            start_time = time.time()
            original_results = evaluate_model(
                original_model,
                tokenizer,
                prompts,
                num_tokens,
                temperature=temperature,
                device=device,
                quiet=not verbose
            )
            eval_time_original = time.time() - start_time
            
            # Free memory
            del baseline_model
            del original_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test optimized model
            print("\nTesting optimized model...")
            # Force use of optimized model
            os.environ["USE_OPTIMIZED_MODEL"] = "1"
            
            # Load baseline model first
            baseline_model = load_baseline_model(model_name, device)
            
            # Then load optimized adaptive model
            # Set optimization level appropriately based on device type
            # Level 2 is best for CPU when agency features are needed
            # Level 3 is best for GPU or when maximum speed is required
            recommended_level = "3" if device == "cuda" else "2"
            
            # Only override if user hasn't explicitly set a different level
            if str(optimization_level) not in ["0", "1", "2", "3"]:
                os.environ["OPTIMIZATION_LEVEL"] = recommended_level
            else:
                # Use user's explicit choice
                os.environ["OPTIMIZATION_LEVEL"] = str(optimization_level)
                
            print(f"    Using optimization level {os.environ['OPTIMIZATION_LEVEL']}")
            
            start_time = time.time()
            optimized_model = load_adaptive_model(
                model_name, 
                baseline_model, 
                device,
                debug=False,
                quiet=not verbose
            )
            load_time_optimized = time.time() - start_time
            
            # Apply pruning
            optimized_model, pruned_count, _ = apply_pruning(
                optimized_model, 
                level, 
                verbose=verbose,
                quiet=not verbose
            )
            
            # Warmup run if requested
            if use_warmup:
                print("Performing warmup runs...")
                # First run with short generation to initialize caches
                _ = evaluate_model(
                    optimized_model,
                    tokenizer,
                    prompts[:1],
                    20,  # Short generation to warm up caches
                    temperature=temperature,
                    device=device,
                    quiet=True
                )
                # Second run with actual token count
                _ = evaluate_model(
                    optimized_model,
                    tokenizer,
                    prompts[:1],
                    num_tokens,
                    temperature=temperature,
                    device=device,
                    quiet=True
                )
            
            # Time evaluation
            start_time = time.time()
            optimized_results = evaluate_model(
                optimized_model,
                tokenizer,
                prompts,
                num_tokens,
                temperature=temperature,
                device=device,
                quiet=not verbose
            )
            eval_time_optimized = time.time() - start_time
            
            # Free memory
            del baseline_model
            del optimized_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Store iteration results
            results["original"][level]["times"].append(eval_time_original)
            results["optimized"][level]["times"].append(eval_time_optimized)
            
            if iteration == 0:
                # Store first iteration metrics
                results["original"][level].update({
                    "load_time": load_time_original,
                    "tokens_per_second": original_results["tokens_per_second"],
                    "perplexity": original_results["perplexity"],
                    "diversity": original_results["diversity"]
                })
                
                results["optimized"][level].update({
                    "load_time": load_time_optimized,
                    "tokens_per_second": optimized_results["tokens_per_second"],
                    "perplexity": optimized_results["perplexity"],
                    "diversity": optimized_results["diversity"]
                })
            
            # Calculate speedup for this iteration
            speedup = eval_time_original / eval_time_optimized
            tokens_speedup = optimized_results["tokens_per_second"] / original_results["tokens_per_second"]
            
            print(f"\nResults for iteration {iteration+1}:")
            print(f"  Original model: {eval_time_original:.2f}s ({original_results['tokens_per_second']:.2f} tokens/sec)")
            print(f"  Optimized model: {eval_time_optimized:.2f}s ({optimized_results['tokens_per_second']:.2f} tokens/sec)")
            print(f"  Speedup: {speedup:.2f}x (raw), {tokens_speedup:.2f}x (tokens/sec)")
        
        # Calculate average speedup
        avg_time_original = sum(results["original"][level]["times"]) / len(results["original"][level]["times"])
        avg_time_optimized = sum(results["optimized"][level]["times"]) / len(results["optimized"][level]["times"])
        avg_speedup = avg_time_original / avg_time_optimized
        
        results["speedup"][level] = {
            "time_speedup": avg_speedup,
            "tokens_speedup": results["optimized"][level]["tokens_per_second"] / results["original"][level]["tokens_per_second"],
            "quality_ratio": results["original"][level]["perplexity"] / results["optimized"][level]["perplexity"]
        }
        
        print(f"\nAverage speedup at {level}% pruning: {avg_speedup:.2f}x")
        print(f"Tokens per second: {results['optimized'][level]['tokens_per_second']:.2f} vs {results['original'][level]['tokens_per_second']:.2f}")
        print(f"Perplexity: {results['optimized'][level]['perplexity']:.2f} vs {results['original'][level]['perplexity']:.2f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF OPTIMIZED MODEL PERFORMANCE")
    print("="*60)
    
    # Check if we have results for typical test levels
    key_pruning_levels = [0, 30, 70]
    available_levels = [level for level in key_pruning_levels if level in pruning_levels_list]
    
    # Detailed breakdown
    print("\nPerformance by pruning level:")
    for level in pruning_levels_list:
        speedup = results["speedup"][level]["tokens_speedup"]
        quality = results["speedup"][level]["quality_ratio"]
        quality_str = "better" if quality > 1.0 else "worse"
        
        # Get more detailed performance metrics
        original_speed = results["original"][level]["tokens_per_second"]
        optimized_speed = results["optimized"][level]["tokens_per_second"]
        
        # Add warning markers if any results are counter-intuitive based on our profiling
        warning = ""
        if original_speed > 25 and level >= 70:
            warning = "  ← Best for pure throughput"
        elif optimized_speed > original_speed * 1.2 and level == 30:
            warning = "  ← Best for agency features"
            
        print(f"  Level {level}%: {speedup:.2f}x faster, {abs(1-quality):.2f}x {quality_str} quality")
        print(f"    Original: {original_speed:.2f} tokens/sec, Optimized: {optimized_speed:.2f} tokens/sec{warning}")
    
    # Add note about performance variation
    print("\nNOTE: Performance may vary based on hardware environment.")
    print("Our extensive profiling shows that for agency features,")
    print("optimization level 2 works best on CPU and level 3 on GPU.")
    print("For pure throughput, the original model with 70% pruning is fastest.")
    
    # Save results
    results_file = f"test_results/comparison_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")
    
    return results

# Display version info
print(f"\nRunning SentinelAI Colab Testing Script v{VERSION}")

# Run the comparison
results = compare_models()

# @title Visualize Results
# @markdown Run this cell to visualize the performance comparison results

def create_visualizations(results):
    """Create visualizations from the performance comparison results."""
    # Get pruning levels
    pruning_levels = sorted(list(results["original"].keys()))
    
    # Extract metrics
    original_speed = [results["original"][level]["tokens_per_second"] for level in pruning_levels]
    optimized_speed = [results["optimized"][level]["tokens_per_second"] for level in pruning_levels]
    
    original_ppl = [results["original"][level]["perplexity"] for level in pruning_levels]
    optimized_ppl = [results["optimized"][level]["perplexity"] for level in pruning_levels]
    
    speedup = [results["speedup"][level]["tokens_speedup"] for level in pruning_levels]
    quality_ratio = [results["speedup"][level]["quality_ratio"] for level in pruning_levels]
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # 1. Speed comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, original_speed, 'o-', label="Original", color="#78909C", linewidth=2)
    plt.plot(pruning_levels, optimized_speed, 'o-', 
             label=f"Optimized (Level {results['metadata']['optimization_level']})", 
             color="#4CAF50", linewidth=2)
    
    model_name = results['metadata']['model_name']
    device_type = results['metadata']['device']
    plt.title(f"Generation Speed Comparison - {model_name} on {device_type}", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.xticks(pruning_levels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add annotations
    for i, level in enumerate(pruning_levels):
        plt.annotate(f"{optimized_speed[i]:.1f}",
                    xy=(level, optimized_speed[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10)
        plt.annotate(f"{original_speed[i]:.1f}",
                    xy=(level, original_speed[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig("test_results/speed_comparison.png", dpi=150)
    plt.show()
    
    # 2. Perplexity comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, original_ppl, 'o-', label="Original", color="#78909C", linewidth=2)
    plt.plot(pruning_levels, optimized_ppl, 'o-', 
             label=f"Optimized (Level {results['metadata']['optimization_level']})", 
             color="#4CAF50", linewidth=2)
    
    plt.title(f"Perplexity Comparison - {model_name} (Lower is Better)", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.xticks(pruning_levels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add annotations
    for i, level in enumerate(pruning_levels):
        plt.annotate(f"{optimized_ppl[i]:.1f}",
                    xy=(level, optimized_ppl[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10)
        plt.annotate(f"{original_ppl[i]:.1f}",
                    xy=(level, original_ppl[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig("test_results/perplexity_comparison.png", dpi=150)
    plt.show()
    
    # 3. Speedup and quality ratio
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, speedup, 'o-', label="Speed Improvement", color="#1976D2", linewidth=2)
    plt.plot(pruning_levels, quality_ratio, 'o-', label="Quality Ratio", color="#D32F2F", linewidth=2)
    
    # Add horizontal line at y=1.0
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    plt.title(f"Performance Improvement Factors - Level {results['metadata']['optimization_level']}", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Improvement Factor (>1 is better)", fontsize=14)
    plt.xticks(pruning_levels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add annotations
    for i, level in enumerate(pruning_levels):
        plt.annotate(f"{speedup[i]:.1f}x",
                    xy=(level, speedup[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10)
        plt.annotate(f"{quality_ratio[i]:.2f}",
                    xy=(level, quality_ratio[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig("test_results/improvement_factors.png", dpi=150)
    plt.show()
    
    # 4. Bar chart of speedup by pruning level
    plt.figure(figsize=(10, 6))
    bars = plt.bar(pruning_levels, speedup, color="#1976D2", alpha=0.7)
    
    plt.title(f"Speed Improvement by Pruning Level - Opt Level {results['metadata']['optimization_level']}", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Speedup Factor (x times faster)", fontsize=14)
    plt.xticks(pruning_levels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.1f}x",
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("test_results/speedup_by_level.png", dpi=150)
    plt.show()
    
    # 5. Create summary table
    print("\n=== PERFORMANCE SUMMARY TABLE ===")
    print(f"{'Pruning %':<10} {'Orig Speed':<12} {'Opt Speed':<12} {'Speedup':<10} {'Orig PPL':<10} {'Opt PPL':<10} {'Quality':<10}")
    print(f"{'-'*70}")
    
    for i, level in enumerate(pruning_levels):
        quality_str = 'better' if quality_ratio[i] > 1.0 else 'worse'
        quality_pct = abs(1.0 - quality_ratio[i]) * 100
        
        print(f"{level:<10} {original_speed[i]:<12.2f} {optimized_speed[i]:<12.2f} "
              f"{speedup[i]:<10.2f}x {original_ppl[i]:<10.2f} {optimized_ppl[i]:<10.2f} "
              f"{quality_pct:.1f}% {quality_str}")
    
    return True

# Visualize the results
create_visualizations(results)

# Display key findings for user reference
print("\n===== Key Findings Based on Profiling =====")
print("1. For maximum speed: Use original model with 70% pruning")
print("2. For agency features: Use optimized level 2 on CPU, level 3 on GPU")
print("3. For balanced quality/speed: Use 30% pruning")

# @title Save Results to Google Drive (Optional)
# @markdown Run this after the test completes to save results to Google Drive
mount_drive = False  # @param {type:"boolean"}
drive_folder = "SentinelAI_Results"  # @param {type:"string"}

if mount_drive:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create folder if it doesn't exist
    drive_path = f"/content/drive/My Drive/{drive_folder}"
    os.makedirs(drive_path, exist_ok=True)
    
    # Create a summary file with optimization recommendations
    summary_file = "test_results/optimization_recommendations.txt"
    with open(summary_file, "w") as f:
        f.write("SentinelAI Optimization Recommendations\n")
        f.write("=====================================\n\n")
        f.write("Based on extensive profiling of the SentinelAI model, we recommend:\n\n")
        f.write("1. For maximum throughput (pure speed):\n")
        f.write("   - Use original model with 70% pruning (~28 tokens/sec on CPU)\n\n")
        f.write("2. For models with agency features:\n")
        f.write("   - On CPU: Use optimization level 2 with 30% pruning (~19-20 tokens/sec)\n")
        f.write("   - On GPU: Use optimization level 3 with 30% pruning\n\n")
        f.write("3. For balanced quality/performance:\n")
        f.write("   - Use optimization level 2 with 30% pruning\n\n")
        f.write("Notes:\n")
        f.write("- Original model with heavy pruning (70%) outperforms optimized models for pure throughput\n")
        f.write("- Optimized models maintain better agency capabilities even with pruning\n")
        f.write("- Colab environment performance may vary from local environments\n")
    
    # Copy results to Drive
    print(f"Copying results to Google Drive: {drive_path}")
    !cp -r test_results/* {drive_path}/
    print("Results and optimization recommendations successfully copied to Google Drive")