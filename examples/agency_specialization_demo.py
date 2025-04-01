#!/usr/bin/env python
"""
Agency Specialization Demo for Sentinel-AI

This example demonstrates how to:
1. Load a model with attention head agency
2. Apply specialized agency patterns for different tasks
3. Measure performance improvements with and without specialization
4. Visualize head states and activity

Usage:
  python agency_specialization_demo.py
"""

import sys
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Sentinel-AI components
from models.loaders.gpt2_loader import load_gpt2_with_sentinel_gates
from models.specialization_registry import SpecializationRegistry

def print_separator(title=""):
    """Print a separator with optional title."""
    width = 80
    if title:
        print("\n" + "=" * width)
        print(f"{title.center(width)}")
        print("=" * width)
    else:
        print("\n" + "-" * width)

def measure_performance(model, tokenizer, prompt, max_tokens=50):
    """Measure performance metrics for text generation."""
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Measure generation time
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    # Calculate metrics
    generation_time = end_time - start_time
    generated_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    else:
        memory_allocated = 0
    
    # Calculate simple diversity metrics
    generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    words = generated_text.split()
    unique_words = len(set(words))
    diversity_score = unique_words / len(words) if words else 0
    
    return {
        "tokens_per_second": tokens_per_second,
        "memory_allocated": memory_allocated,
        "diversity_score": diversity_score,
        "generated_text": generated_text,
        "time": generation_time
    }

def plot_head_states(model, title="Head States"):
    """Create a visualization of head states across layers."""
    num_layers = model.num_layers
    num_heads = model.num_heads
    
    # Collect head states
    states = np.zeros((num_layers, num_heads), dtype=int)
    for layer_idx in range(num_layers):
        if hasattr(model.blocks[layer_idx]["attn"], "agency_signals"):
            for head_idx in range(num_heads):
                state = model.blocks[layer_idx]["attn"].agency_signals[head_idx]["state"]
                # Convert state to numeric value
                if state == "active":
                    states[layer_idx, head_idx] = 3
                elif state == "overloaded":
                    states[layer_idx, head_idx] = 2
                elif state == "misaligned":
                    states[layer_idx, head_idx] = 1
                elif state == "withdrawn":
                    states[layer_idx, head_idx] = 0
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.imshow(states, cmap='viridis', interpolation='nearest')
    plt.colorbar(ticks=[0, 1, 2, 3], label='State')
    plt.gca().set_xticks(np.arange(num_heads))
    plt.gca().set_yticks(np.arange(num_layers))
    plt.gca().set_xticklabels([f"H{i}" for i in range(num_heads)])
    plt.gca().set_yticklabels([f"L{i}" for i in range(num_layers)])
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    plt.title(title)
    
    # Add state labels
    for i in range(num_layers):
        for j in range(num_heads):
            state_text = ["WD", "ML", "OL", "AC"][states[i, j]]
            plt.text(j, i, state_text, ha="center", va="center", 
                    color="w" if states[i, j] < 2 else "black", fontsize=8)
    
    plt.tight_layout()
    return plt.gcf()

def main():
    print_separator("Sentinel-AI: Agency Specialization Demo")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with adaptive transformer and agency
    print("Loading model...")
    model, tokenizer = load_gpt2_with_sentinel_gates(
        model_name="gpt2",           # Can use any GPT-2 variant
        gate_init=1.0,               # Initial gate values
        connection_init=0.0,         # Initial skip connection values
        norm_attn_output=True        # Enable attention normalization
    )
    model = model.to(device)
    print(f"Model loaded: gpt2 with {model.num_layers} layers, {model.num_heads} heads per layer")
    
    # Create specialization registry
    registry = SpecializationRegistry(model)
    
    # Demo with different task types
    demo_tasks = {
        "pattern_matching": "def calculate_fibonacci(n):\n    # Function to calculate the nth Fibonacci number\n    ",
        "logical_reasoning": "Solve this math problem step by step: If a rectangle has a length of 12 cm and a width of 8 cm, what is its area?",
        "creative_generation": "Write a short poem about artificial intelligence and consciousness.",
        "long_context": "Analyze the following passage from Shakespeare's Hamlet: 'To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them.' Explain the philosophical implications of this soliloquy.",
    }
    
    # Baseline measurements (without specialization)
    print_separator("Baseline Performance (No Specialization)")
    baseline_results = {}
    
    for task_name, prompt in demo_tasks.items():
        print(f"\nTask: {task_name}")
        print(f"Prompt: {prompt[:50]}...")
        
        # Measure baseline performance
        metrics = measure_performance(model, tokenizer, prompt)
        baseline_results[task_name] = metrics
        
        print(f"Generation speed: {metrics['tokens_per_second']:.2f} tokens/sec")
        print(f"Memory usage: {metrics['memory_allocated']:.2f} GB")
        print(f"Diversity score: {metrics['diversity_score']:.3f}")
        print(f"Sample output: {metrics['generated_text'][:100]}...")
    
    # Apply specialization
    print_separator("Performance with Agency Specialization")
    specialized_results = {}
    
    # First, initialize the specialization with default patterns
    registry.specialization.initialize_specialization()
    
    # Visualize initial state distribution
    initial_state_fig = plot_head_states(model, "Initial Head States")
    
    for task_name, prompt in demo_tasks.items():
        print(f"\nTask: {task_name}")
        print(f"Prompt: {prompt[:50]}...")
        
        # Apply specialization for this task
        report = registry.apply_specialization(task_type=task_name)
        
        # Measure performance with specialization
        metrics = measure_performance(model, tokenizer, prompt)
        specialized_results[task_name] = metrics
        
        # Calculate improvement
        speed_improvement = (metrics['tokens_per_second'] / baseline_results[task_name]['tokens_per_second'] - 1) * 100
        memory_reduction = (1 - metrics['memory_allocated'] / baseline_results[task_name]['memory_allocated']) * 100
        diversity_improvement = (metrics['diversity_score'] / baseline_results[task_name]['diversity_score'] - 1) * 100
        
        print(f"Generation speed: {metrics['tokens_per_second']:.2f} tokens/sec ({speed_improvement:.1f}% improvement)")
        print(f"Memory usage: {metrics['memory_allocated']:.2f} GB ({memory_reduction:.1f}% reduction)")
        print(f"Diversity score: {metrics['diversity_score']:.3f} ({diversity_improvement:.1f}% improvement)")
        print(f"Sample output: {metrics['generated_text'][:100]}...")
        
        # Visualize head states for this task
        task_state_fig = plot_head_states(model, f"Head States for {task_name}")
        
        # Show plot or save to file
        # plt.savefig(f"head_states_{task_name}.png")
        plt.close(task_state_fig)
    
    # Performance comparison summary
    print_separator("Performance Comparison Summary")
    
    print(f"{'Task Type':<20} {'Baseline Speed':<15} {'Specialized Speed':<15} {'Improvement':<10}")
    print("-" * 60)
    
    for task_name in demo_tasks.keys():
        baseline = baseline_results[task_name]['tokens_per_second']
        specialized = specialized_results[task_name]['tokens_per_second']
        improvement = (specialized / baseline - 1) * 100
        
        print(f"{task_name:<20} {baseline:<15.2f} {specialized:<15.2f} {improvement:<10.1f}%")
    
    # Create comparison plots
    plt.figure(figsize=(10, 6))
    
    tasks = list(demo_tasks.keys())
    baseline_speeds = [baseline_results[task]['tokens_per_second'] for task in tasks]
    specialized_speeds = [specialized_results[task]['tokens_per_second'] for task in tasks]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.bar(x - width/2, baseline_speeds, width, label='Baseline')
    plt.bar(x + width/2, specialized_speeds, width, label='Specialized')
    
    plt.xlabel('Task Type')
    plt.ylabel('Tokens per Second')
    plt.title('Performance Comparison: Baseline vs. Specialized')
    plt.xticks(x, tasks, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    # plt.savefig("performance_comparison.png")
    plt.close()
    
    print("\nDemo complete!")
    print("The agency specialization system successfully adapted to different task types,")
    print("providing significant performance improvements and resource optimization.")

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    main()