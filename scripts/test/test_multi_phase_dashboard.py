#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Multi-Phase Dashboard with ANN Controller Integration.

This script creates a synthetic experiment with multiple phases and cycles,
and generates a comprehensive dashboard to demonstrate the visualization
capabilities of the multi-phase dashboard with controller integration.

Version: v0.0.1 (2025-04-20 19:54:00)
"""

import os
import sys
import numpy as np
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our dashboard
from sentinel.plasticity.visualization.multi_phase.dashboard import MultiPhaseDashboard

def create_synthetic_experiment_data(num_steps=100, cycles=2, num_layers=6, num_heads=8):
    """
    Create synthetic experiment data for demonstration purposes.
    
    Args:
        num_steps: Total number of steps in the experiment
        cycles: Number of pruning cycles
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        
    Returns:
        Dictionary of experiment data
    """
    # Create base metrics with realistic patterns
    steps = list(range(num_steps))
    
    # Create loss curve with realistic noise and improvements after pruning
    base_loss = np.linspace(5.0, 2.0, num_steps)  # General downward trend
    noise = np.random.normal(0, 0.1, num_steps)   # Add noise
    loss = base_loss + noise
    
    # Add jumps after pruning events for realism
    steps_per_cycle = num_steps // cycles
    for i in range(1, cycles):
        pruning_step = i * steps_per_cycle
        loss[pruning_step:pruning_step+5] += 0.5  # Loss spike after pruning
        
    # Perplexity (exponential of loss)
    perplexity = np.exp(loss * 0.5)  # Scale down for realistic values
    
    # Sparsity increases with pruning events
    sparsity = np.zeros(num_steps)
    for i in range(1, cycles + 1):
        pruning_step = i * steps_per_cycle
        if pruning_step < num_steps:
            sparsity[pruning_step:] += 0.15  # 15% sparsity increase per cycle
    
    # Phase transitions
    phase_transitions = []
    phases = []
    
    # Warmup phase
    warmup_end = int(num_steps * 0.2)
    phase_transitions.append(0)
    phases.append("warmup")
    
    # Cycles of pruning and fine-tuning
    for i in range(cycles):
        # Analysis phase before pruning
        analysis_start = warmup_end if i == 0 else fine_tuning_end
        phase_transitions.append(analysis_start)
        phases.append("analysis")
        
        # Pruning phase
        pruning_start = analysis_start + int(num_steps * 0.05)
        phase_transitions.append(pruning_start)
        phases.append("pruning")
        
        # Fine-tuning phase
        fine_tuning_start = pruning_start + 1  # Right after pruning
        phase_transitions.append(fine_tuning_start)
        phases.append("fine-tuning")
        
        fine_tuning_end = (
            pruning_start + int(num_steps * 0.2) 
            if i < cycles - 1 
            else num_steps - int(num_steps * 0.05)
        )
    
    # Evaluation phase at the end
    evaluation_start = num_steps - int(num_steps * 0.05)
    phase_transitions.append(evaluation_start)
    phases.append("evaluation")
    
    # Create pruning events
    pruning_events = []
    for i in range(cycles):
        pruning_step = warmup_end + i * steps_per_cycle
        if pruning_step < num_steps:
            # Randomly select heads to prune
            num_pruned = int(num_heads * num_layers * 0.15)  # 15% pruning per cycle
            pruned_heads = []
            all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
            already_pruned = set()
            
            for _ in range(num_pruned):
                # Find a head that's not already pruned
                candidates = [h for h in all_heads if h not in already_pruned]
                if candidates:
                    head = random.choice(candidates)
                    pruned_heads.append(f"{head[0]}.{head[1]}")
                    already_pruned.add(head)
            
            pruning_events.append({
                "step": pruning_step,
                "pruned_heads": pruned_heads,
                "strategy": "entropy",
                "pruning_level": 0.15,
                "cycle": i + 1
            })
    
    # Create head metrics and controller decisions
    head_metrics = {}
    controller_decisions = []
    
    # Initialize head sets
    all_head_ids = [f"{l}.{h}" for l in range(num_layers) for h in range(num_heads)]
    head_importances = {h_id: random.uniform(0.3, 0.9) for h_id in all_head_ids}
    head_entropies = {h_id: random.uniform(0.5, 2.0) for h_id in all_head_ids}
    head_magnitudes = {h_id: random.uniform(0.1, 0.8) for h_id in all_head_ids}
    
    # Record multiple steps of head metrics
    metric_steps = list(range(0, num_steps, max(1, num_steps // 10)))
    for step in metric_steps:
        # Add noise to make metrics change over time
        noise_factor = 0.05
        
        # Record entropy at this step
        entropy_data = {}
        for h_id in all_head_ids:
            entropy_data[h_id] = head_entropies[h_id] + random.uniform(-noise_factor, noise_factor)
        
        # Record magnitude at this step
        magnitude_data = {}
        for h_id in all_head_ids:
            magnitude_data[h_id] = head_magnitudes[h_id] + random.uniform(-noise_factor, noise_factor)
        
        # Record importance at this step
        importance_data = {}
        for h_id in all_head_ids:
            importance_data[h_id] = head_importances[h_id] + random.uniform(-noise_factor, noise_factor)
        
        # Add to head metrics
        for h_id in all_head_ids:
            if h_id not in head_metrics:
                head_metrics[h_id] = {"steps": [], "metrics": {}}
            
            head_metrics[h_id]["steps"].append(step)
            
            if "entropy" not in head_metrics[h_id]["metrics"]:
                head_metrics[h_id]["metrics"]["entropy"] = []
            head_metrics[h_id]["metrics"]["entropy"].append(entropy_data[h_id])
            
            if "magnitude" not in head_metrics[h_id]["metrics"]:
                head_metrics[h_id]["metrics"]["magnitude"] = []
            head_metrics[h_id]["metrics"]["magnitude"].append(magnitude_data[h_id])
            
            if "importance" not in head_metrics[h_id]["metrics"]:
                head_metrics[h_id]["metrics"]["importance"] = []
            head_metrics[h_id]["metrics"]["importance"].append(importance_data[h_id])
        
        # Make some controller decisions
        active_heads = [h_id for h_id in all_head_ids 
                        if head_importances[h_id] > 0.5 + random.uniform(-0.1, 0.1)]
        
        controller_decisions.append({
            "step": step,
            "active_heads": active_heads,
            "controller_loss": 0.5 - (step / num_steps) * 0.3  # Decreasing loss
        })
    
    # Create recovery events - some heads get "recovered" later
    recovery_data = {}
    for pruning_event in pruning_events[:-1]:  # Exclude last pruning event
        cycle = pruning_event["cycle"]
        pruned_heads = pruning_event["pruned_heads"]
        
        # Randomly select some heads to recover in next cycle
        recover_count = len(pruned_heads) // 3  # Recover about 1/3 of pruned heads
        heads_to_recover = random.sample(pruned_heads, min(recover_count, len(pruned_heads)))
        
        recovery_step = pruning_event["step"] + steps_per_cycle // 2
        
        for head_id in heads_to_recover:
            recovery_data[head_id] = random.uniform(0.6, 0.9)  # Recovery score
    
    # Create text samples and comparisons
    text_samples = []
    comparisons = []
    
    # Sample prompts
    prompts = [
        "The future of artificial intelligence will be",
        "Neural networks have revolutionized how we approach",
        "Transformer models excel at natural language processing because"
    ]
    
    # Create sample texts
    for i, prompt in enumerate(prompts):
        baseline_text = prompt + " significantly influenced by advancements in neural plasticity research."
        pruned_text = prompt + " dramatically transformed by our ability to efficiently prune and adapt models."
        
        text_samples.append({
            "prompt": prompt,
            "output": pruned_text,
            "model_type": "pruned",
            "step": num_steps - 10
        })
        
        text_samples.append({
            "prompt": prompt,
            "output": baseline_text,
            "model_type": "baseline",
            "step": 5
        })
        
        comparisons.append({
            "prompt": prompt,
            "baseline_text": baseline_text,
            "pruned_text": pruned_text,
            "metrics": {
                "baseline_perplexity": perplexity[5],
                "pruned_perplexity": perplexity[-10],
                "improvement_percent": (perplexity[5] - perplexity[-10]) / perplexity[5] * 100
            },
            "step": num_steps - 5
        })
    
    return {
        "steps": steps,
        "loss": loss,
        "perplexity": perplexity,
        "sparsity": sparsity,
        "phase_transitions": phase_transitions,
        "phases": phases,
        "pruning_events": pruning_events,
        "head_metrics": head_metrics,
        "controller_decisions": controller_decisions,
        "recovery_data": recovery_data,
        "text_samples": text_samples,
        "comparisons": comparisons
    }

def main():
    """
    Main function to test the multi-phase dashboard with synthetic data.
    """
    print("Generating synthetic experiment data...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        project_root,
        "output",
        f"test_dashboard_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create dashboard
    dashboard = MultiPhaseDashboard(
        output_dir=output_dir,
        experiment_name="synthetic_experiment",
        config={"test": True, "timestamp": timestamp}
    )
    
    # Generate synthetic data
    data = create_synthetic_experiment_data(
        num_steps=200,
        cycles=3,
        num_layers=6,
        num_heads=12
    )
    
    # Record phase transitions
    print("Recording phase transitions...")
    for phase, step in zip(data["phases"], data["phase_transitions"]):
        dashboard.record_phase_transition(phase, step)
        print(f"  Phase: {phase}, Step: {step}")
    
    # Record all metrics step by step
    print("Recording metrics...")
    for i, step in enumerate(data["steps"]):
        metrics = {
            "loss": data["loss"][i],
            "perplexity": data["perplexity"][i],
            "sparsity": data["sparsity"][i],
            "phase": data["phases"][np.searchsorted(data["phase_transitions"], step, side="right") - 1]
        }
        dashboard.record_step(metrics, step)
        
        # Mark stabilization point at 20% of warmup
        if step == int(data["phase_transitions"][1] * 0.2):
            metrics["stabilized"] = True
            dashboard.record_step(metrics, step)
            print(f"  Stabilization point at step {step}")
    
    # Record pruning events
    print("Recording pruning events...")
    for event in data["pruning_events"]:
        dashboard.record_pruning_event(event, event["step"])
        print(f"  Pruning at step {event['step']}, cycle {event['cycle']}")
    
    # Record head metrics
    print("Recording head metrics...")
    for head_id, metrics in data["head_metrics"].items():
        dashboard.record_head_metrics({head_id: metrics["metrics"]}, metrics["steps"][0])
    
    # Record controller decisions
    print("Recording controller decisions...")
    for decision in data["controller_decisions"]:
        dashboard.record_controller_decision(decision, decision["step"])
    
    # Record attention entropy and magnitude for analysis
    steps_to_record = sorted(list(set([d["step"] for d in data["controller_decisions"]])))
    for step in steps_to_record:
        entropy_data = {}
        magnitude_data = {}
        importance_data = {}
        
        for head_id in data["head_metrics"]:
            metrics = data["head_metrics"][head_id]["metrics"]
            steps = data["head_metrics"][head_id]["steps"]
            
            if step in steps:
                idx = steps.index(step)
                if idx < len(metrics.get("entropy", [])):
                    entropy_data[head_id] = metrics["entropy"][idx]
                if idx < len(metrics.get("magnitude", [])):
                    magnitude_data[head_id] = metrics["magnitude"][idx]
                if idx < len(metrics.get("importance", [])):
                    importance_data[head_id] = metrics["importance"][idx]
        
        dashboard.record_attention_entropy(entropy_data, step)
        dashboard.record_attention_magnitude(magnitude_data, step)
        dashboard.record_head_importance(importance_data, step)
    
    # Record head recovery data
    print("Recording head recovery data...")
    dashboard.record_head_recovery(data["recovery_data"], data["steps"][-30])
    
    # Record text samples and comparisons
    print("Recording text samples and comparisons...")
    dashboard.text_samples = data["text_samples"]
    dashboard.comparisons = data["comparisons"]
    
    # Generate visualizations
    print("Generating visualizations...")
    
    dashboard.visualize_complete_process(
        os.path.join(output_dir, "complete_process.png")
    )
    
    dashboard.generate_multi_cycle_dashboard(
        os.path.join(output_dir, "multi_cycle_process.png")
    )
    
    dashboard.visualize_head_metrics(
        os.path.join(output_dir, "head_metrics.png")
    )
    
    dashboard.generate_controller_dashboard(
        os.path.join(output_dir, "controller_dashboard.png")
    )
    
    # Generate HTML dashboard
    print("Generating HTML dashboard...")
    dashboard_path = dashboard.generate_standalone_dashboard(
        os.path.join(output_dir, "dashboard")
    )
    
    # Save dashboard data for later analysis
    print("Saving dashboard data...")
    data_path = dashboard.save_dashboard_data(
        os.path.join(output_dir, "dashboard_data")
    )
    
    print(f"Dashboard generated at: {dashboard_path}")
    print(f"Dashboard data saved at: {data_path}")
    
    # Try to open the dashboard in a browser
    try:
        import webbrowser
        print(f"Opening dashboard in browser: file://{os.path.abspath(dashboard_path)}")
        webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
    except Exception as e:
        print(f"Could not open browser: {e}")

if __name__ == "__main__":
    main()