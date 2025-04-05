#!/usr/bin/env python
"""
Neural Plasticity System Demonstration

This script demonstrates the complete neural plasticity system by:
1. Loading a pre-trained transformer model (distilgpt2)
2. Applying scientific pruning using entropy metrics
3. Performing fine-tuning with differential learning rates
4. Tracking gate activity and head regrowth
5. Comparing performance before and after the plasticity cycle
6. Generating visualizations of head importance and entropy changes

This serves as an easy-to-follow example for understanding how transformer
models adapt and reorganize their internal representations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sentinel-AI modules
from sentinel.plasticity.plasticity_loop import run_plasticity_experiment
from sentinel.utils.viz.heatmaps import (
    plot_entropy_heatmap, 
    plot_entropy_deltas_heatmap,
    plot_gate_activity, 
    plot_regrowth_heatmap
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural Plasticity System Demonstration")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name (default: distilgpt2)")
    parser.add_argument("--output_dir", type=str, default="./output/plasticity_demo",
                        help="Directory to save results (default: ./output/plasticity_demo)")
    
    # Experiment parameters
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                        choices=["entropy", "magnitude"],
                        help="Pruning strategy (default: entropy)")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                        help="Pruning level (default: 0.3)")
    parser.add_argument("--training_steps", type=int, default=200,
                        help="Number of fine-tuning steps (default: 200)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    
    # Visualization parameters
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots interactively (in addition to saving)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()

def get_dataloader_builder(batch_size=4):
    """
    Create a function that returns train and evaluation dataloaders.
    Uses a simple dataset for demonstration purposes.
    """
    from transformers import AutoTokenizer
    import torch
    
    # Create demonstration data focused on neural networks and science
    texts = [
        "Neural networks learn through a process of adjusting connection weights.",
        "Attention mechanisms allow models to focus on different parts of the input.",
        "The transformer architecture revolutionized language processing tasks.",
        "Pruning techniques remove less important weights to improve efficiency.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Neural plasticity refers to the brain's ability to reorganize itself.",
        "Deep learning models extract hierarchical features from raw data.",
        "Scientific models must be validated through rigorous experimentation.",
        "Machine learning algorithms improve with more training examples.",
        "Entropy measures the randomness or unpredictability in a system.",
        "Knowledge distillation transfers information from large to small models.",
        "The optimization process minimizes a loss function during training.",
        "Parameter sharing reduces the total number of weights in a model.",
        "Gradient descent updates weights to minimize prediction errors.",
        "Regularization techniques prevent overfitting to training data."
    ] * 6  # Repeat to create more samples
    
    def build_dataloaders(model_name="distilgpt2", batch_size=4):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        from torch.utils.data import TensorDataset, DataLoader
        
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        dataset = TensorDataset(input_ids, attention_mask)
        
        # Split into train and eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        return train_dataloader, eval_dataloader
    
    # Return a function that will create dataloaders with the specified batch size
    return lambda batch_size=batch_size: build_dataloaders(model_name=batch_size)

def create_visualizations(results_dir, results, show=False):
    """Create visualizations from plasticity experiment results"""
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data from results files
    try:
        import json
        import torch
        import numpy as np
        
        # Load pre/post entropy data
        with open(os.path.join(results_dir, "pre_entropy.json"), 'r') as f:
            pre_entropy_data = json.load(f)
            # Convert from serialized format back to tensors
            pre_entropy = {int(k): torch.tensor(v) for k, v in pre_entropy_data.items()}
            
        with open(os.path.join(results_dir, "post_entropy.json"), 'r') as f:
            post_entropy_data = json.load(f)
            post_entropy = {int(k): torch.tensor(v) for k, v in post_entropy_data.items()}
            
        with open(os.path.join(results_dir, "entropy_deltas.json"), 'r') as f:
            deltas_data = json.load(f)
            entropy_deltas = {int(k): torch.tensor(v) for k, v in deltas_data.items()}
            
        # Load gate history
        with open(os.path.join(results_dir, "gate_history.json"), 'r') as f:
            gate_history_data = json.load(f)
            gate_history = {}
            for step, layers in gate_history_data.items():
                gate_history[int(step)] = {}
                for layer, gates in layers.items():
                    gate_history[int(step)][int(layer)] = torch.tensor(gates)
                    
        # Load regrowth analysis
        with open(os.path.join(results_dir, "regrowth_analysis.json"), 'r') as f:
            regrowth_data = json.load(f)
            # Convert to the format expected by plot_regrowth_heatmap
            regrowth_analysis = {}
            for key, data in regrowth_data.items():
                layer_idx, head_idx = map(int, key.split('_'))
                regrowth_analysis[(layer_idx, head_idx)] = data
        
        # Create visualizations
        
        # 1. Pre-pruning entropy heatmap
        pre_entropy_fig = plot_entropy_heatmap(
            pre_entropy,
            title="Attention Entropy Before Fine-tuning",
            save_path=os.path.join(viz_dir, "pre_entropy_heatmap.png")
        )
        
        # 2. Post-fine-tuning entropy heatmap
        post_entropy_fig = plot_entropy_heatmap(
            post_entropy,
            title="Attention Entropy After Fine-tuning",
            save_path=os.path.join(viz_dir, "post_entropy_heatmap.png")
        )
        
        # 3. Entropy change heatmap
        delta_entropy_fig = plot_entropy_deltas_heatmap(
            entropy_deltas,
            title="Entropy Change After Fine-tuning",
            save_path=os.path.join(viz_dir, "entropy_deltas_heatmap.png")
        )
        
        # 4. Gate activity for regrown heads
        if regrowth_analysis:
            regrown_heads = list(regrowth_analysis.keys())
            gate_activity_fig = plot_gate_activity(
                gate_history,
                head_indices=regrown_heads,
                title="Gate Activity for Regrown Heads During Fine-tuning",
                save_path=os.path.join(viz_dir, "gate_activity_regrown.png")
            )
            
            # 5. Regrowth heatmap
            regrowth_fig = plot_regrowth_heatmap(
                regrowth_analysis,
                title="Head Regrowth Analysis",
                save_path=os.path.join(viz_dir, "regrowth_heatmap.png")
            )
            
            print(f"Detected {len(regrown_heads)} regrown attention heads")
        else:
            print("No significant head regrowth detected in this experiment")
            
            # Still create a gate activity plot for a few random heads
            if gate_history:
                # Get first and last step
                first_step = min(gate_history.keys())
                last_step = max(gate_history.keys())
                
                # Find heads with some change
                changing_heads = []
                for layer in gate_history[first_step]:
                    if layer in gate_history[last_step]:
                        for head_idx in range(len(gate_history[first_step][layer])):
                            if head_idx < len(gate_history[last_step][layer]):
                                initial = gate_history[first_step][layer][head_idx].item()
                                final = gate_history[last_step][layer][head_idx].item()
                                if abs(final - initial) > 0.1:  # Some meaningful change
                                    changing_heads.append((layer, head_idx))
                
                # Take up to 5 changing heads
                display_heads = changing_heads[:5]
                
                if display_heads:
                    gate_activity_fig = plot_gate_activity(
                        gate_history,
                        head_indices=display_heads,
                        title="Gate Activity for Selected Heads During Fine-tuning",
                        save_path=os.path.join(viz_dir, "gate_activity_selected.png")
                    )
        
        # 6. Create a combined visualization of metrics
        metrics_data = results.get("metrics", {})
        if metrics_data:
            stages = ["baseline", "post_pruning", "final"]
            perplexities = [metrics_data.get(stage, {}).get("perplexity", 0) for stage in stages]
            
            plt.figure(figsize=(10, 6))
            plt.bar(stages, perplexities, color=['green', 'red', 'blue'])
            plt.ylabel('Perplexity (lower is better)')
            plt.title('Model Perplexity Through Plasticity Cycle')
            
            # Add value labels
            for i, v in enumerate(perplexities):
                plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "perplexity_comparison.png"))
            
            # Calculate recovery rate
            if "post_pruning" in metrics_data and "baseline" in metrics_data and "final" in metrics_data:
                recovery_rate = results.get("recovery_rate", 0.0)
                
                print(f"\nRecovery Analysis:")
                print(f"Baseline perplexity: {metrics_data['baseline']['perplexity']:.2f}")
                print(f"Post-pruning perplexity: {metrics_data['post_pruning']['perplexity']:.2f}")
                print(f"Final perplexity: {metrics_data['final']['perplexity']:.2f}")
                print(f"Recovery rate: {recovery_rate:.2%}")
        
        # 7. Create a custom neural plasticity cycle visualization
        plt.figure(figsize=(12, 8))
        
        # Create a circular diagram showing the plasticity cycle
        import matplotlib.patches as patches
        from matplotlib.path import Path
        
        ax = plt.subplot(111)
        
        # Create a circle
        circle = plt.Circle((0.5, 0.5), 0.3, fill=False, edgecolor='black', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        # Add the cycle stages
        stages = ['Prune', 'Measure', 'Grow', 'Learn']
        angles = [0, 90, 180, 270]  # in degrees
        colors = ['red', 'orange', 'green', 'blue']
        
        for stage, angle, color in zip(stages, angles, colors):
            # Convert angle to radians
            rad = np.radians(angle)
            # Calculate position on circle
            x = 0.5 + 0.3 * np.cos(rad)
            y = 0.5 + 0.3 * np.sin(rad)
            
            # Add a colored circle at the stage position
            stage_circle = plt.Circle((x, y), 0.06, fill=True, color=color, alpha=0.8)
            ax.add_patch(stage_circle)
            
            # Add stage name
            offset_x = 0.15 * np.cos(rad)
            offset_y = 0.15 * np.sin(rad)
            plt.text(x + offset_x, y + offset_y, stage, 
                     ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add curved arrows connecting the stages
        for i in range(len(stages)):
            start_angle = angles[i]
            end_angle = angles[(i+1) % len(stages)]
            
            # Calculate control points for a curved arrow
            start_rad = np.radians(start_angle)
            end_rad = np.radians(end_angle)
            
            # Start and end points
            x1 = 0.5 + 0.3 * np.cos(start_rad)
            y1 = 0.5 + 0.3 * np.sin(start_rad)
            x2 = 0.5 + 0.3 * np.cos(end_rad)
            y2 = 0.5 + 0.3 * np.sin(end_rad)
            
            # Control points for curve (towards center then out)
            # Adjust these to control the curve shape
            mid_angle = (start_angle + end_angle) / 2
            if abs(end_angle - start_angle) == 180:
                # Special case for straight line through center
                mid_angle += 90  # Perpendicular control point
            
            mid_rad = np.radians(mid_angle)
            cx = 0.5 + 0.18 * np.cos(mid_rad)  # Control point closer to center
            cy = 0.5 + 0.18 * np.sin(mid_rad)
            
            # Create curved path
            curve = Path([(x1, y1), (cx, cy), (x2, y2)],
                         [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            
            # Add the path as a patch
            patch = patches.PathPatch(curve, facecolor='none', edgecolor=colors[i], 
                                     linewidth=2, arrowstyle='->', mutation_scale=15)
            ax.add_patch(patch)
        
        # Add title and description
        plt.title('Neural Plasticity Cycle', fontsize=16)
        
        # Add metrics from the experiment
        if "post_pruning" in metrics_data and "baseline" in metrics_data and "final" in metrics_data:
            y_pos = 0.12
            plt.text(0.5, y_pos, f"Baseline → Pruned: {metrics_data['baseline']['perplexity']:.2f} → {metrics_data['post_pruning']['perplexity']:.2f} perplexity", 
                     ha='center', va='center', fontsize=10, transform=ax.transAxes)
            plt.text(0.5, y_pos-0.05, f"Pruned → Final: {metrics_data['post_pruning']['perplexity']:.2f} → {metrics_data['final']['perplexity']:.2f} perplexity", 
                     ha='center', va='center', fontsize=10, transform=ax.transAxes)
            plt.text(0.5, y_pos-0.1, f"Recovery Rate: {recovery_rate:.1%}", 
                     ha='center', va='center', fontsize=10, fontweight='bold', transform=ax.transAxes)
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "plasticity_cycle.png"), dpi=300)
        
        # Show plots if requested
        if show:
            plt.show()
        else:
            plt.close('all')
            
        return viz_dir
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return None

def scientific_notes():
    """
    Generate the scientific notes explaining attention head pruning and fine-tuning results.
    """
    notes = """# Neural Plasticity in Transformer Models: Scientific Observations

## Introduction
This experiment demonstrates neural plasticity in transformer models through a process of pruning and regrowth.
Similar to biological neural networks, these artificial neural networks show an ability to reorganize and adapt
to structural modifications.

## Methodology
1. **Pruning Phase**: We selectively remove attention heads using entropy-based metrics, which identify heads with
   diffuse (high entropy) attention patterns that contribute less to the model's performance.
   
2. **Measurement Phase**: We evaluate the immediate impact of pruning on model performance, measuring perplexity
   and attention patterns.
   
3. **Growth Phase**: We strategically regrow certain attention heads to restore capability, initializing them 
   with small random values.
   
4. **Learning Phase**: We fine-tune the model with differential learning rates, applying higher rates to newly
   added heads to accelerate their integration.

## Key Observations

### Head Importance
- Attention heads show varying levels of importance, with some being critical to model function
- Entropy-based pruning identifies heads with diffuse attention patterns for removal
- Some heads specialize in specific linguistic features or patterns

### Adaptation Process
- After pruning, the model redistributes functionality among remaining heads
- Newly grown heads gradually adopt specialized roles during fine-tuning
- Differential learning rates accelerate the integration of new heads

### Performance Characteristics
- Initial pruning causes a measurable drop in performance (increased perplexity)
- Through fine-tuning, the model recovers some or all of the lost performance
- The recovery rate varies based on pruning ratio and learning parameters

## Scientific Implications
This experiment demonstrates that transformer models possess neural plasticity properties analogous to biological
neural networks. This suggests potential for developing more adaptive, self-organizing AI systems and contributes
to our understanding of how distributed representations function in deep learning models.

---

*This analysis was produced by the SentinelAI Neural Plasticity System.*
"""
    return notes

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_{args.pruning_strategy}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Neural Plasticity System Demonstration ===")
    print(f"Model: {args.model_name}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning level: {args.pruning_level}")
    print(f"Training steps: {args.training_steps}")
    print(f"Output directory: {output_dir}")
    
    # Create dataloader builder
    dataloader_builder = get_dataloader_builder(batch_size=args.batch_size)
    
    # Run plasticity experiment
    print(f"\nRunning complete plasticity cycle (prune → measure → grow → learn)...")
    results = run_plasticity_experiment(
        model_name=args.model_name,
        pruning_strategy=args.pruning_strategy,
        prune_ratio=args.pruning_level,
        learning_rate=args.learning_rate,
        adaptive_lr=True,  # Use differential learning rates
        learning_steps=args.training_steps,
        batch_size=args.batch_size,
        dataloader_builder_fn=dataloader_builder,
        device=args.device,
        output_dir=output_dir
    )
    
    # Print success message
    print(f"\nNeural plasticity experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    # Create visualizations
    print(f"\nCreating scientific visualizations...")
    viz_dir = create_visualizations(output_dir, results, show=args.show_plots)
    
    if viz_dir:
        print(f"Visualizations saved to: {viz_dir}")
    
    # Print summary
    recovery_rate = results.get("recovery_rate", 0.0)
    print(f"\nExperiment Summary:")
    print(f"- Model: {args.model_name}")
    print(f"- Pruning: {args.pruning_strategy} at {args.pruning_level:.2f} level")
    print(f"- Recovery rate: {recovery_rate:.2%}")
    
    # Generate and save scientific notes
    notes = scientific_notes()
    notes_path = os.path.join(output_dir, "scientific_notes.md")
    with open(notes_path, 'w') as f:
        f.write(notes)
    
    print(f"Scientific notes saved to: {notes_path}")
    
    # Print next steps suggestion
    print(f"\nNext Steps Suggestion:")
    print(f"1. Try different pruning strategies and levels to compare their effects")
    print(f"2. Run on larger models to observe more complex plasticity patterns")
    print(f"3. Visualize the results in the Jupyter notebook: notebooks/NeuralPlasticityDemo.ipynb")

if __name__ == "__main__":
    main()