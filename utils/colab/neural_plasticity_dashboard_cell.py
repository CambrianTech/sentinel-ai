"""
Neural Plasticity Dashboard Cell for Colab Notebooks

This script can be copied into a Colab notebook cell to generate a comprehensive 
visualization of the neural plasticity process.

Usage:
1. Copy this entire file into a Colab notebook cell
2. Run the cell to generate visualizations
3. Call display_neural_plasticity_dashboard(experiment) with your experiment object

Author: Claude <noreply@anthropic.com>
Version: v0.0.1 (2025-04-20)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import sys

def display_neural_plasticity_dashboard(
    experiment,
    output_dir: Optional[str] = None,
    show_quote: bool = True,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Display a comprehensive dashboard of the complete neural plasticity process.
    
    This function can be used in any Colab notebook to visualize the results
    of a neural plasticity experiment, including warmup, stabilization,
    pruning, and fine-tuning phases.
    
    Args:
        experiment: NeuralPlasticityExperiment object or dictionary with experiment results
        output_dir: Optional directory to save the visualization
        show_quote: Whether to display an inspirational quote
        figsize: Figure size (width, height) in inches
        
    Returns:
        None - displays the visualization in the notebook
    """
    # Try importing from installed modules first
    try:
        from utils.colab.visualizations import visualize_complete_training_process
    except ImportError:
        # If not available, check if we need to add project root to path
        try:
            visualize_complete_training_process
        except NameError:
            # Define helper functions locally if not available from imports
            
            def extract_complete_training_data(experiment):
                """Extract comprehensive training data from the experiment object."""
                data = {
                    "warmup_losses": [],
                    "warmup_steps": [],
                    "stabilization_point": None,
                    "pruning_losses": [],
                    "pruning_steps": [],
                    "fine_tuning_losses": [],
                    "fine_tuning_steps": [],
                    "perplexity_scores": [],
                    "sparsity_history": [],
                    "pruned_head_counts": [],
                    "phase_markers": {}
                }
                
                # Extract from NeuralPlasticityExperiment object
                if hasattr(experiment, 'results') and isinstance(experiment.results, dict):
                    results = experiment.results
                elif isinstance(experiment, dict):
                    results = experiment
                else:
                    # Try to access experiment data through other common attributes
                    results = {}
                    if hasattr(experiment, 'warmup_results'):
                        results['warmup'] = experiment.warmup_results
                    if hasattr(experiment, 'pruning_results'):
                        results['pruning'] = experiment.pruning_results
                    if hasattr(experiment, 'fine_tuning_results'):
                        results['fine_tuning'] = experiment.fine_tuning_results
                
                # Extract warmup phase data
                warmup_data = results.get('warmup', {})
                if isinstance(warmup_data, dict):
                    data['warmup_losses'] = warmup_data.get('losses', [])
                    data['warmup_steps'] = list(range(len(data['warmup_losses'])))
                    data['stabilization_point'] = warmup_data.get('stabilization_point')
                    
                    # If stabilization_point is not directly available, calculate from other fields
                    if data['stabilization_point'] is None and warmup_data.get('is_stable'):
                        steps_without_decrease = warmup_data.get('steps_without_decrease', 0)
                        if steps_without_decrease > 0:
                            data['stabilization_point'] = len(data['warmup_losses']) - steps_without_decrease
                
                # Extract pruning phase data
                pruning_data = results.get('pruning', {})
                if isinstance(pruning_data, dict):
                    # Extract training metrics during pruning phase
                    metrics = pruning_data.get('training_metrics', {})
                    if metrics:
                        data['pruning_losses'] = metrics.get('train_loss', [])
                        data['pruning_steps'] = metrics.get('step', list(range(len(data['pruning_losses']))))
                        pruning_perplexity = metrics.get('perplexity', [])
                        data['perplexity_scores'].extend(pruning_perplexity)
                        data['sparsity_history'] = metrics.get('sparsity', [])
                        data['pruned_head_counts'] = metrics.get('pruned_heads', [])
                
                # Extract fine-tuning phase data
                fine_tuning_data = results.get('fine_tuning', {})
                if isinstance(fine_tuning_data, dict):
                    # Extract training metrics during fine-tuning phase
                    metrics = fine_tuning_data.get('training_metrics', {})
                    if metrics:
                        data['fine_tuning_losses'] = metrics.get('train_loss', [])
                        data['fine_tuning_steps'] = metrics.get('step', list(range(len(data['fine_tuning_losses']))))
                        ft_perplexity = metrics.get('perplexity', [])
                        data['perplexity_scores'].extend(ft_perplexity)
                
                # Calculate phase markers for visualization
                # Start of warmup is always 0
                data['phase_markers']['warmup_start'] = 0
                
                # End of warmup / start of pruning
                warmup_length = len(data['warmup_losses'])
                data['phase_markers']['warmup_end'] = warmup_length
                data['phase_markers']['pruning_start'] = warmup_length
                
                # End of pruning / start of fine-tuning
                pruning_length = len(data['pruning_losses'])
                data['phase_markers']['pruning_end'] = warmup_length + pruning_length
                data['phase_markers']['fine_tuning_start'] = warmup_length + pruning_length
                
                # End of fine-tuning
                fine_tuning_length = len(data['fine_tuning_losses'])
                data['phase_markers']['fine_tuning_end'] = warmup_length + pruning_length + fine_tuning_length
                
                return data
            
            def generate_inspirational_quote():
                """Generate an inspirational quote about neural networks and plasticity."""
                quotes = [
                    "The brain's plasticity is its most fascinating feature - so is your model's.",
                    "Adapt, prune, refine, learn. That is the path to AI excellence.",
                    "In neural network pruning, less is often more.",
                    "The most efficient networks are not built, they are grown and pruned like gardens.",
                    "A carefully pruned network is like a well-written sentence - nothing extra, nothing missing.",
                    "Plasticity is not an accident of nature, but its core design principle.",
                    "The art of AI is knowing which connections to strengthen and which to prune.",
                    "In the neural world, connections that fire together wire together, and those that don't, don't.",
                    "The wise gardener prunes to promote growth. The wise AI researcher does the same.",
                    "Every neural connection serves a purpose or makes way for one that does.",
                    "Neural plasticity teaches us that adaptation is not just a response to the environment - it is survival.",
                    "Complexity should be pruned, not pursued."
                ]
                
                # Choose random quote
                import random
                return random.choice(quotes)
            
            def visualize_complete_training_process(
                experiment,
                output_dir: Optional[str] = None,
                filename: Optional[str] = None,
                title: str = "Complete Neural Plasticity Training Process",
                show_plot: bool = True,
                figsize: Tuple[int, int] = (14, 10),
                show_quote: bool = True
            ) -> plt.Figure:
                """Create a comprehensive visualization of the entire neural plasticity process."""
                # Extract all training data from experiment
                training_data = extract_complete_training_data(experiment)
                
                # Concatenate all losses and steps for comprehensive visualization
                all_losses = []
                all_steps = []
                
                # Add warmup phase data
                warmup_losses = training_data['warmup_losses']
                warmup_steps = training_data['warmup_steps']
                all_losses.extend(warmup_losses)
                all_steps.extend(warmup_steps)
                
                # Add pruning phase data
                pruning_losses = training_data['pruning_losses']
                pruning_steps = [step + len(warmup_steps) for step in training_data['pruning_steps']]
                all_losses.extend(pruning_losses)
                all_steps.extend(pruning_steps)
                
                # Add fine-tuning phase data
                fine_tuning_losses = training_data['fine_tuning_losses']
                fine_tuning_steps = [step + len(warmup_steps) + len(pruning_steps) for step in training_data['fine_tuning_steps']]
                all_losses.extend(fine_tuning_losses)
                all_steps.extend(fine_tuning_steps)
                
                # Create a multi-panel figure
                fig = plt.figure(figsize=figsize)
                gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[3, 2, 2, 1])
                
                # 1. Main Plot: Complete Training Process
                ax_main = fig.add_subplot(gs[0, :])
                
                # Plot all losses
                if all_losses:
                    ax_main.plot(all_steps, all_losses, 'b-', label='Training Loss')
                
                # Add phase markers
                markers = training_data['phase_markers']
                
                # Warmup phase highlighting
                if 'warmup_start' in markers and 'warmup_end' in markers:
                    ax_main.axvspan(markers['warmup_start'], markers['warmup_end'], 
                                   alpha=0.2, color='blue', label='Warmup Phase')
                
                # Pruning phase highlighting
                if 'pruning_start' in markers and 'pruning_end' in markers:
                    ax_main.axvspan(markers['pruning_start'], markers['pruning_end'], 
                                   alpha=0.2, color='red', label='Pruning Phase')
                
                # Fine-tuning phase highlighting
                if 'fine_tuning_start' in markers and 'fine_tuning_end' in markers:
                    ax_main.axvspan(markers['fine_tuning_start'], markers['fine_tuning_end'], 
                                   alpha=0.2, color='green', label='Fine-tuning Phase')
                
                # Add stabilization point if available
                stabilization_point = training_data['stabilization_point']
                if stabilization_point is not None and stabilization_point < len(warmup_losses):
                    ax_main.axvline(x=stabilization_point, color='green', linestyle='--', 
                                   label='Stabilization Point')
                    # Add marker at the stabilization point
                    stab_y = warmup_losses[stabilization_point] if stabilization_point < len(warmup_losses) else None
                    if stab_y is not None:
                        ax_main.plot(stabilization_point, stab_y, 'go', markersize=8)
                
                # Configure main plot
                ax_main.set_title(title, fontsize=16)
                ax_main.set_xlabel('Steps')
                ax_main.set_ylabel('Loss')
                ax_main.grid(True, alpha=0.3)
                ax_main.legend(loc='upper right')
                
                # 2. Perplexity Plot
                ax_perplexity = fig.add_subplot(gs[1, :2])
                
                # Plot perplexity if available
                perplexity_scores = training_data['perplexity_scores']
                if perplexity_scores:
                    perplexity_steps = list(range(len(perplexity_scores)))
                    ax_perplexity.plot(perplexity_steps, perplexity_scores, 'purple', label='Perplexity')
                    ax_perplexity.set_title('Model Perplexity')
                    ax_perplexity.set_xlabel('Evaluation Steps')
                    ax_perplexity.set_ylabel('Perplexity')
                    ax_perplexity.grid(True, alpha=0.3)
                else:
                    ax_perplexity.text(0.5, 0.5, 'No perplexity data available', 
                                      ha='center', va='center')
                
                # 3. Pruning Statistics
                ax_sparsity = fig.add_subplot(gs[1, 2])
                
                # Plot sparsity/pruning stats if available
                sparsity_history = training_data['sparsity_history']
                pruned_head_counts = training_data['pruned_head_counts']
                
                if sparsity_history:
                    ax_sparsity.plot(sparsity_history, 'r-', label='Sparsity')
                    ax_sparsity.set_title('Model Sparsity')
                    ax_sparsity.set_xlabel('Pruning Steps')
                    ax_sparsity.set_ylabel('Sparsity (%)')
                    ax_sparsity.grid(True, alpha=0.3)
                elif pruned_head_counts:
                    ax_sparsity.plot(pruned_head_counts, 'r-', label='Pruned Heads')
                    ax_sparsity.set_title('Pruned Attention Heads')
                    ax_sparsity.set_xlabel('Pruning Steps')
                    ax_sparsity.set_ylabel('Head Count')
                    ax_sparsity.grid(True, alpha=0.3)
                else:
                    ax_sparsity.text(0.5, 0.5, 'No pruning statistics available', 
                                    ha='center', va='center')
                
                # 4. Warmup Detail Plot
                ax_warmup = fig.add_subplot(gs[2, 0])
                
                # Plot warmup phase in detail
                if warmup_losses:
                    ax_warmup.plot(warmup_steps, warmup_losses, 'b-')
                    ax_warmup.set_title('Warmup Phase Detail')
                    ax_warmup.set_xlabel('Steps')
                    ax_warmup.set_ylabel('Loss')
                    ax_warmup.grid(True, alpha=0.3)
                    
                    # Highlight stabilization point
                    if stabilization_point is not None and stabilization_point < len(warmup_losses):
                        ax_warmup.axvline(x=stabilization_point, color='green', linestyle='--')
                        ax_warmup.plot(stabilization_point, warmup_losses[stabilization_point], 'go', markersize=8)
                else:
                    ax_warmup.text(0.5, 0.5, 'No warmup data available', 
                                  ha='center', va='center')
                
                # 5. Pruning Detail Plot
                ax_pruning = fig.add_subplot(gs[2, 1])
                
                # Plot pruning phase in detail
                if pruning_losses:
                    pruning_relative_steps = list(range(len(pruning_losses)))
                    ax_pruning.plot(pruning_relative_steps, pruning_losses, 'r-')
                    ax_pruning.set_title('Pruning Phase Detail')
                    ax_pruning.set_xlabel('Steps')
                    ax_pruning.set_ylabel('Loss')
                    ax_pruning.grid(True, alpha=0.3)
                else:
                    ax_pruning.text(0.5, 0.5, 'No pruning data available', 
                                   ha='center', va='center')
                
                # 6. Fine-tuning Detail Plot
                ax_finetune = fig.add_subplot(gs[2, 2])
                
                # Plot fine-tuning phase in detail
                if fine_tuning_losses:
                    finetune_relative_steps = list(range(len(fine_tuning_losses)))
                    ax_finetune.plot(finetune_relative_steps, fine_tuning_losses, 'g-')
                    ax_finetune.set_title('Fine-tuning Phase Detail')
                    ax_finetune.set_xlabel('Steps')
                    ax_finetune.set_ylabel('Loss')
                    ax_finetune.grid(True, alpha=0.3)
                else:
                    ax_finetune.text(0.5, 0.5, 'No fine-tuning data available', 
                                    ha='center', va='center')
                
                # 7. Summary Box
                ax_summary = fig.add_subplot(gs[3, :])
                ax_summary.axis('off')
                
                # Create summary text
                summary_parts = []
                
                # Add total steps per phase
                summary_parts.append(f"Warmup: {len(warmup_losses)} steps")
                
                if pruning_losses:
                    summary_parts.append(f"Pruning: {len(pruning_losses)} steps")
                
                if fine_tuning_losses:
                    summary_parts.append(f"Fine-tuning: {len(fine_tuning_losses)} steps")
                
                # Add stabilization info
                if stabilization_point is not None:
                    summary_parts.append(f"Stabilization at step {stabilization_point}")
                
                # Add pruning stats
                if sparsity_history and len(sparsity_history) > 0:
                    summary_parts.append(f"Final sparsity: {sparsity_history[-1]:.1f}%")
                
                if pruned_head_counts and len(pruned_head_counts) > 0:
                    summary_parts.append(f"Total pruned heads: {pruned_head_counts[-1]}")
                
                # Add perplexity improvement if available
                if perplexity_scores and len(perplexity_scores) > 1:
                    initial_perplexity = perplexity_scores[0]
                    final_perplexity = perplexity_scores[-1]
                    improvement = (initial_perplexity - final_perplexity) / initial_perplexity * 100
                    summary_parts.append(f"Perplexity improvement: {improvement:.1f}%")
                
                # Format the summary
                summary_text = " | ".join(summary_parts)
                
                # Add an inspirational quote if enabled
                if show_quote:
                    quote = generate_inspirational_quote()
                    formatted_quote = f'"{quote}"'
                    ax_summary.text(0.5, 0.2, formatted_quote, ha='center', va='center', 
                                   fontsize=11, fontstyle='italic', color='darkblue')
                
                # Add the summary stats
                ax_summary.text(0.5, 0.7, summary_text, ha='center', va='center', 
                               fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                
                # Add timestamp
                from datetime import datetime
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ax_summary.text(0.99, 0.05, f"Generated: {current_time}", ha='right', va='bottom', 
                               fontsize=8, fontweight='light', transform=ax_summary.transAxes)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save visualization if requested
                if output_dir:
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Set default filename if not provided
                    if filename is None:
                        current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        filename = f"neural_plasticity_process_{current_timestamp}.png"
                    
                    # Ensure filename has .png extension
                    if not filename.endswith(".png"):
                        filename += ".png"
                        
                    save_path = os.path.join(output_dir, filename)
                    plt.savefig(save_path, dpi=120, bbox_inches='tight')
                    print(f"✅ Visualization saved to: {save_path}")
                
                return fig
    
    # Create the visualization
    fig = visualize_complete_training_process(
        experiment=experiment,
        output_dir=output_dir,
        title="Complete Neural Plasticity Training Process",
        show_plot=True,
        figsize=figsize,
        show_quote=show_quote
    )
    
    # Display using IPython display if available
    try:
        from IPython.display import display
        # The display function will make the figure visible in the notebook
        display(fig)
    except ImportError:
        # If not in IPython environment, show the plot directly
        plt.show()
    
    return fig

# Example of how to create synthetic data for testing
def create_synthetic_experiment_data():
    """
    Create synthetic data to mimic a neural plasticity experiment.
    This is useful for testing and demonstration purposes.
    """
    import numpy as np
    
    # Generate simulated loss curves with realistic patterns
    # Warmup: initial rapid decrease, then plateau
    warmup_steps = 20
    warmup_losses = [5.0 - 0.5 * np.sqrt(i) + 0.05 * np.random.randn() for i in range(1, warmup_steps + 1)]
    
    # Pruning: slight increase, then decrease
    pruning_steps = 15
    pruning_losses = [warmup_losses[-1] + 0.1 * np.sin(i/5) + 0.05 * np.random.randn() for i in range(1, pruning_steps + 1)]
    
    # Fine-tuning: steady decrease
    fine_tuning_steps = 25
    fine_tuning_losses = [pruning_losses[-1] - 0.03 * i + 0.05 * np.random.randn() for i in range(1, fine_tuning_steps + 1)]
    
    # Generate perplexity scores (inverse relationship with loss, roughly)
    warmup_perplexity = [np.exp(loss) for loss in warmup_losses][-5:]  # Only measure at end of warmup
    pruning_perplexity = [np.exp(loss) for loss in pruning_losses]
    fine_tuning_perplexity = [np.exp(loss) for loss in fine_tuning_losses]
    
    # Generate pruning metrics
    sparsity_values = [0.0]
    for i in range(1, pruning_steps):
        # Gradually increase sparsity to about 40%
        if i < pruning_steps - 3:
            sparsity_values.append(min(40.0, sparsity_values[-1] + np.random.uniform(2.0, 4.0)))
        else:
            sparsity_values.append(sparsity_values[-1])  # Plateau at the end
    
    # Calculate pruned head counts (assuming 144 total heads = 12 layers * 12 heads)
    total_heads = 144
    pruned_head_counts = [int(total_heads * sparsity / 100.0) for sparsity in sparsity_values]
    
    # Create experiment data structure
    experiment_data = {
        'warmup': {
            'losses': warmup_losses,
            'is_stable': True,
            'steps_without_decrease': 3,
            'initial_loss': warmup_losses[0],
            'final_loss': warmup_losses[-1],
            'stabilization_point': warmup_steps - 3,
            'segment_analysis': {
                'segment_size': 5,
                'first_segment_avg': np.mean(warmup_losses[:5]),
                'last_segment_avg': np.mean(warmup_losses[-5:]),
                'improvement': 100 * (1 - np.mean(warmup_losses[-5:]) / np.mean(warmup_losses[:5])),
                'still_improving': False
            }
        },
        'pruning': {
            'training_metrics': {
                'train_loss': pruning_losses,
                'step': list(range(len(pruning_losses))),
                'perplexity': pruning_perplexity,
                'pruned_heads': pruned_head_counts,
                'sparsity': sparsity_values
            }
        },
        'fine_tuning': {
            'training_metrics': {
                'train_loss': fine_tuning_losses,
                'step': list(range(len(fine_tuning_losses))),
                'perplexity': fine_tuning_perplexity
            }
        }
    }
    
    return experiment_data

# Example usage - uncomment to run this cell standalone in a notebook
# experiment = create_synthetic_experiment_data()
# display_neural_plasticity_dashboard(experiment, output_dir="neural_plasticity_output")