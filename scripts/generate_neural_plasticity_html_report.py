#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity HTML Report Generator

This script creates a comprehensive HTML report for the neural plasticity
process, including interactive visualizations of warmup, stabilization,
pruning, and fine-tuning phases.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization utilities
from utils.colab.visualizations import visualize_complete_training_process

def create_simulated_data():
    """
    Create simulated but realistic training data for demonstration purposes.
    This follows the patterns of real neural plasticity training, but can be
    generated quickly without running a full experiment.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create experiment dictionary structure
    experiment = {
        'warmup': {
            'losses': [],
            'stabilization_point': 75,
            'is_stable': True,
            'steps_without_decrease': 25,
            'polynomial_fit': {'coefficients': [0.0001, -0.02, 1.0, 4.5]}
        },
        'pruning': {
            'training_metrics': {
                'train_loss': [],
                'perplexity': [],
                'sparsity': [],
                'pruned_heads': []
            },
            'pruned_heads': [(i, j) for i in range(6) for j in range(2)],
            'total_heads': 72,
            'entropy_values': np.random.uniform(0.2, 0.8, size=(6, 12)),
            'gradient_values': np.random.uniform(0.1, 0.9, size=(6, 12))
        },
        'fine_tuning': {
            'training_metrics': {
                'train_loss': [],
                'perplexity': []
            }
        },
        'baseline_metrics': {
            'loss': 4.2,
            'perplexity': 66.7
        },
        'final_metrics': {
            'loss': 3.5,
            'perplexity': 33.1
        },
        'improvements': {
            'loss': 16.7,
            'perplexity': 50.4
        }
    }
    
    # Generate warmup phase losses: starting high and gradually stabilizing
    initial_warmup_loss = 6.5
    num_warmup_steps = 150
    
    for i in range(num_warmup_steps):
        # Exponential decay with noise
        decay_rate = 0.97
        noise_factor = 0.4
        loss = initial_warmup_loss * (decay_rate ** (i/10))
        loss += np.random.normal(0, noise_factor * (1 - i/num_warmup_steps))
        loss = max(0.5, loss)  # Keep it positive
        experiment['warmup']['losses'].append(float(loss))
    
    # Generate pruning phase losses: spike after pruning, gradual recovery
    num_pruning_steps = 100
    last_warmup_loss = experiment['warmup']['losses'][-1]
    pruning_spike = last_warmup_loss * 1.8  # Loss spike after pruning
    
    for i in range(num_pruning_steps):
        # Recovery curve with noise
        recovery_rate = 0.98
        noise_factor = 0.2
        loss = pruning_spike * (recovery_rate ** i) 
        loss += np.random.normal(0, noise_factor)
        loss = max(0.5, loss)
        experiment['pruning']['training_metrics']['train_loss'].append(float(loss))
        
        # Generate perplexity scores: related to loss values
        perplexity = np.exp(loss) - np.random.uniform(0, 1)
        experiment['pruning']['training_metrics']['perplexity'].append(float(perplexity))
        
        # Generate sparsity increasing over time
        max_sparsity = 20.0  # 20% sparsity
        step_sparsity = max_sparsity * min(1.0, (i+1) / (num_pruning_steps * 0.7))
        experiment['pruning']['training_metrics']['sparsity'].append(float(step_sparsity))
        
        # Generate pruned head counts increasing over time
        max_pruned = 12  # Total heads to prune
        step_pruned = int(max_pruned * min(1.0, (i+1) / (num_pruning_steps * 0.7)))
        experiment['pruning']['training_metrics']['pruned_heads'].append(step_pruned)
    
    # Generate fine-tuning phase losses: continued improvement
    num_fine_tuning_steps = 200
    last_pruning_loss = experiment['pruning']['training_metrics']['train_loss'][-1]
    
    for i in range(num_fine_tuning_steps):
        # Gradual improvement curve with noise
        improvement_rate = 0.997
        noise_factor = 0.1
        loss = last_pruning_loss * (improvement_rate ** i)
        loss += np.random.normal(0, noise_factor)
        loss = max(0.5, loss)
        experiment['fine_tuning']['training_metrics']['train_loss'].append(float(loss))
        
        # Generate perplexity scores: related to loss values
        perplexity = np.exp(loss) - np.random.uniform(0, 1)
        experiment['fine_tuning']['training_metrics']['perplexity'].append(float(perplexity))
    
    return experiment


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str


def generate_entropy_heatmap(entropy_values, pruned_heads=None):
    """Generate a heatmap visualization of attention head entropy values"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy if tensor
    if isinstance(entropy_values, torch.Tensor):
        entropy_data = entropy_values.detach().cpu().numpy()
    else:
        entropy_data = entropy_values
    
    # Plot heatmap
    im = ax.imshow(entropy_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Entropy')
    
    # Mark pruned heads if provided
    if pruned_heads:
        for layer, head in pruned_heads:
            if layer < entropy_data.shape[0] and head < entropy_data.shape[1]:
                ax.plot(head, layer, 'rx', markersize=8)
    
    ax.set_title('Attention Head Entropy Heatmap')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    fig.tight_layout()
    return fig


def generate_gradient_heatmap(gradient_values, pruned_heads=None):
    """Generate a heatmap visualization of attention head gradient values"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy if tensor
    if isinstance(gradient_values, torch.Tensor):
        grad_data = gradient_values.detach().cpu().numpy()
    else:
        grad_data = gradient_values
    
    # Plot heatmap
    im = ax.imshow(grad_data, cmap='plasma', aspect='auto')
    plt.colorbar(im, ax=ax, label='Gradient Magnitude')
    
    # Mark pruned heads if provided
    if pruned_heads:
        for layer, head in pruned_heads:
            if layer < grad_data.shape[0] and head < grad_data.shape[1]:
                ax.plot(head, layer, 'rx', markersize=8)
    
    ax.set_title('Attention Head Gradient Heatmap')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    fig.tight_layout()
    return fig


def generate_polynomial_fit_visualization(losses, polynomial_coeffs=None):
    """Generate a visualization of loss curve with polynomial fit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original loss curve
    steps = list(range(len(losses)))
    ax.plot(steps, losses, 'b-', label='Training Loss')
    
    # If polynomial coefficients are provided, plot the fit
    if polynomial_coeffs is not None:
        # Generate polynomial function
        def poly_func(x):
            result = 0
            for i, coef in enumerate(polynomial_coeffs[::-1]):
                result += coef * (x ** i)
            return result
        
        # Generate points for fit curve
        fit_x = np.linspace(0, len(losses)-1, 100)
        fit_y = [poly_func(x) for x in fit_x]
        
        # Plot fit curve
        ax.plot(fit_x, fit_y, 'r--', label='Polynomial Fit')
    
    ax.set_title('Warmup Loss with Polynomial Fit')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    return fig


def generate_html_report(experiment_data, output_dir):
    """
    Generate a comprehensive HTML report with interactive visualizations.
    
    Args:
        experiment_data: Dictionary containing experiment results
        output_dir: Directory to save the HTML report
    
    Returns:
        Path to the generated HTML file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations and convert to base64 for embedding
    main_visualization = visualize_complete_training_process(
        experiment=experiment_data,
        title="Neural Plasticity Complete Training Process",
        show_plot=False
    )
    main_vis_base64 = fig_to_base64(main_visualization)
    
    # Generate additional visualizations
    
    # 1. Entropy heatmap
    entropy_values = experiment_data.get('pruning', {}).get('entropy_values', None)
    pruned_heads = experiment_data.get('pruning', {}).get('pruned_heads', [])
    if entropy_values is not None:
        entropy_fig = generate_entropy_heatmap(entropy_values, pruned_heads)
        entropy_base64 = fig_to_base64(entropy_fig)
    else:
        entropy_base64 = None
    
    # 2. Gradient heatmap
    gradient_values = experiment_data.get('pruning', {}).get('gradient_values', None)
    if gradient_values is not None:
        gradient_fig = generate_gradient_heatmap(gradient_values, pruned_heads)
        gradient_base64 = fig_to_base64(gradient_fig)
    else:
        gradient_base64 = None
    
    # 3. Polynomial fit visualization
    warmup_losses = experiment_data.get('warmup', {}).get('losses', [])
    poly_coeffs = experiment_data.get('warmup', {}).get('polynomial_fit', {}).get('coefficients', None)
    if warmup_losses and poly_coeffs:
        poly_fig = generate_polynomial_fit_visualization(warmup_losses, poly_coeffs)
        poly_base64 = fig_to_base64(poly_fig)
    else:
        poly_base64 = None
    
    # Extract key metrics and statistics
    warmup_steps = len(experiment_data.get('warmup', {}).get('losses', []))
    pruning_steps = len(experiment_data.get('pruning', {}).get('training_metrics', {}).get('train_loss', []))
    fine_tuning_steps = len(experiment_data.get('fine_tuning', {}).get('training_metrics', {}).get('train_loss', []))
    stabilization_point = experiment_data.get('warmup', {}).get('stabilization_point', None)
    
    pruned_head_count = len(experiment_data.get('pruning', {}).get('pruned_heads', []))
    total_head_count = experiment_data.get('pruning', {}).get('total_heads', 0)
    sparsity = experiment_data.get('pruning', {}).get('training_metrics', {}).get('sparsity', [])
    final_sparsity = sparsity[-1] if sparsity else 0
    
    baseline_metrics = experiment_data.get('baseline_metrics', {})
    final_metrics = experiment_data.get('final_metrics', {})
    improvements = experiment_data.get('improvements', {})
    
    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neural Plasticity Experiment Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            .header-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .timestamp {{
                font-size: 0.8em;
                color: #7f8c8d;
                text-align: right;
            }}
            .dashboard-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 30px;
            }}
            .metrics-container {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 15px;
                text-align: center;
            }}
            .metric-title {{
                font-weight: bold;
                margin-bottom: 5px;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .metric-value {{
                font-size: 1.4em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-improvement {{
                font-size: 0.9em;
                color: #27ae60;
            }}
            .metric-degradation {{
                font-size: 0.9em;
                color: #e74c3c;
            }}
            .phase-section {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .phase-section h3 {{
                margin-top: 0;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }}
            .visualization-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .visualization-caption {{
                font-style: italic;
                color: #7f8c8d;
                margin-top: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .quote {{
                font-style: italic;
                color: #3498db;
                text-align: center;
                margin: 30px 0;
                font-size: 1.1em;
            }}
        </style>
    </head>
    <body>
        <div class="header-container">
            <h1>Neural Plasticity Experiment Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>
        
        <div class="dashboard-container">
            <h2>Complete Training Process</h2>
            <div class="visualization-container">
                <img src="data:image/png;base64,{main_vis_base64}" alt="Complete Training Process Visualization" style="max-width:100%;">
                <div class="visualization-caption">
                    Comprehensive visualization of the neural plasticity training process showing warmup, pruning, and fine-tuning phases.
                </div>
            </div>
        </div>
        
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-title">WARMUP STEPS</div>
                <div class="metric-value">{warmup_steps}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">PRUNING STEPS</div>
                <div class="metric-value">{pruning_steps}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">FINE-TUNING STEPS</div>
                <div class="metric-value">{fine_tuning_steps}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">STABILIZATION POINT</div>
                <div class="metric-value">{stabilization_point if stabilization_point is not None else 'N/A'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">PRUNED HEADS</div>
                <div class="metric-value">{pruned_head_count} / {total_head_count}</div>
                <div class="metric-improvement">{(pruned_head_count/total_head_count*100):.1f}% of model pruned</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">FINAL SPARSITY</div>
                <div class="metric-value">{final_sparsity:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">INITIAL PERPLEXITY</div>
                <div class="metric-value">{baseline_metrics.get('perplexity', 'N/A'):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">FINAL PERPLEXITY</div>
                <div class="metric-value">{final_metrics.get('perplexity', 'N/A'):.2f}</div>
                <div class="metric-improvement">↓ {improvements.get('perplexity', 0):.1f}% improvement</div>
            </div>
        </div>
        
        <div class="phase-section">
            <h3>Warmup Phase Analysis</h3>
            <p>
                The warmup phase trained the model for {warmup_steps} steps until the loss stabilized.
                Stabilization was detected at step {stabilization_point if stabilization_point is not None else 'N/A'},
                indicating that the model had reached a point where continued training would yield diminishing returns.
            </p>
            
            {f'''
            <div class="visualization-container">
                <img src="data:image/png;base64,{poly_base64}" alt="Warmup Polynomial Fit" style="max-width:100%;">
                <div class="visualization-caption">
                    Polynomial curve fitting was used to identify the point of stabilization in the loss curve.
                </div>
            </div>
            ''' if poly_base64 else ''}
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Initial Loss</td>
                    <td>{experiment_data.get('warmup', {}).get('losses', [0])[0]:.4f}</td>
                </tr>
                <tr>
                    <td>Final Loss</td>
                    <td>{experiment_data.get('warmup', {}).get('losses', [-1])[-1]:.4f}</td>
                </tr>
                <tr>
                    <td>Loss Improvement</td>
                    <td>{(1 - experiment_data.get('warmup', {}).get('losses', [-1])[-1] / experiment_data.get('warmup', {}).get('losses', [1])[0]) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Stabilization Point</td>
                    <td>{stabilization_point if stabilization_point is not None else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Steps Without Improvement</td>
                    <td>{experiment_data.get('warmup', {}).get('steps_without_decrease', 'N/A')}</td>
                </tr>
            </table>
        </div>
        
        <div class="phase-section">
            <h3>Pruning Phase Analysis</h3>
            <p>
                During the pruning phase, {pruned_head_count} attention heads were pruned based on entropy and gradient metrics,
                representing {(pruned_head_count/total_head_count*100):.1f}% of the model's heads. Pruning caused an initial
                performance degradation, which was gradually recovered during continued training.
            </p>
            
            <div class="visualization-container" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;">
                {f'''
                <div style="flex: 1; min-width: 300px;">
                    <img src="data:image/png;base64,{entropy_base64}" alt="Entropy Heatmap" style="max-width:100%;">
                    <div class="visualization-caption">
                        Attention head entropy values. Higher values (brighter colors) indicate more dispersed attention.
                    </div>
                </div>
                ''' if entropy_base64 else ''}
                
                {f'''
                <div style="flex: 1; min-width: 300px;">
                    <img src="data:image/png;base64,{gradient_base64}" alt="Gradient Heatmap" style="max-width:100%;">
                    <div class="visualization-caption">
                        Attention head gradient magnitudes. Lower values (darker colors) indicate less important heads.
                    </div>
                </div>
                ''' if gradient_base64 else ''}
            </div>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Heads</td>
                    <td>{total_head_count}</td>
                </tr>
                <tr>
                    <td>Pruned Heads</td>
                    <td>{pruned_head_count}</td>
                </tr>
                <tr>
                    <td>Sparsity</td>
                    <td>{final_sparsity:.1f}%</td>
                </tr>
                <tr>
                    <td>Pruning Strategy</td>
                    <td>Combined Entropy-Gradient</td>
                </tr>
                <tr>
                    <td>Performance After Pruning</td>
                    <td>
                        Perplexity initially increased by 
                        {((experiment_data.get('pruning', {}).get('training_metrics', {}).get('perplexity', [0])[0] / 
                        baseline_metrics.get('perplexity', 1) - 1) * 100):.1f}%
                    </td>
                </tr>
            </table>
        </div>
        
        <div class="phase-section">
            <h3>Fine-tuning Phase Analysis</h3>
            <p>
                The fine-tuning phase continued training the pruned model for {fine_tuning_steps} steps with a reduced learning rate.
                This phase helped recover and further improve model performance, demonstrating that the pruned architecture
                can match or exceed the performance of the original model.
            </p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Before Pruning</th>
                    <th>After Fine-tuning</th>
                    <th>Improvement</th>
                </tr>
                <tr>
                    <td>Loss</td>
                    <td>{baseline_metrics.get('loss', 'N/A'):.4f}</td>
                    <td>{final_metrics.get('loss', 'N/A'):.4f}</td>
                    <td>{improvements.get('loss', 0):.1f}%</td>
                </tr>
                <tr>
                    <td>Perplexity</td>
                    <td>{baseline_metrics.get('perplexity', 'N/A'):.2f}</td>
                    <td>{final_metrics.get('perplexity', 'N/A'):.2f}</td>
                    <td>{improvements.get('perplexity', 0):.1f}%</td>
                </tr>
            </table>
        </div>
        
        <div class="quote">
            "The wise gardener prunes to promote growth. The wise AI researcher does the same."
        </div>
        
        <div class="footer">
            Neural Plasticity Experiment Report | Generated on {timestamp} | Sentinel-AI v0.0.59
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    html_path = output_path / "neural_plasticity_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated at: {html_path}")
    
    # Save experiment data as JSON for future reference
    json_path = output_path / "experiment_data.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = json.loads(json.dumps(experiment_data, default=lambda o: o.tolist() if isinstance(o, (np.ndarray, np.generic)) else repr(o)))
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)
    
    return html_path


def main():
    """
    Main function to generate the HTML report.
    """
    print("Generating Neural Plasticity HTML Report...")
    
    # Create output directory
    output_dir = Path("neural_plasticity_report")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create simulated but realistic experiment data
    experiment_data = create_simulated_data()
    
    # Generate HTML report
    html_path = generate_html_report(experiment_data, output_dir)
    
    print("\nNeural Plasticity HTML Report generated successfully!")
    print(f"Report file: {html_path}")


if __name__ == "__main__":
    main()