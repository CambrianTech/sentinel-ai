#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Dashboard Demo

This script demonstrates the neural plasticity dashboard capabilities
including warmup, analysis, pruning, and inference phase visualizations.

Version: v0.0.1 (2025-04-20 19:00:00)
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard_demo")

# Import dashboard integration
try:
    from utils.neural_plasticity.dashboard.wandb_integration import WandbDashboard
except ImportError:
    # Add parent directory to path to find the module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    sys.path.insert(0, parent_dir)
    
    from utils.neural_plasticity.dashboard.wandb_integration import WandbDashboard

def generate_mock_data(num_layers=12, num_heads=12):
    """Generate mock data for demonstration."""
    # Generate mock attention maps
    mock_attention = np.random.random((num_layers, num_heads, 20, 20))
    
    # Generate mock entropy values
    entropy_values = np.random.random((num_layers, num_heads)) * 5
    
    # Generate mock gradient values
    gradient_values = np.random.random((num_layers, num_heads)) * 10
    
    # Generate mock pruning decisions
    pruned_heads = []
    for _ in range(int(num_layers * num_heads * 0.2)):  # Prune ~20% of heads
        layer = np.random.randint(0, num_layers)
        head = np.random.randint(0, num_heads)
        if (layer, head) not in pruned_heads:
            pruned_heads.append((layer, head))
    
    # Sort pruned heads for nicer display
    pruned_heads.sort()
    
    return {
        "attention": mock_attention,
        "entropy": entropy_values,
        "gradients": gradient_values,
        "pruned_heads": pruned_heads
    }

def run_dashboard_demo(use_online=False, open_browser=True):
    """Run the neural plasticity dashboard demo."""
    logger.info("Starting neural plasticity dashboard demo")
    
    # Create timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"dashboard_demo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the dashboard
    mode = "online" if use_online else "offline"
    dashboard = WandbDashboard(
        project_name="neural-plasticity-demo",
        experiment_name=f"dashboard-demo-{timestamp}",
        output_dir=output_dir,
        config={
            "model_name": "gpt2",
            "dataset": "wikitext/wikitext-2",
            "pruning_strategy": "entropy",
            "pruning_level": 0.2,
            "demo_mode": True
        },
        mode=mode,
        tags=["demo", "neural-plasticity"]
    )
    
    # Generate mock data
    num_layers = 12
    num_heads = 12
    mock_data = generate_mock_data(num_layers, num_heads)
    
    try:
        # Start with setup phase
        dashboard.set_phase("setup")
        dashboard.log_metrics({
            "status": "running", 
            "message": "Setting up experiment environment"
        })
        
        logger.info("Setting up experiment environment")
        time.sleep(1)  # Simulate setup time
        
        # Generate model architecture visualization
        logger.info("Loading model and creating visualizations")
        dashboard.log_metrics({
            "model_layers": num_layers,
            "model_heads": num_heads,
            "model_size": "124M",
            "message": "Model loaded successfully"
        })
        
        # ===== WARMUP PHASE =====
        dashboard.set_phase("warmup")
        dashboard.log_metrics({
            "status": "running", 
            "message": "Starting warmup phase"
        })
        
        logger.info("Running warmup phase")
        
        # Generate warmup loss curve
        time_steps = np.arange(50)
        
        # Create a declining loss curve with some noise
        base_loss = 5.0 * np.exp(-0.05 * time_steps) + 2.5
        noise = np.random.normal(0, 0.1, size=len(time_steps))
        losses = base_loss + noise
        
        # Log warmup metrics
        for step, loss in enumerate(losses):
            # Add step delay for visualization
            if step % 5 == 0:
                time.sleep(0.2)
                
            perplexity = np.exp(loss)
            
            # Log to dashboard
            dashboard.log_metrics({
                "step": step,
                "train_loss": loss,
                "perplexity": perplexity,
                "smoothed_loss": base_loss[step]
            }, step=step)
            
            # Every 10 steps, generate a text sample
            if step % 10 == 0:
                prompt = "The future of artificial intelligence"
                generated_text = "The future of artificial intelligence is looking increasingly bright, with new breakthroughs in natural language processing, computer vision, and reinforcement learning. Researchers are making significant progress in developing models that can understand and generate human language, recognize objects and scenes in images, and learn complex tasks through trial and error."
                
                dashboard.log_text_sample(prompt, generated_text, step=step)
        
        # Complete warmup phase
        final_loss = losses[-1]
        final_perplexity = np.exp(final_loss)
        
        dashboard.log_metrics({
            "status": "completed", 
            "message": "Warmup phase completed",
            "baseline_loss": final_loss,
            "baseline_perplexity": final_perplexity
        })
        
        # ===== ANALYSIS PHASE =====
        dashboard.set_phase("analysis")
        dashboard.log_metrics({
            "status": "running", 
            "message": "Analyzing attention patterns"
        })
        
        logger.info("Running analysis phase")
        
        # Log entropy heatmap
        dashboard.log_entropy_heatmap(mock_data["entropy"])
        
        # Log gradient heatmap
        dashboard.log_gradient_heatmap(mock_data["gradients"])
        
        # Log combined metrics
        dashboard.log_metrics({
            "status": "completed", 
            "message": "Analysis phase completed",
            "avg_entropy": np.mean(mock_data["entropy"]),
            "max_entropy": np.max(mock_data["entropy"]),
            "avg_gradient": np.mean(mock_data["gradients"]),
            "max_gradient": np.max(mock_data["gradients"])
        })
        
        # ===== PRUNING PHASE =====
        dashboard.set_phase("pruning")
        dashboard.log_metrics({
            "status": "running", 
            "message": "Applying pruning strategy"
        })
        
        logger.info("Running pruning phase")
        
        # Log pruning decision
        pruning_decision = {
            "strategy": "entropy",
            "pruning_level": 0.2,
            "pruned_heads": mock_data["pruned_heads"],
            "cycle": 1
        }
        dashboard.log_pruning_decision(pruning_decision)
        
        # Simulate training with pruned model
        pruned_time_steps = np.arange(30)
        pruned_base_loss = 3.0 * np.exp(-0.03 * pruned_time_steps) + 2.0
        pruned_noise = np.random.normal(0, 0.1, size=len(pruned_time_steps))
        pruned_losses = pruned_base_loss + pruned_noise
        
        # Log pruned model training metrics
        for step, loss in enumerate(pruned_losses):
            # Add step delay for visualization
            if step % 5 == 0:
                time.sleep(0.2)
                
            perplexity = np.exp(loss)
            sparsity = len(mock_data["pruned_heads"]) / (num_layers * num_heads)
            
            # Log to dashboard
            dashboard.log_metrics({
                "step": step + 50,  # Continue from warmup
                "train_loss": loss,
                "perplexity": perplexity,
                "smoothed_loss": pruned_base_loss[step],
                "sparsity": sparsity
            }, step=step + 50)
        
        # Complete pruning phase
        pruned_final_loss = pruned_losses[-1]
        pruned_final_perplexity = np.exp(pruned_final_loss)
        
        dashboard.log_metrics({
            "status": "completed", 
            "message": "Pruning phase completed",
            "pruned_loss": pruned_final_loss,
            "pruned_perplexity": pruned_final_perplexity,
            "sparsity": len(mock_data["pruned_heads"]) / (num_layers * num_heads)
        })
        
        # ===== INFERENCE PHASE =====
        dashboard.set_phase("evaluation")
        dashboard.log_metrics({
            "status": "running", 
            "message": "Running inference evaluation"
        })
        
        logger.info("Running inference evaluation phase")
        
        # Create inference samples
        prompts = [
            "The future of artificial intelligence is",
            "Neural networks have revolutionized",
            "The key challenge in machine learning today is"
        ]
        
        baseline_samples = {
            "The future of artificial intelligence is": "The future of artificial intelligence is bright. Researchers are making significant progress in developing models that can understand and generate human language, recognize objects and scenes in images, and learn complex tasks through trial and error. As these capabilities continue to improve, AI systems will become increasingly integrated into our daily lives, transforming industries from healthcare to transportation.",
            "Neural networks have revolutionized": "Neural networks have revolutionized many fields of artificial intelligence, enabling breakthroughs in computer vision, natural language processing, and reinforcement learning. By mimicking the structure and function of the human brain, these powerful mathematical models can learn complex patterns from data, making them ideal for tasks that were previously thought to require human intelligence.",
            "The key challenge in machine learning today is": "The key challenge in machine learning today is creating systems that can generalize well to new, unseen scenarios beyond their training data. While current models excel at pattern recognition within their training distribution, they often struggle when faced with novel situations or adversarial examples. This limitation highlights the gap between artificial and human intelligence, which adapts robustly to new environments."
        }
        
        pruned_samples = {
            "The future of artificial intelligence is": "The future of artificial intelligence is increasingly intertwined with human society. As models become more capable, we're seeing applications ranging from healthcare diagnostics to creative content generation. The most promising developments combine neural approaches with symbolic reasoning, creating systems that can both recognize patterns in data and apply logical reasoning to new situations.",
            "Neural networks have revolutionized": "Neural networks have revolutionized computing by providing a flexible framework for solving previously intractable problems. From image recognition to language translation, these brain-inspired computational systems learn directly from data rather than following explicit programming. This paradigm shift has enabled computers to master tasks that were once the exclusive domain of human cognition.",
            "The key challenge in machine learning today is": "The key challenge in machine learning today is creating trustworthy and transparent systems. While models continue to achieve impressive results, their black-box nature raises concerns about reliability, fairness, and potential biases. Researchers are actively developing techniques for explainable AI and robust evaluation methods to ensure these powerful tools benefit humanity while minimizing potential harms."
        }
        
        # Generate text comparisons
        for i, prompt in enumerate(prompts):
            baseline_text = baseline_samples[prompt]
            pruned_text = pruned_samples[prompt]
            
            # Log individual samples
            dashboard.log_text_sample(prompt, baseline_text, model_type="baseline", step=80+i)
            dashboard.log_text_sample(prompt, pruned_text, model_type="pruned", step=80+i)
            
            # Create comparison visualization
            dashboard.log_inference_comparison(
                prompt, baseline_text, pruned_text,
                metrics={
                    "baseline_perplexity": final_perplexity,
                    "pruned_perplexity": pruned_final_perplexity,
                    "improvement_percent": ((final_perplexity - pruned_final_perplexity) / final_perplexity) * 100
                },
                step=80+i
            )
            
            # Only need one detailed visualization for the demo
            if i == 0:
                # Generate mock attention maps
                seq_len = 20
                baseline_attention = np.random.random((seq_len, seq_len))
                pruned_attention = np.random.random((seq_len, seq_len))
                
                # Add structure to make it look more realistic
                for i in range(seq_len):
                    baseline_attention[i, i] = 0.8  # Self-attention
                    pruned_attention[i, i] = 0.9    # Stronger self-attention in pruned model
                
                # Create attention visualization
                dashboard.log_attention_visualization(
                    baseline_attention,
                    pruned_attention,
                    prompt,
                    layer_idx=5,
                    head_idx=3,
                    step=90
                )
        
        # Create comprehensive inference dashboard
        perplexity_data = {
            "baseline": final_perplexity,
            "pruned": pruned_final_perplexity,
            "model_size_reduction": 20.0,  # 20% of heads pruned
            "inference_speedup": 1.15      # 15% speedup
        }
        
        generation_samples = {
            "baseline": {
                "prompt": prompts[0],
                "output": baseline_samples[prompts[0]]
            },
            "pruned": {
                "prompt": prompts[0],
                "output": pruned_samples[prompts[0]]
            }
        }
        
        # Create mock attention data
        attention_data = {
            "layer5_head3": {
                "baseline": baseline_attention,
                "pruned": pruned_attention,
                "text": prompts[0],
                "layer": 5,
                "head": 3
            }
        }
        
        # Log comprehensive inference dashboard
        dashboard.log_inference_dashboard(
            perplexity_data=perplexity_data,
            generation_samples=generation_samples,
            attention_data=attention_data,
            step=100
        )
        
        # Log perplexity comparison with sample texts
        dashboard.log_perplexity_comparison(
            baseline_perplexity=final_perplexity,
            pruned_perplexity=pruned_final_perplexity,
            texts=[prompts[0], prompts[1]],
            step=101
        )
        
        # Complete inference phase
        dashboard.log_metrics({
            "status": "completed", 
            "message": "Inference evaluation completed",
            "final_perplexity": pruned_final_perplexity,
            "improvement_percent": ((final_perplexity - pruned_final_perplexity) / final_perplexity) * 100,
            "model_size_reduction": 20.0,
            "inference_speedup": 1.15
        })
        
        # ===== COMPLETE EXPERIMENT =====
        dashboard.set_phase("complete")
        dashboard.log_metrics({
            "status": "completed", 
            "message": "Experiment completed successfully"
        })
        
        logger.info("Dashboard demo completed successfully")
        
        # Handle browser opening differently based on online/offline mode
        import webbrowser
        
        if use_online and open_browser and dashboard.run and hasattr(dashboard.run, 'url'):
            # For online mode, open the wandb.ai URL
            dashboard_url = dashboard.run.url
            logger.info(f"Opening online dashboard at: {dashboard_url}")
            webbrowser.open(dashboard_url)
        elif not use_online and open_browser:
            # For offline mode, start a local wandb server and open it
            try:
                # Export a standalone HTML file as a backup
                html_path = os.path.join(output_dir, "dashboard.html")
                logger.info(f"Exporting standalone HTML dashboard to: {html_path}")
                
                # Save warmup loss curve to a file
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(losses, label="Loss")
                plt.plot(base_loss, label="Smoothed Loss")
                plt.title("Warmup Training Loss")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, alpha=0.3)
                warmup_img_path = os.path.join(output_dir, "warmup_loss.png")
                plt.savefig(warmup_img_path)
                plt.close()
                
                # Create entropy heatmap
                plt.figure(figsize=(10, 6))
                plt.imshow(mock_data["entropy"], cmap="viridis")
                plt.colorbar(label="Entropy")
                plt.title("Attention Head Entropy")
                plt.xlabel("Head Index")
                plt.ylabel("Layer Index")
                entropy_img_path = os.path.join(output_dir, "entropy_heatmap.png")
                plt.savefig(entropy_img_path)
                plt.close()
                
                # Create pruning visualization
                # Before pruning - all heads active
                before_mask = np.ones((num_layers, num_heads))
                
                # Create after pruning mask - pruned heads shown as 0
                after_mask = np.ones((num_layers, num_heads))
                for layer, head in mock_data["pruned_heads"]:
                    after_mask[layer, head] = 0
                    
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                ax1.imshow(before_mask, cmap="Greens")
                ax1.set_title('Model Architecture Before Pruning')
                ax1.set_xlabel("Head Index")
                ax1.set_ylabel("Layer Index")
                
                # Custom colormap: red for pruned, green for active
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(['red', 'green'])
                ax2.imshow(after_mask, cmap=cmap)
                ax2.set_title(f'Model Architecture After Pruning ({len(mock_data["pruned_heads"])} heads pruned)')
                ax2.set_xlabel("Head Index")
                ax2.set_ylabel("Layer Index")
                
                plt.tight_layout()
                pruning_img_path = os.path.join(output_dir, "pruning_visualization.png")
                plt.savefig(pruning_img_path)
                plt.close()
                
                # Create perplexity comparison
                plt.figure(figsize=(10, 6))
                models = ['Baseline Model', 'Pruned Model']
                values = [final_perplexity, pruned_final_perplexity]
                bars = plt.bar(models, values, color=['#3498db', '#9b59b6'])
                plt.title('Perplexity Comparison')
                plt.ylabel('Perplexity (lower is better)')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                
                perplexity_img_path = os.path.join(output_dir, "perplexity_comparison.png")
                plt.savefig(perplexity_img_path)
                plt.close()
                
                # Create standalone HTML dashboard with embedded images
                with open(html_path, 'w') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Neural Plasticity Dashboard</title>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                            h1, h2 {{ color: #3f51b5; }}
                            .container {{ max-width: 1200px; margin: 0 auto; }}
                            .dashboard-section {{ 
                                background-color: white; 
                                border-radius: 8px; 
                                padding: 20px; 
                                margin: 20px 0; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }}
                            .metrics {{ 
                                display: flex; 
                                flex-wrap: wrap; 
                                gap: 15px; 
                                margin: 15px 0;
                            }}
                            .metric-card {{
                                flex: 1;
                                min-width: 200px;
                                background-color: white;
                                padding: 15px;
                                border-radius: 5px;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                                border-left: 5px solid #3f51b5;
                            }}
                            .metric-title {{
                                font-size: 14px;
                                color: #7f8c8d;
                                margin-bottom: 5px;
                            }}
                            .metric-value {{
                                font-size: 24px;
                                font-weight: bold;
                                color: #2c3e50;
                            }}
                            .text-sample {{
                                background-color: #f9f9f9;
                                border-left: 5px solid #3f51b5;
                                padding: 15px;
                                margin: 15px 0;
                                border-radius: 5px;
                            }}
                            .prompt {{
                                font-weight: bold;
                                margin-bottom: 10px;
                            }}
                            .generated-text {{
                                font-family: monospace;
                                padding: 10px;
                                background-color: #f1f1f1;
                                border-radius: 5px;
                                white-space: pre-wrap;
                            }}
                            .text-comparison {{
                                display: flex;
                                flex-wrap: wrap;
                                gap: 20px;
                                margin: 15px 0;
                            }}
                            .text-column {{
                                flex: 1;
                                min-width: 300px;
                            }}
                            .baseline {{
                                border-left: 5px solid #3498db;
                            }}
                            .pruned {{
                                border-left: 5px solid #9b59b6;
                            }}
                            img {{
                                max-width: 100%;
                                height: auto;
                                border-radius: 5px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                margin: 10px 0;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Neural Plasticity Dashboard</h1>
                            <p>Experiment generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            
                            <!-- Summary Metrics -->
                            <div class="dashboard-section">
                                <h2>Experiment Summary</h2>
                                <div class="metrics">
                                    <div class="metric-card">
                                        <div class="metric-title">Baseline Perplexity</div>
                                        <div class="metric-value">{final_perplexity:.2f}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-title">Pruned Perplexity</div>
                                        <div class="metric-value">{pruned_final_perplexity:.2f}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-title">Heads Pruned</div>
                                        <div class="metric-value">{len(mock_data["pruned_heads"])}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-title">Model Size Reduction</div>
                                        <div class="metric-value">20.0%</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-title">Inference Speedup</div>
                                        <div class="metric-value">1.15x</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Warmup Phase -->
                            <div class="dashboard-section">
                                <h2>Warmup Phase</h2>
                                <p>Training loss curve during model warmup before pruning:</p>
                                <img src="warmup_loss.png" alt="Warmup Loss Curve">
                            </div>
                            
                            <!-- Analysis Phase -->
                            <div class="dashboard-section">
                                <h2>Analysis Phase</h2>
                                <p>Entropy heatmap showing importance of each attention head:</p>
                                <img src="entropy_heatmap.png" alt="Entropy Heatmap">
                            </div>
                            
                            <!-- Pruning Phase -->
                            <div class="dashboard-section">
                                <h2>Pruning Phase</h2>
                                <p>Visualization of pruned attention heads:</p>
                                <img src="pruning_visualization.png" alt="Pruning Visualization">
                                <p>Pruned {len(mock_data["pruned_heads"])} heads out of {num_layers * num_heads} total heads.</p>
                            </div>
                            
                            <!-- Evaluation Phase -->
                            <div class="dashboard-section">
                                <h2>Evaluation Phase</h2>
                                <p>Performance comparison between baseline and pruned models:</p>
                                <img src="perplexity_comparison.png" alt="Perplexity Comparison">
                                
                                <h3>Text Generation Examples</h3>
                                <div class="text-comparison">
                                    <div class="text-column">
                                        <div class="text-sample baseline">
                                            <h4>Baseline Model</h4>
                                            <div class="prompt">Prompt: "The future of artificial intelligence is"</div>
                                            <div class="generated-text">The future of artificial intelligence is bright. Researchers are making significant progress in developing models that can understand and generate human language, recognize objects and scenes in images, and learn complex tasks through trial and error.</div>
                                        </div>
                                    </div>
                                    <div class="text-column">
                                        <div class="text-sample pruned">
                                            <h4>Pruned Model</h4>
                                            <div class="prompt">Prompt: "The future of artificial intelligence is"</div>
                                            <div class="generated-text">The future of artificial intelligence is increasingly intertwined with human society. As models become more capable, we're seeing applications ranging from healthcare diagnostics to creative content generation. The most promising developments combine neural approaches with symbolic reasoning.</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="text-comparison">
                                    <div class="text-column">
                                        <div class="text-sample baseline">
                                            <h4>Baseline Model</h4>
                                            <div class="prompt">Prompt: "Neural networks have revolutionized"</div>
                                            <div class="generated-text">Neural networks have revolutionized many fields of artificial intelligence, enabling breakthroughs in computer vision, natural language processing, and reinforcement learning. By mimicking the structure and function of the human brain, these powerful mathematical models can learn complex patterns from data.</div>
                                        </div>
                                    </div>
                                    <div class="text-column">
                                        <div class="text-sample pruned">
                                            <h4>Pruned Model</h4>
                                            <div class="prompt">Prompt: "Neural networks have revolutionized"</div>
                                            <div class="generated-text">Neural networks have revolutionized computing by providing a flexible framework for solving previously intractable problems. From image recognition to language translation, these brain-inspired computational systems learn directly from data rather than following explicit programming.</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="dashboard-section">
                                <h2>Run Full Experiment</h2>
                                <p>You can run a full neural plasticity experiment with real models using:</p>
                                <pre>python scripts/run_neural_plasticity.py --model_name distilgpt2 --pruning_strategy entropy --pruning_level 0.2 --use_dashboard</pre>
                            </div>
                        </div>
                    </body>
                    </html>
                    """)
                
                # Open the HTML file directly
                local_url = f"file://{os.path.abspath(html_path)}"
                logger.info(f"Opening local HTML dashboard at: {local_url}")
                webbrowser.open(local_url)
                
                # Also provide instructions for the interactive wandb UI
                logger.info("\n" + "=" * 70)
                logger.info("To view the full interactive dashboard locally:")
                logger.info(f"1. Run: wandb server --port=8080 --directory={output_dir}")
                logger.info("2. Open: http://localhost:8080")
                logger.info("=" * 70 + "\n")
                
            except Exception as e:
                logger.error(f"Failed to create local dashboard: {e}")
                logger.info(f"Dashboard data is available in: {output_dir}")
                
                # Try to open the output directory at least
                try:
                    webbrowser.open(f"file://{os.path.abspath(output_dir)}")
                except:
                    pass
    
    finally:
        # Ensure dashboard is properly closed
        dashboard.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural plasticity dashboard demo")
    parser.add_argument("--online", action="store_true", help="Use online mode for wandb")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser window")
    
    args = parser.parse_args()
    
    run_dashboard_demo(
        use_online=args.online,
        open_browser=not args.no_browser
    )