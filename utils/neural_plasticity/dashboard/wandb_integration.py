"""
Neural Plasticity Weights & Biases Integration

This module provides integration with Weights & Biases for neural plasticity experiments,
allowing real-time tracking of metrics, visualizations, and model artifacts both in
local environments and Google Colab.

Version: v0.0.2 (2025-04-20 18:45:00)
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path

# Conditional imports for Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
# Check if we're running in Colab
IS_COLAB = False
try:
    import google.colab
    from IPython.display import display, HTML
    IS_COLAB = True
except ImportError:
    pass

def setup_wandb_in_colab():
    """
    Helper function to set up wandb in Colab environment.
    Shows login instructions and authentication information.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if not IS_COLAB:
        return False
    
    if not WANDB_AVAILABLE:
        print("⚠️ Weights & Biases (wandb) is not installed in this Colab environment.")
        print("Install it with: !pip install wandb")
        return False
    
    try:
        # Check if already logged in
        try:
            wandb.ensure_login()
            logged_in = True
        except Exception:
            logged_in = False
        
        if not logged_in:
            print("🔑 You need to log in to Weights & Biases to use the real-time dashboard.")
            print("Follow these steps:")
            print("1. Go to https://wandb.ai/ and create an account or log in")
            print("2. Get your API key from your account settings page")
            print("3. Use the code below to log in:")
            print("\nimport wandb")
            print("wandb.login()\n")
            
            # Display visual instructions
            display(HTML("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="margin-top: 0;">Weights & Biases Integration</h3>
                <p>For real-time dashboard tracking in this neural plasticity experiment, 
                we recommend using Weights & Biases.</p>
                <p><b>To log in:</b></p>
                <pre>import wandb
wandb.login()</pre>
                <p>This will provide a link and authentication code to use.</p>
            </div>
            """))
            return False
        
        print("✅ Successfully set up Weights & Biases in Colab environment!")
        print("🔗 Your dashboard links will appear when the experiment starts.")
        return True
            
    except Exception as e:
        print(f"⚠️ Error setting up wandb in Colab: {e}")
        return False

logger = logging.getLogger(__name__)

class WandbDashboard:
    """
    Integration with Weights & Biases for neural plasticity experiments.
    
    This class wraps the wandb API to provide a consistent interface for
    tracking metrics, visualizations, and model artifacts throughout
    the neural plasticity experiment process.
    """
    
    def __init__(
        self,
        project_name: str = "neural-plasticity",
        experiment_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_login: bool = True,
        mode: str = "online",
        tags: Optional[List[str]] = None
    ):
        """
        Initialize the wandb dashboard.
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Name of this specific experiment run (auto-generated if None)
            output_dir: Directory for wandb files (used for offline mode)
            config: Configuration parameters for the experiment
            auto_login: Whether to automatically log in to wandb
            mode: wandb mode ("online", "offline", or "disabled")
            tags: Tags to associate with this run
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.config = config or {}
        self.auto_login = auto_login
        self.mode = mode
        self.tags = tags or []
        
        # Add default tags
        if IS_COLAB:
            self.tags.append("colab")
        
        # Add neural plasticity tag
        if "neural-plasticity" not in self.tags:
            self.tags.append("neural-plasticity")
        
        # Initialize wandb state
        self.initialized = False
        self.run = None
        
        # Try to initialize if wandb is available
        if WANDB_AVAILABLE and auto_login:
            self.initialize()
        elif not WANDB_AVAILABLE:
            logger.warning("Weights & Biases (wandb) not available. Install with: pip install wandb")
    
    def initialize(self):
        """Initialize the wandb run."""
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases (wandb) not available. Install with: pip install wandb")
            if IS_COLAB:
                print("⚠️ Weights & Biases (wandb) is not installed in this Colab environment.")
                print("Run the following cell to install wandb:")
                print("!pip install wandb")
            return False
        
        if self.initialized:
            return True
        
        try:
            # If in Colab, make sure user is logged in
            if IS_COLAB:
                try:
                    wandb.ensure_login()
                except Exception:
                    setup_wandb_in_colab()
                    return False
            
            # Initialize wandb
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                dir=self.output_dir,
                mode=self.mode,
                tags=self.tags,
                reinit=True
            )
            
            # Add experiment start timestamp
            wandb.config.update({"experiment_start_time": time.strftime("%Y-%m-%d %H:%M:%S")})
            
            self.initialized = True
            logger.info(f"Initialized wandb dashboard for project: {self.project_name}")
            
            # Special instructions for Colab users
            if IS_COLAB:
                dashboard_url = f"https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}"
                logger.info(f"🔗 To view the wandb dashboard in Colab, visit: {dashboard_url}")
                
                # Display clickable link in Colab
                try:
                    from IPython.display import display, HTML
                    display(HTML(f"""
                    <div style="background-color: #f0f7fb; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #3f51b5;">
                        <h3 style="margin-top: 0;">Neural Plasticity Dashboard</h3>
                        <p>Your experiment dashboard is now live! Click the link below to view real-time metrics:</p>
                        <p><a href="{dashboard_url}" target="_blank" style="background-color: #3f51b5; color: white; padding: 8px 15px; text-decoration: none; border-radius: 5px;">
                            📊 View Dashboard
                        </a></p>
                    </div>
                    """))
                except Exception as e:
                    # Fall back to regular output if display fails
                    print(f"📊 Neural Plasticity Dashboard: {dashboard_url}")
            else:
                dashboard_url = wandb.run.url
                logger.info(f"🔗 View experiment dashboard at: {dashboard_url}")
                
                # Try to open the browser
                try:
                    import webbrowser
                    logger.info("Opening dashboard in browser...")
                    webbrowser.open(dashboard_url)
                except Exception as e:
                    logger.warning(f"Failed to open browser: {e}")
                    
                # If we have colab integration available, show sharing options
                try:
                    from . import colab_integration
                    if hasattr(colab_integration, 'create_shareable_link'):
                        shareable_link = colab_integration.create_shareable_link(dashboard_url)
                        logger.info(f"Shareable dashboard link: {shareable_link}")
                except Exception:
                    pass
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step for the metrics (if None, wandb uses internal step counter)
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            # Clean up non-scalar metrics for wandb
            clean_metrics = {}
            for key, value in metrics.items():
                # Convert numpy arrays or tensors to lists
                if isinstance(value, (np.ndarray, list)) and key not in ["entropy_values", "grad_norm_values"]:
                    # Skip large arrays which should be logged as images instead
                    if isinstance(value, np.ndarray) and value.size > 100:
                        continue
                    
                    # Convert to regular Python types for wandb
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                        
                # Keep scalar values
                if isinstance(value, (int, float, bool, str)) or (isinstance(value, list) and len(value) < 100):
                    clean_metrics[key] = value
            
            # Log to wandb
            wandb.log(clean_metrics, step=step)
            
            # Print key metrics to console for real-time visibility
            console_metrics = []
            if 'train_loss' in clean_metrics:
                console_metrics.append(f"Train Loss: {clean_metrics['train_loss']:.4f}")
            if 'eval_loss' in clean_metrics:
                console_metrics.append(f"Eval Loss: {clean_metrics['eval_loss']:.4f}")
            if 'perplexity' in clean_metrics:
                console_metrics.append(f"Perplexity: {clean_metrics['perplexity']:.2f}")
            if 'sparsity' in clean_metrics:
                console_metrics.append(f"Sparsity: {clean_metrics['sparsity']:.2%}")
            if 'phase' in clean_metrics:
                console_metrics.append(f"Phase: {clean_metrics['phase']}")
                
            # Print metrics if we have any to display
            if console_metrics:
                metrics_str = " | ".join(console_metrics)
                print(f"Step {step}: {metrics_str}")
                
            # Print messages if provided
            if 'message' in clean_metrics:
                print(f"📊 {clean_metrics['message']}")
                
            # Handle status updates
            if 'status' in clean_metrics:
                status = clean_metrics['status']
                if status == 'running':
                    print(f"🔄 Running: {clean_metrics.get('message', '')}")
                elif status == 'completed':
                    print(f"✅ Completed: {clean_metrics.get('message', '')}")
                elif status == 'error':
                    print(f"❌ Error: {clean_metrics.get('message', '')}")
                
        except Exception as e:
            logger.error(f"Failed to log metrics to wandb: {e}")
    
    def log_entropy_heatmap(self, entropy_values: np.ndarray, step: Optional[int] = None):
        """
        Log entropy heatmap to wandb.
        
        Args:
            entropy_values: Numpy array of entropy values [layers, heads]
            step: Optional step for the image
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot heatmap
            im = ax.imshow(entropy_values, cmap="viridis")
            plt.colorbar(im, ax=ax, label="Entropy")
            ax.set_title("Attention Head Entropy")
            ax.set_xlabel("Head Index")
            ax.set_ylabel("Layer Index")
            
            # Log to wandb
            wandb.log({"entropy_heatmap": wandb.Image(fig)}, step=step)
            
            # Clean up
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to log entropy heatmap to wandb: {e}")
    
    def log_gradient_heatmap(self, gradient_values: np.ndarray, step: Optional[int] = None):
        """
        Log gradient heatmap to wandb.
        
        Args:
            gradient_values: Numpy array of gradient values [layers, heads]
            step: Optional step for the image
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot heatmap
            im = ax.imshow(gradient_values, cmap="plasma")
            plt.colorbar(im, ax=ax, label="Gradient Norm")
            ax.set_title("Attention Head Gradient Norms")
            ax.set_xlabel("Head Index")
            ax.set_ylabel("Layer Index")
            
            # Log to wandb
            wandb.log({"gradient_heatmap": wandb.Image(fig)}, step=step)
            
            # Clean up
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to log gradient heatmap to wandb: {e}")
    
    def log_pruning_decision(self, decision_data: Dict[str, Any], step: Optional[int] = None):
        """
        Log pruning decision to wandb.
        
        Args:
            decision_data: Dictionary with pruning decision details
            step: Optional step for the decision
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            # Extract key metrics
            pruned_heads = decision_data.get("pruned_heads", [])
            strategy = decision_data.get("strategy", "unknown")
            cycle = decision_data.get("cycle", 0)
            
            # Log simple metrics to wandb
            wandb.log({
                "pruning/pruned_heads_count": len(pruned_heads),
                "pruning/strategy": strategy,
                "pruning/cycle": cycle
            }, step=step)
            
            # Log pruned heads as a table in wandb
            if pruned_heads:
                table = wandb.Table(columns=["layer", "head"])
                for layer, head in pruned_heads:
                    table.add_data(int(layer), int(head))
                wandb.log({"pruning/pruned_heads": table}, step=step)
            
            # Print pruning decision to console for real-time visibility
            print("\n" + "=" * 60)
            print(f"🔪 PRUNING DECISION (Cycle {cycle})")
            print(f"Strategy: {strategy}")
            print(f"Pruned {len(pruned_heads)} heads:")
            
            # Format pruned heads in a readable way
            if pruned_heads:
                # Group by layer for cleaner display
                layers = {}
                for layer, head in pruned_heads:
                    if layer not in layers:
                        layers[layer] = []
                    layers[layer].append(head)
                
                # Print grouped by layer
                for layer in sorted(layers.keys()):
                    heads = sorted(layers[layer])
                    print(f"  Layer {layer}: Heads {', '.join(str(h) for h in heads)}")
            else:
                print("  No heads pruned in this cycle")
                
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Failed to log pruning decision to wandb: {e}")
    
    def log_text_sample(self, prompt: str, generated_text: str, step: Optional[int] = None, model_type: str = "current"):
        """
        Log text generation sample to wandb with enhanced visualization.
        
        Args:
            prompt: Text prompt
            generated_text: Generated text
            step: Optional step for the sample
            model_type: Type of model that generated the text ("baseline", "current", "pruned", etc.)
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            # Create nicely formatted HTML for visualization
            # Use different styling based on model type
            border_color = {
                "baseline": "#3498db",  # Blue for baseline
                "current": "#2ecc71",   # Green for current model
                "pruned": "#9b59b6",    # Purple for pruned model
                "finetuned": "#e74c3c"  # Red for fine-tuned model
            }.get(model_type, "#2ecc71")
            
            title = {
                "baseline": "Baseline Model",
                "current": "Current Model",
                "pruned": "Pruned Model",
                "finetuned": "Fine-tuned Model"
            }.get(model_type, "Model") + " Output"
            
            # Format the HTML with syntax highlighting and styling
            html_content = f"""
            <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                <h3 style="color: {border_color}; margin-top: 0;">{title}</h3>
                <div style="background-color: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Prompt:</strong> <span style="color: #e6db74;">{prompt}</span>
                </div>
                <div style="background-color: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace;">
                    {generated_text}
                </div>
            </div>
            """
            
            # Log to wandb with improved visualization
            wandb.log({
                f"samples/{model_type}_generation": wandb.Html(html_content)
            }, step=step)
            
            # Print to console for real-time visibility
            print("\n" + "-" * 50)
            print(f"📝 Text Generation Sample ({model_type.title()} Model, Step {step}):")
            print(f"Prompt: \"{prompt}\"")
            print(f"Generated: \"{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}\"")
            print("-" * 50 + "\n")
        except Exception as e:
            logger.error(f"Failed to log text sample to wandb: {e}")
    
    def set_phase(self, phase: str, step: Optional[int] = None):
        """
        Set the current phase of the experiment.
        
        Args:
            phase: Current phase (setup, warmup, analysis, pruning, evaluation, complete)
            step: Optional step for the phase change
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            # Log phase to wandb
            wandb.log({"phase": phase}, step=step)
            
            # Print readable phase message to console
            phase_emojis = {
                "setup": "🛠️",
                "warmup": "🔥",
                "analysis": "🔍",
                "pruning": "✂️",
                "evaluation": "📊",
                "complete": "✅"
            }
            
            emoji = phase_emojis.get(phase, "🔄")
            
            # Print with separator for visibility
            print("\n" + "-" * 60)
            print(f"{emoji} PHASE CHANGE: Now entering {phase.upper()} phase")
            
            # Add helpful context based on the phase
            if phase == "setup":
                print("Setting up experiment environment and loading model...")
            elif phase == "warmup":
                print("Running warmup training to stabilize model before pruning...")
            elif phase == "analysis":
                print("Analyzing attention patterns and head importance...")
            elif phase == "pruning":
                print("Applying pruning strategy to remove less important heads...")
            elif phase == "evaluation":
                print("Evaluating model performance after pruning...")
            elif phase == "complete":
                print("Experiment completed successfully!")
                
            print("-" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to set phase in wandb: {e}")
    
    def finish(self):
        """Finish the wandb run."""
        if self.initialized and self.run:
            try:
                # Print dashboard summary
                print("\n" + "=" * 70)
                print("📊 DASHBOARD SUMMARY")
                print("=" * 70)
                
                # Show dashboard URL
                if self.run.url:
                    print(f"View complete experiment results at: {self.run.url}")
                
                # Show key metrics if we have them
                if hasattr(wandb, "run") and wandb.run.summary:
                    if "final_perplexity" in wandb.run.summary:
                        print(f"Final Perplexity: {wandb.run.summary['final_perplexity']:.2f}")
                    if "improvement_percent" in wandb.run.summary:
                        print(f"Improvement: {wandb.run.summary['improvement_percent']:.2f}%")
                    if "pruned_heads_count" in wandb.run.summary:
                        print(f"Pruned Heads: {wandb.run.summary['pruned_heads_count']}")
                
                print("\nFinishing and syncing dashboard data...")
                
                # Finish the run
                self.run.finish()
                
                # Final message
                print("✅ Dashboard tracking completed successfully")
                print("=" * 70 + "\n")
                
                logger.info("Finished wandb dashboard tracking")
            except Exception as e:
                logger.error(f"Failed to finish wandb run: {e}")
    
    def get_metrics_callback(self) -> Callable:
        """
        Get a callback function for logging metrics during training.
        
        Returns:
            Function that can be used as a metrics_callback
        """
        def metrics_callback(step: int, metrics: Dict[str, Any]):
            # Handle special cases
            if "entropy_values" in metrics:
                entropy_values = metrics.pop("entropy_values")
                if isinstance(entropy_values, np.ndarray):
                    self.log_entropy_heatmap(entropy_values, step)
            
            if "grad_norm_values" in metrics:
                grad_values = metrics.pop("grad_norm_values")
                if isinstance(grad_values, np.ndarray):
                    self.log_gradient_heatmap(grad_values, step)
            
            if "phase" in metrics:
                phase = metrics.get("phase")
                self.set_phase(phase, step)
            
            # Log all other metrics
            self.log_metrics(metrics, step)
        
        return metrics_callback
    
    def get_sample_callback(self) -> Callable:
        """
        Get a callback function for logging text samples during training.
        
        Returns:
            Function that can be used as a sample_callback
        """
        def sample_callback(step: int, sample_data: Dict[str, Any]):
            input_text = sample_data.get("input_text", "")
            generated_text = sample_data.get("generated_text", "")
            model_type = sample_data.get("model_type", "current")
            
            if input_text and generated_text:
                self.log_text_sample(input_text, generated_text, step, model_type)
        
        return sample_callback
        
    def log_inference_comparison(self, 
                               prompt: str, 
                               baseline_text: str, 
                               pruned_text: str, 
                               step: Optional[int] = None,
                               metrics: Optional[Dict[str, float]] = None):
        """
        Log a side-by-side comparison of baseline and pruned model outputs.
        
        Args:
            prompt: The input prompt
            baseline_text: Text generated by the baseline model
            pruned_text: Text generated by the pruned model
            step: Optional step for the visualization
            metrics: Optional dictionary of comparison metrics
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            # Create split-panel HTML visualization
            html_content = f"""
            <div style="display: flex; flex-direction: column; gap: 20px; margin: 15px 0;">
                <h3 style="margin-top: 0; text-align: center;">Model Output Comparison</h3>
                
                <div style="background-color: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Prompt:</strong> <span style="color: #e6db74;">{prompt}</span>
                </div>
                
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <!-- Baseline Model Output -->
                    <div style="flex: 1; min-width: 45%; border: 2px solid #3498db; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                        <h4 style="color: #3498db; margin-top: 0;">Baseline Model</h4>
                        <div style="background-color: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; height: 250px; overflow-y: auto;">
                            {baseline_text}
                        </div>
                    </div>
                    
                    <!-- Pruned Model Output -->
                    <div style="flex: 1; min-width: 45%; border: 2px solid #9b59b6; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                        <h4 style="color: #9b59b6; margin-top: 0;">Pruned Model</h4>
                        <div style="background-color: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; height: 250px; overflow-y: auto;">
                            {pruned_text}
                        </div>
                    </div>
                </div>
            """
            
            # Add metrics section if provided
            if metrics:
                html_content += """
                <div style="margin-top: 20px; border: 2px solid #2c3e50; border-radius: 8px; padding: 15px; background-color: #ecf0f1;">
                    <h4 style="margin-top: 0; color: #2c3e50;">Comparative Metrics</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                """
                
                for name, value in metrics.items():
                    if isinstance(value, float):
                        # Format the metric name to be more readable
                        display_name = " ".join(word.capitalize() for word in name.split("_"))
                        html_content += f"""
                        <div style="flex: 1; min-width: 200px; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="font-size: 14px; color: #7f8c8d;">{display_name}</div>
                            <div style="font-size: 18px; font-weight: bold; color: #2c3e50;">{value:.4f}</div>
                        </div>
                        """
                
                html_content += """
                    </div>
                </div>
                """
            
            # Close the main container
            html_content += "</div>"
            
            # Log to wandb
            wandb.log({
                "inference/model_comparison": wandb.Html(html_content)
            }, step=step)
            
            # Also log metrics separately for plotting
            if metrics:
                wandb.log({f"comparison/{k}": v for k, v in metrics.items()}, step=step)
            
            # Print to console
            print("\n" + "=" * 60)
            print("📋 MODEL COMPARISON")
            print(f"Prompt: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
            
            if metrics:
                print("\nMetrics:")
                for name, value in metrics.items():
                    print(f"  {name}: {value:.4f}")
            
            print("=" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log inference comparison: {e}")
            
    def log_perplexity_comparison(self, 
                                baseline_perplexity: float, 
                                pruned_perplexity: float,
                                texts: Optional[List[str]] = None,
                                step: Optional[int] = None):
        """
        Create a visualization comparing perplexity between baseline and pruned models.
        
        Args:
            baseline_perplexity: Perplexity score of the baseline model
            pruned_perplexity: Perplexity score of the pruned model
            texts: Optional list of text samples evaluated
            step: Optional step for the visualization
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Calculate improvement percentage
            improvement = ((baseline_perplexity - pruned_perplexity) / baseline_perplexity) * 100
            
            # Create figure for bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot data
            models = ['Baseline Model', 'Pruned Model']
            values = [baseline_perplexity, pruned_perplexity]
            
            # Create bar colors
            colors = ['#3498db', '#9b59b6']  # blue for baseline, purple for pruned
            
            # Create bars
            bars = ax.bar(models, values, color=colors, width=0.6)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=12)
            
            # Add improvement note
            if improvement > 0:
                ax.text(1.5, max(values) * 0.6, f'Improvement: {improvement:.2f}%', 
                       fontsize=14, ha='center', bbox=dict(facecolor='#2ecc71', alpha=0.2))
            else:
                ax.text(1.5, max(values) * 0.6, f'Degradation: {-improvement:.2f}%', 
                       fontsize=14, ha='center', bbox=dict(facecolor='#e74c3c', alpha=0.2))
            
            # Customize plot
            ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
            ax.set_title('Perplexity Comparison: Baseline vs. Pruned Model', fontsize=14)
            
            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-axis to start from 0
            ax.set_ylim(0, max(values) * 1.2)
            
            # Log to wandb
            wandb.log({
                "inference/perplexity_comparison": wandb.Image(fig)
            }, step=step)
            
            # Also log as metrics for plotting
            wandb.log({
                "perplexity/baseline": baseline_perplexity,
                "perplexity/pruned": pruned_perplexity,
                "perplexity/improvement_percent": improvement
            }, step=step)
            
            # Clean up
            plt.close(fig)
            
            # Print to console
            print("\n" + "-" * 60)
            print("📊 PERPLEXITY COMPARISON")
            print(f"Baseline: {baseline_perplexity:.2f}")
            print(f"Pruned: {pruned_perplexity:.2f}")
            print(f"Improvement: {improvement:.2f}%")
            print("-" * 60 + "\n")
            
            # If text samples are provided, also create a detailed visualization
            if texts and len(texts) > 0:
                self._create_detailed_perplexity_visualization(
                    baseline_perplexity, pruned_perplexity, texts, step)
                
        except Exception as e:
            logger.error(f"Failed to log perplexity comparison: {e}")
            
    def _create_detailed_perplexity_visualization(self, 
                                                baseline: float, 
                                                pruned: float, 
                                                texts: List[str],
                                                step: Optional[int] = None):
        """
        Create a more detailed visualization of perplexity results with sample texts.
        Private helper method for log_perplexity_comparison.
        """
        try:
            # Create HTML content with perplexity results and text samples
            improvement = ((baseline - pruned) / baseline) * 100
            color = "#2ecc71" if improvement > 0 else "#e74c3c"  # green for improvement, red for degradation
            
            html_content = f"""
            <div style="border: 2px solid #2c3e50; border-radius: 8px; padding: 20px; margin: 15px 0; background-color: #f8f9fa;">
                <h3 style="margin-top: 0;">Detailed Perplexity Analysis</h3>
                
                <div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px; background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 5px solid #3498db;">
                        <div style="font-size: 16px; color: #7f8c8d;">Baseline Perplexity</div>
                        <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{baseline:.2f}</div>
                    </div>
                    
                    <div style="flex: 1; min-width: 200px; background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 5px solid #9b59b6;">
                        <div style="font-size: 16px; color: #7f8c8d;">Pruned Perplexity</div>
                        <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{pruned:.2f}</div>
                    </div>
                    
                    <div style="flex: 1; min-width: 200px; background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 5px solid {color};">
                        <div style="font-size: 16px; color: #7f8c8d;">{"Improvement" if improvement > 0 else "Degradation"}</div>
                        <div style="font-size: 24px; font-weight: bold; color: {color};">{abs(improvement):.2f}%</div>
                    </div>
                </div>
                
                <h4 style="margin-top: 30px;">Evaluation Text Samples</h4>
                <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px;">
            """
            
            # Add sample texts
            for i, text in enumerate(texts[:5]):  # Limit to first 5 samples for brevity
                html_content += f"""
                <div style="padding: 15px; background-color: {"#fff" if i % 2 == 0 else "#f9f9f9"}; border-bottom: 1px solid #ddd;">
                    <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Sample {i+1}</div>
                    <div style="font-family: monospace; white-space: pre-wrap; font-size: 14px;">
                        {text[:500]}{"..." if len(text) > 500 else ""}
                    </div>
                </div>
                """
            
            # If there are more samples than we're showing
            if len(texts) > 5:
                html_content += f"""
                <div style="padding: 15px; text-align: center; background-color: #f1f1f1;">
                    +{len(texts) - 5} more samples (not shown)
                </div>
                """
            
            # Close containers
            html_content += """
                </div>
            </div>
            """
            
            # Log to wandb
            wandb.log({
                "inference/detailed_perplexity": wandb.Html(html_content)
            }, step=step)
            
        except Exception as e:
            logger.error(f"Failed to create detailed perplexity visualization: {e}")
    
    def log_attention_visualization(self, 
                                  baseline_attention: np.ndarray, 
                                  pruned_attention: np.ndarray,
                                  text: str,
                                  layer_idx: int, 
                                  head_idx: int,
                                  step: Optional[int] = None):
        """
        Create a visualization comparing attention patterns between baseline and pruned models.
        
        Args:
            baseline_attention: Attention weights from the baseline model [seq_len, seq_len]
            pruned_attention: Attention weights from the pruned model [seq_len, seq_len]
            text: The text used for visualization
            layer_idx: Index of the layer visualized
            head_idx: Index of the attention head visualized
            step: Optional step for the visualization
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
            
            # Process the text into tokens (simple whitespace tokenization for visualization)
            tokens = text.split()
            if len(tokens) > 30:  # Limit to 30 tokens for readability
                tokens = tokens[:30]
                baseline_attention = baseline_attention[:30, :30]
                pruned_attention = pruned_attention[:30, :30]
            
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Custom colormap for attention visualization
            colors = [(0.95, 0.95, 0.95), (0.8, 0.8, 1), (0, 0, 0.9)]  # light gray to dark blue
            cmap = LinearSegmentedColormap.from_list('attention_cmap', colors, N=100)
            
            # Plot baseline attention
            im1 = ax1.imshow(baseline_attention, cmap=cmap)
            ax1.set_title(f'Baseline Model Attention\nLayer {layer_idx}, Head {head_idx}')
            
            # Plot pruned model attention
            im2 = ax2.imshow(pruned_attention, cmap=cmap)
            ax2.set_title(f'Pruned Model Attention\nLayer {layer_idx}, Head {head_idx}')
            
            # Add labels for both plots
            for ax in [ax1, ax2]:
                ax.set_xticks(np.arange(len(tokens)))
                ax.set_yticks(np.arange(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
                ax.set_yticklabels(tokens, fontsize=10)
                
                # Add grid
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which="minor", bottom=False, left=False)
            
            # Add colorbars
            plt.colorbar(im1, ax=ax1, shrink=0.8, label='Attention Weight')
            plt.colorbar(im2, ax=ax2, shrink=0.8, label='Attention Weight')
            
            # Adjust layout
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                f"attention/layer{layer_idx}_head{head_idx}": wandb.Image(fig)
            }, step=step)
            
            # Clean up
            plt.close(fig)
            
            # Print info to console
            print(f"📊 Visualized attention patterns for Layer {layer_idx}, Head {head_idx}")
            
        except Exception as e:
            logger.error(f"Failed to log attention visualization: {e}")
            
    def log_inference_dashboard(self, 
                               perplexity_data: Dict[str, float],
                               generation_samples: Dict[str, Dict[str, str]],
                               attention_data: Optional[Dict[str, Any]] = None,
                               step: Optional[int] = None):
        """
        Create a comprehensive inference dashboard.
        
        Args:
            perplexity_data: Dict with 'baseline', 'pruned', and optional 'finetuned' perplexity values
            generation_samples: Dict of model_type -> {'prompt': str, 'output': str}
            attention_data: Optional attention visualization data
            step: Optional step for the dashboard
        """
        if not self.initialized and not self.initialize():
            return
        
        try:
            # Log performance metrics
            self.log_perplexity_comparison(
                perplexity_data.get('baseline', 0.0),
                perplexity_data.get('pruned', 0.0),
                step=step
            )
            
            # Log generation samples
            for model_type, sample in generation_samples.items():
                if 'prompt' in sample and 'output' in sample:
                    self.log_text_sample(
                        sample['prompt'],
                        sample['output'],
                        step=step,
                        model_type=model_type
                    )
            
            # For baseline and pruned, create a comparison
            if ('baseline' in generation_samples and 'pruned' in generation_samples and
                'prompt' in generation_samples['baseline'] and 'prompt' in generation_samples['pruned']):
                
                # Use the same prompt for comparison
                prompt = generation_samples['baseline']['prompt']
                
                # Log comparison
                self.log_inference_comparison(
                    prompt,
                    generation_samples['baseline']['output'],
                    generation_samples['pruned']['output'],
                    step=step,
                    metrics={
                        'baseline_perplexity': perplexity_data.get('baseline', 0.0),
                        'pruned_perplexity': perplexity_data.get('pruned', 0.0),
                        'improvement_percent': ((perplexity_data.get('baseline', 0.0) - perplexity_data.get('pruned', 0.0)) / 
                                               perplexity_data.get('baseline', 1.0)) * 100
                    }
                )
            
            # Log attention visualizations if provided
            if attention_data and isinstance(attention_data, dict):
                for key, data in attention_data.items():
                    if all(k in data for k in ['baseline', 'pruned', 'text', 'layer', 'head']):
                        self.log_attention_visualization(
                            data['baseline'],
                            data['pruned'],
                            data['text'],
                            data['layer'],
                            data['head'],
                            step=step
                        )
            
            # Log summary metrics
            wandb.log({
                "inference/summary/baseline_perplexity": perplexity_data.get('baseline', 0.0),
                "inference/summary/pruned_perplexity": perplexity_data.get('pruned', 0.0),
                "inference/summary/model_size_reduction": perplexity_data.get('model_size_reduction', 0.0),
                "inference/summary/inference_speedup": perplexity_data.get('inference_speedup', 0.0)
            }, step=step)
            
            # Print summary to console
            print("\n" + "=" * 70)
            print("📊 INFERENCE DASHBOARD UPDATED")
            print("=" * 70)
            print(f"Baseline Perplexity: {perplexity_data.get('baseline', 0.0):.2f}")
            print(f"Pruned Perplexity: {perplexity_data.get('pruned', 0.0):.2f}")
            
            if 'model_size_reduction' in perplexity_data:
                print(f"Model Size Reduction: {perplexity_data['model_size_reduction']:.2f}%")
                
            if 'inference_speedup' in perplexity_data:
                print(f"Inference Speedup: {perplexity_data['inference_speedup']:.2f}x")
                
            print("=" * 70 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log inference dashboard: {e}")