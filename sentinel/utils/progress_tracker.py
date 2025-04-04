# utils/progress_tracker.py

import matplotlib.pyplot as plt


class ProgressTracker:
    def __init__(self, total_epochs=None, steps_per_epoch=None, log_interval=10, eval_interval=100, quiet=False):
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.baseline_perplexities = []
        self.active_heads = []
        self.param_counts = []
        
        # Configuration for display
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.quiet = quiet

    def log_train_loss(self, loss):
        self.train_losses.append(loss)

    def log_val_metrics(self, val_loss, perplexity, baseline_ppl):
        self.val_losses.append(val_loss)
        self.perplexities.append(perplexity)
        self.baseline_perplexities.append(baseline_ppl)

    def log_model_state(self, active_head_count, param_count):
        self.active_heads.append(active_head_count)
        self.param_counts.append(param_count)

    def log_train_step(self, epoch, step, metrics):
        """Log metrics from a training step"""
        if self.quiet:
            return
        
        # Store for plotting
        self.log_train_loss(metrics.get("loss", 0))
        
        # Display progress
        progress_str = f"Epoch {epoch+1}/{self.total_epochs} - Step {step+1}/{self.steps_per_epoch}"
        metrics_str = f"Loss: {metrics.get('loss', 0):.4f}"
        
        # Add controller metrics if available
        if "pruned_percent" in metrics:
            metrics_str += f" | Pruned: {metrics.get('pruned_percent', 0)*100:.1f}%"
        if "controller_lr" in metrics:
            metrics_str += f" | Ctrl LR: {metrics.get('controller_lr', 0):.5f}"
        
        print(f"{progress_str} - {metrics_str}")
    
    def log_eval_step(self, epoch, step, metrics):
        """Log metrics from an evaluation step"""
        # Always show eval results, even in quiet mode as they're less frequent
        
        # Store for plotting if perplexity available
        if "loss" in metrics:
            self.log_val_metrics(
                metrics.get("loss", 0),
                metrics.get("perplexity", 0),
                metrics.get("baseline_perplexity", 0)
            )
        
        # Display evaluation results
        progress_str = f"[Evaluation] Epoch {epoch+1}/{self.total_epochs}"
        metrics_str = f"Loss: {metrics.get('loss', 0):.4f}"
        
        # Add perplexity if available
        if "perplexity" in metrics:
            metrics_str += f" | PPL: {metrics.get('perplexity', 0):.2f}"
        
        # Add pruning metrics if available
        if "pruned_percent" in metrics:
            metrics_str += f" | Pruned: {metrics.get('pruned_percent', 0)*100:.1f}%"
        
        print(f"{progress_str} - {metrics_str}")
    
    def plot(self):
        if not self.train_losses:
            print("No data to plot yet.")
            return

        steps = list(range(len(self.train_losses)))

        plt.figure(figsize=(10, 5))
        plt.plot(steps, self.train_losses, label="Train Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

        if self.val_losses:
            eval_steps = list(range(0, len(self.train_losses), max(1, len(self.train_losses) // len(self.val_losses))))
            plt.figure(figsize=(10, 5))
            plt.plot(eval_steps, self.val_losses, label="Validation Loss")
            plt.plot(eval_steps, self.perplexities, label="Adaptive Perplexity")
            plt.plot(eval_steps, self.baseline_perplexities, label="Baseline Perplexity", linestyle="--")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss / Perplexity")
            plt.title("Validation Metrics")
            plt.legend()
            plt.show()

        if self.active_heads:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(self.active_heads)), self.active_heads, label="Active Heads")
            plt.xlabel("Training Steps")
            plt.ylabel("Number of Active Heads")
            plt.title("Active Heads Over Time")
            plt.legend()
            plt.show()

        if self.param_counts:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(self.param_counts)), self.param_counts, label="Trainable Parameters")
            plt.xlabel("Training Steps")
            plt.ylabel("Parameter Count")
            plt.title("Parameter Count Over Time")
            plt.legend()
            plt.show()
