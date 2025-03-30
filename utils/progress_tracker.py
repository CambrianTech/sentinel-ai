# utils/progress_tracker.py

import matplotlib.pyplot as plt


class ProgressTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.baseline_perplexities = []
        self.active_heads = []
        self.param_counts = []

    def log_train_loss(self, loss):
        self.train_losses.append(loss)

    def log_val_metrics(self, val_loss, perplexity, baseline_ppl):
        self.val_losses.append(val_loss)
        self.perplexities.append(perplexity)
        self.baseline_perplexities.append(baseline_ppl)

    def log_model_state(self, active_head_count, param_count):
        self.active_heads.append(active_head_count)
        self.param_counts.append(param_count)

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
