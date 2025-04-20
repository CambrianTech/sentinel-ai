#!/usr/bin/env python
# Simulate a neural plasticity experiment with the modular API

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import time

# Set up output directory
OUTPUT_DIR = "modular_demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class MockDataLoader:
    """Mock data loader to simulate dataset batches"""
    def __init__(self, num_batches=10, batch_size=4, seq_length=128):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __next__(self):
        if self.current_batch < self.num_batches:
            self.current_batch += 1
            # Create a random batch with input_ids and attention_mask
            return {
                "input_ids": torch.randint(0, 50257, (self.batch_size, self.seq_length)),
                "attention_mask": torch.ones(self.batch_size, self.seq_length),
                "labels": torch.randint(0, 50257, (self.batch_size, self.seq_length))
            }
        else:
            raise StopIteration
            
    def __len__(self):
        return self.num_batches

class MockModel(torch.nn.Module):
    """Mock model to simulate a transformer with attention heads"""
    def __init__(self, num_layers=6, num_heads=12):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Create a mock config object
        self.config = type('obj', (object,), {
            'num_hidden_layers': num_layers,
            'num_attention_heads': num_heads
        })
        
        # Create random parameters to simulate a real model
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(128, 128) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, output_attentions=False):
        # Simulate loss
        loss = torch.tensor(3.0 - 0.1 * random.random())
        
        # Simulate logits
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_length = input_ids.shape[1] if input_ids is not None else 10
        logits = torch.randn(batch_size, seq_length, 50257)
        
        # Create random attention outputs if requested
        if output_attentions:
            attentions = tuple(
                torch.rand(batch_size, self.num_heads, seq_length, seq_length)
                for _ in range(self.num_layers)
            )
            # Create a named tuple to simulate the transformer output
            from collections import namedtuple
            Output = namedtuple('Output', ['loss', 'logits', 'attentions'])
            return Output(loss=loss, logits=logits, attentions=attentions)
        else:
            from collections import namedtuple
            Output = namedtuple('Output', ['loss', 'logits'])
            return Output(loss=loss, logits=logits)
            
    def generate(self, input_ids=None, attention_mask=None, max_length=100, **kwargs):
        # Simulate text generation
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        return torch.randint(0, 50257, (batch_size, max_length))


class MockTokenizer:
    """Mock tokenizer to simulate encoding and decoding"""
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "</s>"
        
    def encode(self, text, return_tensors=None):
        # Simulate encoding
        if text == "Once upon a time":
            base_tokens = [345, 567, 789, 234]
        elif text == "The future of artificial intelligence":
            base_tokens = [123, 456, 789, 234, 567, 890]
        else:
            # Generate random tokens based on text length
            base_tokens = [random.randint(100, 1000) for _ in range(len(text) // 2)]
            
        # Pad to a minimum length
        if len(base_tokens) < 5:
            base_tokens = base_tokens + [0] * (5 - len(base_tokens))
            
        # Return tensor if requested
        if return_tensors == "pt":
            return torch.tensor([base_tokens])
        else:
            return base_tokens
            
    def decode(self, tokens, skip_special_tokens=True):
        # Simulate decoding
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            
        if isinstance(tokens[0], list):
            tokens = tokens[0]
            
        # Generate some fake text based on token values
        words = []
        word_list = ["the", "of", "and", "to", "a", "in", "that", "is", 
                     "was", "for", "on", "are", "with", "as", "by", "this",
                     "but", "not", "or", "from", "an", "they", "which", 
                     "one", "you", "all", "were", "we", "when", "have", "had",
                     "there", "been", "many", "some", "artificial", "intelligence",
                     "neural", "network", "learning", "model", "data", "training"]
        
        # Create a coherent looking sequence based on the tokens
        for i, t in enumerate(tokens[:20]):  # Limit to first 20 tokens
            if t % 5 == 0:
                words.append(".")
            if t % 7 == 0:
                words.append("\n")
            word_idx = t % len(word_list)
            words.append(word_list[word_idx])
            
        return " ".join(words)


class MockNeuralPlasticityExperiment:
    """
    A mock implementation of the NeuralPlasticityExperiment class
    to demonstrate the modular API without actual model training.
    """
    
    def __init__(
        self,
        model_name="mock-model",
        dataset="mock-dataset",
        dataset_config="mock-config",
        output_dir=OUTPUT_DIR,
        batch_size=4,
        max_length=128,
        pruning_level=0.2,
        pruning_strategy="combined",
        learning_rate=5e-5,
        verbose=True,
        save_results=True
    ):
        """
        Initialize the mock experiment.
        
        Args:
            model_name: Name of the model (for display only)
            dataset: Name of the dataset (for display only)
            dataset_config: Dataset configuration (for display only)
            output_dir: Directory to save results
            batch_size: Batch size for training
            max_length: Maximum sequence length
            pruning_level: Percentage of heads to prune (0-1)
            pruning_strategy: Strategy name (for display only)
            learning_rate: Learning rate (for display only)
            verbose: Whether to print progress
            save_results: Whether to save results
        """
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.pruning_level = pruning_level
        self.pruning_strategy = pruning_strategy
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.save_results = save_results
        
        # Create simulation metrics
        self.baseline_loss = 4.5
        self.baseline_perplexity = 90.0
        self.final_loss = None
        self.final_perplexity = None
        
        # Create output directory
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Initialize mock components
        self.model = None
        self.tokenizer = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.pruned_heads = []
        self.device = torch.device("cpu")
        self.reporter = None
        
        # Track cycles
        self.current_cycle = 0
        
        if self.verbose:
            print(f"Neural Plasticity Experiment initialized with:")
            print(f"  Model: {model_name}")
            print(f"  Dataset: {dataset}/{dataset_config}")
            print(f"  Output directory: {output_dir}")
            print(f"  Pruning strategy: {pruning_strategy} at {pruning_level*100:.1f}% level")
            
    def setup(self):
        """Set up the experiment by creating model and dataloaders"""
        if self.verbose:
            print("\nSetting up experiment...")
            print(f"Loading model: {self.model_name}")
        
        # Create mock model
        self.model = MockModel(num_layers=6, num_heads=12)
        
        # Create mock tokenizer
        self.tokenizer = MockTokenizer()
        
        # Create mock dataloaders
        self.train_dataloader = MockDataLoader(num_batches=20, batch_size=self.batch_size)
        self.validation_dataloader = MockDataLoader(num_batches=5, batch_size=self.batch_size)
        
        if self.verbose:
            print(f"Model loaded with {self.model.config.num_hidden_layers} layers and {self.model.config.num_attention_heads} heads per layer")
            print(f"Created train dataloader with {len(self.train_dataloader)} batches")
            print(f"Created validation dataloader with {len(self.validation_dataloader)} batches")
            
        return self
        
    def run_warmup(self, max_epochs=1, patience=15, min_steps=50, max_steps=150):
        """Simulate running warmup training"""
        if self.verbose:
            print("\n=== Running Warmup Phase ===")
            
        # Simulate some processing time
        time.sleep(1)
        
        # Simulate decreasing loss
        initial_loss = 5.0
        final_loss = 4.5
        
        num_steps = min(max_steps, len(self.train_dataloader) * max_epochs)
        
        # Simulate training steps
        if self.verbose:
            print(f"Running {num_steps} warmup steps...")
            
        for step in range(num_steps):
            # Simulate progress
            if self.verbose and step % 5 == 0:
                loss = initial_loss - (initial_loss - final_loss) * (step / num_steps)
                print(f"  Step {step+1}/{num_steps} - Loss: {loss:.4f}")
                
        # Update baseline metrics
        self.baseline_loss = final_loss
        self.baseline_perplexity = np.exp(final_loss)
        
        # Create a fake warmup visualization
        if self.save_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            steps = list(range(num_steps))
            losses = [initial_loss - (initial_loss - final_loss) * (step / num_steps) for step in range(num_steps)]
            ax.plot(steps, losses)
            ax.set_title("Warmup Training Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            warmup_dir = os.path.join(self.output_dir, "warmup")
            os.makedirs(warmup_dir, exist_ok=True)
            fig.savefig(os.path.join(warmup_dir, "warmup_loss.png"))
            plt.close(fig)
        
        if self.verbose:
            print(f"Warmup completed with baseline perplexity: {self.baseline_perplexity:.2f}")
            
        return {
            "steps": num_steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss
        }
        
    def analyze_attention(self):
        """Simulate analyzing attention patterns"""
        if self.verbose:
            print("\n=== Analyzing Attention Patterns ===")
            
        # Simulate some processing time
        time.sleep(1)
        
        # Create fake entropy and gradient data
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        
        entropy_values = torch.rand(num_layers, num_heads) * 0.5  # Lower is more focused
        grad_norm_values = torch.rand(num_layers, num_heads) * 2.0  # Higher is more learning
        
        # Create visualizations
        if self.save_results:
            # Entropy heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(entropy_values.numpy(), cmap="viridis")
            plt.colorbar(im, ax=ax)
            ax.set_title("Attention Head Entropy (Lower = More Focused)")
            ax.set_xlabel("Head Index")
            ax.set_ylabel("Layer Index")
            
            # Save the figure
            analysis_dir = os.path.join(self.output_dir, "attention_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            fig.savefig(os.path.join(analysis_dir, "entropy_heatmap.png"))
            plt.close(fig)
            
            # Gradient heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(grad_norm_values.numpy(), cmap="plasma")
            plt.colorbar(im, ax=ax)
            ax.set_title("Gradient Norms (Higher = More Learning)")
            ax.set_xlabel("Head Index")
            ax.set_ylabel("Layer Index")
            
            # Save the figure
            fig.savefig(os.path.join(analysis_dir, "gradient_heatmap.png"))
            plt.close(fig)
        
        if self.verbose:
            print(f"Analyzed attention patterns across {num_layers} layers with {num_heads} heads each")
            
        return {
            "entropy_values": entropy_values,
            "grad_norm_values": grad_norm_values
        }
    
    def run_pruning_cycle(self, training_steps=100):
        """Simulate running a single pruning cycle"""
        self.current_cycle += 1
        
        if self.verbose:
            print(f"\n=== Running Pruning Cycle {self.current_cycle} ===")
            
        # Simulate processing time
        time.sleep(1)
        
        # Select random heads to prune
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        
        # Determine number of heads to prune
        total_heads = num_layers * num_heads
        heads_to_prune = int(total_heads * self.pruning_level * 0.33)  # Prune a third of the target each cycle
        
        # Generate random heads to prune
        newly_pruned = []
        for _ in range(heads_to_prune):
            layer = random.randint(0, num_layers - 1)
            head = random.randint(0, num_heads - 1)
            head_tuple = (layer, head)
            if head_tuple not in self.pruned_heads and head_tuple not in newly_pruned:
                newly_pruned.append(head_tuple)
                
        # Update pruned heads list
        self.pruned_heads.extend(newly_pruned)
        
        # Simulate training after pruning
        if self.verbose:
            print(f"Pruned {len(newly_pruned)} new heads, running {training_steps} training steps...")
            
        # Simulate training with decreasing loss
        cycle_start_loss = 4.5 - 0.5 * self.current_cycle
        cycle_end_loss = cycle_start_loss - 0.3
        
        for step in range(training_steps):
            # Simulate progress
            if self.verbose and step % (training_steps // 4) == 0:
                loss = cycle_start_loss - (cycle_start_loss - cycle_end_loss) * (step / training_steps)
                print(f"  Step {step+1}/{training_steps} - Loss: {loss:.4f}")
                
        # Update final metrics
        self.final_loss = cycle_end_loss
        self.final_perplexity = np.exp(cycle_end_loss)
        
        # Create a pruning cycle visualization
        if self.save_results:
            # Training metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            steps = list(range(training_steps))
            losses = [cycle_start_loss - (cycle_start_loss - cycle_end_loss) * (step / training_steps) for step in steps]
            ax.plot(steps, losses)
            ax.set_title(f"Training Loss (Cycle {self.current_cycle})")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            cycle_dir = os.path.join(self.output_dir, f"cycle_{self.current_cycle}")
            os.makedirs(cycle_dir, exist_ok=True)
            fig.savefig(os.path.join(cycle_dir, "training_loss.png"))
            plt.close(fig)
            
            # Pruning mask visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a matrix of zeros
            mask = np.zeros((num_layers, num_heads))
            
            # Mark newly pruned heads with 2 (for red)
            for layer, head in newly_pruned:
                mask[layer, head] = 2
                
            # Mark previously pruned heads with 1 (for green)
            for layer, head in self.pruned_heads:
                if (layer, head) not in newly_pruned:
                    mask[layer, head] = 1
                    
            # Create custom colormap
            import matplotlib.colors as mcolors
            colors = [(1,1,1), (0.8,0.2,0.2), (0.2,0.8,0.2)]  # white, red, green
            cmap_custom = mcolors.ListedColormap(colors)
            bounds = [0, 0.5, 1.5, 2.5]
            norm = mcolors.BoundaryNorm(bounds, cmap_custom.N)
            
            # Create heatmap
            im = ax.imshow(mask, cmap=cmap_custom, norm=norm)
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
            cbar.set_ticklabels(['Active', 'Previously Pruned', 'Newly Pruned'])
            
            ax.set_title(f"Pruning State After Cycle {self.current_cycle}")
            ax.set_xlabel("Head Index")
            ax.set_ylabel("Layer Index")
            
            # Save the figure
            fig.savefig(os.path.join(cycle_dir, "pruning_mask.png"))
            plt.close(fig)
            
        if self.verbose:
            print(f"Pruning cycle completed, new perplexity: {self.final_perplexity:.2f}")
            print(f"Total pruned heads: {len(self.pruned_heads)}/{total_heads} ({len(self.pruned_heads)/total_heads*100:.1f}%)")
            
        return {
            "pruned_heads": newly_pruned,
            "final_metrics": {
                "loss": self.final_loss,
                "perplexity": self.final_perplexity
            }
        }
    
    def run_multiple_pruning_cycles(self, num_cycles=3, training_steps=100):
        """
        Run multiple pruning cycles with continuous tracking.
        
        This demonstrates the key method we added to the experiment class,
        which encapsulates the continuous pruning process.
        
        Args:
            num_cycles: Number of pruning cycles to run
            training_steps: Training steps per cycle
            
        Returns:
            Results dictionary with metrics across all cycles
        """
        if self.verbose:
            print(f"\n=== Running {num_cycles} Continuous Pruning Cycles ===")
            
        # Tracking structures for pruning state
        all_pruning_results = []
        cumulative_pruned_heads = []
        cycle_metrics = {
            "perplexity": [],
            "pruned_heads_count": [],
            "loss": [],
            "newly_pruned_count": []
        }
        
        # Get model structure info
        total_heads = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        
        for cycle in range(num_cycles):
            # Run a pruning cycle with training
            pruning_results = self.run_pruning_cycle(training_steps=training_steps)
            
            # Store the results for this cycle
            all_pruning_results.append(pruning_results)
            
            # Extract newly pruned heads
            newly_pruned = pruning_results.get("pruned_heads", [])
            
            # Track cumulative pruned heads (avoiding duplicates)
            newly_pruned_count = 0
            for head in newly_pruned:
                if head not in cumulative_pruned_heads:
                    cumulative_pruned_heads.append(head)
                    newly_pruned_count += 1
            
            # Track metrics for this cycle
            cycle_metrics["perplexity"].append(pruning_results.get("final_metrics", {}).get("perplexity", 0))
            cycle_metrics["pruned_heads_count"].append(len(cumulative_pruned_heads))
            cycle_metrics["loss"].append(pruning_results.get("final_metrics", {}).get("loss", 0))
            cycle_metrics["newly_pruned_count"].append(newly_pruned_count)
            
            # Calculate sparsity
            if "sparsity" not in cycle_metrics:
                cycle_metrics["sparsity"] = []
            cycle_metrics["sparsity"].append(len(cumulative_pruned_heads) / total_heads * 100)
                
        # Create visualization of metrics evolution across cycles
        if num_cycles > 1 and self.save_results:
            self._create_metrics_evolution_plot(cycle_metrics, num_cycles)
        
        # Return comprehensive results
        return {
            "all_pruning_results": all_pruning_results,
            "cumulative_pruned_heads": cumulative_pruned_heads,
            "cycle_metrics": cycle_metrics,
            "num_cycles": num_cycles,
            "total_pruned": len(cumulative_pruned_heads),
            "sparsity": len(cumulative_pruned_heads) / total_heads * 100
        }
    
    def _create_metrics_evolution_plot(self, cycle_metrics, num_cycles):
        """Create visualization of metrics evolution across pruning cycles"""
        if self.verbose:
            print("\nCreating metrics evolution visualization...")
            
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Create a 2x2 grid of subplots
        plt.subplot(2, 2, 1)
        plt.plot(range(1, num_cycles+1), cycle_metrics["perplexity"], marker='o')
        plt.title("Perplexity Evolution")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Perplexity")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(range(1, num_cycles+1), cycle_metrics["pruned_heads_count"], marker='o', color='r')
        plt.title("Cumulative Pruned Heads")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Number of Pruned Heads")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(range(1, num_cycles+1), cycle_metrics["sparsity"], marker='o', color='g')
        plt.title("Model Sparsity")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Sparsity (%)")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(range(1, num_cycles+1), cycle_metrics["loss"], marker='o', color='purple')
        plt.title("Loss Evolution")
        plt.xlabel("Pruning Cycle")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle("Neural Plasticity Evolution Across Cycles", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        fig.savefig(os.path.join(self.output_dir, "metrics_evolution.png"))
        plt.close(fig)
            
    def evaluate(self):
        """Simulate evaluation of the model"""
        if self.verbose:
            print("\n=== Evaluating Model ===")
            
        # Simulate some processing time
        time.sleep(1)
        
        # Calculate final perplexity if not done yet
        if self.final_perplexity is None:
            self.final_loss = 4.0
            self.final_perplexity = np.exp(self.final_loss)
            
        # Calculate improvement
        if self.baseline_perplexity is not None:
            improvement = (self.baseline_perplexity - self.final_perplexity) / self.baseline_perplexity * 100
        else:
            improvement = 0
            
        if self.verbose:
            print(f"Baseline perplexity: {self.baseline_perplexity:.2f}")
            print(f"Final perplexity: {self.final_perplexity:.2f}")
            print(f"Improvement: {improvement:.2f}%")
            
        return {
            "loss": self.final_loss,
            "perplexity": self.final_perplexity,
            "baseline_loss": self.baseline_loss,
            "baseline_perplexity": self.baseline_perplexity,
            "improvement_percent": improvement
        }
        
    def generate_examples(self, prompts=None, max_length=100):
        """Simulate text generation with the model"""
        if self.verbose:
            print("\n=== Generating Text Examples ===")
            
        # Use default prompts if none provided
        if prompts is None:
            prompts = {
                "story": "Once upon a time",
                "ai": "The future of artificial intelligence"
            }
            
        # Generate text for each prompt
        generation_results = {}
        
        for name, prompt in prompts.items():
            if self.verbose:
                print(f"\nPrompt: {prompt}")
                
            # Simulate some processing time
            time.sleep(0.5)
            
            # Encode the prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate text
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length
            )
            
            # Decode the output
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Store the result
            generation_results[name] = generated_text
            
            if self.verbose:
                print(f"Generated: {generated_text[:100]}...")
                
            # Save to file if requested
            if self.save_results:
                generation_dir = os.path.join(self.output_dir, "generation")
                os.makedirs(generation_dir, exist_ok=True)
                
                with open(os.path.join(generation_dir, f"{name}.txt"), 'w') as f:
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Generated text:\n{generated_text}")
                    
        return generation_results
        
    def visualize_metrics_dashboard(self, figsize=(15, 10), save_path=None):
        """Create a comprehensive dashboard of metrics"""
        if self.verbose:
            print("\n=== Creating Metrics Dashboard ===")
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create a grid layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 1: Perplexity improvement
        ax1 = fig.add_subplot(gs[0, 0])
        perplexities = [self.baseline_perplexity]
        for result in self.cycle_results if hasattr(self, 'cycle_results') else []:
            perplexities.append(result.get("final_metrics", {}).get("perplexity", perplexities[-1]))
        
        ax1.plot(range(len(perplexities)), perplexities, marker='o', color='blue')
        ax1.set_title("Perplexity Evolution")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Perplexity")
        ax1.set_xticks(range(len(perplexities)))
        ax1.set_xticklabels(["Baseline"] + [f"Cycle {i+1}" for i in range(len(perplexities)-1)])
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pruning progress
        ax2 = fig.add_subplot(gs[0, 1])
        total_heads = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        pruned_count = len(self.pruned_heads)
        
        ax2.bar(["Active", "Pruned"], [total_heads - pruned_count, pruned_count], color=['green', 'red'])
        ax2.set_title("Attention Head Status")
        ax2.set_ylabel("Number of Heads")
        
        # Add sparsity percentage
        sparsity = pruned_count / total_heads * 100
        ax2.text(1, pruned_count/2, f"{sparsity:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        
        # Plot 3: Layer-wise pruning heatmap
        ax3 = fig.add_subplot(gs[1, :])
        
        # Create a matrix representation of pruned heads
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        pruning_state = np.zeros((num_layers, num_heads))
        
        for layer, head in self.pruned_heads:
            pruning_state[layer, head] = 1
            
        im = ax3.imshow(pruning_state, cmap='Reds')
        ax3.set_title("Pruned Heads Heatmap")
        ax3.set_xlabel("Head Index")
        ax3.set_ylabel("Layer Index")
        
        # Add colorbar
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Performance comparison
        ax4 = fig.add_subplot(gs[2, :])
        
        metrics = ["Baseline", "After Pruning"]
        values = [self.baseline_perplexity, self.final_perplexity]
        
        ax4.bar(metrics, values, color=['blue', 'green'])
        ax4.set_title("Perplexity Comparison")
        ax4.set_ylabel("Perplexity (Lower is Better)")
        
        # Add improvement percentage
        if self.baseline_perplexity and self.final_perplexity:
            improvement = (self.baseline_perplexity - self.final_perplexity) / self.baseline_perplexity * 100
            ax4.text(1, self.final_perplexity/2, f"{improvement:.1f}% better", ha='center', va='center', fontweight='bold')
        
        # Add title
        fig.suptitle(f"Neural Plasticity Dashboard - {self.model_name}", fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)
        
        # Save the figure
        if save_path:
            fig.savefig(save_path)
        elif self.save_results:
            fig.savefig(os.path.join(self.output_dir, "metrics_dashboard.png"))
            
        plt.close(fig)
        
        if self.verbose:
            print(f"Metrics dashboard created")
            
        return fig
    
    def run_full_experiment(self, warmup_epochs=1, pruning_cycles=3, training_steps=100):
        """
        Run a complete neural plasticity experiment pipeline.
        
        This is a convenience method that chains together the individual
        experiment steps into a complete workflow.
        
        Args:
            warmup_epochs: Number of warmup epochs
            pruning_cycles: Number of pruning cycles to run
            training_steps: Number of training steps per pruning cycle
            
        Returns:
            Dictionary with full experiment results
        """
        if self.verbose:
            print("\n=== Running Full Neural Plasticity Experiment ===")
            
        # Run each step of the experiment
        self.setup()
        self.run_warmup(max_epochs=warmup_epochs)
        self.analyze_attention()
        
        # Run multiple pruning cycles with continuous tracking
        multi_cycle_results = self.run_multiple_pruning_cycles(
            num_cycles=pruning_cycles,
            training_steps=training_steps
        )
        
        # Final evaluation and generation
        eval_metrics = self.evaluate()
        generated_texts = self.generate_examples()
        
        # Create dashboard
        self.visualize_metrics_dashboard()
        
        if self.verbose:
            print("\n=== Full Experiment Completed Successfully ===")
            
        return {
            "baseline_metrics": {
                "loss": self.baseline_loss,
                "perplexity": self.baseline_perplexity
            },
            "final_metrics": {
                "loss": self.final_loss,
                "perplexity": self.final_perplexity
            },
            "improvement_percent": eval_metrics["improvement_percent"],
            "pruned_heads": self.pruned_heads,
            "multi_cycle_results": multi_cycle_results
        }


def run_demo():
    """Run a full demonstration of the neural plasticity experiment"""
    
    print("=" * 80)
    print("NEURAL PLASTICITY MODULAR API DEMONSTRATION")
    print("=" * 80)
    print("""
This script simulates a complete neural plasticity experiment using the modular API.
It demonstrates the key feature we implemented: run_multiple_pruning_cycles.

In a real experiment on Colab, this would run for hours/days with a T4 GPU, showing:
- Continuous pruning evolution with cycles of prune → train → evaluate
- Visualizations that update as heads are pruned
- Text generation demonstrating model capability is preserved
- Metrics tracking showing how pruning affects model performance
    """)
    
    experiment = MockNeuralPlasticityExperiment(
        model_name="mock-gpt2",
        dataset="wikitext",
        dataset_config="wikitext-2-raw-v1",
        output_dir=OUTPUT_DIR,
        pruning_level=0.15,
        verbose=True
    )
    
    # Run the experiment
    print("\n" + "=" * 80)
    print("Running step-by-step experiment...")
    print("=" * 80)
    
    # Setup
    experiment.setup()
    
    # Warmup
    experiment.run_warmup(max_epochs=1, min_steps=10, max_steps=20)
    
    # Analyze attention
    experiment.analyze_attention()
    
    # Run multiple pruning cycles (key new functionality)
    num_cycles = 3
    multi_cycle_results = experiment.run_multiple_pruning_cycles(
        num_cycles=num_cycles,
        training_steps=20
    )
    
    # Evaluation
    experiment.evaluate()
    
    # Generate examples
    experiment.generate_examples({
        "story": "Once upon a time",
        "ai": "The future of artificial intelligence"
    })
    
    # Create dashboard
    experiment.visualize_metrics_dashboard()
    
    # Display results
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"Baseline perplexity: {experiment.baseline_perplexity:.2f}")
    print(f"Final perplexity: {experiment.final_perplexity:.2f}")
    
    if experiment.baseline_perplexity is not None and experiment.final_perplexity is not None:
        improvement = (experiment.baseline_perplexity - experiment.final_perplexity) / experiment.baseline_perplexity * 100
        print(f"Perplexity improvement: {improvement:.2f}%")
    
    total_heads = experiment.model.config.num_hidden_layers * experiment.model.config.num_attention_heads
    print(f"Total pruned heads: {len(experiment.pruned_heads)}/{total_heads} ({len(experiment.pruned_heads)/total_heads*100:.1f}%)")
    
    # Now run the one-shot version
    print("\n" + "=" * 80)
    print("Running one-shot experiment (using run_full_experiment)...")
    print("=" * 80)
    
    # Create a new experiment for the one-shot approach
    one_shot_dir = os.path.join(OUTPUT_DIR, "one_shot")
    os.makedirs(one_shot_dir, exist_ok=True)
    
    one_shot_experiment = MockNeuralPlasticityExperiment(
        model_name="mock-gpt2",
        dataset="wikitext",
        dataset_config="wikitext-2-raw-v1",
        output_dir=one_shot_dir,
        pruning_level=0.15,
        verbose=True
    )
    
    # Run the full experiment in one call
    results = one_shot_experiment.run_full_experiment(
        warmup_epochs=1,
        pruning_cycles=num_cycles,
        training_steps=20
    )
    
    print("\n" + "=" * 80)
    print("NEURAL PLASTICITY API BENEFITS")
    print("=" * 80)
    print("""
The run_multiple_pruning_cycles method provides several key benefits:

1. Fully Modular: Encapsulates the entire continuous pruning process in one call
2. Cross-Platform: Works the same on Apple Silicon, CPU, or T4 GPU
3. Visualization Handling: Automatically creates pruning state visualizations
4. Tracking & Analysis: Monitors metrics across cycles without custom code
5. Long-Running Support: Designed for multi-day runs with continuous updates

In a notebook environment, where a T4 GPU would run this for days, the user
can see pruning evolution graphics and text generation samples while the
experiment is running, without explicit update code in the notebook itself.
    """)
    
    # Show where outputs are stored
    print("\n" + "=" * 80)
    print("OUTPUT VISUALIZATIONS")
    print("=" * 80)
    print(f"All visualizations are stored in: {OUTPUT_DIR}")
    print("\nKey files:")
    print(f"- Metrics Dashboard: {os.path.join(OUTPUT_DIR, 'metrics_dashboard.png')}")
    print(f"- Metrics Evolution: {os.path.join(OUTPUT_DIR, 'metrics_evolution.png')}")
    print(f"- Pruning State Heatmaps: {os.path.join(OUTPUT_DIR, 'cycle_N/pruning_mask.png')}")
    
    return {
        "step_by_step_results": multi_cycle_results,
        "one_shot_results": results
    }

if __name__ == "__main__":
    results = run_demo()