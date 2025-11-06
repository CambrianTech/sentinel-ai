#!/usr/bin/env python
"""
Adaptive RL Tuning for Neural Plasticity

This script runs an experiment that uses reinforcement learning to tune
plasticity decisions (pruning strategies, ratios) based on feedback from
previous cycles. It demonstrates how Sentinel-AI can learn to optimize
its own structural adaptation process.

The RL controller learns to:
1. Select pruning strategies based on model state
2. Adjust pruning ratios based on recovery patterns
3. Adapt its policies based on function preservation and task performance
"""

import os
import sys
import logging
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import model components
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import MultiCycleRunner 
from scripts.multi_cycle_runner import MultiCycleRunner

# Import RL controller
from sentinel.plasticity.controller.rl_controller import (
    RLController, RLControllerConfig, create_controller
)


class AdaptiveRLExperiment:
    """
    Experiment that uses reinforcement learning to optimize plasticity decisions.
    
    This class integrates the RL controller with the multi-cycle plasticity runner
    to create a closed-loop system that adapts its own structural changes.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        num_episodes: int = 10,
        cycles_per_episode: int = 5,
        steps_per_cycle: int = 50,
        batch_size: int = 4,
        device: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the adaptive RL experiment.
        
        Args:
            model_name: Hugging Face model name
            output_dir: Directory to save results
            num_episodes: Number of episodes (complete plasticity runs)
            cycles_per_episode: Number of plasticity cycles per episode
            steps_per_cycle: Training steps per cycle
            batch_size: Batch size for training
            device: Device for computation (cpu/cuda)
            experiment_name: Custom name for the experiment
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.cycles_per_episode = cycles_per_episode
        self.steps_per_cycle = steps_per_cycle
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"adaptive_rl_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize episode results
        self.episode_results = []
        
        # Initialize RL controller
        self._init_rl_controller()
        
        logger.info(f"Initialized AdaptiveRLExperiment (experiment: {self.experiment_name})")
        logger.info(f"Output directory: {self.experiment_dir}")
    
    def _init_rl_controller(self):
        """Initialize the RL controller"""
        rl_dir = os.path.join(self.experiment_dir, "rl_controller")
        
        # Create RL controller configuration
        rl_config = RLControllerConfig(
            output_dir=rl_dir,
            experiment_name=self.experiment_name,
            # State space configuration
            use_entropy_features=True,
            use_performance_features=True,
            use_function_features=True,
            use_pruning_history=True,
            # Action space configuration
            available_strategies=["entropy", "magnitude", "random"],
            min_pruning_ratio=0.1,
            max_pruning_ratio=0.5,
            pruning_ratio_steps=5,
            # RL configuration
            learning_rate=0.01,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.95,
            # Reward configuration
            recovery_weight=1.0,
            function_weight=1.0,
            performance_weight=1.0,
            entropy_weight=0.5
        )
        
        # Create controller
        self.rl_controller = create_controller(rl_config, self.device)
    
    def _prepare_metrics_for_controller(self, cycle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare metrics from cycle results for the RL controller.
        
        Args:
            cycle_results: List of results from plasticity cycles
            
        Returns:
            Dictionary of metrics suitable for the RL controller
        """
        metrics = {}
        
        # Handle case with only one cycle (initial state)
        if len(cycle_results) <= 1:
            return {
                "avg_entropy": 0.5,
                "entropy_change": 0.0,
                "entropy_std": 0.0,
                "perplexity": 10.0,
                "perplexity_change": 0.0,
                "recovery_rate": 0.0,
                "function_preservation": 0.0,
                "function_change": 0.0,
                "last_pruning_ratio": 0.3,
                "last_strategy": "entropy"
            }
        
        # Get current and previous cycle results
        current_cycle = cycle_results[-1]
        prev_cycle = cycle_results[-2]
        
        # Calculate metrics
        
        # 1. Entropy metrics (from entropy journal)
        if "entropy_journal" in current_cycle:
            entropy_data = current_cycle["entropy_journal"]
            prev_entropy_data = prev_cycle.get("entropy_journal", {})
            
            metrics["avg_entropy"] = entropy_data.get("avg_entropy", 0.5)
            
            # Calculate entropy change
            if "avg_entropy" in prev_entropy_data:
                metrics["entropy_change"] = (
                    entropy_data.get("avg_entropy", 0.5) - 
                    prev_entropy_data.get("avg_entropy", 0.5)
                )
            else:
                metrics["entropy_change"] = 0.0
            
            metrics["entropy_std"] = entropy_data.get("entropy_std", 0.0)
        
        # 2. Performance metrics
        metrics["perplexity"] = current_cycle.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", 10.0)
        
        # Calculate perplexity change
        prev_perplexity = prev_cycle.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", 10.0)
        metrics["perplexity_change"] = (metrics["perplexity"] - prev_perplexity) / max(1.0, prev_perplexity)
        
        # Calculate recovery rate
        if (isinstance(current_cycle.get("metrics", {}).get("perplexity", None), (int, float)) and
            isinstance(current_cycle.get("post_pruning", {}).get("metrics", {}).get("perplexity", None), (int, float)) and
            isinstance(current_cycle.get("post_fine_tuning", {}).get("metrics", {}).get("perplexity", None), (int, float))):
            
            init_ppl = current_cycle["metrics"]["perplexity"]
            pruned_ppl = current_cycle["post_pruning"]["metrics"]["perplexity"]
            ft_ppl = current_cycle["post_fine_tuning"]["metrics"]["perplexity"]
            
            if pruned_ppl > init_ppl:  # Performance degraded after pruning
                recovery = (pruned_ppl - ft_ppl) / (pruned_ppl - init_ppl)
                recovery = max(0.0, min(1.0, recovery))  # Clamp to [0, 1]
            else:
                recovery = 1.0  # No degradation = perfect recovery
                
            metrics["recovery_rate"] = recovery
        else:
            metrics["recovery_rate"] = 0.0
        
        # 3. Function preservation metrics
        if "function_tracking" in current_cycle:
            metrics["function_preservation"] = current_cycle["function_tracking"].get("overall_score", 0.0)
            
            # Calculate function change
            if "function_tracking" in prev_cycle:
                metrics["function_change"] = (
                    current_cycle["function_tracking"].get("overall_score", 0.0) - 
                    prev_cycle["function_tracking"].get("overall_score", 0.0)
                )
            else:
                metrics["function_change"] = 0.0
        
        # 4. Pruning history
        if "pruning" in current_cycle:
            metrics["last_pruning_ratio"] = current_cycle["pruning"].get("ratio", 0.3)
            metrics["last_strategy"] = current_cycle["pruning"].get("strategy", "entropy")
        
        return metrics
    
    def run_experiment(self):
        """
        Run the complete adaptive RL experiment.
        
        This method orchestrates the entire experiment:
        1. For each episode:
           a. Initialize a new model
           b. Create a multi-cycle runner
           c. For each cycle:
              i. Get pruning action from RL controller
              ii. Run a plasticity cycle with that action
              iii. Calculate reward and update controller
           d. Save results and visualizations
        
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting adaptive RL experiment: {self.experiment_name}")
        
        # Run episodes
        for episode in range(1, self.num_episodes + 1):
            logger.info(f"Starting episode {episode}/{self.num_episodes}")
            
            # Create episode directory
            episode_dir = os.path.join(self.experiment_dir, f"episode_{episode}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # Initialize episode metrics
            episode_metrics = {
                "episode": episode,
                "actions": [],
                "rewards": [],
                "cycle_results": []
            }
            
            # Create multi-cycle runner with default settings
            runner = MultiCycleRunner(
                model_name=self.model_name,
                output_dir=episode_dir,
                num_cycles=self.cycles_per_episode,
                # Default values that will be overridden by RL controller
                pruning_strategy="entropy",
                pruning_ratio=0.3,
                steps_per_cycle=self.steps_per_cycle,
                batch_size=self.batch_size,
                device=self.device,
                experiment_name=f"episode_{episode}"
            )
            
            # Create datasets
            logger.info("Creating datasets")
            datasets = runner._create_datasets()
            train_dataset, train_collator = datasets["train"]
            eval_dataset, eval_collator = datasets["eval"]
            test_prompts = datasets["test_prompts"]
            
            # Create data loaders
            train_loader = runner._create_data_loader(train_dataset)
            eval_loader = runner._create_data_loader(eval_dataset)
            
            # Record initial state
            logger.info("Recording initial model state")
            initial_metrics = runner._evaluate(runner.model, eval_loader)
            runner.entropy_journal.record_model_state(
                runner.model, 
                eval_loader, 
                cycle_idx=0, 
                cycle_name="Initial",
                metadata={"metrics": initial_metrics}
            )
            
            # Store model version
            model_versions = {
                "initial": runner.model.state_dict().copy()
            }
            
            # Store baseline result
            cycle_results = [{
                "cycle": 0,
                "phase": "initial",
                "metrics": initial_metrics,
                "pruning": None
            }]
            
            # Run plasticity cycles
            for cycle in range(1, self.cycles_per_episode + 1):
                logger.info(f"Running episode {episode}, cycle {cycle}")
                
                # Prepare metrics for controller
                controller_metrics = self._prepare_metrics_for_controller(cycle_results)
                
                # Select action from RL controller
                action = self.rl_controller.select_action(
                    controller_metrics, 
                    cycle, 
                    self.cycles_per_episode
                )
                
                # Record action
                episode_metrics["actions"].append(action)
                
                # Override runner settings with controller action
                pruning_strategy = action["strategy"]
                pruning_ratio = action["ratio"]
                
                # 1. Prune the model
                pruned_model, pruning_info = runner._prune_model(
                    runner.model, train_loader, cycle
                )
                cycle_result = {
                    "cycle": cycle,
                    "pruning": pruning_info
                }
                
                # 2. Record post-pruning state
                logger.info(f"Recording post-pruning state for cycle {cycle}")
                post_pruning_metrics = runner._evaluate(pruned_model, eval_loader)
                runner.entropy_journal.record_model_state(
                    pruned_model, 
                    eval_loader, 
                    cycle_idx=cycle, 
                    cycle_name=f"Cycle_{cycle}_PostPruning",
                    metadata={
                        "phase": "post_pruning",
                        "metrics": post_pruning_metrics,
                        "pruning": {
                            "strategy": pruning_strategy,
                            "ratio": pruning_ratio,
                            "pruned_heads_count": len(pruning_info["pruned_heads"])
                        }
                    }
                )
                
                # Store post-pruning metrics
                cycle_result["post_pruning"] = {
                    "metrics": post_pruning_metrics
                }
                
                # 3. Fine-tune the model
                logger.info(f"Fine-tuning model for cycle {cycle}")
                fine_tuned_model = runner._fine_tune(
                    pruned_model, train_dataset, train_collator, cycle
                )
                
                # Store model version
                model_versions[f"cycle_{cycle}"] = fine_tuned_model.state_dict().copy()
                
                # 4. Record post-fine-tuning state
                logger.info(f"Recording post-fine-tuning state for cycle {cycle}")
                post_ft_metrics = runner._evaluate(fine_tuned_model, eval_loader)
                
                # Get entropy journal metrics
                entropy_state = runner.entropy_journal.record_model_state(
                    fine_tuned_model, 
                    eval_loader, 
                    cycle_idx=cycle, 
                    cycle_name=f"Cycle_{cycle}_PostFineTuning",
                    metadata={
                        "phase": "post_fine_tuning",
                        "metrics": post_ft_metrics,
                    }
                )
                
                # Store post-fine-tuning metrics
                cycle_result["post_fine_tuning"] = {
                    "metrics": post_ft_metrics
                }
                
                # Extract entropy stats
                cycle_result["entropy_journal"] = {
                    "avg_entropy": np.mean([
                        np.mean(v) for v in entropy_state.get("entropy", {}).values()
                    ]),
                    "entropy_std": np.mean([
                        np.std(v) for v in entropy_state.get("entropy", {}).values()
                    ])
                }
                
                # 5. Track function preservation (compare with previous cycle)
                if cycle > 1:
                    logger.info(f"Tracking function preservation for cycle {cycle}")
                    
                    # Create previous model
                    prev_model = AutoModelForCausalLM.from_pretrained(runner.model_name)
                    prev_model.load_state_dict(model_versions[f"cycle_{cycle-1}"])
                    prev_model = prev_model.to(runner.device)
                    
                    # Track function
                    function_results = runner.function_tracker.track_function(
                        prev_model,
                        fine_tuned_model,
                        test_prompts,
                        runner.tokenizer,
                        cycle_idx=cycle,
                        cycle_name=f"Cycle_{cycle}"
                    )
                    
                    # Store function tracking results
                    cycle_result["function_tracking"] = {
                        "overall_score": function_results["summary"].get("overall_preservation_score", None),
                        "output_similarity": function_results["summary"].get("output", {}).get("avg_cosine_similarity", None)
                    }
                
                # Store cycle result
                cycle_results.append(cycle_result)
                
                # Calculate reward and update controller
                reward = self.rl_controller.observe_reward(
                    self._prepare_metrics_for_controller(cycle_results),
                    cycle,
                    self.cycles_per_episode,
                    done=(cycle == self.cycles_per_episode)
                )
                
                # Record reward
                episode_metrics["rewards"].append(reward)
                
                # Update model for next cycle
                runner.model = fine_tuned_model
            
            # Create summary visualizations
            logger.info("Creating summary visualizations")
            runner._create_summary_visualizations()
            
            # Create summary report
            logger.info("Creating summary report")
            runner._create_summary_report()
            
            # Create entropy visualizations
            logger.info("Creating entropy visualizations")
            runner.entropy_journal.visualize_entropy_evolution()
            runner.entropy_journal.visualize_gate_evolution()
            runner.entropy_journal.create_summary_report()
            
            # Create function tracking summary
            logger.info("Creating function tracking summary")
            runner.function_tracker.create_summary_report()
            
            # Store cycle results in episode metrics
            episode_metrics["cycle_results"] = cycle_results
            
            # Store episode results
            self.episode_results.append(episode_metrics)
            
            logger.info(f"Completed episode {episode}")
        
        # Create experiment summary
        self._create_experiment_summary()
        
        logger.info(f"Experiment completed successfully. Results saved to {self.experiment_dir}")
        
        return {
            "experiment_name": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "num_episodes": self.num_episodes,
            "cycles_per_episode": self.cycles_per_episode,
            "episode_results": self.episode_results
        }
    
    def _create_experiment_summary(self):
        """Create a summary of the experiment"""
        # Create visualizations
        viz_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot episode rewards
        if self.episode_results:
            # Calculate average reward per episode
            avg_rewards = [
                sum(episode["rewards"]) / len(episode["rewards"])
                for episode in self.episode_results
            ]
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, 'o-')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title('Learning Progress Over Episodes')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(viz_dir, "episode_rewards.png"))
            plt.close()
            
            # Plot strategy selection evolution
            strategy_counts = {}
            for episode in self.episode_results:
                episode_idx = episode["episode"]
                strategies = [action["strategy"] for action in episode["actions"]]
                
                for strategy in set(strategies):
                    if strategy not in strategy_counts:
                        strategy_counts[strategy] = [0] * len(self.episode_results)
                    
                    strategy_counts[strategy][episode_idx - 1] = strategies.count(strategy)
            
            plt.figure(figsize=(12, 6))
            
            episodes = range(1, len(self.episode_results) + 1)
            bottom = np.zeros(len(episodes))
            
            for strategy, counts in strategy_counts.items():
                plt.bar(episodes, counts, bottom=bottom, label=strategy)
                bottom += np.array(counts)
            
            plt.xlabel('Episode')
            plt.ylabel('Strategy Count')
            plt.title('Evolution of Strategy Selection')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "strategy_evolution.png"))
            plt.close()
            
            # Plot average pruning ratio evolution
            avg_ratios = [
                sum(action["ratio"] for action in episode["actions"]) / len(episode["actions"])
                for episode in self.episode_results
            ]
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(avg_ratios) + 1), avg_ratios, 'o-')
            plt.xlabel('Episode')
            plt.ylabel('Average Pruning Ratio')
            plt.title('Evolution of Pruning Ratios')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(viz_dir, "pruning_ratio_evolution.png"))
            plt.close()
        
        # Create summary report
        summary_file = os.path.join(self.experiment_dir, "experiment_summary.md")
        
        with open(summary_file, 'w') as f:
            f.write(f"# Adaptive RL Plasticity Experiment Summary\n\n")
            f.write(f"Experiment: {self.experiment_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Experiment Configuration\n\n")
            f.write(f"- Model: {self.model_name}\n")
            f.write(f"- Number of episodes: {self.num_episodes}\n")
            f.write(f"- Cycles per episode: {self.cycles_per_episode}\n")
            f.write(f"- Steps per cycle: {self.steps_per_cycle}\n")
            f.write(f"- Batch size: {self.batch_size}\n")
            f.write(f"- Device: {self.device}\n\n")
            
            if self.episode_results:
                f.write(f"## Learning Progress\n\n")
                
                # Create table for episode rewards
                f.write(f"| Episode | Avg Reward | Best Strategy | Avg Pruning Ratio |\n")
                f.write(f"|---------|------------|---------------|-------------------|\n")
                
                for episode in self.episode_results:
                    episode_idx = episode["episode"]
                    rewards = episode["rewards"]
                    actions = episode["actions"]
                    
                    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                    
                    # Count strategy selections
                    strategy_counts = {}
                    for action in actions:
                        strategy = action["strategy"]
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
                    # Find most common strategy
                    best_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else "N/A"
                    
                    # Calculate average pruning ratio
                    avg_ratio = sum(action["ratio"] for action in actions) / len(actions) if actions else 0.0
                    
                    f.write(f"| {episode_idx} | {avg_reward:.3f} | {best_strategy} | {avg_ratio:.2f} |\n")
                
                # Key observations
                f.write(f"\n## Key Observations\n\n")
                
                # Look for trends in rewards
                first_reward = sum(self.episode_results[0]["rewards"]) / len(self.episode_results[0]["rewards"])
                last_reward = sum(self.episode_results[-1]["rewards"]) / len(self.episode_results[-1]["rewards"])
                
                if last_reward > first_reward * 1.1:
                    f.write(f"- **Reward improved significantly** ({(last_reward - first_reward) / first_reward * 100:.1f}%) over episodes, indicating effective learning.\n")
                elif last_reward > first_reward:
                    f.write(f"- **Reward improved slightly** ({(last_reward - first_reward) / first_reward * 100:.1f}%) over episodes.\n")
                elif last_reward < first_reward * 0.9:
                    f.write(f"- **Reward declined** ({(last_reward - first_reward) / first_reward * 100:.1f}%) over episodes, suggesting challenges in learning or converging.\n")
                else:
                    f.write(f"- **Reward remained stable** over episodes.\n")
                
                # Look for strategy preferences
                all_actions = [action for episode in self.episode_results for action in episode["actions"]]
                strategy_counts = {}
                for action in all_actions:
                    strategy = action["strategy"]
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                if strategy_counts:
                    total = sum(strategy_counts.values())
                    f.write(f"- **Strategy preferences**:\n")
                    for strategy, count in strategy_counts.items():
                        f.write(f"  - {strategy}: {count / total * 100:.1f}% of decisions\n")
                
                # Look for pruning ratio trends
                first_ratios = [action["ratio"] for action in self.episode_results[0]["actions"]]
                last_ratios = [action["ratio"] for action in self.episode_results[-1]["actions"]]
                
                avg_first = sum(first_ratios) / len(first_ratios) if first_ratios else 0.0
                avg_last = sum(last_ratios) / len(last_ratios) if last_ratios else 0.0
                
                if avg_last > avg_first * 1.1:
                    f.write(f"- **Pruning ratios increased** from {avg_first:.2f} to {avg_last:.2f} on average, suggesting more aggressive pruning was beneficial.\n")
                elif avg_last < avg_first * 0.9:
                    f.write(f"- **Pruning ratios decreased** from {avg_first:.2f} to {avg_last:.2f} on average, suggesting more conservative pruning was beneficial.\n")
                else:
                    f.write(f"- **Pruning ratios remained stable** around {avg_last:.2f} on average.\n")
            
            f.write(f"\n## Conclusions\n\n")
            f.write("*Add your interpretation of these results here.*\n")
        
        logger.info(f"Created experiment summary at {summary_file}")


def run_adaptive_rl_experiment(args):
    """Run adaptive RL experiment with command line arguments"""
    # Create experiment
    experiment = AdaptiveRLExperiment(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        cycles_per_episode=args.cycles,
        steps_per_cycle=args.steps_per_cycle,
        batch_size=args.batch_size,
        device=args.device,
        experiment_name=args.experiment_name
    )
    
    # Run experiment
    results = experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RL Tuning for Neural Plasticity")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes (complete plasticity runs)")
    parser.add_argument("--cycles", type=int, default=5, help="Number of plasticity cycles per episode")
    parser.add_argument("--steps_per_cycle", type=int, default=50, help="Training steps per cycle")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom name for the experiment")
    
    args = parser.parse_args()
    
    run_adaptive_rl_experiment(args)