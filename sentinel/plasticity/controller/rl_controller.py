#!/usr/bin/env python
"""
RL Controller for Neural Plasticity

This module implements a reinforcement learning controller that learns
to optimize pruning and plasticity decisions based on feedback from
previous cycles. It enables the model to learn how to adapt its own
structure for optimal performance and recovery.

The controller observes metrics like recovery rate, function preservation,
and entropy changes, then decides on pruning strategies, ratios, and other
plasticity parameters.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RLControllerConfig:
    """Configuration for the RL controller"""
    
    # Output configuration
    output_dir: str = "./output/rl_controller"
    experiment_name: Optional[str] = None
    
    # State space configuration
    use_entropy_features: bool = True
    use_performance_features: bool = True
    use_function_features: bool = True
    use_pruning_history: bool = True
    
    # Action space configuration
    available_strategies: List[str] = field(default_factory=lambda: ["entropy", "magnitude", "random"])
    min_pruning_ratio: float = 0.1
    max_pruning_ratio: float = 0.5
    pruning_ratio_steps: int = 5  # Number of discrete pruning ratio options
    
    # RL configuration
    learning_rate: float = 0.01
    gamma: float = 0.95  # Discount factor
    epsilon_start: float = 1.0  # Exploration rate start
    epsilon_end: float = 0.1  # Exploration rate end
    epsilon_decay: float = 0.95  # Decay per episode
    batch_size: int = 32
    memory_size: int = 10000
    
    # Reward configuration
    recovery_weight: float = 1.0
    function_weight: float = 1.0
    performance_weight: float = 1.0
    entropy_weight: float = 0.5
    
    # Training settings
    update_frequency: int = 1  # Update target network every N episodes
    min_samples_for_training: int = 100
    
    # Advanced settings
    hidden_size: int = 128  # Hidden layer size for neural networks
    
    def __post_init__(self):
        """Set experiment name if not provided"""
        if self.experiment_name is None:
            self.experiment_name = f"rl_controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate discrete pruning ratios
        self.pruning_ratio_options = np.linspace(
            self.min_pruning_ratio, 
            self.max_pruning_ratio, 
            self.pruning_ratio_steps
        ).tolist()


class ReplayMemory:
    """Memory buffer for experience replay"""
    
    def __init__(self, capacity: int):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, next_state, reward):
        """
        Add an experience to memory.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, next_states, rewards)
        """
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q-Network for plasticity control"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        """
        Initialize DQN.
        
        Args:
            input_size: Size of state vector
            output_size: Number of possible actions
            hidden_size: Size of hidden layer
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RLController:
    """
    Reinforcement learning controller for neural plasticity.
    
    This controller learns to optimize pruning and plasticity decisions
    based on feedback from previous cycles. It enables the model to
    learn how to adapt its own structure for optimal performance and recovery.
    """
    
    def __init__(
        self,
        config: Optional[RLControllerConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the RL controller.
        
        Args:
            config: Configuration for the controller
            device: Device to run computations on
        """
        self.config = config or RLControllerConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize state and action dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        
        # Initialize networks
        self.policy_net = DQN(self.state_dim, self.action_dim, self.config.hidden_size).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.config.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        
        # Initialize replay memory
        self.memory = ReplayMemory(self.config.memory_size)
        
        # Initialize exploration rate
        self.epsilon = self.config.epsilon_start
        
        # Initialize step counter
        self.steps_done = 0
        
        # Initialize history
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        self.q_value_history = []
        
        # Initialize current state and action
        self.current_state = None
        self.current_action = None
        self.current_action_idx = None
        
        logger.info(f"Initialized RLController (device: {self.device})")
        logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _calculate_state_dim(self):
        """Calculate the dimension of the state vector"""
        state_dim = 0
        
        # Cycle index and count features (always included)
        state_dim += 2
        
        # Entropy features
        if self.config.use_entropy_features:
            state_dim += 3  # avg_entropy, entropy_change, entropy_std
        
        # Performance features
        if self.config.use_performance_features:
            state_dim += 3  # perplexity, perplexity_change, recovery_rate
        
        # Function preservation features
        if self.config.use_function_features:
            state_dim += 2  # function_preservation, function_change
        
        # Pruning history features
        if self.config.use_pruning_history:
            state_dim += 2  # last_pruning_ratio, last_strategy_onehot (simplified)
        
        return state_dim
    
    def _calculate_action_dim(self):
        """Calculate the dimension of the action space"""
        # Number of strategies x number of pruning ratios
        return len(self.config.available_strategies) * len(self.config.pruning_ratio_options)
    
    def _build_state_vector(self, metrics: Dict[str, Any], cycle_idx: int, num_cycles: int) -> torch.Tensor:
        """
        Build state vector from metrics.
        
        Args:
            metrics: Dictionary of metrics from current and previous cycles
            cycle_idx: Current cycle index
            num_cycles: Total number of cycles
            
        Returns:
            Tensor representing the current state
        """
        features = []
        
        # Always include cycle information
        features.append(cycle_idx / max(1, num_cycles))  # Normalized cycle index
        features.append(num_cycles / 10)  # Normalized cycle count (assuming max 10 cycles)
        
        # Entropy features
        if self.config.use_entropy_features:
            features.append(metrics.get("avg_entropy", 0.0))
            features.append(metrics.get("entropy_change", 0.0))
            features.append(metrics.get("entropy_std", 0.0))
        
        # Performance features
        if self.config.use_performance_features:
            # Normalize perplexity (assuming range 0-100)
            perplexity = min(100, metrics.get("perplexity", 0.0)) / 100.0
            features.append(perplexity)
            features.append(metrics.get("perplexity_change", 0.0))
            features.append(metrics.get("recovery_rate", 0.0))
        
        # Function preservation features
        if self.config.use_function_features:
            features.append(metrics.get("function_preservation", 0.0))
            features.append(metrics.get("function_change", 0.0))
        
        # Pruning history features
        if self.config.use_pruning_history:
            features.append(metrics.get("last_pruning_ratio", 0.3))
            
            # One-hot for last strategy (simplified to one value)
            strategies = self.config.available_strategies
            last_strategy = metrics.get("last_strategy", strategies[0])
            last_strategy_idx = strategies.index(last_strategy) if last_strategy in strategies else 0
            features.append(last_strategy_idx / len(strategies))
        
        # Convert to tensor
        state = torch.tensor([features], dtype=torch.float).to(self.device)
        return state
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """
        Decode action index to pruning parameters.
        
        Args:
            action_idx: Index of the action
            
        Returns:
            Dictionary with pruning strategy and ratio
        """
        num_ratios = len(self.config.pruning_ratio_options)
        strategy_idx = action_idx // num_ratios
        ratio_idx = action_idx % num_ratios
        
        strategy = self.config.available_strategies[strategy_idx]
        ratio = self.config.pruning_ratio_options[ratio_idx]
        
        return {
            "strategy": strategy,
            "ratio": ratio,
            "action_idx": action_idx
        }
    
    def _encode_action(self, action: Dict[str, Any]) -> int:
        """
        Encode pruning parameters to action index.
        
        Args:
            action: Dictionary with pruning strategy and ratio
            
        Returns:
            Action index
        """
        strategy = action["strategy"]
        ratio = action["ratio"]
        
        strategy_idx = self.config.available_strategies.index(strategy)
        ratio_idx = self.config.pruning_ratio_options.index(ratio)
        
        num_ratios = len(self.config.pruning_ratio_options)
        action_idx = strategy_idx * num_ratios + ratio_idx
        
        return action_idx
    
    def select_action(self, metrics: Dict[str, Any], cycle_idx: int, num_cycles: int) -> Dict[str, Any]:
        """
        Select the next pruning action based on current state.
        
        Args:
            metrics: Dictionary of metrics from current and previous cycles
            cycle_idx: Current cycle index
            num_cycles: Total number of cycles
            
        Returns:
            Dictionary with selected pruning strategy and ratio
        """
        # Build state vector
        state = self._build_state_vector(metrics, cycle_idx, num_cycles)
        self.current_state = state
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_dim)
            logger.info(f"Selecting random action (epsilon: {self.epsilon:.3f})")
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()
                self.q_value_history.append(q_values.max(1)[0].item())
                logger.info(f"Selecting greedy action (Q-value: {q_values.max(1)[0].item():.3f})")
        
        # Decode action
        action = self._decode_action(action_idx)
        
        # Store current action
        self.current_action = action
        self.current_action_idx = action_idx
        
        # Save to history
        self.action_history.append(action)
        self.state_history.append(state.cpu().numpy())
        
        logger.info(f"Selected action: strategy={action['strategy']}, ratio={action['ratio']:.2f}")
        
        return action
    
    def observe_reward(
        self, 
        metrics: Dict[str, Any], 
        cycle_idx: int, 
        num_cycles: int, 
        done: bool = False
    ) -> float:
        """
        Observe reward after taking an action and update the controller.
        
        Args:
            metrics: Dictionary of metrics after taking the action
            cycle_idx: Current cycle index
            num_cycles: Total number of cycles
            done: Whether this is the final cycle
            
        Returns:
            The calculated reward
        """
        if self.current_state is None or self.current_action is None:
            logger.warning("Cannot observe reward: no action has been taken")
            return 0.0
        
        # Calculate reward
        reward = self._calculate_reward(metrics)
        logger.info(f"Observed reward: {reward:.3f}")
        
        # Build next state
        next_state = self._build_state_vector(metrics, cycle_idx, num_cycles)
        
        # Store transition in memory
        self.memory.push(
            self.current_state,
            self.current_action_idx,
            next_state,
            torch.tensor([reward], device=self.device)
        )
        
        # Save reward to history
        self.reward_history.append(reward)
        
        # Perform optimization step
        self._optimize_model()
        
        # Update epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        # Update target network
        if self.steps_done % self.config.update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Save model if done
        if done:
            self._save_model()
            self._create_visualizations()
        
        self.steps_done += 1
        
        return reward
    
    def _calculate_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate reward based on metrics.
        
        Args:
            metrics: Dictionary of metrics after taking the action
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Recovery rate component
        if "recovery_rate" in metrics and self.config.recovery_weight > 0:
            recovery_rate = metrics["recovery_rate"]
            recovery_reward = recovery_rate * self.config.recovery_weight
            reward += recovery_reward
            logger.debug(f"Recovery reward: {recovery_reward:.3f}")
        
        # Function preservation component
        if "function_preservation" in metrics and self.config.function_weight > 0:
            function_score = metrics["function_preservation"]
            function_reward = function_score * self.config.function_weight
            reward += function_reward
            logger.debug(f"Function reward: {function_reward:.3f}")
        
        # Performance component
        if "perplexity_change" in metrics and self.config.performance_weight > 0:
            # Negative change in perplexity is good
            perplexity_change = metrics["perplexity_change"]
            # Bound and normalize
            performance_reward = -min(1.0, max(-1.0, perplexity_change)) * self.config.performance_weight
            reward += performance_reward
            logger.debug(f"Performance reward: {performance_reward:.3f}")
        
        # Entropy component
        if "entropy_change" in metrics and self.config.entropy_weight > 0:
            # Negative change in entropy indicates specialization
            entropy_change = metrics["entropy_change"]
            entropy_reward = -min(1.0, max(-1.0, entropy_change)) * self.config.entropy_weight
            reward += entropy_reward
            logger.debug(f"Entropy reward: {entropy_reward:.3f}")
        
        return reward
    
    def _optimize_model(self):
        """Optimize the model using experience replay"""
        # Skip if not enough samples
        if len(self.memory) < self.config.min_samples_for_training:
            return
        
        # Skip if batch size is too large
        batch_size = min(self.config.batch_size, len(self.memory))
        
        # Sample batch
        states, actions, next_states, rewards = self.memory.sample(batch_size)
        
        # Convert to tensors
        state_batch = torch.cat(states)
        action_batch = torch.tensor(actions, device=self.device)
        reward_batch = torch.cat(rewards)
        non_final_mask = torch.tensor(
            [s is not None for s in next_states],
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(batch_size, device=self.device)
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.config.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def _save_model(self):
        """Save the model and history"""
        # Save model
        model_path = self.output_dir / "policy_net.pt"
        torch.save(self.policy_net.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save history
        history = {
            "actions": [
                {
                    "cycle": i,
                    "strategy": a["strategy"],
                    "ratio": a["ratio"],
                    "action_idx": a["action_idx"]
                } for i, a in enumerate(self.action_history)
            ],
            "rewards": self.reward_history,
            "q_values": self.q_value_history,
            "config": {k: str(v) if isinstance(v, (list, dict)) else v for k, v in vars(self.config).items()}
        }
        
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved history to {history_path}")
    
    def _create_visualizations(self):
        """Create visualizations of controller performance"""
        viz_dir = self.output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot rewards
        if self.reward_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.reward_history, 'o-')
            plt.xlabel('Cycle')
            plt.ylabel('Reward')
            plt.title('Rewards Over Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(viz_dir / "rewards.png")
            plt.close()
        
        # Plot actions
        if self.action_history:
            plt.figure(figsize=(10, 6))
            
            # Plot pruning ratios
            ratios = [a["ratio"] for a in self.action_history]
            plt.plot(ratios, 'o-', label='Pruning Ratio')
            
            plt.xlabel('Cycle')
            plt.ylabel('Pruning Ratio')
            plt.title('Pruning Decisions Over Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig(viz_dir / "pruning_ratios.png")
            plt.close()
            
            # Plot strategies
            plt.figure(figsize=(10, 6))
            strategies = [a["strategy"] for a in self.action_history]
            strategy_indices = [self.config.available_strategies.index(s) for s in strategies]
            
            plt.plot(strategy_indices, 'o-')
            plt.xlabel('Cycle')
            plt.ylabel('Strategy Index')
            plt.yticks(
                range(len(self.config.available_strategies)),
                self.config.available_strategies
            )
            plt.title('Pruning Strategies Over Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(viz_dir / "pruning_strategies.png")
            plt.close()
        
        # Plot Q-values
        if self.q_value_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.q_value_history, 'o-')
            plt.xlabel('Decision Step')
            plt.ylabel('Max Q-Value')
            plt.title('Q-Value Evolution')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(viz_dir / "q_values.png")
            plt.close()
        
        # Plot epsilon decay
        steps = range(self.steps_done + 1)
        epsilon_values = [
            max(self.config.epsilon_end, self.config.epsilon_start * (self.config.epsilon_decay ** s))
            for s in steps
        ]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, epsilon_values)
        plt.xlabel('Step')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate Decay')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(viz_dir / "epsilon_decay.png")
        plt.close()
        
        logger.info(f"Created visualizations in {viz_dir}")
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.info(f"Loaded model from {model_path}")


def create_controller(config=None, device=None):
    """
    Convenience function to create an RL controller.
    
    Args:
        config: Controller configuration
        device: Device to run computations on
        
    Returns:
        RLController instance
    """
    return RLController(config, device)


if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Controller for Neural Plasticity")
    parser.add_argument("--output_dir", type=str, default="./output/rl_controller", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create controller
    config = RLControllerConfig(output_dir=args.output_dir)
    controller = RLController(config, args.device)
    
    # Simulate a few actions and rewards
    for i in range(5):
        # Simulate metrics
        metrics = {
            "avg_entropy": 0.7 - (i * 0.1),
            "entropy_change": -0.1,
            "entropy_std": 0.2,
            "perplexity": 10.0 - i,
            "perplexity_change": -0.5,
            "recovery_rate": 0.6 + (i * 0.05),
            "function_preservation": 0.8,
            "function_change": -0.02,
            "last_pruning_ratio": 0.3,
            "last_strategy": "entropy"
        }
        
        # Select action
        action = controller.select_action(metrics, i, 5)
        print(f"Action {i}: {action}")
        
        # Simulate reward
        reward = controller.observe_reward(metrics, i, 5, done=(i == 4))
        print(f"Reward {i}: {reward:.3f}")
    
    # Create visualizations
    controller._create_visualizations()
    
    logger.info("Simulation complete")