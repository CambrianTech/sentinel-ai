"""
Adaptive Neural Plasticity System

This module implements a self-improving neural network architecture that:
1. Continuously evaluates its own output quality
2. Adjusts its structure through neural plasticity (pruning, measuring, growing, learning)
3. Uses degeneration detection as a feedback mechanism
4. Adapts its growth and pruning strategies based on historical performance
5. Remembers successful transformations and repeats them

The system aims to optimize itself without human intervention while maintaining 
output quality above defined thresholds.
"""

import os
import time
import random
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

# Import plasticity mechanisms
from utils.pruning.fixed_pruning_module import FixedPruningModule
from utils.pruning.strategies import get_strategy as get_pruning_strategy
from utils.pruning.growth import (
    grow_attention_heads_gradually, 
    determine_active_heads,
    get_strategy as get_growth_strategy
)
from utils.pruning.head_lr_manager import HeadLRManager
from utils.train_utils import FineTuner
from utils.metrics_logger import MetricsLogger

# Import degeneration detection and display tools
from utils.pruning.inference_utils import (
    check_for_degeneration,
    apply_degeneration_penalty,
    display_generation,
    display_side_by_side
)

class AdaptivePlasticitySystem:
    """
    Adaptive system that improves neural networks through iterative cycles of
    plasticity while monitoring output quality and adapting strategies.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset: Any,
        output_dir: str = "./output/adaptive_plasticity",
        device: str = "cuda",
        max_degeneration_score: float = 3.0,
        max_perplexity_increase: float = 0.15,  # 15% max degradation
        learning_rate: float = 5e-5,
        memory_capacity: int = 10,
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Initialize the adaptive plasticity system.
        
        Args:
            model_name: Name or path of the model to optimize
            dataset: Dataset to use for training and evaluation
            output_dir: Directory to save results and checkpoints
            device: Device to use ('cuda' or 'cpu')
            max_degeneration_score: Maximum acceptable degeneration score
            max_perplexity_increase: Maximum acceptable perplexity increase ratio
            learning_rate: Base learning rate for training
            memory_capacity: Number of past successful transformations to remember
            verbose: Whether to print detailed information
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.device = device
        self.max_degeneration_score = max_degeneration_score
        self.max_perplexity_increase = max_perplexity_increase
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.verbose = verbose
        self.seed = seed
        
        # Setup directories
        self.setup_directories()
        
        # Initialize pruning module
        self.pruning_module = FixedPruningModule(model_name)
        success = self.pruning_module.load_model()
        if not success:
            raise RuntimeError(f"Failed to load model {model_name}")
            
        # Save model architecture details
        self.num_layers = self.pruning_module.num_layers
        self.num_heads = self.pruning_module.num_heads
        self.total_heads = self.num_layers * self.num_heads
        
        # Initialize metrics logger
        metrics_file = os.path.join(self.metrics_dir, "metrics.jsonl")
        self.metrics_logger = MetricsLogger(metrics_file)
        
        # Set up memory for successful transformations
        self.transformation_memory = []
        
        # Current model state
        self.current_params = self.pruning_module.model.params
        self.current_active_heads = determine_active_heads(self.pruning_module, self.current_params)
        self.current_perplexity = None
        
        # Initialize strategy ratings based on performance
        self.strategy_ratings = {
            "pruning": {
                "entropy": 1.0,
                "magnitude": 1.0,
                "random": 0.5
            },
            "growth": {
                "gradient_sensitivity": 1.0,
                "entropy_gap": 1.0,
                "balanced": 1.0, 
                "random": 0.5
            }
        }
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize baseline metrics
        self.baseline_evaluation = None
        
        # Print initialization info
        if self.verbose:
            print(f"Initialized AdaptivePlasticitySystem for model: {model_name}")
            print(f"Model architecture: Layers={self.num_layers}, Heads per layer={self.num_heads}")
            print(f"Total heads: {self.total_heads}")
            print(f"Maximum acceptable degeneration score: {self.max_degeneration_score}")
            print(f"Maximum acceptable perplexity increase: {self.max_perplexity_increase*100:.1f}%")
            print(f"Output directory: {self.output_dir}")
    
    def setup_directories(self):
        """Create output directories for results and checkpoints."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_short = self.model_name.split("/")[-1]
        run_name = f"adaptive_{model_short}_{timestamp}"
        
        self.run_dir = os.path.join(self.output_dir, run_name)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.visualizations_dir = os.path.join(self.run_dir, "visualizations")
        self.memory_dir = os.path.join(self.run_dir, "memory")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)
        
        if self.verbose:
            print(f"Created output directories in {self.run_dir}")
    
    def evaluate_model(
        self, 
        params: Any, 
        num_samples: int = 5, 
        generate_length: int = 100, 
        show_generations: bool = False,
        use_baseline_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model's performance with comprehensive degeneration detection
        and baseline model comparison.
        
        Args:
            params: Model parameters to evaluate
            num_samples: Number of text samples to evaluate
            generate_length: Length of text to generate
            show_generations: Whether to display generation samples
            use_baseline_comparison: Whether to use baseline model to evaluate quality
            
        Returns:
            Dictionary with evaluation results including perplexity, 
            degeneration scores, and quality metrics
        """
        # Get baseline parameters for comparison if needed
        baseline_params = None
        if use_baseline_comparison and hasattr(self, "baseline_params"):
            baseline_params = self.baseline_params
        elif use_baseline_comparison:
            # Create a new instance with the original model parameters
            try:
                baseline_model = FixedPruningModule(self.model_name)
                baseline_model.load_model()
                baseline_params = baseline_model.model.params
                # Store for future use
                self.baseline_params = baseline_params
            except Exception as e:
                if self.verbose:
                    print(f"Could not load baseline model for comparison: {e}")
                baseline_params = None
        
        # Get evaluation samples
        eval_samples = []
        
        try:
            if hasattr(self.dataset, 'get_evaluation_samples'):
                eval_samples = self.dataset.get_evaluation_samples(num_samples)
            else:
                # Default samples if dataset doesn't provide them
                eval_samples = [
                    "The transformer model processes data through multiple layers of computation.",
                    "Artificial intelligence systems can learn from experience and improve over time.",
                    "The neural network was trained to recognize patterns in complex datasets.",
                    "Language models predict the next token based on previous context.",
                    "The attention mechanism allows the model to focus on relevant parts of the input."
                ][:num_samples]
        except Exception as e:
            print(f"Error getting evaluation samples: {e}")
            # Fallback samples
            eval_samples = [
                "The transformer model processes data through multiple layers of computation.",
                "Artificial intelligence systems can learn from experience and improve over time."
            ]
        
        # Ensure we have enough samples
        if len(eval_samples) < num_samples:
            # Duplicate samples if necessary
            eval_samples = (eval_samples * ((num_samples // len(eval_samples)) + 1))[:num_samples]
        
        results = []
        perplexities = []
        degeneration_scores = []
        quality_scores = []
        
        for sample in eval_samples:
            # Calculate perplexity
            perplexity = self.pruning_module.evaluate_perplexity(params, sample)
            if not np.isnan(perplexity) and not np.isinf(perplexity):
                perplexities.append(perplexity)
            
            # Generate text
            prompt = sample[:30]  # Use beginning of sample as prompt
            generation = self.pruning_module.generate_text(params, prompt, max_length=generate_length)
            
            # Check for degeneration
            degeneration_detected, degeneration_score, degeneration_info = check_for_degeneration(
                generation, 
                repetition_threshold=3, 
                diversity_threshold=0.4
            )
            
            # Store degeneration score
            degeneration_scores.append(degeneration_score)
            
            # Compare with baseline model if available
            baseline_generation = None
            quality_score = 1.0  # Default quality score (neutral)
            
            if baseline_params is not None:
                # Generate text with baseline model
                try:
                    baseline_generation = self.pruning_module.generate_text(
                        baseline_params, prompt, max_length=generate_length
                    )
                    
                    # Use baseline model as quality judge
                    quality_score = self._assess_quality_vs_baseline(baseline_generation, generation)
                    quality_scores.append(quality_score)
                    
                    # Apply baseline-based quality penalty to perplexity
                    # Lower quality = higher perplexity
                    if quality_score < 0.7 and len(perplexities) > 0:  # Only if quality is significantly worse
                        quality_penalty = 1.0 + max(0.0, 0.7 - quality_score) * 2.0  # Up to 2x penalty for very poor quality
                        adjusted_perplexity = perplexities[-1] * quality_penalty
                        perplexities[-1] = adjusted_perplexity
                        
                        if self.verbose:
                            print(f"Applied quality penalty: {perplexities[-1]/quality_penalty:.2f} → "
                                  f"{adjusted_perplexity:.2f} (quality score: {quality_score:.2f})")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"Error in baseline comparison: {e}")
            
            # Apply perplexity penalty if degeneration detected
            if degeneration_detected and len(perplexities) > 0:
                adjusted_perplexity = apply_degeneration_penalty(
                    perplexities[-1], 
                    degeneration_score,
                    max_penalty=1.0
                )
                # Replace the normal perplexity with the penalized one
                perplexities[-1] = adjusted_perplexity
            
            # Display generation if requested
            if show_generations:
                if baseline_generation:
                    # Show side by side comparison
                    display_side_by_side(prompt, baseline_generation, generation)
                else:
                    # Show just the current generation
                    display_generation(prompt, generation)
            
            # Store results
            results.append({
                "prompt": prompt,
                "perplexity": float(perplexity) if not np.isnan(perplexity) and not np.isinf(perplexity) else None,
                "generation": generation,
                "baseline_generation": baseline_generation,
                "degeneration_detected": degeneration_detected,
                "degeneration_score": degeneration_score,
                "degeneration_info": degeneration_info,
                "quality_score": quality_score
            })
        
        # Calculate average metrics
        avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
        avg_degeneration = sum(degeneration_scores) / len(degeneration_scores) if degeneration_scores else 0.0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
        
        # Calculate percentage of degenerated samples
        degeneration_rate = sum(1 for r in results if r["degeneration_detected"]) / len(results)
        
        return {
            "samples": results,
            "average_perplexity": float(avg_perplexity),
            "perplexities": [float(p) for p in perplexities],
            "average_degeneration_score": float(avg_degeneration),
            "average_quality_score": float(avg_quality),
            "degeneration_rate": float(degeneration_rate),
            "baseline_comparison_used": baseline_params is not None,
            "evaluation_time": datetime.now().isoformat()
        }
        
    def _assess_quality_vs_baseline(self, baseline_text: str, optimized_text: str) -> float:
        """
        Compare the quality of generated text against baseline model output.
        Uses heuristics to detect quality differences.
        
        Args:
            baseline_text: Text generated by baseline model
            optimized_text: Text generated by optimized model
            
        Returns:
            Quality score (0.0-1.0) where:
            - 1.0 means equal or better quality than baseline
            - 0.0 means significantly worse quality than baseline
        """
        # If either text is missing, can't compare
        if not baseline_text or not optimized_text:
            return 1.0
            
        # Clean texts for comparison
        baseline_text = ''.join(c for c in baseline_text if c.isprintable() or c in [' ', '\n'])
        optimized_text = ''.join(c for c in optimized_text if c.isprintable() or c in [' ', '\n'])
        
        # 1. Compare word counts (penalize if too short)
        baseline_words = baseline_text.split()
        optimized_words = optimized_text.split()
        
        baseline_word_count = len(baseline_words)
        optimized_word_count = len(optimized_words)
        
        # Penalize if significantly shorter
        length_ratio = min(1.0, optimized_word_count / max(1, baseline_word_count))
        if length_ratio < 0.7:  # Less than 70% of baseline length
            length_penalty = 0.3 + 0.7 * length_ratio  # Linear penalty between 0.3-1.0
        else:
            length_penalty = 1.0  # No penalty
            
        # 2. Compare word diversity
        baseline_unique = len(set(baseline_words))
        optimized_unique = len(set(optimized_words))
        
        baseline_diversity = baseline_unique / max(1, baseline_word_count)
        optimized_diversity = optimized_unique / max(1, optimized_word_count)
        
        # Penalize if diversity is much lower
        diversity_ratio = min(1.0, optimized_diversity / max(0.01, baseline_diversity))
        if diversity_ratio < 0.8:  # Less than 80% of baseline diversity
            diversity_penalty = 0.5 + 0.5 * diversity_ratio  # Linear penalty between 0.5-1.0
        else:
            diversity_penalty = 1.0  # No penalty
        
        # 3. Check for non-word tokens (e.g., unicode errors, broken text)
        # Count tokens that don't look like words
        import re
        word_pattern = re.compile(r'^[a-zA-Z0-9\'-]+$')
        
        baseline_non_words = sum(1 for w in baseline_words if w and not word_pattern.match(w))
        optimized_non_words = sum(1 for w in optimized_words if w and not word_pattern.match(w))
        
        baseline_non_word_ratio = baseline_non_words / max(1, baseline_word_count)
        optimized_non_word_ratio = optimized_non_words / max(1, optimized_word_count)
        
        # Penalize if more non-words than baseline
        if optimized_non_word_ratio > baseline_non_word_ratio + 0.1:  # Allow some leeway
            non_word_penalty = 0.7  # Fixed penalty for excess non-words
        else:
            non_word_penalty = 1.0  # No penalty
        
        # 4. Check for completeness of sentences
        # Simple heuristic: Count ending punctuation vs. number of lines
        baseline_sentences = len(re.findall(r'[.!?]', baseline_text))
        optimized_sentences = len(re.findall(r'[.!?]', optimized_text))
        
        # Penalize if significantly fewer completed sentences
        sentence_ratio = min(1.0, optimized_sentences / max(1, baseline_sentences))
        if sentence_ratio < 0.7:  # Less than 70% of baseline sentences
            sentence_penalty = 0.7 + 0.3 * sentence_ratio  # Linear penalty between 0.7-1.0
        else:
            sentence_penalty = 1.0  # No penalty
            
        # 5. Check if the optimized text has more repetition
        # Simple repetition detection
        def count_repeated_phrases(text, phrase_length=3):
            words = text.split()
            if len(words) < phrase_length * 2:
                return 0
                
            phrases = [' '.join(words[i:i+phrase_length]) for i in range(len(words)-phrase_length+1)]
            phrase_counts = {}
            for phrase in phrases:
                if phrase not in phrase_counts:
                    phrase_counts[phrase] = 0
                phrase_counts[phrase] += 1
                
            # Count phrases that repeat 3+ times
            return sum(1 for count in phrase_counts.values() if count >= 3)
            
        baseline_repetitions = count_repeated_phrases(baseline_text)
        optimized_repetitions = count_repeated_phrases(optimized_text)
        
        # Penalize if significantly more repetitions
        if optimized_repetitions > baseline_repetitions + 1:  # Allow one more repetition
            repetition_penalty = 0.7  # Fixed penalty for excess repetition
        else:
            repetition_penalty = 1.0  # No penalty
            
        # Combine all penalties
        # Prioritize: length > repetition > diversity > sentences > non-words
        quality_score = (
            0.35 * length_penalty +    # 35% weight to length
            0.25 * repetition_penalty + # 25% weight to repetition
            0.20 * diversity_penalty +  # 20% weight to diversity
            0.15 * sentence_penalty +   # 15% weight to sentences
            0.05 * non_word_penalty     # 5% weight to non-words
        )
        
        return quality_score
    
    def train_model(
        self, 
        params: Any, 
        num_steps: int, 
        head_lr_manager: Optional[HeadLRManager] = None, 
        eval_every: int = 50
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the model for a specified number of steps.
        
        Args:
            params: Initial model parameters
            num_steps: Number of training steps
            head_lr_manager: Optional HeadLRManager for differential learning rates
            eval_every: Evaluate model every N steps
            
        Returns:
            Tuple of (trained_params, training_metrics)
        """
        if self.verbose:
            print(f"Training model for {num_steps} steps...")
        
        # Initialize trainer
        trainer = FineTuner(
            pruning_module=self.pruning_module,
            dataset=self.dataset,
            learning_rate=self.learning_rate,
            head_lr_manager=head_lr_manager
        )
        
        # Set initial parameters
        if params is not None:
            trainer.set_params(params)
        
        # Track metrics
        train_losses = []
        perplexities = []
        degeneration_scores = []
        
        # Training loop
        from tqdm import tqdm
        for step in tqdm(range(num_steps)):
            # Train for one step
            train_loss = trainer.train_step()
            train_losses.append(float(train_loss))
            
            # Log metrics
            self.metrics_logger.log({
                "phase": "training",
                "step": step,
                "train_loss": float(train_loss)
            })
            
            # Evaluate periodically
            if step % eval_every == 0 or step == num_steps - 1:
                # Get current parameters
                current_params = trainer.get_params()
                
                # Evaluate model
                eval_results = self.evaluate_model(
                    params=current_params,
                    num_samples=2,  # Use a small number for quick evaluation
                    generate_length=50
                )
                
                # Store metrics
                perplexities.append(eval_results["average_perplexity"])
                degeneration_scores.append(eval_results["average_degeneration_score"])
                
                # Log evaluation metrics
                self.metrics_logger.log({
                    "phase": "training_evaluation",
                    "step": step,
                    "perplexity": eval_results["average_perplexity"],
                    "degeneration_score": eval_results["average_degeneration_score"],
                    "degeneration_rate": eval_results["degeneration_rate"]
                })
                
                # Print progress
                if self.verbose:
                    print(f"Step {step}: Loss = {train_loss:.4f}, Perplexity = {eval_results['average_perplexity']:.4f}")
        
        # Return final parameters and metrics
        final_params = trainer.get_params()
        
        training_metrics = {
            "train_losses": train_losses,
            "perplexities": perplexities,
            "degeneration_scores": degeneration_scores,
            "final_loss": train_losses[-1] if train_losses else None,
            "final_perplexity": perplexities[-1] if perplexities else None,
            "final_degeneration_score": degeneration_scores[-1] if degeneration_scores else None
        }
        
        return final_params, training_metrics
    
    def select_pruning_strategy(self) -> str:
        """
        Select a pruning strategy based on past performance.
        Uses a weighted probability distribution.
        
        Returns:
            Name of the selected pruning strategy
        """
        strategies = list(self.strategy_ratings["pruning"].keys())
        weights = list(self.strategy_ratings["pruning"].values())
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select strategy based on weights
        selected_strategy = random.choices(strategies, probabilities, k=1)[0]
        
        if self.verbose:
            print(f"Selected pruning strategy: {selected_strategy} (probability: {probabilities[strategies.index(selected_strategy)]:.2f})")
        
        return selected_strategy
    
    def select_growth_strategy(self) -> str:
        """
        Select a growth strategy based on past performance.
        Uses a weighted probability distribution.
        
        Returns:
            Name of the selected growth strategy
        """
        strategies = list(self.strategy_ratings["growth"].keys())
        weights = list(self.strategy_ratings["growth"].values())
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select strategy based on weights
        selected_strategy = random.choices(strategies, probabilities, k=1)[0]
        
        if self.verbose:
            print(f"Selected growth strategy: {selected_strategy} (probability: {probabilities[strategies.index(selected_strategy)]:.2f})")
        
        return selected_strategy
    
    def update_strategy_ratings(
        self, 
        phase: str, 
        strategy: str, 
        success: bool, 
        improvement: float
    ):
        """
        Update strategy ratings based on success or failure.
        
        Args:
            phase: 'pruning' or 'growth'
            strategy: Strategy name
            success: Whether the strategy was successful
            improvement: Improvement metric (can be negative)
        """
        if phase not in self.strategy_ratings or strategy not in self.strategy_ratings[phase]:
            return
        
        current_rating = self.strategy_ratings[phase][strategy]
        
        if success:
            # Increase rating based on improvement
            multiplier = 1.0 + max(0.0, min(0.5, improvement))
            new_rating = current_rating * multiplier
        else:
            # Decrease rating
            multiplier = 0.8  # 20% reduction
            new_rating = current_rating * multiplier
        
        # Ensure rating stays in reasonable bounds
        new_rating = max(0.1, min(10.0, new_rating))
        
        # Update rating
        self.strategy_ratings[phase][strategy] = new_rating
        
        if self.verbose:
            print(f"Updated {phase} strategy '{strategy}' rating: {current_rating:.2f} → {new_rating:.2f}")
    
    def prune_model(
        self,
        params: Any,
        pruning_level: float,
        strategy_name: Optional[str] = None
    ) -> Tuple[Any, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Prune the model using the specified or selected strategy.
        
        Args:
            params: Model parameters to prune
            pruning_level: Fraction of heads to prune
            strategy_name: Pruning strategy to use (if None, will select one)
            
        Returns:
            Tuple of (pruned_params, pruned_heads, active_heads)
        """
        # Select strategy if not provided
        if strategy_name is None:
            strategy_name = self.select_pruning_strategy()
        
        if self.verbose:
            print(f"Pruning with strategy '{strategy_name}' at level {pruning_level:.2f}")
        
        # Get pruning strategy
        strategy = get_pruning_strategy(strategy_name, self.pruning_module)
        
        # Calculate head importance
        head_importance = strategy.get_head_importance(params)
        
        # Sort by importance (ascending, least important first)
        head_importance.sort(key=lambda x: x[2])
        
        # Determine active heads before pruning
        active_heads_before = determine_active_heads(self.pruning_module, params)
        
        # Calculate number of heads to prune
        num_active = len(active_heads_before)
        num_to_prune = max(1, int(num_active * pruning_level))
        
        # Select heads to prune (least important first)
        heads_to_prune = [(layer_idx, head_idx) for layer_idx, head_idx, _ in head_importance[:num_to_prune]]
        
        # Prune the selected heads
        pruned_params = params.copy()
        for layer_idx, head_idx in heads_to_prune:
            pruned_params = self.pruning_module.prune_head(pruned_params, layer_idx, head_idx)
        
        # Determine active heads after pruning
        active_heads_after = determine_active_heads(self.pruning_module, pruned_params)
        
        # Log metrics
        self.metrics_logger.log({
            "phase": "pruning",
            "pruning_level": pruning_level,
            "pruning_strategy": strategy_name,
            "heads_before": len(active_heads_before),
            "heads_after": len(active_heads_after),
            "heads_pruned": len(heads_to_prune)
        })
        
        if self.verbose:
            print(f"Pruned {len(heads_to_prune)} heads, {len(active_heads_after)} remaining")
        
        return pruned_params, heads_to_prune, active_heads_after
    
    def grow_model(
        self,
        params: Any,
        active_heads: List[Tuple[int, int]],
        growth_ratio: float,
        strategy_name: Optional[str] = None
    ) -> Tuple[Any, int, List[Tuple[int, int]], List[Tuple[int, int]], Callable]:
        """
        Grow new attention heads in the model.
        
        Args:
            params: Model parameters after pruning
            active_heads: List of active heads after pruning
            growth_ratio: Ratio of pruned heads to grow back
            strategy_name: Growth strategy to use (if None, will select one)
            
        Returns:
            Tuple of (grown_params, num_added, added_heads, active_heads_after, warmup_schedule)
        """
        # Select strategy if not provided
        if strategy_name is None:
            strategy_name = self.select_growth_strategy()
        
        # Calculate growth percentage
        pruned_heads_count = self.total_heads - len(active_heads)
        growth_percentage = growth_ratio * (pruned_heads_count / self.total_heads)
        
        if self.verbose:
            print(f"Growing with strategy '{strategy_name}' at ratio {growth_ratio:.2f} (percentage: {growth_percentage:.2f})")
        
        # Grow attention heads
        grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
            self.pruning_module,
            params=params,
            active_heads=active_heads,
            growth_percentage=growth_percentage,
            strategy=strategy_name,
            initial_scale=0.01,  # Start small
            warmup_steps=100
        )
        
        # Determine active heads after growth
        active_heads_after = determine_active_heads(self.pruning_module, grown_params)
        
        # Log metrics
        self.metrics_logger.log({
            "phase": "growth",
            "growth_ratio": growth_ratio,
            "growth_percentage": growth_percentage,
            "growth_strategy": strategy_name,
            "heads_before": len(active_heads),
            "heads_after": len(active_heads_after),
            "heads_added": added_count
        })
        
        if self.verbose:
            print(f"Added {added_count} heads, now have {len(active_heads_after)} active heads")
        
        return grown_params, added_count, added_heads, active_heads_after, warmup_schedule
    
    def record_successful_transformation(
        self, 
        transformation: Dict[str, Any]
    ):
        """
        Record a successful transformation in memory.
        
        Args:
            transformation: Dictionary describing the transformation
        """
        # Add timestamp
        transformation["timestamp"] = datetime.now().isoformat()
        
        # Add to memory
        self.transformation_memory.append(transformation)
        
        # Trim memory if it exceeds capacity
        if len(self.transformation_memory) > self.memory_capacity:
            self.transformation_memory = self.transformation_memory[-self.memory_capacity:]
        
        # Save memory to disk
        memory_path = os.path.join(self.memory_dir, "transformations.json")
        with open(memory_path, 'w') as f:
            json.dump(self.transformation_memory, f, indent=2)
        
        if self.verbose:
            print(f"Recorded successful transformation in memory (total: {len(self.transformation_memory)})")
    
    def load_transformation_memory(self):
        """Load transformation memory from disk if available."""
        memory_path = os.path.join(self.memory_dir, "transformations.json")
        if os.path.exists(memory_path):
            with open(memory_path, 'r') as f:
                self.transformation_memory = json.load(f)
            
            if self.verbose:
                print(f"Loaded {len(self.transformation_memory)} transformations from memory")
    
    def select_from_memory(self) -> Optional[Dict[str, Any]]:
        """
        Select a transformation from memory to replay.
        
        Returns:
            Selected transformation or None if memory is empty
        """
        if not self.transformation_memory:
            return None
        
        # Select based on success metrics
        weighted_memory = [(t, t.get("improvement", 0.0)) for t in self.transformation_memory]
        
        # Ensure all weights are positive
        min_weight = min(w for _, w in weighted_memory)
        adjusted_weights = [(t, max(0.1, w - min_weight + 0.1)) for t, w in weighted_memory]
        
        # Normalize weights
        total_weight = sum(w for _, w in adjusted_weights)
        normalized_weights = [(t, w / total_weight) for t, w in adjusted_weights]
        
        # Select transformation based on weights
        transformations = [t for t, _ in normalized_weights]
        weights = [w for _, w in normalized_weights]
        
        selected = random.choices(transformations, weights, k=1)[0]
        
        if self.verbose:
            print(f"Selected transformation from memory: {selected.get('description', 'Unknown')}")
        
        return selected
    
    def run_plasticity_cycle(
        self, 
        pruning_level: float = 0.2,
        growth_ratio: float = 0.5,
        training_steps: int = 100,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete plasticity cycle (prune → measure → grow → learn).
        
        Args:
            pruning_level: Fraction of heads to prune
            growth_ratio: Ratio of pruned heads to grow back
            training_steps: Number of training steps after growth
            use_memory: Whether to use transformation memory
            
        Returns:
            Dictionary with cycle results and metrics
        """
        cycle_start_time = datetime.now()
        
        if self.verbose:
            print("\n=== Starting Neural Plasticity Cycle ===\n")
        
        # Get current model state
        params = self.current_params
        active_heads = self.current_active_heads
        
        # Store initial metrics
        initial_head_count = len(active_heads)
        
        # 1. Initial Evaluation
        if self.verbose:
            print("Evaluating model before cycle...")
        
        initial_eval = self.evaluate_model(
            params=params,
            num_samples=5,
            generate_length=100,
            show_generations=self.verbose
        )
        
        initial_perplexity = initial_eval["average_perplexity"]
        initial_degeneration = initial_eval["average_degeneration_score"]
        
        if self.verbose:
            print(f"Initial metrics - Perplexity: {initial_perplexity:.2f}, " +
                  f"Degeneration: {initial_degeneration:.2f}, " +
                  f"Active heads: {initial_head_count}")
        
        # Check if we should use a transformation from memory
        selected_transformation = None
        if use_memory and random.random() < 0.7:  # 70% chance to use memory if enabled
            selected_transformation = self.select_from_memory()
        
        # 2. Pruning Phase
        if selected_transformation:
            # Use pruning strategy from memory
            pruning_strategy = selected_transformation.get("pruning_strategy", self.select_pruning_strategy())
            pruning_level = selected_transformation.get("pruning_level", pruning_level)
        else:
            # Select new pruning strategy
            pruning_strategy = self.select_pruning_strategy()
        
        if self.verbose:
            print(f"\n--- Pruning Phase (strategy: {pruning_strategy}, level: {pruning_level:.2f}) ---\n")
        
        pruned_params, pruned_heads, active_heads_after_pruning = self.prune_model(
            params=params,
            pruning_level=pruning_level,
            strategy_name=pruning_strategy
        )
        
        # 3. Measurement Phase
        if self.verbose:
            print("\n--- Measurement Phase ---\n")
        
        pruned_eval = self.evaluate_model(
            params=pruned_params,
            num_samples=5,
            generate_length=100,
            show_generations=self.verbose
        )
        
        pruned_perplexity = pruned_eval["average_perplexity"]
        pruned_degeneration = pruned_eval["average_degeneration_score"]
        
        # Calculate perplexity change
        perplexity_change = (pruned_perplexity - initial_perplexity) / initial_perplexity
        
        if self.verbose:
            print(f"After pruning - Perplexity: {pruned_perplexity:.2f} ({perplexity_change*100:+.1f}%), " +
                  f"Degeneration: {pruned_degeneration:.2f}, " +
                  f"Active heads: {len(active_heads_after_pruning)}")
        
        # 4. Growth Phase
        if selected_transformation:
            # Use growth strategy from memory
            growth_strategy = selected_transformation.get("growth_strategy", self.select_growth_strategy())
            growth_ratio = selected_transformation.get("growth_ratio", growth_ratio)
        else:
            # Select new growth strategy
            growth_strategy = self.select_growth_strategy()
        
        if self.verbose:
            print(f"\n--- Growth Phase (strategy: {growth_strategy}, ratio: {growth_ratio:.2f}) ---\n")
        
        grown_params, added_count, added_heads, active_heads_after_growth, warmup_schedule = self.grow_model(
            params=pruned_params,
            active_heads=active_heads_after_pruning,
            growth_ratio=growth_ratio,
            strategy_name=growth_strategy
        )
        
        # Evaluate after growth but before learning
        grown_eval = self.evaluate_model(
            params=grown_params,
            num_samples=5,
            generate_length=100,
            show_generations=self.verbose
        )
        
        grown_perplexity = grown_eval["average_perplexity"]
        grown_degeneration = grown_eval["average_degeneration_score"]
        
        if self.verbose:
            print(f"After growth - Perplexity: {grown_perplexity:.2f}, " +
                  f"Degeneration: {grown_degeneration:.2f}, " +
                  f"Active heads: {len(active_heads_after_growth)}")
        
        # 5. Learning Phase
        if self.verbose:
            print("\n--- Learning Phase ---\n")
        
        # Create head learning rate manager for differential learning
        head_lr_manager = HeadLRManager(
            base_lr=self.learning_rate,
            new_head_multiplier=5.0,  # Learn faster in new heads
            new_heads=added_heads
        )
        
        learned_params, training_metrics = self.train_model(
            params=grown_params,
            num_steps=training_steps,
            head_lr_manager=head_lr_manager,
            eval_every=max(10, training_steps // 10)
        )
        
        # 6. Final Evaluation
        if self.verbose:
            print("\n--- Final Evaluation ---\n")
        
        final_eval = self.evaluate_model(
            params=learned_params,
            num_samples=5,
            generate_length=100,
            show_generations=self.verbose
        )
        
        final_perplexity = final_eval["average_perplexity"]
        final_degeneration = final_eval["average_degeneration_score"]
        final_head_count = len(active_heads_after_growth)  # Should be the same after learning
        
        # Calculate metrics
        perplexity_improvement = (initial_perplexity - final_perplexity) / initial_perplexity
        head_reduction_pct = (initial_head_count - final_head_count) / initial_head_count
        
        # Calculate efficiency (perplexity/head ratio improvement)
        initial_efficiency = initial_perplexity / initial_head_count
        final_efficiency = final_perplexity / final_head_count
        efficiency_improvement = (initial_efficiency - final_efficiency) / initial_efficiency
        
        if self.verbose:
            print(f"\n=== Cycle Results ===")
            print(f"Perplexity: {initial_perplexity:.2f} → {final_perplexity:.2f} ({perplexity_improvement*100:+.1f}%)")
            print(f"Degeneration: {initial_degeneration:.2f} → {final_degeneration:.2f}")
            print(f"Active heads: {initial_head_count} → {final_head_count} ({head_reduction_pct*100:.1f}% reduction)")
            print(f"Efficiency improvement: {efficiency_improvement*100:+.1f}%")
        
        # Determine if this cycle was successful
        cycle_success = (
            final_perplexity <= initial_perplexity * (1.0 + self.max_perplexity_increase) and 
            final_degeneration <= self.max_degeneration_score
        )
        
        # Update strategy ratings based on results
        self.update_strategy_ratings(
            phase="pruning",
            strategy=pruning_strategy,
            success=cycle_success,
            improvement=perplexity_improvement
        )
        
        self.update_strategy_ratings(
            phase="growth",
            strategy=growth_strategy,
            success=cycle_success,
            improvement=perplexity_improvement
        )
        
        # If successful, record this transformation
        if cycle_success and not selected_transformation:
            # This was a new successful transformation, record it
            transformation = {
                "description": f"Prune {pruning_level:.2f} with {pruning_strategy}, grow {growth_ratio:.2f} with {growth_strategy}",
                "pruning_strategy": pruning_strategy,
                "pruning_level": pruning_level,
                "growth_strategy": growth_strategy,
                "growth_ratio": growth_ratio,
                "perplexity_improvement": perplexity_improvement,
                "head_reduction": head_reduction_pct,
                "efficiency_improvement": efficiency_improvement,
                "degeneration_change": final_degeneration - initial_degeneration,
                "improvement": perplexity_improvement + efficiency_improvement * 0.5  # Combined score
            }
            
            self.record_successful_transformation(transformation)
        
        # Update current model state if cycle was successful
        if cycle_success:
            self.current_params = learned_params
            self.current_active_heads = active_heads_after_growth
            self.current_perplexity = final_perplexity
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoints_dir, f"model_cycle_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            with open(checkpoint_path, 'wb') as f:
                import pickle
                pickle.dump(learned_params, f)
                
            if self.verbose:
                print(f"Cycle successful! Updated model state and saved checkpoint.")
        else:
            if self.verbose:
                print(f"Cycle did not improve model quality. Reverting to previous state.")
        
        # Compile and return cycle results
        cycle_results = {
            "cycle_time": (datetime.now() - cycle_start_time).total_seconds(),
            "success": cycle_success,
            "replay_from_memory": selected_transformation is not None,
            
            # Strategy information
            "pruning_strategy": pruning_strategy,
            "pruning_level": pruning_level,
            "growth_strategy": growth_strategy,
            "growth_ratio": growth_ratio,
            
            # Metrics at each stage
            "initial": {
                "perplexity": initial_perplexity,
                "degeneration": initial_degeneration,
                "head_count": initial_head_count
            },
            "pruned": {
                "perplexity": pruned_perplexity,
                "degeneration": pruned_degeneration,
                "head_count": len(active_heads_after_pruning)
            },
            "grown": {
                "perplexity": grown_perplexity,
                "degeneration": grown_degeneration,
                "head_count": len(active_heads_after_growth)
            },
            "final": {
                "perplexity": final_perplexity,
                "degeneration": final_degeneration,
                "head_count": final_head_count
            },
            
            # Improvement metrics
            "perplexity_improvement": perplexity_improvement,
            "head_reduction": head_reduction_pct,
            "efficiency_improvement": efficiency_improvement,
            
            # Training details
            "training_steps": training_steps,
            "final_loss": training_metrics.get("final_loss")
        }
        
        # Log cycle results
        self.metrics_logger.log({
            "phase": "cycle_complete",
            "timestamp": datetime.now().isoformat(),
            "results": cycle_results
        })
        
        return cycle_results
    
    def run_adaptive_optimization(
        self, 
        max_cycles: int = 10,
        initial_pruning_level: float = 0.2,
        initial_growth_ratio: float = 0.5,
        initial_training_steps: int = 100,
        patience: int = 3,
        min_improvement: float = 0.01,  # 1% improvement threshold
        memory_usage_probability: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run adaptive optimization through multiple plasticity cycles.
        
        Args:
            max_cycles: Maximum number of plasticity cycles to run
            initial_pruning_level: Starting pruning level
            initial_growth_ratio: Starting growth ratio
            initial_training_steps: Starting training steps per cycle
            patience: Number of cycles without improvement before stopping
            min_improvement: Minimum improvement to count as progress
            memory_usage_probability: Probability of using memory in a cycle
            
        Returns:
            Dictionary with optimization results and metrics
        """
        optimization_start_time = datetime.now()
        
        if self.verbose:
            print("\n=== Starting Adaptive Optimization ===\n")
            print(f"Max cycles: {max_cycles}")
            print(f"Initial pruning level: {initial_pruning_level:.2f}")
            print(f"Initial growth ratio: {initial_growth_ratio:.2f}")
            print(f"Initial training steps: {initial_training_steps}")
            print(f"Early stopping patience: {patience}")
        
        # Load transformation memory if available
        self.load_transformation_memory()
        
        # Establish baseline
        if self.baseline_evaluation is None:
            if self.verbose:
                print("\n--- Establishing Baseline ---\n")
                
            self.baseline_evaluation = self.evaluate_model(
                params=self.current_params,
                num_samples=10,  # More samples for baseline
                generate_length=100,
                show_generations=self.verbose
            )
            
            self.current_perplexity = self.baseline_evaluation["average_perplexity"]
            
            if self.verbose:
                print(f"Baseline - Perplexity: {self.current_perplexity:.2f}, " +
                      f"Degeneration: {self.baseline_evaluation['average_degeneration_score']:.2f}, " +
                      f"Active heads: {len(self.current_active_heads)}")
        
        # Store best metrics
        best_perplexity = self.current_perplexity
        best_efficiency = self.current_perplexity / len(self.current_active_heads)
        best_params = self.current_params
        best_active_heads = self.current_active_heads.copy()
        
        # Track cycles without improvement
        cycles_without_improvement = 0
        
        # Cycle results
        all_cycle_results = []
        
        # Current parameters for adaptive adjustment
        pruning_level = initial_pruning_level
        growth_ratio = initial_growth_ratio
        training_steps = initial_training_steps
        
        for cycle in range(max_cycles):
            if self.verbose:
                print(f"\n=== Adaptive Optimization Cycle {cycle+1}/{max_cycles} ===\n")
                
            # Adjust parameters based on previous cycles
            if cycle > 0 and all_cycle_results:
                last_result = all_cycle_results[-1]
                
                if last_result["success"]:
                    # Successful cycle - potentially increase pruning
                    if last_result["perplexity_improvement"] > 0.05:
                        # Very successful - be more aggressive
                        pruning_level = min(0.5, pruning_level * 1.2)  # Increase pruning up to 50%
                        training_steps = int(training_steps * 0.9)  # Reduce training time
                    else:
                        # Moderately successful - be slightly more aggressive
                        pruning_level = min(0.4, pruning_level * 1.1)
                else:
                    # Failed cycle - be more conservative
                    pruning_level = max(0.05, pruning_level * 0.8)  # Reduce pruning but keep at least 5%
                    growth_ratio = min(0.9, growth_ratio * 1.2)  # Grow back more heads
                    training_steps = int(training_steps * 1.2)  # Train longer
                    
                if self.verbose:
                    print(f"Adjusted parameters - Pruning: {pruning_level:.2f}, Growth: {growth_ratio:.2f}, Training: {training_steps}")
            
            # Run plasticity cycle
            use_memory = random.random() < memory_usage_probability and self.transformation_memory
            cycle_result = self.run_plasticity_cycle(
                pruning_level=pruning_level,
                growth_ratio=growth_ratio,
                training_steps=training_steps,
                use_memory=use_memory
            )
            
            all_cycle_results.append(cycle_result)
            
            # Check for improvement
            current_perplexity = cycle_result["final"]["perplexity"]
            current_head_count = cycle_result["final"]["head_count"]
            current_efficiency = current_perplexity / current_head_count
            
            perplexity_improvement = (best_perplexity - current_perplexity) / best_perplexity
            efficiency_improvement = (best_efficiency - current_efficiency) / best_efficiency
            
            # Update best metrics if improved
            if (perplexity_improvement > min_improvement or 
                (perplexity_improvement > -self.max_perplexity_increase and efficiency_improvement > min_improvement)):
                # Improved either perplexity or efficiency without hurting perplexity too much
                best_perplexity = current_perplexity
                best_efficiency = current_efficiency
                best_params = self.current_params.copy()
                best_active_heads = self.current_active_heads.copy()
                
                cycles_without_improvement = 0
                
                if self.verbose:
                    print(f"New best model! Perplexity: {best_perplexity:.2f}, Efficiency: {best_efficiency:.6f}")
                
                # Save best model
                best_path = os.path.join(self.checkpoints_dir, "model_best.pth")
                with open(best_path, 'wb') as f:
                    import pickle
                    pickle.dump(best_params, f)
            else:
                # No improvement
                cycles_without_improvement += 1
                
                if self.verbose:
                    print(f"No improvement for {cycles_without_improvement} cycles")
                
                # Check early stopping
                if cycles_without_improvement >= patience:
                    if self.verbose:
                        print(f"Early stopping after {cycle+1} cycles (no improvement for {patience} cycles)")
                    break
        
        # After all cycles, restore best model
        self.current_params = best_params
        self.current_active_heads = best_active_heads
        self.current_perplexity = best_perplexity
        
        # Final evaluation
        final_evaluation = self.evaluate_model(
            params=self.current_params,
            num_samples=10,  # More samples for final evaluation
            generate_length=100,
            show_generations=self.verbose
        )
        
        # Calculate overall improvements
        baseline_perplexity = self.baseline_evaluation["average_perplexity"]
        baseline_head_count = self.total_heads  # Assuming we started with all heads
        baseline_efficiency = baseline_perplexity / baseline_head_count
        
        final_perplexity = final_evaluation["average_perplexity"]
        final_head_count = len(self.current_active_heads)
        final_efficiency = final_perplexity / final_head_count
        
        overall_perplexity_improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity
        overall_head_reduction = (baseline_head_count - final_head_count) / baseline_head_count
        overall_efficiency_improvement = (baseline_efficiency - final_efficiency) / baseline_efficiency
        
        # Print final summary
        if self.verbose:
            print("\n=== Adaptive Optimization Complete ===")
            print(f"Total cycles: {len(all_cycle_results)}")
            print(f"Successful cycles: {sum(1 for r in all_cycle_results if r['success'])}")
            print(f"Final metrics:")
            print(f"  Perplexity: {baseline_perplexity:.2f} → {final_perplexity:.2f} ({overall_perplexity_improvement*100:+.1f}%)")
            print(f"  Active heads: {baseline_head_count} → {final_head_count} ({overall_head_reduction*100:.1f}% reduction)")
            print(f"  Efficiency: {baseline_efficiency:.6f} → {final_efficiency:.6f} ({overall_efficiency_improvement*100:+.1f}%)")
            print(f"  Degeneration score: {final_evaluation['average_degeneration_score']:.2f}")
        
        # Compile and return optimization results
        optimization_results = {
            "optimization_time": (datetime.now() - optimization_start_time).total_seconds(),
            "cycles_completed": len(all_cycle_results),
            "successful_cycles": sum(1 for r in all_cycle_results if r["success"]),
            "early_stopped": cycles_without_improvement >= patience,
            
            # Baseline metrics
            "baseline": {
                "perplexity": baseline_perplexity,
                "head_count": baseline_head_count,
                "efficiency": baseline_efficiency,
                "degeneration": self.baseline_evaluation["average_degeneration_score"]
            },
            
            # Final metrics
            "final": {
                "perplexity": final_perplexity,
                "head_count": final_head_count,
                "efficiency": final_efficiency,
                "degeneration": final_evaluation["average_degeneration_score"]
            },
            
            # Improvements
            "perplexity_improvement": overall_perplexity_improvement,
            "head_reduction": overall_head_reduction,
            "efficiency_improvement": overall_efficiency_improvement,
            
            # Cycle details
            "cycle_results": all_cycle_results,
            
            # Strategy ratings
            "final_strategy_ratings": self.strategy_ratings,
            
            # Memory size
            "transformation_memory_size": len(self.transformation_memory)
        }
        
        # Save optimization results
        results_path = os.path.join(self.run_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            # Convert any numpy values to Python types
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            json.dump(optimization_results, f, indent=2, cls=NumpyEncoder)
        
        return optimization_results
    
    def display_model_comparison(self, prompts=None, num_prompts=3):
        """
        Display side-by-side comparison between baseline and current model.
        
        Args:
            prompts: Optional list of prompts to use
            num_prompts: Number of prompts to use if not provided
        """
        # Get baseline parameters if available
        baseline_params = None
        if hasattr(self, "baseline_params"):
            baseline_params = self.baseline_params
        else:
            # Try to recreate baseline
            try:
                original_model = FixedPruningModule(self.model_name)
                original_model.load_model()
                baseline_params = original_model.model.params
            except Exception as e:
                print(f"Could not load baseline parameters: {e}")
                baseline_params = self.pruning_module.model.params
        
        # Use default prompts if not provided
        if prompts is None:
            from utils.pruning.inference_utils import get_test_prompts
            prompts = get_test_prompts(category="transformer", length="medium")[:num_prompts]
        
        print("\n=== Model Comparison ===\n")
        
        for prompt in prompts:
            try:
                # Generate with baseline model
                baseline_generation = self.pruning_module.generate_text(
                    params=baseline_params,
                    prompt=prompt,
                    max_length=100
                )
                
                # Generate with current model
                current_generation = self.pruning_module.generate_text(
                    params=self.current_params,
                    prompt=prompt,
                    max_length=100
                )
                
                # Display side by side comparison
                display_side_by_side(prompt, baseline_generation, current_generation)
                
            except Exception as e:
                print(f"Error generating comparison: {e}")
                
        # Print summary metrics
        baseline_head_count = self.total_heads
        current_head_count = len(self.current_active_heads)
        head_reduction_pct = (baseline_head_count - current_head_count) / baseline_head_count * 100
        
        print("\n=== Model Metrics ===")
        print(f"Baseline active heads: {baseline_head_count}")
        print(f"Current active heads: {current_head_count} ({head_reduction_pct:.1f}% reduction)")
        
        if hasattr(self, "baseline_evaluation") and self.current_perplexity is not None:
            baseline_perplexity = self.baseline_evaluation["average_perplexity"]
            perplexity_change_pct = (self.current_perplexity - baseline_perplexity) / baseline_perplexity * 100
            
            print(f"Baseline perplexity: {baseline_perplexity:.2f}")
            print(f"Current perplexity: {self.current_perplexity:.2f} ({perplexity_change_pct:+.1f}%)")
            
            # Efficiency metrics
            baseline_efficiency = baseline_perplexity / baseline_head_count
            current_efficiency = self.current_perplexity / current_head_count
            efficiency_change_pct = (current_efficiency - baseline_efficiency) / baseline_efficiency * 100
            
            print(f"Efficiency: {baseline_efficiency:.6f} → {current_efficiency:.6f} perplexity/head ({efficiency_change_pct:+.1f}%)")


def run_adaptive_system(
    model_name: str,
    dataset: Any,
    output_dir: str = "./output/adaptive_plasticity",
    max_cycles: int = 10,
    device: str = "cuda",
    initial_pruning_level: float = 0.2,
    initial_growth_ratio: float = 0.5,
    initial_training_steps: int = 100,
    patience: int = 3,
    verbose: bool = True
):
    """
    Run the adaptive plasticity system on a model.
    
    Args:
        model_name: Name or path of the model to optimize
        dataset: Dataset to use for training and evaluation
        output_dir: Directory to save results and checkpoints
        max_cycles: Maximum number of plasticity cycles to run
        device: Device to use ('cuda' or 'cpu')
        initial_pruning_level: Starting pruning level
        initial_growth_ratio: Starting growth ratio
        initial_training_steps: Starting training steps per cycle
        patience: Number of cycles without improvement before stopping
        verbose: Whether to print detailed information
    
    Returns:
        AdaptivePlasticitySystem instance after optimization
    """
    # Create adaptive system
    system = AdaptivePlasticitySystem(
        model_name=model_name,
        dataset=dataset,
        output_dir=output_dir,
        device=device,
        verbose=verbose
    )
    
    # Run adaptive optimization
    optimization_results = system.run_adaptive_optimization(
        max_cycles=max_cycles,
        initial_pruning_level=initial_pruning_level,
        initial_growth_ratio=initial_growth_ratio,
        initial_training_steps=initial_training_steps,
        patience=patience
    )
    
    # Display comparison between baseline and optimized model
    system.display_model_comparison()
    
    return system