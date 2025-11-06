import json
import os
import torch
import numpy as np
from sentinel.models.agency_specialization import AgencySpecialization

class SpecializationRegistry:
    """
    Registry for task-specific agency specialization patterns.
    
    This class enables:
    1. Loading pre-configured specialization patterns for different task types
    2. Automatic task detection based on input characteristics
    3. Runtime adaptation of agency profiles
    4. Persistence of effective patterns
    
    Based on the empirical validation showing different tasks benefit 
    from different specialization patterns.
    """
    
    def __init__(self, model=None, debug=False):
        """
        Initialize the specialization registry.
        
        Args:
            model: Optional AdaptiveCausalLmWrapper instance
            debug: Whether to print debug information
        """
        self.debug = debug
        self.model = model
        
        # Default specialization patterns based on model research
        self.specialization_patterns = {
            "general": {},  # Will be populated on first access
            "summarization": {
                "description": "Optimized for text summarization tasks",
                "pattern": {}  # Will be populated dynamically
            },
            "qa": {
                "description": "Optimized for question answering tasks",
                "pattern": {}
            },
            "code": {
                "description": "Optimized for code generation tasks",
                "pattern": {}
            },
            "creative": {
                "description": "Optimized for creative writing tasks",
                "pattern": {}
            }
        }
        
        # Initialize with default patterns
        self._initialize_default_patterns()
        
        # Task detection keywords
        self.task_keywords = {
            "summarization": ["summarize", "summary", "tldr", "overview", "main points", "briefly", "condense"],
            "qa": ["question", "answer", "why", "what", "how", "when", "where", "who", "explain", "reason"],
            "code": ["code", "function", "class", "program", "algorithm", "bug", "implement", "syntax"],
            "creative": ["story", "creative", "write", "novel", "imagine", "fiction", "poem", "dialogue"]
        }
        
        # Performance tracking for patterns
        self.pattern_performance = {}
        
    def _initialize_default_patterns(self):
        """Initialize default specialization patterns based on research findings."""
        # Abstract pattern assignments into 4 roles with different distributions for tasks
        specialization_roles = [
            "pattern_recognition",
            "logical_reasoning",
            "memory_context", 
            "creative_synthesis"
        ]
        
        # For each task, create specialization distributions
        task_distributions = {
            "general": [0.35, 0.25, 0.25, 0.15],
            "summarization": [0.5, 0.3, 0.1, 0.1],
            "qa": [0.2, 0.5, 0.2, 0.1],
            "code": [0.3, 0.4, 0.2, 0.1],
            "creative": [0.1, 0.2, 0.2, 0.5]
        }
        
        # Initialize patterns
        if self.model is not None:
            # Extract model dimensions
            num_layers = len(self.model.transformer.blocks)
            num_heads = self.model.transformer.blocks[0].attn.num_heads
            
            for task_name, distribution in task_distributions.items():
                # Generate head indices
                head_indices = [(l, h) for l in range(num_layers) for h in range(num_heads)]
                total_heads = len(head_indices)
                
                # Calculate allocations
                allocations = []
                for i, percentage in enumerate(distribution):
                    count = int(total_heads * percentage)
                    allocations.extend([specialization_roles[i]] * count)
                
                # Fill any remaining slots with balanced allocations
                remaining = total_heads - len(allocations)
                if remaining > 0:
                    for i in range(remaining):
                        allocations.append(specialization_roles[i % len(specialization_roles)])
                
                # Shuffle the allocations for balanced distribution
                np.random.seed(42 + hash(task_name) % 1000)  # Stable but different per task
                np.random.shuffle(allocations)
                
                # Create the pattern
                pattern = {}
                for i, head_idx in enumerate(head_indices):
                    pattern[head_idx] = allocations[i]
                
                # Store in specialization patterns
                self.specialization_patterns[task_name]["pattern"] = pattern
        else:
            # Placeholder for when model isn't available
            for task_name in task_distributions:
                self.specialization_patterns[task_name]["pattern"] = {}
    
    def get_specialization_pattern(self, pattern_name="general"):
        """
        Get a specialization pattern by name.
        
        Args:
            pattern_name: Name of the pattern to retrieve
            
        Returns:
            Dictionary mapping (layer_idx, head_idx) to specialization
        """
        if pattern_name not in self.specialization_patterns:
            if self.debug:
                print(f"Pattern '{pattern_name}' not found, using 'general'")
            pattern_name = "general"
        
        pattern = self.specialization_patterns[pattern_name]["pattern"]
        
        # Re-initialize if empty and model is available
        if not pattern and self.model is not None:
            self._initialize_default_patterns()
            pattern = self.specialization_patterns[pattern_name]["pattern"]
        
        return pattern
    
    def set_model(self, model):
        """
        Update the model reference and initialize patterns.
        
        Args:
            model: AdaptiveCausalLmWrapper instance
        """
        self.model = model
        self._initialize_default_patterns()
    
    def detect_task(self, prompt):
        """
        Automatically detect task type from input prompt.
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            Detected task name or "general" if uncertain
        """
        prompt_lower = prompt.lower()
        
        # Count keyword occurrences for each task
        task_scores = {task: 0 for task in self.task_keywords}
        
        for task, keywords in self.task_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    task_scores[task] += 1
        
        # Normalize by keyword count
        for task in task_scores:
            task_scores[task] /= len(self.task_keywords[task])
            
        # Get the highest scoring task
        max_score = max(task_scores.values())
        max_tasks = [task for task, score in task_scores.items() if score == max_score]
        
        # If we have a clear winner with score > 0
        if len(max_tasks) == 1 and max_score > 0:
            return max_tasks[0]
            
        # Default to general
        return "general"
    
    def apply_specialization(self, prompt=None, task=None):
        """
        Apply appropriate specialization pattern based on prompt or specified task.
        
        Args:
            prompt: Text prompt to analyze
            task: Explicit task name to use (overrides detection)
            
        Returns:
            Specialization pattern that was applied
        """
        if self.model is None:
            if self.debug:
                print("Cannot apply specialization without model reference")
            return None
        
        # Determine task
        if task is not None:
            task_name = task
        elif prompt is not None:
            task_name = self.detect_task(prompt)
        else:
            task_name = "general"
            
        # Get specialization pattern
        pattern = self.get_specialization_pattern(task_name)
        
        # Apply to model
        specialization = AgencySpecialization(self.model, debug=self.debug)
        specialization.assign_specializations(pattern)
        
        # Apply default state settings based on task
        if task_name == "summarization":
            specialization.apply_task_specialization("summarization")
        elif task_name == "qa":
            specialization.apply_task_specialization("question_answering")
        elif task_name == "code":
            specialization.apply_task_specialization("code_generation")
        elif task_name == "creative":
            specialization.apply_task_specialization("creative_writing")
            
        if self.debug:
            print(f"Applied specialization pattern for task: {task_name}")
            
        return {
            "task": task_name,
            "pattern": pattern
        }
    
    def track_performance(self, task, metrics):
        """
        Track performance of a specialization pattern.
        
        Args:
            task: The task name
            metrics: Dictionary of performance metrics
        """
        if task not in self.pattern_performance:
            self.pattern_performance[task] = []
            
        self.pattern_performance[task].append(metrics)
        
        if self.debug:
            print(f"Tracked performance for {task}: {metrics}")
    
    def get_default_pattern(self):
        """Get the default pattern name."""
        return "general"
    
    def save_registry(self, file_path):
        """
        Save registry patterns and performance data to a file.
        
        Args:
            file_path: Path to save the registry data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert patterns to serializable format
            serializable_patterns = {}
            for task, data in self.specialization_patterns.items():
                pattern_dict = {}
                for head_key, spec in data.get("pattern", {}).items():
                    # Convert tuple keys to string representation
                    pattern_dict[f"{head_key[0]},{head_key[1]}"] = spec
                    
                serializable_patterns[task] = {
                    "description": data.get("description", ""),
                    "pattern": pattern_dict
                }
                
            # Prepare data
            registry_data = {
                "specialization_patterns": serializable_patterns,
                "pattern_performance": self.pattern_performance,
                "task_keywords": self.task_keywords
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
            if self.debug:
                print(f"Registry saved to {file_path}")
                
            return True
                
        except Exception as e:
            if self.debug:
                print(f"Error saving registry: {e}")
            return False
    
    def load_registry(self, file_path):
        """
        Load registry patterns and performance data from a file.
        
        Args:
            file_path: Path to load the registry data from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                if self.debug:
                    print(f"Registry file not found: {file_path}")
                return False
                
            # Load from file
            with open(file_path, 'r') as f:
                registry_data = json.load(f)
                
            # Convert patterns back from serializable format
            for task, data in registry_data.get("specialization_patterns", {}).items():
                pattern_dict = {}
                for head_key_str, spec in data.get("pattern", {}).items():
                    # Convert string key back to tuple
                    layer_idx, head_idx = map(int, head_key_str.split(","))
                    pattern_dict[(layer_idx, head_idx)] = spec
                    
                self.specialization_patterns[task] = {
                    "description": data.get("description", ""),
                    "pattern": pattern_dict
                }
                
            # Load performance data
            self.pattern_performance = registry_data.get("pattern_performance", {})
            
            # Load task keywords
            self.task_keywords = registry_data.get("task_keywords", self.task_keywords)
            
            if self.debug:
                print(f"Registry loaded from {file_path}")
                
            return True
                
        except Exception as e:
            if self.debug:
                print(f"Error loading registry: {e}")
            return False