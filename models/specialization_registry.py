import json
import os
import torch
import numpy as np
from models.agency_specialization import AgencySpecialization

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
        self.specialization = None
        if model:
            self.specialization = AgencySpecialization(model, debug=debug)
        
        # Define standard task types
        self.task_types = [
            "pattern_matching",  # e.g., code completion, template filling
            "logical_reasoning", # e.g., math, logic puzzles, SQL
            "long_context",      # e.g., document summarization, analysis
            "creative_generation", # e.g., stories, poetry
            "constrained_resources" # e.g., low-memory environments
        ]
        
        # Initialize registry storage
        self.registry = {}
        self.custom_patterns = {}
        self.task_history = []
        self.current_task = None
        
        # Load default patterns
        self._load_default_patterns()
        
        # Path for storing custom patterns
        self.registry_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            '../config/specialization_patterns.json'
        )
        
        # Try to load existing custom patterns
        self._load_custom_patterns()
    
    def _load_default_patterns(self):
        """Load the default specialization patterns based on validation results."""
        self.registry = {
            # Pattern matching: emphasize pattern recognition heads
            "pattern_matching": {
                "description": "Optimized for pattern recognition tasks like code completion",
                "head_distribution": {
                    "pattern_recognition": 0.5,  # Increased from default 0.35
                    "logical_reasoning": 0.2,    # Reduced from default
                    "memory_context": 0.2,       # Reduced from default
                    "creative_synthesis": 0.1    # Reduced from default
                },
                "withdrawn_percentages": {
                    "logical_reasoning": 0.6,    # Higher withdrawal of logical heads
                    "creative_synthesis": 0.7    # High withdrawal of creative heads
                },
                "expected_metrics": {
                    "performance_improvement": 0.38, # 38%
                    "resource_reduction": 0.27,      # 27%
                    "quality_enhancement": 0.20      # 20%
                }
            },
            
            # Logical reasoning: emphasize logical reasoning heads
            "logical_reasoning": {
                "description": "Optimized for logical reasoning tasks like math or SQL",
                "head_distribution": {
                    "pattern_recognition": 0.25,
                    "logical_reasoning": 0.45,    # Increased from default 0.25
                    "memory_context": 0.20,
                    "creative_synthesis": 0.10
                },
                "withdrawn_percentages": {
                    "pattern_recognition": 0.3,
                    "creative_synthesis": 0.6
                },
                "expected_metrics": {
                    "performance_improvement": 0.35, # 35%
                    "resource_reduction": 0.28,      # 28%
                    "quality_enhancement": 0.22      # 22%
                }
            },
            
            # Long context: emphasize memory context heads
            "long_context": {
                "description": "Optimized for long context processing like document analysis",
                "head_distribution": {
                    "pattern_recognition": 0.20,
                    "logical_reasoning": 0.20,
                    "memory_context": 0.50,       # Increased from default 0.25
                    "creative_synthesis": 0.10
                },
                "withdrawn_percentages": {
                    "pattern_recognition": 0.3,
                    "creative_synthesis": 0.5
                },
                "expected_metrics": {
                    "performance_improvement": 0.32, # 32%
                    "resource_reduction": 0.25,      # 25%
                    "quality_enhancement": 0.18      # 18%
                }
            },
            
            # Creative generation: emphasize creative synthesis heads
            "creative_generation": {
                "description": "Optimized for creative text generation like stories or poetry",
                "head_distribution": {
                    "pattern_recognition": 0.30,
                    "logical_reasoning": 0.15,
                    "memory_context": 0.20,
                    "creative_synthesis": 0.35    # Increased from default 0.15
                },
                "withdrawn_percentages": {
                    "logical_reasoning": 0.4
                },
                "expected_metrics": {
                    "performance_improvement": 0.30, # 30%
                    "resource_reduction": 0.22,      # 22%
                    "quality_enhancement": 0.25      # 25%
                }
            },
            
            # Constrained resources: implement strategy from validation
            "constrained_resources": {
                "description": "Optimized for resource-constrained environments",
                "head_distribution": {
                    "pattern_recognition": 0.35,
                    "logical_reasoning": 0.25,
                    "memory_context": 0.25,
                    "creative_synthesis": 0.15
                },
                "critical_percentage": 0.15,  # Keep only 15% of heads active
                "expected_metrics": {
                    "performance_improvement": 0.14, # 14%
                    "resource_reduction": 0.33,      # 33%
                    "quality_enhancement": 0.04      # 4%
                }
            }
        }
    
    def _load_custom_patterns(self):
        """Load custom specialization patterns from file if available."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.custom_patterns = json.load(f)
                if self.debug:
                    print(f"Loaded {len(self.custom_patterns)} custom specialization patterns")
            except Exception as e:
                if self.debug:
                    print(f"Error loading custom patterns: {e}")
    
    def save_custom_patterns(self):
        """Save custom specialization patterns to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.custom_patterns, f, indent=2)
            if self.debug:
                print(f"Saved {len(self.custom_patterns)} custom specialization patterns")
        except Exception as e:
            if self.debug:
                print(f"Error saving custom patterns: {e}")
    
    def set_model(self, model):
        """Set or update the model for this registry."""
        self.model = model
        self.specialization = AgencySpecialization(model, debug=self.debug)
    
    def get_available_patterns(self):
        """Get all available specialization patterns."""
        # Combine default and custom patterns
        all_patterns = {**self.registry, **self.custom_patterns}
        return {
            task: pattern.get("description", "No description available") 
            for task, pattern in all_patterns.items()
        }
    
    def detect_task_type(self, input_text):
        """
        Automatically detect the most appropriate task type based on input text.
        
        Args:
            input_text: Text input to analyze
            
        Returns:
            String with detected task type
        """
        # Simple heuristic-based detection
        input_lower = input_text.lower()
        
        # Code or pattern detection
        code_indicators = ["def ", "class ", "function", "return", "import ", "var ", "const ", 
                         "{", "}", "()", "=>", ";", "for(", "while(", "<div", "<p>", "</", "SELECT "]
        if any(indicator in input_lower for indicator in code_indicators):
            return "pattern_matching"
        
        # Logical reasoning detection
        logical_indicators = ["solve", "calculate", "what is", "how many", "prove", "explain why",
                            "sql", "query", "database", "logic", "math", "equation", "=", "+", "-", "*", "/"]
        if any(indicator in input_lower for indicator in logical_indicators):
            return "logical_reasoning"
        
        # Long context detection
        if len(input_text.split()) > 200:  # If input is longer than ~200 words
            return "long_context"
        
        # Creative generation detection
        creative_indicators = ["story", "poem", "write a", "create a", "imagine", "fiction", 
                             "creative", "narrative", "describe", "fantasy"]
        if any(indicator in input_lower for indicator in creative_indicators):
            return "creative_generation"
        
        # Default to pattern matching as the most common case
        return "pattern_matching"
    
    def apply_specialization(self, task_type=None, input_text=None):
        """
        Apply a specialization pattern based on task type.
        
        Args:
            task_type: String indicating task type, or None to auto-detect
            input_text: Text input for task detection if task_type is None
            
        Returns:
            Dict with report of applied specialization
        """
        if self.model is None or self.specialization is None:
            return {"error": "No model set for specialization"}
        
        # Detect task if not specified
        if task_type is None:
            if input_text is None:
                return {"error": "Either task_type or input_text must be specified"}
            task_type = self.detect_task_type(input_text)
        
        # Check if task type exists
        all_patterns = {**self.registry, **self.custom_patterns}
        if task_type not in all_patterns:
            return {"error": f"Unknown task type: {task_type}"}
        
        # Track task history
        self.task_history.append(task_type)
        self.current_task = task_type
        
        # Get pattern configuration
        pattern = all_patterns[task_type]
        
        # Apply specialization
        if task_type == "constrained_resources":
            # Special handling for constrained resources case
            self._apply_constrained_resources_pattern(pattern)
        else:
            # Standard pattern application
            self._apply_standard_pattern(task_type, pattern)
        
        # Get report on applied specialization
        report = self.specialization.get_specialization_report()
        
        # Add task info to report
        report["task_type"] = task_type
        report["description"] = pattern.get("description", "No description available")
        report["expected_metrics"] = pattern.get("expected_metrics", {})
        
        return report
    
    def _apply_standard_pattern(self, task_type, pattern):
        """Apply a standard specialization pattern."""
        # Initialize specialization if not already done
        if not self.specialization.initialized:
            self.specialization.initialize_specialization()
        
        # Reset all heads to active state
        for category, assignments in self.specialization.head_assignments.items():
            for layer_idx, head_idx in assignments:
                self.model.set_head_state(layer_idx, head_idx, "active", True)
        
        # Apply withdrawn percentages for each category
        withdrawn_percentages = pattern.get("withdrawn_percentages", {})
        for category, percentage in withdrawn_percentages.items():
            if category in self.specialization.head_assignments:
                assignments = self.specialization.head_assignments[category]
                withdraw_count = int(len(assignments) * percentage)
                
                for i, (layer_idx, head_idx) in enumerate(assignments):
                    if i < withdraw_count:
                        self.model.set_head_state(layer_idx, head_idx, "withdrawn", True)
    
    def _apply_constrained_resources_pattern(self, pattern):
        """Apply the constrained resources specialization pattern."""
        # Initialize specialization if not already done
        if not self.specialization.initialized:
            self.specialization.initialize_specialization()
        
        # Get critical percentage (how many heads to keep active)
        critical_percentage = pattern.get("critical_percentage", 0.15)
        
        # Identify critical heads across all categories
        critical_heads = []
        
        # Select critical heads from each category proportionally
        for category, assignments in self.specialization.head_assignments.items():
            critical_count = max(1, int(len(assignments) * critical_percentage))
            critical_heads.extend(assignments[:critical_count])
        
        # Set all heads to withdrawn by default
        for layer_idx in range(self.specialization.num_layers):
            for head_idx in range(self.specialization.num_heads):
                self.model.set_head_state(layer_idx, head_idx, "withdrawn", True)
        
        # Set critical heads to active with high utilization
        for layer_idx, head_idx in critical_heads:
            self.model.set_head_state(layer_idx, head_idx, "active", True)
            if hasattr(self.model.blocks[layer_idx]["attn"], "agency_signals"):
                self.model.blocks[layer_idx]["attn"].agency_signals[head_idx]["utilization"] = 0.92
    
    def create_custom_pattern(self, name, description, head_distribution=None, 
                            withdrawn_percentages=None, expected_metrics=None):
        """
        Create a custom specialization pattern.
        
        Args:
            name: Pattern name/identifier
            description: Pattern description
            head_distribution: Optional dict with category percentages
            withdrawn_percentages: Optional dict with withdrawal percentages by category
            expected_metrics: Optional dict with expected performance metrics
            
        Returns:
            Dict with the created pattern
        """
        # Create pattern with provided or default values
        pattern = {
            "description": description,
            "head_distribution": head_distribution or {
                "pattern_recognition": 0.35,
                "logical_reasoning": 0.25,
                "memory_context": 0.25,
                "creative_synthesis": 0.15
            },
            "withdrawn_percentages": withdrawn_percentages or {},
            "expected_metrics": expected_metrics or {
                "performance_improvement": 0.0,
                "resource_reduction": 0.0,
                "quality_enhancement": 0.0
            }
        }
        
        # Store in custom patterns
        self.custom_patterns[name] = pattern
        
        # Save to file
        self.save_custom_patterns()
        
        return pattern
    
    def update_pattern_metrics(self, task_type, metrics):
        """
        Update expected metrics for a pattern based on observed performance.
        
        Args:
            task_type: Task type to update
            metrics: Dict with observed performance metrics
            
        Returns:
            Dict with updated pattern
        """
        # Check if pattern exists
        all_patterns = {**self.registry, **self.custom_patterns}
        if task_type not in all_patterns:
            return {"error": f"Unknown task type: {task_type}"}
            
        # Determine where the pattern is stored
        if task_type in self.custom_patterns:
            pattern = self.custom_patterns[task_type]
            is_custom = True
        else:
            pattern = self.registry[task_type]
            # Copy to custom patterns to avoid modifying defaults
            self.custom_patterns[task_type] = pattern.copy()
            pattern = self.custom_patterns[task_type]
            is_custom = True
        
        # Update metrics with exponential moving average
        alpha = 0.7  # Weight for new metrics vs old metrics
        
        if "expected_metrics" not in pattern:
            pattern["expected_metrics"] = {}
            
        for metric, value in metrics.items():
            if metric in pattern["expected_metrics"]:
                # Blend old and new values
                old_value = pattern["expected_metrics"][metric]
                pattern["expected_metrics"][metric] = alpha * value + (1 - alpha) * old_value
            else:
                # Set new value
                pattern["expected_metrics"][metric] = value
        
        # Save if this is a custom pattern
        if is_custom:
            self.save_custom_patterns()
        
        return pattern
    
    def get_task_history(self):
        """Get the history of applied task types."""
        return self.task_history
    
    def get_current_task(self):
        """Get the currently applied task type."""
        return self.current_task
    
    def get_pattern_details(self, task_type):
        """Get detailed information about a specific pattern."""
        all_patterns = {**self.registry, **self.custom_patterns}
        if task_type not in all_patterns:
            return {"error": f"Unknown task type: {task_type}"}
        return all_patterns[task_type]