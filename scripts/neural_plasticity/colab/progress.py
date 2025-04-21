"""
Progress tracking utilities for neural plasticity experiments.

This module provides functions and classes for tracking and displaying
progress of neural plasticity experiments in both Colab and local environments.
"""

import time
import sys
import datetime
from typing import Dict, Any, Optional, Union, List, Callable

# Local imports
from scripts.neural_plasticity.colab.integration import is_colab

class ProgressTracker:
    """Track and display progress of neural plasticity experiments."""
    
    def __init__(self, 
                 description: str = "Progress", 
                 total: int = 100, 
                 disable_tqdm: bool = False,
                 notebook_display: bool = None):
        """
        Initialize progress tracker.
        
        Args:
            description: Description of the task
            total: Total number of steps
            disable_tqdm: Whether to disable tqdm progress bar
            notebook_display: Force notebook display mode (True/False)
                             If None, auto-detect based on environment
        """
        self.description = description
        self.total = total
        self.start_time = time.time()
        self.metrics = {}
        self.current = 0
        self.disable_tqdm = disable_tqdm
        
        # Determine if we should use notebook display
        if notebook_display is None:
            notebook_display = is_colab()
        self.notebook_display = notebook_display
        
        # Initialize progress bar
        self.progress_bar = None
        self.initialize_progress_bar()
    
    def initialize_progress_bar(self):
        """Initialize the progress bar based on environment."""
        try:
            from tqdm.auto import tqdm
            
            if self.notebook_display:
                from tqdm.notebook import tqdm as notebook_tqdm
                self.progress_bar = notebook_tqdm(
                    total=self.total,
                    desc=self.description,
                    disable=self.disable_tqdm
                )
            else:
                self.progress_bar = tqdm(
                    total=self.total, 
                    desc=self.description,
                    disable=self.disable_tqdm
                )
        except ImportError:
            # Fall back to simple progress display if tqdm is not available
            self.progress_bar = None
            self._print_progress()
    
    def update(self, 
               step: int = 1, 
               metrics: Optional[Dict[str, float]] = None,
               force_display: bool = False):
        """
        Update progress tracker.
        
        Args:
            step: Current step or increment
            metrics: Dictionary of metrics to track
            force_display: Force display of progress
        """
        if isinstance(step, int):
            # step is an increment
            self.current += step
        else:
            # step is the current position
            self.current = step
        
        # Update metrics
        if metrics:
            for key, value in metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
        
        # Update progress bar
        if self.progress_bar:
            postfix = {}
            if metrics:
                for key, values in self.metrics.items():
                    if values:
                        postfix[key] = f"{values[-1]:.4f}"
            
            self.progress_bar.update(step if isinstance(step, int) else step - self.progress_bar.n)
            if postfix:
                self.progress_bar.set_postfix(**postfix)
        else:
            # Simple progress display
            self._print_progress(force=force_display)
    
    def _print_progress(self, force=False):
        """Simple progress display when tqdm is not available."""
        if self.current == 0 or self.current >= self.total or force or self.current % max(1, self.total // 20) == 0:
            elapsed = time.time() - self.start_time
            percent = 100.0 * self.current / self.total
            
            # Format metrics string
            metrics_str = ""
            for key, values in self.metrics.items():
                if values:
                    metrics_str += f"{key}: {values[-1]:.4f} "
            
            # Print progress
            sys.stdout.write(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%) "
                            f"[{self._format_time(elapsed)}] {metrics_str}")
            sys.stdout.flush()
            
            if self.current >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
    
    def _format_time(self, seconds):
        """Format time in seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)}m {int(seconds % 60)}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)}h {int(minutes)}m"
    
    def close(self):
        """Close progress bar and print summary."""
        if self.progress_bar:
            self.progress_bar.close()
        
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} completed in {self._format_time(elapsed)}")
        
        # Print final metrics
        if self.metrics:
            print("\nFinal metrics:")
            for key, values in self.metrics.items():
                if values:
                    print(f"  {key}: {values[-1]:.6f}")
    
    def __enter__(self):
        """Enable use as a context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        self.close()

def track_progress(iterable, desc="Progress", total=None, metrics_fn=None):
    """
    Track progress through an iterable.
    
    Args:
        iterable: Iterable to track
        desc: Description of the task
        total: Total number of items (if not available from iterable)
        metrics_fn: Function to extract metrics from the current item
        
    Returns:
        An iterator that yields the same items as the input iterable
    """
    try:
        total = len(iterable) if total is None and hasattr(iterable, "__len__") else total
    except (TypeError, AttributeError):
        total = None
    
    tracker = ProgressTracker(description=desc, total=total or 100)
    
    try:
        for i, item in enumerate(iterable):
            # Extract metrics if a metrics function is provided
            metrics = metrics_fn(item) if metrics_fn else None
            
            # Update progress
            if total is None:
                tracker.update(i, metrics=metrics)
            else:
                tracker.update(1, metrics=metrics)
            
            yield item
    finally:
        tracker.close()

def log_time(func_or_name=None):
    """
    Decorator to log execution time of a function.
    
    Can be used as:
    @log_time
    def my_function():
        ...
    
    Or:
    @log_time("Custom description")
    def my_function():
        ...
    
    Or:
    with log_time("Operation"):
        # do something
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            desc = func.__name__ if func_or_name is None else func_or_name
            start_time = time.time()
            print(f"Starting {desc}...")
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Format time
            if elapsed < 60:
                time_str = f"{elapsed:.2f} seconds"
            elif elapsed < 3600:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.2f} seconds"
            else:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                time_str = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
            
            print(f"{desc} completed in {time_str}")
            return result
        return wrapper
    
    # Handle both @log_time and @log_time("Description") syntax
    if callable(func_or_name):
        return decorator(func_or_name)
    
    # Handle context manager case
    if func_or_name is None:
        desc = "Operation"
    else:
        desc = func_or_name
    
    class LogTimeContext:
        def __enter__(self):
            print(f"Starting {desc}...")
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                time_str = f"{elapsed:.2f} seconds"
            elif elapsed < 3600:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.2f} seconds"
            else:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                time_str = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
            
            print(f"{desc} completed in {time_str}")
    
    if func_or_name is None:
        return LogTimeContext()
    else:
        return decorator