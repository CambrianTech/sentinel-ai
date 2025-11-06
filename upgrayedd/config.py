"""
Upgrayedd config module - Configuration handling for the Upgrayedd system
"""

import os
import yaml
import json
import argparse
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

@dataclass
class UpgrayeddConfig:
    """
    Configuration for Upgrayedd model transformation.
    
    This class stores all the configuration parameters for the Upgrayedd system,
    including pruning, growth, training, and output settings.
    """
    
    # Model settings
    model_name: str = "distilgpt2"
    output_dir: Optional[str] = None
    device: Optional[str] = None
    
    # Dataset settings
    dataset: str = "tiny_shakespeare"
    custom_dataset_path: Optional[str] = None
    
    # Optimization settings
    cycles: int = 3
    pruning_level: float = 0.3
    growth_ratio: float = 0.5
    learning_rate: float = 5e-5
    
    # Controller settings
    controller_config: Dict[str, Any] = field(default_factory=lambda: {
        "controller_type": "ann",
        "controller_lr": 0.01,
        "update_frequency": 50,
        "warmup_steps": 100,
        "entropy_threshold": 0.7,
        "gradient_scale": 1.0
    })
    
    # Plasticity settings
    plasticity_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_degeneration_score": 3.0,
        "max_perplexity_increase": 0.15,
        "training_steps": 100,
        "memory_capacity": 5,
        "entropy_weighted": True
    })
    
    # Training settings
    training_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    
    # Output settings
    compress_model: bool = True
    compression_type: str = "mask"
    run_inference: bool = True
    plot: bool = True
    log_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UpgrayeddConfig':
        """Create a config from a dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save the config to a file."""
        # Determine file format from extension
        ext = os.path.splitext(path)[1].lower()
        
        with open(path, 'w') as f:
            if ext == '.yaml' or ext == '.yml':
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            elif ext == '.json':
                json.dump(self.to_dict(), f, indent=2)
            else:
                # Default to JSON
                json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'UpgrayeddConfig':
        """Load a config from a file."""
        # Determine file format from extension
        ext = os.path.splitext(path)[1].lower()
        
        with open(path, 'r') as f:
            if ext == '.yaml' or ext == '.yml':
                config_dict = yaml.safe_load(f)
            elif ext == '.json':
                config_dict = json.load(f)
            else:
                # Try to parse as JSON first, then YAML
                try:
                    config_dict = json.load(f)
                except json.JSONDecodeError:
                    # Reset file position and try YAML
                    f.seek(0)
                    config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)


def load_config(path: str) -> UpgrayeddConfig:
    """
    Load a configuration from a file.
    
    Args:
        path: Path to the configuration file (YAML or JSON)
        
    Returns:
        UpgrayeddConfig: The loaded configuration
    """
    return UpgrayeddConfig.load(path)

def save_config(config: UpgrayeddConfig, path: str) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: The configuration to save
        path: Path to save the configuration to (YAML or JSON)
    """
    config.save(path)

def parse_args() -> UpgrayeddConfig:
    """
    Parse command-line arguments into a configuration.
    
    Returns:
        UpgrayeddConfig: Configuration from command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Upgrayedd: Transform models into adaptive, self-optimizing networks"
    )
    
    # Model settings
    parser.add_argument("--model", type=str, default="distilgpt2",
                      help="HuggingFace model name or path")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save results")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda or cpu)")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset to use for optimization")
    parser.add_argument("--custom-dataset", type=str, default=None,
                      help="Path to custom dataset")
    
    # Optimization settings
    parser.add_argument("--cycles", type=int, default=3,
                      help="Number of optimization cycles")
    parser.add_argument("--pruning-level", type=float, default=0.3,
                      help="Initial pruning level (0.0-1.0)")
    parser.add_argument("--growth-ratio", type=float, default=0.5,
                      help="Growth ratio (0.0-1.0)")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                      help="Learning rate for fine-tuning")
    
    # Controller settings
    parser.add_argument("--controller-type", type=str, default="ann",
                      choices=["ann", "static", "rule"],
                      help="Controller type")
    parser.add_argument("--controller-lr", type=float, default=0.01,
                      help="Controller learning rate")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=3,
                      help="Training epochs per cycle")
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                      help="Gradient accumulation steps")
    
    # Output settings
    parser.add_argument("--compress", action="store_true",
                      help="Compress the model after optimization")
    parser.add_argument("--compression-type", type=str, default="mask",
                      choices=["mask", "remove", "distill"],
                      help="Type of compression to apply")
    parser.add_argument("--run-inference", action="store_true",
                      help="Run inference after optimization")
    parser.add_argument("--plot", action="store_true",
                      help="Generate plots")
    parser.add_argument("--log-metrics", action="store_true",
                      help="Log detailed metrics")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                      help="Path to configuration file (overrides command-line arguments)")
    
    args = parser.parse_args()
    
    # Load from config file if specified
    if args.config:
        config = load_config(args.config)
    else:
        # Convert args to dict, removing None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        
        # Create config from args
        config = UpgrayeddConfig()
        
        # Update config with args
        if 'model' in args_dict:
            config.model_name = args_dict['model']
        if 'output_dir' in args_dict:
            config.output_dir = args_dict['output_dir']
        if 'device' in args_dict:
            config.device = args_dict['device']
        if 'dataset' in args_dict:
            config.dataset = args_dict['dataset']
        if 'custom_dataset' in args_dict:
            config.custom_dataset_path = args_dict['custom_dataset']
        if 'cycles' in args_dict:
            config.cycles = args_dict['cycles']
        if 'pruning_level' in args_dict:
            config.pruning_level = args_dict['pruning_level']
        if 'growth_ratio' in args_dict:
            config.growth_ratio = args_dict['growth_ratio']
        if 'learning_rate' in args_dict:
            config.learning_rate = args_dict['learning_rate']
        if 'epochs' in args_dict:
            config.training_epochs = args_dict['epochs']
        if 'batch_size' in args_dict:
            config.batch_size = args_dict['batch_size']
        if 'gradient_accumulation' in args_dict:
            config.gradient_accumulation = args_dict['gradient_accumulation']
        if 'compress' in args_dict:
            config.compress_model = args_dict['compress']
        if 'compression_type' in args_dict:
            config.compression_type = args_dict['compression_type']
        if 'run_inference' in args_dict:
            config.run_inference = args_dict['run_inference']
        if 'plot' in args_dict:
            config.plot = args_dict['plot']
        if 'log_metrics' in args_dict:
            config.log_metrics = args_dict['log_metrics']
        
        # Update controller config
        if 'controller_type' in args_dict:
            config.controller_config['controller_type'] = args_dict['controller_type']
        if 'controller_lr' in args_dict:
            config.controller_config['controller_lr'] = args_dict['controller_lr']
    
    return config