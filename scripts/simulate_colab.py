#!/usr/bin/env python
"""
Simulate Upgrayedd experiments locally as if running on Colab.

This script provides a local testing environment that mimics Colab
execution, making it easier to validate experiments before deployment.

Usage:
  python scripts/simulate_colab.py --config CONFIG [--cycles CYCLES] [--mock-gpu]
  
Options:
  --config CONFIG     Path to experiment configuration file (JSON or YAML)
  --cycles CYCLES     Maximum number of cycles to run (default: 2)
  --mock-gpu          Mock GPU environment for testing CUDA code paths
  --output-dir DIR    Directory to save results (default: ./simulated_colab_output)
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Upgrayedd")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simulate Upgrayedd experiments locally")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to experiment configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--cycles", 
        type=int, 
        default=2,
        help="Maximum number of cycles to run (default: 2)"
    )
    parser.add_argument(
        "--mock-gpu", 
        action="store_true", 
        help="Mock GPU environment for testing CUDA code paths"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./simulated_colab_output",
        help="Directory to save results (default: ./simulated_colab_output)"
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Determine file type based on extension
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.json']:
        # Load JSON
        with open(config_path, 'r') as f:
            return json.load(f)
            
    elif ext.lower() in ['.yaml', '.yml']:
        # Load YAML
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML is required to load YAML files")
            logger.error("Install with: pip install pyyaml")
            sys.exit(1)
            
    else:
        raise ValueError(f"Unsupported config file format: {ext}")


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Import utilities
        from upgrayedd.utils.local_testing import (
            create_test_config,
            simulate_colab_experiment
        )
        
        # Create test directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Prepare test configuration
        logger.info("Preparing test configuration")
        test_config = create_test_config(
            config,
            force_cpu=not args.mock_gpu,
            use_small_model=True
        )
        
        # Override output directory
        test_config["output_dir"] = args.output_dir
        
        # Print test configuration
        print("\nTest Configuration:")
        print("=" * 80)
        for key, value in test_config.items():
            print(f"{key}: {value}")
            
        # Run simulation
        logger.info(f"Running simulated experiment (max cycles: {args.cycles})")
        result = simulate_colab_experiment(
            test_config,
            max_cycles=args.cycles,
            mock_gpu=args.mock_gpu
        )
        
        # Print results
        if result["success"]:
            print("\nSimulation Results:")
            print("=" * 80)
            
            if "results" in result and result["results"]:
                for key, value in result["results"].items():
                    if isinstance(value, (int, float, str, bool)):
                        print(f"{key}: {value}")
                        
            print("\nMetrics Files:")
            for filename, content in result.get("metrics_files", {}).items():
                print(f"- {filename}")
                
            print(f"\nOutput saved to: {args.output_dir}")
            print("Simulation completed successfully.")
            return 0
        else:
            print("\nSimulation Failed:")
            print("=" * 80)
            if "error" in result:
                print(f"Error: {result['error']}")
            return 1
            
    except ImportError as e:
        logger.error(f"Failed to import necessary modules: {e}")
        logger.error("Make sure upgrayedd package is installed")
        return 1
        
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())