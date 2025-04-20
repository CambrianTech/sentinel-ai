"""
Base Experiment Framework for Sentinel-AI

This module provides the foundation for all experiments in the Sentinel-AI
framework, with a consistent interface for running experiments in various
environments including command line, notebooks, and Colab.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import sys
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    """
    Abstract base class for all Sentinel-AI experiments.
    
    This class provides:
    1. A common interface for experiment execution
    2. Standardized command-line argument handling
    3. Environment detection (Colab, CPU/GPU, etc.)
    4. Consistent output organization and logging
    5. Proper visualization separation
    """
    
    def __init__(
        self,
        output_dir: str,
        device: Optional[str] = None,
        experiment_name: str = "experiment",
        enable_colab_integration: bool = True
    ):
        """
        Initialize the base experiment.
        
        Args:
            output_dir: Directory to save results
            device: Device to run on (auto-detected if None)
            experiment_name: Name of the experiment type
            enable_colab_integration: Whether to enable Colab-specific features
        """
        # Base configuration
        self.output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        self.experiment_name = experiment_name
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Device handling
        self.device = self._setup_device(device)
        
        # Environment detection
        self.in_colab = self._detect_colab()
        self.has_gpu = torch.cuda.is_available()
        self.is_apple_silicon = self._detect_apple_silicon()
        
        # Configure environment
        if enable_colab_integration and self.in_colab:
            self._setup_colab_environment()
            
        # Visualization is handled separately
        self.visualizer = None
        
        # Setup logging
        self._setup_logging()
        
        # Log environment details
        self._log_environment()
        
    def _setup_device(self, device: Optional[str]) -> str:
        """
        Set up the device for computation.
        
        Args:
            device: Requested device or None for auto-detection
            
        Returns:
            Device string ("cuda", "cpu", etc.)
        """
        if device is not None:
            # Use specified device if possible
            return device
            
        # Auto-detect optimal device
        if torch.cuda.is_available():
            device = "cuda"
        # Check for Apple Silicon - use MPS if available
        elif hasattr(torch, "has_mps") and torch.has_mps:
            device = "mps"  # Apple Silicon GPU
        else:
            device = "cpu"
            
        return device
        
    def _detect_colab(self) -> bool:
        """
        Detect if running in Google Colab.
        
        Returns:
            True if in Colab, False otherwise
        """
        try:
            import google.colab
            return True
        except (ImportError, ModuleNotFoundError):
            return False
            
    def _detect_apple_silicon(self) -> bool:
        """
        Detect if running on Apple Silicon.
        
        Returns:
            True if on Apple Silicon, False otherwise
        """
        import platform
        return platform.system() == "Darwin" and platform.processor() == "arm"
        
    def _setup_colab_environment(self):
        """Set up Google Colab environment."""
        try:
            # Try importing display components for Colab
            from IPython.display import display, HTML
            from google.colab import output
            
            # Style notebook for better visibility
            display(HTML("""
            <style>
            .experiment-header {
                background: #f0f8ff;
                padding: 10px;
                border-radius: 5px;
                border-left: 5px solid #4285F4;
                margin: 20px 0;
            }
            .experiment-section {
                margin: 10px 0;
                padding: 5px;
                border-left: 3px solid #34A853;
            }
            </style>
            """))
            
            # Print Colab-specific header
            display(HTML(f"""
            <div class="experiment-header">
                <h2>🧠 {self.experiment_name.title()} Experiment</h2>
                <p><b>Device:</b> {self.device}</p>
                <p><b>Output Directory:</b> {self.output_dir}</p>
                <p><b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """))
            
            # Setup progress display area
            self.colab_progress_output = output.js_eval_cell("""document.createElement('div')""")
            display(self.colab_progress_output)
        except (ImportError, AttributeError):
            logger.warning("Failed to set up Colab display environment")
            self.colab_progress_output = None
            
    def _setup_logging(self):
        """Set up logging for the experiment."""
        # Create log directory
        log_dir = self.output_dir / "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.experiment_name}_{timestamp}.log"
        
        # Add file handler to logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        self.log_file = log_file
        
    def _log_environment(self):
        """Log information about the execution environment."""
        logger.info(f"Starting {self.experiment_name} experiment")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Environment: {'Google Colab' if self.in_colab else 'Local'}")
        logger.info(f"GPU available: {self.has_gpu}")
        if self.has_gpu and torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Apple Silicon: {self.is_apple_silicon}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
    def create_experiment_run(self, run_name: Optional[str] = None) -> str:
        """
        Create a new experiment run with unique ID and directories.
        
        Args:
            run_name: Optional name for this specific run
            
        Returns:
            Experiment run ID
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment ID
        if run_name:
            experiment_id = f"{run_name}_{timestamp}"
        else:
            experiment_id = f"{self.experiment_name}_{timestamp}"
            
        # Create experiment directory
        experiment_dir = self.output_dir / experiment_id
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(experiment_dir / "visualizations", exist_ok=True)
        os.makedirs(experiment_dir / "data", exist_ok=True)
        os.makedirs(experiment_dir / "models", exist_ok=True)
        
        logger.info(f"Created experiment run: {experiment_id}")
        
        return experiment_id
        
    def save_experiment_params(
        self,
        params: Dict[str, Any],
        experiment_id: str
    ) -> str:
        """
        Save experiment parameters to JSON file.
        
        Args:
            params: Dictionary of experiment parameters
            experiment_id: Experiment run ID
            
        Returns:
            Path to saved parameters file
        """
        # Create experiment directory
        experiment_dir = self.output_dir / experiment_id
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Add timestamp to parameters
        params["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save parameters to JSON file
        params_path = experiment_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
            
        logger.info(f"Saved experiment parameters to {params_path}")
        
        return str(params_path)
        
    def save_experiment_results(
        self,
        results: Dict[str, Any],
        experiment_id: str,
        filename: str = "results.json"
    ) -> str:
        """
        Save experiment results to JSON file.
        
        Args:
            results: Dictionary of experiment results
            experiment_id: Experiment run ID
            filename: Name of the results file
            
        Returns:
            Path to saved results file
        """
        # Create experiment directory
        experiment_dir = self.output_dir / experiment_id
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Process results to make them JSON serializable
        processed_results = self._process_results_for_json(results)
        
        # Save results to JSON file
        results_path = experiment_dir / filename
        with open(results_path, "w") as f:
            json.dump(processed_results, f, indent=2)
            
        logger.info(f"Saved experiment results to {results_path}")
        
        return str(results_path)
        
    def _process_results_for_json(self, data: Any) -> Any:
        """
        Process results to make them JSON serializable.
        
        Args:
            data: Input data of any type
            
        Returns:
            JSON-serializable version of the data
        """
        if isinstance(data, dict):
            return {k: self._process_results_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_results_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return [self._process_results_for_json(item) for item in data]
        elif isinstance(data, (np.ndarray, np.number)):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, (datetime, Path)):
            return str(data)
        elif torch.is_tensor(data):
            return data.detach().cpu().numpy().tolist()
        else:
            # Try to convert to a basic type, if that fails, convert to string
            try:
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                return str(data)
                
    def load_experiment_results(
        self,
        experiment_id: str,
        filename: str = "results.json"
    ) -> Dict[str, Any]:
        """
        Load experiment results from JSON file.
        
        Args:
            experiment_id: Experiment run ID
            filename: Name of the results file
            
        Returns:
            Dictionary of experiment results
        """
        # Create experiment directory path
        experiment_dir = self.output_dir / experiment_id
        
        # Load results from JSON file
        results_path = experiment_dir / filename
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
                
            logger.info(f"Loaded experiment results from {results_path}")
            
            return results
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load experiment results: {e}")
            return {}
            
    def list_experiment_runs(self) -> List[str]:
        """
        List all experiment runs in the output directory.
        
        Returns:
            List of experiment run IDs
        """
        # Get all subdirectories in the output directory
        try:
            runs = [d.name for d in self.output_dir.iterdir() 
                   if d.is_dir() and (d / "params.json").exists()]
            
            logger.info(f"Found {len(runs)} experiment runs")
            
            return sorted(runs)
        except FileNotFoundError:
            logger.error(f"Output directory {self.output_dir} not found")
            return []
            
    def update_colab_progress(self, message: str, progress: float = None):
        """
        Update progress display in Colab notebook.
        
        Args:
            message: Progress message to display
            progress: Optional progress value (0.0 to 1.0)
        """
        if not self.in_colab or self.colab_progress_output is None:
            return
            
        try:
            from IPython.display import display, HTML
            from google.colab import output
            
            progress_html = f"<p>{message}</p>"
            
            if progress is not None:
                # Convert progress to percentage
                percent = int(progress * 100)
                progress_html += f"""
                <div style="width:100%; background-color:#f0f0f0; height:20px; border-radius:5px;">
                    <div style="width:{percent}%; background-color:#4285F4; height:20px; border-radius:5px;">
                    </div>
                </div>
                <p style="text-align:center;">{percent}%</p>
                """
                
            # Update progress display
            self.colab_progress_output.update(HTML(progress_html))
        except Exception as e:
            logger.warning(f"Failed to update Colab progress: {e}")
            
    def initialize_visualizer(self, visualizer_class, output_dir: Optional[str] = None):
        """
        Initialize a visualizer if not already initialized.
        
        Args:
            visualizer_class: The visualizer class to initialize
            output_dir: Optional output directory for visualizations
        """
        if self.visualizer is None:
            try:
                viz_output_dir = output_dir or str(self.output_dir)
                self.visualizer = visualizer_class(viz_output_dir)
                logger.info(f"Initialized {visualizer_class.__name__}")
            except Exception as e:
                logger.warning(f"Failed to initialize visualizer: {e}")
                
    @abstractmethod
    def run_experiment(self, *args, **kwargs):
        """
        Run the experiment.
        
        This method must be implemented by subclasses.
        """
        pass
        
    @classmethod
    def from_args(cls, args: Optional[List[str]] = None):
        """
        Create an experiment instance from command-line arguments.
        
        Args:
            args: Command-line arguments (uses sys.argv if None)
            
        Returns:
            Experiment instance
        """
        # Parse command-line arguments
        parser = cls.get_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Create output directory if not exists
        if parsed_args.output_dir is None:
            # Always put output in the /output directory in the project root
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            project_root = os.path.dirname(script_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parsed_args.output_dir = os.path.join(project_root, "output", f"{cls.__name__.lower()}_{timestamp}")
            
        os.makedirs(parsed_args.output_dir, exist_ok=True)
        
        # Create instance
        instance = cls(
            output_dir=parsed_args.output_dir,
            device=parsed_args.device,
            experiment_name=cls.__name__.lower()
        )
        
        # Set instance attributes from parsed arguments
        for key, value in vars(parsed_args).items():
            if hasattr(instance, key) or key == 'quick_test':
                setattr(instance, key, value)
                
        return instance
        
    @classmethod
    def get_argument_parser(cls):
        """
        Get the argument parser for command-line arguments.
        
        Returns:
            Argument parser
        """
        parser = argparse.ArgumentParser(description=f"{cls.__name__} Experiment")
        
        # Common arguments for all experiments
        parser.add_argument(
            "--output_dir", 
            type=str, 
            default=None,
            help="Output directory (default: auto-generated)"
        )
        
        parser.add_argument(
            "--device", 
            type=str, 
            default=None,
            help="Device (default: auto-detect)"
        )
        
        parser.add_argument(
            "--seed", 
            type=int, 
            default=42,
            help="Random seed (default: 42)"
        )
        
        parser.add_argument(
            "--no_visualize", 
            action="store_true", 
            default=False,
            help="Disable visualization generation"
        )
        
        return parser
        
    @classmethod
    def main(cls):
        """
        Main function for running the experiment from command line.
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create experiment instance from command-line arguments
        experiment = cls.from_args()
        
        try:
            # Run the experiment
            results = experiment.run_experiment()
            
            # Print success message
            print(f"\nExperiment completed successfully!")
            print(f"Results saved to: {experiment.output_dir}")
            
            # Print helpful next steps
            print("\nNext steps:")
            print(f"- View log file: {experiment.log_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running experiment: {e}", exc_info=True)
            print(f"Error running experiment: {e}")
            return None