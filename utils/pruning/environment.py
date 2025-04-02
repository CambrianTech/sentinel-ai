"""
Environment detection and configuration
"""

import os
import sys
import platform

class Environment:
    """Detect and configure the runtime environment"""
    
    def __init__(self):
        # Check if we're running in Colab
        self.in_colab = 'google.colab' in sys.modules
        
        # Check if we're on a Mac
        self.is_mac = platform.system() == "Darwin"
        self.is_arm_mac = self.is_mac and platform.machine().startswith("arm")
        
        # Initialize JAX-related properties
        self.has_gpu = False
        self.has_tpu = False
        self.default_device = "cpu"
        self.memory_limit = 4  # Default memory limit in GB
        
        # Configure environment
        self._configure()
        
    def _configure(self):
        """Configure the environment based on detected hardware"""
        if self.is_arm_mac:
            # Mac-specific settings to avoid BLAS issues
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            self.memory_limit = 8  # M1/M2 Macs can handle more
        elif self.in_colab:
            # For Colab, try to detect and use TPU if available
            try:
                import jax
                import jax.tools.colab_tpu
                jax.tools.colab_tpu.setup_tpu()
                self.has_tpu = True
                self.default_device = "tpu"
                self.memory_limit = 24  # TPUs have more memory
                print("TPU configured for JAX")
            except Exception:
                # Check for GPU
                try:
                    import jax
                    jax.config.update('jax_platform_name', 'gpu')
                    if len(jax.devices('gpu')) > 0:
                        self.has_gpu = True
                        self.default_device = "gpu"
                        self.memory_limit = 12  # GPUs have decent memory
                        print("GPU configured for JAX")
                except Exception:
                    print("No TPU or GPU detected, using CPU")
    
    def get_suitable_models(self):
        """Return a list of models suitable for this environment"""
        all_models = {
            # Model name: approximate memory needed in GB
            "distilgpt2": 0.5,
            "gpt2": 1.5,
            "gpt2-medium": 3.0,
            "gpt2-large": 6.0,
            "gpt2-xl": 12.0,
            "facebook/opt-125m": 0.5,
            "facebook/opt-350m": 1.5,
            "facebook/opt-1.3b": 5.0,
            "EleutherAI/pythia-160m": 0.7,
            "EleutherAI/pythia-410m": 1.8,
            "EleutherAI/pythia-1b": 4.0
        }
        
        # Filter models based on memory limit
        suitable_models = {k: v for k, v in all_models.items() if v <= self.memory_limit}
        
        # Sort by size (smallest first)
        return sorted(suitable_models.keys(), key=lambda x: all_models[x])
    
    def print_info(self):
        """Print environment information"""
        print(f"Platform: {platform.platform()}")
        print(f"Python version: {platform.python_version()}")
        print(f"Running in Google Colab: {self.in_colab}")
        print(f"Running on Mac: {self.is_mac}, Apple Silicon: {self.is_arm_mac}")
        print(f"Default device: {self.default_device}")
        print(f"Memory limit (GB): {self.memory_limit}")
        print(f"\nModels available for this environment:")
        for model in self.get_suitable_models():
            print(f"  - {model}")