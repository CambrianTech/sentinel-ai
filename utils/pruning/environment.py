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
        self.has_high_ram = False
        self.default_device = "cpu"
        self.memory_limit = 4  # Default memory limit in GB
        
        # Configure environment
        self._configure()
        
    def _configure(self):
        """Configure the environment based on detected hardware"""
        import jax
        
        # First attempt to detect available memory for any platform
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            self.has_high_ram = total_ram > 12
            
            # Adjust memory limit based on available RAM
            if total_ram > 24:
                self.memory_limit = 16
            elif total_ram > 12:
                self.memory_limit = 8
            else:
                self.memory_limit = 4
                
            print(f"Detected {total_ram:.1f}GB RAM, setting memory limit to {self.memory_limit}GB")
        except Exception:
            print("Could not detect system memory, using default memory limits")
        
        if self.is_arm_mac:
            # Mac-specific settings to avoid BLAS issues
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            self.memory_limit = max(8, self.memory_limit)  # M1/M2 Macs can handle more
            
        elif self.in_colab:
            # For Colab, try to detect and use TPU if available
            try:
                import jax.tools.colab_tpu
                jax.tools.colab_tpu.setup_tpu()
                # Check if TPU devices are really available
                if len(jax.devices('tpu')) > 0:
                    self.has_tpu = True
                    self.default_device = "tpu"
                    self.memory_limit = max(24, self.memory_limit)  # TPUs have more memory
                    print("TPU configured for JAX")
            except Exception:
                # TPU not available, continue to check for GPU
                pass
                
            # Check for GPU if TPU wasn't detected
            if not self.has_tpu:
                try:
                    # First try to explicitly set the platform
                    try:
                        jax.config.update('jax_platform_name', 'gpu')
                    except:
                        pass
                        
                    # Now check if GPU devices are available
                    try:
                        if len(jax.devices('gpu')) > 0:
                            self.has_gpu = True
                            self.default_device = "gpu"
                            self.memory_limit = max(12, self.memory_limit)  # GPUs have decent memory
                            print("GPU configured for JAX")
                    except:
                        # Fall back to any available device and check if it's a GPU
                        devices = jax.devices()
                        self.has_gpu = any("gpu" in str(d).lower() for d in devices)
                        if self.has_gpu:
                            self.default_device = "gpu"
                            self.memory_limit = max(12, self.memory_limit)
                            print("GPU detected for JAX")
                except Exception as e:
                    print(f"Error detecting GPU: {e}")
                    
        # Fallback to CPU if no GPU or TPU was detected
        if not self.has_gpu and not self.has_tpu:
            try:
                # Make sure we're using the CPU backend
                jax.config.update('jax_platform_name', 'cpu')
                print(f"Using CPU backend with {self.memory_limit}GB memory limit")
            except Exception as e:
                print(f"Error configuring CPU backend: {e}")
                
        # Always ensure we can get the devices, even on CPU
        try:
            print(f"JAX devices: {jax.devices()}")
            print(f"JAX default backend: {jax.default_backend()}")
        except Exception as e:
            print(f"Error accessing JAX devices: {e}")
    
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
        
        # Apply safety factor for CPU-only
        effective_limit = self.memory_limit
        if not self.has_gpu and not self.has_tpu:
            # CPU is slower and might need more memory overhead
            effective_limit = self.memory_limit * 0.75
            print(f"Using reduced memory limit of {effective_limit:.1f}GB for CPU-only mode")
        
        # Filter models based on memory limit
        suitable_models = {k: v for k, v in all_models.items() if v <= effective_limit}
        
        # If we have no models (rare case), at least include the smallest ones
        if not suitable_models:
            smallest_models = {
                "distilgpt2": 0.5,
                "facebook/opt-125m": 0.5
            }
            suitable_models = smallest_models
            print("Limited to smallest models due to memory constraints")
            
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