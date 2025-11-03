"""
Install Weights & Biases for neural plasticity dashboard in Colab

This module provides a simple function to install and set up Weights & Biases
for real-time dashboard visualization in Google Colab.

Version: v0.0.1 (2025-04-20 25:35:00)
"""

import sys
import subprocess
import logging
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

def install_wandb(version: Optional[str] = None) -> bool:
    """
    Install the Weights & Biases package.
    
    Args:
        version: Optional specific version to install
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        # Check if wandb is already installed
        try:
            import wandb
            logger.info(f"Weights & Biases already installed (version: {wandb.__version__})")
            
            if version and wandb.__version__ != version:
                logger.info(f"Upgrading wandb from {wandb.__version__} to {version}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"wandb=={version}"])
                
                # Re-import to check version
                import importlib
                wandb = importlib.reload(wandb)
                logger.info(f"Upgraded to wandb version {wandb.__version__}")
            
            return True
            
        except ImportError:
            # Install wandb
            logger.info(f"Installing Weights & Biases {'(version: ' + version + ')' if version else ''}")
            
            # Install the package
            if version:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"wandb=={version}"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])
            
            # Check installation
            try:
                import wandb
                logger.info(f"Successfully installed Weights & Biases (version: {wandb.__version__})")
                return True
            except ImportError:
                logger.error("Failed to import wandb after installation")
                return False
                
    except Exception as e:
        logger.error(f"Error installing Weights & Biases: {e}")
        return False

def setup_wandb_login() -> bool:
    """
    Set up wandb login for dashboard access.
    
    Returns:
        bool: True if login was successful, False otherwise
    """
    try:
        import wandb
        
        # Check if already logged in
        try:
            wandb.ensure_login()
            logger.info("Already logged in to Weights & Biases")
            return True
        except Exception:
            pass
        
        # Detect if running in Colab
        is_colab = False
        try:
            import google.colab
            is_colab = True
        except ImportError:
            pass
        
        # Print login instructions
        if is_colab:
            print("🔑 Logging in to Weights & Biases...")
            print("You will need to authenticate in the prompt that appears.")
            print("If no prompt appears, run `wandb.login()` in a separate cell.")
        else:
            print("🔑 Logging in to Weights & Bibes...")
            print("You will need to authenticate in the browser window that opens.")
        
        # Attempt login
        wandb.login()
        
        # Check login success
        try:
            wandb.ensure_login()
            logger.info("Successfully logged in to Weights & Biases")
            return True
        except Exception:
            logger.error("Failed to log in to Weights & Bibes")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up Weights & Bibes login: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Install wandb if run directly
    success = install_wandb()
    
    # Set up login if successful
    if success:
        setup_wandb_login()