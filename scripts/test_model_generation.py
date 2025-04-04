#!/usr/bin/env python
"""
Comprehensive test script for model generation with detailed logging and error handling.

This script tests both direct Hugging Face model loading and the sentinel-ai adaptive models,
with verbose output at each step to diagnose any issues.

Usage:
    python scripts/test_model_generation.py [--model_name MODEL] [--use_adaptive] [--debug]
"""

import os
import sys
import argparse
import logging
import traceback
import time
from contextlib import redirect_stdout, redirect_stderr
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_generation_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
logger.info(f"Added {ROOT_DIR} to Python path")
logger.info(f"Current working directory: {os.getcwd()}")

def capture_output(func, *args, **kwargs):
    """Capture stdout and stderr from a function call"""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    result = None
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Exception in {func.__name__}: {e}")
        logger.error(traceback.format_exc())
    
    stdout_output = stdout_buffer.getvalue()
    stderr_output = stderr_buffer.getvalue()
    
    if stdout_output:
        logger.info(f"STDOUT from {func.__name__}:\n{stdout_output}")
    if stderr_output:
        logger.error(f"STDERR from {func.__name__}:\n{stderr_output}")
        
    return result, stdout_output, stderr_output

def test_huggingface_model(model_name, prompt, max_length=30, device=None, debug=False):
    """Test text generation with a standard Hugging Face model"""
    logger.info(f"Testing HuggingFace model: {model_name}")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Track timing
    start_time = time.time()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    logger.info(f"Model loaded. Type: {type(model).__name__}")
    
    if debug:
        # Print model information
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {param_count:,} parameters")
        logger.info(f"Model config: {model.config}")
    
    # Tokenize input
    logger.info(f"Tokenizing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    logger.info("Generating text...")
    try:
        with torch.no_grad():
            # Use context manager to prevent gradient computation
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode output
        logger.info("Decoding output...")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generation successful!")
        logger.info(f"Generated text: {generated_text}")
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        logger.info(f"Total generation time: {elapsed_time:.2f} seconds")
        
        return {
            "success": True,
            "text": generated_text,
            "time": elapsed_time
        }
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

def test_adaptive_model(model_name, prompt, max_length=30, device=None, debug=False):
    """Test text generation with the adaptive transformer model"""
    logger.info(f"Testing adaptive model based on {model_name}")
    
    import torch
    import sys
    from transformers import AutoTokenizer
    
    # Track timing
    start_time = time.time()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Try importing the model loader
    try:
        logger.info("Importing model loaders from sentinel namespace...")
        from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
        logger.info("Successfully imported from sentinel namespace")
    except ImportError as e:
        logger.error(f"Import error from sentinel namespace: {e}")
        try:
            # Fallback to old import path
            logger.info("Trying import from old namespace...")
            from models.loaders.loader import load_baseline_model, load_adaptive_model
            logger.info("Successfully imported from old namespace")
        except ImportError as e2:
            logger.error(f"Import error from old namespace too: {e2}")
            return {
                "success": False,
                "error": f"Cannot import model loaders: {e2}"
            }
    
    # Load baseline model
    logger.info(f"Loading baseline model {model_name}...")
    baseline_model = load_baseline_model(model_name, device)
    logger.info(f"Baseline model loaded. Type: {type(baseline_model).__name__}")
    
    # Convert to adaptive model
    logger.info("Converting to adaptive model...")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device, debug=debug)
    logger.info(f"Adaptive model loaded. Type: {type(adaptive_model).__name__}")
    
    if debug:
        # Print model structure
        if hasattr(adaptive_model, "blocks"):
            num_blocks = len(adaptive_model.blocks)
            num_heads = adaptive_model.blocks[0]["attn"].num_heads if num_blocks > 0 else 0
            logger.info(f"Adaptive model has {num_blocks} blocks with {num_heads} heads each")
            
            # Print gate values
            logger.info("Gate values:")
            for i, block in enumerate(adaptive_model.blocks):
                gate_values = block["attn"].gate.detach().cpu().tolist()
                logger.info(f"  Block {i}: {gate_values}")
    
    # Tokenize input
    logger.info(f"Tokenizing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    logger.info("Generating text with adaptive model...")
    try:
        with torch.no_grad():
            # First check if the model has a generate method
            if hasattr(adaptive_model, "generate"):
                outputs = adaptive_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                # If not, implement a simple custom generate function
                logger.info("No generate method found, implementing manual generation...")
                
                current_input = inputs.input_ids
                
                for _ in range(max_length - current_input.shape[1]):
                    # Forward pass to get next token probabilities
                    with torch.no_grad():
                        outputs = adaptive_model(current_input)
                    
                    # Get next token logits
                    if isinstance(outputs, torch.Tensor):
                        next_token_logits = outputs[:, -1, :] / 0.7  # Apply temperature
                    else:
                        next_token_logits = outputs.logits[:, -1, :] / 0.7
                    
                    # Sample from the distribution
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to the sequence
                    current_input = torch.cat([current_input, next_token], dim=1)
                
                outputs = current_input
        
        # Decode output
        logger.info("Decoding output...")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generation successful!")
        logger.info(f"Generated text: {generated_text}")
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        logger.info(f"Total generation time: {elapsed_time:.2f} seconds")
        
        return {
            "success": True,
            "text": generated_text,
            "time": elapsed_time
        }
    
    except Exception as e:
        logger.error(f"Error during adaptive model generation: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

def compare_versions():
    """Compare versions of key dependencies"""
    versions = {}
    
    logger.info("Checking versions of key dependencies...")
    
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "Not installed"
    
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "Not installed"
    
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        versions["numpy"] = "Not installed"
    
    try:
        import accelerate
        versions["accelerate"] = accelerate.__version__
    except ImportError:
        versions["accelerate"] = "Not installed"
    
    # Check Python version
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Log the versions
    logger.info("Dependency versions:")
    for name, version in versions.items():
        logger.info(f"  {name}: {version}")
    
    return versions

def main():
    """Run the test script"""
    parser = argparse.ArgumentParser(description="Test model generation")
    parser.add_argument("--model_name", type=str, default="distilgpt2", 
                        help="Model name to test (default: distilgpt2)")
    parser.add_argument("--prompt", type=str, default="The future of AI is",
                        help="Prompt for generation (default: 'The future of AI is')")
    parser.add_argument("--max_length", type=int, default=30,
                        help="Maximum length of generated text (default: 30)")
    parser.add_argument("--use_adaptive", action="store_true",
                        help="Test the adaptive model instead of standard HuggingFace")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug output")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--check_versions", action="store_true",
                        help="Check versions of key dependencies")
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Starting model generation test")
    logger.info(f"Arguments: {args}")
    logger.info("=" * 50)
    
    # Check dependency versions
    compare_versions()
    
    # Run the appropriate test
    if args.use_adaptive:
        result = test_adaptive_model(
            args.model_name, 
            args.prompt, 
            max_length=args.max_length, 
            device=args.device, 
            debug=args.debug
        )
    else:
        result = test_huggingface_model(
            args.model_name, 
            args.prompt, 
            max_length=args.max_length, 
            device=args.device, 
            debug=args.debug
        )
    
    # Print final result
    logger.info("=" * 50)
    if result["success"]:
        logger.info("TEST SUCCEEDED")
        logger.info(f"Generated text: {result['text']}")
        logger.info(f"Generation time: {result['time']:.2f} seconds")
    else:
        logger.error("TEST FAILED")
        logger.error(f"Error: {result['error']}")
    logger.info("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)