#!/usr/bin/env python
"""
Test script that logs all output to a file to avoid terminal stderr issues.
"""

import os
import sys
import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Create log directory
os.makedirs("test_logs", exist_ok=True)
log_file = f"test_logs/test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
    ]
)
logger = logging.getLogger(__name__)

# Redirect stdout and stderr to log file
sys.stdout = open(log_file, 'a')
sys.stderr = open(log_file, 'a')

logger.info("TESTING MODEL FUNCTIONALITY")
logger.info("==========================")

try:
    # Add root to path
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT_DIR)
    
    # Import sentinel loaders
    try:
        from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
        logger.info("Using sentinel namespace imports")
    except ImportError:
        from models.loaders.loader import load_baseline_model, load_adaptive_model
        logger.info("Using models directory imports")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Test BLOOM (to verify cross-model compatibility)
    model_name = "bigscience/bloom-560m"
    logger.info(f"Testing model: {model_name}")
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    logger.info("Loading baseline model")
    baseline_model = load_baseline_model(model_name, device)
    logger.info(f"Baseline model loaded, type: {type(baseline_model)}")
    
    # Convert to adaptive model
    logger.info("Converting to adaptive model")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device)
    logger.info(f"Adaptive model loaded, type: {type(adaptive_model)}")
    
    # Test generation
    prompt = "The future of AI is"
    logger.info(f"Generating text for prompt: '{prompt}'")
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with adaptive model
    logger.info("Generating with adaptive model")
    with torch.no_grad():
        outputs = adaptive_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=30,
            do_sample=True
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    
    # Check output quality
    if len(generated_text) > len(prompt) + 5:
        logger.info("✅ SUCCESS: Model successfully generated text")
    else:
        logger.error("❌ FAILURE: Model failed to generate meaningful text")
    
    logger.info("Test completed successfully")
    
except Exception as e:
    logger.error(f"Error during test: {e}")
    import traceback
    logger.error(traceback.format_exc())
    logger.error("Test failed")

finally:
    # Print path to log file
    print(f"Test completed. See results in: {os.path.abspath(log_file)}")