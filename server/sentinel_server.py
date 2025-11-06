#!/usr/bin/env python3
"""
Sentinel HTTP Server
====================

Simple HTTP server for Sentinel model inference.
Mimics Ollama's API design for easy integration.

Endpoints:
- POST /api/generate - Generate text
- GET /api/tags - List available models
- GET /api/health - Health check

Default port: 11435

Usage:
    python server/sentinel_server.py
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, Response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sentinel-server')

app = Flask(__name__)

# Global model cache
class ModelCache:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}

        # Prefer MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

    def load_model(self, model_name: str) -> tuple:
        """Load model and tokenizer, caching for reuse"""
        if model_name in self.models:
            logger.info(f"Using cached model: {model_name}")
            return self.models[model_name], self.tokenizers[model_name]

        logger.info(f"Loading model: {model_name}")
        start_time = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Use float16 for GPU (CUDA/MPS), float32 for CPU
            use_fp16 = self.device in ["cuda", "mps"]
            dtype = torch.float16 if use_fp16 else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)

            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded in {load_time:.2f}s")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

cache = ModelCache()

# Available models (can be extended)
AVAILABLE_MODELS = [
    {
        "name": "gpt2",
        "size": "124M",
        "family": "gpt2",
        "modified_at": "2025-01-01T00:00:00Z"
    },
    {
        "name": "distilgpt2",
        "size": "82M",
        "family": "gpt2",
        "modified_at": "2025-01-01T00:00:00Z"
    },
    {
        "name": "microsoft/phi-2",
        "size": "2.7B",
        "family": "phi",
        "modified_at": "2025-01-01T00:00:00Z"
    },
]

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate text from prompt

    Request body:
    {
        "model": "gpt2",
        "prompt": "Once upon a time",
        "system": "You are a helpful assistant",  # optional
        "temperature": 0.7,
        "num_predict": 150,
        "stream": false
    }

    Response:
    {
        "model": "gpt2",
        "response": "generated text...",
        "done": true,
        "context": [token_ids],
        "total_duration": 1234567890,
        "load_duration": 123456,
        "prompt_eval_duration": 12345,
        "eval_duration": 123456
    }
    """
    try:
        data = request.get_json()

        model_name = data.get('model', 'gpt2')
        prompt = data.get('prompt', '')
        system = data.get('system', '')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('num_predict', 150)

        logger.info(f"Generate request: model={model_name}, prompt_len={len(prompt)}, max_tokens={max_tokens}")
        logger.info(f"PROMPT: {repr(prompt)[:500]}")

        # Load model
        load_start = time.time()
        model, tokenizer = cache.load_model(model_name)
        load_duration = int((time.time() - load_start) * 1e9)  # nanoseconds

        # Build full prompt
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # Tokenize
        prompt_eval_start = time.time()
        inputs = tokenizer(full_prompt, return_tensors="pt").to(cache.device)
        prompt_eval_duration = int((time.time() - prompt_eval_start) * 1e9)

        # Generate
        eval_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        eval_duration = int((time.time() - eval_start) * 1e9)

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output (return only generated part)
        if generated_text.startswith(full_prompt):
            response_text = generated_text[len(full_prompt):].strip()
        else:
            response_text = generated_text

        total_duration = load_duration + prompt_eval_duration + eval_duration

        result = {
            "model": model_name,
            "response": response_text,
            "done": True,
            "context": outputs[0].tolist(),
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_duration": eval_duration
        }

        logger.info(f"Generated {len(response_text)} chars in {total_duration/1e9:.2f}s")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/tags', methods=['GET'])
def list_models():
    """
    List available models

    Response:
    {
        "models": [
            {
                "name": "gpt2",
                "size": "124M",
                "family": "gpt2",
                "modified_at": "2025-01-01T00:00:00Z"
            },
            ...
        ]
    }
    """
    return jsonify({"models": AVAILABLE_MODELS})

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": cache.device,
        "loaded_models": list(cache.models.keys())
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - server info"""
    return jsonify({
        "name": "Sentinel AI Server",
        "version": "0.1.0",
        "endpoints": {
            "generate": "/api/generate",
            "models": "/api/tags",
            "health": "/api/health"
        }
    })

def main():
    """Start the server"""
    port = int(os.environ.get('SENTINEL_PORT', 11435))
    host = os.environ.get('SENTINEL_HOST', '127.0.0.1')

    logger.info("=" * 60)
    logger.info("Sentinel AI HTTP Server")
    logger.info("=" * 60)
    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"Device: {cache.device}")
    logger.info(f"Available models: {len(AVAILABLE_MODELS)}")
    logger.info("=" * 60)

    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    main()
