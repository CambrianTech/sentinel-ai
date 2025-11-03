"""
Pytest configuration file for sentinel-ai tests.

This file sets up fixtures and configuration for all tests in the project.
"""

import os
import sys
import pytest
import torch
from unittest.mock import MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def mock_model():
    """Create a mock transformer model for testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.num_hidden_layers = 4
    model.config.num_attention_heads = 4
    model.config.model_type = "gpt2"
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 50256
    return tokenizer


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    # Create mock data
    input_ids = torch.randint(0, 1000, (10, 32))
    attention_mask = torch.ones_like(input_ids)
    batch = [input_ids, attention_mask]
    
    # Create mock dataloader
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([batch])
    dataloader.__len__.return_value = 5
    return dataloader


@pytest.fixture
def temp_output_dir(tmpdir):
    """Create a temporary output directory."""
    output_dir = tmpdir.mkdir("output")
    return str(output_dir)