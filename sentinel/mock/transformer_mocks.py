# sentinel/mock/transformer_mocks.py
"""
Mock implementations of transformers classes.

This module provides mock implementations of transformers classes for testing
without requiring the actual transformers library.
"""

import sys
from typing import Dict, List, Any, Optional, Union, Callable

class MockPreTrainedTokenizer:
    """Mock implementation of PreTrainedTokenizer"""
    
    def __init__(self, *args, **kwargs):
        self.vocab_size = 50000
        self.model_max_length = 1024
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"
        
    def encode(self, text, **kwargs):
        """Mock encode method"""
        # Just return a simple list of tokens for testing
        return [i for i in range(min(10, len(text)))]
        
    def decode(self, token_ids, **kwargs):
        """Mock decode method"""
        # Just return a simple string for testing
        return "Decoded text"
        
    def batch_encode_plus(self, texts, **kwargs):
        """Mock batch encode method"""
        return {
            "input_ids": [[i for i in range(10)] for _ in texts],
            "attention_mask": [[1] * 10 for _ in texts]
        }

    def save_pretrained(self, path):
        """Mock save_pretrained method"""
        pass

class MockOutput:
    """Mock output from model forward pass"""
    
    def __init__(self, loss=0.5):
        self.loss = loss
        self.logits = [[0.1, 0.2, 0.7] for _ in range(10)]

class MockPreTrainedModel:
    """Mock implementation of PreTrainedModel"""
    
    def __init__(self, config=None):
        self.config = config or MockConfig()
        
    def __call__(self, **kwargs):
        """Mock forward pass"""
        return MockOutput()
        
    def generate(self, *args, **kwargs):
        """Mock generate method"""
        # Just return a simple list of token ids
        return [[i for i in range(10)] for _ in range(kwargs.get("batch_size", 1))]
        
    def to(self, device):
        """Mock to method"""
        return self
        
    def eval(self):
        """Mock eval method"""
        return self
        
    def train(self):
        """Mock train method"""
        return self
        
    def parameters(self):
        """Mock parameters method"""
        # Yield a few dummy parameters
        for _ in range(3):
            yield [0.1, 0.2, 0.3]
            
    def save_pretrained(self, path):
        """Mock save_pretrained method"""
        pass

class MockTrainingArguments:
    """Mock implementation of TrainingArguments"""
    
    def __init__(
        self,
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        **kwargs
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        
        # Add any other kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockTrainer:
    """Mock implementation of Trainer"""
    
    def __init__(
        self,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        **kwargs
    ):
        self.model = model or MockPreTrainedModel()
        self.args = args or MockTrainingArguments()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
    def train(self, *args, **kwargs):
        """Mock train method"""
        return {
            "training_loss": 0.5,
            "epoch": 3.0
        }
        
    def evaluate(self, *args, **kwargs):
        """Mock evaluate method"""
        return {
            "eval_loss": 0.4,
            "perplexity": 10.0,
            "epoch": 3.0
        }

class MockConfig:
    """Mock implementation of PretrainedConfig"""
    
    def __init__(self, **kwargs):
        self.vocab_size = 50000
        self.n_positions = 1024
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        
        # Add any other kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

def setup_transformer_mocks():
    """
    Set up mock modules for transformers.
    
    This function should be called before any imports of transformers
    to ensure that the mock implementations are used.
    """
    # Create a simple mock module system
    class MockModule:
        def __init__(self, name):
            self.__name__ = name
            
        def __getattr__(self, name):
            return MockModule(f"{self.__name__}.{name}")
    
    # Create transformers mock module
    transformers_mock = MockModule("transformers")
    
    # Add specific classes to the mock
    transformers_mock.PreTrainedTokenizer = MockPreTrainedTokenizer
    transformers_mock.PreTrainedModel = MockPreTrainedModel
    transformers_mock.Trainer = MockTrainer
    transformers_mock.TrainingArguments = MockTrainingArguments
    
    # Register the mock module
    sys.modules["transformers"] = transformers_mock