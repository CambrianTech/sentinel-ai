# sentinel/mock/dataset_mocks.py
"""
Mock implementations of datasets classes.

This module provides mock implementations of datasets classes for testing
without requiring the actual datasets library.
"""

import sys
from typing import Dict, List, Any, Optional, Union, Callable

class MockFeatures:
    """Mock implementation of datasets.Features"""
    
    def __init__(self, features_dict=None):
        self.features = features_dict or {}
        
    def __getitem__(self, key):
        return self.features.get(key)

class MockArrowTable:
    """Mock implementation of datasets arrow table"""
    
    def __init__(self, data=None):
        self.data = data or {}
        
    def __getitem__(self, key):
        return self.data.get(key, [])
        
    def to_pandas(self):
        """Convert to pandas DataFrame (mock)"""
        return self.data

class MockDataset:
    """Mock implementation of datasets.Dataset"""
    
    def __init__(self, data=None, features=None):
        self.data = data or {"text": ["Sample text 1", "Sample text 2"]}
        self.features = features or MockFeatures()
        self.shape = (len(list(self.data.values())[0]), len(self.data))
        
    def __len__(self):
        return self.shape[0]
        
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {k: v[idx] for k, v in self.data.items()}
        return self
        
    def map(self, function, **kwargs):
        """Mock map method"""
        return self
        
    def filter(self, function, **kwargs):
        """Mock filter method"""
        return self
        
    def select(self, indices, **kwargs):
        """Mock select method"""
        return self
        
    def shuffle(self, **kwargs):
        """Mock shuffle method"""
        return self
        
    def train_test_split(self, **kwargs):
        """Mock train_test_split method"""
        return {
            "train": MockDataset(self.data, self.features),
            "test": MockDataset(self.data, self.features)
        }

def mock_load_dataset(path, **kwargs):
    """Mock implementation of datasets.load_dataset"""
    return MockDataset()

def setup_dataset_mocks():
    """
    Set up mock modules for datasets.
    
    This function should be called before any imports of datasets
    to ensure that the mock implementations are used.
    """
    # Create a simple mock module system
    class MockModule:
        def __init__(self, name):
            self.__name__ = name
            
        def __getattr__(self, name):
            return MockModule(f"{self.__name__}.{name}")
    
    # Create datasets mock module
    datasets_mock = MockModule("datasets")
    
    # Add specific classes and functions to the mock
    datasets_mock.Dataset = MockDataset
    datasets_mock.Features = MockFeatures
    datasets_mock.load_dataset = mock_load_dataset
    
    # Mock the arrow module
    arrow_mock = MockModule("datasets.arrow")
    arrow_mock.Table = MockArrowTable
    
    # Add arrow module to datasets
    datasets_mock.arrow = arrow_mock
    
    # Register the mock module
    sys.modules["datasets"] = datasets_mock