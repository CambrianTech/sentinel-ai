#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the modular pruning experiment framework.

These tests verify that the PruningExperiment and PruningFineTuningExperiment classes
work correctly with various configurations. Tests are designed to run locally without
requiring a GPU, using small models and minimal iterations.

To run these tests:
    python -m unittest examples/pruning_experiments/tests/test_pruning_experiment.py
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parents[3].absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Need to ensure HuggingFace datasets is imported before our local datasets module
try:
    from datasets import load_dataset
except ImportError as e:
    # If our local import is shadowing HuggingFace, we need to fix sys.path
    if 'cannot import name' in str(e) and 'from datasets' in str(e):
        # Temporarily remove the project root from path and import HuggingFace datasets
        sys.path.remove(str(project_root))
        import datasets
        # Then restore the path
        sys.path.append(str(project_root))
    else:
        # Re-raise other import errors
        raise

# Import experiment framework
from utils.pruning import PruningExperiment, PruningFineTuningExperiment, Environment
from utils.pruning.pruning_module import PruningModule

# Disable most logging during tests
logging.basicConfig(level=logging.ERROR)


class TestPruningExperiment(unittest.TestCase):
    """Test cases for the PruningExperiment class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_model = "distilgpt2"  # Small model for faster testing
        cls.test_strategy = "random"   # Fast strategy for testing
        cls.test_prompt = "Testing is"
        
        # Create a temp directory for test results
        cls.temp_dir = tempfile.mkdtemp()
        cls.results_dir = os.path.join(cls.temp_dir, "test_results")
        os.makedirs(cls.results_dir, exist_ok=True)
        
        # Only run extensive tests if the model can be loaded
        test_module = PruningModule(cls.test_model)
        cls.skip_model_tests = not test_module.load_model()
        if cls.skip_model_tests:
            print(f"WARNING: Could not load model {cls.test_model}, skipping intensive tests")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove temp directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_experiment_initialization(self):
        """Test basic initialization of PruningExperiment"""
        experiment = PruningExperiment(
            results_dir=self.results_dir,
            use_improved_fine_tuner=True,
            detect_environment=True,
            optimize_memory=True
        )
        
        # Verify basic properties
        self.assertEqual(experiment.results_dir, Path(self.results_dir))
        self.assertTrue(experiment.use_improved_fine_tuner)
        self.assertTrue(experiment.optimize_memory)
        self.assertIsNotNone(experiment.env)
        
        # Verify results directory was created
        self.assertTrue(os.path.exists(self.results_dir))
    
    def test_environment_detection(self):
        """Test environment detection functionality"""
        experiment = PruningExperiment(
            results_dir=self.results_dir,
            detect_environment=True
        )
        
        # Verify hardware detection
        self.assertIsNotNone(experiment.env)
        self.assertIsInstance(experiment.gpu_memory_gb, float)
        
        # Check if the right models are available
        if experiment.available_models:
            # At minimum, small models should be available
            self.assertIn("distilgpt2", experiment.available_models)
    
    @unittest.skipIf(not os.environ.get("RUN_INTENSIVE_TESTS"), "Skipping intensive test")
    def test_single_experiment_run(self):
        """Test running a single experiment (intensive test)"""
        if self.skip_model_tests:
            self.skipTest("Model could not be loaded")
            
        experiment = PruningExperiment(
            results_dir=self.results_dir,
            use_improved_fine_tuner=True,
            detect_environment=True,
            optimize_memory=True
        )
        
        # Run a minimal experiment (no fine-tuning)
        result = experiment.run_single_experiment(
            model=self.test_model,
            strategy=self.test_strategy,
            pruning_level=0.1,  # Minimal pruning
            prompt=self.test_prompt,
            fine_tuning_epochs=0,  # Skip fine-tuning to make test faster
            save_results=True
        )
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertEqual(result["model"], self.test_model)
        self.assertEqual(result["strategy"], self.test_strategy)
        self.assertEqual(result["pruning_level"], 0.1)
        
        # Verify stages
        self.assertIn("baseline", result["stages"])
        self.assertIn("pruned", result["stages"])
        
        # Check that results were saved
        result_files = os.listdir(self.results_dir)
        self.assertTrue(any(f.endswith(".json") for f in result_files))

    def test_multi_experiment_initialization(self):
        """Test initialization of PruningFineTuningExperiment"""
        experiment = PruningFineTuningExperiment(
            results_dir=self.results_dir,
            use_improved_fine_tuner=True,
            detect_environment=True,
            optimize_memory=True
        )
        
        # Verify basic properties
        self.assertEqual(experiment.results_dir, Path(self.results_dir))
        self.assertTrue(experiment.use_improved_fine_tuner)
        self.assertTrue(experiment.optimize_memory)
        self.assertIsNotNone(experiment.env)
        
        # Verify model size limits are set
        self.assertIsNotNone(experiment.model_size_limits)
        self.assertGreater(len(experiment.model_size_limits), 0)
        
        # Check distilgpt2 is allowed at full pruning level
        self.assertEqual(experiment.model_size_limits.get("distilgpt2", 0), 1.0)
    
    def test_update_model_size_limits(self):
        """Test dynamic updating of model size limits"""
        experiment = PruningFineTuningExperiment(
            results_dir=self.results_dir,
            detect_environment=True
        )
        
        # Store original limits
        original_limits = experiment.model_size_limits.copy()
        
        # Update limits
        experiment.model_size_limits["test_model"] = 0.5
        experiment.update_model_size_limits()
        
        # Verify update preserved our custom setting
        self.assertEqual(experiment.model_size_limits.get("test_model", 0), 0.5)
        
        # Verify base models are still there
        for model, limit in original_limits.items():
            self.assertIn(model, experiment.model_size_limits)


# Only run this more intensive test class if specifically requested
@unittest.skipIf(not os.environ.get("RUN_INTENSIVE_TESTS"), "Skipping intensive tests")
class TestIntegrationExperiment(unittest.TestCase):
    """Integration tests for experiment framework with real models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_model = "distilgpt2"  # Small model for faster testing
        
        # Create a temp directory for test results
        cls.temp_dir = tempfile.mkdtemp()
        cls.results_dir = os.path.join(cls.temp_dir, "integration_results")
        os.makedirs(cls.results_dir, exist_ok=True)
        
        # Verify model loading
        test_module = PruningModule(cls.test_model)
        cls.skip_tests = not test_module.load_model()
        if cls.skip_tests:
            print(f"WARNING: Could not load model {cls.test_model}, skipping integration tests")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove temp directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_full_single_experiment(self):
        """Test a full experiment cycle with pruning and fine-tuning"""
        if self.skip_tests:
            self.skipTest("Model could not be loaded")
            
        experiment = PruningExperiment(
            results_dir=self.results_dir,
            use_improved_fine_tuner=True,
            detect_environment=True,
            optimize_memory=True
        )
        
        # Run a minimal but complete experiment
        result = experiment.run_single_experiment(
            model=self.test_model,
            strategy="random",
            pruning_level=0.1,
            prompt="Testing the experiment framework is",
            fine_tuning_epochs=1,  # Minimal fine-tuning
            save_results=True
        )
        
        # Verify all stages are present
        self.assertIn("baseline", result["stages"])
        self.assertIn("pruned", result["stages"])
        self.assertIn("fine_tuned", result["stages"])
        
        # Verify perplexity data
        self.assertIn("perplexity", result["stages"]["baseline"])
        self.assertIn("perplexity", result["stages"]["pruned"])
        self.assertIn("perplexity", result["stages"]["fine_tuned"])
        
        # Verify metrics
        if "recovery_percentage" in result["stages"]["fine_tuned"]:
            # Recovery means pruning increased perplexity and fine-tuning helped
            recovery = result["stages"]["fine_tuned"]["recovery_percentage"]
            self.assertIsInstance(recovery, float)
        elif "improvement_percentage" in result["stages"]["fine_tuned"]:
            # Improvement means pruning helped and fine-tuning helped more
            improvement = result["stages"]["fine_tuned"]["improvement_percentage"]
            self.assertIsInstance(improvement, float)
    
    def test_multi_experiment_config(self):
        """Test multi-experiment configuration logic"""
        if self.skip_tests:
            self.skipTest("Model could not be loaded")
            
        experiment = PruningFineTuningExperiment(
            results_dir=self.results_dir,
            use_improved_fine_tuner=True,
            detect_environment=True,
            optimize_memory=True
        )
        
        # Test with minimal configuration
        strategies = ["random"]
        pruning_levels = [0.1]
        
        # Configure with minimal runtime
        experiment.run_experiment(
            strategies=strategies,
            pruning_levels=pruning_levels,
            prompt="Testing multi-experiment is",
            fine_tuning_epochs=0,  # Skip fine-tuning
            max_runtime=1,  # 1 second runtime limit to test timeout logic
            models=[self.test_model]
        )
        
        # Verify results data structure
        self.assertIsNotNone(experiment.results_df)


if __name__ == "__main__":
    # Enable intensive tests if run directly
    os.environ["RUN_INTENSIVE_TESTS"] = "1"
    unittest.main()