"""
Tests for metrics handling in the experiment runner.

This ensures that the experiment runner correctly handles different metric formats
(tuples vs dictionaries) to prevent TypeError issues.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import the function to test
# Use a more isolated import approach to avoid circular imports
import importlib.util
import sys

# Dynamically import the module we need to test
spec = importlib.util.spec_from_file_location(
    "experiment_runner", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../utils/pruning/experiment_runner.py'))
)
experiment_runner = importlib.util.module_from_spec(spec)
sys.modules["experiment_runner"] = experiment_runner
spec.loader.exec_module(experiment_runner)

# Get the functions/classes we need
run_experiment = experiment_runner.run_experiment
ExperimentConfig = experiment_runner.ExperimentConfig


class TestMetricsHandling(unittest.TestCase):
    """Test the metrics handling in experiment_runner.py."""

    @patch('experiment_runner.evaluate_model')
    @patch('experiment_runner.fine_tune')
    @patch('experiment_runner.entropy_based_pruning')
    @patch('experiment_runner.collect_attention_distributions')
    @patch('experiment_runner.prepare_data')
    @patch('experiment_runner.generate_text')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_tuple_metrics_handling(self, mock_model, mock_tokenizer, mock_generate, 
                                    mock_prepare_data, mock_collect_attn, mock_pruning,
                                    mock_fine_tune, mock_evaluate):
        """Test that the experiment runner handles tuple-format metrics correctly."""
        # Setup mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_prepare_data.return_value = MagicMock()
        mock_collect_attn.return_value = {}
        mock_pruning.return_value = []
        mock_fine_tune.return_value = (1.5, 4.5)  # (loss, perplexity)
        
        # Mock evaluate_model to return tuples (loss, perplexity)
        mock_evaluate.side_effect = [
            (2.0, 7.4),  # Baseline metrics (loss, perplexity)
            (2.2, 9.1),  # Pruned metrics
            (1.8, 6.0)   # Fine-tuned metrics
        ]
        
        # Create config
        config = ExperimentConfig(
            model_name="dummy-model",
            pruning_percent=0.3,
            pruning_strategy="entropy",
            num_epochs=1,
            batch_size=2,
            use_test_data=True,
            device=torch.device("cpu")
        )
        
        # Run the experiment
        model, tokenizer, summary = run_experiment(config)
        
        # Verify metrics were handled correctly
        self.assertIn("baseline", summary)
        self.assertIn("loss", summary["baseline"])
        self.assertIn("perplexity", summary["baseline"])
        self.assertEqual(summary["baseline"]["loss"], 2.0)
        self.assertEqual(summary["baseline"]["perplexity"], 7.4)
        
        self.assertIn("pruned", summary)
        self.assertIn("loss", summary["pruned"])
        self.assertIn("perplexity", summary["pruned"])
        self.assertEqual(summary["pruned"]["loss"], 2.2)
        self.assertEqual(summary["pruned"]["perplexity"], 9.1)
        
        self.assertIn("finetuned", summary)
        self.assertIn("loss", summary["finetuned"])
        self.assertIn("perplexity", summary["finetuned"])
        self.assertEqual(summary["finetuned"]["loss"], 1.8)
        self.assertEqual(summary["finetuned"]["perplexity"], 6.0)
        
        self.assertIn("improvement", summary)
        # Overall improvement: ((baseline_loss - finetuned_loss) / baseline_loss) * 100
        expected_improvement = ((2.0 - 1.8) / 2.0) * 100
        self.assertAlmostEqual(summary["improvement"]["overall_percent"], expected_improvement)

    @patch('experiment_runner.evaluate_model')
    @patch('experiment_runner.fine_tune')
    @patch('experiment_runner.entropy_based_pruning')
    @patch('experiment_runner.collect_attention_distributions')
    @patch('experiment_runner.prepare_data')
    @patch('experiment_runner.generate_text')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_dict_metrics_handling(self, mock_model, mock_tokenizer, mock_generate, 
                                   mock_prepare_data, mock_collect_attn, mock_pruning,
                                   mock_fine_tune, mock_evaluate):
        """Test that the experiment runner handles dictionary-format metrics correctly."""
        # Setup mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_prepare_data.return_value = MagicMock()
        mock_collect_attn.return_value = {}
        mock_pruning.return_value = []
        mock_fine_tune.return_value = (1.5, 4.5)  # (loss, perplexity)
        
        # Mock evaluate_model to return dictionaries
        mock_evaluate.side_effect = [
            {"loss": 2.0, "perplexity": 7.4},  # Baseline metrics
            {"loss": 2.2, "perplexity": 9.1},  # Pruned metrics
            {"loss": 1.8, "perplexity": 6.0}   # Fine-tuned metrics
        ]
        
        # Create config
        config = ExperimentConfig(
            model_name="dummy-model",
            pruning_percent=0.3,
            pruning_strategy="entropy",
            num_epochs=1,
            batch_size=2,
            use_test_data=True,
            device=torch.device("cpu")
        )
        
        # Run the experiment
        model, tokenizer, summary = run_experiment(config)
        
        # Verify metrics were handled correctly
        self.assertIn("baseline", summary)
        self.assertIn("loss", summary["baseline"])
        self.assertIn("perplexity", summary["baseline"])
        self.assertEqual(summary["baseline"]["loss"], 2.0)
        self.assertEqual(summary["baseline"]["perplexity"], 7.4)
        
        self.assertIn("pruned", summary)
        self.assertIn("loss", summary["pruned"])
        self.assertIn("perplexity", summary["pruned"])
        self.assertEqual(summary["pruned"]["loss"], 2.2)
        self.assertEqual(summary["pruned"]["perplexity"], 9.1)
        
        self.assertIn("finetuned", summary)
        self.assertIn("loss", summary["finetuned"])
        self.assertIn("perplexity", summary["finetuned"])
        self.assertEqual(summary["finetuned"]["loss"], 1.8)
        self.assertEqual(summary["finetuned"]["perplexity"], 6.0)
        
        self.assertIn("improvement", summary)
        # Overall improvement: ((baseline_loss - finetuned_loss) / baseline_loss) * 100
        expected_improvement = ((2.0 - 1.8) / 2.0) * 100
        self.assertAlmostEqual(summary["improvement"]["overall_percent"], expected_improvement)

    @patch('experiment_runner.evaluate_model')
    @patch('experiment_runner.fine_tune')
    @patch('experiment_runner.entropy_based_pruning')
    @patch('experiment_runner.collect_attention_distributions')
    @patch('experiment_runner.prepare_data')
    @patch('experiment_runner.generate_text')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_mixed_metrics_handling(self, mock_model, mock_tokenizer, mock_generate, 
                                    mock_prepare_data, mock_collect_attn, mock_pruning,
                                    mock_fine_tune, mock_evaluate):
        """Test that the experiment runner handles mixed format metrics correctly."""
        # Setup mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_prepare_data.return_value = MagicMock()
        mock_collect_attn.return_value = {}
        mock_pruning.return_value = []
        mock_fine_tune.return_value = (1.5, 4.5)  # (loss, perplexity)
        
        # Mock evaluate_model to return mixed formats
        mock_evaluate.side_effect = [
            (2.0, 7.4),  # Baseline metrics as tuple
            {"loss": 2.2, "perplexity": 9.1},  # Pruned metrics as dict
            (1.8, 6.0)   # Fine-tuned metrics as tuple
        ]
        
        # Create config
        config = ExperimentConfig(
            model_name="dummy-model",
            pruning_percent=0.3,
            pruning_strategy="entropy",
            num_epochs=1,
            batch_size=2,
            use_test_data=True,
            device=torch.device("cpu")
        )
        
        # Run the experiment
        model, tokenizer, summary = run_experiment(config)
        
        # Verify metrics were handled correctly
        self.assertIn("baseline", summary)
        self.assertIn("loss", summary["baseline"])
        self.assertIn("perplexity", summary["baseline"])
        self.assertEqual(summary["baseline"]["loss"], 2.0)
        self.assertEqual(summary["baseline"]["perplexity"], 7.4)
        
        self.assertIn("pruned", summary)
        self.assertIn("loss", summary["pruned"])
        self.assertIn("perplexity", summary["pruned"])
        self.assertEqual(summary["pruned"]["loss"], 2.2)
        self.assertEqual(summary["pruned"]["perplexity"], 9.1)
        
        self.assertIn("finetuned", summary)
        self.assertIn("loss", summary["finetuned"])
        self.assertIn("perplexity", summary["finetuned"])
        self.assertEqual(summary["finetuned"]["loss"], 1.8)
        self.assertEqual(summary["finetuned"]["perplexity"], 6.0)
        
        self.assertIn("improvement", summary)
        # Overall improvement: ((baseline_loss - finetuned_loss) / baseline_loss) * 100
        expected_improvement = ((2.0 - 1.8) / 2.0) * 100
        self.assertAlmostEqual(summary["improvement"]["overall_percent"], expected_improvement)


if __name__ == '__main__':
    unittest.main()