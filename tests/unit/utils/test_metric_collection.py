#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the metric collection module.

Tests the functionality of the MetricCollector class and related analysis functions.
"""

import unittest
import os
import tempfile
import shutil
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from sentinel.utils.metric_collection import MetricCollector, analyze_pruning_strategy, compare_pruning_strategies


class MockModel(torch.nn.Module):
    """Mock model for testing MetricCollector."""
    
    def __init__(self, num_layers=3, num_heads=4, hidden_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # Create mock blocks with attention heads and gates
        self.blocks = torch.nn.ModuleList([
            self._create_block(i) for i in range(num_layers)
        ])
    
    def _create_block(self, layer_idx):
        block = torch.nn.ModuleDict({
            "attn": self._create_attention(layer_idx),
            "ffn": torch.nn.Linear(self.hidden_size, self.hidden_size)
        })
        return block
    
    def _create_attention(self, layer_idx):
        attn = torch.nn.Module()
        attn.num_heads = self.num_heads
        attn.gate = torch.nn.Parameter(torch.ones(self.num_heads))
        
        # Add mock weights for testing
        attn.W_q = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size, self.hidden_size // self.num_heads)
            for _ in range(self.num_heads)
        ])
        attn.W_k = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size, self.hidden_size // self.num_heads)
            for _ in range(self.num_heads)
        ])
        attn.W_v = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size, self.hidden_size // self.num_heads)
            for _ in range(self.num_heads)
        ])
        attn.W_o = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size // self.num_heads, self.hidden_size)
            for _ in range(self.num_heads)
        ])
        
        # Add attention weights for testing
        attention_weights = {}
        for head_idx in range(self.num_heads):
            # Create dummy attention weights: [batch_size, seq_len, seq_len]
            weights = torch.softmax(torch.randn(2, 8, 8), dim=-1)
            attention_weights[head_idx] = weights
        
        attn.attention_weights = attention_weights
        
        return attn
    
    def forward(self, input_ids, attention_mask=None):
        # Mock forward pass that returns logits
        batch_size, seq_len = input_ids.shape
        # Create dummy logits: [batch_size, seq_len, vocab_size]
        logits = torch.randn(batch_size, seq_len, 1000)
        
        # Return an object with a logits attribute for compatibility
        class OutputWithLogits:
            def __init__(self, logits):
                self.logits = logits
        
        return OutputWithLogits(logits)


class TestMetricCollector(unittest.TestCase):
    """Test cases for the MetricCollector class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model
        self.model = MockModel()
        
        # Create test inputs
        self.input_ids = torch.randint(0, 1000, (2, 8))
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.randint(0, 1000, (2, 8))
        self.logits = torch.randn(2, 8, 1000)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test MetricCollector initialization."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Check that config file was created
        config_path = os.path.join(self.temp_dir, "metric_collector_config.json")
        self.assertTrue(os.path.exists(config_path))
        
        # Check config content
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.assertEqual(config["model_name"], "test_model")
        self.assertTrue(config["track_gate_values"])
        self.assertTrue(config["track_head_metrics"])
    
    def test_collect_step_metrics(self):
        """Test collecting metrics for a single step."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        # Collect metrics
        metrics = collector.collect_step_metrics(
            model=self.model,
            step=1,
            phase="test",
            inputs={"input_ids": self.input_ids, "attention_mask": self.attention_mask},
            labels=self.labels,
            logits=self.logits
        )
        
        # Check that metrics were collected
        self.assertIn("step", metrics)
        self.assertIn("phase", metrics)
        self.assertIn("timestamp", metrics)
        
        # Flush the metrics logger to ensure file is written
        collector.metrics_logger.flush()
        
        # Check that JSONL log file was created
        log_path = os.path.join(self.temp_dir, "test_model_metrics.jsonl")
        self.assertTrue(os.path.exists(log_path), f"Log file {log_path} doesn't exist")
        
        # Check that gate value files were created
        gate_files = [f for f in os.listdir(self.temp_dir) if "gate_values" in f]
        self.assertTrue(len(gate_files) > 0)
    
    def test_analyze_pruning_patterns(self):
        """Test analyzing pruning patterns."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        # Collect metrics for multiple steps
        for step in range(3):
            # Set different gate values for each step
            for i, block in enumerate(self.model.blocks):
                gate = block["attn"].gate
                # Create a pattern where more heads get pruned over time
                block["attn"].gate.data = torch.ones_like(gate) - 0.1 * step * (i + 1) / len(self.model.blocks)
            
            collector.collect_step_metrics(
                model=self.model,
                step=step,
                phase="test",
                inputs={"input_ids": self.input_ids, "attention_mask": self.attention_mask},
                labels=self.labels,
                logits=self.logits
            )
        
        # Analyze pruning patterns
        results = collector.analyze_pruning_patterns()
        
        # Check that results were generated
        self.assertIsInstance(results, dict)
        
        # Check that analysis file was created
        analysis_path = os.path.join(self.temp_dir, "pruning_pattern_analysis.json")
        self.assertTrue(os.path.exists(analysis_path))
    
    def test_gate_performance_correlation(self):
        """Test correlation analysis between gate values and performance."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        # Collect metrics for multiple steps with performance metrics
        for step in range(5):
            # Set different gate values for each step
            for i, block in enumerate(self.model.blocks):
                gate = block["attn"].gate
                # Create a pattern where gates decrease over time
                block["attn"].gate.data = torch.ones_like(gate) - 0.1 * step
            
            # Loss decreases as gates decrease (to simulate correlation)
            loss = 1.0 - 0.1 * step
            perplexity = torch.exp(torch.tensor(loss)).item()
            
            collector.collect_step_metrics(
                model=self.model,
                step=step,
                phase="test",
                inputs={"input_ids": self.input_ids, "attention_mask": self.attention_mask},
                labels=self.labels,
                logits=self.logits,
                additional_metrics={
                    "test/loss": loss,
                    "test/perplexity": perplexity
                }
            )
        
        # Analyze correlation
        results = collector.analyze_gate_performance_correlation()
        
        # Check that results were generated
        self.assertIsInstance(results, dict)
        
        # Check that correlation analysis file was created
        analysis_path = os.path.join(self.temp_dir, "gate_performance_correlation.json")
        self.assertTrue(os.path.exists(analysis_path))
    
    def test_compare_with_static_pruning(self):
        """Test comparison with static pruning strategies."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model",
            compare_with_static=True
        )
        
        # Collect performance metrics for adaptive pruning
        for step in range(3):
            collector.collect_step_metrics(
                model=self.model,
                step=step,
                phase="test",
                inputs={"input_ids": self.input_ids, "attention_mask": self.attention_mask},
                labels=self.labels,
                logits=self.logits,
                additional_metrics={
                    "test/loss": 0.5 - 0.1 * step,
                    "test/perplexity": 1.5 - 0.1 * step,
                    "test/accuracy": 0.7 + 0.1 * step
                }
            )
        
        # Register metrics for static pruning strategies
        collector.register_static_pruning_metrics("random_0.3", {
            "test/loss": 0.6,
            "test/perplexity": 1.8,
            "test/accuracy": 0.65
        })
        
        collector.register_static_pruning_metrics("entropy_0.3", {
            "test/loss": 0.4,
            "test/perplexity": 1.5,
            "test/accuracy": 0.75
        })
        
        # Compare with static pruning
        results = collector.compare_with_static_pruning(collector.static_pruning_metrics)
        
        # Check that results were generated
        self.assertIsInstance(results, dict)
        self.assertIn("comparisons", results)
        
        # Check that comparison file was created
        comparison_path = os.path.join(self.temp_dir, "pruning_strategy_comparison.json")
        self.assertTrue(os.path.exists(comparison_path))
    
    def test_generate_report(self):
        """Test generating a comprehensive report."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        # Collect some metrics data
        collector.collect_step_metrics(
            model=self.model,
            step=0,
            phase="test",
            inputs={"input_ids": self.input_ids, "attention_mask": self.attention_mask},
            labels=self.labels,
            logits=self.logits,
            additional_metrics={
                "test/loss": 0.5,
                "test/perplexity": 1.6,
                "test/accuracy": 0.75
            }
        )
        
        # Generate report
        report = collector.generate_report()
        
        # Check that report was generated
        self.assertIsInstance(report, dict)
        
        # Check that report file was created
        report_path = os.path.join(self.temp_dir, "comprehensive_analysis_report.json")
        self.assertTrue(os.path.exists(report_path))
    
    def test_save_metrics_csv(self):
        """Test saving metrics to CSV."""
        collector = MetricCollector(
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        # Collect some metrics
        collector.collect_step_metrics(
            model=self.model,
            step=0,
            phase="test",
            inputs={"input_ids": self.input_ids, "attention_mask": self.attention_mask},
            labels=self.labels,
            logits=self.logits,
            additional_metrics={
                "test/loss": 0.5,
                "test/perplexity": 1.6
            }
        )
        
        # Save to CSV
        csv_path = os.path.join(self.temp_dir, "test_metrics.csv")
        collector.save_metrics_csv(csv_path)
        
        # Check that CSV file was created
        self.assertTrue(os.path.exists(csv_path))


class TestAnalysisFunctions(unittest.TestCase):
    """Test cases for standalone analysis functions."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create mock metrics data
        self.metrics_file = os.path.join(self.temp_dir, "metrics.jsonl")
        with open(self.metrics_file, "w") as f:
            # Write several metrics entries
            for step in range(5):
                # Create mock metrics that improve over time
                metrics = {
                    "step": step,
                    "phase": "test",
                    "timestamp": "2025-04-01T12:00:00",
                    "test/loss": 1.0 - 0.1 * step,
                    "test/perplexity": 2.7 - 0.2 * step,
                    "test/accuracy": 0.6 + 0.05 * step,
                    "test/active_heads_ratio": 0.8 - 0.05 * step
                }
                f.write(json.dumps(metrics) + "\n")
        
        # Create directories for strategy comparison
        self.strategy_dirs = {}
        strategies = ["random", "entropy", "magnitude"]
        
        for strategy in strategies:
            strategy_dir = os.path.join(self.temp_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Create metrics file for this strategy
            metrics_file = os.path.join(strategy_dir, f"{strategy}_metrics.jsonl")
            with open(metrics_file, "w") as f:
                for step in range(5):
                    # Create metrics with different performance for each strategy
                    if strategy == "random":
                        loss = 1.0 - 0.05 * step  # Worst
                    elif strategy == "entropy":
                        loss = 0.9 - 0.1 * step   # Best
                    else:
                        loss = 0.95 - 0.08 * step  # Middle
                    
                    metrics = {
                        "step": step,
                        "phase": "test",
                        "timestamp": "2025-04-01T12:00:00",
                        "test/loss": loss,
                        "test/perplexity": 2.5 - 0.1 * step,
                        "test/accuracy": 0.65 + 0.03 * step
                    }
                    f.write(json.dumps(metrics) + "\n")
            
            self.strategy_dirs[strategy] = strategy_dir
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_pruning_strategy(self):
        """Test the analyze_pruning_strategy function."""
        results = analyze_pruning_strategy(self.temp_dir, self.output_dir)
        
        # Check that results were generated
        self.assertIsInstance(results, dict)
        self.assertIn("summary", results)
        self.assertIn("improvement", results)
        
        # Check that analysis files were created
        analysis_file = os.path.join(self.output_dir, "pruning_analysis.json")
        self.assertTrue(os.path.exists(analysis_file))
        
        # Check analysis content
        self.assertIn("test", results["summary"])
        self.assertIn("test/loss", results["summary"]["test"])
    
    def test_compare_pruning_strategies(self):
        """Test the compare_pruning_strategies function."""
        results = compare_pruning_strategies(self.strategy_dirs, self.output_dir)
        
        # Check that results were generated
        self.assertIsInstance(results, dict)
        self.assertIn("individual_analyses", results)
        self.assertIn("metric_comparison", results)
        self.assertIn("win_counts", results)
        
        # Check that comparison files were created
        comparison_file = os.path.join(self.output_dir, "strategy_comparison.json")
        self.assertTrue(os.path.exists(comparison_file))
        
        # Check that a best strategy was identified
        self.assertIn("overall_best_strategy", results)


if __name__ == "__main__":
    unittest.main()