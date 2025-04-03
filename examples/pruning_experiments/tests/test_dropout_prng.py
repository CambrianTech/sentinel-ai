import sys
import unittest
import unittest.mock as mock
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parents[3].absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import mock dependencies to avoid actual model loading
from examples.pruning_experiments.tests.mock_dependencies import apply_mocks
apply_mocks()

# Now import the module to test
import jax
import jax.numpy as jnp
from utils.pruning.fine_tuner_improved import ImprovedFineTuner


class TestDropoutPRNGHandling(unittest.TestCase):
    """Test cases for dropout PRNG key handling in ImprovedFineTuner."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the pruning module
        self.mock_pruning_module = mock.MagicMock()
        self.mock_pruning_module.model_name = "test-model"
        self.mock_pruning_module.model = mock.MagicMock()
        
        # Create tokenizer mock
        tokenizer_mock = mock.MagicMock()
        tokenizer_mock.pad_token_id = 0
        tokenizer_mock.vocab_size = 50000
        self.mock_pruning_module.tokenizer = tokenizer_mock
        
        # Create the fine tuner instance
        self.fine_tuner = ImprovedFineTuner(
            pruning_module=self.mock_pruning_module,
            batch_size=2,
            dataset_name="test_dataset"
        )
        
        # Ensure we use synthetic data
        self.fine_tuner.use_synthetic_data = True
        
        # Mock the model outputs
        outputs_mock = mock.MagicMock()
        # Create fake logits (batch_size x sequence_length x vocab_size)
        outputs_mock.logits = jnp.zeros((2, 10, 50000))
        self.mock_pruning_module.model.return_value = outputs_mock
        
    def test_train_step_dropout_rng_handling(self):
        """Test that _train_step correctly handles dropout PRNG keys."""
        # Enable RNG keys for dropout
        self.fine_tuner.use_rng_keys_for_dropout = True
        
        # Create a fake batch
        batch = {
            "input_ids": jnp.ones((2, 10), dtype=jnp.int32),
            "attention_mask": jnp.ones((2, 10), dtype=jnp.int32),
            "labels": jnp.ones((2, 10), dtype=jnp.int32)
        }
        
        # Create a train state
        params = {}  # Empty params for testing
        self.fine_tuner.train_state = self.fine_tuner._create_train_state(params)
        
        # Create initial PRNG key
        initial_rng = jax.random.PRNGKey(42)
        
        # Run a training step
        new_state, loss, new_rng = self.fine_tuner._train_step(
            self.fine_tuner.train_state, batch, initial_rng
        )
        
        # Verify that:
        # 1. A new RNG key was returned
        self.assertIsNotNone(new_rng)
        # 2. The new key is different from the initial key
        self.assertFalse(jnp.array_equal(initial_rng, new_rng))
        
        # Verify the model was called with dropout_rng parameter
        model_call_args = self.mock_pruning_module.model.call_args
        self.assertIsNotNone(model_call_args)
        
        # Check kwargs contains dropout_rng
        _, kwargs = model_call_args
        self.assertIn("dropout_rng", kwargs)
    
    def test_loss_fn_dropout_rng_passing(self):
        """Test that _loss_fn correctly passes dropout PRNG key to the model."""
        # Enable RNG keys for dropout
        self.fine_tuner.use_rng_keys_for_dropout = True
        
        # Create a fake batch with dropout_rng
        dropout_rng = jax.random.PRNGKey(42)
        batch = {
            "input_ids": jnp.ones((2, 10), dtype=jnp.int32),
            "attention_mask": jnp.ones((2, 10), dtype=jnp.int32),
            "labels": jnp.ones((2, 10), dtype=jnp.int32),
            "dropout_rng": dropout_rng
        }
        
        # Call loss_fn
        params = {}  # Empty params for testing
        loss = self.fine_tuner._loss_fn(params, batch.copy())
        
        # Verify the model was called with dropout_rng parameter
        model_call_args = self.mock_pruning_module.model.call_args
        self.assertIsNotNone(model_call_args)
        
        # Check kwargs contains dropout_rng
        _, kwargs = model_call_args
        self.assertIn("dropout_rng", kwargs)
        
        # Verify the dropout_rng passed to model is the same as our test key
        self.assertTrue(jnp.array_equal(kwargs["dropout_rng"], dropout_rng))
    
    def test_fine_tune_rng_propagation(self):
        """Test that fine_tune creates and propagates PRNG keys properly."""
        # Enable RNG keys for dropout
        self.fine_tuner.use_rng_keys_for_dropout = True
        
        # Mock _train_step to track RNG key changes
        original_train_step = self.fine_tuner._train_step
        rng_history = []
        
        def mock_train_step(state, batch, rng=None, grad_clip_norm=1.0):
            rng_history.append(rng)
            # Return a new RNG key, simulating what the real method would do
            new_rng = jax.random.split(rng)[0] if rng is not None else None
            return state, jnp.array(0.0), new_rng
            
        self.fine_tuner._train_step = mock_train_step
        
        # Mock _prepare_dataset to return a list of 5 fake batches
        def mock_prepare_dataset():
            return [
                {
                    "input_ids": jnp.ones((2, 10), dtype=jnp.int32),
                    "attention_mask": jnp.ones((2, 10), dtype=jnp.int32),
                    "labels": jnp.ones((2, 10), dtype=jnp.int32)
                }
                for _ in range(5)
            ]
        self.fine_tuner._prepare_dataset = mock_prepare_dataset
        
        # Mock generate_text and evaluate_perplexity to avoid errors
        self.mock_pruning_module.generate_text = mock.MagicMock(return_value="test text")
        self.mock_pruning_module.evaluate_perplexity = mock.MagicMock(return_value=10.0)
        
        # Run fine_tune with 1 epoch (should process 5 batches)
        params = {}  # Empty params for testing
        self.fine_tuner.fine_tune(params, num_epochs=1)
        
        # Verify:
        # 1. We have RNG keys recorded for each batch
        self.assertEqual(len(rng_history), 5)
        
        # 2. None of the RNG keys are None
        for rng in rng_history:
            self.assertIsNotNone(rng)
            
        # 3. Each RNG key is different from the previous one
        for i in range(1, len(rng_history)):
            self.assertFalse(jnp.array_equal(rng_history[i-1], rng_history[i]))


if __name__ == "__main__":
    unittest.main()