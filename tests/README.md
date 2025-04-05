# Sentinel-AI Tests

This directory contains tests for the Sentinel-AI codebase.

## Running Tests

### Individual Test Scripts

You can run individual test scripts directly:

```bash
# Model loader tests
python tests/test_bloom_loader.py

# Other tests may follow similar patterns
```

### Using unittest

You can use Python's unittest framework to discover and run all tests:

```bash
python -m unittest discover
```

### Test Structure

- `test_bloom_loader.py`: Tests for the BLOOM model loader in both old and new package locations
- Unit tests: Tests for individual components
- Integration tests: Tests for interactions between components

## Writing New Tests

When writing new tests, follow these guidelines:

1. Place tests in the appropriate directory:
   - Unit tests of modules in the `tests/unit/` directory
   - Integration tests in the `tests/integration/` directory

2. Name test files with the prefix `test_`.

3. Include docstrings that describe what the test is checking.

4. Use assertions to verify expected behavior.

5. Clean up resources after tests (e.g., close files, delete temporary data).

## Test Coverage

The test suite aims to cover:

- Model loaders for different architectures
- Pruning strategies
- Training and inference workflows
- Controller functionality
- Utility functions

Additional tests will be added as the codebase evolves.