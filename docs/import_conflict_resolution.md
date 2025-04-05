# Import Conflict Resolution

This document explains how we resolved the import conflict between the Hugging Face datasets library and our project's module.

## The Problem

The Hugging Face datasets library tries to import a module called `sentinel_data.table`, which conflicts with our own module name `sentinel_data`. This causes import errors when trying to use both the datasets library and our own code. Additionally, the datasets library has circular imports in its dependencies which further complicate the resolution.

The specific errors include:

```
ModuleNotFoundError: No module named 'sentinel_data.table'
```

and 

```
NameError: name 'datasets' is not defined
```

These errors occur because the datasets library expects a specific module structure that we don't provide, but the module name collides with ours.

## The Solution

We implemented three key approaches to resolve this conflict:

1. **Renamed our module**: We renamed our `sentinel_data` module to `sdata` to avoid the name collision.

2. **Created compatibility module**: We created a stub implementation of `sentinel_data.table` to satisfy the import requirement of the datasets library, even though it should never be called in practice.

3. **Used mock imports for testing**: For our testing scripts, we use mock versions of the `transformers` and `datasets` modules to avoid import issues when testing our own code.

## Implementation Details

### 1. Module Renaming

- Created a new `sdata` module with the same functionality as `sentinel_data`
- Updated all imports in the project to use `sdata` instead of `sentinel_data` using `update_imports.py`
- Provided clear documentation in `sdata/README.md` explaining the change

### 2. Compatibility Module

- Created `sentinel_data/table.py` with a stub implementation of `table_cast()` 
- Created additional stub modules needed by datasets
- Implemented a full folder structure to match what datasets expects

### 3. Testing with Mock Imports

For test scripts that need to test our code without actually importing datasets:

- Created mock implementations of the `transformers` and `datasets` modules
- Used these mocks to test our task suite implementation without triggering import errors
- Created `scripts/simple_test_with_mock.py` to demonstrate this approach

## Testing Approaches

We have several ways to test our code with this conflict resolution:

1. **Isolated mock tests**: For the most reliable testing, use completely isolated mock implementations that don't import any problematic modules (see `scripts/isolated_mock_test.py`).

2. **Mock imports**: For tests that don't need actual transformers or datasets functionality, use mock versions (see `scripts/simple_test_with_mock.py`).

3. **Import order**: For scripts that need both, ensure `sdata` is imported before any imports that might trigger the datasets import.

4. **Dynamic patching**: For more complex cases, use scripts like `scripts/apply_comprehensive_patch.py` that can dynamically patch the datasets module.

The recommended approach for most testing is to use isolated mock tests, as they provide the most reliable way to test your code without external dependencies.

## Usage

Before:
```python
from sentinel_data import load_dataset, evaluate_model
```

After:
```python
from sdata import load_dataset, evaluate_model
```

## Future Considerations

1. **Long-term solution**: The ideal solution would be to completely eliminate the name conflict by renaming our module to something unique. We've already started this process by using `sdata`.

2. **Testing implications**: When writing tests, we need to be careful about import order and consider using mock imports where appropriate.

3. **Handling updates**: If the Hugging Face datasets library changes its import structure in the future, we may need to update our compatibility module.

## Recommended Approach for Development

1. For most development, use the `sdata` module directly and avoid importing the datasets library when not needed.

2. For testing basic functionality, use the mock modules approach as demonstrated in `scripts/simple_test_with_mock.py`.

3. For full system tests that require both our code and the datasets library, consider using the patching approach or running tests in separate processes.