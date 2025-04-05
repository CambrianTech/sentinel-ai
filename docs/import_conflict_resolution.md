# Import Conflict Resolution

This document explains how we resolved the import conflict between the Hugging Face datasets library and our project's module.

## The Problem

The Hugging Face datasets library tries to import a module called `sentinel_data.table`, which conflicts with our own module name `sentinel_data`. This causes import errors when trying to use both the datasets library and our own code.

The specific error is:

```
ModuleNotFoundError: No module named 'sentinel_data.table'
```

This error occurs because the datasets library expects a specific module structure that we don't provide, but the module name collides with ours.

## The Solution

We implemented two key changes to resolve this conflict:

1. **Renamed our module**: We renamed our `sentinel_data` module to `sdata` to avoid the name collision.

2. **Created compatibility module**: We created a stub implementation of `sentinel_data.table` to satisfy the import requirement of the datasets library, even though it should never be called in practice.

## Implementation Details

### 1. Module Renaming

- Created a new `sdata` module with the same functionality as `sentinel_data`
- Updated all imports in the project to use `sdata` instead of `sentinel_data`
- Provided clear documentation in `sdata/README.md` explaining the change

### 2. Compatibility Module

- Created a stub implementation of `sentinel_data.table.table_cast()` that raises `NotImplementedError` if called
- This satisfies the import requirement of the datasets library without affecting our code
- The function is documented to explain its purpose

## Impact

This solution allows our code to coexist with the Hugging Face datasets library without conflicts. Users of our library should import from `sdata` instead of `sentinel_data`, and the transition should be transparent.

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

If the Hugging Face datasets library changes its import structure in the future, we may need to update our compatibility module. However, our renamed module (`sdata`) will continue to work correctly regardless of such changes.