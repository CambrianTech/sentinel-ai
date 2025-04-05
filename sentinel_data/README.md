# Sentinel Data Compatibility Module

This module is a compatibility layer to resolve import conflicts with the HuggingFace datasets library.

## Warning

**This module should not be used directly in your code.**

This is a stub implementation designed solely to satisfy import requirements of external libraries. For actual functionality, please use the `sdata` module instead.

## Purpose

The HuggingFace datasets library attempts to import a module called `sentinel_data.table`, which conflicts with our own module name. This compatibility layer provides the necessary stubs to satisfy those imports without affecting our actual codebase.

## Usage

Don't import from this module directly. Instead, use the renamed module:

```python
# Don't use this:
from sentinel_data import load_dataset  # Will raise a warning

# Use this instead:
from sdata import load_dataset  # Correct usage
```

## Implementation

This module provides:

1. Stub implementations of various datasets features
2. A table module for compatibility with datasets.arrow
3. Stubs for other datasets internal imports

See `docs/import_conflict_resolution.md` for more details about the import conflict resolution strategy.

## Testing

When writing tests that might import HuggingFace datasets, consider using one of the mock approaches documented in `docs/testing_with_mocks.md`.