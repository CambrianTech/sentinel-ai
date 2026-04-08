# Reproducibility test cache

This directory holds **pinned snapshots** of every published `continuum-ai/*`
alloy referenced by `test_published_alloys_dispatch.py`. The test fetches
each alloy from HuggingFace on first run and caches it here, then asserts
the local family adapters can dispatch it.

## Why pin?

The adapter set is built **against these exact bytes**. If a published alloy
on HuggingFace is later edited, the local adapters must keep dispatching the
*pinned* version, not whatever HF currently serves — otherwise we lose the
"every shipped artifact reproduces from its alloy alone" guarantee.

A pinned snapshot is the contract. The test compares the local adapter set
against the contract, not against a moving target.

## How to refresh

To re-fetch from HuggingFace (e.g. after a published alloy is intentionally
updated):

    rm -rf tests/reproducibility/_cache/
    python -m pytest tests/reproducibility/test_published_alloys_dispatch.py

The next test run downloads fresh copies, and the diff in `git status` shows
exactly which alloys changed. **Review every diff before committing.** A
silent change in a published alloy is the kind of drift the reproducibility
gate exists to catch.

## File naming

Cache files are named `<repo-with-slashes-replaced-by-underscores>__<alloy-filename>`.
For example, `continuum-ai/qwen3.5-9b-general-forged/forge-alloy.json` becomes
`continuum-ai_qwen3.5-9b-general-forged__forge-alloy.json`.

## What's NOT cached here

- The actual safetensors / GGUF / MLX model weights — Tier 2 reproducibility
  hashes those against the alloy's `integrity.modelHash` but doesn't store
  the weights locally (multi-gigabyte).
- Eval samples / `samplesPath` references — fetched on demand by Tier 3.
