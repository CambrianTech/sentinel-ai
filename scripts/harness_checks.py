"""
Validation harness assertion helpers.

These functions implement the no-fallback discipline at the harness layer:
every check fails loudly with an error message that names *why* the check
exists, so a future engineer hitting the assertion is routed to the correct
understanding instead of being tempted to soften the check.

Used by:
- scripts/iter_defrag.py
- scripts/v1_to_v15_pad_upscale.py
- scripts/forge_v2_pipeline.sh (stage 2 inline checks)
- forge_model.py post-defrag verification
"""

from __future__ import annotations


def assert_explicit_head_dim(config) -> int:
    """The post-defrag config MUST carry head_dim explicitly.

    Computing head_dim from hidden_size // num_attention_heads is the v1 bug
    (Finding 6): after head pruning, num_attention_heads no longer equals
    hidden_size / head_dim, so the implicit computation gives the wrong
    answer and llama.cpp refuses to load. Any reload that doesn't carry
    head_dim explicitly will silently propagate the wrong value.
    """
    if not hasattr(config, "head_dim"):
        raise AssertionError(
            "head_dim must be explicit on the config. Computing it from "
            "hidden_size // num_attention_heads is wrong after defrag and is "
            "the v1 bug VALIDATED-TENSOR-SURGERY Finding 6 documents. The "
            "fix in defrag_inline.py:_update_config writes head_dim "
            "explicitly. If you are seeing this assertion, the post-defrag "
            "config update did not run, or save_pretrained dropped the "
            "field, or you are loading from a config that predates the fix."
        )
    if config.head_dim is None:
        raise AssertionError(
            "head_dim is present on config but is None. The fix in "
            "defrag_inline.py is supposed to write the integer head dim. "
            "None means the field exists but the value was not set."
        )
    return int(config.head_dim)


def assert_q_proj_invariant(q_proj_weight, hidden_size: int):
    """The q_proj wire shape must be (hidden_size, hidden_size).

    This is the llama.cpp invariant from Finding 6. If it does not hold,
    the artifact will load in transformers/vLLM but fail in llama.cpp's
    GGUF loader. The pad-mode defrag preserves this; slice-mode does not.
    """
    if tuple(q_proj_weight.shape) != (hidden_size, hidden_size):
        raise AssertionError(
            f"q_proj.shape == {tuple(q_proj_weight.shape)} but the llama.cpp "
            f"invariant requires (hidden_size, hidden_size) == "
            f"({hidden_size}, {hidden_size}). This artifact will fail to "
            f"load in llama.cpp / Ollama / LM Studio with 'wrong shape' "
            f"errors. See VALIDATED-TENSOR-SURGERY Finding 6. The fix is "
            f"--defrag-mode pad in the v2 forge."
        )


def assert_o_proj_invariant(o_proj_weight, hidden_size: int):
    """The o_proj wire shape must be (hidden_size, hidden_size). Mirror of q_proj."""
    if tuple(o_proj_weight.shape) != (hidden_size, hidden_size):
        raise AssertionError(
            f"o_proj.shape == {tuple(o_proj_weight.shape)} but expected "
            f"({hidden_size}, {hidden_size}). Same Finding 6 root cause as "
            f"the q_proj invariant — llama.cpp's GGUF loader hardcodes the "
            f"square shape on both Q and O projections."
        )


def assert_nondegenerate_output(text: str, prompt: str, *,
                                 unique_token_ratio: float = 0.4,
                                 max_repeat_ratio: float = 0.30):
    """Reject degenerate generation output.

    Two checks:
    1. Unique-token ratio: at least `unique_token_ratio` of the GENERATED
       (non-prompt) words must be unique. Catches cases like
       'def f(n): 0\\n elif 0\\n elif 0...'.
    2. Longest-repeated-substring: the longest substring that appears more
       than once in the generation must be at most `max_repeat_ratio` of
       the generation length. Catches looped repetition that
       diverse-tokens-alone misses, e.g. 'print(f(5))\\nprint(f(6))\\n
       print(f(7))\\n...' where each line is novel but the structure loops.

    Both checks operate on the GENERATION ONLY (text minus the prompt),
    so a prompt that contains repeated tokens doesn't bias the result.

    The thresholds are intentionally generous (40% / 30%); they catch
    obvious degeneracy but allow normal code which has natural repetition
    (keywords, indentation, common names).

    KNOWN INCOMPLETE: these two checks together do not detect every
    degenerate output mode. They are a smoke test, not a full quality
    gate. Real evaluation lives in EvalPlus / HumanEval+, which is a
    separate stage in the pipeline.
    """
    # Strip prompt prefix if present (we want to test the generation, not the echo)
    generation = text[len(prompt):] if text.startswith(prompt) else text
    if len(generation.strip()) < 5:
        raise AssertionError(
            f"Generation is empty or near-empty (len={len(generation.strip())}). "
            f"text={text!r}"
        )

    # Check 1: unique token ratio
    tokens = generation.split()
    if len(tokens) >= 8:  # only meaningful with enough tokens to measure
        unique = len(set(tokens))
        ratio = unique / len(tokens)
        if ratio < unique_token_ratio:
            raise AssertionError(
                f"Degenerate generation: unique-token ratio {ratio:.2f} < "
                f"{unique_token_ratio} (only {unique} unique tokens out of "
                f"{len(tokens)}). Output is dominated by repetition. "
                f"generation={generation[:200]!r}"
            )

    # Check 2: longest repeated substring (cheap O(n^2) approach, fine for
    # short smoke-test outputs)
    def longest_repeated_substring(s: str) -> int:
        n = len(s)
        if n < 8:
            return 0
        best = 0
        # Use a small step to make this fast on a few-hundred-char string
        for length in range(min(n // 2, 50), 3, -1):
            seen = set()
            for i in range(n - length + 1):
                sub = s[i : i + length]
                if sub in seen:
                    return length
                seen.add(sub)
            if length <= best:
                break
        return best

    lrs = longest_repeated_substring(generation)
    if lrs > 0 and lrs / len(generation) > max_repeat_ratio:
        raise AssertionError(
            f"Degenerate generation: longest repeated substring is {lrs} "
            f"chars, which is {lrs/len(generation):.0%} of the generation "
            f"({len(generation)} chars). max_repeat_ratio = "
            f"{max_repeat_ratio}. The output contains a long looped "
            f"sequence. generation={generation[:200]!r}"
        )
