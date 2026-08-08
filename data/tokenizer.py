"""
tokenizer.py — Character-level tokenizer for the Tiny Shakespeare dataset.

Reads input.txt, builds a vocabulary of unique characters, and provides
encode/decode functions to convert between strings and integer token IDs.
"""

import os

# ---------------------------------------------------------------------------
# 1. Read the dataset
# ---------------------------------------------------------------------------

# Resolve input.txt relative to this script's directory so it works
# regardless of the current working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_input_path = os.path.join(_script_dir, "input.txt")

with open(_input_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset length: {len(text)} characters")

# ---------------------------------------------------------------------------
# 2. Build the character-level vocabulary
# ---------------------------------------------------------------------------

# Get every unique character, sort them so the mapping is deterministic,
# and assign each one a unique integer ID starting from 0.
chars = sorted(set(text))
vocab_size = len(chars)

# Lookup tables for fast conversion in both directions:
#   stoi  — string (char) → integer ID
#   itos  — integer ID → string (char)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# ---------------------------------------------------------------------------
# 3. Encode and decode functions
# ---------------------------------------------------------------------------


def encode(s: str) -> list[int]:
    """Convert a string into a list of integer token IDs.

    Each character is mapped to its ID via the `stoi` lookup table.

    Args:
        s: The input string to encode.

    Returns:
        A list of integers representing the token IDs.
    """
    return [stoi[ch] for ch in s]


def decode(ids: list[int]) -> str:
    """Convert a list of integer token IDs back into a string.

    Each ID is mapped back to its character via the `itos` lookup table.

    Args:
        ids: A list of integer token IDs.

    Returns:
        The reconstructed string.
    """
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 4. Quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(chars)}")
    print()

    # Encode the first 100 characters and display the token IDs
    sample = text[:100]
    encoded = encode(sample)
    print(f"First 100 characters of text:\n{sample}")
    print(f"\nEncoded as token IDs:\n{encoded}")

    # Round-trip check: decode the IDs back and verify they match
    decoded = decode(encoded)
    assert decoded == sample, "Round-trip encode→decode failed!"
    print("\n[OK] Round-trip encode/decode check passed.")
