"""
data.py — Data loading and batching for character-level language modelling.

Encodes the Tiny Shakespeare dataset into a PyTorch tensor of token IDs,
splits it into training / validation sets, and provides a get_batch()
function that produces random (input, target) pairs for training.
"""

import os
import torch

# ---------------------------------------------------------------------------
# 1. Import vocabulary and encode/decode from our tokenizer
# ---------------------------------------------------------------------------

from tokenizer import encode, decode, vocab_size, text

# ---------------------------------------------------------------------------
# 2. Encode the entire dataset into a single 1-D tensor of token IDs
# ---------------------------------------------------------------------------

# encode() returns a plain Python list of ints; wrapping it in a LongTensor
# makes it ready for embedding lookups and GPU acceleration later on.
data = torch.tensor(encode(text), dtype=torch.long)

print(f"Encoded dataset shape: {data.shape}  dtype: {data.dtype}")
# e.g. torch.Size([1115394])  — one integer per character

# ---------------------------------------------------------------------------
# 3. Train / validation split (90 / 10)
# ---------------------------------------------------------------------------

n = int(0.9 * len(data))       # index where we cut
train_data = data[:n]          # first 90 %
val_data   = data[n:]          # remaining 10 %

print(f"Train size: {len(train_data)} tokens")
print(f"Val   size: {len(val_data)} tokens")

# ---------------------------------------------------------------------------
# 4. Automatically pick the best available device (GPU > CPU)
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 5. get_batch — sample random chunks for training or validation
# ---------------------------------------------------------------------------


def get_batch(split: str, block_size: int = 8, batch_size: int = 4):
    """Return a random batch of input–target pairs from the dataset.

    How the x / y shift works
    -------------------------
    Suppose block_size = 4 and a random starting index lands on position 10
    in the token-ID sequence:

        positions:  10  11  12  13  14
        tokens:      A   B   C   D   E

        x = [A, B, C, D]        (positions 10..13  — the context)
        y = [B, C, D, E]        (positions 11..14  — shifted right by 1)

    Each position in x is asking: "given the characters up to *here*,
    what is the *next* character?"  The matching position in y is the
    correct answer.  This single pair actually encodes block_size
    individual training examples of increasing context length:

        context [A]          → predict B
        context [A, B]       → predict C
        context [A, B, C]    → predict D
        context [A, B, C, D] → predict E

    Parameters
    ----------
    split : str
        "train" or "val" — selects which portion of the data to sample from.
    block_size : int
        Number of consecutive tokens in each input chunk (context length).
    batch_size : int
        Number of independent chunks to sample in parallel.

    Returns
    -------
    x : Tensor of shape (batch_size, block_size)
        Input token IDs (the context).
    y : Tensor of shape (batch_size, block_size)
        Target token IDs (the context shifted by one position).
    """
    # Pick the right dataset
    dataset = train_data if split == "train" else val_data

    # Randomly choose `batch_size` starting indices.
    # The maximum valid start is len(dataset) - block_size - 1 because
    # we need block_size tokens for x *plus* one more for the last target.
    ix = torch.randint(len(dataset) - block_size, (batch_size,))

    # Stack the chunks into (batch_size, block_size) tensors
    x = torch.stack([dataset[i   : i + block_size]     for i in ix])
    y = torch.stack([dataset[i + 1: i + block_size + 1] for i in ix])

    # Move tensors to GPU if available
    x, y = x.to(device), y.to(device)

    return x, y


# ---------------------------------------------------------------------------
# 6. Quick verification — print a sample batch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BLOCK_SIZE = 8
    BATCH_SIZE = 4

    xb, yb = get_batch("train", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    print(f"\nSample batch (block_size={BLOCK_SIZE}, batch_size={BATCH_SIZE}):")
    print(f"  x shape: {xb.shape}  (batch_size x block_size)")
    print(f"  y shape: {yb.shape}  (batch_size x block_size)")
    print(f"  x device: {xb.device}")

    # Show one example from the batch to illustrate the shift
    print(f"\n--- Example from batch row 0 ---")
    print(f"  x (input):  {xb[0].tolist()}")
    print(f"  y (target): {yb[0].tolist()}")
    print(f"  x decoded:  {decode(xb[0].tolist())!r}")
    print(f"  y decoded:  {decode(yb[0].tolist())!r}")

    print(f"\n  Step-by-step predictions this row teaches:")
    for t in range(BLOCK_SIZE):
        context = xb[0, : t + 1].tolist()
        target  = yb[0, t].item()
        print(f"    context {str(context):>30s}  ->  next token {target} ({decode([target])!r})")
