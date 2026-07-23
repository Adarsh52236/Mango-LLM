"""
data.py — Data loading and batching for BPE-tokenised language modelling.

Encodes the TinyStories dataset into a PyTorch tensor of BPE token IDs,
splits it into training / validation sets, and provides a get_batch()
function that produces random (input, target) pairs for training.
"""

import os
import torch
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# 1. Load the trained BPE tokenizer
# ---------------------------------------------------------------------------

_script_dir     = os.path.dirname(os.path.abspath(__file__))
_tokenizer_path = os.path.join(_script_dir, "tinystories_bpe.json")

bpe_tokenizer = Tokenizer.from_file(_tokenizer_path)
vocab_size    = bpe_tokenizer.get_vocab_size()

print(f"Loaded BPE tokenizer  |  vocab size: {vocab_size}")

# Convenience wrappers matching the old char-level API
def encode(text: str) -> list[int]:
    """Encode a string into a list of BPE token IDs."""
    return bpe_tokenizer.encode(text).ids

def decode(ids: list[int]) -> str:
    """Decode a list of BPE token IDs back into a string."""
    return bpe_tokenizer.decode(ids)

# ---------------------------------------------------------------------------
# 2. Read and encode the full TinyStories dataset (in chunks)
# ---------------------------------------------------------------------------
# The raw text is ~1.8 GB.  Encoding it all in one call would require the
# tokenizer to allocate a massive internal buffer (~17 GB), which will fail
# on most machines.  Instead we read and encode in manageable chunks, then
# concatenate the resulting token-ID lists.

_data_path  = os.path.join(_script_dir, "tinystories.txt")
_chunk_size = 10 * 1024 * 1024   # 10 MB per chunk

print(f"Reading and encoding {_data_path} in {_chunk_size // (1024*1024)} MB chunks...")
print("(This may take a few minutes for a 1.8 GB file.)")

all_ids: list[int] = []
total_chars = 0

with open(_data_path, "r", encoding="utf-8") as f:
    chunk_num = 0
    while True:
        chunk = f.read(_chunk_size)
        if not chunk:
            break
        total_chars += len(chunk)
        all_ids.extend(bpe_tokenizer.encode(chunk).ids)
        chunk_num += 1
        if chunk_num % 20 == 0:   # progress every ~200 MB
            print(f"  ...encoded {total_chars / 1e9:.2f} GB  ({len(all_ids):,} tokens so far)")

print(f"  ...done! {total_chars:,} characters total.")

# Wrap in a PyTorch LongTensor for embedding lookups and GPU acceleration.
data = torch.tensor(all_ids, dtype=torch.long)

print(f"Encoded dataset shape: {data.shape}  dtype: {data.dtype}")
print(f"Compression: {total_chars:,} chars -> {len(data):,} tokens "
      f"(~{total_chars / len(data):.1f}x)")

# ---------------------------------------------------------------------------
# 3. Train / validation split (90 / 10)
# ---------------------------------------------------------------------------

n = int(0.9 * len(data))       # index where we cut
train_data = data[:n]          # first 90 %
val_data   = data[n:]          # remaining 10 %

print(f"Train size: {len(train_data):,} tokens")
print(f"Val   size: {len(val_data):,} tokens")

# ---------------------------------------------------------------------------
# 4. Automatically pick the best available device (GPU > CPU)
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 5. get_batch — sample random chunks for training or validation
# ---------------------------------------------------------------------------


def get_batch(split: str, block_size: int = 8, batch_size: int = 4):
    """Return a random batch of input-target pairs from the dataset.

    How the x / y shift works
    -------------------------
    Suppose block_size = 4 and a random starting index lands on position 10
    in the token-ID sequence:

        positions:  10  11  12  13  14
        tokens:      A   B   C   D   E

        x = [A, B, C, D]        (positions 10..13  -- the context)
        y = [B, C, D, E]        (positions 11..14  -- shifted right by 1)

    Each position in x is asking: "given the tokens up to *here*,
    what is the *next* token?"  The matching position in y is the
    correct answer.  This single pair actually encodes block_size
    individual training examples of increasing context length:

        context [A]          -> predict B
        context [A, B]       -> predict C
        context [A, B, C]    -> predict D
        context [A, B, C, D] -> predict E

    Parameters
    ----------
    split : str
        "train" or "val" -- selects which portion of the data to sample from.
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
# 6. Quick verification -- print a sample batch
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
