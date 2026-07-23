"""
data.py — Memory-efficient data loading and batching for BPE language modelling.

Instead of loading all ~449M tokens into RAM as a PyTorch tensor, this module
uses numpy.memmap to access tokens.bin directly from disk.  Memmap maps the
file into virtual memory so the OS loads only the pages we actually touch —
typically just the tiny slices needed for each batch.  This lets us train on
datasets far larger than available RAM.

Prerequisites:
    Run `python prepare_data.py` first to create tokens.bin and tokens.meta.
"""

import os
import numpy as np
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

# Convenience wrappers
def encode(text: str) -> list[int]:
    """Encode a string into a list of BPE token IDs."""
    return bpe_tokenizer.encode(text).ids

def decode(ids: list[int]) -> str:
    """Decode a list of BPE token IDs back into a string."""
    return bpe_tokenizer.decode(ids)

# ---------------------------------------------------------------------------
# 2. Load tokens.bin via numpy memmap
# ---------------------------------------------------------------------------
# Why memmap instead of loading the full file?
# ---------------------------------------------
# tokens.bin is ~850 MB (449M uint16 values).  Loading it all into RAM as a
# PyTorch int64 tensor would consume ~3.4 GB.  With memmap the OS maps the
# file into virtual address space without reading it up front.  When
# get_batch() slices a small region, only those disk pages are loaded into
# RAM (typically 4 KB each).  This means:
#
#   - Startup is instant (no multi-minute encoding step)
#   - RAM usage is proportional to batch_size * block_size, not dataset size
#   - The same code works whether the dataset is 1 MB or 100 GB

_tokens_path = os.path.join(_script_dir, "tokens.bin")
_meta_path   = os.path.join(_script_dir, "tokens.meta")

# Read the total token count written by prepare_data.py
with open(_meta_path, "r") as f:
    total_tokens = int(f.read().strip())

# Memory-map the binary file as a flat array of uint16 values.
# mode='r' means read-only — we never modify the prepared data.
data = np.memmap(_tokens_path, dtype=np.uint16, mode="r", shape=(total_tokens,))

print(f"Memory-mapped tokens.bin  |  {total_tokens:,} tokens  |  dtype: uint16")

# ---------------------------------------------------------------------------
# 3. Train / validation split (90 / 10)
# ---------------------------------------------------------------------------
# With memmap we don't actually copy anything — train_data and val_data are
# just *views* into different regions of the same memory-mapped file.

n = int(0.9 * total_tokens)
train_data = data[:n]
val_data   = data[n:]

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
    correct answer.

    The memmap array is uint16 (to save disk space), but PyTorch embeddings
    need int64 (LongTensor).  We cast when constructing the tensors.

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
    # Pick the right dataset (a memmap view — no copy)
    dataset = train_data if split == "train" else val_data

    # Randomly choose `batch_size` starting indices.
    ix = torch.randint(len(dataset) - block_size, (batch_size,))

    # Slice the memmap for each sample, convert uint16 -> int64 tensor.
    # Only the touched pages (~block_size * 2 bytes per sample) are read
    # from disk; the rest of the 850 MB file stays untouched.
    x = torch.stack([
        torch.from_numpy(dataset[i   : i + block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(dataset[i + 1: i + block_size + 1].astype(np.int64))
        for i in ix
    ])

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
