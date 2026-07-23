"""
prepare_data.py — Tokenise TinyStories and write the result to a compact
binary file (tokens.bin) without holding all tokens in RAM at once.

The output file stores token IDs as unsigned 16-bit integers (uint16).
Our BPE vocabulary has 8,000 tokens, which fits comfortably within
uint16's range of 0–65,535.  Using uint16 instead of int64 shrinks the
file from ~3.4 GB (int64) to ~0.85 GB — a 4x saving.

Run this script once before training:
    python prepare_data.py

It produces:
    tokens.bin   — flat binary array of uint16 token IDs
    tokens.meta  — tiny text file recording the total token count
"""

import os
import struct
import numpy as np
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_script_dir     = os.path.dirname(os.path.abspath(__file__))
_tokenizer_path = os.path.join(_script_dir, "tinystories_bpe.json")
_data_path      = os.path.join(_script_dir, "tinystories.txt")
_output_path    = os.path.join(_script_dir, "tokens.bin")
_meta_path      = os.path.join(_script_dir, "tokens.meta")

# ---------------------------------------------------------------------------
# 1. Load the BPE tokenizer
# ---------------------------------------------------------------------------

tokenizer = Tokenizer.from_file(_tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

print(f"Loaded BPE tokenizer  |  vocab size: {vocab_size}")
assert vocab_size <= 65535, (
    f"vocab_size ({vocab_size}) exceeds uint16 max (65535)!"
)

# ---------------------------------------------------------------------------
# 2. Process tinystories.txt in chunks and write token IDs to tokens.bin
# ---------------------------------------------------------------------------
# Strategy:
#   - Read 50 MB of text at a time (keeps RAM usage low)
#   - Encode each chunk with the BPE tokenizer
#   - Immediately write the resulting uint16 IDs to disk and discard them
#   - At no point do we hold more than ~50 MB of text + one chunk of IDs

_chunk_size = 50 * 1024 * 1024   # 50 MB per chunk

print(f"\nReading {_data_path} in {_chunk_size // (1024*1024)} MB chunks...")
print(f"Writing token IDs (uint16) to {_output_path}")
print("(This may take a few minutes for a 1.8 GB file.)\n")

total_chars  = 0
total_tokens = 0

with open(_data_path, "r", encoding="utf-8") as f_in, \
     open(_output_path, "wb") as f_out:

    chunk_num = 0
    while True:
        chunk = f_in.read(_chunk_size)
        if not chunk:
            break

        total_chars += len(chunk)

        # Encode this chunk of text into BPE token IDs
        ids = tokenizer.encode(chunk).ids

        # Convert to a numpy uint16 array and write raw bytes to disk
        arr = np.array(ids, dtype=np.uint16)
        f_out.write(arr.tobytes())

        total_tokens += len(ids)
        chunk_num += 1

        # Progress update every chunk (~50 MB of text)
        print(
            f"  chunk {chunk_num:>3d} | "
            f"{total_chars / 1e9:.2f} GB processed | "
            f"{total_tokens:>12,} tokens so far"
        )

# ---------------------------------------------------------------------------
# 3. Write metadata (token count) so data.py knows the array length
# ---------------------------------------------------------------------------

with open(_meta_path, "w") as f:
    f.write(str(total_tokens))

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

file_size_mb = os.path.getsize(_output_path) / (1024 * 1024)

print(f"\nDone!")
print(f"Total characters processed: {total_chars:,}")
print(f"Total tokens written:       {total_tokens:,}")
print(f"Compression:                ~{total_chars / total_tokens:.1f}x (chars -> tokens)")
print(f"tokens.bin size:            {file_size_mb:.2f} MB (uint16)")
print(f"tokens.meta:                {_meta_path}")
