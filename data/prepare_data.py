"""
prepare_data.py — Tokenise general web text and conversational turns and write
the result to a compact binary file (tokens.bin) without holding all tokens in RAM at once.

The output file stores token IDs as unsigned 16-bit integers (uint16).
Our BPE vocabulary has 20,000 tokens, which fits comfortably within
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
_tokenizer_path = os.path.join(_script_dir, "general_bpe.json")
_general_path   = os.path.join(_script_dir, "general_text.txt")
_conv_path      = os.path.join(_script_dir, "conversations.txt")
_alpaca_path    = os.path.join(_script_dir, "alpaca.txt")
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
# 2. Process both datasets in chunks and write token IDs to tokens.bin
# ---------------------------------------------------------------------------

_chunk_size = 5 * 1024 * 1024   # 5 million characters per chunk

input_files = [_general_path, _conv_path, _alpaca_path]
print(f"\nWriting token IDs (uint16) to {_output_path}")

total_chars  = 0
total_tokens = 0
chunk_num    = 0

general_chars = 0
general_tokens = 0
conv_chars = 0
conv_tokens = 0

with open(_output_path, "wb") as f_out:
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist, skipping.")
            continue
            
        print(f"Processing source file: {file_path}...", flush=True)
        is_general = ("general_text.txt" in file_path)
        file_chars = 0
        
        with open(file_path, "r", encoding="utf-8") as f_in:
            while True:
                chunk_to_read = _chunk_size
                if is_general:
                    max_chars = 500 * 1024 * 1024
                    if file_chars >= max_chars:
                        break
                    chunk_to_read = min(_chunk_size, max_chars - file_chars)
                    
                chunk = f_in.read(chunk_to_read)
                if not chunk:
                    break

                chars_read = len(chunk)
                file_chars += chars_read
                total_chars += chars_read

                if is_general:
                    general_chars += chars_read
                else:
                    conv_chars += chars_read

                ids = tokenizer.encode(chunk).ids
                
                arr = np.array(ids, dtype=np.uint16)
                f_out.write(arr.tobytes())

                toks_read = len(ids)
                total_tokens += toks_read
                
                if is_general:
                    general_tokens += toks_read
                else:
                    conv_tokens += toks_read
                    
                chunk_num += 1

                print(
                    f"  chunk {chunk_num:>3d} | "
                    f"{total_chars / 1e9:.2f} GB processed | "
                    f"{total_tokens:>12,} tokens so far", flush=True
                )

# ---------------------------------------------------------------------------
# 3. Write metadata (token count) so data.py knows the array length
# ---------------------------------------------------------------------------

with open(_meta_path, "w") as f:
    f.write(str(total_tokens))

print("\nCleaning up source text files...")
for file_path in input_files:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"  Deleted {file_path} to free disk space.")

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

file_size_mb = os.path.getsize(_output_path) / (1024 * 1024)

print(f"\nDone!")
print(f"Total characters processed: {total_chars:,}")
print(f"Total tokens written:       {total_tokens:,}")
print(f"Compression:                ~{total_chars / max(1, total_tokens):.1f}x (chars -> tokens)")
print(f"tokens.bin size:            {file_size_mb:.2f} MB (uint16)")
print(f"tokens.meta:                {_meta_path}")

print("\n" + "=" * 60)
print("Data Mix Ratio (General Text vs Instruction Data)")
print("=" * 60)
print(f"General Text (general_text.txt):")
print(f"  - Characters : {general_chars:,}")
print(f"  - Tokens     : {general_tokens:,} ({general_tokens / max(1, total_tokens) * 100:.1f}%)")
print(f"Instruction Data (OASST1 x8 + Alpaca):")
print(f"  - Characters : {conv_chars:,}")
print(f"  - Tokens     : {conv_tokens:,} ({conv_tokens / max(1, total_tokens) * 100:.1f}%)")
print("=" * 60)
