"""
download_tinystories_legacy.py — Download the TinyStories dataset from Hugging Face and save
it as a single text file for character-level language model training.
"""

import os
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Load the TinyStories dataset (train split)
# ---------------------------------------------------------------------------

print("Loading TinyStories dataset from Hugging Face...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Loaded {len(dataset):,} stories.\n")

# ---------------------------------------------------------------------------
# 2. Concatenate all stories into one large string, separated by newlines
# ---------------------------------------------------------------------------

print("Concatenating stories...")
combined_text = "\n".join(story["text"] for story in dataset)

# ---------------------------------------------------------------------------
# 3. Write to tinystories.txt (in the same directory as this script)
# ---------------------------------------------------------------------------

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinystories.txt")

print(f"Writing to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(combined_text)

# ---------------------------------------------------------------------------
# 4. Print stats
# ---------------------------------------------------------------------------

total_chars = len(combined_text)
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

print(f"\nDone!")
print(f"Total characters: {total_chars:,}")
print(f"File size: {file_size_mb:.2f} MB")
