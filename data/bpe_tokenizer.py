"""
bpe_tokenizer.py — Train a Byte-Pair Encoding (BPE) tokenizer on TinyStories.

BPE starts with individual characters and iteratively merges the most frequent
adjacent pair into a new token, building up a vocabulary of common subwords.
This strikes a good balance: common words become single tokens (efficient),
while rare words are split into recognisable pieces (no "unknown" tokens).

This script uses the Hugging Face `tokenizers` library, which is written in
Rust and can train on gigabytes of text in minutes.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_script_dir   = os.path.dirname(os.path.abspath(__file__))
_data_path    = os.path.join(_script_dir, "tinystories.txt")
_save_path    = os.path.join(_script_dir, "tinystories_bpe.json")

# ---------------------------------------------------------------------------
# 1. Create a blank BPE tokenizer
# ---------------------------------------------------------------------------

# Start with an empty BPE model — it has no vocabulary yet.
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# ---------------------------------------------------------------------------
# 2. Configure the trainer
# ---------------------------------------------------------------------------

# vocab_size controls how many unique tokens the tokenizer will learn.
#
# Why 8,000?
# ----------
# - GPT-2 uses 50,257 tokens, but it was trained on a massive, diverse web
#   corpus (WebText) with millions of unique words, technical terms, code,
#   URLs, etc.  A large vocabulary is needed to represent all of that
#   efficiently.
#
# - TinyStories is a *much* simpler dataset: short children's stories with
#   a limited vocabulary of common English words.  There are far fewer
#   unique word forms, so 8,000 tokens is plenty to cover virtually all
#   words as single tokens (or at most two pieces).
#
# - Our model is also small (~200K-1M parameters).  A huge vocabulary would
#   mean a huge embedding table (vocab_size * embedding_dim parameters),
#   eating into the parameter budget without benefit.
#
# Rule of thumb: match vocabulary size to the complexity of your data and
# the capacity of your model.  8,000 is a sweet spot for our scale.

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=special_tokens,
    # Show a progress bar while training
    show_progress=True,
)

# ---------------------------------------------------------------------------
# 3. Set the pre-tokenizer
# ---------------------------------------------------------------------------

# The pre-tokenizer runs *before* BPE and decides how to initially split
# the raw text.  Whitespace splits on spaces and newlines, so BPE only
# merges characters *within* individual words — it won't merge across word
# boundaries (e.g., it won't create a token "the cat" spanning two words).
tokenizer.pre_tokenizer = Whitespace()

# ---------------------------------------------------------------------------
# 4. Train on tinystories.txt
# ---------------------------------------------------------------------------

print(f"Training BPE tokenizer on {_data_path}...")
print(f"Target vocab size: 8,000 tokens\n")

tokenizer.train([_data_path], trainer)

print(f"\nTraining complete!")
print(f"Final vocab size: {tokenizer.get_vocab_size()}")

# ---------------------------------------------------------------------------
# 5. Save the trained tokenizer
# ---------------------------------------------------------------------------

tokenizer.save(_save_path)
print(f"Saved tokenizer to {_save_path}\n")

# ---------------------------------------------------------------------------
# 6. Load it back and run a quick test
# ---------------------------------------------------------------------------

print("=" * 60)
print("Loading tokenizer from file and testing...")
print("=" * 60)

loaded_tokenizer = Tokenizer.from_file(_save_path)

test_text = "Once upon a time, there was a little cat."

# Encode: text -> token IDs + token strings
encoding = loaded_tokenizer.encode(test_text)

print(f"\nOriginal text: {test_text!r}")
print(f"Token IDs:     {encoding.ids}")
print(f"Token strings: {encoding.tokens}")
print(f"Num tokens:    {len(encoding.ids)}")

# Decode: token IDs -> text
decoded = loaded_tokenizer.decode(encoding.ids)

print(f"\nDecoded text:  {decoded!r}")

# Verify round-trip
if decoded == test_text:
    print("\n[OK] Round-trip encode/decode matches perfectly.")
else:
    # BPE decode may have minor whitespace differences depending on config;
    # this is expected and not necessarily an error.
    print(f"\n[NOTE] Decoded text differs slightly (common with BPE whitespace handling).")
    print(f"  Original: {test_text!r}")
    print(f"  Decoded:  {decoded!r}")
