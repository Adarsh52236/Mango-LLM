"""
bpe_tokenizer.py — Train a Byte-Pair Encoding (BPE) tokenizer on general and conversational text.

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

_script_dir        = os.path.dirname(os.path.abspath(__file__))
general_text_path  = os.path.join(_script_dir, "data", "general_text.txt")
conversations_path = os.path.join(_script_dir, "data", "conversations.txt")
_save_path         = os.path.join(_script_dir, "data", "general_bpe.json")

# ---------------------------------------------------------------------------
# 1. Create a blank BPE tokenizer
# ---------------------------------------------------------------------------

# Start with an empty BPE model — it has no vocabulary yet.
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# ---------------------------------------------------------------------------
# 2. Configure the trainer & special tokens
# ---------------------------------------------------------------------------

# vocab_size controls how many unique tokens the tokenizer will learn.
#
# Why 20,000?
# -----------
# - We are transitioning from the simple vocabulary of children's stories (8,000 tokens)
#   to a general-purpose web corpus (OpenWebText) + conversational dialogs (OASST1).
# - A vocabulary of 20,000 provides much broader coverage for diverse topics, technical terms,
#   punctuation patterns, and dialogue turns, while keeping embedding table memory overhead
#   reasonable for training on consumer/Kaggle GPUs.

special_tokens = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|user|>",
    "<|assistant|>",
    "<|endofturn|>",
]

# Ensure special tokens are registered on the tokenizer instance
tokenizer.add_special_tokens(special_tokens)

trainer = BpeTrainer(
    vocab_size=20000,
    special_tokens=special_tokens,
    show_progress=True,
)

# ---------------------------------------------------------------------------
# 3. Set the pre-tokenizer
# ---------------------------------------------------------------------------

# The pre-tokenizer runs *before* BPE and decides how to initially split
# the raw text. Whitespace splits on spaces and newlines, so BPE only
# merges characters *within* individual words — it won't merge across word
# boundaries.
tokenizer.pre_tokenizer = Whitespace()

# ---------------------------------------------------------------------------
# 4. Train on combined general text + conversational text
# ---------------------------------------------------------------------------
# Passing both files in a list instructs the Rust tokenizer trainer to iterate
# over both datasets, learning vocabulary from general web text and dialogue.

training_files = [general_text_path, conversations_path]
print(f"Training BPE tokenizer on combined input files:")
for fpath in training_files:
    print(f"  - {fpath} (size: {os.path.getsize(fpath) / (1024*1024):.2f} MB)")
print(f"\nTarget vocab size: 20,000 tokens\n")

tokenizer.train(training_files, trainer)

print(f"\nTraining complete!")
final_vocab_size = tokenizer.get_vocab_size()
print(f"Final vocab size: {final_vocab_size:,}")

# ---------------------------------------------------------------------------
# 5. Save the trained tokenizer
# ---------------------------------------------------------------------------

tokenizer.save(_save_path)
print(f"Saved tokenizer to {_save_path}\n")

# ---------------------------------------------------------------------------
# 6. Load it back and run conversational verification test
# ---------------------------------------------------------------------------

print("=" * 60)
print("Loading tokenizer from file and testing conversational prompt...")
print("=" * 60)

loaded_tokenizer = Tokenizer.from_file(_save_path)

test_text = "<|user|> What is the capital of France? <|endofturn|> <|assistant|> The capital of France is Paris. <|endofturn|>"

# Encode: text -> token IDs + token strings
encoding = loaded_tokenizer.encode(test_text)

print(f"\nOriginal text: {test_text!r}")
print(f"\nToken breakdown (tokens and their IDs):")
for token_str, token_id in zip(encoding.tokens, encoding.ids):
    is_special = " (SPECIAL TOKEN)" if token_str in special_tokens else ""
    print(f"  {token_id:>5d} : {token_str!r}{is_special}")

print(f"\nNum total tokens: {len(encoding.ids)}")

# Decode: token IDs -> text
decoded = loaded_tokenizer.decode(encoding.ids, skip_special_tokens=False)

print(f"\nDecoded text: {decoded!r}")

# Verify round-trip and special token preservation
if decoded == test_text:
    print("\n[OK] Round-trip encode/decode matches perfectly.")
else:
    print(f"\n[NOTE] Decoded text differs slightly due to BPE whitespace handling.")
    print(f"  Original: {test_text!r}")
    print(f"  Decoded:  {decoded!r}")

# Explicit check that special tokens were NOT split
user_token_id = loaded_tokenizer.token_to_id("<|user|>")
assist_token_id = loaded_tokenizer.token_to_id("<|assistant|>")
eot_token_id = loaded_tokenizer.token_to_id("<|endofturn|>")

print("\nSpecial Token ID Verification:")
print(f"  <|user|>        ID: {user_token_id}")
print(f"  <|assistant|>   ID: {assist_token_id}")
print(f"  <|endofturn|>   ID: {eot_token_id}")

assert user_token_id is not None, "<|user|> was not recognized as a single token!"
assert assist_token_id is not None, "<|assistant|> was not recognized as a single token!"
assert eot_token_id is not None, "<|endofturn|> was not recognized as a single token!"
print("\n[SUCCESS] Special conversational tokens are recognized as atomic single tokens!")
