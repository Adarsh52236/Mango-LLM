"""
download_general_data.py — Data preparation script for general-purpose language modeling
and instruction fine-tuning.

1. Streams a subset of Skylion007/openwebtext (up to ~500,000 documents or ~2.5 GB)
   and saves it as general_text.txt.
2. Loads the OpenAssistant/oasst1 dataset, extracts valid user->assistant reply pairs,
   and saves them formatted with conversational turn tokens into conversations.txt.
"""

import os
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_script_dir            = os.path.dirname(os.path.abspath(__file__))
general_text_path      = os.path.join(_script_dir, "general_text.txt")
conversations_path     = os.path.join(_script_dir, "conversations.txt")

# ---------------------------------------------------------------------------
# 1. Stream Skylion007/openwebtext subset
# ---------------------------------------------------------------------------
# Why stream OpenWebText instead of downloading it fully?
# --------------------------------------------------------
# The full OpenWebText dataset is over 40 GB compressed (and ~100 GB uncompressed),
# which would take hours to download and quickly exhaust local disk space and RAM.
# By setting streaming=True in load_dataset(), we download and process records
# on-the-fly over HTTP without saving the entire massive archive to disk.
# This allows us to cleanly extract only the subset we need (~2.5 GB / ~500,000 docs)
# with a tiny memory footprint and zero wasted storage!

print("=" * 60)
print("Step 1: Streaming Skylion007/openwebtext subset")
print("=" * 60)
print(f"Target: ~500,000 documents or ~2.5 GB of text")
print(f"Output file: {general_text_path}\n")

owt_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

max_docs  = 500_000
max_bytes = int(2.5 * 1024 * 1024 * 1024)  # 2.5 GB in bytes

total_owt_chars = 0
owt_doc_count   = 0

with open(general_text_path, "w", encoding="utf-8") as f_out:
    for doc in owt_stream:
        text = doc.get("text", "")
        if not text:
            continue

        # Write each document followed by a double newline separator
        chunk = text.strip() + "\n\n"
        f_out.write(chunk)

        total_owt_chars += len(chunk)
        owt_doc_count += 1

        if owt_doc_count % 25_000 == 0:
            gb_written = total_owt_chars / (1024 ** 3)
            print(f"  [OpenWebText] Processed {owt_doc_count:>7,} docs | {gb_written:.2f} GB written")

        if owt_doc_count >= max_docs or total_owt_chars >= max_bytes:
            break

print(f"\nFinished OpenWebText extraction!")

# ---------------------------------------------------------------------------
# 2. Load OpenAssistant/oasst1 and extract conversational pairs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Processing OpenAssistant/oasst1 conversations")
print("=" * 60)
print(f"Output file: {conversations_path}\n")

print("Loading full OpenAssistant/oasst1 dataset...")
oasst_ds = load_dataset("OpenAssistant/oasst1")

# Index all messages by message_id across all dataset splits (train + validation)
messages = {}
for split_name, split_data in oasst_ds.items():
    for msg in split_data:
        messages[msg["message_id"]] = msg

# Extract valid user -> assistant reply pairs
# Skip deleted messages, system messages, or malformed/broken threads
turn_pairs = []
for msg_id, msg in messages.items():
    if msg.get("role") == "assistant" and not msg.get("deleted", False):
        parent_id = msg.get("parent_id")
        if parent_id and parent_id in messages:
            parent = messages[parent_id]
            if parent.get("role") == "prompter" and not parent.get("deleted", False):
                user_msg = parent.get("text", "").strip()
                assist_msg = msg.get("text", "").strip()
                if user_msg and assist_msg:
                    turn_pairs.append((user_msg, assist_msg))

print(f"Extracted {len(turn_pairs):,} valid user->assistant conversation pairs.")
print(f"Writing formatted turns to {conversations_path}...")

total_conv_chars = 0
with open(conversations_path, "w", encoding="utf-8") as f_out:
    for user_msg, assist_msg in turn_pairs:
        formatted_pair = (
            f"<|user|> {user_msg} <|endofturn|>\n"
            f"<|assistant|> {assist_msg} <|endofturn|>\n\n"
        )
        f_out.write(formatted_pair)
        total_conv_chars += len(formatted_pair)

print("Finished writing OASST1 conversations!")

# ---------------------------------------------------------------------------
# 3. Print summary of final file sizes and character counts
# ---------------------------------------------------------------------------
owt_file_size_mb  = os.path.getsize(general_text_path) / (1024 * 1024)
conv_file_size_mb = os.path.getsize(conversations_path) / (1024 * 1024)

print("\n" + "=" * 60)
print("Final Dataset Summary")
print("=" * 60)
print(f"1. OpenWebText Subset ({general_text_path}):")
print(f"   - Total Documents : {owt_doc_count:,}")
print(f"   - Character Count : {total_owt_chars:,} characters")
print(f"   - File Size       : {owt_file_size_mb:.2f} MB ({owt_file_size_mb / 1024:.2f} GB)")
print("-" * 60)
print(f"2. OASST1 Conversations ({conversations_path}):")
print(f"   - Turn Pairs      : {len(turn_pairs):,} user->assistant pairs")
print(f"   - Character Count : {total_conv_chars:,} characters")
print(f"   - File Size       : {conv_file_size_mb:.2f} MB ({conv_file_size_mb / 1024:.2f} GB)")
print("=" * 60)
