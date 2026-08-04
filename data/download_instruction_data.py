import os
from datasets import load_dataset

_script_dir = os.path.dirname(os.path.abspath(__file__))
conversations_path = os.path.join(_script_dir, "conversations.txt")
alpaca_path = os.path.join(_script_dir, "alpaca.txt")

# ---------------------------------------------------------------------------
# 1. Load OASST1
# ---------------------------------------------------------------------------
print("Loading full OpenAssistant/oasst1 dataset...")
oasst_ds = load_dataset("OpenAssistant/oasst1")

messages = {}
for split_name, split_data in oasst_ds.items():
    for msg in split_data:
        messages[msg["message_id"]] = msg

total_pairs = 0
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
                    total_pairs += 1
                    if msg.get("lang") == "en" and parent.get("lang") == "en":
                        turn_pairs.append((user_msg, assist_msg))

print(f"Extracted {len(turn_pairs):,} English-only valid user->assistant conversation pairs (out of {total_pairs:,} total pairs).")
print(f"Writing formatted turns to {conversations_path} (oversampled 8x)...")

with open(conversations_path, "w", encoding="utf-8") as f_out:
    for _ in range(8):
        for user_msg, assist_msg in turn_pairs:
            formatted_pair = (
                f"<|user|> {user_msg} <|endofturn|>\n"
                f"<|assistant|> {assist_msg} <|endofturn|>\n\n"
            )
            f_out.write(formatted_pair)

print("Finished writing OASST1 conversations!")

# ---------------------------------------------------------------------------
# 2. Load Alpaca
# ---------------------------------------------------------------------------
print("\nLoading tatsu-lab/alpaca dataset...")
alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")

print(f"Extracted {len(alpaca_ds):,} Alpaca instructions.")
print(f"Writing formatted turns to {alpaca_path}...")

with open(alpaca_path, "w", encoding="utf-8") as f_out:
    for row in alpaca_ds:
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        output = row.get("output", "").strip()
        
        user_msg = f"{instruction} {inp}".strip()
        
        formatted_pair = (
            f"<|user|> {user_msg} <|endofturn|>\n"
            f"<|assistant|> {output} <|endofturn|>\n\n"
        )
        f_out.write(formatted_pair)

print("Finished writing Alpaca instructions!")

# ---------------------------------------------------------------------------
# 3. Print final sizes
# ---------------------------------------------------------------------------
conv_file_size_mb = os.path.getsize(conversations_path) / (1024 * 1024)
alpaca_file_size_mb = os.path.getsize(alpaca_path) / (1024 * 1024)

print("\n" + "=" * 60)
print("Final Dataset Summary")
print("=" * 60)
print(f"1. OASST1 Conversations x8 ({conversations_path}):")
print(f"   - File Size       : {conv_file_size_mb:.2f} MB")
print(f"2. Alpaca Instructions ({alpaca_path}):")
print(f"   - File Size       : {alpaca_file_size_mb:.2f} MB")
print("=" * 60)
