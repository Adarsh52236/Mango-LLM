"""
train_colab.py — Colab-optimised training script for the GPT language model.

Designed to run on Google Colab with a GPU runtime.  Key differences from
train.py:
  - Much larger model (1024-dim, 16 heads, 20 layers, 512 context)
  - Mixed-precision training (float16) to fit the model in limited VRAM
  - Checkpoint saving/loading to Kaggle working directory so progress survives
    session disconnects and runtime resets

Usage (in a Kaggle notebook cell):
    !python train_colab.py
"""

import os
import glob
import re
import torch


from model import GPTLanguageModel
from data import get_batch, vocab_size, decode

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Scaled up significantly compared to train.py.  These are sized for a
# single Colab T4/A100 GPU with 16-40 GB VRAM.

embedding_dim  = 1024       # size of token / positional embeddings
num_heads      = 16         # number of parallel attention heads per block
num_layers     = 20         # number of stacked transformer blocks
block_size     = 512        # maximum context length (tokens the model can see)
batch_size     = 4          # number of independent sequences per training step (reduced from 32 to avoid CUDA OOM)
gradient_accumulation_steps = 8  # number of steps to accumulate gradients (4 * 8 = 32 effective batch size)
learning_rate  = 3e-4       # AdamW learning rate
max_iters      = 20000      # total number of training iterations
eval_interval  = 500        # how often (in steps) to print train/val loss
eval_iters     = 100        # batches to average over when estimating loss
save_interval  = 2000       # how often (in steps) to save a checkpoint

# ---------------------------------------------------------------------------
# Checkpoint directory (Kaggle)
# ---------------------------------------------------------------------------
# Why save to /kaggle/working/checkpoints/?
# --------------------------------------------------------
# In Kaggle Notebooks, /kaggle/working/ is the persistent output directory
# for the session. We check both the uploaded read-only checkpoint in
# /kaggle/input/ and any newer checkpoints saved in /kaggle/working/checkpoints/
# to resume from the latest training step.

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("WARNING: No GPU detected. This script is designed for Colab GPU "
          "runtimes.  Training will be extremely slow on CPU.\n")

# ---------------------------------------------------------------------------
# 1. Instantiate the model
# ---------------------------------------------------------------------------

model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} parameters")

# ---------------------------------------------------------------------------
# 2. Create the optimizer
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3. Mixed-precision training setup
# ---------------------------------------------------------------------------
# Why mixed precision?
# --------------------
# By default, PyTorch uses float32 (32-bit) for all computations.  Mixed
# precision uses float16 (16-bit) for most operations:
#
#   - Memory:  float16 tensors use half the VRAM.  This lets us fit a model
#     that would otherwise OOM (out of memory) on a 16 GB GPU.  Activations,
#     gradients, and intermediate tensors all shrink.
#
#   - Speed:  Modern GPUs (Tensor Cores on T4/A100) execute float16 matrix
#     multiplications 2-8x faster than float32.
#
#   - Accuracy:  A GradScaler prevents the tiny float16 values from
#     underflowing to zero during backprop.  It scales the loss *up* before
#     backward() so gradients stay in a representable range, then scales
#     them back *down* before the optimizer step.  The master weights are
#     kept in float32 for numerical stability.
#
# Net effect: train a much bigger model, faster, with virtually no loss in
# quality.

scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

# ---------------------------------------------------------------------------
# 4. Checkpoint loading (resume from latest if available)
# ---------------------------------------------------------------------------

start_iter = 0

# Check both the read-only input checkpoint and any working checkpoints
checkpoint_files = []

input_ckpt = "/kaggle/input/mango-llm-checkpoint/checkpoint_014000.pt"
if os.path.exists(input_ckpt):
    checkpoint_files.append(input_ckpt)
checkpoint_files.extend(glob.glob("/kaggle/input/mango-llm-checkpoint/checkpoint_*.pt"))
checkpoint_files.extend(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.pt")))
checkpoint_files = list(set(checkpoint_files))


def get_checkpoint_iter(path: str) -> int:
    match = re.search(r"checkpoint_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


if checkpoint_files:
    checkpoint_files.sort(key=get_checkpoint_iter)
    latest_ckpt = checkpoint_files[-1]
    print(f"\nFound checkpoint: {latest_ckpt}")
    print("Loading and resuming training...")

    checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iteration"] + 1

    # If a scaler state was saved, restore it too
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"Resumed from iteration {start_iter}\n")
else:
    print("\nNo existing checkpoints found. Starting training from scratch.\n")

# ---------------------------------------------------------------------------
# 5. Loss estimation helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """Compute the average loss over several batches for train and val splits.

    Why not just print the loss from the latest training step?
    ----------------------------------------------------------
    A single batch's loss is *noisy* — it depends heavily on which random
    chunk of text happened to be sampled.  One batch might be easy (low loss)
    and the very next might be hard (high loss), making the printed numbers
    jump around wildly.

    By averaging over `eval_iters` independent batches we smooth out that
    randomness and get a much more reliable picture of how well the model is
    actually learning.
    """
    results = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, block_size=block_size, batch_size=batch_size)
            with torch.amp.autocast('cuda', dtype=torch.float16,
                               enabled=(device == "cuda")):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        results[split] = losses.mean().item()

    model.train()
    return results


# ---------------------------------------------------------------------------
# 6. Checkpoint saving helper
# ---------------------------------------------------------------------------


def save_checkpoint(iteration: int):
    """Save model, optimizer, and scaler state to the checkpoint directory.

    The filename includes the iteration number so multiple checkpoints
    can coexist and we can always identify the most recent one.
    """
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{iteration:06d}.pt")
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )
    print(f"  >> Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# 7. Training loop (with mixed precision and gradient accumulation)
# ---------------------------------------------------------------------------
# Why gradient accumulation?
# --------------------------
# Training a 20-layer, 1024-dim model with batch_size=32 requires storing a huge
# number of intermediate activations for backpropagation, which causes CUDA
# out-of-memory (OOM) errors on GPUs with limited VRAM.
#
# Gradient accumulation solves this by breaking the target effective batch size (32)
# into smaller micro-batches (batch_size=4) run sequentially over multiple steps
# (gradient_accumulation_steps=8).
#
#   - Memory: Because we only process 4 sequences at a time in the forward/backward
#     pass, peak VRAM usage is cut drastically (only activations for 4 sequences
#     are stored instead of 32).
#
#   - Effective Batch Size: Instead of updating weights after each micro-batch, we
#     let gradients accumulate (add up) in PyTorch's parameter `.grad` tensors over
#     8 iterations without calling zero_grad(). By dividing the loss by 8 before
#     each backward pass, the accumulated gradients equal the exact mathematical
#     average of gradients over all 32 sequences (4 * 8 = 32).
#
# Net effect: We achieve the exact same gradient update and training stability as a
# physical batch size of 32, but with a fraction of the peak VRAM!

print(f"Training from step {start_iter} to {max_iters}...\n")

for step in range(start_iter, max_iters):

    # --- Periodic evaluation ---
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step:>5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    # --- Periodic checkpointing ---
    if step > 0 and step % save_interval == 0:
        save_checkpoint(step)

    # --- Forward pass (mixed precision) ---
    xb, yb = get_batch("train", block_size=block_size, batch_size=batch_size)

    # autocast tells PyTorch to run eligible operations (matmuls, convolutions,
    # etc.) in float16 while keeping sensitive ones (reductions, softmax,
    # loss computation) in float32 — all automatically.
    with torch.amp.autocast('cuda', dtype=torch.float16,
                           enabled=(device == "cuda")):
        logits, loss = model(xb, yb)

    # --- Backward pass (with gradient scaling and accumulation) ---
    # Only zero gradients at the start of each accumulation cycle
    if step % gradient_accumulation_steps == 0:
        optimizer.zero_grad(set_to_none=True)

    # We divide the loss by gradient_accumulation_steps before calling backward().
    # Because PyTorch sums gradients across .backward() calls when they aren't zeroed,
    # scaling the loss down by 1/N ensures the accumulated gradients equal the average
    # gradient over the full effective batch size (N * batch_size).
    loss = loss / gradient_accumulation_steps

    # scaler.scale(loss) multiplies the loss by a large factor so that
    # float16 gradients don't underflow to zero during backward().
    scaler.scale(loss).backward()

    # Only step the optimizer and update the scaler after accumulating gradients
    # over gradient_accumulation_steps iterations (or on the final step).
    if (step + 1) % gradient_accumulation_steps == 0 or step == max_iters - 1:
        # scaler.step() first *unscales* the gradients back to float32 range,
        # checks for infs/NaNs (skipping the step if found), then calls
        # optimizer.step() with the corrected gradients.
        scaler.step(optimizer)

        # scaler.update() adjusts the scale factor for the next iteration,
        # increasing it when training is stable and decreasing it after
        # inf/NaN events.
        scaler.update()

# --- Final evaluation ---
losses = estimate_loss()
print(
    f"step {max_iters:>5d} | "
    f"train loss {losses['train']:.4f} | "
    f"val loss {losses['val']:.4f}"
)

# --- Save final checkpoint ---
save_checkpoint(max_iters)

# ---------------------------------------------------------------------------
# 8. Generate sample text
# ---------------------------------------------------------------------------

print("\n--- Generated text (200 tokens) ---\n")

# Start with a <bos> (beginning-of-sequence) token or a simple pad token.
# Token ID 0 is <pad> in our BPE vocab; we use it as a neutral starting point.
start = torch.zeros((1, 1), dtype=torch.long, device=device)

generated_ids = model.generate(start, max_new_tokens=200)
generated_text = decode(generated_ids[0].tolist())
print(generated_text)
