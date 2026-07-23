"""
train.py — Training script for the GPT character-level language model.

Trains the model on the Tiny Shakespeare dataset and generates a sample
of text once training is complete.
"""

import torch

from model import GPTLanguageModel
from data import get_batch, vocab_size, device
from tokenizer import decode

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# All tunable knobs are gathered here for easy experimentation.

embedding_dim  = 64        # size of token / positional embeddings
num_heads      = 4         # number of parallel attention heads per block
num_layers     = 4         # number of stacked transformer blocks
block_size     = 32        # maximum context length (tokens the model can see)
batch_size     = 16        # number of independent sequences per training step
learning_rate  = 3e-4      # AdamW learning rate
max_iters      = 3000      # total number of training iterations
eval_interval  = 300       # how often (in steps) to print train/val loss
eval_iters     = 100       # batches to average over when estimating loss

# ---------------------------------------------------------------------------
# 1. Instantiate the model and move it to the best available device
# ---------------------------------------------------------------------------

model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
).to(device)

# Count and display total trainable parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} parameters")
print(f"Device: {device}\n")

# ---------------------------------------------------------------------------
# 2. Create the optimizer
# ---------------------------------------------------------------------------
# AdamW is the standard optimizer for training transformers.  It combines:
#   - Adam (adaptive per-parameter learning rates using first & second
#     moment estimates of gradients)
#   - Weight decay (regularisation that gently pushes unused weights toward
#     zero, helping prevent over-fitting)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3. Loss estimation helper
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
    actually learning.  This is the same idea as averaging multiple
    measurements in a science experiment.

    We also evaluate in model.eval() mode and with torch.no_grad() to:
      - Disable dropout / batch-norm training behaviour (if any).
      - Skip gradient computation, saving memory and time.
    """
    results = {}
    model.eval()  # switch to evaluation mode

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, block_size=block_size, batch_size=batch_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        results[split] = losses.mean().item()

    model.train()  # switch back to training mode
    return results


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

print("Starting training...\n")

for step in range(max_iters):

    # --- Periodic evaluation ---
    # Every eval_interval steps (and at step 0) print a stable loss estimate.
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step:>5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    # --- Forward pass ---
    # Sample a random batch of training data.
    xb, yb = get_batch("train", block_size=block_size, batch_size=batch_size)

    # Run the batch through the model to get predictions and loss.
    logits, loss = model(xb, yb)

    # --- Backward pass ---
    # Zero out gradients from the previous step (PyTorch accumulates them
    # by default, which is useful in some scenarios but not here).
    optimizer.zero_grad(set_to_none=True)

    # Compute gradients of the loss with respect to every model parameter.
    loss.backward()

    # Update parameters using the computed gradients.
    optimizer.step()

# Final evaluation after all training is done.
losses = estimate_loss()
print(
    f"step {max_iters:>5d} | "
    f"train loss {losses['train']:.4f} | "
    f"val loss {losses['val']:.4f}"
)

# ---------------------------------------------------------------------------
# 5. Generate text
# ---------------------------------------------------------------------------

print("\n--- Generated text (200 tokens) ---\n")

# Start from a single newline character (token ID 0 in our vocabulary).
# Shape (1, 1): one batch element, one starting token.
start = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate 200 new tokens autoregressively.
generated_ids = model.generate(start, max_new_tokens=200)

# Decode the token IDs back into a readable string.
generated_text = decode(generated_ids[0].tolist())
print(generated_text)
